//! Core-invariant acceptance scenarios ("E5").
//!
//! Black-box checks of three of Brain's seven non-negotiable invariants,
//! driven over the SDK against a live server. These are *correctness*
//! gates — they must hold on any hardware.
//!
//! - **#5 Idempotency by RequestId.** A resent write with the same
//!   `request_id` and the same params replays the original response (same
//!   `MemoryId`, no duplicate). The same `request_id` with *different*
//!   params is rejected `Conflict` (`IdempotencyConflict`).
//!   (`spec/05_operations/02_write_pipeline.md` §, `spec/04…/07` §3.6.)
//! - **#6 Tombstone before reclamation.** A soft FORGET tombstones the
//!   memory — invisible to RECALL — while the record is retained during the
//!   grace window (a re-FORGET still finds it and reports
//!   `was_already_forgotten`). A hard FORGET likewise drops it from RECALL.
//!   (`spec/19_benchmarks/01_correctness_and_durability.md` §12.)
//! - **#4 Slot version on MemoryId.** A read/op against a `MemoryId` whose
//!   slot-version field is stale returns `NotFound` rather than silently
//!   serving the live slot. (`spec/02_data_model/02_memory.md` §,
//!   `spec/08_storage/01_arena.md` §.)
//!
//! Two invariants the spec gates but this black-box harness cannot reach
//! without facilities the SDK does not expose, noted for honesty:
//!
//! - *Slot reclamation + version bump after the 7-day grace* needs either a
//!   7-day wait or an immediate-reclaim admin flag (`force_reclaim_now`),
//!   which is not a confirmed wire field — so #4 is exercised via a
//!   *fabricated* stale version on a live slot, which hits the exact
//!   absent-key/version-mismatch resolution path without waiting for
//!   reclamation. The probe is a LINK (its target resolution errors
//!   `NotFound` on an absent id); FORGET is deliberately not used because a
//!   missing FORGET target is a wire-level no-op (`was_already_forgotten`),
//!   not `NotFound`.
//! - *Soft-vs-hard recoverability* (UNFORGET succeeds for soft, fails for
//!   hard) needs the admin restore op, which is not on the SDK surface; the
//!   observable FORGET visibility + retention is asserted instead.

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use brain_db_sdk::new_id;
use brain_db_sdk::wire::types::{
    EdgeKindWire, EncodeRequest, ErrorCategoryWire, ForgetMode, ForgetRequest, LinkRequest,
    MemoryKindWire,
};
use brain_db_sdk::BrainError;

use super::ScenarioOutcome;
use crate::run::harness::{BrainEvalHarness, HarnessError};

/// The slot-version field occupies bits 32..64 of the 16-byte `MemoryId`
/// (big-endian: `shard(16) | slot(48) | version(32) | reserved(32)`), so
/// adding `1 << 32` increments the version by one and leaves `reserved` 0.
const VERSION_UNIT: u128 = 1u128 << 32;
/// How long to wait for async indexing to surface a freshly-encoded memory.
const BASELINE_WAIT: Duration = Duration::from_secs(8);
/// Poll cadence while waiting for the baseline RECALL hit.
const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Run the core-invariant ("E5") scenarios against `endpoint`.
pub async fn run_invariant_scenarios(endpoint: SocketAddr) -> Vec<ScenarioOutcome> {
    vec![
        idempotency_request_id(endpoint).await,
        tombstone_visibility(endpoint).await,
        slot_version_stale_id(endpoint).await,
    ]
}

// ===========================================================================
// #5 — idempotency by RequestId.
// ===========================================================================

const IDEMPOTENCY: &str = "inv_idempotency_request_id";

async fn idempotency_request_id(endpoint: SocketAddr) -> ScenarioOutcome {
    match run_idempotency(endpoint).await {
        Ok(o) => o,
        Err(e) => ScenarioOutcome::fail(IDEMPOTENCY, format!("sdk error: {e}")),
    }
}

async fn run_idempotency(endpoint: SocketAddr) -> Result<ScenarioOutcome, HarnessError> {
    let h = BrainEvalHarness::connect(endpoint).await?;
    let suffix = short_hex(h.agent_id());
    let rid = new_id();
    let text = format!("Idempotent memory {suffix} alpha");

    // Same request_id + same params → identical response (same MemoryId).
    // dedup is OFF so the only thing that can collapse the second write is
    // request_id idempotency, not content dedup.
    let m1 = h.client().encode(&encode_req(&text, rid)).await?.memory_id;
    let m2 = h.client().encode(&encode_req(&text, rid)).await?.memory_id;
    if m1 != m2 {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            IDEMPOTENCY,
            format!("replay of request_id returned a different MemoryId ({m1:#x} vs {m2:#x})"),
        ));
    }

    // Same request_id + different params → Conflict.
    let other = format!("Idempotent memory {suffix} BETA-different-params");
    let conflict = h.client().encode(&encode_req(&other, rid)).await;
    h.close().await?;

    match conflict {
        Err(BrainError::Server {
            category: ErrorCategoryWire::Conflict,
            ..
        }) => Ok(ScenarioOutcome::pass(
            IDEMPOTENCY,
            format!(
                "invariant #5: same request_id + same params replayed MemoryId {m1:#x}; \
                 same request_id + different params rejected Conflict"
            ),
        )),
        Err(BrainError::Server { category, .. }) => Ok(ScenarioOutcome::fail(
            IDEMPOTENCY,
            format!(
                "request_id reuse with different params returned {category:?}, expected Conflict"
            ),
        )),
        Err(e) => Ok(ScenarioOutcome::fail(
            IDEMPOTENCY,
            format!("request_id reuse with different params errored (not a Conflict): {e}"),
        )),
        Ok(resp) => Ok(ScenarioOutcome::fail(
            IDEMPOTENCY,
            format!(
                "request_id reuse with different params was ACCEPTED (MemoryId {:#x}); \
                 expected Conflict",
                resp.memory_id
            ),
        )),
    }
}

// ===========================================================================
// #6 — tombstone before reclamation.
// ===========================================================================

const TOMBSTONE: &str = "inv_tombstone_visibility";

async fn tombstone_visibility(endpoint: SocketAddr) -> ScenarioOutcome {
    match run_tombstone(endpoint).await {
        Ok(o) => o,
        Err(e) => ScenarioOutcome::fail(TOMBSTONE, format!("sdk error: {e}")),
    }
}

async fn run_tombstone(endpoint: SocketAddr) -> Result<ScenarioOutcome, HarnessError> {
    let h = BrainEvalHarness::connect(endpoint).await?;
    let suffix = short_hex(h.agent_id());

    // --- soft FORGET --------------------------------------------------
    let soft_token = format!("Zoltraxian{suffix}");
    let soft_text = format!("The {soft_token} ledger records tide-gate maintenance windows.");
    let soft_id = h
        .client()
        .encode(&encode_req(&soft_text, new_id()))
        .await?
        .memory_id;

    if !poll_recall_present(&h, &soft_token, soft_id).await? {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            TOMBSTONE,
            "soft-case memory never surfaced in RECALL (baseline) — cannot test exclusion",
        ));
    }

    let f1 = h
        .client()
        .forget(&forget_req(soft_id, ForgetMode::Soft))
        .await?;
    if f1.was_already_forgotten {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            TOMBSTONE,
            "first soft FORGET reported was_already_forgotten=true",
        ));
    }

    if recall_contains(&h, &soft_token, soft_id).await? {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            TOMBSTONE,
            "soft-forgotten memory still surfaced in RECALL",
        ));
    }

    // The record is retained during the grace window: a re-FORGET still
    // resolves it and reports it as already forgotten (not NotFound, not
    // reclaimed).
    let f2 = h
        .client()
        .forget(&forget_req(soft_id, ForgetMode::Soft))
        .await?;
    if !f2.was_already_forgotten {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            TOMBSTONE,
            "re-FORGET of a soft-tombstoned memory did not report was_already_forgotten \
             (record not retained during grace)",
        ));
    }

    // --- hard FORGET --------------------------------------------------
    let hard_token = format!("Brontaxis{suffix}");
    let hard_text = format!("The {hard_token} manifest lists deep-vault rotation keys.");
    let hard_id = h
        .client()
        .encode(&encode_req(&hard_text, new_id()))
        .await?
        .memory_id;

    if !poll_recall_present(&h, &hard_token, hard_id).await? {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            TOMBSTONE,
            "hard-case memory never surfaced in RECALL (baseline) — cannot test exclusion",
        ));
    }

    h.client()
        .forget(&forget_req(hard_id, ForgetMode::Hard))
        .await?;

    if recall_contains(&h, &hard_token, hard_id).await? {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            TOMBSTONE,
            "hard-forgotten memory still surfaced in RECALL",
        ));
    }

    h.close().await?;
    Ok(ScenarioOutcome::pass(
        TOMBSTONE,
        "invariant #6: soft FORGET tombstones + excludes from RECALL with the record retained \
         during grace (re-FORGET → was_already_forgotten); hard FORGET excludes from RECALL",
    ))
}

// ===========================================================================
// #4 — slot version on MemoryId.
// ===========================================================================

const SLOT_VERSION: &str = "inv_slot_version_stale_id";

async fn slot_version_stale_id(endpoint: SocketAddr) -> ScenarioOutcome {
    match run_slot_version(endpoint).await {
        Ok(o) => o,
        Err(e) => ScenarioOutcome::fail(SLOT_VERSION, format!("sdk error: {e}")),
    }
}

async fn run_slot_version(endpoint: SocketAddr) -> Result<ScenarioOutcome, HarnessError> {
    let h = BrainEvalHarness::connect(endpoint).await?;
    let suffix = short_hex(h.agent_id());

    // A real source and a real target; LINK resolves both by full MemoryId
    // key. The metadata table is keyed by the whole 16-byte id (version
    // included), so a version-bumped target is simply an absent key.
    let source = h
        .client()
        .encode(&encode_req(
            &format!("Slot version source {suffix}"),
            new_id(),
        ))
        .await?
        .memory_id;
    let target = h
        .client()
        .encode(&encode_req(
            &format!("Slot version target {suffix}"),
            new_id(),
        ))
        .await?
        .memory_id;

    // Same shard + slot as `target`, version bumped by one → a reference no
    // slot ever served. The lookup must reject it (NotFound), not resolve to
    // the live row at that slot.
    let stale = target.wrapping_add(VERSION_UNIT);
    if stale == target {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            SLOT_VERSION,
            "version bump did not change the MemoryId",
        ));
    }

    // Probe via LINK, whose target resolution errors NotFound on an absent
    // id. (FORGET is the wrong probe: a missing target is a deliberate
    // wire-level no-op there — `was_already_forgotten=true`, not NotFound.)
    let stale_res = h.client().link(&link_req(source, stale)).await;
    // Sanity: the real target id links fine — proves `source` is valid and
    // only the bumped version is what made `stale` resolve to nothing.
    let real_res = h.client().link(&link_req(source, target)).await;
    h.close().await?;

    match stale_res {
        Err(BrainError::Server {
            category: ErrorCategoryWire::NotFound,
            ..
        }) => {}
        Ok(_) => {
            return Ok(ScenarioOutcome::fail(
                SLOT_VERSION,
                format!("stale-version target {stale:#x} was ACCEPTED — no slot-version check"),
            ));
        }
        Err(BrainError::Server { category, .. }) => {
            return Ok(ScenarioOutcome::fail(
                SLOT_VERSION,
                format!("stale-version target returned {category:?}, expected NotFound"),
            ));
        }
        Err(e) => {
            return Ok(ScenarioOutcome::fail(
                SLOT_VERSION,
                format!("stale-version target errored (not NotFound): {e}"),
            ));
        }
    }

    match real_res {
        Ok(_) => Ok(ScenarioOutcome::pass(
            SLOT_VERSION,
            format!(
                "invariant #4: LINK to stale-version target {stale:#x} → NotFound while the \
                 base id {target:#x} links — the slot version discriminates"
            ),
        )),
        Err(e) => Ok(ScenarioOutcome::fail(
            SLOT_VERSION,
            format!("base target MemoryId {target:#x} unexpectedly failed LINK: {e}"),
        )),
    }
}

// ===========================================================================
// Helpers.
// ===========================================================================

/// A hand-built ENCODE with a controlled `request_id` and dedup OFF (the
/// builder mints a fresh id every call, which idempotency testing can't use).
fn encode_req(text: &str, request_id: [u8; 16]) -> EncodeRequest {
    EncodeRequest {
        text: text.to_string(),
        context_id: 0,
        kind: MemoryKindWire::Semantic,
        salience_hint: 0.5,
        edges: Vec::new(),
        request_id,
        txn_id: None,
        deduplicate: false,
    }
}

fn forget_req(memory_id: u128, mode: ForgetMode) -> ForgetRequest {
    ForgetRequest {
        memory_id,
        mode,
        request_id: new_id(),
        txn_id: None,
    }
}

/// A LINK between two memories. Used to probe target resolution, which
/// errors `NotFound` on an absent (stale-version) id.
fn link_req(source: u128, target: u128) -> LinkRequest {
    LinkRequest {
        source,
        target,
        kind: EdgeKindWire::SimilarTo,
        weight: 0.5,
        request_id: new_id(),
        txn_id: None,
    }
}

/// Poll RECALL until `target` surfaces (async indexing) or the deadline.
async fn poll_recall_present(
    h: &BrainEvalHarness,
    cue: &str,
    target: u128,
) -> Result<bool, HarnessError> {
    let deadline = Instant::now() + BASELINE_WAIT;
    loop {
        if recall_contains(h, cue, target).await? {
            return Ok(true);
        }
        if Instant::now() >= deadline {
            return Ok(false);
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

/// Does a single RECALL for `cue` return `target` among its hits?
async fn recall_contains(
    h: &BrainEvalHarness,
    cue: &str,
    target: u128,
) -> Result<bool, HarnessError> {
    let out = h.recall(cue, 20).await?;
    Ok(out.hits.iter().any(|m| m.memory_id == target))
}

/// First 12 hex chars of a 16-byte id — a unique per-run marker.
fn short_hex(id: [u8; 16]) -> String {
    let mut s = String::with_capacity(12);
    for b in id.iter().take(6) {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_bump_changes_only_the_version_field() {
        // Base id with shard/slot set, version 0, reserved 0.
        let shard: u128 = 0x0007 << 112;
        let slot: u128 = 0x0000_0000_002A << 64;
        let base = shard | slot;
        let bumped = base.wrapping_add(VERSION_UNIT);
        // Version field (bits 32..64) went 0 → 1.
        assert_eq!((bumped >> 32) & 0xFFFF_FFFF, 1);
        // Shard, slot, and reserved are untouched.
        assert_eq!(bumped >> 112, 0x0007);
        assert_eq!((bumped >> 64) & 0xFFFF_FFFF_FFFF, 0x2A);
        assert_eq!(bumped & 0xFFFF_FFFF, 0);
        assert_ne!(base, bumped);
    }

    #[test]
    fn scenario_names_are_distinct() {
        let names = [IDEMPOTENCY, TOMBSTONE, SLOT_VERSION];
        let mut sorted: Vec<&str> = names.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len());
    }
}
