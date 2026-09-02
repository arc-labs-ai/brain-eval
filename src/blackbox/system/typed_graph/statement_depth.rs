//! Statement-depth scenario: tombstone, retract-with-grace, and history chain.
//!
//! The lifecycle scenario ([`super::statement`]) covers create/get/list and
//! auto-supersession. This covers the §19/06 depth items that were missing:
//!
//! - **Tombstone** — a tombstoned statement drops out of the default
//!   `list_statements` view but reappears with `include_tombstoned = true`,
//!   flagged `tombstoned`. (Soft delete: hidden, not yet gone.)
//! - **Retract with grace** — `retract_statement` returns a `retracted_at`
//!   and a strictly-later `will_zero_at`, i.e. the tombstone-grace window
//!   (core invariant #6) is honoured rather than zeroing immediately.
//! - **History chain** — after a stateful re-create supersedes the prior
//!   statement, `statement_history` on either id walks the full chain
//!   (`total_versions >= 2`, both ids present).
//!
//! Built entirely on the system-declared `brain:behavior_tone` Preference
//! (a stateful `Value<text>` predicate) so nothing here depends on a
//! user-uploaded schema — the same proven shape the lifecycle scenario uses.

use std::net::SocketAddr;

use brain_db_sdk::new_id;
use brain_db_sdk::wire::types::{
    EntityCreateRequest, EvidenceRefWire, StatementCreateRequest, StatementHistoryRequest,
    StatementKindWire, StatementListRequest, StatementObjectWire, StatementTombstoneRequest,
    StatementValueWire,
};

use super::super::ScenarioOutcome;
use super::PERSON_TYPE_ID;
use crate::blackbox::system::hex16;
use crate::client::harness::{BrainEvalHarness, HarnessError};

const NAME: &str = "tg_statement_depth";
const PREDICATE: &str = "brain:behavior_tone";

/// Tombstone, retract-with-grace, and history-chain checks.
pub async fn statement_depth(endpoint: SocketAddr) -> ScenarioOutcome {
    match run(endpoint).await {
        Ok(o) => o,
        Err(e) => ScenarioOutcome::fail(NAME, format!("sdk error: {e}")),
    }
}

async fn run(endpoint: SocketAddr) -> Result<ScenarioOutcome, HarnessError> {
    let h = BrainEvalHarness::connect(endpoint).await?;
    let suffix = &hex16(h.space_id())[..12];
    let now = unix_nanos_now();

    // Independent subjects so the three checks don't interact: the predicate
    // is stateful, so two creates on one subject would auto-supersede each
    // other. One subject per check keeps tombstone, retract, and history clean.
    let subj_tomb = make_person(&h, &format!("Tomb Subject {suffix}")).await?;
    let subj_retract = make_person(&h, &format!("Retract Subject {suffix}")).await?;
    let subj_hist = make_person(&h, &format!("Hist Subject {suffix}")).await?;

    // --- tombstone -------------------------------------------------------
    let tomb_id = create_tone(&h, subj_tomb, &format!("brisk {suffix}"), now).await?;

    // Present in the default view before tombstoning.
    let before = list_ids(&h, subj_tomb, false).await?;
    if !before.contains(&tomb_id) {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            "statement not visible in list before tombstone",
        ));
    }

    h.client()
        .tombstone_statement(&StatementTombstoneRequest {
            statement_id: tomb_id,
            reason: 0,
            reason_message: "eval tombstone".to_string(),
            request_id: new_id(),
        })
        .await?;

    let after_default = list_ids(&h, subj_tomb, false).await?;
    if after_default.contains(&tomb_id) {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            "tombstoned statement still visible in the default (include_tombstoned=false) list",
        ));
    }
    // Still recoverable with include_tombstoned, and flagged.
    let listed_tomb = h
        .client()
        .list_statements(&list_req(subj_tomb, true))
        .await?;
    let tomb_row = listed_tomb.iter().find(|s| s.statement_id == tomb_id);
    match tomb_row {
        None => {
            h.close().await?;
            return Ok(ScenarioOutcome::fail(
                NAME,
                "tombstoned statement absent even with include_tombstoned=true (lost, not soft-deleted)",
            ));
        }
        Some(row) if !row.tombstoned => {
            h.close().await?;
            return Ok(ScenarioOutcome::fail(
                NAME,
                "statement resurfaced with include_tombstoned but tombstoned flag is false",
            ));
        }
        Some(_) => {}
    }

    // --- retract with grace ---------------------------------------------
    let retract_id = create_tone(&h, subj_retract, &format!("clipped {suffix}"), now + 1).await?;
    let retracted = h
        .client()
        .retract_statement(&brain_db_sdk::wire::types::StatementRetractRequest {
            statement_id: retract_id,
            reason: 0,
            reason_message: "eval retract".to_string(),
            request_id: new_id(),
        })
        .await?;
    if retracted.retracted_at_unix_nanos == 0 {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            "retract returned retracted_at = 0",
        ));
    }
    // Invariant #6: a soft retract sets a strictly-later zeroing deadline
    // (the grace window), it does not zero on the spot.
    if retracted.will_zero_at_unix_nanos <= retracted.retracted_at_unix_nanos {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "retract will_zero_at ({}) is not after retracted_at ({}) — grace window not honoured",
                retracted.will_zero_at_unix_nanos, retracted.retracted_at_unix_nanos
            ),
        ));
    }

    // --- history chain ---------------------------------------------------
    let v1 = create_tone(&h, subj_hist, &format!("tone-v1 {suffix}"), now).await?;
    // Stateful re-create supersedes v1, forming a two-version chain.
    let v2 = create_tone(&h, subj_hist, &format!("tone-v2 {suffix}"), now + 2).await?;
    let history = h
        .client()
        .statement_history(&StatementHistoryRequest {
            anchor_id: v2,
            include_tombstoned: true,
        })
        .await?;
    let has_v1 = history.iter().any(|s| s.statement_id == v1);
    let has_v2 = history.iter().any(|s| s.statement_id == v2);

    h.close().await?;

    if history.len() < 2 || !has_v1 || !has_v2 {
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "history chain incomplete: {} versions, v1_present={has_v1}, v2_present={has_v2}",
                history.len()
            ),
        ));
    }

    Ok(ScenarioOutcome::pass(
        NAME,
        format!(
            "§19/06 statement depth: tombstone hides from default view + flagged under \
             include_tombstoned; retract honours grace (zero_at > retracted_at); history \
             chain walks {} versions",
            history.len()
        ),
    ))
}

fn list_req(subject: [u8; 16], include_tombstoned: bool) -> StatementListRequest {
    StatementListRequest {
        subject,
        predicate: String::new(),
        kind: 0,
        min_confidence: 0.0,
        time_range_start_unix_nanos: 0,
        time_range_end_unix_nanos: 0,
        only_current: false,
        include_tombstoned,
        limit: 1000,
        cursor: Vec::new(),
        act_as: None,
    }
}

async fn list_ids(
    h: &BrainEvalHarness,
    subject: [u8; 16],
    include_tombstoned: bool,
) -> Result<Vec<[u8; 16]>, HarnessError> {
    let rows = h
        .client()
        .list_statements(&list_req(subject, include_tombstoned))
        .await?;
    Ok(rows.into_iter().map(|s| s.statement_id).collect())
}

async fn create_tone(
    h: &BrainEvalHarness,
    subject: [u8; 16],
    value: &str,
    valid_from: u64,
) -> Result<[u8; 16], HarnessError> {
    Ok(h.client()
        .create_statement(&StatementCreateRequest {
            session_id: 0,
            kind: StatementKindWire::Preference,
            subject,
            predicate: PREDICATE.to_string(),
            object: StatementObjectWire::Value(StatementValueWire::Text(value.to_string())),
            confidence: 0.95,
            evidence: EvidenceRefWire::Inline(Vec::new()),
            extractor_id: 0,
            valid_from_unix_nanos: valid_from,
            valid_to_unix_nanos: 0,
            event_at_unix_nanos: 0,
            schema_version: 0,
            request_id: new_id(),
            act_as: None,
        })
        .await?
        .statement_id)
}

async fn make_person(h: &BrainEvalHarness, canonical: &str) -> Result<[u8; 16], HarnessError> {
    Ok(h.client()
        .create_entity(&EntityCreateRequest {
            session_id: 0,
            entity_type_id: PERSON_TYPE_ID,
            canonical_name: canonical.to_string(),
            aliases: Vec::new(),
            attributes_blob: Vec::new(),
            request_id: new_id(),
            act_as: None,
        })
        .await?
        .entity_id)
}

fn unix_nanos_now() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}
