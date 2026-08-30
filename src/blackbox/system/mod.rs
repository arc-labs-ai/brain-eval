//! System & durability scenarios.
//!
//! Black-box behavioural checks that drive a real `brain-server` through
//! the SDK and assert end-to-end contracts the substrate must hold:
//!
//! - [`multi_space_isolation`] — one space's memories are invisible to
//!   another.
//! - [`encode_recall_forget`] — a memory is recallable, then gone after
//!   FORGET.
//! - [`txn_read_your_writes`] — a write buffered in a txn is visible to
//!   reads inside that txn, and to everyone after commit.
//!
//! Each returns a [`ScenarioOutcome`]; [`run_core_scenarios`] runs the
//! set. Heavier durability scenarios (restart-recovery, backfill, chaos)
//! build on [`crate::client::server::ServerHandle`] and land next.

use std::net::SocketAddr;

use brain_db_sdk::wire::types::{TxnBeginRequest, TxnCommitRequest};
use brain_db_sdk::{new_id, EncodeBuilder, ForgetBuilder, RecallBuilder};

use crate::client::harness::{BrainEvalHarness, HarnessError};

pub mod audit;
pub use audit::audit_derivation_logged;

pub mod backfill;
pub use backfill::resumable_backfill;

pub mod chaos;
pub use chaos::kill_during_write;

pub mod durability;
pub use durability::restart_recovery;

pub mod invariants;
pub use invariants::run_invariant_scenarios;

pub mod metrics_families;
pub use metrics_families::metrics_families_present;

pub mod rebuild;
pub use rebuild::index_rebuild;

pub mod storage;
pub use storage::storage_footprint;

pub mod typed_graph;
pub use typed_graph::run_typed_graph_scenarios;

/// Pass/fail result of one scenario.
#[derive(Debug, Clone)]
pub struct ScenarioOutcome {
    /// Scenario name.
    pub name: &'static str,
    /// Whether the scenario passed.
    pub passed: bool,
    /// Human-readable explanation (why it passed / how it failed).
    pub detail: String,
}

impl ScenarioOutcome {
    fn pass(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            passed: true,
            detail: detail.into(),
        }
    }
    fn fail(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            passed: false,
            detail: detail.into(),
        }
    }
}

/// Did any returned hit's text contain `needle`?
fn any_text_contains(hits: &[brain_db_sdk::wire::types::MemoryResult], needle: &str) -> bool {
    hits.iter().any(|m| m.text.contains(needle))
}

/// One space's memories must not surface in another space's recall.
pub async fn multi_space_isolation(endpoint: SocketAddr) -> ScenarioOutcome {
    const NAME: &str = "multi_space_isolation";
    match run_isolation(endpoint).await {
        Ok(outcome) => outcome,
        Err(e) => ScenarioOutcome::fail(NAME, format!("sdk error: {e}")),
    }
}

async fn run_isolation(endpoint: SocketAddr) -> Result<ScenarioOutcome, HarnessError> {
    const NAME: &str = "multi_space_isolation";
    let space_a = BrainEvalHarness::connect(endpoint).await?;
    let space_b = BrainEvalHarness::connect(endpoint).await?;

    // A unique marker keyed by space A's id so repeated runs don't collide.
    let marker = format!("isolation-marker-{}", hex16(space_a.space_id()));
    let secret = format!("{marker}: the vault code is alpha-tango-niner");
    space_a
        .client()
        .encode(&EncodeBuilder::new(secret.as_str()).build())
        .await?;

    let b_hits = space_b.recall(&marker, 10).await?;
    let a_hits = space_a.recall(&marker, 10).await?;

    space_a.close().await?;
    space_b.close().await?;

    if any_text_contains(&b_hits.memories, &marker) {
        return Ok(ScenarioOutcome::fail(
            NAME,
            "space B recalled space A's memory — isolation breach",
        ));
    }
    if !any_text_contains(&a_hits.memories, &marker) {
        return Ok(ScenarioOutcome::fail(
            NAME,
            "space A could not recall its own just-written memory",
        ));
    }
    Ok(ScenarioOutcome::pass(
        NAME,
        "A sees its memory; B (different space) does not",
    ))
}

/// A memory is recallable after ENCODE and gone after FORGET.
pub async fn encode_recall_forget(endpoint: SocketAddr) -> ScenarioOutcome {
    const NAME: &str = "encode_recall_forget";
    match run_forget(endpoint).await {
        Ok(outcome) => outcome,
        Err(e) => ScenarioOutcome::fail(NAME, format!("sdk error: {e}")),
    }
}

async fn run_forget(endpoint: SocketAddr) -> Result<ScenarioOutcome, HarnessError> {
    const NAME: &str = "encode_recall_forget";
    let h = BrainEvalHarness::connect(endpoint).await?;
    let marker = format!("forget-marker-{}", hex16(h.space_id()));
    let text = format!("{marker}: the migration ran at midnight");

    let enc = h
        .client()
        .encode(&EncodeBuilder::new(text.as_str()).build())
        .await?;

    let before = h.recall(&marker, 10).await?;
    if !any_text_contains(&before.memories, &marker) {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            "memory was not recallable immediately after ENCODE",
        ));
    }

    h.client()
        .forget(&ForgetBuilder::new(enc.memory_id).hard().build())
        .await?;

    let after = h.recall(&marker, 10).await?;
    h.close().await?;

    if any_text_contains(&after.memories, &marker) {
        return Ok(ScenarioOutcome::fail(
            NAME,
            "memory still recalled after a hard FORGET",
        ));
    }
    Ok(ScenarioOutcome::pass(
        NAME,
        "recalled before FORGET, gone after",
    ))
}

/// A write buffered in a txn is visible to reads in that txn, and to
/// everyone after commit.
pub async fn txn_read_your_writes(endpoint: SocketAddr) -> ScenarioOutcome {
    const NAME: &str = "txn_read_your_writes";
    match run_txn(endpoint).await {
        Ok(outcome) => outcome,
        Err(e) => ScenarioOutcome::fail(NAME, format!("sdk error: {e}")),
    }
}

async fn run_txn(endpoint: SocketAddr) -> Result<ScenarioOutcome, HarnessError> {
    const NAME: &str = "txn_read_your_writes";
    let h = BrainEvalHarness::connect(endpoint).await?;
    let marker = format!("txn-marker-{}", hex16(h.space_id()));
    let text = format!("{marker}: provisional note inside a transaction");

    let txn_id = new_id();
    h.client()
        .txn_begin(&TxnBeginRequest {
            txn_id,
            timeout_seconds: 30,
        })
        .await?;

    let mut enc = EncodeBuilder::new(text.as_str()).build();
    enc.txn_id = Some(txn_id);
    h.client().encode(&enc).await?;

    // Read inside the txn must see the buffered write.
    let mut in_txn = RecallBuilder::new(marker.as_str())
        .max_results(10)
        .include_text(true)
        .build();
    in_txn.txn_id = Some(txn_id);
    let in_txn_hits = h.client().recall(&in_txn).await?.memories;

    h.client().txn_commit(&TxnCommitRequest { txn_id }).await?;

    // After commit, a plain (no-txn) read must see it.
    let after = h.recall(&marker, 10).await?;
    h.close().await?;

    if !any_text_contains(&in_txn_hits, &marker) {
        return Ok(ScenarioOutcome::fail(
            NAME,
            "in-txn read did not see the txn's buffered write",
        ));
    }
    if !any_text_contains(&after.memories, &marker) {
        return Ok(ScenarioOutcome::fail(
            NAME,
            "committed write not visible to a plain read",
        ));
    }
    Ok(ScenarioOutcome::pass(
        NAME,
        "buffered write visible in-txn; visible to all after commit",
    ))
}

/// Run the core black-box scenarios against `endpoint`.
pub async fn run_core_scenarios(endpoint: SocketAddr) -> Vec<ScenarioOutcome> {
    vec![
        multi_space_isolation(endpoint).await,
        encode_recall_forget(endpoint).await,
        txn_read_your_writes(endpoint).await,
    ]
}

/// Run the operator-admin acceptance scenarios ("E7") — the audit log,
/// derived-index rebuild, and metrics-family completeness. These drive the
/// admin HTTP plane (audit / rebuild) and the `/metrics` endpoint in addition
/// to the wire path, so they read the admin secret + ports from the
/// environment (`BRAIN_EVAL_ADMIN_*`, `BRAIN_EVAL_METRICS_PORT`).
pub async fn run_admin_scenarios(endpoint: SocketAddr) -> Vec<ScenarioOutcome> {
    vec![
        audit_derivation_logged(endpoint).await,
        index_rebuild(endpoint).await,
        resumable_backfill(endpoint).await,
        metrics_families_present(endpoint).await,
    ]
}

/// Lowercase hex of a 16-byte id — a unique, collision-free marker suffix.
fn hex16(id: [u8; 16]) -> String {
    let mut s = String::with_capacity(32);
    for b in id {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex16_is_32_chars() {
        assert_eq!(hex16([0xab; 16]).len(), 32);
        assert_eq!(hex16([0x00; 16]), "0".repeat(32));
    }

    #[test]
    fn outcome_constructors() {
        assert!(ScenarioOutcome::pass("x", "ok").passed);
        assert!(!ScenarioOutcome::fail("x", "bad").passed);
    }
}
