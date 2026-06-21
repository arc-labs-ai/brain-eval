//! Membership-recall INVARIANTS — properties that must hold for ANY input,
//! checked over freshly-constructed corpora rather than a fixed benchmark.
//! These are the anti-overfit gate: they describe how recall must *behave*,
//! not which specific questions must pass, so a change that games one dataset
//! cannot satisfy them.
//!
//! Live: needs a running brain-server (semantic-only is cleanest — no async
//! extraction mutating results between reads). `#[ignore]`d; run with:
//!
//! ```bash
//! BRAIN_EVAL_ENDPOINT=127.0.0.1:18080 \
//!   cargo test --test membership_invariants -- --ignored --nocapture
//! ```

use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::time::Duration;

use brain_eval::core::instance::TurnRecord;
use brain_eval::run::harness::BrainEvalHarness;

fn endpoint() -> SocketAddr {
    std::env::var("BRAIN_EVAL_ENDPOINT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| "127.0.0.1:18080".parse().expect("default endpoint"))
}

fn turns(lines: &[&str]) -> Vec<TurnRecord> {
    lines
        .iter()
        .map(|l| TurnRecord {
            role: "user".to_owned(),
            content: (*l).to_owned(),
        })
        .collect()
}

fn id_set(memories: &[brain_db_sdk::wire::types::MemoryResult]) -> BTreeSet<u128> {
    memories.iter().map(|m| m.memory_id).collect()
}

fn contains_token(memories: &[brain_db_sdk::wire::types::MemoryResult], token: &str) -> bool {
    let t = token.to_lowercase();
    memories
        .iter()
        .any(|m| m.text.to_lowercase().contains(&t))
}

/// Recall, retrying briefly so a just-ingested vector has time to land in the
/// HNSW (ENCODE acks on WAL; index insert may trail by a beat).
async fn recall_settled(
    h: &BrainEvalHarness,
    cue: &str,
) -> Vec<brain_db_sdk::wire::types::MemoryResult> {
    for _ in 0..10 {
        let out = h.recall(cue, 50).await.expect("recall");
        if !out.memories.is_empty() {
            return out.memories;
        }
        tokio::time::sleep(Duration::from_millis(700)).await;
    }
    Vec::new()
}

/// INVARIANT 1 — DETERMINISM: same store + same query → identical answer set.
#[tokio::test]
#[ignore]
async fn determinism_same_query_same_set() {
    let h = BrainEvalHarness::connect(endpoint()).await.expect("connect");
    h.ingest(&turns(&[
        "The Zephyr-9 reactor was commissioned in Helsinki.",
        "Maple syrup production peaked during the cold snap.",
        "The orchestra rehearsed the nocturne until midnight.",
    ]))
    .await
    .expect("ingest");
    tokio::time::sleep(Duration::from_secs(2)).await;
    let a = id_set(&recall_settled(&h, "Where was the Zephyr-9 reactor commissioned?").await);
    let b = id_set(
        &h.recall("Where was the Zephyr-9 reactor commissioned?", 50)
            .await
            .expect("recall")
            .memories,
    );
    assert_eq!(a, b, "identical query over identical store must return the identical set");
    assert!(!a.is_empty(), "the answer set must not be empty for a present fact");
}

/// INVARIANT 2 — MONOTONE RECALL: a distinctive stored fact is returned for a
/// query that asks for it, even amid unrelated distractors.
#[tokio::test]
#[ignore]
async fn monotone_recall_distinctive_fact() {
    let h = BrainEvalHarness::connect(endpoint()).await.expect("connect");
    h.ingest(&turns(&[
        "Quentin adopted a three-legged greyhound named Pascal.",
        "The quarterly budget review was postponed to next Tuesday.",
        "Rainfall in the highlands exceeded the seasonal average.",
        "The bakery introduced a sourdough with toasted walnuts.",
    ]))
    .await
    .expect("ingest");
    tokio::time::sleep(Duration::from_secs(2)).await;
    let mems = recall_settled(&h, "What pet did Quentin adopt?").await;
    assert!(
        contains_token(&mems, "greyhound") || contains_token(&mems, "Pascal"),
        "the memory stating the fact must be in the returned set; got: {:?}",
        mems.iter().map(|m| &m.text).collect::<Vec<_>>()
    );
}

/// INVARIANT 3 — IRRELEVANCE INSENSITIVITY: adding unrelated memories does not
/// drop the answer to an unrelated query. Two isolated agents: base vs
/// base+noise; the gold memory must surface in both.
#[tokio::test]
#[ignore]
async fn irrelevance_insensitivity() {
    let base = [
        "Ingrid defended her dissertation on glacier dynamics in Tromsø.",
        "The ferry schedule changes every equinox.",
    ];
    let noise = [
        "A new ramen shop opened beside the old cinema.",
        "The marathon route avoids the bridge this year.",
        "He alphabetised the spice rack on Sunday.",
        "The printer on the third floor jams on cardstock.",
    ];
    let q = "What was Ingrid's dissertation about?";

    let a = BrainEvalHarness::connect(endpoint()).await.expect("connect a");
    a.ingest(&turns(&base)).await.expect("ingest a");
    tokio::time::sleep(Duration::from_secs(2)).await;
    let ma = recall_settled(&a, q).await;

    let b = BrainEvalHarness::connect(endpoint()).await.expect("connect b");
    let mut both: Vec<&str> = base.to_vec();
    both.extend_from_slice(&noise);
    b.ingest(&turns(&both)).await.expect("ingest b");
    tokio::time::sleep(Duration::from_secs(2)).await;
    let mb = recall_settled(&b, q).await;

    let in_a = contains_token(&ma, "glacier") || contains_token(&ma, "dissertation");
    let in_b = contains_token(&mb, "glacier") || contains_token(&mb, "dissertation");
    assert!(in_a, "gold memory must surface without noise");
    assert!(in_b, "gold memory must STILL surface when unrelated memories are added");
}

/// INVARIANT 4 (diagnostic) — ABSTENTION: a query with no supporting memory
/// should not commit to a confident Single. May surface floor miscalibration.
#[tokio::test]
#[ignore]
async fn abstention_on_unsupported_query() {
    let h = BrainEvalHarness::connect(endpoint()).await.expect("connect");
    h.ingest(&turns(&[
        "The lighthouse keeper repainted the railings in June.",
        "Tidal charts for the strait were updated last week.",
        "The cargo manifest listed forty crates of citrus.",
    ]))
    .await
    .expect("ingest");
    tokio::time::sleep(Duration::from_secs(2)).await;
    let out = h
        .recall("What programming language does the database use internally?", 50)
        .await
        .expect("recall");
    // Strong form: honest abstention. Reported, not asserted hard, since the
    // absolute floor is a deliberately-low default — a failure here is a signal
    // to raise it, not necessarily a bug.
    eprintln!(
        "[abstention] kind={:?} returned={} (expect None / empty for an unsupported query)",
        out.answer_kind,
        out.memories.len()
    );
    assert!(
        !matches!(out.answer_kind, brain_db_sdk::wire::types::AnswerKindWire::Single),
        "an unsupported query must not commit to a confident Single"
    );
}
