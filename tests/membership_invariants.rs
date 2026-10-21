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
use std::time::Duration;

use brain_eval::bench::core::instance::TurnRecord;
use brain_eval::client::harness::BrainEvalHarness;
use brain_eval::client::server::{DockerServerOpts, ServerHandle};

/// Boot a throwaway server for one test — or attach to the operator's, when
/// `BRAIN_EVAL_ENDPOINT` is set.
///
/// These four used to call `BrainEvalHarness::connect(endpoint())`, which
/// assumes a server is already listening on `127.0.0.1:18080` and mints
/// against `BRAIN_EVAL_ADMIN_SECRET`. Nothing in CI provides either, so all
/// four failed in under 20ms with `Mint(NoSecret)` — before asserting
/// anything about membership. Every other live test boots its own container,
/// which is the documented contract for the suite: "Every `#[ignore]` live
/// test boots its own full-stack container (distinct ports / names /
/// volumes) and cleans up after itself."
///
/// `from_env_or_docker` keeps the operator's escape hatch: set
/// `BRAIN_EVAL_ENDPOINT` and it attaches instead of booting.
async fn server_for(
    name: &str,
    data_port: u16,
    metrics_port: u16,
    admin_port: u16,
) -> ServerHandle {
    let tag = std::env::var("BRAIN_EVAL_IMAGE_TAG").unwrap_or_else(|_| "latest".to_string());
    ServerHandle::from_env_or_docker(DockerServerOpts {
        image_tag: tag,
        container_name: format!("brain-eval-membership-{name}"),
        data_port,
        metrics_port,
        // Distinct admin port too. `DockerServerOpts::default()` pins 19092,
        // so leaving it defaulted collides with any other live test whose
        // container is still up — the failure is a docker bind error, not
        // anything about the invariant under test.
        admin_port,
        health_timeout: Duration::from_secs(180),
        ..DockerServerOpts::default()
    })
    .await
    .expect("server should be reachable (external) or boot and become healthy (docker)")
}

/// A harness bound to a fresh space on `server`.
async fn harness_on(server: &ServerHandle) -> BrainEvalHarness {
    BrainEvalHarness::connect_minted(
        &server.key_minter(),
        server.endpoint(),
        brain_db_sdk::new_id(),
    )
    .await
    .expect("connect to booted server")
}

fn turns(lines: &[&str]) -> Vec<TurnRecord> {
    lines
        .iter()
        .map(|l| TurnRecord {
            role: "user".to_owned(),
            content: (*l).to_owned(),
            occurred_at_unix_nanos: None,
        })
        .collect()
}

fn id_set(memories: &[brain_db_sdk::wire::types::MemoryResult]) -> BTreeSet<u128> {
    memories.iter().map(|m| m.memory_id).collect()
}

fn contains_token(memories: &[brain_db_sdk::wire::types::MemoryResult], token: &str) -> bool {
    let t = token.to_lowercase();
    memories.iter().any(|m| m.text.to_lowercase().contains(&t))
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
    let server = server_for("determinism", 38090, 39100, 39110).await;
    let h = harness_on(&server).await;
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
    assert_eq!(
        a, b,
        "identical query over identical store must return the identical set"
    );
    assert!(
        !a.is_empty(),
        "the answer set must not be empty for a present fact"
    );
}

/// INVARIANT 2 — MONOTONE RECALL: a distinctive stored fact is returned for a
/// query that asks for it, even amid unrelated distractors.
#[tokio::test]
#[ignore]
async fn monotone_recall_distinctive_fact() {
    let server = server_for("monotone", 38091, 39101, 39111).await;
    let h = harness_on(&server).await;
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
/// drop the answer to an unrelated query. Two isolated spaces: base vs
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

    let server = server_for("irrelevance", 38092, 39102, 39112).await;
    let a = harness_on(&server).await;
    a.ingest(&turns(&base)).await.expect("ingest a");
    tokio::time::sleep(Duration::from_secs(2)).await;
    let ma = recall_settled(&a, q).await;

    // A second space on the SAME server: distractors must be isolated by
    // space, not by server, or the test would prove nothing about recall.
    let b = harness_on(&server).await;
    let mut both: Vec<&str> = base.to_vec();
    both.extend_from_slice(&noise);
    b.ingest(&turns(&both)).await.expect("ingest b");
    tokio::time::sleep(Duration::from_secs(2)).await;
    let mb = recall_settled(&b, q).await;

    let in_a = contains_token(&ma, "glacier") || contains_token(&ma, "dissertation");
    let in_b = contains_token(&mb, "glacier") || contains_token(&mb, "dissertation");
    assert!(in_a, "gold memory must surface without noise");
    assert!(
        in_b,
        "gold memory must STILL surface when unrelated memories are added"
    );
}

/// INVARIANT 4 (diagnostic) — ABSTENTION: a query with no supporting memory
/// should not commit to a confident Single. May surface floor miscalibration.
#[tokio::test]
#[ignore]
async fn abstention_on_unsupported_query() {
    let server = server_for("abstention", 38093, 39103, 39113).await;
    let h = harness_on(&server).await;
    h.ingest(&turns(&[
        "The lighthouse keeper repainted the railings in June.",
        "Tidal charts for the strait were updated last week.",
        "The cargo manifest listed forty crates of citrus.",
    ]))
    .await
    .expect("ingest");
    tokio::time::sleep(Duration::from_secs(2)).await;
    let out = h
        .recall(
            "What programming language does the database use internally?",
            50,
        )
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
        !matches!(
            out.answer_kind,
            brain_db_sdk::wire::types::AnswerKindWire::Single
        ),
        "an unsupported query must not commit to a confident Single"
    );
}
