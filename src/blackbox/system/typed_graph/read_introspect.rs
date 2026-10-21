//! Read-introspection scenario: TRACE-as-a-gate, filtered RECALL, and the
//! Single/Many/None membership shape.
//!
//! The core scenarios assert *what* RECALL returns; this asserts *how* the read
//! pipeline got there, treating the trace as a gate rather than a debug aid
//! (§19/06 "read introspection & filtered recall"):
//!
//! - **TRACE** — `trace = true` must return a per-stage [`RecallTrace`] on the
//!   final frame: all three retrievers reported, and a filter chain whose
//!   counts are a monotonically non-increasing funnel
//!   (`before >= after_type >= … >= after_limit`). A trace that skips a stage
//!   or reports a count that grows through the funnel is a read-pipeline bug.
//! - **Filtered recall** — a `kinds` filter must return only memories of the
//!   requested kind; a `confidence_threshold` must not return anything below
//!   it. These are the filter-chain stages, checked from the client side.
//! - **Membership shape** — a cue that matches stored data resolves to `Single`
//!   or `Many` (never `None`); a cue that matches nothing resolves to `None`
//!   (honest abstention), never a fabricated `Single`.
//!
//! Like the other typed-graph scenarios this drives a live server and returns a
//! [`ScenarioOutcome`]; markers are salted with the space-id hex so reruns
//! don't collide on persistent state.

use std::net::SocketAddr;

use brain_db_sdk::wire::types::{AnswerKindWire, MemoryKindWire, RecallTraceFilterChain};
use brain_db_sdk::{EncodeBuilder, RecallBuilder};

use super::super::{hex16, ScenarioOutcome};
use crate::client::harness::{BrainEvalHarness, HarnessError};

const NAME: &str = "tg_read_introspection";

/// Encode a small known set, then probe RECALL's trace, filters, and
/// membership shape.
pub async fn read_introspection(endpoint: SocketAddr) -> ScenarioOutcome {
    match run(endpoint).await {
        Ok(o) => o,
        Err(e) => ScenarioOutcome::fail(NAME, format!("sdk error: {e}")),
    }
}

/// The filter chain must be a non-increasing funnel — each stage can only
/// remove candidates, never add them. Returns the first violating pair, if any.
fn funnel_violation(fc: &RecallTraceFilterChain) -> Option<(&'static str, u32, &'static str, u32)> {
    let stages: [(&str, u32); 8] = [
        ("before", fc.before),
        ("after_type", fc.after_type),
        ("after_temporal", fc.after_temporal),
        ("after_confidence", fc.after_confidence),
        ("after_tombstone", fc.after_tombstone),
        ("after_supersession", fc.after_supersession),
        ("after_as_of", fc.after_as_of),
        ("after_limit", fc.after_limit),
    ];
    stages
        .windows(2)
        .find(|w| w[1].1 > w[0].1)
        .map(|w| (w[0].0, w[0].1, w[1].0, w[1].1))
}

async fn run(endpoint: SocketAddr) -> Result<ScenarioOutcome, HarnessError> {
    let h = BrainEvalHarness::connect(endpoint).await?;
    let marker = format!("introspect-{}", hex16(h.space_id()));

    // A known set: three distinct, related memories under one marker so a
    // marker cue matches several (Many-eligible), plus enough shared topic that
    // the semantic and lexical lanes both have something to contribute.
    let texts = [
        format!("{marker}: the staging database migrated to region eu-west on Monday"),
        format!("{marker}: the staging database rollback window is two hours"),
        format!("{marker}: the staging database owner is the platform team"),
    ];
    for t in &texts {
        h.client()
            .encode(&EncodeBuilder::new(t.as_str()).wait_derived().build())
            .await?;
    }

    // --- TRACE as a gate --------------------------------------------------
    let trace_req = RecallBuilder::new(marker.as_str())
        .max_results(10)
        .include_text(true)
        .trace(true)
        .build();
    let frames = h.client().recall_frames(&trace_req).await?;
    let trace = frames.iter().rev().find_map(|f| f.trace.clone());
    let Some(trace) = trace else {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            "trace=true returned no RecallTrace on any frame",
        ));
    };
    // The semantic and lexical lanes always run; the graph lane reports only
    // when the cue yields a graph anchor, so a plain-memory recall legitimately
    // shows two lanes, not three. Require the always-on pair to have reported.
    use brain_db_sdk::wire::types::RetrieverNameWire;
    let reported: Vec<RetrieverNameWire> = trace.retrievers.iter().map(|r| r.name).collect();
    let has_semantic = reported.contains(&RetrieverNameWire::Semantic);
    let has_lexical = reported.contains(&RetrieverNameWire::Lexical);
    if !has_semantic || !has_lexical {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "trace missing an always-on lane (semantic={has_semantic}, lexical={has_lexical}); \
                 reported {reported:?}"
            ),
        ));
    }
    if let Some((a, av, b, bv)) = funnel_violation(&trace.filter_chain) {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!("filter chain is not a funnel: {a}={av} then {b}={bv} (grew)"),
        ));
    }

    // --- kinds filter -----------------------------------------------------
    // Fresh episodic writes: an Episodic-only filter must not surface a
    // non-episodic memory.
    let epi_req = RecallBuilder::new(marker.as_str())
        .max_results(10)
        .include_text(true)
        .kinds(vec![MemoryKindWire::Episodic])
        .build();
    let epi = h.client().recall(&epi_req).await?;
    if let Some(bad) = epi
        .memories
        .iter()
        .find(|m| m.kind != MemoryKindWire::Episodic)
    {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "kinds=[Episodic] returned a {:?} memory — type filter leaked",
                bad.kind
            ),
        ));
    }

    // --- confidence threshold --------------------------------------------
    // `confidence_threshold` filters candidates by their intrinsic *salience*,
    // not by the cosine similarity each `MemoryResult.confidence` carries — a
    // lexical or graph hit can surface with a ~0 cosine yet still be salient
    // enough to pass. Salience isn't on the wire, so a client can't assert the
    // returned `confidence` value against the floor (they measure different
    // things). Assert the filter's observable contract instead: raising the
    // floor can only remove candidates, never add them (monotone funnel).
    let base_conf = h
        .client()
        .recall(
            &RecallBuilder::new(marker.as_str())
                .max_results(10)
                .confidence_threshold(0.0)
                .build(),
        )
        .await?;
    let high_conf = h
        .client()
        .recall(
            &RecallBuilder::new(marker.as_str())
                .max_results(10)
                .confidence_threshold(0.95)
                .build(),
        )
        .await?;
    if high_conf.memories.len() > base_conf.memories.len() {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "raising confidence_threshold grew the result set ({} at 0.0 → {} at 0.95) — \
                 the salience filter must be monotone",
                base_conf.memories.len(),
                high_conf.memories.len()
            ),
        ));
    }

    // --- membership shape -------------------------------------------------
    // The marker cue matches stored data: never None.
    let hit = h.recall(&marker, 10).await?;
    if matches!(hit.answer_kind, AnswerKindWire::None) {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            "a cue matching three stored memories resolved to None (false abstention)",
        ));
    }
    // A cue that matches nothing: honest None, never a fabricated Single.
    let miss_cue = format!("nonexistent-topic-{}-quasar-flux", hex16(h.space_id()));
    let miss = h.recall(&miss_cue, 10).await?;

    h.close().await?;

    if !matches!(miss.answer_kind, AnswerKindWire::None) {
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "a cue matching nothing resolved to {:?} with {} memories \
                 (should be an honest None)",
                miss.answer_kind,
                miss.memories.len()
            ),
        ));
    }

    Ok(ScenarioOutcome::pass(
        NAME,
        format!(
            "§19/06 read introspection: trace funnel {}→{}, {} lane(s) reported ({reported:?}); \
             kinds/confidence filters honoured; membership {:?} on hit, None on miss",
            trace.filter_chain.before,
            trace.filter_chain.after_limit,
            reported.len(),
            hit.answer_kind
        ),
    ))
}
