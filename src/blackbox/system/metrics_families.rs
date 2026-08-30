//! Metrics-family completeness gate ("E7", §19/06:132 — *observability: the
//! metric families emit and their counters advance*).
//!
//! After a mixed read/write run the server's `/metrics` must expose every
//! observability family the spec names — `worker_*`, `extractor_*`,
//! `retriever_*`, `query_*` (all under Brain's `brain_` prefix) — and their
//! counters must be monotonic (a counter never goes backwards). The gate
//! scrapes `/metrics` twice around a burst of reads and asserts each family is
//! present and its representative counter did not regress; the read counter
//! must have strictly advanced, proving the read path is actually wired into
//! the metrics.

use std::net::SocketAddr;

use brain_db_sdk::EncodeBuilder;

use super::{hex16, ScenarioOutcome};
use crate::client::harness::{BrainEvalHarness, HarnessError};
use crate::client::metrics::{metrics_addr_from_env, PrometheusMetrics};

const NAME: &str = "metrics_families_present";

/// The four observability family prefixes §19/06 requires (Brain namespaces
/// every metric under `brain_`).
const FAMILY_PREFIXES: &[(&str, &str)] = &[
    ("worker", "brain_worker_"),
    ("extractor", "brain_extractor_"),
    ("retriever", "brain_retriever_"),
    ("query", "brain_query_"),
];

/// Representative counter of each family — asserted monotonic across the two
/// scrapes. These are cumulative `_total` counters, so a later scrape must be
/// `>=` an earlier one.
const MONOTONIC_COUNTERS: &[&str] = &[
    "brain_worker_cycles_total",
    "brain_extractor_tier_runs_total",
    "brain_retriever_invocations_total",
    "brain_query_total",
];

/// The read-path counter that must strictly advance across the burst of
/// recalls between the two scrapes.
const READ_COUNTER: &str = "brain_query_total";

/// Number of recalls issued between the two scrapes.
const READ_BURST: u32 = 5;

/// Scrape `/metrics` before and after a read burst; assert every family emits
/// and its counter is monotonic.
pub async fn metrics_families_present(endpoint: SocketAddr) -> ScenarioOutcome {
    match run(endpoint).await {
        Ok(o) => o,
        Err(e) => ScenarioOutcome::fail(NAME, format!("harness: {e}")),
    }
}

async fn run(endpoint: SocketAddr) -> Result<ScenarioOutcome, HarnessError> {
    let metrics_addr = metrics_addr_from_env(endpoint);

    let h = BrainEvalHarness::connect(endpoint).await?;
    let marker = format!("metrics-marker-{}", hex16(h.space_id()));

    // Seed a couple of memories with proper nouns so extraction/retriever
    // families have activity, then a warm-up recall so read families exist.
    for s in [
        format!("{marker}: Ada Lovelace wrote the first algorithm."),
        format!("{marker}: the summit is in Geneva this spring."),
    ] {
        h.client()
            .encode(&EncodeBuilder::new(s.as_str()).build())
            .await?;
    }
    let _ = h.recall(&marker, 10).await?;

    // First scrape.
    let before = match PrometheusMetrics::scrape(metrics_addr).await {
        Ok(m) => m,
        Err(e) => {
            h.close().await?;
            return Ok(ScenarioOutcome::fail(
                NAME,
                format!("baseline /metrics scrape ({metrics_addr}) failed: {e}"),
            ));
        }
    };

    // A burst of reads to advance the read counter.
    for _ in 0..READ_BURST {
        let _ = h.recall(&marker, 10).await?;
    }

    // Second scrape.
    let after = match PrometheusMetrics::scrape(metrics_addr).await {
        Ok(m) => m,
        Err(e) => {
            h.close().await?;
            return Ok(ScenarioOutcome::fail(
                NAME,
                format!("post-burst /metrics scrape ({metrics_addr}) failed: {e}"),
            ));
        }
    };
    h.close().await?;

    // Every family prefix must be present.
    let mut missing = Vec::new();
    for (label, prefix) in FAMILY_PREFIXES {
        if !after.has_prefix(prefix) {
            missing.push(*label);
        }
    }
    if !missing.is_empty() {
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!("missing metric families: {missing:?} (prefixes absent from /metrics)"),
        ));
    }

    // Counters monotonic (never regress) across the two scrapes.
    for counter in MONOTONIC_COUNTERS {
        let b = before.sum(counter).unwrap_or(0.0);
        let a = after.sum(counter).unwrap_or(0.0);
        if a < b {
            return Ok(ScenarioOutcome::fail(
                NAME,
                format!("counter {counter} regressed: {a} < {b} (not monotonic)"),
            ));
        }
    }

    // The read counter must have strictly advanced over the burst.
    let read_before = before.sum(READ_COUNTER).unwrap_or(0.0);
    let read_after = after.sum(READ_COUNTER).unwrap_or(0.0);
    if read_after <= read_before {
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "{READ_COUNTER} did not advance over {READ_BURST} recalls \
                 ({read_before} → {read_after}) — read path not wired to metrics"
            ),
        ));
    }

    Ok(ScenarioOutcome::pass(
        NAME,
        format!(
            "§19/06 metrics: worker/extractor/retriever/query families present; \
             counters monotonic; {READ_COUNTER} advanced {read_before} → {read_after} \
             over {READ_BURST} recalls"
        ),
    ))
}
