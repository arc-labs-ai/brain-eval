//! The write-drain barrier: block until Brain's asynchronous write pipeline
//! has settled, so a read sees a fully built graph.
//!
//! This lives in [`crate::client`] — the shared adapter layer — because every
//! surface that writes then reads needs the same barrier: the per-question
//! benchmark runner, the black-box suites, and the live integration tests.
//! Keeping it here is also what keeps the layering honest: `client` is the
//! base layer, so the barrier cannot reach up into `bench::report` for its
//! own result type, and [`DrainBarrierStats`] is defined here and re-exported
//! by the report module that serialises it.

use std::net::SocketAddr;

use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::client::metrics::PrometheusMetrics;

/// The drain-barrier plateau — what the async write pipeline had produced
/// when reads were allowed to start.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DrainBarrierStats {
    /// `brain_extractor_items_written_total` at the plateau (all item kinds).
    pub extractor_items_written: u64,
    /// `brain_statement_embed_rows_embedded_total` at the plateau.
    pub statement_rows_embedded: u64,
    /// `true` if the plateau was reached; `false` if the wait hit its cap
    /// (or never scraped) and proceeded anyway.
    pub flushed: bool,
}

/// Block until the server's async write pipeline has fully flushed, so a
/// conversation's entity / statement / relation graph — and its
/// embeddings — are reflected in the read indexes before we query it.
///
/// A write is two-phase: the synchronous memory record (WAL-acked at
/// ENCODE) and the asynchronous tail (extraction of entities /
/// statements / relations, then statement embedding). Grounded recall
/// depends on that async tail, so reads must not fire until it settles.
/// We poll the COMBINED progress of both async stages —
/// `brain_extractor_items_written_total` + `brain_statement_embed_rows_embedded_total`
/// — and treat a plateau (no increase across a stable window, after at
/// least one write has landed) as flushed. Watching only extraction
/// would fire reads while statement embeddings are still being computed.
/// Best-effort: failed polls keep retrying; the cap just proceeds.
/// Workers whose queue must be empty before a read sees a settled graph.
///
/// The write pipeline is: ENCODE acks, then the extractor builds the typed
/// graph, then statement embedding makes it matchable, then the derived-edge
/// workers link it. Anything else on the scheduler (decay, GC, snapshots) is
/// background housekeeping that never gates a read, so waiting on it would
/// only add dead time.
pub const DRAIN_WORKERS: [&str; 5] = [
    "extractor",
    "statement_embed",
    "auto_edge",
    "temporal_edge",
    "causal_edge",
];

/// Poll cadence. The barrier now reads a queue depth rather than guessing
/// from a counter plateau, so it can afford to look often and finish the
/// moment the queues are empty.
pub const DRAIN_INTERVAL: std::time::Duration = std::time::Duration::from_millis(250);

/// Consecutive all-empty observations required before declaring the pipeline
/// settled.
///
/// One zero is not enough: there is a window between ENCODE acking and the
/// extractor dequeuing where every queue is legitimately empty and the work
/// has not started. Four polls (1 s) spans that handoff while costing a
/// fraction of the old 60 s plateau.
pub const DRAIN_STABLE_POLLS: u32 = 4;

/// Floor before any verdict, so the workers have demonstrably picked up.
pub const DRAIN_MIN_WAIT: std::time::Duration = std::time::Duration::from_secs(1);

/// Hard cap. Reaching it is reported as `flushed: false` so a report can
/// never silently claim a settled read it did not get.
pub const DRAIN_MAX_WAIT: std::time::Duration = std::time::Duration::from_secs(600);

/// Total queued work across the write-path workers, or `None` when the
/// server publishes no `brain_worker_pending_work` gauge at all.
///
/// The `None` case is the one that matters for safety. An absent family and
/// an empty queue are both "no samples", and conflating them would make the
/// barrier declare a settling pipeline flushed — every read would then race
/// the extractor and the benchmark would silently measure a half-built
/// graph. So absence is reported as absence, and the caller falls back to
/// the slower counter plateau rather than guessing.
#[must_use]
pub fn pending_write_work(m: &PrometheusMetrics) -> Option<f64> {
    let mut total: Option<f64> = None;
    for w in DRAIN_WORKERS {
        if let Some(v) = m.sum_where("brain_worker_pending_work", w) {
            *total.get_or_insert(0.0) += v;
        }
    }
    total
}

/// Block until Brain's async write pipeline has settled.
///
/// Reads `brain_worker_pending_work` — the scheduler's own queue depth —
/// and returns once every write-path worker reports an empty queue for
/// [`DRAIN_STABLE_POLLS`] consecutive polls.
///
/// This used to infer settlement from `brain_extractor_items_written_total`
/// going unchanged for 60 s, polled every 5 s, with a 30 s floor. That is a
/// guess ("nothing has happened lately, so probably nothing will"), and it
/// is an expensive one: a 12-question smoke run spent 139 s of its 142 s
/// wall clock inside this function for ~3 s of actual work. The queue depth
/// answers the same question exactly, so the barrier now costs about a
/// second.
///
/// The plateau fallback is kept for a server that does not publish the
/// gauge: `None` from `sum_where` means the family is absent, not that the
/// queue is empty, and treating absence as empty would silently drop the
/// barrier and make every read race the extractor.
pub async fn wait_for_write_flush(metrics_addr: SocketAddr, conv: &str) -> DrainBarrierStats {
    let started = std::time::Instant::now();
    let mut stable: u32 = 0;
    let mut last_ex: u64 = 0;
    let mut last_em: u64 = 0;
    // Plateau-fallback state, used only when the gauge is unavailable.
    let mut plateau_last: i64 = -1;
    let mut plateau_stable: u32 = 0;
    let mut gauge_seen = false;

    while started.elapsed() < DRAIN_MAX_WAIT {
        // A scrape failure is transient: skip the poll without resetting
        // progress, exactly as the previous implementation did.
        let Some(m) = PrometheusMetrics::scrape(metrics_addr).await.ok() else {
            tokio::time::sleep(DRAIN_INTERVAL).await;
            continue;
        };

        let ex = m.sum("brain_extractor_items_written_total").unwrap_or(0.0) as u64;
        let em = m
            .sum("brain_statement_embed_rows_embedded_total")
            .unwrap_or(0.0) as u64;
        last_ex = ex;
        last_em = em;

        let pending = pending_write_work(&m);

        if let Some(pending_total) = pending {
            gauge_seen = true;
            if pending_total == 0.0 {
                stable += 1;
            } else {
                stable = 0;
            }
            if started.elapsed() >= DRAIN_MIN_WAIT && stable >= DRAIN_STABLE_POLLS {
                tracing::info!(
                    conversation = %conv,
                    extracted = ex,
                    statement_rows_embedded = em,
                    waited_ms = started.elapsed().as_millis() as u64,
                    "async write pipeline flushed (queue depth)"
                );
                return DrainBarrierStats {
                    extractor_items_written: ex,
                    statement_rows_embedded: em,
                    flushed: true,
                };
            }
        } else if !gauge_seen {
            // `pending` was `None`: the gauge is absent, not zero.
            // Legacy server: fall back to the counter plateau. Slower, but
            // correct, and it is the only signal such a server offers.
            let cur = ex as i64 + em as i64;
            if cur == plateau_last {
                plateau_stable += 1;
            } else {
                plateau_stable = 0;
                plateau_last = cur;
            }
            // 60 s of no movement, at this interval.
            let needed = (60.0 / DRAIN_INTERVAL.as_secs_f64()) as u32;
            if cur > 0 && plateau_stable >= needed {
                tracing::info!(
                    conversation = %conv,
                    extracted = ex,
                    statement_rows_embedded = em,
                    "async write pipeline flushed (counter plateau fallback)"
                );
                return DrainBarrierStats {
                    extractor_items_written: ex,
                    statement_rows_embedded: em,
                    flushed: true,
                };
            }
        }

        tokio::time::sleep(DRAIN_INTERVAL).await;
    }

    warn!(
        conversation = %conv,
        waited_s = started.elapsed().as_secs(),
        "write-flush wait hit max; proceeding"
    );
    DrainBarrierStats {
        extractor_items_written: last_ex,
        statement_rows_embedded: last_em,
        flushed: false,
    }
}

#[cfg(test)]
mod drain_barrier_tests {
    use super::{pending_write_work, DRAIN_WORKERS};
    use crate::client::metrics::PrometheusMetrics;

    fn metrics(body: &str) -> PrometheusMetrics {
        PrometheusMetrics::parse(body)
    }

    #[test]
    fn an_idle_pipeline_reports_zero_queued_work() {
        let text = DRAIN_WORKERS
            .iter()
            .map(|w| format!("brain_worker_pending_work{{shard=\"0\",worker=\"{w}\"}} 0\n"))
            .collect::<String>();
        assert_eq!(pending_write_work(&metrics(&text)), Some(0.0));
    }

    #[test]
    fn queued_work_on_any_write_worker_is_counted() {
        for w in DRAIN_WORKERS {
            let text = format!("brain_worker_pending_work{{shard=\"0\",worker=\"{w}\"}} 7\n");
            assert_eq!(
                pending_write_work(&metrics(&text)),
                Some(7.0),
                "{w} must gate the barrier"
            );
        }
    }

    #[test]
    fn an_absent_gauge_is_none_not_zero() {
        // The safety property. If absence collapsed to `Some(0.0)` the
        // barrier would declare a settling pipeline flushed and every read
        // in the benchmark would race the extractor — the measurement would
        // silently be of a half-built graph.
        let text = "brain_extractor_items_written_total{shard=\"0\"} 58\n";
        assert_eq!(pending_write_work(&metrics(text)), None);
    }

    #[test]
    fn housekeeping_workers_do_not_gate_the_barrier() {
        // decay / GC / snapshots run forever on their own cadence; waiting
        // for them would mean waiting for a queue that is never reliably
        // empty, which is how a barrier turns into a timeout.
        let text = "brain_worker_pending_work{shard=\"0\",worker=\"decay\"} 500\n\
                    brain_worker_pending_work{shard=\"0\",worker=\"snapshot\"} 12\n";
        assert_eq!(
            pending_write_work(&metrics(text)),
            None,
            "only write-path workers may gate a read"
        );
    }

    #[test]
    fn queued_work_sums_across_shards() {
        let text = "brain_worker_pending_work{shard=\"0\",worker=\"extractor\"} 3\n\
                    brain_worker_pending_work{shard=\"1\",worker=\"extractor\"} 4\n";
        assert_eq!(pending_write_work(&metrics(text)), Some(7.0));
    }
}
