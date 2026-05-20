//! `BenchmarkReport` — the assembled output of a single benchmark run.
//!
//! `EvalRunner::run` returns one of these and hands it to the configured
//! reporters. The shape is deliberately stable and `serde`-able so the
//! JSON sidecar is easy for CI to consume.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::core::outcome::QuestionResult;
use crate::score::metrics::EvalMetrics;

/// Full report — metadata + metrics + competitor table + per-question
/// results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// Metadata about the run itself (date, benchmark id, judge type).
    pub meta: BenchmarkMeta,
    /// Aggregated metrics.
    pub metrics: EvalMetrics,
    /// Optional comparison table of published competitor scores.
    /// Empty when we don't have public numbers for the benchmark.
    pub competitors: Vec<CompetitorRow>,
    /// One row per question — the full per-question detail for the
    /// JSON sidecar.
    pub per_question: Vec<QuestionResult>,
}

impl BenchmarkReport {
    /// Bundle the pieces into a report. No I/O.
    #[must_use]
    pub fn build(
        meta: BenchmarkMeta,
        metrics: EvalMetrics,
        competitors: Vec<CompetitorRow>,
        per_question: Vec<QuestionResult>,
    ) -> Self {
        Self {
            meta,
            metrics,
            competitors,
            per_question,
        }
    }
}

/// Run metadata. Stable shape; downstream dashboards parse this.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMeta {
    /// Stable id: `"dmr"`, `"longmemeval-s"`, etc.
    pub benchmark_id: String,
    /// Display name for reports.
    pub benchmark_name: String,
    /// Paper / dataset URL.
    pub benchmark_url: String,
    /// `"heuristic"` or `"llm"`.
    pub judge_type: String,
    /// Run start time, as Unix nanos. Reports format this for humans.
    pub run_started_unix_nanos: u128,
    /// How many instances were loaded.
    pub instance_count: usize,
    /// Brain version reporting this run (cargo pkg version).
    pub brain_version: String,
}

impl BenchmarkMeta {
    /// Build a fresh meta block. `run_started_unix_nanos` is filled
    /// from the wall clock; pass `instance_count` from the loaded
    /// dataset.
    #[must_use]
    pub fn new(
        benchmark_id: &str,
        benchmark_name: &str,
        benchmark_url: &str,
        judge_type: &str,
        instance_count: usize,
    ) -> Self {
        let run_started_unix_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        Self {
            benchmark_id: benchmark_id.to_owned(),
            benchmark_name: benchmark_name.to_owned(),
            benchmark_url: benchmark_url.to_owned(),
            judge_type: judge_type.to_owned(),
            run_started_unix_nanos,
            instance_count,
            brain_version: env!("CARGO_PKG_VERSION").to_owned(),
        }
    }
}

/// One published competitor row. Honest field: cite where the number
/// came from so anyone reading the report can verify (and we can call
/// out misleading methodology on the spot).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetitorRow {
    /// Competitor system name (e.g. `"mem0"`, `"Zep"`, `"OpenAI Memory"`).
    pub system: String,
    /// Reported accuracy on this benchmark.
    pub accuracy: f64,
    /// Source — paper, blog, repo URL.
    pub source: String,
    /// Free-text note. Use this to flag methodology issues (e.g.
    /// "excluded adversarial questions from the denominator").
    pub note: String,
}
