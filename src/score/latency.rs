//! Latency aggregation — `LatencyStats` and the p50/p95/mean
//! computer that produces it from `QuestionResult` records.

use serde::{Deserialize, Serialize};

use crate::core::outcome::QuestionResult;

/// Latency percentiles in milliseconds.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LatencyStats {
    /// Write-pipeline (per-question ingestion total) p50.
    pub write_p50_ms: u64,
    /// Write-pipeline p95.
    pub write_p95_ms: u64,
    /// Write-pipeline mean.
    pub write_mean_ms: u64,
    /// Read-pipeline (per-question RECALL) p50.
    pub read_p50_ms: u64,
    /// Read-pipeline p95.
    pub read_p95_ms: u64,
    /// Read-pipeline mean.
    pub read_mean_ms: u64,
}

/// Compute the LatencyStats over a full run.
#[must_use]
pub fn compute_latency_stats(results: &[QuestionResult]) -> LatencyStats {
    if results.is_empty() {
        return LatencyStats::default();
    }
    let n = results.len();
    let mut write_ms: Vec<u64> = results.iter().map(|r| r.write_latency_ms).collect();
    let mut read_ms: Vec<u64> = results.iter().map(|r| r.read_latency_ms).collect();
    write_ms.sort_unstable();
    read_ms.sort_unstable();
    let p50 = (n / 2).min(n - 1);
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let p95 = (((n as f64) * 0.95) as usize).min(n - 1);
    LatencyStats {
        write_p50_ms: write_ms[p50],
        write_p95_ms: write_ms[p95],
        write_mean_ms: write_ms.iter().sum::<u64>() / n as u64,
        read_p50_ms: read_ms[p50],
        read_p95_ms: read_ms[p95],
        read_mean_ms: read_ms.iter().sum::<u64>() / n as u64,
    }
}
