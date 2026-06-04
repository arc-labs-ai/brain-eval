//! Warm-shard performance targets the substrate commits to.
//!
//! These are the numbers a single warm shard must hold on reference
//! hardware (16 cores / 64 GB / NVMe / Linux) on the default text-input
//! path (CPU embedding). They're the gate the acceptance scale-run checks
//! against. Two notes on the values used here:
//!
//! - Latency assumes the CPU-embedding ENCODE path (the GPU and
//!   pre-supplied-vector paths have lower targets, measured separately).
//! - RECALL latency is the fused-retrieval read path. The eval harness
//!   runs the reranker disabled (embed-only deploy), so this is the
//!   RRF-only read budget — which is what the read target is set against.

/// A latency budget for one verb, in milliseconds.
#[derive(Debug, Clone, Copy)]
pub struct VerbTarget {
    pub p50_ms: f64,
    pub p99_ms: f64,
}

/// The full target set a scale run checks against.
#[derive(Debug, Clone, Copy)]
pub struct Targets {
    /// ENCODE (text, CPU embedding) latency budget.
    pub encode: VerbTarget,
    /// RECALL (fused retrieval) latency budget.
    pub recall: VerbTarget,
    /// Sustained ENCODE throughput floor, ops/second per shard.
    pub encode_ops_per_sec: f64,
    /// Sustained RECALL/QUERY throughput floor, ops/second per shard.
    pub recall_ops_per_sec: f64,
}

impl Default for Targets {
    fn default() -> Self {
        Self {
            // ENCODE text/CPU: p50 ≤ 12 ms, p99 ≤ 25 ms.
            encode: VerbTarget {
                p50_ms: 12.0,
                p99_ms: 25.0,
            },
            // RECALL retrieval: p50 ≤ 10 ms, p99 ≤ 50 ms.
            recall: VerbTarget {
                p50_ms: 10.0,
                p99_ms: 50.0,
            },
            // ENCODE ≥ 100/s, QUERY ≥ 1000/s per shard.
            encode_ops_per_sec: 100.0,
            recall_ops_per_sec: 1_000.0,
        }
    }
}
