//! Performance & scale pillar.
//!
//! Drives a load through a running `brain-server` over the SDK and
//! measures the per-verb latency distribution and sustained throughput,
//! checking each against the warm-shard targets the substrate commits to.
//!
//! Pipeline:
//! 1. [`LoadGenerator`] ingests N synthetic, structured memories.
//! 2. Latency probes time M individual ENCODE / RECALL ops → p50/p99.
//! 3. Throughput probes measure sustained ops/s over a window.
//! 4. [`ScaleReport`] collects everything with a pass/fail per metric.
//!
//! The absolute numbers are only meaningful on quiet reference hardware;
//! under an emulated dev container they're a smoke signal (the harness,
//! the math, and the report shape are what's exercised). The acceptance
//! orchestrator decides whether a threshold miss is fatal.

use std::time::Instant;

use brain_db_sdk::{BrainClient, EncodeBuilder, RecallBuilder};

use crate::run::harness::HarnessError;

pub mod recall;
mod targets;
pub use recall::{no_regression, run_recall_quality, RecallQualityReport, RecallTargets};
pub use targets::{Targets, VerbTarget};

/// What to run.
#[derive(Debug, Clone)]
pub struct ScaleConfig {
    /// Number of memories to ingest before probing.
    pub ingest_n: usize,
    /// Number of individual ops to time per latency probe.
    pub probe_n: usize,
    /// `top_k` for RECALL probes.
    pub top_k: u32,
}

impl Default for ScaleConfig {
    fn default() -> Self {
        Self {
            ingest_n: 1_000,
            probe_n: 200,
            top_k: 10,
        }
    }
}

/// Latency outcome for one verb, in milliseconds, against its target.
#[derive(Debug, Clone)]
pub struct LatencyResult {
    /// The verb measured (`encode` / `recall`).
    pub verb: &'static str,
    /// Number of timed samples.
    pub samples: usize,
    /// Measured median latency.
    pub p50_ms: f64,
    /// Measured 99th-percentile latency.
    pub p99_ms: f64,
    /// Target median budget.
    pub target_p50_ms: f64,
    /// Target 99th-percentile budget.
    pub target_p99_ms: f64,
}

impl LatencyResult {
    /// True iff both p50 and p99 met their targets.
    #[must_use]
    pub fn pass(&self) -> bool {
        self.p50_ms <= self.target_p50_ms && self.p99_ms <= self.target_p99_ms
    }
}

/// Sustained-throughput outcome for one verb, in ops/second.
#[derive(Debug, Clone)]
pub struct ThroughputResult {
    /// The verb measured (`encode` / `recall`).
    pub verb: &'static str,
    /// Number of ops in the measured window.
    pub ops: usize,
    /// Measured sustained ops/second.
    pub ops_per_sec: f64,
    /// Target ops/second floor.
    pub target_ops_per_sec: f64,
}

impl ThroughputResult {
    /// True iff measured throughput met or exceeded the floor.
    #[must_use]
    pub fn pass(&self) -> bool {
        self.ops_per_sec >= self.target_ops_per_sec
    }
}

/// The full result of a scale run.
#[derive(Debug, Clone)]
pub struct ScaleReport {
    /// Memories successfully ingested before probing.
    pub ingested: usize,
    /// Per-verb latency results.
    pub latency: Vec<LatencyResult>,
    /// Per-verb throughput results.
    pub throughput: Vec<ThroughputResult>,
}

impl ScaleReport {
    /// True iff every latency + throughput metric met its target.
    #[must_use]
    pub fn all_pass(&self) -> bool {
        self.latency.iter().all(LatencyResult::pass)
            && self.throughput.iter().all(ThroughputResult::pass)
    }

    /// Human-readable summary.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "=== scale report — {} memories ingested ===\n",
            self.ingested
        ));
        s.push_str("latency (ms):\n");
        for l in &self.latency {
            s.push_str(&format!(
                "  {:<8} p50 {:>8.3} (≤ {:>6.1})  p99 {:>8.3} (≤ {:>6.1})  [{}]  n={}\n",
                l.verb,
                l.p50_ms,
                l.target_p50_ms,
                l.p99_ms,
                l.target_p99_ms,
                verdict(l.pass()),
                l.samples,
            ));
        }
        s.push_str("throughput (ops/s):\n");
        for t in &self.throughput {
            s.push_str(&format!(
                "  {:<8} {:>10.1} (≥ {:>7.1})  [{}]  ops={}\n",
                t.verb,
                t.ops_per_sec,
                t.target_ops_per_sec,
                verdict(t.pass()),
                t.ops,
            ));
        }
        s.push_str(&format!("overall: {}\n", verdict(self.all_pass())));
        s
    }
}

fn verdict(pass: bool) -> &'static str {
    if pass {
        "PASS"
    } else {
        "FAIL"
    }
}

/// Generate N deterministic, structured memory strings. Structured (not
/// random) so recall has real signal and runs are reproducible.
pub struct LoadGenerator;

impl LoadGenerator {
    /// The i-th synthetic memory.
    #[must_use]
    pub fn memory(i: usize) -> String {
        const TOPICS: &[&str] = &[
            "billing", "search", "auth", "ingest", "planner", "indexer", "embedder", "gateway",
        ];
        const FEATURES: &[&str] = &[
            "rate limiting",
            "fuzzy matching",
            "token rotation",
            "batch upserts",
            "cost caps",
            "warm caches",
        ];
        const MONTHS: &[&str] = &[
            "January", "March", "June", "September", "November", "December",
        ];
        let topic = TOPICS[i % TOPICS.len()];
        let feature = FEATURES[(i / TOPICS.len()) % FEATURES.len()];
        let month = MONTHS[(i / (TOPICS.len() * FEATURES.len())) % MONTHS.len()];
        format!("Fact {i}: the {topic} team shipped {feature} in {month}.")
    }

    /// A cue that should match the i-th memory.
    #[must_use]
    pub fn cue(i: usize) -> String {
        const TOPICS: &[&str] = &[
            "billing", "search", "auth", "ingest", "planner", "indexer", "embedder", "gateway",
        ];
        const FEATURES: &[&str] = &[
            "rate limiting",
            "fuzzy matching",
            "token rotation",
            "batch upserts",
            "cost caps",
            "warm caches",
        ];
        let topic = TOPICS[i % TOPICS.len()];
        let feature = FEATURES[(i / TOPICS.len()) % FEATURES.len()];
        format!("what did the {topic} team ship about {feature}")
    }
}

/// Ingest `cfg.ingest_n` memories, then probe ENCODE + RECALL latency and
/// throughput against `targets`. The client must already be connected.
pub async fn run_scale(
    client: &BrainClient,
    cfg: &ScaleConfig,
    targets: &Targets,
) -> Result<ScaleReport, HarnessError> {
    // --- ingest -------------------------------------------------------
    let mut ingested = 0usize;
    for i in 0..cfg.ingest_n {
        let req = EncodeBuilder::new(LoadGenerator::memory(i).as_str())
            .deduplicate(false)
            .build();
        client.encode(&req).await?;
        ingested += 1;
    }

    // --- ENCODE latency ----------------------------------------------
    let mut encode_ms = Vec::with_capacity(cfg.probe_n);
    for i in 0..cfg.probe_n {
        // Fresh content per probe so we time a real write, not a dedup hit.
        let text = format!("probe encode {i}: {}", LoadGenerator::memory(cfg.ingest_n + i));
        let req = EncodeBuilder::new(text.as_str()).deduplicate(false).build();
        let start = Instant::now();
        client.encode(&req).await?;
        encode_ms.push(ms(start));
    }

    // --- RECALL latency ----------------------------------------------
    let mut recall_ms = Vec::with_capacity(cfg.probe_n);
    for i in 0..cfg.probe_n {
        let req = RecallBuilder::new(LoadGenerator::cue(i).as_str())
            .top_k(cfg.top_k)
            .include_text(false)
            .build();
        let start = Instant::now();
        client.recall(&req).await?;
        recall_ms.push(ms(start));
    }

    // --- throughput (sustained, sequential) --------------------------
    let encode_tput = throughput("encode", cfg.probe_n, targets.encode_ops_per_sec, &encode_ms);
    let recall_tput = throughput("recall", cfg.probe_n, targets.recall_ops_per_sec, &recall_ms);

    Ok(ScaleReport {
        ingested,
        latency: vec![
            latency("encode", &mut encode_ms, &targets.encode),
            latency("recall", &mut recall_ms, &targets.recall),
        ],
        throughput: vec![encode_tput, recall_tput],
    })
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

/// Nearest-rank percentile over a slice (sorts in place).
fn percentile(samples: &mut [f64], q: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (((samples.len() as f64) * q).ceil() as usize)
        .saturating_sub(1)
        .min(samples.len() - 1);
    samples[idx]
}

fn latency(verb: &'static str, samples: &mut [f64], target: &VerbTarget) -> LatencyResult {
    LatencyResult {
        verb,
        samples: samples.len(),
        p50_ms: percentile(samples, 0.50),
        p99_ms: percentile(samples, 0.99),
        target_p50_ms: target.p50_ms,
        target_p99_ms: target.p99_ms,
    }
}

fn throughput(
    verb: &'static str,
    ops: usize,
    target_ops_per_sec: f64,
    sample_ms: &[f64],
) -> ThroughputResult {
    let total_ms: f64 = sample_ms.iter().sum();
    let ops_per_sec = if total_ms > 0.0 {
        (ops as f64) * 1000.0 / total_ms
    } else {
        0.0
    };
    ThroughputResult {
        verb,
        ops,
        ops_per_sec,
        target_ops_per_sec,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_nearest_rank() {
        let mut v: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        assert_eq!(percentile(&mut v, 0.50), 50.0);
        assert_eq!(percentile(&mut v, 0.99), 99.0);
        assert_eq!(percentile(&mut v.clone(), 1.0), 100.0);
    }

    #[test]
    fn percentile_empty_is_zero() {
        let mut v: Vec<f64> = vec![];
        assert_eq!(percentile(&mut v, 0.5), 0.0);
    }

    #[test]
    fn latency_pass_requires_both_p50_and_p99() {
        let t = VerbTarget {
            p50_ms: 10.0,
            p99_ms: 50.0,
        };
        let mut ok = vec![1.0, 2.0, 3.0];
        let r = latency("recall", &mut ok, &t);
        assert!(r.pass());

        let mut slow = vec![100.0; 10];
        let r2 = latency("recall", &mut slow, &t);
        assert!(!r2.pass());
    }

    #[test]
    fn generator_is_deterministic_and_structured() {
        assert_eq!(LoadGenerator::memory(0), LoadGenerator::memory(0));
        assert!(LoadGenerator::memory(0).contains("billing"));
        assert!(LoadGenerator::cue(0).contains("billing"));
    }

    #[test]
    fn throughput_pass_requires_meeting_floor() {
        let fast = throughput("encode", 100, 100.0, &[1.0; 100]); // 100 ops in 100ms = 1000/s
        assert!(fast.pass());
        let slow = throughput("encode", 10, 1000.0, &[100.0; 10]); // 10 ops in 1000ms = 10/s
        assert!(!slow.pass());
    }
}
