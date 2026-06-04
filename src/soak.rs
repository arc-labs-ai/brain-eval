//! Soak harness — sustained mixed workload over a long window.
//!
//! Drives a steady encode/recall mix against a running server for a set
//! duration, sampling at intervals to catch slow degradation: rising
//! errors, or recall quality drifting down as the index ages and
//! tombstones accumulate. The full gate is a 48 h run on reference
//! hardware; the same harness runs for seconds on a dev box as a smoke.

use std::net::SocketAddr;
use std::time::Duration;

use brain_db_sdk::{EncodeBuilder, RecallBuilder};
use tokio::time::Instant;

use crate::run::harness::{BrainEvalHarness, HarnessError};

/// What to run.
#[derive(Debug, Clone)]
pub struct SoakConfig {
    /// Total wall-clock duration of the soak.
    pub duration: Duration,
    /// How often to record a sample (and run a small recall-drift probe).
    pub sample_every: Duration,
    /// Encode/recall ops per work batch between samples.
    pub batch: usize,
    /// `top_k` for recall ops.
    pub top_k: u32,
    /// Minimum acceptable known-answer recall@1 at every sample.
    pub recall_floor: f64,
}

impl SoakConfig {
    /// A few-second dev-box smoke of the harness. The recall floor is
    /// deliberately lenient: a cold first sample on an emulated box is
    /// noisy, and the smoke only proves the harness runs. A reference-
    /// hardware soak sets a strict floor (≈0.95) on its own config.
    #[must_use]
    pub fn smoke() -> Self {
        Self {
            duration: Duration::from_secs(5),
            sample_every: Duration::from_secs(1),
            batch: 20,
            top_k: 10,
            recall_floor: 0.5,
        }
    }
}

/// One periodic sample.
#[derive(Debug, Clone)]
pub struct SoakSample {
    /// Seconds since the soak started.
    pub elapsed_s: u64,
    /// Cumulative successful encodes so far.
    pub cumulative_encodes: u64,
    /// Cumulative successful recalls so far.
    pub cumulative_recalls: u64,
    /// Cumulative errors so far.
    pub cumulative_errors: u64,
    /// Known-answer recall@1 measured at this sample (drift signal).
    pub recall_at_1: f64,
}

/// Full soak result.
#[derive(Debug, Clone)]
pub struct SoakReport {
    /// Per-interval samples.
    pub samples: Vec<SoakSample>,
    /// Total successful encodes over the run.
    pub total_encodes: u64,
    /// Total successful recalls over the run.
    pub total_recalls: u64,
    /// Total errors over the run.
    pub total_errors: u64,
    /// Recall@1 floor every sample must hold.
    pub recall_floor: f64,
}

impl SoakReport {
    /// No errors, and recall@1 stayed at or above the floor at every
    /// sample (no degradation over the window).
    #[must_use]
    pub fn healthy(&self) -> bool {
        self.total_errors == 0
            && !self.samples.is_empty()
            && self.samples.iter().all(|s| s.recall_at_1 >= self.recall_floor)
    }

    /// Human-readable summary.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut s = format!(
            "=== soak report — {} encodes / {} recalls / {} errors ===\n",
            self.total_encodes, self.total_recalls, self.total_errors
        );
        for sample in &self.samples {
            s.push_str(&format!(
                "  t+{:>5}s  enc={:>8} rec={:>8} err={:>4}  recall@1 {:.3}\n",
                sample.elapsed_s,
                sample.cumulative_encodes,
                sample.cumulative_recalls,
                sample.cumulative_errors,
                sample.recall_at_1,
            ));
        }
        s.push_str(&format!(
            "healthy: {}\n",
            if self.healthy() { "yes" } else { "no" }
        ));
        s
    }
}

/// Run the soak against `endpoint` for `cfg.duration`.
pub async fn run_soak(endpoint: SocketAddr, cfg: &SoakConfig) -> Result<SoakReport, HarnessError> {
    let harness = BrainEvalHarness::connect(endpoint).await?;
    let salt = hex16(harness.agent_id());

    let start = Instant::now();
    let mut next_sample = start + cfg.sample_every;
    let mut samples = Vec::new();
    let mut encodes = 0u64;
    let mut recalls = 0u64;
    let mut errors = 0u64;
    let mut seq = 0usize;

    while start.elapsed() < cfg.duration {
        // One work batch: interleave encodes and recalls.
        for _ in 0..cfg.batch {
            let text = format!("soak {salt} item {seq}: steady-state workload note");
            seq += 1;
            let enc = EncodeBuilder::new(text.as_str()).deduplicate(false).build();
            match harness.client().encode(&enc).await {
                Ok(_) => encodes += 1,
                Err(_) => errors += 1,
            }
            let cue = format!("soak {salt} item {}", seq.saturating_sub(1));
            let rec = RecallBuilder::new(cue.as_str())
                .top_k(cfg.top_k)
                .include_text(false)
                .build();
            match harness.client().recall(&rec).await {
                Ok(_) => recalls += 1,
                Err(_) => errors += 1,
            }
        }

        if Instant::now() >= next_sample {
            let recall_at_1 = drift_probe(&harness, &salt, cfg.top_k).await;
            samples.push(SoakSample {
                elapsed_s: start.elapsed().as_secs(),
                cumulative_encodes: encodes,
                cumulative_recalls: recalls,
                cumulative_errors: errors,
                recall_at_1,
            });
            next_sample = Instant::now() + cfg.sample_every;
        }
    }

    let _ = harness.close().await;
    Ok(SoakReport {
        samples,
        total_encodes: encodes,
        total_recalls: recalls,
        total_errors: errors,
        recall_floor: cfg.recall_floor,
    })
}

/// Encode a small known-answer set and measure recall@1 right now — the
/// drift signal at one sample point.
async fn drift_probe(harness: &BrainEvalHarness, salt: &str, top_k: u32) -> f64 {
    const PROBE: usize = 10;
    const DRIFT_TOPICS: &[&str] = &[
        "billing", "search", "auth", "ingest", "planner", "indexer", "embedder", "gateway",
        "cache", "router",
    ];
    let tag = format!("{salt}drift");
    for i in 0..PROBE {
        let needle = format!("zd{tag}{i:04}xz");
        let text = format!(
            "Drift reference {i} for topic {}: the unique marker is {needle}, set amid ordinary prose so retrieval must discriminate.",
            DRIFT_TOPICS[i % DRIFT_TOPICS.len()]
        );
        let enc = EncodeBuilder::new(text.as_str()).deduplicate(false).build();
        if harness.client().encode(&enc).await.is_err() {
            return 0.0;
        }
    }
    let mut hit_at_1 = 0usize;
    for i in 0..PROBE {
        let needle = format!("zd{tag}{i:04}xz");
        let req = RecallBuilder::new(needle.as_str())
            .top_k(top_k)
            .include_text(true)
            .build();
        if let Ok(hits) = harness.client().recall(&req).await {
            if hits.first().is_some_and(|m| m.text.contains(&needle)) {
                hit_at_1 += 1;
            }
        }
    }
    hit_at_1 as f64 / PROBE as f64
}

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

    fn sample(recall: f64, errors: u64) -> SoakSample {
        SoakSample {
            elapsed_s: 1,
            cumulative_encodes: 10,
            cumulative_recalls: 10,
            cumulative_errors: errors,
            recall_at_1: recall,
        }
    }

    #[test]
    fn healthy_requires_no_errors_and_recall_above_floor() {
        let ok = SoakReport {
            samples: vec![sample(0.95, 0), sample(0.93, 0)],
            total_encodes: 100,
            total_recalls: 100,
            total_errors: 0,
            recall_floor: 0.90,
        };
        assert!(ok.healthy());

        let drifted = SoakReport {
            samples: vec![sample(0.95, 0), sample(0.80, 0)],
            recall_floor: 0.90,
            ..ok.clone()
        };
        assert!(!drifted.healthy(), "recall dropping below floor is unhealthy");

        let errored = SoakReport {
            total_errors: 3,
            ..ok.clone()
        };
        assert!(!errored.healthy(), "errors are unhealthy");
    }

    #[test]
    fn empty_samples_is_not_healthy() {
        let r = SoakReport {
            samples: vec![],
            total_encodes: 0,
            total_recalls: 0,
            total_errors: 0,
            recall_floor: 0.9,
        };
        assert!(!r.healthy());
    }
}
