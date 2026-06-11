//! Soak harness — sustained mixed workload over a long window.
//!
//! Drives a steady encode/recall mix against a running server for a set
//! duration, sampling at intervals to catch slow degradation:
//!
//! - **errors** — any failed op fails the run.
//! - **recall drift** — known-answer recall@1 must stay at/above a floor.
//! - **latency drift** — per-window op p99 must not balloon over the run
//!   (a worst-vs-best ratio guard); catches an index/allocator that slows
//!   as it ages.
//! - **memory leak** — process RSS (scraped from the server's `/metrics`,
//!   when a `metrics_endpoint` is configured) must not grow beyond a
//!   tolerance across the run (spec resource target: ≤ ~10% over 48 h).
//!
//! The full gate is a 48 h run on reference hardware; the same harness runs
//! for seconds on a dev box as a smoke. The trend guards only engage once
//! there are enough samples, and the smoke's tolerances are deliberately
//! loose — the smoke proves the harness + sampling work, not the absolute
//! numbers.
//!
//! Disk growth is not asserted here: it is observed at scale by the
//! storage-footprint gate ([`crate::system::storage_footprint`]) via the
//! `brain_wal_size_bytes` / `brain_metadata_size_bytes` gauges.

use std::net::SocketAddr;
use std::time::Duration;

use brain_db_sdk::{EncodeBuilder, RecallBuilder};
use tokio::time::Instant;

use crate::run::harness::{BrainEvalHarness, HarnessError};
use crate::run::metrics::Metrics;

/// Trend guards (latency drift, RSS leak) need at least this many samples
/// before they engage — fewer points is too noisy to judge a trend.
const MIN_TREND_SAMPLES: usize = 3;

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
    /// Server metrics endpoint (`host:metrics_port`) to scrape RSS from. If
    /// `None`, RSS isn't sampled and the leak guard is skipped.
    pub metrics_endpoint: Option<SocketAddr>,
    /// Max allowed RSS growth ratio over the run (max/min). 1.10 ≈ the
    /// spec's "≤ 10% over the run" leak target.
    pub max_rss_growth: f64,
    /// Max allowed per-window p99 latency drift ratio over the run
    /// (worst/best).
    pub max_latency_drift: f64,
}

impl SoakConfig {
    /// A few-second dev-box smoke of the harness. The recall floor and trend
    /// tolerances are deliberately lenient: a cold first sample on an
    /// emulated box is noisy, and the smoke only proves the harness runs. A
    /// reference-hardware soak sets strict values on its own config.
    #[must_use]
    pub fn smoke() -> Self {
        Self {
            duration: Duration::from_secs(5),
            sample_every: Duration::from_secs(1),
            batch: 20,
            top_k: 10,
            recall_floor: 0.5,
            metrics_endpoint: None,
            max_rss_growth: 2.0,
            max_latency_drift: 8.0,
        }
    }

    /// Point the leak guard at a server's metrics plane.
    #[must_use]
    pub fn with_metrics(mut self, endpoint: SocketAddr) -> Self {
        self.metrics_endpoint = Some(endpoint);
        self
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
    /// p99 op latency over the window since the previous sample, ms.
    pub p99_ms: f64,
    /// Process RSS at this sample, bytes (when a metrics endpoint is set).
    pub rss_bytes: Option<u64>,
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
    /// Max allowed RSS growth ratio (max/min).
    pub max_rss_growth: f64,
    /// Max allowed p99 latency drift ratio (worst/best).
    pub max_latency_drift: f64,
}

impl SoakReport {
    /// Healthy iff: no errors; recall@1 held the floor at every sample; and
    /// — once there are enough samples — latency p99 and RSS stayed within
    /// their drift/growth tolerances.
    #[must_use]
    pub fn healthy(&self) -> bool {
        if self.total_errors != 0 || self.samples.is_empty() {
            return false;
        }
        if !self
            .samples
            .iter()
            .all(|s| s.recall_at_1 >= self.recall_floor)
        {
            return false;
        }
        if !self.latency_ok() {
            return false;
        }
        self.rss_ok()
    }

    /// Per-window p99 never drifted past `max_latency_drift × best`.
    #[must_use]
    pub fn latency_ok(&self) -> bool {
        if self.samples.len() < MIN_TREND_SAMPLES {
            return true;
        }
        let p99s: Vec<f64> = self
            .samples
            .iter()
            .map(|s| s.p99_ms)
            .filter(|v| *v > 0.0)
            .collect();
        if p99s.len() < MIN_TREND_SAMPLES {
            return true;
        }
        let best = p99s.iter().copied().fold(f64::INFINITY, f64::min);
        let worst = p99s.iter().copied().fold(0.0_f64, f64::max);
        best <= 0.0 || worst <= best * self.max_latency_drift
    }

    /// RSS never grew past `max_rss_growth × min` (leak guard). Skipped
    /// when RSS wasn't sampled.
    #[must_use]
    pub fn rss_ok(&self) -> bool {
        let rss: Vec<u64> = self.samples.iter().filter_map(|s| s.rss_bytes).collect();
        if rss.len() < MIN_TREND_SAMPLES {
            return true;
        }
        let min = *rss.iter().min().unwrap_or(&0);
        let max = *rss.iter().max().unwrap_or(&0);
        min == 0 || (max as f64) <= (min as f64) * self.max_rss_growth
    }

    /// Human-readable summary.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut s = format!(
            "=== soak report — {} encodes / {} recalls / {} errors ===\n",
            self.total_encodes, self.total_recalls, self.total_errors
        );
        for sample in &self.samples {
            let rss = sample
                .rss_bytes
                .map(|b| format!("{:.0}MiB", b as f64 / (1024.0 * 1024.0)))
                .unwrap_or_else(|| "-".to_string());
            s.push_str(&format!(
                "  t+{:>5}s  enc={:>8} rec={:>8} err={:>4}  recall@1 {:.3}  p99 {:>7.2}ms  rss {}\n",
                sample.elapsed_s,
                sample.cumulative_encodes,
                sample.cumulative_recalls,
                sample.cumulative_errors,
                sample.recall_at_1,
                sample.p99_ms,
                rss,
            ));
        }
        s.push_str(&format!(
            "latency-ok: {}  rss-ok: {}  healthy: {}\n",
            yn(self.latency_ok()),
            yn(self.rss_ok()),
            yn(self.healthy()),
        ));
        s
    }
}

fn yn(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "no"
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
    // Per-window op latencies, drained into each sample's p99.
    let mut window_ms: Vec<f64> = Vec::new();

    while start.elapsed() < cfg.duration {
        // One work batch: interleave encodes and recalls, timing each op.
        for _ in 0..cfg.batch {
            let text = format!("soak {salt} item {seq}: steady-state workload note");
            seq += 1;
            let enc = EncodeBuilder::new(text.as_str()).deduplicate(false).build();
            let t = Instant::now();
            match harness.client().encode(&enc).await {
                Ok(_) => {
                    encodes += 1;
                    window_ms.push(elapsed_ms(t));
                }
                Err(_) => errors += 1,
            }
            let cue = format!("soak {salt} item {}", seq.saturating_sub(1));
            let rec = RecallBuilder::new(cue.as_str())
                .top_k(cfg.top_k)
                .include_text(false)
                .build();
            let t = Instant::now();
            match harness.client().recall(&rec).await {
                Ok(_) => {
                    recalls += 1;
                    window_ms.push(elapsed_ms(t));
                }
                Err(_) => errors += 1,
            }
        }

        if Instant::now() >= next_sample {
            let recall_at_1 = drift_probe(&harness, &salt, cfg.top_k).await;
            let p99_ms = percentile(&mut window_ms, 0.99);
            window_ms.clear();
            let rss_bytes = match cfg.metrics_endpoint {
                Some(addr) => Metrics::scrape(addr)
                    .await
                    .ok()
                    .and_then(|m| m.get("process_memory_resident_bytes"))
                    .map(|v| v as u64),
                None => None,
            };
            samples.push(SoakSample {
                elapsed_s: start.elapsed().as_secs(),
                cumulative_encodes: encodes,
                cumulative_recalls: recalls,
                cumulative_errors: errors,
                recall_at_1,
                p99_ms,
                rss_bytes,
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
        max_rss_growth: cfg.max_rss_growth,
        max_latency_drift: cfg.max_latency_drift,
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

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

/// Nearest-rank percentile (sorts in place). Empty → 0.0.
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

    fn sample(recall: f64, errors: u64, p99: f64, rss: Option<u64>) -> SoakSample {
        SoakSample {
            elapsed_s: 1,
            cumulative_encodes: 10,
            cumulative_recalls: 10,
            cumulative_errors: errors,
            recall_at_1: recall,
            p99_ms: p99,
            rss_bytes: rss,
        }
    }

    fn report(samples: Vec<SoakSample>) -> SoakReport {
        SoakReport {
            samples,
            total_encodes: 100,
            total_recalls: 100,
            total_errors: 0,
            recall_floor: 0.90,
            max_rss_growth: 1.10,
            max_latency_drift: 3.0,
        }
    }

    #[test]
    fn healthy_requires_no_errors_and_recall_above_floor() {
        let ok = report(vec![
            sample(0.95, 0, 5.0, None),
            sample(0.93, 0, 5.0, None),
            sample(0.94, 0, 5.0, None),
        ]);
        assert!(ok.healthy());

        let drifted = report(vec![
            sample(0.95, 0, 5.0, None),
            sample(0.80, 0, 5.0, None),
            sample(0.94, 0, 5.0, None),
        ]);
        assert!(!drifted.healthy(), "recall below floor is unhealthy");

        let errored = SoakReport {
            total_errors: 3,
            ..ok.clone()
        };
        assert!(!errored.healthy(), "errors are unhealthy");
    }

    #[test]
    fn latency_drift_beyond_tolerance_is_unhealthy() {
        // best 5ms, worst 30ms → 6× > 3× tolerance.
        let drifted = report(vec![
            sample(0.95, 0, 5.0, None),
            sample(0.95, 0, 10.0, None),
            sample(0.95, 0, 30.0, None),
        ]);
        assert!(!drifted.latency_ok());
        assert!(!drifted.healthy());

        // within 3×.
        let steady = report(vec![
            sample(0.95, 0, 5.0, None),
            sample(0.95, 0, 8.0, None),
            sample(0.95, 0, 12.0, None),
        ]);
        assert!(steady.latency_ok());
    }

    #[test]
    fn rss_growth_beyond_tolerance_is_unhealthy() {
        // 100MiB → 200MiB = 2× > 1.10 tolerance.
        let leak = report(vec![
            sample(0.95, 0, 5.0, Some(100 << 20)),
            sample(0.95, 0, 5.0, Some(150 << 20)),
            sample(0.95, 0, 5.0, Some(200 << 20)),
        ]);
        assert!(!leak.rss_ok());
        assert!(!leak.healthy());

        // flat RSS.
        let flat = report(vec![
            sample(0.95, 0, 5.0, Some(100 << 20)),
            sample(0.95, 0, 5.0, Some(101 << 20)),
            sample(0.95, 0, 5.0, Some(102 << 20)),
        ]);
        assert!(flat.rss_ok());
    }

    #[test]
    fn trend_guards_skip_with_too_few_samples() {
        // Two samples: trend guards don't engage even with a big jump.
        let r = report(vec![
            sample(0.95, 0, 5.0, Some(100 << 20)),
            sample(0.95, 0, 100.0, Some(500 << 20)),
        ]);
        assert!(r.latency_ok());
        assert!(r.rss_ok());
        assert!(r.healthy());
    }

    #[test]
    fn empty_samples_is_not_healthy() {
        assert!(!report(vec![]).healthy());
    }
}
