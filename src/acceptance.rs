//! The v1.0 acceptance orchestrator.
//!
//! Composes the pillars into one gated run and a single pass/fail report:
//! per-verb latency + throughput (Phase 2), recall quality (Phase 4), the
//! core black-box scenarios (Phase 3), and restart-recovery (Phase 3b).
//! This is the artifact that closes the acceptance scale-run gate.
//!
//! Two tiers of gate live in the same report:
//! - **Correctness** gates (scenarios, recall@1, restart-recovery) must
//!   pass anywhere, including a slow dev box.
//! - **Performance** gates (latency, throughput) only carry their spec
//!   meaning on quiet reference hardware; under emulation they report
//!   measured-vs-target honestly and will usually FAIL — that's expected.
//!
//! `all_pass()` is the release gate; run it at the spec scale (1M) on
//! reference hardware.

use std::net::SocketAddr;

use crate::run::harness::BrainEvalHarness;
use crate::run::server::DockerServerOpts;
use crate::scale::{
    run_concurrent_throughput, run_recall_quality, run_scale, ConcurrentConfig, RecallTargets,
    ScaleConfig, Targets,
};
use crate::system::{
    kill_during_write, restart_recovery, run_core_scenarios, run_invariant_scenarios,
    run_typed_graph_scenarios, storage_footprint,
};

/// One acceptance gate.
#[derive(Debug, Clone)]
pub struct Gate {
    /// Gate name (e.g. `latency:recall`, `restart_recovery`).
    pub name: String,
    /// `true` for performance gates whose verdict is only meaningful on
    /// reference hardware; informational on a dev box.
    pub perf: bool,
    /// Whether this gate passed.
    pub passed: bool,
    /// Human-readable explanation (why it passed / how it failed).
    pub detail: String,
}

/// Full acceptance run result.
#[derive(Debug, Clone)]
pub struct AcceptanceReport {
    /// All gates, in run order.
    pub gates: Vec<Gate>,
}

impl AcceptanceReport {
    /// Every gate passed — the release gate.
    #[must_use]
    pub fn all_pass(&self) -> bool {
        self.gates.iter().all(|g| g.passed)
    }

    /// Only the correctness (non-perf) gates passed — the meaningful
    /// verdict on non-reference hardware.
    #[must_use]
    pub fn correctness_pass(&self) -> bool {
        self.gates.iter().filter(|g| !g.perf).all(|g| g.passed)
    }

    /// Human-readable per-gate report.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut s = String::from("=== v1.0 acceptance report ===\n");
        for g in &self.gates {
            s.push_str(&format!(
                "  [{}]{} {} — {}\n",
                if g.passed { "PASS" } else { "FAIL" },
                if g.perf { " (perf)" } else { "" },
                g.name,
                g.detail,
            ));
        }
        s.push_str(&format!(
            "correctness: {}   overall: {}\n",
            yn(self.correctness_pass()),
            yn(self.all_pass()),
        ));
        s
    }
}

fn yn(b: bool) -> &'static str {
    if b {
        "PASS"
    } else {
        "FAIL"
    }
}

/// What to run.
#[derive(Debug, Clone)]
pub struct AcceptanceConfig {
    /// Server to run scale / recall / scenarios against.
    pub endpoint: SocketAddr,
    /// Load + probe sizes.
    pub scale: ScaleConfig,
    /// Concurrent-throughput run parameters.
    pub concurrent: ConcurrentConfig,
    /// Recall-quality corpus size.
    pub recall_n: usize,
    /// `top_k` for the recall-quality queries.
    pub recall_top_k: u32,
    /// Run the restart-recovery gate (boots its own server on a volume).
    pub run_restart_recovery: bool,
    /// Boot opts for the restart-recovery server (unique ports / volume).
    pub recovery_opts: DockerServerOpts,
    /// Run the kill-during-write chaos gate (boots its own server on a volume).
    pub run_chaos: bool,
    /// Boot opts for the chaos server (unique ports / volume).
    pub chaos_opts: DockerServerOpts,
    /// Run the storage-footprint gate (boots its own server on a volume).
    pub run_storage: bool,
    /// Boot opts for the storage server (unique ports / volume).
    pub storage_opts: DockerServerOpts,
    /// Memories to ingest for the storage-footprint gate.
    pub storage_n: usize,
    /// Disk-budget ceiling (bytes/memory) for the storage gate; `None` =
    /// informational (the dev-box default).
    pub storage_max_bytes_per_memory: Option<f64>,
}

impl AcceptanceConfig {
    /// A small, fast configuration for a dev-box smoke of the orchestrator.
    #[must_use]
    pub fn smoke(endpoint: SocketAddr) -> Self {
        Self {
            endpoint,
            scale: ScaleConfig {
                ingest_n: 200,
                probe_n: 50,
                top_k: 10,
            },
            concurrent: ConcurrentConfig::smoke(),
            recall_n: 100,
            recall_top_k: 10,
            run_restart_recovery: false,
            recovery_opts: DockerServerOpts::default(),
            run_chaos: false,
            chaos_opts: DockerServerOpts::default(),
            run_storage: false,
            storage_opts: DockerServerOpts::default(),
            storage_n: 2_000,
            storage_max_bytes_per_memory: None,
        }
    }

    /// The reference-hardware acceptance config at the spec's primary scale
    /// (1M memories + 500K statements, §19/06). The perf gates carry their
    /// spec meaning here; the storage gate enforces the ~10 KB/memory disk
    /// budget. Meant for a quiet 16-core / 64 GiB / NVMe box — it is far too
    /// heavy for an emulated dev container.
    #[must_use]
    pub fn full_scale(endpoint: SocketAddr) -> Self {
        let volume_opts = |name: &str, data_port, metrics_port, vol: &str| DockerServerOpts {
            container_name: name.to_string(),
            data_port,
            metrics_port,
            data_volume: Some(vol.to_string()),
            // 1M slots × 1600 B ≈ 1.6 GiB of arena; size with headroom.
            arena_capacity: "2GiB".to_string(),
            ..DockerServerOpts::default()
        };
        Self {
            endpoint,
            scale: ScaleConfig {
                ingest_n: 1_000_000,
                probe_n: 1_000,
                top_k: 10,
            },
            concurrent: ConcurrentConfig::default(),
            recall_n: 10_000,
            recall_top_k: 10,
            run_restart_recovery: true,
            recovery_opts: volume_opts(
                "brain-acc-recovery",
                28080,
                29091,
                "brain-acc-recovery-data",
            ),
            run_chaos: true,
            chaos_opts: volume_opts("brain-acc-chaos", 28081, 29092, "brain-acc-chaos-data"),
            run_storage: true,
            storage_opts: volume_opts("brain-acc-storage", 28082, 29093, "brain-acc-storage-data"),
            storage_n: 1_000_000,
            // Spec disk budget: ~8-10 GB per shard at 1M ⇒ ≤ ~10 KiB/memory.
            storage_max_bytes_per_memory: Some(10.0 * 1024.0),
        }
    }
}

/// Run the acceptance suite and return the gated report.
pub async fn run_acceptance(cfg: AcceptanceConfig) -> AcceptanceReport {
    let mut gates = Vec::new();

    // --- scale: latency + throughput (perf gates) --------------------
    match BrainEvalHarness::connect(cfg.endpoint).await {
        Ok(harness) => {
            match run_scale(harness.client(), &cfg.scale, &Targets::default()).await {
                Ok(report) => {
                    for l in &report.latency {
                        gates.push(Gate {
                            name: format!("latency:{}", l.verb),
                            perf: true,
                            passed: l.pass(),
                            detail: format!(
                                "p50 {:.3}ms (≤{:.0}) p99 {:.3}ms (≤{:.0})",
                                l.p50_ms, l.target_p50_ms, l.p99_ms, l.target_p99_ms
                            ),
                        });
                    }
                    for t in &report.throughput {
                        gates.push(Gate {
                            name: format!("throughput:{}", t.verb),
                            perf: true,
                            passed: t.pass(),
                            detail: format!(
                                "{:.1} ops/s (≥{:.0})",
                                t.ops_per_sec, t.target_ops_per_sec
                            ),
                        });
                    }
                }
                Err(e) => gates.push(Gate {
                    name: "scale".into(),
                    perf: true,
                    passed: false,
                    detail: format!("scale run errored: {e}"),
                }),
            }

            // --- concurrent throughput (perf ops/s + correctness no-error)
            match run_concurrent_throughput(cfg.endpoint, &cfg.concurrent).await {
                Ok(report) => {
                    for r in &report.results {
                        gates.push(Gate {
                            name: format!("tput:{}", r.verb),
                            perf: true,
                            passed: r.meets_floor(),
                            detail: format!(
                                "{:.1} ops/s (≥{:.0}) over {} clients; p50 {:.2}ms p99 {:.2}ms; \
                                 ops={} err={} timeout={}",
                                r.ops_per_sec,
                                r.target_ops_per_sec,
                                r.clients,
                                r.p50_ms,
                                r.p99_ms,
                                r.ops,
                                r.errors,
                                r.timeouts,
                            ),
                        });
                    }
                    // Handling N concurrent clients with zero failed ops is a
                    // correctness property — it must hold on any hardware.
                    gates.push(Gate {
                        name: "concurrent_no_errors".into(),
                        perf: false,
                        passed: report.no_errors(),
                        detail: report.error_summary(),
                    });
                }
                Err(e) => gates.push(Gate {
                    name: "concurrent_no_errors".into(),
                    perf: false,
                    passed: false,
                    detail: format!("concurrent throughput run errored: {e}"),
                }),
            }

            // --- recall quality (correctness gate) -------------------
            let salt = hex16(harness.agent_id());
            match run_recall_quality(
                harness.client(),
                cfg.recall_n,
                cfg.recall_top_k,
                &salt,
                &RecallTargets::default(),
            )
            .await
            {
                Ok(rq) => gates.push(Gate {
                    name: "recall_quality".into(),
                    perf: false,
                    passed: rq.pass(),
                    detail: rq.to_text(),
                }),
                Err(e) => gates.push(Gate {
                    name: "recall_quality".into(),
                    perf: false,
                    passed: false,
                    detail: format!("recall probe errored: {e}"),
                }),
            }
            let _ = harness.close().await;
        }
        Err(e) => gates.push(Gate {
            name: "connect".into(),
            perf: false,
            passed: false,
            detail: format!("could not connect to {}: {e}", cfg.endpoint),
        }),
    }

    // --- core scenarios (correctness gates) --------------------------
    for o in run_core_scenarios(cfg.endpoint).await {
        gates.push(Gate {
            name: o.name.to_string(),
            perf: false,
            passed: o.passed,
            detail: o.detail,
        });
    }

    // --- typed-graph "E2" functional suite (correctness gates) -------
    for o in run_typed_graph_scenarios(cfg.endpoint).await {
        gates.push(Gate {
            name: o.name.to_string(),
            perf: false,
            passed: o.passed,
            detail: o.detail,
        });
    }

    // --- core-invariant "E5" suite (correctness gates) --------------
    for o in run_invariant_scenarios(cfg.endpoint).await {
        gates.push(Gate {
            name: o.name.to_string(),
            perf: false,
            passed: o.passed,
            detail: o.detail,
        });
    }

    // --- restart-recovery (correctness gate; boots its own server) ---
    if cfg.run_restart_recovery {
        let o = restart_recovery(cfg.recovery_opts.clone()).await;
        gates.push(Gate {
            name: o.name.to_string(),
            perf: false,
            passed: o.passed,
            detail: o.detail,
        });
    }

    // --- kill-during-write chaos (correctness gate; boots its own server) ---
    if cfg.run_chaos {
        let o = kill_during_write(cfg.chaos_opts.clone()).await;
        gates.push(Gate {
            name: o.name.to_string(),
            perf: false,
            passed: o.passed,
            detail: o.detail,
        });
    }

    // --- storage footprint (correctness gate; boots its own server) ---
    if cfg.run_storage {
        let o = storage_footprint(
            cfg.storage_opts.clone(),
            cfg.storage_n,
            cfg.storage_max_bytes_per_memory,
        )
        .await;
        gates.push(Gate {
            name: o.name.to_string(),
            perf: false,
            passed: o.passed,
            detail: o.detail,
        });
    }

    AcceptanceReport { gates }
}

/// Lowercase hex of a 16-byte id.
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

    fn gate(name: &str, perf: bool, passed: bool) -> Gate {
        Gate {
            name: name.into(),
            perf,
            passed,
            detail: String::new(),
        }
    }

    #[test]
    fn correctness_pass_ignores_perf_gates() {
        let r = AcceptanceReport {
            gates: vec![
                gate("latency:recall", true, false), // perf fail
                gate("recall_quality", false, true),
                gate("multi_agent_isolation", false, true),
            ],
        };
        assert!(r.correctness_pass(), "perf fail must not sink correctness");
        assert!(!r.all_pass(), "a failing perf gate fails overall");
    }

    #[test]
    fn all_pass_requires_everything() {
        let r = AcceptanceReport {
            gates: vec![gate("a", true, true), gate("b", false, true)],
        };
        assert!(r.all_pass());
        assert!(r.correctness_pass());
    }
}
