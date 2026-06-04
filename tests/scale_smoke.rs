//! Phase 2 smoke test — a small scale run against a live server.
//!
//! `#[ignore]`d: needs a running brain-server. Point it with
//! `BRAIN_EVAL_ENDPOINT` (default `127.0.0.1:18080`, the `just
//! serve-local` data plane):
//!
//! ```bash
//! BRAIN_EVAL_ENDPOINT=127.0.0.1:18080 \
//!   cargo test --test scale_smoke -- --ignored --nocapture
//! ```
//!
//! It exercises the load generator, the latency/throughput probes, and the
//! report math against a real server. It asserts the report is *well-
//! formed* — NOT that the thresholds pass, since an emulated dev container
//! is far slower than the reference hardware the targets assume.

#![cfg(not(miri))]

use brain_eval::run::harness::BrainEvalHarness;
use brain_eval::scale::{run_scale, ScaleConfig, Targets};

fn endpoint() -> std::net::SocketAddr {
    std::env::var("BRAIN_EVAL_ENDPOINT")
        .unwrap_or_else(|_| "127.0.0.1:18080".to_string())
        .parse()
        .expect("BRAIN_EVAL_ENDPOINT must be host:port")
}

#[tokio::test]
#[ignore = "requires a running brain-server (set BRAIN_EVAL_ENDPOINT)"]
async fn scale_smoke_produces_a_well_formed_report() {
    let harness = BrainEvalHarness::connect(endpoint())
        .await
        .expect("connect to server");

    let cfg = ScaleConfig {
        ingest_n: 200,
        probe_n: 50,
        top_k: 10,
    };
    let report = run_scale(harness.client(), &cfg, &Targets::default())
        .await
        .expect("scale run");

    println!("{}", report.to_text());

    assert_eq!(report.ingested, cfg.ingest_n, "all memories should ingest");
    assert_eq!(report.latency.len(), 2, "encode + recall latency");
    assert_eq!(report.throughput.len(), 2, "encode + recall throughput");
    for l in &report.latency {
        assert_eq!(l.samples, cfg.probe_n);
        assert!(l.p50_ms >= 0.0 && l.p99_ms >= l.p50_ms, "p99 >= p50 >= 0");
    }
    for t in &report.throughput {
        assert!(t.ops_per_sec > 0.0, "throughput must be positive");
    }

    harness.close().await.expect("clean close");
}
