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

use brain_eval::blackbox::scale::{run_scale, ScaleConfig, Targets};
use brain_eval::client::harness::BrainEvalHarness;
use brain_eval::client::server::{DockerServerOpts, ServerHandle};
use std::time::Duration;

/// Boot a throwaway server — or attach to the operator's, when
/// `BRAIN_EVAL_ENDPOINT` is set.
///
/// This used to call `endpoint()`, which assumes a server is already
/// listening on `127.0.0.1:18080` and mints against `BRAIN_EVAL_ADMIN_SECRET`.
/// CI provides neither, so it failed in under 20ms with `Mint(NoSecret)` —
/// before exercising anything it is meant to test. Every other live test
/// boots its own container, which is the suite's documented contract.
async fn server() -> ServerHandle {
    let tag = std::env::var("BRAIN_EVAL_IMAGE_TAG").unwrap_or_else(|_| "latest".to_string());
    ServerHandle::from_env_or_docker(DockerServerOpts {
        image_tag: tag,
        container_name: "brain-eval-scale-smoke".to_string(),
        data_port: 38094,
        metrics_port: 39104,
        // Distinct from the 19092 default, which collides with any other
        // live container still up.
        admin_port: 39114,
        health_timeout: Duration::from_secs(180),
        ..DockerServerOpts::default()
    })
    .await
    .expect("server should be reachable (external) or boot and become healthy (docker)")
}

#[tokio::test]
#[ignore = "requires docker + a brain:<tag> image + the model volume"]
async fn scale_smoke_produces_a_well_formed_report() {
    let server = server().await;
    let harness = BrainEvalHarness::connect_minted(
        &server.key_minter(),
        server.endpoint(),
        brain_db_sdk::new_id(),
    )
    .await
    .expect("connect to booted server");

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
