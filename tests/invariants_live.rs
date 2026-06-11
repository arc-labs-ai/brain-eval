//! E5 core-invariant scenarios, live against a full-stack server.
//!
//! Boots the production image (E1 `ServerHandle::start_docker`) and runs the
//! invariant scenarios (idempotency-by-RequestId, tombstone visibility,
//! slot-version staleness) against it. These are correctness gates — every
//! scenario must pass.
//!
//! `#[ignore]`d: needs Docker + a `brain:<tag>` image + the `brain-models`
//! volume. Run:
//!
//! ```bash
//! cargo test --test invariants_live -- --ignored --nocapture
//! ```
//!
//! Override the image tag with `BRAIN_EVAL_IMAGE_TAG` (default `latest`).

#![cfg(not(miri))]

use std::time::Duration;

use brain_eval::run::server::{DockerServerOpts, ServerHandle};
use brain_eval::system::run_invariant_scenarios;

#[tokio::test]
#[ignore = "requires docker + a brain:<tag> image + the brain-models volume"]
async fn invariant_scenarios_all_pass() {
    let tag = std::env::var("BRAIN_EVAL_IMAGE_TAG").unwrap_or_else(|_| "latest".to_string());
    let opts = DockerServerOpts {
        image_tag: tag,
        container_name: "brain-eval-invariants".to_string(),
        data_port: 38082,
        metrics_port: 39093,
        health_timeout: Duration::from_secs(180),
        ..DockerServerOpts::default()
    };

    let server = ServerHandle::start_docker(opts)
        .await
        .expect("full-stack server should boot and become healthy");

    let outcomes = run_invariant_scenarios(server.endpoint()).await;
    server.stop().await;

    assert!(!outcomes.is_empty(), "expected invariant scenarios to run");
    let mut failed = Vec::new();
    for o in &outcomes {
        println!(
            "  [{}] {} — {}",
            if o.passed { "PASS" } else { "FAIL" },
            o.name,
            o.detail
        );
        if !o.passed {
            failed.push(o.name);
        }
    }
    assert!(failed.is_empty(), "invariant scenarios failed: {failed:?}");
}
