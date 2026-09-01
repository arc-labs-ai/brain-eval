//! E5 core-invariant scenarios, live against a full-stack server.
//!
//! Boots the production image (E1 `ServerHandle::start_docker`) and runs the
//! invariant scenarios (idempotency-by-RequestId, tombstone visibility,
//! slot-version staleness) against it. These are correctness gates — every
//! scenario must pass.
//!
//! Two modes, selected by `BRAIN_EVAL_ENDPOINT`:
//!
//! - **unset** — boots its own throwaway `brain:<tag>` container (needs Docker
//!   + a `brain:<tag>` image + the `brain-models` volume; override the tag with
//!   `BRAIN_EVAL_IMAGE_TAG`, default `latest`).
//! - **set** — runs the same scenarios against that already-running server,
//!   minting keys through its admin plane (`BRAIN_EVAL_ADMIN_SECRET` /
//!   `BRAIN_EVAL_ADMIN_PORT`).
//!
//! `#[ignore]`d. Run:
//!
//! ```bash
//! # docker mode
//! cargo test --test invariants_live -- --ignored --nocapture
//! # external mode
//! BRAIN_EVAL_ENDPOINT=127.0.0.1:9090 BRAIN_EVAL_ADMIN_SECRET=eval-admin-secret \
//!   cargo test --test invariants_live -- --ignored --nocapture
//! ```

#![cfg(not(miri))]

use std::time::Duration;

use brain_eval::blackbox::system::run_invariant_scenarios;
use brain_eval::client::server::{DockerServerOpts, ServerHandle};

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

    let server = ServerHandle::from_env_or_docker(opts)
        .await
        .expect("server should be reachable (external) or boot and become healthy (docker)");

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
