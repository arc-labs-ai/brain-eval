//! Phase 3 — core system/durability scenarios against a live server.
//!
//! `#[ignore]`d: needs a running brain-server. Point it with
//! `BRAIN_EVAL_ENDPOINT` (default `127.0.0.1:18080`):
//!
//! ```bash
//! BRAIN_EVAL_ENDPOINT=127.0.0.1:18080 \
//!   cargo test --test system_scenarios -- --ignored --nocapture
//! ```

#![cfg(not(miri))]

use brain_eval::blackbox::system::run_core_scenarios;
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
        container_name: "brain-eval-system-scenarios".to_string(),
        data_port: 38095,
        metrics_port: 39105,
        // Distinct from the 19092 default, which collides with any other
        // live container still up.
        admin_port: 39115,
        health_timeout: Duration::from_secs(180),
        ..DockerServerOpts::default()
    })
    .await
    .expect("server should be reachable (external) or boot and become healthy (docker)")
}

#[tokio::test]
#[ignore = "requires docker + a brain:<tag> image + the model volume"]
async fn core_scenarios_all_pass() {
    let server = server().await;
    let outcomes = run_core_scenarios(server.endpoint()).await;
    assert!(!outcomes.is_empty(), "expected scenarios to run");

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
    assert!(failed.is_empty(), "scenarios failed: {failed:?}");
}
