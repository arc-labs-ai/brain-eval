//! Typed-graph scenarios against an **already-running** server.
//!
//! The sibling `typed_graph_live` boots its own `brain:<tag>` container. That
//! is the right shape for CI, but it pins the run to whatever image happens to
//! be on the box — which silently diverges from source between image builds —
//! and it forces the key minter and the data plane onto different ports (see
//! `BRAIN_EVAL_ADMIN_PORT`, which must match the booted container's admin
//! port or every scenario fails with a confusing "unknown API key").
//!
//! This variant runs the same scenarios against a server you already have up
//! (e.g. the devcontainer's `cargo run --bin brain-server`), so the gate
//! reflects *current source* rather than a cached image.
//!
//! ```bash
//! BRAIN_EVAL_ENDPOINT=127.0.0.1:9090 \
//! BRAIN_EVAL_ADMIN_SECRET=dev-admin-token \
//! BRAIN_EVAL_ADMIN_PORT=9092 \
//!   cargo test --test typed_graph_external -- --ignored --nocapture
//! ```

#![cfg(not(miri))]

use std::time::Duration;

use brain_eval::client::server::{DockerServerOpts, ServerHandle};

use brain_eval::blackbox::system::run_typed_graph_scenarios;

/// The data-plane endpoint to test against. `None` (unset) skips.
/// Attach to `BRAIN_EVAL_ENDPOINT` when the operator set one, otherwise boot
/// a throwaway `brain:<tag>` container for this suite.
///
/// This used to return `Option` and the test returned early when it was
/// `None` — which reported a PASS without contacting a server at all. A suite
/// that goes green having exercised nothing is worse than one that fails:
/// it hides every regression it was written to catch. Booting is the same
/// path the other live suites take, so "no endpoint configured" is now a
/// slower run rather than a silent skip.
fn server_opts() -> DockerServerOpts {
    DockerServerOpts {
        container_name: "brain-eval-tg-external".to_string(),
        data_port: 38093,
        metrics_port: 39113,
        admin_port: 39123,
        health_timeout: Duration::from_secs(180),
        ..DockerServerOpts::default()
    }
}

#[tokio::test]
#[ignore = "requires a running brain-server; set BRAIN_EVAL_ENDPOINT"]
async fn typed_graph_scenarios_against_external_server() {
    let server = ServerHandle::from_env_or_docker(server_opts())
        .await
        .expect("server should be reachable (external) or boot and become healthy (docker)");
    let addr = server.endpoint();

    let outcomes = run_typed_graph_scenarios(addr).await;

    assert!(
        !outcomes.is_empty(),
        "expected typed-graph scenarios to run"
    );
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
    assert!(
        failed.is_empty(),
        "typed-graph scenarios failed: {failed:?}"
    );
}
