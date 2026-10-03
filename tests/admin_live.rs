//! Operator-admin acceptance scenarios ("E7") against an **already-running**
//! server: the historical audit log (`GET /v1/audit`), derived-index rebuild
//! (`POST /v1/rebuild[-ann]`), and metrics-family completeness (`/metrics`).
//!
//! `#[ignore]`d: needs a running brain-server with its admin plane reachable
//! and the admin bearer secret. These scenarios drive the operator HTTP plane
//! and the `/metrics` endpoint in addition to the wire path, so they need the
//! admin port + secret (and the metrics port) as well as the data endpoint:
//!
//! ```bash
//! BRAIN_EVAL_ENDPOINT=127.0.0.1:9090 \
//! BRAIN_EVAL_ADMIN_SECRET=dev-admin-token \
//! BRAIN_EVAL_ADMIN_PORT=9092 \
//! BRAIN_EVAL_METRICS_PORT=9091 \
//!   cargo test --test admin_live -- --ignored --nocapture
//! ```

#![cfg(not(miri))]

use std::time::Duration;

use brain_eval::client::server::{DockerServerOpts, ServerHandle};

use brain_eval::blackbox::system::run_admin_scenarios;

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
        container_name: "brain-eval-admin".to_string(),
        data_port: 38090,
        metrics_port: 39110,
        admin_port: 39120,
        health_timeout: Duration::from_secs(180),
        ..DockerServerOpts::default()
    }
}

#[tokio::test]
#[ignore = "requires a running brain-server + admin secret; set BRAIN_EVAL_ENDPOINT + BRAIN_EVAL_ADMIN_SECRET"]
async fn admin_scenarios_against_external_server() {
    let server = ServerHandle::from_env_or_docker(server_opts())
        .await
        .expect("server should be reachable (external) or boot and become healthy (docker)");
    let addr = server.endpoint();

    let outcomes = run_admin_scenarios(addr).await;

    assert!(!outcomes.is_empty(), "expected admin scenarios to run");
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
    assert!(failed.is_empty(), "admin scenarios failed: {failed:?}");
}
