//! Deferred §19/06 scenario scaffolds, run live for observation.
//!
//! These are the items Wave 3/4 deferred because gating them without a live
//! server would bake in an unverified behavioural assumption (schema
//! transitions, contradiction detection, fuzzy/embedding resolution tiers,
//! confidence aggregation / schema-version stamping, extraction tiers, and
//! extraction guardrails). Each scaffold hard-asserts the *safe* invariants it
//! is confident about and records the flagged, behaviour-dependent part as an
//! **advisory** outcome.
//!
//! The gate here is therefore asymmetric: a non-advisory failure (a broken safe
//! invariant — lost data, a crashed pipeline, an undeclared write that
//! succeeded) fails the run; an advisory outcome only prints its observation,
//! for a human to turn into a real gate once the live behaviour is confirmed.
//! This keeps the scaffolds runnable and honest without polluting the
//! acceptance gate in `typed_graph_live` / `typed_graph_external`.
//!
//! Runs against an already-running server (same env as `typed_graph_external`):
//!
//! ```bash
//! BRAIN_EVAL_ENDPOINT=127.0.0.1:9090 \
//! BRAIN_EVAL_ADMIN_SECRET=dev-admin-token \
//! BRAIN_EVAL_ADMIN_PORT=9092 \
//!   cargo test --test typed_graph_scaffolds -- --ignored --nocapture
//! ```

#![cfg(not(miri))]

use std::time::Duration;

use brain_eval::client::server::{DockerServerOpts, ServerHandle};

use brain_eval::blackbox::system::run_typed_graph_scaffolds;

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
        container_name: "brain-eval-tg-scaffold".to_string(),
        data_port: 38094,
        metrics_port: 39114,
        admin_port: 39124,
        health_timeout: Duration::from_secs(180),
        ..DockerServerOpts::default()
    }
}

#[tokio::test]
#[ignore = "requires a running brain-server; set BRAIN_EVAL_ENDPOINT"]
async fn typed_graph_scaffolds_observe() {
    let server = ServerHandle::from_env_or_docker(server_opts())
        .await
        .expect("server should be reachable (external) or boot and become healthy (docker)");
    let addr = server.endpoint();

    let outcomes = run_typed_graph_scaffolds(addr).await;
    assert!(!outcomes.is_empty(), "expected scaffolds to run");

    // A non-advisory failure = a broken safe invariant; advisory outcomes are
    // observations only.
    let mut hard_failures = Vec::new();
    for o in &outcomes {
        let tag = match (o.passed, o.advisory) {
            (_, true) => "OBSERVE",
            (true, false) => "PASS",
            (false, false) => "FAIL",
        };
        println!("  [{tag}] {} — {}", o.name, o.detail);
        if !o.passed && !o.advisory {
            hard_failures.push(o.name);
        }
    }
    assert!(
        hard_failures.is_empty(),
        "scaffold safe-invariant failures: {hard_failures:?}"
    );
}
