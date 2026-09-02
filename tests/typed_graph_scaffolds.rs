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

use std::net::SocketAddr;

use brain_eval::blackbox::system::run_typed_graph_scaffolds;

/// The data-plane endpoint to test against. `None` (unset) skips.
fn endpoint() -> Option<SocketAddr> {
    std::env::var("BRAIN_EVAL_ENDPOINT")
        .ok()
        .and_then(|s| s.parse().ok())
}

#[tokio::test]
#[ignore = "requires a running brain-server; set BRAIN_EVAL_ENDPOINT"]
async fn typed_graph_scaffolds_observe() {
    let Some(addr) = endpoint() else {
        eprintln!("BRAIN_EVAL_ENDPOINT not set — skipping");
        return;
    };

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
