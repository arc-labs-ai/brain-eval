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

use std::net::SocketAddr;

use brain_eval::blackbox::system::run_typed_graph_scenarios;

/// The data-plane endpoint to test against. `None` (unset) skips.
fn endpoint() -> Option<SocketAddr> {
    std::env::var("BRAIN_EVAL_ENDPOINT")
        .ok()
        .and_then(|s| s.parse().ok())
}

#[tokio::test]
#[ignore = "requires a running brain-server; set BRAIN_EVAL_ENDPOINT"]
async fn typed_graph_scenarios_against_external_server() {
    let Some(addr) = endpoint() else {
        eprintln!("BRAIN_EVAL_ENDPOINT not set — skipping");
        return;
    };

    let outcomes = run_typed_graph_scenarios(addr).await;

    assert!(!outcomes.is_empty(), "expected typed-graph scenarios to run");
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
    assert!(failed.is_empty(), "typed-graph scenarios failed: {failed:?}");
}
