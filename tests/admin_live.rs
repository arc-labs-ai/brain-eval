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

use std::net::SocketAddr;

use brain_eval::blackbox::system::run_admin_scenarios;

/// The data-plane endpoint to test against. `None` (unset) skips.
fn endpoint() -> Option<SocketAddr> {
    std::env::var("BRAIN_EVAL_ENDPOINT")
        .ok()
        .and_then(|s| s.parse().ok())
}

#[tokio::test]
#[ignore = "requires a running brain-server + admin secret; set BRAIN_EVAL_ENDPOINT + BRAIN_EVAL_ADMIN_SECRET"]
async fn admin_scenarios_against_external_server() {
    let Some(addr) = endpoint() else {
        eprintln!("BRAIN_EVAL_ENDPOINT not set — skipping");
        return;
    };

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
