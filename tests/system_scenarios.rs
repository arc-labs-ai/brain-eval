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

use brain_eval::system::run_core_scenarios;

fn endpoint() -> std::net::SocketAddr {
    std::env::var("BRAIN_EVAL_ENDPOINT")
        .unwrap_or_else(|_| "127.0.0.1:18080".to_string())
        .parse()
        .expect("BRAIN_EVAL_ENDPOINT must be host:port")
}

#[tokio::test]
#[ignore = "requires a running brain-server (set BRAIN_EVAL_ENDPOINT)"]
async fn core_scenarios_all_pass() {
    let outcomes = run_core_scenarios(endpoint()).await;
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
