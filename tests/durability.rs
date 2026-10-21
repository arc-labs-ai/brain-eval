//! Phase 3b — restart-recovery durability scenario (owns the server).
//!
//! `#[ignore]`d: boots `brain:<tag>` twice on a persistent volume, so it
//! needs Docker + a `brain:<tag>` image + the BGE model on disk. Run:
//!
//! ```bash
//! cargo test --test durability -- --ignored --nocapture
//! ```

#![cfg(not(miri))]

use std::time::Duration;

use brain_eval::blackbox::system::{graceful_shutdown_drain, restart_recovery};
use brain_eval::client::server::DockerServerOpts;

#[tokio::test]
#[ignore = "requires docker + a brain:<tag> image + the BGE model on disk"]
async fn restart_recovery_no_data_loss() {
    let tag = std::env::var("BRAIN_EVAL_IMAGE_TAG").unwrap_or_else(|_| "latest".to_string());
    let opts = DockerServerOpts {
        image_tag: tag,
        container_name: "brain-eval-recovery".to_string(),
        // Non-default ports to avoid colliding with serve-local (18080).
        data_port: 38080,
        metrics_port: 39091,
        data_volume: Some("brain-eval-recovery-data".to_string()),
        health_timeout: Duration::from_secs(180),
        ..DockerServerOpts::default()
    };

    let outcome = restart_recovery(opts).await;
    println!(
        "  [{}] {} — {}",
        if outcome.passed { "PASS" } else { "FAIL" },
        outcome.name,
        outcome.detail
    );
    assert!(outcome.passed, "{}", outcome.detail);
}

/// Graceful-shutdown-drain scaffold: no data loss after a SIGTERM stop; the
/// derivation-drain behaviour is advisory (observed, not gated). Distinct
/// ports/name/volume from `restart_recovery` so the two can run back to back.
#[tokio::test]
#[ignore = "requires docker + a brain:<tag> image + the BGE model on disk"]
async fn graceful_shutdown_drains_without_loss() {
    let tag = std::env::var("BRAIN_EVAL_IMAGE_TAG").unwrap_or_else(|_| "latest".to_string());
    let opts = DockerServerOpts {
        image_tag: tag,
        container_name: "brain-eval-drain".to_string(),
        data_port: 38082,
        metrics_port: 39093,
        data_volume: Some("brain-eval-drain-data".to_string()),
        health_timeout: Duration::from_secs(180),
        ..DockerServerOpts::default()
    };

    let outcome = graceful_shutdown_drain(opts).await;
    let tag = if outcome.advisory {
        "OBSERVE"
    } else if outcome.passed {
        "PASS"
    } else {
        "FAIL"
    };
    println!("  [{tag}] {} — {}", outcome.name, outcome.detail);
    // Only a broken safe invariant (data loss) fails the run; the drain
    // observation is advisory.
    assert!(outcome.passed || outcome.advisory, "{}", outcome.detail);
}
