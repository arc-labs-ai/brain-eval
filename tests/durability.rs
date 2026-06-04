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

use brain_eval::run::server::DockerServerOpts;
use brain_eval::system::restart_recovery;

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
