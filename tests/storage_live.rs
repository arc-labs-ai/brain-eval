//! E6 storage-footprint gate, live against a full-stack server.
//!
//! Boots the production image on a fresh volume, ingests a batch of
//! memories, and reads the per-shard storage gauges off `/metrics`
//! (`brain_wal_size_bytes`, `brain_metadata_size_bytes`, the arena gauges),
//! asserting they emit and the accounting is sane. The byte budget is
//! informational at this small scale (footprint is dominated by a
//! preallocated WAL segment + redb's initial pages); the 1M-scale budget
//! gate runs on reference hardware via `AcceptanceConfig::full_scale`.
//!
//! `#[ignore]`d: needs Docker + a `brain:<tag>` image + the `brain-models`
//! volume. Run:
//!
//! ```bash
//! cargo test --test storage_live -- --ignored --nocapture
//! ```

#![cfg(not(miri))]

use std::time::Duration;

use brain_eval::run::server::{remove_volume, DockerServerOpts};
use brain_eval::system::storage_footprint;

#[tokio::test]
#[ignore = "requires docker + a brain:<tag> image + the brain-models volume"]
async fn storage_gauges_live_and_sane() {
    let tag = std::env::var("BRAIN_EVAL_IMAGE_TAG").unwrap_or_else(|_| "latest".to_string());
    let volume = "brain-eval-storage-data".to_string();
    remove_volume(&volume).await;

    let opts = DockerServerOpts {
        image_tag: tag,
        container_name: "brain-eval-storage".to_string(),
        data_port: 38085,
        metrics_port: 39096,
        data_volume: Some(volume),
        health_timeout: Duration::from_secs(180),
        ..DockerServerOpts::default()
    };

    // Small batch + informational budget (None) for the dev-box smoke.
    let outcome = storage_footprint(opts, 500, None).await;
    println!(
        "  [{}] {} — {}",
        if outcome.passed { "PASS" } else { "FAIL" },
        outcome.name,
        outcome.detail
    );
    assert!(outcome.passed, "storage gate failed: {}", outcome.detail);
}
