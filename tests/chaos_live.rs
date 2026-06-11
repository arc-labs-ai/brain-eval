//! E4 kill-during-write chaos, live against a full-stack server.
//!
//! Boots the production image on a **persistent** data volume, SIGKILLs it
//! mid-write across a couple of cycles, and asserts every ACKed write
//! survives every restart (WAL-before-ack, invariant #1; recovery
//! idempotency across repeated crashes).
//!
//! `#[ignore]`d: needs Docker + a `brain:<tag>` image + the `brain-models`
//! volume. Run:
//!
//! ```bash
//! cargo test --test chaos_live -- --ignored --nocapture
//! ```
//!
//! Override the image tag with `BRAIN_EVAL_IMAGE_TAG` (default `latest`).

#![cfg(not(miri))]

use std::time::Duration;

use brain_eval::run::server::{remove_volume, DockerServerOpts};
use brain_eval::system::kill_during_write;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires docker + a brain:<tag> image + the brain-models volume"]
async fn kill_during_write_loses_no_acked_data() {
    let tag = std::env::var("BRAIN_EVAL_IMAGE_TAG").unwrap_or_else(|_| "latest".to_string());
    let volume = "brain-eval-chaos-data".to_string();
    // Best-effort: clear any volume left by a prior aborted run.
    remove_volume(&volume).await;

    let opts = DockerServerOpts {
        image_tag: tag,
        container_name: "brain-eval-chaos".to_string(),
        data_port: 38083,
        metrics_port: 39094,
        data_volume: Some(volume),
        health_timeout: Duration::from_secs(180),
        ..DockerServerOpts::default()
    };

    let outcome = kill_during_write(opts).await;
    println!(
        "  [{}] {} — {}",
        if outcome.passed { "PASS" } else { "FAIL" },
        outcome.name,
        outcome.detail
    );
    assert!(outcome.passed, "chaos gate failed: {}", outcome.detail);
}
