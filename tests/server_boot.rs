//! Phase 1 integration test for [`ServerHandle::start_docker`].
//!
//! Boots the production image (`brain:<tag>`) in a container, waits for
//! its healthcheck, drives one encode → recall round-trip through the
//! eval harness, then tears the container down.
//!
//! `#[ignore]`d: needs Docker + a `brain:<tag>` image + the bootstrapped
//! BGE model on disk. Run explicitly:
//!
//! ```bash
//! cargo test --test server_boot -- --ignored --nocapture
//! ```
//!
//! Override the image tag with `BRAIN_EVAL_IMAGE_TAG` (default `latest`).

#![cfg(not(miri))]

use std::time::Duration;

use brain_eval::run::harness::BrainEvalHarness;
use brain_eval::run::server::{DockerServerOpts, ServerHandle};

#[tokio::test]
#[ignore = "requires docker + a brain:<tag> image + the BGE model on disk"]
async fn start_docker_boots_a_usable_server() {
    let tag = std::env::var("BRAIN_EVAL_IMAGE_TAG").unwrap_or_else(|_| "latest".to_string());
    // Non-default ports so the test never collides with a `just
    // serve-local` / `brain-local` container already on 18080/19091.
    let opts = DockerServerOpts {
        image_tag: tag,
        container_name: "brain-eval-boot-test".to_string(),
        data_port: 28080,
        metrics_port: 29091,
        health_timeout: Duration::from_secs(180),
        ..DockerServerOpts::default()
    };

    let server = ServerHandle::start_docker(opts)
        .await
        .expect("server should boot and become healthy");

    // Drive a real round-trip through the harness against the booted server.
    let harness = BrainEvalHarness::connect(server.endpoint())
        .await
        .expect("connect to booted server");

    let request =
        brain_db_sdk::EncodeBuilder::new("the sky over Lisbon turned amber at dusk").build();
    harness
        .client()
        .encode(&request)
        .await
        .expect("encode against booted server");

    let recall = harness
        .recall("amber dusk over Lisbon", 5)
        .await
        .expect("recall against booted server");
    assert!(
        !recall.memories.is_empty(),
        "expected the just-encoded memory to be recallable"
    );

    harness.close().await.expect("clean harness close");
    server.stop().await;
}
