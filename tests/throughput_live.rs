//! E3 concurrent-throughput probe, live against a full-stack server.
//!
//! Boots the production image (E1 `ServerHandle::start_docker` → full
//! capability stack) and drives each verb (ENCODE / STATEMENT_CREATE /
//! RELATION_CREATE / QUERY / ENTITY_RESOLVE) from N concurrent connections
//! for a fixed window.
//!
//! The assertion is the **correctness** half of the probe — the server
//! handled every concurrent op with zero errors. The achieved ops/s is
//! printed for visibility but not asserted: absolute throughput is only
//! meaningful on quiet reference hardware, and this runs under an emulated
//! dev container.
//!
//! `#[ignore]`d: needs Docker + a `brain:<tag>` image + the `brain-models`
//! volume. Run:
//!
//! ```bash
//! cargo test --test throughput_live -- --ignored --nocapture
//! ```
//!
//! Override the image tag with `BRAIN_EVAL_IMAGE_TAG` (default `latest`).

#![cfg(not(miri))]

use std::time::Duration;

use brain_eval::run::server::{DockerServerOpts, ServerHandle};
use brain_eval::scale::{run_concurrent_throughput, ConcurrentConfig};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires docker + a brain:<tag> image + the brain-models volume"]
async fn concurrent_throughput_no_errors() {
    let tag = std::env::var("BRAIN_EVAL_IMAGE_TAG").unwrap_or_else(|_| "latest".to_string());
    let opts = DockerServerOpts {
        image_tag: tag,
        container_name: "brain-eval-throughput".to_string(),
        data_port: 38081,
        metrics_port: 39092,
        health_timeout: Duration::from_secs(180),
        ..DockerServerOpts::default()
    };

    let server = ServerHandle::start_docker(opts)
        .await
        .expect("full-stack server should boot and become healthy");

    // A modest concurrent load — enough to exercise group commit and the
    // connection layer without saturating a laptop's Docker VM.
    let cfg = ConcurrentConfig {
        clients: 24,
        window: Duration::from_secs(2),
        query_corpus: 48,
        ..ConcurrentConfig::default()
    };

    let report = run_concurrent_throughput(server.endpoint(), &cfg).await;
    server.stop().await;

    let report = report.expect("concurrent throughput run should complete");
    println!("{}", report.to_text());

    assert!(
        report.no_errors(),
        "concurrent ops must all succeed (any error is a real defect): {}",
        report.error_summary()
    );
}
