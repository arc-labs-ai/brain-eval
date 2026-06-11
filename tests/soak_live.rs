//! E6 soak smoke, live against a full-stack server.
//!
//! Boots the production image and runs a short soak with RSS sampling from
//! the server's `/metrics`, asserting the run is healthy (no errors, recall
//! held, latency + RSS within tolerance). Absolute numbers are noisy on an
//! emulated box; the smoke proves the harness, the sampling, and the
//! report shape — the 48 h gate runs on reference hardware.
//!
//! `#[ignore]`d: needs Docker + a `brain:<tag>` image + the `brain-models`
//! volume. Run:
//!
//! ```bash
//! cargo test --test soak_live -- --ignored --nocapture
//! ```

#![cfg(not(miri))]

use std::time::Duration;

use brain_eval::run::server::{DockerServerOpts, ServerHandle};
use brain_eval::soak::{run_soak, SoakConfig};

#[tokio::test]
#[ignore = "requires docker + a brain:<tag> image + the brain-models volume"]
async fn soak_smoke_is_healthy() {
    let tag = std::env::var("BRAIN_EVAL_IMAGE_TAG").unwrap_or_else(|_| "latest".to_string());
    let metrics_port = 39095;
    let opts = DockerServerOpts {
        image_tag: tag,
        container_name: "brain-eval-soak".to_string(),
        data_port: 38084,
        metrics_port,
        health_timeout: Duration::from_secs(180),
        ..DockerServerOpts::default()
    };

    let server = ServerHandle::start_docker(opts)
        .await
        .expect("full-stack server should boot and become healthy");

    let metrics_addr = format!("127.0.0.1:{metrics_port}").parse().unwrap();
    let cfg = SoakConfig::smoke().with_metrics(metrics_addr);
    let report = run_soak(server.endpoint(), &cfg).await;
    server.stop().await;

    let report = report.expect("soak run should complete");
    println!("{}", report.to_text());

    // RSS must have been sampled (proves the /metrics scrape works live).
    assert!(
        report.samples.iter().any(|s| s.rss_bytes.is_some()),
        "expected RSS to be scraped from /metrics"
    );
    assert!(
        report.healthy(),
        "soak smoke was not healthy:\n{}",
        report.to_text()
    );
}
