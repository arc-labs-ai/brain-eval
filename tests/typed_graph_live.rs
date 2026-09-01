//! E2 typed-graph functional acceptance, live against a full-stack server.
//!
//! Boots the production image (E1 `ServerHandle::start_docker` → full
//! capability stack: embed + reranker + gliner from the `brain-models`
//! volume) and runs the typed-graph scenarios (schema / entity / statement /
//! relation / extraction) against it.
//!
//! Two modes, selected by `BRAIN_EVAL_ENDPOINT`:
//!
//! - **unset** — boots its own throwaway `brain:<tag>` container (the CI
//!   shape). Needs Docker + a `brain:<tag>` image + the `brain-models` volume
//!   (provisioned by brain-db's `.devcontainer/bootstrap-model.sh`). Override
//!   the image tag with `BRAIN_EVAL_IMAGE_TAG` (default `latest`).
//! - **set** — runs the same scenarios against that already-running server
//!   (no container is booted), minting keys through its admin plane
//!   (`BRAIN_EVAL_ADMIN_SECRET` / `BRAIN_EVAL_ADMIN_PORT`).
//!
//! `#[ignore]`d. Run:
//!
//! ```bash
//! # docker mode
//! cargo test --test typed_graph_live -- --ignored --nocapture
//! # external mode
//! BRAIN_EVAL_ENDPOINT=127.0.0.1:9090 BRAIN_EVAL_ADMIN_SECRET=eval-admin-secret \
//!   cargo test --test typed_graph_live -- --ignored --nocapture
//! ```

#![cfg(not(miri))]

use std::time::Duration;

use brain_eval::blackbox::system::run_typed_graph_scenarios;
use brain_eval::client::server::{DockerServerOpts, ServerHandle};

#[tokio::test]
#[ignore = "requires docker + a brain:<tag> image + the brain-models volume"]
async fn typed_graph_scenarios_all_pass() {
    let tag = std::env::var("BRAIN_EVAL_IMAGE_TAG").unwrap_or_else(|_| "latest".to_string());
    // Distinct ports/name so this never collides with other eval containers.
    let opts = DockerServerOpts {
        image_tag: tag,
        container_name: "brain-eval-typed-graph".to_string(),
        data_port: 38080,
        metrics_port: 39091,
        health_timeout: Duration::from_secs(180),
        ..DockerServerOpts::default()
    };

    let server = ServerHandle::from_env_or_docker(opts)
        .await
        .expect("server should be reachable (external) or boot and become healthy (docker)");

    let outcomes = run_typed_graph_scenarios(server.endpoint()).await;
    server.stop().await;

    assert!(
        !outcomes.is_empty(),
        "expected typed-graph scenarios to run"
    );
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
    assert!(
        failed.is_empty(),
        "typed-graph scenarios failed: {failed:?}"
    );
}
