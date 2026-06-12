//! E7 recall-quality regression guard, live against a full-stack server.
//!
//! Boots the production image, measures known-answer recall@1 / recall@K,
//! and compares against the committed baseline
//! (`baselines/recall_quality.txt`). A drop of more than `TOLERANCE` below
//! the baseline on either metric fails the build — this is the perf/recall
//! regression gate the CI live tier runs.
//!
//! The committed baseline is a deliberately **conservative floor**, not a
//! best-observed value: with the cross-encoder reranker on, known-answer
//! recall@1 over the (near-duplicate) probe texts swings run-to-run
//! (~0.84-0.96 at the small smoke N), while recall@K stays at 1.0. A floor,
//! with a tolerance wide enough to absorb that noise, keeps the gate from
//! flapping while still catching a real collapse. Re-bless to a tighter
//! measured baseline from a stable run (larger N is cheap on x86_64 CI;
//! it's slow only under emulation) with:
//!
//! ```bash
//! BRAIN_EVAL_BLESS_BASELINE=1 cargo test --test regression_live -- --ignored --nocapture
//! ```
//!
//! `#[ignore]`d: needs Docker + a `brain:<tag>` image + the `brain-models`
//! volume.

#![cfg(not(miri))]

use std::time::Duration;

use brain_db_sdk::{BrainClient, ClientConfig};
use brain_eval::run::server::{DockerServerOpts, ServerHandle};
use brain_eval::scale::{no_regression, run_recall_quality, RecallQualityReport, RecallTargets};

const BASELINE_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/baselines/recall_quality.txt");
/// Allowed absolute drop below baseline before the gate fails. Wide enough
/// to absorb the reranker's run-to-run recall@1 jitter at the smoke N, so a
/// real recall collapse is what trips it — not noise.
const TOLERANCE: f64 = 0.10;
const RECALL_N: usize = 50;
const RECALL_TOP_K: u32 = 10;
/// The reranker makes a single recall slow on emulated hardware; a generous
/// per-request deadline keeps the probe from tripping the SDK's 30s default
/// (on x86_64 CI this never binds).
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

#[tokio::test]
#[ignore = "requires docker + a brain:<tag> image + the brain-models volume"]
async fn recall_quality_no_regression() {
    let tag = std::env::var("BRAIN_EVAL_IMAGE_TAG").unwrap_or_else(|_| "latest".to_string());
    let opts = DockerServerOpts {
        image_tag: tag,
        container_name: "brain-eval-regression".to_string(),
        data_port: 38086,
        metrics_port: 39097,
        health_timeout: Duration::from_secs(180),
        ..DockerServerOpts::default()
    };

    let server = ServerHandle::start_docker(opts)
        .await
        .expect("full-stack server should boot and become healthy");

    let current = {
        let config = ClientConfig {
            request_timeout: Some(REQUEST_TIMEOUT),
            ..ClientConfig::default()
        };
        let client = BrainClient::connect_with(server.endpoint(), config)
            .await
            .expect("connect");
        let salt = hex16(client.agent_id());
        let report = run_recall_quality(
            &client,
            RECALL_N,
            RECALL_TOP_K,
            &salt,
            &RecallTargets::default(),
        )
        .await
        .expect("recall-quality probe");
        let _ = client.close().await;
        report
    };
    server.stop().await;

    println!("{}", current.to_text());

    if std::env::var("BRAIN_EVAL_BLESS_BASELINE").as_deref() == Ok("1") {
        write_baseline(&current);
        println!("baseline blessed → {BASELINE_PATH}");
        return;
    }

    let baseline = read_baseline();
    assert!(
        no_regression(&baseline, &current, TOLERANCE),
        "recall regressed beyond {TOLERANCE} vs baseline: baseline @1={:.4} @k={:.4}, \
         current @1={:.4} @k={:.4}",
        baseline.recall_at_1,
        baseline.recall_at_k,
        current.recall_at_1,
        current.recall_at_k,
    );
}

/// A baseline report carrying only the two metrics `no_regression` reads.
fn read_baseline() -> RecallQualityReport {
    let raw = std::fs::read_to_string(BASELINE_PATH).unwrap_or_else(|e| {
        panic!(
            "missing baseline {BASELINE_PATH} ({e}); regenerate with \
             BRAIN_EVAL_BLESS_BASELINE=1"
        )
    });
    let mut at_1 = None;
    let mut at_k = None;
    for line in raw.lines() {
        let line = line.trim();
        if let Some(v) = line.strip_prefix("recall_at_1=") {
            at_1 = v.trim().parse::<f64>().ok();
        } else if let Some(v) = line.strip_prefix("recall_at_k=") {
            at_k = v.trim().parse::<f64>().ok();
        }
    }
    let (recall_at_1, recall_at_k) = (
        at_1.expect("baseline recall_at_1"),
        at_k.expect("baseline recall_at_k"),
    );
    RecallQualityReport {
        queries: RECALL_N,
        top_k: RECALL_TOP_K,
        recall_at_1,
        recall_at_k,
        target_at_1: recall_at_1,
        target_at_k: recall_at_k,
    }
}

fn write_baseline(r: &RecallQualityReport) {
    let body = format!(
        "# brain-eval recall-quality baseline (n={}, top_k={})\n\
         recall_at_1={:.4}\n\
         recall_at_k={:.4}\n",
        RECALL_N, RECALL_TOP_K, r.recall_at_1, r.recall_at_k
    );
    if let Some(dir) = std::path::Path::new(BASELINE_PATH).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(BASELINE_PATH, body).expect("write baseline");
}

fn hex16(id: [u8; 16]) -> String {
    let mut s = String::with_capacity(32);
    for b in id {
        s.push_str(&format!("{b:02x}"));
    }
    s
}
