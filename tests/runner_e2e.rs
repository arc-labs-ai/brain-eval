//! End-to-end test of the full `EvalRunner` against a real running
//! `brain-server`. Ignored by default — requires `BRAIN_EVAL_ENDPOINT`
//! to point at a live server. Run with:
//!
//! ```bash
//! BRAIN_EVAL_ENDPOINT=127.0.0.1:9090 \
//!   cargo test --test runner_e2e -- --ignored --nocapture
//! ```
//!
//! Unlike `basic_e2e` (which drives the harness directly), this runs
//! the complete runner: load → group → ingest-once → recall →
//! synthesize → judge → aggregate, on the compiled-in smoke corpus.
//! It asserts the run is structurally sound and prints the headline
//! numbers (Recall@1 / @5 / accuracy) so a human reading `--nocapture`
//! sees Brain's actual retrieval quality. A hard target lives in the
//! regression tier (follow-up); this tier proves the real vertical
//! works and produces non-degenerate numbers.

use std::time::Duration;

use brain_eval::client::server::{DockerServerOpts, ServerHandle};

use brain_eval::bench::datasets::smoke::SmokeBenchmark;
use brain_eval::bench::report::smoke_competitor_baselines;
use brain_eval::bench::{EvalRunner, ReporterKind, RunConfig};

/// Attach to `BRAIN_EVAL_ENDPOINT` when the operator set one, otherwise boot
/// a throwaway `brain:<tag>` container for this suite.
///
/// This used to return `Option` and the test returned early when it was
/// `None` — which reported a PASS without contacting a server at all. A suite
/// that goes green having exercised nothing is worse than one that fails:
/// it hides every regression it was written to catch. Booting is the same
/// path the other live suites take, so "no endpoint configured" is now a
/// slower run rather than a silent skip.
fn server_opts() -> DockerServerOpts {
    DockerServerOpts {
        container_name: "brain-eval-runner".to_string(),
        data_port: 38092,
        metrics_port: 39112,
        admin_port: 39122,
        health_timeout: Duration::from_secs(180),
        ..DockerServerOpts::default()
    }
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires a running brain-server; set BRAIN_EVAL_ENDPOINT"]
async fn smoke_benchmark_runs_against_live_server() {
    let server = ServerHandle::from_env_or_docker(server_opts())
        .await
        .expect("server should be reachable (external) or boot and become healthy (docker)");
    let addr = server.endpoint();

    let mut config = RunConfig::default_for(addr);
    // No report files written during the test.
    config.reporters = Vec::<ReporterKind>::new();

    let report = EvalRunner::new(config, smoke_competitor_baselines)
        .run(&SmokeBenchmark)
        .await
        .expect("smoke benchmark should run end-to-end against the server");

    let m = &report.metrics;
    assert_eq!(m.total_questions, 12, "all 12 smoke questions ran");
    assert_eq!(m.ingestion_errors, 0, "no ingest failures expected");
    assert_eq!(m.retrieval_errors, 0, "no recall failures expected");

    let r = m
        .retrieval
        .as_ref()
        .expect("retrieval stats present (the corpus is non-empty)");

    eprintln!("=== smoke @ {addr} ===");
    eprintln!("accuracy  : {:.4}", m.accuracy);
    eprintln!(
        "recall@1  : {:.4}    recall@5 : {:.4}    recall@10 : {:.4}",
        r.recall_at_1, r.recall_at_5, r.recall_at_10
    );

    // The smoke corpus is tiny and each gold is distinctive, so a
    // healthy substrate should surface every gold within the top 5.
    // Recall@1 is the rerank-sensitive number — printed, not gated —
    // because it depends on the server's rerank config.
    assert!(
        r.recall_at_5 >= 0.9,
        "recall@5 should be ≥ 0.9 on the smoke corpus, got {:.4}",
        r.recall_at_5,
    );
}
