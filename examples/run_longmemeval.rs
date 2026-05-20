//! End-to-end runner for the LongMemEval-S benchmark.
//!
//! ## What it does
//!
//! 1. Builds a [`RunConfig`] from env vars (with a sensible default
//!    endpoint).
//! 2. Loads LongMemEval-S from `$BRAIN_EVAL_DATASETS_DIR/longmemeval/longmemeval_s.json`.
//! 3. Drives the full ingest → recall → judge loop against a running
//!    `brain-server`.
//! 4. Writes a JSON sidecar + text summary to
//!    `$BRAIN_EVAL_OUTPUT_DIR` (default `target/eval-reports/`).
//! 5. Prints the headline accuracy + report paths to stdout.
//!
//! ## Running
//!
//! ```bash
//! # 1. Start a brain-server somewhere
//! cargo run --bin brain-server --manifest-path crates/brain-server/Cargo.toml
//!
//! # 2. Point at your dataset directory and (optionally) cap question count
//! export BRAIN_EVAL_DATASETS_DIR=/path/to/datasets
//! export BRAIN_EVAL_ENDPOINT=127.0.0.1:7878
//! export BRAIN_EVAL_MAX_QUESTIONS=10        # smoke run
//!
//! # 3. Drive it
//! cargo run --release --example run_longmemeval \
//!   --manifest-path crates/brain-eval/Cargo.toml
//! ```
//!
//! ## What "honest" means here
//!
//! LongMemEval expects free-form natural-language answers; the
//! heuristic judge wired today is a directional signal, not a number
//! you should compare to GPT-4o or Zep. The JSON report carries
//! `meta.judge_type = "heuristic"` so consumers can see what they're
//! looking at. The LLM judge follow-up is what turns these numbers
//! into something publishable.

use std::error::Error;
use std::net::SocketAddr;
use std::process::ExitCode;

use brain_eval::datasets::longmemeval::LongMemEvalS;
use brain_eval::report::longmemeval_s_competitor_baselines;
use brain_eval::run::{EvalRunner, RunConfig};

const DEFAULT_ENDPOINT: &str = "127.0.0.1:7878";

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    // tracing-subscriber is intentionally NOT wired in this example
    // to keep the dep surface minimal. Set `RUST_LOG=brain_eval=warn`
    // and pipe through `env_logger` in a future wrapper if you want
    // structured progress logs.
    match try_main().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            let mut src = e.source();
            while let Some(inner) = src {
                eprintln!("  caused by: {inner}");
                src = inner.source();
            }
            ExitCode::from(1)
        }
    }
}

async fn try_main() -> Result<(), Box<dyn Error>> {
    let default_endpoint: SocketAddr = DEFAULT_ENDPOINT
        .parse()
        .expect("default endpoint must parse");
    let config = RunConfig::from_env(default_endpoint)?;

    eprintln!("brain-eval :: LongMemEval-S");
    eprintln!("  endpoint    : {}", config.endpoint);
    eprintln!(
        "  max_q       : {}",
        config
            .max_questions
            .map_or_else(|| "all".to_owned(), |n| n.to_string())
    );
    eprintln!("  top_k       : {}", config.top_k_retrieve);
    eprintln!("  output_dir  : {}", config.output_dir.display());
    eprintln!();

    let runner = EvalRunner::new(config, longmemeval_s_competitor_baselines);
    let report = runner.run(&LongMemEvalS).await?;

    println!();
    println!("=== LongMemEval-S — heuristic judge ===");
    println!(
        "instances          : {}",
        report.metrics.total_questions
    );
    println!("accuracy           : {:.4}", report.metrics.accuracy);
    println!(
        "  correct/partial/incorrect : {}/{}/{}",
        report.metrics.correct, report.metrics.partial, report.metrics.incorrect
    );
    println!(
        "ingestion errors   : {}    retrieval errors : {}",
        report.metrics.ingestion_errors, report.metrics.retrieval_errors
    );
    println!(
        "write p50/p95 (ms) : {}/{}     read p50/p95 (ms) : {}/{}",
        report.metrics.latency.write_p50_ms,
        report.metrics.latency.write_p95_ms,
        report.metrics.latency.read_p50_ms,
        report.metrics.latency.read_p95_ms,
    );
    if let Some(r) = &report.metrics.retrieval {
        println!(
            "Recall@5 / @10     : {:.4} / {:.4}",
            r.recall_at_5, r.recall_at_10
        );
    }
    println!();
    println!("Reports written under: {}", config_output_dir(&report.meta));
    println!(
        "Tip: numbers from a heuristic judge are directional. Wire the LLM \
         judge before quoting them in a comparison."
    );

    Ok(())
}

fn config_output_dir(meta: &brain_eval::report::BenchmarkMeta) -> String {
    // EvalRunner names files as `<benchmark_id>-<run_started_unix_nanos>.<ext>`
    // — surface the stem so the user can `cat` the JSON sidecar directly.
    format!(
        "(filename stem: {}-{})",
        meta.benchmark_id, meta.run_started_unix_nanos
    )
}
