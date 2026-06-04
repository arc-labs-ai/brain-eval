//! `brain-eval` — run a benchmark against a live `brain-server`.
//!
//! ```text
//! brain-eval smoke                       # zero-config Recall@1 canary
//! brain-eval smoke --endpoint 127.0.0.1:9090
//! brain-eval dmr                         # needs BRAIN_EVAL_DATASETS_DIR
//! brain-eval longmemeval-s
//! brain-eval locomo
//! ```
//!
//! Env vars (parsed by [`RunConfig::from_env`]) still apply:
//! `BRAIN_EVAL_ENDPOINT`, `BRAIN_EVAL_MAX_QUESTIONS`, `BRAIN_EVAL_TOP_K`,
//! `BRAIN_EVAL_OUTPUT_DIR`, `BRAIN_EVAL_FORMATS`. An explicit
//! `--endpoint` flag overrides `BRAIN_EVAL_ENDPOINT`.

use std::net::SocketAddr;
use std::process::ExitCode;

use brain_eval::core::benchmark::Benchmark;
use brain_eval::datasets::{
    dmr::DmrBenchmark, locomo::LocomoBenchmark, longmemeval::LongMemEvalS, smoke::SmokeBenchmark,
};
use brain_eval::report::{
    dmr_competitor_baselines, locomo_competitor_baselines, longmemeval_s_competitor_baselines,
    smoke_competitor_baselines, CompetitorBaselines,
};
use brain_eval::run::{EvalRunner, RunConfig};

const DEFAULT_ENDPOINT: &str = "127.0.0.1:9090";

fn main() -> ExitCode {
    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: failed to build tokio runtime: {e}");
            return ExitCode::from(1);
        }
    };
    rt.block_on(async_main())
}

async fn async_main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || matches!(args[0].as_str(), "-h" | "--help" | "help") {
        print_usage();
        return ExitCode::SUCCESS;
    }

    let benchmark_name = args[0].as_str();
    let endpoint_override = parse_endpoint_flag(&args[1..]);

    // Resolve the benchmark + its competitor table.
    let (benchmark, competitors): (Box<dyn Benchmark>, CompetitorBaselines) = match benchmark_name {
        "smoke" => (Box::new(SmokeBenchmark), smoke_competitor_baselines),
        "dmr" => (Box::new(DmrBenchmark), dmr_competitor_baselines),
        "longmemeval-s" | "longmemeval" | "lme" => {
            (Box::new(LongMemEvalS), longmemeval_s_competitor_baselines)
        }
        "locomo" => (Box::new(LocomoBenchmark), locomo_competitor_baselines),
        other => {
            eprintln!("error: unknown benchmark '{other}'");
            print_usage();
            return ExitCode::from(2);
        }
    };

    // Build config: env first, then apply the --endpoint override.
    let default_endpoint: SocketAddr = match DEFAULT_ENDPOINT.parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: bad default endpoint: {e}");
            return ExitCode::from(1);
        }
    };
    let mut config = match RunConfig::from_env(default_endpoint) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: bad BRAIN_EVAL_ENDPOINT: {e}");
            return ExitCode::from(1);
        }
    };
    if let Some(ep) = endpoint_override {
        config.endpoint = ep;
    }

    println!(
        "brain-eval :: {} ({})",
        benchmark.display_name(),
        benchmark.id()
    );
    println!("  endpoint : {}", config.endpoint);
    println!();

    match EvalRunner::new(config, competitors).run(benchmark.as_ref()).await {
        Ok(report) => {
            print_summary(&report);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: eval run failed: {e}");
            ExitCode::from(1)
        }
    }
}

/// Pull `--endpoint HOST:PORT` (or `--endpoint=HOST:PORT`) out of the
/// trailing args. Returns `None` when absent; exits the process is left
/// to the caller — a malformed value is reported and treated as absent
/// so the env/default still applies.
fn parse_endpoint_flag(rest: &[String]) -> Option<SocketAddr> {
    let mut it = rest.iter();
    while let Some(arg) = it.next() {
        let raw = if let Some(v) = arg.strip_prefix("--endpoint=") {
            Some(v.to_owned())
        } else if arg == "--endpoint" {
            it.next().cloned()
        } else {
            None
        };
        if let Some(v) = raw {
            match v.parse::<SocketAddr>() {
                Ok(a) => return Some(a),
                Err(e) => {
                    eprintln!("warning: ignoring bad --endpoint '{v}': {e}");
                    return None;
                }
            }
        }
    }
    None
}

fn print_summary(report: &brain_eval::report::BenchmarkReport) {
    let m = &report.metrics;
    println!("=== {} ===", report.meta.benchmark_name);
    println!("questions : {}", m.total_questions);
    println!(
        "accuracy  : {:.4}   ({}/{}/{} correct/partial/incorrect)",
        m.accuracy, m.correct, m.partial, m.incorrect
    );
    if let Some(r) = &m.retrieval {
        println!(
            "recall@1  : {:.4}    recall@5 : {:.4}    recall@10 : {:.4}",
            r.recall_at_1, r.recall_at_5, r.recall_at_10
        );
    }
    println!(
        "latency   : write p50/p95 {}/{} ms    read p50/p95 {}/{} ms",
        m.latency.write_p50_ms, m.latency.write_p95_ms, m.latency.read_p50_ms, m.latency.read_p95_ms,
    );
}

fn print_usage() {
    eprintln!(
        "usage: brain-eval <benchmark> [--endpoint HOST:PORT]\n\
         \n\
         benchmarks:\n\
         \x20 smoke           compiled-in Aurora corpus (zero config; Recall@1 canary)\n\
         \x20 dmr             DMR / MemGPT (needs BRAIN_EVAL_DATASETS_DIR)\n\
         \x20 longmemeval-s   LongMemEval-S (needs BRAIN_EVAL_DATASETS_DIR)\n\
         \x20 locomo          LoCoMo (needs BRAIN_EVAL_DATASETS_DIR)\n\
         \n\
         env: BRAIN_EVAL_ENDPOINT, BRAIN_EVAL_MAX_QUESTIONS, BRAIN_EVAL_TOP_K,\n\
         \x20    BRAIN_EVAL_OUTPUT_DIR, BRAIN_EVAL_FORMATS"
    );
}
