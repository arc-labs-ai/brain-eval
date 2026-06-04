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
use brain_eval::acceptance::{run_acceptance, AcceptanceConfig};
use brain_eval::run::{EvalRunner, RunConfig};
use brain_eval::soak::{run_soak, SoakConfig};

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

    // System commands that drive the server directly (not dataset evals).
    match benchmark_name {
        "acceptance" => return acceptance_cmd(resolve_endpoint(endpoint_override)).await,
        "soak" => return soak_cmd(resolve_endpoint(endpoint_override)).await,
        _ => {}
    }

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

/// Resolve the endpoint: explicit `--endpoint` flag, else
/// `BRAIN_EVAL_ENDPOINT`, else the default.
fn resolve_endpoint(override_: Option<SocketAddr>) -> SocketAddr {
    override_
        .or_else(|| {
            std::env::var("BRAIN_EVAL_ENDPOINT")
                .ok()
                .and_then(|s| s.parse().ok())
        })
        .unwrap_or_else(|| {
            DEFAULT_ENDPOINT
                .parse()
                .expect("invariant: default endpoint literal parses")
        })
}

/// `brain-eval acceptance` — run the v1.0 acceptance suite and gate on it.
/// Correctness gates must pass everywhere; performance gates are
/// informational off reference hardware. `BRAIN_EVAL_RESTART_RECOVERY=1`
/// adds the (slower, docker-booting) restart-recovery gate.
async fn acceptance_cmd(endpoint: SocketAddr) -> ExitCode {
    println!("brain-eval :: v1.0 acceptance");
    println!("  endpoint : {endpoint}");
    println!();

    let mut cfg = AcceptanceConfig::smoke(endpoint);
    cfg.run_restart_recovery = std::env::var("BRAIN_EVAL_RESTART_RECOVERY").as_deref() == Ok("1");

    let report = run_acceptance(cfg).await;
    print!("{}", report.to_text());

    if report.all_pass() {
        println!("acceptance: PASS (all gates, including performance)");
        ExitCode::SUCCESS
    } else if report.correctness_pass() {
        println!(
            "acceptance: correctness PASS; performance gates need reference \
             hardware (16c/64GB/NVMe) to be meaningful — see the report above"
        );
        ExitCode::SUCCESS
    } else {
        eprintln!("acceptance: FAIL — a correctness gate did not pass");
        ExitCode::from(1)
    }
}

/// `brain-eval soak` — sustained workload + drift sampling.
/// `BRAIN_EVAL_SOAK_SECS` sets the duration (default 5 s smoke).
async fn soak_cmd(endpoint: SocketAddr) -> ExitCode {
    let secs: u64 = std::env::var("BRAIN_EVAL_SOAK_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    println!("brain-eval :: soak ({secs}s)");
    println!("  endpoint : {endpoint}");
    println!();

    let mut cfg = SoakConfig::smoke();
    cfg.duration = std::time::Duration::from_secs(secs);

    match run_soak(endpoint, &cfg).await {
        Ok(report) => {
            print!("{}", report.to_text());
            if report.healthy() {
                ExitCode::SUCCESS
            } else {
                eprintln!("soak: FAIL — errors or recall drift detected");
                ExitCode::from(1)
            }
        }
        Err(e) => {
            eprintln!("error: soak run failed: {e}");
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
         system commands:\n\
         \x20 acceptance      v1.0 acceptance suite (latency/throughput/recall/scenarios)\n\
         \x20                 BRAIN_EVAL_RESTART_RECOVERY=1 adds the restart-recovery gate\n\
         \x20 soak            sustained workload + drift sampling (BRAIN_EVAL_SOAK_SECS)\n\
         \n\
         env: BRAIN_EVAL_ENDPOINT, BRAIN_EVAL_MAX_QUESTIONS, BRAIN_EVAL_TOP_K,\n\
         \x20    BRAIN_EVAL_OUTPUT_DIR, BRAIN_EVAL_FORMATS"
    );
}
