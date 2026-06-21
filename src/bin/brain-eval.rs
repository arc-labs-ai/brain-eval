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

use brain_eval::acceptance::{run_acceptance, AcceptanceConfig};
use brain_eval::core::benchmark::Benchmark;
use brain_eval::datasets::{
    dmr::DmrBenchmark, lexical_stress::LexicalStressBenchmark, locomo::LocomoBenchmark,
    longmemeval::LongMemEvalS, paraphrase_stress::ParaphraseStressBenchmark, smoke::SmokeBenchmark,
    supersession_stress::SupersessionStressBenchmark,
};
use brain_eval::report::{
    dmr_competitor_baselines, lexical_stress_competitor_baselines, locomo_competitor_baselines,
    longmemeval_s_competitor_baselines, paraphrase_stress_competitor_baselines,
    smoke_competitor_baselines, supersession_stress_competitor_baselines, CompetitorBaselines,
};
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

    // System / offline commands that don't run a dataset eval.
    match benchmark_name {
        "acceptance" => return acceptance_cmd(resolve_endpoint(endpoint_override)).await,
        "soak" => return soak_cmd(resolve_endpoint(endpoint_override)).await,
        // Recompute a full report from a streamed partial-results file —
        // recovers metrics from a run that was interrupted mid-flight.
        "summary" | "report" => return summary_cmd(&args[1..]),
        _ => {}
    }

    // Resolve the benchmark + its competitor table.
    let (benchmark, competitors): (Box<dyn Benchmark>, CompetitorBaselines) = match benchmark_name {
        "smoke" => (Box::new(SmokeBenchmark), smoke_competitor_baselines),
        "lexical-stress" => (
            Box::new(LexicalStressBenchmark),
            lexical_stress_competitor_baselines,
        ),
        "paraphrase-stress" => (
            Box::new(ParaphraseStressBenchmark),
            paraphrase_stress_competitor_baselines,
        ),
        "supersession-stress" => (
            Box::new(SupersessionStressBenchmark),
            supersession_stress_competitor_baselines,
        ),
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

    match EvalRunner::new(config, competitors)
        .run(benchmark.as_ref())
        .await
    {
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

/// `brain-eval summary <partial.jsonl>` — recompute the full metrics
/// from a streamed per-question results file. The runner streams every
/// graded question to `<output_dir>/<id>-<ts>.partial.jsonl` as it goes,
/// so even an interrupted run leaves a complete record to summarize.
fn summary_cmd(rest: &[String]) -> ExitCode {
    let Some(path) = rest.iter().find(|a| !a.starts_with("--")) else {
        eprintln!("usage: brain-eval summary <partial.jsonl>");
        return ExitCode::from(2);
    };
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: cannot read {path}: {e}");
            return ExitCode::from(1);
        }
    };
    let mut results = Vec::new();
    let mut unparseable = 0usize;
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<brain_eval::core::outcome::QuestionResult>(line) {
            Ok(r) => results.push(r),
            Err(_) => unparseable += 1,
        }
    }
    if results.is_empty() {
        eprintln!("error: no parseable question results in {path}");
        return ExitCode::from(1);
    }
    let metrics = brain_eval::score::metrics::compute_full_metrics(&results);
    println!("=== summary :: {path} ===");
    println!(
        "parsed    : {} graded questions ({unparseable} unparseable lines skipped)",
        results.len()
    );
    print_metrics(&metrics);
    ExitCode::SUCCESS
}

fn print_summary(report: &brain_eval::report::BenchmarkReport) {
    println!("=== {} ===", report.meta.benchmark_name);
    print_metrics(&report.metrics);
}

/// Render the headline metrics, the answer-shape breakdown (one memory
/// vs a set vs honest abstention — how the router actually behaves), and
/// the committed-answer precision (the hard invariant: never confidently
/// wrong).
fn print_metrics(m: &brain_eval::score::metrics::EvalMetrics) {
    println!("questions : {}", m.total_questions);
    println!(
        "accuracy  : {:.4}   ({}/{}/{} correct/partial/incorrect)",
        m.accuracy, m.correct, m.partial, m.incorrect
    );
    let s = &m.answer_shape;
    println!(
        "shape     : single {} (acc {:.3}) | many {} (acc {:.3}) | abstained {} (acc {:.3}) | errored {}",
        s.single, s.single_accuracy, s.many, s.many_accuracy, s.abstained, s.abstained_accuracy, s.errored,
    );
    println!(
        "precision : {:.4}   ({} committed answers; fraction never scored incorrect)",
        s.committed_precision, s.committed
    );
    // Headline retrieval metric: did Brain surface the supporting context?
    // (LLM-judged, semantic — see Kamalloo 2023 / NoLiMa.)
    if let Some(c) = &m.context_recall {
        println!(
            "ctx-recall: {:.4}   ({}/{} answer-supporting; LLM-judged headline)",
            c.supported_rate, c.n_supported, c.n_judged
        );
    } else {
        println!("ctx-recall: n/a    (no LLM judge configured — run with --features live-llm)");
    }
    // Substring recall@k is a DEPRECATED diagnostic kept only to contrast
    // with ctx-recall; it rewards lexical overlap, not retrieval.
    if let Some(r) = &m.retrieval {
        println!(
            "recall@k  : @1 {:.4}   @5 {:.4}   @10 {:.4}   (DEPRECATED substring diagnostic)",
            r.recall_at_1, r.recall_at_5, r.recall_at_10
        );
    }
    println!(
        "latency   : write p50/p95 {}/{} ms    read p50/p95 {}/{} ms",
        m.latency.write_p50_ms,
        m.latency.write_p95_ms,
        m.latency.read_p50_ms,
        m.latency.read_p95_ms,
    );
}

fn print_usage() {
    eprintln!(
        "usage: brain-eval <benchmark> [--endpoint HOST:PORT]\n\
         \n\
         benchmarks:\n\
         \x20 smoke           compiled-in Aurora corpus (zero config; Recall@1 canary)\n\
         \x20 lexical-stress  compiled-in no-overlap set (proves semantic, not substring, retrieval)\n\
         \x20 paraphrase-stress    compiled-in generated no-overlap triples + distractors (anti-overfit)\n\
         \x20 supersession-stress  compiled-in generated OLD->NEW updates (current-vs-prior direction)\n\
         \x20 dmr             DMR / MemGPT (needs BRAIN_EVAL_DATASETS_DIR)\n\
         \x20 longmemeval-s   LongMemEval-S (needs BRAIN_EVAL_DATASETS_DIR)\n\
         \x20 locomo          LoCoMo (needs BRAIN_EVAL_DATASETS_DIR)\n\
         \n\
         system commands:\n\
         \x20 acceptance      v1.0 acceptance suite (latency/throughput/recall/scenarios)\n\
         \x20                 BRAIN_EVAL_RESTART_RECOVERY=1 adds the restart-recovery gate\n\
         \x20 soak            sustained workload + drift sampling (BRAIN_EVAL_SOAK_SECS)\n\
         \x20 summary <file>  recompute metrics from a streamed .partial.jsonl\n\
         \x20                 (recover results from an interrupted run)\n\
         \n\
         env: BRAIN_EVAL_ENDPOINT, BRAIN_EVAL_MAX_QUESTIONS, BRAIN_EVAL_TOP_K,\n\
         \x20    BRAIN_EVAL_OUTPUT_DIR, BRAIN_EVAL_FORMATS, BRAIN_EVAL_QUESTION_TYPES,\n\
         \x20    BRAIN_EVAL_EXTRACTION_DRAIN"
    );
}
