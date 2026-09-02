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

use brain_eval::bench::core::benchmark::Benchmark;
use brain_eval::bench::datasets::{
    dmr::DmrBenchmark, lexical_stress::LexicalStressBenchmark, locomo::LocomoBenchmark,
    longmemeval::LongMemEvalS, paraphrase_stress::ParaphraseStressBenchmark, smoke::SmokeBenchmark,
    supersession_stress::SupersessionStressBenchmark,
};
use brain_eval::bench::report::{
    dmr_competitor_baselines, lexical_stress_competitor_baselines, locomo_competitor_baselines,
    longmemeval_s_competitor_baselines, paraphrase_stress_competitor_baselines,
    smoke_competitor_baselines, supersession_stress_competitor_baselines, CompetitorBaselines,
};
use brain_eval::bench::{EvalRunner, RunConfig};
use brain_eval::blackbox::acceptance::{run_acceptance, AcceptanceConfig};
use brain_eval::blackbox::soak::{run_soak, SoakConfig};

const DEFAULT_ENDPOINT: &str = "127.0.0.1:9090";

fn main() -> ExitCode {
    init_tracing();

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

/// Install the tracing subscriber. Without this, every `tracing::warn!`/
/// `info!`/`debug!` call in the library is silently dropped by the
/// default no-op dispatcher — this is the one place that turns those
/// calls into actual diagnostics. Writes to stderr so it never interleaves
/// with the report output on stdout. `RUST_LOG` controls verbosity
/// (defaults to `info` when unset or invalid).
fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
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
        "probe" => {
            // `probe` alone → single controlled pair; `probe <benchmark>` →
            // staged per-stage run over that benchmark.
            let staged = args.get(1).filter(|a| !a.starts_with("--")).cloned();
            // Connect to a running server by default (the `just docker-serve`
            // Brain on the persistent `brain-data` volume, at DEFAULT_ENDPOINT
            // 127.0.0.1:9090, or `--endpoint` / BRAIN_EVAL_ENDPOINT). Spinning
            // up a throwaway `brain:latest` container writes to an anonymous
            // volume that vanishes on teardown, so it is now opt-in only —
            // `--self-boot` or BRAIN_EVAL_SELF_BOOT=1 — never the default.
            let self_boot = parse_bool_flag(&args[1..], "--self-boot")
                || matches!(
                    std::env::var("BRAIN_EVAL_SELF_BOOT").ok().as_deref(),
                    Some("1" | "true" | "yes" | "on")
                );
            let external_endpoint = if self_boot {
                None
            } else {
                Some(resolve_endpoint(endpoint_override))
            };
            return probe_cmd(staged.as_deref(), external_endpoint).await;
        }
        "acceptance" => {
            let full =
                parse_bool_flag(&args[1..], "--full") || parse_bool_flag(&args[1..], "--scale");
            return acceptance_cmd(resolve_endpoint(endpoint_override), full).await;
        }
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
    // `--only-conversation <id>` / `--verify-memory-list` override the
    // env-derived config, so the flag wins over BRAIN_EVAL_*.
    if let Some(id) = parse_only_conversation_flag(&args[1..]) {
        config.only_conversation = Some(id);
    }
    if parse_bool_flag(&args[1..], "--verify-memory-list") {
        config.verify_memory_list = true;
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

/// `brain-eval probe` — one controlled write + one controlled read, with
/// a stage-by-stage "what we wanted vs what we got" table. The system
/// smoke test before a full eval: confirms the pipeline works end to end
/// and that the per-stage journal (`brain_debug::stage`) is wired.
/// `probe` — targets an already-running server by default (DEFAULT_ENDPOINT
/// `127.0.0.1:9090`, i.e. the `just docker-serve` Brain on the persistent
/// `brain-data` volume; override with `--endpoint` / `BRAIN_EVAL_ENDPOINT`).
/// Self-booting a throwaway `brain:latest` container (full pipeline, models
/// from the `brain-models` volume, LLM key from `BRAIN__LLM__API_KEY`, torn
/// down after) is opt-in only — `--self-boot` / `BRAIN_EVAL_SELF_BOOT=1` —
/// so a run never silently writes to an anonymous volume.
///
/// `benchmark = None` → single controlled write+read (a correctness gate).
/// `Some(name)` → staged per-stage run; the report is the deliverable, so
/// it exits 0 whatever the per-question verdicts. `BRAIN_EVAL_MAX_QUESTIONS`
/// caps it (the runner ingests a conversation before asking its questions
/// and stops at the cap). File-backed benchmarks need
/// `BRAIN_EVAL_DATASETS_DIR`.
async fn probe_cmd(benchmark: Option<&str>, external_endpoint: Option<SocketAddr>) -> ExitCode {
    use brain_eval::client::{DockerServerOpts, KeyMinter, ServerHandle};

    // Validate the benchmark name before booting anything.
    let bench: Option<Box<dyn Benchmark>> = match benchmark {
        None => None,
        Some("smoke") => Some(Box::new(SmokeBenchmark)),
        Some("locomo") => Some(Box::new(LocomoBenchmark)),
        Some("longmemeval-s" | "longmemeval" | "lme") => Some(Box::new(LongMemEvalS)),
        Some("dmr") => Some(Box::new(DmrBenchmark)),
        Some(other) => {
            eprintln!(
                "probe: unknown benchmark '{other}' (supported: smoke, locomo, longmemeval-s, dmr)"
            );
            return ExitCode::from(2);
        }
    };

    // Reads-only reuse: BRAIN_EVAL_REUSE_SPACE=<uuid> targets an
    // already-ingested space (skip the write). Pair with --endpoint against
    // a kept server.
    let reuse_space: Option<[u8; 16]> = std::env::var("BRAIN_EVAL_REUSE_SPACE")
        .ok()
        .and_then(|s| uuid::Uuid::parse_str(s.trim()).ok())
        .map(uuid::Uuid::into_bytes);
    // Keep a self-booted server up afterward (so a later reads-only run can
    // reuse the ingest) instead of tearing it down.
    let keep = matches!(
        std::env::var("BRAIN_EVAL_KEEP_SERVER").ok().as_deref(),
        Some("1" | "true" | "yes" | "on")
    );

    // Resolve the server: external (named) or self-boot Docker. Also the
    // metrics-plane address, used to drain the async write pipeline before
    // reading.
    let (endpoint, minter, mut handle, metrics_addr) = match external_endpoint {
        Some(ep) => {
            // Metrics plane defaults to 9091 on the same host; override with
            // BRAIN_EVAL_DRAIN_ADMIN_PORT (mirrors the dataset runner).
            let port = std::env::var("BRAIN_EVAL_DRAIN_ADMIN_PORT")
                .ok()
                .and_then(|p| p.parse::<u16>().ok())
                .unwrap_or(9091);
            let metrics = SocketAddr::new(ep.ip(), port);
            (ep, KeyMinter::from_external_env(ep), None, metrics)
        }
        None => {
            println!("brain-eval :: booting brain:latest in Docker (no --endpoint given)…");
            // Default arena (256MiB) is ample — even a 400-turn conversation
            // plus HyPE question vectors uses <10MB of arena. Keeping it small
            // leaves headroom for the embed + GLiNER models under Docker
            // Desktop's memory ceiling.
            let opts = DockerServerOpts::default();
            // The image publishes its metrics plane (9091) at 19091 on the host.
            let metrics: SocketAddr = "127.0.0.1:19091"
                .parse()
                .expect("invariant: literal metrics addr parses");
            match ServerHandle::start_docker(opts).await {
                Ok(h) => (h.endpoint(), h.key_minter(), Some(h), metrics),
                Err(e) => {
                    eprintln!("error: could not boot docker server: {e}");
                    eprintln!(
                        "  need: BRAIN__LLM__API_KEY set, the `brain:latest` image, and the `brain-models` volume"
                    );
                    return ExitCode::from(1);
                }
            }
        }
    };

    // Keep the self-booted container alive after the run when asked.
    if keep {
        if let Some(h) = handle.as_mut() {
            h.keep_alive();
        }
    }

    let is_staged = bench.is_some();
    let result = match bench {
        None => brain_eval::bench::probe::run_probe(&minter, endpoint).await,
        Some(b) => {
            let max_q = std::env::var("BRAIN_EVAL_MAX_QUESTIONS")
                .ok()
                .and_then(|s| s.parse().ok());
            brain_eval::bench::probe::staged::run_staged(
                &minter,
                endpoint,
                Some(metrics_addr),
                reuse_space,
                b.as_ref(),
                max_q,
            )
            .await
        }
    };

    // Tear down a self-booted container — unless keep_alive() was set, in
    // which case stop() is a no-op and we print how to reuse/clean it up.
    if let Some(h) = handle {
        h.stop().await;
        if keep {
            println!();
            println!("server kept for reuse (data lives only while it runs):");
            println!(
                "  reads-only re-run: BRAIN_EVAL_REUSE_SPACE=<space printed above> \\
    BRAIN_EVAL_ADMIN_SECRET=eval-admin-secret BRAIN_EVAL_ADMIN_PORT=19092 \\
    BRAIN_EVAL_DRAIN_ADMIN_PORT=19091 … probe <bench> --endpoint 127.0.0.1:18080"
            );
            println!("  stop when done:    docker rm -f brain-eval-server");
        }
    }

    match result {
        Ok(true) => ExitCode::SUCCESS,
        // Staged report is the deliverable — completing is success.
        Ok(false) if is_staged => ExitCode::SUCCESS,
        Ok(false) => {
            eprintln!("probe: FAIL — a stage did not produce the expected result");
            ExitCode::from(1)
        }
        Err(e) => {
            eprintln!("error: probe failed: {e}");
            ExitCode::from(1)
        }
    }
}

/// Select the acceptance config: `--full`/`--scale` picks the
/// reference-hardware full-scale config (1M memories, with the storage and
/// chaos gates wired on); the default is the dev-box smoke config. Split out
/// so the flag→config mapping is unit-testable without a live server.
fn acceptance_config(endpoint: SocketAddr, full: bool) -> AcceptanceConfig {
    if full {
        AcceptanceConfig::full_scale(endpoint)
    } else {
        AcceptanceConfig::smoke(endpoint)
    }
}

/// `brain-eval acceptance` — run the v1.0 acceptance suite and gate on it.
/// Correctness gates must pass everywhere; performance gates are
/// informational off reference hardware. `BRAIN_EVAL_RESTART_RECOVERY=1`
/// adds the (slower, docker-booting) restart-recovery gate. `--full`
/// (alias `--scale`) selects the reference-hardware full-scale config,
/// which additionally reaches the storage-footprint and kill-during-write
/// chaos gates — far too heavy for an emulated dev container.
async fn acceptance_cmd(endpoint: SocketAddr, full: bool) -> ExitCode {
    println!(
        "brain-eval :: v1.0 acceptance{}",
        if full { " (full-scale)" } else { "" }
    );
    println!("  endpoint : {endpoint}");
    println!();

    let mut cfg = acceptance_config(endpoint, full);
    // The env var opts smoke into the restart-recovery gate; it must not
    // turn OFF the gate that `full_scale` already enables.
    cfg.run_restart_recovery = cfg.run_restart_recovery
        || std::env::var("BRAIN_EVAL_RESTART_RECOVERY").as_deref() == Ok("1");

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

/// Pull `--only-conversation ID` (or `--only-conversation=ID`) out of the
/// trailing args. Returns `None` when absent.
fn parse_only_conversation_flag(rest: &[String]) -> Option<String> {
    let mut it = rest.iter();
    while let Some(arg) = it.next() {
        if let Some(v) = arg.strip_prefix("--only-conversation=") {
            return Some(v.to_owned());
        } else if arg == "--only-conversation" {
            return it.next().cloned();
        }
    }
    None
}

/// True when a bare boolean flag (e.g. `--verify-memory-list`) is present in
/// the trailing args.
fn parse_bool_flag(rest: &[String], flag: &str) -> bool {
    rest.iter().any(|a| a == flag)
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
        match serde_json::from_str::<brain_eval::bench::core::outcome::QuestionResult>(line) {
            Ok(r) => results.push(r),
            Err(_) => unparseable += 1,
        }
    }
    if results.is_empty() {
        eprintln!("error: no parseable question results in {path}");
        return ExitCode::from(1);
    }
    let metrics = brain_eval::bench::score::metrics::compute_full_metrics(&results);
    println!("=== summary :: {path} ===");
    println!(
        "parsed    : {} graded questions ({unparseable} unparseable lines skipped)",
        results.len()
    );
    print_metrics(&metrics);
    ExitCode::SUCCESS
}

fn print_summary(report: &brain_eval::bench::report::BenchmarkReport) {
    println!("=== {} ===", report.meta.benchmark_name);
    print_metrics(&report.metrics);
}

/// Render the headline metrics, the answer-shape breakdown (one memory
/// vs a set vs honest abstention — how the router actually behaves), and
/// the committed-answer precision (the hard invariant: never confidently
/// wrong).
fn print_metrics(m: &brain_eval::bench::score::metrics::EvalMetrics) {
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
         \x20 probe [bench]   stage-by-stage want-vs-got; self-boots brain:latest in Docker\n\
         \x20                 (no bench = one controlled write+read; bench = smoke|locomo|longmemeval-s|dmr)\n\
         \x20                 needs BRAIN__LLM__API_KEY; file datasets need BRAIN_EVAL_DATASETS_DIR;\n\
         \x20                 BRAIN_EVAL_MAX_QUESTIONS caps it; --endpoint targets an external server\n\
         \x20 acceptance      v1.0 acceptance suite (latency/throughput/recall/scenarios)\n\
         \x20                 --full (alias --scale) runs the reference-scale config\n\
         \x20                 (1M memories; adds the storage-footprint + chaos gates)\n\
         \x20                 BRAIN_EVAL_RESTART_RECOVERY=1 adds the restart-recovery gate\n\
         \x20 soak            sustained workload + drift sampling (BRAIN_EVAL_SOAK_SECS)\n\
         \x20 summary <file>  recompute metrics from a streamed .partial.jsonl\n\
         \x20                 (recover results from an interrupted run)\n\
         \n\
         env: BRAIN_EVAL_ENDPOINT, BRAIN_EVAL_MAX_QUESTIONS, BRAIN_EVAL_TOP_K,\n\
         \x20    BRAIN_EVAL_OUTPUT_DIR, BRAIN_EVAL_FORMATS, BRAIN_EVAL_QUESTION_TYPES,\n\
         \x20    BRAIN_EVAL_EXTRACTION_DRAIN\n\
         \n\
         auth (external server only; the docker-boot harness sets these itself):\n\
         \x20    BRAIN_EVAL_ADMIN_SECRET   required to mint data-plane keys via the admin plane\n\
         \x20    BRAIN_EVAL_ADMIN_PORT     admin listener port (default 9092)\n\
         \x20    BRAIN_EVAL_NAMESPACE      tenant the run writes under (default eval)"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ep() -> SocketAddr {
        DEFAULT_ENDPOINT
            .parse()
            .expect("default endpoint literal parses")
    }

    #[test]
    fn full_flag_selects_full_scale_with_storage_and_chaos_gates() {
        let cfg = acceptance_config(ep(), true);
        // The full-scale config is the only one that reaches these gates.
        assert!(cfg.run_storage, "full-scale must enable the storage gate");
        assert!(cfg.run_chaos, "full-scale must enable the chaos gate");
        // Sanity: it is the reference-scale ingest, not the smoke ingest.
        assert_eq!(cfg.scale.ingest_n, 1_000_000);
    }

    #[test]
    fn default_no_flag_is_smoke_without_storage_or_chaos() {
        let cfg = acceptance_config(ep(), false);
        assert!(!cfg.run_storage, "smoke must not run the storage gate");
        assert!(!cfg.run_chaos, "smoke must not run the chaos gate");
        assert_eq!(cfg.scale.ingest_n, 200);
    }

    #[test]
    fn full_and_scale_aliases_both_parse() {
        assert!(parse_bool_flag(&["--full".to_string()], "--full"));
        assert!(parse_bool_flag(&["--scale".to_string()], "--scale"));
    }
}
