//! Pipeline validator — runs the smoke fixture through every stage
//! of the eval pipeline EXCEPT the live `brain-server` roundtrip.
//!
//! ## Why this exists
//!
//! `brain-server` requires Linux (Glommio + io_uring). On macOS / non-
//! Linux dev hosts the server exits cleanly as a stub, so the full
//! end-to-end example (`run_longmemeval`) can't actually produce
//! numbers. This binary fills that gap by:
//!
//! 1. Loading the smoke fixture via the real [`LongMemEvalS`] parser.
//! 2. Generating a deterministic, plausible "RECALL response" per
//!    question — simulating what brain-server would surface for each
//!    cue text by reading the matching user-turn from the haystack.
//! 3. Running the resulting `QuestionResult` records through the real
//!    `compute_full_metrics` + reporter pipeline.
//! 4. Printing the actual numbers + writing JSON / text reports.
//!
//! Steps (1), (3), and (4) are byte-identical to what
//! `run_longmemeval` does on Linux. Step (2) is the substitution — a
//! deterministic stand-in for the wire roundtrip that yields the same
//! shape of `MemoryResult` the harness would.
//!
//! ## Honest framing
//!
//! Numbers from this run validate the **pipeline**, not Brain's recall
//! quality. Treat them as a smoke test of the eval code, not as a
//! benchmark score. The honest brain-server run requires Linux.

use std::error::Error;
use std::path::PathBuf;
use std::process::ExitCode;

use brain_eval::core::benchmark::Benchmark;
use brain_eval::core::instance::EvalInstance;
use brain_eval::core::outcome::{QuestionResult, Verdict};
use brain_eval::datasets::longmemeval::LongMemEvalS;
use brain_eval::report::format::{json::JsonReporter, text::TextReporter, Reporter};
use brain_eval::report::shape::{BenchmarkMeta, BenchmarkReport};
use brain_eval::report::longmemeval_s_competitor_baselines;
use brain_eval::run::synthesize_answer;
use brain_eval::score::judge::judge_answer_heuristic;
use brain_eval::score::metrics::compute_full_metrics;
use brain_protocol::request::MemoryKindWire;
use brain_protocol::response::MemoryResult;

fn main() -> ExitCode {
    match try_main() {
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

fn try_main() -> Result<(), Box<dyn Error>> {
    // ---- 1. Stage the smoke fixture as if it were the real dataset
    let staging = tempdir_under_target()?;
    std::fs::create_dir_all(staging.join("longmemeval"))?;
    let fixture_src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("longmemeval_smoke.json");
    let fixture_dst = staging.join("longmemeval").join("longmemeval_s.json");
    std::fs::copy(&fixture_src, &fixture_dst)?;

    // ---- 2. Run the real parser
    let instances: Vec<EvalInstance> = LongMemEvalS.load(&staging)?;
    println!("brain-eval :: validate_pipeline");
    println!("  fixture     : {}", fixture_dst.display());
    println!("  instances   : {}", instances.len());
    println!();

    // ---- 3. For each instance, run synthesize → judge using a
    //         deterministic mock RECALL response derived from the
    //         instance's haystack.
    let top_k: u32 = 10;
    let mut question_results: Vec<QuestionResult> = Vec::with_capacity(instances.len());

    for inst in &instances {
        let hits = mock_recall_hits(inst);
        let memories_retrieved = hits.len();
        let retrieved_memory_contents: Vec<String> = hits.iter().map(|m| m.text.clone()).collect();

        let system_answer = synthesize_answer(
            &inst.question,
            &hits,
            inst.question_type,
            top_k as usize,
        );

        let judged = judge_answer_heuristic(
            &inst.question_id,
            inst.question_type,
            &inst.answer,
            &system_answer,
        );

        question_results.push(QuestionResult {
            question_id: inst.question_id.clone(),
            question_type: inst.question_type,
            question: inst.question.clone(),
            ground_truth: inst.answer.clone(),
            system_answer: system_answer.clone(),
            verdict: judged.verdict,
            score: judged.score,
            // Synthetic but plausible latencies so the percentile math
            // produces non-degenerate numbers.
            write_latency_ms: 50 + (inst.question_id.len() as u64 % 30),
            read_latency_ms: 3 + (inst.question_id.len() as u64 % 5),
            tokens_write: 0,
            tokens_read: 0,
            memories_retrieved,
            retrieved_memory_contents,
            judge_reasoning: judged.reasoning,
            ingestion_failed: false,
            retrieval_failed: false,
            // Each user turn = one ENCODE attempt.
            write_attempted: count_user_turns(inst),
            write_stored: count_user_turns(inst),
            write_deduplicated: 0,
        });

        // Compact per-question line, so a human reading stdout can
        // see what happened.
        println!(
            "  {qid:<12} [{qtype:<16}] verdict={verdict:?}  hits={n}",
            qid = inst.question_id,
            qtype = inst.question_type.tag(),
            verdict = judged.verdict,
            n = memories_retrieved,
        );
    }
    println!();

    // ---- 4. Real metrics aggregation
    let metrics = compute_full_metrics(&question_results);

    // ---- 5. Build a real BenchmarkReport
    let meta = BenchmarkMeta::new(
        LongMemEvalS.id(),
        LongMemEvalS.display_name(),
        LongMemEvalS.url(),
        // Tag the judge type honestly — we used the heuristic judge.
        "heuristic",
        instances.len(),
    );
    let competitors = longmemeval_s_competitor_baselines();
    let report = BenchmarkReport::build(meta, metrics, competitors, question_results);

    // ---- 6. Write reports
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("eval-reports");
    std::fs::create_dir_all(&out_dir)?;
    let stem = format!(
        "{}-validate-{}",
        report.meta.benchmark_id, report.meta.run_started_unix_nanos
    );
    let json_path = out_dir.join(format!("{stem}.json"));
    let text_path = out_dir.join(format!("{stem}.txt"));
    JsonReporter.write(&report, &json_path)?;
    TextReporter.write(&report, &text_path)?;

    // ---- 7. Headline summary
    println!("=== Pipeline validation — smoke fixture, mocked recall ===");
    println!("instances          : {}", report.metrics.total_questions);
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
    println!("Per-dimension breakdown:");
    let mut entries: Vec<_> = report.metrics.per_dimension.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    for (tag, d) in entries {
        println!("  {tag:<18} acc={:.3}  ({}/{})", d.accuracy, d.correct, d.count);
    }
    println!();
    println!("Reports written:");
    println!("  json : {}", json_path.display());
    println!("  text : {}", text_path.display());
    println!();
    println!(
        "Note: recall is mocked here (Brain server requires Linux + io_uring). \
         The pipeline — parser, judge, metrics, reporters — is real."
    );

    // Best-effort cleanup of the staging dir.
    let _ = std::fs::remove_dir_all(&staging);
    Ok(())
}

/// Build a deterministic "RECALL response" for a single
/// `EvalInstance`. For each user turn in its haystack, emit one
/// `MemoryResult` whose `text` is the turn content. This is what the
/// substrate would return if the embedding model put the haystack
/// turns near the cue (which is what we expect on these obvious
/// single-fact prompts).
///
/// For abstention rows the haystack is intentionally unrelated; we
/// still emit the available turns to mirror what brain-server would
/// — the judge then sees the wrong content and (correctly) marks the
/// answer as incorrect because the system can't honestly say "I don't
/// know" without an LLM synthesizer.
fn mock_recall_hits(inst: &EvalInstance) -> Vec<MemoryResult> {
    let mut hits = Vec::new();
    for session in &inst.sessions {
        for turn in &session.turns {
            if turn.role != "user" {
                continue;
            }
            if turn.content.trim().is_empty() {
                continue;
            }
            hits.push(mock_memory(&turn.content));
        }
    }
    hits
}

fn mock_memory(text: &str) -> MemoryResult {
    MemoryResult {
        memory_id: 0,
        text: text.to_owned(),
        similarity_score: 0.85,
        confidence: 0.85,
        salience: 0.5,
        kind: MemoryKindWire::Episodic,
        context_id: 0,
        created_at_unix_nanos: 0,
        last_accessed_at_unix_nanos: 0,
        vector_offset: 0,
        vector_dim: 0,
        edges: None,
        contributing_retrievers: Vec::new(),
        fused_score: 0.85,
        salience_initial: 0.5,
        access_count: 0,
        lsn: 0,
        flags: 0,
        consolidated_at_unix_nanos: None,
        edges_out_count: 0,
        edges_in_count: 0,
    }
}

fn count_user_turns(inst: &EvalInstance) -> u64 {
    let mut n: u64 = 0;
    for session in &inst.sessions {
        for turn in &session.turns {
            if turn.role == "user" && !turn.content.trim().is_empty() {
                n = n.saturating_add(1);
            }
        }
    }
    n
}

/// Create a uniquely-named directory under `target/` so the staged
/// fixture doesn't collide with concurrent runs.
fn tempdir_under_target() -> Result<PathBuf, Box<dyn Error>> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!("validate-stage-{nanos}"));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

#[allow(dead_code)]
fn assert_three_verdicts_one_correct_two_incorrect(qr: &[QuestionResult]) {
    // Hook for an inline integration test if we ever want one — the
    // smoke fixture should produce a predictable distribution of
    // verdicts under the heuristic judge.
    let n_correct = qr.iter().filter(|r| r.verdict == Verdict::Correct).count();
    let _ = n_correct;
}
