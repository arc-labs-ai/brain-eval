//! `EvalRunner` — drive a benchmark end-to-end.
//!
//! For each `EvalInstance`:
//! 1. Spin up a fresh harness (= fresh `AgentId` = guaranteed isolation
//!    from other questions per spec §12/02).
//! 2. Ingest every session via [`crate::run::harness::BrainEvalHarness::ingest`].
//! 3. Run a RECALL with the question as the cue.
//! 4. Synthesize an answer from the top-K hits.
//! 5. Judge against ground truth.
//! 6. Record a [`QuestionResult`].
//!
//! Instances sharing a non-empty `conversation_id` are ingested once
//! per conversation and queried multiple times — this is the
//! LongMemEval / LoCoMo shape.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use tracing::warn;

use crate::core::benchmark::{Benchmark, EvalError};
use crate::core::instance::EvalInstance;
use crate::core::outcome::{JudgeResult, QuestionResult, Verdict};
use crate::report::baselines::CompetitorBaselines;
use crate::report::format::{json::JsonReporter, text::TextReporter, Reporter};
use crate::report::shape::{BenchmarkMeta, BenchmarkReport};
use crate::run::config::{ReporterKind, RunConfig};
use crate::run::harness::BrainEvalHarness;
use crate::run::synthesize::synthesize_answer;
use crate::score::judge::judge_answer_heuristic;
use crate::score::metrics::compute_full_metrics;

/// Drives a benchmark to completion and produces a [`BenchmarkReport`].
pub struct EvalRunner {
    config: RunConfig,
    competitor_fn: CompetitorBaselines,
    /// LLM judge, if `live-llm` is enabled and a provider key is set.
    /// `None` keeps scoring on the heuristic judge.
    #[cfg(feature = "live-llm")]
    llm_judge: Option<crate::score::llm_judge::LlmJudge>,
    /// LLM answer synthesizer, same gating. `None` keeps the heuristic
    /// top-K concatenation.
    #[cfg(feature = "live-llm")]
    llm_synth: Option<crate::run::synthesize::LlmSynthesizer>,
}

impl EvalRunner {
    /// New runner with explicit config and competitor table.
    #[must_use]
    pub fn new(config: RunConfig, competitor_fn: CompetitorBaselines) -> Self {
        Self {
            config,
            competitor_fn,
            #[cfg(feature = "live-llm")]
            llm_judge: crate::score::llm_judge::LlmJudge::from_env(),
            #[cfg(feature = "live-llm")]
            llm_synth: crate::run::synthesize::LlmSynthesizer::from_env(),
        }
    }

    /// Compose an answer: the LLM synthesizer when configured, else the
    /// heuristic top-K concatenation.
    #[cfg_attr(not(feature = "live-llm"), allow(clippy::unused_self))]
    async fn synth_answer(
        &self,
        instance: &EvalInstance,
        hits: &[brain_db_sdk::wire::types::MemoryResult],
        top_k: usize,
    ) -> String {
        #[cfg(feature = "live-llm")]
        if let Some(synth) = &self.llm_synth {
            return synth
                .synthesize(&instance.question, hits, instance.question_type, top_k)
                .await;
        }
        synthesize_answer(&instance.question, hits, instance.question_type, top_k)
    }

    /// Score one answer: the LLM judge when configured, else the heuristic.
    #[cfg_attr(not(feature = "live-llm"), allow(clippy::unused_self))]
    async fn judge_answer(&self, instance: &EvalInstance, system_answer: &str) -> JudgeResult {
        #[cfg(feature = "live-llm")]
        if let Some(judge) = &self.llm_judge {
            return judge
                .judge(
                    &instance.question_id,
                    instance.question_type,
                    &instance.question,
                    &instance.answer,
                    system_answer,
                )
                .await;
        }
        judge_answer_heuristic(
            &instance.question_id,
            instance.question_type,
            &instance.answer,
            system_answer,
        )
    }

    /// Run `benchmark` end-to-end. The report is written to
    /// `config.output_dir/<benchmark_id>-<ts>.{json,txt}` and also
    /// returned.
    ///
    /// # Errors
    ///
    /// - [`EvalError::DatasetsDirNotSet`] when `BRAIN_EVAL_DATASETS_DIR`
    ///   is not set.
    /// - [`EvalError::DatasetNotFound`] / [`EvalError::ParseError`]
    ///   from the benchmark's `load`.
    /// - [`EvalError::Harness`] for SDK / network errors that prevent
    ///   the run from making progress at all.
    pub async fn run(&self, benchmark: &dyn Benchmark) -> Result<BenchmarkReport, EvalError> {
        let datasets_dir = if benchmark.requires_datasets_dir() {
            datasets_dir().ok_or(EvalError::DatasetsDirNotSet)?
        } else {
            // Compiled-in benchmarks ignore the path; pass a harmless
            // placeholder so `load` keeps its uniform signature.
            PathBuf::from(".")
        };
        let instances = benchmark.load(&datasets_dir)?;
        let instances = match self.config.max_questions {
            Some(n) => instances.into_iter().take(n).collect::<Vec<_>>(),
            None => instances,
        };

        // Reflect the judge that will actually run, not just the feature:
        // `live-llm` with no API key still scores on the heuristic.
        #[cfg(feature = "live-llm")]
        let judge_type = match &self.llm_judge {
            Some(j) => j.describe(),
            None => "heuristic".to_string(),
        };
        #[cfg(not(feature = "live-llm"))]
        let judge_type = "heuristic".to_string();
        let meta = BenchmarkMeta::new(
            benchmark.id(),
            benchmark.display_name(),
            benchmark.url(),
            &judge_type,
            instances.len(),
        );

        // Group by conversation_id so we ingest once per conversation.
        // BTreeMap keeps deterministic ordering.
        let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (idx, inst) in instances.iter().enumerate() {
            let key = inst
                .conversation_id
                .clone()
                .unwrap_or_else(|| inst.question_id.clone());
            groups.entry(key).or_default().push(idx);
        }

        let mut question_results: Vec<QuestionResult> = Vec::with_capacity(instances.len());

        // Incremental tracking: append each result to a JSONL sidecar and
        // print a running tally as we go, so an interrupted or crashed run
        // still leaves partial metrics on disk instead of nothing.
        if let Err(e) = std::fs::create_dir_all(&self.config.output_dir) {
            warn!(error = %e, "could not create output dir; incremental results disabled");
        }
        let mut tracker = IncrementalTracker::new(&self.config.output_dir, &meta);
        println!(
            "eval: {} questions; streaming partial results to {}",
            tracker.total,
            tracker.path.display()
        );

        for (conv_key, idxs) in &groups {
            let harness = match BrainEvalHarness::connect(self.config.endpoint).await {
                Ok(h) => h,
                Err(e) => {
                    warn!(
                        conversation = %conv_key,
                        error = %e,
                        "harness connect failed; recording every question in this group as failed",
                    );
                    for &idx in idxs {
                        let inst = &instances[idx];
                        let r = failed_question_result(
                            inst,
                            format!("connect failed: {e}"),
                            /* ingest_failed = */ true,
                        );
                        tracker.record(&r);
                        question_results.push(r);
                    }
                    continue;
                }
            };

            // ---- ingest once ----
            let first = &instances[idxs[0]];
            let (write_latency_ms, write_attempted, write_stored, write_deduplicated, ingest_err) =
                ingest_sessions(&harness, first).await;

            // ---- per-question retrieval + judging ----
            for &idx in idxs {
                let inst = &instances[idx];
                let r = self
                    .run_question(
                        &harness,
                        inst,
                        write_latency_ms,
                        write_attempted,
                        write_stored,
                        write_deduplicated,
                        ingest_err,
                    )
                    .await;
                tracker.record(&r);
                question_results.push(r);
            }

            // ---- close harness (best-effort) ----
            if let Err(e) = harness.close().await {
                warn!(error = %e, "harness close failed; continuing");
            }
        }

        let metrics = compute_full_metrics(&question_results);
        let competitors = (self.competitor_fn)();
        let report = BenchmarkReport::build(meta, metrics, competitors, question_results);

        if let Err(e) = self.write_reports(&report) {
            warn!(error = %e, "report write failed");
        }

        Ok(report)
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_question(
        &self,
        harness: &BrainEvalHarness,
        instance: &EvalInstance,
        write_latency_ms: u64,
        write_attempted: u64,
        write_stored: u64,
        write_deduplicated: u64,
        ingestion_failed: bool,
    ) -> QuestionResult {
        let recall = harness
            .recall(&instance.question, self.config.top_k_retrieve)
            .await;
        let (hits, read_latency_ms, retrieval_failed) = match recall {
            Ok(o) => (o.hits, o.latency_ms, false),
            Err(e) => {
                warn!(
                    question_id = %instance.question_id,
                    error = %e,
                    "recall failed; recording as retrieval_failed",
                );
                (Vec::new(), 0, true)
            }
        };

        let retrieved_memory_contents: Vec<String> = hits.iter().map(|m| m.text.clone()).collect();
        let memories_retrieved = hits.len();

        #[allow(clippy::cast_possible_truncation)]
        let cap = self.config.top_k_retrieve as usize;
        let system_answer = self.synth_answer(instance, &hits, cap.max(1)).await;

        let judged = self.judge_answer(instance, &system_answer).await;

        QuestionResult {
            question_id: instance.question_id.clone(),
            question_type: instance.question_type,
            question: instance.question.clone(),
            ground_truth: instance.answer.clone(),
            system_answer,
            verdict: judged.verdict,
            score: judged.score,
            write_latency_ms,
            read_latency_ms,
            tokens_write: 0,
            tokens_read: 0,
            memories_retrieved,
            retrieved_memory_contents,
            judge_reasoning: judged.reasoning,
            ingestion_failed,
            retrieval_failed,
            write_attempted,
            write_stored,
            write_deduplicated,
        }
    }

    fn write_reports(&self, report: &BenchmarkReport) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.config.output_dir)?;
        let stem = format!(
            "{}-{}",
            report.meta.benchmark_id, report.meta.run_started_unix_nanos
        );
        for kind in &self.config.reporters {
            let path = match kind {
                ReporterKind::Json => self.config.output_dir.join(format!("{stem}.json")),
                ReporterKind::Text => self.config.output_dir.join(format!("{stem}.txt")),
            };
            let result: std::io::Result<()> = match kind {
                ReporterKind::Json => JsonReporter.write(report, &path),
                ReporterKind::Text => TextReporter.write(report, &path),
            };
            if let Err(ref e) = result {
                warn!(reporter = ?kind, error = %e, "reporter failed");
            }
        }
        Ok(())
    }
}

/// Streams per-question results to a JSONL sidecar as the run progresses
/// and prints a running tally. The point is durability of *partial*
/// results: if the run is interrupted (crash, Ctrl-C, server death) the
/// `.partial.jsonl` file still holds every question graded so far, so the
/// metrics aren't all-or-nothing on the final report write.
struct IncrementalTracker {
    /// `<output_dir>/<benchmark_id>-<run_started_unix_nanos>.partial.jsonl`.
    path: PathBuf,
    /// Instances loaded for this run (denominator for progress).
    total: usize,
    /// Questions recorded so far.
    done: usize,
    /// Running sum of judge scores (numerator for running accuracy).
    score_sum: f64,
}

impl IncrementalTracker {
    fn new(output_dir: &Path, meta: &BenchmarkMeta) -> Self {
        let path = output_dir.join(format!(
            "{}-{}.partial.jsonl",
            meta.benchmark_id, meta.run_started_unix_nanos
        ));
        Self {
            path,
            total: meta.instance_count,
            done: 0,
            score_sum: 0.0,
        }
    }

    /// Append one result as a JSON line and print a progress line. Both
    /// are best-effort: a write failure warns but never aborts the run.
    fn record(&mut self, result: &QuestionResult) {
        self.done += 1;
        self.score_sum += result.score;
        #[allow(clippy::cast_precision_loss)]
        let running_acc = self.score_sum / self.done as f64;

        if let Err(e) = self.append_line(result) {
            warn!(path = %self.path.display(), error = %e, "incremental result append failed");
        }

        let flag = if result.ingestion_failed {
            " INGEST-FAIL"
        } else if result.retrieval_failed {
            " RETRIEVAL-FAIL"
        } else {
            ""
        };
        println!(
            "[{:>4}/{}] {:<9} acc={:.3} q={}{}",
            self.done,
            self.total,
            format!("{:?}", result.verdict),
            running_acc,
            result.question_id,
            flag,
        );
    }

    fn append_line(&self, result: &QuestionResult) -> std::io::Result<()> {
        use std::io::Write;
        let line = serde_json::to_string(result)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        writeln!(f, "{line}")?;
        f.flush()
    }
}

/// Resolve the datasets directory from `BRAIN_EVAL_DATASETS_DIR`.
#[must_use]
pub fn datasets_dir() -> Option<PathBuf> {
    std::env::var_os("BRAIN_EVAL_DATASETS_DIR").map(PathBuf::from)
}

async fn ingest_sessions(
    harness: &BrainEvalHarness,
    inst: &EvalInstance,
) -> (u64, u64, u64, u64, bool) {
    let mut total_latency_ms = 0u64;
    let mut attempted = 0u64;
    let mut stored = 0u64;
    let mut deduplicated = 0u64;
    let mut failed = false;

    for session in &inst.sessions {
        match harness.ingest(&session.turns).await {
            Ok(out) => {
                total_latency_ms = total_latency_ms.saturating_add(out.latency_ms);
                attempted = attempted.saturating_add(out.attempted);
                #[allow(clippy::cast_possible_truncation)]
                let s = out.stored_ids.len() as u64;
                stored = stored.saturating_add(s);
                deduplicated = deduplicated.saturating_add(out.deduplicated);
            }
            Err(e) => {
                warn!(
                    session = %session.session_id,
                    error = %e,
                    "session ingest failed",
                );
                failed = true;
            }
        }
    }

    (total_latency_ms, attempted, stored, deduplicated, failed)
}

fn failed_question_result(
    inst: &EvalInstance,
    reason: String,
    ingest_failed: bool,
) -> QuestionResult {
    QuestionResult {
        question_id: inst.question_id.clone(),
        question_type: inst.question_type,
        question: inst.question.clone(),
        ground_truth: inst.answer.clone(),
        system_answer: String::new(),
        verdict: Verdict::Incorrect,
        score: 0.0,
        write_latency_ms: 0,
        read_latency_ms: 0,
        tokens_write: 0,
        tokens_read: 0,
        memories_retrieved: 0,
        retrieved_memory_contents: Vec::new(),
        judge_reasoning: reason,
        ingestion_failed: ingest_failed,
        retrieval_failed: !ingest_failed,
        write_attempted: 0,
        write_stored: 0,
        write_deduplicated: 0,
    }
}
