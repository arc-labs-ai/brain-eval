//! `EvalRunner` — drive a benchmark end-to-end.
//!
//! For each `EvalInstance`:
//! 1. Spin up a fresh harness (= fresh `AgentId` = guaranteed isolation
//!    from other questions per spec §12/02).
//! 2. Ingest every session via [`crate::client::harness::BrainEvalHarness::ingest`].
//! 3. Run a RECALL with the question as the cue.
//! 4. Synthesize an answer from the top-K hits.
//! 5. Judge against ground truth.
//! 6. Record a [`QuestionResult`].
//!
//! Instances sharing a non-empty `conversation_id` are ingested once
//! per conversation and queried multiple times — this is the
//! LongMemEval / LoCoMo shape.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use tracing::warn;

use crate::bench::agent_manifest::AgentManifest;
use crate::bench::config::{ReporterKind, RunConfig};
use crate::bench::core::benchmark::{Benchmark, EvalError};
use crate::bench::core::instance::{EvalInstance, QuestionType};
use crate::bench::core::outcome::{AnswerKind, JudgeResult, QuestionResult, Verdict};
use crate::bench::report::baselines::CompetitorBaselines;
use crate::bench::report::format::{json::JsonReporter, text::TextReporter, Reporter};
use crate::bench::report::shape::{BenchmarkMeta, BenchmarkReport};
use crate::bench::score::judge::judge_answer_heuristic;
use crate::bench::score::metrics::compute_full_metrics;
use crate::bench::synthesize::synthesize_answer;
use crate::client::auth::KeyMinter;
use crate::client::harness::{BrainEvalHarness, HarnessError, RecallOutcome};
use crate::client::metrics::PrometheusMetrics;
use brain_db_sdk::wire::types::AnswerKindWire;

/// Wraps the optional `live-llm` judge behind an always-present slot, so
/// every call site treats it uniformly instead of re-checking the feature
/// flag. `#[cfg(feature = "live-llm")]` is confined to this type's body —
/// `EvalRunner`'s own methods and `run()` never branch on the feature.
struct JudgeSlot {
    #[cfg(feature = "live-llm")]
    judge: Option<crate::bench::score::llm_judge::LlmJudge>,
}

impl JudgeSlot {
    fn from_env() -> Self {
        Self {
            #[cfg(feature = "live-llm")]
            judge: crate::bench::score::llm_judge::LlmJudge::from_env(),
        }
    }

    /// `llm:<provider>:<model>@<version>` when an LLM judge is configured,
    /// else `"heuristic"` — reflects the judge that actually ran, not just
    /// whether the feature is compiled in (a `live-llm` build with no
    /// provider key still scores on the heuristic).
    fn describe(&self) -> String {
        #[cfg(feature = "live-llm")]
        {
            self.judge.as_ref().map_or_else(
                || "heuristic".to_string(),
                crate::bench::score::llm_judge::LlmJudge::describe,
            )
        }
        #[cfg(not(feature = "live-llm"))]
        {
            "heuristic".to_string()
        }
    }

    /// How many `judge` calls fell back to the heuristic grader. Always 0
    /// without `live-llm`, or without a configured judge.
    fn heuristic_fallback_count(&self) -> usize {
        #[cfg(feature = "live-llm")]
        {
            self.judge.as_ref().map_or(
                0,
                crate::bench::score::llm_judge::LlmJudge::heuristic_fallback_count,
            )
        }
        #[cfg(not(feature = "live-llm"))]
        {
            0
        }
    }

    /// Grade one answer: the LLM judge when configured, else the heuristic.
    async fn judge(
        &self,
        question_id: &str,
        qtype: QuestionType,
        question: &str,
        ground_truth: &str,
        system_answer: &str,
    ) -> JudgeResult {
        #[cfg(feature = "live-llm")]
        if let Some(judge) = &self.judge {
            return judge
                .judge(question_id, qtype, question, ground_truth, system_answer)
                .await;
        }
        #[cfg(not(feature = "live-llm"))]
        let _ = question;
        judge_answer_heuristic(question_id, qtype, ground_truth, system_answer)
    }

    /// Answer-supporting context recall: ask the support judge whether the
    /// retrieved memories fully support the gold answer. `None` when no LLM
    /// judge is configured — the support verdict needs the LLM, so a
    /// heuristic run leaves the question unjudged rather than guessing.
    /// Returns `(supported, reasoning)`.
    async fn judge_support(
        &self,
        question: &str,
        ground_truth: &str,
        retrieved: &[String],
    ) -> (Option<bool>, String) {
        #[cfg(feature = "live-llm")]
        if let Some(judge) = &self.judge {
            return match judge.judge_support(question, ground_truth, retrieved).await {
                Some(v) => (Some(v.supported), v.reasoning),
                None => (None, String::new()),
            };
        }
        #[cfg(not(feature = "live-llm"))]
        let _ = (question, ground_truth, retrieved);
        (None, String::new())
    }
}

/// Wraps the optional `live-llm` answer synthesizer — same shape and
/// rationale as [`JudgeSlot`].
struct SynthSlot {
    #[cfg(feature = "live-llm")]
    synth: Option<crate::bench::synthesize::LlmSynthesizer>,
}

impl SynthSlot {
    fn from_env() -> Self {
        Self {
            #[cfg(feature = "live-llm")]
            synth: crate::bench::synthesize::LlmSynthesizer::from_env(),
        }
    }

    /// Compose an answer from a RECALL outcome: the LLM synthesizer when
    /// configured, else the heuristic. Both branch on the answer shape
    /// (grounded value / honest abstention / episodic snippets).
    async fn synthesize(
        &self,
        question: &str,
        outcome: &RecallOutcome,
        qtype: QuestionType,
        episodic_cap: usize,
    ) -> String {
        #[cfg(feature = "live-llm")]
        if let Some(synth) = &self.synth {
            return synth
                .synthesize(question, outcome, qtype, episodic_cap)
                .await;
        }
        synthesize_answer(question, outcome, qtype, episodic_cap)
    }
}

/// Drives a benchmark to completion and produces a [`BenchmarkReport`].
pub struct EvalRunner {
    config: RunConfig,
    competitor_fn: CompetitorBaselines,
    judge: JudgeSlot,
    synth: SynthSlot,
}

impl EvalRunner {
    /// New runner with explicit config and competitor table.
    #[must_use]
    pub fn new(config: RunConfig, competitor_fn: CompetitorBaselines) -> Self {
        Self {
            config,
            competitor_fn,
            judge: JudgeSlot::from_env(),
            synth: SynthSlot::from_env(),
        }
    }

    /// Compose an answer from a RECALL outcome.
    async fn synth_answer(&self, instance: &EvalInstance, outcome: &RecallOutcome) -> String {
        let cap = self.config.max_results as usize;
        self.synth
            .synthesize(&instance.question, outcome, instance.question_type, cap)
            .await
    }

    /// Score one answer.
    async fn judge_answer(&self, instance: &EvalInstance, system_answer: &str) -> JudgeResult {
        self.judge
            .judge(
                &instance.question_id,
                instance.question_type,
                &instance.question,
                &instance.answer,
                system_answer,
            )
            .await
    }

    /// Answer-supporting context recall — see [`JudgeSlot::judge_support`].
    async fn judge_support(
        &self,
        instance: &EvalInstance,
        retrieved: &[String],
    ) -> (Option<bool>, String) {
        self.judge
            .judge_support(&instance.question, &instance.answer, retrieved)
            .await
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
        // Type filter runs before the count cap so `max_questions` bounds
        // the filtered set, not the raw load.
        let instances = match &self.config.question_types {
            Some(types) => instances
                .into_iter()
                .filter(|i| types.iter().any(|t| t == i.question_type.tag()))
                .collect::<Vec<_>>(),
            None => instances,
        };
        let instances = match self.config.max_questions {
            Some(n) => instances.into_iter().take(n).collect::<Vec<_>>(),
            None => instances,
        };

        // Reflects the judge that actually ran, not just the feature —
        // `live-llm` with no API key still scores on the heuristic.
        let judge_type = self.judge.describe();
        let mut meta = BenchmarkMeta::new(
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

        // Read-only mode: reconnect to the agents a prior `--persist-agents`
        // run recorded and SKIP ingest entirely, so the read path can be
        // re-measured against the already-populated store with no write cost.
        let read_only = match &self.config.read_only_agents {
            Some(path) => {
                let m = AgentManifest::load(path).map_err(|e| EvalError::ParseError {
                    path: path.display().to_string(),
                    reason: format!("read-only agent manifest: {e}"),
                })?;
                println!(
                    "eval: read-only mode — reusing {} agents from {} (no ingest)",
                    m.len(),
                    path.display()
                );
                Some(m)
            }
            None => None,
        };
        // Persist mode: record each conversation's agent id so a later
        // read-only run can reuse it. Only meaningful on an ingesting run.
        let mut persist = self
            .config
            .persist_agents
            .as_ref()
            .map(|_| AgentManifest::new(benchmark.id(), self.config.endpoint.to_string()));

        // One minter for the whole run; mint a data-plane key per agent and
        // cache it so a reconnect of the same agent reuses its token rather
        // than minting a fresh one each time.
        let minter = self.config.key_minter();
        // Preflight: prove the admin plane can mint BEFORE iterating
        // conversations. Without this, a misconfigured admin plane (missing
        // BRAIN_EVAL_ADMIN_SECRET, wrong admin port, unreachable server) makes
        // every per-conversation mint fail and the run silently records all
        // questions as "failed" with no obvious cause. Fail fast with one clear
        // error instead. The throwaway key lands in the eval namespace and is
        // harmless. The docker-boot harness configures the admin plane itself.
        if let Err(e) = minter.mint(brain_db_sdk::new_id()).await {
            return Err(EvalError::Pipeline(format!(
                "preflight: cannot mint data-plane keys via the admin plane — {e}. \
                 The docker-boot harness sets the admin secret automatically; for an \
                 external server set BRAIN_EVAL_ADMIN_SECRET (and BRAIN_EVAL_ADMIN_PORT \
                 if the admin listener is not on 9092)."
            )));
        }
        let mut tokens: BTreeMap<[u8; 16], String> = BTreeMap::new();

        for (conv_key, idxs) in &groups {
            // Establish the harness and decide whether to ingest. Read-only
            // reconnects to the recorded agent and skips ingest entirely;
            // otherwise mint a fresh agent, ingest once, and (if persisting)
            // record the agent for future read-only reuse.
            #[allow(clippy::type_complexity)]
            let (
                harness,
                write_latency_ms,
                write_attempted,
                write_stored,
                write_deduplicated,
                ingest_err,
            ): (BrainEvalHarness, u64, u64, u64, u64, bool) = if let Some(manifest) = &read_only {
                let Some(agent) = manifest.agent_for(conv_key) else {
                    warn!(
                        conversation = %conv_key,
                        "read-only: no agent recorded for this conversation; recording its questions as failed",
                    );
                    for &idx in idxs {
                        let r = failed_question_result(
                            &instances[idx],
                            "read-only: no persisted agent for conversation".to_owned(),
                            /* ingest_failed = */ true,
                        );
                        tracker.record(&r);
                        question_results.push(r);
                    }
                    continue;
                };
                let reconnect = match mint_or_cached(&minter, &mut tokens, agent).await {
                    Ok(token) => {
                        BrainEvalHarness::connect_with_token(self.config.endpoint, agent, token)
                            .await
                    }
                    Err(e) => Err(e),
                };
                match reconnect {
                    Ok(h) => (h, 0, 0, 0, 0, false),
                    Err(e) => {
                        warn!(conversation = %conv_key, error = %e, "read-only: reconnect failed; recording group as failed");
                        for &idx in idxs {
                            let r = failed_question_result(
                                &instances[idx],
                                format!("read-only reconnect failed: {e}"),
                                true,
                            );
                            tracker.record(&r);
                            question_results.push(r);
                        }
                        continue;
                    }
                }
            } else {
                // Fresh agent for this conversation: mint its key, connect.
                let agent_id = brain_db_sdk::new_id();
                let connect = match mint_or_cached(&minter, &mut tokens, agent_id).await {
                    Ok(token) => {
                        BrainEvalHarness::connect_with_token(self.config.endpoint, agent_id, token)
                            .await
                    }
                    Err(e) => Err(e),
                };
                let harness = match connect {
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

                // Record the agent so a later read-only run can reuse it.
                if let Some(p) = persist.as_mut() {
                    p.record(conv_key, harness.agent_id());
                }

                // ---- ingest once ----
                let first = &instances[idxs[0]];
                let outcome = ingest_sessions(&harness, first).await;

                // ---- wait for async write-time extraction to drain ----
                // Brain's extractors (entity / statement / relation) run
                // asynchronously after ENCODE acks (extraction → statement
                // embedding). Querying immediately would hit a half-built
                // graph / empty embeddings, so we block until that whole
                // async pipeline plateaus before reading — this is the
                // correct default for measurement. Set BRAIN_EVAL_NO_DRAIN
                // to a truthy value (1/true/yes/on) to opt out and read
                // against the live, still-settling graph.
                if !env_truthy("BRAIN_EVAL_NO_DRAIN") {
                    let admin_port = std::env::var("BRAIN_EVAL_DRAIN_ADMIN_PORT")
                        .ok()
                        .and_then(|p| p.parse::<u16>().ok())
                        .unwrap_or(9091);
                    let metrics_addr = SocketAddr::new(self.config.endpoint.ip(), admin_port);
                    wait_for_write_flush(metrics_addr, conv_key).await;
                }
                (
                    harness, outcome.0, outcome.1, outcome.2, outcome.3, outcome.4,
                )
            };

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

        // Persist the conversation→agent manifest so a later
        // `BRAIN_EVAL_READ_ONLY_AGENTS` run can re-query this populated
        // store without re-ingesting.
        if let (Some(p), Some(path)) = (persist.as_ref(), self.config.persist_agents.as_ref()) {
            match p.save(path) {
                Ok(()) => println!(
                    "eval: wrote {} agents to {} (reuse with BRAIN_EVAL_READ_ONLY_AGENTS={})",
                    p.len(),
                    path.display(),
                    path.display()
                ),
                Err(e) => warn!(error = %e, "failed to write agent manifest"),
            }
        }

        // Record how many questions the LLM judge graded with its heuristic
        // fallback, so a run where the judge died partway can never look
        // fully LLM-judged in the saved report.
        meta.judge_heuristic_fallbacks = self.judge.heuristic_fallback_count();

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
            .recall(&instance.question, self.config.max_results)
            .await;

        let (
            answer_kind,
            retrieved_memory_contents,
            read_latency_ms,
            retrieval_failed,
            system_answer,
        ) = match recall {
            Ok(outcome) => {
                let answer_kind = answer_kind_from_wire(outcome.answer_kind);
                let retrieved: Vec<String> =
                    outcome.memories.iter().map(|m| m.text.clone()).collect();
                let latency = outcome.latency_ms;
                let answer = self.synth_answer(instance, &outcome).await;
                (answer_kind, retrieved, latency, false, answer)
            }
            Err(e) => {
                warn!(
                    question_id = %instance.question_id,
                    error = %e,
                    "recall failed; recording as retrieval_failed",
                );
                (AnswerKind::Error, Vec::new(), 0, true, String::new())
            }
        };

        let memories_retrieved = retrieved_memory_contents.len();
        let judged = self.judge_answer(instance, &system_answer).await;
        // Grade retrieval independently of synthesis: did Brain surface the
        // memory that supports the gold answer? Skip on a retrieval error
        // (nothing meaningful to grade).
        let (context_supported, context_support_reasoning) = if retrieval_failed {
            (None, String::new())
        } else {
            self.judge_support(instance, &retrieved_memory_contents)
                .await
        };

        QuestionResult {
            question_id: instance.question_id.clone(),
            question_type: instance.question_type,
            question: instance.question.clone(),
            ground_truth: instance.answer.clone(),
            system_answer,
            answer_kind,
            verdict: judged.verdict,
            score: judged.score,
            write_latency_ms,
            read_latency_ms,
            tokens_write: 0,
            tokens_read: 0,
            memories_retrieved,
            retrieved_memory_contents,
            judge_reasoning: judged.reasoning,
            context_supported,
            context_support_reasoning,
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
            "[{:>4}/{}] {:<9} {:<10} acc={:.3} q={}{}",
            self.done,
            self.total,
            format!("{:?}", result.verdict),
            result.answer_kind,
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

/// Get the cached data-plane token for `agent_id`, minting (and caching) a
/// fresh one on first use. Caching keeps a reconnect of the same agent on the
/// same key instead of minting a new one each time.
async fn mint_or_cached(
    minter: &KeyMinter,
    tokens: &mut BTreeMap<[u8; 16], String>,
    agent_id: [u8; 16],
) -> Result<String, HarnessError> {
    if let Some(t) = tokens.get(&agent_id) {
        return Ok(t.clone());
    }
    let token = minter.mint(agent_id).await?;
    tokens.insert(agent_id, token.clone());
    Ok(token)
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
        answer_kind: AnswerKind::Error,
        verdict: Verdict::Incorrect,
        score: 0.0,
        write_latency_ms: 0,
        read_latency_ms: 0,
        tokens_write: 0,
        tokens_read: 0,
        memories_retrieved: 0,
        retrieved_memory_contents: Vec::new(),
        judge_reasoning: reason,
        context_supported: None,
        context_support_reasoning: String::new(),
        ingestion_failed: ingest_failed,
        retrieval_failed: !ingest_failed,
        write_attempted: 0,
        write_stored: 0,
        write_deduplicated: 0,
    }
}

/// The one place `AnswerKindWire` (wire type) converts to `AnswerKind`
/// (eval-only type, adds the `Error` state the wire can't represent) —
/// keeps `core` free of any SDK dependency.
fn answer_kind_from_wire(kind: AnswerKindWire) -> AnswerKind {
    match kind {
        AnswerKindWire::Single => AnswerKind::Single,
        AnswerKindWire::Many => AnswerKind::Many,
        AnswerKindWire::None => AnswerKind::None,
    }
}

/// Parse an environment variable as a boolean flag. Unset, empty, or any
/// non-affirmative value is `false`; `1`/`true`/`yes`/`on` (case-insensitive)
/// are `true`. Used for opt-out toggles where presence alone must NOT flip
/// the flag — `FLAG=0` and `FLAG=false` keep the default behaviour.
fn env_truthy(var: &str) -> bool {
    std::env::var(var)
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// Block until the server's async write pipeline has fully flushed, so a
/// conversation's entity / statement / relation graph — and its
/// embeddings — are reflected in the read indexes before we query it.
///
/// A write is two-phase: the synchronous memory record (WAL-acked at
/// ENCODE) and the asynchronous tail (extraction of entities /
/// statements / relations, then statement embedding). Grounded recall
/// depends on that async tail, so reads must not fire until it settles.
/// We poll the COMBINED progress of both async stages —
/// `brain_extractor_items_written_total` + `brain_statement_embed_rows_embedded_total`
/// — and treat a plateau (no increase across a stable window, after at
/// least one write has landed) as flushed. Watching only extraction
/// would fire reads while statement embeddings are still being computed.
/// Best-effort: failed polls keep retrying; the cap just proceeds.
async fn wait_for_write_flush(metrics_addr: SocketAddr, conv: &str) {
    use std::time::Duration;
    const INTERVAL: Duration = Duration::from_secs(5);
    const STABLE_NEEDED: u32 = 12; // 60s plateau — tolerates slow LLM batches
    const MIN_POLLS: u32 = 6; // ≥30s floor so the workers have started
    const MAX_POLLS: u32 = 360; // 30 min hard cap

    let mut last: i64 = -1;
    let mut stable: u32 = 0;
    for i in 0..MAX_POLLS {
        // A scrape failure (endpoint unreachable) is `None` here and skips
        // this poll without resetting the plateau — a transient scrape
        // failure shouldn't be mistaken for the pipeline going backwards.
        // A successful scrape with the family simply absent (nothing
        // extracted yet) is `0`, not `None` — `PrometheusMetrics::sum`
        // returning `None` for an absent family is folded into 0 via
        // `unwrap_or(0.0)` below, since "connected but no samples yet" and
        // "not written yet" mean the same thing here.
        let scraped = PrometheusMetrics::scrape(metrics_addr).await.ok();
        if let Some(m) = scraped.as_ref() {
            let ex = m.sum("brain_extractor_items_written_total").unwrap_or(0.0) as i64;
            let em = m
                .sum("brain_statement_embed_rows_embedded_total")
                .unwrap_or(0.0) as i64;
            let cur = ex + em;
            if cur == last {
                stable += 1;
            } else {
                stable = 0;
                last = cur;
            }
            // Require some async work to have actually landed (`cur > 0`)
            // before honoring a plateau, so we don't mistake "workers
            // haven't picked up yet" for "flushed".
            if i + 1 >= MIN_POLLS && cur > 0 && stable >= STABLE_NEEDED {
                tracing::info!(
                    conversation = %conv,
                    extracted = ex,
                    statement_rows_embedded = em,
                    "async write pipeline flushed"
                );
                return;
            }
        }
        tokio::time::sleep(INTERVAL).await;
    }
    warn!(conversation = %conv, "write-flush wait hit max; proceeding");
}
