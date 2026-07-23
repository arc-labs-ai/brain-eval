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
use std::time::{SystemTime, UNIX_EPOCH};

use tracing::warn;

use crate::bench::agent_manifest::AgentManifest;
use crate::bench::config::{ReporterKind, RunConfig};
use crate::bench::core::benchmark::{Benchmark, EvalError};
use crate::bench::core::instance::{EvalInstance, QuestionType};
use crate::bench::core::outcome::{AnswerKind, JudgeResult, QuestionResult, Verdict};
use crate::bench::report::baselines::CompetitorBaselines;
use crate::bench::report::format::{
    json::JsonReporter, run_json::JsonRunReporter, text::TextReporter, Reporter,
};
use crate::bench::report::run_report::{
    ConversationWrite, DrainBarrierStats, ExtractionHealth, FailStage, FailureAttribution,
    MemoryExtraction, MemoryGraph, PlanTrace, QuestionRecord, ReadReport, RetrievedMemory,
    RunMetadata, RunMode, RunReport, WriteReport,
};
use crate::bench::report::shape::{BenchmarkMeta, BenchmarkReport};
use crate::bench::score::metrics::compute_full_metrics;
use crate::bench::synthesize::synthesize_answer;
use crate::client::auth::KeyMinter;
use crate::client::harness::{BrainEvalHarness, HarnessError, RecallOutcome};
use crate::client::metrics::PrometheusMetrics;
use brain_db_sdk::wire::types::{
    AnswerKindWire, MemoryListDirWire, MemoryListRequest, MemoryListSortWire,
    MemoryListTimeAxisWire, QueryRequest, QueryTraceRequest, RetrieverSelectionWire, WireMemoryId,
};
use brain_db_sdk::RecallBuilder;

use crate::bench::score::judge::Judge;

/// Wraps the optional `live-llm` answer synthesizer — the synthesis-side
/// sibling of [`Judge`] (runner-local; only this module synthesizes).
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
    judge: Judge,
    synth: SynthSlot,
}

impl EvalRunner {
    /// New runner with explicit config and competitor table.
    #[must_use]
    pub fn new(config: RunConfig, competitor_fn: CompetitorBaselines) -> Self {
        Self {
            config,
            competitor_fn,
            judge: Judge::from_env(),
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
            .grade(
                &instance.question_id,
                instance.question_type,
                &instance.question,
                &instance.answer,
                system_answer,
            )
            .await
    }

    /// Answer-supporting context recall — see [`Judge::grade_support`].
    async fn judge_support(
        &self,
        instance: &EvalInstance,
        retrieved: &[String],
    ) -> (Option<bool>, String) {
        self.judge
            .grade_support(&instance.question, &instance.answer, retrieved)
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
        // Conversation selector: keep only instances whose conversation_id
        // matches, so a single conversation's haystack is the only thing
        // ingested. Applied before the count cap so `max_questions` bounds
        // the selected conversation, not the raw load.
        let instances = match &self.config.only_conversation {
            Some(target) => {
                let kept = instances
                    .into_iter()
                    .filter(|i| i.conversation_id.as_deref() == Some(target.as_str()))
                    .collect::<Vec<_>>();
                println!(
                    "eval: --only-conversation {target} — {} question(s) selected",
                    kept.len()
                );
                kept
            }
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
        // Exhaustive run-report accumulators, filled alongside the summary
        // results and assembled into the `*.run.json` artifact at the end.
        let mut conversation_writes: Vec<ConversationWrite> = Vec::new();
        let mut question_records: Vec<QuestionRecord> = Vec::with_capacity(instances.len());
        // The most recent drain plateau — extraction counters are cumulative
        // and server-wide, so the last conversation's plateau is the most
        // complete snapshot for the run.
        let mut last_drain: Option<DrainBarrierStats> = None;

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

        // Full-trace capture (default ON): per-read plan trace + graph and
        // per-stored-memory extraction. Each adds server round-trips, so a
        // large run can opt out with `BRAIN_EVAL_FULL_TRACE=0`. Read-only runs
        // still capture reads; only the write-side capture is skipped there.
        let full_trace = full_trace_enabled();
        if !full_trace {
            println!("eval: BRAIN_EVAL_FULL_TRACE off — skipping per-read trace/graph capture");
        }

        // Drain / extraction-health metrics endpoint — identical for every
        // conversation, `None` when drain is opted out. Reused after the loop
        // to scrape run-level extraction health.
        let drain_metrics_addr: Option<SocketAddr> = if env_truthy("BRAIN_EVAL_NO_DRAIN") {
            None
        } else {
            let admin_port = std::env::var("BRAIN_EVAL_DRAIN_ADMIN_PORT")
                .ok()
                .and_then(|p| p.parse::<u16>().ok())
                .unwrap_or(9091);
            Some(SocketAddr::new(self.config.endpoint.ip(), admin_port))
        };

        for (conv_key, idxs) in &groups {
            // Establish the harness and decide whether to ingest. Read-only
            // reconnects to the recorded agent and skips ingest entirely;
            // otherwise mint a fresh agent, ingest once, and (if persisting)
            // record the agent for future read-only reuse.
            let (harness, cw): (BrainEvalHarness, ConvWrite) = if let Some(manifest) = &read_only {
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
                        record_failed(
                            &mut tracker,
                            &mut question_results,
                            &mut question_records,
                            r,
                        );
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
                    Ok(h) => (h, ConvWrite::skipped()),
                    Err(e) => {
                        warn!(conversation = %conv_key, error = %e, "read-only: reconnect failed; recording group as failed");
                        for &idx in idxs {
                            let r = failed_question_result(
                                &instances[idx],
                                format!("read-only reconnect failed: {e}"),
                                true,
                            );
                            record_failed(
                                &mut tracker,
                                &mut question_results,
                                &mut question_records,
                                r,
                            );
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
                            record_failed(
                                &mut tracker,
                                &mut question_results,
                                &mut question_records,
                                r,
                            );
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
                let ingest = ingest_sessions(&harness, first).await;

                // ---- wait for async write-time extraction to drain ----
                // Brain's extractors (entity / statement / relation) run
                // asynchronously after ENCODE acks (extraction → statement
                // embedding). Querying immediately would hit a half-built
                // graph / empty embeddings, so we block until that whole
                // async pipeline plateaus before reading — this is the
                // correct default for measurement. Set BRAIN_EVAL_NO_DRAIN
                // to a truthy value (1/true/yes/on) to opt out and read
                // against the live, still-settling graph.
                let drain = match drain_metrics_addr {
                    Some(addr) => Some(wait_for_write_flush(addr, conv_key).await),
                    None => None,
                };
                (harness, ConvWrite::from_ingest(ingest, drain))
            };

            // Ledger the conversation's write (ingesting runs only — a
            // read-only reconnect performed no write worth recording). The
            // drain barrier has already settled by here, so the per-memory
            // extraction capture reads a fully-built graph.
            if read_only.is_none() {
                if let Some(d) = cw.drain.clone() {
                    last_drain = Some(d);
                }
                let memory_extractions = if full_trace {
                    capture_memory_extractions(
                        harness.client(),
                        &cw.stored_texts,
                        &cw.stored_ids,
                        self.config.max_results,
                    )
                    .await
                } else {
                    Vec::new()
                };
                conversation_writes.push(ConversationWrite {
                    conversation_id: conv_key.clone(),
                    agent_id: uuid::Uuid::from_bytes(harness.agent_id())
                        .hyphenated()
                        .to_string(),
                    write_latency_ms: cw.latency_ms,
                    attempted: cw.attempted,
                    stored: cw.stored,
                    deduplicated: cw.deduplicated,
                    stored_memory_ids: cw.stored_ids.iter().map(ToString::to_string).collect(),
                    memory_extractions,
                    ingest_failed: cw.ingest_failed,
                });
            }

            // ---- MEMORY_LIST verification (opt-in) ----
            // Enumerate the tenant timeline on the SAME authenticated agent.
            // After an ingesting run the drain barrier has settled, so this
            // cross-checks against what ingest reported storing (`cw.stored`).
            // In a read-only run there is no ingest count (`cw.stored == 0`),
            // so it just reports the listed total — the proof that a
            // persisted, migrated corpus is still fully enumerable.
            if self.config.verify_memory_list {
                verify_memory_list(harness.client(), conv_key, cw.stored).await;
            }

            // ---- per-question retrieval + judging ----
            // `--verify-memory-list` is a list-only diagnostic: it proves the
            // tenant timeline is enumerable and skips the (LLM-judged, paid)
            // per-question pass entirely.
            if !self.config.verify_memory_list {
                for &idx in idxs {
                    let inst = &instances[idx];
                    let run = self
                        .run_question(
                            &harness,
                            inst,
                            cw.latency_ms,
                            cw.attempted,
                            cw.stored,
                            cw.deduplicated,
                            cw.ingest_failed,
                            full_trace,
                        )
                        .await;
                    tracker.record(&run.result);
                    question_records.push(QuestionRecord {
                        result: run.result.clone(),
                        retrieved: run.retrieved,
                        fail_stage: run.fail_stage,
                        plan_trace: run.plan_trace,
                    });
                    question_results.push(run.result);
                }
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
        let report = BenchmarkReport::build(meta, metrics, competitors.clone(), question_results);

        // ---- assemble the exhaustive run report ----
        // Extraction health is a single run-level snapshot: the extractor
        // counters are cumulative and server-wide, so one scrape after every
        // conversation has drained is the most complete reading. Only on an
        // ingesting run with a reachable metrics endpoint.
        let extraction = if read_only.is_none() {
            match drain_metrics_addr {
                Some(addr) => PrometheusMetrics::scrape(addr)
                    .await
                    .ok()
                    .map(|m| ExtractionHealth::from_metrics(&m)),
                None => None,
            }
        } else {
            None
        };
        let run_finished = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let run_meta = RunMetadata::from_parts(
            &report.meta,
            &self.config,
            RunMode::from_config(&self.config),
            run_finished,
        );
        let failure_attribution = FailureAttribution::from_records(&question_records);
        let run_report = RunReport::new(
            run_meta,
            WriteReport {
                summary: report.metrics.write_quality.clone(),
                conversations: conversation_writes,
                extraction,
                drain: last_drain,
            },
            ReadReport {
                per_question: question_records,
                failure_attribution,
            },
            report.metrics.clone(),
            competitors,
        );

        if let Err(e) = self.write_reports(&report, &run_report) {
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
        full_trace: bool,
    ) -> QuestionRun {
        let recall = harness
            .recall(&instance.question, self.config.max_results)
            .await;

        // Keep the FULL retrieved memories, not just their text — the run
        // report captures every scored field the SDK returned.
        let (answer_kind, mut retrieved, read_latency_ms, retrieval_failed, system_answer) =
            match recall {
                Ok(outcome) => {
                    let answer_kind = answer_kind_from_wire(outcome.answer_kind);
                    let retrieved: Vec<RetrievedMemory> = outcome
                        .memories
                        .iter()
                        .map(RetrievedMemory::from_wire)
                        .collect();
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

        // Stage-level read capture (full-trace only): the server's per-stage
        // plan/execution trace for this cue, and the extracted typed-graph per
        // retrieved memory (a second recall with graph enrichment, matched to
        // the measured hits by id). Both are extra round-trips, so they are
        // gated on `BRAIN_EVAL_FULL_TRACE`; a failure warns and leaves the
        // capture empty rather than failing the question.
        let mut plan_trace: Option<PlanTrace> = None;
        if full_trace && !retrieval_failed {
            plan_trace = self.capture_plan_trace(harness, instance).await;
            self.attach_retrieved_graph(harness, instance, &mut retrieved)
                .await;
        }

        // The summary result carries texts only (metrics read them); the
        // run-report record carries the full `retrieved` alongside.
        let retrieved_memory_contents: Vec<String> =
            retrieved.iter().map(|m| m.text.clone()).collect();
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

        // Attribute a non-correct answer to the stage that lost it. Skip
        // attribution on an infrastructure retrieval failure — that is not a
        // model/retrieval quality signal.
        let fail_stage = if retrieval_failed {
            None
        } else {
            FailStage::attribute(
                matches!(judged.verdict, Verdict::Correct),
                memories_retrieved,
                context_supported,
            )
        };

        let result = QuestionResult {
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
        };
        QuestionRun {
            result,
            retrieved,
            fail_stage,
            plan_trace,
        }
    }

    /// Capture the server's `QUERY_TRACE` for this question's cue. Best-effort:
    /// a trace failure warns and yields `None`.
    async fn capture_plan_trace(
        &self,
        harness: &BrainEvalHarness,
        instance: &EvalInstance,
    ) -> Option<PlanTrace> {
        let req = QueryTraceRequest {
            query: trace_query_request(&instance.question, self.config.max_results),
        };
        match harness.client().query_trace(&req).await {
            Ok(resp) => Some(PlanTrace::from_wire(&resp)),
            Err(e) => {
                warn!(
                    question_id = %instance.question_id,
                    error = %e,
                    "query_trace capture failed; plan_trace omitted",
                );
                None
            }
        }
    }

    /// Re-recall this question with graph enrichment and attach each returned
    /// memory's extracted graph to the matching entry in `retrieved` (by id).
    /// Best-effort: a failure warns and leaves the graphs unset.
    async fn attach_retrieved_graph(
        &self,
        harness: &BrainEvalHarness,
        instance: &EvalInstance,
        retrieved: &mut [RetrievedMemory],
    ) {
        let req = RecallBuilder::new(instance.question.as_str())
            .max_results(self.config.max_results)
            .include_graph(true)
            .include_text(true)
            .build();
        let enriched = match harness.client().recall(&req).await {
            Ok(a) => a,
            Err(e) => {
                warn!(
                    question_id = %instance.question_id,
                    error = %e,
                    "graph-enriched recall failed; retrieved graphs omitted",
                );
                return;
            }
        };
        for m in &enriched.memories {
            let Some(g) = &m.graph else { continue };
            let id = m.memory_id.to_string();
            if let Some(rm) = retrieved.iter_mut().find(|r| r.memory_id == id) {
                rm.graph = Some(MemoryGraph::from_wire(g));
            }
        }
    }

    fn write_reports(
        &self,
        report: &BenchmarkReport,
        run_report: &RunReport,
    ) -> std::io::Result<()> {
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

        // The exhaustive run report is always written (independent of the
        // configured summary reporters) — it is the lossless artifact.
        let run_path = self.config.output_dir.join(format!("{stem}.run.json"));
        if let Err(ref e) = JsonRunReporter.write(run_report, &run_path) {
            warn!(error = %e, "run-report write failed");
        }
        Ok(())
    }
}

/// The full output of [`EvalRunner::run_question`]: the summary result plus
/// the run-report extras (full retrieved memories + attributed fail stage).
struct QuestionRun {
    result: QuestionResult,
    retrieved: Vec<RetrievedMemory>,
    fail_stage: Option<FailStage>,
    plan_trace: Option<PlanTrace>,
}

/// Per-conversation write state threaded from ingest through to the
/// conversation ledger.
struct ConvWrite {
    latency_ms: u64,
    attempted: u64,
    stored: u64,
    deduplicated: u64,
    ingest_failed: bool,
    stored_ids: Vec<WireMemoryId>,
    stored_texts: Vec<String>,
    drain: Option<DrainBarrierStats>,
}

impl ConvWrite {
    /// A read-only reconnect performed no write.
    fn skipped() -> Self {
        Self {
            latency_ms: 0,
            attempted: 0,
            stored: 0,
            deduplicated: 0,
            ingest_failed: false,
            stored_ids: Vec::new(),
            stored_texts: Vec::new(),
            drain: None,
        }
    }

    /// Build from an ingest aggregate and the drain plateau (if drained).
    fn from_ingest(ingest: IngestAggregate, drain: Option<DrainBarrierStats>) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        let stored = ingest.stored_ids.len() as u64;
        Self {
            latency_ms: ingest.latency_ms,
            attempted: ingest.attempted,
            stored,
            deduplicated: ingest.deduplicated,
            ingest_failed: ingest.failed,
            stored_ids: ingest.stored_ids,
            stored_texts: ingest.stored_texts,
            drain,
        }
    }
}

/// Record an infrastructure-failed question into all three sinks: the
/// progress tracker, the summary results, and the run-report records (with
/// no retrieved memories and no attributable fail stage).
fn record_failed(
    tracker: &mut IncrementalTracker,
    results: &mut Vec<QuestionResult>,
    records: &mut Vec<QuestionRecord>,
    r: QuestionResult,
) {
    tracker.record(&r);
    records.push(QuestionRecord {
        result: r.clone(),
        retrieved: Vec::new(),
        fail_stage: None,
        plan_trace: None,
    });
    results.push(r);
}

/// Whether full-trace stage capture is enabled — default ON. Only an explicit
/// falsey value (`0`/`false`/`no`/`off`) turns it off, so a fresh run captures
/// everything unless the operator opts out for a large batch.
fn full_trace_enabled() -> bool {
    match std::env::var("BRAIN_EVAL_FULL_TRACE") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        ),
        Err(_) => true,
    }
}

/// Build the `QueryRequest` the `QUERY_TRACE` capture traces: Auto routing,
/// default fusion, no filters — the same shape a plain RECALL takes.
fn trace_query_request(text: &str, limit: u32) -> QueryRequest {
    QueryRequest {
        text: text.to_string(),
        entity_anchor: None,
        kind_filter: Vec::new(),
        predicate_filter: Vec::new(),
        time_filter: None,
        as_of_record_time_unix_nanos: None,
        confidence_min: None,
        include_tombstoned: false,
        include_superseded: false,
        limit,
        retrievers: RetrieverSelectionWire::Auto,
        fusion_config: None,
        request_id: brain_db_sdk::new_id(),
    }
}

/// Enumerate the calling agent's memories via MEMORY_LIST, paginating with
/// the opaque keyset cursor until exhausted, and print a verification
/// summary. Best-effort: a failed page is logged and stops the walk, never
/// aborts the run. `stored` is what ingest reported writing for this
/// conversation, used for a MATCH/MISMATCH cross-check.
async fn verify_memory_list(client: &brain_db_sdk::BrainClient, conv_key: &str, stored: u64) {
    const PAGE: u32 = 100;
    let mut cursor: Vec<u8> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    let mut listed: u64 = 0;
    loop {
        let request = MemoryListRequest {
            sort: MemoryListSortWire::Created,
            dir: MemoryListDirWire::Asc,
            limit: PAGE,
            cursor: cursor.clone(),
            kinds: Vec::new(),
            include_tombstoned: false,
            time_axis: MemoryListTimeAxisWire::Created,
            from_unix_nanos: 0,
            to_unix_nanos: 0,
            salience_min: 0.0,
            salience_max: 1.0,
            text_contains: String::new(),
            act_as: None,
        };
        let frames = match client.memory_list_frames(&request).await {
            Ok(f) => f,
            Err(e) => {
                warn!(
                    conversation = %conv_key,
                    error = %e,
                    "MEMORY_LIST verify :: page request failed; stopping walk",
                );
                break;
            }
        };
        let mut next_cursor: Vec<u8> = Vec::new();
        for frame in frames {
            #[allow(clippy::cast_possible_truncation)]
            {
                listed = listed.saturating_add(frame.items.len() as u64);
            }
            for item in frame.items {
                if texts.len() < 5 {
                    texts.push(item.text);
                }
            }
            if !frame.next_cursor.is_empty() {
                next_cursor = frame.next_cursor;
            }
        }
        if next_cursor.is_empty() {
            break;
        }
        cursor = next_cursor;
    }

    println!("MEMORY_LIST verify :: conversation {conv_key}: {listed} memories listed");
    for (i, t) in texts.iter().enumerate() {
        let trimmed: String = t.chars().take(80).collect();
        let ellipsis = if t.chars().count() > 80 { "…" } else { "" };
        println!("MEMORY_LIST verify ::   [{i}] {trimmed}{ellipsis}");
    }
    if stored > 0 {
        let verdict = if listed == stored {
            "MATCH"
        } else {
            "MISMATCH"
        };
        println!("MEMORY_LIST verify :: listed {listed} vs ingest-stored {stored} => {verdict}");
    }
    assert!(
        listed > 0,
        "MEMORY_LIST verify :: conversation {conv_key} listed 0 memories after drain"
    );
}

/// Capture the extracted typed-graph for each stored memory. For every
/// `(text, id)` pair, recall by the memory's own turn text with graph
/// enrichment and pick out the hit whose id matches, recording its graph.
/// Best-effort per memory: a failed recall or a missing hit is skipped, not
/// fatal. Only memories whose graph was found are recorded.
async fn capture_memory_extractions(
    client: &brain_db_sdk::BrainClient,
    stored_texts: &[String],
    stored_ids: &[WireMemoryId],
    max_results: u32,
) -> Vec<MemoryExtraction> {
    let mut out: Vec<MemoryExtraction> = Vec::new();
    for (text, id) in stored_texts.iter().zip(stored_ids.iter()) {
        let req = RecallBuilder::new(text.as_str())
            .max_results(max_results)
            .include_graph(true)
            .include_text(true)
            .build();
        let ans = match client.recall(&req).await {
            Ok(a) => a,
            Err(e) => {
                warn!(memory_id = %id, error = %e, "write-side graph recall failed; skipping");
                continue;
            }
        };
        let hit = ans.memories.iter().find(|m| m.memory_id == *id);
        if let Some(g) = hit.and_then(|m| m.graph.as_ref()) {
            out.push(MemoryExtraction {
                memory_id: id.to_string(),
                graph: MemoryGraph::from_wire(g),
            });
        }
    }
    out
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

/// The aggregate outcome of ingesting one conversation's sessions.
struct IngestAggregate {
    latency_ms: u64,
    attempted: u64,
    deduplicated: u64,
    failed: bool,
    /// Ids of the fresh memories the substrate accepted — previously
    /// discarded after counting, now retained for the run report.
    stored_ids: Vec<WireMemoryId>,
    /// Source turn text per stored id (same order/length as `stored_ids`),
    /// used to re-fetch each stored memory's extracted graph.
    stored_texts: Vec<String>,
}

async fn ingest_sessions(harness: &BrainEvalHarness, inst: &EvalInstance) -> IngestAggregate {
    let mut total_latency_ms = 0u64;
    let mut attempted = 0u64;
    let mut deduplicated = 0u64;
    let mut failed = false;
    let mut stored_ids: Vec<WireMemoryId> = Vec::new();
    let mut stored_texts: Vec<String> = Vec::new();

    for session in &inst.sessions {
        match harness.ingest(&session.turns).await {
            Ok(out) => {
                total_latency_ms = total_latency_ms.saturating_add(out.latency_ms);
                attempted = attempted.saturating_add(out.attempted);
                stored_ids.extend(out.stored_ids);
                stored_texts.extend(out.stored_texts);
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

    IngestAggregate {
        latency_ms: total_latency_ms,
        attempted,
        deduplicated,
        failed,
        stored_ids,
        stored_texts,
    }
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
pub(crate) async fn wait_for_write_flush(
    metrics_addr: SocketAddr,
    conv: &str,
) -> DrainBarrierStats {
    use std::time::Duration;
    const INTERVAL: Duration = Duration::from_secs(5);
    const STABLE_NEEDED: u32 = 12; // 60s plateau — tolerates slow LLM batches
    const MIN_POLLS: u32 = 6; // ≥30s floor so the workers have started
    const MAX_POLLS: u32 = 360; // 30 min hard cap

    let mut last: i64 = -1;
    let mut stable: u32 = 0;
    let mut last_ex: u64 = 0;
    let mut last_em: u64 = 0;
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
            last_ex = ex.max(0) as u64;
            last_em = em.max(0) as u64;
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
                return DrainBarrierStats {
                    extractor_items_written: last_ex,
                    statement_rows_embedded: last_em,
                    flushed: true,
                };
            }
        }
        tokio::time::sleep(INTERVAL).await;
    }
    warn!(conversation = %conv, "write-flush wait hit max; proceeding");
    DrainBarrierStats {
        extractor_items_written: last_ex,
        statement_rows_embedded: last_em,
        flushed: false,
    }
}
