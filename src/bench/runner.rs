//! `EvalRunner` — drive a benchmark end-to-end.
//!
//! For each `EvalInstance`:
//! 1. Spin up a fresh harness (= fresh `SpaceId` = guaranteed isolation
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

use crate::bench::config::{ReporterKind, RunConfig};
use crate::bench::core::benchmark::{Benchmark, EvalError};
use crate::bench::core::instance::{EvalInstance, QuestionType};
use crate::bench::core::outcome::{AnswerKind, JudgeResult, QuestionResult, Verdict};
use crate::bench::report::baselines::CompetitorBaselines;
use crate::bench::report::format::{
    html::HtmlReporter, json::JsonReporter, run_json::JsonRunReporter, text::TextReporter, Reporter,
};
use crate::bench::report::run_report::{
    ConversationWrite, DrainBarrierStats, ExtractionHealth, FailStage, FailureAttribution,
    MemoryExtraction, MemoryGraph, PlanTrace, QuestionRecord, ReadReport, RetrievedMemory,
    RunMetadata, RunMode, RunReport, WriteReport,
};
use crate::bench::report::shape::{BenchmarkMeta, BenchmarkReport};
use crate::bench::score::metrics::compute_full_metrics;
use crate::bench::space_manifest::SpaceManifest;
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
        )
        .with_concurrency(self.config.concurrency.max(1));

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

        // What phase 1 (connect + ingest) hands to phase 2 (read back and
        // judge). The connection is deliberately NOT carried: phase 2
        // reconnects on the cached token, so a run of N conversations never
        // holds N sockets at once.
        struct Prepared {
            conv_key: String,
            idxs: Vec<usize>,
            space_id: [u8; 16],
            cw: ConvWrite,
        }
        let mut prepared: Vec<Prepared> = Vec::with_capacity(groups.len());
        // Whether any conversation actually wrote. A read-only run ingests
        // nothing, so there is nothing to wait for and the barrier is
        // skipped entirely rather than waiting out its floor for no reason.
        let mut ingested_any = false;

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

        // Read-only mode: reconnect to the spaces a prior `--persist-spaces`
        // run recorded and SKIP ingest entirely, so the read path can be
        // re-measured against the already-populated store with no write cost.
        let read_only = match &self.config.read_only_spaces {
            Some(path) => {
                let m = SpaceManifest::load(path).map_err(|e| EvalError::ParseError {
                    path: path.display().to_string(),
                    reason: format!("read-only space manifest: {e}"),
                })?;
                println!(
                    "eval: read-only mode — reusing {} spaces from {} (no ingest)",
                    m.len(),
                    path.display()
                );
                Some(m)
            }
            None => None,
        };
        // Persist mode: record each conversation's space id so a later
        // read-only run can reuse it. Only meaningful on an ingesting run.
        let mut persist = self
            .config
            .persist_spaces
            .as_ref()
            .map(|_| SpaceManifest::new(benchmark.id(), self.config.endpoint.to_string()));

        // One minter for the whole run; mint a data-plane key per space and
        // cache it so a reconnect of the same space reuses its token rather
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

        // ---- phase 0: resolve each conversation's space + token ----
        //
        // Minting is an admin-plane round-trip and it mutates the shared
        // token cache, so it happens once, serially, before any concurrent
        // work. That leaves the concurrent phase with nothing shared to
        // contend on and makes phase 2's reconnect a pure cache hit.
        struct Planned {
            conv_key: String,
            idxs: Vec<usize>,
            /// `None` when this conversation cannot be run at all (read-only
            /// with no recorded space, or minting failed); `reason` says why.
            space: Option<([u8; 16], String)>,
            reason: String,
        }
        let mut planned: Vec<Planned> = Vec::with_capacity(groups.len());
        for (conv_key, idxs) in &groups {
            let (space_id, why) = if let Some(manifest) = &read_only {
                match manifest.space_for(conv_key) {
                    Some(space) => (Some(space), String::new()),
                    None => {
                        warn!(
                            conversation = %conv_key,
                            "read-only: no space recorded for this conversation; recording its questions as failed",
                        );
                        (
                            None,
                            "read-only: no persisted space for conversation".to_owned(),
                        )
                    }
                }
            } else {
                (Some(brain_db_sdk::new_id()), String::new())
            };

            let space = match space_id {
                None => None,
                Some(id) => match mint_or_cached(&minter, &mut tokens, id).await {
                    Ok(token) => {
                        // Record the space so a later read-only run can reuse it.
                        if read_only.is_none() {
                            if let Some(pp) = persist.as_mut() {
                                pp.record(conv_key, id);
                            }
                        }
                        Some((id, token))
                    }
                    Err(e) => {
                        warn!(conversation = %conv_key, error = %e, "key mint failed");
                        None
                    }
                },
            };
            let reason = if space.is_none() && why.is_empty() {
                "data-plane key mint failed".to_owned()
            } else {
                why
            };
            planned.push(Planned {
                conv_key: conv_key.clone(),
                idxs: idxs.clone(),
                space,
                reason,
            });
        }

        // ---- phase 1: connect + ingest, bounded-concurrently ----
        //
        // Each conversation writes into its own `SpaceId`, so no ingest can
        // observe another's data and they may overlap freely. Sessions
        // WITHIN a conversation stay strictly ordered: they share one space,
        // and a stateful predicate's "newer value supersedes the older" is
        // defined by write order — the supersession benchmark exists to
        // check exactly that, so parallelising there would be measuring a
        // race instead.
        struct IngestOutcome {
            conv_key: String,
            idxs: Vec<usize>,
            space_id: Option<[u8; 16]>,
            cw: ConvWrite,
            /// Set when the conversation could not be established at all.
            failure: Option<String>,
        }

        let ingest_concurrency = self.config.concurrency.max(1);
        let endpoint = self.config.endpoint;
        let is_read_only = read_only.is_some();

        // Borrow, don't move: the instance list outlives every phase and is
        // read again by phase 2 and the report.
        let instances_ref = &instances;
        let ingest_futures = planned.into_iter().map(|pl| async move {
            let Some((space_id, token)) = pl.space else {
                return IngestOutcome {
                    conv_key: pl.conv_key,
                    idxs: pl.idxs,
                    space_id: None,
                    cw: ConvWrite::skipped(),
                    failure: Some(pl.reason),
                };
            };

            let harness = match BrainEvalHarness::connect_with_token(endpoint, space_id, token)
                .await
            {
                Ok(h) => h,
                Err(e) => {
                    warn!(
                        conversation = %pl.conv_key,
                        error = %e,
                        "harness connect failed; recording every question in this group as failed",
                    );
                    return IngestOutcome {
                        conv_key: pl.conv_key,
                        idxs: pl.idxs,
                        space_id: None,
                        cw: ConvWrite::skipped(),
                        failure: Some(format!("connect failed: {e}")),
                    };
                }
            };

            // A read-only reconnect performed no write: there is nothing to
            // ingest and nothing for the barrier to wait on.
            let cw = if is_read_only {
                ConvWrite::skipped()
            } else {
                ConvWrite::from_ingest(ingest_sessions(&harness, &instances_ref[pl.idxs[0]]).await)
            };

            // Release the socket before phase 2. Phase 2 reconnects on the
            // cached token, so this costs a handshake rather than an admin
            // mint — and it keeps a run of N conversations from holding N
            // sockets at once, which on a 500-question benchmark is the
            // difference between working and exhausting the client's
            // file-descriptor limit.
            if let Err(e) = harness.close().await {
                warn!(error = %e, "harness close failed; continuing");
            }

            IngestOutcome {
                conv_key: pl.conv_key,
                idxs: pl.idxs,
                space_id: Some(space_id),
                cw,
                failure: None,
            }
        });

        use futures::StreamExt as _;
        let ingest_outcomes: Vec<IngestOutcome> = futures::stream::iter(ingest_futures)
            .buffer_unordered(ingest_concurrency)
            .collect()
            .await;

        // Merge in group order so a failure's questions land where the
        // benchmark put them, not where the scheduler finished them.
        let mut ingest_outcomes = ingest_outcomes;
        ingest_outcomes.sort_by(|a, b| a.conv_key.cmp(&b.conv_key));
        for o in ingest_outcomes {
            match (o.space_id, o.failure) {
                (Some(space_id), None) => {
                    if !is_read_only {
                        ingested_any = true;
                    }
                    prepared.push(Prepared {
                        conv_key: o.conv_key,
                        idxs: o.idxs,
                        space_id,
                        cw: o.cw,
                    });
                }
                (_, failure) => {
                    let reason = failure.unwrap_or_else(|| "conversation unavailable".to_owned());
                    for &idx in &o.idxs {
                        let r = failed_question_result(
                            &instances[idx],
                            reason.clone(),
                            /* ingest_failed = */ true,
                        );
                        record_failed(
                            &mut tracker,
                            &mut question_results,
                            &mut question_records,
                            r,
                        );
                    }
                }
            }
        }

        // ---- phase 1.5: one drain barrier for the whole run ----
        //
        // `brain_worker_pending_work` is a server-wide gauge, so one wait
        // settles every conversation's writes. The old code asked the same
        // global question once per conversation and paid the full wait each
        // time; on a benchmark whose instances carry no conversation id —
        // where every question becomes its own group — that was one barrier
        // PER QUESTION.
        if ingested_any {
            if let Some(addr) = drain_metrics_addr {
                last_drain = Some(wait_for_write_flush(addr, "all-conversations").await);
            }
        }

        // ---- phase 2: read back and judge, bounded-concurrently ----
        //
        // Conversations are independent by construction — each owns a
        // `SpaceId`, so no read can observe another's data — and every unit
        // of work here is network-bound: a RECALL round-trip and, under
        // `live-llm`, a judge call. Run sequentially, a benchmark cost the
        // SUM of those latencies; overlapped, it costs roughly their
        // maximum divided by the concurrency.
        //
        // `buffer_unordered` caps in-flight work at one number, which is
        // simultaneously the connection cap, the server-load cap, and the
        // LLM rate limit. Set `BRAIN_EVAL_CONCURRENCY=1` for strict
        // sequence.
        //
        // Results are returned OWNED and merged by the caller in instance
        // order, never appended from inside a task: a report whose question
        // order depended on which recall happened to finish first would
        // differ run to run, and two runs of the same benchmark have to be
        // diffable.
        struct GroupOutcome {
            conversation_write: Option<ConversationWrite>,
            /// `(instance index, record, result)`, in this group's order.
            questions: Vec<(usize, QuestionRecord, QuestionResult)>,
        }

        let concurrency = self.config.concurrency.max(1);
        // Only the predicate is needed inside the tasks; capturing the
        // manifest itself would move it out of the enclosing scope.
        let ingesting_run = read_only.is_none();
        let token_for = |space: [u8; 16]| tokens.get(&space).cloned();

        let group_futures = prepared.iter().map(|pre| {
            let conv_key = pre.conv_key.as_str();
            let idxs = pre.idxs.as_slice();
            let space_id = pre.space_id;
            let cw = pre.cw.clone();
            let cached = token_for(space_id);
            let instances = &instances;
            async move {
                let mut out = GroupOutcome {
                    conversation_write: None,
                    questions: Vec::with_capacity(idxs.len()),
                };

                // Phase 1 cached this space's token, so the reconnect is a
                // handshake with no admin round-trip. A missing token means
                // phase 1 never got this far, which is already recorded.
                let Some(token) = cached else {
                    for &idx in idxs {
                        let r = failed_question_result(
                            &instances[idx],
                            "read-back: no cached data-plane token".to_owned(),
                            /* ingest_failed = */ false,
                        );
                        out.questions.push((idx, failed_record(r.clone()), r));
                    }
                    return out;
                };

                let harness = match BrainEvalHarness::connect_with_token(
                    self.config.endpoint,
                    space_id,
                    token,
                )
                .await
                {
                    Ok(h) => h,
                    Err(e) => {
                        warn!(
                            conversation = %conv_key,
                            error = %e,
                            "read-back reconnect failed; recording every question in this group as failed",
                        );
                        for &idx in idxs {
                            let r = failed_question_result(
                                &instances[idx],
                                format!("read-back reconnect failed: {e}"),
                                false,
                            );
                            out.questions.push((idx, failed_record(r.clone()), r));
                        }
                        return out;
                    }
                };

                // Ledger the conversation's write (ingesting runs only — a
                // read-only reconnect performed no write worth recording).
                // The drain barrier settled before this phase, so the
                // per-memory extraction capture reads a fully-built graph.
                if ingesting_run {
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
                    out.conversation_write = Some(ConversationWrite {
                        conversation_id: conv_key.to_owned(),
                        space_id: uuid::Uuid::from_bytes(harness.space_id())
                            .hyphenated()
                            .to_string(),
                        write_latency_ms: cw.latency_ms,
                        attempted: cw.attempted,
                        stored: cw.stored,
                        deduplicated: cw.deduplicated,
                        stored_memory_ids: cw
                            .stored_ids
                            .iter()
                            .map(ToString::to_string)
                            .collect(),
                        memory_extractions,
                        ingest_failed: cw.ingest_failed,
                    });
                }

                // ---- MEMORY_LIST verification (opt-in) ----
                if self.config.verify_memory_list {
                    verify_memory_list(harness.client(), conv_key, cw.stored).await;
                }

                // ---- per-question retrieval + judging ----
                // `--verify-memory-list` is a list-only diagnostic: it
                // proves the tenant timeline is enumerable and skips the
                // (LLM-judged, paid) per-question pass entirely.
                if !self.config.verify_memory_list {
                    for &idx in idxs {
                        let run = self
                            .run_question(
                                &harness,
                                &instances[idx],
                                cw.latency_ms,
                                cw.attempted,
                                cw.stored,
                                cw.deduplicated,
                                cw.ingest_failed,
                                full_trace,
                            )
                            .await;
                        out.questions.push((
                            idx,
                            QuestionRecord {
                                result: run.result.clone(),
                                retrieved: run.retrieved,
                                fail_stage: run.fail_stage,
                                plan_trace: run.plan_trace,
                            },
                            run.result,
                        ));
                    }
                }

                if let Err(e) = harness.close().await {
                    warn!(error = %e, "harness close failed; continuing");
                }
                out
            }
        });

        let outcomes: Vec<GroupOutcome> = futures::stream::iter(group_futures)
            .buffer_unordered(concurrency)
            .collect()
            .await;

        // Merge deterministically: conversation order follows `prepared`
        // (itself `BTreeMap` order), and questions are restored to instance
        // order regardless of which group finished first.
        let mut merged: Vec<(usize, QuestionRecord, QuestionResult)> = Vec::new();
        for o in outcomes {
            if let Some(cwrite) = o.conversation_write {
                conversation_writes.push(cwrite);
            }
            merged.extend(o.questions);
        }
        conversation_writes.sort_by(|a, b| a.conversation_id.cmp(&b.conversation_id));
        restore_instance_order(&mut merged, |(idx, _, _)| *idx);
        for (_, record, result) in merged {
            tracker.record(&result);
            question_records.push(record);
            question_results.push(result);
        }

        // Persist the conversation→space manifest so a later
        // `BRAIN_EVAL_READ_ONLY_SPACES` run can re-query this populated
        // store without re-ingesting.
        if let (Some(p), Some(path)) = (persist.as_ref(), self.config.persist_spaces.as_ref()) {
            match p.save(path) {
                Ok(()) => println!(
                    "eval: wrote {} spaces to {} (reuse with BRAIN_EVAL_READ_ONLY_SPACES={})",
                    p.len(),
                    path.display(),
                    path.display()
                ),
                Err(e) => warn!(error = %e, "failed to write space manifest"),
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
                ReporterKind::Html => self.config.output_dir.join(format!("{stem}.html")),
            };
            let result: std::io::Result<()> = match kind {
                ReporterKind::Json => JsonReporter.write(report, &path),
                ReporterKind::Text => TextReporter.write(report, &path),
                ReporterKind::Html => HtmlReporter.write(report, &path),
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
#[derive(Clone)]
struct ConvWrite {
    latency_ms: u64,
    attempted: u64,
    stored: u64,
    deduplicated: u64,
    ingest_failed: bool,
    stored_ids: Vec<WireMemoryId>,
    stored_texts: Vec<String>,
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
        }
    }

    /// Build from an ingest aggregate.
    ///
    /// No drain field: the barrier used to run per conversation and its
    /// result was carried here, but it reads a SERVER-WIDE gauge, so the
    /// run now waits once after every ingest and records that single
    /// result directly on the report.
    fn from_ingest(ingest: IngestAggregate) -> Self {
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
        }
    }
}

/// Record an infrastructure-failed question into all three sinks: the
/// progress tracker, the summary results, and the run-report records (with
/// no retrieved memories and no attributable fail stage).
/// Put concurrently-produced results back into instance order.
///
/// `buffer_unordered` yields groups as they finish, so without this the
/// report's question order would depend on which recall happened to return
/// first — two runs of the same benchmark would produce different files and
/// stop being diffable. The instance index is captured before the work is
/// dispatched, so ordering is restored exactly, independent of scheduling.
///
/// Sorting is stable, so questions within one group keep their emitted
/// order even if two shared an index (they cannot, but the guarantee costs
/// nothing and makes the function total).
fn restore_instance_order<T>(merged: &mut [T], index_of: impl Fn(&T) -> usize) {
    merged.sort_by_key(index_of);
}

/// The record shape for a question that never produced a reading — the
/// same empty-trace form `record_failed` writes, hoisted so the concurrent
/// read-back phase can build one without touching shared accumulators.
fn failed_record(r: QuestionResult) -> QuestionRecord {
    QuestionRecord {
        result: r,
        retrieved: Vec::new(),
        fail_stage: None,
        plan_trace: None,
    }
}

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
        session_filter: None,
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

/// Enumerate the calling space's memories via MEMORY_LIST, paginating with
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

/// Get the cached data-plane token for `space_id`, minting (and caching) a
/// fresh one on first use. Caching keeps a reconnect of the same space on the
/// same key instead of minting a new one each time.
async fn mint_or_cached(
    minter: &KeyMinter,
    tokens: &mut BTreeMap<[u8; 16], String>,
    space_id: [u8; 16],
) -> Result<String, HarnessError> {
    if let Some(t) = tokens.get(&space_id) {
        return Ok(t.clone());
    }
    let token = minter.mint(space_id).await?;
    tokens.insert(space_id, token.clone());
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
/// Workers whose queue must be empty before a read sees a settled graph.
///
/// The write pipeline is: ENCODE acks, then the extractor builds the typed
/// graph, then statement embedding makes it matchable, then the derived-edge
/// workers link it. Anything else on the scheduler (decay, GC, snapshots) is
/// background housekeeping that never gates a read, so waiting on it would
/// only add dead time.
const DRAIN_WORKERS: [&str; 5] = [
    "extractor",
    "statement_embed",
    "auto_edge",
    "temporal_edge",
    "causal_edge",
];

/// Poll cadence. The barrier now reads a queue depth rather than guessing
/// from a counter plateau, so it can afford to look often and finish the
/// moment the queues are empty.
const DRAIN_INTERVAL: std::time::Duration = std::time::Duration::from_millis(250);

/// Consecutive all-empty observations required before declaring the pipeline
/// settled.
///
/// One zero is not enough: there is a window between ENCODE acking and the
/// extractor dequeuing where every queue is legitimately empty and the work
/// has not started. Four polls (1 s) spans that handoff while costing a
/// fraction of the old 60 s plateau.
const DRAIN_STABLE_POLLS: u32 = 4;

/// Floor before any verdict, so the workers have demonstrably picked up.
const DRAIN_MIN_WAIT: std::time::Duration = std::time::Duration::from_secs(1);

/// Hard cap. Reaching it is reported as `flushed: false` so a report can
/// never silently claim a settled read it did not get.
const DRAIN_MAX_WAIT: std::time::Duration = std::time::Duration::from_secs(600);

/// Total queued work across the write-path workers, or `None` when the
/// server publishes no `brain_worker_pending_work` gauge at all.
///
/// The `None` case is the one that matters for safety. An absent family and
/// an empty queue are both "no samples", and conflating them would make the
/// barrier declare a settling pipeline flushed — every read would then race
/// the extractor and the benchmark would silently measure a half-built
/// graph. So absence is reported as absence, and the caller falls back to
/// the slower counter plateau rather than guessing.
#[must_use]
pub(crate) fn pending_write_work(m: &PrometheusMetrics) -> Option<f64> {
    let mut total: Option<f64> = None;
    for w in DRAIN_WORKERS {
        if let Some(v) = m.sum_where("brain_worker_pending_work", w) {
            *total.get_or_insert(0.0) += v;
        }
    }
    total
}

/// Block until Brain's async write pipeline has settled.
///
/// Reads `brain_worker_pending_work` — the scheduler's own queue depth —
/// and returns once every write-path worker reports an empty queue for
/// [`DRAIN_STABLE_POLLS`] consecutive polls.
///
/// This used to infer settlement from `brain_extractor_items_written_total`
/// going unchanged for 60 s, polled every 5 s, with a 30 s floor. That is a
/// guess ("nothing has happened lately, so probably nothing will"), and it
/// is an expensive one: a 12-question smoke run spent 139 s of its 142 s
/// wall clock inside this function for ~3 s of actual work. The queue depth
/// answers the same question exactly, so the barrier now costs about a
/// second.
///
/// The plateau fallback is kept for a server that does not publish the
/// gauge: `None` from `sum_where` means the family is absent, not that the
/// queue is empty, and treating absence as empty would silently drop the
/// barrier and make every read race the extractor.
pub(crate) async fn wait_for_write_flush(
    metrics_addr: SocketAddr,
    conv: &str,
) -> DrainBarrierStats {
    let started = std::time::Instant::now();
    let mut stable: u32 = 0;
    let mut last_ex: u64 = 0;
    let mut last_em: u64 = 0;
    // Plateau-fallback state, used only when the gauge is unavailable.
    let mut plateau_last: i64 = -1;
    let mut plateau_stable: u32 = 0;
    let mut gauge_seen = false;

    while started.elapsed() < DRAIN_MAX_WAIT {
        // A scrape failure is transient: skip the poll without resetting
        // progress, exactly as the previous implementation did.
        let Some(m) = PrometheusMetrics::scrape(metrics_addr).await.ok() else {
            tokio::time::sleep(DRAIN_INTERVAL).await;
            continue;
        };

        let ex = m.sum("brain_extractor_items_written_total").unwrap_or(0.0) as u64;
        let em = m
            .sum("brain_statement_embed_rows_embedded_total")
            .unwrap_or(0.0) as u64;
        last_ex = ex;
        last_em = em;

        let pending = pending_write_work(&m);

        if let Some(pending_total) = pending {
            gauge_seen = true;
            if pending_total == 0.0 {
                stable += 1;
            } else {
                stable = 0;
            }
            if started.elapsed() >= DRAIN_MIN_WAIT && stable >= DRAIN_STABLE_POLLS {
                tracing::info!(
                    conversation = %conv,
                    extracted = ex,
                    statement_rows_embedded = em,
                    waited_ms = started.elapsed().as_millis() as u64,
                    "async write pipeline flushed (queue depth)"
                );
                return DrainBarrierStats {
                    extractor_items_written: ex,
                    statement_rows_embedded: em,
                    flushed: true,
                };
            }
        } else if !gauge_seen {
            // `pending` was `None`: the gauge is absent, not zero.
            // Legacy server: fall back to the counter plateau. Slower, but
            // correct, and it is the only signal such a server offers.
            let cur = ex as i64 + em as i64;
            if cur == plateau_last {
                plateau_stable += 1;
            } else {
                plateau_stable = 0;
                plateau_last = cur;
            }
            // 60 s of no movement, at this interval.
            let needed = (60.0 / DRAIN_INTERVAL.as_secs_f64()) as u32;
            if cur > 0 && plateau_stable >= needed {
                tracing::info!(
                    conversation = %conv,
                    extracted = ex,
                    statement_rows_embedded = em,
                    "async write pipeline flushed (counter plateau fallback)"
                );
                return DrainBarrierStats {
                    extractor_items_written: ex,
                    statement_rows_embedded: em,
                    flushed: true,
                };
            }
        }

        tokio::time::sleep(DRAIN_INTERVAL).await;
    }

    warn!(
        conversation = %conv,
        waited_s = started.elapsed().as_secs(),
        "write-flush wait hit max; proceeding"
    );
    DrainBarrierStats {
        extractor_items_written: last_ex,
        statement_rows_embedded: last_em,
        flushed: false,
    }
}

#[cfg(test)]
mod drain_barrier_tests {
    use super::{pending_write_work, DRAIN_WORKERS};
    use crate::client::metrics::PrometheusMetrics;

    fn metrics(body: &str) -> PrometheusMetrics {
        PrometheusMetrics::parse(body)
    }

    #[test]
    fn an_idle_pipeline_reports_zero_queued_work() {
        let text = DRAIN_WORKERS
            .iter()
            .map(|w| format!("brain_worker_pending_work{{shard=\"0\",worker=\"{w}\"}} 0\n"))
            .collect::<String>();
        assert_eq!(pending_write_work(&metrics(&text)), Some(0.0));
    }

    #[test]
    fn queued_work_on_any_write_worker_is_counted() {
        for w in DRAIN_WORKERS {
            let text = format!("brain_worker_pending_work{{shard=\"0\",worker=\"{w}\"}} 7\n");
            assert_eq!(
                pending_write_work(&metrics(&text)),
                Some(7.0),
                "{w} must gate the barrier"
            );
        }
    }

    #[test]
    fn an_absent_gauge_is_none_not_zero() {
        // The safety property. If absence collapsed to `Some(0.0)` the
        // barrier would declare a settling pipeline flushed and every read
        // in the benchmark would race the extractor — the measurement would
        // silently be of a half-built graph.
        let text = "brain_extractor_items_written_total{shard=\"0\"} 58\n";
        assert_eq!(pending_write_work(&metrics(text)), None);
    }

    #[test]
    fn housekeeping_workers_do_not_gate_the_barrier() {
        // decay / GC / snapshots run forever on their own cadence; waiting
        // for them would mean waiting for a queue that is never reliably
        // empty, which is how a barrier turns into a timeout.
        let text = "brain_worker_pending_work{shard=\"0\",worker=\"decay\"} 500\n\
                    brain_worker_pending_work{shard=\"0\",worker=\"snapshot\"} 12\n";
        assert_eq!(
            pending_write_work(&metrics(text)),
            None,
            "only write-path workers may gate a read"
        );
    }

    #[test]
    fn queued_work_sums_across_shards() {
        let text = "brain_worker_pending_work{shard=\"0\",worker=\"extractor\"} 3\n\
                    brain_worker_pending_work{shard=\"1\",worker=\"extractor\"} 4\n";
        assert_eq!(pending_write_work(&metrics(text)), Some(7.0));
    }
}

#[cfg(test)]
mod ordering_tests {
    use super::restore_instance_order;

    /// The determinism guarantee behind running conversations concurrently.
    ///
    /// `buffer_unordered` yields whichever group finishes first, so results
    /// arrive shuffled. If the report simply appended them, two runs of the
    /// same benchmark would order their questions differently and stop
    /// being diffable — the comparison a benchmark exists for.
    #[test]
    fn completion_order_does_not_survive_into_the_report() {
        // Groups finished 3rd, 1st, 2nd.
        let mut merged = vec![(7usize, "h"), (0, "a"), (3, "d")];
        restore_instance_order(&mut merged, |(i, _)| *i);
        assert_eq!(merged, vec![(0, "a"), (3, "d"), (7, "h")]);
    }

    #[test]
    fn any_completion_permutation_yields_the_same_order() {
        let canonical: Vec<(usize, u32)> = (0..8).map(|i| (i, i as u32 * 10)).collect();
        // Every rotation stands in for a different scheduling outcome.
        for rot in 0..canonical.len() {
            let mut shuffled = canonical.clone();
            shuffled.rotate_left(rot);
            restore_instance_order(&mut shuffled, |(i, _)| *i);
            assert_eq!(shuffled, canonical, "rotation {rot} must normalise");
        }
    }

    #[test]
    fn an_empty_run_is_not_a_special_case() {
        let mut merged: Vec<(usize, ())> = Vec::new();
        restore_instance_order(&mut merged, |(i, _)| *i);
        assert!(merged.is_empty());
    }

    #[test]
    fn ordering_is_by_instance_index_not_by_group() {
        // Two conversations interleave in instance order: a report grouped
        // by conversation would read 0,2,4,1,3,5 and silently reorder the
        // benchmark's own question sequence.
        let mut merged = vec![
            (1usize, "convB-q0"),
            (3, "convB-q1"),
            (5, "convB-q2"),
            (0, "convA-q0"),
            (2, "convA-q1"),
            (4, "convA-q2"),
        ];
        restore_instance_order(&mut merged, |(i, _)| *i);
        assert_eq!(
            merged.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5]
        );
    }
}
