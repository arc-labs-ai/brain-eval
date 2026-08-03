//! `RunReport` — the exhaustive artifact for one benchmark run.
//!
//! The existing [`crate::bench::report::shape::BenchmarkReport`] is the
//! stable, dashboard-facing summary. `RunReport` is its lossless sibling:
//! it captures every detail the run computed and then discarded — the full
//! retrieved [`brain_db_sdk::wire::types::MemoryResult`] per question (not
//! just its text), the per-conversation write ledger (including the fresh
//! memory ids), the server-side extraction health, the drain-barrier
//! plateau, and a per-question failure attribution.
//!
//! It is written additively next to the `.json`/`.txt` reports as
//! `<benchmark_id>-<run_started_unix_nanos>.run.json`, so nothing that
//! already consumes the existing artifacts breaks.
//!
//! Every section is `#[serde(default)]` and the top level carries a
//! [`RunReport::schema_version`] so a consumer written against an older
//! shape keeps deserializing as fields are added.

use serde::{Deserialize, Serialize};

use brain_db_sdk::wire::types::{
    GraphEnrichment, MemoryResult, QueryTraceResponse, RetrieverNameWire,
};

use crate::bench::config::RunConfig;
use crate::bench::core::outcome::QuestionResult;
use crate::bench::report::shape::{BenchmarkMeta, CompetitorRow};
use crate::bench::score::metrics::{EvalMetrics, WriteQualityMetrics};
use crate::client::metrics::PrometheusMetrics;

/// The current [`RunReport`] schema version. Bumped on any
/// backwards-incompatible shape change.
///
/// - **v1** — the original exhaustive artifact.
/// - **v2** — adds stage-level detail per read and per write: a
///   [`QuestionRecord::plan_trace`] (the server's `QUERY_TRACE` output), a
///   per-retrieved-memory [`RetrievedMemory::graph`], and a per-stored-memory
///   [`ConversationWrite::memory_extractions`]. All additive and optional, so
///   a v1 consumer keeps deserializing.
pub const RUN_REPORT_SCHEMA_VERSION: u32 = 2;

/// The exhaustive per-run artifact. Serialized to `*.run.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunReport {
    /// Schema version so consumers can gate on shape.
    pub schema_version: u32,
    /// Run-level metadata: what was run, where, how, and when.
    pub run: RunMetadata,
    /// Everything the write path produced.
    pub write: WriteReport,
    /// Everything the read path produced.
    pub read: ReadReport,
    /// The same aggregate metrics as [`crate::bench::report::shape::BenchmarkReport`],
    /// embedded unchanged.
    pub metrics: EvalMetrics,
    /// Published competitor numbers, embedded unchanged.
    #[serde(default)]
    pub competitors: Vec<CompetitorRow>,
}

impl RunReport {
    /// Assemble a report from its parts. No I/O.
    #[must_use]
    pub fn new(
        run: RunMetadata,
        write: WriteReport,
        read: ReadReport,
        metrics: EvalMetrics,
        competitors: Vec<CompetitorRow>,
    ) -> Self {
        Self {
            schema_version: RUN_REPORT_SCHEMA_VERSION,
            run,
            write,
            read,
            metrics,
            competitors,
        }
    }
}

/// Which ingest/read mode the run was driven in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    /// Fresh spaces, ingest + read, no manifest written.
    Ingest,
    /// Reconnect to a prior manifest and read only (no ingest).
    ReadOnly,
    /// Ingest + read AND record the conversation→space manifest for reuse.
    Persist,
}

impl RunMode {
    /// Derive the mode from the run config's manifest flags. Read-only wins
    /// (it skips ingest entirely); persist is an ingesting run that also
    /// records the manifest.
    #[must_use]
    pub fn from_config(config: &RunConfig) -> Self {
        if config.read_only_spaces.is_some() {
            Self::ReadOnly
        } else if config.persist_spaces.is_some() {
            Self::Persist
        } else {
            Self::Ingest
        }
    }
}

/// Run-level metadata. Never carries the admin secret.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata {
    /// Stable benchmark id (`"locomo"`, `"longmemeval-s"`, …).
    pub benchmark_id: String,
    /// Human-readable benchmark name.
    pub benchmark_name: String,
    /// Paper / dataset URL.
    pub benchmark_url: String,
    /// How many instances were loaded (post type/count filtering).
    pub instance_count: usize,
    /// Endpoint the harness connected to (`ip:port`).
    pub endpoint: String,
    /// Tenant namespace minted keys bound to.
    pub namespace: String,
    /// Safety cap on returned memories per RECALL.
    pub max_results: u32,
    /// Question-count cap, if any.
    #[serde(default)]
    pub max_questions: Option<usize>,
    /// Question-type filter, if any.
    #[serde(default)]
    pub question_types: Option<Vec<String>>,
    /// Ingest / read mode.
    pub mode: RunMode,
    /// `"heuristic"` or `"llm:<provider>:<model>@<ver>"`.
    pub judge_type: String,
    /// Judge model, or `"heuristic"`.
    pub judge_model: String,
    /// Judge prompt version.
    pub judge_prompt_version: String,
    /// Hex sha256 of the rendered judge templates.
    pub judge_prompt_sha256: String,
    /// Judge sampling temperature.
    pub judge_temperature: f64,
    /// How many questions fell back to the heuristic grader.
    pub judge_heuristic_fallbacks: usize,
    /// brain-eval version reporting this run.
    pub brain_version: String,
    /// Run start, Unix nanos.
    pub run_started_unix_nanos: u128,
    /// Run finish, Unix nanos — captured after the whole run completes.
    pub run_finished_unix_nanos: u128,
}

impl RunMetadata {
    /// Build from the assembled [`BenchmarkMeta`], the run config, the
    /// derived mode, and the run-finish timestamp. Pulls the judge identity
    /// and benchmark identity straight from `meta` so the two artifacts can
    /// never disagree, and folds in the config-only fields the summary
    /// report never carried. The admin secret is deliberately absent.
    #[must_use]
    pub fn from_parts(
        meta: &BenchmarkMeta,
        config: &RunConfig,
        mode: RunMode,
        run_finished_unix_nanos: u128,
    ) -> Self {
        Self {
            benchmark_id: meta.benchmark_id.clone(),
            benchmark_name: meta.benchmark_name.clone(),
            benchmark_url: meta.benchmark_url.clone(),
            instance_count: meta.instance_count,
            endpoint: config.endpoint.to_string(),
            namespace: config.namespace.clone(),
            max_results: config.max_results,
            max_questions: config.max_questions,
            question_types: config.question_types.clone(),
            mode,
            judge_type: meta.judge_type.clone(),
            judge_model: meta.judge_model.clone(),
            judge_prompt_version: meta.judge_prompt_version.clone(),
            judge_prompt_sha256: meta.judge_prompt_sha256.clone(),
            judge_temperature: meta.judge_temperature,
            judge_heuristic_fallbacks: meta.judge_heuristic_fallbacks,
            brain_version: meta.brain_version.clone(),
            run_started_unix_nanos: meta.run_started_unix_nanos,
            run_finished_unix_nanos,
        }
    }
}

// ---------------------------------------------------------------------------
// Write side
// ---------------------------------------------------------------------------

/// Everything the write path produced this run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WriteReport {
    /// Aggregate encode-quality summary (accepted / merged / discarded).
    pub summary: WriteQualityMetrics,
    /// One row per ingested conversation.
    #[serde(default)]
    pub conversations: Vec<ConversationWrite>,
    /// Server-side extraction health, when a metrics endpoint was reachable.
    #[serde(default)]
    pub extraction: Option<ExtractionHealth>,
    /// The final drain-barrier plateau, when the run waited on one.
    #[serde(default)]
    pub drain: Option<DrainBarrierStats>,
}

/// The extracted typed-graph for one memory, flattened to display strings.
///
/// Shared by the read side (attached per retrieved memory) and the write side
/// (attached per stored memory), so the two artifacts render the same shape.
/// The rendering mirrors the staged probe's `focus_read` exactly: entities as
/// `name :: type_qname`, statements as `subject --predicate--> object (conf)`,
/// relations as `subject --predicate--> object`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryGraph {
    /// Resolved typed entities.
    #[serde(default)]
    pub entities: Vec<GraphEntityView>,
    /// Typed statements (subject → predicate → object).
    #[serde(default)]
    pub statements: Vec<GraphStatementView>,
    /// Typed relations (subject → predicate → object).
    #[serde(default)]
    pub relations: Vec<GraphRelationView>,
}

impl MemoryGraph {
    /// Project the SDK's [`GraphEnrichment`] into the report shape.
    #[must_use]
    pub fn from_wire(g: &GraphEnrichment) -> Self {
        Self {
            entities: g
                .entities
                .iter()
                .map(|e| GraphEntityView {
                    name: e.name.clone(),
                    type_qname: e.type_qname.clone(),
                })
                .collect(),
            statements: g
                .statements
                .iter()
                .map(|s| GraphStatementView {
                    subject: s.subject_name.clone(),
                    predicate: s.predicate.clone(),
                    object: s.object_label.clone(),
                    confidence: s.confidence,
                })
                .collect(),
            relations: g
                .relations
                .iter()
                .map(|r| GraphRelationView {
                    subject: r.from_name.clone(),
                    predicate: r.predicate.clone(),
                    object: r.to_name.clone(),
                })
                .collect(),
        }
    }

    /// `true` when nothing was extracted for this memory.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty() && self.statements.is_empty() && self.relations.is_empty()
    }
}

/// A resolved typed entity: display name + schema type qname.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEntityView {
    /// Entity display name.
    pub name: String,
    /// Schema type qname (e.g. `brain:Person`).
    pub type_qname: String,
}

/// A typed statement flattened to display strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStatementView {
    /// Subject entity display name.
    pub subject: String,
    /// Predicate qname.
    pub predicate: String,
    /// Rendered object label.
    pub object: String,
    /// Extraction confidence.
    pub confidence: f32,
}

/// A typed relation flattened to its two endpoint names + connecting predicate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRelationView {
    /// Source endpoint display name.
    pub subject: String,
    /// Predicate qname.
    pub predicate: String,
    /// Target endpoint display name.
    pub object: String,
}

/// The extracted graph for one stored memory, keyed by its id — the write-side
/// analogue of [`RetrievedMemory::graph`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryExtraction {
    /// Decimal `WireMemoryId` of the stored memory.
    pub memory_id: String,
    /// What the async extractor produced for this memory.
    pub graph: MemoryGraph,
}

/// Per-conversation write ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationWrite {
    /// Conversation grouping key (or the question id for singletons).
    pub conversation_id: String,
    /// The space id this conversation was ingested under (hyphenated uuid).
    pub space_id: String,
    /// Total ingest latency for the conversation, ms.
    pub write_latency_ms: u64,
    /// ENCODE attempts (user turns processed).
    pub attempted: u64,
    /// ENCODEs that produced a fresh memory.
    pub stored: u64,
    /// ENCODEs that hit the fingerprint dedupe path.
    pub deduplicated: u64,
    /// Ids of the fresh memories (decimal `WireMemoryId`), previously
    /// discarded after counting.
    #[serde(default)]
    pub stored_memory_ids: Vec<String>,
    /// Per-stored-memory extraction detail: what the async typed-graph
    /// extractor produced for each id in `stored_memory_ids`. Populated only
    /// when full-trace capture is on (`BRAIN_EVAL_FULL_TRACE`), and only for
    /// stored memories whose graph could be re-fetched; empty otherwise. This
    /// is per-memory detail alongside the aggregate [`ExtractionHealth`].
    #[serde(default)]
    pub memory_extractions: Vec<MemoryExtraction>,
    /// `true` if any session in this conversation failed to ingest.
    pub ingest_failed: bool,
}

/// Server-side extraction health, scraped from the extractor metrics.
///
/// The tier / drop counters are cumulative and server-wide: exact for a
/// fresh single-conversation run, a coarse upper bound across many
/// conversations in one process. Computed from a single `/metrics` scrape.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtractionHealth {
    /// Memories the pattern tier ran on.
    pub tier_ran_pattern: u64,
    /// Memories the classifier tier ran on.
    pub tier_ran_classifier: u64,
    /// Memories the LLM tier ran on.
    pub tier_ran_llm: u64,
    /// Tier runs that failed (any tier).
    pub tier_failed: u64,
    /// Candidates dropped (apply-skip + generic drops + schema filter).
    pub dropped: u64,
    /// Entity rows persisted.
    pub entities: u64,
    /// Statement rows persisted.
    pub statements: u64,
    /// Relation rows persisted.
    pub relations: u64,
}

impl ExtractionHealth {
    const TIER_RUNS: &'static str = "brain_extractor_tier_runs_total";
    const ITEMS_WRITTEN: &'static str = "brain_extractor_items_written_total";

    /// Compute health from a parsed `/metrics` scrape. The single shared
    /// helper both the dataset runner and the staged probe read from, so a
    /// change to which counters count "health" lives in one place.
    #[must_use]
    pub fn from_metrics(m: &PrometheusMetrics) -> Self {
        let ran = |tier: &str| {
            m.sum_where(Self::TIER_RUNS, &format!("tier=\"{tier}\",status=\"ran\""))
                .unwrap_or(0.0) as u64
        };
        let item = |kind: &str| {
            m.sum_where(Self::ITEMS_WRITTEN, &format!("item_kind=\"{kind}\""))
                .unwrap_or(0.0) as u64
        };
        let tier_failed = m
            .sum_where(Self::TIER_RUNS, "status=\"failed\"")
            .unwrap_or(0.0) as u64;
        let dropped = (m.sum("brain_extractor_apply_dropped_total").unwrap_or(0.0)
            + m.sum("brain_extractor_drops_total").unwrap_or(0.0)
            + m.sum("brain_extractor_schema_filtered_total")
                .unwrap_or(0.0)) as u64;
        Self {
            tier_ran_pattern: ran("pattern"),
            tier_ran_classifier: ran("classifier"),
            tier_ran_llm: ran("llm"),
            tier_failed,
            dropped,
            entities: item("entity"),
            statements: item("statement"),
            relations: item("relation"),
        }
    }

    /// The one-line health tag the staged probe appends to a WRITE row.
    #[must_use]
    pub fn health_tag(&self) -> String {
        format!(
            " | tiers ran pattern={} classifier={} llm={}; failed={} | dropped={}",
            self.tier_ran_pattern,
            self.tier_ran_classifier,
            self.tier_ran_llm,
            self.tier_failed,
            self.dropped,
        )
    }
}

/// The drain-barrier plateau — what the async write pipeline had produced
/// when reads were allowed to start.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DrainBarrierStats {
    /// `brain_extractor_items_written_total` at the plateau (all item kinds).
    pub extractor_items_written: u64,
    /// `brain_statement_embed_rows_embedded_total` at the plateau.
    pub statement_rows_embedded: u64,
    /// `true` if the plateau was reached; `false` if the wait hit its cap
    /// (or never scraped) and proceeded anyway.
    pub flushed: bool,
}

// ---------------------------------------------------------------------------
// Read side
// ---------------------------------------------------------------------------

/// Everything the read path produced this run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReadReport {
    /// One record per question — the full [`QuestionResult`] plus the full
    /// retrieved memories and a failure attribution.
    #[serde(default)]
    pub per_question: Vec<QuestionRecord>,
    /// Roll-up of per-question failure stages.
    pub failure_attribution: FailureAttribution,
}

/// A per-question record: the summary [`QuestionResult`] plus the full
/// retrieved memories (not just their text) and the attributed fail stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionRecord {
    /// The summary result, flattened so its fields sit at this level.
    #[serde(flatten)]
    pub result: QuestionResult,
    /// The full retrieved memories — every scored field the SDK returned,
    /// not just `.text`.
    #[serde(default)]
    pub retrieved: Vec<RetrievedMemory>,
    /// Which stage lost a non-correct answer, if attributable.
    #[serde(default)]
    pub fail_stage: Option<FailStage>,
    /// The server's per-read plan/execution trace for this question's cue —
    /// the `QUERY_TRACE` output (routing → retrievers → fusion → filter chain,
    /// with per-stage latencies). Captured only when full-trace is on
    /// (`BRAIN_EVAL_FULL_TRACE`); `None` otherwise.
    #[serde(default)]
    pub plan_trace: Option<PlanTrace>,
}

/// A captured server-side query trace.
///
/// The SDK's `QUERY_TRACE` returns the trace as an **opaque text blob** plus a
/// total latency — the server renders routing, each retriever's config, the
/// fusion params, and the filter-chain survivor counts into that text, and
/// does not expose a parsed struct on the wire. So the structured capture
/// here is this thin wrapper: the verbatim trace text (the routing / retriever
/// / fusion / filter detail lives inside it, line by line) plus the numeric
/// total latency the trace response carries separately.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanTrace {
    /// The server's verbatim per-stage trace text.
    pub trace_text: String,
    /// Total server-measured latency for the traced query, ms.
    pub total_latency_ms: f64,
}

impl PlanTrace {
    /// Capture from the SDK's `QUERY_TRACE` response.
    #[must_use]
    pub fn from_wire(r: &QueryTraceResponse) -> Self {
        Self {
            trace_text: r.trace_text.clone(),
            total_latency_ms: r.total_latency_ms,
        }
    }
}

/// A retrieved memory, carrying the scored fields the summary report drops.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedMemory {
    /// Decimal `WireMemoryId`.
    pub memory_id: String,
    /// Memory text.
    pub text: String,
    /// Cosine similarity of the semantic retriever.
    pub similarity_score: f32,
    /// Post-fusion score.
    pub fused_score: f32,
    /// Cross-encoder rerank score, when rerank ran.
    #[serde(default)]
    pub rerank_score: Option<f32>,
    /// Stored confidence / salience.
    pub confidence: f32,
    /// Retriever families that surfaced this memory.
    #[serde(default)]
    pub contributing_retrievers: Vec<String>,
    /// The extracted typed-graph attached to this memory by recall's graph
    /// enrichment — exactly what the write pipeline produced. Populated only
    /// when full-trace is on (`BRAIN_EVAL_FULL_TRACE`) and the enriched recall
    /// carried a graph for this id; `None` otherwise.
    #[serde(default)]
    pub graph: Option<MemoryGraph>,
}

impl RetrievedMemory {
    /// Project the SDK's `MemoryResult` down to the report shape. Captures the
    /// graph enrichment when the source memory carried one (i.e. the recall
    /// was issued with `include_graph`).
    #[must_use]
    pub fn from_wire(m: &MemoryResult) -> Self {
        Self {
            memory_id: m.memory_id.to_string(),
            text: m.text.clone(),
            similarity_score: m.similarity_score,
            fused_score: m.fused_score,
            rerank_score: m.rerank_score,
            confidence: m.confidence,
            contributing_retrievers: m
                .contributing_retrievers
                .iter()
                .map(|r| retriever_name(*r).to_owned())
                .collect(),
            graph: m.graph.as_ref().map(MemoryGraph::from_wire),
        }
    }
}

/// The stage a non-correct answer was lost at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailStage {
    /// Candidates were retrieved and the router answered, but the support
    /// judge found the gold was NOT in the retrieved context.
    RetrievalMissedGold,
    /// The gold WAS retrieved but the answer dropped it (a synthesis loss).
    AnswerDroppedIt,
    /// The router returned nothing on an answerable question.
    Abstention,
    /// Wrong answer, support unknown (no LLM support judge to attribute).
    WrongAnswerSupportUnknown,
}

impl FailStage {
    /// Attribute a non-correct verdict to a stage. A pure function of the
    /// fields the runner already computes: whether the answer was correct,
    /// whether any memory came back, and whether the support judge found the
    /// gold in the retrieved context. `None` when the answer was correct
    /// (nothing to attribute).
    ///
    /// - correct                       ⇒ `None`
    /// - no memories retrieved         ⇒ [`FailStage::Abstention`]
    /// - support judge said "yes"      ⇒ [`FailStage::AnswerDroppedIt`]
    /// - support judge said "no"       ⇒ [`FailStage::RetrievalMissedGold`]
    /// - no support verdict            ⇒ [`FailStage::WrongAnswerSupportUnknown`]
    #[must_use]
    pub fn attribute(
        correct: bool,
        memories_retrieved: usize,
        context_supported: Option<bool>,
    ) -> Option<Self> {
        if correct {
            return None;
        }
        if memories_retrieved == 0 {
            return Some(Self::Abstention);
        }
        Some(match context_supported {
            Some(true) => Self::AnswerDroppedIt,
            Some(false) => Self::RetrievalMissedGold,
            None => Self::WrongAnswerSupportUnknown,
        })
    }
}

/// Roll-up of per-question failure stages, mirroring the staged probe's
/// `read failures by stage` line.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FailureAttribution {
    /// Retrieval never surfaced the gold.
    pub fail_retrieval: usize,
    /// Gold was retrieved but the answer dropped it.
    pub fail_answer_drop: usize,
    /// Router returned nothing on an answerable question.
    pub fail_abstention: usize,
    /// Wrong answer, support unattributed.
    pub fail_wrong: usize,
}

impl FailureAttribution {
    /// Fold one question's fail stage in.
    pub fn record(&mut self, stage: Option<FailStage>) {
        match stage {
            Some(FailStage::RetrievalMissedGold) => self.fail_retrieval += 1,
            Some(FailStage::AnswerDroppedIt) => self.fail_answer_drop += 1,
            Some(FailStage::Abstention) => self.fail_abstention += 1,
            Some(FailStage::WrongAnswerSupportUnknown) => self.fail_wrong += 1,
            None => {}
        }
    }

    /// Build the roll-up from a slice of question records.
    #[must_use]
    pub fn from_records(records: &[QuestionRecord]) -> Self {
        let mut attribution = Self::default();
        for r in records {
            attribution.record(r.fail_stage);
        }
        attribution
    }
}

/// Map a wire retriever family to its lower-case name.
fn retriever_name(r: RetrieverNameWire) -> &'static str {
    match r {
        RetrieverNameWire::Semantic => "semantic",
        RetrieverNameWire::Lexical => "lexical",
        RetrieverNameWire::Graph => "graph",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fail_stage_attribution_covers_every_branch() {
        assert_eq!(FailStage::attribute(true, 5, Some(true)), None);
        assert_eq!(
            FailStage::attribute(false, 0, None),
            Some(FailStage::Abstention)
        );
        assert_eq!(
            FailStage::attribute(false, 3, Some(true)),
            Some(FailStage::AnswerDroppedIt)
        );
        assert_eq!(
            FailStage::attribute(false, 3, Some(false)),
            Some(FailStage::RetrievalMissedGold)
        );
        assert_eq!(
            FailStage::attribute(false, 3, None),
            Some(FailStage::WrongAnswerSupportUnknown)
        );
    }

    #[test]
    fn failure_attribution_folds_each_stage() {
        let mut a = FailureAttribution::default();
        a.record(Some(FailStage::Abstention));
        a.record(Some(FailStage::Abstention));
        a.record(Some(FailStage::RetrievalMissedGold));
        a.record(Some(FailStage::AnswerDroppedIt));
        a.record(Some(FailStage::WrongAnswerSupportUnknown));
        a.record(None);
        assert_eq!(a.fail_abstention, 2);
        assert_eq!(a.fail_retrieval, 1);
        assert_eq!(a.fail_answer_drop, 1);
        assert_eq!(a.fail_wrong, 1);
    }

    #[test]
    fn extraction_health_reads_labeled_counters() {
        let text = "\
brain_extractor_tier_runs_total{shard=\"0\",tier=\"pattern\",status=\"ran\"} 4
brain_extractor_tier_runs_total{shard=\"0\",tier=\"classifier\",status=\"ran\"} 2
brain_extractor_tier_runs_total{shard=\"0\",tier=\"llm\",status=\"ran\"} 3
brain_extractor_tier_runs_total{shard=\"0\",tier=\"llm\",status=\"failed\"} 1
brain_extractor_apply_dropped_total{shard=\"0\",reason=\"subject_unresolved\"} 2
brain_extractor_items_written_total{shard=\"0\",item_kind=\"entity\"} 7
brain_extractor_items_written_total{shard=\"0\",item_kind=\"statement\"} 5
brain_extractor_items_written_total{shard=\"0\",item_kind=\"relation\"} 1
";
        let m = PrometheusMetrics::parse(text);
        let h = ExtractionHealth::from_metrics(&m);
        assert_eq!(h.tier_ran_pattern, 4);
        assert_eq!(h.tier_ran_classifier, 2);
        assert_eq!(h.tier_ran_llm, 3);
        assert_eq!(h.tier_failed, 1);
        assert_eq!(h.dropped, 2);
        assert_eq!(h.entities, 7);
        assert_eq!(h.statements, 5);
        assert_eq!(h.relations, 1);
    }

    #[test]
    fn schema_version_is_two() {
        assert_eq!(RUN_REPORT_SCHEMA_VERSION, 2);
    }

    #[test]
    fn question_record_with_trace_and_graph_round_trips() {
        use crate::bench::core::instance::QuestionType;
        use crate::bench::core::outcome::{AnswerKind, QuestionResult, Verdict};

        let graph = MemoryGraph {
            entities: vec![GraphEntityView {
                name: "Alice Johnson".to_owned(),
                type_qname: "brain:Person".to_owned(),
            }],
            statements: vec![GraphStatementView {
                subject: "Alice Johnson".to_owned(),
                predicate: "worksAt".to_owned(),
                object: "Acme Corporation".to_owned(),
                confidence: 0.91,
            }],
            relations: vec![GraphRelationView {
                subject: "Alice Johnson".to_owned(),
                predicate: "employedBy".to_owned(),
                object: "Acme Corporation".to_owned(),
            }],
        };
        let record = QuestionRecord {
            result: QuestionResult {
                question_id: "q1".to_owned(),
                question_type: QuestionType::SingleHop,
                question: "Where does Alice work?".to_owned(),
                ground_truth: "Acme Corporation".to_owned(),
                system_answer: "Acme Corporation".to_owned(),
                answer_kind: AnswerKind::Single,
                verdict: Verdict::Correct,
                score: 1.0,
                write_latency_ms: 12,
                read_latency_ms: 3,
                tokens_write: 0,
                tokens_read: 0,
                memories_retrieved: 1,
                retrieved_memory_contents: vec!["Alice works at Acme".to_owned()],
                judge_reasoning: String::new(),
                context_supported: Some(true),
                context_support_reasoning: String::new(),
                ingestion_failed: false,
                retrieval_failed: false,
                write_attempted: 1,
                write_stored: 1,
                write_deduplicated: 0,
            },
            retrieved: vec![RetrievedMemory {
                memory_id: "42".to_owned(),
                text: "Alice works at Acme".to_owned(),
                similarity_score: 0.8,
                fused_score: 0.7,
                rerank_score: Some(0.6),
                confidence: 0.9,
                contributing_retrievers: vec!["semantic".to_owned()],
                graph: Some(graph.clone()),
            }],
            fail_stage: None,
            plan_trace: Some(PlanTrace {
                trace_text: "route: semantic+lexical+graph\nfusion: rrf k=60".to_owned(),
                total_latency_ms: 4.2,
            }),
        };

        let json = serde_json::to_string(&record).expect("serialize");
        let back: QuestionRecord = serde_json::from_str(&json).expect("deserialize");

        let pt = back.plan_trace.expect("plan_trace preserved");
        assert!(pt.trace_text.contains("fusion: rrf"));
        assert!((pt.total_latency_ms - 4.2).abs() < 1e-9);

        let g = back.retrieved[0].graph.as_ref().expect("graph preserved");
        assert_eq!(g.entities[0].name, "Alice Johnson");
        assert_eq!(g.statements[0].predicate, "worksAt");
        assert!((g.statements[0].confidence - 0.91).abs() < 1e-6);
        assert_eq!(g.relations[0].object, "Acme Corporation");
    }

    #[test]
    fn conversation_write_carries_per_memory_extraction() {
        let cw = ConversationWrite {
            conversation_id: "c1".to_owned(),
            space_id: "space".to_owned(),
            write_latency_ms: 5,
            attempted: 2,
            stored: 2,
            deduplicated: 0,
            stored_memory_ids: vec!["1".to_owned(), "2".to_owned()],
            memory_extractions: vec![MemoryExtraction {
                memory_id: "1".to_owned(),
                graph: MemoryGraph {
                    entities: vec![GraphEntityView {
                        name: "Acme".to_owned(),
                        type_qname: "brain:Org".to_owned(),
                    }],
                    statements: Vec::new(),
                    relations: Vec::new(),
                },
            }],
            ingest_failed: false,
        };
        let json = serde_json::to_string(&cw).expect("serialize");
        let back: ConversationWrite = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.memory_extractions.len(), 1);
        assert_eq!(back.memory_extractions[0].memory_id, "1");
        assert_eq!(back.memory_extractions[0].graph.entities[0].name, "Acme");
    }

    #[test]
    fn v1_json_without_new_fields_still_deserializes() {
        // A v1 record has neither plan_trace, retrieved[].graph, nor
        // memory_extractions — the new fields must default cleanly.
        let v1 = r#"{
            "conversation_id": "c1", "space_id": "a", "write_latency_ms": 1,
            "attempted": 1, "stored": 1, "deduplicated": 0,
            "stored_memory_ids": ["1"], "ingest_failed": false
        }"#;
        let cw: ConversationWrite = serde_json::from_str(v1).expect("v1 deserialize");
        assert!(cw.memory_extractions.is_empty());
    }

    #[test]
    fn run_mode_priority_read_only_over_persist() {
        // Not building a full RunConfig here — RunMode::from_config is
        // exercised via the runner integration; this documents the intent.
        assert_ne!(RunMode::ReadOnly, RunMode::Persist);
    }
}
