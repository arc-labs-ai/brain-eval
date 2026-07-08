//! `brain-eval probe` — one write, one read, every stage.
//!
//! This is the diagnostic sibling of the dataset benchmarks. Instead of
//! grading hundreds of questions, it drives a **single controlled ENCODE
//! and a single controlled RECALL/QUERY** and prints, stage by stage,
//! *what we wanted* versus *what we got*. It exists to answer one
//! question before a full eval run: is the pipeline behaving correctly,
//! end to end, with the operator's real LLM key?
//!
//! ## What it observes, and from where
//!
//! The probe is a wire client, so it sees the wire-observable half of
//! each stage directly (Option A). The deep per-stage internals are
//! emitted by the server as a **stage journal** on the `brain_debug::stage`
//! tracing target, keyed by `memory_id`. Brain does NOT consult `RUST_LOG`
//! — the one log knob is the config `[monitoring.logging] level`, overridden
//! at runtime by the `BRAIN_LOG` env var. Boot the server with
//! `BRAIN_LOG=info,brain_debug=debug,brain_ops::write_trace=debug,brain_planner::executor=debug`
//! and the two views line up: the probe prints the wire "got"; the server
//! log carries the intermediate detail (embed dims, HyPE question count,
//! per-tier extraction proposals, resolver verdict+tier, rerank reordering).
//!
//! The probe prints the `memory_id` so you can `grep` the journal:
//! `docker logs <container> 2>&1 | grep brain_debug::stage | grep <memory_id>`.
//!
//! ## The controlled pair
//!
//! A single relational fact whose entities and relation are unambiguous,
//! so every extraction/resolution stage has a known-correct target — and
//! a question answerable only from that fact.

use std::net::SocketAddr;
use std::time::Duration;

use brain_db_sdk::wire::types::{
    AnswerKindWire, GraphEnrichment, QueryRequest, QueryTraceRequest, RetrieverSelectionWire,
};
use brain_db_sdk::{BrainClient, EncodeBuilder, RecallBuilder};

use crate::client::auth::KeyMinter;
use crate::client::harness::{BrainEvalHarness, HarnessError};

pub mod staged;

/// The controlled write. One relational fact: two entities (`Alice
/// Johnson`, `Acme Corporation`) joined by an employment relation, plus a
/// role attribute. Every extraction/resolution stage has a known target.
const WRITE_TEXT: &str = "Alice Johnson works at Acme Corporation as a staff software engineer.";

/// The subject we pin read-back and recall on.
const WRITE_SUBJECT: &str = "Alice Johnson";

/// The employer name — the token that proves both the write's relation
/// landed and the read surfaced it.
const EXPECT_EMPLOYER: &str = "acme";

/// The person name — proves the subject entity was extracted.
const EXPECT_SUBJECT_TOKEN: &str = "alice";

/// The controlled read. Answerable only from `WRITE_TEXT`.
const READ_QUESTION: &str = "Where does Alice Johnson work?";

/// How long to wait for the asynchronous extractor stage to settle before
/// giving up on read-back. Extraction is LLM-bound on a CPU-only dev box,
/// so this is generous.
const EXTRACTION_POLL_ATTEMPTS: u32 = 40;
const EXTRACTION_POLL_INTERVAL: Duration = Duration::from_secs(3);

/// Errors surfaced by the probe.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ProbeError {
    /// Connecting / minting a key / closing failed.
    #[error("harness: {0}")]
    Harness(#[from] HarnessError),
    /// An op over the wire failed.
    #[error("sdk: {0}")]
    Sdk(#[from] brain_db_sdk::BrainError),
    /// Loading the benchmark's instances failed (staged runs only).
    #[error("load: {0}")]
    Load(String),
}

/// Run the probe against a live server. Returns `true` iff both the write
/// and the read succeeded. Auth is minted from the environment exactly
/// like the dataset benchmarks (`BRAIN_EVAL_ADMIN_SECRET` / `_ADMIN_PORT`
/// / `_NAMESPACE`).
pub async fn run_probe(minter: &KeyMinter, endpoint: SocketAddr) -> Result<bool, ProbeError> {
    println!("brain-eval :: probe — one write, one read, every stage");
    println!("  endpoint : {endpoint}");
    println!("  journal  : boot the server with BRAIN_LOG=info,brain_debug=debug,\\");
    println!("             brain_ops::write_trace=debug,brain_planner::executor=debug");
    println!("             then grep 'brain_debug::stage' in the server log for per-stage detail.");
    println!();

    let harness =
        BrainEvalHarness::connect_minted(minter, endpoint, brain_db_sdk::new_id()).await?;
    let write_ok = probe_write(harness.client()).await?;
    let read_ok = probe_read(harness.client()).await?;
    harness.close().await?;

    println!();
    println!(
        "probe verdict :: WRITE {}   READ {}",
        verdict_word(write_ok),
        verdict_word(read_ok),
    );
    Ok(write_ok && read_ok)
}

// ===========================================================================
// WRITE
// ===========================================================================

async fn probe_write(client: &BrainClient) -> Result<bool, ProbeError> {
    println!("═══════════════════ WRITE ═══════════════════");
    println!("  text : \"{WRITE_TEXT}\"");
    println!();

    let resp = client
        .encode(&EncodeBuilder::new(WRITE_TEXT).build())
        .await?;
    let memory_id = resp.memory_id;
    println!("  memory_id = {memory_id}   (grep the stage journal by this id)");
    println!();

    let mut rows: Vec<StageRow> = Vec::new();

    // S1–S3b are confirmed by the synchronous encode ack: the write path
    // does not return until the WAL record is fsynced (WAL-before-ack) and
    // the memory row / arena slot / HNSW point are live.
    rows.push(StageRow::pass(
        "S1 embed",
        "text → 384-dim BGE vector",
        "vector produced (ack returned)",
    ));
    rows.push(StageRow::check(
        "S2 WAL",
        "record fsynced before ack",
        format!("lsn = {} (ack ⇒ durable)", resp.lsn),
        resp.lsn > 0,
    ));
    rows.push(StageRow::check(
        "S3 persist",
        "arena + redb + memory HNSW live, fresh write (not dedup)",
        format!(
            "memory_id = {memory_id}, was_deduplicated = {}",
            resp.was_deduplicated
        ),
        memory_id != 0 && !resp.was_deduplicated,
    ));
    rows.push(StageRow::check(
        "S3b enqueue",
        "extractor stage queued for async processing",
        format!("pending_stages = {:?}", resp.pending_stages),
        !resp.pending_stages.is_empty(),
    ));

    // S4–S9 are asynchronous. Poll the graph the write built until it
    // settles (or we give up). Read-back is name-based via recall's graph
    // enrichment — no entity-type-id needed.
    println!("  waiting for the extractor stage (S4–S9) to settle …");
    let graph = poll_graph(client, WRITE_SUBJECT).await?;
    println!();

    let entity_names: Vec<&str> = graph.entities.iter().map(|e| e.name.as_str()).collect();
    let has_subject = contains_token(&entity_names, EXPECT_SUBJECT_TOKEN);
    let has_employer = contains_token(&entity_names, EXPECT_EMPLOYER);

    rows.push(StageRow::check(
        "S4 HyPE",
        "≥1 hypothetical question generated (see journal stage=S4_hype)",
        "count is server-side only — see the stage journal for this memory_id".to_string(),
        // We can't see HyPE counts on the wire; mark informational.
        true,
    ));
    rows.push(StageRow::check(
        "S5–S7 extract",
        format!("entities include \"{WRITE_SUBJECT}\" and \"Acme Corporation\""),
        if entity_names.is_empty() {
            "no entities extracted".to_string()
        } else {
            format!("entities = {entity_names:?}")
        },
        has_subject && has_employer,
    ));
    rows.push(StageRow::check(
        "S8 resolve",
        "each surface resolved/created to an entity (tier in journal stage=S8_resolve)",
        format!(
            "{} entit{} resolved",
            graph.entities.len(),
            plural(graph.entities.len())
        ),
        !graph.entities.is_empty(),
    ));

    // S9 apply — the relation/statement that links subject → employer.
    let link = employment_link(&graph);
    rows.push(StageRow::check(
        "S9 apply",
        format!("a statement/relation links \"{WRITE_SUBJECT}\" → \"Acme Corporation\""),
        describe_graph_facts(&graph),
        link,
    ));

    render_rows(&rows);

    // Write success = the memory is durable AND the core fact (subject →
    // employer link) actually landed in the graph. HyPE/resolver detail is
    // observed in the journal but not gated here.
    let write_ok = resp.lsn > 0 && memory_id != 0 && has_subject && has_employer && link;
    Ok(write_ok)
}

/// Poll recall (with graph enrichment) until the async extractor produces
/// statements/relations, or we exhaust the budget. Returns the merged
/// enrichment across all returned memories.
async fn poll_graph(client: &BrainClient, subject: &str) -> Result<GraphEnrichment, ProbeError> {
    let mut merged = empty_graph();
    for attempt in 0..EXTRACTION_POLL_ATTEMPTS {
        let req = RecallBuilder::new(subject)
            .subject(subject)
            .include_graph(true)
            .include_text(true)
            .max_results(10)
            .build();
        let ans = client.recall(&req).await?;

        merged = empty_graph();
        for m in &ans.memories {
            if let Some(g) = &m.graph {
                merged.entities.extend(g.entities.iter().cloned());
                merged.statements.extend(g.statements.iter().cloned());
                merged.relations.extend(g.relations.iter().cloned());
            }
        }
        if !merged.statements.is_empty() || !merged.relations.is_empty() {
            println!("  extraction settled after ~{}s", attempt * 3);
            return Ok(merged);
        }
        tokio::time::sleep(EXTRACTION_POLL_INTERVAL).await;
    }
    println!(
        "  gave up after ~{}s — extraction produced no statements/relations",
        EXTRACTION_POLL_ATTEMPTS * 3
    );
    Ok(merged)
}

/// True iff some statement or relation joins the subject to the employer.
fn employment_link(graph: &GraphEnrichment) -> bool {
    let stmt_link = graph.statements.iter().any(|s| {
        token_in(&s.subject_name, EXPECT_SUBJECT_TOKEN)
            && token_in(&s.object_label, EXPECT_EMPLOYER)
    });
    let rel_link = graph.relations.iter().any(|r| {
        (token_in(&r.from_name, EXPECT_SUBJECT_TOKEN) && token_in(&r.to_name, EXPECT_EMPLOYER))
            || (token_in(&r.from_name, EXPECT_EMPLOYER)
                && token_in(&r.to_name, EXPECT_SUBJECT_TOKEN))
    });
    stmt_link || rel_link
}

/// Render the extracted statements + relations as a compact "got" string.
fn describe_graph_facts(graph: &GraphEnrichment) -> String {
    if graph.statements.is_empty() && graph.relations.is_empty() {
        return "no statements or relations".to_string();
    }
    let mut parts: Vec<String> = Vec::new();
    for s in &graph.statements {
        parts.push(format!(
            "stmt[{}]: {} --{}--> {} (conf {:.2})",
            s.subject_name, s.subject_name, s.predicate, s.object_label, s.confidence
        ));
    }
    for r in &graph.relations {
        parts.push(format!(
            "rel: {} --{}--> {}",
            r.from_name, r.predicate, r.to_name
        ));
    }
    parts.join("; ")
}

// ===========================================================================
// READ
// ===========================================================================

async fn probe_read(client: &BrainClient) -> Result<bool, ProbeError> {
    println!();
    println!("═══════════════════ READ ═══════════════════");
    println!("  question : \"{READ_QUESTION}\"");
    println!();

    let mut rows: Vec<StageRow> = Vec::new();

    // R6 rerank + R7 filter chain live in the server trace text blob (and,
    // for rerank ordering, the stage journal). Print the trace so the
    // operator sees the filter-chain survivor counts + per-retriever line.
    let trace = client
        .query_trace(&QueryTraceRequest {
            query: query_request(READ_QUESTION),
        })
        .await?;
    rows.push(StageRow::info(
        "R6 rerank",
        "cross-encoder reorders the head (if loaded)",
        "ordering is server-side — see journal stage=R6_rerank".to_string(),
    ));
    rows.push(StageRow::info(
        "R7 filter",
        "type/temporal/confidence/tombstone/supersession applied",
        "survivor counts in the trace below".to_string(),
    ));

    // R8: the router's shaped answer + whether it surfaced the employer.
    let rans = client
        .recall(
            &RecallBuilder::new(READ_QUESTION)
                .include_graph(true)
                .include_text(true)
                .max_results(10)
                .build(),
        )
        .await?;
    let found = read_surfaces(&rans, EXPECT_EMPLOYER);
    rows.push(StageRow::check(
        "R8 shape",
        format!("router surfaces the employer (\"{EXPECT_EMPLOYER}\")"),
        format!(
            "answer_kind = {}, {} memor{} returned, employer_present = {found}",
            answer_kind_name(rans.answer_kind),
            rans.memories.len(),
            plural(rans.memories.len()),
        ),
        found,
    ));

    render_rows(&rows);

    println!();
    println!("  ── server trace (QUERY_TRACE) ──");
    for line in trace.trace_text.lines() {
        println!("    {line}");
    }

    Ok(found)
}

/// Build a full `QueryRequest` — there is no builder, so every field is
/// set explicitly: Auto routing, default fusion, no filters.
fn query_request(text: &str) -> QueryRequest {
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
        limit: 10,
        retrievers: RetrieverSelectionWire::Auto,
        fusion_config: None,
        request_id: brain_db_sdk::new_id(),
    }
}

/// True iff any returned memory's text or graph enrichment contains the
/// token (case-insensitive).
fn read_surfaces(ans: &brain_db_sdk::RecallAnswer, token: &str) -> bool {
    ans.memories.iter().any(|m| {
        if token_in(&m.text, token) {
            return true;
        }
        m.graph.as_ref().is_some_and(|g| {
            g.entities.iter().any(|e| token_in(&e.name, token))
                || g.statements
                    .iter()
                    .any(|s| token_in(&s.object_label, token))
                || g.relations.iter().any(|r| token_in(&r.to_name, token))
        })
    })
}

// ===========================================================================
// Rendering + small helpers
// ===========================================================================

/// One row of a stage table: the stage, what we wanted, what we got, and a
/// verdict (`Some(true/false)` for a checked stage, `None` for one whose
/// detail lives server-side and is only informational here).
struct StageRow {
    stage: String,
    want: String,
    got: String,
    verdict: Option<bool>,
}

impl StageRow {
    fn check(stage: impl Into<String>, want: impl Into<String>, got: String, ok: bool) -> Self {
        Self {
            stage: stage.into(),
            want: want.into(),
            got,
            verdict: Some(ok),
        }
    }

    fn pass(stage: impl Into<String>, want: impl Into<String>, got: impl Into<String>) -> Self {
        Self {
            stage: stage.into(),
            want: want.into(),
            got: got.into(),
            verdict: Some(true),
        }
    }

    fn info(stage: impl Into<String>, want: impl Into<String>, got: String) -> Self {
        Self {
            stage: stage.into(),
            want: want.into(),
            got,
            verdict: None,
        }
    }
}

fn render_rows(rows: &[StageRow]) {
    for r in rows {
        let mark = match r.verdict {
            Some(true) => "✓",
            Some(false) => "✗",
            None => "·",
        };
        println!("  [{mark}] {:<14} want: {}", r.stage, r.want);
        println!("               got : {}", r.got);
    }
}

fn empty_graph() -> GraphEnrichment {
    GraphEnrichment {
        entities: Vec::new(),
        statements: Vec::new(),
        relations: Vec::new(),
    }
}

/// Case-insensitive substring test.
fn token_in(haystack: &str, token: &str) -> bool {
    haystack.to_ascii_lowercase().contains(token)
}

/// True iff any string in `names` contains `token` (case-insensitive).
fn contains_token(names: &[&str], token: &str) -> bool {
    names.iter().any(|n| token_in(n, token))
}

fn answer_kind_name(k: AnswerKindWire) -> &'static str {
    match k {
        AnswerKindWire::Single => "Single",
        AnswerKindWire::Many => "Many",
        AnswerKindWire::None => "None",
    }
}

fn verdict_word(ok: bool) -> &'static str {
    if ok {
        "PASS"
    } else {
        "FAIL"
    }
}

fn plural(n: usize) -> &'static str {
    if n == 1 {
        "y"
    } else {
        "ies"
    }
}
