//! Staged multi-item run — the dataset-scale sibling of the single-pair
//! [`super::run_probe`].
//!
//! Where the single probe drives one controlled write + read with a known
//! answer, this drives a whole benchmark and, for **each** write and
//! **each** read, records a per-stage "want vs got" so a failure is
//! attributable to a specific stage rather than lost in an aggregate
//! accuracy number.
//!
//! - **Write** (internal benchmark): a real dataset has no per-turn
//!   extraction gold, so write health is measured two ways — *durability*
//!   per turn (the encode ack proves WAL + persist) and *extraction yield*
//!   per conversation (how many entities / statements / relations the
//!   async pipeline produced from the ingested turns).
//! - **Read** (publishable report): graded against the dataset's gold
//!   answer via the heuristic judge, with the failing stage attributed —
//!   a miss at R2–R4 (retrieval), an abstention at R8 (shaping), or an
//!   answer that surfaced the wrong memory.
//!
//! Deep per-stage internals (S4 HyPE counts, S8 resolver tier, R6 rerank)
//! remain in the server's `brain_debug::stage` journal; this runner is a
//! pure wire client and reports the wire-observable stages.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use std::collections::HashSet;

use brain_db_sdk::wire::types::QueryTraceRequest;
use brain_db_sdk::{EncodeBuilder, RecallBuilder};

use crate::bench::core::benchmark::Benchmark;
use crate::bench::core::instance::EvalInstance;
use crate::bench::core::outcome::Verdict;
use crate::bench::synthesize::synthesize_answer;
use crate::client::auth::KeyMinter;
use crate::client::harness::BrainEvalHarness;

use super::{answer_kind_name, query_request, ProbeError};

/// Number of top memories `synthesize_answer` folds into the answer.
const EPISODIC_CAP: usize = 10;

/// How long to wait for a conversation's async extraction to settle
/// before reading back its graph. Generous: extraction is LLM-bound.
const DRAIN_ATTEMPTS: u32 = 40;
const DRAIN_INTERVAL: Duration = Duration::from_secs(3);

/// Run the staged analysis over `benchmark`, capped at `max_questions`
/// (`None` = all). Returns `true` iff every read passed — but the value
/// of this run is the per-stage report, not the pass/fail bit.
pub async fn run_staged(
    minter: &KeyMinter,
    endpoint: std::net::SocketAddr,
    metrics_addr: Option<std::net::SocketAddr>,
    reuse_agent: Option<[u8; 16]>,
    benchmark: &dyn Benchmark,
    max_questions: Option<usize>,
) -> Result<bool, ProbeError> {
    println!(
        "brain-eval :: staged probe — per-stage want-vs-got over `{}`",
        benchmark.id()
    );
    println!("  endpoint : {endpoint}");
    // The judge grades read answers: the LLM judge when a `live-llm` build
    // has a provider key, else the heuristic substring grader. LoCoMo's
    // temporal/derived golds need the LLM judge to be scored fairly.
    let judge = crate::bench::score::judge::Judge::from_env();
    println!("  judge    : {}", judge.describe());
    if reuse_agent.is_some() {
        println!("  mode     : reads-only (reusing a prior ingest; no write)");
    }
    println!();

    // Compiled-in benchmarks (smoke) ignore the datasets dir; file-backed
    // ones would need it, but the staged runner is validated on smoke.
    let datasets_dir = std::env::var("BRAIN_EVAL_DATASETS_DIR").unwrap_or_default();
    let instances = benchmark
        .load(Path::new(&datasets_dir))
        .map_err(|e| ProbeError::Load(e.to_string()))?;

    let groups = group_by_conversation(instances);

    let mut report = String::new();
    let mut asked = 0usize;
    let mut agg = ReadRollup::default();
    let mut total_durable = 0usize;
    let mut total_turns = 0usize;
    // Focused single-question tracer: dump every read stage for one question.
    let focus_qid = std::env::var("BRAIN_EVAL_FOCUS_QID").ok();

    for (conv, insts) in &groups {
        if max_questions.is_some_and(|m| asked >= m) {
            break;
        }
        // Reuse the given agent id (reads-only against a prior ingest) or a
        // fresh one (normal write+read). Data is scoped per (namespace,
        // agent), so the reuse id must match the run that wrote the data.
        let agent_id = reuse_agent.unwrap_or_else(brain_db_sdk::new_id);
        // Print the agent id so a write-then-keep run surfaces the id needed
        // for a later reads-only reuse (BRAIN_EVAL_REUSE_AGENT=<id>).
        println!(
            "  agent    : {}",
            uuid::Uuid::from_bytes(agent_id).hyphenated()
        );
        let harness = BrainEvalHarness::connect_minted(minter, endpoint, agent_id).await?;

        // Focused tracer: dump every read stage for one question, then stop.
        // Skips the yield count + the rest of the reads.
        if let Some(fq) = &focus_qid {
            match insts.iter().find(|i| &i.question_id == fq) {
                Some(inst) => focus_read(&harness, &judge, inst).await?,
                None => println!("focus question '{fq}' not found in conversation {conv}"),
            }
            harness.close().await?;
            break;
        }

        // Collect the conversation's user turns — needed for the extraction
        // read-back count whether or not we ingest.
        let sessions = insts.first().map(|i| i.sessions.as_slice()).unwrap_or(&[]);
        let turn_texts: Vec<String> = sessions
            .iter()
            .flat_map(|s| s.turns.iter())
            .filter(|t| t.role == "user" && !t.content.trim().is_empty())
            .map(|t| t.content.clone())
            .collect();
        let turns = turn_texts.len();

        let write_line = if reuse_agent.is_some() {
            // ---- reads-only: the write happened in a prior run; skip ingest
            // + drain and just count the graph that is already there. ----
            let yld = drain_and_count(&harness, &turn_texts).await?;
            format!(
                "WRITE [{conv}] REUSED prior ingest ({turns} turns) | extraction: {} entities, {} statements, {} relations",
                yld.entities, yld.statements, yld.relations,
            )
        } else {
            // ---- WRITE: ingest, drain the heavy async pipeline, then count. ----
            let mut durable = 0usize;
            for text in &turn_texts {
                let resp = harness
                    .client()
                    .encode(&EncodeBuilder::new(text.as_str()).build())
                    .await?;
                if resp.lsn > 0 && resp.memory_id != 0 && !resp.was_deduplicated {
                    durable += 1;
                }
            }
            total_durable += durable;
            total_turns += turns;

            // Barrier: block until the extraction + statement-embedding
            // metrics plateau, so we count and read a *fully* built graph for
            // every write, not a half-extracted one — the difference between
            // measuring the system and measuring a race.
            if let Some(addr) = metrics_addr {
                println!("  draining write pipeline (waiting for all stages to settle)…");
                crate::bench::runner::wait_for_write_flush(addr, conv.as_str()).await;
            }

            let yld = drain_and_count(&harness, &turn_texts).await?;
            // Per-tier success + drop counts from the server's own metrics —
            // a low yield is attributable to a failed/dropped stage.
            let health = match metrics_addr {
                Some(addr) => extraction_health(addr).await,
                None => String::new(),
            };
            format!(
                "WRITE [{conv}] durable {durable}/{turns} turns | extraction: {} entities, {} statements, {} relations{health}",
                yld.entities, yld.statements, yld.relations,
            )
        };
        println!("{write_line}");
        report.push_str(&write_line);
        report.push('\n');

        // ---- READ: per-question staged analysis + judge. ----
        for inst in insts {
            if max_questions.is_some_and(|m| asked >= m) {
                break;
            }
            asked += 1;
            let rec = staged_read(&harness, &judge, inst).await?;
            agg.record(&rec);
            let line = rec.render(inst);
            println!("  {line}");
            report.push_str("  ");
            report.push_str(&line);
            report.push('\n');
        }

        harness.close().await?;
        println!();
        report.push('\n');

        // Reads-only reuse targets ONE conversation (the agent that was
        // ingested). Later groups would reuse the same agent, which holds no
        // data for them — so stop after the reused conversation rather than
        // reporting spurious misses.
        if reuse_agent.is_some() {
            break;
        }
    }

    // ---- Rollup. ----
    let rollup = format!(
        "rollup: reads {}/{} correct (score {:.3}) | writes {total_durable}/{total_turns} durable\n  read failures by stage: retrieval-missed-gold={}, answer-dropped-it={}, abstention={}, unattributed={}",
        agg.correct,
        agg.total,
        agg.score(),
        agg.fail_retrieval,
        agg.fail_answer_drop,
        agg.fail_abstention,
        agg.fail_wrong,
    );
    println!("{rollup}");
    report.push_str(&rollup);
    report.push('\n');

    write_report(benchmark.id(), &report);

    Ok(agg.correct == agg.total && agg.total > 0)
}

/// One read's per-stage record.
struct ReadRecord {
    r8_answered: bool,
    verdict: Verdict,
    fail_stage: Option<&'static str>,
}

impl ReadRecord {
    fn render(&self, inst: &EvalInstance) -> String {
        let mark = |b: bool| if b { "✓" } else { "✗" };
        let outcome = match self.verdict {
            Verdict::Correct => "PASS".to_string(),
            other => {
                let stage = self.fail_stage.unwrap_or("?");
                let gold = truncate(&inst.answer, 40);
                format!(
                    "{} ← {stage} (gold=\"{gold}\")",
                    if matches!(other, Verdict::Partial) {
                        "PARTIAL"
                    } else {
                        "FAIL"
                    }
                )
            }
        };
        format!(
            "read {:<14} [{:<14}] R8{}  {outcome}",
            truncate(&inst.question_id, 14),
            inst.question_type.tag(),
            mark(self.r8_answered),
        )
    }
}

/// Trace one read through every stage, printing "what we got" at each so a
/// single failure can be diagnosed end to end: the server's own per-stage
/// trace incl. the retrievers, fusion, and filter chain (R7, via
/// `QUERY_TRACE`), the router's shaped answer + returned memory texts (R8),
/// and finally the grade + support verdict. A gold-retrievability probe
/// re-recalls with the gold phrase to test whether the supporting memory is
/// retrievable *at all* — separating "the gold isn't stored/indexed" from
/// "the question just doesn't match it".
async fn focus_read(
    harness: &BrainEvalHarness,
    judge: &crate::bench::score::judge::Judge,
    inst: &EvalInstance,
) -> Result<(), ProbeError> {
    let client = harness.client();
    println!(
        "═══════════════ FOCUS READ: {} ═══════════════",
        inst.question_id
    );
    println!("  question : {}", inst.question);
    println!("  gold     : {}", inst.answer);
    println!("  type     : {}", inst.question_type.tag());
    println!();

    // R7: the server's own per-stage trace (routing → retrievers → fusion →
    // filter chain), delivered over the wire — no server debug logs needed.
    let tr = client
        .query_trace(&QueryTraceRequest {
            query: query_request(&inst.question),
        })
        .await?;
    println!("  ── server per-stage trace ──");
    for line in tr.trace_text.lines() {
        println!("    {line}");
    }

    // R8: the router's shaped answer + the memory texts it actually returned.
    let recall = harness.recall(&inst.question, EPISODIC_CAP as u32).await?;
    println!(
        "  R8 recall: shape={}, {} memories returned:",
        answer_kind_name(recall.answer_kind),
        recall.memories.len()
    );
    for m in recall.memories.iter().take(8) {
        println!("    · {}", truncate(&m.text, 130));
    }

    // Gold-retrievability probe: can the supporting memory be surfaced by a
    // gold-flavored cue at all? If yes here but no above, it's a
    // question↔memory mismatch, not a storage/index gap.
    let cue = format!("{} {}", subject_hint(&inst.question), inst.answer);
    let probe = harness.recall(&cue, 5).await?;
    println!(
        "  gold-retrievability (cue = \"{cue}\"): {} hits",
        probe.memories.len()
    );
    for m in probe.memories.iter().take(3) {
        println!("    · {}", truncate(&m.text, 130));
    }

    // Extracted typed graph attached to the recalled memories — exactly what
    // the write pipeline produced (entities / statements / relations) for each.
    let enriched = client
        .recall(
            &RecallBuilder::new(&inst.question)
                .max_results(EPISODIC_CAP as u32)
                .include_graph(true)
                .include_text(true)
                .build(),
        )
        .await?;
    println!();
    println!("  ── extracted graph on recalled memories ──");
    for m in enriched.memories.iter().take(6) {
        println!("    · [{}]", truncate(&m.text, 90));
        match &m.graph {
            None => println!("        (no graph enrichment — memory not extracted)"),
            Some(g) => {
                for e in g.entities.iter().take(8) {
                    println!("        entity   : {} :: {}", e.name, e.type_qname);
                }
                for s in g.statements.iter().take(8) {
                    println!(
                        "        statement: {} --{}--> {} (conf {:.2})",
                        s.subject_name, s.predicate, s.object_label, s.confidence
                    );
                }
                for r in g.relations.iter().take(8) {
                    println!(
                        "        relation : {} --{}--> {}",
                        r.from_name, r.predicate, r.to_name
                    );
                }
            }
        }
    }

    // Verdict + support attribution.
    let answer = synthesize_answer(&inst.question, &recall, inst.question_type, EPISODIC_CAP);
    let judged = judge
        .grade(
            &inst.question_id,
            inst.question_type,
            &inst.question,
            &inst.answer,
            &answer,
        )
        .await;
    let retrieved: Vec<String> = recall.memories.iter().map(|m| m.text.clone()).collect();
    let (supported, _why) = judge
        .grade_support(&inst.question, &inst.answer, &retrieved)
        .await;
    println!();
    println!(
        "  verdict  : {:?} | gold-in-retrieved(support-judge) = {:?}",
        judged.verdict, supported
    );
    println!("  synth answer: {}", truncate(&answer, 200));
    Ok(())
}

/// First few capitalized words of the question — a cheap subject cue for the
/// gold-retrievability probe (e.g. "What is Caroline's …" → "Caroline").
fn subject_hint(question: &str) -> String {
    question
        .split_whitespace()
        .filter(|w| w.chars().next().is_some_and(char::is_uppercase))
        .take(2)
        .collect::<Vec<_>>()
        .join(" ")
}

async fn staged_read(
    harness: &BrainEvalHarness,
    judge: &crate::bench::score::judge::Judge,
    inst: &EvalInstance,
) -> Result<ReadRecord, ProbeError> {
    let recall = harness.recall(&inst.question, EPISODIC_CAP as u32).await?;
    let r8_answered = !recall.memories.is_empty();

    let answer = synthesize_answer(&inst.question, &recall, inst.question_type, EPISODIC_CAP);
    let judged = judge
        .grade(
            &inst.question_id,
            inst.question_type,
            &inst.question,
            &inst.answer,
            &answer,
        )
        .await;

    // Attribute a non-correct verdict to the stage that lost it. For the
    // hardest case — candidates retrieved and the router answered, but the
    // answer is wrong — ask the support-judge whether the gold was actually
    // in the retrieved context. That splits "retrieval never surfaced the
    // gold" from "context had it but the answer dropped it" — the difference
    // between a retrieval problem and a synthesis problem.
    let fail_stage: Option<&'static str> = match judged.verdict {
        Verdict::Correct => None,
        _ if !r8_answered => Some("abstention (router returned nothing)"),
        _ => {
            let retrieved: Vec<String> = recall.memories.iter().map(|m| m.text.clone()).collect();
            match judge
                .grade_support(&inst.question, &inst.answer, &retrieved)
                .await
                .0
            {
                Some(true) => Some("answer dropped it (gold WAS retrieved)"),
                Some(false) => Some("retrieval missed the gold"),
                None => Some("wrong answer (support unknown; heuristic judge)"),
            }
        }
    };

    Ok(ReadRecord {
        r8_answered,
        verdict: judged.verdict,
        fail_stage,
    })
}

/// Scrape the server's extractor metrics and format a one-line health tag
/// for the WRITE row: how many memories each tier RAN on, how many tier runs
/// FAILED, and how many candidates were dropped (apply-skip + schema-filter +
/// generic drops). `failed=0 dropped=0` means every stage completed cleanly.
/// Counters are cumulative server-wide — exact for a fresh single-conversation
/// run; a coarse upper bound across multiple conversations in one process.
async fn extraction_health(addr: std::net::SocketAddr) -> String {
    use crate::bench::report::run_report::ExtractionHealth;
    use crate::client::metrics::PrometheusMetrics;
    let Ok(m) = PrometheusMetrics::scrape(addr).await else {
        return String::new();
    };
    ExtractionHealth::from_metrics(&m).health_tag()
}

/// The per-conversation extraction totals (deduped by display string).
struct ExtractionYield {
    entities: usize,
    statements: usize,
    relations: usize,
}

/// Wait for the async extractor to settle, then count the typed graph by
/// recalling **every** ingested turn with graph enrichment and unioning the
/// result. This covers the whole corpus (each written memory is queried by
/// its own text), unlike a single-cue sample. Entity-type ids aren't needed
/// (the enumeration verbs require one in v1.0), and the counts dedup by
/// lowercased display string so the same fact surfacing on multiple hits is
/// counted once.
async fn drain_and_count(
    harness: &BrainEvalHarness,
    turns: &[String],
) -> Result<ExtractionYield, ProbeError> {
    let client = harness.client();
    // Poll the *last* ingested turn — it is extracted last, so its graph
    // carrying facts means the extraction queue has essentially caught up
    // (better than the first turn, which settles almost immediately).
    let probe_cue = turns.last().map(String::as_str).unwrap_or_default();

    // Settle: poll that memory until its graph carries facts.
    for _ in 0..DRAIN_ATTEMPTS {
        let ans = client
            .recall(
                &RecallBuilder::new(probe_cue)
                    .include_graph(true)
                    .max_results(5)
                    .build(),
            )
            .await?;
        let has_facts = ans.memories.iter().any(|m| {
            m.graph
                .as_ref()
                .is_some_and(|g| !g.statements.is_empty() || !g.relations.is_empty())
        });
        if has_facts {
            break;
        }
        tokio::time::sleep(DRAIN_INTERVAL).await;
    }

    // Union the graph across every ingested turn.
    let mut ents: HashSet<String> = HashSet::new();
    let mut stmts: HashSet<(String, String, String)> = HashSet::new();
    let mut rels: HashSet<(String, String, String)> = HashSet::new();
    for text in turns {
        let ans = client
            .recall(
                &RecallBuilder::new(text.as_str())
                    .include_graph(true)
                    .max_results(3)
                    .build(),
            )
            .await?;
        for m in &ans.memories {
            let Some(g) = &m.graph else { continue };
            for e in &g.entities {
                ents.insert(e.name.to_lowercase());
            }
            for s in &g.statements {
                stmts.insert((
                    s.subject_name.to_lowercase(),
                    s.predicate.clone(),
                    s.object_label.to_lowercase(),
                ));
            }
            for r in &g.relations {
                rels.insert((
                    r.from_name.to_lowercase(),
                    r.predicate.clone(),
                    r.to_name.to_lowercase(),
                ));
            }
        }
    }

    Ok(ExtractionYield {
        entities: ents.len(),
        statements: stmts.len(),
        relations: rels.len(),
    })
}

/// Running totals across all reads.
#[derive(Default)]
struct ReadRollup {
    total: usize,
    correct: usize,
    partial: usize,
    score_sum: f64,
    fail_retrieval: usize,
    fail_abstention: usize,
    fail_answer_drop: usize,
    fail_wrong: usize,
}

impl ReadRollup {
    fn record(&mut self, r: &ReadRecord) {
        self.total += 1;
        self.score_sum += r.verdict.score();
        match r.verdict {
            Verdict::Correct => self.correct += 1,
            Verdict::Partial => self.partial += 1,
            Verdict::Incorrect => {}
        }
        match r.fail_stage {
            Some(s) if s.contains("abstention") => self.fail_abstention += 1,
            Some(s) if s.contains("answer dropped") => self.fail_answer_drop += 1,
            Some(s) if s.contains("retrieval") => self.fail_retrieval += 1,
            Some(_) => self.fail_wrong += 1,
            None => {}
        }
    }

    fn score(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.score_sum / self.total as f64
        }
    }
}

/// Group instances by conversation, preserving encounter order. Instances
/// with no `conversation_id` key on their own `question_id` so each is its
/// own isolated session.
fn group_by_conversation(instances: Vec<EvalInstance>) -> BTreeMap<String, Vec<EvalInstance>> {
    let mut groups: BTreeMap<String, Vec<EvalInstance>> = BTreeMap::new();
    for inst in instances {
        let key = inst
            .conversation_id
            .clone()
            .unwrap_or_else(|| inst.question_id.clone());
        groups.entry(key).or_default().push(inst);
    }
    groups
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let head: String = s.chars().take(n.saturating_sub(1)).collect();
        format!("{head}…")
    }
}

/// Write the report text under `BRAIN_EVAL_OUTPUT_DIR` (default `./reports`).
/// Best-effort — a write failure is logged, not fatal, so the run's stdout
/// still stands.
fn write_report(bench_id: &str, body: &str) {
    let dir = std::env::var("BRAIN_EVAL_OUTPUT_DIR").unwrap_or_else(|_| "./reports".to_string());
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let path = format!("{dir}/{bench_id}-staged.txt");
    match std::fs::write(&path, body) {
        Ok(()) => println!("\nreport written: {path}"),
        Err(e) => eprintln!("warning: could not write report to {path}: {e}"),
    }
}
