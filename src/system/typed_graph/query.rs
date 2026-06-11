//! Query scenario: structured QUERY over the typed graph.
//!
//! Exercises §19/06 "typed-graph query" — the two retrieval modes the
//! acceptance gate calls out explicitly:
//!
//! - **Free-text query** → semantic + lexical retrievers invoked and
//!   fused. Proven by encoding a memory with a run-unique token, then
//!   polling a free-text QUERY until that memory surfaces with both
//!   retrievers reporting success (lexical/semantic indexing is async, so
//!   the poll is bounded ~10s).
//! - **Entity-anchored query** → graph retriever invoked and weighted.
//!   Proven by creating a relation `A → B` and asserting an anchored QUERY
//!   on `A` surfaces neighbor `B` (or the relation) via the graph walk.
//!   Relations are read straight from redb, so this leg is immediate (no
//!   async-index lag).
//!
//! A bare, unconnected entity surfaces in NEITHER mode by design: the graph
//! walk omits the anchor itself and emits only its neighbours
//! (`spec/13_retrievers/04_graph_retriever.md` §2-3), and entity
//! `canonical_name`s live in no QUERY-searched corpus — by-name entity
//! lookup is `ENTITY_RESOLVE`'s job, not QUERY's. An earlier version of this
//! scenario asserted the opposite (and used the wrong `kind` discriminant);
//! that was a test bug, not a server bug.

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use brain_db_sdk::new_id;
use brain_db_sdk::wire::types::{
    EntityCreateRequest, EvidenceRefWire, QueryRequest, QueryResponse, RelationCreateRequest,
    RetrieverSelectionWire, RetrieverWire,
};
use brain_db_sdk::EncodeBuilder;

use super::super::ScenarioOutcome;
use super::{hex16, ITEM_KIND_ENTITY, ITEM_KIND_MEMORY, ITEM_KIND_RELATION, PERSON_TYPE_ID};
use crate::run::harness::{BrainEvalHarness, HarnessError};

const NAME: &str = "tg_query";
/// Seeded symmetric many-to-many relation type (same as the relation scenario).
const RELATION_TYPE: &str = "brain:related_to";
/// How long to wait for async lexical/semantic indexing of the memory.
const MAX_WAIT: Duration = Duration::from_secs(10);
/// Poll cadence while waiting for the free-text memory to index.
const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Prove both QUERY retrieval modes against a live full-stack server.
pub async fn query_typed_graph(endpoint: SocketAddr) -> ScenarioOutcome {
    match run(endpoint).await {
        Ok(o) => o,
        Err(e) => ScenarioOutcome::fail(NAME, format!("sdk error: {e}")),
    }
}

async fn run(endpoint: SocketAddr) -> Result<ScenarioOutcome, HarnessError> {
    let h = BrainEvalHarness::connect(endpoint).await?;
    let suffix = &hex16(h.agent_id())[..12];

    // ================================================================
    // Leg 1 — entity-anchored QUERY: graph retriever surfaces a neighbour.
    // ================================================================
    let anchor_name = format!("Zylphara Quintessence {suffix}");
    let entity_a = h
        .client()
        .create_entity(&EntityCreateRequest {
            entity_type_id: PERSON_TYPE_ID,
            canonical_name: anchor_name.clone(),
            aliases: Vec::new(),
            attributes_blob: Vec::new(),
            request_id: new_id(),
        })
        .await?
        .entity_id;
    let entity_b = h
        .client()
        .create_entity(&EntityCreateRequest {
            entity_type_id: PERSON_TYPE_ID,
            canonical_name: format!("Vornak Threnody {suffix}"),
            aliases: Vec::new(),
            attributes_blob: Vec::new(),
            request_id: new_id(),
        })
        .await?
        .entity_id;

    let now = unix_nanos_now();
    let relation_id = h
        .client()
        .create_relation(&RelationCreateRequest {
            relation_type: RELATION_TYPE.to_string(),
            from_entity: entity_a,
            to_entity: entity_b,
            properties_blob: Vec::new(),
            evidence: EvidenceRefWire::Inline(Vec::new()),
            extractor_id: 0,
            confidence: 0.9,
            valid_from_unix_nanos: now,
            valid_to_unix_nanos: 0,
            request_id: new_id(),
        })
        .await?
        .relation_id;

    // Anchor on A; the graph retriever walks A's outgoing relations and
    // surfaces B + the relation. Carry A's name as the text so the
    // semantic/lexical lanes have a cue too (we only assert on graph here).
    let anchored = h
        .client()
        .query(&QueryRequest {
            text: anchor_name.clone(),
            entity_anchor: Some(entity_a),
            kind_filter: Vec::new(),
            predicate_filter: Vec::new(),
            time_filter: None,
            confidence_min: None,
            include_tombstoned: false,
            include_superseded: false,
            limit: 50,
            retrievers: RetrieverSelectionWire::Auto,
            fusion_config: None,
            request_id: new_id(),
        })
        .await?;

    if let Some(bad) = errored_retriever(&anchored) {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!("anchored QUERY had an errored retriever: {bad}"),
        ));
    }
    if !invoked(&anchored, RetrieverWire::Graph) {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "entity-anchored QUERY did not invoke the graph retriever (outcomes: {})",
                outcome_summary(&anchored)
            ),
        ));
    }
    let neighbour_surfaced = anchored.items.iter().any(|it| {
        (it.id.kind == ITEM_KIND_ENTITY && it.id.bytes == entity_b)
            || (it.id.kind == ITEM_KIND_RELATION && it.id.bytes == relation_id)
    });
    if !neighbour_surfaced {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "entity-anchored QUERY on A did not surface neighbour B or the relation \
                 via the graph walk (items={}, outcomes: {})",
                anchored.items.len(),
                outcome_summary(&anchored)
            ),
        ));
    }

    // ================================================================
    // Leg 2 — free-text QUERY: semantic + lexical surface an encoded memory.
    // ================================================================
    let token = format!("Qffluvian{suffix}");
    let mem_text = format!("The {token} protocol governs deep-sea telemetry relays.");
    let mem_id = h
        .client()
        .encode(&EncodeBuilder::new(mem_text.as_str()).deduplicate(false).build())
        .await?
        .memory_id;

    let deadline = Instant::now() + MAX_WAIT;
    let mut polls = 0u32;
    let outcome = loop {
        polls += 1;
        let freetext = h
            .client()
            .query(&QueryRequest {
                text: token.clone(),
                entity_anchor: None,
                kind_filter: Vec::new(),
                predicate_filter: Vec::new(),
                time_filter: None,
                confidence_min: None,
                include_tombstoned: false,
                include_superseded: false,
                limit: 50,
                retrievers: RetrieverSelectionWire::Auto,
                fusion_config: None,
                request_id: new_id(),
            })
            .await?;

        if let Some(bad) = errored_retriever(&freetext) {
            h.close().await?;
            return Ok(ScenarioOutcome::fail(
                NAME,
                format!("free-text QUERY had an errored retriever: {bad}"),
            ));
        }

        let mem_hit = freetext
            .items
            .iter()
            .any(|it| it.id.kind == ITEM_KIND_MEMORY && u128::from_be_bytes(it.id.bytes) == mem_id);
        let both_invoked = invoked(&freetext, RetrieverWire::Semantic)
            && invoked(&freetext, RetrieverWire::Lexical);

        if mem_hit && both_invoked {
            break Ok(ScenarioOutcome::pass(
                NAME,
                format!(
                    "§19/06 query: entity-anchored QUERY surfaced a graph neighbour \
                     (graph invoked, B/relation in results); free-text QUERY surfaced the \
                     encoded memory via semantic+lexical fusion after {polls} poll(s)"
                ),
            ));
        }
        if Instant::now() >= deadline {
            break Ok(ScenarioOutcome::fail(
                NAME,
                format!(
                    "free-text QUERY did not surface the encoded memory with semantic+lexical \
                     contributions within {}s ({polls} polls; last items={}; outcomes: {})",
                    MAX_WAIT.as_secs(),
                    freetext.items.len(),
                    outcome_summary(&freetext)
                ),
            ));
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    };

    h.close().await?;
    outcome
}

/// A retriever counts as "invoked" when it ran to success (`status == 0`).
/// `Skipped` (1) means it was not selected / had no signal; `Timeout` (2)
/// and `Failure` (3) are caught separately by [`errored_retriever`].
fn invoked(resp: &QueryResponse, which: RetrieverWire) -> bool {
    resp.retriever_outcomes
        .iter()
        .any(|o| o.retriever == which && o.status == 0)
}

/// Returns the first hard-errored retriever's message, if any. Status byte:
/// 0=Success, 1=Skipped, 2=Timeout, 3=Failure. Skipped is a legitimate
/// no-signal outcome, so only Timeout / Failure count as errors.
fn errored_retriever(resp: &QueryResponse) -> Option<String> {
    resp.retriever_outcomes
        .iter()
        .find(|o| o.status == 2 || o.status == 3)
        .map(|o| format!("{:?} status={} ({})", o.retriever, o.status, o.message))
}

/// Compact `Retriever=statusN(count)` summary for failure diagnostics.
fn outcome_summary(resp: &QueryResponse) -> String {
    resp.retriever_outcomes
        .iter()
        .map(|o| format!("{:?}=status{}({})", o.retriever, o.status, o.result_count))
        .collect::<Vec<_>>()
        .join(", ")
}

fn unix_nanos_now() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}
