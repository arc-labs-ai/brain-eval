//! Relation scenario: create → list_from → list_to.
//!
//! Exercises §19/06 "relation lifecycle". Creates two Person entities and
//! a `brain:related_to` relation between them, then asserts the relation
//! is visible both from the subject (`list_relations_from`) and to the
//! object (`list_relations_to`).

use std::net::SocketAddr;

use brain_db_sdk::new_id;
use brain_db_sdk::wire::types::{
    EntityCreateRequest, EvidenceRefWire, RelationCreateRequest, RelationListFromRequest,
    RelationListToRequest,
};

use super::super::ScenarioOutcome;
use super::{hex16, PERSON_TYPE_ID};
use crate::run::harness::{BrainEvalHarness, HarnessError};

const NAME: &str = "tg_relation_lifecycle";
/// Seeded symmetric many-to-many relation type.
const RELATION_TYPE: &str = "brain:related_to";

/// Create a relation, then list it from the subject and to the object.
pub async fn relation_lifecycle(endpoint: SocketAddr) -> ScenarioOutcome {
    match run(endpoint).await {
        Ok(o) => o,
        Err(e) => ScenarioOutcome::fail(NAME, format!("sdk error: {e}")),
    }
}

async fn run(endpoint: SocketAddr) -> Result<ScenarioOutcome, HarnessError> {
    let h = BrainEvalHarness::connect(endpoint).await?;
    let suffix = &hex16(h.agent_id())[..12];

    let from_entity = make_person(&h, &format!("Alan Turing {suffix}")).await?;
    let to_entity = make_person(&h, &format!("Christopher Morcom {suffix}")).await?;

    let now = unix_nanos_now();

    // --- create the relation -----------------------------------------
    let created = h
        .client()
        .create_relation(&RelationCreateRequest {
            relation_type: RELATION_TYPE.to_string(),
            from_entity,
            to_entity,
            properties_blob: Vec::new(),
            evidence: EvidenceRefWire::Inline(Vec::new()),
            extractor_id: 0,
            confidence: 0.9,
            valid_from_unix_nanos: now,
            valid_to_unix_nanos: 0,
            request_id: new_id(),
        })
        .await?;
    let relation_id = created.relation_id;
    if relation_id == [0u8; 16] {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            "create_relation returned an all-zero relation_id",
        ));
    }

    // --- list FROM the subject (must include the relation) -----------
    let from_list = h
        .client()
        .list_relations_from(&RelationListFromRequest {
            from_entity,
            relation_type_filter: String::new(),
            time_range_start_unix_nanos: 0,
            time_range_end_unix_nanos: 0,
            include_superseded: false,
            include_tombstoned: false,
            limit: 1000,
            cursor: Vec::new(),
        })
        .await?;
    let in_from = from_list.iter().any(|r| {
        r.relation_id == relation_id || (r.from_entity == from_entity && r.to_entity == to_entity)
    });

    // --- list TO the object (must include the relation) --------------
    let to_list = h
        .client()
        .list_relations_to(&RelationListToRequest {
            to_entity,
            relation_type_filter: String::new(),
            time_range_start_unix_nanos: 0,
            time_range_end_unix_nanos: 0,
            include_superseded: false,
            include_tombstoned: false,
            limit: 1000,
            cursor: Vec::new(),
        })
        .await?;
    let in_to = to_list.iter().any(|r| {
        r.relation_id == relation_id || (r.from_entity == from_entity && r.to_entity == to_entity)
    });

    h.close().await?;

    if !in_from {
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "relation not found via list_relations_from(subject) — {} rows",
                from_list.len()
            ),
        ));
    }
    if !in_to {
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "relation not found via list_relations_to(object) — {} rows",
                to_list.len()
            ),
        ));
    }

    Ok(ScenarioOutcome::pass(
        NAME,
        format!(
            "§19/06 relation: created {RELATION_TYPE}, visible from subject ({} rows) \
             and to object ({} rows)",
            from_list.len(),
            to_list.len()
        ),
    ))
}

async fn make_person(h: &BrainEvalHarness, canonical: &str) -> Result<[u8; 16], HarnessError> {
    Ok(h
        .client()
        .create_entity(&EntityCreateRequest {
            entity_type_id: PERSON_TYPE_ID,
            canonical_name: canonical.to_string(),
            aliases: Vec::new(),
            attributes_blob: Vec::new(),
            request_id: new_id(),
        })
        .await?
        .entity_id)
}

fn unix_nanos_now() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}
