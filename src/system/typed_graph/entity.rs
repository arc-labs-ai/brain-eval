//! Entity scenario: create → get → resolve → list.
//!
//! Exercises §19/06 "entity lifecycle" against the seeded `brain:Person`
//! type: create a Person, read it back by id (fields round-trip), resolve
//! it by exact canonical name (resolution returns the same id), and list
//! Person entities by type filter (the new entity appears).

use std::net::SocketAddr;

use brain_db_sdk::new_id;
use brain_db_sdk::wire::types::{
    EntityCreateRequest, EntityGetRequest, EntityListRequest, EntityResolveRequest,
    ResolutionOutcomeWire,
};

use super::super::ScenarioOutcome;
use super::{hex16, PERSON_TYPE_ID};
use crate::run::harness::{BrainEvalHarness, HarnessError};

const NAME: &str = "tg_entity_lifecycle";

/// Create a Person, get it, resolve it by name, list it by type.
pub async fn entity_lifecycle(endpoint: SocketAddr) -> ScenarioOutcome {
    match run(endpoint).await {
        Ok(o) => o,
        Err(e) => ScenarioOutcome::fail(NAME, format!("sdk error: {e}")),
    }
}

async fn run(endpoint: SocketAddr) -> Result<ScenarioOutcome, HarnessError> {
    let h = BrainEvalHarness::connect(endpoint).await?;
    // Unique canonical name so repeated runs mint distinct entities.
    let suffix = &hex16(h.agent_id())[..12];
    let canonical = format!("Ada Lovelace {suffix}");

    // --- create ------------------------------------------------------
    let created = h
        .client()
        .create_entity(&EntityCreateRequest {
            entity_type_id: PERSON_TYPE_ID,
            canonical_name: canonical.clone(),
            aliases: vec![format!("Ada {suffix}")],
            attributes_blob: Vec::new(),
            request_id: new_id(),
        })
        .await?;
    let entity_id = created.entity_id;
    if entity_id == [0u8; 16] {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            "create_entity returned an all-zero entity_id",
        ));
    }

    // --- get by id (fields must round-trip) --------------------------
    let got = h
        .client()
        .get_entity(&EntityGetRequest { entity_id })
        .await?;
    if got.entity.entity_id != entity_id {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            "get_entity returned a different entity_id than was created",
        ));
    }
    if got.entity.canonical_name != canonical {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "get_entity canonical_name {:?} != created {:?}",
                got.entity.canonical_name, canonical
            ),
        ));
    }
    if got.entity.entity_type_id != PERSON_TYPE_ID {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "get_entity entity_type_id {} != Person ({PERSON_TYPE_ID})",
                got.entity.entity_type_id
            ),
        ));
    }

    // --- resolve by exact canonical name (tier 1) --------------------
    // allow_create=false: we expect to RESOLVE the existing one, not mint.
    let resolved = h
        .client()
        .resolve_entity(&EntityResolveRequest {
            candidate_name: canonical.clone(),
            context: String::new(),
            entity_type_hint: PERSON_TYPE_ID,
            allow_create: false,
            request_id: new_id(),
        })
        .await?;
    if resolved.outcome != ResolutionOutcomeWire::Resolved {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "resolve by exact name gave outcome {:?}, expected Resolved",
                resolved.outcome
            ),
        ));
    }
    if resolved.resolved_entity != entity_id {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            "resolve returned a different entity than the one created",
        ));
    }

    // --- list by type (the new entity must appear) -------------------
    let listed = h
        .client()
        .list_entities(&EntityListRequest {
            entity_type_id: PERSON_TYPE_ID,
            name_prefix: String::new(),
            mention_count_min: 0,
            include_tombstoned: false,
            include_merged: false,
            limit: 1000,
            cursor: Vec::new(),
        })
        .await?;
    let appears = listed.iter().any(|it| it.entity.entity_id == entity_id);

    h.close().await?;

    if !appears {
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "created Person not found in list_entities(type=Person) — {} listed",
                listed.len()
            ),
        ));
    }

    Ok(ScenarioOutcome::pass(
        NAME,
        format!(
            "§19/06 entity: created Person {canonical}, get round-trips, \
             resolve(exact)=Resolved same id, appears in list ({} Person rows)",
            listed.len()
        ),
    ))
}
