//! Entity-resolution-breadth scenario.
//!
//! The lifecycle scenario ([`super::entity`]) resolves an existing entity by
//! its exact canonical name (tier 1, `Resolved`). This covers the §19/06
//! resolution outcomes it does not, gating only the confirmed contract:
//!
//! - **Exact resolve** — an existing entity resolves by its canonical name
//!   (`Resolved`, same id).
//! - **No-match, no create** — resolving a never-seen name with
//!   `allow_create = false` is `NotFound`, not a silent mint.
//! - **No-match, allow_create** — with `allow_create = true` on an unknown
//!   name the resolver's tier-5 create-fallback mints a new entity and returns
//!   `Created`; an exact re-resolve then returns that id. (`allow_create =
//!   false` on an unknown name is `NotFound`, per the earlier bullet.)
//! - **Rename keeps identity** — after `rename_entity(..., move_to_alias =
//!   true)`, the entity resolves under its new canonical name *and* still under
//!   the old name (now an alias): a rename moves the label, not the identity.
//!
//! Not asserted: the fuzzy / embedding tiers (2–3). Tier 2 (trigram) exists but
//! its threshold isn't a stable client contract; tier 3 (embedding) is unwired.
//! The observe-only fuzzy scaffold ([`super::scaffolds`]) records those.

use std::net::SocketAddr;

use brain_db_sdk::new_id;
use brain_db_sdk::wire::types::{
    EntityCreateRequest, EntityRenameRequest, EntityResolveRequest, ResolutionOutcomeWire,
};

use super::super::ScenarioOutcome;
use super::PERSON_TYPE_ID;
use crate::blackbox::system::hex16;
use crate::client::harness::{BrainEvalHarness, HarnessError};

const NAME: &str = "tg_resolution_breadth";

/// Exercise Resolved / NotFound outcomes and rename-preserves-identity.
pub async fn resolution_breadth(endpoint: SocketAddr) -> ScenarioOutcome {
    match run(endpoint).await {
        Ok(o) => o,
        Err(e) => ScenarioOutcome::fail(NAME, format!("sdk error: {e}")),
    }
}

fn resolve_req(name: &str, allow_create: bool) -> EntityResolveRequest {
    EntityResolveRequest {
        candidate_name: name.to_string(),
        resolution_context: String::new(),
        entity_type_hint: PERSON_TYPE_ID,
        allow_create,
        request_id: new_id(),
        act_as: None,
    }
}

async fn run(endpoint: SocketAddr) -> Result<ScenarioOutcome, HarnessError> {
    let h = BrainEvalHarness::connect(endpoint).await?;
    let suffix = &hex16(h.space_id())[..12];
    let name = format!("Rosalind Franklin {suffix}");

    // Create deterministically (resolve-create is a server stub — see below).
    let id = h
        .client()
        .create_entity(&EntityCreateRequest {
            session_id: 0,
            entity_type_id: PERSON_TYPE_ID,
            canonical_name: name.clone(),
            aliases: Vec::new(),
            attributes_blob: Vec::new(),
            request_id: new_id(),
            act_as: None,
        })
        .await?
        .entity_id;

    // --- exact resolve (tier 1) -----------------------------------------
    let hit = h
        .client()
        .resolve_entity(&resolve_req(&name, false))
        .await?;
    if hit.outcome != ResolutionOutcomeWire::Resolved || hit.resolved_entity != id {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "exact resolve gave outcome {:?} id={:?}, expected Resolved id={:?}",
                hit.outcome, hit.resolved_entity, id
            ),
        ));
    }

    // --- no match, no create --------------------------------------------
    let unseen = format!("Nikola Unseen {suffix}");
    let miss = h
        .client()
        .resolve_entity(&resolve_req(&unseen, false))
        .await?;
    if miss.outcome != ResolutionOutcomeWire::NotFound {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "resolve of an unknown name with allow_create=false gave {:?}, expected NotFound",
                miss.outcome
            ),
        ));
    }

    // --- no match, allow_create → tier-5 create-fallback → Created -------
    let unseen_create = format!("Ferdinand Unseen {suffix}");
    let created = h
        .client()
        .resolve_entity(&resolve_req(&unseen_create, true))
        .await?;
    if created.outcome != ResolutionOutcomeWire::Created || created.resolved_entity == [0u8; 16] {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "resolve with allow_create=true on an unknown name gave {:?} id={:?}, \
                 expected Created with a fresh id (tier-5 create-fallback)",
                created.outcome, created.resolved_entity
            ),
        ));
    }
    // The freshly-created entity now resolves exactly to that id.
    let refetch = h
        .client()
        .resolve_entity(&resolve_req(&unseen_create, false))
        .await?;
    if refetch.outcome != ResolutionOutcomeWire::Resolved
        || refetch.resolved_entity != created.resolved_entity
    {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "after create-fallback, exact re-resolve gave {:?} id={:?}, expected Resolved id={:?}",
                refetch.outcome, refetch.resolved_entity, created.resolved_entity
            ),
        ));
    }

    // --- rename keeps identity under both names -------------------------
    let new_name = format!("Rosalind E. Franklin {suffix}");
    let renamed = h
        .client()
        .rename_entity(&EntityRenameRequest {
            entity_id: id,
            new_canonical_name: new_name.clone(),
            move_to_alias: true,
            request_id: new_id(),
        })
        .await?;
    if renamed.entity.entity_id != id {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            "rename returned a different entity id than the one renamed",
        ));
    }

    let by_new = h
        .client()
        .resolve_entity(&resolve_req(&new_name, false))
        .await?;
    if by_new.outcome != ResolutionOutcomeWire::Resolved || by_new.resolved_entity != id {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "resolve by the new canonical name gave {:?} id={:?}, expected Resolved id={:?}",
                by_new.outcome, by_new.resolved_entity, id
            ),
        ));
    }

    // Old name was moved to an alias, so it must still resolve to the same id.
    let by_old = h
        .client()
        .resolve_entity(&resolve_req(&name, false))
        .await?;

    h.close().await?;

    if by_old.outcome != ResolutionOutcomeWire::Resolved || by_old.resolved_entity != id {
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "resolve by the old name (moved to alias) gave {:?} id={:?}, expected Resolved id={:?}",
                by_old.outcome, by_old.resolved_entity, id
            ),
        ));
    }

    Ok(ScenarioOutcome::pass(
        NAME,
        "§19/06 resolution breadth: exact→Resolved; unknown+no-create→NotFound; \
         unknown+allow_create→Created (tier-5 fallback) then re-resolves to that id; \
         rename kept identity under the new canonical name and the old-name alias",
    ))
}
