//! Entity-resolution-breadth scenario.
//!
//! The lifecycle scenario ([`super::entity`]) resolves an existing entity by
//! its exact canonical name (tier 1, `Resolved`). This covers the §19/06
//! resolution outcomes it does not:
//!
//! - **No-match, no create** — resolving a never-seen name with
//!   `allow_create = false` is `NotFound` (tier 0), not a silent mint.
//! - **No-match → create** — the same name with `allow_create = true` is
//!   `Created`, returning a fresh non-zero id.
//! - **Resolve the created one** — an exact re-resolve now returns `Resolved`
//!   with that id (the create persisted; tier 1).
//! - **Rename keeps identity** — after `rename_entity(..., move_to_alias =
//!   true)`, the entity resolves under its new canonical name *and* still under
//!   the old name (now an alias): a rename moves the label, not the identity.
//!
//! The fuzzy / embedding tiers (2–3) are not asserted here: they depend on the
//! resolver's similarity thresholds, which are not a stable client-side
//! contract to gate on. This scenario stays on the deterministic outcomes.

use std::net::SocketAddr;

use brain_db_sdk::new_id;
use brain_db_sdk::wire::types::{EntityRenameRequest, EntityResolveRequest, ResolutionOutcomeWire};

use super::super::ScenarioOutcome;
use super::PERSON_TYPE_ID;
use crate::blackbox::system::hex16;
use crate::client::harness::{BrainEvalHarness, HarnessError};

const NAME: &str = "tg_resolution_breadth";

/// Exercise NotFound / Created / Resolved outcomes and rename-preserves-identity.
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
    // A name unique to this run so the "never seen" precondition holds even
    // across reruns against persistent state.
    let name = format!("Rosalind Franklin {suffix}");

    // --- no match, no create --------------------------------------------
    let miss = h
        .client()
        .resolve_entity(&resolve_req(&name, false))
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

    // --- no match → create ----------------------------------------------
    let created = h.client().resolve_entity(&resolve_req(&name, true)).await?;
    if created.outcome != ResolutionOutcomeWire::Created {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "resolve with allow_create=true gave {:?}, expected Created",
                created.outcome
            ),
        ));
    }
    let id = created.resolved_entity;
    if id == [0u8; 16] {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            "Created outcome returned an all-zero entity id",
        ));
    }

    // --- resolve the created one exactly (tier 1) -----------------------
    let hit = h
        .client()
        .resolve_entity(&resolve_req(&name, false))
        .await?;
    if hit.outcome != ResolutionOutcomeWire::Resolved || hit.resolved_entity != id {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "re-resolve of the created entity gave outcome {:?} id={:?}, expected Resolved id={:?}",
                hit.outcome, hit.resolved_entity, id
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
        "§19/06 resolution breadth: NotFound (no create) → Created (allow_create) → \
         Resolved (exact); rename kept identity under both the new canonical name and \
         the old-name alias",
    ))
}
