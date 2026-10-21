//! Relation-traversal-depth scenario.
//!
//! The lifecycle scenario ([`super::relation`]) covers create + list-from/to.
//! This covers the §19/06 traversal-depth item: a multi-hop chain
//! `A → B → C → D` (via the seeded symmetric `brain:related_to`), traversed
//! outward from `A`:
//!
//! - **Reaches depth** — a `max_depth = 3` outgoing traversal returns at least
//!   one step at depth ≥ 2 (it walked past the immediate neighbour), and never
//!   a step deeper than the requested bound.
//! - **Bounds depth** — the same traversal at `max_depth = 1` returns nothing
//!   deeper than depth 1: the depth cap is enforced, not advisory.
//!
//! `direction = 0` is Outgoing (`crates/brain-protocol/src/ops/relation.rs`:
//! 0=Outgoing / 1=Incoming / 2=Both).

use std::net::SocketAddr;

use brain_db_sdk::new_id;
use brain_db_sdk::wire::types::{
    EntityCreateRequest, EvidenceRefWire, RelationCreateRequest, RelationTraverseRequest,
    TraversalPathWire,
};

use super::super::ScenarioOutcome;
use super::PERSON_TYPE_ID;
use crate::blackbox::system::hex16;
use crate::client::harness::{BrainEvalHarness, HarnessError};

const NAME: &str = "tg_relation_traversal";
const RELATION_TYPE: &str = "brain:related_to";
/// 0 = Outgoing (per the wire protocol's relation-op direction encoding).
const DIR_OUTGOING: u8 = 0;

/// Deepest `depth` reached across every step of every path.
fn max_step_depth(paths: &[TraversalPathWire]) -> u32 {
    paths
        .iter()
        .flat_map(|p| p.steps.iter())
        .map(|s| s.depth)
        .max()
        .unwrap_or(0)
}

/// Traverse a 4-node chain and assert the depth cap is both reachable and
/// enforced.
pub async fn relation_traversal(endpoint: SocketAddr) -> ScenarioOutcome {
    match run(endpoint).await {
        Ok(o) => o,
        Err(e) => ScenarioOutcome::fail(NAME, format!("sdk error: {e}")),
    }
}

async fn run(endpoint: SocketAddr) -> Result<ScenarioOutcome, HarnessError> {
    let h = BrainEvalHarness::connect(endpoint).await?;
    let suffix = &hex16(h.space_id())[..12];
    let now = unix_nanos_now();

    // A → B → C → D
    let a = make_person(&h, &format!("Node A {suffix}")).await?;
    let b = make_person(&h, &format!("Node B {suffix}")).await?;
    let c = make_person(&h, &format!("Node C {suffix}")).await?;
    let d = make_person(&h, &format!("Node D {suffix}")).await?;
    link(&h, a, b, now).await?;
    link(&h, b, c, now).await?;
    link(&h, c, d, now).await?;

    // --- depth reachable at max_depth = 3 --------------------------------
    let deep = h.client().traverse_relations(&traverse_req(a, 3)).await?;
    if deep.is_empty() {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            "outgoing traversal from A returned no paths on a 3-hop chain",
        ));
    }
    let deep_reached = max_step_depth(&deep);
    if deep_reached < 2 {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!("max_depth=3 traversal only reached depth {deep_reached}; expected >= 2"),
        ));
    }
    if deep_reached > 3 {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!("max_depth=3 traversal reached depth {deep_reached} — cap not enforced"),
        ));
    }

    // --- depth bounded at max_depth = 1 ----------------------------------
    let shallow = h.client().traverse_relations(&traverse_req(a, 1)).await?;
    let shallow_reached = max_step_depth(&shallow);

    h.close().await?;

    if shallow_reached > 1 {
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!("max_depth=1 traversal reached depth {shallow_reached} — cap leaked"),
        ));
    }

    Ok(ScenarioOutcome::pass(
        NAME,
        format!(
            "§19/06 relation traversal: A→B→C→D, max_depth=3 reached depth {deep_reached}, \
             max_depth=1 bounded at depth {shallow_reached}"
        ),
    ))
}

fn traverse_req(start: [u8; 16], max_depth: u32) -> RelationTraverseRequest {
    RelationTraverseRequest {
        start_entity: start,
        relation_types: vec![RELATION_TYPE.to_string()],
        direction: DIR_OUTGOING,
        max_depth,
        max_nodes: 1000,
        time_at_unix_nanos: 0,
        include_superseded: false,
        request_id: new_id(),
        act_as: None,
    }
}

async fn link(
    h: &BrainEvalHarness,
    from: [u8; 16],
    to: [u8; 16],
    now: u64,
) -> Result<(), HarnessError> {
    h.client()
        .create_relation(&RelationCreateRequest {
            session_id: 0,
            relation_type: RELATION_TYPE.to_string(),
            from_entity: from,
            to_entity: to,
            properties_blob: Vec::new(),
            evidence: EvidenceRefWire::Inline(Vec::new()),
            extractor_id: 0,
            confidence: 0.9,
            valid_from_unix_nanos: now,
            valid_to_unix_nanos: 0,
            request_id: new_id(),
            act_as: None,
        })
        .await?;
    Ok(())
}

async fn make_person(h: &BrainEvalHarness, canonical: &str) -> Result<[u8; 16], HarnessError> {
    Ok(h.client()
        .create_entity(&EntityCreateRequest {
            session_id: 0,
            entity_type_id: PERSON_TYPE_ID,
            canonical_name: canonical.to_string(),
            aliases: Vec::new(),
            attributes_blob: Vec::new(),
            request_id: new_id(),
            act_as: None,
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
