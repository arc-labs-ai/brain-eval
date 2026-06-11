//! Typed-graph functional acceptance suite ("E2").
//!
//! The core suite ([`super`]) covers the cognitive primitives
//! (ENCODE / RECALL / FORGET / TXN). This module covers the typed-graph
//! half of the v1.0 acceptance doc (`spec/19_benchmarks/06_complete_acceptance.md`):
//! schema, entity, statement, relation, query, and async extraction.
//!
//! Every scenario drives a live `brain-server` through the SDK with real
//! assertions and a meaningful failure `detail`. Two rules keep reruns
//! from colliding on persistent, additive typed-graph state:
//!
//! - Markers (entity / statement names) are suffixed with the hex of a
//!   fresh agent id, like the cognitive scenarios.
//! - Schema namespaces are suffixed with that same hex, because
//!   `SCHEMA_UPLOAD` is additive and persists across the run.
//!
//! The seeded `brain:` system namespace is active from byte zero, so the
//! entity / statement / relation / query scenarios use the built-in
//! types (`Person = EntityTypeId(1)`) and predicates (`brain:works_at`)
//! directly — only the schema scenario uploads a fresh user namespace.

use std::net::SocketAddr;

use super::ScenarioOutcome;

pub mod entity;
pub mod extraction;
pub mod query;
pub mod relation;
pub mod schema;
pub mod statement;

pub use entity::entity_lifecycle;
pub use extraction::extraction_pipeline;
pub use query::query_typed_graph;
pub use relation::relation_lifecycle;
pub use schema::schema_lifecycle;
pub use statement::statement_lifecycle;

/// Built-in `Person` entity type id, seeded by the `brain:` system schema
/// at byte zero (see `crates/brain-metadata/src/system_schema/schema.brain`).
/// Source order in that file pins it: `Person = EntityTypeId(1)`.
pub(crate) const PERSON_TYPE_ID: u32 = 1;

/// `ItemIdWire.kind` discriminants in a QUERY response, matching the
/// server's `item_id_to_wire` mapping (`brain-ops` query handler):
/// `Memory = 0`, `Statement = 1`, `Entity = 2`, `Relation = 3`.
pub(crate) const ITEM_KIND_MEMORY: u8 = 0;
pub(crate) const ITEM_KIND_ENTITY: u8 = 2;
pub(crate) const ITEM_KIND_RELATION: u8 = 3;

/// Lowercase hex of a 16-byte id — a unique, collision-free marker /
/// namespace suffix. Mirrors [`super::hex16`] (private there).
pub(crate) fn hex16(id: [u8; 16]) -> String {
    let mut s = String::with_capacity(32);
    for b in id {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Run the typed-graph ("E2") functional acceptance scenarios against
/// `endpoint`. Mirrors [`super::run_core_scenarios`].
pub async fn run_typed_graph_scenarios(endpoint: SocketAddr) -> Vec<ScenarioOutcome> {
    vec![
        schema_lifecycle(endpoint).await,
        entity_lifecycle(endpoint).await,
        statement_lifecycle(endpoint).await,
        relation_lifecycle(endpoint).await,
        query_typed_graph(endpoint).await,
        extraction_pipeline(endpoint).await,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex16_is_32_chars() {
        assert_eq!(hex16([0xab; 16]).len(), 32);
    }

    #[test]
    fn seeded_person_type_id_matches_system_schema() {
        // Source order in schema.brain pins Person = EntityTypeId(1).
        assert_eq!(PERSON_TYPE_ID, 1);
    }

    #[test]
    fn query_item_kinds_match_server_mapping() {
        // brain-ops `item_id_to_wire`: Memory=0, Statement=1, Entity=2,
        // Relation=3. A QUERY entity item is kind 2, not the type id.
        assert_eq!(ITEM_KIND_MEMORY, 0);
        assert_eq!(ITEM_KIND_ENTITY, 2);
        assert_eq!(ITEM_KIND_RELATION, 3);
    }
}
