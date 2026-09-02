//! Typed-graph functional acceptance suite ("E2").
//!
//! The core suite ([`super`]) covers the cognitive primitives
//! (ENCODE / RECALL / FORGET / TXN). This module covers the typed-graph
//! half of the v1.0 acceptance doc (`spec/19_benchmarks/06_complete_acceptance.md`):
//! schema, entity, statement, relation, and async extraction.
//!
//! Every scenario drives a live `brain-server` through the SDK with real
//! assertions and a meaningful failure `detail`. Two rules keep reruns
//! from colliding on persistent, additive typed-graph state:
//!
//! - Markers (entity / statement names) are suffixed with the hex of a
//!   fresh space id, like the cognitive scenarios.
//! - Schema namespaces are suffixed with that same hex, because
//!   `SCHEMA_UPLOAD` is additive and persists across the run.
//!
//! The seeded `brain:` system namespace is active from byte zero, so the
//! entity / statement / relation scenarios use the built-in types
//! (`Person = EntityTypeId(1)`) and predicates (`brain:works_at`) directly —
//! only the schema scenario uploads a fresh user namespace.

use std::net::SocketAddr;

use super::ScenarioOutcome;

pub mod entity;
pub mod entity_merge;
pub mod extraction;
pub mod read_introspect;
pub mod relation;
pub mod relation_traverse;
pub mod schema;
pub mod schema_drop;
pub mod statement;
pub mod statement_depth;

pub use entity::entity_lifecycle;
pub use entity_merge::entity_merge_unmerge;
pub use extraction::extraction_pipeline;
pub use read_introspect::read_introspection;
pub use relation::relation_lifecycle;
pub use relation_traverse::relation_traversal;
pub use schema::schema_lifecycle;
pub use schema_drop::schema_drop;
pub use statement::statement_lifecycle;
pub use statement_depth::statement_depth;

/// Built-in `Person` entity type id, seeded by the `brain:` system schema
/// at byte zero (see `crates/brain-metadata/src/system_schema/schema.brain`).
/// Source order in that file pins it: `Person = EntityTypeId(1)`.
pub(crate) const PERSON_TYPE_ID: u32 = 1;

/// Run the typed-graph ("E2") functional acceptance scenarios against
/// `endpoint`. Mirrors [`super::run_core_scenarios`].
pub async fn run_typed_graph_scenarios(endpoint: SocketAddr) -> Vec<ScenarioOutcome> {
    vec![
        schema_lifecycle(endpoint).await,
        schema_drop(endpoint).await,
        entity_lifecycle(endpoint).await,
        entity_merge_unmerge(endpoint).await,
        statement_lifecycle(endpoint).await,
        statement_depth(endpoint).await,
        relation_lifecycle(endpoint).await,
        relation_traversal(endpoint).await,
        read_introspection(endpoint).await,
        extraction_pipeline(endpoint).await,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_person_type_id_matches_system_schema() {
        // Source order in schema.brain pins Person = EntityTypeId(1).
        assert_eq!(PERSON_TYPE_ID, 1);
    }
}
