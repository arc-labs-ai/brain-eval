//! Statement scenario: create Fact → get → list → supersession.
//!
//! Exercises §19/06 "statement lifecycle". Creates a Person subject, then
//! a `brain:current_role` Fact (a `stateful` predicate whose object is
//! `Value<text>`), reads it back by id, lists statements for the subject,
//! and finally creates a second `current_role` Fact to drive the
//! auto-supersession path — the stateful predicate makes the new
//! statement supersede the prior one (response carries `auto_superseded`
//! + `chain_root`).
//!
//! There is no dedicated "supersede statement" SDK verb; supersession is
//! exercised implicitly via a second create on the stateful predicate.

use std::net::SocketAddr;

use brain_db_sdk::new_id;
use brain_db_sdk::wire::types::{
    EntityCreateRequest, EvidenceRefWire, StatementCreateRequest, StatementGetRequest,
    StatementKindWire, StatementListRequest, StatementObjectWire, StatementValueWire,
};

use super::super::ScenarioOutcome;
use super::{hex16, PERSON_TYPE_ID};
use crate::run::harness::{BrainEvalHarness, HarnessError};

const NAME: &str = "tg_statement_lifecycle";
/// `brain:current_role` is seeded `stateful: true`, object `Value<text>`.
const PREDICATE: &str = "brain:current_role";

/// Create a Fact statement, get + list it, then supersede it.
pub async fn statement_lifecycle(endpoint: SocketAddr) -> ScenarioOutcome {
    match run(endpoint).await {
        Ok(o) => o,
        Err(e) => ScenarioOutcome::fail(NAME, format!("sdk error: {e}")),
    }
}

async fn run(endpoint: SocketAddr) -> Result<ScenarioOutcome, HarnessError> {
    let h = BrainEvalHarness::connect(endpoint).await?;
    let suffix = &hex16(h.agent_id())[..12];

    // --- subject entity ----------------------------------------------
    let subject = h
        .client()
        .create_entity(&EntityCreateRequest {
            entity_type_id: PERSON_TYPE_ID,
            canonical_name: format!("Grace Hopper {suffix}"),
            aliases: Vec::new(),
            attributes_blob: Vec::new(),
            request_id: new_id(),
        })
        .await?
        .entity_id;

    let now = unix_nanos_now();

    // --- create a Fact statement -------------------------------------
    let role_a = format!("Rear Admiral {suffix}");
    let created = h
        .client()
        .create_statement(&StatementCreateRequest {
            kind: StatementKindWire::Fact,
            subject,
            predicate: PREDICATE.to_string(),
            object: StatementObjectWire::Value(StatementValueWire::Text(role_a.clone())),
            confidence: 0.95,
            evidence: EvidenceRefWire::Inline(Vec::new()),
            extractor_id: 0,
            valid_from_unix_nanos: now,
            valid_to_unix_nanos: 0,
            event_at_unix_nanos: 0,
            schema_version: 0,
            request_id: new_id(),
        })
        .await?;
    let stmt_id = created.statement_id;
    if stmt_id == [0u8; 16] {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            "create_statement returned an all-zero statement_id",
        ));
    }

    // --- get by id ---------------------------------------------------
    let got = h
        .client()
        .get_statement(&StatementGetRequest {
            statement_id: stmt_id,
            follow_supersession: false,
        })
        .await?;
    if got.statement.statement_id != stmt_id {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            "get_statement returned a different statement_id than was created",
        ));
    }
    if got.statement.subject != subject {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            "get_statement subject != the created subject entity",
        ));
    }
    if got.statement.predicate != PREDICATE {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "get_statement predicate {:?} != {PREDICATE:?}",
                got.statement.predicate
            ),
        ));
    }

    // --- list statements for the subject (it must appear) ------------
    let listed = h
        .client()
        .list_statements(&StatementListRequest {
            subject,
            predicate: String::new(),
            kind: 0,
            min_confidence: 0.0,
            time_range_start_unix_nanos: 0,
            time_range_end_unix_nanos: 0,
            only_current: false,
            include_tombstoned: false,
            limit: 1000,
            cursor: Vec::new(),
        })
        .await?;
    if !listed.iter().any(|s| s.statement_id == stmt_id) {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "created statement not found in list_statements(subject) — {} listed",
                listed.len()
            ),
        ));
    }

    // --- supersession: a second create on the stateful predicate -----
    // `brain:current_role` is stateful, so the new statement supersedes
    // the first; the response reports the auto-superseded prior id and a
    // non-zero chain root.
    let role_b = format!("Commodore {suffix}");
    let superseding = h
        .client()
        .create_statement(&StatementCreateRequest {
            kind: StatementKindWire::Fact,
            subject,
            predicate: PREDICATE.to_string(),
            object: StatementObjectWire::Value(StatementValueWire::Text(role_b)),
            confidence: 0.97,
            evidence: EvidenceRefWire::Inline(Vec::new()),
            extractor_id: 0,
            valid_from_unix_nanos: now + 1,
            valid_to_unix_nanos: 0,
            event_at_unix_nanos: 0,
            schema_version: 0,
            request_id: new_id(),
        })
        .await?;

    let superseded_prior = superseding.auto_superseded == stmt_id;
    let chain_linked = superseding.chain_root != [0u8; 16];

    h.close().await?;

    if !superseded_prior {
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "second stateful {PREDICATE} create did not auto-supersede the prior \
                 statement (auto_superseded={:?}, expected the first id)",
                superseding.auto_superseded
            ),
        ));
    }
    if !chain_linked {
        return Ok(ScenarioOutcome::fail(
            NAME,
            "superseding statement reported an all-zero chain_root",
        ));
    }

    Ok(ScenarioOutcome::pass(
        NAME,
        format!(
            "§19/06 statement: created Fact {PREDICATE}, get round-trips, appears in list \
             ({} rows), stateful re-create auto-superseded the prior + linked a chain root",
            listed.len()
        ),
    ))
}

fn unix_nanos_now() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}
