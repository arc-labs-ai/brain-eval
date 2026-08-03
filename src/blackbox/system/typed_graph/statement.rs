//! Statement scenario: create Preference → get → list → supersession.
//!
//! Exercises §19/06 "statement lifecycle". Creates a Person subject, then
//! a `brain:behavior_tone` Preference (a `stateful` predicate whose object
//! is `Value<text>`), reads it back by id, lists statements for the
//! subject, and finally creates a second `behavior_tone` Preference to
//! drive the auto-supersession path — the stateful predicate makes the new
//! statement supersede the prior one (response carries `auto_superseded`
//! + `chain_root`).
//!
//! The predicate must be one the *system* schema declares: the server
//! rejects an undeclared `brain:` predicate with `PredicateNotInSchema`.
//! `crates/brain-metadata/src/system_schema/schema.brain` declares exactly
//! five stateful predicates and all are `kind: Preference`, so the kind
//! here must match — a `Fact` on a Preference predicate is not the shape
//! the schema describes.
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
use super::PERSON_TYPE_ID;
use crate::blackbox::system::hex16;
use crate::client::harness::{BrainEvalHarness, HarnessError};

const NAME: &str = "tg_statement_lifecycle";
/// `brain:behavior_tone` is declared `stateful: true`, object `Value<text>`
/// in the system schema. Stateful predicates are what drive supersession.
const PREDICATE: &str = "brain:behavior_tone";

/// Create a Preference statement, get + list it, then supersede it.
pub async fn statement_lifecycle(endpoint: SocketAddr) -> ScenarioOutcome {
    match run(endpoint).await {
        Ok(o) => o,
        Err(e) => ScenarioOutcome::fail(NAME, format!("sdk error: {e}")),
    }
}

async fn run(endpoint: SocketAddr) -> Result<ScenarioOutcome, HarnessError> {
    let h = BrainEvalHarness::connect(endpoint).await?;
    let suffix = &hex16(h.space_id())[..12];

    // --- subject entity ----------------------------------------------
    let subject = h
        .client()
        .create_entity(&EntityCreateRequest {
            session_id: 0,
            entity_type_id: PERSON_TYPE_ID,
            canonical_name: format!("Grace Hopper {suffix}"),
            aliases: Vec::new(),
            attributes_blob: Vec::new(),
            request_id: new_id(),
            act_as: None,
        })
        .await?
        .entity_id;

    let now = unix_nanos_now();

    // --- create a Preference statement -------------------------------
    let tone_a = format!("concise {suffix}");
    let created = h
        .client()
        .create_statement(&StatementCreateRequest {
            session_id: 0,
            kind: StatementKindWire::Preference,
            subject,
            predicate: PREDICATE.to_string(),
            object: StatementObjectWire::Value(StatementValueWire::Text(tone_a.clone())),
            confidence: 0.95,
            evidence: EvidenceRefWire::Inline(Vec::new()),
            extractor_id: 0,
            valid_from_unix_nanos: now,
            valid_to_unix_nanos: 0,
            event_at_unix_nanos: 0,
            schema_version: 0,
            request_id: new_id(),
            act_as: None,
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
            act_as: None,
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
            act_as: None,
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
    // `brain:behavior_tone` is stateful, so the new statement supersedes
    // the first; the response reports the auto-superseded prior id and a
    // non-zero chain root.
    let tone_b = format!("verbose {suffix}");
    let superseding = h
        .client()
        .create_statement(&StatementCreateRequest {
            session_id: 0,
            kind: StatementKindWire::Preference,
            subject,
            predicate: PREDICATE.to_string(),
            object: StatementObjectWire::Value(StatementValueWire::Text(tone_b)),
            confidence: 0.97,
            evidence: EvidenceRefWire::Inline(Vec::new()),
            extractor_id: 0,
            valid_from_unix_nanos: now + 1,
            valid_to_unix_nanos: 0,
            event_at_unix_nanos: 0,
            schema_version: 0,
            request_id: new_id(),
            act_as: None,
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
            "§19/06 statement: created Preference {PREDICATE}, get round-trips, appears in list \
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
