//! Provenance & versioning scenario.
//!
//! The lifecycle scenario ([`super::statement`]) checks a created statement's
//! id / subject / predicate. This gates the provenance and versioning
//! invariants a fresh statement must carry, which nothing else asserts
//! (§19/06 "provenance & versioning"):
//!
//! - **Server-stamped time** — `extracted_at_unix_nanos > 0`: the write path
//!   records when the claim entered, it isn't left zero.
//! - **Version starts at 1** — a brand-new statement is `version == 1`, not 0
//!   and not some inherited number.
//! - **No phantom lineage** — a fresh statement is not tombstoned, has no
//!   `superseded_by` / `supersedes` links, and `get_statement` returns it
//!   directly (`returned_via_supersession == false`): version metadata means
//!   what it says on a first write.
//!
//! Deliberately not asserted: the exact `evidence` bytes (evidence refs are a
//! separate id space from the `u128` memory id, and whether the server echoes
//! or normalises inline refs is not a stable client contract), nor the
//! confidence-aggregation formula or stale-schema flagging (server-internal
//! behaviour that needs a live run to pin down, not a blind gate).

use std::net::SocketAddr;

use brain_db_sdk::new_id;
use brain_db_sdk::wire::types::{
    EntityCreateRequest, EvidenceRefWire, StatementCreateRequest, StatementGetRequest,
    StatementKindWire, StatementObjectWire, StatementValueWire,
};

use super::super::ScenarioOutcome;
use super::PERSON_TYPE_ID;
use crate::blackbox::system::hex16;
use crate::client::harness::{BrainEvalHarness, HarnessError};

const NAME: &str = "tg_provenance_versioning";
const PREDICATE: &str = "brain:behavior_tone";

/// A fresh statement must carry consistent provenance + version-1 metadata.
pub async fn provenance_versioning(endpoint: SocketAddr) -> ScenarioOutcome {
    match run(endpoint).await {
        Ok(o) => o,
        Err(e) => ScenarioOutcome::fail(NAME, format!("sdk error: {e}")),
    }
}

async fn run(endpoint: SocketAddr) -> Result<ScenarioOutcome, HarnessError> {
    let h = BrainEvalHarness::connect(endpoint).await?;
    let suffix = &hex16(h.space_id())[..12];
    let now = unix_nanos_now();

    let subject = h
        .client()
        .create_entity(&EntityCreateRequest {
            session_id: 0,
            entity_type_id: PERSON_TYPE_ID,
            canonical_name: format!("Provenance Subject {suffix}"),
            aliases: Vec::new(),
            attributes_blob: Vec::new(),
            request_id: new_id(),
            act_as: None,
        })
        .await?
        .entity_id;

    let stmt_id = h
        .client()
        .create_statement(&StatementCreateRequest {
            session_id: 0,
            kind: StatementKindWire::Preference,
            subject,
            predicate: PREDICATE.to_string(),
            object: StatementObjectWire::Value(StatementValueWire::Text(format!("even {suffix}"))),
            confidence: 0.9,
            evidence: EvidenceRefWire::Inline(Vec::new()),
            extractor_id: 0,
            valid_from_unix_nanos: now,
            valid_to_unix_nanos: 0,
            event_at_unix_nanos: 0,
            schema_version: 0,
            request_id: new_id(),
            act_as: None,
        })
        .await?
        .statement_id;

    let got = h
        .client()
        .get_statement(&StatementGetRequest {
            statement_id: stmt_id,
            follow_supersession: true,
            act_as: None,
        })
        .await?;
    let s = &got.statement;

    // Collect every violated invariant so the failure detail names them all.
    let mut problems = Vec::new();
    if s.extracted_at_unix_nanos == 0 {
        problems.push("extracted_at is 0 (write time not stamped)".to_string());
    }
    if s.version != 1 {
        problems.push(format!(
            "version is {} (fresh statement must be 1)",
            s.version
        ));
    }
    if s.tombstoned {
        problems.push("fresh statement is already tombstoned".to_string());
    }
    if s.superseded_by != [0u8; 16] {
        problems.push("fresh statement already has a superseded_by link".to_string());
    }
    if s.supersedes != [0u8; 16] {
        problems.push("fresh statement already has a supersedes link".to_string());
    }
    if got.returned_via_supersession {
        problems
            .push("get_statement redirected via supersession for a fresh statement".to_string());
    }

    h.close().await?;

    if !problems.is_empty() {
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "provenance/version invariants violated: {}",
                problems.join("; ")
            ),
        ));
    }

    Ok(ScenarioOutcome::pass(
        NAME,
        format!(
            "§19/06 provenance: fresh statement stamped extracted_at={}, version=1, \
             not tombstoned, no supersession links, direct (non-redirected) get",
            s.extracted_at_unix_nanos
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
