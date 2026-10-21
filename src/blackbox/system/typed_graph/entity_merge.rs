//! Entity merge / unmerge scenario (§19/06:32-35 — entity duplicate
//! reconciliation, reversible within grace).
//!
//! Two entities that turn out to be the same real-world thing are folded with
//! `ENTITY_MERGE`: the merged id redirects to the survivor, and every
//! statement on the merged entity re-routes onto the survivor. The merge is
//! reversible within a grace window via `ENTITY_UNMERGE`. This scenario
//! exercises the full round trip and, in particular, the SDK's
//! `EntityGetResponse.resolved_from` redirect chain:
//!
//! 1. Create two `Person` entities A (survivor) and B (merged), each with a
//!    statement of its own.
//! 2. `ENTITY_MERGE(survivor = A, merged = B)`.
//! 3. `ENTITY_GET(B)` now redirects: `entity.entity_id == A` and
//!    `resolved_from` contains B (the redirect the GET walked through).
//! 4. B's statement is reachable via the survivor: it appears in
//!    `list_statements(subject = A)`.
//! 5. `ENTITY_UNMERGE(B)` within grace restores B: `ENTITY_GET(B)` is a direct
//!    hit again (`entity_id == B`, empty `resolved_from`) and its statement is
//!    reachable under B once more.

use std::net::SocketAddr;

use brain_db_sdk::new_id;
use brain_db_sdk::wire::types::{
    EntityCreateRequest, EntityGetRequest, EntityMergeRequest, EntityUnmergeRequest,
    EvidenceRefWire, StatementCreateRequest, StatementKindWire, StatementListRequest,
    StatementObjectWire, StatementValueWire,
};

use super::super::ScenarioOutcome;
use super::PERSON_TYPE_ID;
use crate::blackbox::system::hex16;
use crate::client::harness::{BrainEvalHarness, HarnessError};

const NAME: &str = "tg_entity_merge_unmerge";
/// A declared stateful Preference predicate (system schema). Subjects differ
/// between the two statements, so no cross-subject supersession fires.
const PREDICATE: &str = "brain:behavior_tone";

/// Create two Persons with statements, merge, assert redirect + reachability,
/// then unmerge and assert restoration.
pub async fn entity_merge_unmerge(endpoint: SocketAddr) -> ScenarioOutcome {
    match run(endpoint).await {
        Ok(o) => o,
        Err(e) => ScenarioOutcome::fail(NAME, format!("sdk error: {e}")),
    }
}

async fn run(endpoint: SocketAddr) -> Result<ScenarioOutcome, HarnessError> {
    let h = BrainEvalHarness::connect(endpoint).await?;
    let suffix = &hex16(h.space_id())[..12];

    // --- two same-type entities --------------------------------------
    let survivor = create_person(&h, &format!("Katherine Johnson {suffix}")).await?;
    let merged = create_person(&h, &format!("Kathryn Johnson {suffix}")).await?;
    if survivor == [0u8; 16] || merged == [0u8; 16] || survivor == merged {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            "failed to create two distinct Person entities to merge",
        ));
    }

    // --- one statement on each side ----------------------------------
    let _stmt_a = create_tone(&h, survivor, &format!("formal {suffix}")).await?;
    let stmt_b = create_tone(&h, merged, &format!("terse {suffix}")).await?;

    // Pre-merge: B's statement is reachable under B, not under A.
    if statement_reachable(&h, survivor, stmt_b).await? {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            "B's statement was already reachable under A before any merge",
        ));
    }
    if !statement_reachable(&h, merged, stmt_b).await? {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            "B's own statement was not reachable under B before merge",
        ));
    }

    // --- merge B into A ----------------------------------------------
    h.client()
        .merge_entities(&EntityMergeRequest {
            survivor,
            merged,
            confidence: 0.95,
            reason: "eval merge/unmerge scenario".to_string(),
            request_id: new_id(),
        })
        .await?;

    // --- ENTITY_GET(B) redirects to the survivor with B in resolved_from
    let got = h
        .client()
        .get_entity(&EntityGetRequest {
            entity_id: merged,
            act_as: None,
        })
        .await?;
    if got.entity.entity_id != survivor {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "after merge, ENTITY_GET(B) returned entity {:?}, expected survivor {:?}",
                got.entity.entity_id, survivor
            ),
        ));
    }
    if !got.resolved_from.contains(&merged) {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "ENTITY_GET(B).resolved_from {:?} does not contain the merged id {:?}",
                got.resolved_from, merged
            ),
        ));
    }

    // --- B's statement is now reachable via the survivor -------------
    if !statement_reachable(&h, survivor, stmt_b).await? {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            "after merge, B's statement was not reachable under the survivor A",
        ));
    }

    // --- unmerge within grace ----------------------------------------
    let restored = h
        .client()
        .unmerge_entity(&EntityUnmergeRequest {
            merged_entity: merged,
            request_id: new_id(),
        })
        .await?;
    if restored.restored_entity_id != merged {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "unmerge restored id {:?}, expected the merged id {:?}",
                restored.restored_entity_id, merged
            ),
        ));
    }

    // --- ENTITY_GET(B) is a direct hit again -------------------------
    let got_after = h
        .client()
        .get_entity(&EntityGetRequest {
            entity_id: merged,
            act_as: None,
        })
        .await?;
    if got_after.entity.entity_id != merged {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "after unmerge, ENTITY_GET(B) returned {:?}, expected B itself {:?}",
                got_after.entity.entity_id, merged
            ),
        ));
    }
    if !got_after.resolved_from.is_empty() {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "after unmerge, ENTITY_GET(B).resolved_from should be empty, was {:?}",
                got_after.resolved_from
            ),
        ));
    }

    // --- B's statement is reachable under B again --------------------
    let restored_reachable = statement_reachable(&h, merged, stmt_b).await?;
    let survivor_clean = !statement_reachable(&h, survivor, stmt_b).await?;

    h.close().await?;

    if !restored_reachable {
        return Ok(ScenarioOutcome::fail(
            NAME,
            "after unmerge, B's statement was not reachable under B again",
        ));
    }
    if !survivor_clean {
        return Ok(ScenarioOutcome::fail(
            NAME,
            "after unmerge, B's statement was still reachable under the survivor A",
        ));
    }

    Ok(ScenarioOutcome::pass(
        NAME,
        "§19/06 merge/unmerge: merged B into A (GET(B)→A, resolved_from∋B, \
         B's statement reachable via survivor); unmerge restored B (direct hit, \
         empty resolved_from, statement reachable under B again)",
    ))
}

/// Create a `Person` and return its id.
async fn create_person(h: &BrainEvalHarness, canonical: &str) -> Result<[u8; 16], HarnessError> {
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

/// Create a `behavior_tone` Preference on `subject` and return its id.
async fn create_tone(
    h: &BrainEvalHarness,
    subject: [u8; 16],
    tone: &str,
) -> Result<[u8; 16], HarnessError> {
    Ok(h.client()
        .create_statement(&StatementCreateRequest {
            session_id: 0,
            kind: StatementKindWire::Preference,
            subject,
            predicate: PREDICATE.to_string(),
            object: StatementObjectWire::Value(StatementValueWire::Text(tone.to_string())),
            confidence: 0.95,
            evidence: EvidenceRefWire::Inline(Vec::new()),
            extractor_id: 0,
            valid_from_unix_nanos: unix_nanos_now(),
            valid_to_unix_nanos: 0,
            event_at_unix_nanos: 0,
            schema_version: 0,
            request_id: new_id(),
            act_as: None,
        })
        .await?
        .statement_id)
}

/// Whether `stmt` appears in `list_statements(subject = subject)` — the test
/// of reachability via a given (possibly survivor) entity.
async fn statement_reachable(
    h: &BrainEvalHarness,
    subject: [u8; 16],
    stmt: [u8; 16],
) -> Result<bool, HarnessError> {
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
    Ok(listed.iter().any(|s| s.statement_id == stmt))
}

fn unix_nanos_now() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}
