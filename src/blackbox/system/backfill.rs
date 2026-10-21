//! Resumable-backfill scenario ("E7" — *the extraction backfill is
//! startable, observable, and cancellable through the operator admin plane*).
//!
//! Brain's resumable extraction **BackfillWorker** re-runs the write-path
//! extractor tiers over already-stored memories (e.g. after a new extractor is
//! registered), checkpointing its progress so it can resume. It is driven from
//! the operator admin HTTP plane — never a wire opcode — so this scenario
//! drives it with the bare-HTTP [`AdminClient`], exactly like the audit and
//! rebuild scenarios:
//!
//! - `POST /v1/backfill?all&extractors=<id>` starts a backfill across every
//!   shard, returning a 32-hex `backfill_id` and a per-shard `progress` array.
//! - `GET /v1/backfill` reports current per-shard progress (one entry per
//!   shard).
//! - `DELETE /v1/backfill/<id>` cancels the run, echoing the id and a
//!   per-shard `cancelled` array.
//!
//! Two negative checks assert the route validates its inputs: a start with no
//! `extractors` list is a `400`, and a cancel of a non-hex id is a `400`.

use std::net::SocketAddr;

use brain_db_sdk::EncodeBuilder;

use super::{hex16, ScenarioOutcome};
use crate::client::admin::{AdminClient, AdminError};
use crate::client::harness::{BrainEvalHarness, HarnessError};

const NAME: &str = "resumable_backfill";

/// The extractor id to backfill. The admin route requires a comma-separated
/// list of *registered* extractor tier ids; the eval repo has no authoritative
/// id registry to discover them from (server-side extractor metrics are keyed
/// by tier *name*, not numeric id), so this uses the smallest id. It MUST
/// match a registered extractor on the server under test — if the server
/// renumbers its extractor registry, update this constant.
const EXTRACTOR_ID: u32 = 1;

#[derive(Debug, thiserror::Error)]
enum BackfillScenarioError {
    #[error("harness: {0}")]
    Harness(#[from] HarnessError),
    #[error("admin: {0}")]
    Admin(#[from] AdminError),
    #[error("sdk: {0}")]
    Sdk(#[from] brain_db_sdk::BrainError),
}

/// Seed a few memories, then start / observe / cancel an extraction backfill
/// through the admin plane and assert the route's input validation.
pub async fn resumable_backfill(endpoint: SocketAddr) -> ScenarioOutcome {
    match run(endpoint).await {
        Ok(o) => o,
        Err(e) => ScenarioOutcome::fail(NAME, format!("{e}")),
    }
}

async fn run(endpoint: SocketAddr) -> Result<ScenarioOutcome, BackfillScenarioError> {
    let admin = AdminClient::from_external_env(endpoint);
    if !admin.has_secret() {
        return Ok(ScenarioOutcome::fail(
            NAME,
            "no admin secret configured (set BRAIN_EVAL_ADMIN_SECRET) — \
             cannot drive /v1/backfill",
        ));
    }

    let h = BrainEvalHarness::connect(endpoint).await?;
    let marker = format!("backfill-marker-{}", hex16(h.space_id()));

    // Seed a handful of memories so the backfill has stored content to
    // re-extract over — proper nouns so the extractor tiers have real work.
    let seeds = [
        format!("{marker}: Alan Turing worked at Bletchley Park."),
        format!("{marker}: the retrospective is scheduled for Friday."),
        format!("{marker}: Globex Corporation opened an office in Berlin."),
    ];
    for s in &seeds {
        h.client()
            .encode(&EncodeBuilder::new(s.as_str()).build())
            .await?;
    }

    // Start a backfill across every shard for the chosen extractor.
    let start = admin
        .post_empty(&format!("/v1/backfill?all&extractors={EXTRACTOR_ID}"))
        .await?;
    if !start.is_success() {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "POST /v1/backfill?all&extractors={EXTRACTOR_ID} returned {} — {}",
                start.status,
                snippet(&start.body)
            ),
        ));
    }
    let (backfill_id, start_progress) = match parse_backfill_start(&start.body) {
        Ok(v) => v,
        Err(e) => {
            h.close().await?;
            return Ok(ScenarioOutcome::fail(
                NAME,
                format!("start body malformed: {e}"),
            ));
        }
    };

    // Observe current per-shard progress.
    let list = admin.get("/v1/backfill").await?;
    if list.status != 200 {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "GET /v1/backfill returned {} — {}",
                list.status,
                snippet(&list.body)
            ),
        ));
    }
    let (list_shards, list_progress) = match parse_backfill_list(&list.body) {
        Ok(v) => v,
        Err(e) => {
            h.close().await?;
            return Ok(ScenarioOutcome::fail(
                NAME,
                format!("list body malformed: {e}"),
            ));
        }
    };

    // Cancel the run; the response must echo the id and a per-shard array.
    let cancel = admin.delete(&format!("/v1/backfill/{backfill_id}")).await?;
    if cancel.status != 200 {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "DELETE /v1/backfill/{backfill_id} returned {} — {}",
                cancel.status,
                snippet(&cancel.body)
            ),
        ));
    }
    let cancel_shards = match parse_backfill_cancel(&cancel.body, &backfill_id) {
        Ok(n) => n,
        Err(e) => {
            h.close().await?;
            return Ok(ScenarioOutcome::fail(
                NAME,
                format!("cancel body malformed: {e}"),
            ));
        }
    };

    // Negative: a start with no `extractors` list is a client error (400).
    let no_extractors = admin.post_empty("/v1/backfill?all").await?;
    if no_extractors.status != 400 {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "POST /v1/backfill?all (no extractors) returned {} — expected 400",
                no_extractors.status
            ),
        ));
    }

    // Negative: a cancel of a non-hex id is a client error (400).
    let bad_id = admin.delete("/v1/backfill/not-hex").await?;
    h.close().await?;
    if bad_id.status != 400 {
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "DELETE /v1/backfill/not-hex returned {} — expected 400",
                bad_id.status
            ),
        ));
    }

    Ok(ScenarioOutcome::pass(
        NAME,
        format!(
            "backfill started (id {backfill_id}, {start_progress} shard(s) of progress), \
             GET reported {list_shards} shard(s)/{list_progress} progress entr(ies), \
             DELETE cancelled across {cancel_shards} shard(s); \
             missing-extractors and non-hex-id both rejected 400"
        ),
    ))
}

/// Every field the documented per-shard progress entry carries.
fn assert_progress_fields(entry: &serde_json::Value) -> Result<(), String> {
    let obj = entry
        .as_object()
        .ok_or_else(|| "progress entry is not an object".to_string())?;
    // Numeric-or-required fields: `shard` and the three counters are always
    // present integers; the three nullable fields must at least be present.
    for f in ["shard", "completed", "failed", "skipped_already_completed"] {
        match obj.get(f).and_then(serde_json::Value::as_u64) {
            Some(_) => {}
            None => return Err(format!("progress entry missing numeric field {f:?}")),
        }
    }
    if obj
        .get("running")
        .and_then(serde_json::Value::as_bool)
        .is_none()
    {
        return Err("progress entry missing bool field \"running\"".to_string());
    }
    // Nullable fields must be present (value may be null or numeric).
    for f in ["request_id", "last_processed_memory_id", "eta_secs"] {
        if !obj.contains_key(f) {
            return Err(format!("progress entry missing field {f:?}"));
        }
    }
    Ok(())
}

/// Parse a `POST /v1/backfill` start response. Returns the 32-hex
/// `backfill_id` and the number of per-shard progress entries. Errors if the
/// id is not 32 hex chars or the progress array is empty / malformed.
fn parse_backfill_start(body: &str) -> Result<(String, usize), String> {
    let v: serde_json::Value =
        serde_json::from_str(body).map_err(|e| format!("not JSON: {e}; body={}", snippet(body)))?;
    let id = v
        .get("backfill_id")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "missing string field \"backfill_id\"".to_string())?;
    if id.len() != 32 || !id.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("backfill_id {id:?} is not 32 hex chars"));
    }
    let progress = v
        .get("progress")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "missing array field \"progress\"".to_string())?;
    if progress.is_empty() {
        return Err("progress array is empty".to_string());
    }
    for entry in progress {
        assert_progress_fields(entry)?;
    }
    Ok((id.to_string(), progress.len()))
}

/// Parse a `GET /v1/backfill` list response. Returns the declared `shards`
/// count and the number of per-shard progress entries, checking the two agree.
fn parse_backfill_list(body: &str) -> Result<(u64, usize), String> {
    let v: serde_json::Value =
        serde_json::from_str(body).map_err(|e| format!("not JSON: {e}; body={}", snippet(body)))?;
    let shards = v
        .get("shards")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| "missing numeric field \"shards\"".to_string())?;
    let progress = v
        .get("progress")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "missing array field \"progress\"".to_string())?;
    for entry in progress {
        assert_progress_fields(entry)?;
    }
    if progress.len() as u64 != shards {
        return Err(format!(
            "progress array length {} != shards {shards}",
            progress.len()
        ));
    }
    Ok((shards, progress.len()))
}

/// Parse a `DELETE /v1/backfill/<id>` cancel response. Checks the `backfill_id`
/// echoes `expected_id` and the `cancelled` array is present (each entry a
/// `{shard, cancelled}` object); returns the number of per-shard entries.
fn parse_backfill_cancel(body: &str, expected_id: &str) -> Result<usize, String> {
    let v: serde_json::Value =
        serde_json::from_str(body).map_err(|e| format!("not JSON: {e}; body={}", snippet(body)))?;
    match v.get("backfill_id").and_then(serde_json::Value::as_str) {
        Some(got) if got.eq_ignore_ascii_case(expected_id) => {}
        other => {
            return Err(format!(
                "backfill_id {other:?} != requested {expected_id:?}"
            ))
        }
    }
    // `any_cancelled` is a documented bool; it must be present.
    if v.get("any_cancelled")
        .and_then(serde_json::Value::as_bool)
        .is_none()
    {
        return Err("missing bool field \"any_cancelled\"".to_string());
    }
    let cancelled = v
        .get("cancelled")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "missing array field \"cancelled\"".to_string())?;
    for entry in cancelled {
        let obj = entry
            .as_object()
            .ok_or_else(|| "cancelled entry is not an object".to_string())?;
        if obj
            .get("shard")
            .and_then(serde_json::Value::as_u64)
            .is_none()
        {
            return Err("cancelled entry missing numeric field \"shard\"".to_string());
        }
        if obj
            .get("cancelled")
            .and_then(serde_json::Value::as_bool)
            .is_none()
        {
            return Err("cancelled entry missing bool field \"cancelled\"".to_string());
        }
    }
    Ok(cancelled.len())
}

/// First 200 chars of a body, for error messages.
fn snippet(s: &str) -> String {
    s.chars().take(200).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const START_OK: &str = r#"{
        "backfill_id":"0123456789abcdef0123456789abcdef",
        "shards":2,
        "progress":[
            {"shard":0,"request_id":42,"running":true,"completed":3,"failed":0,
             "skipped_already_completed":1,"last_processed_memory_id":99,"eta_secs":1.5},
            {"shard":1,"request_id":null,"running":false,"completed":0,"failed":0,
             "skipped_already_completed":0,"last_processed_memory_id":null,"eta_secs":null}
        ]
    }"#;

    #[test]
    fn parse_start_reads_id_and_progress() {
        let (id, n) = parse_backfill_start(START_OK).unwrap();
        assert_eq!(id, "0123456789abcdef0123456789abcdef");
        assert_eq!(n, 2);
    }

    #[test]
    fn parse_start_rejects_short_id() {
        let body = r#"{"backfill_id":"abcd","progress":[]}"#;
        assert!(parse_backfill_start(body).is_err());
    }

    #[test]
    fn parse_start_rejects_non_hex_id() {
        let body = r#"{"backfill_id":"zzzz56789abcdef0123456789abcdef0","progress":[
            {"shard":0,"request_id":null,"running":false,"completed":0,"failed":0,
             "skipped_already_completed":0,"last_processed_memory_id":null,"eta_secs":null}]}"#;
        assert!(parse_backfill_start(body).is_err());
    }

    #[test]
    fn parse_start_rejects_empty_progress() {
        let body = r#"{"backfill_id":"0123456789abcdef0123456789abcdef","progress":[]}"#;
        assert!(parse_backfill_start(body).is_err());
    }

    #[test]
    fn parse_start_rejects_missing_progress_field() {
        // Missing `running` on the entry.
        let body = r#"{"backfill_id":"0123456789abcdef0123456789abcdef","progress":[
            {"shard":0,"request_id":null,"completed":0,"failed":0,
             "skipped_already_completed":0,"last_processed_memory_id":null,"eta_secs":null}]}"#;
        assert!(parse_backfill_start(body).is_err());
    }

    #[test]
    fn parse_start_rejects_non_json() {
        assert!(parse_backfill_start("shard error").is_err());
    }

    #[test]
    fn parse_list_reads_shards_and_progress() {
        let body = r#"{"shards":2,"progress":[
            {"shard":0,"request_id":null,"running":false,"completed":0,"failed":0,
             "skipped_already_completed":0,"last_processed_memory_id":null,"eta_secs":null},
            {"shard":1,"request_id":null,"running":false,"completed":0,"failed":0,
             "skipped_already_completed":0,"last_processed_memory_id":null,"eta_secs":null}]}"#;
        let (shards, n) = parse_backfill_list(body).unwrap();
        assert_eq!(shards, 2);
        assert_eq!(n, 2);
    }

    #[test]
    fn parse_list_rejects_length_mismatch() {
        let body = r#"{"shards":3,"progress":[
            {"shard":0,"request_id":null,"running":false,"completed":0,"failed":0,
             "skipped_already_completed":0,"last_processed_memory_id":null,"eta_secs":null}]}"#;
        assert!(parse_backfill_list(body).is_err());
    }

    #[test]
    fn parse_cancel_reads_shards() {
        let body = r#"{"backfill_id":"0123456789ABCDEF0123456789abcdef","any_cancelled":true,
            "cancelled":[{"shard":0,"cancelled":true},{"shard":1,"cancelled":false}]}"#;
        // Echo check is case-insensitive.
        let n = parse_backfill_cancel(body, "0123456789abcdef0123456789abcdef").unwrap();
        assert_eq!(n, 2);
    }

    #[test]
    fn parse_cancel_rejects_id_mismatch() {
        let body = r#"{"backfill_id":"ffffffffffffffffffffffffffffffff","any_cancelled":false,
            "cancelled":[]}"#;
        assert!(parse_backfill_cancel(body, "0123456789abcdef0123456789abcdef").is_err());
    }

    #[test]
    fn parse_cancel_rejects_missing_cancelled_array() {
        let body = r#"{"backfill_id":"0123456789abcdef0123456789abcdef","any_cancelled":false}"#;
        assert!(parse_backfill_cancel(body, "0123456789abcdef0123456789abcdef").is_err());
    }
}
