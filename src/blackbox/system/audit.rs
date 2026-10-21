//! Audit-log query scenario ("E7", §19/06:133 — *all derivations logged;
//! queryable*).
//!
//! Brain writes a per-call **extraction-audit** row every time the write-path
//! extractor pipeline runs over a memory, and a per-mention **resolution-audit**
//! row every time the entity resolver decides a candidate. Both are readable
//! through the operator admin plane (`GET /v1/audit`), which the data-plane SDK
//! does not surface — so this scenario drives it with the bare-HTTP
//! [`AdminClient`].
//!
//! The flow: ENCODE one memory whose prose carries clear proper nouns (a
//! person + an organization + a place), so the always-on extractor pipeline
//! derives a typed graph and the resolver resolves each mention. Extraction is
//! asynchronous (a background worker), so the scenario polls the audit plane
//! until the derivation lands, then asserts:
//!
//! - `GET /v1/audit?by=memory&memory=<hex>` returns extraction-audit rows for
//!   that exact memory (the derivation was logged and is queryable by memory).
//! - `GET /v1/audit?by=time` returns a well-formed extraction page (queryable
//!   by time window).
//! - `GET /v1/audit?by=resolution` returns resolution-audit rows (per-mention
//!   resolution decisions were logged and are queryable).

use std::net::SocketAddr;
use std::time::Duration;

use brain_db_sdk::EncodeBuilder;

use super::ScenarioOutcome;
use crate::client::admin::{AdminClient, AdminError};
use crate::client::harness::{BrainEvalHarness, HarnessError};

const NAME: &str = "audit_derivation_logged";

/// Max time to wait for asynchronous extraction + resolution to land in the
/// audit tables. Extraction runs on a background worker and, on a CPU-only
/// box, an LLM tier can take many seconds.
const POLL_TIMEOUT: Duration = Duration::from_secs(120);
/// Delay between audit polls.
const POLL_INTERVAL: Duration = Duration::from_secs(3);

/// Errors this scenario can surface before it can produce a pass/fail.
#[derive(Debug, thiserror::Error)]
enum AuditScenarioError {
    #[error("harness: {0}")]
    Harness(#[from] HarnessError),
    #[error("admin: {0}")]
    Admin(#[from] AdminError),
    #[error("sdk: {0}")]
    Sdk(#[from] brain_db_sdk::BrainError),
}

/// Ingest a memory, then assert its extraction + resolution derivations are
/// logged and queryable through `GET /v1/audit`.
pub async fn audit_derivation_logged(endpoint: SocketAddr) -> ScenarioOutcome {
    match run(endpoint).await {
        Ok(o) => o,
        Err(e) => ScenarioOutcome::fail(NAME, format!("{e}")),
    }
}

async fn run(endpoint: SocketAddr) -> Result<ScenarioOutcome, AuditScenarioError> {
    let admin = AdminClient::from_external_env(endpoint);
    if !admin.has_secret() {
        return Ok(ScenarioOutcome::fail(
            NAME,
            "no admin secret configured (set BRAIN_EVAL_ADMIN_SECRET) — \
             cannot query /v1/audit",
        ));
    }

    let h = BrainEvalHarness::connect(endpoint).await?;

    // Prose with unambiguous proper nouns so the always-on extractor pipeline
    // derives entities/statements and the resolver resolves each mention.
    let text = "Grace Hopper works at the United States Navy in Arlington.";
    let enc = h.client().encode(&EncodeBuilder::new(text).build()).await?;
    let memory_hex = format!("{:032x}", enc.memory_id);

    // Poll until the memory's extraction-audit rows appear (extraction is
    // asynchronous). Once they do, the resolution rows have been written in
    // the same pipeline pass.
    let deadline = std::time::Instant::now() + POLL_TIMEOUT;
    let extraction_rows = loop {
        let resp = admin
            .get(&format!("/v1/audit?by=memory&memory={memory_hex}"))
            .await?;
        if resp.status != 200 {
            h.close().await?;
            return Ok(ScenarioOutcome::fail(
                NAME,
                format!(
                    "GET /v1/audit?by=memory returned {} — {}",
                    resp.status,
                    snippet(&resp.body)
                ),
            ));
        }
        let n = match count_rows(&resp.body, "extraction") {
            Ok(n) => n,
            Err(e) => {
                h.close().await?;
                return Ok(ScenarioOutcome::fail(
                    NAME,
                    format!("by=memory body was not a valid audit page: {e}"),
                ));
            }
        };
        if n > 0 || std::time::Instant::now() >= deadline {
            break n;
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    };

    if extraction_rows == 0 {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "no extraction-audit rows for memory {memory_hex} after {}s — \
                 derivation not logged",
                POLL_TIMEOUT.as_secs()
            ),
        ));
    }

    // by=time — the extraction-audit tables are queryable by time window.
    let by_time = admin.get("/v1/audit?by=time&since=0").await?;
    if by_time.status != 200 {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "GET /v1/audit?by=time returned {} — {}",
                by_time.status,
                snippet(&by_time.body)
            ),
        ));
    }
    if let Err(e) = count_rows(&by_time.body, "extraction") {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!("by=time body was not a valid audit page: {e}"),
        ));
    }

    // by=resolution — per-mention resolution decisions are logged + queryable.
    // Resolution rows land in the same async pass; poll briefly in case they
    // trail the extraction rows.
    let res_deadline = std::time::Instant::now() + POLL_INTERVAL * 4;
    let resolution_rows = loop {
        let resp = admin.get("/v1/audit?by=resolution&since=0").await?;
        if resp.status != 200 {
            h.close().await?;
            return Ok(ScenarioOutcome::fail(
                NAME,
                format!(
                    "GET /v1/audit?by=resolution returned {} — {}",
                    resp.status,
                    snippet(&resp.body)
                ),
            ));
        }
        let n = match count_rows(&resp.body, "resolution") {
            Ok(n) => n,
            Err(e) => {
                h.close().await?;
                return Ok(ScenarioOutcome::fail(
                    NAME,
                    format!("by=resolution body was not a valid audit page: {e}"),
                ));
            }
        };
        if n > 0 || std::time::Instant::now() >= res_deadline {
            break n;
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    };

    h.close().await?;

    if resolution_rows == 0 {
        return Ok(ScenarioOutcome::fail(
            NAME,
            "no resolution-audit rows queryable via by=resolution — \
             per-mention resolution decisions not logged",
        ));
    }

    Ok(ScenarioOutcome::pass(
        NAME,
        format!(
            "§19/06 audit: {extraction_rows} extraction-audit row(s) for the ingested memory \
             (by=memory), by=time queryable, {resolution_rows} resolution-audit row(s) \
             (by=resolution) — all derivations logged; queryable"
        ),
    ))
}

/// Parse a `GET /v1/audit` page and return its row count, checking the
/// envelope `kind` matches `expected_kind`. The server serializes the page as
/// `{"kind":<str>,"rows":[...],"next_cursor":<str|null>}`.
fn count_rows(body: &str, expected_kind: &str) -> Result<usize, String> {
    let v: serde_json::Value =
        serde_json::from_str(body).map_err(|e| format!("not JSON: {e}; body={}", snippet(body)))?;
    let kind = v
        .get("kind")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "missing string field \"kind\"".to_string())?;
    if kind != expected_kind {
        return Err(format!("kind={kind:?}, expected {expected_kind:?}"));
    }
    let rows = v
        .get("rows")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "missing array field \"rows\"".to_string())?;
    Ok(rows.len())
}

/// First 200 chars of a body, for error messages.
fn snippet(s: &str) -> String {
    s.chars().take(200).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_rows_reads_extraction_page() {
        let body = r#"{"kind":"extraction","rows":[{"audit_id":"aa"},{"audit_id":"bb"}],"next_cursor":null}"#;
        assert_eq!(count_rows(body, "extraction").unwrap(), 2);
    }

    #[test]
    fn count_rows_empty_page_is_zero() {
        let body = r#"{"kind":"resolution","rows":[],"next_cursor":null}"#;
        assert_eq!(count_rows(body, "resolution").unwrap(), 0);
    }

    #[test]
    fn count_rows_rejects_kind_mismatch() {
        let body = r#"{"kind":"extraction","rows":[],"next_cursor":null}"#;
        assert!(count_rows(body, "resolution").is_err());
    }

    #[test]
    fn count_rows_rejects_non_json() {
        assert!(count_rows("shard out of range", "extraction").is_err());
    }
}
