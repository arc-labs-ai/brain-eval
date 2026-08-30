//! Derived-index rebuild scenario ("E7", §19/06:129 — *derived indexes are
//! rebuildable from authoritative state*).
//!
//! Every ANN / lexical index Brain serves reads from is *derived*: it can be
//! discarded and reconstructed from the redb tables and the arena, which are
//! the authoritative record. The operator admin plane exposes this as
//! `POST /v1/rebuild?index=<target>&shard=N` (plus the `POST /v1/rebuild-ann`
//! back-compat alias for the memory HNSW). This scenario asserts each derived
//! index rebuilds with a `201` + a sane report, across every shard, and that
//! recall still answers afterward — the rebuild is non-destructive.
//!
//! Targets exercised: `memory_hnsw`, `entity_hnsw`, `hype_hnsw`,
//! `statement_question_hnsw`, and the `rebuild-ann` alias.

use std::net::SocketAddr;

use brain_db_sdk::EncodeBuilder;

use super::{hex16, ScenarioOutcome};
use crate::client::admin::{AdminClient, AdminError, AdminResponse};
use crate::client::harness::{BrainEvalHarness, HarnessError};

const NAME: &str = "index_rebuild";

/// The derived-index targets `POST /v1/rebuild?index=` accepts (excluding
/// `all`, which the per-target loop already covers).
const INDEX_TARGETS: &[&str] = &[
    "memory_hnsw",
    "entity_hnsw",
    "hype_hnsw",
    "statement_question_hnsw",
];

#[derive(Debug, thiserror::Error)]
enum RebuildScenarioError {
    #[error("harness: {0}")]
    Harness(#[from] HarnessError),
    #[error("admin: {0}")]
    Admin(#[from] AdminError),
    #[error("sdk: {0}")]
    Sdk(#[from] brain_db_sdk::BrainError),
}

/// Ingest, rebuild every derived index across every shard, and assert recall
/// still answers.
pub async fn index_rebuild(endpoint: SocketAddr) -> ScenarioOutcome {
    match run(endpoint).await {
        Ok(o) => o,
        Err(e) => ScenarioOutcome::fail(NAME, format!("{e}")),
    }
}

async fn run(endpoint: SocketAddr) -> Result<ScenarioOutcome, RebuildScenarioError> {
    let admin = AdminClient::from_external_env(endpoint);
    if !admin.has_secret() {
        return Ok(ScenarioOutcome::fail(
            NAME,
            "no admin secret configured (set BRAIN_EVAL_ADMIN_SECRET) — \
             cannot POST /v1/rebuild",
        ));
    }

    let h = BrainEvalHarness::connect(endpoint).await?;
    let marker = format!("rebuild-marker-{}", hex16(h.space_id()));

    // A handful of memories carrying proper nouns so entity/statement/HyPE
    // indexes have something to rebuild, not just the memory HNSW.
    let seeds = [
        format!("{marker}: Marie Curie discovered polonium in Paris."),
        format!("{marker}: the launch review is scheduled for next Tuesday."),
        format!("{marker}: Acme Robotics acquired a lidar startup last year."),
    ];
    for s in &seeds {
        h.client()
            .encode(&EncodeBuilder::new(s.as_str()).build())
            .await?;
    }

    // Baseline: the marker is recallable before any rebuild.
    let before = h.recall(&marker, 10).await?;
    if before.memories.is_empty() {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            "marker not recallable before rebuild — cannot attribute a post-rebuild miss",
        ));
    }

    // Rebuild the memory HNSW via the back-compat alias across every shard,
    // discovering the shard count as we go (a 404 = shard out of range).
    let mut shard_count = 0usize;
    loop {
        match rebuild_shard(&admin, "/v1/rebuild-ann", shard_count).await? {
            Some(resp) => {
                if resp.status != 201 {
                    h.close().await?;
                    return Ok(ScenarioOutcome::fail(
                        NAME,
                        format!(
                            "POST /v1/rebuild-ann?shard={shard_count} returned {} — {}",
                            resp.status,
                            snippet(&resp.body)
                        ),
                    ));
                }
                if let Err(e) = check_report(&resp.body, None) {
                    h.close().await?;
                    return Ok(ScenarioOutcome::fail(
                        NAME,
                        format!("rebuild-ann shard {shard_count} report unsane: {e}"),
                    ));
                }
                shard_count += 1;
            }
            None => break,
        }
    }
    if shard_count == 0 {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            "no shards accepted /v1/rebuild-ann?shard=0 (404) — server exposes no shard 0",
        ));
    }

    // Rebuild every derived index target across every shard.
    let mut rebuilt = 0usize;
    for target in INDEX_TARGETS {
        for shard in 0..shard_count {
            let path = format!("/v1/rebuild?index={target}");
            let Some(resp) = rebuild_shard(&admin, &path, shard).await? else {
                // Shard count is stable within a run; a 404 here is unexpected.
                h.close().await?;
                return Ok(ScenarioOutcome::fail(
                    NAME,
                    format!("/v1/rebuild?index={target}&shard={shard} 404 — shard vanished"),
                ));
            };
            if resp.status != 201 {
                h.close().await?;
                return Ok(ScenarioOutcome::fail(
                    NAME,
                    format!(
                        "POST /v1/rebuild?index={target}&shard={shard} returned {} — {}",
                        resp.status,
                        snippet(&resp.body)
                    ),
                ));
            }
            if let Err(e) = check_report(&resp.body, Some(target)) {
                h.close().await?;
                return Ok(ScenarioOutcome::fail(
                    NAME,
                    format!("/v1/rebuild?index={target}&shard={shard} report unsane: {e}"),
                ));
            }
            rebuilt += 1;
        }
    }

    // Recall must still answer after the derived indexes were rebuilt.
    let after = h.recall(&marker, 10).await?;
    h.close().await?;

    if after.memories.is_empty() {
        return Ok(ScenarioOutcome::fail(
            NAME,
            "marker no longer recallable after rebuild — rebuild was not faithful",
        ));
    }

    Ok(ScenarioOutcome::pass(
        NAME,
        format!(
            "§19/06 rebuild: rebuilt-ann on {shard_count} shard(s) + {rebuilt} \
             index-rebuild(s) across {targets} targets, all 201 with sane reports; \
             recall still answers after rebuild",
            targets = INDEX_TARGETS.len(),
        ),
    ))
}

/// POST a rebuild for one shard. `Ok(None)` when the server answers `404`
/// (shard out of range) — the caller uses that to stop the shard walk.
async fn rebuild_shard(
    admin: &AdminClient,
    path: &str,
    shard: usize,
) -> Result<Option<AdminResponse>, AdminError> {
    let sep = if path.contains('?') { '&' } else { '?' };
    let resp = admin
        .post_empty(&format!("{path}{sep}shard={shard}"))
        .await?;
    if resp.status == 404 {
        return Ok(None);
    }
    Ok(Some(resp))
}

/// Validate a rebuild report JSON: numeric `entries` + `elapsed_ms`, and — for
/// the `/v1/rebuild?index=` route — an `index` echoing the requested target.
fn check_report(body: &str, expect_index: Option<&str>) -> Result<(), String> {
    let v: serde_json::Value =
        serde_json::from_str(body).map_err(|e| format!("not JSON: {e}; body={}", snippet(body)))?;
    if v.get("entries")
        .and_then(serde_json::Value::as_u64)
        .is_none()
    {
        return Err(format!(
            "missing numeric \"entries\"; body={}",
            snippet(body)
        ));
    }
    if v.get("elapsed_ms")
        .and_then(serde_json::Value::as_u64)
        .is_none()
    {
        return Err(format!(
            "missing numeric \"elapsed_ms\"; body={}",
            snippet(body)
        ));
    }
    if let Some(idx) = expect_index {
        match v.get("index").and_then(serde_json::Value::as_str) {
            Some(got) if got == idx => {}
            other => return Err(format!("index field {other:?} != requested {idx:?}")),
        }
    }
    Ok(())
}

/// First 200 chars of a body, for error messages.
fn snippet(s: &str) -> String {
    s.chars().take(200).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_report_accepts_ann_report() {
        assert!(check_report(r#"{"entries":42,"elapsed_ms":7,"shard":0}"#, None).is_ok());
    }

    #[test]
    fn check_report_accepts_index_report() {
        assert!(check_report(
            r#"{"index":"entity_hnsw","entries":3,"elapsed_ms":1,"shard":0}"#,
            Some("entity_hnsw")
        )
        .is_ok());
    }

    #[test]
    fn check_report_rejects_index_mismatch() {
        assert!(check_report(
            r#"{"index":"memory_hnsw","entries":3,"elapsed_ms":1,"shard":0}"#,
            Some("entity_hnsw")
        )
        .is_err());
    }

    #[test]
    fn check_report_rejects_missing_entries() {
        assert!(check_report(r#"{"elapsed_ms":7,"shard":0}"#, None).is_err());
    }
}
