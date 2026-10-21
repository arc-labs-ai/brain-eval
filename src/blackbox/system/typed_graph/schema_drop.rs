//! Schema-drop scenario: upload → drop a declared predicate → re-drop no-op.
//!
//! The surgical counterpart to the `schema_lifecycle` scenario. Exercises
//! §19/06 "schema lifecycle" from the narrowing side: a user namespace is
//! uploaded with a predicate + relation_type, then one declared predicate is
//! removed with `SCHEMA_DROP`. With no live rows on the target, the drop needs
//! no `force`, reports `dropped = true` / `live_rows = 0`, and bumps the
//! namespace to a version above the uploaded one. A second drop of the now-gone
//! predicate is a no-op — `dropped = false`, and it reports the current active
//! `schema_version` unchanged (the version only advances when a drop actually
//! narrows the document) — mirroring the valid/invalid dual of the lifecycle
//! scenario.

use std::net::SocketAddr;

use brain_db_sdk::new_id;
use brain_db_sdk::wire::types::{
    schema_drop_target, SchemaDropRequest, SchemaGetRequest, SchemaUploadRequest,
};

use super::super::ScenarioOutcome;
use crate::blackbox::system::hex16;
use crate::client::harness::{BrainEvalHarness, HarnessError};

const NAME: &str = "tg_schema_drop";

/// The declared predicate the scenario narrows away.
const PREDICATE_NAME: &str = "built_by";

/// Upload a user schema, drop one declared predicate, then confirm a second
/// drop of the same (now-absent) predicate is a no-op.
pub async fn schema_drop(endpoint: SocketAddr) -> ScenarioOutcome {
    match run(endpoint).await {
        Ok(o) => o,
        Err(e) => ScenarioOutcome::fail(NAME, format!("sdk error: {e}")),
    }
}

async fn run(endpoint: SocketAddr) -> Result<ScenarioOutcome, HarnessError> {
    // Unique namespace per run — SCHEMA_UPLOAD is additive + persistent, so
    // the key is minted into this namespace (a default `eval` key dropping in
    // `e2_…` is a PermissionDenied). Same discipline as `schema_lifecycle`.
    let space_id = brain_db_sdk::new_id();
    let ns = format!("e2_{}", &hex16(space_id)[..12]);
    let h = BrainEvalHarness::connect_in_namespace(endpoint, space_id, ns.clone()).await?;

    // A well-formed schema: one entity_type, one predicate (the drop target),
    // and one relation_type whose `from`/`to` resolve to that entity_type.
    // `define` is the real DSL keyword. The entity_type must be declared — a
    // relation_type referencing an undeclared type is rejected `UnresolvedTypeRef`.
    let doc = format!(
        "namespace {ns}\n\
         define entity_type Widget {{\n\
         \x20\x20\x20\x20attributes {{\n\
         \x20\x20\x20\x20\x20\x20\x20\x20serial: text optional\n\
         \x20\x20\x20\x20}}\n\
         }}\n\
         define predicate {PREDICATE_NAME} {{\n\
         \x20\x20\x20\x20kind: Fact\n\
         \x20\x20\x20\x20object: Value<text>\n\
         }}\n\
         define relation_type depends_on {{\n\
         \x20\x20\x20\x20from: Widget\n\
         \x20\x20\x20\x20to: Widget\n\
         \x20\x20\x20\x20cardinality: many-to-many\n\
         }}\n"
    );

    // --- upload (real apply, not dry-run) ----------------------------
    let up = h
        .client()
        .upload_schema(&SchemaUploadRequest {
            schema_document: doc.clone(),
            dry_run: false,
            allow_breaking: false,
            request_id: new_id(),
            act_as: None,
        })
        .await?;
    if !up.validation_errors.is_empty() {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "upload of a valid schema reported {} validation error(s): {:?}",
                up.validation_errors.len(),
                up.validation_errors
            ),
        ));
    }
    let uploaded_version = up.schema_version;

    // --- drop the predicate (no live rows → no force needed) ---------
    let dropped = h
        .client()
        .drop_schema(&SchemaDropRequest {
            namespace: ns.clone(),
            target_kind: schema_drop_target::PREDICATE,
            target_name: PREDICATE_NAME.to_string(),
            force: false,
            request_id: new_id(),
            act_as: None,
        })
        .await?;

    if !dropped.validation_errors.is_empty() {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "drop of a declared predicate reported errors: {:?}",
                dropped.validation_errors
            ),
        ));
    }
    if !dropped.dropped {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "drop of declared predicate {PREDICATE_NAME:?} reported dropped=false \
                 (live_rows={})",
                dropped.live_rows
            ),
        ));
    }
    if dropped.live_rows != 0 {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "drop reported {} live row(s) on a predicate with none created",
                dropped.live_rows
            ),
        ));
    }
    if dropped.schema_version <= uploaded_version {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "drop version {} did not advance past uploaded version {uploaded_version}",
                dropped.schema_version
            ),
        ));
    }
    let dropped_version = dropped.schema_version;

    // --- get: active version reflects the narrow ---------------------
    let got = h
        .client()
        .get_schema(&SchemaGetRequest {
            namespace: ns.clone(),
            version: 0,
            act_as: None,
        })
        .await?;
    if got.schema_version != dropped_version {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "active version {} != post-drop version {dropped_version}",
                got.schema_version
            ),
        ));
    }

    // --- re-drop the now-absent predicate: a no-op -------------------
    let redrop = h
        .client()
        .drop_schema(&SchemaDropRequest {
            namespace: ns.clone(),
            target_kind: schema_drop_target::PREDICATE,
            target_name: PREDICATE_NAME.to_string(),
            force: false,
            request_id: new_id(),
            act_as: None,
        })
        .await?;

    h.close().await?;

    if redrop.dropped {
        return Ok(ScenarioOutcome::fail(
            NAME,
            "re-drop of an already-absent predicate reported dropped=true (should be a no-op)",
        ));
    }
    // A no-op drop reports the current active version unchanged (it advances
    // only when a drop actually narrows the document). With a schema still
    // active, that is `dropped_version` — not 0 (0 is reserved for the case
    // where no schema is declared at all).
    if redrop.schema_version != dropped_version {
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "no-op re-drop returned schema_version {} (should report the unchanged \
                 active version {dropped_version})",
                redrop.schema_version
            ),
        ));
    }

    Ok(ScenarioOutcome::pass(
        NAME,
        format!(
            "§19/06 schema drop: ns {ns} uploaded @v{uploaded_version}, \
             dropped predicate {PREDICATE_NAME} → v{dropped_version} (live_rows=0), \
             active=v{}, re-drop=no-op",
            got.schema_version
        ),
    ))
}
