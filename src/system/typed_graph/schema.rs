//! Schema scenario: upload → get → validate (valid + invalid).
//!
//! Exercises §19/06 "schema lifecycle": a user namespace can be uploaded
//! (entity type + predicate + relation type), read back at its assigned
//! version, and used to validate documents — both a well-formed one (no
//! errors, a `would_be_version`) and a malformed one (populated
//! `validation_errors`).

use std::net::SocketAddr;

use brain_db_sdk::new_id;
use brain_db_sdk::wire::types::{SchemaGetRequest, SchemaUploadRequest, SchemaValidateRequest};

use super::super::ScenarioOutcome;
use super::hex16;
use crate::run::harness::{BrainEvalHarness, HarnessError};

const NAME: &str = "tg_schema_lifecycle";

/// Upload a valid user schema, read it back, then validate a good and a
/// bad document.
pub async fn schema_lifecycle(endpoint: SocketAddr) -> ScenarioOutcome {
    match run(endpoint).await {
        Ok(o) => o,
        Err(e) => ScenarioOutcome::fail(NAME, format!("sdk error: {e}")),
    }
}

async fn run(endpoint: SocketAddr) -> Result<ScenarioOutcome, HarnessError> {
    let h = BrainEvalHarness::connect(endpoint).await?;
    // Unique namespace per run — SCHEMA_UPLOAD is additive + persistent.
    let ns = format!("e2_{}", &hex16(h.agent_id())[..12]);

    // A well-formed schema: one entity type, one predicate, one relation
    // type. `define` is the real DSL keyword (the SDK mock tests use a
    // shorthand that the real parser rejects).
    let doc = format!(
        "namespace {ns}\n\
         define entity_type Widget {{\n\
         \x20\x20\x20\x20attributes {{\n\
         \x20\x20\x20\x20\x20\x20\x20\x20serial: text optional\n\
         \x20\x20\x20\x20}}\n\
         }}\n\
         define predicate built_by {{\n\
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
    if up.namespace != ns {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "upload returned namespace {:?}, expected {:?}",
                up.namespace, ns
            ),
        ));
    }
    if up.schema_version == 0 {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            "upload assigned schema_version 0; a fresh namespace must start at version >= 1",
        ));
    }
    let uploaded_version = up.schema_version;

    // --- get it back at the active version (version == 0 selects active) ---
    let got = h
        .client()
        .get_schema(&SchemaGetRequest {
            namespace: ns.clone(),
            version: 0,
        })
        .await?;
    if got.namespace != ns {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "get returned namespace {:?}, expected {:?}",
                got.namespace, ns
            ),
        ));
    }
    if got.schema_version != uploaded_version {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "active version {} != uploaded version {uploaded_version}",
                got.schema_version
            ),
        ));
    }

    // --- validate a well-formed document (no errors expected) --------
    let valid = h
        .client()
        .validate_schema(&SchemaValidateRequest {
            schema_document: doc.clone(),
        })
        .await?;
    if !valid.validation_errors.is_empty() {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "validate of a valid document reported errors: {:?}",
                valid.validation_errors
            ),
        ));
    }

    // --- validate a malformed document (errors expected) -------------
    // Garbage that cannot parse as the DSL — must surface >= 1 error and
    // must NOT yield a would_be_version.
    let bad_doc = format!("namespace {ns}\ndefine entity_type {{ this is not valid <<< }}\n");
    let invalid = h
        .client()
        .validate_schema(&SchemaValidateRequest {
            schema_document: bad_doc,
        })
        .await?;

    h.close().await?;

    if invalid.validation_errors.is_empty() {
        return Ok(ScenarioOutcome::fail(
            NAME,
            "validate of a malformed document reported NO errors (should reject)",
        ));
    }
    if invalid.would_be_version != 0 {
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "malformed document got would_be_version {} (should be 0 on failure)",
                invalid.would_be_version
            ),
        ));
    }

    Ok(ScenarioOutcome::pass(
        NAME,
        format!(
            "§19/06 schema: uploaded ns {ns} @v{uploaded_version}, read back active=v{}, \
             validate(valid)=0 errors, validate(bad)={} error(s)",
            got.schema_version,
            invalid.validation_errors.len()
        ),
    ))
}
