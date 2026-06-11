//! Extraction scenario: ENCODE → async extractor → entity appears.
//!
//! Exercises §19/06 "extraction pipeline". Encodes a memory whose text
//! names a person, then polls (bounded retry/sleep, ~10s) until the
//! extractor tier (pattern `brain:entity_mentions` and/or the classifier)
//! has materialized a Person entity for that name. On timeout the scenario
//! fails with a clear message rather than hanging.
//!
//! Requires the server to run the full extraction stack (tiers enabled).
//! The seeded pattern extractor matches two capitalized words
//! (`\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b`), so the probe name is built from
//! two capitalized, run-unique tokens.

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use brain_db_sdk::new_id;
use brain_db_sdk::wire::types::{EntityListRequest, EntityResolveRequest, ResolutionOutcomeWire};
use brain_db_sdk::EncodeBuilder;

use super::super::ScenarioOutcome;
use super::{hex16, PERSON_TYPE_ID};
use crate::run::harness::{BrainEvalHarness, HarnessError};

const NAME: &str = "tg_extraction_pipeline";
/// How long to wait for the async extractor before declaring a timeout.
const MAX_WAIT: Duration = Duration::from_secs(10);
/// Poll cadence while waiting for extraction.
const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Encode a memory naming a person, then poll until the extractor mints it.
pub async fn extraction_pipeline(endpoint: SocketAddr) -> ScenarioOutcome {
    match run(endpoint).await {
        Ok(o) => o,
        Err(e) => ScenarioOutcome::fail(NAME, format!("sdk error: {e}")),
    }
}

async fn run(endpoint: SocketAddr) -> Result<ScenarioOutcome, HarnessError> {
    let h = BrainEvalHarness::connect(endpoint).await?;

    // A run-unique, two-capitalized-word name the pattern extractor will
    // match. Built from the agent id so reruns don't collide and the
    // tokens stay capitalized (the hex marker itself is lowercase and
    // would NOT match the pattern, so we map id bytes to A-Z letters).
    let full_name = unique_person_name(h.agent_id());
    let text =
        format!("During the project kickoff, {full_name} agreed to lead the migration effort.");

    // Encode; deduplicate(false) so the extractor stages always enqueue.
    let enc = h
        .client()
        .encode(&EncodeBuilder::new(text.as_str()).deduplicate(false).build())
        .await?;

    // If the server reports no extractor stage was queued, extraction is
    // disabled on this deployment — skip the assertion honestly.
    let extractor_queued = enc
        .pending_stages
        .iter()
        .any(|s| matches!(s, brain_db_sdk::wire::types::StageKind::Extractor));
    if !extractor_queued {
        h.close().await?;
        return Ok(ScenarioOutcome::fail(
            NAME,
            "ENCODE reported no Extractor stage queued — the extraction tier appears \
             disabled on this server; cannot exercise §19/06 extraction",
        ));
    }

    // --- bounded poll for the extracted entity -----------------------
    let deadline = Instant::now() + MAX_WAIT;
    let mut polls = 0u32;
    loop {
        polls += 1;

        // Resolve by exact canonical name is the cheapest positive signal.
        let resolved = h
            .client()
            .resolve_entity(&EntityResolveRequest {
                candidate_name: full_name.clone(),
                context: String::new(),
                entity_type_hint: PERSON_TYPE_ID,
                allow_create: false,
                request_id: new_id(),
            })
            .await?;
        if resolved.outcome == ResolutionOutcomeWire::Resolved
            && resolved.resolved_entity != [0u8; 16]
        {
            h.close().await?;
            return Ok(ScenarioOutcome::pass(
                NAME,
                format!(
                    "§19/06 extraction: ENCODE of text naming {full_name:?} produced a \
                     Person entity (resolved after {polls} poll(s))"
                ),
            ));
        }

        // Fallback positive signal: a name-prefix list hit (handles the
        // case where the classifier minted it with a slightly different
        // canonical form but the same leading token).
        let listed = h
            .client()
            .list_entities(&EntityListRequest {
                entity_type_id: PERSON_TYPE_ID,
                name_prefix: leading_token(&full_name),
                mention_count_min: 0,
                include_tombstoned: false,
                include_merged: false,
                limit: 1000,
                cursor: Vec::new(),
            })
            .await?;
        if listed
            .iter()
            .any(|it| it.entity.canonical_name.contains(&full_name))
        {
            h.close().await?;
            return Ok(ScenarioOutcome::pass(
                NAME,
                format!(
                    "§19/06 extraction: ENCODE of text naming {full_name:?} produced a \
                     Person entity (found via list after {polls} poll(s))"
                ),
            ));
        }

        if Instant::now() >= deadline {
            h.close().await?;
            return Ok(ScenarioOutcome::fail(
                NAME,
                format!(
                    "extractor did not materialize an entity for {full_name:?} within \
                     {}s ({polls} polls). Either extraction is slow/disabled or the \
                     name did not extract.",
                    MAX_WAIT.as_secs()
                ),
            ));
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

/// Build a two-capitalized-word name from an agent id. Maps id bytes to
/// the A-Z range so the tokens are run-unique yet match the seeded
/// pattern extractor's `[A-Z][a-z]+ [A-Z][a-z]+` shape.
fn unique_person_name(id: [u8; 16]) -> String {
    let h = hex16(id);
    // Two distinct alpha tokens from disjoint hex slices.
    let first = capitalized_token(&h[..6]);
    let last = capitalized_token(&h[6..12]);
    format!("{first} {last}")
}

/// Map a hex slice to a `Xxxxx`-shaped token (leading uppercase, rest
/// lowercase ASCII letters), so it matches `[A-Z][a-z]+`.
fn capitalized_token(hex_slice: &str) -> String {
    let mut out = String::with_capacity(hex_slice.len() + 1);
    // Lead with a fixed uppercase so the first char is always A-Z.
    out.push('Z');
    for c in hex_slice.chars() {
        // Map hex char to a lowercase letter a-p (deterministic, a-z range).
        let v = c.to_digit(16).unwrap_or(0) as u8;
        out.push((b'a' + v) as char);
    }
    out
}

/// The leading token of a two-word name (used as a list name-prefix).
fn leading_token(full_name: &str) -> String {
    full_name
        .split_whitespace()
        .next()
        .unwrap_or(full_name)
        .to_string()
}
