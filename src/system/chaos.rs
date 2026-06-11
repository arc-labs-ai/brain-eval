//! Kill-during-operation chaos ("E4").
//!
//! Restart-recovery ([`super::restart_recovery`]) proves a *quiesced* server
//! survives a restart. This goes further: it SIGKILLs the server **while
//! writes are in flight**, then proves the WAL-before-ack invariant
//! (invariant #1) — every write the server ACKed before the crash is still
//! there after restart, and a write that was in flight at kill time (never
//! ACKed) simply doesn't exist, with no torn/partial row served.
//!
//! `docker rm -f` is a SIGKILL (force-remove kills the container outright,
//! no SIGTERM grace), so there is no graceful flush: only records whose WAL
//! entry was fsynced before the ACK can survive. That is exactly the
//! contract under test.
//!
//! The crash/restart cycle runs `CYCLES` times against one persistent
//! volume, re-verifying **every** ACK from **all** prior cycles after each
//! restart — so it also exercises recovery idempotency across repeated
//! crashes (replaying an already-replayed WAL must not lose or duplicate
//! state).
//!
//! Owns its server lifecycle and cleans up the volume on exit. `opts` must
//! carry a unique container name / ports / `data_volume`.

use std::collections::HashSet;
use std::net::SocketAddr;
use std::time::Duration;

use brain_db_sdk::wire::types::{EncodeRequest, MemoryKindWire};
use brain_db_sdk::{new_id, BrainClient, ClientConfig, RecallBuilder};

use super::ScenarioOutcome;
use crate::run::server::{remove_volume, DockerServerOpts, ServerHandle};

const NAME: &str = "kill_during_write";
/// Crash/restart cycles. Each writes a fresh batch, is SIGKILLed mid-stream,
/// then restarts and re-verifies ALL prior ACKs.
const CYCLES: usize = 2;
/// How long writes flow before the SIGKILL, so ops are genuinely in flight.
const KILL_DELAY: Duration = Duration::from_millis(1500);
/// Settle time for docker to release the port mapping between boots.
const SETTLE: Duration = Duration::from_secs(1);
/// Cap on ACKs collected per cycle, so on fast hardware the total stays well
/// under `MAX_RECALL_TOP_K` (1000) for the single-RECALL survivor count.
const MAX_BATCH: usize = 200;

/// SIGKILL the server mid-write across `CYCLES` cycles; every ACKed write
/// must survive every restart.
pub async fn kill_during_write(opts: DockerServerOpts) -> ScenarioOutcome {
    let Some(volume) = opts.data_volume.clone() else {
        return ScenarioOutcome::fail(
            NAME,
            "kill_during_write requires opts.data_volume (a persistent volume)",
        );
    };
    let result = run(opts, &volume).await;
    // Always try to clean the volume up, whatever happened.
    remove_volume(&volume).await;
    match result {
        Ok(o) => o,
        Err(detail) => ScenarioOutcome::fail(NAME, detail),
    }
}

async fn run(opts: DockerServerOpts, _volume: &str) -> Result<ScenarioOutcome, String> {
    // One agent id for the whole run → one shard slice on the volume.
    let agent_id = new_id();
    let marker = format!("chaos-{}", super::hex16(agent_id));
    let mut acked: Vec<(u128, String)> = Vec::new();
    let mut next_index = 0usize;

    for cycle in 0..CYCLES {
        // Boot on the persistent volume — after cycle 0 this replays the WAL
        // of every prior (crashed) cycle.
        let server = ServerHandle::start_docker(opts.clone())
            .await
            .map_err(|e| format!("cycle {cycle}: boot: {e}"))?;

        // Recovery check: every ACK from prior cycles must still be present.
        if !acked.is_empty() {
            let survived = count_survivors(server.endpoint(), agent_id, &marker, &acked)
                .await
                .map_err(|e| format!("cycle {cycle}: post-restart verify: {e}"))?;
            if survived != acked.len() {
                server.stop().await;
                return Ok(ScenarioOutcome::fail(
                    NAME,
                    format!(
                        "data loss after restart (entering cycle {cycle}): {survived}/{} ACKed \
                         writes survived",
                        acked.len()
                    ),
                ));
            }
        }

        // Write a fresh batch in a spawned task; SIGKILL mid-stream.
        let endpoint = server.endpoint();
        let marker_c = marker.clone();
        let writer = tokio::spawn(async move {
            write_until_dead(endpoint, agent_id, &marker_c, next_index).await
        });
        tokio::time::sleep(KILL_DELAY).await;
        server.stop().await; // SIGKILL — no graceful flush.
        let batch = writer
            .await
            .map_err(|e| format!("cycle {cycle}: writer task join: {e}"))?;
        if batch.is_empty() {
            return Err(format!(
                "cycle {cycle}: no writes were ACKed before the kill"
            ));
        }
        next_index += batch.len() + 16;
        acked.extend(batch);
        tokio::time::sleep(SETTLE).await;
    }

    // Final restart: the last cycle's ACKs must survive too.
    let server = ServerHandle::start_docker(opts.clone())
        .await
        .map_err(|e| format!("final boot: {e}"))?;
    let survived = count_survivors(server.endpoint(), agent_id, &marker, &acked)
        .await
        .map_err(|e| format!("final verify: {e}"))?;
    server.stop().await;

    if survived == acked.len() {
        Ok(ScenarioOutcome::pass(
            NAME,
            format!(
                "invariant #1: {} ACKed writes across {CYCLES} SIGKILL/restart cycles all \
                 survived (WAL-before-ack; recovery idempotent across repeated crashes)",
                acked.len()
            ),
        ))
    } else {
        Ok(ScenarioOutcome::fail(
            NAME,
            format!(
                "data loss after final restart: {survived}/{} ACKed writes survived",
                acked.len()
            ),
        ))
    }
}

/// Encode a stream until the connection dies (the SIGKILL). Returns the
/// `(memory_id, cue)` of every write the server ACKed — the writes that, by
/// the WAL-before-ack contract, MUST survive the crash. A write in flight at
/// kill time errors and is never recorded.
async fn write_until_dead(
    endpoint: SocketAddr,
    agent_id: [u8; 16],
    marker: &str,
    start_index: usize,
) -> Vec<(u128, String)> {
    let mut acked = Vec::new();
    let client = match BrainClient::connect_with(endpoint, client_config(agent_id)).await {
        Ok(c) => c,
        Err(_) => return acked,
    };

    let mut i = start_index;
    while acked.len() < MAX_BATCH {
        let cue = format!("{marker}-{i}");
        let req = encode_req(&format!("{cue}: chaos durable fact"));
        match client.encode(&req).await {
            Ok(r) => acked.push((r.memory_id, cue)),
            Err(_) => break, // connection dropped by the SIGKILL
        }
        i += 1;
    }

    let _ = client.close().await;
    acked
}

/// Count how many of `acked` resurface in a single RECALL on the marker.
async fn count_survivors(
    endpoint: SocketAddr,
    agent_id: [u8; 16],
    marker: &str,
    acked: &[(u128, String)],
) -> Result<usize, String> {
    let client = BrainClient::connect_with(endpoint, client_config(agent_id))
        .await
        .map_err(|e| e.to_string())?;
    let k = (acked.len() * 2).clamp(10, 1000) as u32;
    let req = RecallBuilder::new(marker)
        .top_k(k)
        .include_text(false)
        .build();
    let hits = client.recall(&req).await.map_err(|e| e.to_string())?;
    let ids: HashSet<u128> = hits.iter().map(|m| m.memory_id).collect();
    let survived = acked.iter().filter(|(id, _)| ids.contains(id)).count();
    let _ = client.close().await;
    Ok(survived)
}

fn client_config(agent_id: [u8; 16]) -> ClientConfig {
    ClientConfig {
        agent_id,
        ..ClientConfig::default()
    }
}

/// A non-dedup ENCODE (every call writes a real, durable row).
fn encode_req(text: &str) -> EncodeRequest {
    EncodeRequest {
        text: text.to_string(),
        context_id: 0,
        kind: MemoryKindWire::Semantic,
        salience_hint: 0.5,
        edges: Vec::new(),
        request_id: new_id(),
        txn_id: None,
        deduplicate: false,
    }
}
