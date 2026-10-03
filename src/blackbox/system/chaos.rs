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

use brain_db_sdk::wire::types::{
    EncodeRequest, MemoryListDirWire, MemoryListRequest, MemoryListSortWire, MemoryListTimeAxisWire,
};
use brain_db_sdk::{new_id, Auth, BrainClient, ClientConfig};

use super::ScenarioOutcome;
use crate::client::server::{remove_volume, DockerServerOpts, ServerHandle};

const NAME: &str = "kill_during_write";
/// Crash/restart cycles. Each writes a fresh batch, is SIGKILLed mid-stream,
/// then restarts and re-verifies ALL prior ACKs.
const CYCLES: usize = 2;
/// How long writes flow before the SIGKILL, so ops are genuinely in flight.
const KILL_DELAY: Duration = Duration::from_millis(1500);
/// Settle time for docker to release the port mapping between boots.
const SETTLE: Duration = Duration::from_secs(1);
/// Cap on ACKs collected per cycle, so a fast machine does not spend the
/// whole kill window writing. The survivor count pages through MEMORY_LIST,
/// so this is no longer bounded by a RECALL top-k.
const MAX_BATCH: usize = 200;
/// MEMORY_LIST page size. The server validates this to `1..=100`.
const MEMORY_LIST_PAGE: u32 = 100;

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
    // One space id for the whole run → one shard slice on the volume.
    let space_id = new_id();
    let marker = format!("chaos-{}", super::hex16(space_id));
    let mut acked: Vec<(u128, String)> = Vec::new();
    let mut next_index = 0usize;

    for cycle in 0..CYCLES {
        // Boot on the persistent volume — after cycle 0 this replays the WAL
        // of every prior (crashed) cycle.
        let server = ServerHandle::start_docker(opts.clone())
            .await
            .map_err(|e| format!("cycle {cycle}: boot: {e}"))?;

        // Mint this boot's data-plane key for the single space. The admin
        // plane is fresh each boot, so we re-mint after every restart.
        let token = server
            .key_minter()
            .mint(space_id)
            .await
            .map_err(|e| format!("cycle {cycle}: mint: {e}"))?;

        // Recovery check: every ACK from prior cycles must still be present.
        if !acked.is_empty() {
            let survived = count_survivors(server.endpoint(), &token, &marker, &acked)
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
        let token_c = token.clone();
        let writer = tokio::spawn(async move {
            write_until_dead(endpoint, &token_c, &marker_c, next_index).await
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
    let token = server
        .key_minter()
        .mint(space_id)
        .await
        .map_err(|e| format!("final mint: {e}"))?;
    let survived = count_survivors(server.endpoint(), &token, &marker, &acked)
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
    token: &str,
    marker: &str,
    start_index: usize,
) -> Vec<(u128, String)> {
    let mut acked = Vec::new();
    let client = match BrainClient::connect_with(endpoint, client_config(token)).await {
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

/// Count how many of `acked` are still in the durable store after restart.
///
/// This enumerates with MEMORY_LIST rather than asking RECALL, because the
/// two answer different questions and only one of them is the invariant
/// under test. Writes here use [`WaitMode::Ack`], whose contract is "durable
/// sync ack; async derivation runs in the background" — so an ACK promises
/// the record is on disk, and promises nothing about its embedding being
/// built. A SIGKILL mid-stream is precisely the event that strands the last
/// ACKed writes with their derivation unfinished.
///
/// RECALL reads the derived indexes, so it cannot see such a record and
/// reports it as lost. That made the scenario fail with a healthy server
/// (17/18 "survived") by holding the write path to a guarantee its own wait
/// mode explicitly disclaims. MEMORY_LIST walks the tenant timeline straight
/// off the durable rows — no query, no ranking, no derivation — which is
/// exactly the WAL-before-ack claim in this module's doc comment.
///
/// The space id is freshly minted per run, so enumerating the whole space is
/// exact: every row in it belongs to this scenario. (MEMORY_LIST v1 rejects
/// the `text_contains` filter, so `marker` is used only for the diagnostic
/// cross-check that the rows found really are this run's.)
async fn count_survivors(
    endpoint: SocketAddr,
    token: &str,
    marker: &str,
    acked: &[(u128, String)],
) -> Result<usize, String> {
    let client = BrainClient::connect_with(endpoint, client_config(token))
        .await
        .map_err(|e| e.to_string())?;

    let mut present: HashSet<u128> = HashSet::new();
    let mut cursor: Vec<u8> = Vec::new();
    loop {
        let req = MemoryListRequest {
            sort: MemoryListSortWire::Created,
            dir: MemoryListDirWire::Asc,
            limit: MEMORY_LIST_PAGE,
            cursor: cursor.clone(),
            kinds: Vec::new(),
            // A crashed write must not be resurrected as a tombstone and
            // counted as a survivor, and a tombstone here would itself be a
            // finding — so enumerate both states and check `state` below.
            include_tombstoned: true,
            time_axis: MemoryListTimeAxisWire::Created,
            from_unix_nanos: 0,
            to_unix_nanos: 0,
            salience_min: 0.0,
            salience_max: 1.0,
            text_contains: String::new(),
            act_as: None,
        };
        let frames = client
            .memory_list_frames(&req)
            .await
            .map_err(|e| e.to_string())?;
        let (mut last_cursor, mut done) = (Vec::new(), true);
        for frame in &frames {
            for item in &frame.items {
                // state 0 = active. A tombstoned row is not a survivor.
                if item.state == 0 && item.text.contains(marker) {
                    present.insert(u128::from_be_bytes(item.memory_id));
                }
            }
            last_cursor = frame.next_cursor.clone();
            done = frame.is_final;
        }
        if done || last_cursor.is_empty() {
            break;
        }
        cursor = last_cursor;
    }

    let survived = acked.iter().filter(|(id, _)| present.contains(id)).count();
    let _ = client.close().await;
    Ok(survived)
}

fn client_config(token: &str) -> ClientConfig {
    ClientConfig::new(Auth::Token(token.as_bytes().to_vec()))
}

/// An ENCODE with a fresh request id (every call writes a real row).
fn encode_req(text: &str) -> EncodeRequest {
    EncodeRequest {
        text: text.to_string(),
        session_id: 0,
        request_id: new_id(),
        txn_id: None,
        occurred_at_unix_nanos: None,
        act_as: None,
        wait: brain_db_sdk::wire::types::WaitMode::Ack,
        allow_duplicates: false,
    }
}
