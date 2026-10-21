//! Durability scenarios that own the server lifecycle.
//!
//! Unlike the core scenarios (which take an endpoint), these boot the
//! server themselves via [`ServerHandle`] so they can kill and restart it.
//! The headline check is **restart-recovery / no-data-loss**: the WAL
//! must replay so memories written before a crash survive it.

use std::time::Duration;

use brain_db_sdk::EncodeBuilder;

use super::ScenarioOutcome;
use crate::client::harness::BrainEvalHarness;
use crate::client::server::{remove_volume, DockerServerOpts, ServerHandle};

/// Memories written before a restart must survive it (WAL replay), with no
/// data loss. Boots a server on a persistent volume, writes N memories,
/// removes the container (keeping the volume), boots a fresh container on
/// the same volume, and recalls — every memory must come back.
///
/// Owns its server lifecycle and cleans up the volume on the way out.
/// `opts` should carry a unique container name / ports / `data_volume` so
/// the scenario can run alongside other servers.
pub async fn restart_recovery(opts: DockerServerOpts) -> ScenarioOutcome {
    const NAME: &str = "restart_recovery";
    let Some(volume) = opts.data_volume.clone() else {
        return ScenarioOutcome::fail(
            NAME,
            "restart_recovery requires opts.data_volume (a persistent volume)",
        );
    };

    let result = run_restart_recovery(NAME, opts, &volume).await;
    // Always try to clean the volume up, whatever happened.
    remove_volume(&volume).await;
    match result {
        Ok(outcome) => outcome,
        Err(detail) => ScenarioOutcome::fail(NAME, detail),
    }
}

async fn run_restart_recovery(
    name: &'static str,
    opts: DockerServerOpts,
    _volume: &str,
) -> Result<ScenarioOutcome, String> {
    const N: usize = 25;

    // --- boot #1, write N memories ------------------------------------
    let server1 = ServerHandle::start_docker(opts.clone())
        .await
        .map_err(|e| format!("first boot: {e}"))?;

    // One space id, reused across the restart so the second connection
    // queries the same shard slice the first wrote to.
    let space_id = brain_db_sdk::new_id();
    let marker = format!("recovery-marker-{}", super::hex16(space_id));

    {
        let h =
            BrainEvalHarness::connect_minted(&server1.key_minter(), server1.endpoint(), space_id)
                .await
                .map_err(|e| format!("connect #1: {e}"))?;
        for i in 0..N {
            let text = format!("{marker}: durable fact number {i}");
            let req = EncodeBuilder::new(text.as_str()).build();
            h.client()
                .encode(&req)
                .await
                .map_err(|e| format!("encode #{i}: {e}"))?;
        }
        h.close().await.map_err(|e| format!("close #1: {e}"))?;
    }

    // --- kill the container, keep the volume --------------------------
    server1.stop().await;
    // Brief settle so docker fully releases the port mapping before re-bind.
    tokio::time::sleep(Duration::from_secs(1)).await;

    // --- boot #2 on the same volume -----------------------------------
    let server2 = ServerHandle::start_docker(opts.clone())
        .await
        .map_err(|e| format!("restart boot: {e}"))?;

    let recovered = {
        let h =
            BrainEvalHarness::connect_minted(&server2.key_minter(), server2.endpoint(), space_id)
                .await
                .map_err(|e| format!("connect #2: {e}"))?;
        // Count the durable rows, not RECALL hits. A restart interrupts the
        // async derivation a recall depends on, so asking RECALL whether a
        // write survived reports durable rows as lost — it failed here at
        // 24/25 against a server that had lost nothing. See
        // [`super::durable_active_memory_ids`].
        let hits = super::durable_active_memory_ids(h.client(), &marker)
            .await
            .map_err(|e| format!("memory list after restart: {e}"))?
            .len();
        h.close().await.map_err(|e| format!("close #2: {e}"))?;
        hits
    };

    server2.stop().await;

    if recovered >= N {
        Ok(ScenarioOutcome::pass(
            name,
            format!("all {N} memories survived a container restart (WAL replay)"),
        ))
    } else {
        Ok(ScenarioOutcome::fail(
            name,
            format!("data loss after restart: recovered {recovered}/{N} memories"),
        ))
    }
}

/// Graceful-shutdown-drain scaffold (§19/06 storage/shutdown depth).
///
/// `restart_recovery` proves the *crash* path (SIGKILL → WAL replay). This
/// probes the *graceful* path: `docker stop` sends SIGTERM, so the server gets
/// its shutdown-drain window before dying. Boots on a persistent volume, writes
/// N memories with the default `Ack` wait (WAL-durable, not necessarily
/// derived), stops gracefully, reboots on the same volume, and recalls.
///
/// - **Safe invariant (hard):** no data loss — every memory comes back after a
///   graceful stop, exactly as after a crash.
/// - **Flagged (advisory):** whether the drain *also* flushed the async
///   derivation queue, so the memories are immediately recallable on reboot
///   with no re-derivation lag. That's the difference between "graceful" and
///   "just WAL-durable"; whether Brain's shutdown drains derivation is a
///   behaviour to confirm on a live run, not gate blind — so a shortfall here
///   is reported as an observation, not a failure.
///
/// Owns its server lifecycle and cleans up the volume. `opts` should carry a
/// unique name / ports / `data_volume`.
pub async fn graceful_shutdown_drain(opts: DockerServerOpts) -> ScenarioOutcome {
    const NAME: &str = "graceful_shutdown_drain";
    let Some(volume) = opts.data_volume.clone() else {
        return ScenarioOutcome::fail(
            NAME,
            "graceful_shutdown_drain requires opts.data_volume (a persistent volume)",
        );
    };
    let result = run_graceful_drain(NAME, opts).await;
    remove_volume(&volume).await;
    match result {
        Ok(outcome) => outcome,
        Err(detail) => ScenarioOutcome::fail(NAME, detail),
    }
}

async fn run_graceful_drain(
    name: &'static str,
    opts: DockerServerOpts,
) -> Result<ScenarioOutcome, String> {
    const N: usize = 25;

    let server1 = ServerHandle::start_docker(opts.clone())
        .await
        .map_err(|e| format!("first boot: {e}"))?;

    let space_id = brain_db_sdk::new_id();
    let marker = format!("drain-marker-{}", super::hex16(space_id));
    {
        let h =
            BrainEvalHarness::connect_minted(&server1.key_minter(), server1.endpoint(), space_id)
                .await
                .map_err(|e| format!("connect #1: {e}"))?;
        for i in 0..N {
            let text = format!("{marker}: drained fact number {i}");
            // Default Ack wait: WAL-durable, derivation may still be in flight —
            // exactly the state a graceful drain is supposed to settle.
            h.client()
                .encode(&EncodeBuilder::new(text.as_str()).build())
                .await
                .map_err(|e| format!("encode #{i}: {e}"))?;
        }
        h.close().await.map_err(|e| format!("close #1: {e}"))?;
    }

    // Graceful stop (SIGTERM + drain window), not a kill.
    let stopped_cleanly = server1.stop_graceful().await;
    tokio::time::sleep(Duration::from_secs(1)).await;

    let server2 = ServerHandle::start_docker(opts.clone())
        .await
        .map_err(|e| format!("restart boot: {e}"))?;

    // First read immediately after reboot: if the drain flushed derivation,
    // everything is recallable now with no settle.
    let (immediate, settled) = {
        let h =
            BrainEvalHarness::connect_minted(&server2.key_minter(), server2.endpoint(), space_id)
                .await
                .map_err(|e| format!("connect #2: {e}"))?;
        let count = |out: &crate::client::harness::RecallOutcome| {
            out.memories
                .iter()
                .filter(|m| m.text.contains(&marker))
                .count()
        };
        let first = h
            .recall(&marker, (N as u32) * 2)
            .await
            .map_err(|e| format!("recall after restart: {e}"))?;
        let immediate = count(&first);
        // Allow a brief settle for any re-derivation, then read again.
        tokio::time::sleep(Duration::from_secs(2)).await;
        let second = h
            .recall(&marker, (N as u32) * 2)
            .await
            .map_err(|e| format!("recall after settle: {e}"))?;
        let settled = count(&second);
        h.close().await.map_err(|e| format!("close #2: {e}"))?;
        (immediate, settled)
    };

    server2.stop().await;

    // Safe invariant: no data loss after a graceful stop.
    if settled < N {
        return Ok(ScenarioOutcome::fail(
            name,
            format!("data loss after graceful shutdown: recovered {settled}/{N} (immediate {immediate})"),
        ));
    }

    // Flagged: did the drain flush derivation (immediate == settled == N)?
    Ok(ScenarioOutcome::advisory(
        name,
        format!(
            "graceful stop clean={stopped_cleanly}; no data loss ({settled}/{N}). \
             OBSERVED immediate-recall={immediate}/{N} vs after-settle={settled}/{N}. \
             FLAGGED: immediate<settled means the drain left derivation to finish post-reboot \
             rather than flushing it — confirm the intended drain contract live."
        ),
    ))
}
