//! Durability scenarios that own the server lifecycle.
//!
//! Unlike the core scenarios (which take an endpoint), these boot the
//! server themselves via [`ServerHandle`] so they can kill and restart it.
//! The headline check is **restart-recovery / no-data-loss**: the WAL
//! must replay so memories written before a crash survive it.

use std::time::Duration;

use brain_db_sdk::EncodeBuilder;

use super::ScenarioOutcome;
use crate::run::harness::BrainEvalHarness;
use crate::run::server::{remove_volume, DockerServerOpts, ServerHandle};

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
    volume: &str,
) -> Result<ScenarioOutcome, String> {
    const N: usize = 25;

    // --- boot #1, write N memories ------------------------------------
    let server1 = ServerHandle::start_docker(opts.clone())
        .await
        .map_err(|e| format!("first boot: {e}"))?;

    // One agent id, reused across the restart so the second connection
    // queries the same shard slice the first wrote to.
    let agent_id = brain_db_sdk::new_id();
    let marker = format!("recovery-marker-{}", super::hex16(agent_id));

    {
        let h = BrainEvalHarness::connect_with_agent(server1.endpoint(), agent_id)
            .await
            .map_err(|e| format!("connect #1: {e}"))?;
        for i in 0..N {
            let text = format!("{marker}: durable fact number {i}");
            let req = EncodeBuilder::new(text.as_str()).deduplicate(false).build();
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
        let h = BrainEvalHarness::connect_with_agent(server2.endpoint(), agent_id)
            .await
            .map_err(|e| format!("connect #2: {e}"))?;
        // Pull more than N so we can count distinct recovered facts.
        let out = h
            .recall(&marker, (N as u32) * 2)
            .await
            .map_err(|e| format!("recall after restart: {e}"))?;
        let hits = out
            .hits
            .iter()
            .filter(|m| m.text.contains(&marker))
            .count();
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
