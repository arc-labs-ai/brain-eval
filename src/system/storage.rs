//! Storage-footprint gate ("E6").
//!
//! Boots a server on a fresh volume, ingests `n` memories, and reads the
//! per-shard storage gauges off `/metrics`
//! (`brain_wal_size_bytes`, `brain_wal_segments`, `brain_metadata_size_bytes`,
//! `brain_arena_slots_used`, `brain_arena_used_bytes`,
//! `brain_arena_capacity_bytes`). It asserts the gauges are live and the
//! accounting is sane — every memory took a durable arena slot, metadata
//! grew with the data — and reports the per-memory disk footprint.
//!
//! At the spec's 1M scale on reference hardware this is the disk-budget
//! gate (arena ~6 GB / metadata ~1 GB / WAL ~0.5-1 GB; ~8-10 GB total per
//! shard). Pass `max_bytes_per_memory` to enforce a ceiling there. On a
//! dev-box smoke at small `n` the footprint is dominated by fixed overhead
//! (a preallocated WAL segment, redb's initial pages), so leave the ceiling
//! `None` and treat the numbers as informational — the point of the smoke
//! is to prove the gauges emit and the accounting holds.
//!
//! Owns its server lifecycle and cleans up the volume on exit. `opts` must
//! carry a unique container name / ports / `data_volume`.

use std::net::SocketAddr;

use brain_db_sdk::EncodeBuilder;

use super::ScenarioOutcome;
use crate::run::harness::BrainEvalHarness;
use crate::run::metrics::Metrics;
use crate::run::server::{remove_volume, DockerServerOpts, ServerHandle};

const NAME: &str = "storage_footprint";

/// Ingest `n` memories and assert the storage gauges are live and sane.
/// `max_bytes_per_memory` enforces the disk-budget ceiling when `Some`
/// (use at the 1M reference scale); `None` reports footprint informationally.
pub async fn storage_footprint(
    opts: DockerServerOpts,
    n: usize,
    max_bytes_per_memory: Option<f64>,
) -> ScenarioOutcome {
    let Some(volume) = opts.data_volume.clone() else {
        return ScenarioOutcome::fail(NAME, "storage_footprint requires opts.data_volume");
    };
    let metrics_addr: SocketAddr = match format!("127.0.0.1:{}", opts.metrics_port).parse() {
        Ok(a) => a,
        Err(e) => return ScenarioOutcome::fail(NAME, format!("bad metrics addr: {e}")),
    };

    let result = run(opts, metrics_addr, n, max_bytes_per_memory).await;
    remove_volume(&volume).await;
    match result {
        Ok(o) => o,
        Err(detail) => ScenarioOutcome::fail(NAME, detail),
    }
}

async fn run(
    opts: DockerServerOpts,
    metrics_addr: SocketAddr,
    n: usize,
    max_bytes_per_memory: Option<f64>,
) -> Result<ScenarioOutcome, String> {
    let server = ServerHandle::start_docker(opts)
        .await
        .map_err(|e| format!("boot: {e}"))?;

    // Baseline metadata size on the empty store.
    let baseline = Metrics::scrape(metrics_addr)
        .await
        .map_err(|e| format!("baseline scrape: {e}"))?;
    let meta_before = baseline.sum("brain_metadata_size_bytes").unwrap_or(0.0);

    // Ingest n memories under one agent (one shard slice).
    let agent_id = brain_db_sdk::new_id();
    {
        let h = BrainEvalHarness::connect_with_agent(server.endpoint(), agent_id)
            .await
            .map_err(|e| format!("connect: {e}"))?;
        for i in 0..n {
            let text = format!("footprint item {i}: a durable memory carrying ordinary prose");
            let req = EncodeBuilder::new(text.as_str()).build();
            if let Err(e) = h.client().encode(&req).await {
                let _ = h.close().await;
                server.stop().await;
                return Err(format!("encode #{i}: {e}"));
            }
        }
        let _ = h.close().await;
    }

    // Read the storage gauges after ingest.
    let m = Metrics::scrape(metrics_addr)
        .await
        .map_err(|e| format!("post-ingest scrape: {e}"))?;
    server.stop().await;

    // Every storage gauge must be present — that's the end-to-end proof the
    // new /metrics block emits.
    let wal = required(&m, "brain_wal_size_bytes")?;
    let segments = required(&m, "brain_wal_segments")?;
    let metadata = required(&m, "brain_metadata_size_bytes")?;
    let slots_used = required(&m, "brain_arena_slots_used")?;
    let arena_used = required(&m, "brain_arena_used_bytes")?;
    let capacity = required(&m, "brain_arena_capacity_bytes")?;

    // Accounting sanity (scale-invariant).
    if (slots_used as usize) < n {
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!("arena_slots_used={slots_used} < {n} ingested — a memory took no slot"),
        ));
    }
    if metadata <= meta_before {
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!("metadata did not grow with data (before={meta_before}, after={metadata})"),
        ));
    }
    if wal <= 0.0 || segments < 1.0 {
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!("WAL gauges look empty (size={wal}, segments={segments})"),
        ));
    }
    if capacity < arena_used {
        return Ok(ScenarioOutcome::fail(
            NAME,
            format!("arena capacity {capacity} < used {arena_used}"),
        ));
    }

    // Per-memory footprint of the data-bearing files.
    let total = wal + metadata + arena_used;
    let per_memory = total / n as f64;

    if let Some(ceiling) = max_bytes_per_memory {
        if per_memory > ceiling {
            return Ok(ScenarioOutcome::fail(
                NAME,
                format!(
                    "disk footprint {per_memory:.0} B/memory exceeds the {ceiling:.0} B budget \
                     (wal={wal:.0} meta={metadata:.0} arena_used={arena_used:.0}, n={n})"
                ),
            ));
        }
    }

    Ok(ScenarioOutcome::pass(
        NAME,
        format!(
            "storage gauges live; {n} memories → {} arena slots; footprint {per_memory:.0} B/mem \
             (wal={:.1}MiB segs={segments:.0} meta={:.1}MiB arena_used={:.1}MiB cap={:.1}MiB){}",
            slots_used as u64,
            wal / MIB,
            metadata / MIB,
            arena_used / MIB,
            capacity / MIB,
            match max_bytes_per_memory {
                Some(c) => format!("; under {c:.0} B/mem budget"),
                None => " [informational]".to_string(),
            },
        ),
    ))
}

const MIB: f64 = 1024.0 * 1024.0;

/// Require a gauge family to be present (the new metrics must actually emit).
fn required(m: &Metrics, family: &str) -> Result<f64, String> {
    m.sum(family)
        .ok_or_else(|| format!("storage gauge {family} absent from /metrics"))
}
