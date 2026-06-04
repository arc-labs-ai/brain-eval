//! `BrainEvalHarness` — the eval-side wrapper around
//! [`brain_db_sdk::BrainClient`].
//!
//! Each harness instance owns one client bound to a fresh agent id.
//! Per-test isolation is achieved by spawning a new harness per scope —
//! every agent maps to its own shard slice, so agent A's memories are
//! never visible to agent B (Brain's natural routing rule).
//!
//! Two methods cover the eval surface today:
//!
//! - [`BrainEvalHarness::ingest`] — one ENCODE per user turn. Returns the
//!   ids of the freshly written memories so callers can correlate
//!   downstream.
//! - [`BrainEvalHarness::recall`] — issue a RECALL with `include_text`
//!   set, returning a [`RecallOutcome`] with hits + per-call latency.

use std::net::SocketAddr;
use std::time::Instant;

use brain_db_sdk::wire::types::{MemoryResult, WireMemoryId, WireUuid};
use brain_db_sdk::{BrainClient, BrainError, ClientConfig, EncodeBuilder, RecallBuilder};

use crate::core::instance::TurnRecord;

/// Errors surfaced by the harness.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HarnessError {
    /// The underlying SDK client returned an error.
    #[error("brain-db-sdk error: {0}")]
    Sdk(#[from] BrainError),
}

/// One driver bound to one agent.
///
/// Not `Clone`: the SDK client owns a live connection. A harness drives
/// one conversation's ingest plus its N per-question recalls through
/// `&self`, so cloning was never needed.
pub struct BrainEvalHarness {
    client: BrainClient,
    agent_id: WireUuid,
}

impl BrainEvalHarness {
    /// Open a fresh connection with a fresh agent id. Drives the
    /// HELLO → WELCOME → AUTH → AUTH_OK handshake before returning.
    pub async fn connect(addr: SocketAddr) -> Result<Self, HarnessError> {
        let agent_id = brain_db_sdk::new_id();
        Self::connect_with_agent(addr, agent_id).await
    }

    /// Connect with an explicit agent id (useful for tests asserting
    /// isolation, or for resuming an existing eval scope).
    pub async fn connect_with_agent(
        addr: SocketAddr,
        agent_id: WireUuid,
    ) -> Result<Self, HarnessError> {
        let config = ClientConfig {
            agent_id,
            ..ClientConfig::default()
        };
        let client = BrainClient::connect_with(addr, config).await?;
        Ok(Self { client, agent_id })
    }

    /// The agent id this harness was bound to.
    #[must_use]
    pub fn agent_id(&self) -> WireUuid {
        self.agent_id
    }

    /// Borrow the underlying SDK client (escape hatch for tests touching
    /// ops we haven't surfaced here yet, e.g. SUBSCRIBE).
    #[must_use]
    pub fn client(&self) -> &BrainClient {
        &self.client
    }

    /// Ingest a session's turns. Each user turn becomes one ENCODE;
    /// assistant turns are dropped because the dataset's ground truth
    /// was authored against user utterances. The returned
    /// [`IngestOutcome`] carries per-turn outcomes plus wall-clock
    /// latency for the entire ingest.
    pub async fn ingest(&self, turns: &[TurnRecord]) -> Result<IngestOutcome, HarnessError> {
        let start = Instant::now();
        let mut stored_ids: Vec<WireMemoryId> = Vec::new();
        let mut attempted = 0u64;
        let mut deduplicated = 0u64;

        for turn in turns {
            if turn.role != "user" {
                continue;
            }
            if turn.content.trim().is_empty() {
                continue;
            }
            attempted += 1;
            let request = EncodeBuilder::new(turn.content.as_str())
                .deduplicate(true)
                .build();
            let resp = self.client.encode(&request).await?;
            if resp.was_deduplicated {
                deduplicated += 1;
            } else {
                stored_ids.push(resp.memory_id);
            }
        }

        let latency_ms = elapsed_ms(start);
        Ok(IngestOutcome {
            stored_ids,
            attempted,
            deduplicated,
            latency_ms,
        })
    }

    /// Run a RECALL with `include_text` so the eval can read memory
    /// contents back for substring scoring.
    pub async fn recall(&self, cue: &str, top_k: u32) -> Result<RecallOutcome, HarnessError> {
        let start = Instant::now();
        let request = RecallBuilder::new(cue)
            .top_k(top_k)
            .include_text(true)
            .build();
        let hits = self.client.recall(&request).await?;
        let latency_ms = elapsed_ms(start);
        Ok(RecallOutcome { hits, latency_ms })
    }

    /// Close the underlying client.
    pub async fn close(self) -> Result<(), HarnessError> {
        self.client.close().await?;
        Ok(())
    }
}

/// Outcome of one [`BrainEvalHarness::ingest`] call.
#[derive(Debug, Clone)]
pub struct IngestOutcome {
    /// Ids of fresh memories the substrate accepted.
    pub stored_ids: Vec<WireMemoryId>,
    /// Number of ENCODE attempts (user turns processed).
    pub attempted: u64,
    /// Number of attempts that hit the fingerprint dedupe path.
    pub deduplicated: u64,
    /// Wall-clock time for the whole ingest, in milliseconds.
    pub latency_ms: u64,
}

/// Outcome of one [`BrainEvalHarness::recall`] call.
#[derive(Debug, Clone)]
pub struct RecallOutcome {
    /// Memory hits returned by the substrate, in server order.
    pub hits: Vec<MemoryResult>,
    /// Wall-clock time for the RECALL, in milliseconds.
    pub latency_ms: u64,
}

fn elapsed_ms(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}
