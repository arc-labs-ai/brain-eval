//! `BrainEvalHarness` — the eval-side wrapper around
//! [`brain_db_sdk::BrainClient`].
//!
//! Each harness instance owns one client bound to a fresh agent id.
//! Per-test isolation is achieved by spawning a new harness per scope —
//! every agent maps to its own shard slice, so agent A's memories are
//! never visible to agent B (Brain's natural routing rule).
//!
//! Two methods cover the eval surface:
//!
//! - [`BrainEvalHarness::ingest`] — one ENCODE per user turn. Returns the
//!   ids of the freshly written memories so callers can correlate
//!   downstream.
//! - [`BrainEvalHarness::recall`] — issue a RECALL and return a
//!   [`RecallOutcome`] carrying the server's *answer shape*: the memories
//!   the smart router decided to return, tagged `Single` (one memory),
//!   `Many` (a set), or `None` (honest abstention). There is no read mode
//!   and no separate retrieval lane exposed on the wire — the server runs
//!   one unified real-memory path and the router shapes the result.

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use brain_db_sdk::wire::types::{AnswerKindWire, MemoryResult, WireMemoryId, WireUuid};
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
        // The eval write path triggers full server-side extraction
        // (pattern + LLM) on a CPU-only dev box, where a single ENCODE can
        // far exceed the SDK's 30s default. A short deadline fails the whole
        // session on one slow turn (the dreaded INGEST-FAIL); give ingest
        // generous headroom so latency, not a timeout, is what we measure.
        let config = ClientConfig {
            agent_id,
            request_timeout: Some(Duration::from_secs(300)),
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
    ///
    /// Deduplication is not a client knob: Brain's write router decides
    /// whether a write merges into an existing near-duplicate. We still
    /// count the server's `was_deduplicated` verdict for write-quality
    /// reporting.
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
            let request = EncodeBuilder::new(turn.content.as_str()).build();
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

    /// Run a RECALL and return the server's answer shape.
    ///
    /// There is no read mode: the server always runs the unified
    /// real-memory path behind a smart router that decides whether to
    /// return one memory (`Single`), a set (`Many`), or nothing at all
    /// (`None`, an honest abstention). `max_results` is a safety cap on
    /// the returned set size — not a ranking knob; the answer's shape
    /// comes from the stored data, never from this count.
    pub async fn recall(
        &self,
        cue: &str,
        max_results: u32,
    ) -> Result<RecallOutcome, HarnessError> {
        let start = Instant::now();
        let request = RecallBuilder::new(cue)
            .max_results(max_results)
            .include_text(true)
            .build();
        let answer = self.client.recall(&request).await?;
        let latency_ms = elapsed_ms(start);
        Ok(RecallOutcome {
            answer_kind: answer.answer_kind,
            memories: answer.memories,
            latency_ms,
        })
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
    /// Number of attempts the server merged into a near-duplicate.
    pub deduplicated: u64,
    /// Wall-clock time for the whole ingest, in milliseconds.
    pub latency_ms: u64,
}

/// Outcome of one [`BrainEvalHarness::recall`] call — the memories the
/// server's smart router returned, tagged with the shape it chose.
///
/// `answer_kind` is the router's verdict: `Single` (exactly one memory),
/// `Many` (a set), or `None` (it declined — `memories` is empty). The
/// caller never sees a separate retrieval lane; the router has already
/// decided what to surface.
#[derive(Debug, Clone)]
pub struct RecallOutcome {
    /// The shape of the answer the router returned.
    pub answer_kind: AnswerKindWire,
    /// The memories the router surfaced. Empty iff `answer_kind` is
    /// `None`.
    pub memories: Vec<MemoryResult>,
    /// Wall-clock time for the RECALL, in milliseconds.
    pub latency_ms: u64,
}

impl RecallOutcome {
    /// The router declined — no memory answers this cue.
    #[must_use]
    pub fn is_abstention(&self) -> bool {
        matches!(self.answer_kind, AnswerKindWire::None)
    }
}

fn elapsed_ms(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}
