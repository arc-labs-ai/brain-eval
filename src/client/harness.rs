//! `BrainEvalHarness` — the eval-side wrapper around
//! [`brain_db_sdk::BrainClient`].
//!
//! Each harness instance owns one client bound to a fresh space id.
//! Per-test isolation is achieved by spawning a new harness per scope —
//! every space maps to its own shard slice, so space A's memories are
//! never visible to space B (Brain's natural routing rule).
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

use brain_db_sdk::wire::types::{
    AnswerKindWire, MemoryResult, RecallScopeWire, WireMemoryId, WireUuid,
};
use brain_db_sdk::{Auth, BrainClient, BrainError, ClientConfig, EncodeBuilder, RecallBuilder};

use crate::bench::core::instance::TurnRecord;
use crate::client::auth::{KeyMinter, MintError};

/// Generous per-request deadline. The eval write path triggers full
/// server-side extraction (pattern + LLM) on a CPU-only dev box, where a
/// single ENCODE can far exceed the SDK's 30s default. A short deadline fails
/// the whole session on one slow turn (the dreaded INGEST-FAIL); this gives
/// ingest headroom so latency, not a timeout, is what we measure.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

/// Errors surfaced by the harness.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HarnessError {
    /// The underlying SDK client returned an error.
    #[error("brain-db-sdk error: {0}")]
    Sdk(#[from] BrainError),
    /// Minting a data-plane key failed.
    #[error("key mint failed: {0}")]
    Mint(#[from] MintError),
}

/// One driver bound to one space.
///
/// Not `Clone`: the SDK client owns a live connection. A harness drives
/// one conversation's ingest plus its N per-question recalls through
/// `&self`, so cloning was never needed.
pub struct BrainEvalHarness {
    client: BrainClient,
    space_id: WireUuid,
}

impl BrainEvalHarness {
    /// Open a fresh connection with a fresh space id, minting its data-plane
    /// key against an external server's admin plane (resolved from the
    /// environment — see [`KeyMinter::from_external_env`]). Drives the
    /// HELLO → WELCOME → AUTH → AUTH_OK handshake before returning.
    pub async fn connect(addr: SocketAddr) -> Result<Self, HarnessError> {
        let space_id = brain_db_sdk::new_id();
        Self::connect_with_space(addr, space_id).await
    }

    /// [`connect_with_space`](Self::connect_with_space), but minting the key
    /// into `namespace` instead of the environment's default.
    ///
    /// For scenarios that operate on a namespace they mint themselves —
    /// SCHEMA_UPLOAD is additive and persistent, so each run needs a fresh
    /// one. Brain authorizes a namespace-scoped write against the namespace
    /// the *key* is bound to, so connecting with the default `eval` key and
    /// then writing to `e2_<hex>` is a `PermissionDenied`, not a valid
    /// setup. The space id is taken as an argument because the namespace
    /// name is usually derived from it.
    pub async fn connect_in_namespace(
        addr: SocketAddr,
        space_id: WireUuid,
        namespace: impl Into<String>,
    ) -> Result<Self, HarnessError> {
        let minter = KeyMinter::from_external_env(addr).with_namespace(namespace);
        Self::connect_minted(&minter, addr, space_id).await
    }

    /// Connect with an explicit space id against an external server, minting
    /// its key from the environment. For a server the eval owns, prefer
    /// [`connect_minted`](Self::connect_minted) with the handle's
    /// [`KeyMinter`].
    pub async fn connect_with_space(
        addr: SocketAddr,
        space_id: WireUuid,
    ) -> Result<Self, HarnessError> {
        let minter = KeyMinter::from_external_env(addr);
        Self::connect_minted(&minter, addr, space_id).await
    }

    /// Mint a data-plane key for `space_id` through `minter`, then connect
    /// with it. The space id is what the key is minted *for* — it rides the
    /// token, not the client config (Brain assigns identity from auth).
    pub async fn connect_minted(
        minter: &KeyMinter,
        addr: SocketAddr,
        space_id: WireUuid,
    ) -> Result<Self, HarnessError> {
        let token = minter.mint(space_id).await?;
        Self::connect_with_token(addr, space_id, token).await
    }

    /// Connect with an already-minted token. The credential is the
    /// connection's whole identity; `space_id` is retained only so callers
    /// can correlate the harness with the space its key was minted for.
    pub async fn connect_with_token(
        addr: SocketAddr,
        space_id: WireUuid,
        token: String,
    ) -> Result<Self, HarnessError> {
        let mut config = ClientConfig::new(Auth::Token(token.into_bytes()));
        config.request_timeout = Some(REQUEST_TIMEOUT);
        let client = BrainClient::connect_with(addr, config).await?;
        Ok(Self { client, space_id })
    }

    /// The space id this harness was bound to.
    #[must_use]
    pub fn space_id(&self) -> WireUuid {
        self.space_id
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
        let mut stored_texts: Vec<String> = Vec::new();
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
            let mut b = EncodeBuilder::new(turn.content.as_str());
            if let Some(ts) = turn.occurred_at_unix_nanos {
                b = b.occurred_at(ts);
            }
            let request = b.build();
            let resp = self.client.encode(&request).await?;
            if resp.was_deduplicated {
                deduplicated += 1;
            } else {
                stored_ids.push(resp.memory_id);
                stored_texts.push(turn.content.clone());
            }
        }

        let latency_ms = elapsed_ms(start);
        Ok(IngestOutcome {
            stored_ids,
            stored_texts,
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
    pub async fn recall(&self, cue: &str, max_results: u32) -> Result<RecallOutcome, HarnessError> {
        self.recall_scoped(cue, max_results, RecallScopeWire::Space)
            .await
    }

    /// Run a RECALL at an explicit [`RecallScopeWire`]. `Space` (the default via
    /// [`Self::recall`]) searches this harness's own space; `Namespace` fans
    /// out across every space in the caller's namespace (never crossing
    /// namespaces). The namespace-wide eval scenario uses this to compare the
    /// two read scopes over one persona split across spaces.
    pub async fn recall_scoped(
        &self,
        cue: &str,
        max_results: u32,
        scope: RecallScopeWire,
    ) -> Result<RecallOutcome, HarnessError> {
        let start = Instant::now();
        let request = RecallBuilder::new(cue)
            .max_results(max_results)
            .include_text(true)
            .scope(scope)
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
    /// Source turn text for each id in `stored_ids`, same order and length —
    /// lets a caller re-fetch a stored memory's extracted graph by recalling
    /// its own text and matching the id.
    pub stored_texts: Vec<String>,
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
