//! Agent manifest — the `conversation_key → agent_id` map that lets an
//! eval **ingest once and read many times** without re-ingesting.
//!
//! Each benchmark conversation is ingested under its own agent id (so a
//! question can only retrieve its own haystack). A normal run mints a
//! fresh agent per conversation and throws it away. With
//! `BRAIN_EVAL_PERSIST_AGENTS=<path>` the run records each
//! `conversation_key → agent_id` here; a later run with
//! `BRAIN_EVAL_READ_ONLY_AGENTS=<path>` reconnects to those same agents,
//! **skips ingest entirely**, and runs only retrieval + judging against
//! the already-populated store. That makes read-path iteration free of
//! the (slow, sometimes rate-limited) write pipeline.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use brain_db_sdk::wire::types::WireUuid;

/// Persisted `conversation_key → agent_id` map plus the context it was
/// captured in (so a mismatched reuse is at least visible).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentManifest {
    /// Benchmark id the agents were ingested for (e.g. `longmemeval-s`).
    pub benchmark: String,
    /// Endpoint the data was ingested into. Advisory — the read-only run
    /// uses its own configured endpoint; this just documents provenance.
    pub endpoint: String,
    /// `conversation_key → 32-char lowercase hex of the 16-byte agent id`.
    pub agents: BTreeMap<String, String>,
}

impl AgentManifest {
    /// A new, empty manifest tagged with the benchmark id and endpoint it
    /// was produced against.
    #[must_use]
    pub fn new(benchmark: impl Into<String>, endpoint: impl Into<String>) -> Self {
        Self {
            benchmark: benchmark.into(),
            endpoint: endpoint.into(),
            agents: BTreeMap::new(),
        }
    }

    /// Record the agent a conversation's haystack was ingested under.
    pub fn record(&mut self, conversation_key: &str, agent: WireUuid) {
        self.agents
            .insert(conversation_key.to_owned(), hex16(&agent));
    }

    /// The agent id for a conversation key, if recorded and well-formed.
    #[must_use]
    pub fn agent_for(&self, conversation_key: &str) -> Option<WireUuid> {
        self.agents.get(conversation_key).and_then(|h| unhex16(h))
    }

    /// Number of conversations recorded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.agents.len()
    }

    /// Whether no conversations are recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    /// Write the manifest as pretty JSON, creating parent dirs.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Load a manifest written by a prior `--persist-agents` run.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

/// 16 bytes → 32-char lowercase hex.
fn hex16(b: &[u8; 16]) -> String {
    let mut s = String::with_capacity(32);
    for x in b {
        s.push_str(&format!("{x:02x}"));
    }
    s
}

/// 32-char hex → 16 bytes, or `None` on bad length / non-hex.
fn unhex16(s: &str) -> Option<WireUuid> {
    if s.len() != 32 {
        return None;
    }
    let mut out = [0u8; 16];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = u8::from_str_radix(s.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trips() {
        let id: WireUuid = [
            0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54,
            0x32, 0x10,
        ];
        let h = hex16(&id);
        assert_eq!(h.len(), 32);
        assert_eq!(unhex16(&h), Some(id));
    }

    #[test]
    fn unhex_rejects_malformed() {
        assert_eq!(unhex16("abc"), None); // wrong length
        assert_eq!(unhex16(&"zz".repeat(16)), None); // non-hex
    }

    #[test]
    fn manifest_record_and_lookup() {
        let mut m = AgentManifest::new("longmemeval-s", "127.0.0.1:8080");
        let agent: WireUuid = [7u8; 16];
        m.record("conv-1", agent);
        assert_eq!(m.agent_for("conv-1"), Some(agent));
        assert_eq!(m.agent_for("missing"), None);
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn manifest_save_load_round_trip() {
        let path = std::env::temp_dir().join(format!(
            "brain_eval_agents_test_{}.json",
            std::process::id()
        ));
        let mut m = AgentManifest::new("locomo", "127.0.0.1:18080");
        m.record("conv-a", [1u8; 16]);
        m.record("conv-b", [2u8; 16]);
        m.save(&path).unwrap();
        let loaded = AgentManifest::load(&path).unwrap();
        assert_eq!(loaded.benchmark, "locomo");
        assert_eq!(loaded.agent_for("conv-a"), Some([1u8; 16]));
        assert_eq!(loaded.agent_for("conv-b"), Some([2u8; 16]));
        let _ = std::fs::remove_file(&path);
    }
}
