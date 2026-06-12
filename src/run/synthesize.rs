//! Answer synthesis — turning retrieved memories into a candidate
//! answer string.
//!
//! Two synthesizers:
//!
//! - **Heuristic** ([`synthesize_answer`], always available): concatenates
//!   the top-K memory texts as a numbered list. Fine for the substring
//!   judge on fact-style benchmarks, but it buries exact facts (dates,
//!   counts) in raw dialogue, which caps LoCoMo / LongMemEval accuracy.
//! - **LLM** ([`LlmSynthesizer`], behind `live-llm` + a key): asks a model
//!   to compose a concise answer from the retrieved snippets, or to decline
//!   when they don't contain it. This is the answer the (LLM) judge then
//!   grades. Falls back to the heuristic on a failed call.

use brain_db_sdk::wire::types::MemoryResult;

use crate::core::instance::QuestionType;

/// Build a candidate answer from the top-K retrieved memories.
///
/// `top_k` clamps the number of memories considered. Empty input or
/// abstention questions return a "don't know" sentinel.
#[must_use]
pub fn synthesize_answer(
    question: &str,
    memories: &[MemoryResult],
    qtype: QuestionType,
    top_k: usize,
) -> String {
    let _ = question; // reserved for future LLM synthesizer
    if memories.is_empty() {
        return match qtype {
            QuestionType::Abstention => "I don't know.".to_owned(),
            _ => "I don't know.".to_owned(),
        };
    }

    let cap = memories.len().min(top_k.max(1));
    let mut out = String::new();
    for (i, m) in memories.iter().take(cap).enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&format!("{}. {}", i + 1, m.text));
    }
    out
}

/// Format the top-K memory snippets as a numbered block for a prompt.
#[cfg(feature = "live-llm")]
fn memory_block(memories: &[MemoryResult], top_k: usize) -> String {
    let cap = memories.len().min(top_k.max(1));
    let mut out = String::new();
    for (i, m) in memories.iter().take(cap).enumerate() {
        out.push_str(&format!("{}. {}\n", i + 1, m.text));
    }
    out
}

/// LLM answer synthesizer: composes a concise answer from the retrieved
/// memories. Compiled only under `live-llm`; falls back to
/// [`synthesize_answer`] on a failed call.
#[cfg(feature = "live-llm")]
pub struct LlmSynthesizer {
    client: crate::llm::LlmClient,
    warned: std::sync::atomic::AtomicBool,
}

#[cfg(feature = "live-llm")]
impl LlmSynthesizer {
    /// Build from the environment, or `None` if no provider key is set.
    /// Model overridable via `BRAIN_EVAL_SYNTH_MODEL`.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        Some(Self {
            client: crate::llm::LlmClient::from_env("BRAIN_EVAL_SYNTH_MODEL")?,
            warned: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// `llm:<provider>:<model>` identity for logging.
    #[must_use]
    pub fn describe(&self) -> String {
        format!("llm:{}", self.client.describe())
    }

    /// Compose an answer from the top-K memories. Empty input → "I don't
    /// know."; a failed/empty LLM reply → the heuristic concatenation.
    pub async fn synthesize(
        &self,
        question: &str,
        memories: &[MemoryResult],
        qtype: QuestionType,
        top_k: usize,
    ) -> String {
        if memories.is_empty() {
            return "I don't know.".to_owned();
        }
        let prompt = format!(
            "Answer the question using ONLY the retrieved memory snippets below. \
             Be concise and direct — give just the answer, not an explanation. \
             If the snippets do not contain the answer, reply exactly \"I don't know.\"\n\n\
             Question: {question}\n\n\
             Memories:\n{}\n\
             Answer:",
            memory_block(memories, top_k)
        );
        match self.client.complete(&prompt, 512).await {
            Ok(answer) if !answer.trim().is_empty() => answer.trim().to_owned(),
            Ok(_) => {
                self.warn_once("empty reply");
                synthesize_answer(question, memories, qtype, top_k)
            }
            Err(e) => {
                self.warn_once(&e);
                synthesize_answer(question, memories, qtype, top_k)
            }
        }
    }

    fn warn_once(&self, message: &str) {
        use std::sync::atomic::Ordering;
        tracing::warn!(error = %message, "llm synthesizer failed; heuristic fallback");
        if !self.warned.swap(true, Ordering::Relaxed) {
            eprintln!(
                "warning: LLM synthesizer call failed ({message}). Falling back to the \
                 raw top-K concatenation for unsynthesized answers. Check the API key / \
                 credit balance, or set BRAIN_EVAL_SYNTH_MODEL."
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_db_sdk::wire::types::{MemoryKindWire, MemoryResult};

    fn mem(text: &str) -> MemoryResult {
        MemoryResult {
            memory_id: 0,
            text: text.to_owned(),
            similarity_score: 0.0,
            confidence: 0.0,
            salience: 0.0,
            kind: MemoryKindWire::Episodic,
            agent_id: [0u8; 16],
            context_id: 0,
            created_at_unix_nanos: 0,
            last_accessed_at_unix_nanos: 0,
            edges: None,
            contributing_retrievers: Vec::new(),
            fused_score: 0.0,
            rerank_score: None,
            salience_initial: 0.0,
            access_count: 0,
            lsn: 0,
            flags: 0,
            consolidated_at_unix_nanos: None,
            edges_out_count: 0,
            edges_in_count: 0,
            graph: None,
        }
    }

    #[test]
    fn empty_memories_yields_dont_know() {
        let a = synthesize_answer("q", &[], QuestionType::SingleHop, 5);
        assert!(a.to_lowercase().contains("don't know"));
    }

    #[test]
    fn abstention_with_empty_memories_yields_dont_know() {
        let a = synthesize_answer("q", &[], QuestionType::Abstention, 5);
        assert!(a.to_lowercase().contains("don't know"));
    }

    #[test]
    fn concatenates_top_k() {
        let m = vec![mem("Paris"), mem("Berlin"), mem("Rome")];
        let a = synthesize_answer("q", &m, QuestionType::SingleHop, 2);
        assert!(a.contains("Paris"));
        assert!(a.contains("Berlin"));
        assert!(!a.contains("Rome"));
    }
}
