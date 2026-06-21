//! Answer synthesis — turning a RECALL [`RecallOutcome`] into a
//! candidate answer string the judge can grade.
//!
//! The read path is a smart router that returns memories, not a flat
//! top-k list. Synthesis is therefore uniform over whatever the router
//! surfaced:
//!
//! - **Abstention** (`None`, empty `memories`): the router honestly has
//!   no answer → "I don't know." We never fabricate one.
//! - **Single / Many**: the router returned the memories that answer the
//!   cue. Synthesis composes the answer from their text:
//!   - **Heuristic** ([`synthesize_answer`]): concatenates the memory
//!     texts as a numbered list. Cheap and deterministic.
//!   - **LLM** ([`LlmSynthesizer`], behind `live-llm` + a key): composes
//!     a concise answer from the snippets, or declines when they don't
//!     contain it. Falls back to the heuristic on a failed call.

use brain_db_sdk::wire::types::MemoryResult;

use crate::core::instance::QuestionType;
use crate::run::harness::RecallOutcome;

/// Sentinel answer for "no answer available".
const DONT_KNOW: &str = "I don't know.";

/// Build a candidate answer from a RECALL outcome (heuristic path).
///
/// Abstention (no memories) → "I don't know."; otherwise the top-`cap`
/// memory texts as a numbered list.
#[must_use]
pub fn synthesize_answer(
    question: &str,
    outcome: &RecallOutcome,
    qtype: QuestionType,
    episodic_cap: usize,
) -> String {
    let _ = (question, qtype); // reserved; the shape decides the branch

    if outcome.memories.is_empty() {
        return DONT_KNOW.to_owned();
    }
    numbered_block(&outcome.memories, episodic_cap)
}

/// Top-`cap` memory texts as a `1. … 2. …` numbered list.
fn numbered_block(results: &[MemoryResult], cap: usize) -> String {
    let n = results.len().min(cap.max(1));
    let mut out = String::new();
    for (i, m) in results.iter().take(n).enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&format!("{}. {}", i + 1, m.text));
    }
    out
}

/// Format the top-`cap` memory snippets as a numbered block for a prompt.
#[cfg(feature = "live-llm")]
fn memory_block(results: &[MemoryResult], cap: usize) -> String {
    let n = results.len().min(cap.max(1));
    let mut out = String::new();
    for (i, m) in results.iter().take(n).enumerate() {
        out.push_str(&format!("{}. {}\n", i + 1, m.text));
    }
    out
}

/// LLM answer synthesizer for the episodic path. Compiled only under
/// `live-llm`; grounded answers and abstentions bypass it entirely
/// (the stored value / honest decline is already the answer).
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

    /// Compose an answer from a RECALL outcome.
    ///
    /// Abstention (no memories) → "I don't know."; otherwise an
    /// LLM-composed answer over the returned memory snippets, falling
    /// back to the heuristic numbered block on a failed / empty reply.
    pub async fn synthesize(
        &self,
        question: &str,
        outcome: &RecallOutcome,
        qtype: QuestionType,
        episodic_cap: usize,
    ) -> String {
        if outcome.memories.is_empty() {
            return DONT_KNOW.to_owned();
        }
        let prompt = format!(
            "Answer the question using the retrieved memory snippets below. \
             Each snippet is prefixed with the date it was said, e.g. \
             \"[8 May, 2023] Alice: ...\".\n\n\
             Rules:\n\
             - Give just the answer, concise and direct — no explanation.\n\
             - USE ALL MEMORIES: the answer may be in any snippet, not just the \
             first. Read every snippet before answering; the relevant one is not \
             necessarily at the top.\n\
             - REASON over the memories. When they clearly imply the answer, give \
             it rather than abstaining. Apply ordinary reasoning:\n\
             - TIME: resolve relative time expressions (\"yesterday\", \"last \
             week\", \"this month\", \"last year\", \"the week before X\", \"the \
             third month\", \"N years/months ago\") against the bracketed date of \
             the snippet that states them, and answer with the absolute date. \
             Example: \"this month\" said on [3 July, 2023] means July 2023; \
             \"4 years ago\" said in 2023 means 2019.\n\
             - WORLD KNOWLEDGE: apply common knowledge to interpret the memories \
             (e.g. a season or equinox maps to a month; a holiday maps to a date). \
             Use it only to interpret what the memories say, never to invent facts \
             not grounded in them — and do NOT manufacture a falsely-precise value \
             (an exact day or number) the memories do not state; if only an \
             approximate value is supported, give the approximate, not a guess.\n\
             - ANSWER SLOT: the answer must match what the question asks for — \
             \"where\" wants a place, \"when\" a date/time, \"who\" a person, \
             \"which/what\" the named thing. If no snippet provides a value of \
             that kind, reply \"I don't know.\" rather than substituting a \
             different kind of fact.\n\
             - INFERENCE: combine facts stated across multiple snippets to derive \
             an answer the memories jointly support.\n\
             - LISTS: when the question asks for multiple things or a set (e.g. \
             \"what activities\", \"where has she ...\", \"which ...\"), gather EVERY \
             matching item across ALL snippets and answer with the full \
             comma-separated list, not just the first one.\n\
             - FAITHFULNESS: do not fabricate. Base the answer on the memories; \
             do not add facts they do not support.\n\
             - Reply exactly \"I don't know.\" ONLY when none of the snippets \
             support an answer.\n\n\
             Question: {question}\n\n\
             Memories:\n{}\n\
             Answer:",
            memory_block(&outcome.memories, episodic_cap)
        );
        match self.client.complete(&prompt, 512).await {
            Ok(answer) if !answer.trim().is_empty() => answer.trim().to_owned(),
            Ok(_) => {
                self.warn_once("empty reply");
                synthesize_answer(question, outcome, qtype, episodic_cap)
            }
            Err(e) => {
                self.warn_once(&e);
                synthesize_answer(question, outcome, qtype, episodic_cap)
            }
        }
    }

    fn warn_once(&self, message: &str) {
        use std::sync::atomic::Ordering;
        tracing::warn!(error = %message, "llm synthesizer failed; heuristic fallback");
        if !self.warned.swap(true, Ordering::Relaxed) {
            eprintln!(
                "warning: LLM synthesizer call failed ({message}). Falling back to the \
                 raw top-K concatenation for episodic answers. Check the API key / \
                 credit balance, or set BRAIN_EVAL_SYNTH_MODEL."
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_db_sdk::wire::types::{AnswerKindWire, MemoryKindWire, MemoryResult};

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
            occurred_at_unix_nanos: None,
            edges_out_count: 0,
            edges_in_count: 0,
            graph: None,
        }
    }

    fn memories(kind: AnswerKindWire, texts: &[&str]) -> RecallOutcome {
        RecallOutcome {
            answer_kind: kind,
            memories: texts.iter().map(|t| mem(t)).collect(),
            latency_ms: 0,
        }
    }

    fn abstain() -> RecallOutcome {
        RecallOutcome {
            answer_kind: AnswerKindWire::None,
            memories: Vec::new(),
            latency_ms: 0,
        }
    }

    #[test]
    fn abstention_yields_dont_know() {
        let a = synthesize_answer("q", &abstain(), QuestionType::SingleHop, 5);
        assert!(a.to_lowercase().contains("don't know"));
    }

    #[test]
    fn single_renders_the_one_memory() {
        let a = synthesize_answer(
            "where?",
            &memories(AnswerKindWire::Single, &["Berlin"]),
            QuestionType::SingleHop,
            5,
        );
        assert!(a.contains("Berlin"));
    }

    #[test]
    fn many_concatenates_top_k() {
        let a = synthesize_answer(
            "q",
            &memories(AnswerKindWire::Many, &["Paris", "Berlin", "Rome"]),
            QuestionType::SingleHop,
            2,
        );
        assert!(a.contains("Paris"));
        assert!(a.contains("Berlin"));
        assert!(!a.contains("Rome"));
    }

    #[test]
    fn empty_is_dont_know() {
        let a = synthesize_answer(
            "q",
            &memories(AnswerKindWire::None, &[]),
            QuestionType::SingleHop,
            5,
        );
        assert!(a.to_lowercase().contains("don't know"));
    }
}
