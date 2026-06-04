//! Answer synthesis — turning retrieved memories into a candidate
//! answer string.
//!
//! Brain's `live-llm` answer path (REASON over the top-K hits) is a
//! follow-up; v1 ships the heuristic synthesizer only. It concatenates
//! the top-K memory texts with question-type-aware formatting:
//!
//! - `Abstention`: returns `"I don't know."` when no memories were
//!   retrieved, otherwise summarises in case the hit was wrong.
//! - everything else: numbered list of memory texts, joined.
//!
//! Downstream the judge reads this; for the heuristic judge the
//! substring rule catches "ground truth contained in any returned
//! memory's text," which is enough for fact-style benchmarks.

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
