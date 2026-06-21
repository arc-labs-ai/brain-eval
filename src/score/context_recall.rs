//! Answer-supporting context recall — the headline retrieval metric.
//!
//! Substring recall@k (see [`crate::score::retrieval`]) asks "does the
//! gold answer string appear in a retrieved memory?" — which rewards
//! lexical overlap and punishes correct-but-paraphrased retrieval. That
//! gap is the whole point of the Kamalloo 2023 / NoLiMa critiques: a
//! retriever can score 0 on substring recall while having returned
//! exactly the memory that supports the answer.
//!
//! This metric instead asks an LLM judge a retrieval-only question: *can
//! the gold answer be derived ENTIRELY from these retrieved memories?*
//! It measures whether Brain surfaced the supporting context, decoupled
//! from whether the synthesizer then phrased the answer well. That makes
//! it the honest "did we retrieve the right memory" number.
//!
//! Per-question support is recorded as
//! [`crate::core::outcome::QuestionResult::context_supported`] — `None`
//! when no LLM judge was configured (the support judge needs the LLM, so
//! a heuristic run leaves it unjudged rather than guessing).

use serde::{Deserialize, Serialize};

use crate::core::outcome::QuestionResult;

/// Answer-supporting context recall over a run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextRecallStats {
    /// Fraction of *judged* questions whose retrieved memories fully
    /// support the gold answer. `n_supported / n_judged`.
    pub supported_rate: f64,
    /// Questions the support judge actually graded (those with a judge
    /// configured and a retrieved set to grade).
    pub n_judged: usize,
    /// Of those, how many had answer-supporting context.
    pub n_supported: usize,
}

/// Aggregate answer-supporting context recall. Returns `None` when no
/// question carries a support verdict (heuristic-only run, or no judge),
/// so the report can fall back to showing substring recall as the only
/// retrieval signal rather than printing a misleading `0.0`.
#[must_use]
pub fn compute_context_recall(results: &[QuestionResult]) -> Option<ContextRecallStats> {
    let mut n_judged = 0usize;
    let mut n_supported = 0usize;
    for r in results {
        if let Some(supported) = r.context_supported {
            n_judged += 1;
            if supported {
                n_supported += 1;
            }
        }
    }
    if n_judged == 0 {
        return None;
    }
    #[allow(clippy::cast_precision_loss)]
    let supported_rate = n_supported as f64 / n_judged as f64;
    Some(ContextRecallStats {
        supported_rate,
        n_judged,
        n_supported,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::instance::QuestionType;
    use crate::core::outcome::Verdict;

    fn qr(context_supported: Option<bool>) -> QuestionResult {
        QuestionResult {
            question_id: "q".into(),
            question_type: QuestionType::SingleHop,
            question: String::new(),
            ground_truth: String::new(),
            system_answer: String::new(),
            answer_kind: "single".into(),
            verdict: Verdict::Correct,
            score: 1.0,
            write_latency_ms: 0,
            read_latency_ms: 0,
            tokens_write: 0,
            tokens_read: 0,
            memories_retrieved: 0,
            retrieved_memory_contents: Vec::new(),
            judge_reasoning: String::new(),
            context_supported,
            context_support_reasoning: String::new(),
            ingestion_failed: false,
            retrieval_failed: false,
            write_attempted: 0,
            write_stored: 0,
            write_deduplicated: 0,
        }
    }

    #[test]
    fn none_when_no_question_is_judged() {
        let results = vec![qr(None), qr(None)];
        assert!(compute_context_recall(&results).is_none());
    }

    #[test]
    fn rate_counts_only_judged_questions() {
        // 2 supported, 1 not, 1 unjudged => 2/3, not 2/4.
        let results = vec![qr(Some(true)), qr(Some(true)), qr(Some(false)), qr(None)];
        let s = compute_context_recall(&results).expect("some judged");
        assert_eq!(s.n_judged, 3);
        assert_eq!(s.n_supported, 2);
        assert!((s.supported_rate - 2.0 / 3.0).abs() < 1e-9);
    }
}
