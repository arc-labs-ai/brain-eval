//! Per-question outcomes — judge verdicts and the runner's
//! [`QuestionResult`] record. Shared across the run / score / report
//! pipeline stages so they live in `core` rather than inside any
//! single stage.

use serde::{Deserialize, Serialize};

use crate::core::instance::QuestionType;

// ---------------------------------------------------------------------------
// Judge verdict
// ---------------------------------------------------------------------------

/// Three-state verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Full credit — answer is correct.
    Correct,
    /// Half credit — answer is partially correct (mentions the right
    /// entity but misses detail, off-by-a-unit, etc.).
    Partial,
    /// Zero credit — answer is wrong, or absent when one was expected.
    Incorrect,
}

impl Verdict {
    /// Numeric score for accuracy aggregation.
    #[must_use]
    pub fn score(self) -> f64 {
        match self {
            Self::Correct => 1.0,
            Self::Partial => 0.5,
            Self::Incorrect => 0.0,
        }
    }
}

/// Judge output for a single question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeResult {
    /// Question id this verdict belongs to.
    pub question_id: String,
    /// Verdict.
    pub verdict: Verdict,
    /// Numeric score (`verdict.score()`).
    pub score: f64,
    /// Free-text reasoning (heuristic explanation or LLM rationale).
    pub reasoning: String,
}

// ---------------------------------------------------------------------------
// Per-question runner record
// ---------------------------------------------------------------------------

/// Per-question record produced by the runner; the JSON reporter
/// emits a `per_question` array of these.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionResult {
    /// Stable id from the dataset.
    pub question_id: String,
    /// Evaluation dimension.
    pub question_type: QuestionType,
    /// Question text.
    pub question: String,
    /// Ground-truth answer from the dataset.
    pub ground_truth: String,
    /// Candidate answer produced by [`crate::run::synthesize::synthesize_answer`].
    pub system_answer: String,
    /// Judge verdict.
    pub verdict: Verdict,
    /// Numeric score (0.0 / 0.5 / 1.0).
    pub score: f64,
    /// Total ingest latency for this conversation (amortised — every
    /// question in a conversation reports the same value).
    pub write_latency_ms: u64,
    /// RECALL latency for this question.
    pub read_latency_ms: u64,
    /// Tokens spent during ingestion (server-reported when wired;
    /// `0` until the wire grows the field).
    pub tokens_write: u64,
    /// Tokens spent during retrieval (same caveat).
    pub tokens_read: u64,
    /// Number of memories returned by the RECALL.
    pub memories_retrieved: usize,
    /// Text content of retrieved memories (for downstream Recall@K
    /// computation).
    pub retrieved_memory_contents: Vec<String>,
    /// Free-text reasoning from the judge.
    pub judge_reasoning: String,
    /// `true` if the ingest pipeline returned an error.
    pub ingestion_failed: bool,
    /// `true` if the RECALL call returned an error.
    pub retrieval_failed: bool,
    /// ENCODE attempts during conversation ingestion (amortised).
    pub write_attempted: u64,
    /// ENCODEs that produced fresh memories.
    pub write_stored: u64,
    /// ENCODEs that hit the fingerprint dedupe path.
    pub write_deduplicated: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdict_scores_round_trip() {
        assert!((Verdict::Correct.score() - 1.0).abs() < f64::EPSILON);
        assert!((Verdict::Partial.score() - 0.5).abs() < f64::EPSILON);
        assert!(Verdict::Incorrect.score().abs() < f64::EPSILON);
    }
}
