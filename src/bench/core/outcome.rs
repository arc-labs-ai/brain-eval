//! Per-question outcomes — judge verdicts and the runner's
//! [`QuestionResult`] record. Shared across the run / score / report
//! pipeline stages so they live in `core` rather than inside any
//! single stage.

use serde::{Deserialize, Serialize};

use crate::bench::core::instance::QuestionType;

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
    /// NOT GRADED — no answer was produced to grade.
    ///
    /// Emitted only by a retrieval-only run, which skips synthesis and both
    /// judges. It is a fourth state rather than a `bool` beside the verdict
    /// because the compiler then forces every consumer to decide what it
    /// means; a flag can be ignored silently, and the one that matters —
    /// accuracy — would have read these as failures.
    ///
    /// It is deliberately NOT zero credit. `Incorrect` means the system got
    /// it wrong; this means nobody asked.
    NotJudged,
}

impl Verdict {
    /// Numeric score for accuracy aggregation.
    ///
    /// `NotJudged` scores 0.0 only because the type must return something —
    /// callers aggregating accuracy must filter it out with [`Self::is_graded`]
    /// FIRST, or an ungraded run reads as a run where everything was wrong.
    #[must_use]
    pub fn score(self) -> f64 {
        match self {
            Self::Correct => 1.0,
            Self::Partial => 0.5,
            Self::Incorrect | Self::NotJudged => 0.0,
        }
    }

    /// Whether this verdict carries a judgement at all.
    ///
    /// The guard for every accuracy aggregation: an ungraded question belongs
    /// in neither the numerator nor the denominator.
    #[must_use]
    pub const fn is_graded(self) -> bool {
        !matches!(self, Self::NotJudged)
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
// Answer shape
// ---------------------------------------------------------------------------

/// How the router shaped an answer. Mirrors
/// `brain_db_sdk::wire::types::AnswerKindWire`'s three real shapes
/// (`Single` / `Many` / `None`), plus an eval-only `Error` state the wire
/// type has no way to represent — a question whose ingest or retrieval
/// failed before the router ever ran. Kept as a real enum (not the bare
/// `String` this used to be) so a typo in a match arm is a compile error,
/// not a silently-miscounted metrics bucket. `core` stays free of any SDK
/// dependency — the `AnswerKindWire` → `AnswerKind` conversion lives in
/// `run::runner`, where the wire type is already in scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerKind {
    /// The router committed to one memory.
    Single,
    /// The router returned a set of memories.
    Many,
    /// The router honestly declined — no supporting memory found.
    None,
    /// Eval-only: ingest or retrieval failed before the router ran.
    Error,
}

impl std::fmt::Display for AnswerKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Single => "single",
            Self::Many => "many",
            Self::None => "none",
            Self::Error => "error",
        })
    }
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
    /// Candidate answer produced by [`crate::bench::synthesize::synthesize_answer`].
    pub system_answer: String,
    /// How the router shaped this answer. Lets the report split accuracy
    /// and precision by whether the router returned one memory, a set, or
    /// honestly declined — the read path is a smart router, not a flat
    /// top-k list.
    pub answer_kind: AnswerKind,
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
    /// Whether the retrieved memories fully support the gold answer, per
    /// the support judge (answer-supporting context recall). `None` when
    /// no LLM judge was configured — the support verdict needs the LLM,
    /// so a heuristic run leaves it unjudged rather than guessing.
    pub context_supported: Option<bool>,
    /// Free-text reasoning behind `context_supported`.
    pub context_support_reasoning: String,
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
