//! Aggregate metrics shape + the top-level `compute_full_metrics`
//! entry point that turns a `Vec<QuestionResult>` into an
//! `EvalMetrics`.
//!
//! This file owns the **aggregate accuracy + token + write-quality**
//! computation. Latency and retrieval-quality stats live in their own
//! sibling files ([`crate::score::latency`], [`crate::score::retrieval`])
//! because they grow on different cadences.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::core::instance::QuestionType;
use crate::core::outcome::{JudgeResult, QuestionResult};
use crate::score::context_recall::{compute_context_recall, ContextRecallStats};
use crate::score::latency::{compute_latency_stats, LatencyStats};
use crate::score::retrieval::{compute_retrieval_stats, RetrievalStats};

/// Full aggregate metrics for one benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalMetrics {
    /// Mean score across all questions (1.0 / 0.5 / 0.0).
    pub accuracy: f64,
    /// Per-dimension accuracy breakdown, keyed by [`QuestionType::tag`].
    pub per_dimension: HashMap<String, DimensionMetrics>,
    /// Total questions evaluated.
    pub total_questions: usize,
    /// Questions scored 1.0.
    pub correct: usize,
    /// Questions scored 0.5.
    pub partial: usize,
    /// Questions scored 0.0.
    pub incorrect: usize,
    /// Write + read latency percentiles.
    pub latency: LatencyStats,
    /// Token usage summary.
    pub tokens: TokenStats,
    /// Answer-supporting context recall — the headline retrieval metric.
    /// Asks an LLM judge whether the retrieved memories fully support the
    /// gold answer, decoupled from how the synthesizer phrased it. `None`
    /// on a heuristic-only run (the support judge needs the LLM). This
    /// replaces substring recall@k as the retrieval headline.
    pub context_recall: Option<ContextRecallStats>,
    /// Substring recall@k (DEPRECATED diagnostic — see Kamalloo 2023 /
    /// NoLiMa). Rewards lexical overlap between the gold answer string and
    /// a retrieved memory, so it scores 0 on correct-but-paraphrased
    /// retrieval. Kept only as a diagnostic against the headline
    /// `context_recall`; do not read it as "did we retrieve the answer".
    /// `None` when no question returned any memories (all abstentions).
    pub retrieval: Option<RetrievalStats>,
    /// Accuracy and precision split by the router's answer shape
    /// (one memory / a set / honest abstention). This is how the read
    /// path actually behaves now that recall is a smart router, not a
    /// flat top-k list.
    pub answer_shape: AnswerShapeMetrics,
    /// Questions where session ingestion failed (infrastructure error,
    /// not model behaviour).
    pub ingestion_errors: usize,
    /// Questions where retrieval failed (infrastructure error).
    pub retrieval_errors: usize,
    /// Encode-side quality (accepted / merged / discarded counters).
    pub write_quality: WriteQualityMetrics,
}

/// Accuracy + precision split by the router's answer shape.
///
/// The headline `EvalMetrics::accuracy` answers "how often is the system
/// right". This breakdown answers the two questions that matter for a
/// memory database: *when it commits to an answer, is it ever wrong*
/// (precision — the hard invariant) and *what shape did the router
/// choose* (one memory / a set / honest abstention).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnswerShapeMetrics {
    /// `single` answers returned (the router surfaced exactly one memory).
    pub single: usize,
    /// Mean score among `single` answers.
    pub single_accuracy: f64,
    /// `many` answers returned (the router surfaced a set of memories).
    pub many: usize,
    /// Mean score among `many` answers.
    pub many_accuracy: f64,
    /// Honest abstentions returned (`none` — the router surfaced nothing).
    pub abstained: usize,
    /// Mean score among abstentions (correct when the truth was
    /// genuinely unanswerable, wrong when the system gave up on an
    /// answerable question).
    pub abstained_accuracy: f64,
    /// Questions whose RECALL errored (infrastructure, not model).
    pub errored: usize,
    /// Committed answers: `single` or `many` where the system actually
    /// asserted something (did not decline).
    pub committed: usize,
    /// Of committed answers, the fraction that were NOT scored
    /// `Incorrect`. The hard invariant: a committed answer must never be
    /// confidently wrong. `1.0` = the system never asserted a falsehood.
    pub committed_precision: f64,
}

/// Accuracy for a single evaluation dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionMetrics {
    /// Mean score for this dimension.
    pub accuracy: f64,
    /// Total questions in this dimension.
    pub count: usize,
    /// Questions scored 1.0.
    pub correct: usize,
    /// Questions scored 0.5.
    pub partial: usize,
    /// Questions scored 0.0.
    pub incorrect: usize,
}

/// Token usage summary across all questions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenStats {
    /// Average write tokens per question.
    pub write_avg: f64,
    /// Average read tokens per question.
    pub read_avg: f64,
    /// Average total tokens per question.
    pub total_avg: f64,
    /// Grand total across the entire run.
    pub grand_total: u64,
}

/// Encode-side quality summary.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WriteQualityMetrics {
    /// Total ENCODE attempts across all questions.
    pub total_attempted: u64,
    /// Total ENCODEs that produced a fresh memory.
    pub total_stored: u64,
    /// Total ENCODEs that hit the fingerprint dedupe path.
    pub total_deduplicated: u64,
    /// `total_stored / total_attempted`.
    pub store_rate: f64,
    /// `total_deduplicated / total_attempted`.
    pub dedup_rate: f64,
}

// ---------------------------------------------------------------------------
// compute_full_metrics
// ---------------------------------------------------------------------------

/// Compute aggregate metrics from a slice of `QuestionResult`.
#[must_use]
pub fn compute_full_metrics(results: &[QuestionResult]) -> EvalMetrics {
    let total_questions = results.len();
    let scores: Vec<f64> = results.iter().map(|r| r.score).collect();

    #[allow(clippy::float_cmp)]
    let correct = scores.iter().filter(|&&s| s == 1.0).count();
    let partial = scores
        .iter()
        .filter(|&&s| (s - 0.5).abs() < f64::EPSILON)
        .count();
    let incorrect = total_questions.saturating_sub(correct + partial);
    let accuracy = if total_questions == 0 {
        0.0
    } else {
        scores.iter().sum::<f64>() / total_questions as f64
    };

    let judge_results: Vec<JudgeResult> = results
        .iter()
        .map(|r| JudgeResult {
            question_id: r.question_id.clone(),
            verdict: r.verdict,
            score: r.score,
            reasoning: r.judge_reasoning.clone(),
        })
        .collect();
    let types: Vec<QuestionType> = results.iter().map(|r| r.question_type).collect();
    let per_dimension = aggregate_dimensions(&judge_results, &types);

    let ingestion_errors = results.iter().filter(|r| r.ingestion_failed).count();
    let retrieval_errors = results.iter().filter(|r| r.retrieval_failed).count();

    EvalMetrics {
        accuracy,
        per_dimension,
        total_questions,
        correct,
        partial,
        incorrect,
        latency: compute_latency_stats(results),
        tokens: compute_token_stats(results),
        context_recall: compute_context_recall(results),
        retrieval: compute_retrieval_stats(results),
        answer_shape: compute_answer_shape(results),
        ingestion_errors,
        retrieval_errors,
        write_quality: compute_write_quality(results),
    }
}

/// A system answer that declines rather than asserts. Mirrors the
/// abstention phrases the judge recognises.
fn is_decline(answer: &str) -> bool {
    let trimmed = answer.trim();
    if trimmed.is_empty() {
        return true;
    }
    let lower = trimmed.to_ascii_lowercase();
    lower.contains("don't know")
        || lower.contains("do not know")
        || lower.contains("not sure")
        || lower.contains("unknown")
        || lower.contains("not mentioned")
        || lower.contains("no information")
}

/// Split accuracy + precision by the router's answer shape.
fn compute_answer_shape(results: &[QuestionResult]) -> AnswerShapeMetrics {
    let mut m = AnswerShapeMetrics::default();
    let (mut single_sum, mut many_sum, mut abstained_sum) = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut committed_not_wrong = 0usize;

    for r in results {
        let committed_assertion = !is_decline(&r.system_answer);
        match r.answer_kind.as_str() {
            "single" => {
                m.single += 1;
                single_sum += r.score;
                if committed_assertion {
                    m.committed += 1;
                    if !matches!(r.verdict, crate::core::outcome::Verdict::Incorrect) {
                        committed_not_wrong += 1;
                    }
                }
            }
            "many" => {
                m.many += 1;
                many_sum += r.score;
                if committed_assertion {
                    m.committed += 1;
                    if !matches!(r.verdict, crate::core::outcome::Verdict::Incorrect) {
                        committed_not_wrong += 1;
                    }
                }
            }
            "none" => {
                m.abstained += 1;
                abstained_sum += r.score;
            }
            _ => m.errored += 1,
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let mean = |sum: f64, n: usize| if n == 0 { 0.0 } else { sum / n as f64 };
    m.single_accuracy = mean(single_sum, m.single);
    m.many_accuracy = mean(many_sum, m.many);
    m.abstained_accuracy = mean(abstained_sum, m.abstained);
    #[allow(clippy::cast_precision_loss)]
    {
        m.committed_precision = if m.committed == 0 {
            1.0
        } else {
            committed_not_wrong as f64 / m.committed as f64
        };
    }
    m
}

fn aggregate_dimensions(
    results: &[JudgeResult],
    types: &[QuestionType],
) -> HashMap<String, DimensionMetrics> {
    let mut by_type: HashMap<QuestionType, Vec<f64>> = HashMap::new();
    for (result, qtype) in results.iter().zip(types.iter()) {
        by_type.entry(*qtype).or_default().push(result.score);
    }
    by_type
        .into_iter()
        .map(|(qtype, scores)| {
            let count = scores.len();
            #[allow(clippy::float_cmp)]
            let correct = scores.iter().filter(|&&s| s == 1.0).count();
            let partial = scores
                .iter()
                .filter(|&&s| (s - 0.5).abs() < f64::EPSILON)
                .count();
            let incorrect = count.saturating_sub(correct + partial);
            let accuracy = scores.iter().sum::<f64>() / count as f64;
            (
                qtype.tag().to_owned(),
                DimensionMetrics {
                    accuracy,
                    count,
                    correct,
                    partial,
                    incorrect,
                },
            )
        })
        .collect()
}

fn compute_token_stats(results: &[QuestionResult]) -> TokenStats {
    if results.is_empty() {
        return TokenStats::default();
    }
    #[allow(clippy::cast_precision_loss)]
    let n = results.len() as f64;
    let write_total: u64 = results.iter().map(|r| r.tokens_write).sum();
    let read_total: u64 = results.iter().map(|r| r.tokens_read).sum();
    let grand_total = write_total + read_total;
    #[allow(clippy::cast_precision_loss)]
    TokenStats {
        write_avg: write_total as f64 / n,
        read_avg: read_total as f64 / n,
        total_avg: grand_total as f64 / n,
        grand_total,
    }
}

fn compute_write_quality(results: &[QuestionResult]) -> WriteQualityMetrics {
    let total_attempted: u64 = results.iter().map(|r| r.write_attempted).sum();
    let total_stored: u64 = results.iter().map(|r| r.write_stored).sum();
    let total_deduplicated: u64 = results.iter().map(|r| r.write_deduplicated).sum();
    #[allow(clippy::cast_precision_loss)]
    let (store_rate, dedup_rate) = if total_attempted > 0 {
        (
            total_stored as f64 / total_attempted as f64,
            total_deduplicated as f64 / total_attempted as f64,
        )
    } else {
        (0.0, 0.0)
    };
    WriteQualityMetrics {
        total_attempted,
        total_stored,
        total_deduplicated,
        store_rate,
        dedup_rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::instance::QuestionType;
    use crate::core::outcome::Verdict;

    fn qr(answer_kind: &str, verdict: Verdict, system_answer: &str) -> QuestionResult {
        QuestionResult {
            question_id: "q".into(),
            question_type: QuestionType::SingleHop,
            question: String::new(),
            ground_truth: String::new(),
            system_answer: system_answer.into(),
            answer_kind: answer_kind.into(),
            verdict,
            score: verdict.score(),
            write_latency_ms: 0,
            read_latency_ms: 0,
            tokens_write: 0,
            tokens_read: 0,
            memories_retrieved: 0,
            retrieved_memory_contents: Vec::new(),
            judge_reasoning: String::new(),
            context_supported: None,
            context_support_reasoning: String::new(),
            ingestion_failed: false,
            retrieval_failed: false,
            write_attempted: 0,
            write_stored: 0,
            write_deduplicated: 0,
        }
    }

    #[test]
    fn answer_shape_splits_by_kind() {
        let results = vec![
            qr("single", Verdict::Correct, "Berlin"),
            qr("many", Verdict::Correct, "a, b"),
            qr("many", Verdict::Partial, "1. ..."),
            qr("none", Verdict::Correct, "I don't know."),
            qr("error", Verdict::Incorrect, ""),
        ];
        let m = compute_answer_shape(&results);
        assert_eq!(m.single, 1);
        assert_eq!(m.many, 2);
        assert_eq!(m.abstained, 1);
        assert_eq!(m.errored, 1);
    }

    #[test]
    fn committed_precision_drops_only_on_a_confident_wrong_answer() {
        // Two correct committed answers + one that's WRONG = a confident
        // falsehood. 2/3 committed answers were not wrong.
        let results = vec![
            qr("single", Verdict::Correct, "Berlin"),
            qr("many", Verdict::Partial, "a, b"),
            qr("many", Verdict::Incorrect, "Paris"),
        ];
        let m = compute_answer_shape(&results);
        assert_eq!(m.committed, 3);
        assert!((m.committed_precision - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn a_decline_is_not_a_committed_answer() {
        // A read that synthesized "I don't know." asserted nothing, so it
        // never counts against precision.
        let results = vec![
            qr("single", Verdict::Correct, "Berlin"),
            qr("many", Verdict::Incorrect, "I don't know."),
        ];
        let m = compute_answer_shape(&results);
        assert_eq!(m.committed, 1);
        assert!((m.committed_precision - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn no_committed_answers_is_vacuously_perfect_precision() {
        let results = vec![qr("none", Verdict::Correct, "I don't know.")];
        let m = compute_answer_shape(&results);
        assert_eq!(m.committed, 0);
        assert!((m.committed_precision - 1.0).abs() < f64::EPSILON);
    }
}
