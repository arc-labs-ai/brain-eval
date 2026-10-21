//! Aggregate metrics shape + the top-level `compute_full_metrics`
//! entry point that turns a `Vec<QuestionResult>` into an
//! `EvalMetrics`.
//!
//! This file owns the **aggregate accuracy + token + write-quality**
//! computation. Latency and retrieval-quality stats live in their own
//! sibling files ([`crate::bench::score::latency`], [`crate::bench::score::retrieval`])
//! because they grow on different cadences.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::bench::core::instance::QuestionType;
use crate::bench::core::outcome::{AnswerKind, JudgeResult, QuestionResult};
use crate::bench::score::context_recall::{compute_context_recall, ContextRecallStats};
use crate::bench::score::latency::{compute_latency_stats, LatencyStats};
use crate::bench::score::retrieval::{compute_retrieval_stats, RetrievalStats};

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
    // ACCURACY IS SCORED OVER GRADED QUESTIONS ONLY.
    //
    // A retrieval-only run synthesises no answer and calls no judge, so every
    // verdict comes back `NotJudged`. Counting those in the denominator would
    // report a flawless retrieval run as 0.0 accuracy — the exact misreading
    // the variant exists to prevent. They are excluded from both sides, so a
    // retrieval-only run reports accuracy over zero questions (0.0, with
    // `correct`/`partial`/`incorrect` all zero) and the retrieval metrics
    // carry the signal.
    let graded: Vec<f64> = results
        .iter()
        .filter(|r| r.verdict.is_graded())
        .map(|r| r.score)
        .collect();
    let graded_questions = graded.len();

    #[allow(clippy::float_cmp)]
    let correct = graded.iter().filter(|&&s| s == 1.0).count();
    let partial = graded
        .iter()
        .filter(|&&s| (s - 0.5).abs() < f64::EPSILON)
        .count();
    let incorrect = graded_questions.saturating_sub(correct + partial);
    let accuracy = if graded_questions == 0 {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let denom = graded_questions as f64;
        graded.iter().sum::<f64>() / denom
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
    let (mut single_graded, mut many_graded, mut abstained_graded) = (0usize, 0usize, 0usize);
    let mut committed_not_wrong = 0usize;
    let mut committed_graded = 0usize;

    // SHAPE COUNTS COUNT EVERY QUESTION; ACCURACIES COUNT ONLY GRADED ONES.
    //
    // Same rule `compute_full_metrics` follows, for the same reason. The shape
    // of an answer — one memory, a set, an honest abstention — is decided by
    // the router and is just as real in a retrieval-only run, so `single`,
    // `many`, `abstained`, `errored` and `committed` keep counting everything.
    //
    // The accuracies cannot. A retrieval-only run judges nothing, so every
    // verdict is `NotJudged` and every score 0.0. Left in the denominator they
    // dragged `single_accuracy`/`many_accuracy` to 0.0, while
    // `committed_precision` went the other way: `NotJudged` is not
    // `Incorrect`, so every ungraded answer counted as "committed and not
    // wrong" and precision reported a flawless 1.0. The same run read as
    // simultaneously perfect and catastrophic — both misreadings the
    // `NotJudged` variant exists to prevent.
    for r in results {
        let graded = r.verdict.is_graded();
        let committed_assertion = !is_decline(&r.system_answer);
        match r.answer_kind {
            AnswerKind::Single => {
                m.single += 1;
                if graded {
                    single_graded += 1;
                    single_sum += r.score;
                }
                if committed_assertion {
                    m.committed += 1;
                    if graded {
                        committed_graded += 1;
                        if !matches!(r.verdict, crate::bench::core::outcome::Verdict::Incorrect) {
                            committed_not_wrong += 1;
                        }
                    }
                }
            }
            AnswerKind::Many => {
                m.many += 1;
                if graded {
                    many_graded += 1;
                    many_sum += r.score;
                }
                if committed_assertion {
                    m.committed += 1;
                    if graded {
                        committed_graded += 1;
                        if !matches!(r.verdict, crate::bench::core::outcome::Verdict::Incorrect) {
                            committed_not_wrong += 1;
                        }
                    }
                }
            }
            AnswerKind::None => {
                m.abstained += 1;
                if graded {
                    abstained_graded += 1;
                    abstained_sum += r.score;
                }
            }
            AnswerKind::Error => m.errored += 1,
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let mean = |sum: f64, n: usize| if n == 0 { 0.0 } else { sum / n as f64 };
    m.single_accuracy = mean(single_sum, single_graded);
    m.many_accuracy = mean(many_sum, many_graded);
    m.abstained_accuracy = mean(abstained_sum, abstained_graded);
    #[allow(clippy::cast_precision_loss)]
    {
        // One rule: precision is measured over GRADED commitments, and when
        // there are none there is nothing to report.
        //
        // The old code answered 1.0 when nothing was committed — "it asserted
        // no falsehood because it asserted nothing". That reads as a perfect
        // score for a run that demonstrated nothing, which is the same
        // flattery as reporting 1.0 for answers no judge ever read. Both are
        // "no denominator", and `compute_full_metrics` already answers that
        // with 0.0; a run with nothing to measure says so in the retrieval
        // metrics, not in a precision of 1.0.
        m.committed_precision = if committed_graded == 0 {
            0.0
        } else {
            committed_not_wrong as f64 / committed_graded as f64
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
    use crate::bench::core::instance::QuestionType;
    use crate::bench::core::outcome::Verdict;

    fn qr(answer_kind: AnswerKind, verdict: Verdict, system_answer: &str) -> QuestionResult {
        QuestionResult {
            question_id: "q".into(),
            question_type: QuestionType::SingleHop,
            question: String::new(),
            ground_truth: String::new(),
            system_answer: system_answer.into(),
            answer_kind,
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
            qr(AnswerKind::Single, Verdict::Correct, "Berlin"),
            qr(AnswerKind::Many, Verdict::Correct, "a, b"),
            qr(AnswerKind::Many, Verdict::Partial, "1. ..."),
            qr(AnswerKind::None, Verdict::Correct, "I don't know."),
            qr(AnswerKind::Error, Verdict::Incorrect, ""),
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
            qr(AnswerKind::Single, Verdict::Correct, "Berlin"),
            qr(AnswerKind::Many, Verdict::Partial, "a, b"),
            qr(AnswerKind::Many, Verdict::Incorrect, "Paris"),
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
            qr(AnswerKind::Single, Verdict::Correct, "Berlin"),
            qr(AnswerKind::Many, Verdict::Incorrect, "I don't know."),
        ];
        let m = compute_answer_shape(&results);
        assert_eq!(m.committed, 1);
        assert!((m.committed_precision - 1.0).abs() < f64::EPSILON);
    }

    /// A run that committed to nothing reports no precision, not perfect
    /// precision.
    ///
    /// This asserted 1.0 until the retrieval-only work, on the reasoning that
    /// the system "asserted no falsehood". True, and useless: it asserted
    /// nothing at all, and a headline 1.0 flatters a run that demonstrated
    /// nothing — the same flattery as reporting 1.0 for answers no judge ever
    /// read. Precision is now measured over graded commitments, and with none
    /// there is nothing to report.
    #[test]
    fn no_committed_answers_reports_no_precision() {
        let results = vec![qr(AnswerKind::None, Verdict::Correct, "I don't know.")];
        let m = compute_answer_shape(&results);
        assert_eq!(m.committed, 0, "an abstention commits to nothing");
        assert!(
            m.committed_precision.abs() < f64::EPSILON,
            "no graded commitments => nothing to report, not a perfect score"
        );
    }

    // ── retrieval-only runs (accuracy plan §6) ────────────────────────────

    /// THE MISREADING THIS GUARDS AGAINST. A retrieval-only run synthesises no
    /// answer and calls no judge, so every verdict is `NotJudged`. If those
    /// counted in the denominator, a flawless retrieval run would report
    /// 0.0 accuracy and read as total failure.
    #[test]
    fn ungraded_questions_are_excluded_from_accuracy() {
        let results = vec![
            qr(AnswerKind::Single, Verdict::NotJudged, ""),
            qr(AnswerKind::Single, Verdict::NotJudged, ""),
            qr(AnswerKind::Many, Verdict::NotJudged, ""),
        ];
        let m = compute_full_metrics(&results);
        assert_eq!(m.total_questions, 3, "the run still saw three questions");
        assert_eq!(m.correct, 0);
        assert_eq!(m.partial, 0);
        assert_eq!(
            m.incorrect, 0,
            "ungraded must NOT be counted as incorrect — that is the bug"
        );
        assert!(
            m.accuracy.abs() < f64::EPSILON,
            "no graded questions => accuracy over an empty set"
        );
    }

    /// The answer-shape breakdown follows the same rule as the headline
    /// accuracy — and used not to.
    ///
    /// `compute_answer_shape` counted every question in its accuracy
    /// denominators and treated `NotJudged` as "not Incorrect", so a
    /// retrieval-only run reported `single_accuracy` 0.0 **and**
    /// `committed_precision` 1.0 at the same time: the same run reading as
    /// catastrophic and flawless at once, in the one mode the variant exists
    /// to support.
    #[test]
    fn answer_shape_excludes_ungraded_from_accuracy_and_precision() {
        let results = vec![
            qr(AnswerKind::Single, Verdict::NotJudged, "Paris"),
            qr(AnswerKind::Many, Verdict::NotJudged, "Paris, Berlin"),
        ];
        let m = compute_answer_shape(&results);

        // Shape counts still describe what the router did.
        assert_eq!(m.single, 1, "the router still returned one single answer");
        assert_eq!(m.many, 1, "and one many answer");
        assert_eq!(m.committed, 2, "both asserted something");

        // Accuracies have nothing to average over.
        assert!(
            m.single_accuracy.abs() < f64::EPSILON,
            "no graded single answers => accuracy over an empty set"
        );
        assert!(m.many_accuracy.abs() < f64::EPSILON, "likewise for many");
        assert!(
            m.committed_precision.abs() < f64::EPSILON,
            "committed answers nobody judged must NOT report perfect precision \
             — claiming 1.0 here is the bug"
        );
    }

    /// Precision over a mixed run divides by the GRADED committed answers, so
    /// an unjudged answer alongside a correct one cannot inflate or deflate it.
    #[test]
    fn answer_shape_precision_divides_by_graded_commitments() {
        let results = vec![
            qr(AnswerKind::Single, Verdict::Correct, "Berlin"),
            qr(AnswerKind::Single, Verdict::NotJudged, "Paris"),
        ];
        let m = compute_answer_shape(&results);

        assert_eq!(m.committed, 2, "both committed, graded or not");
        assert!(
            (m.committed_precision - 1.0).abs() < f64::EPSILON,
            "one graded commitment, not wrong => 1.0, not 0.5"
        );
        assert!(
            (m.single_accuracy - 1.0).abs() < f64::EPSILON,
            "one graded single answer, correct => 1.0, not 0.5"
        );
    }

    /// A mixed run scores only what was graded: one Correct out of one graded
    /// question is 1.0, regardless of how many went unjudged alongside it.
    #[test]
    fn accuracy_is_scored_over_graded_questions_only() {
        let results = vec![
            qr(AnswerKind::Single, Verdict::Correct, "Berlin"),
            qr(AnswerKind::Single, Verdict::NotJudged, ""),
            qr(AnswerKind::Single, Verdict::NotJudged, ""),
        ];
        let m = compute_full_metrics(&results);
        assert_eq!(m.correct, 1);
        assert_eq!(m.incorrect, 0);
        assert!(
            (m.accuracy - 1.0).abs() < f64::EPSILON,
            "1 correct of 1 graded = 1.0, not 1/3; got {}",
            m.accuracy
        );
    }

    #[test]
    fn not_judged_is_not_graded_but_incorrect_is() {
        assert!(!Verdict::NotJudged.is_graded());
        for v in [Verdict::Correct, Verdict::Partial, Verdict::Incorrect] {
            assert!(v.is_graded(), "{v:?} is a judgement");
        }
    }
}
