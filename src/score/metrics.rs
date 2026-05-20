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
    /// Retrieval quality (Recall@K, NDCG@K). `None` when no question
    /// returned any retrieved memories.
    pub retrieval: Option<RetrievalStats>,
    /// Questions where session ingestion failed (infrastructure error,
    /// not model behaviour).
    pub ingestion_errors: usize,
    /// Questions where retrieval failed (infrastructure error).
    pub retrieval_errors: usize,
    /// Encode-side quality (accepted / merged / discarded counters).
    pub write_quality: WriteQualityMetrics,
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
        retrieval: compute_retrieval_stats(results),
        ingestion_errors,
        retrieval_errors,
        write_quality: compute_write_quality(results),
    }
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
