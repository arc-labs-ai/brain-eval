//! Retrieval-quality metrics.
//!
//! - [`RetrievalStats`] — shape carried inside [`crate::score::EvalMetrics`].
//! - [`recall_at_k`] / [`ndcg_at_k`] — pure functions for any caller
//!   that wants the underlying primitives.
//! - [`compute_retrieval_stats`] — aggregate over the full run.
//!
//! Relevance is approximated by case-insensitive substring match
//! against the ground truth. A learned-relevance / position-based
//! mode is a follow-up.

use serde::{Deserialize, Serialize};

use crate::core::outcome::QuestionResult;

/// Retrieval-quality metrics at K=1, 5, and 10.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetrievalStats {
    /// Recall@1: fraction of questions whose rank-1 hit is relevant.
    /// The strictest signal — "did the single best result answer the
    /// question?" — and the headline number for the smoke corpus.
    pub recall_at_1: f64,
    /// Recall@5: fraction of relevant memories in the top-5 results.
    pub recall_at_5: f64,
    /// Recall@10.
    pub recall_at_10: f64,
    /// NDCG@5.
    pub ndcg_at_5: f64,
    /// NDCG@10.
    pub ndcg_at_10: f64,
}

/// Aggregate retrieval stats over the run. Returns `None` when no
/// question returned any retrieved memories.
#[must_use]
pub fn compute_retrieval_stats(results: &[QuestionResult]) -> Option<RetrievalStats> {
    let any_retrieved = results
        .iter()
        .any(|r| !r.retrieved_memory_contents.is_empty());
    if !any_retrieved {
        return None;
    }
    Some(RetrievalStats {
        recall_at_1: mean_recall_at_k(results, 1),
        recall_at_5: mean_recall_at_k(results, 5),
        recall_at_10: mean_recall_at_k(results, 10),
        ndcg_at_5: mean_ndcg_at_k(results, 5),
        ndcg_at_10: mean_ndcg_at_k(results, 10),
    })
}

fn mean_recall_at_k(results: &[QuestionResult], k: usize) -> f64 {
    let mut total = 0.0_f64;
    let mut count = 0_usize;
    for r in results {
        if r.retrieval_failed || r.ingestion_failed {
            continue;
        }
        let ground = r.ground_truth.to_lowercase();
        if ground.is_empty() {
            continue;
        }
        let found = r
            .retrieved_memory_contents
            .iter()
            .take(k)
            .any(|c| c.to_lowercase().contains(&ground));
        total += f64::from(u8::from(found));
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let denom = count as f64;
        total / denom
    }
}

fn mean_ndcg_at_k(results: &[QuestionResult], k: usize) -> f64 {
    let mut total = 0.0_f64;
    let mut count = 0_usize;
    for r in results {
        if r.retrieval_failed || r.ingestion_failed {
            continue;
        }
        let ground = r.ground_truth.to_lowercase();
        if ground.is_empty() {
            continue;
        }
        let retrieved_ids: Vec<String> = r
            .retrieved_memory_contents
            .iter()
            .map(|c| c.to_lowercase())
            .collect();
        let relevant: Vec<String> = retrieved_ids
            .iter()
            .filter(|c| c.contains(&ground))
            .cloned()
            .collect();
        let score = ndcg_at_k(&retrieved_ids, &relevant, k);
        total += score;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let denom = count as f64;
        total / denom
    }
}

/// Recall@K — fraction of `relevant` items present in the top-K of
/// `retrieved`. Returns `1.0` vacuously when `relevant` is empty.
#[must_use]
pub fn recall_at_k(retrieved: &[String], relevant: &[String], k: usize) -> f64 {
    if relevant.is_empty() {
        return 1.0;
    }
    let top_k: Vec<&String> = retrieved.iter().take(k).collect();
    let found = relevant.iter().filter(|r| top_k.contains(r)).count();
    #[allow(clippy::cast_precision_loss)]
    let denom = relevant.len() as f64;
    found as f64 / denom
}

/// NDCG@K with binary relevance. Returns `1.0` vacuously when
/// `relevant` is empty.
#[must_use]
pub fn ndcg_at_k(retrieved: &[String], relevant: &[String], k: usize) -> f64 {
    if relevant.is_empty() {
        return 1.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let dcg: f64 = retrieved
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, id)| {
            let rel = f64::from(u8::from(relevant.contains(id)));
            rel / (i as f64 + 2.0).log2()
        })
        .sum();
    let ideal_k = k.min(relevant.len());
    #[allow(clippy::cast_precision_loss)]
    let idcg: f64 = (0..ideal_k)
        .map(|i| 1.0_f64 / (i as f64 + 2.0).log2())
        .sum();
    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_at_k_perfect() {
        let retrieved = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let relevant = vec!["a".to_owned(), "b".to_owned()];
        assert!((recall_at_k(&retrieved, &relevant, 3) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn recall_at_k_partial() {
        let retrieved = vec!["a".to_owned(), "x".to_owned(), "y".to_owned()];
        let relevant = vec!["a".to_owned(), "b".to_owned()];
        assert!((recall_at_k(&retrieved, &relevant, 3) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn recall_at_k_empty_relevant_is_vacuously_one() {
        assert!(
            (recall_at_k(&["x".to_owned()], &[], 5) - 1.0).abs() < f64::EPSILON,
            "no relevant items => trivially perfect"
        );
    }

    #[test]
    fn ndcg_at_k_perfect_ordering() {
        let retrieved = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let relevant = vec!["a".to_owned(), "b".to_owned()];
        assert!((ndcg_at_k(&retrieved, &relevant, 3) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn ndcg_at_k_empty_relevant_is_vacuously_one() {
        assert!((ndcg_at_k(&["a".to_owned()], &[], 3) - 1.0).abs() < f64::EPSILON);
    }
}
