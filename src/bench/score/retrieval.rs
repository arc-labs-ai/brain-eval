//! Retrieval-quality metrics.
//!
//! - [`RetrievalStats`] — shape carried inside [`crate::bench::score::EvalMetrics`].
//! - [`recall_at_k`] / [`ndcg_at_k`] / [`reciprocal_rank`] — pure functions
//!   for any caller that wants the underlying primitives.
//! - [`compute_retrieval_stats`] — aggregate over the full run.
//!
//! Relevance is approximated by case-insensitive substring match
//! against the ground truth. A learned-relevance / position-based
//! mode is a follow-up.

use serde::{Deserialize, Serialize};

use crate::bench::core::outcome::QuestionResult;

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
    /// Mean Reciprocal Rank over the whole ranked list.
    ///
    /// WHY THIS IS HERE AND NOT JUST ANOTHER RECALL CUT. Recall@K is a step
    /// function: it cannot tell a hit at rank 2 from a hit at rank 5, and
    /// both land in recall@5 as a flat 1.0. Most retrieval changes worth
    /// making — a wider HNSW beam, a different fusion method, a reranker —
    /// move the relevant memory *within* the top-K far more often than they
    /// move it across a K boundary. Measured only by recall@K those changes
    /// read as no-ops.
    ///
    /// MRR is the companion that sees that movement: 1.0 at rank 1, 0.5 at
    /// rank 2, 0.2 at rank 5. It is also the metric that stays informative on
    /// a corpus small enough that recall@10 is saturated at 1.0, which is
    /// where this eval currently sits.
    ///
    /// Unranked (0.0) when no relevant memory appears anywhere in the list.
    pub mrr: f64,
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
        mrr: mean_reciprocal_rank(results),
    })
}

fn mean_recall_at_k(results: &[QuestionResult], k: usize) -> f64 {
    let mut total = 0.0_f64;
    let mut count = 0_usize;
    for r in results {
        if r.retrieval_failed || r.ingestion_failed {
            continue;
        }
        // Episodic-only diagnostic: grounded / abstention answers carry
        // no ranked memory list, so they don't belong in recall@K.
        if r.retrieved_memory_contents.is_empty() {
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
        if r.retrieved_memory_contents.is_empty() {
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

/// Mean Reciprocal Rank across the run.
///
/// Scored over the same population as recall and NDCG — a question is counted
/// only when it produced a ranked list and carries a ground truth, so grounded
/// and abstention answers stay excluded exactly as they are above.
fn mean_reciprocal_rank(results: &[QuestionResult]) -> f64 {
    let mut total = 0.0_f64;
    let mut count = 0_usize;
    for r in results {
        if r.retrieval_failed || r.ingestion_failed {
            continue;
        }
        if r.retrieved_memory_contents.is_empty() {
            continue;
        }
        let ground = r.ground_truth.to_lowercase();
        if ground.is_empty() {
            continue;
        }
        let ranked: Vec<String> = r
            .retrieved_memory_contents
            .iter()
            .map(|c| c.to_lowercase())
            .collect();
        total += reciprocal_rank(&ranked, |c| c.contains(&ground));
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

/// Reciprocal rank of the FIRST item satisfying `is_relevant`.
///
/// `1 / (1-based rank)`, or `0.0` when nothing in the list is relevant. Takes
/// a predicate rather than a slice of ids because relevance here is a
/// substring test against ground truth, not set membership — see the module
/// note about the approximation.
#[must_use]
pub fn reciprocal_rank<T, F>(ranked: &[T], is_relevant: F) -> f64
where
    F: Fn(&T) -> bool,
{
    ranked.iter().position(is_relevant).map_or(0.0, |i| {
        #[allow(clippy::cast_precision_loss)]
        let rank = (i + 1) as f64;
        1.0 / rank
    })
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

    // ── MRR ───────────────────────────────────────────────────────────────

    #[test]
    fn reciprocal_rank_is_one_over_the_first_hit() {
        let ranked = vec!["x".to_owned(), "y".to_owned(), "hit".to_owned()];
        let rr = reciprocal_rank(&ranked, |c| c == "hit");
        assert!((rr - 1.0 / 3.0).abs() < 1e-12, "rank 3 => 1/3, got {rr}");
    }

    #[test]
    fn reciprocal_rank_rewards_rank_one() {
        let ranked = vec!["hit".to_owned(), "x".to_owned()];
        assert!((reciprocal_rank(&ranked, |c| c == "hit") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn reciprocal_rank_is_zero_when_nothing_is_relevant() {
        let ranked = vec!["x".to_owned(), "y".to_owned()];
        assert!(reciprocal_rank(&ranked, |c| c == "hit").abs() < f64::EPSILON);
    }

    #[test]
    fn reciprocal_rank_takes_only_the_first_hit() {
        // Two relevant items: MRR is defined on the FIRST, so a second hit
        // further down must not raise the score.
        let ranked = vec!["x".to_owned(), "hit".to_owned(), "hit".to_owned()];
        let rr = reciprocal_rank(&ranked, |c| c == "hit");
        assert!((rr - 0.5).abs() < f64::EPSILON, "rank 2 => 0.5, got {rr}");
    }

    /// THE REASON MRR IS HERE. Two runs where the relevant memory moves from
    /// rank 5 to rank 1 are IDENTICAL under recall@5 — both 1.0 — and that is
    /// exactly the movement a wider HNSW beam or a reranker produces. MRR is
    /// the metric that can see it.
    #[test]
    fn mrr_separates_runs_that_recall_at_k_cannot() {
        let worse = vec![
            "x".to_owned(),
            "x".to_owned(),
            "x".to_owned(),
            "x".to_owned(),
            "hit".to_owned(),
        ];
        let better = vec![
            "hit".to_owned(),
            "x".to_owned(),
            "x".to_owned(),
            "x".to_owned(),
            "x".to_owned(),
        ];
        let relevant = vec!["hit".to_owned()];

        // Recall@5 cannot tell them apart.
        let r_worse = recall_at_k(&worse, &relevant, 5);
        let r_better = recall_at_k(&better, &relevant, 5);
        assert!(
            (r_worse - r_better).abs() < f64::EPSILON,
            "recall@5 should be blind to the reordering"
        );

        // MRR can: 1/5 against 1/1.
        let m_worse = reciprocal_rank(&worse, |c| c == "hit");
        let m_better = reciprocal_rank(&better, |c| c == "hit");
        assert!((m_worse - 0.2).abs() < 1e-12);
        assert!((m_better - 1.0).abs() < f64::EPSILON);
        assert!(m_better > m_worse);
    }
}
