//! Recall-quality probe — known-answer recall@K.
//!
//! True HNSW-vs-exhaustive recall (the spec's recall@K) is an internal
//! index property best measured in-crate. What a black-box client can
//! measure honestly is **known-answer recall**: encode N memories each
//! carrying a unique nonce, query each nonce, and check whether the one
//! memory that contains it comes back at rank 1 (recall@1) and within the
//! top-K (recall@K). This is correctness, not latency, so the numbers are
//! meaningful even on a slow dev box.

use brain_db_sdk::{BrainClient, EncodeBuilder, RecallBuilder};

use crate::run::harness::HarnessError;

/// Targets the substrate commits to (default-tuned index).
#[derive(Debug, Clone, Copy)]
pub struct RecallTargets {
    /// Minimum acceptable recall@1.
    pub at_1: f64,
    /// Minimum acceptable recall@K.
    pub at_k: f64,
}

impl Default for RecallTargets {
    fn default() -> Self {
        // recall@1 ≥ 0.97, recall@10 ≥ 0.95.
        Self {
            at_1: 0.97,
            at_k: 0.95,
        }
    }
}

/// Outcome of a recall-quality probe.
#[derive(Debug, Clone)]
pub struct RecallQualityReport {
    /// Number of queries issued.
    pub queries: usize,
    /// `top_k` used for the recall queries.
    pub top_k: u32,
    /// Fraction whose intended memory ranked first.
    pub recall_at_1: f64,
    /// Fraction whose intended memory appeared within top-K.
    pub recall_at_k: f64,
    /// Target recall@1.
    pub target_at_1: f64,
    /// Target recall@K.
    pub target_at_k: f64,
}

impl RecallQualityReport {
    /// True iff both recall@1 and recall@K met their targets.
    #[must_use]
    pub fn pass(&self) -> bool {
        self.recall_at_1 >= self.target_at_1 && self.recall_at_k >= self.target_at_k
    }

    /// One-line human-readable summary.
    #[must_use]
    pub fn to_text(&self) -> String {
        format!(
            "recall quality ({} queries, top_k={}): recall@1 {:.3} (≥ {:.2})  recall@{} {:.3} (≥ {:.2})  [{}]",
            self.queries,
            self.top_k,
            self.recall_at_1,
            self.target_at_1,
            self.top_k,
            self.recall_at_k,
            self.target_at_k,
            if self.pass() { "PASS" } else { "FAIL" },
        )
    }
}

/// A nonce unique to query `i` within one run, salted by `salt` (use the
/// connection's agent id hex so reruns don't collide via dedup).
fn nonce(salt: &str, i: usize) -> String {
    format!("zq{salt}{i:06}xz")
}

/// Encode `n` nonce-bearing memories, then query each nonce and measure
/// known-answer recall@1 / recall@K. The client must be connected.
pub async fn run_recall_quality(
    client: &BrainClient,
    n: usize,
    top_k: u32,
    salt: &str,
    targets: &RecallTargets,
) -> Result<RecallQualityReport, HarnessError> {
    // Encode N memories, each carrying its own nonce.
    for i in 0..n {
        let text = format!(
            "Reference note {i}: the marker is {}. Surrounded by ordinary prose so retrieval has to discriminate.",
            nonce(salt, i)
        );
        let req = EncodeBuilder::new(text.as_str()).deduplicate(false).build();
        client.encode(&req).await?;
    }

    let mut hit_at_1 = 0usize;
    let mut hit_at_k = 0usize;
    for i in 0..n {
        let needle = nonce(salt, i);
        let req = RecallBuilder::new(needle.as_str())
            .top_k(top_k)
            .include_text(true)
            .build();
        let hits = client.recall(&req).await?;
        if let Some(rank) = hits.iter().position(|m| m.text.contains(&needle)) {
            hit_at_k += 1;
            if rank == 0 {
                hit_at_1 += 1;
            }
        }
    }

    let denom = n.max(1) as f64;
    Ok(RecallQualityReport {
        queries: n,
        top_k,
        recall_at_1: hit_at_1 as f64 / denom,
        recall_at_k: hit_at_k as f64 / denom,
        target_at_1: targets.at_1,
        target_at_k: targets.at_k,
    })
}

/// Quality-regression gate: a new report must not drop more than
/// `tolerance` (absolute) below the baseline on either metric.
#[must_use]
pub fn no_regression(
    baseline: &RecallQualityReport,
    current: &RecallQualityReport,
    tolerance: f64,
) -> bool {
    current.recall_at_1 + tolerance >= baseline.recall_at_1
        && current.recall_at_k + tolerance >= baseline.recall_at_k
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(at1: f64, atk: f64) -> RecallQualityReport {
        RecallQualityReport {
            queries: 100,
            top_k: 10,
            recall_at_1: at1,
            recall_at_k: atk,
            target_at_1: 0.97,
            target_at_k: 0.95,
        }
    }

    #[test]
    fn pass_requires_both_targets() {
        assert!(report(0.98, 0.96).pass());
        assert!(!report(0.95, 0.96).pass());
        assert!(!report(0.98, 0.90).pass());
    }

    #[test]
    fn regression_gate_tolerates_small_drops() {
        let base = report(0.98, 0.96);
        assert!(no_regression(&base, &report(0.975, 0.955), 0.01));
        assert!(!no_regression(&base, &report(0.95, 0.96), 0.01)); // 3% drop @1
    }

    #[test]
    fn nonce_is_deterministic_and_unique() {
        assert_eq!(nonce("a", 1), nonce("a", 1));
        assert_ne!(nonce("a", 1), nonce("a", 2));
        assert_ne!(nonce("a", 1), nonce("b", 1));
    }
}
