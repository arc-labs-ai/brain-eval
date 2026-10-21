//! Recall-quality probe — known-answer recall@K.
//!
//! True HNSW-vs-exhaustive recall (the spec's recall@K) is an internal
//! index property best measured in-crate. What a black-box client can
//! measure honestly is **known-answer recall**: encode N memories each
//! carrying a unique nonce, query each nonce, and check whether the one
//! memory that contains it comes back at rank 1 (recall@1) and within the
//! top-K (recall@K). This is correctness, not latency, so the numbers are
//! meaningful even on a slow dev box.
//!
//! Alongside the binary recall@K counts we report two rank-quality metrics
//! over the same known-answer results — **MRR** (mean reciprocal rank) and
//! **nDCG@K** (normalized discounted cumulative gain). Recall@K only asks
//! "did the right memory make the cut"; MRR/nDCG ask "how near the top did
//! it land", which is what catches the "correct doc in top-K but never #1"
//! regression class that recall@K alone is blind to.

use brain_db_sdk::{BrainClient, EncodeBuilder, RecallBuilder};

use crate::client::harness::HarnessError;

/// Targets the substrate commits to (default-tuned index).
#[derive(Debug, Clone, Copy)]
pub struct RecallTargets {
    /// Minimum acceptable recall@1.
    pub at_1: f64,
    /// Minimum acceptable recall@K.
    pub at_k: f64,
    /// Advisory floor for MRR (mean reciprocal rank). Informative, not a
    /// hard gate — see [`RecallQualityReport::pass`].
    pub mrr: f64,
    /// Advisory floor for nDCG@K. Informative, not a hard gate.
    pub ndcg: f64,
}

impl Default for RecallTargets {
    fn default() -> Self {
        // recall@1 ≥ 0.97, recall@10 ≥ 0.95. The MRR / nDCG floors are
        // advisory: with recall@1 ≥ 0.97 and a single relevant memory per
        // query, MRR is bounded below by ~0.97 and nDCG likewise, but we
        // keep them off the hard gate so a marginal ranking wobble reports
        // rather than fails the acceptance run.
        Self {
            at_1: 0.97,
            at_k: 0.95,
            mrr: 0.97,
            ndcg: 0.97,
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
    /// Mean reciprocal rank over all queries (missing memory contributes 0).
    pub mrr: f64,
    /// Mean nDCG@K over all queries (missing memory contributes 0).
    pub ndcg_at_k: f64,
    /// Target recall@1.
    pub target_at_1: f64,
    /// Target recall@K.
    pub target_at_k: f64,
    /// Advisory MRR floor.
    pub target_mrr: f64,
    /// Advisory nDCG@K floor.
    pub target_ndcg: f64,
}

impl RecallQualityReport {
    /// True iff both recall@1 and recall@K met their targets.
    #[must_use]
    pub fn pass(&self) -> bool {
        self.recall_at_1 >= self.target_at_1 && self.recall_at_k >= self.target_at_k
    }

    /// True iff both advisory ranking floors (MRR, nDCG@K) are met. Purely
    /// informative — not part of [`pass`](Self::pass).
    #[must_use]
    pub fn meets_ranking_advisories(&self) -> bool {
        self.mrr >= self.target_mrr && self.ndcg_at_k >= self.target_ndcg
    }

    /// One-line human-readable summary.
    #[must_use]
    pub fn to_text(&self) -> String {
        format!(
            "recall quality ({} queries, top_k={}): recall@1 {:.3} (≥ {:.2})  recall@{} {:.3} (≥ {:.2})  MRR {:.3} (≥ {:.2}~)  nDCG@{} {:.3} (≥ {:.2}~)  [{}]",
            self.queries,
            self.top_k,
            self.recall_at_1,
            self.target_at_1,
            self.top_k,
            self.recall_at_k,
            self.target_at_k,
            self.mrr,
            self.target_mrr,
            self.top_k,
            self.ndcg_at_k,
            self.target_ndcg,
            if self.pass() { "PASS" } else { "FAIL" },
        )
    }
}

/// Reciprocal rank of a hit at 0-indexed position `rank`: `1/(rank+1)`.
/// `None` (the relevant memory was not retrieved) contributes 0.
fn reciprocal_rank(rank: Option<usize>) -> f64 {
    match rank {
        Some(r) => 1.0 / (r as f64 + 1.0),
        None => 0.0,
    }
}

/// Discounted cumulative gain over `gains` in returned order, cut at `k`.
/// Position `i` (0-indexed) is discounted by `log2(i + 2)`.
fn dcg_at_k(gains: &[f64], k: usize) -> f64 {
    gains
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, g)| g / (i as f64 + 2.0).log2())
        .sum()
}

/// nDCG@K: DCG over `gains` (returned order) divided by the ideal DCG
/// (`gains` sorted descending). Returns 0 when the ideal DCG is 0 (no
/// relevant items), so an empty or all-zero ranking scores 0.
fn ndcg_at_k(gains: &[f64], k: usize) -> f64 {
    let dcg = dcg_at_k(gains, k);
    let mut ideal = gains.to_vec();
    ideal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let idcg = dcg_at_k(&ideal, k);
    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

/// A nonce unique to query `i` within one run, salted by `salt` (use the
/// connection's space id hex so reruns don't collide via dedup).
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
    // Encode N memories, each carrying its own nonce. ENCODE acks on the
    // durable WAL sync (`WaitMode::Ack`) *before* async derivation runs, so a
    // plain encode-then-recall races the index: the just-written vector may not
    // be in the HNSW / lexical index when we query it, and the probe reads 0
    // recall for a memory that is in fact stored. `wait_derived` blocks each
    // encode until derivation (embed + index insert) completes, so the
    // subsequent recall measures retrieval quality, not indexing lag.
    for i in 0..n {
        let text = format!(
            "Reference note {i}: the marker is {}. Surrounded by ordinary prose so retrieval has to discriminate.",
            nonce(salt, i)
        );
        let req = EncodeBuilder::new(text.as_str()).wait_derived().build();
        client.encode(&req).await?;
    }

    let mut hit_at_1 = 0usize;
    let mut hit_at_k = 0usize;
    let mut sum_rr = 0.0f64;
    let mut sum_ndcg = 0.0f64;
    for i in 0..n {
        let needle = nonce(salt, i);
        let req = RecallBuilder::new(needle.as_str())
            .max_results(top_k)
            .include_text(true)
            .build();
        let hits = client.recall(&req).await?.memories;
        let rank = hits.iter().position(|m| m.text.contains(&needle));
        if let Some(r) = rank {
            hit_at_k += 1;
            if r == 0 {
                hit_at_1 += 1;
            }
        }
        // One relevant memory per query: gain 1.0 at its returned position,
        // 0.0 everywhere else. Ideal ordering puts that single hit first, so
        // nDCG@K collapses to 1/log2(rank+2) when found, 0 when missed.
        sum_rr += reciprocal_rank(rank);
        let gains: Vec<f64> = hits
            .iter()
            .map(|m| if m.text.contains(&needle) { 1.0 } else { 0.0 })
            .collect();
        sum_ndcg += ndcg_at_k(&gains, top_k as usize);
    }

    let denom = n.max(1) as f64;
    Ok(RecallQualityReport {
        queries: n,
        top_k,
        recall_at_1: hit_at_1 as f64 / denom,
        recall_at_k: hit_at_k as f64 / denom,
        mrr: sum_rr / denom,
        ndcg_at_k: sum_ndcg / denom,
        target_at_1: targets.at_1,
        target_at_k: targets.at_k,
        target_mrr: targets.mrr,
        target_ndcg: targets.ndcg,
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
            mrr: at1,
            ndcg_at_k: at1,
            target_at_1: 0.97,
            target_at_k: 0.95,
            target_mrr: 0.97,
            target_ndcg: 0.97,
        }
    }

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
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

    #[test]
    fn reciprocal_rank_by_position() {
        assert!(approx(reciprocal_rank(Some(0)), 1.0)); // rank 1 → 1.0
        assert!(approx(reciprocal_rank(Some(1)), 0.5)); // rank 2 → 1/2
        assert!(approx(reciprocal_rank(Some(2)), 1.0 / 3.0)); // rank 3 → 1/3
        assert!(approx(reciprocal_rank(None), 0.0)); // missed → 0
    }

    #[test]
    fn ndcg_ideal_ordering_is_one() {
        // A single relevant item already at the top is the ideal ordering.
        let gains = [1.0, 0.0, 0.0, 0.0];
        assert!(approx(ndcg_at_k(&gains, 4), 1.0));

        // Graded, already-descending gains are also ideal → 1.0.
        let graded = [3.0, 2.0, 1.0, 0.0];
        assert!(approx(ndcg_at_k(&graded, 4), 1.0));
    }

    #[test]
    fn ndcg_single_relevant_at_rank_matches_discount() {
        // One relevant item at 0-indexed rank r: DCG = 1/log2(r+2),
        // IDCG = 1/log2(2) = 1, so nDCG = 1/log2(r+2).
        let gains = [0.0, 0.0, 1.0]; // rank 3 (index 2)
        let expected = 1.0 / (2.0f64 + 2.0).log2();
        assert!(approx(ndcg_at_k(&gains, 3), expected));
        assert!(approx(ndcg_at_k(&gains, 3), 0.5)); // log2(4) = 2 → 0.5
    }

    #[test]
    fn ndcg_empty_and_all_zero_are_zero() {
        assert!(approx(ndcg_at_k(&[], 10), 0.0));
        assert!(approx(ndcg_at_k(&[0.0, 0.0, 0.0], 10), 0.0));
    }

    #[test]
    fn ndcg_cut_at_k_drops_items_past_k() {
        // Relevant item sits at index 3, beyond a top-3 cut → 0 within k=3,
        // but recovered when k covers it.
        let gains = [0.0, 0.0, 0.0, 1.0];
        assert!(approx(ndcg_at_k(&gains, 3), 0.0));
        assert!(ndcg_at_k(&gains, 4) > 0.0);
    }

    #[test]
    fn dcg_respects_position_discount() {
        // Same total gain, higher = better DCG.
        let top = [1.0, 0.0];
        let bottom = [0.0, 1.0];
        assert!(dcg_at_k(&top, 2) > dcg_at_k(&bottom, 2));
        assert!(approx(dcg_at_k(&top, 2), 1.0)); // 1/log2(2)
    }

    #[test]
    fn ranking_advisories_are_not_part_of_hard_gate() {
        // Recall targets met but ranking floors missed: still passes the
        // hard gate, and only the advisory helper flags it.
        let mut r = report(0.98, 0.96);
        r.mrr = 0.80;
        r.ndcg_at_k = 0.80;
        assert!(r.pass());
        assert!(!r.meets_ranking_advisories());
    }
}
