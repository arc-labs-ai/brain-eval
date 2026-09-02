//! Meta-evaluation of the answer judge.
//!
//! "The LLM judge agrees with itself" is not validation. A judge is only
//! trustworthy once it is measured against a **gold** label set — a small
//! batch of (question, answer) pairs a human (or a task-grounded oracle) has
//! graded — and reported with the same rigour we demand of the system under
//! test:
//!
//! - **Agreement** — exact-match rate against gold, and a lenient
//!   adjacent-agreement rate that forgives only Correct↔Partial /
//!   Partial↔Incorrect confusions (never Correct↔Incorrect).
//! - **Chance-corrected agreement** — Cohen's κ, so a judge that scores well
//!   only because the labels are skewed (almost all Correct) is caught.
//! - **Bias** — mean signed score error (judge − gold). Positive ⇒ the judge
//!   systematically over-credits (grade inflation); negative ⇒ it under-credits.
//! - **Confusion matrix** — the full 3×3, so a specific failure mode (e.g.
//!   Partial graded Correct) is visible, not averaged away.
//! - **Sensitivity** — [`disagreement_rate`] over two runs of the same judge
//!   on the same items, for order- / style-perturbation robustness checks.
//!
//! Everything here is a pure function over verdict pairs, so the meta-eval runs
//! offline against a checked-in label set with no live model.

use serde::{Deserialize, Serialize};

use crate::bench::core::outcome::Verdict;

/// One graded item: what the gold label says, what the judge produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabeledVerdict {
    /// Human / oracle gold label.
    pub gold: Verdict,
    /// The judge's verdict for the same item.
    pub judge: Verdict,
}

impl LabeledVerdict {
    /// Construct a pair.
    #[must_use]
    pub fn new(gold: Verdict, judge: Verdict) -> Self {
        Self { gold, judge }
    }
}

/// Meta-evaluation report for a judge over a gold set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JudgeMetaReport {
    /// Number of graded items.
    pub n: usize,
    /// Exact-match agreement rate in `[0,1]`.
    pub exact_agreement: f64,
    /// Adjacent agreement: exact matches plus one-step confusions
    /// (Correct↔Partial, Partial↔Incorrect). A Correct↔Incorrect flip never
    /// counts. In `[0,1]` and always `>= exact_agreement`.
    pub adjacent_agreement: f64,
    /// Cohen's κ — agreement corrected for chance. `1.0` perfect, `0.0`
    /// chance-level, negative worse than chance.
    pub cohens_kappa: f64,
    /// Mean signed score error (judge − gold), in `[-1,1]`. `> 0` ⇒ the judge
    /// over-credits on average; `< 0` ⇒ under-credits; `≈ 0` ⇒ unbiased.
    pub score_bias: f64,
    /// Mean absolute score error, in `[0,1]` — magnitude of disagreement
    /// regardless of direction.
    pub score_mae: f64,
    /// Row-major 3×3 confusion, indexed `[gold][judge]` with the order
    /// Correct(0), Partial(1), Incorrect(2).
    pub confusion: [[usize; 3]; 3],
}

impl JudgeMetaReport {
    /// Does the judge clear the trust bar? A conventional floor: exact
    /// agreement and κ both at or above the given thresholds, and the bias
    /// magnitude within `max_bias`. Callers pick the numbers per label set;
    /// there is no universal constant.
    #[must_use]
    pub fn passes(&self, min_agreement: f64, min_kappa: f64, max_bias: f64) -> bool {
        self.exact_agreement >= min_agreement
            && self.cohens_kappa >= min_kappa
            && self.score_bias.abs() <= max_bias
    }
}

fn verdict_index(v: Verdict) -> usize {
    match v {
        Verdict::Correct => 0,
        Verdict::Partial => 1,
        Verdict::Incorrect => 2,
    }
}

/// Meta-evaluate a judge from its labeled verdicts. Returns `None` for an empty
/// set — no rate is meaningful over zero items.
#[must_use]
pub fn meta_evaluate(labels: &[LabeledVerdict]) -> Option<JudgeMetaReport> {
    let n = labels.len();
    if n == 0 {
        return None;
    }

    let mut confusion = [[0_usize; 3]; 3];
    let mut exact = 0_usize;
    let mut adjacent = 0_usize;
    let mut signed = 0.0_f64;
    let mut abs = 0.0_f64;

    for l in labels {
        let gi = verdict_index(l.gold);
        let ji = verdict_index(l.judge);
        confusion[gi][ji] += 1;
        if gi == ji {
            exact += 1;
        }
        // Adjacent ⇒ index distance <= 1 (Correct↔Incorrect is distance 2).
        if gi.abs_diff(ji) <= 1 {
            adjacent += 1;
        }
        let d = l.judge.score() - l.gold.score();
        signed += d;
        abs += d.abs();
    }

    #[allow(clippy::cast_precision_loss)]
    let nf = n as f64;
    let exact_agreement = exact as f64 / nf;
    let adjacent_agreement = adjacent as f64 / nf;
    let score_bias = signed / nf;
    let score_mae = abs / nf;
    let cohens_kappa = cohens_kappa(&confusion, n);

    Some(JudgeMetaReport {
        n,
        exact_agreement,
        adjacent_agreement,
        cohens_kappa,
        score_bias,
        score_mae,
        confusion,
    })
}

/// Cohen's κ from a 3×3 confusion matrix. `p_o` is observed agreement (the
/// diagonal); `p_e` is expected-by-chance agreement from the marginals.
/// Returns `1.0` when there is no room to disagree (`p_e == 1`).
fn cohens_kappa(confusion: &[[usize; 3]; 3], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let nf = n as f64;

    let mut agree = 0_usize;
    let mut row_tot = [0_usize; 3];
    let mut col_tot = [0_usize; 3];
    for (gi, row) in confusion.iter().enumerate() {
        for (ji, &c) in row.iter().enumerate() {
            row_tot[gi] += c;
            col_tot[ji] += c;
            if gi == ji {
                agree += c;
            }
        }
    }
    let p_o = agree as f64 / nf;
    #[allow(clippy::cast_precision_loss)]
    let p_e: f64 = (0..3)
        .map(|i| (row_tot[i] as f64 / nf) * (col_tot[i] as f64 / nf))
        .sum();
    if (1.0 - p_e).abs() < f64::EPSILON {
        // No expected disagreement: perfect iff observed is also perfect.
        return if (p_o - 1.0).abs() < f64::EPSILON {
            1.0
        } else {
            0.0
        };
    }
    (p_o - p_e) / (1.0 - p_e)
}

/// Fraction of items on which two verdict streams (e.g. the same judge run
/// twice with perturbed answer order or phrasing) disagree. A robust,
/// low-variance judge keeps this near zero. Returns `None` on length mismatch
/// or empty input — the two runs must be aligned item-for-item.
#[must_use]
pub fn disagreement_rate(run_a: &[Verdict], run_b: &[Verdict]) -> Option<f64> {
    if run_a.is_empty() || run_a.len() != run_b.len() {
        return None;
    }
    let disagreements = run_a
        .iter()
        .zip(run_b.iter())
        .filter(|(a, b)| a != b)
        .count();
    #[allow(clippy::cast_precision_loss)]
    Some(disagreements as f64 / run_a.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use Verdict::{Correct, Incorrect, Partial};

    fn pairs(v: &[(Verdict, Verdict)]) -> Vec<LabeledVerdict> {
        v.iter().map(|&(g, j)| LabeledVerdict::new(g, j)).collect()
    }

    #[test]
    fn perfect_judge_scores_perfectly() {
        let labels = pairs(&[
            (Correct, Correct),
            (Partial, Partial),
            (Incorrect, Incorrect),
            (Correct, Correct),
        ]);
        let r = meta_evaluate(&labels).expect("non-empty");
        assert!((r.exact_agreement - 1.0).abs() < 1e-9);
        assert!((r.adjacent_agreement - 1.0).abs() < 1e-9);
        assert!((r.cohens_kappa - 1.0).abs() < 1e-9);
        assert!(r.score_bias.abs() < 1e-9 && r.score_mae.abs() < 1e-9);
        assert!(r.passes(0.9, 0.8, 0.05));
    }

    #[test]
    fn grade_inflation_shows_positive_bias() {
        // Judge upgrades every Partial to Correct and every Incorrect to Partial.
        let labels = pairs(&[
            (Partial, Correct),
            (Incorrect, Partial),
            (Partial, Correct),
            (Incorrect, Partial),
        ]);
        let r = meta_evaluate(&labels).expect("non-empty");
        assert!(r.score_bias > 0.0, "over-crediting => positive bias: {r:?}");
        assert!((r.exact_agreement).abs() < 1e-9, "never exactly agrees");
        // All confusions are one step, so adjacent agreement is full.
        assert!((r.adjacent_agreement - 1.0).abs() < 1e-9);
        assert!(!r.passes(0.5, 0.0, 0.1));
    }

    #[test]
    fn correct_incorrect_flip_is_not_adjacent() {
        let labels = pairs(&[(Correct, Incorrect), (Incorrect, Correct)]);
        let r = meta_evaluate(&labels).expect("non-empty");
        assert!(r.adjacent_agreement.abs() < 1e-9, "distance-2 flips: {r:?}");
        assert!(r.cohens_kappa < 0.0, "worse than chance");
    }

    #[test]
    fn confusion_matrix_indices() {
        let labels = pairs(&[(Correct, Partial), (Correct, Partial), (Partial, Partial)]);
        let r = meta_evaluate(&labels).expect("non-empty");
        assert_eq!(r.confusion[0][1], 2, "two gold-Correct graded Partial");
        assert_eq!(r.confusion[1][1], 1, "one gold-Partial graded Partial");
    }

    #[test]
    fn empty_set_has_no_report() {
        assert!(meta_evaluate(&[]).is_none());
    }

    #[test]
    fn disagreement_rate_basics() {
        let a = [Correct, Partial, Incorrect];
        let b = [Correct, Incorrect, Incorrect];
        assert!((disagreement_rate(&a, &b).expect("aligned") - 1.0 / 3.0).abs() < 1e-9);
        assert!(
            disagreement_rate(&a, &[Correct]).is_none(),
            "length mismatch"
        );
        assert!(disagreement_rate(&[], &[]).is_none());
    }
}
