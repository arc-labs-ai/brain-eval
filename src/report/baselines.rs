//! Published competitor numbers per benchmark.
//!
//! These tables decay fast — pin every row to a primary source and
//! call out methodology issues (the Zep LoCoMo footnote is the
//! template).
//!
//! Update opportunistically; don't list numbers you can't trace.

use crate::report::shape::CompetitorRow;

/// Function type the `EvalRunner` takes to look up baselines for the
/// benchmark it's about to run.
pub type CompetitorBaselines = fn() -> Vec<CompetitorRow>;

/// The smoke corpus is Brain's own fixture, not a published
/// benchmark — there are no external competitor numbers to compare
/// against. The smoke report stands alone as a Recall@1 canary.
#[must_use]
pub fn smoke_competitor_baselines() -> Vec<CompetitorRow> {
    Vec::new()
}

/// The lexical-stress set is Brain's own adversarial fixture — there
/// are no published competitor numbers. Its value is internal: the gap
/// between substring recall@k (≈0 by construction) and context-recall.
#[must_use]
pub fn lexical_stress_competitor_baselines() -> Vec<CompetitorRow> {
    Vec::new()
}

/// The paraphrase-stress set is Brain's own generated adversarial fixture
/// — there are no published competitor numbers. Its value is internal:
/// the gap between substring recall@k (≈0) and context-recall at scale.
#[must_use]
pub fn paraphrase_stress_competitor_baselines() -> Vec<CompetitorRow> {
    Vec::new()
}

/// The supersession-stress set is Brain's own generated fixture probing
/// knowledge-update direction — there are no published competitor numbers.
#[must_use]
pub fn supersession_stress_competitor_baselines() -> Vec<CompetitorRow> {
    Vec::new()
}

/// Empty placeholder — fill in as DMR numbers from competitors are
/// collected. (The original MemGPT paper reported DMR; that's the
/// natural first row to add.)
#[must_use]
pub fn dmr_competitor_baselines() -> Vec<CompetitorRow> {
    Vec::new()
}

/// Published LongMemEval-S numbers worth comparing against. Source
/// citations are deliberate — every number is something a reader can
/// look up and verify.
#[must_use]
pub fn longmemeval_s_competitor_baselines() -> Vec<CompetitorRow> {
    vec![
        CompetitorRow {
            system: "GPT-4o (full-context)".into(),
            accuracy: 0.50,
            source: "Wu et al., LongMemEval, ICLR 2025 — Table 4".into(),
            note: "Reads the entire ~115k-token haystack each turn. Baseline upper bound on \
                   long-context-only retrieval; published numbers vary by ~3 pp across release \
                   versions."
                .into(),
        },
        CompetitorRow {
            system: "GPT-4o + RAG (default)".into(),
            accuracy: 0.32,
            source: "Wu et al., LongMemEval, ICLR 2025 — Table 4".into(),
            note: "Off-the-shelf BM25+embedding RAG; significantly under full-context on the \
                   multi-session and temporal dimensions."
                .into(),
        },
        CompetitorRow {
            system: "MemGPT".into(),
            accuracy: 0.37,
            source: "Wu et al., LongMemEval, ICLR 2025 — Table 4".into(),
            note: "Tool-augmented memory; reportedly strong on knowledge-update.".into(),
        },
        CompetitorRow {
            system: "Zep".into(),
            accuracy: 0.62,
            source: "Zep blog, 'Zep on LongMemEval' (2024)".into(),
            note: "Public number from the Zep blog post; not independently reproduced — flag this \
                   in any side-by-side until we can run their stack ourselves."
                .into(),
        },
    ]
}

/// Published LoCoMo numbers — with the honest call-out about the
/// Zep methodology bug. We include adversarial questions (category
/// 5) in the denominator per the standard protocol; any reported
/// number that doesn't is comparing different things.
#[must_use]
pub fn locomo_competitor_baselines() -> Vec<CompetitorRow> {
    vec![
        CompetitorRow {
            system: "GPT-4 (full-context)".into(),
            accuracy: 0.58,
            source: "Maharana et al., LoCoMo, ACL 2024 — Table 3".into(),
            note:
                "Full-context baseline. Cross-comparable; uses the standard 5-category denominator."
                    .into(),
        },
        CompetitorRow {
            system: "Zep (reported)".into(),
            accuracy: 0.833,
            source: "Zep paper / blog (2024)".into(),
            note: "Excluded category-5 (adversarial) from the denominator while counting correct \
                   abstentions in the numerator — inflates the score ~25 pp. See \
                   https://github.com/getzep/zep-papers/issues/5."
                .into(),
        },
        CompetitorRow {
            system: "Zep (re-scored, standard protocol)".into(),
            accuracy: 0.5844,
            source: "Same Zep data, re-scored under the LoCoMo standard 5-category denominator"
                .into(),
            note: "Use this when comparing apples-to-apples.".into(),
        },
    ]
}
