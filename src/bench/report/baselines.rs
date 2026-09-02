//! Published competitor numbers per benchmark.
//!
//! These tables decay fast — pin every row to a primary source and
//! call out methodology issues (the Zep LoCoMo footnote is the
//! template).
//!
//! Update opportunistically; don't list numbers you can't trace.

use crate::bench::report::shape::CompetitorRow;

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

/// Published DMR (Deep Memory Retrieval) numbers from the MemGPT paper,
/// which introduced the benchmark. All figures are LLM-judge accuracy from
/// Table 2 ("Deep memory retrieval (DMR) performance"), Packer et al.,
/// *MemGPT: Towards LLMs as Operating Systems*, 2023 (arXiv:2310.08560).
///
/// Note the shape of this benchmark: the fixed-context baselines are weak
/// (~0.32–0.39) because the gold turn falls outside the window, while
/// MemGPT — which can page the past conversation into recall memory —
/// nearly saturates it (~0.93). DMR is therefore a *floor* test (can the
/// system retrieve a single planted fact at all), not a discriminating
/// one; treat a Brain DMR score near MemGPT's as table-stakes, and lean on
/// LongMemEval / LoCoMo for the harder signal.
#[must_use]
pub fn dmr_competitor_baselines() -> Vec<CompetitorRow> {
    vec![
        CompetitorRow {
            system: "GPT-4 (fixed-context)".into(),
            accuracy: 0.321,
            source: "Packer et al., MemGPT, 2023 (arXiv:2310.08560) — Table 2".into(),
            note: "Fixed-context baseline: the gold turn is outside the window, so retrieval \
                   fails on most questions. The natural floor to clear."
                .into(),
        },
        CompetitorRow {
            system: "GPT-4 Turbo (fixed-context)".into(),
            accuracy: 0.353,
            source: "Packer et al., MemGPT, 2023 (arXiv:2310.08560) — Table 2".into(),
            note: "Strongest fixed-context baseline in the paper; still far below any \
                   memory-augmented system."
                .into(),
        },
        CompetitorRow {
            system: "MemGPT + GPT-4".into(),
            accuracy: 0.925,
            source: "Packer et al., MemGPT, 2023 (arXiv:2310.08560) — Table 2".into(),
            note: "Pages past conversation into recall memory; near-saturates DMR. This is the \
                   table-stakes bar, not a stretch target."
                .into(),
        },
        CompetitorRow {
            system: "MemGPT + GPT-4 Turbo".into(),
            accuracy: 0.934,
            source: "Packer et al., MemGPT, 2023 (arXiv:2310.08560) — Table 2".into(),
            note: "Best published DMR number in the introducing paper.".into(),
        },
    ]
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
