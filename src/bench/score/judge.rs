//! Answer judging.
//!
//! Two paths:
//!
//! - **Heuristic** (always available): case-insensitive substring +
//!   token-overlap scoring against the ground truth. Cheap and
//!   deterministic; honest for fact-style benchmarks (DMR) and a
//!   directional signal for multi-hop ones (LME, LoCoMo).
//! - **LLM-as-judge** (behind the `live-llm` feature): proper
//!   evaluation via a real LLM. Required for honest LongMemEval and
//!   LoCoMo numbers.
//!
//! The `live-llm` path is intentionally not wired into the runner
//! yet — Brain doesn't expose a per-request LLM call surface from
//! `brain-db-sdk`. The hook lives here so it can be filled in once
//! the SDK grows a `judge_with_llm` method.

use crate::bench::core::instance::QuestionType;
use crate::bench::core::outcome::{JudgeResult, Verdict};

/// Heuristic judge — substring + token overlap.
///
/// Scoring rules:
///
/// 1. If `ground_truth` is empty or `qtype == Abstention`, a "don't
///    know" answer scores `Correct`, anything else scores `Incorrect`.
/// 2. If `system_answer` contains `ground_truth` as a case-insensitive
///    substring → `Correct`.
/// 3. If at least half the ground-truth tokens (length ≥ 3, lowercased,
///    punctuation-stripped) appear in the answer → `Partial`.
/// 4. Otherwise → `Incorrect`.
#[must_use]
pub fn judge_answer_heuristic(
    question_id: &str,
    qtype: QuestionType,
    ground_truth: &str,
    system_answer: &str,
) -> JudgeResult {
    let gt = ground_truth.trim();
    let ans = system_answer.trim();

    // Abstention rule.
    if matches!(qtype, QuestionType::Abstention) || gt.is_empty() {
        let ans_lower = ans.to_ascii_lowercase();
        let says_dont_know = ans.is_empty()
            || ans_lower.contains("don't know")
            || ans_lower.contains("do not know")
            || ans_lower.contains("not sure")
            || ans_lower.contains("unknown")
            || ans_lower.contains("not mentioned")
            || ans_lower.contains("no information");
        let verdict = if says_dont_know {
            Verdict::Correct
        } else {
            Verdict::Incorrect
        };
        return JudgeResult {
            question_id: question_id.to_owned(),
            verdict,
            score: verdict.score(),
            reasoning: if says_dont_know {
                "abstention: system correctly declined to answer".into()
            } else {
                "abstention: system produced an answer when none was expected".into()
            },
        };
    }

    let gt_lower = gt.to_ascii_lowercase();
    let ans_lower = ans.to_ascii_lowercase();

    if ans_lower.contains(&gt_lower) {
        return JudgeResult {
            question_id: question_id.to_owned(),
            verdict: Verdict::Correct,
            score: Verdict::Correct.score(),
            reasoning: "heuristic: answer contains ground truth as substring".into(),
        };
    }

    let gt_tokens = significant_tokens(&gt_lower);
    if gt_tokens.is_empty() {
        return JudgeResult {
            question_id: question_id.to_owned(),
            verdict: Verdict::Incorrect,
            score: Verdict::Incorrect.score(),
            reasoning: "heuristic: ground truth has no significant tokens".into(),
        };
    }

    let matched = gt_tokens
        .iter()
        .filter(|t| ans_lower.contains(t.as_str()))
        .count();
    #[allow(clippy::cast_precision_loss)]
    let overlap = matched as f64 / gt_tokens.len() as f64;

    let verdict = if overlap >= 0.5 {
        Verdict::Partial
    } else {
        Verdict::Incorrect
    };
    JudgeResult {
        question_id: question_id.to_owned(),
        verdict,
        score: verdict.score(),
        reasoning: format!(
            "heuristic: token overlap {matched}/{} = {:.2}",
            gt_tokens.len(),
            overlap
        ),
    }
}

fn significant_tokens(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| t.len() >= 3)
        .map(str::to_owned)
        .collect()
}

/// The answer-grading strategy — the LLM-as-judge when a `live-llm` build
/// has a provider key, else the always-available [`judge_answer_heuristic`].
///
/// A **closed set** of two strategies with an `async` grade step: the
/// idiomatic Rust shape for that is an enum (static dispatch, native
/// `async fn`) rather than a `dyn` trait object, which `async fn` can't
/// back without an extra dependency. The `#[cfg(feature = "live-llm")]`
/// juggling is confined to the variant and [`Judge::from_env`], so callers
/// (the dataset [`crate::bench::runner`] and the [`crate::bench::probe`]
/// staged probe) grade uniformly and never see the feature flag.
///
/// Scoped `pub(in crate::bench)` — it is the eval subsystem's shared grader,
/// not a crate-wide API.
pub(in crate::bench) enum Judge {
    /// Deterministic substring + token-overlap grader. Always available.
    Heuristic,
    /// LLM-as-judge. Only present in a `live-llm` build with a provider key.
    #[cfg(feature = "live-llm")]
    Llm(super::llm_judge::LlmJudge),
}

impl Judge {
    /// The LLM judge when configured, else the heuristic.
    pub(in crate::bench) fn from_env() -> Self {
        #[cfg(feature = "live-llm")]
        if let Some(llm) = super::llm_judge::LlmJudge::from_env() {
            return Self::Llm(llm);
        }
        Self::Heuristic
    }

    /// `llm:<provider>:<model>@<version>` when the LLM judge is active, else
    /// `"heuristic"` — the judge that actually ran.
    pub(in crate::bench) fn describe(&self) -> String {
        match self {
            Self::Heuristic => "heuristic".to_string(),
            #[cfg(feature = "live-llm")]
            Self::Llm(j) => j.describe(),
        }
    }

    /// Grade one answer against the gold.
    pub(in crate::bench) async fn grade(
        &self,
        question_id: &str,
        qtype: QuestionType,
        question: &str,
        ground_truth: &str,
        system_answer: &str,
    ) -> JudgeResult {
        match self {
            Self::Heuristic => {
                let _ = question; // the heuristic scores against gold alone
                judge_answer_heuristic(question_id, qtype, ground_truth, system_answer)
            }
            #[cfg(feature = "live-llm")]
            Self::Llm(j) => {
                j.judge(question_id, qtype, question, ground_truth, system_answer)
                    .await
            }
        }
    }

    /// Answer-supporting context recall: does the retrieved context fully
    /// support the gold answer? `None` for the heuristic — that verdict needs
    /// the LLM, so a heuristic run leaves it unjudged rather than guessing.
    pub(in crate::bench) async fn grade_support(
        &self,
        question: &str,
        ground_truth: &str,
        retrieved: &[String],
    ) -> (Option<bool>, String) {
        match self {
            Self::Heuristic => {
                let _ = (question, ground_truth, retrieved);
                (None, String::new())
            }
            #[cfg(feature = "live-llm")]
            Self::Llm(j) => match j.judge_support(question, ground_truth, retrieved).await {
                Some(v) => (Some(v.supported), v.reasoning),
                None => (None, String::new()),
            },
        }
    }

    /// How many `grade` calls the LLM judge fell back to the heuristic for.
    /// Always 0 for the heuristic (it is the fallback).
    pub(in crate::bench) fn heuristic_fallback_count(&self) -> usize {
        match self {
            Self::Heuristic => 0,
            #[cfg(feature = "live-llm")]
            Self::Llm(j) => j.heuristic_fallback_count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dmr_qtype() -> QuestionType {
        QuestionType::SingleHop
    }

    #[test]
    fn substring_match_is_correct() {
        let r = judge_answer_heuristic(
            "q1",
            dmr_qtype(),
            "Paris",
            "The user lives in Paris, France.",
        );
        assert_eq!(r.verdict, Verdict::Correct);
    }

    #[test]
    fn half_overlap_is_partial() {
        let r = judge_answer_heuristic(
            "q2",
            dmr_qtype(),
            "blue Toyota Corolla",
            "The car is Toyota and blue.",
        );
        assert!(matches!(r.verdict, Verdict::Partial | Verdict::Correct));
    }

    #[test]
    fn empty_answer_is_incorrect_when_truth_present() {
        let r = judge_answer_heuristic("q3", dmr_qtype(), "Paris", "");
        assert_eq!(r.verdict, Verdict::Incorrect);
    }

    #[test]
    fn abstention_with_dont_know_is_correct() {
        let r = judge_answer_heuristic("q4", QuestionType::Abstention, "", "I don't know.");
        assert_eq!(r.verdict, Verdict::Correct);
    }

    #[test]
    fn abstention_with_made_up_answer_is_incorrect() {
        let r = judge_answer_heuristic(
            "q5",
            QuestionType::Abstention,
            "",
            "The user's favorite color is green.",
        );
        assert_eq!(r.verdict, Verdict::Incorrect);
    }
}
