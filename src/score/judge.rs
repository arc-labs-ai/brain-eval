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
//! `brain-sdk-rust`. The hook lives here so it can be filled in once
//! the SDK grows a `judge_with_llm` method.

use crate::core::instance::QuestionType;
use crate::core::outcome::{JudgeResult, Verdict};

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
