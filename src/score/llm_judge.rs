//! LLM-as-judge — proper scoring for free-form benchmark answers.
//!
//! The heuristic judge ([`super::judge`]) is honest for fact-style
//! questions but too blunt for the multi-hop / paraphrased answers in
//! LongMemEval and LoCoMo. This judge asks a real LLM (via the shared
//! [`crate::llm::LlmClient`]) to grade the system's answer against the
//! reference, returning correct / partial / incorrect with a one-line
//! reason.
//!
//! Compiled only under the `live-llm` feature; without a provider key
//! [`LlmJudge::from_env`] returns `None` and the runner stays on the
//! heuristic judge. A failed call (or unparseable reply) falls back to the
//! heuristic for that one question and prints a one-time stderr warning so
//! a credit-less / wrong key can't masquerade as LLM-judged.

use std::sync::atomic::{AtomicBool, Ordering};

use tracing::warn;

use crate::core::instance::QuestionType;
use crate::core::outcome::{JudgeResult, Verdict};
use crate::llm::{truncate, LlmClient};
use crate::score::judge::judge_answer_heuristic;

/// Token budget for the verdict JSON. Generous so a verbose `reasoning`
/// can't get truncated mid-object (which would leave the JSON unclosed and
/// unparseable, forcing a heuristic fallback).
const MAX_TOKENS: u32 = 512;

/// A configured LLM grader.
pub struct LlmJudge {
    client: LlmClient,
    /// Set once we've surfaced a judge failure, so the "falling back to
    /// heuristic" warning prints to stderr exactly once per run.
    warned: AtomicBool,
}

impl LlmJudge {
    /// Build from the environment, or `None` if no provider key is set.
    /// The model is overridable via `BRAIN_EVAL_JUDGE_MODEL`.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        Some(Self {
            client: LlmClient::from_env("BRAIN_EVAL_JUDGE_MODEL")?,
            warned: AtomicBool::new(false),
        })
    }

    /// `llm:<provider>:<model>` identity for the report header.
    #[must_use]
    pub fn describe(&self) -> String {
        format!("llm:{}", self.client.describe())
    }

    /// Grade one answer. Falls back to the heuristic judge on a failed call
    /// or an unparseable reply.
    pub async fn judge(
        &self,
        question_id: &str,
        qtype: QuestionType,
        question: &str,
        ground_truth: &str,
        system_answer: &str,
    ) -> JudgeResult {
        let prompt = build_prompt(question, ground_truth, system_answer);
        match self.client.complete(&prompt, MAX_TOKENS).await {
            Ok(reply) => parse_verdict(question_id, &reply).unwrap_or_else(|| {
                self.warn_once(&format!("unparseable reply: {}", truncate(&reply, 120)));
                judge_answer_heuristic(question_id, qtype, ground_truth, system_answer)
            }),
            Err(e) => {
                self.warn_once(&e);
                judge_answer_heuristic(question_id, qtype, ground_truth, system_answer)
            }
        }
    }

    /// Surface the first judge failure on stderr (brain-eval installs no
    /// tracing subscriber, so a silent fallback would otherwise hide a bad
    /// key / no credit / wrong model behind heuristic scores).
    fn warn_once(&self, message: &str) {
        warn!(error = %message, "llm judge failed; heuristic fallback");
        if !self.warned.swap(true, Ordering::Relaxed) {
            eprintln!(
                "warning: LLM judge call failed ({message}). Falling back to the HEURISTIC \
                 judge for ungraded questions — reported accuracy is NOT LLM-judged. Check the \
                 API key / credit balance, or set BRAIN_EVAL_JUDGE_MODEL."
            );
        }
    }
}

fn build_prompt(question: &str, ground_truth: &str, system_answer: &str) -> String {
    format!(
        "You are a strict grader for a memory question-answering benchmark. \
         Decide whether the system's answer is correct given the reference answer.\n\n\
         Question: {question}\n\
         Reference answer: {ground_truth}\n\
         System answer: {system_answer}\n\n\
         Grade \"correct\" if the system answer conveys the reference answer's key \
         facts (a paraphrase or extra detail is fine). Grade \"partial\" if it is \
         only partially right or omits a key detail. Grade \"incorrect\" if it is \
         wrong, irrelevant, or empty. If the reference answer indicates the question \
         is unanswerable, grade \"correct\" only when the system declined to answer.\n\n\
         Respond with ONLY a JSON object, no prose:\n\
         {{\"verdict\": \"correct\" | \"partial\" | \"incorrect\", \"reasoning\": \"<one short sentence>\"}}"
    )
}

#[derive(serde::Deserialize)]
struct VerdictReply {
    verdict: String,
    #[serde(default)]
    reasoning: String,
}

/// Parse a verdict out of the model's reply. Tolerates surrounding prose by
/// extracting the first `{ .. }` span. `None` if no usable verdict is found.
fn parse_verdict(question_id: &str, reply: &str) -> Option<JudgeResult> {
    let start = reply.find('{')?;
    let end = reply.rfind('}')?;
    if end < start {
        return None;
    }
    let json = &reply[start..=end];
    let parsed: VerdictReply = serde_json::from_str(json).ok()?;
    let verdict = match parsed.verdict.trim().to_ascii_lowercase().as_str() {
        "correct" => Verdict::Correct,
        "partial" => Verdict::Partial,
        "incorrect" => Verdict::Incorrect,
        _ => return None,
    };
    Some(JudgeResult {
        question_id: question_id.to_owned(),
        verdict,
        score: verdict.score(),
        reasoning: parsed.reasoning,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_clean_json() {
        let r = parse_verdict("q1", r#"{"verdict":"correct","reasoning":"matches"}"#).unwrap();
        assert_eq!(r.verdict, Verdict::Correct);
        assert_eq!(r.reasoning, "matches");
        assert_eq!(r.question_id, "q1");
    }

    #[test]
    fn parses_json_wrapped_in_prose() {
        let reply = "Here is my grade:\n{\"verdict\": \"PARTIAL\", \"reasoning\": \"missing date\"}\nThanks.";
        let r = parse_verdict("q2", reply).unwrap();
        assert_eq!(r.verdict, Verdict::Partial);
        assert!((r.score - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn incorrect_maps_to_zero() {
        let r = parse_verdict("q3", r#"{"verdict":"incorrect","reasoning":"wrong"}"#).unwrap();
        assert_eq!(r.verdict, Verdict::Incorrect);
        assert!(r.score.abs() < f64::EPSILON);
    }

    #[test]
    fn unknown_or_garbage_is_none() {
        assert!(parse_verdict("q", "no json here").is_none());
        assert!(parse_verdict("q", r#"{"verdict":"maybe"}"#).is_none());
    }

    #[test]
    fn missing_reasoning_defaults_empty() {
        let r = parse_verdict("q", r#"{"verdict":"correct"}"#).unwrap();
        assert_eq!(r.verdict, Verdict::Correct);
        assert_eq!(r.reasoning, "");
    }
}
