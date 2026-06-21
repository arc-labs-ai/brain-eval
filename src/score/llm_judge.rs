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

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use tracing::warn;

use crate::core::instance::QuestionType;
use crate::core::outcome::{JudgeResult, Verdict};
use crate::llm::{truncate, LlmClient};
use crate::score::judge::judge_answer_heuristic;
use crate::score::judge_prompt::{
    JUDGE_PROMPT_VERSION, SUPPORT_PROMPT_TEMPLATE, VERDICT_PROMPT_TEMPLATE,
};

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
    /// How many times [`Self::judge`] silently substituted a heuristic
    /// grade because the LLM call failed or returned an unparseable reply.
    /// Surfaced in the report so a run where the judge died partway can
    /// never look fully LLM-judged in the saved artifact.
    heuristic_fallbacks: AtomicUsize,
}

impl LlmJudge {
    /// Build from the environment, or `None` if no provider key is set.
    /// The model is overridable via `BRAIN_EVAL_JUDGE_MODEL`.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        Some(Self {
            client: LlmClient::from_env("BRAIN_EVAL_JUDGE_MODEL")?,
            warned: AtomicBool::new(false),
            heuristic_fallbacks: AtomicUsize::new(0),
        })
    }

    /// `llm:<provider>:<model>@<prompt-version>` identity for the report
    /// header. The prompt version pins which grading instructions ran, so
    /// a methodology change is visible in the report's `judge` line.
    #[must_use]
    pub fn describe(&self) -> String {
        format!("llm:{}@{JUDGE_PROMPT_VERSION}", self.client.describe())
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
        let prompt = render_verdict_prompt(question, ground_truth, system_answer);
        match self.client.complete(&prompt, MAX_TOKENS).await {
            Ok(reply) => parse_verdict(question_id, &reply).unwrap_or_else(|| {
                self.heuristic_fallbacks.fetch_add(1, Ordering::Relaxed);
                self.warn_once(&format!("unparseable reply: {}", truncate(&reply, 120)));
                judge_answer_heuristic(question_id, qtype, ground_truth, system_answer)
            }),
            Err(e) => {
                self.heuristic_fallbacks.fetch_add(1, Ordering::Relaxed);
                self.warn_once(&e);
                judge_answer_heuristic(question_id, qtype, ground_truth, system_answer)
            }
        }
    }

    /// How many `judge` calls fell back to the heuristic grader because the
    /// LLM call failed or returned an unparseable reply. A non-zero count
    /// means the run's accuracy is not fully LLM-judged.
    #[must_use]
    pub fn heuristic_fallback_count(&self) -> usize {
        self.heuristic_fallbacks.load(Ordering::Relaxed)
    }

    /// Answer-supporting context recall: does the *retrieved context*
    /// contain enough to derive the gold answer? This grades retrieval,
    /// not synthesis — separate from [`Self::judge`].
    ///
    /// Returns `None` on a failed call or unparseable reply (so the
    /// caller records the question as unjudged rather than guessing) and
    /// surfaces the failure on stderr once, matching `judge`'s fallback
    /// honesty. An empty retrieved set short-circuits to `Some(false)`:
    /// nothing was retrieved, so nothing can support the answer — no
    /// reason to spend a judge call.
    pub async fn judge_support(
        &self,
        question: &str,
        ground_truth: &str,
        retrieved: &[String],
    ) -> Option<SupportVerdict> {
        if retrieved.is_empty() {
            return Some(SupportVerdict {
                supported: false,
                reasoning: "no memories retrieved".to_owned(),
            });
        }
        let prompt = render_support_prompt(question, ground_truth, retrieved);
        match self.client.complete(&prompt, MAX_TOKENS).await {
            Ok(reply) => parse_support(&reply).or_else(|| {
                self.warn_once(&format!(
                    "unparseable support reply: {}",
                    truncate(&reply, 120)
                ));
                None
            }),
            Err(e) => {
                self.warn_once(&e);
                None
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

/// Render the pinned verdict template with this question's data. Kept as
/// plain replacement (not `format!`) so the hashed template in
/// [`crate::score::judge_prompt`] is the single source of the wording.
fn render_verdict_prompt(question: &str, ground_truth: &str, system_answer: &str) -> String {
    VERDICT_PROMPT_TEMPLATE
        .replace("{question}", question)
        .replace("{ground_truth}", ground_truth)
        .replace("{system_answer}", system_answer)
        // The template's JSON example uses literal `{{`/`}}`; un-escape
        // them now that no `format!` will.
        .replace("{{", "{")
        .replace("}}", "}")
}

/// Render the pinned support template. The retrieved memories are
/// numbered so the judge can reference them and so an empty line can't
/// blur two memories together.
fn render_support_prompt(question: &str, ground_truth: &str, retrieved: &[String]) -> String {
    let numbered = retrieved
        .iter()
        .enumerate()
        .map(|(i, m)| format!("[{}] {}", i + 1, m))
        .collect::<Vec<_>>()
        .join("\n");
    SUPPORT_PROMPT_TEMPLATE
        .replace("{question}", question)
        .replace("{ground_truth}", ground_truth)
        .replace("{retrieved}", &numbered)
        .replace("{{", "{")
        .replace("}}", "}")
}

/// A support verdict: did the retrieved context support the gold answer?
#[derive(Debug, Clone)]
pub struct SupportVerdict {
    /// Whether the gold answer is derivable from the retrieved memories.
    pub supported: bool,
    /// One-line rationale from the judge.
    pub reasoning: String,
}

#[derive(serde::Deserialize)]
struct VerdictReply {
    verdict: String,
    #[serde(default)]
    reasoning: String,
}

#[derive(serde::Deserialize)]
struct SupportReply {
    supported: bool,
    #[serde(default)]
    reasoning: String,
}

/// Parse a `{supported, reasoning}` object out of the model's reply.
/// Mirrors [`parse_verdict`]: tolerate surrounding prose by extracting
/// the first `{ .. }` span. `None` if no usable object is found.
fn parse_support(reply: &str) -> Option<SupportVerdict> {
    let start = reply.find('{')?;
    let end = reply.rfind('}')?;
    if end < start {
        return None;
    }
    let parsed: SupportReply = serde_json::from_str(&reply[start..=end]).ok()?;
    Some(SupportVerdict {
        supported: parsed.supported,
        reasoning: parsed.reasoning,
    })
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

    #[test]
    fn parses_support_yes() {
        let r = parse_support(r#"{"supported": true, "reasoning": "memory [2] states it"}"#)
            .expect("parse");
        assert!(r.supported);
        assert_eq!(r.reasoning, "memory [2] states it");
    }

    #[test]
    fn parses_support_no_wrapped_in_prose() {
        let reply = "Verdict:\n{\"supported\": false, \"reasoning\": \"fact absent\"}\nDone.";
        let r = parse_support(reply).expect("parse");
        assert!(!r.supported);
    }

    #[test]
    fn support_garbage_is_none() {
        assert!(parse_support("no json").is_none());
        assert!(parse_support(r#"{"supported":"maybe"}"#).is_none());
    }

    #[test]
    fn verdict_template_renders_without_leftover_braces() {
        let p = render_verdict_prompt("Q?", "A", "B");
        assert!(p.contains("Question: Q?"));
        assert!(!p.contains("{{"));
        assert!(!p.contains("{question}"));
    }

    #[test]
    fn support_template_numbers_memories() {
        let p = render_support_prompt("Q?", "A", &["mem one".into(), "mem two".into()]);
        assert!(p.contains("[1] mem one"));
        assert!(p.contains("[2] mem two"));
        assert!(!p.contains("{retrieved}"));
    }
}
