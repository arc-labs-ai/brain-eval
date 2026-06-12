//! LLM-as-judge — proper scoring for free-form benchmark answers.
//!
//! The heuristic judge ([`super::judge`]) is honest for fact-style
//! questions but too blunt for the multi-hop / paraphrased answers in
//! LongMemEval and LoCoMo. This judge asks a real LLM to grade the
//! system's answer against the reference, returning correct / partial /
//! incorrect with a one-line reason.
//!
//! Provider-agnostic: it auto-detects `ANTHROPIC_API_KEY` (Claude, the
//! default) or `OPENAI_API_KEY`, with the model overridable via
//! `BRAIN_EVAL_JUDGE_MODEL`. Compiled only under the `live-llm` feature
//! (it pulls in `reqwest`); without a key, [`LlmJudge::from_env`] returns
//! `None` and the runner stays on the heuristic judge.
//!
//! Robustness: each grade retries on transient/HTTP errors, and a final
//! failure (or an unparseable reply) falls back to the heuristic judge for
//! that one question rather than aborting the run — a flaky API call must
//! not throw away an otherwise-complete benchmark.

use std::time::Duration;

use serde::Deserialize;
use tracing::warn;

use crate::core::instance::QuestionType;
use crate::core::outcome::{JudgeResult, Verdict};
use crate::score::judge::judge_answer_heuristic;

const ANTHROPIC_URL: &str = "https://api.anthropic.com/v1/messages";
const OPENAI_URL: &str = "https://api.openai.com/v1/chat/completions";
const DEFAULT_ANTHROPIC_MODEL: &str = "claude-haiku-4-5-20251001";
const DEFAULT_OPENAI_MODEL: &str = "gpt-4o-mini";
const MAX_RETRIES: u32 = 3;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_TOKENS: u32 = 512;

#[derive(Clone, Copy, Debug)]
enum Provider {
    Anthropic,
    OpenAI,
}

/// A configured LLM grader.
pub struct LlmJudge {
    client: reqwest::Client,
    provider: Provider,
    api_key: String,
    model: String,
}

impl LlmJudge {
    /// Build a judge from the environment, or `None` if no provider key is
    /// set. Prefers Anthropic; `BRAIN_EVAL_JUDGE_MODEL` overrides the model.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        let model_override = std::env::var("BRAIN_EVAL_JUDGE_MODEL")
            .ok()
            .filter(|s| !s.trim().is_empty());

        let (provider, api_key, default_model) = if let Some(k) = nonempty_env("ANTHROPIC_API_KEY")
        {
            (Provider::Anthropic, k, DEFAULT_ANTHROPIC_MODEL)
        } else if let Some(k) = nonempty_env("OPENAI_API_KEY") {
            (Provider::OpenAI, k, DEFAULT_OPENAI_MODEL)
        } else {
            return None;
        };

        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .ok()?;

        Some(Self {
            client,
            provider,
            api_key,
            model: model_override.unwrap_or_else(|| default_model.to_string()),
        })
    }

    /// Human-readable judge identity for the report header.
    #[must_use]
    pub fn describe(&self) -> String {
        let p = match self.provider {
            Provider::Anthropic => "anthropic",
            Provider::OpenAI => "openai",
        };
        format!("llm:{p}:{}", self.model)
    }

    /// Grade one answer. Falls back to the heuristic judge on a final API
    /// failure or an unparseable reply.
    pub async fn judge(
        &self,
        question_id: &str,
        qtype: QuestionType,
        question: &str,
        ground_truth: &str,
        system_answer: &str,
    ) -> JudgeResult {
        let prompt = build_prompt(question, ground_truth, system_answer);
        match self.grade_with_retry(&prompt).await {
            Ok(reply) => parse_verdict(question_id, &reply).unwrap_or_else(|| {
                warn!(question_id, reply = %reply, "llm judge reply unparseable; heuristic fallback");
                judge_answer_heuristic(question_id, qtype, ground_truth, system_answer)
            }),
            Err(e) => {
                warn!(question_id, error = %e, "llm judge failed after retries; heuristic fallback");
                judge_answer_heuristic(question_id, qtype, ground_truth, system_answer)
            }
        }
    }

    async fn grade_with_retry(&self, prompt: &str) -> Result<String, String> {
        let mut last_err = String::from("no attempt made");
        for attempt in 0..MAX_RETRIES {
            if attempt > 0 {
                // 0.5s, 1s, 2s backoff.
                let backoff = Duration::from_millis(500u64 << (attempt - 1));
                tokio::time::sleep(backoff).await;
            }
            match self.call(prompt).await {
                Ok(text) => return Ok(text),
                Err(e) => last_err = e,
            }
        }
        Err(last_err)
    }

    async fn call(&self, prompt: &str) -> Result<String, String> {
        match self.provider {
            Provider::Anthropic => self.call_anthropic(prompt).await,
            Provider::OpenAI => self.call_openai(prompt).await,
        }
    }

    async fn call_anthropic(&self, prompt: &str) -> Result<String, String> {
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "messages": [{ "role": "user", "content": prompt }],
        });
        let resp = self
            .client
            .post(ANTHROPIC_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let status = resp.status();
        let text = resp.text().await.map_err(|e| e.to_string())?;
        if !status.is_success() {
            return Err(format!("anthropic {status}: {}", truncate(&text, 300)));
        }
        #[derive(Deserialize)]
        struct Resp {
            content: Vec<Block>,
        }
        #[derive(Deserialize)]
        struct Block {
            #[serde(default)]
            text: String,
        }
        let parsed: Resp = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        Ok(parsed.content.into_iter().map(|b| b.text).collect())
    }

    async fn call_openai(&self, prompt: &str) -> Result<String, String> {
        let body = serde_json::json!({
            "model": self.model,
            "temperature": 0,
            "messages": [{ "role": "user", "content": prompt }],
        });
        let resp = self
            .client
            .post(OPENAI_URL)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let status = resp.status();
        let text = resp.text().await.map_err(|e| e.to_string())?;
        if !status.is_success() {
            return Err(format!("openai {status}: {}", truncate(&text, 300)));
        }
        #[derive(Deserialize)]
        struct Resp {
            choices: Vec<Choice>,
        }
        #[derive(Deserialize)]
        struct Choice {
            message: Message,
        }
        #[derive(Deserialize)]
        struct Message {
            #[serde(default)]
            content: String,
        }
        let parsed: Resp = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        Ok(parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default())
    }
}

fn nonempty_env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
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

#[derive(Deserialize)]
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

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max])
    }
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
