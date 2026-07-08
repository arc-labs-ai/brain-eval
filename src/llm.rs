//! Shared LLM HTTP client for the `live-llm` eval paths (answer synthesis
//! and the answer judge).
//!
//! Provider-agnostic and single-credential: the key comes from the one
//! shared `BRAIN__LLM__API_KEY` (the same var Brain itself uses), and the
//! provider (Anthropic / OpenAI) is derived from the model id. The model
//! is overridable via a caller-named env var (so the synthesizer and judge
//! can use different models), falling back to `BRAIN__LLM__MODEL` then a
//! built-in default. Compiled only under the `live-llm` feature (it pulls
//! in `reqwest`).
//!
//! [`LlmClient::complete`] retries transient failures (5xx / 429 / network)
//! and fails fast on 4xx (bad key, no credit, unknown model), returning the
//! error message so the caller can fall back to its non-LLM path and
//! surface the reason.

use std::time::Duration;

use serde::Deserialize;

const ANTHROPIC_URL: &str = "https://api.anthropic.com/v1/messages";
const OPENAI_URL: &str = "https://api.openai.com/v1/chat/completions";
/// Default model when neither the caller-specific override nor
/// `BRAIN__LLM__MODEL` is set. Provider is derived from the id.
const DEFAULT_MODEL: &str = "gpt-4o-mini";
const MAX_RETRIES: u32 = 3;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone, Copy, Debug)]
enum Provider {
    Anthropic,
    OpenAI,
}

/// A configured provider client.
pub struct LlmClient {
    client: reqwest::Client,
    provider: Provider,
    api_key: String,
    model: String,
}

/// A single API call's failure, with whether retrying could help.
struct CallError {
    message: String,
    retryable: bool,
}

impl LlmClient {
    /// Build from the single shared credential `BRAIN__LLM__API_KEY`, or
    /// `None` if it is unset. The model resolves from `model_override_env`
    /// (e.g. `BRAIN_EVAL_JUDGE_MODEL`) first, then the shared
    /// `BRAIN__LLM__MODEL`, then [`DEFAULT_MODEL`]; the provider is derived
    /// from the resolved model id.
    #[must_use]
    pub fn from_env(model_override_env: &str) -> Option<Self> {
        let api_key = nonempty_env("BRAIN__LLM__API_KEY")?;
        let model = std::env::var(model_override_env)
            .ok()
            .filter(|s| !s.trim().is_empty())
            .or_else(|| nonempty_env("BRAIN__LLM__MODEL"))
            .unwrap_or_else(|| DEFAULT_MODEL.to_string());
        let provider = provider_for_model(&model);

        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .ok()?;

        Some(Self {
            client,
            provider,
            api_key,
            model,
        })
    }

    /// `<provider>:<model>` identity for report headers.
    #[must_use]
    pub fn describe(&self) -> String {
        let p = match self.provider {
            Provider::Anthropic => "anthropic",
            Provider::OpenAI => "openai",
        };
        format!("{p}:{}", self.model)
    }

    /// Send `prompt`, returning the model's text reply. Retries transient
    /// failures; fails fast on 4xx. `Err` carries a human-readable reason.
    pub async fn complete(&self, prompt: &str, max_tokens: u32) -> Result<String, String> {
        self.complete_inner(prompt, max_tokens, false).await
    }

    /// Like [`Self::complete`], but constrains the reply to a JSON object
    /// when the provider supports it (OpenAI `response_format: json_object`).
    /// The prompt must itself mention JSON for OpenAI to honor the mode; ours
    /// do. Anthropic has no equivalent knob — the prompt's explicit
    /// "respond with ONLY a JSON object" instruction carries it there — so
    /// this is a no-op for Anthropic and behaves exactly like `complete`.
    pub async fn complete_json(&self, prompt: &str, max_tokens: u32) -> Result<String, String> {
        self.complete_inner(prompt, max_tokens, true).await
    }

    async fn complete_inner(
        &self,
        prompt: &str,
        max_tokens: u32,
        json_mode: bool,
    ) -> Result<String, String> {
        let mut last_err = String::from("no attempt made");
        for attempt in 0..MAX_RETRIES {
            if attempt > 0 {
                // 0.5s, 1s, 2s backoff.
                tokio::time::sleep(Duration::from_millis(500u64 << (attempt - 1))).await;
            }
            match self.call(prompt, max_tokens, json_mode).await {
                Ok(text) => return Ok(text),
                Err(e) => {
                    let retryable = e.retryable;
                    last_err = e.message;
                    if !retryable {
                        break;
                    }
                }
            }
        }
        Err(last_err)
    }

    async fn call(
        &self,
        prompt: &str,
        max_tokens: u32,
        json_mode: bool,
    ) -> Result<String, CallError> {
        match self.provider {
            Provider::Anthropic => self.call_anthropic(prompt, max_tokens).await,
            Provider::OpenAI => self.call_openai(prompt, max_tokens, json_mode).await,
        }
    }

    async fn call_anthropic(&self, prompt: &str, max_tokens: u32) -> Result<String, CallError> {
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": max_tokens,
            // Deterministic grading/synthesis for reproducible eval numbers.
            "temperature": 0,
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
            .map_err(net_err)?;
        let status = resp.status();
        let text = resp.text().await.map_err(net_err)?;
        if !status.is_success() {
            return Err(http_err("anthropic", status, &text));
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
        let parsed: Resp = serde_json::from_str(&text).map_err(parse_err)?;
        Ok(parsed.content.into_iter().map(|b| b.text).collect())
    }

    async fn call_openai(
        &self,
        prompt: &str,
        max_tokens: u32,
        json_mode: bool,
    ) -> Result<String, CallError> {
        let mut body = serde_json::json!({
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [{ "role": "user", "content": prompt }],
        });
        if json_mode {
            // Constrain the reply to a syntactically valid JSON object so the
            // grader never has to salvage a verdict from prose.
            body["response_format"] = serde_json::json!({ "type": "json_object" });
        }
        let resp = self
            .client
            .post(OPENAI_URL)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(net_err)?;
        let status = resp.status();
        let text = resp.text().await.map_err(net_err)?;
        if !status.is_success() {
            return Err(http_err("openai", status, &text));
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
        let parsed: Resp = serde_json::from_str(&text).map_err(parse_err)?;
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

/// Derive the provider from the model id, matching Brain's own rule:
/// `claude*` → Anthropic, everything else → OpenAI.
fn provider_for_model(model: &str) -> Provider {
    if model.to_ascii_lowercase().starts_with("claude") {
        Provider::Anthropic
    } else {
        Provider::OpenAI
    }
}

/// Network/transport error — worth a retry.
fn net_err(e: reqwest::Error) -> CallError {
    CallError {
        message: e.to_string(),
        retryable: true,
    }
}

/// HTTP error response. 5xx / 429 are transient; 4xx (bad key, no credit,
/// unknown model) are not.
fn http_err(provider: &str, status: reqwest::StatusCode, body: &str) -> CallError {
    CallError {
        message: format!("{provider} {status}: {}", truncate(body, 300)),
        retryable: status.is_server_error() || status.as_u16() == 429,
    }
}

/// Malformed success body — a retry won't change it.
fn parse_err(e: serde_json::Error) -> CallError {
    CallError {
        message: e.to_string(),
        retryable: false,
    }
}

/// Truncate to at most `max` bytes with an ellipsis (for error/log
/// messages). Cuts on a UTF-8 char boundary so a multibyte sequence is
/// never split — this runs on error/fallback paths (API error bodies,
/// unparseable model replies) where the input is untrusted, so slicing
/// `&s[..max]` directly would panic exactly when something is already wrong.
#[must_use]
pub fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let end = (0..=max)
        .rev()
        .find(|&i| s.is_char_boundary(i))
        .unwrap_or(0);
    format!("{}…", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::truncate;

    #[test]
    fn truncate_short_string_is_unchanged() {
        assert_eq!(truncate("hi", 300), "hi");
    }

    #[test]
    fn truncate_never_splits_a_multibyte_codepoint() {
        // "€" is 3 bytes; a byte-cut at max=1 or 2 lands mid-codepoint and
        // would panic on `&s[..max]`. The boundary-safe cut must not.
        let s = "€€€€€"; // 15 bytes, 5 chars
        for max in 0..=15 {
            let out = truncate(s, max); // must not panic for any cut point
            assert!(s.starts_with(out.trim_end_matches('…')));
        }
    }
}
