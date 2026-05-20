//! `ScriptedLlm` — a deterministic mock LLM for offline tests.
//!
//! Brain's substrate doesn't expose an in-process LLM client through the
//! SDK (the extractor stack lives server-side). This struct is a
//! pure-Rust placeholder that future LLM-judge / LLM-answer-synthesis
//! paths can target without a network round trip.
//!
//! Pattern matching is substring-based and ordered: the first
//! `(pattern, response)` pair whose `pattern` appears in the prompt
//! wins. Put more specific patterns first.

use std::sync::Mutex;

/// Deterministic, prompt-driven mock.
pub struct ScriptedLlm {
    rules: Vec<(String, String)>,
    fallback: String,
    /// Recorded prompts in call order — useful for assertions in tests.
    calls: Mutex<Vec<String>>,
}

impl ScriptedLlm {
    /// Build a fresh scripted LLM. `rules` are tried in order on each
    /// `respond_to` call; if none match, `fallback` is returned.
    #[must_use]
    pub fn new(rules: Vec<(String, String)>, fallback: impl Into<String>) -> Self {
        Self {
            rules,
            fallback: fallback.into(),
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Look up a deterministic response for `prompt`. The pattern that
    /// matched (or `"<fallback>"`) is returned alongside the response so
    /// callers can sanity-check pattern hit-rates in tests.
    pub fn respond_to(&self, prompt: &str) -> ScriptedResponse {
        if let Ok(mut calls) = self.calls.lock() {
            calls.push(prompt.to_owned());
        }
        for (pat, resp) in &self.rules {
            if prompt.contains(pat) {
                return ScriptedResponse {
                    matched_pattern: pat.clone(),
                    response: resp.clone(),
                };
            }
        }
        ScriptedResponse {
            matched_pattern: "<fallback>".to_owned(),
            response: self.fallback.clone(),
        }
    }

    /// All prompts seen by this mock since construction.
    #[must_use]
    pub fn recorded_calls(&self) -> Vec<String> {
        self.calls
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }
}

/// Output of a single `ScriptedLlm::respond_to` call.
#[derive(Debug, Clone)]
pub struct ScriptedResponse {
    /// The pattern that matched the prompt, or `"<fallback>"`.
    pub matched_pattern: String,
    /// The configured response for that pattern (or the fallback).
    pub response: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_matching_rule_wins() {
        let llm = ScriptedLlm::new(
            vec![
                ("Paris".to_owned(), "France".to_owned()),
                ("city".to_owned(), "<generic-city>".to_owned()),
            ],
            "unknown".to_owned(),
        );
        let out = llm.respond_to("Where is Paris, the city of lights?");
        assert_eq!(out.matched_pattern, "Paris");
        assert_eq!(out.response, "France");
    }

    #[test]
    fn fallback_when_no_rule_matches() {
        let llm = ScriptedLlm::new(vec![], "unknown".to_owned());
        let out = llm.respond_to("anything");
        assert_eq!(out.matched_pattern, "<fallback>");
        assert_eq!(out.response, "unknown");
    }

    #[test]
    fn records_call_history() {
        let llm = ScriptedLlm::new(vec![], "x".to_owned());
        let _ = llm.respond_to("first");
        let _ = llm.respond_to("second");
        let calls = llm.recorded_calls();
        assert_eq!(calls, vec!["first".to_owned(), "second".to_owned()]);
    }
}
