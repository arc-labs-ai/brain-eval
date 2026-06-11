//! DMR — the Deep Memory Retrieval benchmark from the MemGPT paper
//! (2023), single-hop fact retrieval across multi-session
//! conversations. 500 questions in the canonical release.
//!
//! ## File layout
//!
//! `BRAIN_EVAL_DATASETS_DIR/dmr/dmr.jsonl` — one JSON object per line.
//! Each line decodes to a single `EvalInstance`. Expected shape (the
//! field names follow the MemGPT release; `serde(rename)` glues them to
//! our internal struct):
//!
//! ```jsonc
//! {
//!   "id": "dmr-001",
//!   "question": "What is the user's favourite colour?",
//!   "answer": "blue",
//!   "conversation_id": "conv-007",
//!   "sessions": [
//!     {
//!       "session_id": "s1",
//!       "turns": [
//!         {"role": "user",      "content": "My favourite colour is blue."},
//!         {"role": "assistant", "content": "Got it."}
//!       ]
//!     }
//!   ]
//! }
//! ```
//!
//! Datasets that ship `question_type`-style tags can extend the loader
//! later; today every DMR row is treated as [`QuestionType::SingleHop`].

use std::path::Path;

use serde::Deserialize;

use crate::core::benchmark::{Benchmark, EvalError};
use crate::core::instance::{EvalInstance, QuestionType, Session, TurnRecord};

/// The DMR benchmark.
pub struct DmrBenchmark;

impl Benchmark for DmrBenchmark {
    fn id(&self) -> &'static str {
        "dmr"
    }

    fn display_name(&self) -> &'static str {
        "DMR (MemGPT 2023)"
    }

    fn url(&self) -> &'static str {
        "https://arxiv.org/abs/2310.08560"
    }

    fn load(&self, datasets_dir: &Path) -> Result<Vec<EvalInstance>, EvalError> {
        let path = datasets_dir.join("dmr").join("dmr.jsonl");
        let bytes = std::fs::read(&path).map_err(|_| EvalError::DatasetNotFound {
            path: path.display().to_string(),
        })?;
        let text = std::str::from_utf8(&bytes).map_err(|e| EvalError::ParseError {
            path: path.display().to_string(),
            reason: format!("non-UTF-8: {e}"),
        })?;

        let mut out = Vec::new();
        for (lineno, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let row: DmrRow = serde_json::from_str(trimmed).map_err(|e| EvalError::ParseError {
                path: path.display().to_string(),
                reason: format!("line {}: {e}", lineno + 1),
            })?;
            out.push(row.into_eval_instance());
        }
        Ok(out)
    }

    fn requires_synthesis(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Wire shapes (just for parsing — `From<DmrRow> for EvalInstance`)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct DmrRow {
    id: String,
    question: String,
    answer: String,
    conversation_id: Option<String>,
    #[serde(default)]
    sessions: Vec<DmrSession>,
}

#[derive(Debug, Deserialize)]
struct DmrSession {
    #[serde(default = "default_session_id")]
    session_id: String,
    #[serde(default)]
    turns: Vec<DmrTurn>,
}

#[derive(Debug, Deserialize)]
struct DmrTurn {
    role: String,
    content: String,
}

fn default_session_id() -> String {
    "session-0".to_owned()
}

impl DmrRow {
    fn into_eval_instance(self) -> EvalInstance {
        EvalInstance {
            question_id: self.id,
            question: self.question,
            answer: self.answer,
            question_type: QuestionType::SingleHop,
            conversation_id: self.conversation_id,
            sessions: self
                .sessions
                .into_iter()
                .map(|s| Session {
                    session_id: s.session_id,
                    turns: s
                        .turns
                        .into_iter()
                        .map(|t| TurnRecord {
                            role: t.role,
                            content: t.content,
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_minimal_row() {
        let line = r#"{"id":"dmr-001","question":"Colour?","answer":"blue","sessions":[]}"#;
        let row: DmrRow = serde_json::from_str(line).expect("parse");
        let inst = row.into_eval_instance();
        assert_eq!(inst.question_id, "dmr-001");
        assert_eq!(inst.question, "Colour?");
        assert_eq!(inst.answer, "blue");
        assert!(inst.sessions.is_empty());
        assert_eq!(inst.question_type, QuestionType::SingleHop);
    }

    #[test]
    fn parses_row_with_turns() {
        let line = r#"{"id":"x","question":"q","answer":"a","sessions":[{"session_id":"s","turns":[{"role":"user","content":"hi"}]}]}"#;
        let row: DmrRow = serde_json::from_str(line).expect("parse");
        let inst = row.into_eval_instance();
        assert_eq!(inst.sessions.len(), 1);
        assert_eq!(inst.sessions[0].turns.len(), 1);
        assert_eq!(inst.sessions[0].turns[0].role, "user");
    }

    #[test]
    fn missing_dataset_dir_returns_not_found() {
        let tmp = std::env::temp_dir().join("definitely-not-there-brain-eval");
        let err = DmrBenchmark.load(&tmp).unwrap_err();
        assert!(matches!(err, EvalError::DatasetNotFound { .. }));
    }
}
