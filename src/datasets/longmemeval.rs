//! LongMemEval (ICLR 2025) — the de-facto agent-memory benchmark.
//!
//! Paper: <https://arxiv.org/abs/2410.10813>.
//! Release: <https://github.com/xiaowu0162/LongMemEval>.
//!
//! ## What it tests
//!
//! 500 questions per `_s` (Small) release, each backed by a multi-session
//! conversation "haystack" the model must remember across. Question
//! types cover the five dimensions LongMemEval scores on:
//!
//! - `single-session-user` / `single-session-assistant` — recall of a
//!   fact stated once in a single session.
//! - `single-session-preference` — preference recall.
//! - `multi-session` — synthesis across sessions.
//! - `temporal-reasoning` — time-aware queries.
//! - `knowledge-update` — return the latest value after an update.
//! - `abstention` — refuse when information was never mentioned.
//!
//! ## File layout
//!
//! `$BRAIN_EVAL_DATASETS_DIR/longmemeval/longmemeval_s.json` — a single
//! JSON file containing an array of objects, each matching the wire
//! shape below.
//!
//! ## Wire shape (per the LongMemEval release)
//!
//! ```jsonc
//! {
//!   "question_id":   "abc-001",
//!   "question_type": "multi-session",
//!   "question":      "When did the user move to Berlin?",
//!   "answer":        "March 2024",
//!   "haystack_sessions": [
//!     {
//!       "session_id": "s-1",
//!       "session_date": "2024-02-15",
//!       "turns": [
//!         {"role": "user",      "content": "I'm planning a move."},
//!         {"role": "assistant", "content": "Where to?"}
//!       ]
//!     }
//!   ],
//!   "answer_session_ids": ["s-2"]
//! }
//! ```
//!
//! Fields the loader currently ignores (kept for future enrichment):
//! `session_date`, `answer_session_ids`.

use std::path::Path;

use serde::Deserialize;

use crate::core::benchmark::{Benchmark, EvalError};
use crate::core::instance::{EvalInstance, QuestionType, Session, TurnRecord};

/// LongMemEval-S — the 500-question Small variant. Reads
/// `longmemeval/longmemeval_s.json` under the datasets dir.
pub struct LongMemEvalS;

impl Benchmark for LongMemEvalS {
    fn id(&self) -> &'static str {
        "longmemeval-s"
    }

    fn display_name(&self) -> &'static str {
        "LongMemEval (S variant, ICLR 2025)"
    }

    fn url(&self) -> &'static str {
        "https://arxiv.org/abs/2410.10813"
    }

    fn load(&self, datasets_dir: &Path) -> Result<Vec<EvalInstance>, EvalError> {
        let path = datasets_dir
            .join("longmemeval")
            .join("longmemeval_s.json");
        let bytes = std::fs::read(&path).map_err(|_| EvalError::DatasetNotFound {
            path: path.display().to_string(),
        })?;

        // The release ships either a JSON array or one-object-per-line
        // JSONL depending on the snapshot. Try the array first; fall
        // back to lenient line-by-line if that fails so we can read
        // both shapes without a separate command.
        let rows: Vec<LmeRow> = serde_json::from_slice(&bytes).or_else(|_| {
            let text = std::str::from_utf8(&bytes).map_err(|e| EvalError::ParseError {
                path: path.display().to_string(),
                reason: format!("non-UTF-8: {e}"),
            })?;
            parse_jsonl(text, &path)
        })?;

        Ok(rows.into_iter().map(LmeRow::into_eval_instance).collect())
    }

    /// LongMemEval expects free-form natural-language answers — the
    /// runner should hand candidate retrievals to a real LLM for
    /// synthesis. Heuristic mode still works as a directional signal.
    fn requires_synthesis(&self) -> bool {
        true
    }
}

fn parse_jsonl(text: &str, path: &Path) -> Result<Vec<LmeRow>, EvalError> {
    let mut out = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let row: LmeRow = serde_json::from_str(trimmed).map_err(|e| EvalError::ParseError {
            path: path.display().to_string(),
            reason: format!("line {}: {e}", lineno + 1),
        })?;
        out.push(row);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Wire shapes
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct LmeRow {
    question_id: String,
    #[serde(default = "default_qtype")]
    question_type: String,
    question: String,
    answer: String,
    #[serde(default)]
    haystack_sessions: Vec<LmeSession>,
}

#[derive(Debug, Deserialize)]
struct LmeSession {
    #[serde(default = "default_session_id")]
    session_id: String,
    #[serde(default)]
    turns: Vec<LmeTurn>,
}

#[derive(Debug, Deserialize)]
struct LmeTurn {
    role: String,
    content: String,
}

fn default_qtype() -> String {
    "single-session-user".to_owned()
}

fn default_session_id() -> String {
    "session-0".to_owned()
}

impl LmeRow {
    fn into_eval_instance(self) -> EvalInstance {
        let question_type = map_question_type(&self.question_type);
        EvalInstance {
            question_id: self.question_id,
            question: self.question,
            answer: self.answer,
            question_type,
            // Every LongMemEval question carries its own haystack; the
            // runner ingests fresh sessions per question. No conversation
            // sharing.
            conversation_id: None,
            sessions: self
                .haystack_sessions
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

fn map_question_type(raw: &str) -> QuestionType {
    match raw {
        "single-session-user" | "single-session-assistant" => QuestionType::SingleHop,
        "single-session-preference" => QuestionType::Preference,
        "multi-session" => QuestionType::MultiHop,
        "temporal-reasoning" => QuestionType::Temporal,
        "knowledge-update" => QuestionType::KnowledgeUpdate,
        "abstention" => QuestionType::Abstention,
        _ => QuestionType::Other,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_row() -> &'static str {
        r#"{
            "question_id": "lme-001",
            "question_type": "multi-session",
            "question": "When did the user move to Berlin?",
            "answer": "March 2024",
            "haystack_sessions": [
                {
                    "session_id": "s-1",
                    "turns": [
                        {"role": "user", "content": "I'm planning a move."},
                        {"role": "assistant", "content": "Where to?"}
                    ]
                }
            ]
        }"#
    }

    #[test]
    fn parses_a_row_with_haystack() {
        let row: LmeRow = serde_json::from_str(sample_row()).expect("parse");
        let inst = row.into_eval_instance();
        assert_eq!(inst.question_id, "lme-001");
        assert_eq!(inst.question_type, QuestionType::MultiHop);
        assert_eq!(inst.sessions.len(), 1);
        assert_eq!(inst.sessions[0].turns.len(), 2);
    }

    #[test]
    fn maps_each_known_question_type() {
        assert_eq!(
            map_question_type("single-session-user"),
            QuestionType::SingleHop
        );
        assert_eq!(
            map_question_type("single-session-preference"),
            QuestionType::Preference
        );
        assert_eq!(map_question_type("multi-session"), QuestionType::MultiHop);
        assert_eq!(
            map_question_type("temporal-reasoning"),
            QuestionType::Temporal
        );
        assert_eq!(
            map_question_type("knowledge-update"),
            QuestionType::KnowledgeUpdate
        );
        assert_eq!(map_question_type("abstention"), QuestionType::Abstention);
        assert_eq!(map_question_type("unknown-tag"), QuestionType::Other);
    }

    #[test]
    fn parses_jsonl_alternate() {
        let text = format!(
            "{}\n{}\n",
            sample_row().replace([' ', '\n'], ""),
            sample_row()
                .replace([' ', '\n'], "")
                .replacen("lme-001", "lme-002", 1),
        );
        let rows = parse_jsonl(&text, Path::new("test")).expect("parse");
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn missing_dataset_returns_not_found() {
        let tmp = std::env::temp_dir().join("definitely-not-there-lme");
        let err = LongMemEvalS.load(&tmp).unwrap_err();
        assert!(matches!(err, EvalError::DatasetNotFound { .. }));
    }
}
