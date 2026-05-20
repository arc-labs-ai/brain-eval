//! LoCoMo (ACL 2024) — long conversational memory benchmark with
//! multi-hop, temporal, and adversarial categories.
//!
//! Paper: <https://arxiv.org/abs/2402.17753>.
//! Release: <https://github.com/snap-research/LoCoMo>.
//!
//! ## Shape
//!
//! Each LoCoMo *sample* is a single long conversation between two
//! speakers, broken into multiple sessions, paired with many QA
//! pairs that share that conversation as their haystack. The
//! canonical `locomo10.json` ships 10 samples × ~150 QA each =
//! ~1540 questions.
//!
//! We expand one sample into many [`EvalInstance`]s sharing the
//! same `conversation_id` so [`crate::runner::EvalRunner`] ingests
//! the conversation once and queries it per-question.
//!
//! ## Categories
//!
//! LoCoMo tags each QA with a numeric category 1..=5:
//!
//! | Category | Description           | Maps to                     |
//! |----------|-----------------------|-----------------------------|
//! | 1        | Single-hop            | `QuestionType::SingleHop`   |
//! | 2        | Multi-hop             | `QuestionType::MultiHop`    |
//! | 3        | Temporal              | `QuestionType::Temporal`    |
//! | 4        | Open-domain           | `QuestionType::Other`       |
//! | 5        | Adversarial / unanswerable | `QuestionType::Adversarial` |
//!
//! ## Honest-scoring note
//!
//! Category 5 is the abstention class — the correct answer is to
//! refuse. The Zep team [excluded category 5 from the
//! denominator](https://github.com/getzep/zep-papers/issues/5) while
//! still counting correct abstentions in the numerator, inflating
//! their reported LoCoMo score by ~25 points (0.833 reported vs.
//! 0.5844 corrected).
//!
//! `compute_full_metrics` includes every category in the denominator,
//! matching the standard protocol. The discrepancy is documented in
//! [`crate::report::locomo_competitor_baselines`] so any report we
//! publish ships the call-out alongside the numbers.
//!
//! ## File layout
//!
//! `$BRAIN_EVAL_DATASETS_DIR/locomo/locomo10.json` — JSON array of
//! samples, each matching the wire shape below.
//!
//! ## Wire shape
//!
//! ```jsonc
//! [
//!   {
//!     "sample_id": "sample-0",
//!     "conversation": {
//!       "session_1": [
//!         {"speaker": "Alice", "text": "Hi Bob.", "date_time": "..."},
//!         {"speaker": "Bob",   "text": "Hey."}
//!       ],
//!       "session_2": [ ... ]
//!     },
//!     "qa": [
//!       {
//!         "question": "Who did Alice greet first?",
//!         "answer":   "Bob",
//!         "category": 1
//!       }
//!     ]
//!   }
//! ]
//! ```

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::core::benchmark::{Benchmark, EvalError};
use crate::core::instance::{EvalInstance, QuestionType, Session, TurnRecord};

/// LoCoMo — drives the hybrid path under multi-hop / adversarial load.
pub struct LocomoBenchmark;

impl Benchmark for LocomoBenchmark {
    fn id(&self) -> &'static str {
        "locomo"
    }

    fn display_name(&self) -> &'static str {
        "LoCoMo (ACL 2024)"
    }

    fn url(&self) -> &'static str {
        "https://arxiv.org/abs/2402.17753"
    }

    fn load(&self, datasets_dir: &Path) -> Result<Vec<EvalInstance>, EvalError> {
        let path = datasets_dir.join("locomo").join("locomo10.json");
        let bytes = std::fs::read(&path).map_err(|_| EvalError::DatasetNotFound {
            path: path.display().to_string(),
        })?;
        let samples: Vec<LocomoSample> =
            serde_json::from_slice(&bytes).map_err(|e| EvalError::ParseError {
                path: path.display().to_string(),
                reason: e.to_string(),
            })?;

        let mut out = Vec::new();
        for sample in samples {
            sample.expand_into(&mut out);
        }
        Ok(out)
    }

    /// LoCoMo expects free-form answers; reliable scoring needs an LLM
    /// judge. The heuristic judge is a directional signal only.
    fn requires_synthesis(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Wire shapes
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct LocomoSample {
    #[serde(default = "default_sample_id")]
    sample_id: String,
    /// Map keyed by session label (e.g. `"session_1"`, `"session_2"`)
    /// → turns. Using `BTreeMap` keeps ingest order deterministic
    /// (lexicographic by session label, which matches the natural
    /// `session_1` < `session_2` < ... ordering).
    #[serde(default)]
    conversation: BTreeMap<String, Vec<LocomoTurn>>,
    #[serde(default)]
    qa: Vec<LocomoQa>,
}

#[derive(Debug, Deserialize)]
struct LocomoTurn {
    speaker: String,
    text: String,
    // `date_time` exists in the release but the loader doesn't use
    // it yet — temporal handling lives in the substrate.
    #[serde(default)]
    #[allow(dead_code)]
    date_time: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LocomoQa {
    question: String,
    answer: String,
    #[serde(default)]
    category: Option<u8>,
}

fn default_sample_id() -> String {
    "sample-0".to_owned()
}

impl LocomoSample {
    fn expand_into(self, out: &mut Vec<EvalInstance>) {
        let conv_id = self.sample_id.clone();

        // Build sessions once per sample; every QA in this sample
        // re-uses them via shared `conversation_id`.
        let sessions: Vec<Session> = self
            .conversation
            .into_iter()
            .map(|(session_label, turns)| Session {
                session_id: session_label,
                turns: turns
                    .into_iter()
                    .map(|t| TurnRecord {
                        // Map every speaker to a `user` turn so the
                        // ingest helper accepts both sides — both
                        // contribute facts. Prefix the speaker name
                        // so the substrate text preserves attribution.
                        role: "user".to_owned(),
                        content: format!("{}: {}", t.speaker, t.text),
                    })
                    .collect(),
            })
            .collect();

        for (idx, qa) in self.qa.into_iter().enumerate() {
            let question_id = format!("{}-{idx:04}", self.sample_id);
            let question_type = map_category(qa.category);
            out.push(EvalInstance {
                question_id,
                question: qa.question,
                answer: qa.answer,
                question_type,
                conversation_id: Some(conv_id.clone()),
                sessions: sessions.clone(),
            });
        }
    }
}

fn map_category(cat: Option<u8>) -> QuestionType {
    match cat {
        Some(1) => QuestionType::SingleHop,
        Some(2) => QuestionType::MultiHop,
        Some(3) => QuestionType::Temporal,
        Some(4) => QuestionType::Other,
        Some(5) => QuestionType::Adversarial,
        _ => QuestionType::Other,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_text() -> &'static str {
        r#"[
            {
                "sample_id": "sample-0",
                "conversation": {
                    "session_1": [
                        {"speaker": "Alice", "text": "Hi Bob.", "date_time": "2024-01-01"},
                        {"speaker": "Bob", "text": "Hey."}
                    ],
                    "session_2": [
                        {"speaker": "Alice", "text": "Did you finish the report?"}
                    ]
                },
                "qa": [
                    {"question": "Who greeted whom?", "answer": "Alice greeted Bob.", "category": 1},
                    {"question": "What is on Mars?",  "answer": "I don't know.",      "category": 5}
                ]
            }
        ]"#
    }

    #[test]
    fn expands_sample_into_multiple_instances() {
        let samples: Vec<LocomoSample> = serde_json::from_str(sample_text()).expect("parse");
        let mut out = Vec::new();
        for s in samples {
            s.expand_into(&mut out);
        }
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].question_type, QuestionType::SingleHop);
        assert_eq!(out[1].question_type, QuestionType::Adversarial);
        // Both share the same conversation_id.
        assert_eq!(out[0].conversation_id, out[1].conversation_id);
        assert!(out[0].conversation_id.is_some());
    }

    #[test]
    fn sessions_are_lexicographic() {
        let samples: Vec<LocomoSample> = serde_json::from_str(sample_text()).expect("parse");
        let mut out = Vec::new();
        for s in samples {
            s.expand_into(&mut out);
        }
        let labels: Vec<&str> = out[0]
            .sessions
            .iter()
            .map(|s| s.session_id.as_str())
            .collect();
        assert_eq!(labels, vec!["session_1", "session_2"]);
    }

    #[test]
    fn speaker_prefix_preserved_in_turn_content() {
        let samples: Vec<LocomoSample> = serde_json::from_str(sample_text()).expect("parse");
        let mut out = Vec::new();
        for s in samples {
            s.expand_into(&mut out);
        }
        let first_turn = &out[0].sessions[0].turns[0];
        assert_eq!(first_turn.role, "user");
        assert!(first_turn.content.starts_with("Alice:"));
        assert!(first_turn.content.contains("Hi Bob."));
    }

    #[test]
    fn unknown_category_maps_to_other() {
        assert_eq!(map_category(Some(99)), QuestionType::Other);
        assert_eq!(map_category(None), QuestionType::Other);
    }

    #[test]
    fn missing_dataset_returns_not_found() {
        let tmp = std::env::temp_dir().join("definitely-not-there-locomo");
        let err = LocomoBenchmark.load(&tmp).unwrap_err();
        assert!(matches!(err, EvalError::DatasetNotFound { .. }));
    }
}
