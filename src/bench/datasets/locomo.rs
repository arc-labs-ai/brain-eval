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
//! [`crate::bench::report::locomo_competitor_baselines`] so any report we
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

use crate::bench::core::benchmark::{Benchmark, EvalError};
use crate::bench::core::instance::{EvalInstance, QuestionType, Session, TurnRecord};

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
        let bytes = crate::bench::datasets::read_dataset_file(&path)?;
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
    /// The `conversation` object mixes `session_<N>` turn arrays with
    /// scalar metadata under the same map (`speaker_a`, `speaker_b`,
    /// `session_<N>_date_time`), so values are heterogeneous — keep them as
    /// raw JSON and pick out the session arrays in `expand_into`.
    #[serde(default)]
    conversation: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    qa: Vec<LocomoQa>,
}

#[derive(Debug, Deserialize)]
struct LocomoTurn {
    speaker: String,
    text: String,
    // Per-turn timestamp when present; otherwise the session-level
    // `session_<N>_date_time` is used. Stamped into the ingested text so
    // temporal questions have an absolute date to retrieve.
    #[serde(default)]
    date_time: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LocomoQa {
    question: String,
    // LoCoMo answers are usually strings but are sometimes bare numbers
    // (a year, a count) or booleans; accept any scalar and stringify it.
    // Adversarial (category 5) questions omit `answer` entirely and carry
    // `adversarial_answer` instead — `default` covers the missing case.
    #[serde(default, deserialize_with = "scalar_to_string")]
    answer: String,
    #[serde(default, deserialize_with = "scalar_to_string")]
    adversarial_answer: String,
    #[serde(default)]
    category: Option<u8>,
}

/// Deserialize a JSON scalar (string / number / bool) into a `String`.
/// LoCoMo's `answer` field isn't consistently typed across questions.
fn scalar_to_string<'de, D>(de: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    Ok(match serde_json::Value::deserialize(de)? {
        serde_json::Value::String(s) => s,
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    })
}

fn default_sample_id() -> String {
    "sample-0".to_owned()
}

/// `true` for a `session_<N>` turn-array key (all-digit suffix), not the
/// `session_<N>_date_time` scalar or the `speaker_*` metadata.
fn is_session_label(label: &str) -> bool {
    match label.strip_prefix("session_") {
        Some(rest) => !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()),
        None => false,
    }
}

impl LocomoSample {
    fn expand_into(self, out: &mut Vec<EvalInstance>) {
        let conv_id = self.sample_id.clone();

        // The conversation map interleaves `session_<N>` turn arrays with
        // scalar metadata: `speaker_a/b` and, crucially, the per-session
        // timestamp at `session_<N>_date_time`. Split the two so we can
        // stamp each turn with its session date — LoCoMo's temporal
        // questions ("When did X happen?") need an absolute date in the
        // ingested text, or the conversation only says "yesterday".
        let mut session_arrays: Vec<(String, serde_json::Value)> = Vec::new();
        let mut session_dates: BTreeMap<String, String> = BTreeMap::new();
        for (label, value) in self.conversation {
            if let Some(session) = label.strip_suffix("_date_time") {
                if let serde_json::Value::String(date) = value {
                    session_dates.insert(session.to_owned(), date);
                }
            } else if is_session_label(&label) {
                session_arrays.push((label, value));
            }
            // else: speaker_a / speaker_b and any other scalar metadata.
        }

        // Build sessions once per sample; every QA in this sample
        // re-uses them via shared `conversation_id`.
        let sessions: Vec<Session> = session_arrays
            .into_iter()
            .filter_map(|(label, value)| {
                let turns: Vec<LocomoTurn> = serde_json::from_value(value).ok()?;
                let session_date = session_dates.get(&label).cloned();
                Some(Session {
                    turns: turns
                        .into_iter()
                        .map(|t| {
                            // Prefer a per-turn timestamp; fall back to the
                            // session's. Map every speaker to a `user` turn
                            // (both sides contribute facts) and prefix the
                            // speaker name to preserve attribution.
                            let when = t.date_time.as_deref().or(session_date.as_deref());
                            let content = match when {
                                Some(date) => format!("[{date}] {}: {}", t.speaker, t.text),
                                None => format!("{}: {}", t.speaker, t.text),
                            };
                            // Also pass the date as a structured event time so
                            // the server anchors relative dates in the utterance
                            // ("yesterday") to the message date, not ingest time.
                            let occurred_at_unix_nanos =
                                when.and_then(parse_locomo_datetime);
                            TurnRecord {
                                role: "user".to_owned(),
                                content,
                                occurred_at_unix_nanos,
                            }
                        })
                        .collect(),
                    session_id: label,
                })
            })
            .collect();

        for (idx, qa) in self.qa.into_iter().enumerate() {
            let question_id = format!("{}-{idx:04}", self.sample_id);
            let question_type = map_category(qa.category);
            // Adversarial questions carry `adversarial_answer`, not `answer`.
            let answer = if qa.answer.is_empty() {
                qa.adversarial_answer
            } else {
                qa.answer
            };
            out.push(EvalInstance {
                question_id,
                question: qa.question,
                answer,
                question_type,
                conversation_id: Some(conv_id.clone()),
                sessions: sessions.clone(),
            });
        }
    }
}

/// Parse a LoCoMo timestamp string into unix **nanoseconds** at midnight
/// UTC of the date it names.
///
/// LoCoMo dates look like `"1:56 pm on 8 May, 2023"` — a clock time, the
/// literal `" on "`, then a `"D Month, YYYY"` (or `"D Month YYYY"`) date.
/// Some rows drop the clock time and carry only the date. We anchor to the
/// **date** (midnight UTC); the clock time, when present, is ignored — the
/// date is what the server's temporal extractor needs to resolve relative
/// references in the utterance.
///
/// Returns `None` for anything that doesn't parse. Never panics.
///
/// # Examples
///
/// ```
/// # use brain_eval::bench::datasets::locomo::parse_locomo_datetime;
/// let nanos = parse_locomo_datetime("1:56 pm on 8 May, 2023").unwrap();
/// assert_eq!(nanos, 1_683_504_000_000_000_000);
/// ```
#[must_use]
pub fn parse_locomo_datetime(s: &str) -> Option<u64> {
    // Take the date part: everything after the last " on " if present,
    // otherwise the whole (trimmed) string.
    let date_part = match s.rfind(" on ") {
        Some(idx) => &s[idx + 4..],
        None => s,
    }
    .trim();

    // Expect "<day> <month>[,] <year>": split on whitespace, drop a trailing
    // comma on the month token.
    let mut it = date_part.split_whitespace();
    let day: u32 = it.next()?.parse().ok()?;
    let month_tok = it.next()?.trim_end_matches(',');
    let month = month_from_name(month_tok)?;
    let year: i64 = it.next()?.parse().ok()?;
    // Reject trailing garbage ("8 May 2023 extra").
    if it.next().is_some() {
        return None;
    }

    if day < 1 || day > 31 || !(1..=12).contains(&month) {
        return None;
    }

    let days = days_from_civil(year, month, day);
    let secs = days.checked_mul(86_400)?;
    let nanos = secs.checked_mul(1_000_000_000)?;
    u64::try_from(nanos).ok()
}

/// Month number (1..=12) from a full or 3-letter English month name,
/// case-insensitive. `None` for anything else.
fn month_from_name(name: &str) -> Option<u32> {
    let n = name.to_ascii_lowercase();
    let m = match n.as_str() {
        "january" | "jan" => 1,
        "february" | "feb" => 2,
        "march" | "mar" => 3,
        "april" | "apr" => 4,
        "may" => 5,
        "june" | "jun" => 6,
        "july" | "jul" => 7,
        "august" | "aug" => 8,
        "september" | "sep" | "sept" => 9,
        "october" | "oct" => 10,
        "november" | "nov" => 11,
        "december" | "dec" => 12,
        _ => return None,
    };
    Some(m)
}

/// Days since the unix epoch (1970-01-01) for a proleptic-Gregorian date,
/// via Howard Hinnant's `days_from_civil` algorithm. Valid for any date;
/// returns a negative count for dates before the epoch.
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as i64; // [0, 399]
    let m = i64::from(month);
    let d = i64::from(day);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
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
                    "speaker_a": "Alice",
                    "speaker_b": "Bob",
                    "session_1": [
                        {"speaker": "Alice", "text": "Hi Bob.", "date_time": "2024-01-01"},
                        {"speaker": "Bob", "text": "Hey."}
                    ],
                    "session_2_date_time": "2024-02-02",
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
    fn turn_content_is_date_stamped_and_attributed() {
        let samples: Vec<LocomoSample> = serde_json::from_str(sample_text()).expect("parse");
        let mut out = Vec::new();
        for s in samples {
            s.expand_into(&mut out);
        }
        // session_1 turn carries a per-turn date.
        let first_turn = &out[0].sessions[0].turns[0];
        assert_eq!(first_turn.role, "user");
        assert_eq!(first_turn.content, "[2024-01-01] Alice: Hi Bob.");
        // session_2 turn has no per-turn date → falls back to the
        // session-level `session_2_date_time`.
        let s2_turn = &out[0].sessions[1].turns[0];
        assert!(s2_turn.content.starts_with("[2024-02-02] Alice:"));
        assert!(s2_turn.content.contains("Did you finish the report?"));
    }

    #[test]
    fn speaker_metadata_is_not_a_session() {
        // `speaker_a` / `speaker_b` scalars must not become sessions.
        let samples: Vec<LocomoSample> = serde_json::from_str(sample_text()).expect("parse");
        let mut out = Vec::new();
        for s in samples {
            s.expand_into(&mut out);
        }
        assert_eq!(out[0].sessions.len(), 2);
    }

    #[test]
    fn parses_datetime_with_clock_time() {
        // 2023-05-08 00:00:00 UTC = 1_683_504_000 s.
        let nanos = parse_locomo_datetime("1:56 pm on 8 May, 2023").expect("parse");
        assert_eq!(nanos, 1_683_504_000_000_000_000);
    }

    #[test]
    fn parses_bare_date_without_time() {
        // No clock time / no " on " prefix — the date alone must parse to the
        // same midnight-UTC instant as the timestamped form.
        let nanos = parse_locomo_datetime("8 May, 2023").expect("parse");
        assert_eq!(nanos, 1_683_504_000_000_000_000);
    }

    #[test]
    fn parses_date_without_comma() {
        let nanos = parse_locomo_datetime("8 May 2023").expect("parse");
        assert_eq!(nanos, 1_683_504_000_000_000_000);
    }

    #[test]
    fn malformed_datetime_returns_none() {
        assert_eq!(parse_locomo_datetime("not a date"), None);
        assert_eq!(parse_locomo_datetime(""), None);
        assert_eq!(parse_locomo_datetime("8 Smarch, 2023"), None);
        assert_eq!(parse_locomo_datetime("32 May, 2023"), None);
        assert_eq!(parse_locomo_datetime("8 May, 2023 extra"), None);
    }

    #[test]
    fn locomo_turn_carries_structured_event_time() {
        let samples: Vec<LocomoSample> = serde_json::from_str(sample_text()).expect("parse");
        let mut out = Vec::new();
        for s in samples {
            s.expand_into(&mut out);
        }
        // session_1 turn has a per-turn `date_time` of "2024-01-01"; that ISO
        // form isn't the "D Month, YYYY" LoCoMo shape, so it stays `None` —
        // the field only populates for parseable LoCoMo dates.
        let first_turn = &out[0].sessions[0].turns[0];
        assert_eq!(first_turn.occurred_at_unix_nanos, None);
    }

    #[test]
    fn locomo_turn_parses_prose_date() {
        // A turn whose date is in LoCoMo's prose form populates the field.
        let text = r#"[
            {
                "sample_id": "sample-1",
                "conversation": {
                    "session_1_date_time": "1:56 pm on 8 May, 2023",
                    "session_1": [
                        {"speaker": "Alice", "text": "Hi."}
                    ]
                },
                "qa": [ {"question": "hi?", "answer": "hi", "category": 1} ]
            }
        ]"#;
        let samples: Vec<LocomoSample> = serde_json::from_str(text).expect("parse");
        let mut out = Vec::new();
        for s in samples {
            s.expand_into(&mut out);
        }
        let turn = &out[0].sessions[0].turns[0];
        assert_eq!(turn.occurred_at_unix_nanos, Some(1_683_504_000_000_000_000));
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
