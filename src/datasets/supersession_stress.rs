//! Supersession-stress — a compiled-in benchmark that tests knowledge-
//! update DIRECTION: can recall distinguish the current value from the
//! superseded one?
//!
//! For each of ~20 entities the corpus states an OLD value in one turn and
//! a NEW value in a LATER turn, e.g.
//!
//! ```text
//! Earlier, Maria's role was junior analyst.
//! Maria was later promoted to engineering manager.
//! ```
//!
//! Two questions are emitted per entity:
//!
//! - **current** — "What is Maria's role now?" → gold `engineering manager`
//! - **prior**   — "What was Maria's role before?" → gold `junior analyst`
//!
//! ~40 questions total. A retriever that just surfaces "the memory about
//! Maria's role" without honouring temporal ordering will get exactly half
//! wrong — the benchmark is a direct probe of supersession handling.
//!
//! Generation is fully DETERMINISTIC: every entity/value pair comes from a
//! fixed pool indexed by position. No `Date::now`, no entropy RNG.

use std::path::Path;

use crate::core::benchmark::{Benchmark, EvalError};
use crate::core::instance::{EvalInstance, QuestionType, Session, TurnRecord};

/// Compiled-in supersession-stress benchmark.
pub struct SupersessionStressBenchmark;

/// Shared conversation key — all questions run against the one ingested
/// corpus, so the runner ingests it once.
const CONVERSATION_ID: &str = "supersession-stress-corpus";

/// One updatable attribute and the phrasing for its update / questions.
struct Attribute {
    /// Stable id fragment (e.g. `"role"`).
    tag: &'static str,
    /// "Earlier, {name}'s {noun} was {old}." — the OLD-value turn.
    old_template: &'static str,
    /// "{name} {change} {new}." — the NEW-value turn, stated later.
    new_template: &'static str,
    /// Question asking for the CURRENT value.
    current_question: &'static str,
    /// Question asking for the PRIOR (superseded) value.
    prior_question: &'static str,
    /// `(old-value, new-value)` pairs.
    values: &'static [(&'static str, &'static str)],
}

/// Attribute families. Crossed with the name pool, these yield ~20
/// entities, each producing one current + one prior question.
const ATTRIBUTES: &[Attribute] = &[
    Attribute {
        tag: "role",
        old_template: "Earlier, {name}'s role at the company was {old}.",
        new_template: "{name} was later promoted to {new}.",
        current_question: "What is {name}'s role now?",
        prior_question: "What was {name}'s role before the promotion?",
        values: &[
            ("junior analyst", "engineering manager"),
            ("staff writer", "editor-in-chief"),
            ("line cook", "head chef"),
            ("sales associate", "regional director"),
            ("intern", "team lead"),
            ("teller", "branch manager"),
            ("paralegal", "partner"),
        ],
    },
    Attribute {
        tag: "city",
        old_template: "Earlier, {name} was living in {old}.",
        new_template: "{name} has since moved to {new}.",
        current_question: "Which city does {name} live in now?",
        prior_question: "Which city did {name} live in previously?",
        values: &[
            ("Boston", "Seattle"),
            ("Lyon", "Marseille"),
            ("Nairobi", "Mombasa"),
            ("Toronto", "Vancouver"),
            ("Naples", "Turin"),
            ("Pune", "Bengaluru"),
            ("Cardiff", "Edinburgh"),
        ],
    },
    Attribute {
        tag: "car",
        old_template: "Earlier, {name} drove an old {old}.",
        new_template: "{name} recently traded it in for a {new}.",
        current_question: "What car does {name} drive now?",
        prior_question: "What car did {name} used to drive?",
        values: &[
            ("Volvo", "Tesla"),
            ("Fiat", "Audi"),
            ("Datsun", "Subaru"),
            ("Peugeot", "Renault"),
            ("Saab", "Volkswagen"),
            ("Lada", "Skoda"),
        ],
    },
];

/// Entity-name pool, distinct per slot.
const NAMES: &[&str] = &[
    "Maria", "Theo", "Priya", "Omar", "Lena", "Sasha", "Diego", "Yuki", "Ingrid", "Mateo",
    "Aisha", "Bjorn", "Carmen", "Dmitri", "Esme", "Farah", "Gustav", "Hana", "Ravi", "Nadia",
    "Pablo", "Rosa", "Soren", "Tariq", "Uma", "Viktor", "Wren", "Xenia", "Yara", "Zane",
];

/// One materialized entity update: the two turns plus both questions.
struct UpdatePair {
    /// Stable id fragment, e.g. `"role-01"`.
    pair_id: String,
    old_turn: String,
    new_turn: String,
    current_question: String,
    current_gold: String,
    prior_question: String,
    prior_gold: String,
}

/// Deterministically materialize every update pair. Names are drawn by a
/// running cursor so each attribute family gets a disjoint slice.
fn generate_pairs() -> Vec<UpdatePair> {
    let mut out = Vec::new();
    let mut name_cursor = 0usize;
    for attr in ATTRIBUTES {
        for (vi, (old, new)) in attr.values.iter().enumerate() {
            let name = NAMES[name_cursor % NAMES.len()];
            name_cursor += 1;
            out.push(UpdatePair {
                pair_id: format!("supr-{}-{:02}", attr.tag, vi + 1),
                old_turn: attr
                    .old_template
                    .replace("{name}", name)
                    .replace("{old}", old),
                new_turn: attr
                    .new_template
                    .replace("{name}", name)
                    .replace("{new}", new),
                current_question: attr.current_question.replace("{name}", name),
                current_gold: (*new).to_owned(),
                prior_question: attr.prior_question.replace("{name}", name),
                prior_gold: (*old).to_owned(),
            });
        }
    }
    out
}

impl SupersessionStressBenchmark {
    /// Build the shared session. Turn order matters: every OLD turn is
    /// emitted before its NEW turn so the NEW value is genuinely "later".
    /// We interleave per-entity (old, new) so each update sits adjacent.
    fn corpus_session(pairs: &[UpdatePair]) -> Session {
        let mut turns = Vec::with_capacity(pairs.len() * 2);
        for p in pairs {
            turns.push(TurnRecord {
                role: "user".to_owned(),
                content: p.old_turn.clone(),
            });
            turns.push(TurnRecord {
                role: "user".to_owned(),
                content: p.new_turn.clone(),
            });
        }
        Session {
            session_id: "supersession-stress-session-0".to_owned(),
            turns,
        }
    }
}

impl Benchmark for SupersessionStressBenchmark {
    fn id(&self) -> &'static str {
        "supersession-stress"
    }

    fn display_name(&self) -> &'static str {
        "Supersession stress (knowledge-update direction)"
    }

    fn url(&self) -> &'static str {
        "https://github.com/brain-db-io/brain-eval#supersession-stress"
    }

    fn requires_datasets_dir(&self) -> bool {
        false
    }

    /// The answer must be selected by temporal direction (current vs
    /// prior), not just surfaced — synthesis / judgement is required.
    fn requires_synthesis(&self) -> bool {
        true
    }

    fn load(&self, _datasets_dir: &Path) -> Result<Vec<EvalInstance>, EvalError> {
        let pairs = generate_pairs();
        let session = Self::corpus_session(&pairs);
        let mut instances = Vec::with_capacity(pairs.len() * 2);
        for p in &pairs {
            instances.push(EvalInstance {
                question_id: format!("{}-current", p.pair_id),
                question: p.current_question.clone(),
                answer: p.current_gold.clone(),
                question_type: QuestionType::KnowledgeUpdate,
                conversation_id: Some(CONVERSATION_ID.to_owned()),
                sessions: vec![session.clone()],
            });
            instances.push(EvalInstance {
                question_id: format!("{}-prior", p.pair_id),
                question: p.prior_question.clone(),
                answer: p.prior_gold.clone(),
                question_type: QuestionType::Temporal,
                conversation_id: Some(CONVERSATION_ID.to_owned()),
                sessions: vec![session.clone()],
            });
        }
        Ok(instances)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn loads_all_cases() {
        let insts = SupersessionStressBenchmark
            .load(Path::new("."))
            .expect("load");
        let entities: usize = ATTRIBUTES.iter().map(|a| a.values.len()).sum();
        assert_eq!(insts.len(), entities * 2);
        assert!(
            insts.len() >= 38,
            "expected ~40 questions, got {}",
            insts.len()
        );
        assert!(insts
            .iter()
            .all(|i| i.conversation_id.as_deref() == Some(CONVERSATION_ID)));
        assert!(insts.iter().all(|i| i.sessions.len() == 1));
    }

    /// Each entity must contribute exactly one current + one prior
    /// question with distinct gold answers.
    #[test]
    fn current_and_prior_golds_differ() {
        let pairs = generate_pairs();
        for p in &pairs {
            assert_ne!(
                p.current_gold, p.prior_gold,
                "{}: current and prior gold must differ",
                p.pair_id
            );
        }
    }

    /// Both the OLD and NEW value must appear somewhere in the corpus —
    /// the corpus genuinely contains both, so the test is about picking the
    /// right one, not about missing data.
    #[test]
    fn both_values_present_in_corpus() {
        let pairs = generate_pairs();
        let session = SupersessionStressBenchmark::corpus_session(&pairs);
        let corpus: String = session
            .turns
            .iter()
            .map(|t| t.content.to_lowercase())
            .collect::<Vec<_>>()
            .join(" ");
        for p in &pairs {
            assert!(
                corpus.contains(&p.current_gold.to_lowercase()),
                "{}: new value {:?} missing from corpus",
                p.pair_id,
                p.current_gold
            );
            assert!(
                corpus.contains(&p.prior_gold.to_lowercase()),
                "{}: old value {:?} missing from corpus",
                p.pair_id,
                p.prior_gold
            );
        }
    }

    /// The OLD turn must precede the NEW turn for every entity, so the new
    /// value is genuinely the later one.
    #[test]
    fn old_turn_precedes_new_turn() {
        let pairs = generate_pairs();
        let session = SupersessionStressBenchmark::corpus_session(&pairs);
        for p in &pairs {
            let old_idx = session
                .turns
                .iter()
                .position(|t| t.content == p.old_turn)
                .expect("old turn present");
            let new_idx = session
                .turns
                .iter()
                .position(|t| t.content == p.new_turn)
                .expect("new turn present");
            assert!(
                old_idx < new_idx,
                "{}: old turn must precede new turn",
                p.pair_id
            );
        }
    }

    /// Generation is deterministic.
    #[test]
    fn generation_is_deterministic() {
        let a = SupersessionStressBenchmark
            .load(Path::new("."))
            .expect("load");
        let b = SupersessionStressBenchmark
            .load(Path::new("."))
            .expect("load");
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.question_id, y.question_id);
            assert_eq!(x.question, y.question);
            assert_eq!(x.answer, y.answer);
        }
    }

    /// Question ids must be unique.
    #[test]
    fn question_ids_are_unique() {
        let insts = SupersessionStressBenchmark
            .load(Path::new("."))
            .expect("load");
        let ids: HashSet<_> = insts.iter().map(|i| i.question_id.clone()).collect();
        assert_eq!(ids.len(), insts.len(), "duplicate question_id");
    }

    #[test]
    fn requires_synthesis_is_true() {
        assert!(SupersessionStressBenchmark.requires_synthesis());
    }
}
