//! Paraphrase-stress — a compiled-in benchmark that scales the
//! no-overlap idea to ~40 procedurally-generated instances.
//!
//! Like [`crate::datasets::lexical_stress`], every instance phrases the
//! gold fact with vocabulary DISJOINT from the question: the memory says
//! "{person} relocated to the Bavarian capital"; the question asks "Where
//! does {person} live?"; the gold answer is "Munich". A substring matcher
//! finds nothing.
//!
//! ## Why generated, not hand-authored
//!
//! Hand-authored cases (the lexical-stress set) are small and risk being
//! memorized by a model under test. This benchmark composes its corpus
//! from small pools — entity names crossed with template families — so the
//! population is large and procedurally varied. It tests the recall
//! MECHANISM, not a memorized list of examples.
//!
//! Generation is fully DETERMINISTIC: every triple is derived from its
//! index by integer arithmetic over fixed pools. No `Date::now`, no
//! entropy-seeded RNG — two runs produce byte-identical corpora.
//!
//! ## What it proves
//!
//! Run with `--features live-llm`:
//!
//! - **substring recall@k ≈ 0** — by construction; the gold string is
//!   absent from every memory.
//! - **context-recall + accuracy stay high** — a semantic retriever
//!   surfaces the right memory and the judges confirm it.
//!
//! The ~40 distractor memories (unrelated to any question) make the
//! retrieval problem non-trivial: a degenerate "return everything"
//! retriever would drown the answer in noise.

use std::path::Path;

use crate::core::benchmark::{Benchmark, EvalError};
use crate::core::instance::{EvalInstance, QuestionType, Session, TurnRecord};

/// Compiled-in paraphrase-stress benchmark.
pub struct ParaphraseStressBenchmark;

/// Shared conversation key — all questions run against the one ingested
/// corpus, so the runner ingests it once.
const CONVERSATION_ID: &str = "paraphrase-stress-corpus";

/// One template family: a way to encode a fact by indirection and the
/// matching no-overlap question. The `memory` / `question` strings carry
/// two placeholders, `{name}` and `{para}` (the paraphrase of the gold
/// answer that appears in the memory). The `gold` is the direct answer
/// — it must never appear in either the memory or the question.
struct Family {
    /// Stable id fragment used in question ids (e.g. `"loc"`).
    tag: &'static str,
    /// Memory template: `{name}` + `{para}` (indirect phrasing of gold).
    memory: &'static str,
    /// Question template: `{name}` only; shares no content token with gold.
    question: &'static str,
    /// `(paraphrase-used-in-memory, gold-answer)` pairs for this family.
    values: &'static [(&'static str, &'static str)],
    /// Closest evaluation dimension.
    qtype: QuestionType,
}

/// Six template families. Each crosses with a distinct slice of the name
/// pool, so no entity reuses a family — every triple is unique and its
/// question shares no content token with its memory or gold.
const FAMILIES: &[Family] = &[
    // location — "relocated to {paraphrase of city}" / "where does X live?"
    Family {
        tag: "loc",
        memory: "{name} relocated to {para} a couple of years ago.",
        question: "Where does {name} live?",
        values: &[
            ("the Bavarian capital", "Munich"),
            ("the city of canals and gondolas", "Venice"),
            ("the windy metropolis on Lake Michigan", "Chicago"),
            ("the eternal city on seven hills", "Rome"),
            ("the harbour town beneath the opera sails", "Sydney"),
            ("the misty port at the mouth of the Liffey", "Dublin"),
            ("the imperial seat along the Danube", "Vienna"),
        ],
        qtype: QuestionType::SingleHop,
    },
    // occupation-by-indirection — describe the work, ask the job title.
    Family {
        tag: "job",
        memory: "{name} {para} for a living.",
        question: "What is {name}'s job?",
        values: &[
            ("pilots the night freight run between the two coasts", "Truck driver"),
            ("sets broken bones and reads the films afterward", "Doctor"),
            ("argues cases before the bench in a black robe", "Lawyer"),
            ("draws blueprints for the towers downtown", "Architect"),
            ("keeps the books and files the quarterly returns", "Accountant"),
            ("tends the dough through the small hours before dawn", "Baker"),
            ("guides the great hull past the harbour reefs", "Ship captain"),
        ],
        qtype: QuestionType::SingleHop,
    },
    // language-by-locale — fluent in a place, ask the language.
    Family {
        tag: "lang",
        memory: "{name} reads a menu in {para} without any help at all.",
        question: "Which language does {name} speak?",
        values: &[
            ("Lisbon", "Portuguese"),
            ("Osaka", "Japanese"),
            ("Warsaw", "Polish"),
            ("Athens", "Greek"),
            ("Helsinki", "Finnish"),
            ("Seoul", "Korean"),
            ("Cairo", "Arabic"),
        ],
        qtype: QuestionType::SingleHop,
    },
    // preference-by-avoidance — what is avoided, ask what to skip.
    Family {
        tag: "pref",
        memory: "{name} cannot abide {para}, so the cook always leaves it off the plate.",
        question: "What food should we avoid serving {name}?",
        values: &[
            ("anything from the nightshade family", "Eggplant"),
            ("the briny shellfish that wash up at low tide", "Shrimp"),
            ("the pungent bulb that makes the eyes water", "Onion"),
            ("the fiery red pods from the spice market", "Chili pepper"),
            ("the soft blue-veined wheel from the cellar", "Cheese"),
            ("the fungus that sprouts after autumn rain", "Mushrooms"),
        ],
        qtype: QuestionType::Preference,
    },
    // date-by-event — anchor a date to an event, ask the date.
    Family {
        tag: "date",
        memory: "{name}'s firstborn arrived on {para}.",
        question: "On what date was {name}'s child born?",
        values: &[
            ("the morning of the autumn equinox", "September 22nd"),
            ("the longest day of the year", "June 21st"),
            ("the night the year turns over", "December 31st"),
            ("the first stroke of the new year", "January 1st"),
            ("the day the nation lights its fireworks in midsummer", "July 4th"),
            ("the eve of the spring thaw", "March 20th"),
        ],
        qtype: QuestionType::Temporal,
    },
    // possession — describe the object, ask what it is.
    Family {
        tag: "pos",
        memory: "{name} keeps {para} locked away in the study.",
        question: "What treasured object does {name} own?",
        values: &[
            ("the wind-up timepiece that once belonged to a grandfather", "A pocket watch"),
            ("the six-string box of polished spruce and rosewood", "A guitar"),
            ("the leather-bound first edition signed by the author", "A book"),
            ("the brass scope that brings the night sky close", "A telescope"),
            ("the strung instrument played beneath the chin", "A violin"),
            ("the painted board where ivory and ebony keys lie", "A piano"),
        ],
        qtype: QuestionType::Preference,
    },
];

/// Entity-name pool. Disjoint enough that crossing names with families
/// never reuses a name across families (we slice it per-family below).
const NAMES: &[&str] = &[
    "Maria", "Theo", "Priya", "Omar", "Lena", "Sasha", "Diego", "Yuki", "Ingrid", "Mateo",
    "Aisha", "Bjorn", "Carmen", "Dmitri", "Esme", "Farah", "Gustav", "Hana", "Ravi", "Nadia",
    "Pablo", "Qadir", "Rosa", "Soren", "Tariq", "Uma", "Viktor", "Wren", "Xenia", "Yara",
    "Zane", "Anouk", "Bilal", "Clara", "Dante", "Elif", "Felix", "Greta", "Hugo", "Iris",
    "Jonas", "Kira", "Liam", "Mira", "Noor", "Otis", "Pia", "Quill",
];

/// One fully-materialized triple plus its distractor flag.
struct Triple {
    question_id: String,
    memory: String,
    question: String,
    gold: String,
    qtype: QuestionType,
}

/// Deterministically materialize every (memory, question, gold) triple.
///
/// Names are assigned by a running cursor over [`NAMES`] so each family
/// gets its own disjoint slice — no name is shared between families, which
/// keeps every triple independent. Generation is pure index arithmetic.
fn generate_triples() -> Vec<Triple> {
    let mut out = Vec::new();
    let mut name_cursor = 0usize;
    for family in FAMILIES {
        for (vi, (para, gold)) in family.values.iter().enumerate() {
            let name = NAMES[name_cursor % NAMES.len()];
            name_cursor += 1;
            let memory = family
                .memory
                .replace("{name}", name)
                .replace("{para}", para);
            let question = family.question.replace("{name}", name);
            out.push(Triple {
                question_id: format!("para-{}-{:02}", family.tag, vi + 1),
                memory,
                question,
                gold: (*gold).to_owned(),
                qtype: family.qtype,
            });
        }
    }
    out
}

/// Distractor memories — plausible, unrelated facts that share no entity
/// with any question. They pad the corpus so retrieval must discriminate.
/// Generated deterministically from a small pool crossed with the tail of
/// the name pool (offset so it never collides with answer-bearing names).
const DISTRACTOR_TEMPLATES: &[&str] = &[
    "{name} spent the weekend repainting the garden fence a pale green.",
    "{name} finally finished assembling the model railway in the attic.",
    "{name} signed up for a pottery class that meets on alternate Tuesdays.",
    "{name} adopted a tabby cat from the shelter down the road.",
    "{name} swapped the old kettle for one that whistles two notes.",
    "{name} planted a row of sunflowers along the back wall this year.",
    "{name} switched the morning coffee for a pot of loose-leaf tea.",
    "{name} reorganized the bookshelf strictly by the colour of the spines.",
];

/// Build the distractor memory strings. Uses the same name pool offset by
/// a fixed stride so distractor entities never coincide with question
/// entities (which start at index 0 and run for ~40 names).
fn generate_distractors() -> Vec<String> {
    // Question entities consume the first `answer_count` names; offset the
    // distractor names past them with wraparound, picking a deterministic
    // stride that keeps them clear of the answer-bearing prefix.
    let answer_count: usize = FAMILIES.iter().map(|f| f.values.len()).sum();
    let mut out = Vec::new();
    for i in 0..40 {
        let template = DISTRACTOR_TEMPLATES[i % DISTRACTOR_TEMPLATES.len()];
        let name = NAMES[(answer_count + i) % NAMES.len()];
        out.push(template.replace("{name}", &format!("{name}-D{i:02}")));
    }
    out
}

impl ParaphraseStressBenchmark {
    /// Build the shared session — one user turn per answer-bearing memory,
    /// then one per distractor.
    fn corpus_session(triples: &[Triple], distractors: &[String]) -> Session {
        let mut turns: Vec<TurnRecord> = triples
            .iter()
            .map(|t| TurnRecord {
                role: "user".to_owned(),
                content: t.memory.clone(),
            })
            .collect();
        turns.extend(distractors.iter().map(|d| TurnRecord {
            role: "user".to_owned(),
            content: d.clone(),
        }));
        Session {
            session_id: "paraphrase-stress-session-0".to_owned(),
            turns,
        }
    }
}

impl Benchmark for ParaphraseStressBenchmark {
    fn id(&self) -> &'static str {
        "paraphrase-stress"
    }

    fn display_name(&self) -> &'static str {
        "Paraphrase stress (generated no-overlap retrieval)"
    }

    fn url(&self) -> &'static str {
        "https://github.com/brain-db-io/brain-eval#paraphrase-stress"
    }

    fn requires_datasets_dir(&self) -> bool {
        false
    }

    /// Answers are entailed by paraphrase, never quoted, so synthesis is
    /// required — the heuristic top-K concatenation can't produce them.
    fn requires_synthesis(&self) -> bool {
        true
    }

    fn load(&self, _datasets_dir: &Path) -> Result<Vec<EvalInstance>, EvalError> {
        let triples = generate_triples();
        let distractors = generate_distractors();
        let session = Self::corpus_session(&triples, &distractors);
        let instances = triples
            .iter()
            .map(|t| EvalInstance {
                question_id: t.question_id.clone(),
                question: t.question.clone(),
                answer: t.gold.clone(),
                question_type: t.qtype,
                conversation_id: Some(CONVERSATION_ID.to_owned()),
                sessions: vec![session.clone()],
            })
            .collect();
        Ok(instances)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Generic words that don't count as "content" overlap.
    const STOPWORDS: &[&str] = &[
        "a", "an", "the", "to", "of", "in", "on", "at", "is", "are", "was", "were", "be", "do",
        "does", "did", "what", "where", "when", "who", "which", "how", "why", "we", "i", "my",
        "his", "her", "now", "should", "and", "or", "for", "with", "after", "all", "off", "it",
        "any", "help", "without", "live", "speak", "job", "own", "food", "avoid", "serving",
        "treasured", "object", "child", "born", "date",
    ];

    fn content_tokens(s: &str) -> HashSet<String> {
        s.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty() && !STOPWORDS.contains(t))
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn loads_all_cases() {
        let insts = ParaphraseStressBenchmark
            .load(Path::new("."))
            .expect("load");
        let expected: usize = FAMILIES.iter().map(|f| f.values.len()).sum();
        assert_eq!(insts.len(), expected);
        assert!(insts.len() >= 38, "expected ~40 generated triples, got {}", insts.len());
        assert!(insts
            .iter()
            .all(|i| i.conversation_id.as_deref() == Some(CONVERSATION_ID)));
        assert!(insts.iter().all(|i| i.sessions.len() == 1));
    }

    /// Generation must be deterministic: two loads produce identical
    /// instances in identical order.
    #[test]
    fn generation_is_deterministic() {
        let a = ParaphraseStressBenchmark.load(Path::new(".")).expect("load");
        let b = ParaphraseStressBenchmark.load(Path::new(".")).expect("load");
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
        let insts = ParaphraseStressBenchmark.load(Path::new(".")).expect("load");
        let ids: HashSet<_> = insts.iter().map(|i| i.question_id.clone()).collect();
        assert_eq!(ids.len(), insts.len(), "duplicate question_id");
    }

    /// The core invariant: the question must not lexically leak the gold
    /// answer. A shared SUBJECT anchor (the entity name) is allowed.
    #[test]
    fn question_does_not_leak_the_answer() {
        let triples = generate_triples();
        for t in &triples {
            let q = content_tokens(&t.question);
            let g = content_tokens(&t.gold);
            let leak: Vec<_> = q.intersection(&g).collect();
            assert!(
                leak.is_empty(),
                "{}: question shares content tokens with gold: {leak:?}",
                t.question_id,
            );
        }
    }

    /// The gold answer must NOT appear verbatim in its storing memory, so
    /// substring recall@k scores ~0 by construction.
    #[test]
    fn gold_is_absent_from_memory() {
        let triples = generate_triples();
        for t in &triples {
            assert!(
                !t.memory.to_lowercase().contains(&t.gold.to_lowercase()),
                "{}: gold {:?} leaked verbatim into memory",
                t.question_id,
                t.gold,
            );
        }
    }

    /// Distractor memories must not name any answer-bearing entity as a
    /// standalone whitespace-delimited token, so they can't accidentally
    /// satisfy a question. Distractor entities carry a `-Dnn` suffix
    /// (e.g. `Maria-D09`), so the entity token is never byte-equal to a
    /// bare answer name like `Maria`.
    #[test]
    fn distractors_do_not_collide_with_questions() {
        let triples = generate_triples();
        let distractors = generate_distractors();
        assert!(distractors.len() >= 38, "expected ~40 distractors");
        let answer_names: HashSet<&str> = triples
            .iter()
            .flat_map(|t| t.question.split_whitespace())
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .collect();
        for d in &distractors {
            for word in d.split_whitespace() {
                let token = word.trim_matches(|c: char| !c.is_alphanumeric());
                assert!(
                    !answer_names.contains(token),
                    "distractor reused answer entity name {token:?} verbatim: {d:?}",
                );
            }
        }
    }

    #[test]
    fn requires_synthesis_is_true() {
        assert!(ParaphraseStressBenchmark.requires_synthesis());
    }
}
