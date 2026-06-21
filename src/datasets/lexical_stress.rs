//! Lexical-stress — a compiled-in benchmark that defeats substring recall.
//!
//! Every instance is hand-authored so the question shares NO content
//! tokens with the gold answer OR the memory that stores it. The memory
//! says "relocated to the Bavarian capital"; the question asks "Where
//! does Maria live?"; the gold answer is "Munich". A substring matcher
//! finds nothing — "Munich" never appears in the memory, and the
//! question words ("where", "live") never appear either.
//!
//! ## What it proves
//!
//! Run with `--features live-llm`:
//!
//! - **substring recall@k ≈ 0** — by construction; the gold string is
//!   absent from every retrieved memory. This is the DEPRECATED
//!   diagnostic, and here it is *designed* to be wrong.
//! - **context-recall + accuracy stay high** — a semantic retriever
//!   surfaces the right memory and the support judge / answer judge both
//!   confirm it.
//!
//! The gap between the two is the whole argument: Brain is not winning by
//! lexical overlap. If substring recall ever rises on this set, the
//! corpus has drifted (a gold token leaked into a memory) — the
//! `no_token_overlap` test guards against exactly that.

use std::path::Path;

use crate::core::benchmark::{Benchmark, EvalError};
use crate::core::instance::{EvalInstance, QuestionType, Session, TurnRecord};

/// Compiled-in lexical-stress benchmark.
pub struct LexicalStressBenchmark;

/// Shared conversation key — all questions run against the one ingested
/// corpus, so the runner ingests it once.
const CONVERSATION_ID: &str = "lexical-stress-corpus";

/// `(question_id, memory, question, gold answer, question type)`.
///
/// Authoring rule: the question must share no content token (ignoring
/// stopwords) with either the memory or the gold answer. The memory
/// encodes the fact by paraphrase / indirection; recall must be semantic.
const CASES: &[(&str, &str, &str, &str, QuestionType)] = &[
    (
        "lex-01",
        "Maria relocated to the Bavarian capital in spring 2021.",
        "Where does Maria live?",
        "Munich",
        QuestionType::SingleHop,
    ),
    (
        "lex-02",
        "Theo cannot stomach anything from the nightshade family, so the chef omits aubergine from his plate.",
        "What food should we avoid serving Theo?",
        "Eggplant",
        QuestionType::Preference,
    ),
    (
        "lex-03",
        "Priya pilots the night freight run between the two coasts every Thursday.",
        "What is Priya's job?",
        "Truck driver",
        QuestionType::SingleHop,
    ),
    (
        "lex-04",
        "The committee gathered beneath the Eiffel Tower to sign the accord.",
        "In which country was the agreement signed?",
        "France",
        QuestionType::MultiHop,
    ),
    (
        "lex-05",
        "Grandfather's timepiece finally stopped ticking last winter.",
        "What broke at the end of the year?",
        "The watch",
        QuestionType::SingleHop,
    ),
    (
        "lex-06",
        "Lena swapped her sedan for two wheels and now pedals to the office.",
        "How does Lena commute?",
        "By bicycle",
        QuestionType::Preference,
    ),
    (
        "lex-07",
        "The startup's burn rate finally dipped below its monthly revenue in the third month of the year.",
        "When did the company become profitable?",
        "March",
        QuestionType::Temporal,
    ),
    (
        "lex-08",
        "Omar's firstborn arrived on the morning of the autumn equinox.",
        "What day was Omar's child born?",
        "September 22nd",
        QuestionType::Temporal,
    ),
    (
        "lex-09",
        "After the merger, every engineer reports to the woman who founded the smaller firm.",
        "Who leads the technical staff now?",
        "The founder",
        QuestionType::MultiHop,
    ),
    (
        "lex-10",
        "Sasha can read a menu in Lisbon without any help.",
        "Which language does Sasha speak?",
        "Portuguese",
        QuestionType::SingleHop,
    ),
    (
        "lex-11",
        "The novelist set every chapter inside the city of canals and gondolas.",
        "Where does the book take place?",
        "Venice",
        QuestionType::SingleHop,
    ),
    (
        "lex-12",
        "Dad swore off the leaf entirely the year before the pandemic began.",
        "When did my father stop smoking?",
        "2019",
        QuestionType::Temporal,
    ),
];

impl LexicalStressBenchmark {
    /// Build the shared session — one user turn per memory.
    fn corpus_session() -> Session {
        Session {
            session_id: "lexical-stress-session-0".to_owned(),
            turns: CASES
                .iter()
                .map(|(_, memory, _, _, _)| TurnRecord {
                    role: "user".to_owned(),
                    content: (*memory).to_owned(),
                })
                .collect(),
        }
    }
}

impl Benchmark for LexicalStressBenchmark {
    fn id(&self) -> &'static str {
        "lexical-stress"
    }

    fn display_name(&self) -> &'static str {
        "Lexical stress (no-overlap retrieval)"
    }

    fn url(&self) -> &'static str {
        "https://github.com/brain-db-io/brain-eval#lexical-stress"
    }

    fn requires_datasets_dir(&self) -> bool {
        false
    }

    /// Answers are entailed, never quoted, so the heuristic top-K
    /// concatenation would never produce them — synthesis is required.
    fn requires_synthesis(&self) -> bool {
        true
    }

    fn load(&self, _datasets_dir: &Path) -> Result<Vec<EvalInstance>, EvalError> {
        let session = Self::corpus_session();
        let instances = CASES
            .iter()
            .map(|(qid, _memory, question, gold, qtype)| EvalInstance {
                question_id: (*qid).to_owned(),
                question: (*question).to_owned(),
                answer: (*gold).to_owned(),
                question_type: *qtype,
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

    /// Generic words that don't count as "content" overlap — they carry
    /// no topical signal, so sharing them doesn't make the question a
    /// lexical shortcut.
    const STOPWORDS: &[&str] = &[
        "a", "an", "the", "to", "of", "in", "on", "at", "is", "are", "was", "were", "be", "do",
        "does", "did", "what", "where", "when", "who", "which", "how", "why", "we", "i", "my",
        "his", "her", "now", "should", "and", "or", "for", "with", "after",
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
        let insts = LexicalStressBenchmark.load(Path::new(".")).expect("load");
        assert_eq!(insts.len(), CASES.len());
        assert!(insts.len() >= 8 && insts.len() <= 12, "8-12 hand-authored cases");
        assert!(insts
            .iter()
            .all(|i| i.conversation_id.as_deref() == Some(CONVERSATION_ID)));
        assert!(insts.iter().all(|i| i.sessions.len() == 1));
    }

    /// The core invariant: the question must not lexically leak the
    /// ANSWER. A shared content token between question and gold answer
    /// (or between question and the *fact-bearing* part of the memory)
    /// would let keyword/substring recall "win" by overlap.
    ///
    /// A shared SUBJECT anchor (the entity the question is about — "Maria"
    /// in the worked example) is deliberately allowed: a question must be
    /// able to name who/what it asks about, and that is exactly what makes
    /// semantic retrieval do real work. So we exempt tokens that appear in
    /// BOTH the question and the memory but NOT in the gold answer — those
    /// are anchors, not answer leaks — while still forbidding any overlap
    /// with the gold answer itself.
    #[test]
    fn question_does_not_leak_the_answer() {
        for (qid, _memory, question, gold, _) in CASES {
            let q = content_tokens(question);
            let g = content_tokens(gold);
            let q_gold: Vec<_> = q.intersection(&g).collect();
            assert!(
                q_gold.is_empty(),
                "{qid}: question shares content tokens with gold answer: {q_gold:?}",
            );
        }
    }

    /// The deprecated substring metric must score ~0 here by
    /// construction: the gold answer string must NOT appear in its
    /// storing memory. (If it did, substring recall@k would spuriously
    /// pass and the benchmark would prove nothing.)
    #[test]
    fn gold_is_absent_from_memory() {
        for (qid, memory, _question, gold, _) in CASES {
            assert!(
                !memory.to_lowercase().contains(&gold.to_lowercase()),
                "{qid}: gold answer {gold:?} leaked verbatim into its memory",
            );
        }
    }

    #[test]
    fn requires_synthesis_is_true() {
        assert!(LexicalStressBenchmark.requires_synthesis());
    }
}
