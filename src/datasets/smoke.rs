//! Smoke — a compiled-in micro-benchmark for fast inner-loop signal.
//!
//! Unlike the file-backed datasets (DMR, LongMemEval, LoCoMo), the
//! smoke corpus ships in the binary: 18 short memories about a single
//! fictional engineer ("Sarah Chen" at "Aurora Robotics") plus 12
//! questions whose ground-truth answer is a substring that appears in
//! exactly one memory. That uniqueness is what makes the corpus a
//! clean retrieval signal — Recall@1 == 1.0 iff every question's
//! single best hit is the intended memory.
//!
//! ## Why it exists
//!
//! It runs in seconds against a live `brain-server`, needs no dataset
//! download (`requires_datasets_dir() == false`), and reproduces the
//! exact battery we'd otherwise eyeball by hand in the shell. It's the
//! canary: if a substrate change regresses recall, `brain-eval smoke`
//! catches it before any real benchmark run.
//!
//! ## Design rules for the gold answers
//!
//! Each `answer` is a verbatim substring of its target memory and of
//! NO other memory in the corpus. The heuristic judge and the
//! Recall@K metric both reduce to "did the target memory surface?" —
//! so a green smoke run is an unambiguous statement about retrieval,
//! not about answer synthesis.

use std::path::Path;

use crate::core::benchmark::{Benchmark, EvalError};
use crate::core::instance::{EvalInstance, QuestionType, Session, TurnRecord};

/// Compiled-in smoke benchmark.
pub struct SmokeBenchmark;

/// The shared conversation key — all 12 questions run against the one
/// ingested corpus, so the runner ingests it once.
const CONVERSATION_ID: &str = "smoke-aurora-corpus";

/// The 18-memory corpus. One memory per entry; ingested as 18
/// user-turn ENCODEs.
const CORPUS: &[&str] = &[
    "Sarah Chen is a senior backend engineer at Aurora Robotics. She leads the payments platform team, which owns transaction processing, fraud detection, and the merchant payout pipeline.",
    "Sarah strongly prefers Rust over Go for new backend services. She cites Rust's compile-time memory safety, fearless concurrency, and zero-cost abstractions as the deciding factors for high-throughput systems.",
    "Sarah dislikes synchronous daily standup meetings. She believes they interrupt deep work and pushes her team toward async written status updates posted in Slack each morning.",
    "The Phoenix project is the migration of Aurora's legacy billing monolith into event-sourced microservices. The effort is scoped to finish in Q3 and is expected to cut month-end invoicing time from six hours to under twenty minutes.",
    "Sarah is the technical lead of the Phoenix project. She owns the architecture decision records and personally reviews every schema change to the event store.",
    "On March 14th 2026, the Phoenix project shipped its first service to production: the invoice generator. The rollout used a blue-green deployment and completed with zero customer-facing downtime.",
    "Sarah gave a talk at RustConf 2026 titled \"Scaling Event-Sourced Systems to Millions of Transactions a Day.\" The session covered snapshotting strategies, idempotent consumers, and back-pressure in async pipelines.",
    "Aurora Robotics closed a 40 million dollar Series B financing round led by Sequoia Capital. The company plans to use the funding to expand its warehouse automation division and double its engineering headcount over the next year.",
    "Last week the legacy billing monolith suffered a two-hour outage during peak traffic. The customer support team logged a sharp spike in complaints about failed payments and duplicate charges.",
    "Outside of work, Sarah is learning to play the cello. She practices for thirty minutes every morning before logging on and is preparing to perform a Bach suite at a community recital in the fall.",
    "Marcus Lee is the staff site reliability engineer at Aurora Robotics. He owns the on-call rotation and incident response for the payments platform.",
    "Aurora Robotics uses PostgreSQL for transactional data and ClickHouse for analytics. The Phoenix project introduces Kafka as the event backbone for the new microservices.",
    "Sarah mentors two junior engineers, Priya and Tom, with weekly one-on-ones focused on systems design and code-review habits.",
    "The fraud detection service uses a gradient-boosted decision-tree model retrained nightly on the previous day's labeled transaction outcomes.",
    "Aurora Robotics was founded in 2019 in Boston and now employs about 180 people across engineering, operations, and sales.",
    "Sarah's favorite debugging technique is writing a failing test first, then bisecting with git to find the commit that introduced the regression.",
    "The biggest risk in the Phoenix migration is dual-write consistency between the old monolith and the new event store during the cutover window.",
    "On a typical morning Sarah reviews overnight alerts, triages the team's open pull requests, then blocks two hours for focused architecture work on Phoenix.",
];

/// `(question_id, cue text, gold substring, question type)`. The gold
/// substring is verbatim-unique to one corpus memory; the
/// `corpus_uniqueness` test enforces that invariant at build time.
const QUESTIONS: &[(&str, &str, &str, QuestionType)] = &[
    (
        "smoke-01",
        "Who leads the payments team at Aurora?",
        "leads the payments platform team",
        QuestionType::SingleHop,
    ),
    (
        "smoke-02",
        "What language does Sarah prefer for backend work?",
        "prefers Rust over Go",
        QuestionType::Preference,
    ),
    (
        "smoke-03",
        "What does Sarah think about morning meetings?",
        "dislikes synchronous daily standup",
        QuestionType::Preference,
    ),
    (
        "smoke-04",
        "When did the Phoenix project ship its first service?",
        "invoice generator",
        QuestionType::Temporal,
    ),
    (
        "smoke-05",
        "Who is the technical lead of Phoenix?",
        "technical lead of the Phoenix project",
        QuestionType::SingleHop,
    ),
    (
        "smoke-06",
        "How much did Aurora raise in its Series B?",
        "40 million dollar Series B",
        QuestionType::SingleHop,
    ),
    (
        "smoke-07",
        "How was Aurora Robotics funded?",
        "Sequoia Capital",
        QuestionType::SingleHop,
    ),
    (
        "smoke-08",
        "What music does Sarah practice?",
        "learning to play the cello",
        QuestionType::Preference,
    ),
    (
        "smoke-09",
        "Who handles incident response at Aurora?",
        "staff site reliability engineer",
        QuestionType::SingleHop,
    ),
    (
        "smoke-10",
        "What databases does Aurora use?",
        "PostgreSQL for transactional data",
        QuestionType::SingleHop,
    ),
    (
        "smoke-11",
        "How does Sarah debug regressions?",
        "bisecting with git",
        QuestionType::Preference,
    ),
    (
        "smoke-12",
        "What is the biggest risk in the Phoenix migration?",
        "dual-write consistency",
        QuestionType::MultiHop,
    ),
];

impl SmokeBenchmark {
    /// Build the shared session — every corpus memory as one user turn.
    fn corpus_session() -> Session {
        Session {
            session_id: "smoke-session-0".to_owned(),
            turns: CORPUS
                .iter()
                .map(|line| TurnRecord {
                    role: "user".to_owned(),
                    content: (*line).to_owned(),
                })
                .collect(),
        }
    }
}

impl Benchmark for SmokeBenchmark {
    fn id(&self) -> &'static str {
        "smoke"
    }

    fn display_name(&self) -> &'static str {
        "Smoke (Aurora Robotics corpus)"
    }

    fn url(&self) -> &'static str {
        "https://github.com/brain-db-io/brain-eval#smoke"
    }

    fn requires_datasets_dir(&self) -> bool {
        false
    }

    fn load(&self, _datasets_dir: &Path) -> Result<Vec<EvalInstance>, EvalError> {
        // Every instance carries the same conversation so the runner
        // ingests the corpus once (off the first instance) and runs all
        // 12 questions against it.
        let session = Self::corpus_session();
        let instances = QUESTIONS
            .iter()
            .map(|(qid, cue, gold, qtype)| EvalInstance {
                question_id: (*qid).to_owned(),
                question: (*cue).to_owned(),
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

    #[test]
    fn loads_twelve_questions() {
        let insts = SmokeBenchmark.load(Path::new(".")).expect("load");
        assert_eq!(insts.len(), 12);
        // Every question shares the conversation so ingest happens once.
        assert!(insts
            .iter()
            .all(|i| i.conversation_id.as_deref() == Some(CONVERSATION_ID)));
        // Each instance carries the full 18-memory corpus.
        assert!(insts.iter().all(|i| i.sessions.len() == 1));
        assert!(insts.iter().all(|i| i.sessions[0].turns.len() == 18));
    }

    /// The core invariant: each gold substring appears in EXACTLY one
    /// corpus memory. If this fails, the smoke score stops being an
    /// unambiguous retrieval signal — two memories could satisfy the
    /// same question and Recall@1 would be meaningless.
    #[test]
    fn corpus_uniqueness() {
        for (qid, _cue, gold, _qtype) in QUESTIONS {
            let gold_lower = gold.to_lowercase();
            let matches = CORPUS
                .iter()
                .filter(|m| m.to_lowercase().contains(&gold_lower))
                .count();
            assert_eq!(
                matches, 1,
                "gold substring for {qid} ({gold:?}) matched {matches} memories; must be exactly 1",
            );
        }
    }

    #[test]
    fn does_not_require_datasets_dir() {
        assert!(!SmokeBenchmark.requires_datasets_dir());
    }
}
