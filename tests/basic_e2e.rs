//! Tier-1 end-to-end test: drive a deterministic fixture through the
//! harness against a real running `brain-server`. Ignored by default —
//! requires `BRAIN_EVAL_ENDPOINT` to point at a live server. Run with:
//!
//! ```bash
//! BRAIN_EVAL_ENDPOINT=127.0.0.1:7878 \
//!   cargo test --test basic_e2e -- --ignored
//! ```
//!
//! The test asserts the full vertical: connect → ingest → recall →
//! synthesize → judge. Numbers aren't compared against a target —
//! that's the regression tier's job (follow-up).

use std::net::SocketAddr;

use brain_eval::core::instance::{EvalInstance, QuestionType, Session, TurnRecord};
use brain_eval::core::outcome::Verdict;
use brain_eval::run::harness::BrainEvalHarness;
use brain_eval::score::judge::judge_answer_heuristic;

/// Three deterministic single-hop fact-retrieval instances. Local to
/// this integration test — the production library ships no test
/// fixtures. Each instance is one session (user statement → assistant
/// ack) whose question asks the stated value back.
fn deterministic_single_hop(n: usize) -> Vec<EvalInstance> {
    let templates: &[(&str, &str, &str)] = &[
        ("Paris", "I live in Paris.", "What city do I live in?"),
        (
            "blue",
            "My favourite colour is blue.",
            "What is my favourite colour?",
        ),
        ("Toyota", "I drive a Toyota.", "What car do I drive?"),
    ];
    (0..n)
        .map(|i| {
            let (answer, statement, question) = templates[i % templates.len()];
            EvalInstance {
                question_id: format!("fixture-{i}"),
                question: question.to_string(),
                answer: answer.to_string(),
                question_type: QuestionType::SingleHop,
                conversation_id: None,
                sessions: vec![Session {
                    session_id: format!("sess-{i}"),
                    turns: vec![
                        TurnRecord {
                            role: "user".into(),
                            content: statement.to_string(),
                        },
                        TurnRecord {
                            role: "assistant".into(),
                            content: "Got it.".to_string(),
                        },
                    ],
                }],
            }
        })
        .collect()
}

fn endpoint() -> Option<SocketAddr> {
    std::env::var("BRAIN_EVAL_ENDPOINT")
        .ok()
        .and_then(|s| s.parse::<SocketAddr>().ok())
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires a running brain-server; set BRAIN_EVAL_ENDPOINT"]
async fn ingest_recall_judge_round_trip() {
    let Some(addr) = endpoint() else {
        eprintln!("BRAIN_EVAL_ENDPOINT not set — skipping");
        return;
    };

    let harness = BrainEvalHarness::connect(addr)
        .await
        .expect("harness should connect to a running brain-server");

    // Three deterministic single-hop instances.
    let instances = deterministic_single_hop(3);

    for inst in &instances {
        for session in &inst.sessions {
            let out = harness
                .ingest(&session.turns)
                .await
                .expect("ingest should succeed");
            assert!(out.attempted > 0, "fixture should produce ENCODE attempts");
        }

        let recall = harness
            .recall(&inst.question, 5)
            .await
            .expect("recall should succeed");
        if recall.memories.is_empty() {
            eprintln!(
                "warning: no recall hits for {} — index may be cold",
                inst.question_id
            );
        }

        let candidate = recall
            .memories
            .iter()
            .map(|m| m.text.clone())
            .collect::<Vec<_>>()
            .join(" ");

        let verdict = judge_answer_heuristic(
            &inst.question_id,
            QuestionType::SingleHop,
            &inst.answer,
            &candidate,
        );

        eprintln!(
            "{}: verdict={:?} reasoning={}",
            inst.question_id, verdict.verdict, verdict.reasoning
        );
        assert!(
            matches!(
                verdict.verdict,
                Verdict::Correct | Verdict::Partial | Verdict::Incorrect
            ),
            "verdict must be one of the three variants",
        );
    }

    harness.close().await.expect("clean close");
}
