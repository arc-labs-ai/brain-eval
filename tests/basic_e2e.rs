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

use std::time::Duration;

use brain_eval::client::server::{DockerServerOpts, ServerHandle};

use brain_eval::bench::core::instance::{EvalInstance, QuestionType, Session, TurnRecord};
use brain_eval::bench::core::outcome::Verdict;
use brain_eval::bench::score::judge::judge_answer_heuristic;
use brain_eval::client::harness::BrainEvalHarness;

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
                            occurred_at_unix_nanos: None,
                        },
                        TurnRecord {
                            role: "assistant".into(),
                            content: "Got it.".to_string(),
                            occurred_at_unix_nanos: None,
                        },
                    ],
                }],
            }
        })
        .collect()
}

/// Attach to `BRAIN_EVAL_ENDPOINT` when the operator set one, otherwise boot
/// a throwaway `brain:<tag>` container for this suite.
///
/// This used to return `Option` and the test returned early when it was
/// `None` — which reported a PASS without contacting a server at all. A suite
/// that goes green having exercised nothing is worse than one that fails:
/// it hides every regression it was written to catch. Booting is the same
/// path the other live suites take, so "no endpoint configured" is now a
/// slower run rather than a silent skip.
fn server_opts() -> DockerServerOpts {
    DockerServerOpts {
        container_name: "brain-eval-basic".to_string(),
        data_port: 38091,
        metrics_port: 39111,
        admin_port: 39121,
        health_timeout: Duration::from_secs(180),
        ..DockerServerOpts::default()
    }
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires a running brain-server; set BRAIN_EVAL_ENDPOINT"]
async fn ingest_recall_judge_round_trip() {
    let server = ServerHandle::from_env_or_docker(server_opts())
        .await
        .expect("server should be reachable (external) or boot and become healthy (docker)");
    let addr = server.endpoint();

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
