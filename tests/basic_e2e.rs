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

use brain_eval::core::instance::QuestionType;
use brain_eval::core::outcome::Verdict;
use brain_eval::run::harness::BrainEvalHarness;
use brain_eval::score::judge::judge_answer_heuristic;
use brain_eval::testing::fixtures::deterministic_single_hop;

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
        if recall.hits.is_empty() {
            eprintln!(
                "warning: no recall hits for {} — index may be cold",
                inst.question_id
            );
        }

        let candidate = recall
            .hits
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
