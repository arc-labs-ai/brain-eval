//! Deterministic generators for `EvalInstance` shapes — used by tests
//! that don't want to download a real dataset.
//!
//! Three templates rotate: `Fact`, `Preference`, `Event`. Same input
//! seed always produces the same output `(question, answer, turns)`.

use crate::core::instance::{EvalInstance, QuestionType, Session, TurnRecord};

/// Generate `n` deterministic single-hop fact-retrieval instances.
///
/// Each instance:
/// - one session, two turns (user statement → assistant ack)
/// - question asks back what was stated
/// - answer is the stated value
#[must_use]
pub fn deterministic_single_hop(n: usize) -> Vec<EvalInstance> {
    let templates: &[(&str, &str, &str)] = &[
        ("Paris", "I live in Paris.", "What city do I live in?"),
        ("blue", "My favourite colour is blue.", "What is my favourite colour?"),
        ("Toyota", "I drive a Toyota.", "What car do I drive?"),
    ];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let (answer, statement, question) = templates[i % templates.len()];
        let session = Session {
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
        };
        out.push(EvalInstance {
            question_id: format!("fixture-{i}"),
            question: question.to_string(),
            answer: answer.to_string(),
            question_type: QuestionType::SingleHop,
            conversation_id: None,
            sessions: vec![session],
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_single_hop_is_repeatable() {
        let a = deterministic_single_hop(5);
        let b = deterministic_single_hop(5);
        assert_eq!(a.len(), 5);
        assert_eq!(b.len(), 5);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.question, y.question);
            assert_eq!(x.answer, y.answer);
            assert_eq!(x.question_id, y.question_id);
        }
    }

    #[test]
    fn deterministic_single_hop_rotates_templates() {
        let v = deterministic_single_hop(3);
        let answers: Vec<&str> = v.iter().map(|i| i.answer.as_str()).collect();
        assert_eq!(answers, vec!["Paris", "blue", "Toyota"]);
    }
}
