//! Dataset version-pinning and contamination guards.
//!
//! A benchmark number is only comparable across runs if the corpus behind it is
//! byte-for-byte the one the baseline was measured on. Public memory/RAG
//! datasets (LoCoMo, LongMemEval) get silently re-released, re-ordered, and
//! re-cleaned; an unpinned loader will happily score a different corpus under
//! the same name and call the delta a regression. Two guards:
//!
//! - [`DatasetPin`] + [`pin_instances`] / [`verify_pin`] — a content hash over
//!   the *parsed* instances (question ids, questions, answers, and the turn
//!   texts), order-independent, so a re-shuffled release with identical content
//!   still verifies but an edited answer does not. The pin records the count
//!   and the hash; a mismatch is a loud error, not a quiet drift.
//! - [`contamination_report`] — a leakage check within a corpus: how many
//!   questions have their gold answer sitting verbatim in the conversation
//!   context the system will ingest. Some presence is expected (the answer must
//!   be *recoverable*), but a corpus where the answer is always a trivial
//!   substring of a single turn measures string-matching, not memory — the
//!   report surfaces that so it is a known property, not a hidden inflator.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::bench::core::instance::EvalInstance;

/// A pinned fingerprint of a parsed corpus.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetPin {
    /// Dataset name (e.g. `"locomo"`), for the error message on mismatch.
    pub name: String,
    /// Number of instances the pin was taken over.
    pub instances: usize,
    /// Hex sha256 over the canonicalised instance content — order-independent.
    pub content_sha256: String,
}

/// Per-instance canonical bytes: the fields whose change would change what the
/// benchmark measures. Turn texts are included (a re-cleaned context is a
/// different corpus), timestamps and session ids are not (they are provenance,
/// not content the score depends on).
fn instance_digest(inst: &EvalInstance) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(inst.question_id.as_bytes());
    h.update(b"\x1f");
    h.update(inst.question.as_bytes());
    h.update(b"\x1f");
    h.update(inst.answer.as_bytes());
    h.update(b"\x1f");
    for session in &inst.sessions {
        for turn in &session.turns {
            h.update(turn.role.as_bytes());
            h.update(b"\x1e");
            h.update(turn.content.as_bytes());
            h.update(b"\x1e");
        }
    }
    h.finalize().into()
}

/// Compute a content pin over a parsed corpus. Order-independent: the
/// per-instance digests are XOR-folded, so a re-ordered release with identical
/// instances produces the same pin. (XOR-fold can't detect a duplicated pair of
/// identical instances cancelling out, which real corpora don't contain; the
/// instance count in the pin catches a changed cardinality regardless.)
#[must_use]
pub fn pin_instances(name: &str, instances: &[EvalInstance]) -> DatasetPin {
    let mut fold = [0_u8; 32];
    for inst in instances {
        let d = instance_digest(inst);
        for (f, b) in fold.iter_mut().zip(d.iter()) {
            *f ^= b;
        }
    }
    // Fold in the count so an added-and-removed pair can't silently pass.
    let mut h = Sha256::new();
    h.update(fold);
    h.update((instances.len() as u64).to_le_bytes());
    let digest = h.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(hex, "{byte:02x}");
    }
    DatasetPin {
        name: name.to_owned(),
        instances: instances.len(),
        content_sha256: hex,
    }
}

/// Why a corpus failed to match its pin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PinMismatch {
    /// Instance count differs.
    Count {
        /// Count recorded in the pin.
        expected: usize,
        /// Count in the corpus being verified.
        actual: usize,
    },
    /// Content hash differs (count may still match).
    Hash {
        /// Hash recorded in the pin.
        expected: String,
        /// Hash of the corpus being verified.
        actual: String,
    },
}

impl std::fmt::Display for PinMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Count { expected, actual } => write!(
                f,
                "instance count changed: pinned {expected}, loaded {actual}"
            ),
            Self::Hash { expected, actual } => write!(
                f,
                "content hash changed: pinned {expected}, loaded {actual} \
                 (corpus edited or re-released — re-pin deliberately)"
            ),
        }
    }
}

/// Verify a loaded corpus against a pin. `Ok(())` on match, else the specific
/// mismatch. Count is checked before hash so the message names the simpler
/// cause first.
///
/// # Errors
/// Returns [`PinMismatch`] when the corpus differs from the pin.
pub fn verify_pin(pin: &DatasetPin, instances: &[EvalInstance]) -> Result<(), PinMismatch> {
    if pin.instances != instances.len() {
        return Err(PinMismatch::Count {
            expected: pin.instances,
            actual: instances.len(),
        });
    }
    let actual = pin_instances(&pin.name, instances);
    if actual.content_sha256 != pin.content_sha256 {
        return Err(PinMismatch::Hash {
            expected: pin.content_sha256.clone(),
            actual: actual.content_sha256,
        });
    }
    Ok(())
}

/// Answer-leakage summary for a corpus.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContaminationReport {
    /// Instances considered (those with a non-empty gold answer; abstention
    /// and empty-gold questions are excluded — they have nothing to leak).
    pub considered: usize,
    /// Of those, how many have the gold answer verbatim (case-insensitive) in
    /// at least one ingested turn.
    pub answer_in_context: usize,
    /// Of those, how many have the gold answer verbatim in the *question* text
    /// itself — a harder leak: the system could "answer" without any memory.
    pub answer_in_question: usize,
}

impl ContaminationReport {
    /// Fraction of considered instances whose answer leaks into the context,
    /// in `[0,1]`. `0.0` when nothing was considered.
    #[must_use]
    pub fn context_leak_rate(&self) -> f64 {
        if self.considered == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        {
            self.answer_in_context as f64 / self.considered as f64
        }
    }

    /// Fraction whose answer leaks into the question text, in `[0,1]`. A
    /// non-trivial value here means the benchmark partly measures reading the
    /// question, not the memory.
    #[must_use]
    pub fn question_leak_rate(&self) -> f64 {
        if self.considered == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        {
            self.answer_in_question as f64 / self.considered as f64
        }
    }
}

/// Scan a corpus for gold-answer leakage. See [`ContaminationReport`].
#[must_use]
pub fn contamination_report(instances: &[EvalInstance]) -> ContaminationReport {
    let mut considered = 0_usize;
    let mut in_context = 0_usize;
    let mut in_question = 0_usize;

    for inst in instances {
        let gold = inst.answer.trim().to_ascii_lowercase();
        // Nothing to leak for abstention / empty-gold, and single-token or
        // very short answers ("yes", a year) give false positives — require a
        // gold answer with real content.
        if gold.len() < 4 {
            continue;
        }
        considered += 1;

        if inst.question.to_ascii_lowercase().contains(&gold) {
            in_question += 1;
        }
        let leaks_to_context = inst.sessions.iter().any(|s| {
            s.turns
                .iter()
                .any(|t| t.content.to_ascii_lowercase().contains(&gold))
        });
        if leaks_to_context {
            in_context += 1;
        }
    }

    ContaminationReport {
        considered,
        answer_in_context: in_context,
        answer_in_question: in_question,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::core::instance::{EvalInstance, QuestionType, Session, TurnRecord};

    fn turn(role: &str, content: &str) -> TurnRecord {
        TurnRecord {
            role: role.to_owned(),
            content: content.to_owned(),
            occurred_at_unix_nanos: None,
        }
    }

    fn inst(id: &str, q: &str, a: &str, context: &str) -> EvalInstance {
        EvalInstance {
            question_id: id.to_owned(),
            question: q.to_owned(),
            answer: a.to_owned(),
            question_type: QuestionType::SingleHop,
            conversation_id: Some("c1".to_owned()),
            sessions: vec![Session {
                session_id: "s1".to_owned(),
                turns: vec![turn("user", context)],
            }],
        }
    }

    #[test]
    fn pin_is_order_independent_but_content_sensitive() {
        let a = inst("q1", "where?", "Paris", "I live in Paris");
        let b = inst("q2", "who?", "Ada", "my friend Ada");
        let forward = pin_instances("t", &[a.clone(), b.clone()]);
        let reversed = pin_instances("t", &[b.clone(), a.clone()]);
        assert_eq!(forward, reversed, "reorder must not change the pin");

        let mut edited = a.clone();
        edited.answer = "London".to_owned();
        let changed = pin_instances("t", &[edited, b]);
        assert_ne!(forward.content_sha256, changed.content_sha256);
    }

    #[test]
    fn verify_detects_count_and_hash() {
        let a = inst("q1", "where?", "Paris", "I live in Paris");
        let b = inst("q2", "who?", "Ada", "my friend Ada");
        let pin = pin_instances("t", &[a.clone(), b.clone()]);

        assert!(verify_pin(&pin, &[a.clone(), b.clone()]).is_ok());
        assert!(matches!(
            verify_pin(&pin, &[a.clone()]),
            Err(PinMismatch::Count { .. })
        ));

        let mut edited = b.clone();
        edited.question = "whom?".to_owned();
        assert!(matches!(
            verify_pin(&pin, &[a, edited]),
            Err(PinMismatch::Hash { .. })
        ));
    }

    #[test]
    fn contamination_counts_context_and_question_leaks() {
        let leaks_ctx = inst("q1", "where do they live?", "Paris", "I live in Paris now");
        let leaks_q = inst("q2", "is the code Paris?", "Paris", "some unrelated turn");
        let clean = inst("q3", "where?", "Reykjavik", "far up north somewhere");
        let short = inst("q4", "yes or no?", "yes", "the answer is yes");

        let r = contamination_report(&[leaks_ctx, leaks_q, clean, short]);
        assert_eq!(r.considered, 3, "short 'yes' answer excluded");
        assert_eq!(r.answer_in_context, 1);
        assert_eq!(r.answer_in_question, 1);
        assert!((r.context_leak_rate() - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn empty_corpus_is_clean() {
        let r = contamination_report(&[]);
        assert_eq!(r.considered, 0);
        assert!(r.context_leak_rate().abs() < 1e-9);
    }
}
