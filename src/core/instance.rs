//! Common dataset shapes — every benchmark loader normalizes to these.

use serde::{Deserialize, Serialize};

/// A single (question, ground-truth answer, conversation history) triple.
/// One `EvalInstance` = one row in the benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalInstance {
    /// Benchmark-scoped question identifier (must be unique within
    /// the dataset's run).
    pub question_id: String,
    /// The question text — what the agent will be asked to answer.
    pub question: String,
    /// Ground-truth answer (used by the judge).
    pub answer: String,
    /// The evaluation dimension this question targets (used for the
    /// per-dimension accuracy breakdown).
    pub question_type: QuestionType,
    /// Optional conversation key — instances sharing the same
    /// `conversation_id` are ingested once and queried per-question.
    /// `None` means each question gets its own isolated session.
    pub conversation_id: Option<String>,
    /// The conversation turns that must be ingested before the
    /// question can be asked. Empty when the dataset only provides
    /// the question text.
    pub sessions: Vec<Session>,
}

/// A "session" is a contiguous conversation — a sequence of turns
/// that will be ENCODE'd into the substrate before any RECALL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Human-readable session id (used only for diagnostics).
    pub session_id: String,
    /// Turns in chronological order.
    pub turns: Vec<TurnRecord>,
}

/// One turn — a single utterance with role + content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnRecord {
    /// `"user"` or `"assistant"`.
    pub role: String,
    /// The utterance text.
    pub content: String,
}

/// Evaluation dimension — used to break down accuracy by question
/// kind. The variants cover the common dimensions across LongMemEval,
/// LoCoMo, DMR, and BEAM; benchmarks map their per-question type tags
/// onto one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuestionType {
    /// Recall of a single fact stated once in a single session.
    SingleHop,
    /// Synthesis across multiple sessions or turns.
    MultiHop,
    /// Time-aware queries ("what did I say last week?").
    Temporal,
    /// Returning the latest value after an update ("I moved to Berlin").
    KnowledgeUpdate,
    /// Correctly refusing when information was never mentioned.
    Abstention,
    /// Preference recall (likes / dislikes / favorites).
    Preference,
    /// Adversarial / unanswerable (LoCoMo category 5).
    Adversarial,
    /// Catch-all for benchmark types that don't map cleanly.
    Other,
}

impl QuestionType {
    /// Lower-case stable tag used in JSON output and report tables.
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Self::SingleHop => "single_hop",
            Self::MultiHop => "multi_hop",
            Self::Temporal => "temporal",
            Self::KnowledgeUpdate => "knowledge_update",
            Self::Abstention => "abstention",
            Self::Preference => "preference",
            Self::Adversarial => "adversarial",
            Self::Other => "other",
        }
    }
}
