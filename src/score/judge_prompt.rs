//! Pinned judge prompts — the exact templates the LLM judge renders,
//! plus the version + temperature recorded in every report.
//!
//! These live in their own always-compiled module (not behind
//! `live-llm`) because the report's [`crate::report::shape::BenchmarkMeta`]
//! records the sha256 of the rendered templates even on heuristic runs.
//! A silent edit to a prompt would otherwise let two runs claim the same
//! methodology while grading differently — the hash makes the drift
//! visible in the report.
//!
//! The `{...}` spans are the substitution points the judge fills in. We
//! hash the templates *as written* (placeholders intact): the structure
//! and instructions are what define the methodology, not the per-question
//! data poured into them.

use sha2::{Digest, Sha256};

/// Bumped whenever a judge prompt's wording changes. Reports carry it so
/// a methodology change is never silent.
pub const JUDGE_PROMPT_VERSION: &str = "v1";

/// Grading temperature recorded in the report for transparency. This value
/// is advisory: the judge does not thread it through to the API — the
/// [`crate::llm::LlmClient`] pins temperature 0 on every call, so verdicts
/// are deterministic regardless of what this constant says. Keep it at 0 so
/// the recorded value matches the client's pinned behavior.
pub const JUDGE_TEMPERATURE: f64 = 0.0;

/// Verdict prompt: grade a system answer against the reference.
/// `{question}`, `{ground_truth}`, `{system_answer}` are filled per call.
pub const VERDICT_PROMPT_TEMPLATE: &str = "You are a strict grader for a memory question-answering benchmark. \
Decide whether the system's answer is correct given the reference answer.\n\n\
Question: {question}\n\
Reference answer: {ground_truth}\n\
System answer: {system_answer}\n\n\
Grade \"correct\" if the system answer conveys the reference answer's key \
facts (a paraphrase or extra detail is fine). Grade \"partial\" if it is \
only partially right or omits a key detail. Grade \"incorrect\" if it is \
wrong, irrelevant, or empty. If the reference answer indicates the question \
is unanswerable, grade \"correct\" only when the system declined to answer.\n\n\
Respond with ONLY a JSON object, no prose:\n\
{{\"verdict\": \"correct\" | \"partial\" | \"incorrect\", \"reasoning\": \"<one short sentence>\"}}";

/// Support prompt: does the retrieved context actually contain the gold
/// answer? This is the answer-supporting-context-recall judge — it grades
/// retrieval, not synthesis. `{question}`, `{ground_truth}`, `{retrieved}`
/// are filled per call.
pub const SUPPORT_PROMPT_TEMPLATE: &str = "You are auditing the RETRIEVAL stage of a memory question-answering \
system. You are given a question, its gold answer, and the memories the \
system retrieved. Decide whether the gold answer can be derived ENTIRELY \
from the retrieved memories — i.e. the memories contain enough information \
to support the gold answer on their own, without outside knowledge.\n\n\
Question: {question}\n\
Gold answer: {ground_truth}\n\
Retrieved memories:\n{retrieved}\n\n\
Answer \"yes\" only if a reader could produce the gold answer using ONLY \
the retrieved memories. Answer \"no\" if the memories are missing the key \
fact, contradict it, or are empty. Paraphrase is fine — the fact need not \
be stated verbatim, but it must be present.\n\n\
Respond with ONLY a JSON object, no prose:\n\
{{\"supported\": true | false, \"reasoning\": \"<one short sentence>\"}}";

/// Hex sha256 over the concatenated judge templates. The report records
/// this so a prompt edit (even one that leaves [`JUDGE_PROMPT_VERSION`]
/// untouched) is detectable by hash mismatch.
#[must_use]
pub fn judge_prompt_sha256() -> String {
    let mut hasher = Sha256::new();
    hasher.update(VERDICT_PROMPT_TEMPLATE.as_bytes());
    hasher.update(b"\x00");
    hasher.update(SUPPORT_PROMPT_TEMPLATE.as_bytes());
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_is_stable_and_hex() {
        let a = judge_prompt_sha256();
        let b = judge_prompt_sha256();
        assert_eq!(a, b, "hash must be deterministic");
        assert_eq!(a.len(), 64, "sha256 hex is 64 chars");
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn templates_keep_their_placeholders() {
        assert!(VERDICT_PROMPT_TEMPLATE.contains("{question}"));
        assert!(VERDICT_PROMPT_TEMPLATE.contains("{system_answer}"));
        assert!(SUPPORT_PROMPT_TEMPLATE.contains("{retrieved}"));
        assert!(SUPPORT_PROMPT_TEMPLATE.contains("{ground_truth}"));
    }
}
