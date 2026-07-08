//! Pinned judge prompts — the exact templates the LLM judge renders,
//! plus the version + temperature recorded in every report.
//!
//! These live in their own always-compiled module (not behind
//! `live-llm`) because the report's [`crate::bench::report::shape::BenchmarkMeta`]
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

use crate::bench::core::instance::QuestionType;

/// Bumped whenever a judge prompt's wording changes. Reports carry it so
/// a methodology change is never silent. v3 switches the verdict to a
/// binary, category-aware, structured-JSON grader modeled on LongMemEval's
/// per-category rubric: the model reasons first (reasoning before verdict),
/// emits `{"reasoning", "correct": bool}`, and the per-category rules
/// (temporal off-by-one, knowledge-update latest-value, strict all-present
/// lists, gold-routed abstention) are spliced in by question category.
pub const JUDGE_PROMPT_VERSION: &str = "v3";

/// Grading temperature recorded in the report for transparency. This value
/// is advisory: the judge does not thread it through to the API — the
/// [`crate::llm::LlmClient`] pins temperature 0 on every call, so verdicts
/// are deterministic regardless of what this constant says. Keep it at 0 so
/// the recorded value matches the client's pinned behavior.
pub const JUDGE_TEMPERATURE: f64 = 0.0;

/// Verdict prompt: grade a system answer against the reference. Binary,
/// category-aware, structured-JSON. The model reasons BEFORE it decides so
/// the verdict follows the reasoning rather than being rationalized after
/// the fact. `{question}`, `{ground_truth}`, `{system_answer}`,
/// `{category}`, and `{type_guidance}` are filled per call.
pub const VERDICT_PROMPT_TEMPLATE: &str = "You are a strict grader for a long-term-memory \
question-answering benchmark. Decide whether the SYSTEM ANSWER is correct given the \
REFERENCE ANSWER, judging meaning rather than surface wording.\n\n\
Question: {question}\n\
Reference answer: {ground_truth}\n\
System answer: {system_answer}\n\
Question category: {category}\n\n\
{type_guidance}\
General rules:\n\
- Judge meaning, not form: ignore paraphrase, formatting, and equivalent date-number \
expressions (e.g. \"two weeks\" vs \"14 days\").\n\
- Do not reward verbosity: extra correct context is fine, but any claim that CONTRADICTS \
the reference makes the answer incorrect.\n\
- When the reference enumerates several items, the answer is correct only if EVERY item is \
present (order-independent); extra non-contradicting items are tolerated.\n\
- An empty, evasive, or \"I don't know\" answer is incorrect UNLESS the category rule says \
otherwise.\n\n\
Reason first, then decide. Respond with ONLY a JSON object — no prose, no code fence — \
with \"reasoning\" BEFORE \"correct\":\n\
{{\"reasoning\": \"<1-2 sentences>\", \"correct\": true}}";

/// Per-category grading rule spliced into the verdict prompt at
/// `{type_guidance}`. A category-blind rubric systematically mis-grades the
/// hard categories — an abstention question wants a decline, a
/// knowledge-update question wants the newest value, a temporal count may be
/// off by one and still correct. Each arm ends in a blank line so it drops
/// cleanly ahead of the general rules. Unknown / catch-all categories reuse
/// the single-hop rule. Mirrors the LongMemEval per-category judge.
#[must_use]
pub fn verdict_type_guidance(qtype: QuestionType) -> &'static str {
    match qtype {
        QuestionType::SingleHop | QuestionType::Other => {
            "Category rule (single-hop): mark correct when the system answer states the gold \
             fact; a paraphrase or additional correct context is fine. Missing or contradicting \
             the gold fact is incorrect.\n\n"
        }
        QuestionType::MultiHop => {
            "Category rule (multi-hop): the answer must combine facts across memories. Mark \
             correct when it states the final synthesized fact, or all the intermediate facts \
             that entail it. If the reference lists several items, EVERY gold item must be \
             present (order-independent); missing any one is incorrect.\n\n"
        }
        QuestionType::Temporal => {
            "Category rule (temporal): apply the general rules, but do NOT penalize an off-by-one \
             on a day / week / month COUNT (e.g. 19 vs 18 days is correct). The referenced date or \
             event itself must still match — a wrong date or wrong event is incorrect.\n\n"
        }
        QuestionType::KnowledgeUpdate => {
            "Category rule (knowledge-update): the reference is the MOST RECENT value after an \
             update. Mark correct only if the system states the current value (it may also mention \
             older superseded values); stating only an outdated value is incorrect.\n\n"
        }
        QuestionType::Preference => {
            "Category rule (preference): mark correct when the system captures the user's stated \
             preference (the liked / disliked / favorite thing); reasonable phrasing variants are \
             fine.\n\n"
        }
        QuestionType::Abstention | QuestionType::Adversarial => {
            "Category rule (unanswerable): the reference indicates the information is unavailable. \
             Mark correct ONLY if the system declined or said it does not know; a confident, \
             concrete answer is incorrect.\n\n"
        }
    }
}

/// Support prompt: does the retrieved context actually contain the gold
/// answer? This is the answer-supporting-context-recall judge — it grades
/// retrieval, not synthesis. `{question}`, `{ground_truth}`, `{retrieved}`
/// are filled per call.
pub const SUPPORT_PROMPT_TEMPLATE: &str =
    "You are auditing the RETRIEVAL stage of a memory question-answering \
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
        assert!(VERDICT_PROMPT_TEMPLATE.contains("{category}"));
        assert!(VERDICT_PROMPT_TEMPLATE.contains("{type_guidance}"));
        assert!(SUPPORT_PROMPT_TEMPLATE.contains("{retrieved}"));
        assert!(SUPPORT_PROMPT_TEMPLATE.contains("{ground_truth}"));
    }

    #[test]
    fn verdict_prompt_reasons_before_deciding() {
        // The JSON example must place the quoted "reasoning" key ahead of the
        // quoted "correct" key so the model commits to a rationale first.
        let ri = VERDICT_PROMPT_TEMPLATE
            .find("\"reasoning\"")
            .expect("reasoning key present");
        let ci = VERDICT_PROMPT_TEMPLATE
            .find("\"correct\"")
            .expect("correct key present");
        assert!(
            ri < ci,
            "reasoning must precede correct in the JSON example"
        );
    }

    #[test]
    fn type_guidance_is_category_specific() {
        // Every category yields a non-empty rule (unknown reuses single-hop).
        assert!(verdict_type_guidance(QuestionType::SingleHop).contains("single-hop"));
        assert!(verdict_type_guidance(QuestionType::Other).contains("single-hop"));
        assert!(verdict_type_guidance(QuestionType::MultiHop).contains("multi-hop"));
        assert!(verdict_type_guidance(QuestionType::Temporal).contains("off-by-one"));
        assert!(verdict_type_guidance(QuestionType::KnowledgeUpdate).contains("MOST RECENT"));
        assert!(verdict_type_guidance(QuestionType::Preference).contains("preference"));
        assert!(verdict_type_guidance(QuestionType::Abstention).contains("unanswerable"));
        assert!(verdict_type_guidance(QuestionType::Adversarial).contains("unanswerable"));
    }
}
