//! Dataset implementations.
//!
//! One file per dataset.
//!
//! | Module | Benchmark | Year | Questions | Notes |
//! |---|---|---|---|---|
//! | [`smoke`]        | Smoke (Aurora) | —    | 12    | Compiled-in; zero download; fast inner-loop Recall@1 canary. |
//! | [`lexical_stress`]| Lexical stress | —    | 12    | Compiled-in; question shares no token with gold/memory — proves semantic, not substring, retrieval. |
//! | [`paraphrase_stress`]| Paraphrase stress | — | ~40 | Compiled-in; procedurally-generated no-overlap triples + distractors — anti-overfit retrieval probe. |
//! | [`supersession_stress`]| Supersession stress | — | ~40 | Compiled-in; generated OLD→NEW updates; current-vs-prior questions probe knowledge-update direction. |
//! | [`dmr`]          | DMR (MemGPT)   | 2023 | 500   | Single-hop fact retrieval; simplest shape. |
//! | [`longmemeval`]  | LongMemEval-S  | 2025 | 500   | Multi-session, multi-dimension; heuristic-judge is directional, LLM judge for honest numbers. |
//! | [`locomo`]       | LoCoMo         | 2024 | ~1540 | Multi-hop, temporal, adversarial; samples expand to many instances. |
//!
//! Follow-ups: BEAM (1M–10M scale).

pub mod dmr;
pub mod lexical_stress;
pub mod locomo;
pub mod longmemeval;
pub mod paraphrase_stress;
pub mod smoke;
pub mod supersession_stress;

// Follow-ups:
// pub mod beam;

use std::path::Path;

use crate::bench::core::benchmark::EvalError;

/// Read a dataset file's bytes. Distinguishes a genuinely-missing file
/// ([`EvalError::DatasetNotFound`]) from any other IO failure —
/// permissions, a directory where a file was expected, etc. — which must
/// surface as [`EvalError::ParseError`] rather than being misreported as
/// "dataset not found". Shared by every file-backed loader
/// ([`dmr`], [`locomo`], [`longmemeval`]).
pub(crate) fn read_dataset_file(path: &Path) -> Result<Vec<u8>, EvalError> {
    std::fs::read(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            EvalError::DatasetNotFound {
                path: path.display().to_string(),
            }
        } else {
            EvalError::ParseError {
                path: path.display().to_string(),
                reason: format!("could not read dataset: {e}"),
            }
        }
    })
}

/// Parse one JSON value per non-empty, non-`#`-comment line — the
/// row-oriented JSONL shape some dataset releases ship instead of a
/// single JSON array. A parse failure reports the 1-based line number so
/// a malformed row is easy to find in a 500+ line release file.
pub(crate) fn parse_jsonl<T: serde::de::DeserializeOwned>(
    text: &str,
    path: &Path,
) -> Result<Vec<T>, EvalError> {
    let mut out = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let row: T = serde_json::from_str(trimmed).map_err(|e| EvalError::ParseError {
            path: path.display().to_string(),
            reason: format!("line {}: {e}", lineno + 1),
        })?;
        out.push(row);
    }
    Ok(out)
}
