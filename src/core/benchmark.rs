//! The `Benchmark` trait — one impl per dataset.
//!
//! The [`crate::run::EvalRunner`] is generic over `&dyn Benchmark`,
//! so adding a new dataset is mechanical: define a struct, implement
//! the trait, drop it into `src/datasets/`.

use std::path::Path;

use crate::core::instance::EvalInstance;

/// Errors that surface during dataset loading or benchmark execution.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EvalError {
    /// The dataset file or directory was not found.
    #[error("dataset file not found: {path}")]
    DatasetNotFound {
        /// The path we expected to find the dataset at.
        path: String,
    },
    /// The dataset file could not be parsed.
    #[error("dataset parse error in {path}: {reason}")]
    ParseError {
        /// Path that failed.
        path: String,
        /// Description of the parse failure.
        reason: String,
    },
    /// `BRAIN_EVAL_DATASETS_DIR` is not set — the runner can't locate
    /// dataset files. Run `scripts/download_datasets.sh` (or set the
    /// env var manually) before invoking the eval.
    #[error("BRAIN_EVAL_DATASETS_DIR is not set; run scripts/download_datasets.sh first")]
    DatasetsDirNotSet,
    /// Something went wrong driving the harness (TCP, handshake, op error).
    #[error("harness error: {0}")]
    Harness(#[from] crate::run::harness::HarnessError),
    /// Generic pipeline failure with a free-text description.
    #[error("pipeline error: {0}")]
    Pipeline(String),
}

/// A benchmark dataset that can be loaded and evaluated.
///
/// Implement this for a new dataset; the runner handles ingestion,
/// retrieval, answer synthesis, judging, and report generation.
pub trait Benchmark: Send + Sync {
    /// Short, stable, machine-readable id used in report filenames and
    /// JSON output. Examples: `"dmr"`, `"longmemeval-s"`, `"locomo"`,
    /// `"beam-1m"`.
    fn id(&self) -> &'static str;

    /// Human-readable name for reports. Example: `"DMR (MemGPT 2023)"`.
    fn display_name(&self) -> &'static str;

    /// URL to the paper or dataset repository.
    fn url(&self) -> &'static str;

    /// Load all evaluation instances from `datasets_dir`.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError::DatasetNotFound`] when the expected file
    /// is missing, [`EvalError::ParseError`] when it can't be parsed.
    fn load(&self, datasets_dir: &Path) -> Result<Vec<EvalInstance>, EvalError>;

    /// When `true`, the runner calls an LLM to synthesize the answer
    /// from retrieved memories instead of concatenating them. Requires
    /// the `live-llm` feature; defaults to `false`.
    fn requires_synthesis(&self) -> bool {
        false
    }
}
