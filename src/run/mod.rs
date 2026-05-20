//! How an eval run actually happens.
//!
//! - [`config`]    — `RunConfig`, env-var parsing, [`ReporterKind`].
//! - [`harness`]   — `BrainEvalHarness` wrapping `brain-sdk-rust`.
//! - [`synthesize`] — top-K → candidate answer.
//! - [`runner`]    — `EvalRunner`: orchestrates ingest → recall → judge.

pub mod config;
pub mod harness;
pub mod runner;
pub mod synthesize;

pub use config::{ReporterKind, RunConfig};
pub use harness::{BrainEvalHarness, HarnessError, IngestOutcome, RecallOutcome};
pub use runner::{datasets_dir, EvalRunner};
pub use synthesize::synthesize_answer;
