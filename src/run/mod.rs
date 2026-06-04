//! How an eval run actually happens.
//!
//! - [`config`]    — `RunConfig`, env-var parsing, [`ReporterKind`].
//! - [`server`]    — `ServerHandle`: boot `brain:<tag>` in docker or attach
//!   to an external server.
//! - [`harness`]   — `BrainEvalHarness` wrapping `brain-db-sdk`.
//! - [`synthesize`] — top-K → candidate answer.
//! - [`runner`]    — `EvalRunner`: orchestrates ingest → recall → judge.

pub mod config;
pub mod harness;
pub mod runner;
pub mod server;
pub mod synthesize;

pub use config::{ReporterKind, RunConfig};
pub use harness::{BrainEvalHarness, HarnessError, IngestOutcome, RecallOutcome};
pub use runner::{datasets_dir, EvalRunner};
pub use server::{remove_volume, DockerServerOpts, ServerError, ServerHandle};
pub use synthesize::synthesize_answer;
