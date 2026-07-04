//! Shared adapter layer — talks to a `brain-server`, either process
//! (Docker-managed) or connection (SDK client + admin key minting).
//!
//! Used by both surfaces: [`crate::bench`] (per-question benchmarks) and
//! [`crate::blackbox`] (whole-server suites).
//!
//! - [`auth`]    — admin-plane data key minting (`KeyMinter`) + the
//!   redacting [`auth::Secret`] newtype.
//! - [`harness`] — a connected client + convenience ENCODE/RECALL helpers.
//! - [`metrics`] — Prometheus `/metrics` scraping.
//! - [`server`]  — Docker-managed `brain-server` lifecycle.

pub mod auth;
pub mod harness;
pub mod metrics;
pub mod server;

pub use auth::{KeyMinter, MintError, Secret};
pub use harness::{BrainEvalHarness, HarnessError, IngestOutcome, RecallOutcome};
pub use metrics::PrometheusMetrics;
pub use server::{remove_volume, DockerServerOpts, ServerError, ServerHandle};
