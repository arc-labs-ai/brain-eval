//! # brain-eval
//!
//! Evaluation and benchmarking for the Brain cognitive substrate.
//!
//! ## What this is
//!
//! A client-side test rig that talks to a running `brain-server` over
//! the wire via [`brain_db_sdk::BrainClient`]. Drives a benchmark dataset
//! through the cognitive ops loop (ENCODE → RECALL), judges answers
//! against ground truth, and produces a `BenchmarkReport` in JSON / text
//! form.
//!
//! ## Layout — surface first
//!
//! One shared adapter layer, and two independent surfaces built on top of
//! it:
//!
//! | Module        | What it is                                        |
//! |---------------|----------------------------------------------------|
//! | [`client`]    | Shared adapter layer: connects to and drives a `brain-server` (auth, harness, metrics, the Docker-managed server handle). Used by both surfaces below. |
//! | [`bench`]     | Surface (A): per-question dataset benchmarking — datasets, scoring, reporting, run orchestration. |
//! | [`blackbox`]  | Surface (B): whole-server black-box suites — scale/load, durability + chaos scenarios, the v1.0 acceptance gate, long-running soak. |
//! | [`llm`] (feature `live-llm`) | Shared LLM HTTP client backing the `live-llm` judge + answer synthesis. |
//!
//! ## Quick start
//!
//! ```no_run
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! use brain_eval::bench::{
//!     datasets::dmr::DmrBenchmark,
//!     report::dmr_competitor_baselines,
//!     EvalRunner, RunConfig,
//! };
//!
//! let endpoint = "127.0.0.1:9090".parse()?;
//! let config = RunConfig::default_for(endpoint);
//! let runner = EvalRunner::new(config, dmr_competitor_baselines);
//! let report = runner.run(&DmrBenchmark).await?;
//! println!("accuracy: {:.3}", report.metrics.accuracy);
//! # Ok(()) }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]

pub mod bench;
pub mod blackbox;
pub mod client;
/// Shared LLM HTTP client for the `live-llm` synthesis + judge paths.
#[cfg(feature = "live-llm")]
pub mod llm;
