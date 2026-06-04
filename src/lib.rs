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
//! ## Layout — pipeline first
//!
//! Five top-level folders, each answering one question:
//!
//! | Folder        | Question it answers                              |
//! |---------------|--------------------------------------------------|
//! | [`core`]      | What types does eval revolve around?             |
//! | [`run`]       | How does a run happen?                           |
//! | [`score`]     | How do we score answers?                         |
//! | [`report`]    | What does the output look like?                  |
//! | [`datasets`]  | Which benchmarks do we know how to load?         |
//!
//! ## Quick start
//!
//! ```no_run
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! use brain_eval::{
//!     datasets::dmr::DmrBenchmark,
//!     report::dmr_competitor_baselines,
//!     run::{EvalRunner, RunConfig},
//! };
//!
//! let endpoint = "127.0.0.1:7878".parse()?;
//! let config = RunConfig::default_for(endpoint);
//! let runner = EvalRunner::new(config, dmr_competitor_baselines);
//! let report = runner.run(&DmrBenchmark).await?;
//! println!("accuracy: {:.3}", report.metrics.accuracy);
//! # Ok(()) }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]

pub mod core;
pub mod datasets;
pub mod report;
pub mod run;
pub mod scale;
pub mod score;
