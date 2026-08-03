//! Surface (A): per-question dataset benchmarking.
//!
//! Drives a benchmark dataset through the cognitive ops loop
//! (ENCODE → RECALL) over [`crate::client`], judges answers against
//! ground truth, and produces a `BenchmarkReport`.
//!
//! | Module        | Question it answers                              |
//! |---------------|---------------------------------------------------|
//! | [`core`]      | What types does eval revolve around?             |
//! | [`datasets`]  | Which benchmarks do we know how to load? (DMR, LongMemEval-S, LoCoMo, smoke, plus the compiled-in lexical/paraphrase/supersession stress sets) |
//! | [`score`]     | How do we score answers? (heuristic + `live-llm` judge, retrieval/latency/context-recall metrics) |
//! | [`report`]    | What does the output look like?                  |
//! | `config`, `space_manifest`, `synthesize`, `runner` | How does a run happen? (config, per-space probing, answer synthesis, orchestration) |

pub mod space_manifest;
pub mod config;
pub mod core;
pub mod datasets;
/// Staged per-stage want-vs-got probe — a benchmark-runner variant that
/// instruments each write/read stage instead of only aggregating accuracy.
pub mod probe;
pub mod report;
pub mod runner;
pub mod score;
pub mod synthesize;

pub use config::{ReporterKind, RunConfig};
pub use runner::{datasets_dir, EvalRunner};
pub use synthesize::synthesize_answer;
