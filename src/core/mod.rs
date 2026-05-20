//! Foundational types every other module depends on.
//!
//! - [`benchmark`] — the `Benchmark` trait and `EvalError`.
//! - [`instance`] — the `EvalInstance` shape every loader normalizes to.
//! - [`outcome`] — per-question results + judge verdicts.

pub mod benchmark;
pub mod instance;
pub mod outcome;

pub use benchmark::{Benchmark, EvalError};
pub use instance::{EvalInstance, QuestionType, Session, TurnRecord};
pub use outcome::{JudgeResult, QuestionResult, Verdict};
