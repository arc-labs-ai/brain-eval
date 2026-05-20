//! Test-side scaffolding — mocks and deterministic fixtures.
//!
//! - [`scripted_llm`] — deterministic, prompt-driven mock LLM.
//! - [`fixtures`]     — deterministic `EvalInstance` generators.

pub mod fixtures;
pub mod scripted_llm;

pub use fixtures::deterministic_single_hop;
pub use scripted_llm::{ScriptedLlm, ScriptedResponse};
