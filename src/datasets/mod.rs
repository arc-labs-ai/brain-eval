//! Dataset implementations.
//!
//! One file per dataset.
//!
//! | Module | Benchmark | Year | Questions | Notes |
//! |---|---|---|---|---|
//! | [`smoke`]        | Smoke (Aurora) | —    | 12    | Compiled-in; zero download; fast inner-loop Recall@1 canary. |
//! | [`dmr`]          | DMR (MemGPT)   | 2023 | 500   | Single-hop fact retrieval; simplest shape. |
//! | [`longmemeval`]  | LongMemEval-S  | 2025 | 500   | Multi-session, multi-dimension; heuristic-judge is directional, LLM judge for honest numbers. |
//! | [`locomo`]       | LoCoMo         | 2024 | ~1540 | Multi-hop, temporal, adversarial; samples expand to many instances. |
//!
//! Follow-ups: BEAM (1M–10M scale).

pub mod dmr;
pub mod locomo;
pub mod longmemeval;
pub mod smoke;

// Follow-ups:
// pub mod beam;
