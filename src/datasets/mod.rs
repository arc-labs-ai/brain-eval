//! Dataset implementations.
//!
//! One file per dataset.
//!
//! | Module | Benchmark | Year | Questions | Notes |
//! |---|---|---|---|---|
//! | [`smoke`]        | Smoke (Aurora) | —    | 12    | Compiled-in; zero download; fast inner-loop Recall@1 canary. |
//! | [`lexical_stress`]| Lexical stress | —    | 12    | Compiled-in; question shares no token with gold/memory — proves semantic, not substring, retrieval. |
//! | [`paraphrase_stress`]| Paraphrase stress | — | ~40 | Compiled-in; procedurally-generated no-overlap triples + distractors — anti-overfit retrieval probe. |
//! | [`supersession_stress`]| Supersession stress | — | ~40 | Compiled-in; generated OLD→NEW updates; current-vs-prior questions probe knowledge-update direction. |
//! | [`dmr`]          | DMR (MemGPT)   | 2023 | 500   | Single-hop fact retrieval; simplest shape. |
//! | [`longmemeval`]  | LongMemEval-S  | 2025 | 500   | Multi-session, multi-dimension; heuristic-judge is directional, LLM judge for honest numbers. |
//! | [`locomo`]       | LoCoMo         | 2024 | ~1540 | Multi-hop, temporal, adversarial; samples expand to many instances. |
//!
//! Follow-ups: BEAM (1M–10M scale).

pub mod dmr;
pub mod lexical_stress;
pub mod locomo;
pub mod longmemeval;
pub mod paraphrase_stress;
pub mod smoke;
pub mod supersession_stress;

// Follow-ups:
// pub mod beam;
