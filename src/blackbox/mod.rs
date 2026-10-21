//! Surface (B): whole-server black-box suites.
//!
//! Drives a real `brain-server` (via [`crate::client`]) through scale,
//! durability, chaos, and long-soak scenarios that don't fit the
//! per-question [`crate::bench`] pipeline shape.
//!
//! - [`scale`]      — latency + sustained-throughput load generation.
//! - [`system`]     — behavioural scenarios (isolation, encode/recall/forget,
//!   durability, chaos, typed-graph) asserting end-to-end substrate contracts.
//! - [`acceptance`] — composes `scale` + `system` + [`crate::bench`] into the
//!   v1.0 acceptance gate — one pass/fail report.
//! - [`soak`]       — sustained mixed workload over a long window, sampling
//!   for error/recall/latency drift.

pub mod acceptance;
pub mod namespace_wide;
pub mod scale;
pub mod soak;
pub mod system;
