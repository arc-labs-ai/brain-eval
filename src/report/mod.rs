//! What the eval produces.
//!
//! - [`shape`]      — `BenchmarkReport`, `BenchmarkMeta`, `CompetitorRow`.
//! - [`baselines`]  — published competitor numbers per benchmark.
//! - [`format`]     — writers (`json`, `text`; `html` slots in here).

pub mod baselines;
pub mod format;
pub mod shape;

pub use baselines::{
    dmr_competitor_baselines, locomo_competitor_baselines, longmemeval_s_competitor_baselines,
    CompetitorBaselines,
};
pub use format::Reporter;
pub use shape::{BenchmarkMeta, BenchmarkReport, CompetitorRow};
