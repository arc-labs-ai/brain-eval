//! Pluggable reporters: write a [`crate::report::shape::BenchmarkReport`]
//! to disk in one of several formats. Currently JSON and text.

use std::path::Path;

use crate::report::shape::BenchmarkReport;

pub mod json;
pub mod text;

/// Trait every reporter implements. Stateless.
pub trait Reporter {
    /// Write `report` to `path`. Caller pre-creates the parent dir.
    ///
    /// # Errors
    ///
    /// Returns the underlying `io::Error` on filesystem / serialization
    /// failure.
    fn write(&self, report: &BenchmarkReport, path: &Path) -> std::io::Result<()>;
}
