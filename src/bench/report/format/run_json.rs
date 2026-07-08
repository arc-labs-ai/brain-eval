//! JSON reporter for the exhaustive [`RunReport`] artifact.
//!
//! The sibling [`super::json::JsonReporter`] writes the stable summary
//! [`crate::bench::report::shape::BenchmarkReport`]; this one writes the
//! lossless [`RunReport`] to `<stem>.run.json`. It is a separate reporter
//! (not the [`super::Reporter`] trait) because that trait is typed to the
//! summary report — the two artifacts have different shapes on purpose.

use std::fs::File;
use std::path::Path;

use crate::bench::report::run_report::RunReport;

/// Pretty-JSON writer for a [`RunReport`]. One `*.run.json` per run.
pub struct JsonRunReporter;

impl JsonRunReporter {
    /// Write `report` to `path`. The caller pre-creates the parent dir.
    ///
    /// # Errors
    ///
    /// Returns the underlying `io::Error` on filesystem / serialization
    /// failure.
    pub fn write(&self, report: &RunReport, path: &Path) -> std::io::Result<()> {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, report).map_err(std::io::Error::other)
    }
}
