//! JSON reporter — machine-readable sidecar for CI / dashboards.

use std::fs::File;
use std::path::Path;

use super::Reporter;
use crate::report::shape::BenchmarkReport;

/// Pretty JSON dump. One file per run.
pub struct JsonReporter;

impl Reporter for JsonReporter {
    fn write(&self, report: &BenchmarkReport, path: &Path) -> std::io::Result<()> {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, report)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}
