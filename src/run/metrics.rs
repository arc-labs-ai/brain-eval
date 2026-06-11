//! Minimal Prometheus `/metrics` scraper.
//!
//! The soak and storage gates read the server's own gauges
//! (`process_memory_resident_bytes`, `brain_wal_size_bytes`,
//! `brain_metadata_size_bytes`, the arena gauges) rather than reaching into
//! the container. brain-eval has no HTTP client dependency, so this issues a
//! bare `GET /metrics` over a TCP socket and parses the exposition text.
//!
//! The parser is deliberately tiny: it strips comments, takes the last
//! whitespace token of each line as the value, and the family name as the
//! text before `{` (labels) or the first space. Multiple shard-labeled
//! lines of the same family are summed by [`Metrics::sum`].

use std::net::SocketAddr;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// A parsed `/metrics` scrape: `(family_name, value)` per sample line.
#[derive(Debug, Clone, Default)]
pub struct Metrics {
    samples: Vec<(String, f64)>,
}

impl Metrics {
    /// Scrape `http://addr/metrics` and parse it.
    pub async fn scrape(addr: SocketAddr) -> Result<Self, String> {
        let mut stream = TcpStream::connect(addr)
            .await
            .map_err(|e| format!("connect {addr}: {e}"))?;
        let req = format!("GET /metrics HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
        stream
            .write_all(req.as_bytes())
            .await
            .map_err(|e| format!("write: {e}"))?;
        let mut buf = Vec::with_capacity(16 * 1024);
        stream
            .read_to_end(&mut buf)
            .await
            .map_err(|e| format!("read: {e}"))?;
        Ok(Self::parse(&String::from_utf8_lossy(&buf)))
    }

    /// Parse exposition text (also the body of an HTTP response; non-metric
    /// lines such as headers and `#` comments are ignored).
    #[must_use]
    pub fn parse(text: &str) -> Self {
        let mut samples = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // value is the last whitespace-separated token.
            let Some(sp) = line.rfind(char::is_whitespace) else {
                continue;
            };
            let Ok(value) = line[sp + 1..].trim().parse::<f64>() else {
                continue;
            };
            let lhs = line[..sp].trim();
            let name = lhs.split('{').next().unwrap_or(lhs).trim();
            if !name.is_empty() {
                samples.push((name.to_string(), value));
            }
        }
        Self { samples }
    }

    /// Sum every sample of `family` (e.g. all per-shard lines of
    /// `brain_wal_size_bytes`). `None` if the family is absent.
    #[must_use]
    pub fn sum(&self, family: &str) -> Option<f64> {
        let mut found = false;
        let mut total = 0.0;
        for (name, value) in &self.samples {
            if name == family {
                found = true;
                total += value;
            }
        }
        found.then_some(total)
    }

    /// The first sample of `family` (for scalar, unlabeled gauges).
    #[must_use]
    pub fn get(&self, family: &str) -> Option<f64> {
        self.samples
            .iter()
            .find(|(name, _)| name == family)
            .map(|(_, v)| *v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
# HELP brain_wal_size_bytes Total bytes across the shard's WAL segment files.
# TYPE brain_wal_size_bytes gauge
brain_wal_size_bytes{shard=\"0\"} 268435456
brain_wal_size_bytes{shard=\"1\"} 1024
# TYPE process_memory_resident_bytes gauge
process_memory_resident_bytes 524288000
brain_arena_slots_used{shard=\"0\"} 25
";

    #[test]
    fn sums_labeled_family_across_shards() {
        let m = Metrics::parse(SAMPLE);
        assert_eq!(m.sum("brain_wal_size_bytes"), Some(268_435_456.0 + 1024.0));
        assert_eq!(m.sum("brain_arena_slots_used"), Some(25.0));
    }

    #[test]
    fn reads_scalar_gauge() {
        let m = Metrics::parse(SAMPLE);
        assert_eq!(m.get("process_memory_resident_bytes"), Some(524_288_000.0));
    }

    #[test]
    fn absent_family_is_none() {
        let m = Metrics::parse(SAMPLE);
        assert_eq!(m.sum("brain_nonexistent"), None);
        assert_eq!(m.get("brain_nonexistent"), None);
    }

    #[test]
    fn ignores_comments_and_headers() {
        let m = Metrics::parse("HTTP/1.1 200 OK\r\n# a comment\nfoo 1\n");
        assert_eq!(m.get("foo"), Some(1.0));
    }
}
