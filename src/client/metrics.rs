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
//! lines of the same family are summed by [`PrometheusMetrics::sum`].

use std::net::SocketAddr;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Environment variable overriding the Prometheus `/metrics` listener port
/// for an external server. Defaults to [`DEFAULT_METRICS_PORT`].
pub const ENV_METRICS_PORT: &str = "BRAIN_EVAL_METRICS_PORT";
/// Default `/metrics` listener port (matches the server's `metrics_addr`
/// port in the standard three-port layout: data / metrics / admin).
pub const DEFAULT_METRICS_PORT: u16 = 9091;

/// Resolve the `/metrics` address for an external server: the data-plane IP
/// plus [`ENV_METRICS_PORT`] (default [`DEFAULT_METRICS_PORT`]).
#[must_use]
pub fn metrics_addr_from_env(data_plane: SocketAddr) -> SocketAddr {
    let port = std::env::var(ENV_METRICS_PORT)
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(DEFAULT_METRICS_PORT);
    SocketAddr::new(data_plane.ip(), port)
}

/// A parsed `/metrics` scrape — one [`Sample`] per exposition line.
#[derive(Debug, Clone, Default)]
pub struct PrometheusMetrics {
    samples: Vec<Sample>,
}

/// One exposition line: family name, its raw `{...}` label body (empty when
/// unlabeled), and the value. The label body is kept so callers can sum a
/// single tier/status series, not just the whole family.
#[derive(Debug, Clone)]
struct Sample {
    name: String,
    labels: String,
    value: f64,
}

impl PrometheusMetrics {
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
            let (name, labels) = match lhs.split_once('{') {
                Some((n, rest)) => (n.trim(), rest.to_string()),
                None => (lhs, String::new()),
            };
            if !name.is_empty() {
                samples.push(Sample {
                    name: name.to_string(),
                    labels,
                    value,
                });
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
        for s in &self.samples {
            if s.name == family {
                found = true;
                total += s.value;
            }
        }
        found.then_some(total)
    }

    /// Sum every sample of `family` whose label body contains
    /// `label_contains` (a raw substring like `tier="llm",status="ran"`).
    /// `None` if no such sample exists — lets a caller read one tier/status
    /// series out of a multi-labeled counter.
    #[must_use]
    pub fn sum_where(&self, family: &str, label_contains: &str) -> Option<f64> {
        let mut found = false;
        let mut total = 0.0;
        for s in &self.samples {
            if s.name == family && s.labels.contains(label_contains) {
                found = true;
                total += s.value;
            }
        }
        found.then_some(total)
    }

    /// The first sample of `family` (for scalar, unlabeled gauges).
    #[must_use]
    pub fn get(&self, family: &str) -> Option<f64> {
        self.samples
            .iter()
            .find(|s| s.name == family)
            .map(|s| s.value)
    }

    /// Whether any sampled family name begins with `prefix` (e.g.
    /// `brain_worker_`). Used by the metrics-completeness gate to assert a
    /// whole metric family emits (`worker_*` / `extractor_*` / …).
    #[must_use]
    pub fn has_prefix(&self, prefix: &str) -> bool {
        self.samples.iter().any(|s| s.name.starts_with(prefix))
    }

    /// The distinct family names beginning with `prefix`, sorted — for a
    /// human-readable report of what a prefix covered.
    #[must_use]
    pub fn families_with_prefix(&self, prefix: &str) -> Vec<String> {
        let mut names: Vec<String> = self
            .samples
            .iter()
            .filter(|s| s.name.starts_with(prefix))
            .map(|s| s.name.clone())
            .collect();
        names.sort();
        names.dedup();
        names
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
        let m = PrometheusMetrics::parse(SAMPLE);
        assert_eq!(m.sum("brain_wal_size_bytes"), Some(268_435_456.0 + 1024.0));
        assert_eq!(m.sum("brain_arena_slots_used"), Some(25.0));
    }

    #[test]
    fn reads_scalar_gauge() {
        let m = PrometheusMetrics::parse(SAMPLE);
        assert_eq!(m.get("process_memory_resident_bytes"), Some(524_288_000.0));
    }

    #[test]
    fn absent_family_is_none() {
        let m = PrometheusMetrics::parse(SAMPLE);
        assert_eq!(m.sum("brain_nonexistent"), None);
        assert_eq!(m.get("brain_nonexistent"), None);
    }

    #[test]
    fn ignores_comments_and_headers() {
        let m = PrometheusMetrics::parse("HTTP/1.1 200 OK\r\n# a comment\nfoo 1\n");
        assert_eq!(m.get("foo"), Some(1.0));
    }

    #[test]
    fn has_prefix_matches_family_prefix() {
        let m = PrometheusMetrics::parse(SAMPLE);
        assert!(m.has_prefix("brain_wal_"));
        assert!(m.has_prefix("brain_arena_"));
        assert!(!m.has_prefix("brain_worker_"));
        let fams = m.families_with_prefix("brain_");
        assert!(fams.contains(&"brain_wal_size_bytes".to_string()));
        assert!(fams.contains(&"brain_arena_slots_used".to_string()));
    }

    #[test]
    fn metrics_addr_uses_env_port_default() {
        // No env set: default metrics port applies.
        let a = metrics_addr_from_env("127.0.0.1:9090".parse().unwrap());
        assert_eq!(a.port(), DEFAULT_METRICS_PORT);
        assert_eq!(a.ip().to_string(), "127.0.0.1");
    }
}
