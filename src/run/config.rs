//! Run-time configuration — endpoint, smoke-mode caps, output paths,
//! reporter selection. Also the env-var parser so an operator can
//! drive a run from a shell.

use std::net::SocketAddr;
use std::path::PathBuf;

/// Output format selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReporterKind {
    /// Machine-readable JSON.
    Json,
    /// ASCII-table text summary.
    Text,
}

/// Runtime configuration for [`crate::run::EvalRunner::run`].
#[derive(Debug, Clone)]
pub struct RunConfig {
    /// Brain-server endpoint the harness connects to.
    pub endpoint: SocketAddr,
    /// Cap question count (smoke runs). `None` = run all.
    pub max_questions: Option<usize>,
    /// `top_k` passed to every RECALL. Generous on purpose: long-corpus
    /// benchmarks (LoCoMo ≈ 588 turns/conversation) need a wide candidate
    /// set for the cross-encoder to rerank — pulling only the top few
    /// strands the answer-bearing memory below the cutoff. Measured on
    /// LoCoMo: top-10 → 0.42 accuracy, top-50 → 0.75. Override with
    /// `BRAIN_EVAL_TOP_K`.
    pub top_k_retrieve: u32,
    /// Where to write report files.
    pub output_dir: PathBuf,
    /// Reporters to activate.
    pub reporters: Vec<ReporterKind>,
}

impl RunConfig {
    /// Sensible defaults for the given endpoint.
    #[must_use]
    pub fn default_for(endpoint: SocketAddr) -> Self {
        Self {
            endpoint,
            max_questions: None,
            top_k_retrieve: 50,
            output_dir: PathBuf::from("target/eval-reports"),
            reporters: vec![ReporterKind::Json, ReporterKind::Text],
        }
    }

    /// Build a config from env vars (`BRAIN_EVAL_ENDPOINT`,
    /// `BRAIN_EVAL_MAX_QUESTIONS`, `BRAIN_EVAL_TOP_K`,
    /// `BRAIN_EVAL_OUTPUT_DIR`, `BRAIN_EVAL_FORMATS`).
    ///
    /// # Errors
    ///
    /// Returns an error if `BRAIN_EVAL_ENDPOINT` is set to an
    /// unparseable address.
    pub fn from_env(default_endpoint: SocketAddr) -> Result<Self, std::net::AddrParseError> {
        let endpoint = std::env::var("BRAIN_EVAL_ENDPOINT")
            .ok()
            .map(|s| s.parse::<SocketAddr>())
            .transpose()?
            .unwrap_or(default_endpoint);

        let max_questions = std::env::var("BRAIN_EVAL_MAX_QUESTIONS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());

        let top_k_retrieve = std::env::var("BRAIN_EVAL_TOP_K")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(10);

        let output_dir = std::env::var("BRAIN_EVAL_OUTPUT_DIR")
            .unwrap_or_else(|_| "target/eval-reports".to_owned());

        let formats_str =
            std::env::var("BRAIN_EVAL_FORMATS").unwrap_or_else(|_| "json,text".to_owned());
        let reporters: Vec<ReporterKind> = formats_str
            .split(',')
            .filter_map(|s| match s.trim().to_ascii_lowercase().as_str() {
                "json" => Some(ReporterKind::Json),
                "text" | "txt" => Some(ReporterKind::Text),
                _ => None,
            })
            .collect();
        let reporters = if reporters.is_empty() {
            vec![ReporterKind::Json, ReporterKind::Text]
        } else {
            reporters
        };

        Ok(Self {
            endpoint,
            max_questions,
            top_k_retrieve,
            output_dir: PathBuf::from(output_dir),
            reporters,
        })
    }

    /// Override the endpoint.
    #[must_use]
    pub fn with_endpoint(mut self, ep: SocketAddr) -> Self {
        self.endpoint = ep;
        self
    }

    /// Cap question count.
    #[must_use]
    pub fn with_max_questions(mut self, n: usize) -> Self {
        self.max_questions = Some(n);
        self
    }

    /// Override the output directory.
    #[must_use]
    pub fn with_output_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.output_dir = dir.into();
        self
    }

    /// Override the reporter set.
    #[must_use]
    pub fn with_reporters(mut self, reporters: Vec<ReporterKind>) -> Self {
        self.reporters = reporters;
        self
    }
}
