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
    /// Safety cap on returned set members / episodic hits per RECALL —
    /// NOT a ranking knob. The answer's shape comes from the stored data
    /// (one value, a set, or honest "don't know"); this only bounds how
    /// many episodic snippets the synthesizer sees on the fallback path.
    /// Generous on purpose: long-corpus benchmarks (LoCoMo ≈ 588
    /// turns/conversation) need a wide episodic set for the synthesizer.
    /// Measured on LoCoMo: cap-10 → 0.42 accuracy, cap-50 → 0.75.
    /// Override with `BRAIN_EVAL_TOP_K`.
    pub max_results: u32,
    /// Where to write report files.
    pub output_dir: PathBuf,
    /// Reporters to activate.
    pub reporters: Vec<ReporterKind>,
    /// Restrict the run to these question-type tags (e.g. `single_hop`).
    /// `None` runs every type. Lets a run isolate one dimension — e.g.
    /// single-fact retrieval, where the answer is one item and "is it in
    /// the retrieved list at all" is a clean signal.
    pub question_types: Option<Vec<String>>,
}

impl RunConfig {
    /// Sensible defaults for the given endpoint.
    #[must_use]
    pub fn default_for(endpoint: SocketAddr) -> Self {
        Self {
            endpoint,
            max_questions: None,
            max_results: 50,
            output_dir: PathBuf::from("target/eval-reports"),
            reporters: vec![ReporterKind::Json, ReporterKind::Text],
            question_types: None,
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

        let max_results = std::env::var("BRAIN_EVAL_TOP_K")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(50);

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

        let question_types = std::env::var("BRAIN_EVAL_QUESTION_TYPES")
            .ok()
            .map(|s| {
                s.split(',')
                    .map(|t| t.trim().to_ascii_lowercase())
                    .filter(|t| !t.is_empty())
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty());

        Ok(Self {
            endpoint,
            max_questions,
            max_results,
            output_dir: PathBuf::from(output_dir),
            reporters,
            question_types,
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
