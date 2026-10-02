//! Run-time configuration — endpoint, smoke-mode caps, output paths,
//! reporter selection. Also the env-var parser so an operator can
//! drive a run from a shell.

use std::net::SocketAddr;
use std::path::PathBuf;

use crate::client::auth::{
    KeyMinter, Secret, DEFAULT_ADMIN_PORT, DEFAULT_NAMESPACE, ENV_ADMIN_PORT, ENV_ADMIN_SECRET,
    ENV_NAMESPACE,
};

/// Output format selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReporterKind {
    /// Machine-readable JSON.
    Json,
    /// ASCII-table text summary.
    Text,
    /// Self-contained HTML page (all CSS inlined, no external requests).
    Html,
}

/// Runtime configuration for [`crate::bench::EvalRunner::run`].
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
    /// When set, write the per-conversation `space_id` manifest to this
    /// path during an ingesting run, so a later `read_only_spaces` run can
    /// re-query the populated store without re-ingesting.
    /// `BRAIN_EVAL_PERSIST_SPACES`.
    pub persist_spaces: Option<PathBuf>,
    /// When set, SKIP ingest and reconnect to the spaces in this manifest,
    /// running only retrieval + judging against the already-populated
    /// store — read-path iteration with no write cost.
    /// `BRAIN_EVAL_READ_ONLY_SPACES`.
    pub read_only_spaces: Option<PathBuf>,
    /// Tenant namespace minted data-plane keys bind to. `BRAIN_EVAL_NAMESPACE`,
    /// default `eval`.
    pub namespace: String,
    /// Admin bearer secret for minting data-plane keys against the endpoint's
    /// admin plane. `BRAIN_EVAL_ADMIN_SECRET`. `None` ⇒ minting unavailable
    /// (only meaningful against an external server the operator gated).
    pub admin_secret: Option<Secret>,
    /// Admin listener port on the endpoint host. `BRAIN_EVAL_ADMIN_PORT`,
    /// default `9092`.
    pub admin_port: u16,
    /// When set, restrict the run to instances whose `conversation_id`
    /// equals this value (applied before grouping, so only that
    /// conversation's haystack is ingested). `None` runs every
    /// conversation. `BRAIN_EVAL_ONLY_CONVERSATION`.
    pub only_conversation: Option<String>,
    /// When true, after each conversation's drain barrier completes,
    /// enumerate that conversation's memories via `MEMORY_LIST` on the same
    /// authenticated space, print a summary, and assert the listed count is
    /// > 0. `BRAIN_EVAL_VERIFY_MEMORY_LIST`.
    pub verify_memory_list: bool,
    /// How many conversations the runner drives at once.
    ///
    /// Conversations are independent by construction — each gets its own
    /// `SpaceId`, so their ingests and recalls cannot observe one another.
    /// The runner was nonetheless strictly sequential, which made a run cost
    /// the SUM of every conversation's server round-trips, LLM judge calls,
    /// and extraction-drain wait rather than their overlap.
    ///
    /// `1` reproduces the old strictly-sequential behaviour exactly, and is
    /// the escape hatch when a server is too small to take the load or a run
    /// has to be bisected. Higher values bound in-flight work: connections,
    /// concurrent server requests, and concurrent judge calls are all capped
    /// by this one number, so it is also the LLM rate-limit control.
    ///
    /// `BRAIN_EVAL_CONCURRENCY`.
    pub concurrency: usize,
}

/// Conversations in flight when nothing is configured.
///
/// Chosen to overlap latency without becoming a load test of the server the
/// eval is supposed to be measuring: a benchmark that saturates the box
/// reports its own queueing delay as Brain's latency. Raise it for a
/// dedicated host; set `1` to go back to strict sequence.
pub const DEFAULT_CONCURRENCY: usize = 8;

/// Clamp a configured concurrency into the usable range.
///
/// `0` is meaningless (no work would ever run) and is treated as "unset" →
/// the default, rather than hanging or silently doing nothing.
#[must_use]
pub fn normalize_concurrency(requested: usize) -> usize {
    if requested == 0 {
        DEFAULT_CONCURRENCY
    } else {
        requested
    }
}

impl RunConfig {
    /// Sensible defaults for the given endpoint.
    ///
    /// The admin plane (`admin_secret` / `admin_port`) is read from the
    /// environment (`BRAIN_EVAL_ADMIN_SECRET`, `BRAIN_EVAL_ADMIN_PORT`)
    /// exactly as [`RunConfig::from_env`] and
    /// [`KeyMinter::from_external_env`](crate::client::auth::KeyMinter::from_external_env)
    /// do: `default_for` is the "attach to an already-running external
    /// server" constructor, and for such a server the admin credentials are
    /// inherently environmental — the eval did not boot it, so it cannot know
    /// the secret any other way. Callers that own the boot (docker harness)
    /// set the fields explicitly afterward.
    #[must_use]
    pub fn default_for(endpoint: SocketAddr) -> Self {
        let admin_secret = std::env::var(ENV_ADMIN_SECRET)
            .ok()
            .filter(|s| !s.is_empty())
            .map(Secret::new);
        let admin_port = std::env::var(ENV_ADMIN_PORT)
            .ok()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(DEFAULT_ADMIN_PORT);

        Self {
            endpoint,
            max_questions: None,
            max_results: 50,
            output_dir: PathBuf::from("target/eval-reports"),
            reporters: vec![ReporterKind::Json, ReporterKind::Text, ReporterKind::Html],
            question_types: None,
            persist_spaces: None,
            read_only_spaces: None,
            namespace: DEFAULT_NAMESPACE.to_string(),
            admin_secret,
            admin_port,
            only_conversation: None,
            verify_memory_list: false,
            concurrency: DEFAULT_CONCURRENCY,
        }
    }

    /// A [`KeyMinter`] for this run's endpoint, namespace, and admin plane.
    /// Used to mint per-space data-plane keys before connecting.
    #[must_use]
    pub fn key_minter(&self) -> KeyMinter {
        KeyMinter::new(
            format!("http://{}:{}", self.endpoint.ip(), self.admin_port),
            self.admin_secret.clone().unwrap_or_default(),
            self.namespace.clone(),
        )
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
            std::env::var("BRAIN_EVAL_FORMATS").unwrap_or_else(|_| "json,text,html".to_owned());
        let reporters: Vec<ReporterKind> = formats_str
            .split(',')
            .filter_map(|s| match s.trim().to_ascii_lowercase().as_str() {
                "json" => Some(ReporterKind::Json),
                "text" | "txt" => Some(ReporterKind::Text),
                "html" => Some(ReporterKind::Html),
                _ => None,
            })
            .collect();
        let reporters = if reporters.is_empty() {
            vec![ReporterKind::Json, ReporterKind::Text, ReporterKind::Html]
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

        let persist_spaces = std::env::var("BRAIN_EVAL_PERSIST_SPACES")
            .ok()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        let read_only_spaces = std::env::var("BRAIN_EVAL_READ_ONLY_SPACES")
            .ok()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);

        let namespace =
            std::env::var(ENV_NAMESPACE).unwrap_or_else(|_| DEFAULT_NAMESPACE.to_string());
        let admin_secret = std::env::var(ENV_ADMIN_SECRET)
            .ok()
            .filter(|s| !s.is_empty())
            .map(Secret::new);
        let admin_port = std::env::var(ENV_ADMIN_PORT)
            .ok()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(DEFAULT_ADMIN_PORT);

        let only_conversation = std::env::var("BRAIN_EVAL_ONLY_CONVERSATION")
            .ok()
            .filter(|s| !s.is_empty());
        let verify_memory_list = matches!(
            std::env::var("BRAIN_EVAL_VERIFY_MEMORY_LIST")
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        );
        // An unparseable value falls back to the default rather than
        // failing the run: a typo in a throughput knob must not stop an
        // eval, and the effective value is reported in the run meta.
        let concurrency = normalize_concurrency(
            std::env::var("BRAIN_EVAL_CONCURRENCY")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(DEFAULT_CONCURRENCY),
        );

        Ok(Self {
            endpoint,
            max_questions,
            max_results,
            output_dir: PathBuf::from(output_dir),
            reporters,
            question_types,
            persist_spaces,
            read_only_spaces,
            namespace,
            admin_secret,
            admin_port,
            only_conversation,
            verify_memory_list,
            concurrency,
        })
    }

    /// Override the endpoint.
    #[must_use]
    pub fn with_endpoint(mut self, ep: SocketAddr) -> Self {
        self.endpoint = ep;
        self
    }

    /// Override how many conversations run at once. `0` means "use the
    /// default"; `1` is strict sequence.
    #[must_use]
    pub fn with_concurrency(mut self, n: usize) -> Self {
        self.concurrency = normalize_concurrency(n);
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
