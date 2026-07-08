//! `ServerHandle` — gives an eval suite a `brain-server` to talk to.
//!
//! Two modes:
//!
//! - [`ServerHandle::external`] — connect to an already-running server
//!   (e.g. a deployment on reference hardware). The handle owns nothing;
//!   dropping it does not stop the server.
//! - [`ServerHandle::start_docker`] — boot the production image
//!   (`brain:<tag>`) in a container, wait for its healthcheck, and hand
//!   back the data-plane endpoint. The handle owns the container; dropping
//!   it (or calling [`ServerHandle::stop`]) removes the container.
//!
//! The docker path boots the **full production capability stack**: it mounts
//! the `brain-models` volume (embed + cross-encoder reranker + gliner NER,
//! all provisioned by brain-db's `.devcontainer/bootstrap-model.sh`) and
//! leaves every tier enabled — rerank and classifier load from those models,
//! so the acceptance suite exercises the real read + extraction paths rather
//! than a degraded server. The LLM is the one external dependency, and a
//! hard one: write-time HyPE is mandatory, so the server refuses to boot
//! without a provider key. The eval forwards the single shared credential
//! `BRAIN__LLM__API_KEY` from its own environment into the container
//! (provider derived from `BRAIN__LLM__MODEL`).

use std::net::SocketAddr;
use std::time::Duration;

use tokio::process::Command;
use tokio::time::{sleep, Instant};

use crate::client::auth::{KeyMinter, Secret, DEFAULT_NAMESPACE};

/// Fixed admin bearer secret the eval boots its docker servers with and
/// mints against. The brain image ships a placeholder secret; we override it
/// with a known value so the eval (and only the eval) can reach the admin
/// plane. Not a real secret — the admin listener is published only to the
/// host loopback for the test's lifetime.
const EVAL_ADMIN_SECRET: &str = "eval-admin-secret";

/// How a [`ServerHandle`] obtained its server.
enum Backing {
    /// Connected to a server this handle did not start.
    External,
    /// Owns a container; removed on drop / [`ServerHandle::stop`].
    Container { name: String },
}

/// A live `brain-server` endpoint plus ownership of whatever backs it.
pub struct ServerHandle {
    endpoint: SocketAddr,
    backing: Backing,
    /// Base URL of the admin plane (`http://host:port`) used to mint keys.
    admin_base: String,
    /// Bearer secret for the admin plane.
    admin_secret: Secret,
    /// Tenant namespace minted keys bind to.
    namespace: String,
    /// When `true`, the container is left running on stop/drop (for
    /// read-only reuse across runs). The caller then removes it manually.
    keep: bool,
}

/// Knobs for [`ServerHandle::start_docker`]. `Default` mirrors
/// `just serve-local`: `brain:latest`, embed-only, BGE model from the
/// XDG default path.
#[derive(Debug, Clone)]
pub struct DockerServerOpts {
    /// Image tag to run (`brain:<tag>`).
    pub image_tag: String,
    /// Container name. Made unique by the caller when running in parallel.
    pub container_name: String,
    /// Host port mapped to the container's data plane (`8080`).
    pub data_port: u16,
    /// Host port mapped to the container's metrics/health plane (`9091`).
    pub metrics_port: u16,
    /// Host port mapped to the container's admin plane (`9092`). The eval
    /// mints data-plane keys through this port. Kept distinct from
    /// `data_port` / `metrics_port`.
    pub admin_port: u16,
    /// Named docker volume holding the full model set (embed + reranker +
    /// gliner), mounted read-only at `/models`. Provisioned by brain-db's
    /// `.devcontainer/bootstrap-model.sh`. The three subdirectories
    /// (`bge-small-en-v1.5`, `bge-reranker-base`, `gliner-small-v2.1`) are
    /// pointed at via the model env vars.
    pub models_volume: String,
    /// Per-shard arena size (env `BRAIN__SHARD__ARENA_CAPACITY_BYTES`).
    pub arena_capacity: String,
    /// Named docker volume to mount at the server's data dir
    /// (`/var/lib/brain/data`). `None` = ephemeral (data dies with the
    /// container). `Some(name)` persists across container restarts —
    /// required for restart-recovery scenarios. Remove it with
    /// [`remove_volume`].
    pub data_volume: Option<String>,
    /// How long to wait for the container's healthcheck to go `healthy`.
    pub health_timeout: Duration,
}

impl Default for DockerServerOpts {
    fn default() -> Self {
        Self {
            image_tag: "latest".to_string(),
            container_name: "brain-eval-server".to_string(),
            data_port: 18080,
            metrics_port: 19091,
            admin_port: 19092,
            models_volume: "brain-models".to_string(),
            arena_capacity: "256MiB".to_string(),
            data_volume: None,
            health_timeout: Duration::from_secs(120),
        }
    }
}

/// Remove a named docker volume (best-effort). Use after the last
/// container on it is gone to clean up a restart-recovery scenario.
pub async fn remove_volume(volume: &str) {
    let _ = run_docker(&["volume", "rm", "-f", volume]).await;
}

/// Errors from booting or probing a docker-backed server.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ServerError {
    /// A `docker` invocation failed to spawn or returned non-zero.
    #[error("docker {context}: {detail}")]
    Docker {
        /// The docker subcommand that failed (e.g. `run`, `inspect`).
        context: String,
        /// Stderr / spawn-error detail.
        detail: String,
    },
    /// The container never reached the `healthy` state in time.
    #[error("server did not become healthy within {0:?}")]
    HealthTimeout(Duration),
    /// Failed to write the temporary env-file used to forward a secret
    /// into the container without putting it in the process's argv.
    #[error("failed writing secret env file: {0}")]
    Io(#[from] std::io::Error),
}

/// A temp env-file holding one `KEY=VALUE` line, created with owner-only
/// permissions and removed on drop. Used to forward a real secret (e.g.
/// an LLM provider key) into `docker run` via `--env-file` instead of
/// `-e KEY=VALUE` — `-e` bakes the value into the process's argv, which
/// `ps aux` reveals to *every* user on the host, not just those with
/// docker-socket access. This does **not** hide the value from `docker
/// inspect` afterward: Docker always records resolved env vars in the
/// container's own `Config.Env`, regardless of how they were supplied.
/// Closing that residual exposure needs Docker secrets (swarm-only) or a
/// vault-backed `_FILE` config convention in brain-server itself — out of
/// scope for this harness.
struct SecretEnvFile {
    path: std::path::PathBuf,
}

impl SecretEnvFile {
    fn write(key: &str, value: &str) -> std::io::Result<Self> {
        use std::io::Write;

        let path = std::env::temp_dir().join(format!(
            "brain-eval-env-{}-{}.env",
            std::process::id(),
            uuid::Uuid::now_v7()
        ));
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut file = opts.open(&path)?;
        writeln!(file, "{key}={value}")?;
        Ok(Self { path })
    }

    fn path_str(&self) -> &str {
        self.path
            .to_str()
            .expect("invariant: temp_dir()-derived path is valid UTF-8 on this host")
    }
}

impl Drop for SecretEnvFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

impl ServerHandle {
    /// Use a server this process did not start. Dropping the handle does
    /// not stop it. The admin plane (for [`key_minter`](Self::key_minter)) is
    /// resolved from the environment — see [`KeyMinter::from_external_env`].
    #[must_use]
    pub fn external(endpoint: SocketAddr) -> Self {
        // Mirror KeyMinter::from_external_env so the handle and a directly
        // built minter agree on where the admin plane is.
        let minter = KeyMinter::from_external_env(endpoint);
        // Reconstruct the parts the handle stores. KeyMinter owns them
        // privately, so derive them the same way it does.
        let admin_port = std::env::var(crate::client::auth::ENV_ADMIN_PORT)
            .ok()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(crate::client::auth::DEFAULT_ADMIN_PORT);
        Self {
            endpoint,
            backing: Backing::External,
            admin_base: format!("http://{}:{admin_port}", endpoint.ip()),
            admin_secret: Secret::new(
                std::env::var(crate::client::auth::ENV_ADMIN_SECRET).unwrap_or_default(),
            ),
            namespace: minter.namespace().to_string(),
            keep: false,
        }
    }

    /// Leave the container running when this handle stops or drops, so a
    /// later read-only run can reuse the already-ingested data. The caller
    /// owns cleanup (`docker rm -f <container>`) afterward.
    pub fn keep_alive(&mut self) {
        self.keep = true;
    }

    /// Boot `brain:<tag>` in a container, wait for `healthy`, and return a
    /// handle whose [`endpoint`](Self::endpoint) is the data plane.
    pub async fn start_docker(opts: DockerServerOpts) -> Result<Self, ServerError> {
        // Best-effort: remove any stale container with the same name.
        let _ = run_docker(&["rm", "-f", &opts.container_name]).await;

        let data_map = format!("{}:8080", opts.data_port);
        let metrics_map = format!("{}:9091", opts.metrics_port);
        // Publish the admin plane so the host can mint keys. The image binds
        // it to loopback by default; we rebind it to the container's external
        // interface (see admin_addr_env) and map it out here.
        let admin_map = format!("{}:9092", opts.admin_port);
        // The whole model volume mounts at /models; the env vars below point
        // each tier at its subdirectory.
        let model_mount = format!("{}:/models:ro", opts.models_volume);
        let arena_env = format!("BRAIN__SHARD__ARENA_CAPACITY_BYTES={}", opts.arena_capacity);
        let image = format!("brain:{}", opts.image_tag);
        // Optional persistent data volume (for restart-recovery).
        let data_mount = opts
            .data_volume
            .as_ref()
            .map(|v| format!("{v}:/var/lib/brain/data"));
        // Forward the single shared LLM credential into the container. Brain
        // requires it: write-time HyPE is mandatory, so the server refuses to
        // boot without a provider key. The provider is derived from the model
        // id, so this one var (`BRAIN__LLM__API_KEY`) covers OpenAI or
        // Anthropic — set `BRAIN__LLM__MODEL` to pick. Passed via a 0600 temp
        // env-file (see `SecretEnvFile`), not `-e`, so a real API key never
        // lands in this process's argv.
        let llm_key_file: Option<SecretEnvFile> = match std::env::var("BRAIN__LLM__API_KEY") {
            Ok(v) if !v.is_empty() => Some(SecretEnvFile::write("BRAIN__LLM__API_KEY", &v)?),
            _ => None,
        };
        // Bind the admin plane to the container's external interface (it
        // defaults to loopback) and gate it with a known secret, so the eval
        // can mint per-agent data-plane keys through the published port. The
        // admin listener is fail-closed: it refuses to start without a token.
        let admin_addr_env = "BRAIN__SERVER__ADMIN_ADDR=0.0.0.0:9092".to_string();
        let admin_token_env = format!("BRAIN__ADMIN__TOKEN={EVAL_ADMIN_SECRET}");

        // Pass through non-secret config the caller set in its own env: the
        // LLM model (provider is derived from it) and the log filter (so a
        // staged/probe run can turn on the server's `brain_debug::stage`
        // journal). The API key is NOT here — it rides the 0600 env-file.
        let mut extra_env: Vec<String> = Vec::new();
        for key in ["BRAIN__LLM__MODEL", "BRAIN_LOG"] {
            if let Ok(v) = std::env::var(key) {
                if !v.is_empty() {
                    extra_env.push(format!("{key}={v}"));
                }
            }
        }

        let mut args: Vec<&str> = vec![
            "run",
            "-d",
            "--name",
            &opts.container_name,
            // io_uring under Docker Desktop's VM needs the default seccomp
            // profile relaxed AND the memlock rlimit raised.
            "--security-opt",
            "seccomp=unconfined",
            "--ulimit",
            "memlock=-1",
            "-p",
            &data_map,
            "-p",
            &metrics_map,
            "-p",
            &admin_map,
            "-v",
            &model_mount,
        ];
        if let Some(mount) = data_mount.as_deref() {
            args.push("-v");
            args.push(mount);
        }
        // Full capability stack: embed + reranker + gliner NER all load from
        // the mounted volume; every tier stays enabled so the suite tests the
        // real read + extraction paths. The model dirs stay bespoke deploy
        // paths (`BRAIN_EMBED_MODEL_DIR` / `BRAIN_RERANK_MODEL_DIR`); the
        // classifier model path and HyPE tuning ride the generic
        // `BRAIN__SECTION__FIELD` config override. HyPE itself is mandatory
        // and always-on — there is no enable flag.
        args.extend_from_slice(&[
            "-e",
            "BRAIN_EMBED_MODEL_DIR=/models/bge-small-en-v1.5",
            "-e",
            "BRAIN_RERANK_MODEL_DIR=/models/bge-reranker-base",
            "-e",
            "BRAIN__EXTRACTORS__CLASSIFIER__MODEL_PATH=/models/gliner-small-v2.1",
            "-e",
            "BRAIN__EXTRACTORS__HYPE__NUM_QUESTIONS=6",
            "-e",
            &arena_env,
            "-e",
            &admin_addr_env,
            "-e",
            &admin_token_env,
        ]);
        for e in &extra_env {
            args.push("-e");
            args.push(e);
        }
        if let Some(f) = llm_key_file.as_ref() {
            args.push("--env-file");
            args.push(f.path_str());
        }
        args.push(&image);
        run_docker(&args).await?;
        // `llm_key_file` (if any) is removed here when it drops at the end
        // of this function's scope — its job (getting the secret into the
        // `docker run` invocation) is done once that call returns.

        let handle = Self {
            endpoint: format!("127.0.0.1:{}", opts.data_port)
                .parse()
                .expect("invariant: literal 127.0.0.1:port is a valid SocketAddr"),
            backing: Backing::Container {
                name: opts.container_name.clone(),
            },
            // The admin plane is published on the host loopback at admin_port,
            // gated by the secret we booted with.
            admin_base: format!("http://127.0.0.1:{}", opts.admin_port),
            admin_secret: Secret::new(EVAL_ADMIN_SECRET),
            namespace: DEFAULT_NAMESPACE.to_string(),
            keep: false,
        };

        handle
            .wait_healthy(&opts.container_name, opts.health_timeout)
            .await?;
        Ok(handle)
    }

    /// The data-plane endpoint a client should connect to.
    #[must_use]
    pub fn endpoint(&self) -> SocketAddr {
        self.endpoint
    }

    /// A [`KeyMinter`] bound to this server's admin plane and tenant
    /// namespace. Use it to mint per-agent data-plane keys before connecting.
    #[must_use]
    pub fn key_minter(&self) -> KeyMinter {
        KeyMinter::new(
            self.admin_base.clone(),
            self.admin_secret.clone(),
            self.namespace.clone(),
        )
    }

    /// Stop and remove the container (no-op for an external server).
    /// Idempotent; also runs on drop.
    pub async fn stop(self) {
        if self.keep {
            return;
        }
        if let Backing::Container { name } = &self.backing {
            let _ = run_docker(&["rm", "-f", name]).await;
        }
    }

    /// Poll the container's docker healthcheck until `healthy`.
    async fn wait_healthy(&self, name: &str, timeout: Duration) -> Result<(), ServerError> {
        let deadline = Instant::now() + timeout;
        loop {
            let status = run_docker(&["inspect", "-f", "{{.State.Health.Status}}", name])
                .await
                .map(|out| out.trim().to_string())
                .unwrap_or_default();

            if status == "healthy" {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(ServerError::HealthTimeout(timeout));
            }
            sleep(Duration::from_millis(500)).await;
        }
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        if self.keep {
            return;
        }
        if let Backing::Container { name } = &self.backing {
            // Best-effort synchronous cleanup — Drop can't await.
            let _ = std::process::Command::new("docker")
                .args(["rm", "-f", name])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status();
        }
    }
}

/// Run a `docker` subcommand, returning stdout on success.
async fn run_docker(args: &[&str]) -> Result<String, ServerError> {
    let output = Command::new("docker")
        .args(args)
        .output()
        .await
        .map_err(|e| ServerError::Docker {
            context: args.first().unwrap_or(&"?").to_string(),
            detail: format!("spawn failed: {e}"),
        })?;
    if !output.status.success() {
        return Err(ServerError::Docker {
            context: args.first().unwrap_or(&"?").to_string(),
            detail: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}
