//! `KeyMinter` — mints data-plane API keys via the server's admin HTTP
//! listener.
//!
//! Brain dropped anonymous auth: a connection's whole identity is the
//! bearer token it presents, and the server resolves `(namespace, agent,
//! permissions)` from it. The eval is a black-box client, so it cannot
//! fabricate a token — it asks the operator admin plane to mint one bound
//! to the `(namespace, agent_id)` it wants to drive load as.
//!
//! The admin plane is separate from `/metrics`: `POST {admin_base}/v1/api-keys`
//! with `Authorization: Bearer <admin_secret>`. The response carries the
//! token string (`secret`) the client then presents over the wire.
//!
//! Two ways to obtain a minter:
//! - [`ServerHandle::key_minter`](crate::client::server::ServerHandle::key_minter)
//!   when the eval owns the booted (docker) or attached (external) server.
//! - [`KeyMinter::from_external_env`] when a scenario only has the data-plane
//!   `SocketAddr` and reads the admin plane location + secret from the
//!   environment.

use std::net::SocketAddr;

/// Environment variable holding the admin bearer secret for an external
/// server. Required to mint against a server the eval did not boot.
pub const ENV_ADMIN_SECRET: &str = "BRAIN_EVAL_ADMIN_SECRET";
/// Environment variable overriding the admin listener port for an external
/// server. Defaults to [`DEFAULT_ADMIN_PORT`].
pub const ENV_ADMIN_PORT: &str = "BRAIN_EVAL_ADMIN_PORT";
/// Environment variable overriding the tenant namespace minted keys bind to.
/// Defaults to [`DEFAULT_NAMESPACE`].
pub const ENV_NAMESPACE: &str = "BRAIN_EVAL_NAMESPACE";
/// Default admin listener port (matches the server's `admin_addr` port).
pub const DEFAULT_ADMIN_PORT: u16 = 9092;
/// Default tenant namespace every eval key binds to.
pub const DEFAULT_NAMESPACE: &str = "eval";

/// A secret value that never prints itself via `Debug`. Wrapping every
/// admin-bearer-secret field in this (rather than a bare `String`) means a
/// containing struct can keep `#[derive(Debug)]` without a future
/// `debug!("{:?}", config)` (or similar) silently leaking a real
/// credential into logs — the compiler enforces the redaction, not
/// programmer discipline at each call site.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct Secret(String);

impl Secret {
    /// Wrap a raw secret value.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// The raw value, for the one place that actually needs to send it
    /// (e.g. as a bearer token). Named `expose_*` so a reviewer scanning a
    /// diff immediately sees where a secret leaves the wrapper.
    #[must_use]
    pub fn expose_secret(&self) -> &str {
        &self.0
    }

    /// True if no secret is configured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_empty() {
            f.write_str("Secret(<empty>)")
        } else {
            f.write_str("Secret(<redacted>)")
        }
    }
}

impl From<String> for Secret {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for Secret {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

/// Errors from minting a data-plane key.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MintError {
    /// No admin secret is configured, so minting cannot proceed. For an
    /// external server set [`ENV_ADMIN_SECRET`].
    #[error(
        "no admin secret configured; set {ENV_ADMIN_SECRET} to mint keys against an external server"
    )]
    NoSecret,
    /// The HTTP request to the admin plane failed to send / receive.
    #[error("admin request to {url} failed: {source}")]
    Transport {
        /// The admin endpoint we tried to reach.
        url: String,
        /// The underlying transport error.
        #[source]
        source: reqwest::Error,
    },
    /// The admin plane answered with a non-201 status.
    #[error("admin mint returned {status} from {url}: {body}")]
    Status {
        /// The endpoint that rejected the mint.
        url: String,
        /// HTTP status code.
        status: u16,
        /// A snippet of the response body, for diagnosis.
        body: String,
    },
    /// The 201 body did not carry a usable `secret` token.
    #[error("admin mint response missing a usable secret: {0}")]
    BadResponse(String),
}

/// Mints data-plane keys against one server's admin plane, all bound to a
/// single tenant namespace.
#[derive(Clone, Debug)]
pub struct KeyMinter {
    http: reqwest::Client,
    admin_base: String,
    admin_secret: Secret,
    namespace: String,
}

impl KeyMinter {
    /// A minter for an explicit admin plane. `admin_secret` may be empty —
    /// [`mint`](Self::mint) then fails with [`MintError::NoSecret`] rather
    /// than silently producing an unauthorized request.
    #[must_use]
    pub fn new(admin_base: String, admin_secret: impl Into<Secret>, namespace: String) -> Self {
        Self {
            http: reqwest::Client::new(),
            admin_base,
            admin_secret: admin_secret.into(),
            namespace,
        }
    }

    /// Build a minter for an external (already-running) server from the
    /// environment. The admin base is the data-plane IP plus
    /// [`ENV_ADMIN_PORT`] (default [`DEFAULT_ADMIN_PORT`]); the secret is
    /// [`ENV_ADMIN_SECRET`] (absent ⇒ minting unavailable); the namespace is
    /// [`ENV_NAMESPACE`] (default [`DEFAULT_NAMESPACE`]).
    #[must_use]
    pub fn from_external_env(data_plane: SocketAddr) -> Self {
        let admin_port = std::env::var(ENV_ADMIN_PORT)
            .ok()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(DEFAULT_ADMIN_PORT);
        let admin_base = format!("http://{}:{admin_port}", data_plane.ip());
        let admin_secret = std::env::var(ENV_ADMIN_SECRET).unwrap_or_default();
        let namespace =
            std::env::var(ENV_NAMESPACE).unwrap_or_else(|_| DEFAULT_NAMESPACE.to_string());
        Self::new(admin_base, admin_secret, namespace)
    }

    /// The namespace every key this minter produces is bound to.
    #[must_use]
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Mint a data-plane key bound to `(namespace, agent_id)` with `FULL`
    /// permissions and return its secret token. The token string is what a
    /// client presents over the wire (`Auth::Token(secret.into_bytes())`).
    ///
    /// # Errors
    ///
    /// [`MintError::NoSecret`] when no admin secret is configured;
    /// [`MintError::Transport`] / [`MintError::Status`] /
    /// [`MintError::BadResponse`] for admin-plane failures.
    pub async fn mint(&self, agent_id: [u8; 16]) -> Result<String, MintError> {
        if self.admin_secret.is_empty() {
            return Err(MintError::NoSecret);
        }
        let url = format!("{}/v1/api-keys", self.admin_base);
        let body = serde_json::json!({
            "agent_id_hex": hex_lower(&agent_id),
            "namespace": self.namespace,
            "permissions": ["FULL"],
        });
        let resp = self
            .http
            .post(&url)
            .bearer_auth(self.admin_secret.expose_secret())
            .json(&body)
            .send()
            .await
            .map_err(|source| MintError::Transport {
                url: url.clone(),
                source,
            })?;

        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if status.as_u16() != 201 {
            return Err(MintError::Status {
                url,
                status: status.as_u16(),
                body: snippet(&text),
            });
        }

        let parsed: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
            MintError::BadResponse(format!("not JSON: {e}; body={}", snippet(&text)))
        })?;
        match parsed.get("secret").and_then(serde_json::Value::as_str) {
            Some(s) if !s.is_empty() => Ok(s.to_string()),
            _ => Err(MintError::BadResponse(snippet(&text))),
        }
    }
}

/// Lowercase hex of a 16-byte id (32 chars), the form the admin API expects.
fn hex_lower(id: &[u8; 16]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(32);
    for b in id {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// First 200 chars of a body, for error messages.
fn snippet(s: &str) -> String {
    s.chars().take(200).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_lower_is_32_lowercase_chars() {
        assert_eq!(hex_lower(&[0xab; 16]).len(), 32);
        assert_eq!(hex_lower(&[0x00; 16]), "0".repeat(32));
        assert_eq!(&hex_lower(&[0x0f; 16])[..2], "0f");
    }

    #[tokio::test]
    async fn mint_without_secret_is_no_secret_error() {
        let m = KeyMinter::new(
            "http://127.0.0.1:9092".to_string(),
            String::new(),
            "eval".into(),
        );
        let err = m.mint([1u8; 16]).await.unwrap_err();
        assert!(matches!(err, MintError::NoSecret));
    }

    #[test]
    fn from_external_env_uses_endpoint_ip() {
        // No env set in this test process: defaults apply, secret empty.
        let m = KeyMinter::from_external_env("10.0.0.5:8080".parse().unwrap());
        assert_eq!(
            m.admin_base,
            format!("http://10.0.0.5:{DEFAULT_ADMIN_PORT}")
        );
        assert_eq!(m.namespace(), DEFAULT_NAMESPACE);
    }
}
