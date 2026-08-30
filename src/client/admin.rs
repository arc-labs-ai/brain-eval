//! `AdminClient` — a tiny bare-`reqwest` client for the operator admin
//! HTTP plane.
//!
//! Several acceptance scenarios drive admin-only `/v1/*` routes that the SDK
//! (a data-plane client) does not surface: the historical audit log
//! (`GET /v1/audit`) and derived-index rebuild (`POST /v1/rebuild`,
//! `POST /v1/rebuild-ann`). These are deployment-wide operator surfaces gated
//! by the `/v1` admin bearer token — exactly like [`KeyMinter`] and
//! [`PrometheusMetrics`], which already reach the admin / metrics planes with
//! raw HTTP rather than the wire protocol.
//!
//! This shares [`KeyMinter`]'s environment contract: the admin base is the
//! data-plane IP plus [`ENV_ADMIN_PORT`] (default [`DEFAULT_ADMIN_PORT`]) and
//! the bearer secret is [`ENV_ADMIN_SECRET`]. One client is reused across the
//! audit and rebuild scenarios.
//!
//! [`KeyMinter`]: crate::client::auth::KeyMinter
//! [`PrometheusMetrics`]: crate::client::metrics::PrometheusMetrics

use std::net::SocketAddr;

use crate::client::auth::{Secret, DEFAULT_ADMIN_PORT, ENV_ADMIN_PORT, ENV_ADMIN_SECRET};

/// Errors from an admin-plane request.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AdminError {
    /// No admin secret is configured, so the request cannot be authorized.
    /// Set [`ENV_ADMIN_SECRET`] to reach an external server's admin plane.
    #[error(
        "no admin secret configured; set {ENV_ADMIN_SECRET} to reach an external server's admin plane"
    )]
    NoSecret,
    /// The HTTP request failed to send / receive.
    #[error("admin request to {url} failed: {source}")]
    Transport {
        /// The admin endpoint we tried to reach.
        url: String,
        /// The underlying transport error.
        #[source]
        source: reqwest::Error,
    },
}

/// One admin-plane HTTP response: status code + body text. Kept deliberately
/// small — the scenarios parse the JSON body themselves.
#[derive(Debug, Clone)]
pub struct AdminResponse {
    /// HTTP status code.
    pub status: u16,
    /// Response body, decoded lossily as UTF-8.
    pub body: String,
}

impl AdminResponse {
    /// `true` for any 2xx status.
    #[must_use]
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }
}

/// A bare-HTTP client for one server's operator admin plane. Every request
/// carries the `/v1` admin bearer token.
#[derive(Clone, Debug)]
pub struct AdminClient {
    http: reqwest::Client,
    base: String,
    secret: Secret,
}

impl AdminClient {
    /// A client for an explicit admin base (e.g. `http://127.0.0.1:9092`).
    /// `secret` may be empty — every request then fails with
    /// [`AdminError::NoSecret`] rather than sending an unauthorized request.
    #[must_use]
    pub fn new(base: String, secret: impl Into<Secret>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base,
            secret: secret.into(),
        }
    }

    /// Build a client for an external (already-running) server from the
    /// environment, sharing [`KeyMinter`](crate::client::auth::KeyMinter)'s
    /// contract: the admin base is the data-plane IP plus [`ENV_ADMIN_PORT`]
    /// (default [`DEFAULT_ADMIN_PORT`]); the secret is [`ENV_ADMIN_SECRET`].
    #[must_use]
    pub fn from_external_env(data_plane: SocketAddr) -> Self {
        let admin_port = std::env::var(ENV_ADMIN_PORT)
            .ok()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(DEFAULT_ADMIN_PORT);
        let base = format!("http://{}:{admin_port}", data_plane.ip());
        let secret = std::env::var(ENV_ADMIN_SECRET).unwrap_or_default();
        Self::new(base, secret)
    }

    /// Whether an admin secret is configured. Scenarios can short-circuit
    /// with a clear message instead of a bare `NoSecret` failure.
    #[must_use]
    pub fn has_secret(&self) -> bool {
        !self.secret.is_empty()
    }

    /// `GET {base}{path_and_query}` with the admin bearer token.
    /// `path_and_query` starts with `/` and may carry a query string
    /// (e.g. `/v1/audit?by=time`).
    ///
    /// # Errors
    ///
    /// [`AdminError::NoSecret`] when no secret is configured;
    /// [`AdminError::Transport`] on a send/receive failure. A non-2xx status
    /// is returned as an [`AdminResponse`], not an error — the caller decides.
    pub async fn get(&self, path_and_query: &str) -> Result<AdminResponse, AdminError> {
        if self.secret.is_empty() {
            return Err(AdminError::NoSecret);
        }
        let url = format!("{}{path_and_query}", self.base);
        let resp = self
            .http
            .get(&url)
            .bearer_auth(self.secret.expose_secret())
            .send()
            .await
            .map_err(|source| AdminError::Transport {
                url: url.clone(),
                source,
            })?;
        Self::finish(url, resp).await
    }

    /// `POST {base}{path_and_query}` with the admin bearer token and an empty
    /// body (the rebuild routes take their arguments in the query string).
    ///
    /// # Errors
    ///
    /// As [`get`](Self::get).
    pub async fn post_empty(&self, path_and_query: &str) -> Result<AdminResponse, AdminError> {
        if self.secret.is_empty() {
            return Err(AdminError::NoSecret);
        }
        let url = format!("{}{path_and_query}", self.base);
        let resp = self
            .http
            .post(&url)
            .bearer_auth(self.secret.expose_secret())
            .header("content-length", "0")
            .send()
            .await
            .map_err(|source| AdminError::Transport {
                url: url.clone(),
                source,
            })?;
        Self::finish(url, resp).await
    }

    /// `DELETE {base}{path_and_query}` with the admin bearer token and no
    /// body (the backfill-cancel route takes the backfill id in the path).
    ///
    /// # Errors
    ///
    /// As [`get`](Self::get).
    pub async fn delete(&self, path_and_query: &str) -> Result<AdminResponse, AdminError> {
        if self.secret.is_empty() {
            return Err(AdminError::NoSecret);
        }
        let url = format!("{}{path_and_query}", self.base);
        let resp = self
            .http
            .delete(&url)
            .bearer_auth(self.secret.expose_secret())
            .send()
            .await
            .map_err(|source| AdminError::Transport {
                url: url.clone(),
                source,
            })?;
        Self::finish(url, resp).await
    }

    async fn finish(url: String, resp: reqwest::Response) -> Result<AdminResponse, AdminError> {
        let status = resp.status().as_u16();
        let body = resp
            .text()
            .await
            .map_err(|source| AdminError::Transport { url, source })?;
        Ok(AdminResponse { status, body })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_external_env_uses_endpoint_ip() {
        // No env set: defaults apply, secret empty ⇒ requests would NoSecret.
        let c = AdminClient::from_external_env("10.0.0.7:8080".parse().unwrap());
        assert_eq!(c.base, format!("http://10.0.0.7:{DEFAULT_ADMIN_PORT}"));
        assert!(!c.has_secret());
    }

    #[tokio::test]
    async fn request_without_secret_is_no_secret_error() {
        let c = AdminClient::new("http://127.0.0.1:9092".to_string(), String::new());
        assert!(matches!(
            c.get("/v1/audit?by=time").await,
            Err(AdminError::NoSecret)
        ));
        assert!(matches!(
            c.post_empty("/v1/rebuild-ann?shard=0").await,
            Err(AdminError::NoSecret)
        ));
        assert!(matches!(
            c.delete("/v1/backfill/deadbeef").await,
            Err(AdminError::NoSecret)
        ));
    }

    #[test]
    fn response_success_range() {
        assert!(AdminResponse {
            status: 201,
            body: String::new()
        }
        .is_success());
        assert!(!AdminResponse {
            status: 404,
            body: String::new()
        }
        .is_success());
    }
}
