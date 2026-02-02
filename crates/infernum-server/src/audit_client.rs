//! # Moloch Audit Client
//!
//! HTTP client for submitting audit events to the Moloch audit chain.
//!
//! ## Features
//!
//! - **Event Submission**: Submit signed `AuditEvent` to Moloch REST API
//! - **Retry Logic**: Automatic retry with exponential backoff
//! - **Batch Submission**: Submit multiple events efficiently
//! - **HoloCrypt Encryption**: Selective field encryption for sensitive data
//!
//! ## Architecture
//!
//! ```text
//! Agent signs tool execution
//!            │
//!            ▼
//! ┌─────────────────────┐
//! │   AuditClient       │
//! │                     │
//! │  ┌───────────────┐  │
//! │  │ HTTP Client   │──┼──► POST /v1/events
//! │  │ (reqwest)     │  │
//! │  └───────────────┘  │
//! │                     │
//! │  ┌───────────────┐  │
//! │  │ HoloCrypt     │  │  (optional encryption)
//! │  │ - Selective   │  │
//! │  │ - ZK proofs   │  │
//! │  │ - PQ security │  │
//! │  └───────────────┘  │
//! └─────────────────────┘
//! ```
//!
//! ## Example
//!
//! ```ignore
//! use infernum_server::audit_client::{AuditClient, AuditClientConfig};
//! use infernum_server::agent_identity::AgentIdentity;
//!
//! let client = AuditClient::new(AuditClientConfig {
//!     endpoint: "http://localhost:8090".into(),
//!     ..Default::default()
//! });
//!
//! let identity = AgentIdentity::generate("my-agent");
//! let event = identity.sign_tool_execution("read_file", &json!({"path": "/tmp/test"}))?;
//!
//! client.submit(event).await?;
//! ```
//!
//! ## HoloCrypt Integration
//!
//! HoloCrypt enables selective field encryption for sensitive prompts:
//!
//! ```ignore
//! use infernum_server::audit_client::{
//!     EncryptedEventBuilder, EncryptionPolicy, FieldVisibility,
//!     generate_encryption_keypair, policies,
//! };
//!
//! // Generate encryption keypair
//! let (sealing_key, opening_key) = generate_encryption_keypair("agent-1-keys");
//!
//! // Create encrypted event with tool execution policy
//! let encrypted = EncryptedEventBuilder::new(event)
//!     .with_policy(policies::tool_execution())
//!     .build(&sealing_key)?;
//!
//! // Submit encrypted event
//! client.submit_encrypted(encrypted).await?;
//! ```
//!
//! Encryption policies control field visibility:
//! - `FieldVisibility::Public` - Visible to all
//! - `FieldVisibility::Encrypted` - Encrypted, selectively disclosable
//! - `FieldVisibility::Private` - ZK-provable only, never disclosed

use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info, warn};

use moloch_core::event::AuditEvent;

// Re-export HoloCrypt types for encryption
pub use moloch_holocrypt::{
    EncryptedEvent, EncryptedEventBuilder, EncryptionPolicy, FieldVisibility,
    EventSealingKey, EventOpeningKey, generate_keypair as generate_encryption_keypair,
};

/// Preset encryption policies for common use cases.
pub mod policies {
    use super::{EncryptionPolicy, FieldVisibility};

    /// Policy for tool execution events.
    ///
    /// - Event type and outcome are public (for filtering/analytics)
    /// - Actor is encrypted (agent identity protected)
    /// - Resource and metadata are private (tool args may contain secrets)
    pub fn tool_execution() -> EncryptionPolicy {
        EncryptionPolicy {
            event_type: FieldVisibility::Public,
            actor: FieldVisibility::Encrypted,
            resource: FieldVisibility::Encrypted,
            outcome: FieldVisibility::Public,
            metadata: FieldVisibility::Private,
            timestamp: FieldVisibility::Public,
            key_id: None,
        }
    }

    /// Policy for key lifecycle events (rotation, revocation).
    ///
    /// - Event type and timestamp are public (for audit trail)
    /// - Actor and resource are encrypted (key identities protected)
    /// - Metadata is private (rotation reasons may be sensitive)
    pub fn key_lifecycle() -> EncryptionPolicy {
        EncryptionPolicy {
            event_type: FieldVisibility::Public,
            actor: FieldVisibility::Encrypted,
            resource: FieldVisibility::Encrypted,
            outcome: FieldVisibility::Public,
            metadata: FieldVisibility::Private,
            timestamp: FieldVisibility::Public,
            key_id: None,
        }
    }

    /// Policy for user prompt events.
    ///
    /// - Only timestamp and outcome are public
    /// - Everything else is private (prompts may contain PII)
    pub fn user_prompt() -> EncryptionPolicy {
        EncryptionPolicy {
            event_type: FieldVisibility::Encrypted,
            actor: FieldVisibility::Private,
            resource: FieldVisibility::Private,
            outcome: FieldVisibility::Public,
            metadata: FieldVisibility::Private,
            timestamp: FieldVisibility::Public,
            key_id: None,
        }
    }

    /// Policy for maximum privacy (ZK-provable only).
    ///
    /// All fields are private - only zero-knowledge proofs can
    /// verify event properties.
    pub fn maximum_privacy() -> EncryptionPolicy {
        EncryptionPolicy::all_private()
    }

    /// Policy for full transparency (no encryption).
    ///
    /// All fields are public - useful for non-sensitive events
    /// or when audit transparency is required.
    pub fn transparent() -> EncryptionPolicy {
        EncryptionPolicy::all_public()
    }
}

/// Errors that can occur when interacting with the audit chain.
#[derive(Debug, Error)]
pub enum AuditClientError {
    /// HTTP request failed.
    #[error("HTTP error: {0}")]
    Http(String),

    /// Server returned an error response.
    #[error("server error: {status} - {message}")]
    ServerError {
        /// HTTP status code.
        status: u16,
        /// Error message from server.
        message: String,
    },

    /// Event was rejected by the server.
    #[error("event rejected: {0}")]
    Rejected(String),

    /// Serialization failed.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Connection failed.
    #[error("connection failed: {0}")]
    Connection(String),

    /// Timeout waiting for response.
    #[error("request timed out")]
    Timeout,
}

/// Result type for audit client operations.
pub type Result<T> = std::result::Result<T, AuditClientError>;

/// Configuration for the audit client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditClientConfig {
    /// Moloch API endpoint (e.g., "http://localhost:8090").
    pub endpoint: String,
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Maximum retry attempts.
    pub max_retries: u32,
    /// Initial retry delay in milliseconds.
    pub retry_delay_ms: u64,
}

impl Default for AuditClientConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:8090".to_string(),
            timeout_ms: 30_000,
            max_retries: 3,
            retry_delay_ms: 100,
        }
    }
}

impl AuditClientConfig {
    /// Create a new config with the given endpoint.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            ..Default::default()
        }
    }

    /// Set request timeout.
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Set retry configuration.
    pub fn with_retries(mut self, max_retries: u32, retry_delay_ms: u64) -> Self {
        self.max_retries = max_retries;
        self.retry_delay_ms = retry_delay_ms;
        self
    }
}

/// Request to submit an event to Moloch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitEventRequest {
    /// The signed audit event.
    pub event: AuditEvent,
}

/// Response from event submission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitEventResponse {
    /// Event ID assigned by Moloch.
    pub id: String,
    /// Whether the event was accepted.
    pub accepted: bool,
    /// Optional message (e.g., rejection reason).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// HTTP client for the Moloch audit chain.
pub struct AuditClient {
    config: AuditClientConfig,
    http_client: reqwest::Client,
}

impl AuditClient {
    /// Create a new audit client with the given configuration.
    pub fn new(config: AuditClientConfig) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms))
            .build()
            .expect("failed to build HTTP client");

        Self {
            config,
            http_client,
        }
    }

    /// Submit a signed audit event to Moloch.
    pub async fn submit(&self, event: AuditEvent) -> Result<SubmitEventResponse> {
        // Validate event signature before submission
        event.validate()
            .map_err(|e| AuditClientError::Rejected(format!("invalid event signature: {e}")))?;

        self.submit_with_retry(event).await
    }

    /// Submit multiple events in sequence.
    ///
    /// Returns results for each event (in order).
    pub async fn submit_batch(&self, events: Vec<AuditEvent>) -> Vec<Result<SubmitEventResponse>> {
        let mut results = Vec::with_capacity(events.len());

        for event in events {
            results.push(self.submit(event).await);
        }

        results
    }

    /// Submit an encrypted audit event to Moloch.
    ///
    /// The encrypted event contains a HoloCrypt container with selective
    /// field encryption based on the policy used during encryption.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use infernum_server::audit_client::{
    ///     AuditClient, EncryptedEventBuilder, generate_encryption_keypair, policies,
    /// };
    ///
    /// let (sealing_key, _) = generate_encryption_keypair("my-keys");
    /// let encrypted = EncryptedEventBuilder::new(event)
    ///     .with_policy(policies::tool_execution())
    ///     .build(&sealing_key)?;
    ///
    /// client.submit_encrypted(encrypted).await?;
    /// ```
    pub async fn submit_encrypted(&self, event: EncryptedEvent) -> Result<SubmitEventResponse> {
        let url = format!("{}/v1/events/encrypted", self.config.endpoint);

        debug!(url = %url, "submitting encrypted event");

        let response = self.http_client
            .post(&url)
            .json(&event)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    AuditClientError::Timeout
                } else if e.is_connect() {
                    AuditClientError::Connection(e.to_string())
                } else {
                    AuditClientError::Http(e.to_string())
                }
            })?;

        let status = response.status();

        if status.is_success() {
            response
                .json::<SubmitEventResponse>()
                .await
                .map_err(|e| AuditClientError::Serialization(e.to_string()))
        } else {
            let body = response.text().await.unwrap_or_default();
            Err(AuditClientError::ServerError {
                status: status.as_u16(),
                message: body,
            })
        }
    }

    /// Submit with retry logic.
    async fn submit_with_retry(&self, event: AuditEvent) -> Result<SubmitEventResponse> {
        let mut last_error = None;
        let mut delay = Duration::from_millis(self.config.retry_delay_ms);

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                warn!(attempt, "retrying event submission after delay");
                tokio::time::sleep(delay).await;
                delay *= 2; // Exponential backoff
            }

            match self.submit_once(&event).await {
                Ok(response) => {
                    if response.accepted {
                        info!(event_id = %response.id, "event submitted successfully");
                        return Ok(response);
                    } else {
                        // Event was rejected - don't retry
                        return Err(AuditClientError::Rejected(
                            response.message.unwrap_or_else(|| "unknown reason".to_string())
                        ));
                    }
                }
                Err(e) => {
                    // Only retry on transient errors
                    if matches!(e, AuditClientError::Connection(_) | AuditClientError::Timeout) {
                        last_error = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Err(last_error.unwrap_or(AuditClientError::Timeout))
    }

    /// Submit a single event (no retry).
    async fn submit_once(&self, event: &AuditEvent) -> Result<SubmitEventResponse> {
        let url = format!("{}/v1/events", self.config.endpoint);

        let request = SubmitEventRequest {
            event: event.clone(),
        };

        debug!(url = %url, event_id = %hex::encode(event.id().0.as_bytes()), "submitting event");

        let response = self.http_client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    AuditClientError::Timeout
                } else if e.is_connect() {
                    AuditClientError::Connection(e.to_string())
                } else {
                    AuditClientError::Http(e.to_string())
                }
            })?;

        let status = response.status();

        if status.is_success() {
            response
                .json::<SubmitEventResponse>()
                .await
                .map_err(|e| AuditClientError::Serialization(e.to_string()))
        } else {
            let body = response.text().await.unwrap_or_default();
            Err(AuditClientError::ServerError {
                status: status.as_u16(),
                message: body,
            })
        }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &AuditClientConfig {
        &self.config
    }
}

impl std::fmt::Debug for AuditClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuditClient")
            .field("endpoint", &self.config.endpoint)
            .field("timeout_ms", &self.config.timeout_ms)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AuditClientConfig::default();
        assert_eq!(config.endpoint, "http://localhost:8090");
        assert_eq!(config.timeout_ms, 30_000);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_config_builder() {
        let config = AuditClientConfig::new("http://moloch:8090")
            .with_timeout(5_000)
            .with_retries(5, 200);

        assert_eq!(config.endpoint, "http://moloch:8090");
        assert_eq!(config.timeout_ms, 5_000);
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.retry_delay_ms, 200);
    }

    #[test]
    fn test_client_creation() {
        let config = AuditClientConfig::default();
        let client = AuditClient::new(config);

        assert_eq!(client.config().endpoint, "http://localhost:8090");
    }

    #[test]
    fn test_debug_output() {
        let config = AuditClientConfig::default();
        let client = AuditClient::new(config);
        let debug_str = format!("{:?}", client);

        assert!(debug_str.contains("AuditClient"));
        assert!(debug_str.contains("localhost:8090"));
    }
}
