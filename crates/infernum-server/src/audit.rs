//! Audit logging for security-relevant events.
//!
//! This module provides structured audit logging for authentication,
//! authorization, and key management events. Audit logs are critical
//! for security monitoring, incident response, and compliance.
//!
//! # Usage
//!
//! ```rust,ignore
//! use infernum_server::audit::{AuditLogger, AuditEventType};
//!
//! let logger = AuditLogger::new(true);
//!
//! // Log successful authentication
//! logger.auth_success("req-123", Some("192.168.1.1"), "sk-inf-abc", "/v1/chat/completions");
//!
//! // Log authentication failure
//! logger.auth_failure("req-456", Some("192.168.1.1"), "invalid_key", "/v1/chat/completions");
//! ```
//!
//! # Event Types
//!
//! | Event | Description |
//! |-------|-------------|
//! | `AuthSuccess` | Successful authentication |
//! | `AuthFailure` | Failed authentication (invalid key) |
//! | `AuthExpired` | Key has expired |
//! | `AuthDisabled` | Key is disabled |
//! | `ScopeViolation` | Key lacks required scope |
//! | `RateLimited` | Request was rate limited |
//! | `ModelLoad` | Model loaded (admin action) |
//! | `ModelUnload` | Model unloaded (admin action) |
//! | `KeyCreated` | New API key created |
//! | `KeyRevoked` | API key revoked |

use chrono::{DateTime, Utc};
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Audit event structure for logging.
#[derive(Debug, Clone, Serialize)]
pub struct AuditEvent {
    /// Timestamp of the event.
    pub timestamp: DateTime<Utc>,
    /// Type of audit event.
    pub event_type: AuditEventType,
    /// Request ID for correlation.
    pub request_id: String,
    /// Client IP address (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_ip: Option<String>,
    /// API key prefix (first 8 chars for identification).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key_prefix: Option<String>,
    /// Target endpoint.
    pub endpoint: String,
    /// Whether the operation succeeded.
    pub success: bool,
    /// Additional details about the event.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

/// Types of audit events.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AuditEventType {
    /// Successful authentication.
    AuthSuccess,
    /// Failed authentication (invalid key).
    AuthFailure,
    /// Key has expired.
    AuthExpired,
    /// Key is disabled.
    AuthDisabled,
    /// Key lacks required scope.
    ScopeViolation,
    /// Request was rate limited.
    RateLimited,
    /// Model loaded (admin action).
    ModelLoad,
    /// Model unloaded (admin action).
    ModelUnload,
    /// New API key created.
    KeyCreated,
    /// API key revoked.
    KeyRevoked,
}

impl AuditEventType {
    /// Returns the string name of the event type.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::AuthSuccess => "auth_success",
            Self::AuthFailure => "auth_failure",
            Self::AuthExpired => "auth_expired",
            Self::AuthDisabled => "auth_disabled",
            Self::ScopeViolation => "scope_violation",
            Self::RateLimited => "rate_limited",
            Self::ModelLoad => "model_load",
            Self::ModelUnload => "model_unload",
            Self::KeyCreated => "key_created",
            Self::KeyRevoked => "key_revoked",
        }
    }
}

/// Audit logger configuration.
#[derive(Debug, Clone)]
pub struct AuditConfig {
    /// Whether audit logging is enabled.
    pub enabled: bool,
    /// Log client IP addresses.
    pub log_client_ip: bool,
    /// Log key prefixes.
    pub log_key_prefix: bool,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_client_ip: true,
            log_key_prefix: true,
        }
    }
}

impl AuditConfig {
    /// Creates a new config with audit logging enabled.
    pub fn enabled() -> Self {
        Self::default()
    }

    /// Creates a new config with audit logging disabled.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }
}

/// Audit logger for security-relevant events.
///
/// This logger writes structured audit events to the tracing framework
/// with the target "audit" for easy filtering and routing.
#[derive(Debug, Clone)]
pub struct AuditLogger {
    config: Arc<RwLock<AuditConfig>>,
}

impl Default for AuditLogger {
    fn default() -> Self {
        Self::new(AuditConfig::default())
    }
}

impl AuditLogger {
    /// Creates a new audit logger with the given configuration.
    pub fn new(config: AuditConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
        }
    }

    /// Creates a new audit logger that is enabled.
    pub fn enabled() -> Self {
        Self::new(AuditConfig::enabled())
    }

    /// Creates a new audit logger that is disabled.
    pub fn disabled() -> Self {
        Self::new(AuditConfig::disabled())
    }

    /// Logs an audit event.
    ///
    /// Events are logged to the tracing framework with target "audit"
    /// for easy filtering and routing to audit-specific sinks.
    pub async fn log(&self, event: AuditEvent) {
        let config = self.config.read().await;
        if !config.enabled {
            return;
        }

        // Build the log output
        let event_json = serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_string());

        // Log with structured fields for easy querying
        tracing::info!(
            target: "audit",
            event_type = event.event_type.as_str(),
            request_id = %event.request_id,
            client_ip = ?event.client_ip,
            key_prefix = ?event.key_prefix,
            endpoint = %event.endpoint,
            success = event.success,
            details = ?event.details,
            "{}",
            event_json
        );
    }

    /// Logs a successful authentication event.
    pub async fn auth_success(
        &self,
        request_id: &str,
        client_ip: Option<&str>,
        key: &str,
        endpoint: &str,
    ) {
        self.log(AuditEvent {
            timestamp: Utc::now(),
            event_type: AuditEventType::AuthSuccess,
            request_id: request_id.to_string(),
            client_ip: self.maybe_client_ip(client_ip).await,
            key_prefix: self.maybe_key_prefix(key).await,
            endpoint: endpoint.to_string(),
            success: true,
            details: None,
        })
        .await;
    }

    /// Logs an authentication failure event.
    pub async fn auth_failure(
        &self,
        request_id: &str,
        client_ip: Option<&str>,
        reason: &str,
        endpoint: &str,
    ) {
        self.log(AuditEvent {
            timestamp: Utc::now(),
            event_type: AuditEventType::AuthFailure,
            request_id: request_id.to_string(),
            client_ip: self.maybe_client_ip(client_ip).await,
            key_prefix: None,
            endpoint: endpoint.to_string(),
            success: false,
            details: Some(reason.to_string()),
        })
        .await;
    }

    /// Logs an expired key authentication attempt.
    pub async fn auth_expired(
        &self,
        request_id: &str,
        client_ip: Option<&str>,
        key: &str,
        endpoint: &str,
    ) {
        self.log(AuditEvent {
            timestamp: Utc::now(),
            event_type: AuditEventType::AuthExpired,
            request_id: request_id.to_string(),
            client_ip: self.maybe_client_ip(client_ip).await,
            key_prefix: self.maybe_key_prefix(key).await,
            endpoint: endpoint.to_string(),
            success: false,
            details: Some("API key has expired".to_string()),
        })
        .await;
    }

    /// Logs a disabled key authentication attempt.
    pub async fn auth_disabled(
        &self,
        request_id: &str,
        client_ip: Option<&str>,
        key: &str,
        endpoint: &str,
    ) {
        self.log(AuditEvent {
            timestamp: Utc::now(),
            event_type: AuditEventType::AuthDisabled,
            request_id: request_id.to_string(),
            client_ip: self.maybe_client_ip(client_ip).await,
            key_prefix: self.maybe_key_prefix(key).await,
            endpoint: endpoint.to_string(),
            success: false,
            details: Some("API key is disabled".to_string()),
        })
        .await;
    }

    /// Logs a scope violation (insufficient permissions).
    pub async fn scope_violation(
        &self,
        request_id: &str,
        client_ip: Option<&str>,
        key: &str,
        endpoint: &str,
        required_scope: &str,
    ) {
        self.log(AuditEvent {
            timestamp: Utc::now(),
            event_type: AuditEventType::ScopeViolation,
            request_id: request_id.to_string(),
            client_ip: self.maybe_client_ip(client_ip).await,
            key_prefix: self.maybe_key_prefix(key).await,
            endpoint: endpoint.to_string(),
            success: false,
            details: Some(format!("Required scope: {required_scope}")),
        })
        .await;
    }

    /// Logs a rate limit event.
    pub async fn rate_limited(
        &self,
        request_id: &str,
        client_ip: Option<&str>,
        key: Option<&str>,
        endpoint: &str,
    ) {
        self.log(AuditEvent {
            timestamp: Utc::now(),
            event_type: AuditEventType::RateLimited,
            request_id: request_id.to_string(),
            client_ip: self.maybe_client_ip(client_ip).await,
            key_prefix: match key {
                Some(k) => self.maybe_key_prefix(k).await,
                None => None,
            },
            endpoint: endpoint.to_string(),
            success: false,
            details: Some("Request rate limit exceeded".to_string()),
        })
        .await;
    }

    /// Logs a model load event.
    pub async fn model_load(
        &self,
        request_id: &str,
        client_ip: Option<&str>,
        key: &str,
        model_id: &str,
        success: bool,
    ) {
        self.log(AuditEvent {
            timestamp: Utc::now(),
            event_type: AuditEventType::ModelLoad,
            request_id: request_id.to_string(),
            client_ip: self.maybe_client_ip(client_ip).await,
            key_prefix: self.maybe_key_prefix(key).await,
            endpoint: "/api/models/load".to_string(),
            success,
            details: Some(format!("Model: {model_id}")),
        })
        .await;
    }

    /// Logs a model unload event.
    pub async fn model_unload(
        &self,
        request_id: &str,
        client_ip: Option<&str>,
        key: &str,
        model_id: &str,
        success: bool,
    ) {
        self.log(AuditEvent {
            timestamp: Utc::now(),
            event_type: AuditEventType::ModelUnload,
            request_id: request_id.to_string(),
            client_ip: self.maybe_client_ip(client_ip).await,
            key_prefix: self.maybe_key_prefix(key).await,
            endpoint: "/api/models/unload".to_string(),
            success,
            details: Some(format!("Model: {model_id}")),
        })
        .await;
    }

    /// Logs a key creation event.
    pub async fn key_created(
        &self,
        request_id: &str,
        client_ip: Option<&str>,
        admin_key: &str,
        new_key_prefix: &str,
    ) {
        self.log(AuditEvent {
            timestamp: Utc::now(),
            event_type: AuditEventType::KeyCreated,
            request_id: request_id.to_string(),
            client_ip: self.maybe_client_ip(client_ip).await,
            key_prefix: self.maybe_key_prefix(admin_key).await,
            endpoint: "/api/keys".to_string(),
            success: true,
            details: Some(format!("Created key: {new_key_prefix}...")),
        })
        .await;
    }

    /// Logs a key revocation event.
    pub async fn key_revoked(
        &self,
        request_id: &str,
        client_ip: Option<&str>,
        admin_key: &str,
        revoked_key_prefix: &str,
    ) {
        self.log(AuditEvent {
            timestamp: Utc::now(),
            event_type: AuditEventType::KeyRevoked,
            request_id: request_id.to_string(),
            client_ip: self.maybe_client_ip(client_ip).await,
            key_prefix: self.maybe_key_prefix(admin_key).await,
            endpoint: "/api/keys".to_string(),
            success: true,
            details: Some(format!("Revoked key: {revoked_key_prefix}...")),
        })
        .await;
    }

    /// Returns the key prefix if logging is enabled.
    async fn maybe_key_prefix(&self, key: &str) -> Option<String> {
        let config = self.config.read().await;
        if config.log_key_prefix {
            let end = std::cmp::min(8, key.len());
            Some(key[..end].to_string())
        } else {
            None
        }
    }

    /// Returns the client IP if logging is enabled.
    async fn maybe_client_ip(&self, ip: Option<&str>) -> Option<String> {
        let config = self.config.read().await;
        if config.log_client_ip {
            ip.map(String::from)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_event_type_as_str() {
        assert_eq!(AuditEventType::AuthSuccess.as_str(), "auth_success");
        assert_eq!(AuditEventType::AuthFailure.as_str(), "auth_failure");
        assert_eq!(AuditEventType::AuthExpired.as_str(), "auth_expired");
        assert_eq!(AuditEventType::AuthDisabled.as_str(), "auth_disabled");
        assert_eq!(AuditEventType::ScopeViolation.as_str(), "scope_violation");
        assert_eq!(AuditEventType::RateLimited.as_str(), "rate_limited");
        assert_eq!(AuditEventType::ModelLoad.as_str(), "model_load");
        assert_eq!(AuditEventType::ModelUnload.as_str(), "model_unload");
        assert_eq!(AuditEventType::KeyCreated.as_str(), "key_created");
        assert_eq!(AuditEventType::KeyRevoked.as_str(), "key_revoked");
    }

    #[test]
    fn test_audit_config_default() {
        let config = AuditConfig::default();
        assert!(config.enabled);
        assert!(config.log_client_ip);
        assert!(config.log_key_prefix);
    }

    #[test]
    fn test_audit_config_disabled() {
        let config = AuditConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_audit_event_serialization() {
        let event = AuditEvent {
            timestamp: Utc::now(),
            event_type: AuditEventType::AuthSuccess,
            request_id: "req-123".to_string(),
            client_ip: Some("192.168.1.1".to_string()),
            key_prefix: Some("sk-inf-a".to_string()),
            endpoint: "/v1/chat/completions".to_string(),
            success: true,
            details: None,
        };

        let json = serde_json::to_string(&event).expect("Should serialize");
        assert!(json.contains("auth_success"));
        assert!(json.contains("req-123"));
        assert!(json.contains("192.168.1.1"));
        assert!(json.contains("sk-inf-a"));
    }

    #[tokio::test]
    async fn test_audit_logger_disabled() {
        let logger = AuditLogger::disabled();
        // Should not panic even when disabled
        logger
            .auth_success("req-1", Some("127.0.0.1"), "sk-test", "/test")
            .await;
        logger
            .auth_failure("req-2", Some("127.0.0.1"), "bad key", "/test")
            .await;
    }

    #[tokio::test]
    async fn test_audit_logger_key_prefix() {
        let logger = AuditLogger::enabled();
        let prefix = logger.maybe_key_prefix("sk-inf-abcdefghijklmnop").await;
        assert_eq!(prefix, Some("sk-inf-a".to_string()));
    }

    #[tokio::test]
    async fn test_audit_logger_short_key() {
        let logger = AuditLogger::enabled();
        let prefix = logger.maybe_key_prefix("short").await;
        assert_eq!(prefix, Some("short".to_string()));
    }
}
