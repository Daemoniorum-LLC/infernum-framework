//! Authentication and authorization middleware.
//!
//! This module provides comprehensive authentication and authorization for the
//! inference server, including:
//!
//! - **API Key Authentication**: Bearer token and X-API-Key header support
//! - **Permission Levels**: User and Admin permission tiers
//! - **Scope-Based Authorization**: Fine-grained access control per endpoint
//! - **Key Management**: Runtime key addition/removal and secure hashing
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use infernum_server::auth::{AuthConfig, ApiKey, AuthState};
//!
//! // Create auth configuration
//! let config = AuthConfig::enabled()
//!     .add_key(ApiKey::user("sk-inf-user123"))
//!     .add_key(ApiKey::admin("sk-adm-admin456"));
//!
//! // Create shared state
//! let auth_state = AuthState::new(config);
//!
//! // Validate a key
//! if let Some(permission) = auth_state.validate_key("sk-inf-user123").await {
//!     println!("Key has {:?} permission", permission);
//! }
//! ```
//!
//! # Scope-Based Authorization
//!
//! API keys can be created with specific scopes using the format `sk-{scope}-{random}`:
//!
//! | Scope | Abbreviation | Endpoints |
//! |-------|--------------|-----------|
//! | Inference | `inf` | `/v1/chat/completions`, `/v1/embeddings`, `/v1/models` |
//! | Admin | `adm` | `/api/models/load`, `/api/models/unload`, `/api/keys` |
//! | Metrics | `met` | `/metrics` |
//!
//! ```rust,ignore
//! use infernum_server::auth::{ApiKey, Scope};
//!
//! // Create a key with specific scopes
//! let key = ApiKey::with_scopes("sk-inf-abc123", vec![Scope::Inference]);
//!
//! // Parse scope from key format
//! let scope = ApiKey::parse_scope_from_key("sk-inf-abc123");
//! assert_eq!(scope, Some(Scope::Inference));
//! ```
//!
//! # Security Considerations
//!
//! - Keys should be stored hashed, not in plaintext
//! - Use [`ApiKey::hash_key`] and [`ApiKey::verify_key`] for secure storage
//! - All header parsing uses safe string operations (no `unwrap()` calls)
//! - Rate limiting should be combined with authentication for DoS protection

use crate::audit::{AuditConfig, AuditLogger};
use axum::{
    extract::Request,
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

// ============================================================================
// Authentication Metrics (Sprint 6 Day 19.6)
// ============================================================================

/// Metrics for authentication events.
///
/// Tracks authentication failures for Prometheus monitoring.
#[derive(Debug, Default)]
pub struct AuthMetrics {
    /// Total authentication failures (missing key, invalid key, expired, disabled).
    pub failures_total: AtomicU64,
    /// Failures due to missing API key.
    pub failures_missing_key: AtomicU64,
    /// Failures due to invalid API key.
    pub failures_invalid_key: AtomicU64,
    /// Failures due to expired API key.
    pub failures_expired_key: AtomicU64,
    /// Failures due to disabled API key.
    pub failures_disabled_key: AtomicU64,
    /// Failures due to insufficient permissions.
    pub failures_insufficient_scope: AtomicU64,
}

impl AuthMetrics {
    /// Creates a new metrics instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records an authentication failure.
    pub fn record_failure(&self, reason: AuthFailureReason) {
        self.failures_total.fetch_add(1, Ordering::Relaxed);
        match reason {
            AuthFailureReason::MissingKey => {
                self.failures_missing_key.fetch_add(1, Ordering::Relaxed);
            }
            AuthFailureReason::InvalidKey => {
                self.failures_invalid_key.fetch_add(1, Ordering::Relaxed);
            }
            AuthFailureReason::ExpiredKey => {
                self.failures_expired_key.fetch_add(1, Ordering::Relaxed);
            }
            AuthFailureReason::DisabledKey => {
                self.failures_disabled_key.fetch_add(1, Ordering::Relaxed);
            }
            AuthFailureReason::InsufficientScope => {
                self.failures_insufficient_scope.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Returns the total number of authentication failures.
    pub fn total_failures(&self) -> u64 {
        self.failures_total.load(Ordering::Relaxed)
    }

    /// Renders metrics in Prometheus text exposition format.
    pub fn render_prometheus(&self) -> String {
        let mut output = String::new();

        output.push_str("# HELP infernum_auth_failures_total Total authentication failures.\n");
        output.push_str("# TYPE infernum_auth_failures_total counter\n");
        output.push_str(&format!(
            "infernum_auth_failures_total{{reason=\"missing_key\"}} {}\n",
            self.failures_missing_key.load(Ordering::Relaxed)
        ));
        output.push_str(&format!(
            "infernum_auth_failures_total{{reason=\"invalid_key\"}} {}\n",
            self.failures_invalid_key.load(Ordering::Relaxed)
        ));
        output.push_str(&format!(
            "infernum_auth_failures_total{{reason=\"expired_key\"}} {}\n",
            self.failures_expired_key.load(Ordering::Relaxed)
        ));
        output.push_str(&format!(
            "infernum_auth_failures_total{{reason=\"disabled_key\"}} {}\n",
            self.failures_disabled_key.load(Ordering::Relaxed)
        ));
        output.push_str(&format!(
            "infernum_auth_failures_total{{reason=\"insufficient_scope\"}} {}\n",
            self.failures_insufficient_scope.load(Ordering::Relaxed)
        ));

        output
    }
}

/// Reason for authentication failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthFailureReason {
    /// No API key provided.
    MissingKey,
    /// API key not found.
    InvalidKey,
    /// API key has expired.
    ExpiredKey,
    /// API key is disabled.
    DisabledKey,
    /// API key lacks required scope.
    InsufficientScope,
}

/// Permission level for API access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    /// Read-only access (inference, embeddings, model info).
    User,
    /// Full access including model management.
    Admin,
}

impl Default for Permission {
    fn default() -> Self {
        Self::User
    }
}

// ============================================================================
// Scope-Based Authorization (Phase 3)
// ============================================================================

/// API key scope for fine-grained access control.
///
/// Scopes determine which endpoints an API key can access.
/// The key format is `sk-{scope}-{random}` where scope is a 3-letter prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Scope {
    /// Inference scope: chat completions, embeddings, model info.
    Inference,
    /// Admin scope: model loading, configuration, key management.
    Admin,
    /// Metrics scope: access to /metrics endpoint.
    Metrics,
}

impl Scope {
    /// Returns the string representation of the scope.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Inference => "inference",
            Self::Admin => "admin",
            Self::Metrics => "metrics",
        }
    }

    /// Parses a scope from a string (supports both full names and abbreviations).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "inference" | "inf" => Some(Self::Inference),
            "admin" | "adm" => Some(Self::Admin),
            "metrics" | "met" => Some(Self::Metrics),
            _ => None,
        }
    }
}

/// Returns the required scope for a given path, or None for public paths.
pub fn required_scope_for_path(path: &str) -> Option<Scope> {
    // Public paths
    if path == "/health" || path == "/ready" {
        return None;
    }

    // Admin-only paths
    if path.starts_with("/api/models/load")
        || path.starts_with("/api/models/unload")
        || path.starts_with("/api/keys")
        || path.starts_with("/api/config")
        || path.starts_with("/admin/models")
    {
        return Some(Scope::Admin);
    }

    // Inference paths
    if path.starts_with("/v1/chat")
        || path.starts_with("/v1/completions")
        || path.starts_with("/v1/embeddings")
        || path.starts_with("/v1/models")
    {
        return Some(Scope::Inference);
    }

    // Metrics path
    if path.starts_with("/metrics") {
        return Some(Scope::Metrics);
    }

    // Default to inference for unknown API paths
    if path.starts_with("/v1/") || path.starts_with("/api/") {
        return Some(Scope::Inference);
    }

    None
}

/// API key configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApiKey {
    /// The API key value (plaintext for validation, should be hashed for storage).
    pub key: String,
    /// Hashed version of the key (if stored hashed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key_hash: Option<String>,
    /// Permission level for this key (legacy, use scopes for new keys).
    pub permission: Permission,
    /// Scopes this key has access to.
    #[serde(default)]
    pub scopes: Vec<Scope>,
    /// Optional name for identification.
    pub name: Option<String>,
    /// Whether the key is enabled.
    pub enabled: bool,
    /// Expiration time (ISO 8601 or duration like "30d").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Creation time.
    #[serde(default = "chrono::Utc::now")]
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl ApiKey {
    /// Creates a new API key with user permissions (inference scope only).
    pub fn user(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            key_hash: None,
            permission: Permission::User,
            scopes: vec![Scope::Inference],
            name: None,
            enabled: true,
            expires_at: None,
            created_at: chrono::Utc::now(),
        }
    }

    /// Creates a new API key with admin permissions (all scopes).
    pub fn admin(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            key_hash: None,
            permission: Permission::Admin,
            scopes: vec![Scope::Inference, Scope::Admin, Scope::Metrics],
            name: None,
            enabled: true,
            expires_at: None,
            created_at: chrono::Utc::now(),
        }
    }

    /// Creates a new API key with specific scopes.
    pub fn with_scopes(key: impl Into<String>, scopes: Vec<Scope>) -> Self {
        let permission = if scopes.contains(&Scope::Admin) {
            Permission::Admin
        } else {
            Permission::User
        };
        Self {
            key: key.into(),
            key_hash: None,
            permission,
            scopes,
            name: None,
            enabled: true,
            expires_at: None,
            created_at: chrono::Utc::now(),
        }
    }

    /// Sets the expiration time for this key.
    pub fn with_expiration(mut self, expires_at: chrono::DateTime<chrono::Utc>) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// Sets the expiration time from a duration string (e.g., "30d", "24h").
    pub fn with_expiration_duration(mut self, duration: &str) -> Self {
        self.expires_at = Self::parse_duration_to_expiry(duration);
        self
    }

    /// Parses a duration string to an expiry timestamp.
    fn parse_duration_to_expiry(duration: &str) -> Option<chrono::DateTime<chrono::Utc>> {
        use chrono::Duration;

        if let Some(days) = duration.strip_suffix('d') {
            if let Ok(d) = days.parse::<i64>() {
                return Some(chrono::Utc::now() + Duration::days(d));
            }
        }
        if let Some(hours) = duration.strip_suffix('h') {
            if let Ok(h) = hours.parse::<i64>() {
                return Some(chrono::Utc::now() + Duration::hours(h));
            }
        }
        if let Some(mins) = duration.strip_suffix('m') {
            if let Ok(m) = mins.parse::<i64>() {
                return Some(chrono::Utc::now() + Duration::minutes(m));
            }
        }
        // Try ISO 8601 parsing
        chrono::DateTime::parse_from_rfc3339(duration)
            .ok()
            .map(|dt| dt.with_timezone(&chrono::Utc))
    }

    /// Checks if this key has expired.
    pub fn is_expired(&self) -> bool {
        if let Some(expires) = self.expires_at {
            chrono::Utc::now() > expires
        } else {
            false
        }
    }

    /// Parses the scope from a key in the format `sk-{scope}-{random}`.
    ///
    /// Returns the scope if the key matches the format, otherwise None.
    pub fn parse_scope_from_key(key: &str) -> Option<Scope> {
        // Format: sk-{scope_abbrev}-{random}
        // e.g., sk-inf-abc123, sk-adm-xyz789, sk-met-qrs456
        if !key.starts_with("sk-") {
            return None;
        }

        let parts: Vec<&str> = key.split('-').collect();
        if parts.len() < 3 {
            return None;
        }

        Scope::from_str(parts[1])
    }

    /// Generates a SHA-256 hash of the API key for secure storage.
    ///
    /// Returns a hex-encoded hash string with "sha256:" prefix.
    ///
    /// # Security
    ///
    /// This method uses cryptographic SHA-256 hashing, which is suitable
    /// for API key storage. For password hashing, consider Argon2 instead.
    pub fn hash_key(&self) -> String {
        Self::hash_key_sha256(&self.key)
    }

    /// Hashes an API key using SHA-256.
    ///
    /// Returns a hex-encoded hash string with "sha256:" prefix.
    #[must_use]
    pub fn hash_key_sha256(key: &str) -> String {
        use sha2::{Sha256, Digest};

        let mut hasher = Sha256::new();
        hasher.update(key.as_bytes());
        let result = hasher.finalize();
        format!("sha256:{}", hex::encode(result))
    }

    /// Verifies a plaintext key against a hash using constant-time comparison.
    ///
    /// # Security
    ///
    /// This method uses constant-time comparison to prevent timing attacks.
    /// It automatically detects the hash algorithm from the prefix.
    pub fn verify_key(plaintext: &str, hash: &str) -> bool {
        use subtle::ConstantTimeEq;

        if let Some(sha_hash) = hash.strip_prefix("sha256:") {
            // SHA-256 verification with constant-time comparison
            let computed = Self::hash_key_sha256(plaintext);
            if let Some(computed_hash) = computed.strip_prefix("sha256:") {
                // Convert to bytes for constant-time comparison
                let expected = hex::decode(sha_hash).unwrap_or_default();
                let actual = hex::decode(computed_hash).unwrap_or_default();

                if expected.len() != actual.len() {
                    return false;
                }

                expected.ct_eq(&actual).into()
            } else {
                false
            }
        } else {
            // Legacy format: compare directly with constant-time comparison
            // This supports old keys that weren't hashed with SHA-256
            tracing::warn!("Verifying key against non-prefixed hash (legacy format)");
            let a = plaintext.as_bytes();
            let b = hash.as_bytes();
            if a.len() != b.len() {
                return false;
            }
            a.ct_eq(b).into()
        }
    }

    /// Creates a hashed version of this key for secure storage.
    ///
    /// Returns a new `ApiKey` with the `key_hash` field set and the
    /// plaintext key replaced with "[HASHED]".
    #[must_use]
    pub fn hashed(mut self) -> Self {
        self.key_hash = Some(self.hash_key());
        self.key = "[HASHED]".to_string();
        self
    }

    /// Gets the key prefix (first 8 characters) for logging without exposing the full key.
    #[must_use]
    pub fn key_prefix(&self) -> &str {
        let end = std::cmp::min(8, self.key.len());
        &self.key[..end]
    }

    /// Sets the name for this key.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Checks if this key has the specified scope.
    pub fn has_scope(&self, scope: Scope) -> bool {
        self.scopes.contains(&scope)
    }
}

/// Authentication configuration.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AuthConfig {
    /// Whether authentication is enabled.
    pub enabled: bool,
    /// API keys (key value -> ApiKey config).
    pub api_keys: HashMap<String, ApiKey>,
    /// Paths that don't require authentication.
    pub public_paths: Vec<String>,
}

impl AuthConfig {
    /// Creates a new auth config with authentication disabled.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Creates a new auth config with authentication enabled.
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            public_paths: vec![
                "/health".to_string(),
                "/ready".to_string(),
                "/metrics".to_string(),
            ],
            ..Default::default()
        }
    }

    /// Adds an API key.
    pub fn add_key(mut self, api_key: ApiKey) -> Self {
        self.api_keys.insert(api_key.key.clone(), api_key);
        self
    }

    /// Adds a public path that doesn't require authentication.
    pub fn add_public_path(mut self, path: impl Into<String>) -> Self {
        self.public_paths.push(path.into());
        self
    }

    /// Creates an auth config from environment variable.
    ///
    /// Reads `INFERNUM_API_KEYS` as comma-separated key:permission pairs.
    /// Example: `sk-key1:admin,sk-key2:user`
    pub fn from_env() -> Self {
        let mut config = Self::enabled();

        if let Ok(keys_str) = std::env::var("INFERNUM_API_KEYS") {
            for pair in keys_str.split(',') {
                let parts: Vec<&str> = pair.trim().split(':').collect();
                match parts.as_slice() {
                    [key, "admin"] => {
                        config = config.add_key(ApiKey::admin(*key));
                    }
                    [key, "user"] | [key] => {
                        config = config.add_key(ApiKey::user(*key));
                    }
                    _ => {
                        tracing::warn!("Invalid API key format: {}", pair);
                    }
                }
            }
        }

        // If no keys configured but auth is expected, check for single key
        if config.api_keys.is_empty() {
            if let Ok(key) = std::env::var("INFERNUM_API_KEY") {
                config = config.add_key(ApiKey::admin(key));
            }
        }

        // If still no keys, disable auth
        if config.api_keys.is_empty() {
            tracing::warn!("No API keys configured, authentication disabled");
            config.enabled = false;
        }

        config
    }

    /// Returns whether authentication is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Returns the number of configured API keys.
    pub fn key_count(&self) -> usize {
        self.api_keys.len()
    }
}

/// Result of key validation with detailed failure reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationResult {
    /// Key is valid with the given permission.
    Valid(Permission),
    /// Key is not found.
    NotFound,
    /// Key is disabled.
    Disabled,
    /// Key has expired.
    Expired,
}

/// Shared authentication state.
#[derive(Debug, Clone)]
pub struct AuthState {
    config: Arc<RwLock<AuthConfig>>,
    audit_logger: AuditLogger,
    metrics: Arc<AuthMetrics>,
}

impl AuthState {
    /// Creates a new auth state with the given config.
    pub fn new(config: AuthConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            audit_logger: AuditLogger::new(AuditConfig::default()),
            metrics: Arc::new(AuthMetrics::new()),
        }
    }

    /// Creates a new auth state with custom audit logging configuration.
    pub fn with_audit_config(config: AuthConfig, audit_config: AuditConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            audit_logger: AuditLogger::new(audit_config),
            metrics: Arc::new(AuthMetrics::new()),
        }
    }

    /// Returns a reference to the audit logger.
    pub fn audit_logger(&self) -> &AuditLogger {
        &self.audit_logger
    }

    /// Returns a reference to the metrics.
    pub fn metrics(&self) -> &AuthMetrics {
        &self.metrics
    }

    /// Checks if a path is public (doesn't require auth).
    pub async fn is_public_path(&self, path: &str) -> bool {
        let config = self.config.read().await;
        if !config.enabled {
            return true;
        }
        config.public_paths.iter().any(|p| path.starts_with(p))
    }

    /// Validates an API key and returns the permission level.
    ///
    /// Returns `None` if the key is invalid, disabled, or expired.
    pub async fn validate_key(&self, key: &str) -> Option<Permission> {
        match self.validate_key_detailed(key).await {
            ValidationResult::Valid(permission) => Some(permission),
            _ => None,
        }
    }

    /// Validates an API key with detailed failure reason.
    ///
    /// Returns a `ValidationResult` indicating success or the specific
    /// reason for failure (not found, disabled, or expired).
    pub async fn validate_key_detailed(&self, key: &str) -> ValidationResult {
        let config = self.config.read().await;

        match config.api_keys.get(key) {
            None => ValidationResult::NotFound,
            Some(api_key) => {
                // Check if key is enabled
                if !api_key.enabled {
                    tracing::debug!(key_prefix = api_key.key_prefix(), "Key is disabled");
                    return ValidationResult::Disabled;
                }

                // Check if key has expired
                if api_key.is_expired() {
                    tracing::debug!(
                        key_prefix = api_key.key_prefix(),
                        expires_at = ?api_key.expires_at,
                        "Key has expired"
                    );
                    return ValidationResult::Expired;
                }

                ValidationResult::Valid(api_key.permission)
            }
        }
    }

    /// Checks if authentication is enabled.
    pub async fn is_enabled(&self) -> bool {
        self.config.read().await.enabled
    }

    /// Adds a new API key at runtime.
    pub async fn add_key(&self, api_key: ApiKey) {
        let mut config = self.config.write().await;
        config.api_keys.insert(api_key.key.clone(), api_key);
    }

    /// Removes an API key at runtime.
    pub async fn remove_key(&self, key: &str) {
        let mut config = self.config.write().await;
        config.api_keys.remove(key);
    }

    /// Checks if an API key has a specific scope.
    pub async fn has_scope(&self, key: &str, scope: Scope) -> bool {
        let config = self.config.read().await;
        config
            .api_keys
            .get(key)
            .map_or(false, |api_key| api_key.enabled && api_key.scopes.contains(&scope))
    }
}

impl Default for AuthState {
    fn default() -> Self {
        Self::new(AuthConfig::disabled())
    }
}

/// Error response for authentication failures.
#[derive(Debug, Serialize)]
struct AuthError {
    error: AuthErrorDetail,
}

#[derive(Debug, Serialize)]
struct AuthErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    code: String,
}

impl AuthError {
    fn unauthorized(message: &str) -> Self {
        Self {
            error: AuthErrorDetail {
                message: message.to_string(),
                error_type: "authentication_error".to_string(),
                code: "invalid_api_key".to_string(),
            },
        }
    }

    fn forbidden(message: &str) -> Self {
        Self {
            error: AuthErrorDetail {
                message: message.to_string(),
                error_type: "authorization_error".to_string(),
                code: "insufficient_permissions".to_string(),
            },
        }
    }
}

/// Extracts API key from request headers.
///
/// Supports:
/// - `Authorization: Bearer sk-xxx`
/// - `X-API-Key: sk-xxx`
fn extract_api_key(request: &Request) -> Option<String> {
    // Try Authorization header first
    if let Some(auth_header) = request.headers().get(header::AUTHORIZATION) {
        if let Ok(auth_str) = auth_header.to_str() {
            if let Some(token) = auth_str.strip_prefix("Bearer ") {
                return Some(token.trim().to_string());
            }
        }
    }

    // Try X-API-Key header
    if let Some(api_key_header) = request.headers().get("x-api-key") {
        if let Ok(key_str) = api_key_header.to_str() {
            return Some(key_str.trim().to_string());
        }
    }

    None
}

/// Extracts the client IP from request headers.
fn extract_client_ip(request: &Request) -> Option<String> {
    // Try X-Forwarded-For first (for reverse proxies)
    if let Some(forwarded) = request.headers().get("x-forwarded-for") {
        if let Ok(value) = forwarded.to_str() {
            // Take the first IP (original client)
            if let Some(ip) = value.split(',').next() {
                return Some(ip.trim().to_string());
            }
        }
    }

    // Try X-Real-IP
    if let Some(real_ip) = request.headers().get("x-real-ip") {
        if let Ok(ip) = real_ip.to_str() {
            return Some(ip.trim().to_string());
        }
    }

    None
}

/// Generates a request ID for audit logging.
fn generate_request_id() -> String {
    format!("req-{}", uuid::Uuid::new_v4().as_simple())
}

/// Authentication middleware.
///
/// Validates API keys and sets permission level in request extensions.
/// Logs all authentication events to the audit log.
pub async fn auth_middleware(
    auth_state: AuthState,
    request: Request,
    next: Next,
) -> Response {
    let path = request.uri().path().to_string();
    let client_ip = extract_client_ip(&request);
    let request_id = generate_request_id();

    // Check if path is public
    if auth_state.is_public_path(&path).await {
        return next.run(request).await;
    }

    // Check if auth is enabled
    if !auth_state.is_enabled().await {
        return next.run(request).await;
    }

    // Extract and validate API key
    let api_key = match extract_api_key(&request) {
        Some(key) => key,
        None => {
            auth_state.metrics().record_failure(AuthFailureReason::MissingKey);
            auth_state
                .audit_logger()
                .auth_failure(
                    &request_id,
                    client_ip.as_deref(),
                    "missing_api_key",
                    &path,
                )
                .await;

            return (
                StatusCode::UNAUTHORIZED,
                Json(AuthError::unauthorized(
                    "Missing API key. Include it in Authorization header as 'Bearer sk-xxx' or X-API-Key header.",
                )),
            )
                .into_response();
        }
    };

    // Validate the key with detailed result
    let validation_result = auth_state.validate_key_detailed(&api_key).await;

    let permission = match validation_result {
        ValidationResult::Valid(perm) => perm,
        ValidationResult::NotFound => {
            auth_state.metrics().record_failure(AuthFailureReason::InvalidKey);
            auth_state
                .audit_logger()
                .auth_failure(&request_id, client_ip.as_deref(), "invalid_key", &path)
                .await;

            return (
                StatusCode::UNAUTHORIZED,
                Json(AuthError::unauthorized("Invalid API key")),
            )
                .into_response();
        }
        ValidationResult::Disabled => {
            auth_state.metrics().record_failure(AuthFailureReason::DisabledKey);
            auth_state
                .audit_logger()
                .auth_disabled(&request_id, client_ip.as_deref(), &api_key, &path)
                .await;

            return (
                StatusCode::UNAUTHORIZED,
                Json(AuthError::unauthorized("API key is disabled")),
            )
                .into_response();
        }
        ValidationResult::Expired => {
            auth_state.metrics().record_failure(AuthFailureReason::ExpiredKey);
            auth_state
                .audit_logger()
                .auth_expired(&request_id, client_ip.as_deref(), &api_key, &path)
                .await;

            return (
                StatusCode::UNAUTHORIZED,
                Json(AuthError::unauthorized("API key has expired")),
            )
                .into_response();
        }
    };

    // Check if admin permission is required for this path
    if requires_admin_permission(&path) && permission != Permission::Admin {
        auth_state.metrics().record_failure(AuthFailureReason::InsufficientScope);
        auth_state
            .audit_logger()
            .scope_violation(
                &request_id,
                client_ip.as_deref(),
                &api_key,
                &path,
                "admin",
            )
            .await;

        return (
            StatusCode::FORBIDDEN,
            Json(AuthError::forbidden(
                "This endpoint requires admin permissions",
            )),
        )
            .into_response();
    }

    // Log successful authentication
    auth_state
        .audit_logger()
        .auth_success(&request_id, client_ip.as_deref(), &api_key, &path)
        .await;

    // Continue with the request
    next.run(request).await
}

/// Checks if a path requires admin permission.
fn requires_admin_permission(path: &str) -> bool {
    // Admin-only paths
    let admin_paths = [
        "/api/models/load",
        "/api/models/unload",
        "/api/keys",
        "/api/config",
    ];

    admin_paths.iter().any(|p| path.starts_with(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_key_user() {
        let key = ApiKey::user("sk-test123");
        assert_eq!(key.key, "sk-test123");
        assert_eq!(key.permission, Permission::User);
        assert!(key.enabled);
    }

    #[test]
    fn test_api_key_admin() {
        let key = ApiKey::admin("sk-admin456");
        assert_eq!(key.key, "sk-admin456");
        assert_eq!(key.permission, Permission::Admin);
        assert!(key.enabled);
    }

    #[test]
    fn test_api_key_with_name() {
        let key = ApiKey::user("sk-test").with_name("Production Key");
        assert_eq!(key.name, Some("Production Key".to_string()));
    }

    #[test]
    fn test_auth_config_disabled() {
        let config = AuthConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_auth_config_enabled() {
        let config = AuthConfig::enabled();
        assert!(config.enabled);
        assert!(config.public_paths.contains(&"/health".to_string()));
        assert!(config.public_paths.contains(&"/ready".to_string()));
    }

    #[test]
    fn test_auth_config_add_key() {
        let config = AuthConfig::enabled()
            .add_key(ApiKey::user("sk-user1"))
            .add_key(ApiKey::admin("sk-admin1"));

        assert_eq!(config.api_keys.len(), 2);
        assert!(config.api_keys.contains_key("sk-user1"));
        assert!(config.api_keys.contains_key("sk-admin1"));
    }

    #[tokio::test]
    async fn test_auth_state_validate_key() {
        let config = AuthConfig::enabled()
            .add_key(ApiKey::user("sk-user"))
            .add_key(ApiKey::admin("sk-admin"));

        let state = AuthState::new(config);

        assert_eq!(state.validate_key("sk-user").await, Some(Permission::User));
        assert_eq!(state.validate_key("sk-admin").await, Some(Permission::Admin));
        assert_eq!(state.validate_key("sk-invalid").await, None);
    }

    #[tokio::test]
    async fn test_auth_state_public_path() {
        let config = AuthConfig::enabled()
            .add_public_path("/custom/public");

        let state = AuthState::new(config);

        assert!(state.is_public_path("/health").await);
        assert!(state.is_public_path("/ready").await);
        assert!(state.is_public_path("/custom/public").await);
        assert!(!state.is_public_path("/v1/chat/completions").await);
    }

    #[tokio::test]
    async fn test_auth_state_disabled() {
        let config = AuthConfig::disabled();
        let state = AuthState::new(config);

        // All paths are "public" when auth is disabled
        assert!(state.is_public_path("/v1/chat/completions").await);
        assert!(state.is_public_path("/api/models/load").await);
    }

    #[test]
    fn test_requires_admin_permission() {
        assert!(requires_admin_permission("/api/models/load"));
        assert!(requires_admin_permission("/api/models/unload"));
        assert!(requires_admin_permission("/api/keys"));
        assert!(!requires_admin_permission("/v1/chat/completions"));
        assert!(!requires_admin_permission("/v1/embeddings"));
        assert!(!requires_admin_permission("/health"));
    }

    #[test]
    fn test_permission_default() {
        let perm = Permission::default();
        assert_eq!(perm, Permission::User);
    }

    // === Phase 3: Scope-Based Authorization Tests ===

    #[test]
    fn test_scope_display() {
        assert_eq!(Scope::Inference.as_str(), "inference");
        assert_eq!(Scope::Admin.as_str(), "admin");
        assert_eq!(Scope::Metrics.as_str(), "metrics");
    }

    #[test]
    fn test_scope_from_str() {
        assert_eq!(Scope::from_str("inference"), Some(Scope::Inference));
        assert_eq!(Scope::from_str("inf"), Some(Scope::Inference));
        assert_eq!(Scope::from_str("admin"), Some(Scope::Admin));
        assert_eq!(Scope::from_str("adm"), Some(Scope::Admin));
        assert_eq!(Scope::from_str("metrics"), Some(Scope::Metrics));
        assert_eq!(Scope::from_str("met"), Some(Scope::Metrics));
        assert_eq!(Scope::from_str("invalid"), None);
    }

    #[test]
    fn test_api_key_with_scopes() {
        let key = ApiKey::with_scopes("sk-inf-test123", vec![Scope::Inference]);
        assert_eq!(key.key, "sk-inf-test123");
        assert!(key.scopes.contains(&Scope::Inference));
        assert!(!key.scopes.contains(&Scope::Admin));
    }

    #[test]
    fn test_api_key_admin_has_all_scopes() {
        let key = ApiKey::admin("sk-adm-test456");
        assert!(key.scopes.contains(&Scope::Inference));
        assert!(key.scopes.contains(&Scope::Admin));
        assert!(key.scopes.contains(&Scope::Metrics));
    }

    #[test]
    fn test_api_key_parse_from_format() {
        // sk-{scope}-{random}
        let key_str = "sk-inf-abc123def456";
        let scope = ApiKey::parse_scope_from_key(key_str);
        assert_eq!(scope, Some(Scope::Inference));

        let admin_key = "sk-adm-xyz789";
        assert_eq!(ApiKey::parse_scope_from_key(admin_key), Some(Scope::Admin));

        let metrics_key = "sk-met-qrs456";
        assert_eq!(ApiKey::parse_scope_from_key(metrics_key), Some(Scope::Metrics));

        let legacy_key = "sk-oldkey123";
        assert_eq!(ApiKey::parse_scope_from_key(legacy_key), None);
    }

    #[test]
    fn test_endpoint_scope_requirements() {
        // Inference endpoints require Inference scope
        assert_eq!(required_scope_for_path("/v1/chat/completions"), Some(Scope::Inference));
        assert_eq!(required_scope_for_path("/v1/completions"), Some(Scope::Inference));
        assert_eq!(required_scope_for_path("/v1/embeddings"), Some(Scope::Inference));
        assert_eq!(required_scope_for_path("/v1/models"), Some(Scope::Inference));

        // Admin endpoints require Admin scope
        assert_eq!(required_scope_for_path("/api/models/load"), Some(Scope::Admin));
        assert_eq!(required_scope_for_path("/api/models/unload"), Some(Scope::Admin));
        assert_eq!(required_scope_for_path("/api/keys"), Some(Scope::Admin));
        assert_eq!(required_scope_for_path("/api/config"), Some(Scope::Admin));
        assert_eq!(required_scope_for_path("/admin/models/load"), Some(Scope::Admin));
        assert_eq!(required_scope_for_path("/admin/models/unload"), Some(Scope::Admin));
        assert_eq!(required_scope_for_path("/admin/models/status"), Some(Scope::Admin));
        assert_eq!(required_scope_for_path("/admin/models/warmup"), Some(Scope::Admin));

        // Public paths require no scope
        assert_eq!(required_scope_for_path("/health"), None);
        assert_eq!(required_scope_for_path("/ready"), None);
    }

    #[tokio::test]
    async fn test_auth_state_validate_scopes() {
        let config = AuthConfig::enabled()
            .add_key(ApiKey::with_scopes("sk-inf-user", vec![Scope::Inference]))
            .add_key(ApiKey::admin("sk-adm-admin"));

        let state = AuthState::new(config);

        // Inference-only key should work for inference endpoints
        assert!(state.has_scope("sk-inf-user", Scope::Inference).await);
        assert!(!state.has_scope("sk-inf-user", Scope::Admin).await);

        // Admin key should have all scopes
        assert!(state.has_scope("sk-adm-admin", Scope::Inference).await);
        assert!(state.has_scope("sk-adm-admin", Scope::Admin).await);
        assert!(state.has_scope("sk-adm-admin", Scope::Metrics).await);
    }

    #[test]
    fn test_api_key_hashed_storage() {
        // Keys should be stored hashed, not plaintext
        let key = ApiKey::with_scopes("sk-inf-secret123", vec![Scope::Inference]);
        let hashed = key.hash_key();

        // Hashed key should not equal plaintext
        assert_ne!(hashed, "sk-inf-secret123");
        // Should be able to verify against hash
        assert!(ApiKey::verify_key("sk-inf-secret123", &hashed));
        assert!(!ApiKey::verify_key("wrong-key", &hashed));
    }

    // ========================================================================
    // Sprint 6 Day 19.6: Auth Metrics Tests
    // ========================================================================

    #[test]
    fn test_auth_metrics_new() {
        let metrics = AuthMetrics::new();
        assert_eq!(metrics.total_failures(), 0);
    }

    #[test]
    fn test_auth_metrics_record_failures() {
        let metrics = AuthMetrics::new();

        metrics.record_failure(AuthFailureReason::MissingKey);
        metrics.record_failure(AuthFailureReason::InvalidKey);
        metrics.record_failure(AuthFailureReason::InvalidKey);
        metrics.record_failure(AuthFailureReason::ExpiredKey);
        metrics.record_failure(AuthFailureReason::DisabledKey);
        metrics.record_failure(AuthFailureReason::InsufficientScope);

        assert_eq!(metrics.total_failures(), 6);
        assert_eq!(metrics.failures_missing_key.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.failures_invalid_key.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.failures_expired_key.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.failures_disabled_key.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.failures_insufficient_scope.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_auth_metrics_prometheus_format() {
        let metrics = AuthMetrics::new();
        metrics.record_failure(AuthFailureReason::InvalidKey);
        metrics.record_failure(AuthFailureReason::InvalidKey);
        metrics.record_failure(AuthFailureReason::MissingKey);

        let output = metrics.render_prometheus();

        assert!(output.contains("# HELP infernum_auth_failures_total"));
        assert!(output.contains("# TYPE infernum_auth_failures_total counter"));
        assert!(output.contains("infernum_auth_failures_total{reason=\"invalid_key\"} 2"));
        assert!(output.contains("infernum_auth_failures_total{reason=\"missing_key\"} 1"));
        assert!(output.contains("infernum_auth_failures_total{reason=\"expired_key\"} 0"));
    }

    #[test]
    fn test_auth_state_has_metrics() {
        let config = AuthConfig::enabled();
        let state = AuthState::new(config);

        // Metrics should be accessible
        assert_eq!(state.metrics().total_failures(), 0);

        // Record a failure
        state.metrics().record_failure(AuthFailureReason::InvalidKey);
        assert_eq!(state.metrics().total_failures(), 1);
    }
}
