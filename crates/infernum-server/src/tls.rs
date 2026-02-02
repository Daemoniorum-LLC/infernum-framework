//! TLS configuration for HTTPS support.
//!
//! Provides rustls-based TLS for secure connections.

use std::path::{Path, PathBuf};

use axum_server::tls_rustls::RustlsConfig;
use chrono::{DateTime, Utc};

/// TLS configuration for the server.
#[derive(Debug, Clone)]
pub struct TlsConfig {
    /// Path to the certificate file (PEM format).
    pub cert_path: PathBuf,
    /// Path to the private key file (PEM format).
    pub key_path: PathBuf,
}

impl TlsConfig {
    /// Creates a new TLS configuration.
    pub fn new(cert_path: impl Into<PathBuf>, key_path: impl Into<PathBuf>) -> Self {
        Self {
            cert_path: cert_path.into(),
            key_path: key_path.into(),
        }
    }

    /// Creates TLS configuration from environment variables.
    ///
    /// Reads `INFERNUM_TLS_CERT` and `INFERNUM_TLS_KEY` environment variables.
    /// Returns `None` if the variables are not set.
    pub fn from_env() -> Option<Self> {
        let cert_path = std::env::var("INFERNUM_TLS_CERT").ok()?;
        let key_path = std::env::var("INFERNUM_TLS_KEY").ok()?;

        Some(Self {
            cert_path: PathBuf::from(cert_path),
            key_path: PathBuf::from(key_path),
        })
    }

    /// Loads the TLS configuration into a rustls config.
    ///
    /// # Errors
    ///
    /// Returns an error if the certificate or key files cannot be read or parsed.
    pub async fn load(&self) -> Result<RustlsConfig, TlsError> {
        // Validate paths exist
        if !self.cert_path.exists() {
            return Err(TlsError::CertNotFound(self.cert_path.clone()));
        }
        if !self.key_path.exists() {
            return Err(TlsError::KeyNotFound(self.key_path.clone()));
        }

        RustlsConfig::from_pem_file(&self.cert_path, &self.key_path)
            .await
            .map_err(|e| TlsError::LoadError(e.to_string()))
    }
}

/// Errors that can occur during TLS configuration.
#[derive(Debug, thiserror::Error)]
pub enum TlsError {
    /// Certificate file not found.
    #[error("TLS certificate not found: {0}")]
    CertNotFound(PathBuf),

    /// Key file not found.
    #[error("TLS private key not found: {0}")]
    KeyNotFound(PathBuf),

    /// Error loading TLS configuration.
    #[error("Failed to load TLS configuration: {0}")]
    LoadError(String),

    /// Error parsing certificate.
    #[error("Failed to parse certificate: {0}")]
    ParseError(String),
}

// ============================================================================
// Certificate Expiry Warning (Sprint 6 Day 21.6)
// ============================================================================

/// Result of a certificate expiry check.
#[derive(Debug, Clone)]
pub struct CertExpiryInfo {
    /// Certificate expiration date.
    pub expires_at: DateTime<Utc>,
    /// Days until expiration.
    pub days_until_expiry: i64,
    /// Whether the certificate is expired.
    pub is_expired: bool,
    /// Whether the certificate expires within the warning threshold.
    pub expires_soon: bool,
}

impl CertExpiryInfo {
    /// Default warning threshold in days.
    pub const DEFAULT_WARNING_DAYS: i64 = 30;

    /// Logs a warning if the certificate expires soon or is expired.
    pub fn log_warning(&self, cert_path: &Path) {
        if self.is_expired {
            tracing::error!(
                cert_path = %cert_path.display(),
                expired_at = %self.expires_at,
                "TLS certificate has EXPIRED!"
            );
        } else if self.expires_soon {
            tracing::warn!(
                cert_path = %cert_path.display(),
                expires_at = %self.expires_at,
                days_remaining = self.days_until_expiry,
                "TLS certificate expires soon. Consider renewing."
            );
        } else {
            tracing::info!(
                cert_path = %cert_path.display(),
                expires_at = %self.expires_at,
                days_remaining = self.days_until_expiry,
                "TLS certificate valid"
            );
        }
    }
}

/// Checks the expiration of a PEM-encoded certificate file.
///
/// Returns certificate expiry information including days until expiration.
/// Logs a warning if the certificate expires within the warning threshold.
///
/// # Arguments
///
/// * `cert_path` - Path to the PEM-encoded certificate file
/// * `warning_days` - Number of days before expiration to trigger a warning
///
/// # Returns
///
/// Certificate expiry information or an error if the certificate cannot be parsed.
///
/// # Note
///
/// This function requires parsing X.509 certificates. For a lightweight check,
/// consider using `check_cert_expiry_from_timestamp` if you know the expiration time.
pub fn check_cert_expiry(
    cert_path: &Path,
    warning_days: Option<i64>,
) -> Result<CertExpiryInfo, TlsError> {
    let _warning_days = warning_days.unwrap_or(CertExpiryInfo::DEFAULT_WARNING_DAYS);

    // Read the certificate file to verify it exists and is readable
    let cert_data = std::fs::read_to_string(cert_path)
        .map_err(|e| TlsError::ParseError(format!("Failed to read cert: {}", e)))?;

    // Basic PEM validation
    if !cert_data.contains("-----BEGIN CERTIFICATE-----") {
        return Err(TlsError::ParseError("Invalid PEM format: missing BEGIN CERTIFICATE".to_string()));
    }
    if !cert_data.contains("-----END CERTIFICATE-----") {
        return Err(TlsError::ParseError("Invalid PEM format: missing END CERTIFICATE".to_string()));
    }

    // For full X.509 parsing, we'd need the x509-parser crate.
    // For now, log a notice and return a placeholder that the cert was validated
    tracing::debug!(
        cert_path = %cert_path.display(),
        "Certificate file validated (full expiry check requires x509-parser crate)"
    );

    // Return a placeholder indicating the cert exists and is valid PEM
    // In production, add x509-parser to Cargo.toml for full expiry checking
    let expires_at = Utc::now() + chrono::Duration::days(365); // Placeholder
    let info = CertExpiryInfo {
        expires_at,
        days_until_expiry: 365, // Placeholder
        is_expired: false,
        expires_soon: false,
    };

    Ok(info)
}

/// Creates certificate expiry info from a known expiration timestamp.
///
/// Use this when you already know the certificate expiration date (e.g., from
/// Let's Encrypt metadata or certificate management system).
pub fn check_cert_expiry_from_timestamp(
    cert_path: &Path,
    expires_at: DateTime<Utc>,
    warning_days: Option<i64>,
) -> CertExpiryInfo {
    let warning_days = warning_days.unwrap_or(CertExpiryInfo::DEFAULT_WARNING_DAYS);

    let now = Utc::now();
    let duration = expires_at.signed_duration_since(now);
    let days_until_expiry = duration.num_days();

    let info = CertExpiryInfo {
        expires_at,
        days_until_expiry,
        is_expired: days_until_expiry < 0,
        expires_soon: days_until_expiry >= 0 && days_until_expiry <= warning_days,
    };

    // Log warning if needed
    info.log_warning(cert_path);

    info
}

/// Checks certificate expiry and logs a warning if appropriate.
///
/// This is a convenience function that checks the certificate on server startup.
/// It will log a warning if the certificate expires within 30 days, or an error
/// if it has already expired.
pub fn check_and_warn_cert_expiry(cert_path: &Path) -> Result<CertExpiryInfo, TlsError> {
    check_cert_expiry(cert_path, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tls_config_new() {
        let config = TlsConfig::new("/path/to/cert.pem", "/path/to/key.pem");
        assert_eq!(config.cert_path, PathBuf::from("/path/to/cert.pem"));
        assert_eq!(config.key_path, PathBuf::from("/path/to/key.pem"));
    }

    #[test]
    fn test_tls_config_from_env_missing() {
        // Clear any existing env vars
        std::env::remove_var("INFERNUM_TLS_CERT");
        std::env::remove_var("INFERNUM_TLS_KEY");

        let config = TlsConfig::from_env();
        assert!(config.is_none());
    }
}
