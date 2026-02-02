//! Unified configuration for the inference server.
//!
//! All settings can be loaded from environment variables or TOML config files.
//!
//! # Environment Variables
//!
//! | Variable | Description | Default |
//! |----------|-------------|---------|
//! | `INFERNUM_HOST` | Server bind address | `0.0.0.0` |
//! | `INFERNUM_PORT` | Server port | `8080` |
//! | `INFERNUM_CORS` | Enable CORS (`true`/`false`) | `true` |
//! | `INFERNUM_MODEL` | Model to load on startup | None |
//! | `INFERNUM_MAX_CONCURRENT` | Max concurrent requests | `64` |
//! | `INFERNUM_MAX_QUEUE` | Max queued requests | `256` |
//! | `INFERNUM_TLS_CERT` | TLS certificate path | None |
//! | `INFERNUM_TLS_KEY` | TLS private key path | None |
//! | `INFERNUM_RATE_LIMIT_ENABLED` | Enable rate limiting | `true` |
//! | `INFERNUM_RATE_LIMIT_REQUESTS` | Max requests per window | `100` |
//! | `INFERNUM_RATE_LIMIT_WINDOW_SECS` | Rate limit window in seconds | `60` |
//! | `INFERNUM_CORS_ORIGINS` | Allowed CORS origins (comma-separated) | Any |
//! | `INFERNUM_API_KEYS` | Valid API keys (comma-separated) | None (auth disabled) |
//! | `INFERNUM_CUDA_DEVICE` | CUDA device index | `0` |
//! | `INFERNUM_CONFIG` | Path to TOML config file | None |
//! | `INFERNUM_DRAFT_MODEL` | Draft model for speculative decoding | None |
//! | `INFERNUM_SPECULATIVE_TOKENS` | Tokens per speculative round (1-16) | `5` |
//!
//! # Configuration File
//!
//! ```toml
//! [server]
//! host = "0.0.0.0"
//! port = 8080
//! max_concurrent_requests = 64
//! max_queue_size = 256
//!
//! [auth]
//! enabled = true
//! api_keys = ["sk-adm-admin123"]
//! ```
//!
//! # Priority
//!
//! Configuration is loaded with the following priority (highest to lowest):
//! 1. Environment variables
//! 2. TOML config file
//! 3. Default values

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::auth::AuthConfig;
use crate::config_error::ConfigError;
use crate::security::{CorsConfig, RateLimitConfig, SecurityHeadersConfig};
use crate::tls::TlsConfig;

/// Complete server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Server host address.
    pub host: IpAddr,
    /// Server port.
    pub port: u16,
    /// Enable CORS.
    pub cors_enabled: bool,
    /// CORS configuration.
    #[serde(skip)]
    pub cors: CorsConfig,
    /// Model to load on startup.
    pub model: Option<String>,
    /// Draft model for speculative decoding (smaller model, fits in VRAM).
    /// When set alongside `model`, enables speculative decoding for 3-4x speedup.
    pub draft_model: Option<String>,
    /// Number of draft tokens per speculative round (default: 5).
    pub speculative_tokens: u32,
    /// Maximum concurrent requests.
    pub max_concurrent_requests: usize,
    /// Maximum queued requests.
    pub max_queue_size: usize,
    /// TLS configuration (enables HTTPS).
    #[serde(skip)]
    pub tls: Option<TlsConfig>,
    /// Rate limiting configuration.
    #[serde(skip)]
    pub rate_limit: RateLimitConfig,
    /// Security headers configuration.
    #[serde(skip)]
    pub security_headers: SecurityHeadersConfig,
    /// Authentication configuration.
    #[serde(skip)]
    pub auth: Option<AuthConfig>,
    /// CUDA device index.
    pub cuda_device: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            host: IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            port: 8080,
            cors_enabled: true,
            cors: CorsConfig::default(),
            model: None,
            draft_model: None,
            speculative_tokens: 5,
            max_concurrent_requests: 64,
            max_queue_size: 256,
            tls: None,
            rate_limit: RateLimitConfig::default(),
            security_headers: SecurityHeadersConfig::api_only(),
            auth: None,
            cuda_device: 0,
        }
    }
}

impl Config {
    /// Creates a new configuration builder.
    pub fn builder() -> ConfigBuilder {
        ConfigBuilder::default()
    }

    /// Loads configuration from environment variables.
    ///
    /// Returns an error if any configuration value is invalid.
    /// Missing values use defaults.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError` if any environment variable contains an invalid value.
    pub fn from_env() -> Result<Self, ConfigError> {
        let mut config = Self::default();
        let mut errors = Vec::new();

        // Server address
        if let Ok(host) = std::env::var("INFERNUM_HOST") {
            match host.parse() {
                Ok(addr) => config.host = addr,
                Err(_) => errors.push(ConfigError::invalid_value(
                    "INFERNUM_HOST",
                    "valid IP address (e.g., 0.0.0.0, 127.0.0.1)",
                    &host,
                )),
            }
        }

        if let Ok(port) = std::env::var("INFERNUM_PORT") {
            match port.parse::<u16>() {
                Ok(0) => errors.push(ConfigError::out_of_range("INFERNUM_PORT", 0, 1, 65535)),
                Ok(p) => config.port = p,
                Err(_) => errors.push(ConfigError::invalid_value(
                    "INFERNUM_PORT",
                    "integer 1-65535",
                    &port,
                )),
            }
        }

        // CORS
        if let Ok(cors) = std::env::var("INFERNUM_CORS") {
            config.cors_enabled = cors.to_lowercase() == "true" || cors == "1";
        }
        config.cors = CorsConfig::from_env();

        // Model - auto-detect HCT directories and convert to holo:// URL
        if let Ok(model) = std::env::var("INFERNUM_MODEL") {
            if !model.is_empty() {
                let model_source = if !model.starts_with("holo://") && is_hct_model(&model) {
                    // Get HoloTensor quality parameters from env, with defaults
                    let min_quality = std::env::var("INFERNUM_HOLO_MIN_QUALITY")
                        .ok()
                        .and_then(|v| v.parse::<f32>().ok())
                        .unwrap_or(0.7);
                    let target_quality = std::env::var("INFERNUM_HOLO_TARGET_QUALITY")
                        .ok()
                        .and_then(|v| v.parse::<f32>().ok())
                        .unwrap_or(0.95);

                    let holo_url = format!(
                        "holo://{}?min={}&target={}",
                        model, min_quality, target_quality
                    );
                    tracing::info!(
                        original = %model,
                        holo_url = %holo_url,
                        "Auto-detected HCT model directory, using HoloTensor loader"
                    );
                    holo_url
                } else {
                    model
                };
                config.model = Some(model_source);
            }
        }

        // Draft model for speculative decoding
        if let Ok(draft_model) = std::env::var("INFERNUM_DRAFT_MODEL") {
            if !draft_model.is_empty() {
                tracing::info!(
                    draft_model = %draft_model,
                    "Draft model configured for speculative decoding"
                );
                config.draft_model = Some(draft_model);
            }
        }

        // Number of speculative tokens per round
        if let Ok(tokens) = std::env::var("INFERNUM_SPECULATIVE_TOKENS") {
            match tokens.parse::<u32>() {
                Ok(0) => errors.push(ConfigError::out_of_range(
                    "INFERNUM_SPECULATIVE_TOKENS",
                    0,
                    1,
                    16,
                )),
                Ok(n) if n > 16 => errors.push(ConfigError::out_of_range(
                    "INFERNUM_SPECULATIVE_TOKENS",
                    n,
                    1,
                    16,
                )),
                Ok(n) => config.speculative_tokens = n,
                Err(_) => errors.push(ConfigError::invalid_value(
                    "INFERNUM_SPECULATIVE_TOKENS",
                    "integer 1-16",
                    &tokens,
                )),
            }
        }

        // Concurrency
        if let Ok(max) = std::env::var("INFERNUM_MAX_CONCURRENT") {
            match max.parse::<usize>() {
                Ok(0) => errors.push(ConfigError::out_of_range(
                    "INFERNUM_MAX_CONCURRENT",
                    0,
                    1,
                    10000,
                )),
                Ok(n) if n > 10000 => errors.push(ConfigError::out_of_range(
                    "INFERNUM_MAX_CONCURRENT",
                    n,
                    1,
                    10000,
                )),
                Ok(n) => config.max_concurrent_requests = n,
                Err(_) => errors.push(ConfigError::invalid_value(
                    "INFERNUM_MAX_CONCURRENT",
                    "integer 1-10000",
                    &max,
                )),
            }
        }

        // Queue size
        if let Ok(max) = std::env::var("INFERNUM_MAX_QUEUE") {
            match max.parse::<usize>() {
                Ok(0) => errors.push(ConfigError::out_of_range(
                    "INFERNUM_MAX_QUEUE",
                    0,
                    1,
                    100000,
                )),
                Ok(n) if n > 100000 => errors.push(ConfigError::out_of_range(
                    "INFERNUM_MAX_QUEUE",
                    n,
                    1,
                    100000,
                )),
                Ok(n) => config.max_queue_size = n,
                Err(_) => errors.push(ConfigError::invalid_value(
                    "INFERNUM_MAX_QUEUE",
                    "integer 1-100000",
                    &max,
                )),
            }
        }

        // TLS
        config.tls = TlsConfig::from_env();

        // If TLS is enabled, update security headers
        if config.tls.is_some() {
            config.security_headers = config.security_headers.with_https();
        }

        // Rate limiting
        config.rate_limit = RateLimitConfig::from_env();

        // Authentication
        let auth = AuthConfig::from_env();
        config.auth = if auth.is_enabled() { Some(auth) } else { None };

        // CUDA device
        if let Ok(device) = std::env::var("INFERNUM_CUDA_DEVICE") {
            match device.parse::<u32>() {
                Ok(d) if d > 15 => errors.push(ConfigError::out_of_range(
                    "INFERNUM_CUDA_DEVICE",
                    d,
                    0,
                    15,
                )),
                Ok(d) => config.cuda_device = d,
                Err(_) => errors.push(ConfigError::invalid_value(
                    "INFERNUM_CUDA_DEVICE",
                    "integer 0-15",
                    &device,
                )),
            }
        }

        // Log all errors for debugging
        for err in &errors {
            tracing::error!("Configuration error: {}", err);
        }

        // Return errors if any
        if !errors.is_empty() {
            return Err(ConfigError::multiple(errors));
        }

        // Validate configuration consistency
        config.validate()?;

        Ok(config)
    }

    /// Loads configuration from a TOML file.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError` if the file cannot be read or parsed.
    pub fn from_file(path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path).map_err(|_| ConfigError::FileNotFound {
            path: path.display().to_string(),
        })?;

        let config: Self = toml::from_str(&content).map_err(|e| ConfigError::ParseError {
            message: e.to_string(),
        })?;

        config.validate()?;
        Ok(config)
    }

    /// Loads configuration with priority: env vars > file > defaults.
    ///
    /// Tries to load from:
    /// 1. `INFERNUM_CONFIG` environment variable path
    /// 2. `/etc/infernum/config.toml`
    /// 3. `./config.toml`
    /// 4. `./infernum.toml`
    ///
    /// Then overrides with environment variables.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError` if configuration is invalid.
    pub fn load() -> Result<Self, ConfigError> {
        // Start with defaults
        let mut config = Self::default();

        // Try to load from file
        if let Ok(path) = std::env::var("INFERNUM_CONFIG") {
            config = Self::from_file(Path::new(&path))?;
        } else {
            // Try standard locations
            for path in &[
                "/etc/infernum/config.toml",
                "./config.toml",
                "./infernum.toml",
            ] {
                if Path::new(path).exists() {
                    match Self::from_file(Path::new(path)) {
                        Ok(c) => {
                            config = c;
                            tracing::info!("Loaded configuration from {}", path);
                            break;
                        }
                        Err(e) => {
                            tracing::warn!("Failed to load config from {}: {}", path, e);
                        }
                    }
                }
            }
        }

        // Override with environment variables
        config.apply_env_overrides()?;
        config.validate()?;

        Ok(config)
    }

    /// Applies environment variable overrides to the current config.
    fn apply_env_overrides(&mut self) -> Result<(), ConfigError> {
        let mut errors = Vec::new();

        if let Ok(host) = std::env::var("INFERNUM_HOST") {
            match host.parse() {
                Ok(addr) => self.host = addr,
                Err(_) => errors.push(ConfigError::invalid_value(
                    "INFERNUM_HOST",
                    "valid IP address",
                    &host,
                )),
            }
        }

        if let Ok(port) = std::env::var("INFERNUM_PORT") {
            match port.parse::<u16>() {
                Ok(0) => errors.push(ConfigError::out_of_range("INFERNUM_PORT", 0, 1, 65535)),
                Ok(p) => self.port = p,
                Err(_) => errors.push(ConfigError::invalid_value(
                    "INFERNUM_PORT",
                    "integer 1-65535",
                    &port,
                )),
            }
        }

        if let Ok(max) = std::env::var("INFERNUM_MAX_CONCURRENT") {
            match max.parse::<usize>() {
                Ok(n) if n > 0 && n <= 10000 => self.max_concurrent_requests = n,
                Ok(n) => errors.push(ConfigError::out_of_range(
                    "INFERNUM_MAX_CONCURRENT",
                    n,
                    1,
                    10000,
                )),
                Err(_) => errors.push(ConfigError::invalid_value(
                    "INFERNUM_MAX_CONCURRENT",
                    "integer 1-10000",
                    &max,
                )),
            }
        }

        if !errors.is_empty() {
            return Err(ConfigError::multiple(errors));
        }

        Ok(())
    }

    /// Validates configuration consistency.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError` if configuration is inconsistent.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let mut errors = Vec::new();

        // Port validation
        if self.port == 0 {
            errors.push(ConfigError::out_of_range("server.port", 0, 1, 65535));
        }

        // Concurrent requests validation
        if self.max_concurrent_requests == 0 {
            errors.push(ConfigError::out_of_range(
                "server.max_concurrent_requests",
                0,
                1,
                10000,
            ));
        }

        // Queue size should be >= concurrent requests
        if self.max_queue_size < self.max_concurrent_requests {
            errors.push(ConfigError::conflict(&format!(
                "max_queue_size ({}) should be >= max_concurrent_requests ({})",
                self.max_queue_size, self.max_concurrent_requests
            )));
        }

        // TLS + port 80 warning (not error, just log)
        if self.tls.is_some() && self.port == 80 {
            tracing::warn!(
                "TLS enabled but using port 80. Consider using port 443 for HTTPS."
            );
        }

        // Auth enabled but no keys
        if let Some(ref auth) = self.auth {
            if auth.is_enabled() && auth.key_count() == 0 {
                errors.push(ConfigError::conflict(
                    "Authentication enabled but no API keys configured",
                ));
            }
        }

        // Rate limit validation
        if self.rate_limit.enabled {
            if self.rate_limit.max_requests == 0 {
                errors.push(ConfigError::out_of_range(
                    "rate_limit.max_requests",
                    0,
                    1,
                    1000000,
                ));
            }
            if self.rate_limit.window.as_secs() == 0 {
                errors.push(ConfigError::out_of_range(
                    "rate_limit.window",
                    "0s",
                    "1s",
                    "86400s",
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(ConfigError::multiple(errors))
        }
    }

    /// Returns the socket address for binding.
    pub fn socket_addr(&self) -> SocketAddr {
        SocketAddr::new(self.host, self.port)
    }

    /// Returns whether TLS is enabled.
    pub fn is_tls_enabled(&self) -> bool {
        self.tls.is_some()
    }

    /// Returns whether authentication is required.
    pub fn is_auth_enabled(&self) -> bool {
        self.auth.is_some()
    }

    /// Prints a configuration summary to stdout.
    pub fn print_summary(&self) {
        println!("╔══════════════════════════════════════════════════════════╗");
        println!("║                  Infernum Configuration                   ║");
        println!("╠══════════════════════════════════════════════════════════╣");
        println!(
            "║ Server: {}:{:<43} ║",
            self.host,
            self.port
        );
        println!(
            "║ Max Concurrent: {:<42} ║",
            self.max_concurrent_requests
        );
        println!(
            "║ Max Queue: {:<47} ║",
            self.max_queue_size
        );
        println!(
            "║ Auth Enabled: {:<44} ║",
            self.is_auth_enabled()
        );
        println!(
            "║ Rate Limiting: {:<43} ║",
            self.rate_limit.enabled
        );
        println!(
            "║ TLS Enabled: {:<45} ║",
            self.is_tls_enabled()
        );
        if let Some(ref model) = self.model {
            let truncated = if model.len() > 50 {
                format!("{}...", &model[..47])
            } else {
                model.clone()
            };
            println!("║ Model: {:<51} ║", truncated);
        }
        if let Some(ref draft) = self.draft_model {
            let truncated = if draft.len() > 45 {
                format!("{}...", &draft[..42])
            } else {
                draft.clone()
            };
            println!("║ Draft Model: {:<45} ║", truncated);
            println!("║ Speculative Tokens: {:<38} ║", self.speculative_tokens);
        }
        println!("╚══════════════════════════════════════════════════════════╝");
    }

    /// Returns whether speculative decoding is configured.
    pub fn is_speculative_enabled(&self) -> bool {
        self.draft_model.is_some() && self.model.is_some()
    }
}

/// Builder for server configuration.
#[derive(Debug, Default)]
pub struct ConfigBuilder {
    host: Option<IpAddr>,
    port: Option<u16>,
    cors_enabled: Option<bool>,
    cors: Option<CorsConfig>,
    model: Option<String>,
    draft_model: Option<String>,
    speculative_tokens: Option<u32>,
    max_concurrent_requests: Option<usize>,
    max_queue_size: Option<usize>,
    tls: Option<TlsConfig>,
    rate_limit: Option<RateLimitConfig>,
    security_headers: Option<SecurityHeadersConfig>,
    auth: Option<AuthConfig>,
    cuda_device: Option<u32>,
}

impl ConfigBuilder {
    /// Sets the server host address.
    pub fn host(mut self, host: IpAddr) -> Self {
        self.host = Some(host);
        self
    }

    /// Sets the server port.
    pub fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    /// Sets the socket address (host and port).
    pub fn addr(mut self, addr: SocketAddr) -> Self {
        self.host = Some(addr.ip());
        self.port = Some(addr.port());
        self
    }

    /// Enables or disables CORS.
    pub fn cors(mut self, enabled: bool) -> Self {
        self.cors_enabled = Some(enabled);
        self
    }

    /// Sets the CORS configuration.
    pub fn cors_config(mut self, config: CorsConfig) -> Self {
        self.cors = Some(config);
        self
    }

    /// Sets the model to load on startup.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the draft model for speculative decoding.
    /// This should be a small model (1B-8B) that fits entirely in VRAM.
    pub fn draft_model(mut self, model: impl Into<String>) -> Self {
        self.draft_model = Some(model.into());
        self
    }

    /// Sets the number of speculative tokens per round (1-16).
    pub fn speculative_tokens(mut self, tokens: u32) -> Self {
        self.speculative_tokens = Some(tokens.clamp(1, 16));
        self
    }

    /// Sets the maximum concurrent requests.
    pub fn max_concurrent_requests(mut self, max: usize) -> Self {
        self.max_concurrent_requests = Some(max);
        self
    }

    /// Sets the maximum queue size.
    pub fn max_queue_size(mut self, max: usize) -> Self {
        self.max_queue_size = Some(max);
        self
    }

    /// Sets the TLS configuration.
    pub fn tls(mut self, config: TlsConfig) -> Self {
        self.tls = Some(config);
        self
    }

    /// Sets the rate limiting configuration.
    pub fn rate_limit(mut self, config: RateLimitConfig) -> Self {
        self.rate_limit = Some(config);
        self
    }

    /// Sets the security headers configuration.
    pub fn security_headers(mut self, config: SecurityHeadersConfig) -> Self {
        self.security_headers = Some(config);
        self
    }

    /// Sets the authentication configuration.
    pub fn auth(mut self, config: AuthConfig) -> Self {
        self.auth = Some(config);
        self
    }

    /// Sets the CUDA device index.
    pub fn cuda_device(mut self, device: u32) -> Self {
        self.cuda_device = Some(device);
        self
    }

    /// Builds the configuration, using defaults for unset values.
    pub fn build(self) -> Config {
        let defaults = Config::default();
        let mut config = Config {
            host: self.host.unwrap_or(defaults.host),
            port: self.port.unwrap_or(defaults.port),
            cors_enabled: self.cors_enabled.unwrap_or(defaults.cors_enabled),
            cors: self.cors.unwrap_or(defaults.cors),
            model: self.model.or(defaults.model),
            draft_model: self.draft_model.or(defaults.draft_model),
            speculative_tokens: self.speculative_tokens.unwrap_or(defaults.speculative_tokens),
            max_concurrent_requests: self
                .max_concurrent_requests
                .unwrap_or(defaults.max_concurrent_requests),
            max_queue_size: self.max_queue_size.unwrap_or(defaults.max_queue_size),
            tls: self.tls.or(defaults.tls),
            rate_limit: self.rate_limit.unwrap_or(defaults.rate_limit),
            security_headers: self.security_headers.unwrap_or(defaults.security_headers),
            auth: self.auth.or(defaults.auth),
            cuda_device: self.cuda_device.unwrap_or(defaults.cuda_device),
        };

        // Auto-enable HSTS if TLS is configured
        if config.tls.is_some() && config.security_headers.strict_transport_security.is_none() {
            config.security_headers = config.security_headers.with_https();
        }

        config
    }

    /// Builds the configuration with validation.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError` if the configuration is invalid.
    pub fn build_validated(self) -> Result<Config, ConfigError> {
        let config = self.build();
        config.validate()?;
        Ok(config)
    }
}

/// Checks if a path is a HoloTensor Compressed (HCT) model directory.
///
/// Returns true if the path exists, is a directory, and contains at least
/// one `.hct` file.
fn is_hct_model(model: &str) -> bool {
    use std::path::Path;

    let path = Path::new(model);
    if !path.exists() || !path.is_dir() {
        return false;
    }

    // Check if directory contains .hct files
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Some(ext) = entry.path().extension() {
                if ext == "hct" {
                    return true;
                }
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.host, "0.0.0.0".parse::<IpAddr>().expect("valid"));
        assert_eq!(config.port, 8080);
        assert!(config.cors_enabled);
        assert!(config.model.is_none());
        assert_eq!(config.max_concurrent_requests, 64);
        assert_eq!(config.max_queue_size, 256);
        assert!(config.tls.is_none());
        assert!(config.auth.is_none());
        assert_eq!(config.cuda_device, 0);
    }

    #[test]
    fn test_config_builder() {
        let config = Config::builder()
            .host("127.0.0.1".parse().expect("valid"))
            .port(3000)
            .cors(false)
            .model("test-model")
            .max_concurrent_requests(32)
            .max_queue_size(128)
            .cuda_device(1)
            .build();

        assert_eq!(config.host, "127.0.0.1".parse::<IpAddr>().expect("valid"));
        assert_eq!(config.port, 3000);
        assert!(!config.cors_enabled);
        assert_eq!(config.model, Some("test-model".to_string()));
        assert_eq!(config.max_concurrent_requests, 32);
        assert_eq!(config.max_queue_size, 128);
        assert_eq!(config.cuda_device, 1);
    }

    #[test]
    fn test_config_socket_addr() {
        let config = Config::builder()
            .host("192.168.1.1".parse().expect("valid"))
            .port(9000)
            .build();

        assert_eq!(
            config.socket_addr(),
            "192.168.1.1:9000".parse::<SocketAddr>().expect("valid")
        );
    }

    #[test]
    fn test_config_builder_addr() {
        let config = Config::builder()
            .addr("10.0.0.1:5000".parse().expect("valid"))
            .build();

        assert_eq!(config.host, "10.0.0.1".parse::<IpAddr>().expect("valid"));
        assert_eq!(config.port, 5000);
    }

    #[test]
    fn test_config_tls_auto_hsts() {
        let config = Config::builder()
            .tls(TlsConfig::new("/cert.pem", "/key.pem"))
            .build();

        assert!(config.is_tls_enabled());
        assert!(config.security_headers.strict_transport_security.is_some());
    }

    #[test]
    fn test_config_validation_port_zero() {
        let config = Config::builder().port(0).build();
        let result = config.validate();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("server.port"));
    }

    #[test]
    fn test_config_validation_concurrent_zero() {
        let config = Config::builder().max_concurrent_requests(0).build();
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_config_validation_queue_too_small() {
        let config = Config::builder()
            .max_concurrent_requests(100)
            .max_queue_size(50)
            .build();
        let result = config.validate();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("max_queue_size"));
    }

    #[test]
    fn test_config_validation_valid() {
        let config = Config::builder()
            .port(8080)
            .max_concurrent_requests(64)
            .max_queue_size(256)
            .build();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_build_validated() {
        let result = Config::builder()
            .port(0)
            .build_validated();
        assert!(result.is_err());

        let result = Config::builder()
            .port(8080)
            .build_validated();
        assert!(result.is_ok());
    }

    #[test]
    fn test_config_print_summary() {
        // Just ensure it doesn't panic
        let config = Config::default();
        config.print_summary();
    }
}
