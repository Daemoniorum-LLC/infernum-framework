//! Configuration hot reload functionality.
//!
//! Watches configuration files for changes and reloads them at runtime.
//!
//! # Example
//!
//! ```ignore
//! use infernum_server::config_reload::{ConfigWatcher, ReloadableConfig};
//! use std::sync::Arc;
//!
//! let config = Arc::new(ReloadableConfig::load()?);
//! let watcher = ConfigWatcher::new(config.clone())?;
//!
//! // Config will be automatically reloaded when the file changes
//! watcher.start().await?;
//! ```
//!
//! # What Can Be Hot Reloaded
//!
//! The following settings can be changed without restarting:
//! - Rate limiting configuration (limits, windows)
//! - Authentication configuration (API keys)
//! - CORS configuration (origins, methods)
//! - Security headers
//!
//! The following require a restart:
//! - Server host/port
//! - TLS certificates
//! - Model selection

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use parking_lot::RwLock;
use tokio::sync::mpsc;

use crate::auth::AuthConfig;
use crate::config::Config;
use crate::config_error::ConfigError;
use crate::security::{CorsConfig, RateLimitConfig, SecurityHeadersConfig};

/// Configuration that can be reloaded at runtime.
pub struct ReloadableConfig {
    /// The current configuration.
    inner: RwLock<Config>,
    /// Path to the config file being watched.
    config_path: Option<PathBuf>,
    /// Whether reload is enabled.
    reload_enabled: AtomicBool,
    /// Reload callbacks.
    callbacks: RwLock<Vec<Box<dyn Fn(&Config) + Send + Sync>>>,
}

impl std::fmt::Debug for ReloadableConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReloadableConfig")
            .field("config_path", &self.config_path)
            .field(
                "reload_enabled",
                &self.reload_enabled.load(Ordering::SeqCst),
            )
            .field("callbacks_count", &self.callbacks.read().len())
            .finish_non_exhaustive()
    }
}

impl ReloadableConfig {
    /// Creates a new reloadable config from an existing config.
    pub fn new(config: Config) -> Self {
        Self {
            inner: RwLock::new(config),
            config_path: None,
            reload_enabled: AtomicBool::new(false),
            callbacks: RwLock::new(Vec::new()),
        }
    }

    /// Loads configuration and prepares for hot reload.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError` if configuration cannot be loaded.
    pub fn load() -> Result<Self, ConfigError> {
        let config = Config::load()?;
        let config_path = find_config_path();

        Ok(Self {
            inner: RwLock::new(config),
            config_path,
            reload_enabled: AtomicBool::new(true),
            callbacks: RwLock::new(Vec::new()),
        })
    }

    /// Loads from a specific config file.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError` if file cannot be loaded.
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let config = Config::from_file(path.as_ref())?;

        Ok(Self {
            inner: RwLock::new(config),
            config_path: Some(path.as_ref().to_path_buf()),
            reload_enabled: AtomicBool::new(true),
            callbacks: RwLock::new(Vec::new()),
        })
    }

    /// Returns the path to the config file being watched.
    pub fn config_path(&self) -> Option<&Path> {
        self.config_path.as_deref()
    }

    /// Returns a read lock on the current configuration.
    pub fn read(&self) -> parking_lot::RwLockReadGuard<'_, Config> {
        self.inner.read()
    }

    /// Returns the current rate limit configuration.
    pub fn rate_limit(&self) -> RateLimitConfig {
        self.inner.read().rate_limit.clone()
    }

    /// Returns the current auth configuration.
    pub fn auth(&self) -> Option<AuthConfig> {
        self.inner.read().auth.clone()
    }

    /// Returns the current CORS configuration.
    pub fn cors(&self) -> CorsConfig {
        self.inner.read().cors.clone()
    }

    /// Returns the current security headers configuration.
    pub fn security_headers(&self) -> SecurityHeadersConfig {
        self.inner.read().security_headers.clone()
    }

    /// Enables or disables hot reload.
    pub fn set_reload_enabled(&self, enabled: bool) {
        self.reload_enabled.store(enabled, Ordering::SeqCst);
    }

    /// Returns whether hot reload is enabled.
    pub fn is_reload_enabled(&self) -> bool {
        self.reload_enabled.load(Ordering::SeqCst)
    }

    /// Registers a callback to be called when configuration is reloaded.
    pub fn on_reload<F>(&self, callback: F)
    where
        F: Fn(&Config) + Send + Sync + 'static,
    {
        self.callbacks.write().push(Box::new(callback));
    }

    /// Attempts to reload configuration from the file.
    ///
    /// Only reloads runtime-configurable settings.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError` if reload fails.
    pub fn reload(&self) -> Result<ReloadResult, ConfigError> {
        if !self.is_reload_enabled() {
            return Ok(ReloadResult::Disabled);
        }

        let Some(path) = &self.config_path else {
            return Ok(ReloadResult::NoConfigFile);
        };

        let new_config = Config::from_file(path)?;
        let mut changes = Vec::new();

        // Update runtime-configurable settings
        {
            let mut current = self.inner.write();

            // Rate limiting
            if current.rate_limit != new_config.rate_limit {
                changes.push(ConfigChange::RateLimit);
                current.rate_limit = new_config.rate_limit.clone();
            }

            // Authentication
            if current.auth != new_config.auth {
                changes.push(ConfigChange::Auth);
                current.auth = new_config.auth.clone();
            }

            // CORS
            if current.cors_enabled != new_config.cors_enabled {
                changes.push(ConfigChange::Cors);
                current.cors_enabled = new_config.cors_enabled;
            }

            // Security headers
            if current.security_headers != new_config.security_headers {
                changes.push(ConfigChange::SecurityHeaders);
                current.security_headers = new_config.security_headers.clone();
            }

            // Log warnings for settings that require restart
            if current.port != new_config.port || current.host != new_config.host {
                tracing::warn!(
                    "Server address changed in config ({} -> {}), restart required",
                    current.socket_addr(),
                    new_config.socket_addr()
                );
            }

            if current.tls.is_some() != new_config.tls.is_some() {
                tracing::warn!("TLS configuration changed, restart required");
            }

            if current.model != new_config.model {
                tracing::warn!("Model changed in config, restart required");
            }

            if current.max_concurrent_requests != new_config.max_concurrent_requests {
                tracing::warn!(
                    "Max concurrent requests changed ({} -> {}), restart required",
                    current.max_concurrent_requests,
                    new_config.max_concurrent_requests
                );
            }
        }

        // Call reload callbacks
        if !changes.is_empty() {
            let config = self.inner.read();
            for callback in self.callbacks.read().iter() {
                callback(&config);
            }

            tracing::info!(
                "Configuration reloaded: {:?}",
                changes.iter().map(|c| c.as_str()).collect::<Vec<_>>()
            );
        }

        Ok(ReloadResult::Reloaded(changes))
    }
}

/// Result of a configuration reload attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReloadResult {
    /// Reload is disabled.
    Disabled,
    /// No config file is being watched.
    NoConfigFile,
    /// Configuration was reloaded with the specified changes.
    Reloaded(Vec<ConfigChange>),
}

/// A configuration change that was applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigChange {
    /// Rate limiting configuration changed.
    RateLimit,
    /// Authentication configuration changed.
    Auth,
    /// CORS configuration changed.
    Cors,
    /// Security headers changed.
    SecurityHeaders,
}

impl ConfigChange {
    /// Returns a string representation of the change.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RateLimit => "rate_limit",
            Self::Auth => "auth",
            Self::Cors => "cors",
            Self::SecurityHeaders => "security_headers",
        }
    }
}

/// Watches configuration files for changes and triggers reloads.
pub struct ConfigWatcher {
    config: Arc<ReloadableConfig>,
    watcher: Option<RecommendedWatcher>,
    shutdown: Arc<AtomicBool>,
}

impl ConfigWatcher {
    /// Creates a new config watcher.
    ///
    /// # Errors
    ///
    /// Returns an error if the watcher cannot be created.
    pub fn new(config: Arc<ReloadableConfig>) -> Result<Self, ConfigError> {
        Ok(Self {
            config,
            watcher: None,
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Starts watching for configuration changes.
    ///
    /// This spawns a background task that watches the config file.
    ///
    /// # Errors
    ///
    /// Returns an error if watching cannot be started.
    pub async fn start(&mut self) -> Result<(), ConfigError> {
        let Some(path) = self.config.config_path() else {
            tracing::info!("No config file to watch, hot reload disabled");
            return Ok(());
        };

        let path = path.to_path_buf();
        let config = self.config.clone();
        let shutdown = self.shutdown.clone();

        // Create a channel for file events
        let (tx, mut rx) = mpsc::channel::<()>(1);

        // Create the watcher
        let tx_clone = tx.clone();
        let path_clone = path.clone();
        let watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
            if let Ok(event) = res {
                match event.kind {
                    EventKind::Modify(_) | EventKind::Create(_) => {
                        if event.paths.iter().any(|p| p == &path_clone) {
                            let _ = tx_clone.blocking_send(());
                        }
                    },
                    _ => {},
                }
            }
        })
        .map_err(|e| ConfigError::ParseError {
            message: format!("Failed to create file watcher: {}", e),
        })?;

        self.watcher = Some(watcher);

        // Watch the config file's parent directory
        if let Some(watcher) = &mut self.watcher {
            let watch_path = path.parent().unwrap_or(&path);
            watcher
                .watch(watch_path, RecursiveMode::NonRecursive)
                .map_err(|e| ConfigError::ParseError {
                    message: format!("Failed to watch config file: {}", e),
                })?;

            tracing::info!("Watching config file for changes: {}", path.display());
        }

        // Spawn the reload task
        tokio::spawn(async move {
            let debounce = Duration::from_millis(500);
            let mut last_reload = std::time::Instant::now();

            while !shutdown.load(Ordering::SeqCst) {
                tokio::select! {
                    Some(()) = rx.recv() => {
                        // Debounce rapid changes
                        if last_reload.elapsed() < debounce {
                            continue;
                        }
                        last_reload = std::time::Instant::now();

                        // Small delay to ensure file is fully written
                        tokio::time::sleep(Duration::from_millis(100)).await;

                        match config.reload() {
                            Ok(ReloadResult::Reloaded(changes)) if !changes.is_empty() => {
                                tracing::info!(
                                    "Hot reload complete: {} settings updated",
                                    changes.len()
                                );
                            }
                            Ok(_) => {
                                tracing::debug!("Config file changed but no runtime changes detected");
                            }
                            Err(e) => {
                                tracing::error!("Failed to reload configuration: {}", e);
                            }
                        }
                    }
                    _ = tokio::time::sleep(Duration::from_secs(1)) => {
                        // Periodic check for shutdown
                    }
                }
            }
        });

        Ok(())
    }

    /// Stops the config watcher.
    pub fn stop(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
}

impl Drop for ConfigWatcher {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Finds the configuration file path.
fn find_config_path() -> Option<PathBuf> {
    // Check environment variable first
    if let Ok(path) = std::env::var("INFERNUM_CONFIG") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }

    // Check standard locations
    for path in &[
        "/etc/infernum/config.toml",
        "./config.toml",
        "./infernum.toml",
    ] {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_config() -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("create temp file");
        writeln!(
            file,
            r#"
            host = "127.0.0.1"
            port = 8080
            max_concurrent_requests = 64
            max_queue_size = 256
            "#
        )
        .expect("write config");
        file
    }

    #[test]
    fn test_reloadable_config_new() {
        let config = Config::default();
        let reloadable = ReloadableConfig::new(config);
        assert!(!reloadable.is_reload_enabled());
        assert!(reloadable.config_path().is_none());
    }

    #[test]
    fn test_reloadable_config_from_file() {
        let file = create_test_config();
        let reloadable = ReloadableConfig::load_from_file(file.path()).expect("load");
        assert!(reloadable.is_reload_enabled());
        assert!(reloadable.config_path().is_some());
    }

    #[test]
    fn test_reload_disabled() {
        let config = Config::default();
        let reloadable = ReloadableConfig::new(config);
        reloadable.set_reload_enabled(false);

        let result = reloadable.reload().expect("reload");
        assert_eq!(result, ReloadResult::Disabled);
    }

    #[test]
    fn test_reload_no_config_file() {
        let config = Config::default();
        let reloadable = ReloadableConfig::new(config);
        reloadable.set_reload_enabled(true);

        let result = reloadable.reload().expect("reload");
        assert_eq!(result, ReloadResult::NoConfigFile);
    }

    #[test]
    fn test_reload_from_file() {
        let file = create_test_config();
        let reloadable = ReloadableConfig::load_from_file(file.path()).expect("load");

        // Initial reload should show no changes
        let result = reloadable.reload().expect("reload");
        assert!(matches!(result, ReloadResult::Reloaded(changes) if changes.is_empty()));
    }

    #[test]
    fn test_config_change_as_str() {
        assert_eq!(ConfigChange::RateLimit.as_str(), "rate_limit");
        assert_eq!(ConfigChange::Auth.as_str(), "auth");
        assert_eq!(ConfigChange::Cors.as_str(), "cors");
        assert_eq!(ConfigChange::SecurityHeaders.as_str(), "security_headers");
    }

    #[test]
    fn test_set_reload_enabled() {
        let config = Config::default();
        let reloadable = ReloadableConfig::new(config);

        reloadable.set_reload_enabled(true);
        assert!(reloadable.is_reload_enabled());

        reloadable.set_reload_enabled(false);
        assert!(!reloadable.is_reload_enabled());
    }

    #[test]
    fn test_reload_callback() {
        use std::sync::atomic::AtomicU32;

        let file = create_test_config();
        let reloadable = ReloadableConfig::load_from_file(file.path()).expect("load");

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        reloadable.on_reload(move |_| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        // Reload should not call callback if nothing changed
        reloadable.reload().expect("reload");
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }
}
