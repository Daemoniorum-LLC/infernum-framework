//! Configuration management for the Infernum CLI.
//!
//! Configuration is loaded from (in order of precedence):
//! 1. Command-line arguments
//! 2. Environment variables (INFERNUM_*)
//! 3. Config file (~/.config/infernum/config.toml)
//! 4. Default values

use std::path::PathBuf;

use figment::{
    providers::{Env, Format, Serialized, Toml},
    Figment,
};
use serde::{Deserialize, Serialize};

/// CLI configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Default model to use when --model is not specified.
    #[serde(default)]
    pub default_model: Option<String>,

    /// Default temperature for generation.
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Default maximum tokens for generation.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Server host.
    #[serde(default = "default_host")]
    pub server_host: String,

    /// Server port.
    #[serde(default = "default_port")]
    pub server_port: u16,
}

fn default_temperature() -> f32 {
    0.7
}

fn default_max_tokens() -> u32 {
    256
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8080
}

impl Default for Config {
    fn default() -> Self {
        Self {
            default_model: None,
            temperature: default_temperature(),
            max_tokens: default_max_tokens(),
            server_host: default_host(),
            server_port: default_port(),
        }
    }
}

impl Config {
    /// Loads configuration from all sources.
    ///
    /// Reports warnings for configuration errors but falls back to defaults.
    pub fn load() -> Self {
        let config_path = Self::config_path();

        let figment = Figment::new()
            .merge(Serialized::defaults(Config::default()))
            .merge(Toml::file(&config_path))
            .merge(Env::prefixed("INFERNUM_"));

        match figment.extract::<Config>() {
            Ok(config) => config,
            Err(e) => {
                // Report the error clearly to the user
                eprintln!("\x1b[33mWarning:\x1b[0m Configuration error, using defaults");
                eprintln!("  Config file: {}", config_path.display());
                eprintln!("  Error: {}", e);
                eprintln!();
                eprintln!("  To fix, edit or delete the config file:");
                eprintln!("    rm {}", config_path.display());
                eprintln!();
                Config::default()
            },
        }
    }

    /// Returns the path to the config file.
    pub fn config_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("infernum")
            .join("config.toml")
    }

    /// Returns the path to the config directory.
    pub fn config_dir() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("infernum")
    }

    /// Saves the current configuration to the config file.
    pub fn save(&self) -> Result<(), std::io::Error> {
        let config_dir = Self::config_dir();
        std::fs::create_dir_all(&config_dir)?;

        let config_path = Self::config_path();
        let toml_str = toml::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        std::fs::write(&config_path, toml_str)?;
        Ok(())
    }

    /// Sets the default model and saves.
    pub fn set_default_model(&mut self, model: &str) -> Result<(), std::io::Error> {
        self.default_model = Some(model.to_string());
        self.save()
    }

    /// Clears the default model and saves.
    pub fn clear_default_model(&mut self) -> Result<(), std::io::Error> {
        self.default_model = None;
        self.save()
    }
}

/// Prints the current configuration and its sources.
pub fn show_config() {
    let config = Config::load();
    let config_path = Config::config_path();

    println!("Infernum Configuration");
    println!("======================\n");

    println!("Config file: {}", config_path.display());
    if config_path.exists() {
        println!("Status: Found\n");
    } else {
        println!("Status: Not found (using defaults)\n");
    }

    println!("Current settings:");
    println!(
        "  default_model: {}",
        config.default_model.as_deref().unwrap_or("(not set)")
    );
    println!("  temperature: {}", config.temperature);
    println!("  max_tokens: {}", config.max_tokens);
    println!("  server_host: {}", config.server_host);
    println!("  server_port: {}", config.server_port);

    println!("\nEnvironment variables:");
    println!("  INFERNUM_DEFAULT_MODEL");
    println!("  INFERNUM_TEMPERATURE");
    println!("  INFERNUM_MAX_TOKENS");
    println!("  INFERNUM_SERVER_HOST");
    println!("  INFERNUM_SERVER_PORT");
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_config_default() {
        let config = Config::default();

        assert!(config.default_model.is_none());
        assert!((config.temperature - 0.7).abs() < 0.001);
        assert_eq!(config.max_tokens, 256);
        assert_eq!(config.server_host, "0.0.0.0");
        assert_eq!(config.server_port, 8080);
    }

    #[test]
    fn test_config_serialization() {
        let config = Config {
            default_model: Some("llama".to_string()),
            temperature: 0.5,
            max_tokens: 1024,
            server_host: "127.0.0.1".to_string(),
            server_port: 9090,
        };

        // Serialize to TOML
        let toml_str = toml::to_string_pretty(&config).expect("serialize");
        assert!(toml_str.contains("default_model = \"llama\""));
        assert!(toml_str.contains("temperature = 0.5"));
        assert!(toml_str.contains("max_tokens = 1024"));
        assert!(toml_str.contains("server_host = \"127.0.0.1\""));
        assert!(toml_str.contains("server_port = 9090"));

        // Deserialize from TOML
        let parsed: Config = toml::from_str(&toml_str).expect("deserialize");
        assert_eq!(parsed.default_model, config.default_model);
        assert!((parsed.temperature - config.temperature).abs() < 0.001);
        assert_eq!(parsed.max_tokens, config.max_tokens);
        assert_eq!(parsed.server_host, config.server_host);
        assert_eq!(parsed.server_port, config.server_port);
    }

    #[test]
    fn test_config_partial_toml() {
        // Test that partial TOML uses defaults for missing fields
        let toml_str = r#"
            default_model = "custom-model"
        "#;

        let config: Config = toml::from_str(toml_str).expect("parse");
        assert_eq!(config.default_model, Some("custom-model".to_string()));
        // Other fields should use defaults
        assert!((config.temperature - 0.7).abs() < 0.001);
        assert_eq!(config.max_tokens, 256);
    }

    #[test]
    fn test_config_path() {
        let path = Config::config_path();
        assert!(path.to_string_lossy().contains("infernum"));
        assert!(path.to_string_lossy().contains("config.toml"));
    }

    #[test]
    fn test_config_dir() {
        let dir = Config::config_dir();
        assert!(dir.to_string_lossy().contains("infernum"));
    }

    #[test]
    fn test_config_save_and_load() {
        let temp = TempDir::new().expect("temp dir");
        let config_path = temp.path().join("config.toml");

        let config = Config {
            default_model: Some("test-model".to_string()),
            temperature: 0.9,
            max_tokens: 512,
            server_host: "localhost".to_string(),
            server_port: 3000,
        };

        // Manually save to temp path
        let toml_str = toml::to_string_pretty(&config).expect("serialize");
        std::fs::write(&config_path, toml_str).expect("write");

        // Load and verify
        let content = std::fs::read_to_string(&config_path).expect("read");
        let loaded: Config = toml::from_str(&content).expect("parse");

        assert_eq!(loaded.default_model, Some("test-model".to_string()));
        assert!((loaded.temperature - 0.9).abs() < 0.001);
        assert_eq!(loaded.max_tokens, 512);
        assert_eq!(loaded.server_host, "localhost");
        assert_eq!(loaded.server_port, 3000);
    }

    #[test]
    fn test_config_clone() {
        let config = Config {
            default_model: Some("model".to_string()),
            temperature: 0.8,
            max_tokens: 100,
            server_host: "host".to_string(),
            server_port: 1234,
        };

        let cloned = config.clone();
        assert_eq!(cloned.default_model, config.default_model);
        assert!((cloned.temperature - config.temperature).abs() < 0.001);
        assert_eq!(cloned.max_tokens, config.max_tokens);
        assert_eq!(cloned.server_host, config.server_host);
        assert_eq!(cloned.server_port, config.server_port);
    }

    #[test]
    fn test_config_debug() {
        let config = Config::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("Config"));
        assert!(debug_str.contains("temperature"));
        assert!(debug_str.contains("max_tokens"));
    }

    #[test]
    fn test_default_functions() {
        assert!((default_temperature() - 0.7).abs() < 0.001);
        assert_eq!(default_max_tokens(), 256);
        assert_eq!(default_host(), "0.0.0.0");
        assert_eq!(default_port(), 8080);
    }

    #[test]
    fn test_config_load_fallback() {
        // Config::load() should return defaults when config file doesn't exist
        // This test just verifies it doesn't panic
        let config = Config::load();
        // Should have reasonable defaults
        assert!(config.temperature > 0.0);
        assert!(config.max_tokens > 0);
    }

    #[test]
    fn test_config_empty_model() {
        let config = Config {
            default_model: None,
            ..Default::default()
        };

        assert!(config.default_model.is_none());

        // Serialize and deserialize - None should be omitted or handled
        let toml_str = toml::to_string_pretty(&config).expect("serialize");
        let parsed: Config = toml::from_str(&toml_str).expect("parse");
        assert!(parsed.default_model.is_none());
    }

    #[test]
    fn test_config_json_serialization() {
        // Also test JSON serialization since Config derives Serialize/Deserialize
        let config = Config {
            default_model: Some("gpt-4".to_string()),
            temperature: 0.3,
            max_tokens: 2048,
            server_host: "api.example.com".to_string(),
            server_port: 443,
        };

        let json = serde_json::to_string(&config).expect("serialize json");
        assert!(json.contains("gpt-4"));
        assert!(json.contains("2048"));

        let parsed: Config = serde_json::from_str(&json).expect("parse json");
        assert_eq!(parsed.default_model, config.default_model);
        assert_eq!(parsed.max_tokens, config.max_tokens);
    }

    #[test]
    fn test_config_temperature_range() {
        // Temperature can be any positive float
        let config = Config {
            temperature: 1.5,
            ..Default::default()
        };
        assert!((config.temperature - 1.5).abs() < 0.001);

        let config = Config {
            temperature: 0.0,
            ..Default::default()
        };
        assert!(config.temperature.abs() < 0.001);
    }
}
