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
            }
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
    println!("  default_model: {}", config.default_model.as_deref().unwrap_or("(not set)"));
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
