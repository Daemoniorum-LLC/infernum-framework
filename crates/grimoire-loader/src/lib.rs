//! # Grimoire Loader
//!
//! Integration with the Grimoire prompt management system.
//!
//! This crate provides utilities for loading personas and prompts
//! from the Grimoire filesystem structure.

#![warn(missing_docs)]
#![warn(clippy::all)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use infernum_core::Result;
use serde::{Deserialize, Serialize};

/// Default Grimoire personas path.
pub const DEFAULT_GRIMOIRE_PATH: &str = "/home/lilith/development/projects/grimoire/personas/";

/// A loaded Grimoire persona.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrimoirePersona {
    /// Persona identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// System prompt.
    pub system_prompt: String,
    /// Variants of the persona.
    pub variants: HashMap<String, String>,
    /// Metadata.
    pub metadata: HashMap<String, String>,
}

/// Loader for Grimoire personas.
pub struct GrimoireLoader {
    base_path: PathBuf,
    cache: dashmap::DashMap<String, GrimoirePersona>,
}

impl GrimoireLoader {
    /// Creates a new loader with the default path.
    #[must_use]
    pub fn new() -> Self {
        Self::with_path(DEFAULT_GRIMOIRE_PATH)
    }

    /// Creates a new loader with a custom path.
    #[must_use]
    pub fn with_path(path: impl Into<PathBuf>) -> Self {
        Self {
            base_path: path.into(),
            cache: dashmap::DashMap::new(),
        }
    }

    /// Loads a persona by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the persona cannot be loaded.
    pub async fn load(&self, persona_id: &str) -> Result<GrimoirePersona> {
        // Check cache first
        if let Some(cached) = self.cache.get(persona_id) {
            return Ok(cached.clone());
        }

        // Build path
        let path = self.base_path.join(persona_id);
        let prompt_path = if path.is_dir() {
            path.join("prompt.md")
        } else {
            path.with_extension("md")
        };

        // Load file
        let content = tokio::fs::read_to_string(&prompt_path)
            .await
            .map_err(|e| infernum_core::Error::Io(e))?;

        // Parse persona
        let persona = self.parse_persona(persona_id, &content)?;

        // Cache it
        self.cache.insert(persona_id.to_string(), persona.clone());

        Ok(persona)
    }

    /// Parses persona content.
    fn parse_persona(&self, id: &str, content: &str) -> Result<GrimoirePersona> {
        // Simple parsing - in production, handle frontmatter YAML
        Ok(GrimoirePersona {
            id: id.to_string(),
            name: id.to_string(),
            system_prompt: content.to_string(),
            variants: HashMap::new(),
            metadata: HashMap::new(),
        })
    }

    /// Lists all available personas.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read.
    pub async fn list(&self) -> Result<Vec<String>> {
        let mut personas = Vec::new();

        if !self.base_path.exists() {
            return Ok(personas);
        }

        let mut entries = tokio::fs::read_dir(&self.base_path).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_dir() || path.extension().map_or(false, |e| e == "md") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    personas.push(name.to_string());
                }
            }
        }

        Ok(personas)
    }

    /// Clears the cache.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Returns the base path.
    #[must_use]
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }
}

impl Default for GrimoireLoader {
    fn default() -> Self {
        Self::new()
    }
}
