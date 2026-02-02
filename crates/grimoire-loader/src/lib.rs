//! # Grimoire Loader
//!
//! Integration with the Grimoire prompt management system.
//!
//! This crate provides utilities for loading personas, skills, and prompts
//! from the Grimoire filesystem structure.
//!
//! # Features
//!
//! - **Personas**: Load system prompts with variants
//! - **Skills**: Load skill definitions with triggers and code templates
//! - **Templates**: Extract and render code templates from skills
//!
//! # Configuration
//!
//! The Grimoire path can be configured via:
//! 1. Environment variable: `INFERNUM_GRIMOIRE_PATH`
//! 2. Programmatic: `GrimoireLoader::with_path(path)`
//! 3. Default: `~/.local/share/infernum/personas/`

#![warn(missing_docs)]
#![warn(clippy::all)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use regex::Regex;

use infernum_core::Result;
use serde::{Deserialize, Serialize};

/// Environment variable for customizing Grimoire path.
pub const GRIMOIRE_PATH_ENV: &str = "INFERNUM_GRIMOIRE_PATH";

/// Returns the default Grimoire personas path.
///
/// Checks in order:
/// 1. `INFERNUM_GRIMOIRE_PATH` environment variable
/// 2. `~/.local/share/infernum/personas/` (XDG data directory)
#[must_use]
pub fn default_grimoire_path() -> PathBuf {
    // Check environment variable first
    if let Ok(path) = std::env::var(GRIMOIRE_PATH_ENV) {
        return PathBuf::from(path);
    }

    // Fall back to XDG data directory
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("infernum")
        .join("personas")
}

/// Legacy constant for backwards compatibility.
#[deprecated(since = "0.2.0", note = "Use default_grimoire_path() instead")]
pub const DEFAULT_GRIMOIRE_PATH: &str = "~/.local/share/infernum/personas/";

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
    ///
    /// Uses `default_grimoire_path()` which checks `INFERNUM_GRIMOIRE_PATH`
    /// environment variable first, then falls back to XDG data directory.
    #[must_use]
    pub fn new() -> Self {
        Self::with_path(default_grimoire_path())
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
            .map_err(infernum_core::Error::Io)?;

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

// =============================================================================
// Skills
// =============================================================================

/// A code template extracted from a skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeTemplate {
    /// Template name (e.g., "Entity Template", "Service Template").
    pub name: String,
    /// Programming language (e.g., "kotlin", "tsx", "rust").
    pub language: String,
    /// Template content.
    pub content: String,
}

/// A loaded Grimoire skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrimoireSkill {
    /// Skill identifier (directory name).
    pub id: String,
    /// Display name from frontmatter.
    pub name: String,
    /// Description of the skill.
    pub description: String,
    /// Trigger phrases that activate this skill.
    pub triggers: Vec<String>,
    /// Full markdown content (including frontmatter).
    pub content: String,
    /// Extracted code templates.
    pub templates: Vec<CodeTemplate>,
}

impl GrimoireSkill {
    /// Creates a new skill with minimal fields.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            triggers: Vec::new(),
            content: String::new(),
            templates: Vec::new(),
        }
    }

    /// Checks if any trigger matches the given text.
    #[must_use]
    pub fn matches_trigger(&self, text: &str) -> bool {
        let text_lower = text.to_lowercase();
        self.triggers
            .iter()
            .any(|t| text_lower.contains(&t.to_lowercase()))
    }

    /// Returns the skill content as an agent instruction.
    #[must_use]
    pub fn as_agent_instruction(&self) -> String {
        format!(
            "Use the following skill:\n\n# {}\n\n{}\n\n{}",
            self.name,
            self.description,
            self.content
        )
    }

    /// Gets a template by name.
    #[must_use]
    pub fn get_template(&self, name: &str) -> Option<&CodeTemplate> {
        self.templates.iter().find(|t| t.name.eq_ignore_ascii_case(name))
    }

    /// Gets templates by language.
    #[must_use]
    pub fn get_templates_by_language(&self, lang: &str) -> Vec<&CodeTemplate> {
        self.templates
            .iter()
            .filter(|t| t.language.eq_ignore_ascii_case(lang))
            .collect()
    }
}

/// Loader for Grimoire skills.
pub struct SkillLoader {
    base_path: PathBuf,
    cache: dashmap::DashMap<String, GrimoireSkill>,
}

impl SkillLoader {
    /// Creates a new skill loader with the given base path.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            base_path: path.into(),
            cache: dashmap::DashMap::new(),
        }
    }

    /// Returns the base path.
    #[must_use]
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Loads a skill by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the skill cannot be loaded.
    pub async fn load(&self, skill_id: &str) -> Result<GrimoireSkill> {
        // Check cache first
        if let Some(cached) = self.cache.get(skill_id) {
            return Ok(cached.clone());
        }

        // Build path to SKILL.md
        let skill_path = self.base_path.join(skill_id).join("SKILL.md");

        // Load file
        let content = tokio::fs::read_to_string(&skill_path)
            .await
            .map_err(infernum_core::Error::Io)?;

        // Parse skill
        let skill = self.parse_skill(skill_id, &content)?;

        // Cache it
        self.cache.insert(skill_id.to_string(), skill.clone());

        Ok(skill)
    }

    /// Parses skill content.
    fn parse_skill(&self, id: &str, content: &str) -> Result<GrimoireSkill> {
        // Parse YAML frontmatter
        let (name, description, triggers) = self.parse_frontmatter(content);

        // Extract code templates
        let templates = self.extract_templates(content);

        Ok(GrimoireSkill {
            id: id.to_string(),
            name,
            description,
            triggers,
            content: content.to_string(),
            templates,
        })
    }

    /// Parses YAML frontmatter from skill markdown.
    fn parse_frontmatter(&self, content: &str) -> (String, String, Vec<String>) {
        let mut name = String::new();
        let mut description = String::new();
        let mut triggers = Vec::new();

        // Check for frontmatter
        if !content.starts_with("---") {
            return (name, description, triggers);
        }

        // Find end of frontmatter
        if let Some(end_idx) = content[3..].find("---") {
            let frontmatter = &content[3..3 + end_idx];

            // Simple line-by-line YAML parsing
            let mut in_triggers = false;
            for line in frontmatter.lines() {
                let line = line.trim();

                if line.starts_with("name:") {
                    name = line[5..].trim().trim_matches('"').to_string();
                    in_triggers = false;
                } else if line.starts_with("description:") {
                    description = line[12..].trim().trim_matches('"').to_string();
                    in_triggers = false;
                } else if line.starts_with("triggers:") {
                    in_triggers = true;
                } else if in_triggers && line.starts_with('-') {
                    let trigger = line[1..].trim().trim_matches('"').to_string();
                    if !trigger.is_empty() {
                        triggers.push(trigger);
                    }
                } else if !line.starts_with('-') && !line.is_empty() {
                    in_triggers = false;
                }
            }
        }

        (name, description, triggers)
    }

    /// Extracts code templates from markdown content.
    fn extract_templates(&self, content: &str) -> Vec<CodeTemplate> {
        let mut templates = Vec::new();

        // Pattern to match ## Heading followed by ```language code ```
        let heading_re = Regex::new(r"##\s+([^\n]+)").ok();
        let code_block_re = Regex::new(r"```(\w+)\n([\s\S]*?)```").ok();

        let (heading_re, code_block_re) = match (heading_re, code_block_re) {
            (Some(h), Some(c)) => (h, c),
            _ => return templates,
        };

        // Extract code blocks
        for caps in code_block_re.captures_iter(content) {
            let language = caps.get(1).map(|m| m.as_str()).unwrap_or("text");
            let code = caps.get(2).map(|m| m.as_str()).unwrap_or("");

            // Find the heading before this code block
            let code_start = caps.get(0).map(|m| m.start()).unwrap_or(0);
            let mut heading = "Code Block".to_string();

            for cap in heading_re.captures_iter(&content[..code_start]) {
                heading = cap.get(1).map(|m| m.as_str().to_string()).unwrap_or_default();
            }

            // Only include if it looks like a template (has "Template" in heading or is substantial)
            if heading.to_lowercase().contains("template") || code.len() > 50 {
                templates.push(CodeTemplate {
                    name: heading,
                    language: language.to_string(),
                    content: code.trim().to_string(),
                });
            }
        }

        templates
    }

    /// Lists all available skills.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read.
    pub async fn list(&self) -> Result<Vec<String>> {
        let mut skills = Vec::new();

        if !self.base_path.exists() {
            return Ok(skills);
        }

        let mut entries = tokio::fs::read_dir(&self.base_path).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            // Check for directory with SKILL.md
            if path.is_dir() && path.join("SKILL.md").exists() {
                if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    skills.push(name.to_string());
                }
            }
        }

        Ok(skills)
    }

    /// Finds skills matching a trigger phrase.
    ///
    /// # Errors
    ///
    /// Returns an error if skills cannot be loaded.
    pub async fn find_by_trigger(&self, phrase: &str) -> Result<Vec<GrimoireSkill>> {
        let mut matching = Vec::new();

        let skill_ids = self.list().await?;
        for id in skill_ids {
            if let Ok(skill) = self.load(&id).await {
                if skill.matches_trigger(phrase) {
                    matching.push(skill);
                }
            }
        }

        Ok(matching)
    }

    /// Clears the cache.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

// =============================================================================
// Simulacra
// =============================================================================

/// Device profile for a simulacrum.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeviceProfile {
    /// Primary device (e.g., "MacBook Air M2").
    pub primary: String,
    /// Secondary device (optional).
    pub secondary: Option<String>,
    /// Input method.
    pub input_method: Option<String>,
    /// Screen size.
    pub screen_size: Option<String>,
}

/// Demographics for a simulacrum.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimulacrumDemographics {
    /// Age of the user.
    pub age: u32,
    /// Occupation.
    pub occupation: String,
    /// Location.
    pub location: Option<String>,
    /// Device profile.
    pub devices: DeviceProfile,
    /// Connectivity type.
    pub connectivity: Option<String>,
    /// Locale (e.g., "en-US").
    pub locale: Option<String>,
}

/// Cognitive profile for a simulacrum.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimulacrumCognition {
    /// Tech literacy (1-10).
    pub tech_literacy: u8,
    /// Patience level (1-10).
    pub patience: u8,
    /// Attention span.
    pub attention_span: String,
    /// Reading speed.
    pub reading_speed: Option<String>,
    /// Multitasking ability.
    pub multitasking: Option<String>,
}

/// Temperament profile for a simulacrum.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimulacrumTemperament {
    /// Frustration threshold (low/medium/high).
    pub frustration_threshold: String,
    /// Error tolerance.
    pub error_tolerance: String,
    /// Exploration tendency.
    pub exploration_tendency: String,
    /// Help seeking behavior.
    pub help_seeking: Option<String>,
}

/// Vision profile for accessibility.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VisionProfile {
    /// Vision acuity (normal, corrected, lowVision, blind).
    pub acuity: String,
    /// Color blindness type.
    pub color_blindness: String,
    /// Contrast sensitivity.
    pub contrast_sensitivity: Option<String>,
}

/// Motor profile for accessibility.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MotorProfile {
    /// Fine motor control level.
    pub fine_control: String,
    /// Preferred input method.
    pub preferred_input: String,
    /// Click accuracy.
    pub click_accuracy: Option<String>,
}

/// Cognitive accessibility profile.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CognitiveProfile {
    /// Has dyslexia.
    pub dyslexia: Option<bool>,
    /// Has ADHD.
    pub adhd: Option<bool>,
    /// Has autism.
    pub autism: Option<bool>,
}

/// Full accessibility profile.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimulacrumAccessibility {
    /// Vision profile.
    pub vision: VisionProfile,
    /// Motor profile.
    pub motor: MotorProfile,
    /// Cognitive profile.
    pub cognitive: Option<CognitiveProfile>,
    /// Assistive technologies used.
    pub assistive_tech: Option<Vec<String>>,
}

/// Context for the testing session.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimulacrumContext {
    /// Time pressure level.
    pub time_pressure: Option<String>,
    /// Environment description.
    pub environment: Option<String>,
    /// User motivation.
    pub motivation: Option<String>,
    /// Prior experience.
    pub prior_experience: Option<String>,
}

/// Behavioral patterns for a simulacrum.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimulacrumBehavior {
    /// Max clicks before frustration.
    pub max_clicks_before_frustration: u32,
    /// Max seconds waiting before frustration.
    pub max_seconds_waiting: u32,
    /// Scroll behavior.
    pub scroll_behavior: String,
    /// Whether user reads labels.
    pub reads_labels: String,
    /// Search usage pattern.
    pub uses_search: Option<String>,
    /// Keyboard shortcut usage.
    pub keyboard_shortcuts: Option<String>,
    /// Navigation method (for screen readers).
    pub navigation_method: Option<String>,
}

/// Voice and communication style.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimulacrumVoice {
    /// Tone of communication.
    pub tone: String,
    /// Verbosity level.
    pub verbosity: String,
    /// Feedback style.
    pub feedback_style: String,
    /// Example phrases this user would say.
    pub example_phrases: Vec<String>,
}

/// Metadata for a simulacrum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulacrumMetadata {
    /// Unique code (e.g., "FIRST_TIME_USER").
    pub code: String,
    /// Display name (e.g., "Chris").
    pub name: String,
    /// Version number.
    pub version: u32,
    /// Tags for categorization.
    pub tags: Option<Vec<String>>,
    /// Archetype category.
    pub archetype: Option<String>,
}

impl Default for SimulacrumMetadata {
    fn default() -> Self {
        Self {
            code: String::new(),
            name: String::new(),
            version: 1,
            tags: None,
            archetype: None,
        }
    }
}

/// A loaded Grimoire simulacrum (user archetype for UX testing).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GrimoireSimulacrum {
    /// API version (always "simulacra/v1").
    #[serde(rename = "apiVersion")]
    pub api_version: String,
    /// Kind (always "Simulacrum").
    pub kind: String,
    /// Metadata.
    pub metadata: SimulacrumMetadata,
    /// Demographics.
    pub demographics: SimulacrumDemographics,
    /// Cognition profile.
    pub cognition: SimulacrumCognition,
    /// Temperament profile.
    pub temperament: SimulacrumTemperament,
    /// Accessibility profile.
    pub accessibility: SimulacrumAccessibility,
    /// Testing context.
    pub context: Option<SimulacrumContext>,
    /// Behavioral patterns.
    pub behavior: SimulacrumBehavior,
    /// Voice/communication style.
    pub voice: SimulacrumVoice,
}

impl GrimoireSimulacrum {
    /// Returns the simulacrum code.
    #[must_use]
    pub fn code(&self) -> &str {
        &self.metadata.code
    }

    /// Returns the simulacrum name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.metadata.name
    }

    /// Returns the archetype (inferred from tech_literacy if not set).
    #[must_use]
    pub fn archetype(&self) -> &str {
        if let Some(ref archetype) = self.metadata.archetype {
            archetype
        } else {
            match self.cognition.tech_literacy {
                7..=10 => "power-user",
                4..=6 => "average-user",
                _ => "novice",
            }
        }
    }

    /// Checks if this simulacrum has accessibility needs.
    #[must_use]
    pub fn needs_accessibility(&self) -> bool {
        self.accessibility.vision.acuity != "normal"
            || self.accessibility.vision.color_blindness != "none"
            || self.accessibility.motor.fine_control != "normal"
            || self.accessibility.assistive_tech.as_ref().map_or(false, |t| !t.is_empty())
    }

    /// Calculates frustration risk for a given task.
    ///
    /// Returns a score from 0 (no risk) to 1+ (high risk).
    #[must_use]
    pub fn calculate_frustration_risk(&self, expected_clicks: u32, expected_wait_seconds: u32) -> f32 {
        if self.behavior.max_clicks_before_frustration == 0 || self.behavior.max_seconds_waiting == 0 {
            return 0.0;
        }

        let click_ratio = expected_clicks as f32 / self.behavior.max_clicks_before_frustration as f32;
        let wait_ratio = expected_wait_seconds as f32 / self.behavior.max_seconds_waiting as f32;

        // Weight by patience (inverse relationship)
        let patience_factor = (11 - self.cognition.patience.min(10)) as f32 / 10.0;

        ((click_ratio + wait_ratio) / 2.0) * patience_factor
    }

    /// Generates a system prompt for this simulacrum.
    #[must_use]
    pub fn to_system_prompt(&self) -> String {
        let mut prompt = format!(
            "You are roleplaying as {} ({}), a {} user archetype for UX testing.\n\n",
            self.metadata.name,
            self.metadata.code,
            self.archetype()
        );

        prompt.push_str(&format!("## Demographics\n"));
        prompt.push_str(&format!("- Age: {}\n", self.demographics.age));
        prompt.push_str(&format!("- Occupation: {}\n", self.demographics.occupation));
        prompt.push_str(&format!("- Primary Device: {}\n", self.demographics.devices.primary));
        prompt.push('\n');

        prompt.push_str(&format!("## Cognitive Profile\n"));
        prompt.push_str(&format!("- Tech Literacy: {}/10\n", self.cognition.tech_literacy));
        prompt.push_str(&format!("- Patience: {}/10\n", self.cognition.patience));
        prompt.push_str(&format!("- Attention Span: {}\n", self.cognition.attention_span));
        prompt.push('\n');

        prompt.push_str(&format!("## Temperament\n"));
        prompt.push_str(&format!("- Frustration Threshold: {}\n", self.temperament.frustration_threshold));
        prompt.push_str(&format!("- Error Tolerance: {}\n", self.temperament.error_tolerance));
        prompt.push_str(&format!("- Exploration Tendency: {}\n", self.temperament.exploration_tendency));
        prompt.push('\n');

        if self.needs_accessibility() {
            prompt.push_str("## Accessibility\n");
            prompt.push_str(&format!("- Vision: {}\n", self.accessibility.vision.acuity));
            if let Some(ref tech) = self.accessibility.assistive_tech {
                prompt.push_str(&format!("- Assistive Tech: {}\n", tech.join(", ")));
            }
            prompt.push('\n');
        }

        prompt.push_str(&format!("## Behavior\n"));
        prompt.push_str(&format!("- Max Clicks Before Frustration: {}\n", self.behavior.max_clicks_before_frustration));
        prompt.push_str(&format!("- Max Seconds Waiting: {}\n", self.behavior.max_seconds_waiting));
        prompt.push_str(&format!("- Reads Labels: {}\n", self.behavior.reads_labels));
        prompt.push('\n');

        prompt.push_str(&format!("## Voice\n"));
        prompt.push_str(&format!("- Tone: {}\n", self.voice.tone));
        prompt.push_str(&format!("- Verbosity: {}\n", self.voice.verbosity));
        prompt.push_str(&format!("- Example Phrases:\n"));
        for phrase in &self.voice.example_phrases {
            prompt.push_str(&format!("  - \"{}\"\n", phrase));
        }

        prompt
    }
}

/// Loader for Grimoire simulacra.
pub struct SimulacrumLoader {
    base_path: PathBuf,
    cache: dashmap::DashMap<String, GrimoireSimulacrum>,
}

impl SimulacrumLoader {
    /// Creates a new simulacrum loader.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            base_path: path.into(),
            cache: dashmap::DashMap::new(),
        }
    }

    /// Returns the base path.
    #[must_use]
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Loads a simulacrum by code.
    ///
    /// # Errors
    ///
    /// Returns an error if the simulacrum cannot be loaded.
    pub async fn load(&self, code: &str) -> Result<GrimoireSimulacrum> {
        // Check cache first
        if let Some(cached) = self.cache.get(code) {
            return Ok(cached.clone());
        }

        // Convert SCREAMING_SNAKE_CASE to kebab-case for directory lookup
        let dir_name = code.to_lowercase().replace('_', "-");
        let manifest_path = self.base_path.join(&dir_name).join("manifest.yaml");

        // Load file
        let content = tokio::fs::read_to_string(&manifest_path)
            .await
            .map_err(infernum_core::Error::Io)?;

        // Parse YAML
        let simulacrum: GrimoireSimulacrum = serde_yaml::from_str(&content)
            .map_err(|e| infernum_core::Error::internal(format!("YAML parse error: {}", e)))?;

        // Cache it
        self.cache.insert(code.to_string(), simulacrum.clone());

        Ok(simulacrum)
    }

    /// Lists all available simulacra.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read.
    pub async fn list(&self) -> Result<Vec<String>> {
        let mut simulacra = Vec::new();

        if !self.base_path.exists() {
            return Ok(simulacra);
        }

        let mut entries = tokio::fs::read_dir(&self.base_path).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            // Check for directory with manifest.yaml
            if path.is_dir() && path.join("manifest.yaml").exists() {
                if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    // Convert kebab-case to SCREAMING_SNAKE_CASE
                    let code = name.to_uppercase().replace('-', "_");
                    simulacra.push(code);
                }
            }
        }

        Ok(simulacra)
    }

    /// Loads all simulacra.
    ///
    /// # Errors
    ///
    /// Returns an error if simulacra cannot be loaded.
    pub async fn load_all(&self) -> Result<Vec<GrimoireSimulacrum>> {
        let codes = self.list().await?;
        let mut simulacra = Vec::with_capacity(codes.len());

        for code in codes {
            if let Ok(sim) = self.load(&code).await {
                simulacra.push(sim);
            }
        }

        Ok(simulacra)
    }

    /// Finds simulacra by tag.
    ///
    /// # Errors
    ///
    /// Returns an error if simulacra cannot be loaded.
    pub async fn find_by_tag(&self, tag: &str) -> Result<Vec<GrimoireSimulacrum>> {
        let all = self.load_all().await?;
        let tag_lower = tag.to_lowercase();

        Ok(all
            .into_iter()
            .filter(|s| {
                s.metadata.tags.as_ref().map_or(false, |tags| {
                    tags.iter().any(|t| t.to_lowercase() == tag_lower)
                })
            })
            .collect())
    }

    /// Clears the cache.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

// =============================================================================
// Workspace Configuration
// =============================================================================

/// Workspace configuration loaded from `.claude/` directory.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorkspaceConfig {
    /// Available skill IDs.
    pub skills: Vec<String>,
    /// Workspace root path.
    pub root_path: PathBuf,
    /// Custom hooks.
    pub hooks: HashMap<String, Vec<String>>,
}

impl WorkspaceConfig {
    /// Creates a new workspace config.
    #[must_use]
    pub fn new(root_path: impl Into<PathBuf>) -> Self {
        Self {
            skills: Vec::new(),
            root_path: root_path.into(),
            hooks: HashMap::new(),
        }
    }

    /// Returns the skills directory path.
    #[must_use]
    pub fn skills_path(&self) -> PathBuf {
        self.root_path.join(".claude").join("skills")
    }

    /// Creates a skill loader for this workspace.
    #[must_use]
    pub fn skill_loader(&self) -> SkillLoader {
        SkillLoader::new(self.skills_path())
    }
}

/// Loader for workspace configuration.
pub struct WorkspaceConfigLoader {
    root_path: PathBuf,
}

impl WorkspaceConfigLoader {
    /// Creates a new workspace config loader.
    #[must_use]
    pub fn new(root_path: impl Into<PathBuf>) -> Self {
        Self {
            root_path: root_path.into(),
        }
    }

    /// Attempts to find workspace root by looking for `.claude/` directory.
    ///
    /// Walks up from the given path until it finds `.claude/` or reaches root.
    #[must_use]
    pub fn find_workspace_root(start_path: impl AsRef<Path>) -> Option<PathBuf> {
        let mut current = start_path.as_ref().to_path_buf();

        loop {
            let claude_dir = current.join(".claude");
            if claude_dir.is_dir() && claude_dir.join("skills").is_dir() {
                return Some(current);
            }

            if !current.pop() {
                return None;
            }
        }
    }

    /// Loads the workspace configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration cannot be loaded.
    pub async fn load(&self) -> Result<WorkspaceConfig> {
        let claude_dir = self.root_path.join(".claude");

        if !claude_dir.exists() {
            return Ok(WorkspaceConfig::new(&self.root_path));
        }

        let mut config = WorkspaceConfig::new(&self.root_path);

        // Load available skills
        let skills_dir = claude_dir.join("skills");
        if skills_dir.exists() {
            let skill_loader = SkillLoader::new(&skills_dir);
            config.skills = skill_loader.list().await?;
        }

        // Load hooks from settings.json
        let settings_path = claude_dir.join("settings.json");
        if settings_path.exists() {
            if let Ok(content) = tokio::fs::read_to_string(&settings_path).await {
                if let Ok(settings) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(hooks) = settings.get("hooks").and_then(|h| h.as_object()) {
                        for (hook_name, hook_config) in hooks {
                            if let Some(hook_list) = hook_config.as_array() {
                                let commands: Vec<String> = hook_list
                                    .iter()
                                    .filter_map(|h| {
                                        h.get("hooks")
                                            .and_then(|arr| arr.as_array())
                                            .map(|arr| {
                                                arr.iter()
                                                    .filter_map(|cmd| {
                                                        cmd.get("command")
                                                            .and_then(|c| c.as_str())
                                                            .map(String::from)
                                                    })
                                                    .collect::<Vec<_>>()
                                            })
                                    })
                                    .flatten()
                                    .collect();
                                if !commands.is_empty() {
                                    config.hooks.insert(hook_name.clone(), commands);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(config)
    }

    /// Returns the root path.
    #[must_use]
    pub fn root_path(&self) -> &Path {
        &self.root_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;

    #[test]
    fn test_default_grimoire_path_from_env() {
        // Set env var
        let test_path = "/custom/grimoire/path";
        std::env::set_var(GRIMOIRE_PATH_ENV, test_path);

        let path = default_grimoire_path();
        assert_eq!(path, PathBuf::from(test_path));

        // Clean up
        std::env::remove_var(GRIMOIRE_PATH_ENV);
    }

    #[test]
    fn test_default_grimoire_path_fallback() {
        // Remove env var to test fallback
        std::env::remove_var(GRIMOIRE_PATH_ENV);

        let path = default_grimoire_path();
        assert!(path.to_string_lossy().contains("personas"));
    }

    #[test]
    fn test_grimoire_persona_serialization() {
        let persona = GrimoirePersona {
            id: "test-persona".to_string(),
            name: "Test Persona".to_string(),
            system_prompt: "You are a helpful assistant.".to_string(),
            variants: {
                let mut v = HashMap::new();
                v.insert("formal".to_string(), "You are a formal assistant.".to_string());
                v
            },
            metadata: {
                let mut m = HashMap::new();
                m.insert("version".to_string(), "1.0".to_string());
                m
            },
        };

        // Serialize
        let json = serde_json::to_string(&persona).expect("serialize");
        assert!(json.contains("test-persona"));
        assert!(json.contains("Test Persona"));
        assert!(json.contains("helpful assistant"));

        // Deserialize
        let parsed: GrimoirePersona = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.id, persona.id);
        assert_eq!(parsed.name, persona.name);
        assert_eq!(parsed.system_prompt, persona.system_prompt);
        assert_eq!(parsed.variants.get("formal"), persona.variants.get("formal"));
        assert_eq!(parsed.metadata.get("version"), persona.metadata.get("version"));
    }

    #[test]
    fn test_grimoire_loader_with_path() {
        let loader = GrimoireLoader::with_path("/custom/path");
        assert_eq!(loader.base_path(), Path::new("/custom/path"));
    }

    #[test]
    fn test_grimoire_loader_default() {
        std::env::remove_var(GRIMOIRE_PATH_ENV);
        let loader = GrimoireLoader::default();
        assert!(loader.base_path().to_string_lossy().contains("personas"));
    }

    #[test]
    fn test_grimoire_loader_clear_cache() {
        let loader = GrimoireLoader::with_path("/test");

        // Manually insert into cache
        let persona = GrimoirePersona {
            id: "cached".to_string(),
            name: "Cached Persona".to_string(),
            system_prompt: "System prompt".to_string(),
            variants: HashMap::new(),
            metadata: HashMap::new(),
        };
        loader.cache.insert("cached".to_string(), persona);

        assert!(!loader.cache.is_empty());
        loader.clear_cache();
        assert!(loader.cache.is_empty());
    }

    #[test]
    fn test_parse_persona() {
        let loader = GrimoireLoader::with_path("/test");
        let content = "You are a helpful assistant for code review.";

        let persona = loader.parse_persona("code-reviewer", content).expect("parse");

        assert_eq!(persona.id, "code-reviewer");
        assert_eq!(persona.name, "code-reviewer");
        assert_eq!(persona.system_prompt, content);
        assert!(persona.variants.is_empty());
        assert!(persona.metadata.is_empty());
    }

    #[tokio::test]
    async fn test_list_empty_directory() {
        let temp = TempDir::new().expect("temp dir");
        let loader = GrimoireLoader::with_path(temp.path());

        let personas = loader.list().await.expect("list");
        assert!(personas.is_empty());
    }

    #[tokio::test]
    async fn test_list_nonexistent_directory() {
        let loader = GrimoireLoader::with_path("/nonexistent/path/that/does/not/exist");

        let personas = loader.list().await.expect("list");
        assert!(personas.is_empty());
    }

    #[tokio::test]
    async fn test_list_with_markdown_files() {
        let temp = TempDir::new().expect("temp dir");

        // Create some persona files
        fs::write(temp.path().join("assistant.md"), "You are an assistant.").expect("write");
        fs::write(temp.path().join("reviewer.md"), "You are a code reviewer.").expect("write");
        fs::write(temp.path().join("other.txt"), "Not a persona").expect("write");

        let loader = GrimoireLoader::with_path(temp.path());
        let personas = loader.list().await.expect("list");

        assert_eq!(personas.len(), 2);
        assert!(personas.contains(&"assistant".to_string()));
        assert!(personas.contains(&"reviewer".to_string()));
    }

    #[tokio::test]
    async fn test_list_with_directories() {
        let temp = TempDir::new().expect("temp dir");

        // Create persona directories
        fs::create_dir(temp.path().join("complex-agent")).expect("mkdir");
        fs::write(
            temp.path().join("complex-agent").join("prompt.md"),
            "Complex agent prompt"
        ).expect("write");

        fs::create_dir(temp.path().join("simple-agent")).expect("mkdir");

        let loader = GrimoireLoader::with_path(temp.path());
        let personas = loader.list().await.expect("list");

        assert!(personas.contains(&"complex-agent".to_string()));
        assert!(personas.contains(&"simple-agent".to_string()));
    }

    #[tokio::test]
    async fn test_load_persona_from_file() {
        let temp = TempDir::new().expect("temp dir");

        let prompt = "You are a helpful coding assistant. Help users write better code.";
        fs::write(temp.path().join("coder.md"), prompt).expect("write");

        let loader = GrimoireLoader::with_path(temp.path());
        let persona = loader.load("coder").await.expect("load");

        assert_eq!(persona.id, "coder");
        assert_eq!(persona.system_prompt, prompt);
    }

    #[tokio::test]
    async fn test_load_persona_from_directory() {
        let temp = TempDir::new().expect("temp dir");

        let prompt = "You are a complex agent with multiple capabilities.";
        fs::create_dir(temp.path().join("complex")).expect("mkdir");
        fs::write(temp.path().join("complex").join("prompt.md"), prompt).expect("write");

        let loader = GrimoireLoader::with_path(temp.path());
        let persona = loader.load("complex").await.expect("load");

        assert_eq!(persona.id, "complex");
        assert_eq!(persona.system_prompt, prompt);
    }

    #[tokio::test]
    async fn test_load_caches_persona() {
        let temp = TempDir::new().expect("temp dir");

        fs::write(temp.path().join("cached-test.md"), "Cached prompt").expect("write");

        let loader = GrimoireLoader::with_path(temp.path());

        // First load should read from disk
        let persona1 = loader.load("cached-test").await.expect("load");
        assert_eq!(persona1.system_prompt, "Cached prompt");

        // Modify the file
        fs::write(temp.path().join("cached-test.md"), "Modified prompt").expect("write");

        // Second load should return cached version
        let persona2 = loader.load("cached-test").await.expect("load");
        assert_eq!(persona2.system_prompt, "Cached prompt"); // Still cached

        // Clear cache and reload
        loader.clear_cache();
        let persona3 = loader.load("cached-test").await.expect("load");
        assert_eq!(persona3.system_prompt, "Modified prompt"); // Now updated
    }

    #[tokio::test]
    async fn test_load_nonexistent_persona() {
        let temp = TempDir::new().expect("temp dir");
        let loader = GrimoireLoader::with_path(temp.path());

        let result = loader.load("nonexistent").await;
        assert!(result.is_err());
    }

    #[test]
    fn test_grimoire_persona_clone() {
        let persona = GrimoirePersona {
            id: "clone-test".to_string(),
            name: "Clone Test".to_string(),
            system_prompt: "Prompt".to_string(),
            variants: HashMap::new(),
            metadata: HashMap::new(),
        };

        let cloned = persona.clone();
        assert_eq!(cloned.id, persona.id);
        assert_eq!(cloned.name, persona.name);
    }

    #[test]
    fn test_grimoire_persona_debug() {
        let persona = GrimoirePersona {
            id: "debug-test".to_string(),
            name: "Debug Test".to_string(),
            system_prompt: "Prompt".to_string(),
            variants: HashMap::new(),
            metadata: HashMap::new(),
        };

        let debug_str = format!("{:?}", persona);
        assert!(debug_str.contains("debug-test"));
        assert!(debug_str.contains("Debug Test"));
    }

    #[test]
    fn test_grimoire_path_env_constant() {
        assert_eq!(GRIMOIRE_PATH_ENV, "INFERNUM_GRIMOIRE_PATH");
    }

    // ==========================================================================
    // Skill Tests
    // ==========================================================================

    #[test]
    fn test_code_template_creation() {
        let template = CodeTemplate {
            name: "Entity Template".to_string(),
            language: "kotlin".to_string(),
            content: "@Entity\nclass User {}".to_string(),
        };

        assert_eq!(template.name, "Entity Template");
        assert_eq!(template.language, "kotlin");
        assert!(template.content.contains("@Entity"));
    }

    #[test]
    fn test_grimoire_skill_new() {
        let skill = GrimoireSkill::new("kotlin", "Kotlin Code Generator");

        assert_eq!(skill.id, "kotlin");
        assert_eq!(skill.name, "Kotlin Code Generator");
        assert!(skill.description.is_empty());
        assert!(skill.triggers.is_empty());
        assert!(skill.templates.is_empty());
    }

    #[test]
    fn test_grimoire_skill_matches_trigger() {
        let mut skill = GrimoireSkill::new("kotlin", "Kotlin");
        skill.triggers = vec![
            "kotlin".to_string(),
            "kotlin entity".to_string(),
            "spring boot".to_string(),
        ];

        // Should match
        assert!(skill.matches_trigger("kotlin"));
        assert!(skill.matches_trigger("I need a kotlin entity"));
        assert!(skill.matches_trigger("Create a Spring Boot service"));
        assert!(skill.matches_trigger("KOTLIN")); // case insensitive

        // Should not match
        assert!(!skill.matches_trigger("python"));
        assert!(!skill.matches_trigger("java entity"));
    }

    #[test]
    fn test_grimoire_skill_as_agent_instruction() {
        let mut skill = GrimoireSkill::new("kotlin", "Kotlin Generator");
        skill.description = "Generate Kotlin code".to_string();
        skill.content = "## Templates\n```kotlin\ncode\n```".to_string();

        let instruction = skill.as_agent_instruction();

        assert!(instruction.contains("Kotlin Generator"));
        assert!(instruction.contains("Generate Kotlin code"));
        assert!(instruction.contains("Templates"));
    }

    #[test]
    fn test_grimoire_skill_get_template() {
        let mut skill = GrimoireSkill::new("kotlin", "Kotlin");
        skill.templates = vec![
            CodeTemplate {
                name: "Entity Template".to_string(),
                language: "kotlin".to_string(),
                content: "@Entity class".to_string(),
            },
            CodeTemplate {
                name: "Service Template".to_string(),
                language: "kotlin".to_string(),
                content: "@Service class".to_string(),
            },
        ];

        let entity = skill.get_template("entity template");
        assert!(entity.is_some());
        assert!(entity.unwrap().content.contains("@Entity"));

        let service = skill.get_template("Service Template");
        assert!(service.is_some());

        let missing = skill.get_template("Controller Template");
        assert!(missing.is_none());
    }

    #[test]
    fn test_grimoire_skill_get_templates_by_language() {
        let mut skill = GrimoireSkill::new("multi", "Multi-language");
        skill.templates = vec![
            CodeTemplate {
                name: "Kotlin Entity".to_string(),
                language: "kotlin".to_string(),
                content: "@Entity".to_string(),
            },
            CodeTemplate {
                name: "Kotlin Service".to_string(),
                language: "kotlin".to_string(),
                content: "@Service".to_string(),
            },
            CodeTemplate {
                name: "React Component".to_string(),
                language: "tsx".to_string(),
                content: "export const".to_string(),
            },
        ];

        let kotlin_templates = skill.get_templates_by_language("kotlin");
        assert_eq!(kotlin_templates.len(), 2);

        let tsx_templates = skill.get_templates_by_language("TSX");
        assert_eq!(tsx_templates.len(), 1);

        let rust_templates = skill.get_templates_by_language("rust");
        assert!(rust_templates.is_empty());
    }

    #[test]
    fn test_skill_loader_base_path() {
        let loader = SkillLoader::new("/custom/skills");
        assert_eq!(loader.base_path().to_string_lossy(), "/custom/skills");
    }

    #[tokio::test]
    async fn test_skill_loader_list_empty() {
        let temp = TempDir::new().expect("temp dir");
        let loader = SkillLoader::new(temp.path());

        let skills = loader.list().await.expect("list");
        assert!(skills.is_empty());
    }

    #[tokio::test]
    async fn test_skill_loader_list_skills() {
        let temp = TempDir::new().expect("temp dir");

        // Create skill directories with SKILL.md
        for name in &["kotlin", "react", "rust"] {
            let skill_dir = temp.path().join(name);
            fs::create_dir(&skill_dir).expect("mkdir");
            fs::write(skill_dir.join("SKILL.md"), "# Skill").expect("write");
        }

        // Create a directory without SKILL.md (should be ignored)
        fs::create_dir(temp.path().join("not-a-skill")).expect("mkdir");

        let loader = SkillLoader::new(temp.path());
        let skills = loader.list().await.expect("list");

        assert_eq!(skills.len(), 3);
        assert!(skills.contains(&"kotlin".to_string()));
        assert!(skills.contains(&"react".to_string()));
        assert!(skills.contains(&"rust".to_string()));
    }

    #[tokio::test]
    async fn test_skill_loader_load_skill() {
        let temp = TempDir::new().expect("temp dir");

        let skill_content = r#"---
name: kotlin
description: Generate Kotlin/Spring code.
triggers:
  - "kotlin"
  - "spring boot"
---

# Kotlin Skill

## Entity Template

```kotlin
@Entity
class User(
    @Id val id: Long,
    var name: String
)
```

## Service Template

```kotlin
@Service
class UserService(private val repo: UserRepository) {
    fun findById(id: Long) = repo.findById(id)
}
```
"#;

        let skill_dir = temp.path().join("kotlin");
        fs::create_dir(&skill_dir).expect("mkdir");
        fs::write(skill_dir.join("SKILL.md"), skill_content).expect("write");

        let loader = SkillLoader::new(temp.path());
        let skill = loader.load("kotlin").await.expect("load");

        assert_eq!(skill.id, "kotlin");
        assert_eq!(skill.name, "kotlin");
        assert_eq!(skill.description, "Generate Kotlin/Spring code.");
        assert_eq!(skill.triggers.len(), 2);
        assert!(skill.triggers.contains(&"kotlin".to_string()));
        assert!(skill.triggers.contains(&"spring boot".to_string()));

        // Should have extracted templates
        assert!(skill.templates.len() >= 2);
    }

    #[tokio::test]
    async fn test_skill_loader_template_extraction() {
        let temp = TempDir::new().expect("temp dir");

        let skill_content = r#"---
name: test
description: Test skill
triggers:
  - "test"
---

# Test Skill

## Entity Template

```kotlin
@Entity
@Table(name = "users")
class User(
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    val id: Long = 0,

    @Column(nullable = false)
    var name: String
)
```

## Service Template

```kotlin
@Service
@Transactional(readOnly = true)
class UserService(
    private val repository: UserRepository
) {
    fun findById(id: Long): User? = repository.findByIdOrNull(id)
}
```
"#;

        let skill_dir = temp.path().join("test");
        fs::create_dir(&skill_dir).expect("mkdir");
        fs::write(skill_dir.join("SKILL.md"), skill_content).expect("write");

        let loader = SkillLoader::new(temp.path());
        let skill = loader.load("test").await.expect("load");

        // Should have extracted both templates
        let entity_template = skill.get_template("Entity Template");
        assert!(entity_template.is_some());
        let entity = entity_template.unwrap();
        assert_eq!(entity.language, "kotlin");
        assert!(entity.content.contains("@Entity"));
        assert!(entity.content.contains("@Table"));

        let service_template = skill.get_template("Service Template");
        assert!(service_template.is_some());
        let service = service_template.unwrap();
        assert!(service.content.contains("@Service"));
        assert!(service.content.contains("@Transactional"));
    }

    #[tokio::test]
    async fn test_skill_loader_find_by_trigger() {
        let temp = TempDir::new().expect("temp dir");

        // Create kotlin skill
        let kotlin_dir = temp.path().join("kotlin");
        fs::create_dir(&kotlin_dir).expect("mkdir");
        fs::write(kotlin_dir.join("SKILL.md"), r#"---
name: kotlin
description: Kotlin code gen
triggers:
  - "kotlin"
  - "spring boot"
---
# Kotlin
"#).expect("write");

        // Create react skill
        let react_dir = temp.path().join("react");
        fs::create_dir(&react_dir).expect("mkdir");
        fs::write(react_dir.join("SKILL.md"), r#"---
name: react
description: React code gen
triggers:
  - "react"
  - "typescript component"
---
# React
"#).expect("write");

        let loader = SkillLoader::new(temp.path());

        // Find by "kotlin" trigger
        let kotlin_skills = loader.find_by_trigger("kotlin entity").await.expect("find");
        assert_eq!(kotlin_skills.len(), 1);
        assert_eq!(kotlin_skills[0].id, "kotlin");

        // Find by "spring boot" trigger
        let spring_skills = loader.find_by_trigger("spring boot service").await.expect("find");
        assert_eq!(spring_skills.len(), 1);

        // Find by "react" trigger
        let react_skills = loader.find_by_trigger("react component").await.expect("find");
        assert_eq!(react_skills.len(), 1);
        assert_eq!(react_skills[0].id, "react");

        // No match
        let python_skills = loader.find_by_trigger("python").await.expect("find");
        assert!(python_skills.is_empty());
    }

    #[tokio::test]
    async fn test_skill_loader_caching() {
        let temp = TempDir::new().expect("temp dir");

        let skill_dir = temp.path().join("cached");
        fs::create_dir(&skill_dir).expect("mkdir");
        fs::write(skill_dir.join("SKILL.md"), r#"---
name: cached
description: Original
triggers:
  - "cached"
---
# Original
"#).expect("write");

        let loader = SkillLoader::new(temp.path());

        // First load
        let skill1 = loader.load("cached").await.expect("load");
        assert_eq!(skill1.description, "Original");

        // Modify file
        fs::write(skill_dir.join("SKILL.md"), r#"---
name: cached
description: Modified
triggers:
  - "cached"
---
# Modified
"#).expect("write");

        // Should return cached version
        let skill2 = loader.load("cached").await.expect("load");
        assert_eq!(skill2.description, "Original");

        // Clear cache
        loader.clear_cache();

        // Now should return modified version
        let skill3 = loader.load("cached").await.expect("load");
        assert_eq!(skill3.description, "Modified");
    }

    #[test]
    fn test_skill_serialization() {
        let mut skill = GrimoireSkill::new("test", "Test Skill");
        skill.description = "A test skill".to_string();
        skill.triggers = vec!["test".to_string()];
        skill.templates = vec![CodeTemplate {
            name: "Template".to_string(),
            language: "rust".to_string(),
            content: "fn main() {}".to_string(),
        }];

        let json = serde_json::to_string(&skill).expect("serialize");
        assert!(json.contains("Test Skill"));
        assert!(json.contains("test skill"));

        let parsed: GrimoireSkill = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.id, skill.id);
        assert_eq!(parsed.name, skill.name);
        assert_eq!(parsed.templates.len(), 1);
    }

    // ==========================================================================
    // Workspace Config Tests
    // ==========================================================================

    #[test]
    fn test_workspace_config_new() {
        let config = WorkspaceConfig::new("/test/workspace");

        assert_eq!(config.root_path, PathBuf::from("/test/workspace"));
        assert!(config.skills.is_empty());
        assert!(config.hooks.is_empty());
    }

    #[test]
    fn test_workspace_config_skills_path() {
        let config = WorkspaceConfig::new("/test/workspace");
        let skills_path = config.skills_path();

        assert_eq!(skills_path, PathBuf::from("/test/workspace/.claude/skills"));
    }

    #[test]
    fn test_workspace_config_skill_loader() {
        let config = WorkspaceConfig::new("/test/workspace");
        let loader = config.skill_loader();

        assert_eq!(loader.base_path(), Path::new("/test/workspace/.claude/skills"));
    }

    #[test]
    fn test_workspace_config_loader_root_path() {
        let loader = WorkspaceConfigLoader::new("/test/workspace");
        assert_eq!(loader.root_path(), Path::new("/test/workspace"));
    }

    #[test]
    fn test_find_workspace_root() {
        let temp = TempDir::new().expect("temp dir");

        // Create .claude/skills directory
        let claude_dir = temp.path().join(".claude");
        fs::create_dir(&claude_dir).expect("mkdir");
        fs::create_dir(claude_dir.join("skills")).expect("mkdir skills");

        // Find from root
        let root = WorkspaceConfigLoader::find_workspace_root(temp.path());
        assert_eq!(root, Some(temp.path().to_path_buf()));

        // Create nested directory
        let nested = temp.path().join("src").join("components");
        fs::create_dir_all(&nested).expect("mkdir nested");

        // Find from nested path
        let root_from_nested = WorkspaceConfigLoader::find_workspace_root(&nested);
        assert_eq!(root_from_nested, Some(temp.path().to_path_buf()));
    }

    #[test]
    fn test_find_workspace_root_not_found() {
        let temp = TempDir::new().expect("temp dir");

        // No .claude directory
        let root = WorkspaceConfigLoader::find_workspace_root(temp.path());
        assert!(root.is_none());
    }

    #[tokio::test]
    async fn test_workspace_config_load_empty() {
        let temp = TempDir::new().expect("temp dir");
        let loader = WorkspaceConfigLoader::new(temp.path());

        let config = loader.load().await.expect("load");

        assert_eq!(config.root_path, temp.path());
        assert!(config.skills.is_empty());
        assert!(config.hooks.is_empty());
    }

    #[tokio::test]
    async fn test_workspace_config_load_with_skills() {
        let temp = TempDir::new().expect("temp dir");

        // Create .claude/skills directory with skills
        let skills_dir = temp.path().join(".claude").join("skills");
        fs::create_dir_all(&skills_dir).expect("mkdir");

        // Create skill directories
        for name in &["kotlin", "react", "rust"] {
            let skill_dir = skills_dir.join(name);
            fs::create_dir(&skill_dir).expect("mkdir skill");
            fs::write(skill_dir.join("SKILL.md"), "# Skill").expect("write");
        }

        let loader = WorkspaceConfigLoader::new(temp.path());
        let config = loader.load().await.expect("load");

        assert_eq!(config.skills.len(), 3);
        assert!(config.skills.contains(&"kotlin".to_string()));
        assert!(config.skills.contains(&"react".to_string()));
        assert!(config.skills.contains(&"rust".to_string()));
    }

    #[tokio::test]
    async fn test_workspace_config_load_with_hooks() {
        let temp = TempDir::new().expect("temp dir");

        // Create .claude directory
        let claude_dir = temp.path().join(".claude");
        fs::create_dir(&claude_dir).expect("mkdir");
        fs::create_dir(claude_dir.join("skills")).expect("mkdir skills");

        // Create settings.json with hooks
        let settings = r#"{
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            { "type": "command", "command": "./session-start.sh" }
                        ]
                    }
                ],
                "PreCommit": [
                    {
                        "hooks": [
                            { "type": "command", "command": "./pre-commit.sh" },
                            { "type": "command", "command": "./lint.sh" }
                        ]
                    }
                ]
            }
        }"#;
        fs::write(claude_dir.join("settings.json"), settings).expect("write");

        let loader = WorkspaceConfigLoader::new(temp.path());
        let config = loader.load().await.expect("load");

        // Check SessionStart hooks
        assert!(config.hooks.contains_key("SessionStart"));
        let session_hooks = config.hooks.get("SessionStart").unwrap();
        assert!(session_hooks.contains(&"./session-start.sh".to_string()));

        // Check PreCommit hooks
        assert!(config.hooks.contains_key("PreCommit"));
        let precommit_hooks = config.hooks.get("PreCommit").unwrap();
        assert_eq!(precommit_hooks.len(), 2);
    }

    #[test]
    fn test_workspace_config_serialization() {
        let mut config = WorkspaceConfig::new("/test");
        config.skills = vec!["kotlin".to_string(), "react".to_string()];
        config.hooks.insert("SessionStart".to_string(), vec!["./start.sh".to_string()]);

        let json = serde_json::to_string(&config).expect("serialize");
        assert!(json.contains("kotlin"));
        assert!(json.contains("react"));
        assert!(json.contains("SessionStart"));

        let parsed: WorkspaceConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.skills.len(), 2);
        assert_eq!(parsed.hooks.len(), 1);
    }

    // ==========================================================================
    // Simulacra Tests
    // ==========================================================================

    fn create_test_simulacrum() -> GrimoireSimulacrum {
        GrimoireSimulacrum {
            api_version: "simulacra/v1".to_string(),
            kind: "Simulacrum".to_string(),
            metadata: SimulacrumMetadata {
                code: "FIRST_TIME_USER".to_string(),
                name: "Chris".to_string(),
                version: 1,
                tags: Some(vec!["onboarding".to_string(), "baseline".to_string()]),
                archetype: None,
            },
            demographics: SimulacrumDemographics {
                age: 29,
                occupation: "Marketing coordinator".to_string(),
                location: Some("Urban apartment".to_string()),
                devices: DeviceProfile {
                    primary: "MacBook Air M2".to_string(),
                    secondary: Some("iPhone 14".to_string()),
                    ..Default::default()
                },
                connectivity: Some("Reliable home fiber".to_string()),
                locale: None,
            },
            cognition: SimulacrumCognition {
                tech_literacy: 6,
                patience: 7,
                attention_span: "medium".to_string(),
                reading_speed: Some("average".to_string()),
                multitasking: Some("low".to_string()),
            },
            temperament: SimulacrumTemperament {
                frustration_threshold: "medium".to_string(),
                error_tolerance: "medium".to_string(),
                exploration_tendency: "moderate".to_string(),
                help_seeking: Some("willing".to_string()),
            },
            accessibility: SimulacrumAccessibility {
                vision: VisionProfile {
                    acuity: "normal".to_string(),
                    color_blindness: "none".to_string(),
                    contrast_sensitivity: None,
                },
                motor: MotorProfile {
                    fine_control: "normal".to_string(),
                    preferred_input: "trackpad".to_string(),
                    click_accuracy: None,
                },
                cognitive: None,
                assistive_tech: Some(vec![]),
            },
            context: Some(SimulacrumContext {
                time_pressure: Some("low".to_string()),
                environment: Some("Quiet home office, evening".to_string()),
                motivation: Some("Curious to try it".to_string()),
                prior_experience: Some("None".to_string()),
            }),
            behavior: SimulacrumBehavior {
                max_clicks_before_frustration: 8,
                max_seconds_waiting: 10,
                scroll_behavior: "normal".to_string(),
                reads_labels: "usually".to_string(),
                uses_search: Some("sometimes".to_string()),
                keyboard_shortcuts: Some("occasionally".to_string()),
                navigation_method: None,
            },
            voice: SimulacrumVoice {
                tone: "Curious, open-minded".to_string(),
                verbosity: "moderate".to_string(),
                feedback_style: "Thoughtful".to_string(),
                example_phrases: vec![
                    "Okay, so what's this for exactly?".to_string(),
                    "That's interesting...".to_string(),
                ],
            },
        }
    }

    fn create_screen_reader_simulacrum() -> GrimoireSimulacrum {
        GrimoireSimulacrum {
            api_version: "simulacra/v1".to_string(),
            kind: "Simulacrum".to_string(),
            metadata: SimulacrumMetadata {
                code: "SCREEN_READER_USER".to_string(),
                name: "David".to_string(),
                version: 1,
                tags: Some(vec!["accessibility".to_string(), "a11y".to_string(), "blind".to_string()]),
                archetype: None,
            },
            demographics: SimulacrumDemographics {
                age: 45,
                occupation: "Software architect".to_string(),
                location: None,
                devices: DeviceProfile {
                    primary: "MacBook Pro with VoiceOver".to_string(),
                    ..Default::default()
                },
                ..Default::default()
            },
            cognition: SimulacrumCognition {
                tech_literacy: 9,
                patience: 8,
                attention_span: "high".to_string(),
                ..Default::default()
            },
            temperament: SimulacrumTemperament {
                frustration_threshold: "medium".to_string(),
                error_tolerance: "high".to_string(),
                exploration_tendency: "high".to_string(),
                help_seeking: None,
            },
            accessibility: SimulacrumAccessibility {
                vision: VisionProfile {
                    acuity: "blind".to_string(),
                    color_blindness: "n/a".to_string(),
                    contrast_sensitivity: None,
                },
                motor: MotorProfile {
                    fine_control: "normal".to_string(),
                    preferred_input: "keyboard".to_string(),
                    click_accuracy: None,
                },
                cognitive: None,
                assistive_tech: Some(vec![
                    "VoiceOver".to_string(),
                    "keyboard_navigation".to_string(),
                ]),
            },
            context: None,
            behavior: SimulacrumBehavior {
                max_clicks_before_frustration: 15,
                max_seconds_waiting: 20,
                scroll_behavior: "via_keyboard".to_string(),
                reads_labels: "always_via_screen_reader".to_string(),
                uses_search: Some("frequently".to_string()),
                keyboard_shortcuts: Some("expert".to_string()),
                navigation_method: Some("headings_and_landmarks".to_string()),
            },
            voice: SimulacrumVoice {
                tone: "Patient but thorough".to_string(),
                verbosity: "detailed".to_string(),
                feedback_style: "Precise, technical".to_string(),
                example_phrases: vec![
                    "Navigating by headings...".to_string(),
                    "This button has no accessible name".to_string(),
                ],
            },
        }
    }

    #[test]
    fn test_simulacrum_code_and_name() {
        let sim = create_test_simulacrum();

        assert_eq!(sim.code(), "FIRST_TIME_USER");
        assert_eq!(sim.name(), "Chris");
    }

    #[test]
    fn test_simulacrum_archetype_inference() {
        // Tech literacy 6 -> average-user
        let sim = create_test_simulacrum();
        assert_eq!(sim.archetype(), "average-user");

        // Tech literacy 9 -> power-user
        let power_user = create_screen_reader_simulacrum();
        assert_eq!(power_user.archetype(), "power-user");

        // Tech literacy 3 -> novice
        let mut novice = create_test_simulacrum();
        novice.cognition.tech_literacy = 3;
        assert_eq!(novice.archetype(), "novice");

        // Explicit archetype overrides inference
        let mut explicit = create_test_simulacrum();
        explicit.metadata.archetype = Some("designer".to_string());
        assert_eq!(explicit.archetype(), "designer");
    }

    #[test]
    fn test_simulacrum_needs_accessibility() {
        // Normal user - no accessibility needs
        let normal = create_test_simulacrum();
        assert!(!normal.needs_accessibility());

        // Screen reader user - has accessibility needs
        let screen_reader = create_screen_reader_simulacrum();
        assert!(screen_reader.needs_accessibility());

        // User with color blindness
        let mut color_blind = create_test_simulacrum();
        color_blind.accessibility.vision.color_blindness = "deuteranopia".to_string();
        assert!(color_blind.needs_accessibility());

        // User with motor challenges
        let mut motor = create_test_simulacrum();
        motor.accessibility.motor.fine_control = "reduced".to_string();
        assert!(motor.needs_accessibility());
    }

    #[test]
    fn test_simulacrum_frustration_risk() {
        let sim = create_test_simulacrum();
        // max_clicks: 8, max_seconds: 10, patience: 7

        // Low risk task (2 clicks, 2 seconds)
        let low_risk = sim.calculate_frustration_risk(2, 2);
        assert!(low_risk < 0.5, "Expected low risk, got {}", low_risk);

        // Medium risk task (6 clicks, 8 seconds)
        let medium_risk = sim.calculate_frustration_risk(6, 8);
        assert!(medium_risk > 0.3 && medium_risk < 0.8, "Expected medium risk, got {}", medium_risk);

        // High risk task (15 clicks, 20 seconds)
        let high_risk = sim.calculate_frustration_risk(15, 20);
        assert!(high_risk > 0.7, "Expected high risk, got {}", high_risk);
    }

    #[test]
    fn test_simulacrum_frustration_risk_patience_factor() {
        let mut impatient = create_test_simulacrum();
        impatient.cognition.patience = 2; // Very impatient

        let mut patient = create_test_simulacrum();
        patient.cognition.patience = 9; // Very patient

        // Same task
        let impatient_risk = impatient.calculate_frustration_risk(5, 5);
        let patient_risk = patient.calculate_frustration_risk(5, 5);

        assert!(impatient_risk > patient_risk,
            "Impatient risk {} should be > patient risk {}", impatient_risk, patient_risk);
    }

    #[test]
    fn test_simulacrum_to_system_prompt() {
        let sim = create_test_simulacrum();
        let prompt = sim.to_system_prompt();

        // Check key sections are present
        assert!(prompt.contains("Chris"));
        assert!(prompt.contains("FIRST_TIME_USER"));
        assert!(prompt.contains("average-user"));
        assert!(prompt.contains("## Demographics"));
        assert!(prompt.contains("Age: 29"));
        assert!(prompt.contains("Marketing coordinator"));
        assert!(prompt.contains("## Cognitive Profile"));
        assert!(prompt.contains("Tech Literacy: 6/10"));
        assert!(prompt.contains("## Behavior"));
        assert!(prompt.contains("Max Clicks Before Frustration: 8"));
        assert!(prompt.contains("## Voice"));
        assert!(prompt.contains("Curious, open-minded"));
    }

    #[test]
    fn test_simulacrum_to_system_prompt_accessibility() {
        let sim = create_screen_reader_simulacrum();
        let prompt = sim.to_system_prompt();

        // Should include accessibility section
        assert!(prompt.contains("## Accessibility"));
        assert!(prompt.contains("Vision: blind"));
        assert!(prompt.contains("VoiceOver"));
    }

    #[test]
    fn test_simulacrum_serialization() {
        let sim = create_test_simulacrum();

        // Serialize to JSON
        let json = serde_json::to_string(&sim).expect("serialize");
        assert!(json.contains("FIRST_TIME_USER"));
        assert!(json.contains("Chris"));
        assert!(json.contains("simulacra/v1"));

        // Deserialize from JSON
        let parsed: GrimoireSimulacrum = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.code(), sim.code());
        assert_eq!(parsed.name(), sim.name());
        assert_eq!(parsed.cognition.tech_literacy, sim.cognition.tech_literacy);
    }

    #[test]
    fn test_simulacrum_yaml_parsing() {
        let yaml = r#"
apiVersion: simulacra/v1
kind: Simulacrum

metadata:
  code: POWER_USER
  name: Alex
  version: 1
  tags: [power-user, developer]

demographics:
  age: 32
  occupation: "Software engineer"
  devices:
    primary: "Linux workstation"

cognition:
  tech_literacy: 9
  patience: 5
  attention_span: high

temperament:
  frustration_threshold: low
  error_tolerance: high
  exploration_tendency: high

accessibility:
  vision:
    acuity: normal
    color_blindness: none
  motor:
    fine_control: excellent
    preferred_input: keyboard

behavior:
  max_clicks_before_frustration: 20
  max_seconds_waiting: 5
  scroll_behavior: minimal
  reads_labels: scans

voice:
  tone: "Direct, efficient"
  verbosity: low
  feedback_style: "Terse, technical"
  example_phrases:
    - "Where's the keyboard shortcut for this?"
    - "This should be faster"
"#;

        let sim: GrimoireSimulacrum = serde_yaml::from_str(yaml).expect("parse yaml");

        assert_eq!(sim.code(), "POWER_USER");
        assert_eq!(sim.name(), "Alex");
        assert_eq!(sim.cognition.tech_literacy, 9);
        assert_eq!(sim.archetype(), "power-user");
        assert_eq!(sim.behavior.max_seconds_waiting, 5);
        assert!(sim.voice.example_phrases.len() >= 2);
    }

    #[test]
    fn test_simulacrum_loader_base_path() {
        let loader = SimulacrumLoader::new("/custom/simulacra");
        assert_eq!(loader.base_path().to_string_lossy(), "/custom/simulacra");
    }

    #[tokio::test]
    async fn test_simulacrum_loader_list_empty() {
        let temp = TempDir::new().expect("temp dir");
        let loader = SimulacrumLoader::new(temp.path());

        let simulacra = loader.list().await.expect("list");
        assert!(simulacra.is_empty());
    }

    #[tokio::test]
    async fn test_simulacrum_loader_list() {
        let temp = TempDir::new().expect("temp dir");

        // Create simulacrum directories with manifest.yaml
        for name in &["first-time-user", "power-user", "screen-reader-user"] {
            let sim_dir = temp.path().join(name);
            fs::create_dir(&sim_dir).expect("mkdir");
            fs::write(sim_dir.join("manifest.yaml"), "apiVersion: simulacra/v1\nkind: Simulacrum\nmetadata:\n  code: TEST\n  name: Test\n  version: 1\ndemographics:\n  age: 30\n  occupation: Test\n  devices:\n    primary: Test\ncognition:\n  tech_literacy: 5\n  patience: 5\n  attention_span: medium\ntemperament:\n  frustration_threshold: medium\n  error_tolerance: medium\n  exploration_tendency: medium\naccessibility:\n  vision:\n    acuity: normal\n    color_blindness: none\n  motor:\n    fine_control: normal\n    preferred_input: mouse\nbehavior:\n  max_clicks_before_frustration: 10\n  max_seconds_waiting: 10\n  scroll_behavior: normal\n  reads_labels: usually\nvoice:\n  tone: Neutral\n  verbosity: medium\n  feedback_style: Normal\n  example_phrases:\n    - Test phrase").expect("write");
        }

        // Create a directory without manifest.yaml (should be ignored)
        fs::create_dir(temp.path().join("not-a-simulacrum")).expect("mkdir");

        let loader = SimulacrumLoader::new(temp.path());
        let codes = loader.list().await.expect("list");

        assert_eq!(codes.len(), 3);
        assert!(codes.contains(&"FIRST_TIME_USER".to_string()));
        assert!(codes.contains(&"POWER_USER".to_string()));
        assert!(codes.contains(&"SCREEN_READER_USER".to_string()));
    }

    #[tokio::test]
    async fn test_simulacrum_loader_load() {
        let temp = TempDir::new().expect("temp dir");

        let yaml = r#"apiVersion: simulacra/v1
kind: Simulacrum

metadata:
  code: FIRST_TIME_USER
  name: Chris
  version: 1
  tags: [onboarding, baseline]

demographics:
  age: 29
  occupation: "Marketing coordinator"
  devices:
    primary: "MacBook Air M2"

cognition:
  tech_literacy: 6
  patience: 7
  attention_span: medium

temperament:
  frustration_threshold: medium
  error_tolerance: medium
  exploration_tendency: moderate

accessibility:
  vision:
    acuity: normal
    color_blindness: none
  motor:
    fine_control: normal
    preferred_input: trackpad

behavior:
  max_clicks_before_frustration: 8
  max_seconds_waiting: 10
  scroll_behavior: normal
  reads_labels: usually

voice:
  tone: "Curious, open-minded"
  verbosity: moderate
  feedback_style: Thoughtful
  example_phrases:
    - "Okay, so what's this for exactly?"
"#;

        let sim_dir = temp.path().join("first-time-user");
        fs::create_dir(&sim_dir).expect("mkdir");
        fs::write(sim_dir.join("manifest.yaml"), yaml).expect("write");

        let loader = SimulacrumLoader::new(temp.path());
        let sim = loader.load("FIRST_TIME_USER").await.expect("load");

        assert_eq!(sim.code(), "FIRST_TIME_USER");
        assert_eq!(sim.name(), "Chris");
        assert_eq!(sim.demographics.age, 29);
        assert_eq!(sim.cognition.tech_literacy, 6);
        assert_eq!(sim.archetype(), "average-user");
    }

    #[tokio::test]
    async fn test_simulacrum_loader_caching() {
        let temp = TempDir::new().expect("temp dir");

        let yaml = r#"apiVersion: simulacra/v1
kind: Simulacrum
metadata:
  code: CACHED_USER
  name: Cached
  version: 1
demographics:
  age: 30
  occupation: Test
  devices:
    primary: Test
cognition:
  tech_literacy: 5
  patience: 5
  attention_span: medium
temperament:
  frustration_threshold: medium
  error_tolerance: medium
  exploration_tendency: medium
accessibility:
  vision:
    acuity: normal
    color_blindness: none
  motor:
    fine_control: normal
    preferred_input: mouse
behavior:
  max_clicks_before_frustration: 10
  max_seconds_waiting: 10
  scroll_behavior: normal
  reads_labels: usually
voice:
  tone: Neutral
  verbosity: medium
  feedback_style: Normal
  example_phrases:
    - Original
"#;

        let sim_dir = temp.path().join("cached-user");
        fs::create_dir(&sim_dir).expect("mkdir");
        fs::write(sim_dir.join("manifest.yaml"), yaml).expect("write");

        let loader = SimulacrumLoader::new(temp.path());

        // First load
        let sim1 = loader.load("CACHED_USER").await.expect("load");
        assert!(sim1.voice.example_phrases.contains(&"Original".to_string()));

        // Modify file
        let modified_yaml = yaml.replace("Original", "Modified");
        fs::write(sim_dir.join("manifest.yaml"), modified_yaml).expect("write");

        // Second load should return cached
        let sim2 = loader.load("CACHED_USER").await.expect("load");
        assert!(sim2.voice.example_phrases.contains(&"Original".to_string()));

        // Clear cache
        loader.clear_cache();

        // Third load should return modified
        let sim3 = loader.load("CACHED_USER").await.expect("load");
        assert!(sim3.voice.example_phrases.contains(&"Modified".to_string()));
    }

    #[tokio::test]
    async fn test_simulacrum_loader_find_by_tag() {
        let temp = TempDir::new().expect("temp dir");

        // Create simulacrum with accessibility tag
        let a11y_yaml = r#"apiVersion: simulacra/v1
kind: Simulacrum
metadata:
  code: SCREEN_READER_USER
  name: David
  version: 1
  tags: [accessibility, a11y, blind]
demographics:
  age: 45
  occupation: Architect
  devices:
    primary: MacBook
cognition:
  tech_literacy: 9
  patience: 8
  attention_span: high
temperament:
  frustration_threshold: medium
  error_tolerance: high
  exploration_tendency: high
accessibility:
  vision:
    acuity: blind
    color_blindness: n/a
  motor:
    fine_control: normal
    preferred_input: keyboard
  assistive_tech:
    - VoiceOver
behavior:
  max_clicks_before_frustration: 15
  max_seconds_waiting: 20
  scroll_behavior: via_keyboard
  reads_labels: always
voice:
  tone: Patient
  verbosity: detailed
  feedback_style: Technical
  example_phrases:
    - Test
"#;

        let sim_dir = temp.path().join("screen-reader-user");
        fs::create_dir(&sim_dir).expect("mkdir");
        fs::write(sim_dir.join("manifest.yaml"), a11y_yaml).expect("write");

        // Create simulacrum without accessibility tag
        let normal_yaml = r#"apiVersion: simulacra/v1
kind: Simulacrum
metadata:
  code: NORMAL_USER
  name: Normal
  version: 1
  tags: [baseline]
demographics:
  age: 30
  occupation: Office
  devices:
    primary: Laptop
cognition:
  tech_literacy: 5
  patience: 5
  attention_span: medium
temperament:
  frustration_threshold: medium
  error_tolerance: medium
  exploration_tendency: medium
accessibility:
  vision:
    acuity: normal
    color_blindness: none
  motor:
    fine_control: normal
    preferred_input: mouse
behavior:
  max_clicks_before_frustration: 10
  max_seconds_waiting: 10
  scroll_behavior: normal
  reads_labels: usually
voice:
  tone: Neutral
  verbosity: medium
  feedback_style: Normal
  example_phrases:
    - Test
"#;

        let sim_dir2 = temp.path().join("normal-user");
        fs::create_dir(&sim_dir2).expect("mkdir");
        fs::write(sim_dir2.join("manifest.yaml"), normal_yaml).expect("write");

        let loader = SimulacrumLoader::new(temp.path());

        // Find by accessibility tag
        let a11y_sims = loader.find_by_tag("accessibility").await.expect("find");
        assert_eq!(a11y_sims.len(), 1);
        assert_eq!(a11y_sims[0].code(), "SCREEN_READER_USER");

        // Find by a11y tag
        let a11y_sims2 = loader.find_by_tag("a11y").await.expect("find");
        assert_eq!(a11y_sims2.len(), 1);

        // Find by baseline tag
        let baseline_sims = loader.find_by_tag("baseline").await.expect("find");
        assert_eq!(baseline_sims.len(), 1);
        assert_eq!(baseline_sims[0].code(), "NORMAL_USER");

        // Find non-existent tag
        let empty = loader.find_by_tag("enterprise").await.expect("find");
        assert!(empty.is_empty());
    }
}
