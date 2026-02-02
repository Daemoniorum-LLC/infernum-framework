//! Prompt Studio - Version-controlled prompt engineering with A/B testing.
//!
//! The Prompt Studio provides:
//! - Version-controlled prompt templates
//! - Variable interpolation and validation
//! - A/B testing framework
//! - Performance analytics per prompt version

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{info, info_span};
use uuid::Uuid;

use crate::persistence::StudioDatabase;

/// Errors that can occur in the prompt studio.
#[derive(Debug, Error)]
pub enum PromptError {
    /// Template not found.
    #[error("Template not found: {0}")]
    NotFound(String),

    /// Version not found.
    #[error("Version not found: {0}")]
    VersionNotFound(String),

    /// Invalid template syntax.
    #[error("Invalid template syntax: {0}")]
    InvalidSyntax(String),

    /// Missing variable.
    #[error("Missing variable: {0}")]
    MissingVariable(String),

    /// Test not found.
    #[error("Test not found: {0}")]
    TestNotFound(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Result type for prompt operations.
pub type Result<T> = std::result::Result<T, PromptError>;

/// A prompt template with versioning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    /// Unique identifier.
    pub id: String,
    /// Template name.
    pub name: String,
    /// Description.
    pub description: Option<String>,
    /// All versions of this template.
    pub versions: Vec<PromptVersion>,
    /// Currently active version ID.
    pub active_version_id: Option<String>,
    /// Tags for categorization.
    pub tags: Vec<String>,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp.
    pub updated_at: DateTime<Utc>,
}

impl PromptTemplate {
    /// Creates a new prompt template.
    pub fn new(name: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.into(),
            description: None,
            versions: Vec::new(),
            active_version_id: None,
            tags: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Sets the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Adds tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Creates a new version of this template.
    pub fn create_version(&mut self, content: impl Into<String>, message: impl Into<String>) -> &PromptVersion {
        let mut version = PromptVersion::new(content, message);
        // Assign the next version number (1-indexed)
        version.version_number = self.versions.len() as u32 + 1;
        let is_first = self.versions.is_empty();

        self.versions.push(version);
        self.updated_at = Utc::now();

        // Auto-activate first version
        if is_first {
            self.active_version_id = Some(self.versions[0].id.clone());
        }

        self.versions.last().expect("just pushed")
    }

    /// Gets the active version.
    pub fn active_version(&self) -> Option<&PromptVersion> {
        self.active_version_id.as_ref()
            .and_then(|id| self.versions.iter().find(|v| &v.id == id))
    }

    /// Gets a version by ID.
    pub fn get_version(&self, version_id: &str) -> Option<&PromptVersion> {
        self.versions.iter().find(|v| v.id == version_id)
    }

    /// Sets the active version.
    pub fn set_active_version(&mut self, version_id: &str) -> Result<()> {
        if !self.versions.iter().any(|v| v.id == version_id) {
            return Err(PromptError::VersionNotFound(version_id.to_string()));
        }
        self.active_version_id = Some(version_id.to_string());
        self.updated_at = Utc::now();
        Ok(())
    }

    /// Renders the active version with variables.
    pub fn render(&self, variables: &HashMap<String, String>) -> Result<String> {
        let version = self.active_version()
            .ok_or_else(|| PromptError::NotFound("No active version".to_string()))?;
        version.render(variables)
    }

    /// Gets all variable names in the active version.
    pub fn variables(&self) -> Vec<String> {
        self.active_version()
            .map(|v| v.variables())
            .unwrap_or_default()
    }
}

/// A specific version of a prompt template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptVersion {
    /// Version ID.
    pub id: String,
    /// Version number (auto-incremented).
    pub version_number: u32,
    /// The prompt content with {{variable}} placeholders.
    pub content: String,
    /// System prompt (if separate).
    pub system_prompt: Option<String>,
    /// Change message.
    pub message: String,
    /// Author.
    pub author: Option<String>,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Performance metrics for this version.
    pub metrics: VersionMetrics,
}

impl PromptVersion {
    /// Creates a new version.
    pub fn new(content: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            version_number: 1, // Will be set by template
            content: content.into(),
            system_prompt: None,
            message: message.into(),
            author: None,
            created_at: Utc::now(),
            metrics: VersionMetrics::default(),
        }
    }

    /// Sets the system prompt.
    pub fn with_system_prompt(mut self, system: impl Into<String>) -> Self {
        self.system_prompt = Some(system.into());
        self
    }

    /// Sets the author.
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Renders the prompt with variables.
    pub fn render(&self, variables: &HashMap<String, String>) -> Result<String> {
        let mut result = self.content.clone();

        // Find all variables in the template
        let required_vars = self.variables();

        // Check all required variables are provided
        for var in &required_vars {
            if !variables.contains_key(var) {
                return Err(PromptError::MissingVariable(var.clone()));
            }
        }

        // Replace variables
        for (key, value) in variables {
            let placeholder = format!("{{{{{}}}}}", key);
            result = result.replace(&placeholder, value);
        }

        Ok(result)
    }

    /// Extracts variable names from the template.
    pub fn variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        let content = &self.content;

        // Find all {{variable}} patterns using simple string matching
        let mut i = 0;
        let bytes = content.as_bytes();
        while i < bytes.len().saturating_sub(3) {
            // Look for "{{"
            if bytes[i] == b'{' && bytes.get(i + 1) == Some(&b'{') {
                // Find the closing "}}"
                let start = i + 2;
                let mut end = start;
                while end < bytes.len().saturating_sub(1) {
                    if bytes[end] == b'}' && bytes.get(end + 1) == Some(&b'}') {
                        // Extract variable name
                        if end > start {
                            if let Ok(var_name) = std::str::from_utf8(&bytes[start..end]) {
                                vars.push(var_name.to_string());
                            }
                        }
                        i = end + 2;
                        break;
                    }
                    end += 1;
                }
                if end >= bytes.len().saturating_sub(1) {
                    break;
                }
            } else {
                i += 1;
            }
        }

        vars.sort();
        vars.dedup();
        vars
    }

    /// Records a usage of this version.
    pub fn record_usage(&mut self, latency_ms: u64, success: bool, quality_score: Option<f32>) {
        self.metrics.total_uses += 1;
        if success {
            self.metrics.successful_uses += 1;
        }
        self.metrics.total_latency_ms += latency_ms;
        if let Some(score) = quality_score {
            self.metrics.quality_scores.push(score);
        }
    }
}

/// Performance metrics for a prompt version.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VersionMetrics {
    /// Total times used.
    pub total_uses: u64,
    /// Successful uses.
    pub successful_uses: u64,
    /// Total latency in milliseconds.
    pub total_latency_ms: u64,
    /// Quality scores from evaluations.
    pub quality_scores: Vec<f32>,
}

impl VersionMetrics {
    /// Calculates success rate.
    pub fn success_rate(&self) -> f32 {
        if self.total_uses == 0 {
            return 0.0;
        }
        self.successful_uses as f32 / self.total_uses as f32
    }

    /// Calculates average latency.
    pub fn avg_latency_ms(&self) -> f64 {
        if self.total_uses == 0 {
            return 0.0;
        }
        self.total_latency_ms as f64 / self.total_uses as f64
    }

    /// Calculates average quality score.
    pub fn avg_quality_score(&self) -> Option<f32> {
        if self.quality_scores.is_empty() {
            return None;
        }
        Some(self.quality_scores.iter().sum::<f32>() / self.quality_scores.len() as f32)
    }
}

/// Result of an A/B test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Test ID.
    pub test_id: String,
    /// Test name.
    pub name: String,
    /// Version A ID.
    pub version_a_id: String,
    /// Version B ID.
    pub version_b_id: String,
    /// Results for version A.
    pub version_a_results: TestVersionResults,
    /// Results for version B.
    pub version_b_results: TestVersionResults,
    /// Winner (if determined).
    pub winner: Option<String>,
    /// Statistical significance.
    pub significance: Option<f32>,
    /// Test status.
    pub status: TestStatus,
    /// Start timestamp.
    pub started_at: DateTime<Utc>,
    /// End timestamp.
    pub ended_at: Option<DateTime<Utc>>,
}

impl TestResult {
    /// Creates a new test result.
    pub fn new(name: impl Into<String>, version_a_id: String, version_b_id: String) -> Self {
        Self {
            test_id: Uuid::new_v4().to_string(),
            name: name.into(),
            version_a_id,
            version_b_id,
            version_a_results: TestVersionResults::default(),
            version_b_results: TestVersionResults::default(),
            winner: None,
            significance: None,
            status: TestStatus::Running,
            started_at: Utc::now(),
            ended_at: None,
        }
    }

    /// Records a result for version A.
    pub fn record_a(&mut self, success: bool, quality_score: f32, latency_ms: u64) {
        self.version_a_results.record(success, quality_score, latency_ms);
    }

    /// Records a result for version B.
    pub fn record_b(&mut self, success: bool, quality_score: f32, latency_ms: u64) {
        self.version_b_results.record(success, quality_score, latency_ms);
    }

    /// Determines the winner based on quality scores.
    pub fn determine_winner(&mut self, min_samples: usize, significance_threshold: f32) {
        let a_samples = self.version_a_results.quality_scores.len();
        let b_samples = self.version_b_results.quality_scores.len();

        if a_samples < min_samples || b_samples < min_samples {
            return;
        }

        let a_mean = self.version_a_results.avg_quality();
        let b_mean = self.version_b_results.avg_quality();

        // Simple statistical test (would use proper t-test in production)
        let diff = (a_mean - b_mean).abs();
        let pooled_std = ((self.version_a_results.quality_variance()
            + self.version_b_results.quality_variance()) / 2.0).sqrt();

        if pooled_std > 0.0 {
            let effect_size = diff / pooled_std;
            self.significance = Some(effect_size);

            if effect_size > significance_threshold {
                self.winner = Some(if a_mean > b_mean {
                    self.version_a_id.clone()
                } else {
                    self.version_b_id.clone()
                });
                self.status = TestStatus::Completed;
                self.ended_at = Some(Utc::now());
            }
        }
    }
}

/// Results for one version in an A/B test.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TestVersionResults {
    /// Total trials.
    pub trials: u64,
    /// Successful trials.
    pub successes: u64,
    /// Quality scores.
    pub quality_scores: Vec<f32>,
    /// Latencies in ms.
    pub latencies_ms: Vec<u64>,
}

impl TestVersionResults {
    /// Records a trial result.
    pub fn record(&mut self, success: bool, quality_score: f32, latency_ms: u64) {
        self.trials += 1;
        if success {
            self.successes += 1;
        }
        self.quality_scores.push(quality_score);
        self.latencies_ms.push(latency_ms);
    }

    /// Average quality score.
    pub fn avg_quality(&self) -> f32 {
        if self.quality_scores.is_empty() {
            return 0.0;
        }
        self.quality_scores.iter().sum::<f32>() / self.quality_scores.len() as f32
    }

    /// Quality score variance.
    pub fn quality_variance(&self) -> f32 {
        if self.quality_scores.len() < 2 {
            return 0.0;
        }
        let mean = self.avg_quality();
        let sum_sq: f32 = self.quality_scores.iter()
            .map(|x| (x - mean).powi(2))
            .sum();
        sum_sq / (self.quality_scores.len() - 1) as f32
    }

    /// Success rate.
    pub fn success_rate(&self) -> f32 {
        if self.trials == 0 {
            return 0.0;
        }
        self.successes as f32 / self.trials as f32
    }

    /// Average latency.
    pub fn avg_latency_ms(&self) -> f64 {
        if self.latencies_ms.is_empty() {
            return 0.0;
        }
        self.latencies_ms.iter().sum::<u64>() as f64 / self.latencies_ms.len() as f64
    }
}

/// Status of an A/B test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestStatus {
    /// Test is running.
    Running,
    /// Test completed with a winner.
    Completed,
    /// Test cancelled.
    Cancelled,
    /// Test ended without significant results.
    Inconclusive,
}

/// The Prompt Studio manages all prompt templates and tests.
pub struct PromptStudio {
    /// All templates.
    templates: Arc<RwLock<HashMap<String, PromptTemplate>>>,
    /// Active A/B tests.
    tests: Arc<RwLock<HashMap<String, TestResult>>>,
    /// Storage directory.
    storage_dir: PathBuf,
    /// Optional database for persistence.
    db: Option<Arc<StudioDatabase>>,
}

impl PromptStudio {
    /// Creates a new Prompt Studio.
    pub fn new(storage_dir: impl Into<PathBuf>) -> Self {
        Self {
            templates: Arc::new(RwLock::new(HashMap::new())),
            tests: Arc::new(RwLock::new(HashMap::new())),
            storage_dir: storage_dir.into(),
            db: None,
        }
    }

    /// Creates a Prompt Studio with SQLite persistence.
    pub fn with_database(storage_dir: impl Into<PathBuf>, db: Arc<StudioDatabase>) -> Self {
        Self {
            templates: Arc::new(RwLock::new(HashMap::new())),
            tests: Arc::new(RwLock::new(HashMap::new())),
            storage_dir: storage_dir.into(),
            db: Some(db),
        }
    }

    /// Returns the number of prompt templates.
    pub async fn count(&self) -> usize {
        if let Some(ref db) = self.db {
            db.count_prompt_templates().unwrap_or(0)
        } else {
            self.templates.read().len()
        }
    }

    /// Creates a new template.
    pub async fn create_template(&self, name: impl Into<String>) -> Result<PromptTemplate> {
        let _span = info_span!("prompt_studio.create_template").entered();

        let template = PromptTemplate::new(name);
        let id = template.id.clone();

        info!(template_id = %id, name = %template.name, "Created template");

        // Save to database if available
        if let Some(ref db) = self.db {
            db.save_prompt_template(&template)
                .map_err(|e| PromptError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
        } else {
            // Save to disk
            self.save_template_to_disk(&template).await?;
            self.templates.write().insert(id, template.clone());
        }

        Ok(template)
    }

    /// Saves a template to disk.
    async fn save_template_to_disk(&self, template: &PromptTemplate) -> Result<()> {
        // Ensure directory exists
        if !self.storage_dir.exists() {
            tokio::fs::create_dir_all(&self.storage_dir).await?;
        }
        let path = self.storage_dir.join(format!("{}.json", template.id));
        let json = serde_json::to_string_pretty(template)?;
        tokio::fs::write(path, json).await?;
        Ok(())
    }

    /// Gets a template by ID.
    pub async fn get_template(&self, id: &str) -> Option<PromptTemplate> {
        if let Some(ref db) = self.db {
            return db.load_prompt_template(id).ok().flatten();
        }

        // Try in-memory first
        if let Some(template) = self.templates.read().get(id).cloned() {
            return Some(template);
        }

        // Try loading from disk
        let path = self.storage_dir.join(format!("{}.json", id));
        if path.exists() {
            if let Ok(content) = tokio::fs::read_to_string(&path).await {
                if let Ok(template) = serde_json::from_str::<PromptTemplate>(&content) {
                    return Some(template);
                }
            }
        }

        None
    }

    /// Gets a template by name.
    pub async fn get_template_by_name(&self, name: &str) -> Option<PromptTemplate> {
        let templates = self.list_templates().await;
        templates.into_iter().find(|t| t.name == name)
    }

    /// Lists all templates.
    pub async fn list_templates(&self) -> Vec<PromptTemplate> {
        if let Some(ref db) = self.db {
            let summaries = db.list_prompt_templates().unwrap_or_default();
            let mut templates = Vec::new();
            for (id, _, _) in summaries {
                if let Some(template) = db.load_prompt_template(&id).ok().flatten() {
                    templates.push(template);
                }
            }
            return templates;
        }

        // Load from disk
        let mut templates = Vec::new();
        if self.storage_dir.exists() {
            if let Ok(mut entries) = tokio::fs::read_dir(&self.storage_dir).await {
                while let Ok(Some(entry)) = entries.next_entry().await {
                    let path = entry.path();
                    if path.extension().map_or(false, |e| e == "json") {
                        if let Ok(content) = tokio::fs::read_to_string(&path).await {
                            if let Ok(template) = serde_json::from_str::<PromptTemplate>(&content) {
                                templates.push(template);
                            }
                        }
                    }
                }
            }
        }
        templates
    }

    /// Deletes a template by ID.
    pub fn delete_template(&self, id: &str) -> Result<()> {
        let _span = info_span!("prompt_studio.delete_template", id = %id).entered();

        if let Some(ref db) = self.db {
            db.delete_prompt_template(id)
                .map_err(|e| PromptError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
        } else {
            self.templates.write().remove(id);
        }

        info!("Deleted template");
        Ok(())
    }

    /// Adds a version to a template.
    pub async fn add_version(
        &self,
        template_id: &str,
        content: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<PromptVersion> {
        let _span = info_span!("prompt_studio.add_version", template_id = %template_id).entered();

        if let Some(ref db) = self.db {
            // Load template from database
            let mut template = db.load_prompt_template(template_id)
                .map_err(|e| PromptError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?
                .ok_or_else(|| PromptError::NotFound(template_id.to_string()))?;

            let version = template.create_version(content, message).clone();

            // Save updated template back to database
            db.save_prompt_template(&template)
                .map_err(|e| PromptError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;

            info!(version_id = %version.id, "Added version");
            Ok(version)
        } else {
            // Load template from disk first
            let mut template = self.get_template(template_id).await
                .ok_or_else(|| PromptError::NotFound(template_id.to_string()))?;

            let version = template.create_version(content, message).clone();

            // Save updated template to disk
            self.save_template_to_disk(&template).await?;

            // Update in-memory cache
            self.templates.write().insert(template_id.to_string(), template);

            info!(version_id = %version.id, "Added version");
            Ok(version)
        }
    }

    /// Sets the active version for a template.
    pub fn set_active_version(&self, template_id: &str, version_id: &str) -> Result<()> {
        let _span = info_span!(
            "prompt_studio.set_active_version",
            template_id = %template_id,
            version_id = %version_id
        ).entered();

        if let Some(ref db) = self.db {
            let mut template = db.load_prompt_template(template_id)
                .map_err(|e| PromptError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?
                .ok_or_else(|| PromptError::NotFound(template_id.to_string()))?;

            template.set_active_version(version_id)?;

            db.save_prompt_template(&template)
                .map_err(|e| PromptError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
        } else {
            let mut templates = self.templates.write();
            let template = templates.get_mut(template_id)
                .ok_or_else(|| PromptError::NotFound(template_id.to_string()))?;

            template.set_active_version(version_id)?;
        }

        info!("Set active version");
        Ok(())
    }

    /// Renders a template with variables.
    pub async fn render(
        &self,
        template_id: &str,
        variables: &HashMap<String, String>,
    ) -> Result<String> {
        let template = self.get_template(template_id).await
            .ok_or_else(|| PromptError::NotFound(template_id.to_string()))?;

        template.render(variables)
    }

    /// Starts an A/B test between two versions.
    pub async fn start_ab_test(
        &self,
        template_id: &str,
        name: impl Into<String>,
        version_a_id: &str,
        version_b_id: &str,
    ) -> Result<TestResult> {
        let _span = info_span!(
            "prompt_studio.start_ab_test",
            template_id = %template_id,
            version_a = %version_a_id,
            version_b = %version_b_id
        ).entered();

        // Verify versions exist
        let template = self.get_template(template_id).await
            .ok_or_else(|| PromptError::NotFound(template_id.to_string()))?;

        if template.get_version(version_a_id).is_none() {
            return Err(PromptError::VersionNotFound(version_a_id.to_string()));
        }
        if template.get_version(version_b_id).is_none() {
            return Err(PromptError::VersionNotFound(version_b_id.to_string()));
        }

        let test = TestResult::new(name, version_a_id.to_string(), version_b_id.to_string());
        let test_id = test.test_id.clone();

        info!(test_id = %test_id, "Started A/B test");

        // Save to database or in-memory
        if let Some(ref db) = self.db {
            db.save_prompt_test(Some(template_id), &test)
                .map_err(|e| PromptError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
        } else {
            self.tests.write().insert(test_id, test.clone());
        }

        Ok(test)
    }

    /// Records a result for an A/B test.
    pub fn record_test_result(
        &self,
        template_id: &str,
        test_id: &str,
        is_version_a: bool,
        success: bool,
        quality_score: f32,
        latency_ms: u64,
    ) -> Result<()> {
        if let Some(ref db) = self.db {
            let mut test = db.load_prompt_test(test_id)
                .map_err(|e| PromptError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?
                .ok_or_else(|| PromptError::TestNotFound(test_id.to_string()))?;

            if is_version_a {
                test.record_a(success, quality_score, latency_ms);
            } else {
                test.record_b(success, quality_score, latency_ms);
            }

            // Check if we can determine a winner
            test.determine_winner(30, 0.5); // min 30 samples, effect size > 0.5

            db.save_prompt_test(Some(template_id), &test)
                .map_err(|e| PromptError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
        } else {
            let mut tests = self.tests.write();
            let test = tests.get_mut(test_id)
                .ok_or_else(|| PromptError::TestNotFound(test_id.to_string()))?;

            if is_version_a {
                test.record_a(success, quality_score, latency_ms);
            } else {
                test.record_b(success, quality_score, latency_ms);
            }

            // Check if we can determine a winner
            test.determine_winner(30, 0.5); // min 30 samples, effect size > 0.5
        }

        Ok(())
    }

    /// Gets an A/B test by ID.
    pub fn get_test(&self, test_id: &str) -> Option<TestResult> {
        if let Some(ref db) = self.db {
            db.load_prompt_test(test_id).ok().flatten()
        } else {
            self.tests.read().get(test_id).cloned()
        }
    }

    /// Lists all active tests.
    pub async fn list_tests(&self) -> Vec<TestResult> {
        if let Some(ref db) = self.db {
            // Load all tests from all templates
            let templates = self.list_templates().await;
            let mut all_tests = Vec::new();
            for template in templates {
                if let Ok(tests) = db.list_prompt_tests(&template.id) {
                    for (test_id, _name) in tests {
                        if let Some(test) = db.load_prompt_test(&test_id).ok().flatten() {
                            all_tests.push(test);
                        }
                    }
                }
            }
            all_tests
        } else {
            self.tests.read().values().cloned().collect()
        }
    }

    /// Ends an A/B test.
    pub fn end_test(&self, template_id: &str, test_id: &str) -> Result<TestResult> {
        let _span = info_span!("prompt_studio.end_test", test_id = %test_id).entered();

        if let Some(ref db) = self.db {
            let mut test = db.load_prompt_test(test_id)
                .map_err(|e| PromptError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?
                .ok_or_else(|| PromptError::TestNotFound(test_id.to_string()))?;

            if test.status == TestStatus::Running {
                if test.winner.is_none() {
                    test.status = TestStatus::Inconclusive;
                } else {
                    test.status = TestStatus::Completed;
                }
                test.ended_at = Some(Utc::now());
            }

            db.save_prompt_test(Some(template_id), &test)
                .map_err(|e| PromptError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;

            info!(status = ?test.status, winner = ?test.winner, "Ended A/B test");
            Ok(test)
        } else {
            let mut tests = self.tests.write();
            let test = tests.get_mut(test_id)
                .ok_or_else(|| PromptError::TestNotFound(test_id.to_string()))?;

            if test.status == TestStatus::Running {
                if test.winner.is_none() {
                    test.status = TestStatus::Inconclusive;
                } else {
                    test.status = TestStatus::Completed;
                }
                test.ended_at = Some(Utc::now());
            }

            info!(status = ?test.status, winner = ?test.winner, "Ended A/B test");
            Ok(test.clone())
        }
    }

    /// Records usage metrics for a version.
    pub fn record_usage(
        &self,
        template_id: &str,
        version_id: &str,
        latency_ms: u64,
        success: bool,
        quality_score: Option<f32>,
    ) -> Result<()> {
        if let Some(ref db) = self.db {
            let mut template = db.load_prompt_template(template_id)
                .map_err(|e| PromptError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?
                .ok_or_else(|| PromptError::NotFound(template_id.to_string()))?;

            let version = template.versions.iter_mut()
                .find(|v| v.id == version_id)
                .ok_or_else(|| PromptError::VersionNotFound(version_id.to_string()))?;

            version.record_usage(latency_ms, success, quality_score);

            db.save_prompt_template(&template)
                .map_err(|e| PromptError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
        } else {
            let mut templates = self.templates.write();
            let template = templates.get_mut(template_id)
                .ok_or_else(|| PromptError::NotFound(template_id.to_string()))?;

            let version = template.versions.iter_mut()
                .find(|v| v.id == version_id)
                .ok_or_else(|| PromptError::VersionNotFound(version_id.to_string()))?;

            version.record_usage(latency_ms, success, quality_score);
        }
        Ok(())
    }

    /// Gets performance comparison across versions.
    pub async fn version_comparison(&self, template_id: &str) -> Result<VersionComparison> {
        let template = self.get_template(template_id).await
            .ok_or_else(|| PromptError::NotFound(template_id.to_string()))?;

        let mut versions = Vec::new();
        for v in &template.versions {
            versions.push(VersionStats {
                version_id: v.id.clone(),
                version_number: v.version_number,
                total_uses: v.metrics.total_uses,
                success_rate: v.metrics.success_rate(),
                avg_latency_ms: v.metrics.avg_latency_ms(),
                avg_quality: v.metrics.avg_quality_score(),
                is_active: Some(&v.id) == template.active_version_id.as_ref(),
            });
        }

        Ok(VersionComparison {
            template_id: template_id.to_string(),
            template_name: template.name.clone(),
            versions,
        })
    }
}

impl Default for PromptStudio {
    fn default() -> Self {
        Self::new(std::path::PathBuf::from("."))
    }
}

/// Comparison of prompt versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionComparison {
    /// Template ID.
    pub template_id: String,
    /// Template name.
    pub template_name: String,
    /// Version statistics.
    pub versions: Vec<VersionStats>,
}

/// Statistics for a single version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionStats {
    /// Version ID.
    pub version_id: String,
    /// Version number.
    pub version_number: u32,
    /// Total uses.
    pub total_uses: u64,
    /// Success rate.
    pub success_rate: f32,
    /// Average latency.
    pub avg_latency_ms: f64,
    /// Average quality score.
    pub avg_quality: Option<f32>,
    /// Whether this is the active version.
    pub is_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_template() {
        let template = PromptTemplate::new("test-template")
            .with_description("A test template")
            .with_tags(vec!["test".to_string()]);

        assert_eq!(template.name, "test-template");
        assert!(template.versions.is_empty());
    }

    #[test]
    fn test_template_versioning() {
        let mut template = PromptTemplate::new("test");

        let v1_id = template.create_version(
            "Hello {{name}}!",
            "Initial version"
        ).id.clone();
        assert_eq!(v1_id, template.active_version_id.clone().unwrap());

        let _ = template.create_version(
            "Hi {{name}}, welcome!",
            "More friendly"
        );

        assert_eq!(template.versions.len(), 2);
        // First version should still be active
        assert_eq!(template.active_version_id, Some(template.versions[0].id.clone()));
    }

    #[test]
    fn test_variable_extraction() {
        let version = PromptVersion::new(
            "Hello {{name}}, your order {{order_id}} is ready. Thank you, {{name}}!",
            "test"
        );

        let vars = version.variables();
        assert_eq!(vars, vec!["name", "order_id"]);
    }

    #[test]
    fn test_render() {
        let mut template = PromptTemplate::new("test");
        template.create_version(
            "Hello {{name}}! Your order {{order_id}} is ready.",
            "Initial"
        );

        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "Alice".to_string());
        vars.insert("order_id".to_string(), "12345".to_string());

        let rendered = template.render(&vars).expect("render");
        assert_eq!(rendered, "Hello Alice! Your order 12345 is ready.");
    }

    #[test]
    fn test_render_missing_variable() {
        let mut template = PromptTemplate::new("test");
        template.create_version("Hello {{name}}!", "Initial");

        let vars = HashMap::new();
        let result = template.render(&vars);

        assert!(matches!(result, Err(PromptError::MissingVariable(_))));
    }

    #[test]
    fn test_ab_test() {
        let mut test = TestResult::new("test", "v1".to_string(), "v2".to_string());

        // Record some results
        for _ in 0..35 {
            test.record_a(true, 0.8, 100);
            test.record_b(true, 0.6, 120);
        }

        test.determine_winner(30, 0.5);

        assert_eq!(test.status, TestStatus::Completed);
        assert_eq!(test.winner, Some("v1".to_string()));
    }

    #[tokio::test]
    async fn test_prompt_studio() {
        let studio = PromptStudio::default();

        let template = studio.create_template("greeting").await.expect("create template");
        let version = studio.add_version(
            &template.id,
            "Hello {{name}}!",
            "Initial version"
        ).await.expect("add version");

        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "World".to_string());

        let rendered = studio.render(&template.id, &vars).await.expect("render");
        assert_eq!(rendered, "Hello World!");

        // Record usage
        studio.record_usage(&template.id, &version.id, 50, true, Some(0.9))
            .expect("record usage");

        let comparison = studio.version_comparison(&template.id).await.expect("comparison");
        assert_eq!(comparison.versions.len(), 1);
        assert_eq!(comparison.versions[0].total_uses, 1);
    }

    #[tokio::test]
    async fn test_prompt_studio_with_database() {
        use std::sync::Arc;
        use crate::persistence::StudioDatabase;

        let db = Arc::new(StudioDatabase::in_memory().expect("create db"));
        let studio = PromptStudio::with_database(".", db);

        // Create template
        let template = studio.create_template("db-greeting").await.expect("create template");
        assert_eq!(template.name, "db-greeting");

        // Add version
        let version = studio.add_version(
            &template.id,
            "Hello {{name}}!",
            "Initial version"
        ).await.expect("add version");

        // Get template
        let retrieved = studio.get_template(&template.id).await.expect("template exists");
        assert_eq!(retrieved.versions.len(), 1);

        // Render
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "Database".to_string());
        let rendered = studio.render(&template.id, &vars).await.expect("render");
        assert_eq!(rendered, "Hello Database!");

        // Record usage and verify
        studio.record_usage(&template.id, &version.id, 50, true, Some(0.9))
            .expect("record usage");

        let comparison = studio.version_comparison(&template.id).await.expect("comparison");
        assert_eq!(comparison.versions.len(), 1);
        assert_eq!(comparison.versions[0].total_uses, 1);

        // List templates
        let templates = studio.list_templates().await;
        assert_eq!(templates.len(), 1);

        // Count templates
        assert_eq!(studio.count().await, 1);

        // Delete template
        studio.delete_template(&template.id).expect("delete");
        assert_eq!(studio.count().await, 0);
    }

    #[tokio::test]
    async fn test_prompt_studio_persistence_across_instances() {
        use std::sync::Arc;
        use crate::persistence::StudioDatabase;

        let temp = tempfile::TempDir::new().expect("temp dir");
        let db_path = temp.path().join("test.db");

        let template_id;

        // Create template with first studio instance
        {
            let db = Arc::new(StudioDatabase::new(&db_path).expect("create db"));
            let studio = PromptStudio::with_database(".", db);

            let template = studio.create_template("persistent-template").await.expect("create");
            studio.add_version(&template.id, "Version 1", "Initial").await.expect("add version");
            template_id = template.id.clone();
        }

        // Verify data persists with new studio instance
        {
            let db = Arc::new(StudioDatabase::new(&db_path).expect("reopen db"));
            let studio = PromptStudio::with_database(".", db);

            assert_eq!(studio.count().await, 1);

            let loaded = studio.get_template(&template_id).await.expect("template exists");
            assert_eq!(loaded.name, "persistent-template");
            assert_eq!(loaded.versions.len(), 1);
        }
    }

    #[test]
    fn test_version_metrics() {
        let mut version = PromptVersion::new("test", "test");

        version.record_usage(100, true, Some(0.8));
        version.record_usage(150, true, Some(0.9));
        version.record_usage(200, false, Some(0.5));

        assert_eq!(version.metrics.total_uses, 3);
        assert_eq!(version.metrics.successful_uses, 2);
        assert!((version.metrics.success_rate() - 0.666).abs() < 0.01);
        assert!((version.metrics.avg_latency_ms() - 150.0).abs() < 0.01);
    }
}
