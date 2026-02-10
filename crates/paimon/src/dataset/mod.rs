//! Dataset management for LLM fine-tuning.
//!
//! Provides tools for uploading, validating, splitting, and augmenting
//! training datasets with agent-powered assistance.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{info, info_span};
use uuid::Uuid;

use crate::agents::DataCuratorAgent;

/// Errors that can occur during dataset operations.
#[derive(Debug, Error)]
pub enum DatasetError {
    /// Dataset not found.
    #[error("Dataset not found: {0}")]
    NotFound(String),

    /// Invalid dataset format.
    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    /// Validation failed.
    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Result type for dataset operations.
pub type Result<T> = std::result::Result<T, DatasetError>;

/// A single training example.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    /// Unique identifier.
    pub id: String,

    /// Input/prompt text.
    pub input: String,

    /// Expected output/completion.
    pub output: String,

    /// System prompt (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,

    /// Additional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,

    /// Quality score (0.0 - 1.0), set by validation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality_score: Option<f32>,

    /// Whether this is a synthetic example.
    #[serde(default)]
    pub synthetic: bool,
}

impl Example {
    /// Creates a new example.
    pub fn new(input: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            input: input.into(),
            output: output.into(),
            system: None,
            metadata: HashMap::new(),
            quality_score: None,
            synthetic: false,
        }
    }

    /// Sets the system prompt.
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Adds metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Marks as synthetic.
    pub fn as_synthetic(mut self) -> Self {
        self.synthetic = true;
        self
    }

    /// Returns the total character count.
    pub fn char_count(&self) -> usize {
        self.input.len() + self.output.len() + self.system.as_ref().map_or(0, String::len)
    }
}

/// Dataset format specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetFormat {
    /// JSON Lines format (one JSON object per line).
    JsonLines,
    /// Single JSON array.
    JsonArray,
    /// CSV with input/output columns.
    Csv,
    /// Alpaca format (instruction, input, output).
    Alpaca,
    /// ShareGPT format (conversations).
    ShareGpt,
    /// OpenAI fine-tuning format.
    OpenAI,
}

/// Configuration for creating a dataset.
#[derive(Debug, Clone)]
pub struct DatasetConfig {
    /// Dataset name.
    pub name: String,

    /// Description.
    pub description: Option<String>,

    /// Source format.
    pub format: DatasetFormat,

    /// Tags for organization.
    pub tags: Vec<String>,

    /// Whether to validate on creation.
    pub validate: bool,
}

impl DatasetConfig {
    /// Creates a new dataset configuration.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            format: DatasetFormat::JsonLines,
            tags: Vec::new(),
            validate: true,
        }
    }

    /// Sets the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Sets the format.
    pub fn with_format(mut self, format: DatasetFormat) -> Self {
        self.format = format;
        self
    }

    /// Adds tags.
    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags = tags.into_iter().map(Into::into).collect();
        self
    }
}

/// A training dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    /// Unique identifier.
    pub id: String,

    /// Human-readable name.
    pub name: String,

    /// Description.
    pub description: Option<String>,

    /// Source format.
    pub format: DatasetFormat,

    /// Tags.
    pub tags: Vec<String>,

    /// Training examples.
    pub examples: Vec<Example>,

    /// Creation timestamp.
    pub created_at: DateTime<Utc>,

    /// Last modified timestamp.
    pub updated_at: DateTime<Utc>,

    /// Validation report.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation: Option<ValidationReport>,

    /// Dataset statistics.
    pub stats: DatasetStats,
}

impl Dataset {
    /// Creates a new dataset.
    pub fn new(config: DatasetConfig, examples: Vec<Example>) -> Self {
        let stats = DatasetStats::compute(&examples);
        let now = Utc::now();

        Self {
            id: Uuid::new_v4().to_string(),
            name: config.name,
            description: config.description,
            format: config.format,
            tags: config.tags,
            examples,
            created_at: now,
            updated_at: now,
            validation: None,
            stats,
        }
    }

    /// Returns the number of examples.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Returns true if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Adds examples to the dataset.
    pub fn add_examples(&mut self, examples: impl IntoIterator<Item = Example>) {
        self.examples.extend(examples);
        self.stats = DatasetStats::compute(&self.examples);
        self.updated_at = Utc::now();
    }

    /// Filters examples by a predicate.
    pub fn filter<F>(&self, predicate: F) -> Self
    where
        F: Fn(&Example) -> bool,
    {
        let filtered: Vec<Example> = self
            .examples
            .iter()
            .filter(|e| predicate(e))
            .cloned()
            .collect();
        let stats = DatasetStats::compute(&filtered);

        Self {
            id: Uuid::new_v4().to_string(),
            name: format!("{}_filtered", self.name),
            description: self.description.clone(),
            format: self.format,
            tags: self.tags.clone(),
            examples: filtered,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            validation: None,
            stats,
        }
    }

    /// Splits the dataset into train/validation/test sets.
    pub fn split(&self, config: SplitConfig) -> DatasetSplit {
        let total = self.examples.len();
        let train_end = (total as f32 * config.train_ratio) as usize;
        let val_end = train_end + (total as f32 * config.val_ratio) as usize;

        let mut examples = self.examples.clone();
        if config.shuffle {
            // Simple shuffle using hash-based sorting for reproducibility
            examples.sort_by(|a, b| {
                let hash_a =
                    a.id.as_bytes()
                        .iter()
                        .fold(0u64, |acc, &b| acc.wrapping_add(b as u64));
                let hash_b =
                    b.id.as_bytes()
                        .iter()
                        .fold(0u64, |acc, &b| acc.wrapping_add(b as u64));
                hash_a.cmp(&hash_b)
            });
        }

        DatasetSplit {
            train: examples[..train_end].to_vec(),
            validation: examples[train_end..val_end].to_vec(),
            test: examples[val_end..].to_vec(),
        }
    }
}

/// Configuration for splitting a dataset.
#[derive(Debug, Clone)]
pub struct SplitConfig {
    /// Ratio for training set (0.0 - 1.0).
    pub train_ratio: f32,
    /// Ratio for validation set (0.0 - 1.0).
    pub val_ratio: f32,
    /// Whether to shuffle before splitting.
    pub shuffle: bool,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            train_ratio: 0.8,
            val_ratio: 0.1,
            shuffle: true,
        }
    }
}

/// Result of splitting a dataset.
#[derive(Debug, Clone)]
pub struct DatasetSplit {
    /// Training examples.
    pub train: Vec<Example>,
    /// Validation examples.
    pub validation: Vec<Example>,
    /// Test examples.
    pub test: Vec<Example>,
}

impl DatasetSplit {
    /// Returns the total number of examples across all splits.
    pub fn total(&self) -> usize {
        self.train.len() + self.validation.len() + self.test.len()
    }
}

/// Statistics about a dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStats {
    /// Total number of examples.
    pub example_count: usize,
    /// Number of synthetic examples.
    pub synthetic_count: usize,
    /// Total character count.
    pub total_chars: usize,
    /// Average input length.
    pub avg_input_len: f32,
    /// Average output length.
    pub avg_output_len: f32,
    /// Examples with system prompts.
    pub with_system_count: usize,
}

impl DatasetStats {
    /// Computes statistics from examples.
    pub fn compute(examples: &[Example]) -> Self {
        if examples.is_empty() {
            return Self {
                example_count: 0,
                synthetic_count: 0,
                total_chars: 0,
                avg_input_len: 0.0,
                avg_output_len: 0.0,
                with_system_count: 0,
            };
        }

        let total_input: usize = examples.iter().map(|e| e.input.len()).sum();
        let total_output: usize = examples.iter().map(|e| e.output.len()).sum();

        Self {
            example_count: examples.len(),
            synthetic_count: examples.iter().filter(|e| e.synthetic).count(),
            total_chars: examples.iter().map(Example::char_count).sum(),
            avg_input_len: total_input as f32 / examples.len() as f32,
            avg_output_len: total_output as f32 / examples.len() as f32,
            with_system_count: examples.iter().filter(|e| e.system.is_some()).count(),
        }
    }
}

/// Validation report for a dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Overall validation passed.
    pub passed: bool,
    /// Quality score (0.0 - 1.0).
    pub quality_score: f32,
    /// Issues found.
    pub issues: Vec<ValidationIssue>,
    /// Suggestions for improvement.
    pub suggestions: Vec<String>,
    /// Timestamp.
    pub validated_at: DateTime<Utc>,
}

/// A validation issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// Severity level.
    pub severity: IssueSeverity,
    /// Issue description.
    pub message: String,
    /// Affected example IDs.
    pub affected_examples: Vec<String>,
}

/// Severity of a validation issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Informational.
    Info,
    /// Warning, may affect quality.
    Warning,
    /// Error, should be fixed.
    Error,
}

/// Manages datasets for the studio.
pub struct DatasetManager {
    base_dir: PathBuf,
    datasets: RwLock<HashMap<String, Dataset>>,
    curator: Option<Arc<DataCuratorAgent>>,
    db: Option<Arc<crate::persistence::StudioDatabase>>,
}

impl DatasetManager {
    /// Creates a new dataset manager.
    pub fn new(base_dir: PathBuf) -> Self {
        Self {
            base_dir,
            datasets: RwLock::new(HashMap::new()),
            curator: None,
            db: None,
        }
    }

    /// Creates a dataset manager with SQLite persistence.
    pub fn with_database(base_dir: PathBuf, db: Arc<crate::persistence::StudioDatabase>) -> Self {
        Self {
            base_dir,
            datasets: RwLock::new(HashMap::new()),
            curator: None,
            db: Some(db),
        }
    }

    /// Sets the data curator agent.
    pub fn with_curator(mut self, curator: Arc<DataCuratorAgent>) -> Self {
        self.curator = Some(curator);
        self
    }

    /// Returns the number of datasets.
    pub async fn count(&self) -> usize {
        if let Some(ref db) = self.db {
            db.count_datasets().unwrap_or(0)
        } else {
            self.datasets.read().len()
        }
    }

    /// Creates a new dataset.
    pub async fn create(&self, config: DatasetConfig, examples: Vec<Example>) -> Result<Dataset> {
        let _span = info_span!("dataset.create", name = %config.name).entered();

        let mut dataset = Dataset::new(config, examples);

        // Validate if requested
        if !dataset.examples.is_empty() {
            let validation = self.validate_internal(&dataset).await;
            dataset.validation = Some(validation);
        }

        // Save to database or disk
        if let Some(ref db) = self.db {
            db.save_dataset(&dataset).map_err(|e| {
                DatasetError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    e.to_string(),
                ))
            })?;
        } else {
            self.save(&dataset).await?;
            self.datasets
                .write()
                .insert(dataset.id.clone(), dataset.clone());
        }

        info!(id = %dataset.id, examples = dataset.len(), "Created dataset");
        Ok(dataset)
    }

    /// Gets a dataset by ID.
    pub async fn get(&self, id: &str) -> Result<Dataset> {
        // Try database first
        if let Some(ref db) = self.db {
            return db
                .load_dataset(id)
                .map_err(|e| {
                    DatasetError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e.to_string(),
                    ))
                })?
                .ok_or_else(|| DatasetError::NotFound(id.to_string()));
        }

        // Try in-memory first
        if let Some(dataset) = self.datasets.read().get(id).cloned() {
            return Ok(dataset);
        }

        // Try loading from disk
        let path = self.dataset_path(id);
        if path.exists() {
            let content = tokio::fs::read_to_string(&path).await?;
            let dataset: Dataset = serde_json::from_str(&content)?;
            return Ok(dataset);
        }

        Err(DatasetError::NotFound(id.to_string()))
    }

    /// Lists all datasets.
    pub async fn list(&self) -> Vec<Dataset> {
        if let Some(ref db) = self.db {
            // For list, we only return metadata. Load full datasets on demand.
            let summaries = db.list_datasets().unwrap_or_default();
            let mut datasets = Vec::new();
            for (id, _, _) in summaries {
                if let Ok(Some(dataset)) = db.load_dataset(&id) {
                    datasets.push(dataset);
                }
            }
            return datasets;
        }

        // Load datasets from disk
        let mut datasets = Vec::new();
        if self.base_dir.exists() {
            if let Ok(mut entries) = tokio::fs::read_dir(&self.base_dir).await {
                while let Ok(Some(entry)) = entries.next_entry().await {
                    let path = entry.path();
                    if path.extension().map_or(false, |e| e == "json") {
                        if let Ok(content) = tokio::fs::read_to_string(&path).await {
                            if let Ok(dataset) = serde_json::from_str::<Dataset>(&content) {
                                datasets.push(dataset);
                            }
                        }
                    }
                }
            }
        }
        datasets
    }

    /// Deletes a dataset.
    pub async fn delete(&self, id: &str) -> Result<()> {
        let _span = info_span!("dataset.delete", id = %id).entered();

        if let Some(ref db) = self.db {
            db.delete_dataset(id).map_err(|e| {
                DatasetError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    e.to_string(),
                ))
            })?;
        } else {
            self.datasets.write().remove(id);
            let path = self.dataset_path(id);
            if path.exists() {
                tokio::fs::remove_file(path).await?;
            }
        }

        info!("Deleted dataset");
        Ok(())
    }

    /// Validates a dataset.
    pub async fn validate(&self, id: &str) -> Result<ValidationReport> {
        let dataset = self.get(id).await?;
        Ok(self.validate_internal(&dataset).await)
    }

    /// Augments a dataset with synthetic examples.
    ///
    /// Requires the data curator agent to be configured.
    pub async fn augment(&self, id: &str, count: usize) -> Result<Dataset> {
        let _span = info_span!("dataset.augment", id = %id, count = count).entered();

        let curator = self.curator.as_ref().ok_or_else(|| {
            DatasetError::ValidationFailed("Data curator agent not configured".to_string())
        })?;

        let dataset = self.get(id).await?;

        info!("Generating {} synthetic examples", count);
        let synthetic = curator
            .generate_examples(&dataset.examples, count)
            .await
            .map_err(|e| DatasetError::ValidationFailed(e.to_string()))?;

        let mut augmented = dataset;
        augmented.add_examples(synthetic);
        augmented.updated_at = Utc::now();

        // Re-validate
        let validation = self.validate_internal(&augmented).await;
        augmented.validation = Some(validation);

        // Save to database or disk
        if let Some(ref db) = self.db {
            db.save_dataset(&augmented).map_err(|e| {
                DatasetError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    e.to_string(),
                ))
            })?;
        } else {
            self.save(&augmented).await?;
            self.datasets
                .write()
                .insert(augmented.id.clone(), augmented.clone());
        }

        info!(new_total = augmented.len(), "Augmented dataset");
        Ok(augmented)
    }

    /// Internal validation logic.
    async fn validate_internal(&self, dataset: &Dataset) -> ValidationReport {
        let mut issues = Vec::new();
        let mut suggestions = Vec::new();

        // Check for empty examples
        let empty_inputs: Vec<_> = dataset
            .examples
            .iter()
            .filter(|e| e.input.trim().is_empty())
            .map(|e| e.id.clone())
            .collect();
        if !empty_inputs.is_empty() {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Error,
                message: "Examples with empty inputs".to_string(),
                affected_examples: empty_inputs,
            });
        }

        let empty_outputs: Vec<_> = dataset
            .examples
            .iter()
            .filter(|e| e.output.trim().is_empty())
            .map(|e| e.id.clone())
            .collect();
        if !empty_outputs.is_empty() {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Error,
                message: "Examples with empty outputs".to_string(),
                affected_examples: empty_outputs,
            });
        }

        // Check for very short examples
        let short_examples: Vec<_> = dataset
            .examples
            .iter()
            .filter(|e| e.char_count() < 20)
            .map(|e| e.id.clone())
            .collect();
        if !short_examples.is_empty() {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Warning,
                message: "Very short examples (< 20 chars)".to_string(),
                affected_examples: short_examples,
            });
        }

        // Check for potential duplicates
        let mut seen = std::collections::HashSet::new();
        let duplicates: Vec<_> = dataset
            .examples
            .iter()
            .filter(|e| !seen.insert(&e.input))
            .map(|e| e.id.clone())
            .collect();
        if !duplicates.is_empty() {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Warning,
                message: "Potential duplicate inputs".to_string(),
                affected_examples: duplicates,
            });
            suggestions.push("Consider removing duplicate examples".to_string());
        }

        // Suggest system prompts if few have them
        let system_ratio =
            dataset.stats.with_system_count as f32 / dataset.stats.example_count.max(1) as f32;
        if system_ratio < 0.5 && dataset.stats.example_count > 10 {
            suggestions
                .push("Consider adding system prompts to improve model behavior".to_string());
        }

        // Calculate quality score
        let error_count = issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .count();
        let warning_count = issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Warning)
            .count();
        let quality_score = 1.0 - (error_count as f32 * 0.2) - (warning_count as f32 * 0.05);

        ValidationReport {
            passed: error_count == 0,
            quality_score: quality_score.max(0.0),
            issues,
            suggestions,
            validated_at: Utc::now(),
        }
    }

    /// Saves a dataset to disk (async I/O).
    async fn save(&self, dataset: &Dataset) -> Result<()> {
        // Ensure base directory exists
        if !self.base_dir.exists() {
            tokio::fs::create_dir_all(&self.base_dir).await?;
        }
        let path = self.dataset_path(&dataset.id);
        let json = serde_json::to_string_pretty(dataset)?;
        tokio::fs::write(path, json).await?;
        Ok(())
    }

    /// Returns the path for a dataset file.
    fn dataset_path(&self, id: &str) -> PathBuf {
        self.base_dir.join(format!("{}.json", id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_creation() {
        let example = Example::new("What is 2+2?", "4")
            .with_system("You are a math tutor")
            .with_metadata("subject", serde_json::json!("math"));

        assert!(!example.id.is_empty());
        assert_eq!(example.input, "What is 2+2?");
        assert_eq!(example.output, "4");
        assert_eq!(example.system, Some("You are a math tutor".to_string()));
        assert!(!example.synthetic);
    }

    #[test]
    fn test_dataset_stats() {
        let examples = vec![
            Example::new("input1", "output1"),
            Example::new("input2", "output2").as_synthetic(),
            Example::new("input3", "output3").with_system("system"),
        ];

        let stats = DatasetStats::compute(&examples);

        assert_eq!(stats.example_count, 3);
        assert_eq!(stats.synthetic_count, 1);
        assert_eq!(stats.with_system_count, 1);
    }

    #[test]
    fn test_dataset_split() {
        let examples: Vec<Example> = (0..100)
            .map(|i| Example::new(format!("input{}", i), format!("output{}", i)))
            .collect();

        let config = DatasetConfig::new("test");
        let dataset = Dataset::new(config, examples);

        let split = dataset.split(SplitConfig::default());

        assert_eq!(split.train.len(), 80);
        assert_eq!(split.validation.len(), 10);
        assert_eq!(split.test.len(), 10);
        assert_eq!(split.total(), 100);
    }

    #[tokio::test]
    async fn test_dataset_manager() {
        let temp = tempfile::TempDir::new().expect("temp dir");
        let manager = DatasetManager::new(temp.path().to_path_buf());

        let examples = vec![
            Example::new("Hello", "Hi there!"),
            Example::new("How are you?", "I'm doing well, thanks!"),
        ];

        let config = DatasetConfig::new("test_dataset")
            .with_description("A test dataset")
            .with_tags(vec!["test", "demo"]);

        let dataset = manager.create(config, examples).await.expect("create");

        assert_eq!(dataset.name, "test_dataset");
        assert_eq!(dataset.len(), 2);
        assert!(dataset.validation.is_some());

        let retrieved = manager.get(&dataset.id).await.expect("get");
        assert_eq!(retrieved.id, dataset.id);

        assert_eq!(manager.count().await, 1);
    }

    #[tokio::test]
    async fn test_dataset_manager_with_database() {
        use crate::persistence::StudioDatabase;
        use std::sync::Arc;

        let temp = tempfile::TempDir::new().expect("temp dir");
        let db = Arc::new(StudioDatabase::in_memory().expect("create db"));
        let manager = DatasetManager::with_database(temp.path().to_path_buf(), db);

        let examples = vec![
            Example::new("What is 2+2?", "4"),
            Example::new("What is 3+3?", "6"),
        ];

        let config = DatasetConfig::new("math-dataset").with_description("Math examples");

        // Create dataset
        let dataset = manager.create(config, examples).await.expect("create");
        assert_eq!(dataset.name, "math-dataset");
        assert_eq!(dataset.len(), 2);

        // Get dataset
        let retrieved = manager.get(&dataset.id).await.expect("get");
        assert_eq!(retrieved.id, dataset.id);
        assert_eq!(retrieved.examples.len(), 2);

        // List datasets
        let list = manager.list().await;
        assert_eq!(list.len(), 1);

        // Count datasets
        assert_eq!(manager.count().await, 1);

        // Delete dataset
        manager.delete(&dataset.id).await.expect("delete");
        assert_eq!(manager.count().await, 0);
    }

    #[tokio::test]
    async fn test_dataset_manager_persistence_across_instances() {
        use crate::persistence::StudioDatabase;
        use std::sync::Arc;

        let temp = tempfile::TempDir::new().expect("temp dir");
        let db_path = temp.path().join("test.db");

        let dataset_id;

        // Create dataset with first manager instance
        {
            let db = Arc::new(StudioDatabase::new(&db_path).expect("create db"));
            let manager = DatasetManager::with_database(temp.path().to_path_buf(), db);

            let examples = vec![Example::new("input", "output")];
            let config = DatasetConfig::new("persistent-dataset");

            let dataset = manager.create(config, examples).await.expect("create");
            dataset_id = dataset.id.clone();
        }

        // Verify data persists with new manager instance
        {
            let db = Arc::new(StudioDatabase::new(&db_path).expect("reopen db"));
            let manager = DatasetManager::with_database(temp.path().to_path_buf(), db);

            assert_eq!(manager.count().await, 1);

            let loaded = manager.get(&dataset_id).await.expect("get");
            assert_eq!(loaded.name, "persistent-dataset");
            assert_eq!(loaded.examples.len(), 1);
        }
    }

    #[tokio::test]
    async fn test_async_file_operations() {
        // Test that async file operations (save/delete) work correctly
        // without blocking the async runtime
        let temp = tempfile::TempDir::new().expect("temp dir");
        let manager = DatasetManager::new(temp.path().to_path_buf());

        let examples = vec![
            Example::new("async input 1", "async output 1"),
            Example::new("async input 2", "async output 2"),
        ];

        let config = DatasetConfig::new("async-test-dataset");

        // Create should use async file write
        let dataset = manager.create(config, examples).await.expect("create");
        let dataset_id = dataset.id.clone();

        // Verify file was created on disk
        let file_path = temp.path().join(format!("{}.json", dataset_id));
        assert!(file_path.exists(), "Dataset file should be created");

        // Read file content asynchronously to verify
        let content = tokio::fs::read_to_string(&file_path)
            .await
            .expect("read file");
        assert!(content.contains("async-test-dataset"));
        assert!(content.contains("async input 1"));

        // Delete should use async file remove
        manager.delete(&dataset_id).await.expect("delete");
        assert!(!file_path.exists(), "Dataset file should be deleted");
    }
}
