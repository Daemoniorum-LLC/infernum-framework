//! Sigil Specialist Model Training
//!
//! This module provides fine-tuning capabilities specifically for creating
//! Sigil-specialist models from Jormungandr research data.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Training data pair for Sigil specialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingPair {
    /// Unique ID.
    pub id: String,
    /// Input prompt/context.
    pub input: String,
    /// Expected output.
    pub output: String,
    /// Source (checkpoint ID, pattern name, etc.).
    pub source: TrainingSource,
    /// Quality score (0.0-1.0).
    pub quality_score: f32,
    /// Specialization category.
    pub specialization: Specialization,
    /// Metadata.
    pub metadata: HashMap<String, String>,
}

impl TrainingPair {
    /// Creates a new training pair.
    pub fn new(
        input: impl Into<String>,
        output: impl Into<String>,
        source: TrainingSource,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            input: input.into(),
            output: output.into(),
            source,
            quality_score: 0.5,
            specialization: Specialization::General,
            metadata: HashMap::new(),
        }
    }

    /// Sets quality score.
    pub fn with_quality(mut self, score: f32) -> Self {
        self.quality_score = score.clamp(0.0, 1.0);
        self
    }

    /// Sets specialization.
    pub fn with_specialization(mut self, spec: Specialization) -> Self {
        self.specialization = spec;
        self
    }
}

/// Source of training data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingSource {
    /// From a successful conversion checkpoint.
    Checkpoint {
        /// Checkpoint ID.
        checkpoint_id: String,
        /// Project name.
        project: String,
    },
    /// From a resolved friction.
    ResolvedFriction {
        /// Friction description.
        friction: String,
        /// Resolution description.
        resolution: String,
    },
    /// From a discovered pattern.
    Pattern {
        /// Pattern name.
        pattern_name: String,
    },
    /// From documentation.
    Documentation {
        /// Document path.
        doc_path: String,
    },
    /// Synthetic/generated.
    Synthetic {
        /// Generation method.
        method: String,
    },
    /// Human-curated.
    Curated {
        /// Curator identifier.
        curator: String,
    },
}

/// Specialization areas for Sigil models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Specialization {
    /// General Sigil understanding.
    General,
    /// Sigil syntax completion.
    SyntaxCompletion,
    /// Evidentiality inference (!/~/? semantics).
    EvidentialityInference,
    /// Morpheme chain understanding (tau/phi/sigma/rho).
    MorphemeChains,
    /// Pattern recognition and idioms.
    PatternRecognition,
    /// Sigil compiler error diagnosis.
    ErrorDiagnosis,
    /// X-to-Sigil migration.
    MigrationExpertise,
    /// Sigil-to-X migration (reverse).
    ReverseMigration,
}

impl Specialization {
    /// Returns weight for this specialization in training.
    pub fn training_weight(&self) -> f32 {
        match self {
            Self::General => 0.5,
            Self::SyntaxCompletion => 1.0,
            Self::EvidentialityInference => 1.5, // High priority - unique to Sigil
            Self::MorphemeChains => 1.5,         // High priority - unique to Sigil
            Self::PatternRecognition => 1.0,
            Self::ErrorDiagnosis => 1.2,
            Self::MigrationExpertise => 1.3,
            Self::ReverseMigration => 0.8,
        }
    }
}

/// Configuration for Sigil specialist training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SigilTrainingConfig {
    /// Base model to fine-tune.
    pub base_model: String,
    /// Output model name.
    pub output_model: String,
    /// Training epochs.
    pub epochs: u32,
    /// Batch size.
    pub batch_size: u32,
    /// Learning rate.
    pub learning_rate: f64,
    /// LoRA rank (if using LoRA).
    pub lora_rank: Option<u32>,
    /// LoRA alpha (if using LoRA).
    pub lora_alpha: Option<f32>,
    /// Minimum quality score for training data.
    pub min_quality_score: f32,
    /// Specializations to focus on.
    pub focus_specializations: Vec<Specialization>,
    /// Whether to use curriculum learning.
    pub curriculum_learning: bool,
    /// Validation split ratio.
    pub validation_split: f32,
    /// Early stopping patience.
    pub early_stopping_patience: u32,
}

impl Default for SigilTrainingConfig {
    fn default() -> Self {
        Self {
            base_model: "llama-3.2-8b".to_string(),
            output_model: "sigil-specialist-v1".to_string(),
            epochs: 3,
            batch_size: 4,
            learning_rate: 2e-5,
            lora_rank: Some(16),
            lora_alpha: Some(32.0),
            min_quality_score: 0.6,
            focus_specializations: vec![
                Specialization::SyntaxCompletion,
                Specialization::EvidentialityInference,
                Specialization::MorphemeChains,
            ],
            curriculum_learning: true,
            validation_split: 0.1,
            early_stopping_patience: 3,
        }
    }
}

impl SigilTrainingConfig {
    /// Creates a lightweight config for quick experiments.
    pub fn lightweight() -> Self {
        Self {
            epochs: 1,
            batch_size: 2,
            lora_rank: Some(8),
            ..Default::default()
        }
    }

    /// Creates a full training config.
    pub fn full() -> Self {
        Self {
            epochs: 5,
            batch_size: 8,
            lora_rank: Some(32),
            lora_alpha: Some(64.0),
            ..Default::default()
        }
    }
}

/// Training dataset for Sigil specialist.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SigilDataset {
    /// Training pairs.
    pub pairs: Vec<TrainingPair>,
    /// Dataset metadata.
    pub metadata: DatasetMetadata,
}

/// Dataset metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Dataset name.
    pub name: String,
    /// Dataset version.
    pub version: String,
    /// Creation timestamp.
    pub created_at: Option<DateTime<Utc>>,
    /// Number of checkpoints sourced.
    pub checkpoint_count: u32,
    /// Number of patterns sourced.
    pub pattern_count: u32,
    /// Number of frictions sourced.
    pub friction_count: u32,
    /// Projects included.
    pub projects: Vec<String>,
    /// Total pairs.
    pub total_pairs: usize,
}

impl SigilDataset {
    /// Creates a new empty dataset.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            pairs: Vec::new(),
            metadata: DatasetMetadata {
                name: name.into(),
                version: "1.0.0".to_string(),
                created_at: Some(Utc::now()),
                ..Default::default()
            },
        }
    }

    /// Adds a training pair.
    pub fn add(&mut self, pair: TrainingPair) {
        self.pairs.push(pair);
        self.metadata.total_pairs = self.pairs.len();
    }

    /// Filters pairs by minimum quality score.
    pub fn filter_by_quality(&self, min_score: f32) -> Self {
        let pairs: Vec<_> = self
            .pairs
            .iter()
            .filter(|p| p.quality_score >= min_score)
            .cloned()
            .collect();

        Self {
            pairs,
            metadata: self.metadata.clone(),
        }
    }

    /// Filters pairs by specialization.
    pub fn filter_by_specialization(&self, specs: &[Specialization]) -> Self {
        let pairs: Vec<_> = self
            .pairs
            .iter()
            .filter(|p| specs.contains(&p.specialization))
            .cloned()
            .collect();

        Self {
            pairs,
            metadata: self.metadata.clone(),
        }
    }

    /// Returns statistics about the dataset.
    pub fn stats(&self) -> DatasetStats {
        let mut by_specialization: HashMap<Specialization, usize> = HashMap::new();
        let mut by_source_type: HashMap<String, usize> = HashMap::new();
        let mut total_quality = 0.0f64;

        for pair in &self.pairs {
            *by_specialization.entry(pair.specialization).or_default() += 1;
            let source_type = match &pair.source {
                TrainingSource::Checkpoint { .. } => "checkpoint",
                TrainingSource::ResolvedFriction { .. } => "friction",
                TrainingSource::Pattern { .. } => "pattern",
                TrainingSource::Documentation { .. } => "documentation",
                TrainingSource::Synthetic { .. } => "synthetic",
                TrainingSource::Curated { .. } => "curated",
            };
            *by_source_type.entry(source_type.to_string()).or_default() += 1;
            total_quality += pair.quality_score as f64;
        }

        DatasetStats {
            total_pairs: self.pairs.len(),
            avg_quality: if self.pairs.is_empty() {
                0.0
            } else {
                total_quality / self.pairs.len() as f64
            },
            by_specialization,
            by_source_type,
        }
    }

    /// Saves dataset to file.
    pub fn save(&self, path: &PathBuf) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)
    }

    /// Loads dataset from file.
    pub fn load(path: &PathBuf) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }
}

/// Dataset statistics.
#[derive(Debug, Clone)]
pub struct DatasetStats {
    /// Total training pairs.
    pub total_pairs: usize,
    /// Average quality score.
    pub avg_quality: f64,
    /// Pairs by specialization.
    pub by_specialization: HashMap<Specialization, usize>,
    /// Pairs by source type.
    pub by_source_type: HashMap<String, usize>,
}

/// Data collector for Jormungandr checkpoints.
pub struct CheckpointCollector {
    /// Output dataset.
    dataset: RwLock<SigilDataset>,
    /// Configuration.
    config: CollectorConfig,
}

/// Collector configuration.
#[derive(Debug, Clone)]
pub struct CollectorConfig {
    /// Minimum joy intensity to include.
    pub min_joy_intensity: f32,
    /// Include resolved frictions.
    pub include_frictions: bool,
    /// Include patterns.
    pub include_patterns: bool,
    /// Minimum friction severity to include.
    pub min_friction_severity: String,
}

impl Default for CollectorConfig {
    fn default() -> Self {
        Self {
            min_joy_intensity: 0.5,
            include_frictions: true,
            include_patterns: true,
            min_friction_severity: "minor".to_string(),
        }
    }
}

impl CheckpointCollector {
    /// Creates a new collector.
    pub fn new(config: CollectorConfig) -> Self {
        Self {
            dataset: RwLock::new(SigilDataset::new("jormungandr-collected")),
            config,
        }
    }

    /// Collects training data from a successful conversion.
    pub fn collect_from_conversion(
        &self,
        checkpoint_id: &str,
        project: &str,
        rust_code: &str,
        sigil_code: &str,
    ) {
        // Create conversion pair
        let pair = TrainingPair::new(
            format!(
                "Convert this Rust code to Sigil:\n\n```rust\n{}\n```",
                rust_code
            ),
            format!("```sigil\n{}\n```", sigil_code),
            TrainingSource::Checkpoint {
                checkpoint_id: checkpoint_id.to_string(),
                project: project.to_string(),
            },
        )
        .with_quality(0.8)
        .with_specialization(Specialization::MigrationExpertise);

        self.dataset.write().add(pair);
    }

    /// Collects training data from a resolved friction.
    pub fn collect_from_friction(
        &self,
        friction_desc: &str,
        resolution: &str,
        example_code: Option<&str>,
    ) {
        if !self.config.include_frictions {
            return;
        }

        let input = format!(
            "I encountered this issue with Sigil:\n{}\n\nHow do I resolve it?",
            friction_desc
        );
        let output = match example_code {
            Some(code) => format!("{}\n\nExample:\n```sigil\n{}\n```", resolution, code),
            None => resolution.to_string(),
        };

        let pair = TrainingPair::new(
            input,
            output,
            TrainingSource::ResolvedFriction {
                friction: friction_desc.to_string(),
                resolution: resolution.to_string(),
            },
        )
        .with_quality(0.7)
        .with_specialization(Specialization::ErrorDiagnosis);

        self.dataset.write().add(pair);
    }

    /// Collects training data from a discovered pattern.
    pub fn collect_from_pattern(&self, pattern_name: &str, description: &str, example: &str) {
        if !self.config.include_patterns {
            return;
        }

        let input = format!(
            "What is the {} pattern in Sigil and when should I use it?",
            pattern_name
        );
        let output = format!("{}\n\nExample:\n```sigil\n{}\n```", description, example);

        let pair = TrainingPair::new(
            input,
            output,
            TrainingSource::Pattern {
                pattern_name: pattern_name.to_string(),
            },
        )
        .with_quality(0.9)
        .with_specialization(Specialization::PatternRecognition);

        self.dataset.write().add(pair);
    }

    /// Collects syntax completion examples.
    pub fn collect_syntax_completion(&self, partial: &str, completed: &str) {
        let pair = TrainingPair::new(
            format!("Complete this Sigil code:\n```sigil\n{}\n```", partial),
            format!("```sigil\n{}\n```", completed),
            TrainingSource::Synthetic {
                method: "syntax_completion".to_string(),
            },
        )
        .with_quality(0.8)
        .with_specialization(Specialization::SyntaxCompletion);

        self.dataset.write().add(pair);
    }

    /// Collects evidentiality examples.
    pub fn collect_evidentiality_example(
        &self,
        scenario: &str,
        explanation: &str,
        code_example: &str,
    ) {
        let pair = TrainingPair::new(
            format!(
                "Explain the evidentiality markers in this Sigil scenario:\n{}",
                scenario
            ),
            format!(
                "{}\n\nCode example:\n```sigil\n{}\n```",
                explanation, code_example
            ),
            TrainingSource::Curated {
                curator: "jormungandr".to_string(),
            },
        )
        .with_quality(0.95)
        .with_specialization(Specialization::EvidentialityInference);

        self.dataset.write().add(pair);
    }

    /// Returns the collected dataset.
    pub fn dataset(&self) -> SigilDataset {
        self.dataset.read().clone()
    }

    /// Clears collected data.
    pub fn clear(&self) {
        *self.dataset.write() = SigilDataset::new("jormungandr-collected");
    }
}

/// Training job status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingStatus {
    /// Job is queued.
    Queued,
    /// Job is preparing data.
    PreparingData,
    /// Job is training.
    Training {
        /// Current epoch.
        epoch: u32,
        /// Total epochs.
        total_epochs: u32,
    },
    /// Job is validating.
    Validating,
    /// Job completed successfully.
    Completed,
    /// Job failed.
    Failed,
    /// Job was cancelled.
    Cancelled,
}

/// Training job for Sigil specialist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJob {
    /// Job ID.
    pub id: String,
    /// Configuration.
    pub config: SigilTrainingConfig,
    /// Status.
    pub status: TrainingStatus,
    /// Created timestamp.
    pub created_at: DateTime<Utc>,
    /// Started timestamp.
    pub started_at: Option<DateTime<Utc>>,
    /// Completed timestamp.
    pub completed_at: Option<DateTime<Utc>>,
    /// Training metrics.
    pub metrics: TrainingMetrics,
    /// Error message (if failed).
    pub error: Option<String>,
}

impl TrainingJob {
    /// Creates a new training job.
    pub fn new(config: SigilTrainingConfig) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            config,
            status: TrainingStatus::Queued,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            metrics: TrainingMetrics::default(),
            error: None,
        }
    }
}

/// Training metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Training loss per epoch.
    pub train_loss: Vec<f64>,
    /// Validation loss per epoch.
    pub val_loss: Vec<f64>,
    /// Training pairs processed.
    pub pairs_processed: u64,
    /// Training time in seconds.
    pub training_time_secs: u64,
    /// Best validation loss achieved.
    pub best_val_loss: Option<f64>,
    /// Best epoch.
    pub best_epoch: Option<u32>,
}

/// Sigil specialist trainer.
pub struct SigilTrainer {
    /// Active jobs.
    jobs: RwLock<HashMap<String, Arc<TrainingJob>>>,
    /// Output directory for models.
    output_dir: PathBuf,
}

impl SigilTrainer {
    /// Creates a new trainer.
    pub fn new(output_dir: PathBuf) -> std::io::Result<Self> {
        std::fs::create_dir_all(&output_dir)?;
        Ok(Self {
            jobs: RwLock::new(HashMap::new()),
            output_dir,
        })
    }

    /// Submits a training job.
    pub fn submit(&self, config: SigilTrainingConfig, dataset: &SigilDataset) -> Arc<TrainingJob> {
        // Filter dataset by config
        let filtered = dataset
            .filter_by_quality(config.min_quality_score)
            .filter_by_specialization(&config.focus_specializations);

        let job = Arc::new(TrainingJob::new(config));
        self.jobs.write().insert(job.id.clone(), Arc::clone(&job));

        // In a real implementation, this would spawn actual training
        tracing::info!(
            job_id = %job.id,
            pairs = filtered.pairs.len(),
            "Training job submitted"
        );

        job
    }

    /// Gets a job by ID.
    pub fn get_job(&self, job_id: &str) -> Option<Arc<TrainingJob>> {
        self.jobs.read().get(job_id).cloned()
    }

    /// Returns the output directory for trained models.
    #[must_use]
    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }

    /// Lists all jobs.
    pub fn list_jobs(&self) -> Vec<Arc<TrainingJob>> {
        self.jobs.read().values().cloned().collect()
    }

    /// Cancels a job.
    pub fn cancel(&self, job_id: &str) -> bool {
        // In a real implementation, this would stop the training process
        self.jobs.write().remove(job_id).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_pair_creation() {
        let pair = TrainingPair::new(
            "Convert this to Sigil",
            "Here is the Sigil code",
            TrainingSource::Checkpoint {
                checkpoint_id: "cp-001".to_string(),
                project: "infernum".to_string(),
            },
        )
        .with_quality(0.9)
        .with_specialization(Specialization::MigrationExpertise);

        assert_eq!(pair.quality_score, 0.9);
        assert_eq!(pair.specialization, Specialization::MigrationExpertise);
    }

    #[test]
    fn test_dataset_operations() {
        let mut dataset = SigilDataset::new("test");

        dataset.add(
            TrainingPair::new(
                "input1",
                "output1",
                TrainingSource::Curated {
                    curator: "test".to_string(),
                },
            )
            .with_quality(0.8),
        );

        dataset.add(
            TrainingPair::new(
                "input2",
                "output2",
                TrainingSource::Curated {
                    curator: "test".to_string(),
                },
            )
            .with_quality(0.4),
        );

        let filtered = dataset.filter_by_quality(0.6);
        assert_eq!(filtered.pairs.len(), 1);
    }

    #[test]
    fn test_checkpoint_collector() {
        let collector = CheckpointCollector::new(CollectorConfig::default());

        collector.collect_from_conversion("cp-001", "infernum", "fn hello() {}", "fn hello() {}");

        collector.collect_from_pattern(
            "builder",
            "The builder pattern for fluent APIs",
            "struct Builder { ... }",
        );

        let dataset = collector.dataset();
        assert_eq!(dataset.pairs.len(), 2);
    }

    #[test]
    fn test_training_config() {
        let config = SigilTrainingConfig::default();
        assert_eq!(config.epochs, 3);
        assert_eq!(config.focus_specializations.len(), 3);

        let lightweight = SigilTrainingConfig::lightweight();
        assert_eq!(lightweight.epochs, 1);
    }

    #[test]
    fn test_dataset_stats() {
        let mut dataset = SigilDataset::new("test");

        dataset.add(
            TrainingPair::new(
                "a",
                "b",
                TrainingSource::Curated {
                    curator: "test".to_string(),
                },
            )
            .with_quality(0.8)
            .with_specialization(Specialization::SyntaxCompletion),
        );

        dataset.add(
            TrainingPair::new(
                "c",
                "d",
                TrainingSource::Pattern {
                    pattern_name: "test".to_string(),
                },
            )
            .with_quality(0.6)
            .with_specialization(Specialization::PatternRecognition),
        );

        let stats = dataset.stats();
        assert_eq!(stats.total_pairs, 2);
        assert!((stats.avg_quality - 0.7).abs() < 0.001);
    }
}
