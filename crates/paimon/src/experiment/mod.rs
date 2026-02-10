//! Experiment Tracker - Track, compare, and analyze training experiments.
//!
//! The experiment tracker provides:
//! - Run tracking with full metrics history
//! - Artifact management (checkpoints, logs, configs)
//! - Comparison across runs
//! - Integration with Training Coach for real-time monitoring

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{info, info_span, warn};
use uuid::Uuid;

/// Errors that can occur in experiment tracking.
#[derive(Debug, Error)]
pub enum ExperimentError {
    /// Experiment not found.
    #[error("Experiment not found: {0}")]
    NotFound(String),

    /// Run not found.
    #[error("Run not found: {0}")]
    RunNotFound(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Invalid state transition.
    #[error("Invalid state transition: {from:?} -> {to:?}")]
    InvalidStateTransition {
        /// The current run status.
        from: RunStatus,
        /// The attempted target status.
        to: RunStatus,
    },
}

/// Result type for experiment operations.
pub type Result<T> = std::result::Result<T, ExperimentError>;

/// Configuration for experiment tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// Name of the experiment.
    pub name: String,
    /// Description.
    pub description: Option<String>,
    /// Base model being fine-tuned.
    pub base_model: String,
    /// Dataset ID.
    pub dataset_id: String,
    /// Tags for categorization.
    pub tags: Vec<String>,
}

impl ExperimentConfig {
    /// Creates a new experiment configuration.
    pub fn new(
        name: impl Into<String>,
        base_model: impl Into<String>,
        dataset_id: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: None,
            base_model: base_model.into(),
            dataset_id: dataset_id.into(),
            tags: Vec::new(),
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
}

/// An experiment tracking multiple runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    /// Unique identifier.
    pub id: String,
    /// Configuration.
    pub config: ExperimentConfig,
    /// All runs in this experiment.
    pub runs: Vec<Run>,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp.
    pub updated_at: DateTime<Utc>,
    /// Best run ID (based on primary metric).
    pub best_run_id: Option<String>,
    /// Primary metric for comparison.
    pub primary_metric: Option<String>,
}

impl Experiment {
    /// Creates a new experiment.
    pub fn new(config: ExperimentConfig) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            config,
            runs: Vec::new(),
            created_at: now,
            updated_at: now,
            best_run_id: None,
            primary_metric: None,
        }
    }

    /// Sets the primary metric for comparison.
    pub fn set_primary_metric(&mut self, metric: impl Into<String>) {
        self.primary_metric = Some(metric.into());
        self.update_best_run();
    }

    /// Adds a run to this experiment.
    pub fn add_run(&mut self, run: Run) {
        self.runs.push(run);
        self.updated_at = Utc::now();
        self.update_best_run();
    }

    /// Gets a run by ID.
    pub fn get_run(&self, run_id: &str) -> Option<&Run> {
        self.runs.iter().find(|r| r.id == run_id)
    }

    /// Gets a mutable run by ID.
    pub fn get_run_mut(&mut self, run_id: &str) -> Option<&mut Run> {
        self.runs.iter_mut().find(|r| r.id == run_id)
    }

    /// Updates the best run based on primary metric.
    fn update_best_run(&mut self) {
        if let Some(ref metric) = self.primary_metric {
            let best = self
                .runs
                .iter()
                .filter(|r| r.status == RunStatus::Completed)
                .filter_map(|r| r.final_metrics.get(metric).map(|v| (r.id.clone(), *v)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            self.best_run_id = best.map(|(id, _)| id);
        }
    }

    /// Gets completed runs sorted by a metric.
    pub fn runs_sorted_by(&self, metric: &str, descending: bool) -> Vec<&Run> {
        let mut runs: Vec<_> = self
            .runs
            .iter()
            .filter(|r| r.status == RunStatus::Completed)
            .filter(|r| r.final_metrics.contains_key(metric))
            .collect();

        runs.sort_by(|a, b| {
            let va = a.final_metrics.get(metric).unwrap_or(&0.0);
            let vb = b.final_metrics.get(metric).unwrap_or(&0.0);
            if descending {
                vb.partial_cmp(va).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                va.partial_cmp(vb).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        runs
    }
}

/// A single training run within an experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    /// Unique identifier.
    pub id: String,
    /// Run name (optional, auto-generated if not provided).
    pub name: String,
    /// Hyperparameters used.
    pub hyperparameters: HashMap<String, HyperparamValue>,
    /// Current status.
    pub status: RunStatus,
    /// Metrics history (metric_name -> [(step, value)]).
    pub metrics_history: HashMap<String, Vec<(u64, f64)>>,
    /// Final metrics at end of training.
    pub final_metrics: HashMap<String, f64>,
    /// Artifacts (checkpoints, logs, etc.).
    pub artifacts: Vec<Artifact>,
    /// Start timestamp.
    pub started_at: DateTime<Utc>,
    /// End timestamp.
    pub ended_at: Option<DateTime<Utc>>,
    /// Error message if failed.
    pub error_message: Option<String>,
    /// Tags.
    pub tags: Vec<String>,
    /// Notes.
    pub notes: Option<String>,
}

impl Run {
    /// Creates a new run.
    pub fn new(name: Option<String>) -> Self {
        let id = Uuid::new_v4().to_string();
        let name = name.unwrap_or_else(|| format!("run-{}", &id[..8]));

        Self {
            id,
            name,
            hyperparameters: HashMap::new(),
            status: RunStatus::Pending,
            metrics_history: HashMap::new(),
            final_metrics: HashMap::new(),
            artifacts: Vec::new(),
            started_at: Utc::now(),
            ended_at: None,
            error_message: None,
            tags: Vec::new(),
            notes: None,
        }
    }

    /// Sets a hyperparameter.
    pub fn set_hyperparam(&mut self, name: impl Into<String>, value: HyperparamValue) {
        self.hyperparameters.insert(name.into(), value);
    }

    /// Sets multiple hyperparameters.
    pub fn set_hyperparams(&mut self, params: HashMap<String, HyperparamValue>) {
        self.hyperparameters.extend(params);
    }

    /// Logs a metric value.
    pub fn log_metric(&mut self, name: impl Into<String>, step: u64, value: f64) {
        let name = name.into();
        self.metrics_history
            .entry(name)
            .or_default()
            .push((step, value));
    }

    /// Logs multiple metrics at once.
    pub fn log_metrics(&mut self, step: u64, metrics: HashMap<String, f64>) {
        for (name, value) in metrics {
            self.log_metric(name, step, value);
        }
    }

    /// Gets the latest value for a metric.
    pub fn latest_metric(&self, name: &str) -> Option<f64> {
        self.metrics_history
            .get(name)
            .and_then(|h| h.last())
            .map(|(_, v)| *v)
    }

    /// Starts the run.
    pub fn start(&mut self) -> Result<()> {
        if self.status != RunStatus::Pending {
            return Err(ExperimentError::InvalidStateTransition {
                from: self.status,
                to: RunStatus::Running,
            });
        }
        self.status = RunStatus::Running;
        self.started_at = Utc::now();
        Ok(())
    }

    /// Completes the run successfully.
    pub fn complete(&mut self, final_metrics: HashMap<String, f64>) {
        self.status = RunStatus::Completed;
        self.final_metrics = final_metrics;
        self.ended_at = Some(Utc::now());
    }

    /// Fails the run.
    pub fn fail(&mut self, error: impl Into<String>) {
        self.status = RunStatus::Failed;
        self.error_message = Some(error.into());
        self.ended_at = Some(Utc::now());
    }

    /// Pauses the run.
    pub fn pause(&mut self) {
        if self.status == RunStatus::Running {
            self.status = RunStatus::Paused;
        }
    }

    /// Resumes a paused run.
    pub fn resume(&mut self) {
        if self.status == RunStatus::Paused {
            self.status = RunStatus::Running;
        }
    }

    /// Adds an artifact.
    pub fn add_artifact(&mut self, artifact: Artifact) {
        self.artifacts.push(artifact);
    }

    /// Gets duration of the run.
    pub fn duration(&self) -> Option<chrono::Duration> {
        self.ended_at.map(|end| end - self.started_at)
    }
}

/// Status of a training run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunStatus {
    /// Not yet started.
    Pending,
    /// Currently running.
    Running,
    /// Paused.
    Paused,
    /// Completed successfully.
    Completed,
    /// Failed with an error.
    Failed,
    /// Cancelled by user.
    Cancelled,
}

impl RunStatus {
    /// Returns true if the run is finished (completed, failed, or cancelled).
    pub fn is_finished(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }

    /// Returns a human-readable status string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "Pending",
            Self::Running => "Running",
            Self::Paused => "Paused",
            Self::Completed => "Completed",
            Self::Failed => "Failed",
            Self::Cancelled => "Cancelled",
        }
    }
}

/// A hyperparameter value (supports multiple types).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum HyperparamValue {
    /// Integer value.
    Int(i64),
    /// Float value.
    Float(f64),
    /// String value.
    String(String),
    /// Boolean value.
    Bool(bool),
    /// List value.
    List(Vec<HyperparamValue>),
}

impl From<i64> for HyperparamValue {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}

impl From<f64> for HyperparamValue {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}

impl From<String> for HyperparamValue {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}

impl From<&str> for HyperparamValue {
    fn from(v: &str) -> Self {
        Self::String(v.to_string())
    }
}

impl From<bool> for HyperparamValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

/// An artifact from a training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// Artifact name.
    pub name: String,
    /// Artifact type.
    pub artifact_type: ArtifactType,
    /// Path to the artifact.
    pub path: PathBuf,
    /// Size in bytes.
    pub size_bytes: Option<u64>,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Associated step (if applicable).
    pub step: Option<u64>,
    /// Metadata.
    pub metadata: HashMap<String, String>,
}

impl Artifact {
    /// Creates a new artifact.
    pub fn new(
        name: impl Into<String>,
        artifact_type: ArtifactType,
        path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            name: name.into(),
            artifact_type,
            path: path.into(),
            size_bytes: None,
            created_at: Utc::now(),
            step: None,
            metadata: HashMap::new(),
        }
    }

    /// Sets the step.
    pub fn with_step(mut self, step: u64) -> Self {
        self.step = Some(step);
        self
    }

    /// Sets the size.
    pub fn with_size(mut self, size: u64) -> Self {
        self.size_bytes = Some(size);
        self
    }

    /// Adds metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Types of artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArtifactType {
    /// Model checkpoint.
    Checkpoint,
    /// Training logs.
    Logs,
    /// Configuration file.
    Config,
    /// Metrics export.
    Metrics,
    /// Trained model weights.
    Model,
    /// LoRA adapter.
    LoraAdapter,
    /// Other artifact.
    Other,
}

/// The experiment tracker manages all experiments and runs.
pub struct ExperimentTracker {
    /// All experiments.
    experiments: Arc<RwLock<HashMap<String, Experiment>>>,
    /// Storage directory.
    storage_dir: PathBuf,
    /// Optional database for persistence.
    db: Option<Arc<crate::persistence::StudioDatabase>>,
}

impl ExperimentTracker {
    /// Creates a new experiment tracker.
    pub fn new(storage_dir: impl Into<PathBuf>) -> Self {
        Self {
            experiments: Arc::new(RwLock::new(HashMap::new())),
            storage_dir: storage_dir.into(),
            db: None,
        }
    }

    /// Creates an experiment tracker with SQLite persistence.
    pub fn with_database(
        storage_dir: impl Into<PathBuf>,
        db: Arc<crate::persistence::StudioDatabase>,
    ) -> Self {
        Self {
            experiments: Arc::new(RwLock::new(HashMap::new())),
            storage_dir: storage_dir.into(),
            db: Some(db),
        }
    }

    /// Returns the number of experiments.
    pub async fn count(&self) -> usize {
        if let Some(ref db) = self.db {
            db.count_experiments().unwrap_or(0)
        } else {
            self.experiments.read().len()
        }
    }

    /// Creates a new experiment.
    pub async fn create_experiment(&self, config: ExperimentConfig) -> Result<Experiment> {
        let _span = info_span!("tracker.create_experiment", name = %config.name).entered();

        let experiment = Experiment::new(config);
        let id = experiment.id.clone();

        // Save to database if available
        if let Some(ref db) = self.db {
            if let Err(e) = db.save_experiment(&experiment) {
                warn!(error = %e, "Failed to persist experiment to database");
            }
        } else {
            // Save to disk
            self.save_experiment_to_disk(&experiment).await?;
        }

        info!(experiment_id = %id, "Created experiment");

        self.experiments.write().insert(id, experiment.clone());
        Ok(experiment)
    }

    /// Saves a single experiment to disk.
    async fn save_experiment_to_disk(&self, experiment: &Experiment) -> Result<()> {
        // Ensure directory exists
        if !self.storage_dir.exists() {
            tokio::fs::create_dir_all(&self.storage_dir).await?;
        }
        let path = self.storage_dir.join(format!("{}.json", experiment.id));
        let json = serde_json::to_string_pretty(experiment)?;
        tokio::fs::write(path, json).await?;
        Ok(())
    }

    /// Gets an experiment by ID.
    pub async fn get_experiment(&self, id: &str) -> Option<Experiment> {
        // Try in-memory first
        if let Some(experiment) = self.experiments.read().get(id).cloned() {
            return Some(experiment);
        }

        // Try loading from disk
        let path = self.storage_dir.join(format!("{}.json", id));
        if path.exists() {
            if let Ok(content) = tokio::fs::read_to_string(&path).await {
                if let Ok(experiment) = serde_json::from_str::<Experiment>(&content) {
                    return Some(experiment);
                }
            }
        }

        None
    }

    /// Lists all experiments.
    pub async fn list_experiments(&self) -> Vec<Experiment> {
        // If using database, experiments are in memory (loaded on startup)
        if self.db.is_some() {
            return self.experiments.read().values().cloned().collect();
        }

        // Load from disk
        let mut experiments = Vec::new();
        if self.storage_dir.exists() {
            if let Ok(mut entries) = tokio::fs::read_dir(&self.storage_dir).await {
                while let Ok(Some(entry)) = entries.next_entry().await {
                    let path = entry.path();
                    if path.extension().map_or(false, |e| e == "json") {
                        if let Ok(content) = tokio::fs::read_to_string(&path).await {
                            if let Ok(experiment) = serde_json::from_str::<Experiment>(&content) {
                                experiments.push(experiment);
                            }
                        }
                    }
                }
            }
        }
        experiments
    }

    /// Deletes an experiment.
    pub fn delete_experiment(&self, id: &str) -> Result<()> {
        let _span = info_span!("tracker.delete_experiment", experiment_id = %id).entered();

        if self.experiments.write().remove(id).is_none() {
            return Err(ExperimentError::NotFound(id.to_string()));
        }

        info!("Deleted experiment");
        Ok(())
    }

    /// Starts a new run in an experiment.
    pub fn start_run(&self, experiment_id: &str, name: Option<String>) -> Result<Run> {
        let _span = info_span!("tracker.start_run", experiment_id = %experiment_id).entered();

        let mut experiments = self.experiments.write();
        let experiment = experiments
            .get_mut(experiment_id)
            .ok_or_else(|| ExperimentError::NotFound(experiment_id.to_string()))?;

        let mut run = Run::new(name);
        run.start()?;

        // Save run to database if available
        if let Some(ref db) = self.db {
            if let Err(e) = db.save_run(experiment_id, &run) {
                warn!(error = %e, "Failed to persist run to database");
            }
        }

        info!(run_id = %run.id, run_name = %run.name, "Started run");

        experiment.add_run(run.clone());
        Ok(run)
    }

    /// Logs metrics for a run.
    pub fn log_metrics(
        &self,
        experiment_id: &str,
        run_id: &str,
        step: u64,
        metrics: HashMap<String, f64>,
    ) -> Result<()> {
        let mut experiments = self.experiments.write();
        let experiment = experiments
            .get_mut(experiment_id)
            .ok_or_else(|| ExperimentError::NotFound(experiment_id.to_string()))?;

        let run = experiment
            .get_run_mut(run_id)
            .ok_or_else(|| ExperimentError::RunNotFound(run_id.to_string()))?;

        run.log_metrics(step, metrics.clone());

        // Log to database if available
        if let Some(ref db) = self.db {
            let metric_pairs: Vec<(&str, f64)> =
                metrics.iter().map(|(k, v)| (k.as_str(), *v)).collect();
            if let Err(e) = db.log_run_metrics(run_id, &metric_pairs) {
                warn!(error = %e, "Failed to persist metrics to database");
            }
        }

        Ok(())
    }

    /// Completes a run.
    pub fn complete_run(
        &self,
        experiment_id: &str,
        run_id: &str,
        final_metrics: HashMap<String, f64>,
    ) -> Result<()> {
        let _span =
            info_span!("tracker.complete_run", experiment_id = %experiment_id, run_id = %run_id)
                .entered();

        let mut experiments = self.experiments.write();
        let experiment = experiments
            .get_mut(experiment_id)
            .ok_or_else(|| ExperimentError::NotFound(experiment_id.to_string()))?;

        let run = experiment
            .get_run_mut(run_id)
            .ok_or_else(|| ExperimentError::RunNotFound(run_id.to_string()))?;

        run.complete(final_metrics);
        experiment.update_best_run();

        // Update status in database
        if let Some(ref db) = self.db {
            if let Err(e) = db.update_run_status(run_id, RunStatus::Completed, None) {
                warn!(error = %e, "Failed to update run status in database");
            }
        }

        info!("Completed run");
        Ok(())
    }

    /// Fails a run.
    pub fn fail_run(
        &self,
        experiment_id: &str,
        run_id: &str,
        error: impl Into<String>,
    ) -> Result<()> {
        let _span =
            info_span!("tracker.fail_run", experiment_id = %experiment_id, run_id = %run_id)
                .entered();

        let error_msg = error.into();
        warn!(error = %error_msg, "Run failed");

        let mut experiments = self.experiments.write();
        let experiment = experiments
            .get_mut(experiment_id)
            .ok_or_else(|| ExperimentError::NotFound(experiment_id.to_string()))?;

        let run = experiment
            .get_run_mut(run_id)
            .ok_or_else(|| ExperimentError::RunNotFound(run_id.to_string()))?;

        run.fail(error_msg.clone());

        // Update status in database
        if let Some(ref db) = self.db {
            if let Err(e) = db.update_run_status(run_id, RunStatus::Failed, Some(&error_msg)) {
                warn!(error = %e, "Failed to update run status in database");
            }
        }

        Ok(())
    }

    /// Adds an artifact to a run.
    pub fn add_artifact(
        &self,
        experiment_id: &str,
        run_id: &str,
        artifact: Artifact,
    ) -> Result<()> {
        let _span = info_span!(
            "tracker.add_artifact",
            experiment_id = %experiment_id,
            run_id = %run_id,
            artifact = %artifact.name
        )
        .entered();

        let mut experiments = self.experiments.write();
        let experiment = experiments
            .get_mut(experiment_id)
            .ok_or_else(|| ExperimentError::NotFound(experiment_id.to_string()))?;

        let run = experiment
            .get_run_mut(run_id)
            .ok_or_else(|| ExperimentError::RunNotFound(run_id.to_string()))?;

        info!(artifact_type = ?artifact.artifact_type, "Added artifact");
        run.add_artifact(artifact);
        Ok(())
    }

    /// Compares runs across experiments.
    pub fn compare_runs(&self, run_ids: &[(String, String)]) -> RunComparison {
        let _span = info_span!("tracker.compare_runs", count = run_ids.len()).entered();

        let experiments = self.experiments.read();
        let mut runs = Vec::new();

        for (exp_id, run_id) in run_ids {
            if let Some(exp) = experiments.get(exp_id) {
                if let Some(run) = exp.get_run(run_id) {
                    runs.push((exp.config.name.clone(), run.clone()));
                }
            }
        }

        // Find common metrics
        let common_metrics: Vec<String> = if runs.is_empty() {
            Vec::new()
        } else {
            let first_metrics: std::collections::HashSet<_> =
                runs[0].1.final_metrics.keys().cloned().collect();
            runs.iter()
                .skip(1)
                .fold(first_metrics, |acc, (_, r)| {
                    acc.intersection(&r.final_metrics.keys().cloned().collect())
                        .cloned()
                        .collect()
                })
                .into_iter()
                .collect()
        };

        // Build comparison table
        let mut metric_values: HashMap<String, Vec<(String, f64)>> = HashMap::new();
        for metric in &common_metrics {
            let values: Vec<_> = runs
                .iter()
                .filter_map(|(name, run)| run.final_metrics.get(metric).map(|v| (name.clone(), *v)))
                .collect();
            metric_values.insert(metric.clone(), values);
        }

        RunComparison {
            runs: runs.into_iter().map(|(name, run)| (name, run)).collect(),
            common_metrics,
            metric_values,
        }
    }

    /// Saves experiments to disk.
    pub async fn save(&self) -> Result<()> {
        let _span = info_span!("tracker.save").entered();

        tokio::fs::create_dir_all(&self.storage_dir).await?;

        let experiments = self.experiments.read();
        for (id, exp) in experiments.iter() {
            let path = self.storage_dir.join(format!("{}.json", id));
            let json = serde_json::to_string_pretty(exp)?;
            tokio::fs::write(&path, json).await?;
        }

        info!(count = experiments.len(), "Saved experiments");
        Ok(())
    }

    /// Loads experiments from disk.
    pub async fn load(&self) -> Result<()> {
        let _span = info_span!("tracker.load").entered();

        if !self.storage_dir.exists() {
            return Ok(());
        }

        let mut entries = tokio::fs::read_dir(&self.storage_dir).await?;
        let mut loaded = 0;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "json") {
                let json = tokio::fs::read_to_string(&path).await?;
                let exp: Experiment = serde_json::from_str(&json)?;
                self.experiments.write().insert(exp.id.clone(), exp);
                loaded += 1;
            }
        }

        info!(count = loaded, "Loaded experiments");
        Ok(())
    }
}

/// Comparison of multiple runs.
#[derive(Debug, Clone)]
pub struct RunComparison {
    /// Runs being compared (experiment_name, run).
    pub runs: Vec<(String, Run)>,
    /// Metrics common to all runs.
    pub common_metrics: Vec<String>,
    /// Metric values by run (metric_name -> [(experiment_name, value)]).
    pub metric_values: HashMap<String, Vec<(String, f64)>>,
}

impl RunComparison {
    /// Gets the best run for a given metric.
    pub fn best_for_metric(&self, metric: &str, higher_is_better: bool) -> Option<&(String, Run)> {
        if let Some(values) = self.metric_values.get(metric) {
            let best_name = if higher_is_better {
                values
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            } else {
                values
                    .iter()
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            };

            best_name.and_then(|(name, _)| self.runs.iter().find(|(n, _)| n == name))
        } else {
            None
        }
    }

    /// Generates a summary table as markdown.
    pub fn to_markdown_table(&self) -> String {
        if self.runs.is_empty() {
            return "No runs to compare".to_string();
        }

        let mut table = String::new();

        // Header
        table.push_str("| Metric |");
        for (name, _) in &self.runs {
            table.push_str(&format!(" {} |", name));
        }
        table.push('\n');

        // Separator
        table.push_str("|--------|");
        for _ in &self.runs {
            table.push_str("--------|");
        }
        table.push('\n');

        // Metrics
        for metric in &self.common_metrics {
            table.push_str(&format!("| {} |", metric));
            if let Some(values) = self.metric_values.get(metric) {
                for (name, _) in &self.runs {
                    let value = values
                        .iter()
                        .find(|(n, _)| n == name)
                        .map(|(_, v)| format!("{:.4}", v))
                        .unwrap_or_else(|| "-".to_string());
                    table.push_str(&format!(" {} |", value));
                }
            }
            table.push('\n');
        }

        table
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === ExperimentConfig Tests ===

    #[test]
    fn test_experiment_config_new() {
        let config = ExperimentConfig::new("exp", "llama", "ds-1");
        assert_eq!(config.name, "exp");
        assert_eq!(config.base_model, "llama");
        assert_eq!(config.dataset_id, "ds-1");
        assert!(config.description.is_none());
        assert!(config.tags.is_empty());
    }

    #[test]
    fn test_experiment_config_with_description() {
        let config =
            ExperimentConfig::new("exp", "llama", "ds-1").with_description("My experiment");
        assert_eq!(config.description, Some("My experiment".to_string()));
    }

    #[test]
    fn test_experiment_config_with_tags() {
        let config = ExperimentConfig::new("exp", "llama", "ds-1")
            .with_tags(vec!["tag1".to_string(), "tag2".to_string()]);
        assert_eq!(config.tags.len(), 2);
    }

    // === RunStatus Tests ===

    #[test]
    fn test_run_status_is_finished() {
        assert!(!RunStatus::Pending.is_finished());
        assert!(!RunStatus::Running.is_finished());
        assert!(!RunStatus::Paused.is_finished());
        assert!(RunStatus::Completed.is_finished());
        assert!(RunStatus::Failed.is_finished());
        assert!(RunStatus::Cancelled.is_finished());
    }

    #[test]
    fn test_run_status_as_str() {
        assert_eq!(RunStatus::Pending.as_str(), "Pending");
        assert_eq!(RunStatus::Running.as_str(), "Running");
        assert_eq!(RunStatus::Paused.as_str(), "Paused");
        assert_eq!(RunStatus::Completed.as_str(), "Completed");
        assert_eq!(RunStatus::Failed.as_str(), "Failed");
        assert_eq!(RunStatus::Cancelled.as_str(), "Cancelled");
    }

    // === HyperparamValue Tests ===

    #[test]
    fn test_hyperparam_from_i64() {
        let v: HyperparamValue = 42i64.into();
        assert!(matches!(v, HyperparamValue::Int(42)));
    }

    #[test]
    fn test_hyperparam_from_f64() {
        let v: HyperparamValue = 3.14f64.into();
        assert!(matches!(v, HyperparamValue::Float(f) if (f - 3.14).abs() < f64::EPSILON));
    }

    #[test]
    fn test_hyperparam_from_string() {
        let v: HyperparamValue = "hello".to_string().into();
        assert!(matches!(v, HyperparamValue::String(s) if s == "hello"));
    }

    #[test]
    fn test_hyperparam_from_str() {
        let v: HyperparamValue = "world".into();
        assert!(matches!(v, HyperparamValue::String(s) if s == "world"));
    }

    #[test]
    fn test_hyperparam_from_bool() {
        let v: HyperparamValue = true.into();
        assert!(matches!(v, HyperparamValue::Bool(true)));
    }

    // === Artifact Tests ===

    #[test]
    fn test_artifact_new() {
        let artifact = Artifact::new("model", ArtifactType::Model, "/path/model.bin");
        assert_eq!(artifact.name, "model");
        assert_eq!(artifact.artifact_type, ArtifactType::Model);
        assert!(artifact.step.is_none());
        assert!(artifact.size_bytes.is_none());
    }

    #[test]
    fn test_artifact_with_step() {
        let artifact = Artifact::new("ckpt", ArtifactType::Checkpoint, "/path").with_step(500);
        assert_eq!(artifact.step, Some(500));
    }

    #[test]
    fn test_artifact_with_size() {
        let artifact = Artifact::new("log", ArtifactType::Logs, "/path").with_size(1024);
        assert_eq!(artifact.size_bytes, Some(1024));
    }

    #[test]
    fn test_artifact_types() {
        assert_eq!(ArtifactType::Checkpoint, ArtifactType::Checkpoint);
        assert_ne!(ArtifactType::Checkpoint, ArtifactType::Model);
    }

    // === Run Tests ===

    #[test]
    fn test_run_new_auto_name() {
        let run = Run::new(None);
        assert!(run.name.starts_with("run-"));
        assert_eq!(run.status, RunStatus::Pending);
    }

    #[test]
    fn test_run_new_custom_name() {
        let run = Run::new(Some("custom-run".to_string()));
        assert_eq!(run.name, "custom-run");
    }

    #[test]
    fn test_run_set_hyperparams() {
        let mut run = Run::new(None);
        let mut params = HashMap::new();
        params.insert("lr".to_string(), HyperparamValue::Float(0.001));
        params.insert("batch".to_string(), HyperparamValue::Int(32));
        run.set_hyperparams(params);
        assert_eq!(run.hyperparameters.len(), 2);
    }

    #[test]
    fn test_run_log_metric() {
        let mut run = Run::new(None);
        run.log_metric("loss", 1, 1.0);
        run.log_metric("loss", 2, 0.8);
        run.log_metric("loss", 3, 0.6);

        let history = run.metrics_history.get("loss").unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0], (1, 1.0));
        assert_eq!(history[2], (3, 0.6));
    }

    #[test]
    fn test_run_latest_metric_none() {
        let run = Run::new(None);
        assert_eq!(run.latest_metric("loss"), None);
    }

    #[test]
    fn test_run_pause_resume() {
        let mut run = Run::new(None);
        run.start().unwrap();
        assert_eq!(run.status, RunStatus::Running);

        run.pause();
        assert_eq!(run.status, RunStatus::Paused);

        run.resume();
        assert_eq!(run.status, RunStatus::Running);
    }

    #[test]
    fn test_run_pause_wrong_state() {
        let mut run = Run::new(None);
        run.pause(); // Should do nothing when Pending
        assert_eq!(run.status, RunStatus::Pending);
    }

    #[test]
    fn test_run_resume_wrong_state() {
        let mut run = Run::new(None);
        run.resume(); // Should do nothing when Pending
        assert_eq!(run.status, RunStatus::Pending);
    }

    #[test]
    fn test_run_duration_incomplete() {
        let run = Run::new(None);
        assert!(run.duration().is_none());
    }

    #[test]
    fn test_run_duration_complete() {
        let mut run = Run::new(None);
        run.start().unwrap();
        run.complete(HashMap::new());
        assert!(run.duration().is_some());
    }

    #[test]
    fn test_run_start_invalid_state() {
        let mut run = Run::new(None);
        run.start().unwrap();
        // Starting again should fail
        let result = run.start();
        assert!(result.is_err());
    }

    #[test]
    fn test_run_add_artifact() {
        let mut run = Run::new(None);
        let artifact = Artifact::new("ckpt", ArtifactType::Checkpoint, "/path");
        run.add_artifact(artifact);
        assert_eq!(run.artifacts.len(), 1);
    }

    // === Experiment Tests ===

    #[test]
    fn test_experiment_get_run() {
        let config = ExperimentConfig::new("test", "llama", "ds");
        let mut exp = Experiment::new(config);

        let run = Run::new(Some("my-run".to_string()));
        let run_id = run.id.clone();
        exp.add_run(run);

        assert!(exp.get_run(&run_id).is_some());
        assert!(exp.get_run("nonexistent").is_none());
    }

    #[test]
    fn test_experiment_get_run_mut() {
        let config = ExperimentConfig::new("test", "llama", "ds");
        let mut exp = Experiment::new(config);

        let run = Run::new(None);
        let run_id = run.id.clone();
        exp.add_run(run);

        let run_mut = exp.get_run_mut(&run_id).unwrap();
        run_mut.tags.push("modified".to_string());
        assert_eq!(exp.get_run(&run_id).unwrap().tags[0], "modified");
    }

    #[test]
    fn test_experiment_runs_sorted_by() {
        let config = ExperimentConfig::new("test", "llama", "ds");
        let mut exp = Experiment::new(config);

        // Add runs with different accuracy values
        for (i, acc) in [(1, 0.7), (2, 0.9), (3, 0.8)] {
            let mut run = Run::new(Some(format!("run-{}", i)));
            run.start().unwrap();
            let mut metrics = HashMap::new();
            metrics.insert("accuracy".to_string(), acc);
            run.complete(metrics);
            exp.add_run(run);
        }

        let sorted = exp.runs_sorted_by("accuracy", true); // descending
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].final_metrics.get("accuracy"), Some(&0.9));
        assert_eq!(sorted[2].final_metrics.get("accuracy"), Some(&0.7));
    }

    #[test]
    fn test_experiment_runs_sorted_by_ascending() {
        let config = ExperimentConfig::new("test", "llama", "ds");
        let mut exp = Experiment::new(config);

        for (i, loss) in [(1, 0.5), (2, 0.3), (3, 0.8)] {
            let mut run = Run::new(Some(format!("run-{}", i)));
            run.start().unwrap();
            let mut metrics = HashMap::new();
            metrics.insert("loss".to_string(), loss);
            run.complete(metrics);
            exp.add_run(run);
        }

        let sorted = exp.runs_sorted_by("loss", false); // ascending
        assert_eq!(sorted[0].final_metrics.get("loss"), Some(&0.3));
        assert_eq!(sorted[2].final_metrics.get("loss"), Some(&0.8));
    }

    // === RunComparison Tests ===

    #[test]
    fn test_run_comparison_best_for_metric_higher() {
        let mut runs = Vec::new();
        let mut metric_values = HashMap::new();

        for (name, acc) in [("run1", 0.7), ("run2", 0.9), ("run3", 0.8)] {
            let mut run = Run::new(Some(name.to_string()));
            run.start().unwrap();
            let mut metrics = HashMap::new();
            metrics.insert("accuracy".to_string(), acc);
            run.complete(metrics);
            runs.push((name.to_string(), run));
        }

        metric_values.insert(
            "accuracy".to_string(),
            vec![
                ("run1".to_string(), 0.7),
                ("run2".to_string(), 0.9),
                ("run3".to_string(), 0.8),
            ],
        );

        let comparison = RunComparison {
            runs,
            common_metrics: vec!["accuracy".to_string()],
            metric_values,
        };

        let best = comparison.best_for_metric("accuracy", true);
        assert!(best.is_some());
        assert_eq!(best.unwrap().0, "run2");
    }

    #[test]
    fn test_run_comparison_best_for_metric_lower() {
        let mut runs = Vec::new();
        let mut metric_values = HashMap::new();

        for (name, loss) in [("run1", 0.5), ("run2", 0.3), ("run3", 0.8)] {
            let mut run = Run::new(Some(name.to_string()));
            run.start().unwrap();
            let mut metrics = HashMap::new();
            metrics.insert("loss".to_string(), loss);
            run.complete(metrics);
            runs.push((name.to_string(), run));
        }

        metric_values.insert(
            "loss".to_string(),
            vec![
                ("run1".to_string(), 0.5),
                ("run2".to_string(), 0.3),
                ("run3".to_string(), 0.8),
            ],
        );

        let comparison = RunComparison {
            runs,
            common_metrics: vec!["loss".to_string()],
            metric_values,
        };

        let best = comparison.best_for_metric("loss", false); // lower is better
        assert!(best.is_some());
        assert_eq!(best.unwrap().0, "run2");
    }

    #[test]
    fn test_run_comparison_to_markdown_empty() {
        let comparison = RunComparison {
            runs: Vec::new(),
            common_metrics: Vec::new(),
            metric_values: HashMap::new(),
        };
        assert_eq!(comparison.to_markdown_table(), "No runs to compare");
    }

    #[test]
    fn test_run_comparison_to_markdown() {
        let mut runs = Vec::new();
        let mut run = Run::new(Some("run1".to_string()));
        run.start().unwrap();
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.85);
        run.complete(metrics);
        runs.push(("run1".to_string(), run));

        let mut metric_values = HashMap::new();
        metric_values.insert("accuracy".to_string(), vec![("run1".to_string(), 0.85)]);

        let comparison = RunComparison {
            runs,
            common_metrics: vec!["accuracy".to_string()],
            metric_values,
        };

        let table = comparison.to_markdown_table();
        assert!(table.contains("| Metric |"));
        assert!(table.contains("| accuracy |"));
        assert!(table.contains("0.8500"));
    }

    // === Original Tests ===

    #[test]
    fn test_create_experiment() {
        let config = ExperimentConfig::new("test-exp", "llama-7b", "dataset-1")
            .with_description("Test experiment")
            .with_tags(vec!["test".to_string()]);

        let exp = Experiment::new(config);

        assert!(!exp.id.is_empty());
        assert_eq!(exp.config.name, "test-exp");
        assert_eq!(exp.config.base_model, "llama-7b");
        assert!(exp.runs.is_empty());
    }

    #[test]
    fn test_run_lifecycle() {
        let mut run = Run::new(Some("test-run".to_string()));
        assert_eq!(run.status, RunStatus::Pending);

        run.start().expect("start");
        assert_eq!(run.status, RunStatus::Running);

        run.log_metric("loss", 1, 1.5);
        run.log_metric("loss", 2, 1.2);
        run.log_metric("loss", 3, 0.9);

        assert_eq!(run.latest_metric("loss"), Some(0.9));

        let mut final_metrics = HashMap::new();
        final_metrics.insert("loss".to_string(), 0.9);
        final_metrics.insert("accuracy".to_string(), 0.85);
        run.complete(final_metrics);

        assert_eq!(run.status, RunStatus::Completed);
        assert!(run.ended_at.is_some());
    }

    #[test]
    fn test_run_failure() {
        let mut run = Run::new(None);
        run.start().expect("start");
        run.fail("Out of memory");

        assert_eq!(run.status, RunStatus::Failed);
        assert_eq!(run.error_message, Some("Out of memory".to_string()));
    }

    #[test]
    fn test_experiment_best_run() {
        let config = ExperimentConfig::new("test", "llama", "ds");
        let mut exp = Experiment::new(config);
        exp.set_primary_metric("accuracy");

        // Add first run
        let mut run1 = Run::new(Some("run1".to_string()));
        run1.start().expect("start");
        let mut metrics1 = HashMap::new();
        metrics1.insert("accuracy".to_string(), 0.8);
        run1.complete(metrics1);
        exp.add_run(run1);

        // Add second (better) run
        let mut run2 = Run::new(Some("run2".to_string()));
        run2.start().expect("start");
        let mut metrics2 = HashMap::new();
        metrics2.insert("accuracy".to_string(), 0.9);
        run2.complete(metrics2);
        let run2_id = run2.id.clone();
        exp.add_run(run2);

        assert_eq!(exp.best_run_id, Some(run2_id));
    }

    #[test]
    fn test_hyperparams() {
        let mut run = Run::new(None);

        run.set_hyperparam("learning_rate", HyperparamValue::Float(0.001));
        run.set_hyperparam("epochs", HyperparamValue::Int(10));
        run.set_hyperparam("model", HyperparamValue::String("llama-7b".to_string()));

        assert_eq!(run.hyperparameters.len(), 3);
    }

    #[test]
    fn test_artifacts() {
        let artifact = Artifact::new(
            "checkpoint-100",
            ArtifactType::Checkpoint,
            "/path/to/checkpoint",
        )
        .with_step(100)
        .with_size(1024 * 1024)
        .with_metadata("format", "safetensors");

        assert_eq!(artifact.step, Some(100));
        assert_eq!(artifact.size_bytes, Some(1024 * 1024));
        assert_eq!(
            artifact.metadata.get("format"),
            Some(&"safetensors".to_string())
        );
    }

    #[tokio::test]
    async fn test_experiment_tracker() {
        let tracker = ExperimentTracker::new("/tmp/test-experiments");

        let config = ExperimentConfig::new("test", "llama", "dataset");
        let exp = tracker
            .create_experiment(config)
            .await
            .expect("create experiment");

        let run = tracker
            .start_run(&exp.id, Some("test-run".to_string()))
            .expect("start run");

        let mut metrics = HashMap::new();
        metrics.insert("loss".to_string(), 0.5);
        tracker
            .log_metrics(&exp.id, &run.id, 1, metrics)
            .expect("log");

        let mut final_metrics = HashMap::new();
        final_metrics.insert("loss".to_string(), 0.3);
        tracker
            .complete_run(&exp.id, &run.id, final_metrics)
            .expect("complete");

        let exp = tracker
            .get_experiment(&exp.id)
            .await
            .expect("get experiment");
        let run = exp.get_run(&run.id).expect("get run");

        assert_eq!(run.status, RunStatus::Completed);
        assert_eq!(run.final_metrics.get("loss"), Some(&0.3));
    }
}
