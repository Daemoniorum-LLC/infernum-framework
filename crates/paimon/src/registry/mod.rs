//! Model Registry - Version, deploy, and manage fine-tuned models.
//!
//! The Model Registry provides:
//! - Model versioning with metadata
//! - Deployment management (staging, production)
//! - Rollback capabilities
//! - Model lineage tracking

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{info, info_span, warn};
use uuid::Uuid;

use crate::persistence::StudioDatabase;

/// Errors that can occur in the model registry.
#[derive(Debug, Error)]
pub enum RegistryError {
    /// Model not found.
    #[error("Model not found: {0}")]
    NotFound(String),

    /// Version not found.
    #[error("Version not found: {0}")]
    VersionNotFound(String),

    /// Deployment not found.
    #[error("Deployment not found: {0}")]
    DeploymentNotFound(String),

    /// Invalid transition.
    #[error("Invalid state transition: {from:?} -> {to:?}")]
    InvalidTransition {
        /// The current model stage.
        from: ModelStage,
        /// The attempted target stage.
        to: ModelStage,
    },

    /// Already deployed.
    #[error("Model version {0} is already deployed to {1}")]
    AlreadyDeployed(String, String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Result type for registry operations.
pub type Result<T> = std::result::Result<T, RegistryError>;

/// A registered model with multiple versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    /// Unique identifier.
    pub id: String,
    /// Model name.
    pub name: String,
    /// Description.
    pub description: Option<String>,
    /// Base model used for fine-tuning.
    pub base_model: String,
    /// Task type (e.g., text-generation, classification).
    pub task_type: String,
    /// All versions.
    pub versions: Vec<ModelVersion>,
    /// Tags.
    pub tags: Vec<String>,
    /// Owner/creator.
    pub owner: Option<String>,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp.
    pub updated_at: DateTime<Utc>,
}

impl Model {
    /// Creates a new model.
    pub fn new(name: impl Into<String>, base_model: impl Into<String>, task_type: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.into(),
            description: None,
            base_model: base_model.into(),
            task_type: task_type.into(),
            versions: Vec::new(),
            tags: Vec::new(),
            owner: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Sets the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Sets the owner.
    pub fn with_owner(mut self, owner: impl Into<String>) -> Self {
        self.owner = Some(owner.into());
        self
    }

    /// Adds tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Creates a new version.
    pub fn create_version(&mut self, metadata: ModelMetadata) -> &ModelVersion {
        let version_num = self.versions.len() as u32 + 1;
        let version = ModelVersion::new(version_num, metadata);

        self.versions.push(version);
        self.updated_at = Utc::now();

        self.versions.last().expect("just pushed")
    }

    /// Gets a version by number.
    pub fn get_version(&self, version_num: u32) -> Option<&ModelVersion> {
        self.versions.iter().find(|v| v.version == version_num)
    }

    /// Gets a mutable version by number.
    pub fn get_version_mut(&mut self, version_num: u32) -> Option<&mut ModelVersion> {
        self.versions.iter_mut().find(|v| v.version == version_num)
    }

    /// Gets the latest version.
    pub fn latest_version(&self) -> Option<&ModelVersion> {
        self.versions.last()
    }

    /// Gets versions in a specific stage.
    pub fn versions_in_stage(&self, stage: ModelStage) -> Vec<&ModelVersion> {
        self.versions.iter()
            .filter(|v| v.stage == stage)
            .collect()
    }

    /// Gets the production version.
    pub fn production_version(&self) -> Option<&ModelVersion> {
        self.versions_in_stage(ModelStage::Production).into_iter().last()
    }
}

/// A specific version of a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    /// Version ID.
    pub id: String,
    /// Version number.
    pub version: u32,
    /// Current stage.
    pub stage: ModelStage,
    /// Model metadata.
    pub metadata: ModelMetadata,
    /// Path to model artifacts.
    pub artifact_path: Option<PathBuf>,
    /// Model metrics.
    pub metrics: HashMap<String, f64>,
    /// Linked experiment run ID.
    pub experiment_run_id: Option<String>,
    /// Linked dataset ID.
    pub dataset_id: Option<String>,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Stage transition history.
    pub stage_history: Vec<StageTransition>,
}

impl ModelVersion {
    /// Creates a new version.
    pub fn new(version: u32, metadata: ModelMetadata) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            version,
            stage: ModelStage::Development,
            metadata,
            artifact_path: None,
            metrics: HashMap::new(),
            experiment_run_id: None,
            dataset_id: None,
            created_at: Utc::now(),
            stage_history: vec![StageTransition {
                from: None,
                to: ModelStage::Development,
                timestamp: Utc::now(),
                reason: Some("Initial creation".to_string()),
            }],
        }
    }

    /// Sets the artifact path.
    pub fn with_artifact_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.artifact_path = Some(path.into());
        self
    }

    /// Links to an experiment run.
    pub fn with_experiment_run(mut self, run_id: impl Into<String>) -> Self {
        self.experiment_run_id = Some(run_id.into());
        self
    }

    /// Links to a dataset.
    pub fn with_dataset(mut self, dataset_id: impl Into<String>) -> Self {
        self.dataset_id = Some(dataset_id.into());
        self
    }

    /// Sets metrics.
    pub fn with_metrics(mut self, metrics: HashMap<String, f64>) -> Self {
        self.metrics = metrics;
        self
    }

    /// Transitions to a new stage.
    pub fn transition_to(&mut self, stage: ModelStage, reason: Option<String>) -> Result<()> {
        // Validate transition
        let valid = match (&self.stage, &stage) {
            (ModelStage::Development, ModelStage::Staging) => true,
            (ModelStage::Staging, ModelStage::Production) => true,
            (ModelStage::Staging, ModelStage::Development) => true, // Reject
            (ModelStage::Production, ModelStage::Archived) => true,
            (ModelStage::Production, ModelStage::Staging) => true, // Rollback
            (_, ModelStage::Archived) => true, // Can always archive
            _ => false,
        };

        if !valid {
            return Err(RegistryError::InvalidTransition {
                from: self.stage,
                to: stage,
            });
        }

        self.stage_history.push(StageTransition {
            from: Some(self.stage),
            to: stage,
            timestamp: Utc::now(),
            reason,
        });

        self.stage = stage;
        Ok(())
    }
}

/// Metadata about a model version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Training configuration used.
    pub training_config: Option<serde_json::Value>,
    /// LoRA configuration (if applicable).
    pub lora_config: Option<LoraMetadata>,
    /// Quantization info.
    pub quantization: Option<QuantizationInfo>,
    /// Model size in bytes.
    pub size_bytes: Option<u64>,
    /// Number of parameters.
    pub num_parameters: Option<u64>,
    /// Framework used (e.g., transformers, vllm).
    pub framework: Option<String>,
    /// Format (e.g., safetensors, gguf).
    pub format: Option<String>,
    /// Additional custom metadata.
    pub custom: HashMap<String, serde_json::Value>,
}

impl ModelMetadata {
    /// Creates new empty metadata.
    pub fn new() -> Self {
        Self {
            training_config: None,
            lora_config: None,
            quantization: None,
            size_bytes: None,
            num_parameters: None,
            framework: None,
            format: None,
            custom: HashMap::new(),
        }
    }

    /// Sets LoRA configuration.
    pub fn with_lora(mut self, lora: LoraMetadata) -> Self {
        self.lora_config = Some(lora);
        self
    }

    /// Sets quantization info.
    pub fn with_quantization(mut self, quant: QuantizationInfo) -> Self {
        self.quantization = Some(quant);
        self
    }

    /// Sets model size.
    pub fn with_size(mut self, size_bytes: u64) -> Self {
        self.size_bytes = Some(size_bytes);
        self
    }

    /// Sets format.
    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// LoRA adapter metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraMetadata {
    /// LoRA rank.
    pub rank: u32,
    /// LoRA alpha.
    pub alpha: u32,
    /// Dropout rate.
    pub dropout: f32,
    /// Target modules.
    pub target_modules: Vec<String>,
}

/// Quantization information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationInfo {
    /// Quantization method (e.g., GPTQ, AWQ, GGUF).
    pub method: String,
    /// Bits (e.g., 4, 8).
    pub bits: u8,
    /// Group size.
    pub group_size: Option<u32>,
}

/// Stage/environment of a model version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelStage {
    /// In development/testing.
    Development,
    /// Staged for production testing.
    Staging,
    /// Deployed to production.
    Production,
    /// Archived/retired.
    Archived,
}

impl ModelStage {
    /// Returns a human-readable name.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Development => "Development",
            Self::Staging => "Staging",
            Self::Production => "Production",
            Self::Archived => "Archived",
        }
    }
}

/// A stage transition event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageTransition {
    /// Previous stage.
    pub from: Option<ModelStage>,
    /// New stage.
    pub to: ModelStage,
    /// When the transition occurred.
    pub timestamp: DateTime<Utc>,
    /// Reason for transition.
    pub reason: Option<String>,
}

/// A deployment of a model version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Deployment {
    /// Deployment ID.
    pub id: String,
    /// Deployment name/alias.
    pub name: String,
    /// Model ID.
    pub model_id: String,
    /// Model version.
    pub model_version: u32,
    /// Environment.
    pub environment: DeploymentEnvironment,
    /// Status.
    pub status: DeploymentStatus,
    /// Endpoint URL (if deployed).
    pub endpoint_url: Option<String>,
    /// Resource configuration.
    pub resources: DeploymentResources,
    /// Created timestamp.
    pub created_at: DateTime<Utc>,
    /// Last updated.
    pub updated_at: DateTime<Utc>,
    /// Deployment history.
    pub history: Vec<DeploymentEvent>,
}

impl Deployment {
    /// Creates a new deployment.
    pub fn new(
        name: impl Into<String>,
        model_id: impl Into<String>,
        model_version: u32,
        environment: DeploymentEnvironment,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.into(),
            model_id: model_id.into(),
            model_version,
            environment,
            status: DeploymentStatus::Pending,
            endpoint_url: None,
            resources: DeploymentResources::default(),
            created_at: now,
            updated_at: now,
            history: vec![DeploymentEvent {
                event_type: DeploymentEventType::Created,
                timestamp: now,
                details: None,
            }],
        }
    }

    /// Records a deployment event.
    pub fn record_event(&mut self, event_type: DeploymentEventType, details: Option<String>) {
        self.history.push(DeploymentEvent {
            event_type,
            timestamp: Utc::now(),
            details,
        });
        self.updated_at = Utc::now();
    }

    /// Updates to a new model version.
    pub fn update_version(&mut self, new_version: u32) {
        let old_version = self.model_version;
        self.model_version = new_version;
        self.record_event(
            DeploymentEventType::Updated,
            Some(format!("Version {} -> {}", old_version, new_version)),
        );
    }
}

/// Deployment environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeploymentEnvironment {
    /// Development/testing.
    Development,
    /// Staging.
    Staging,
    /// Production.
    Production,
}

/// Deployment status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeploymentStatus {
    /// Pending deployment.
    Pending,
    /// Deploying.
    Deploying,
    /// Running.
    Running,
    /// Failed.
    Failed,
    /// Stopped.
    Stopped,
}

/// Resource configuration for deployment.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeploymentResources {
    /// Number of replicas.
    pub replicas: u32,
    /// GPU type.
    pub gpu_type: Option<String>,
    /// Number of GPUs per replica.
    pub gpus_per_replica: u32,
    /// Memory limit (e.g., "16Gi").
    pub memory_limit: Option<String>,
    /// CPU limit (e.g., "4").
    pub cpu_limit: Option<String>,
}

/// A deployment event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentEvent {
    /// Event type.
    pub event_type: DeploymentEventType,
    /// When the event occurred.
    pub timestamp: DateTime<Utc>,
    /// Event details.
    pub details: Option<String>,
}

/// Types of deployment events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeploymentEventType {
    /// Deployment created.
    Created,
    /// Deployment started.
    Started,
    /// Deployment updated.
    Updated,
    /// Deployment stopped.
    Stopped,
    /// Deployment failed.
    Failed,
    /// Health check passed.
    HealthCheckPassed,
    /// Health check failed.
    HealthCheckFailed,
    /// Scaled up/down.
    Scaled,
    /// Rolled back.
    RolledBack,
}

/// The Model Registry manages all models and deployments.
pub struct ModelRegistry {
    /// All models.
    models: Arc<RwLock<HashMap<String, Model>>>,
    /// All deployments.
    deployments: Arc<RwLock<HashMap<String, Deployment>>>,
    /// Storage directory.
    storage_dir: PathBuf,
    /// Optional database for persistence.
    db: Option<Arc<StudioDatabase>>,
}

impl ModelRegistry {
    /// Creates a new model registry.
    pub fn new(storage_dir: impl Into<PathBuf>) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            deployments: Arc::new(RwLock::new(HashMap::new())),
            storage_dir: storage_dir.into(),
            db: None,
        }
    }

    /// Creates a model registry with SQLite persistence.
    pub fn with_database(storage_dir: impl Into<PathBuf>, db: Arc<StudioDatabase>) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            deployments: Arc::new(RwLock::new(HashMap::new())),
            storage_dir: storage_dir.into(),
            db: Some(db),
        }
    }

    /// Returns the number of models.
    pub async fn count(&self) -> usize {
        if let Some(ref db) = self.db {
            db.count_models().unwrap_or(0)
        } else {
            self.models.read().len()
        }
    }

    /// Registers a new model.
    pub fn register_model(&self, model: Model) -> Result<String> {
        let _span = info_span!("registry.register_model", name = %model.name).entered();

        let id = model.id.clone();
        info!(model_id = %id, "Registered model");

        if let Some(ref db) = self.db {
            db.save_model(&model)
                .map_err(|e| RegistryError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
        } else {
            self.models.write().insert(id.clone(), model);
        }

        Ok(id)
    }

    /// Deletes a model by ID.
    pub fn delete_model(&self, id: &str) -> Result<()> {
        let _span = info_span!("registry.delete_model", id = %id).entered();

        if let Some(ref db) = self.db {
            db.delete_model(id)
                .map_err(|e| RegistryError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
        } else {
            self.models.write().remove(id);
        }

        info!("Deleted model");
        Ok(())
    }

    /// Gets a model by ID.
    pub fn get_model(&self, id: &str) -> Option<Model> {
        if let Some(ref db) = self.db {
            db.load_model(id).ok().flatten()
        } else {
            self.models.read().get(id).cloned()
        }
    }

    /// Gets a model by name.
    pub fn get_model_by_name(&self, name: &str) -> Option<Model> {
        if self.db.is_some() {
            let models = self.list_models();
            models.into_iter().find(|m| m.name == name)
        } else {
            self.models.read().values()
                .find(|m| m.name == name)
                .cloned()
        }
    }

    /// Lists all models.
    pub fn list_models(&self) -> Vec<Model> {
        if let Some(ref db) = self.db {
            let summaries = db.list_models().unwrap_or_default();
            let mut models = Vec::new();
            for (id, _, _) in summaries {
                if let Some(model) = db.load_model(&id).ok().flatten() {
                    models.push(model);
                }
            }
            models
        } else {
            self.models.read().values().cloned().collect()
        }
    }

    /// Lists models by tag.
    pub fn list_models_by_tag(&self, tag: &str) -> Vec<Model> {
        let models = self.list_models();
        models.into_iter()
            .filter(|m| m.tags.contains(&tag.to_string()))
            .collect()
    }

    /// Creates a new version for a model.
    pub fn create_version(
        &self,
        model_id: &str,
        metadata: ModelMetadata,
    ) -> Result<ModelVersion> {
        let _span = info_span!("registry.create_version", model_id = %model_id).entered();

        if let Some(ref db) = self.db {
            let mut model = db.load_model(model_id)
                .map_err(|e| RegistryError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?
                .ok_or_else(|| RegistryError::NotFound(model_id.to_string()))?;

            let version = model.create_version(metadata).clone();

            db.save_model(&model)
                .map_err(|e| RegistryError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;

            info!(version = version.version, "Created model version");
            Ok(version)
        } else {
            let mut models = self.models.write();
            let model = models.get_mut(model_id)
                .ok_or_else(|| RegistryError::NotFound(model_id.to_string()))?;

            let version = model.create_version(metadata).clone();
            info!(version = version.version, "Created model version");

            Ok(version)
        }
    }

    /// Transitions a model version to a new stage.
    pub fn transition_stage(
        &self,
        model_id: &str,
        version_num: u32,
        stage: ModelStage,
        reason: Option<String>,
    ) -> Result<()> {
        let _span = info_span!(
            "registry.transition_stage",
            model_id = %model_id,
            version = version_num,
            stage = ?stage
        ).entered();

        if let Some(ref db) = self.db {
            let mut model = db.load_model(model_id)
                .map_err(|e| RegistryError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?
                .ok_or_else(|| RegistryError::NotFound(model_id.to_string()))?;

            let version = model.get_version_mut(version_num)
                .ok_or_else(|| RegistryError::VersionNotFound(format!("v{}", version_num)))?;

            version.transition_to(stage, reason)?;

            db.save_model(&model)
                .map_err(|e| RegistryError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
        } else {
            let mut models = self.models.write();
            let model = models.get_mut(model_id)
                .ok_or_else(|| RegistryError::NotFound(model_id.to_string()))?;

            let version = model.get_version_mut(version_num)
                .ok_or_else(|| RegistryError::VersionNotFound(format!("v{}", version_num)))?;

            version.transition_to(stage, reason)?;
        }

        info!("Transitioned to {:?}", stage);
        Ok(())
    }

    /// Sets metrics for a model version.
    pub fn set_metrics(
        &self,
        model_id: &str,
        version_num: u32,
        metrics: HashMap<String, f64>,
    ) -> Result<()> {
        if let Some(ref db) = self.db {
            let mut model = db.load_model(model_id)
                .map_err(|e| RegistryError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?
                .ok_or_else(|| RegistryError::NotFound(model_id.to_string()))?;

            let version = model.get_version_mut(version_num)
                .ok_or_else(|| RegistryError::VersionNotFound(format!("v{}", version_num)))?;

            version.metrics = metrics;

            db.save_model(&model)
                .map_err(|e| RegistryError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
        } else {
            let mut models = self.models.write();
            let model = models.get_mut(model_id)
                .ok_or_else(|| RegistryError::NotFound(model_id.to_string()))?;

            let version = model.get_version_mut(version_num)
                .ok_or_else(|| RegistryError::VersionNotFound(format!("v{}", version_num)))?;

            version.metrics = metrics;
        }
        Ok(())
    }

    /// Creates a deployment.
    pub fn create_deployment(
        &self,
        name: impl Into<String>,
        model_id: &str,
        version_num: u32,
        environment: DeploymentEnvironment,
    ) -> Result<Deployment> {
        let _span = info_span!(
            "registry.create_deployment",
            model_id = %model_id,
            version = version_num,
            environment = ?environment
        ).entered();

        // Verify model and version exist
        let model = self.get_model(model_id)
            .ok_or_else(|| RegistryError::NotFound(model_id.to_string()))?;

        if model.get_version(version_num).is_none() {
            return Err(RegistryError::VersionNotFound(format!("v{}", version_num)));
        }

        let deployment = Deployment::new(name, model_id, version_num, environment);
        let id = deployment.id.clone();

        info!(deployment_id = %id, "Created deployment");

        if let Some(ref db) = self.db {
            db.save_deployment(&deployment)
                .map_err(|e| RegistryError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
        } else {
            self.deployments.write().insert(id, deployment.clone());
        }

        Ok(deployment)
    }

    /// Gets a deployment by ID.
    pub fn get_deployment(&self, id: &str) -> Option<Deployment> {
        if let Some(ref db) = self.db {
            db.load_deployment(id).ok().flatten()
        } else {
            self.deployments.read().get(id).cloned()
        }
    }

    /// Gets a deployment by name.
    pub fn get_deployment_by_name(&self, name: &str) -> Option<Deployment> {
        let deployments = self.list_deployments();
        deployments.into_iter().find(|d| d.name == name)
    }

    /// Lists all deployments.
    pub fn list_deployments(&self) -> Vec<Deployment> {
        if let Some(ref db) = self.db {
            let summaries = db.list_deployments().unwrap_or_default();
            let mut deployments = Vec::new();
            for (id, _, _) in summaries {
                if let Some(deployment) = db.load_deployment(&id).ok().flatten() {
                    deployments.push(deployment);
                }
            }
            deployments
        } else {
            self.deployments.read().values().cloned().collect()
        }
    }

    /// Lists deployments for a model.
    pub fn list_deployments_for_model(&self, model_id: &str) -> Vec<Deployment> {
        let deployments = self.list_deployments();
        deployments.into_iter()
            .filter(|d| d.model_id == model_id)
            .collect()
    }

    /// Deletes a deployment by ID.
    pub fn delete_deployment(&self, id: &str) -> Result<()> {
        let _span = info_span!("registry.delete_deployment", id = %id).entered();

        if let Some(ref db) = self.db {
            db.delete_deployment(id)
                .map_err(|e| RegistryError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
        } else {
            self.deployments.write().remove(id);
        }

        info!("Deleted deployment");
        Ok(())
    }

    /// Updates a deployment's status.
    pub fn update_deployment_status(
        &self,
        deployment_id: &str,
        status: DeploymentStatus,
        details: Option<String>,
    ) -> Result<()> {
        let _span = info_span!(
            "registry.update_deployment_status",
            deployment_id = %deployment_id,
            status = ?status
        ).entered();

        if let Some(ref db) = self.db {
            let mut deployment = db.load_deployment(deployment_id)
                .map_err(|e| RegistryError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?
                .ok_or_else(|| RegistryError::DeploymentNotFound(deployment_id.to_string()))?;

            let event_type = match status {
                DeploymentStatus::Running => DeploymentEventType::Started,
                DeploymentStatus::Failed => DeploymentEventType::Failed,
                DeploymentStatus::Stopped => DeploymentEventType::Stopped,
                _ => DeploymentEventType::Updated,
            };

            deployment.status = status;
            deployment.record_event(event_type, details);

            db.save_deployment(&deployment)
                .map_err(|e| RegistryError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
        } else {
            let mut deployments = self.deployments.write();
            let deployment = deployments.get_mut(deployment_id)
                .ok_or_else(|| RegistryError::DeploymentNotFound(deployment_id.to_string()))?;

            let event_type = match status {
                DeploymentStatus::Running => DeploymentEventType::Started,
                DeploymentStatus::Failed => DeploymentEventType::Failed,
                DeploymentStatus::Stopped => DeploymentEventType::Stopped,
                _ => DeploymentEventType::Updated,
            };

            deployment.status = status;
            deployment.record_event(event_type, details);
        }

        info!("Updated deployment status");
        Ok(())
    }

    /// Rolls back a deployment to a previous version.
    pub fn rollback_deployment(
        &self,
        deployment_id: &str,
        target_version: u32,
    ) -> Result<()> {
        let _span = info_span!(
            "registry.rollback_deployment",
            deployment_id = %deployment_id,
            target_version = target_version
        ).entered();

        // Verify version exists
        let deployment = self.get_deployment(deployment_id)
            .ok_or_else(|| RegistryError::DeploymentNotFound(deployment_id.to_string()))?;

        let model = self.get_model(&deployment.model_id)
            .ok_or_else(|| RegistryError::NotFound(deployment.model_id.clone()))?;

        if model.get_version(target_version).is_none() {
            return Err(RegistryError::VersionNotFound(format!("v{}", target_version)));
        }

        if let Some(ref db) = self.db {
            let mut deployment = db.load_deployment(deployment_id)
                .map_err(|e| RegistryError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?
                .ok_or_else(|| RegistryError::DeploymentNotFound(deployment_id.to_string()))?;

            let old_version = deployment.model_version;
            deployment.model_version = target_version;
            deployment.record_event(
                DeploymentEventType::RolledBack,
                Some(format!("Rolled back from v{} to v{}", old_version, target_version)),
            );

            db.save_deployment(&deployment)
                .map_err(|e| RegistryError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;

            warn!(from = old_version, to = target_version, "Rolled back deployment");
        } else {
            let mut deployments = self.deployments.write();
            let deployment = deployments.get_mut(deployment_id)
                .ok_or_else(|| RegistryError::DeploymentNotFound(deployment_id.to_string()))?;

            let old_version = deployment.model_version;
            deployment.model_version = target_version;
            deployment.record_event(
                DeploymentEventType::RolledBack,
                Some(format!("Rolled back from v{} to v{}", old_version, target_version)),
            );

            warn!(from = old_version, to = target_version, "Rolled back deployment");
        }

        Ok(())
    }

    /// Gets model lineage (what was used to create this model).
    pub fn get_lineage(&self, model_id: &str, version_num: u32) -> Result<ModelLineage> {
        let model = self.get_model(model_id)
            .ok_or_else(|| RegistryError::NotFound(model_id.to_string()))?;

        let version = model.get_version(version_num)
            .ok_or_else(|| RegistryError::VersionNotFound(format!("v{}", version_num)))?;

        Ok(ModelLineage {
            model_id: model_id.to_string(),
            model_name: model.name.clone(),
            version: version_num,
            base_model: model.base_model.clone(),
            dataset_id: version.dataset_id.clone(),
            experiment_run_id: version.experiment_run_id.clone(),
            created_at: version.created_at,
            stage_history: version.stage_history.clone(),
        })
    }

    /// Compares two model versions.
    pub fn compare_versions(
        &self,
        model_id: &str,
        version_a: u32,
        version_b: u32,
    ) -> Result<VersionComparison> {
        let model = self.get_model(model_id)
            .ok_or_else(|| RegistryError::NotFound(model_id.to_string()))?;

        let va = model.get_version(version_a)
            .ok_or_else(|| RegistryError::VersionNotFound(format!("v{}", version_a)))?;
        let vb = model.get_version(version_b)
            .ok_or_else(|| RegistryError::VersionNotFound(format!("v{}", version_b)))?;

        // Find common metrics
        let common_metrics: Vec<String> = va.metrics.keys()
            .filter(|k| vb.metrics.contains_key(*k))
            .cloned()
            .collect();

        let mut metric_comparison = Vec::new();
        for metric in &common_metrics {
            let val_a = va.metrics.get(metric).copied().unwrap_or(0.0);
            let val_b = vb.metrics.get(metric).copied().unwrap_or(0.0);
            metric_comparison.push(MetricComparison {
                name: metric.clone(),
                version_a_value: val_a,
                version_b_value: val_b,
                difference: val_b - val_a,
                percent_change: if val_a != 0.0 { ((val_b - val_a) / val_a) * 100.0 } else { 0.0 },
            });
        }

        Ok(VersionComparison {
            model_id: model_id.to_string(),
            version_a,
            version_b,
            metric_comparison,
        })
    }

    /// Saves registry to disk.
    pub async fn save(&self) -> Result<()> {
        let _span = info_span!("registry.save").entered();

        tokio::fs::create_dir_all(&self.storage_dir).await?;

        // Save models
        let models_path = self.storage_dir.join("models.json");
        let models = self.models.read();
        let models_json = serde_json::to_string_pretty(&*models)?;
        tokio::fs::write(&models_path, models_json).await?;

        // Save deployments
        let deployments_path = self.storage_dir.join("deployments.json");
        let deployments = self.deployments.read();
        let deployments_json = serde_json::to_string_pretty(&*deployments)?;
        tokio::fs::write(&deployments_path, deployments_json).await?;

        info!("Saved registry");
        Ok(())
    }

    /// Loads registry from disk.
    pub async fn load(&self) -> Result<()> {
        let _span = info_span!("registry.load").entered();

        // Load models
        let models_path = self.storage_dir.join("models.json");
        if models_path.exists() {
            let json = tokio::fs::read_to_string(&models_path).await?;
            let models: HashMap<String, Model> = serde_json::from_str(&json)?;
            *self.models.write() = models;
        }

        // Load deployments
        let deployments_path = self.storage_dir.join("deployments.json");
        if deployments_path.exists() {
            let json = tokio::fs::read_to_string(&deployments_path).await?;
            let deployments: HashMap<String, Deployment> = serde_json::from_str(&json)?;
            *self.deployments.write() = deployments;
        }

        info!(
            models = self.models.read().len(),
            deployments = self.deployments.read().len(),
            "Loaded registry"
        );
        Ok(())
    }
}

/// Model lineage information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLineage {
    /// Model ID.
    pub model_id: String,
    /// Model name.
    pub model_name: String,
    /// Version number.
    pub version: u32,
    /// Base model used.
    pub base_model: String,
    /// Dataset used for training.
    pub dataset_id: Option<String>,
    /// Experiment run that produced this.
    pub experiment_run_id: Option<String>,
    /// When created.
    pub created_at: DateTime<Utc>,
    /// Stage history.
    pub stage_history: Vec<StageTransition>,
}

/// Comparison between two versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionComparison {
    /// Model ID.
    pub model_id: String,
    /// First version.
    pub version_a: u32,
    /// Second version.
    pub version_b: u32,
    /// Metric comparisons.
    pub metric_comparison: Vec<MetricComparison>,
}

/// Comparison of a single metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComparison {
    /// Metric name.
    pub name: String,
    /// Value in version A.
    pub version_a_value: f64,
    /// Value in version B.
    pub version_b_value: f64,
    /// Absolute difference.
    pub difference: f64,
    /// Percent change.
    pub percent_change: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_model() {
        let model = Model::new("test-model", "llama-7b", "text-generation")
            .with_description("A test model")
            .with_owner("test-user")
            .with_tags(vec!["test".to_string()]);

        assert_eq!(model.name, "test-model");
        assert_eq!(model.base_model, "llama-7b");
        assert!(model.versions.is_empty());
    }

    #[test]
    fn test_model_versioning() {
        let mut model = Model::new("test", "llama", "gen");

        let metadata = ModelMetadata::new()
            .with_format("safetensors");

        let v1 = model.create_version(metadata.clone());
        assert_eq!(v1.version, 1);
        assert_eq!(v1.stage, ModelStage::Development);

        let v2 = model.create_version(metadata);
        assert_eq!(v2.version, 2);

        assert_eq!(model.versions.len(), 2);
        assert_eq!(model.latest_version().map(|v| v.version), Some(2));
    }

    #[test]
    fn test_stage_transitions() {
        let mut version = ModelVersion::new(1, ModelMetadata::new());

        // Valid: Dev -> Staging
        version.transition_to(ModelStage::Staging, Some("Ready for testing".to_string()))
            .expect("transition to staging");
        assert_eq!(version.stage, ModelStage::Staging);

        // Valid: Staging -> Production
        version.transition_to(ModelStage::Production, Some("Passed QA".to_string()))
            .expect("transition to production");
        assert_eq!(version.stage, ModelStage::Production);

        // History should have 3 entries (initial + 2 transitions)
        assert_eq!(version.stage_history.len(), 3);
    }

    #[test]
    fn test_invalid_transition() {
        let mut version = ModelVersion::new(1, ModelMetadata::new());

        // Invalid: Dev -> Production (must go through Staging)
        let result = version.transition_to(ModelStage::Production, None);
        assert!(matches!(result, Err(RegistryError::InvalidTransition { .. })));
    }

    #[test]
    fn test_deployment() {
        let mut deployment = Deployment::new(
            "prod-deployment",
            "model-123",
            1,
            DeploymentEnvironment::Production,
        );

        assert_eq!(deployment.status, DeploymentStatus::Pending);

        deployment.record_event(DeploymentEventType::Started, None);
        deployment.status = DeploymentStatus::Running;

        assert_eq!(deployment.history.len(), 2); // Created + Started
    }

    #[test]
    fn test_registry() {
        let registry = ModelRegistry::new("/tmp/test-registry");

        let model = Model::new("test-model", "llama-7b", "generation");
        let model_id = registry.register_model(model).expect("register model");

        let metadata = ModelMetadata::new().with_format("safetensors");
        let version = registry.create_version(&model_id, metadata).expect("create version");

        assert_eq!(version.version, 1);

        // Transition to staging
        registry.transition_stage(&model_id, 1, ModelStage::Staging, None)
            .expect("transition");

        let model = registry.get_model(&model_id).expect("get model");
        assert_eq!(model.get_version(1).unwrap().stage, ModelStage::Staging);
    }

    #[test]
    fn test_deployment_lifecycle() {
        let registry = ModelRegistry::new("/tmp/test-registry");

        let model = Model::new("test-model", "llama-7b", "generation");
        let model_id = registry.register_model(model).expect("register model");

        registry.create_version(&model_id, ModelMetadata::new()).expect("version");

        let deployment = registry.create_deployment(
            "test-deploy",
            &model_id,
            1,
            DeploymentEnvironment::Staging,
        ).expect("deploy");

        registry.update_deployment_status(&deployment.id, DeploymentStatus::Running, None)
            .expect("update status");

        let deployment = registry.get_deployment(&deployment.id).expect("get deployment");
        assert_eq!(deployment.status, DeploymentStatus::Running);
    }

    #[test]
    fn test_rollback() {
        let registry = ModelRegistry::new("/tmp/test-registry");

        let model = Model::new("test-model", "llama-7b", "generation");
        let model_id = registry.register_model(model).expect("register model");

        registry.create_version(&model_id, ModelMetadata::new()).expect("v1");
        registry.create_version(&model_id, ModelMetadata::new()).expect("v2");

        let deployment = registry.create_deployment(
            "test-deploy",
            &model_id,
            2,
            DeploymentEnvironment::Production,
        ).expect("deploy");

        registry.rollback_deployment(&deployment.id, 1).expect("rollback");

        let deployment = registry.get_deployment(&deployment.id).expect("get");
        assert_eq!(deployment.model_version, 1);
        assert!(deployment.history.iter().any(|e| e.event_type == DeploymentEventType::RolledBack));
    }

    #[test]
    fn test_version_comparison() {
        let registry = ModelRegistry::new("/tmp/test-registry");

        let model = Model::new("test-model", "llama-7b", "generation");
        let model_id = registry.register_model(model).expect("register model");

        registry.create_version(&model_id, ModelMetadata::new()).expect("v1");
        registry.create_version(&model_id, ModelMetadata::new()).expect("v2");

        let mut metrics1 = HashMap::new();
        metrics1.insert("accuracy".to_string(), 0.85);
        metrics1.insert("loss".to_string(), 0.3);
        registry.set_metrics(&model_id, 1, metrics1).expect("set metrics v1");

        let mut metrics2 = HashMap::new();
        metrics2.insert("accuracy".to_string(), 0.90);
        metrics2.insert("loss".to_string(), 0.25);
        registry.set_metrics(&model_id, 2, metrics2).expect("set metrics v2");

        let comparison = registry.compare_versions(&model_id, 1, 2).expect("compare");

        assert_eq!(comparison.metric_comparison.len(), 2);

        let accuracy = comparison.metric_comparison.iter()
            .find(|m| m.name == "accuracy")
            .expect("accuracy metric");
        assert!((accuracy.difference - 0.05).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_registry_with_database() {
        use std::sync::Arc;
        use crate::persistence::StudioDatabase;

        let db = Arc::new(StudioDatabase::in_memory().expect("create db"));
        let registry = ModelRegistry::with_database("/tmp/test-registry", db);

        // Register model
        let model = Model::new("db-model", "llama-7b", "generation");
        let model_id = registry.register_model(model).expect("register model");

        // Create version
        let metadata = ModelMetadata::new().with_format("safetensors");
        let version = registry.create_version(&model_id, metadata).expect("create version");
        assert_eq!(version.version, 1);

        // Get model
        let retrieved = registry.get_model(&model_id).expect("model exists");
        assert_eq!(retrieved.versions.len(), 1);

        // Transition to staging
        registry.transition_stage(&model_id, 1, ModelStage::Staging, None)
            .expect("transition");

        let updated = registry.get_model(&model_id).expect("get model");
        assert_eq!(updated.get_version(1).unwrap().stage, ModelStage::Staging);

        // Create deployment
        let deployment = registry.create_deployment(
            "db-deployment",
            &model_id,
            1,
            DeploymentEnvironment::Staging,
        ).expect("deploy");

        // List models and deployments
        assert_eq!(registry.list_models().len(), 1);
        assert_eq!(registry.list_deployments().len(), 1);

        // Count
        assert_eq!(registry.count().await, 1);

        // Delete model (should cascade to version)
        registry.delete_model(&model_id).expect("delete model");
        assert_eq!(registry.count().await, 0);
    }

    #[tokio::test]
    async fn test_registry_persistence_across_instances() {
        use std::sync::Arc;
        use crate::persistence::StudioDatabase;

        let temp = tempfile::TempDir::new().expect("temp dir");
        let db_path = temp.path().join("test.db");

        let model_id;
        let deployment_id;

        // Create model and deployment with first registry instance
        {
            let db = Arc::new(StudioDatabase::new(&db_path).expect("create db"));
            let registry = ModelRegistry::with_database("/tmp/test-registry", db);

            let model = Model::new("persistent-model", "llama-7b", "generation");
            model_id = registry.register_model(model).expect("register");
            registry.create_version(&model_id, ModelMetadata::new()).expect("version");

            let deployment = registry.create_deployment(
                "persistent-deployment",
                &model_id,
                1,
                DeploymentEnvironment::Production,
            ).expect("deploy");
            deployment_id = deployment.id.clone();
        }

        // Verify data persists with new registry instance
        {
            let db = Arc::new(StudioDatabase::new(&db_path).expect("reopen db"));
            let registry = ModelRegistry::with_database("/tmp/test-registry", db);

            assert_eq!(registry.count().await, 1);

            let loaded_model = registry.get_model(&model_id).expect("model exists");
            assert_eq!(loaded_model.name, "persistent-model");
            assert_eq!(loaded_model.versions.len(), 1);

            let loaded_deployment = registry.get_deployment(&deployment_id).expect("deployment exists");
            assert_eq!(loaded_deployment.name, "persistent-deployment");
            assert_eq!(loaded_deployment.model_version, 1);
        }
    }
}
