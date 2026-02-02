//! Core LLM Studio orchestration.
//!
//! The `Studio` struct is the main entry point for all LLM development workflows,
//! coordinating datasets, experiments, prompts, and AI agents.

use std::path::PathBuf;
use std::sync::Arc;

use thiserror::Error;
use tracing::{info, info_span};

use crate::agents::{DataCuratorAgent, EvalAnalystAgent, HyperparamOptimizerAgent, TrainingCoachAgent};
use crate::dataset::DatasetManager;
use crate::experiment::ExperimentTracker;
use crate::prompt::PromptStudio;
use crate::registry::ModelRegistry;

/// Errors that can occur in the LLM Studio.
#[derive(Debug, Error)]
pub enum StudioError {
    /// Dataset-related error.
    #[error("Dataset error: {0}")]
    Dataset(String),

    /// Experiment-related error.
    #[error("Experiment error: {0}")]
    Experiment(String),

    /// Training-related error.
    #[error("Training error: {0}")]
    Training(String),

    /// Agent-related error.
    #[error("Agent error: {0}")]
    Agent(String),

    /// Model registry error.
    #[error("Registry error: {0}")]
    Registry(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Result type for studio operations.
pub type Result<T> = std::result::Result<T, StudioError>;

/// Configuration for the LLM Studio.
#[derive(Debug, Clone)]
pub struct StudioConfig {
    /// Base directory for studio data.
    pub data_dir: PathBuf,

    /// Directory for datasets.
    pub datasets_dir: PathBuf,

    /// Directory for experiments.
    pub experiments_dir: PathBuf,

    /// Directory for model registry.
    pub models_dir: PathBuf,

    /// Directory for prompts.
    pub prompts_dir: PathBuf,

    /// Enable agent-powered features.
    pub enable_agents: bool,

    /// Default model for agent operations.
    pub agent_model: Option<String>,

    /// Maximum concurrent training runs.
    pub max_concurrent_runs: usize,
}

impl Default for StudioConfig {
    fn default() -> Self {
        let base = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("infernum")
            .join("studio");

        Self {
            data_dir: base.clone(),
            datasets_dir: base.join("datasets"),
            experiments_dir: base.join("experiments"),
            models_dir: base.join("models"),
            prompts_dir: base.join("prompts"),
            enable_agents: true,
            agent_model: None,
            max_concurrent_runs: 2,
        }
    }
}

impl StudioConfig {
    /// Creates a new studio configuration with the given base directory.
    pub fn with_base_dir(base: PathBuf) -> Self {
        Self {
            data_dir: base.clone(),
            datasets_dir: base.join("datasets"),
            experiments_dir: base.join("experiments"),
            models_dir: base.join("models"),
            prompts_dir: base.join("prompts"),
            ..Default::default()
        }
    }

    /// Sets the agent model to use.
    pub fn with_agent_model(mut self, model: impl Into<String>) -> Self {
        self.agent_model = Some(model.into());
        self
    }

    /// Disables agent-powered features.
    pub fn without_agents(mut self) -> Self {
        self.enable_agents = false;
        self
    }
}

/// The main LLM Studio orchestrator.
///
/// Coordinates all studio components and provides a unified interface
/// for model development workflows.
pub struct Studio {
    config: StudioConfig,
    dataset_manager: Arc<DatasetManager>,
    experiment_tracker: Arc<ExperimentTracker>,
    prompt_studio: Arc<PromptStudio>,
    model_registry: Arc<ModelRegistry>,

    // Agent familiars
    data_curator: Option<Arc<DataCuratorAgent>>,
    training_coach: Option<Arc<TrainingCoachAgent>>,
    eval_analyst: Option<Arc<EvalAnalystAgent>>,
    hyperparam_optimizer: Option<Arc<HyperparamOptimizerAgent>>,
}

impl Studio {
    /// Creates a new LLM Studio with the given configuration.
    pub async fn new(config: StudioConfig) -> Result<Self> {
        let _span = info_span!("studio.init").entered();

        // Ensure directories exist (async I/O)
        tokio::fs::create_dir_all(&config.datasets_dir).await?;
        tokio::fs::create_dir_all(&config.experiments_dir).await?;
        tokio::fs::create_dir_all(&config.models_dir).await?;
        tokio::fs::create_dir_all(&config.prompts_dir).await?;

        info!(data_dir = %config.data_dir.display(), "Initializing LLM Studio");

        // Initialize core components
        let dataset_manager = Arc::new(DatasetManager::new(config.datasets_dir.clone()));
        let experiment_tracker = Arc::new(ExperimentTracker::new(config.experiments_dir.clone()));
        let prompt_studio = Arc::new(PromptStudio::new(config.prompts_dir.clone()));
        let model_registry = Arc::new(ModelRegistry::new(config.models_dir.clone()));

        // Initialize agents if enabled
        let (data_curator, training_coach, eval_analyst, hyperparam_optimizer) = if config.enable_agents {
            info!("Initializing agent familiars");
            (
                Some(Arc::new(DataCuratorAgent::new(config.agent_model.clone()))),
                Some(Arc::new(TrainingCoachAgent::new(config.agent_model.clone()))),
                Some(Arc::new(EvalAnalystAgent::new(config.agent_model.clone()))),
                Some(Arc::new(HyperparamOptimizerAgent::new(config.agent_model.clone()))),
            )
        } else {
            (None, None, None, None)
        };

        Ok(Self {
            config,
            dataset_manager,
            experiment_tracker,
            prompt_studio,
            model_registry,
            data_curator,
            training_coach,
            eval_analyst,
            hyperparam_optimizer,
        })
    }

    /// Creates a studio with default configuration.
    pub async fn with_defaults() -> Result<Self> {
        Self::new(StudioConfig::default()).await
    }

    /// Returns the studio configuration.
    pub fn config(&self) -> &StudioConfig {
        &self.config
    }

    /// Returns the dataset manager.
    pub fn datasets(&self) -> &Arc<DatasetManager> {
        &self.dataset_manager
    }

    /// Returns the experiment tracker.
    pub fn experiments(&self) -> &Arc<ExperimentTracker> {
        &self.experiment_tracker
    }

    /// Returns the prompt studio.
    pub fn prompts(&self) -> &Arc<PromptStudio> {
        &self.prompt_studio
    }

    /// Returns the model registry.
    pub fn models(&self) -> &Arc<ModelRegistry> {
        &self.model_registry
    }

    /// Returns the data curator agent, if enabled.
    pub fn data_curator(&self) -> Option<&Arc<DataCuratorAgent>> {
        self.data_curator.as_ref()
    }

    /// Returns the training coach agent, if enabled.
    pub fn training_coach(&self) -> Option<&Arc<TrainingCoachAgent>> {
        self.training_coach.as_ref()
    }

    /// Returns the evaluation analyst agent, if enabled.
    pub fn eval_analyst(&self) -> Option<&Arc<EvalAnalystAgent>> {
        self.eval_analyst.as_ref()
    }

    /// Returns the hyperparameter optimizer agent, if enabled.
    pub fn hyperparam_optimizer(&self) -> Option<&Arc<HyperparamOptimizerAgent>> {
        self.hyperparam_optimizer.as_ref()
    }

    /// Checks if agent-powered features are enabled.
    pub fn agents_enabled(&self) -> bool {
        self.config.enable_agents
    }

    /// Gets studio statistics.
    pub async fn stats(&self) -> StudioStats {
        StudioStats {
            datasets_count: self.dataset_manager.count().await,
            experiments_count: self.experiment_tracker.count().await,
            models_count: self.model_registry.count().await,
            prompts_count: self.prompt_studio.count().await,
            agents_enabled: self.config.enable_agents,
        }
    }
}

/// Statistics about the studio.
#[derive(Debug, Clone)]
pub struct StudioStats {
    /// Number of datasets.
    pub datasets_count: usize,
    /// Number of experiments.
    pub experiments_count: usize,
    /// Number of registered models.
    pub models_count: usize,
    /// Number of prompt templates.
    pub prompts_count: usize,
    /// Whether agents are enabled.
    pub agents_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_studio_creation() {
        let temp = TempDir::new().expect("Failed to create temp dir");
        let config = StudioConfig::with_base_dir(temp.path().to_path_buf());

        let studio = Studio::new(config).await.expect("Failed to create studio");

        assert!(studio.agents_enabled());
        assert!(studio.data_curator().is_some());
        assert!(studio.training_coach().is_some());
    }

    #[tokio::test]
    async fn test_studio_without_agents() {
        let temp = TempDir::new().expect("Failed to create temp dir");
        let config = StudioConfig::with_base_dir(temp.path().to_path_buf())
            .without_agents();

        let studio = Studio::new(config).await.expect("Failed to create studio");

        assert!(!studio.agents_enabled());
        assert!(studio.data_curator().is_none());
    }

    #[test]
    fn test_config_builder() {
        let config = StudioConfig::default()
            .with_agent_model("qwen-2.5-7b")
            .without_agents();

        assert_eq!(config.agent_model, Some("qwen-2.5-7b".to_string()));
        assert!(!config.enable_agents);
    }
}
