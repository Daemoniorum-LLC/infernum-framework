//! # Paimon - LLM Studio
//!
//! *"Teaches arts, sciences, and secret things; gives good familiars"*
//!
//! Paimon is the LLM Studio for the Infernum ecosystem, providing a comprehensive
//! platform for custom model development supercharged by AI agents.
//!
//! ## Features
//!
//! - **Dataset Management**: Upload, curate, validate, and augment training data
//! - **Experiment Tracking**: Track training runs, compare metrics, hyperparameter search
//! - **Prompt Studio**: Version-controlled prompts with A/B testing
//! - **Model Registry**: Version, deploy, and rollback models
//! - **Agent-Powered Development**: AI agents that accelerate every step
//!
//! ## Agent Familiars
//!
//! Paimon provides specialized AI agents ("familiars") for each phase:
//!
//! - **Data Curator**: Generates synthetic data, validates quality, suggests augmentations
//! - **Training Coach**: Monitors runs, detects issues, suggests interventions
//! - **Eval Analyst**: Interprets benchmarks, compares models, suggests improvements
//! - **Hyperparam Optimizer**: Analyzes datasets, suggests configurations
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                         PAIMON                               │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Dataset Manager ─────┬───── Experiment Tracker             │
//! │         │             │              │                       │
//! │         ▼             ▼              ▼                       │
//! │  ┌─────────────────────────────────────────────┐            │
//! │  │              AGENT FAMILIARS                 │            │
//! │  │  Curator │ Coach │ Analyst │ Optimizer      │            │
//! │  └─────────────────────────────────────────────┘            │
//! │         │             │              │                       │
//! │         ▼             ▼              ▼                       │
//! │     ASMODEUS       BELETH        STOLAS                     │
//! │   (Fine-tuning)   (Agents)       (RAG)                      │
//! └─────────────────────────────────────────────────────────────┘
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![deny(clippy::unwrap_used)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod agents;
pub mod dataset;
pub mod experiment;
pub mod llm;
pub mod persistence;
pub mod prompt;
pub mod registry;
pub mod studio;

// Re-export main types
pub use studio::{Studio, StudioConfig, StudioError};
pub use dataset::{Dataset, DatasetConfig, DatasetManager, DatasetSplit, Example};
pub use experiment::{Experiment, ExperimentConfig, ExperimentTracker, Run, RunStatus};
pub use persistence::{StudioDatabase, DatabaseConfig, PersistenceError};
pub use prompt::{PromptStudio, PromptTemplate, PromptVersion, TestResult};
pub use registry::{Model, ModelMetadata, ModelRegistry, ModelStage, ModelVersion};
pub use agents::{
    DataCuratorAgent, TrainingCoachAgent, EvalAnalystAgent, HyperparamOptimizerAgent,
    AgentSuggestion, AugmentationStrategy, TrainingIssue, ImprovementPlan,
    coach::{TrainingMetrics, RunHealth, RunAnalysis},
};

/// Re-export analyst types for benchmark/roadmap functionality.
pub mod analyst {
    pub use crate::agents::analyst::{
        BenchmarkResults, BenchmarkScore, NarrativeReport, CompetitiveReport, ModelComparison,
    };
}
