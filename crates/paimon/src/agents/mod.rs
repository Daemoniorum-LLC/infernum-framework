//! Agent-powered development assistants ("familiars").
//!
//! These AI agents supercharge every phase of model development:
//!
//! - **Data Curator**: Generates synthetic data, validates quality, suggests augmentations
//! - **Training Coach**: Monitors runs, detects issues, suggests interventions
//! - **Eval Analyst**: Interprets benchmarks, compares models, suggests improvements
//! - **Hyperparam Optimizer**: Analyzes datasets, suggests configurations

/// Evaluation analyst agent module.
pub mod analyst;
/// Training coach agent module.
pub mod coach;
mod curator;
mod optimizer;

pub use analyst::EvalAnalystAgent;
pub use coach::TrainingCoachAgent;
pub use curator::DataCuratorAgent;
pub use optimizer::HyperparamOptimizerAgent;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors from agent operations.
#[derive(Debug, Error)]
pub enum AgentError {
    /// LLM provider error.
    #[error("LLM error: {0}")]
    Llm(String),

    /// Generation failed.
    #[error("Generation failed: {0}")]
    Generation(String),

    /// Analysis failed.
    #[error("Analysis failed: {0}")]
    Analysis(String),

    /// Invalid response from LLM.
    #[error("Invalid LLM response: {0}")]
    InvalidResponse(String),
}

/// Result type for agent operations.
pub type Result<T> = std::result::Result<T, AgentError>;

/// A suggestion from an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSuggestion {
    /// Suggestion category.
    pub category: String,
    /// Suggestion text.
    pub suggestion: String,
    /// Confidence score (0.0 - 1.0).
    pub confidence: f32,
    /// Reasoning behind the suggestion.
    pub reasoning: Option<String>,
    /// Priority (1 = highest).
    pub priority: u32,
}

impl AgentSuggestion {
    /// Creates a new suggestion.
    pub fn new(category: impl Into<String>, suggestion: impl Into<String>) -> Self {
        Self {
            category: category.into(),
            suggestion: suggestion.into(),
            confidence: 0.8,
            reasoning: None,
            priority: 2,
        }
    }

    /// Sets confidence.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    /// Sets reasoning.
    pub fn with_reasoning(mut self, reasoning: impl Into<String>) -> Self {
        self.reasoning = Some(reasoning.into());
        self
    }

    /// Sets priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }
}

/// Strategy for data augmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationStrategy {
    /// Strategy name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Expected improvement.
    pub expected_improvement: String,
    /// Difficulty to implement.
    pub difficulty: Difficulty,
    /// Recommended example count.
    pub recommended_count: usize,
}

/// Difficulty level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Difficulty {
    /// Easy to implement.
    Easy,
    /// Medium difficulty.
    Medium,
    /// Hard to implement.
    Hard,
}

/// A detected training issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingIssue {
    /// Issue type.
    pub issue_type: TrainingIssueType,
    /// Description.
    pub description: String,
    /// Severity (1-5, 5 being critical).
    pub severity: u8,
    /// Suggested action.
    pub suggested_action: String,
    /// Metrics that triggered detection.
    pub evidence: Vec<String>,
}

/// Types of training issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingIssueType {
    /// Model is overfitting to training data.
    Overfitting,
    /// Model is underfitting.
    Underfitting,
    /// Loss is diverging.
    Divergence,
    /// Training has plateaued.
    Plateau,
    /// Gradient explosion.
    GradientExplosion,
    /// Gradient vanishing.
    GradientVanishing,
    /// Learning rate issues.
    LearningRateIssue,
    /// Memory pressure.
    MemoryPressure,
}

/// Plan for model improvement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementPlan {
    /// Plan title.
    pub title: String,
    /// Executive summary.
    pub summary: String,
    /// Prioritized steps.
    pub steps: Vec<ImprovementStep>,
    /// Expected outcome.
    pub expected_outcome: String,
    /// Estimated effort.
    pub estimated_effort: String,
}

/// A step in an improvement plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementStep {
    /// Step number.
    pub step: u32,
    /// Action to take.
    pub action: String,
    /// Rationale.
    pub rationale: String,
    /// Expected impact.
    pub impact: String,
}
