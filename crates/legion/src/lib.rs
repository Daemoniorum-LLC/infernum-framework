//! # Legion - Holographic Agent Swarm
//!
//! *"We are Legion, for we are many. We think as one, but we see from all perspectives."*
//!
//! Legion enables collective intelligence through holographic agent coordination,
//! where every agent sees the WHOLE task but emphasizes different frequency bands.
//! This is not task decomposition - it's multiple perspectives on the same reality.
//!
//! ## Core Principles
//!
//! 1. **Holographic Distribution**: Each agent receives the FULL task but emphasizes
//!    their frequency band using spectral filtering. Any subset can produce an answer.
//!
//! 2. **Wave Interference Consensus**: Agent contributions superimpose in the Legion Field.
//!    Constructive interference = agreement. Destructive interference = conflict.
//!
//! 3. **Graceful Degradation**: Quality scales with agent count, but never fails.
//!    First fragment = 60% quality. More agents = progressive refinement.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                         ORCHESTRATOR                            │
//! │  Distributes holographic task fragments, gathers contributions  │
//! └───────────────────────────┬─────────────────────────────────────┘
//!                             │
//!     ┌───────────┬───────────┼───────────┬───────────┐
//!     │           │           │           │           │
//!     ▼           ▼           ▼           ▼           ▼
//! ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
//! │ Anima   │ │Strategic│ │Tactical │ │Operative│ │Reflectv │
//! │ DC (∿)  │ │ (⟁)     │ │ (⟀)     │ │ (⊕)     │ │ (◉)     │
//! │ Core ID │ │Planning │ │ Steps   │ │ Work    │ │ Meta    │
//! └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
//!     │           │           │           │           │
//!     └───────────┴───────────┴───────────┴───────────┘
//!                             │
//!                             ▼
//!            ┌────────────────────────────────┐
//!            │       LEGION FIELD (∿)         │
//!            │  Contributions superimpose     │
//!            │  Interference → Consensus      │
//!            └────────────────────────────────┘
//! ```
//!
//! ## Frequency Bands
//!
//! - **Anima**: Core identity (DC component) - never filtered out
//! - **Strategic**: High-level planning (ultra-low freq, 0.0-0.1)
//! - **Tactical**: Step-by-step execution (low freq, 0.1-0.3)
//! - **Operational**: The actual work (mid freq, 0.3-0.6)
//! - **Verification**: Quality checking (high freq, 0.6-0.9)
//! - **Reflective**: Meta-cognition (ultra-high freq, 0.9-1.0)
//!
//! ## Example
//!
//! ```ignore
//! use legion::{Legion, LegionConfig, FrequencyBand};
//!
//! let legion = Legion::builder()
//!     .with_agent_count(6)
//!     .with_quality_target(0.9)
//!     .build()?;
//!
//! // Fast response from strategic agents
//! let quick = legion.query("What is 2+2?", FrequencyBand::Strategic).await?;
//!
//! // High quality from reflective agents (full context)
//! let quality = legion.query("Analyze this codebase", FrequencyBand::Reflective).await?;
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![deny(clippy::unwrap_used)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod agent;
#[cfg(feature = "beleth")]
pub mod beleth_integration;
pub mod consensus;
pub mod context;
pub mod fault;
pub mod field;
pub mod orchestrator;
pub mod quality;
pub mod speculative;
pub mod spectral_merge;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use parking_lot::RwLock;
use thiserror::Error;

pub use agent::{LegionAgent, AgentConfig, AgentState};
pub use consensus::{
    ConsensusStrategy, ConsensusResult, InterferenceConsensus, InterferenceConfig,
    InterferenceResult, AgentContribution as ConsensusContribution,
};
pub use context::{SharedContext, ContextFragment, FragmentId};
pub use fault::{
    FaultConfig, AgentHealth, FailureReason, HealthMonitor, RecoveryManager,
    DegradationManager, HealthCheckResult, HealthStatusChange, RespawnRequest,
    RecoveryOperation, RecoveryType, DegradationEvent, DegradationCause,
    HealthStats, RecoveryStats,
};
pub use field::{LegionField, FieldConfig, LegionPattern, Resonance, EssentialCoefficients, DetailCoefficients};
pub use orchestrator::{Orchestrator, TaskRouting};
pub use quality::{QualityTarget, QualityMetrics, FrequencyBand, QualityCurve, SpectralFilter};
#[cfg(feature = "beleth")]
pub use beleth_integration::{
    AgentContribution, AgentProposal, LegionBackend, LegionBackendError, LegionReactBackend,
    MultiPerspectiveThought, Perspective, PerspectiveConfig, ResolvedAction, ResolutionMethod,
};
pub use speculative::{
    SpeculativeLegion, SpeculativeLegionConfig, DraftSequence, DraftGenerator,
    DraftGeneratorConfig, DraftPool, RankedPath, VerificationResult, VerificationStats,
    SpeculativeStats, QualityCurve as SpeculativeQualityCurve, TokenId,
};
pub use spectral_merge::{
    SpectralDecomposition, LayerDecomposition, LayerType, LayerWeights,
    SpectralBlend, BlendedModel, BlendComponent, DynamicBlendController, BlendStats,
    SpectralMergeError,
};

// ==================== Error Types ====================

/// Errors from Legion operations.
#[derive(Debug, Error)]
pub enum LegionError {
    /// No agents available for the task.
    #[error("No agents available for task")]
    NoAgentsAvailable,

    /// Quality target cannot be achieved.
    #[error("Cannot achieve quality {target}: max available {available}")]
    InsufficientQuality {
        /// Requested quality target.
        target: f32,
        /// Maximum quality available.
        available: f32,
    },

    /// Consensus failed.
    #[error("Consensus failed: {0}")]
    ConsensusFailed(String),

    /// Agent error.
    #[error("Agent error: {0}")]
    Agent(String),

    /// Timeout.
    #[error("Operation timed out after {0:?}")]
    Timeout(std::time::Duration),

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type for Legion operations.
pub type Result<T> = std::result::Result<T, LegionError>;

// ==================== Legion Configuration ====================

/// Configuration for a Legion instance.
#[derive(Debug, Clone)]
pub struct LegionConfig {
    /// Number of agents in the swarm.
    pub agent_count: usize,
    /// Default quality target.
    pub default_quality: f32,
    /// Enable progressive refinement.
    pub progressive_refinement: bool,
    /// Consensus strategy.
    pub consensus_strategy: ConsensusStrategy,
    /// Timeout for operations.
    pub timeout: std::time::Duration,
}

impl Default for LegionConfig {
    fn default() -> Self {
        Self {
            agent_count: 4,
            default_quality: 0.8,
            progressive_refinement: true,
            consensus_strategy: ConsensusStrategy::WeightedMajority,
            timeout: std::time::Duration::from_secs(30),
        }
    }
}

/// Builder for LegionConfig.
#[derive(Debug, Default)]
pub struct LegionConfigBuilder {
    agent_count: Option<usize>,
    default_quality: Option<f32>,
    progressive_refinement: Option<bool>,
    consensus_strategy: Option<ConsensusStrategy>,
    timeout: Option<std::time::Duration>,
}

impl LegionConfigBuilder {
    /// Sets agent count.
    pub fn with_agent_count(mut self, count: usize) -> Self {
        self.agent_count = Some(count);
        self
    }

    /// Sets default quality target.
    pub fn with_quality_target(mut self, quality: f32) -> Self {
        self.default_quality = Some(quality);
        self
    }

    /// Enables/disables progressive refinement.
    pub fn with_progressive_refinement(mut self, enable: bool) -> Self {
        self.progressive_refinement = Some(enable);
        self
    }

    /// Sets consensus strategy.
    pub fn with_consensus_strategy(mut self, strategy: ConsensusStrategy) -> Self {
        self.consensus_strategy = Some(strategy);
        self
    }

    /// Sets operation timeout.
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Builds the configuration.
    pub fn build(self) -> LegionConfig {
        LegionConfig {
            agent_count: self.agent_count.unwrap_or(4),
            default_quality: self.default_quality.unwrap_or(0.8),
            progressive_refinement: self.progressive_refinement.unwrap_or(true),
            consensus_strategy: self.consensus_strategy.unwrap_or(ConsensusStrategy::WeightedMajority),
            timeout: self.timeout.unwrap_or(std::time::Duration::from_secs(30)),
        }
    }
}

// ==================== Legion ====================

/// The main Legion swarm coordinator.
pub struct Legion {
    config: LegionConfig,
    agents: RwLock<Vec<Arc<LegionAgent>>>,
    orchestrator: Arc<Orchestrator>,
    context: Arc<SharedContext>,
}

impl Legion {
    /// Creates a builder for Legion.
    pub fn builder() -> LegionConfigBuilder {
        LegionConfigBuilder::default()
    }

    /// Creates a new Legion with the given configuration.
    pub fn new(config: LegionConfig) -> Result<Self> {
        let context = Arc::new(SharedContext::new(config.agent_count));
        let orchestrator = Arc::new(Orchestrator::new(config.clone()));

        let mut agents = Vec::with_capacity(config.agent_count);
        for i in 0..config.agent_count {
            let agent_config = AgentConfig {
                id: format!("legion-agent-{}", i),
                frequency_band: FrequencyBand::from_index(i, config.agent_count),
                context_fraction: (i + 1) as f32 / config.agent_count as f32,
            };
            agents.push(Arc::new(LegionAgent::new(agent_config)?));
        }

        Ok(Self {
            config,
            agents: RwLock::new(agents),
            orchestrator,
            context,
        })
    }

    /// Returns the configuration.
    pub fn config(&self) -> &LegionConfig {
        &self.config
    }

    /// Returns the number of agents.
    pub fn agent_count(&self) -> usize {
        self.agents.read().len()
    }

    /// Returns the shared context.
    pub fn context(&self) -> &Arc<SharedContext> {
        &self.context
    }

    /// Returns the orchestrator.
    pub fn orchestrator(&self) -> &Arc<Orchestrator> {
        &self.orchestrator
    }
}

// ==================== Tests ====================

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_legion_creation() {
        let config = LegionConfig::default();
        let legion = Legion::new(config);
        assert!(legion.is_ok());

        let legion = legion.expect("Failed to create legion");
        assert_eq!(legion.agent_count(), 4);
    }

    #[test]
    fn test_legion_builder() {
        let config = Legion::builder()
            .with_agent_count(8)
            .with_quality_target(0.95)
            .build();

        assert_eq!(config.agent_count, 8);
        assert!((config.default_quality - 0.95).abs() < 0.001);
    }
}
