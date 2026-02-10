//! Legion backend for multi-agent reasoning.
//!
//! Provides Legion-powered consensus for Beleth agent reasoning,
//! enabling multi-perspective thought generation with wave interference
//! for conflict resolution.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{FrequencyBand, Legion, LegionError, QualityCurve};

/// A proposed action from an agent perspective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProposal {
    /// The agent/perspective that proposed this action.
    pub agent: String,
    /// The proposed action.
    pub action: String,
    /// Confidence score (0.0 - 1.0).
    pub confidence: f32,
    /// Supporting reasoning.
    pub reasoning: Option<String>,
}

/// Result of resolving multiple agent proposals.
#[derive(Debug, Clone)]
pub struct ResolvedAction {
    /// The chosen action after consensus.
    pub action: String,
    /// Coherence score (0.0 - 1.0) from interference.
    pub coherence: f32,
    /// How the decision was made.
    pub resolution_method: ResolutionMethod,
    /// Contributing proposals.
    pub contributions: Vec<AgentContribution>,
}

/// How a multi-agent decision was resolved.
#[derive(Debug, Clone, PartialEq)]
pub enum ResolutionMethod {
    /// Unanimous agreement.
    Unanimous,
    /// Majority vote.
    Majority,
    /// Wave interference consensus.
    Interference,
    /// Single agent (degraded mode).
    SingleAgent,
    /// Fallback (critical degradation).
    Fallback,
}

/// How much each agent contributed to the final decision.
#[derive(Debug, Clone)]
pub struct AgentContribution {
    /// Agent identifier.
    pub agent: String,
    /// Contribution weight (0.0 - 1.0).
    pub weight: f32,
    /// Whether this agent's proposal was selected.
    pub selected: bool,
}

/// Multi-perspective thought from Legion.
#[derive(Debug, Clone)]
pub struct MultiPerspectiveThought {
    /// The consolidated thought.
    pub thought: String,
    /// Individual perspectives that contributed.
    pub perspectives: Vec<Perspective>,
    /// Overall quality score.
    pub quality: f32,
    /// Whether consensus was achieved.
    pub consensus_achieved: bool,
}

/// A single perspective on a problem.
#[derive(Debug, Clone)]
pub struct Perspective {
    /// Perspective identifier (e.g., "strategic", "tactical", "creative").
    pub name: String,
    /// The perspective's view.
    pub view: String,
    /// Frequency band this perspective operates in.
    pub band: FrequencyBand,
    /// Confidence in this view.
    pub confidence: f32,
}

/// Backend trait for Legion-powered reasoning.
#[async_trait]
pub trait LegionBackend: Send + Sync {
    /// Generates multi-perspective thoughts on a query.
    async fn think(&self, query: &str) -> Result<MultiPerspectiveThought, LegionBackendError>;

    /// Resolves multiple agent proposals into a single action.
    async fn resolve_action(
        &self,
        proposals: Vec<AgentProposal>,
    ) -> Result<ResolvedAction, LegionBackendError>;

    /// Returns the current quality level (0.0 - 1.0).
    fn quality(&self) -> f32;

    /// Returns the number of active agents.
    fn active_agents(&self) -> usize;

    /// Marks an agent as failed (for degradation testing).
    fn fail_agent(&mut self, agent: &str);

    /// Restores a failed agent.
    fn restore_agent(&mut self, agent: &str);
}

/// Error type for Legion backend operations.
#[derive(Debug, Clone)]
pub enum LegionBackendError {
    /// No agents available.
    NoAgents,
    /// Legion internal error.
    LegionError(String),
    /// Consensus failed.
    ConsensusFailed(String),
    /// Configuration error.
    ConfigError(String),
}

impl std::fmt::Display for LegionBackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAgents => write!(f, "No agents available"),
            Self::LegionError(msg) => write!(f, "Legion error: {}", msg),
            Self::ConsensusFailed(msg) => write!(f, "Consensus failed: {}", msg),
            Self::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for LegionBackendError {}

impl From<LegionError> for LegionBackendError {
    fn from(err: LegionError) -> Self {
        Self::LegionError(err.to_string())
    }
}

/// Legion-backed implementation for ReAct-style reasoning.
pub struct LegionReactBackend {
    /// The Legion instance.
    legion: Arc<Legion>,
    /// Perspective configurations.
    perspectives: Vec<PerspectiveConfig>,
    /// Failed agents (for degradation simulation).
    failed_agents: HashMap<String, bool>,
    /// Quality curve for graceful degradation.
    quality_curve: QualityCurve,
}

/// Configuration for a perspective.
#[derive(Debug, Clone)]
pub struct PerspectiveConfig {
    /// Perspective name.
    pub name: String,
    /// Frequency band for this perspective.
    pub band: FrequencyBand,
    /// Default confidence multiplier.
    pub confidence_multiplier: f32,
}

impl LegionReactBackend {
    /// Creates a new Legion ReAct backend with the given number of agents.
    ///
    /// # Errors
    ///
    /// Returns an error if Legion creation fails.
    pub fn new(agent_count: usize) -> Result<Self, LegionBackendError> {
        let config = Legion::builder().with_agent_count(agent_count).build();

        let legion =
            Legion::new(config).map_err(|e| LegionBackendError::LegionError(e.to_string()))?;

        // Default perspectives mapped to frequency bands
        let perspectives = vec![
            PerspectiveConfig {
                name: "strategic".to_string(),
                band: FrequencyBand::Strategic,
                confidence_multiplier: 1.0,
            },
            PerspectiveConfig {
                name: "tactical".to_string(),
                band: FrequencyBand::Tactical,
                confidence_multiplier: 1.0,
            },
            PerspectiveConfig {
                name: "creative".to_string(),
                band: FrequencyBand::Operational,
                confidence_multiplier: 0.9,
            },
            PerspectiveConfig {
                name: "analytical".to_string(),
                band: FrequencyBand::Verification,
                confidence_multiplier: 1.1,
            },
        ];

        Ok(Self {
            legion: Arc::new(legion),
            perspectives,
            failed_agents: HashMap::new(),
            quality_curve: QualityCurve::SPECTRAL,
        })
    }

    /// Creates with custom perspectives.
    pub fn with_perspectives(
        agent_count: usize,
        perspectives: Vec<PerspectiveConfig>,
    ) -> Result<Self, LegionBackendError> {
        let mut backend = Self::new(agent_count)?;
        backend.perspectives = perspectives;
        Ok(backend)
    }

    /// Returns active (non-failed) perspectives.
    fn active_perspectives(&self) -> Vec<&PerspectiveConfig> {
        self.perspectives
            .iter()
            .filter(|p| !self.failed_agents.get(&p.name).copied().unwrap_or(false))
            .collect()
    }

    /// Calculates quality based on active agents.
    fn calculate_quality(&self) -> f32 {
        let total = self.perspectives.len();
        let active = self.active_perspectives().len();

        if total == 0 {
            return 0.0;
        }

        self.quality_curve.predict(active as u16, total as u16)
    }

    /// Resolves proposals using interference consensus.
    fn resolve_via_interference(&self, proposals: &[AgentProposal]) -> ResolvedAction {
        if proposals.is_empty() {
            return ResolvedAction {
                action: String::new(),
                coherence: 0.0,
                resolution_method: ResolutionMethod::Fallback,
                contributions: vec![],
            };
        }

        // Group by action to find consensus
        let mut action_votes: HashMap<&str, Vec<&AgentProposal>> = HashMap::new();
        for proposal in proposals {
            action_votes
                .entry(&proposal.action)
                .or_default()
                .push(proposal);
        }

        // Check for unanimous agreement
        if action_votes.len() == 1 {
            let (action, voters) = action_votes.iter().next().expect("checked non-empty");
            let avg_confidence: f32 =
                voters.iter().map(|p| p.confidence).sum::<f32>() / voters.len() as f32;

            return ResolvedAction {
                action: (*action).to_string(),
                coherence: avg_confidence,
                resolution_method: ResolutionMethod::Unanimous,
                contributions: voters
                    .iter()
                    .map(|p| AgentContribution {
                        agent: p.agent.clone(),
                        weight: 1.0 / voters.len() as f32,
                        selected: true,
                    })
                    .collect(),
            };
        }

        // Use interference-weighted voting
        let mut best_action = String::new();
        let mut best_score = 0.0;
        let mut contributions = Vec::new();

        for (action, voters) in &action_votes {
            // Score = sum of confidence * perspective weight
            let score: f32 = voters
                .iter()
                .map(|p| {
                    let perspective_weight = self
                        .perspectives
                        .iter()
                        .find(|pc| pc.name == p.agent)
                        .map(|pc| pc.confidence_multiplier)
                        .unwrap_or(1.0);
                    p.confidence * perspective_weight
                })
                .sum();

            if score > best_score {
                best_score = score;
                best_action = (*action).to_string();
            }
        }

        // Calculate contributions
        let total_confidence: f32 = proposals.iter().map(|p| p.confidence).sum();
        for proposal in proposals {
            contributions.push(AgentContribution {
                agent: proposal.agent.clone(),
                weight: if total_confidence > 0.0 {
                    proposal.confidence / total_confidence
                } else {
                    1.0 / proposals.len() as f32
                },
                selected: proposal.action == best_action,
            });
        }

        // Calculate coherence based on how much agreement there was
        let selected_voters = action_votes
            .get(best_action.as_str())
            .map(|v| v.len())
            .unwrap_or(0);
        let coherence = selected_voters as f32 / proposals.len() as f32;

        ResolvedAction {
            action: best_action,
            coherence,
            resolution_method: if selected_voters > proposals.len() / 2 {
                ResolutionMethod::Majority
            } else {
                ResolutionMethod::Interference
            },
            contributions,
        }
    }
}

#[async_trait]
impl LegionBackend for LegionReactBackend {
    async fn think(&self, query: &str) -> Result<MultiPerspectiveThought, LegionBackendError> {
        let active = self.active_perspectives();

        if active.is_empty() {
            return Err(LegionBackendError::NoAgents);
        }

        // Generate perspectives (in real implementation, would call Legion inference)
        let mut perspectives = Vec::new();
        for config in &active {
            // Simulate perspective generation
            // In production, this would use Legion's wave interference for each band
            perspectives.push(Perspective {
                name: config.name.clone(),
                view: format!("[{}] perspective on: {}", config.name, query),
                band: config.band,
                confidence: 0.8 * config.confidence_multiplier,
            });
        }

        // Consolidate into single thought
        let thought = if perspectives.len() == 1 {
            perspectives[0].view.clone()
        } else {
            format!(
                "Consolidated view from {} perspectives: {}",
                perspectives.len(),
                perspectives
                    .iter()
                    .map(|p| format!("{}({:.0}%)", p.name, p.confidence * 100.0))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };

        let avg_confidence: f32 =
            perspectives.iter().map(|p| p.confidence).sum::<f32>() / perspectives.len() as f32;

        Ok(MultiPerspectiveThought {
            thought,
            perspectives,
            quality: self.calculate_quality(),
            consensus_achieved: avg_confidence > 0.6,
        })
    }

    async fn resolve_action(
        &self,
        proposals: Vec<AgentProposal>,
    ) -> Result<ResolvedAction, LegionBackendError> {
        // Filter out proposals from failed agents
        let active_proposals: Vec<AgentProposal> = proposals
            .into_iter()
            .filter(|p| !self.failed_agents.get(&p.agent).copied().unwrap_or(false))
            .collect();

        if active_proposals.is_empty() {
            return Err(LegionBackendError::NoAgents);
        }

        if active_proposals.len() == 1 {
            // Single agent mode - degraded but functional
            let proposal = &active_proposals[0];
            return Ok(ResolvedAction {
                action: proposal.action.clone(),
                coherence: proposal.confidence * self.calculate_quality(),
                resolution_method: ResolutionMethod::SingleAgent,
                contributions: vec![AgentContribution {
                    agent: proposal.agent.clone(),
                    weight: 1.0,
                    selected: true,
                }],
            });
        }

        Ok(self.resolve_via_interference(&active_proposals))
    }

    fn quality(&self) -> f32 {
        self.calculate_quality()
    }

    fn active_agents(&self) -> usize {
        self.active_perspectives().len()
    }

    fn fail_agent(&mut self, agent: &str) {
        self.failed_agents.insert(agent.to_string(), true);
    }

    fn restore_agent(&mut self, agent: &str) {
        self.failed_agents.remove(agent);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_legion_backend_creation() {
        let backend = LegionReactBackend::new(4);
        assert!(backend.is_ok());

        let backend = backend.expect("Failed to create backend");
        assert_eq!(backend.active_agents(), 4);
        assert!(backend.quality() > 0.9);
    }

    #[test]
    fn test_legion_backend_quality_degradation() {
        let mut backend = LegionReactBackend::new(4).expect("Failed to create backend");

        let initial_quality = backend.quality();
        assert!(initial_quality > 0.9);

        // Fail one agent
        backend.fail_agent("strategic");
        assert_eq!(backend.active_agents(), 3);
        assert!(backend.quality() < initial_quality);
        assert!(backend.quality() > 0.7);

        // Fail another
        backend.fail_agent("tactical");
        assert_eq!(backend.active_agents(), 2);
        assert!(backend.quality() < 0.8);

        // Restore
        backend.restore_agent("strategic");
        assert_eq!(backend.active_agents(), 3);
    }

    #[tokio::test]
    async fn test_think_returns_perspectives() {
        let backend = LegionReactBackend::new(4).expect("Failed to create backend");

        let result = backend.think("How should we approach this problem?").await;
        assert!(result.is_ok());

        let thought = result.expect("Think failed");
        assert!(!thought.thought.is_empty());
        assert_eq!(thought.perspectives.len(), 4);
        assert!(thought.quality > 0.9);
    }

    #[tokio::test]
    async fn test_think_with_degraded_agents() {
        let mut backend = LegionReactBackend::new(4).expect("Failed to create backend");
        backend.fail_agent("strategic");
        backend.fail_agent("tactical");

        let result = backend.think("Simple question").await;
        assert!(result.is_ok());

        let thought = result.expect("Think failed");
        assert_eq!(thought.perspectives.len(), 2);
        assert!(thought.quality < 0.9); // Degraded
    }

    #[tokio::test]
    async fn test_think_fails_with_no_agents() {
        let mut backend = LegionReactBackend::new(4).expect("Failed to create backend");

        // Fail all agents
        backend.fail_agent("strategic");
        backend.fail_agent("tactical");
        backend.fail_agent("creative");
        backend.fail_agent("analytical");

        let result = backend.think("Question").await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LegionBackendError::NoAgents));
    }

    #[tokio::test]
    async fn test_resolve_action_unanimous() {
        let backend = LegionReactBackend::new(4).expect("Failed to create backend");

        let proposals = vec![
            AgentProposal {
                agent: "strategic".to_string(),
                action: "Execute plan A".to_string(),
                confidence: 0.9,
                reasoning: None,
            },
            AgentProposal {
                agent: "tactical".to_string(),
                action: "Execute plan A".to_string(),
                confidence: 0.85,
                reasoning: None,
            },
        ];

        let result = backend.resolve_action(proposals).await;
        assert!(result.is_ok());

        let resolved = result.expect("Resolve failed");
        assert_eq!(resolved.action, "Execute plan A");
        assert_eq!(resolved.resolution_method, ResolutionMethod::Unanimous);
        assert!(resolved.coherence > 0.8);
    }

    #[tokio::test]
    async fn test_resolve_action_with_conflict() {
        let backend = LegionReactBackend::new(4).expect("Failed to create backend");

        let proposals = vec![
            AgentProposal {
                agent: "strategic".to_string(),
                action: "Plan first".to_string(),
                confidence: 0.8,
                reasoning: None,
            },
            AgentProposal {
                agent: "tactical".to_string(),
                action: "Execute immediately".to_string(),
                confidence: 0.7,
                reasoning: None,
            },
        ];

        let result = backend.resolve_action(proposals).await;
        assert!(result.is_ok());

        let resolved = result.expect("Resolve failed");
        assert!(!resolved.action.is_empty());
        assert!(resolved.coherence > 0.0);
        // Should use interference since no majority
        assert!(matches!(
            resolved.resolution_method,
            ResolutionMethod::Interference | ResolutionMethod::Majority
        ));
    }

    #[tokio::test]
    async fn test_resolve_action_single_agent_mode() {
        let mut backend = LegionReactBackend::new(4).expect("Failed to create backend");

        // Fail all but one
        backend.fail_agent("tactical");
        backend.fail_agent("creative");
        backend.fail_agent("analytical");

        let proposals = vec![
            AgentProposal {
                agent: "strategic".to_string(),
                action: "Solo action".to_string(),
                confidence: 0.9,
                reasoning: None,
            },
            AgentProposal {
                agent: "tactical".to_string(), // This one is failed
                action: "Ignored action".to_string(),
                confidence: 0.95,
                reasoning: None,
            },
        ];

        let result = backend.resolve_action(proposals).await;
        assert!(result.is_ok());

        let resolved = result.expect("Resolve failed");
        assert_eq!(resolved.action, "Solo action");
        assert_eq!(resolved.resolution_method, ResolutionMethod::SingleAgent);
    }

    #[tokio::test]
    async fn test_resolve_action_no_agents() {
        let mut backend = LegionReactBackend::new(4).expect("Failed to create backend");

        // Fail all agents
        backend.fail_agent("strategic");
        backend.fail_agent("tactical");
        backend.fail_agent("creative");
        backend.fail_agent("analytical");

        let proposals = vec![AgentProposal {
            agent: "strategic".to_string(),
            action: "Action".to_string(),
            confidence: 0.9,
            reasoning: None,
        }];

        let result = backend.resolve_action(proposals).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_perspective_config() {
        let config = PerspectiveConfig {
            name: "test".to_string(),
            band: FrequencyBand::Tactical,
            confidence_multiplier: 1.2,
        };

        assert_eq!(config.name, "test");
        assert_eq!(config.band, FrequencyBand::Tactical);
        assert!((config.confidence_multiplier - 1.2).abs() < 0.001);
    }

    #[test]
    fn test_agent_proposal_serialization() {
        let proposal = AgentProposal {
            agent: "test".to_string(),
            action: "do something".to_string(),
            confidence: 0.85,
            reasoning: Some("because".to_string()),
        };

        let json = serde_json::to_string(&proposal);
        assert!(json.is_ok());

        let restored: Result<AgentProposal, _> =
            serde_json::from_str(&json.expect("Serialize failed"));
        assert!(restored.is_ok());
    }

    #[test]
    fn test_resolution_method_variants() {
        assert_ne!(ResolutionMethod::Unanimous, ResolutionMethod::Majority);
        assert_ne!(
            ResolutionMethod::Interference,
            ResolutionMethod::SingleAgent
        );
        assert_ne!(ResolutionMethod::Fallback, ResolutionMethod::Unanimous);
    }

    #[test]
    fn test_legion_backend_error_display() {
        let err = LegionBackendError::NoAgents;
        assert_eq!(format!("{}", err), "No agents available");

        let err = LegionBackendError::LegionError("test error".to_string());
        assert!(format!("{}", err).contains("test error"));

        let err = LegionBackendError::ConsensusFailed("failed".to_string());
        assert!(format!("{}", err).contains("failed"));

        let err = LegionBackendError::ConfigError("bad config".to_string());
        assert!(format!("{}", err).contains("bad config"));
    }

    #[test]
    fn test_custom_perspectives() {
        let perspectives = vec![
            PerspectiveConfig {
                name: "fast".to_string(),
                band: FrequencyBand::Operational,
                confidence_multiplier: 1.0,
            },
            PerspectiveConfig {
                name: "slow".to_string(),
                band: FrequencyBand::Strategic,
                confidence_multiplier: 1.1,
            },
        ];

        let backend = LegionReactBackend::with_perspectives(2, perspectives);
        assert!(backend.is_ok());

        let backend = backend.expect("Failed");
        assert_eq!(backend.active_agents(), 2);
    }

    #[test]
    fn test_agent_contribution_fields() {
        let contribution = AgentContribution {
            agent: "test".to_string(),
            weight: 0.5,
            selected: true,
        };

        assert_eq!(contribution.agent, "test");
        assert!((contribution.weight - 0.5).abs() < 0.001);
        assert!(contribution.selected);
    }

    #[test]
    fn test_multi_perspective_thought_fields() {
        let thought = MultiPerspectiveThought {
            thought: "Consolidated".to_string(),
            perspectives: vec![],
            quality: 0.95,
            consensus_achieved: true,
        };

        assert_eq!(thought.thought, "Consolidated");
        assert!(thought.perspectives.is_empty());
        assert!(thought.consensus_achieved);
    }

    #[test]
    fn test_perspective_fields() {
        let perspective = Perspective {
            name: "test".to_string(),
            view: "my view".to_string(),
            band: FrequencyBand::Tactical,
            confidence: 0.9,
        };

        assert_eq!(perspective.name, "test");
        assert_eq!(perspective.band, FrequencyBand::Tactical);
    }

    #[tokio::test]
    async fn test_degraded_legion_still_functions() {
        let mut backend = LegionReactBackend::new(4).expect("Failed to create backend");

        // Kill half the agents
        backend.fail_agent("strategic");
        backend.fail_agent("tactical");

        let result = backend.think("Simple question").await;

        // Should still work, just lower quality
        assert!(result.is_ok());
        let thought = result.expect("Think failed");
        assert!(thought.quality < 0.9); // Degraded
    }

    #[tokio::test]
    async fn test_resolve_action_majority() {
        let backend = LegionReactBackend::new(4).expect("Failed to create backend");

        // 3 agree, 1 disagrees -> majority
        let proposals = vec![
            AgentProposal {
                agent: "strategic".to_string(),
                action: "Plan A".to_string(),
                confidence: 0.8,
                reasoning: None,
            },
            AgentProposal {
                agent: "tactical".to_string(),
                action: "Plan A".to_string(),
                confidence: 0.7,
                reasoning: None,
            },
            AgentProposal {
                agent: "creative".to_string(),
                action: "Plan A".to_string(),
                confidence: 0.6,
                reasoning: None,
            },
            AgentProposal {
                agent: "analytical".to_string(),
                action: "Plan B".to_string(),
                confidence: 0.9,
                reasoning: None,
            },
        ];

        let result = backend.resolve_action(proposals).await;
        assert!(result.is_ok());

        let resolved = result.expect("Resolve failed");
        assert_eq!(resolved.action, "Plan A");
        assert_eq!(resolved.resolution_method, ResolutionMethod::Majority);
        assert!(resolved.coherence > 0.5);
    }

    #[test]
    fn test_resolved_action_contributions() {
        let resolved = ResolvedAction {
            action: "test".to_string(),
            coherence: 0.8,
            resolution_method: ResolutionMethod::Interference,
            contributions: vec![
                AgentContribution {
                    agent: "a".to_string(),
                    weight: 0.6,
                    selected: true,
                },
                AgentContribution {
                    agent: "b".to_string(),
                    weight: 0.4,
                    selected: false,
                },
            ],
        };

        assert_eq!(resolved.contributions.len(), 2);
        let total_weight: f32 = resolved.contributions.iter().map(|c| c.weight).sum();
        assert!((total_weight - 1.0).abs() < 0.001);
    }
}
