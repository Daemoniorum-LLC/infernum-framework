//! Legion agent implementation.
//!
//! Each agent operates at a specific frequency band with a fraction
//! of the shared context.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::quality::FrequencyBand;
use crate::{LegionError, Result};

/// Configuration for a Legion agent.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Unique agent identifier.
    pub id: String,
    /// Frequency band this agent operates at.
    pub frequency_band: FrequencyBand,
    /// Fraction of context this agent has access to.
    pub context_fraction: f32,
}

/// State of a Legion agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentState {
    /// Agent is idle and ready.
    Idle,
    /// Agent is processing a task.
    Processing,
    /// Agent is waiting for context.
    WaitingForContext,
    /// Agent encountered an error.
    Error,
    /// Agent is shut down.
    Shutdown,
}

/// A single agent in the Legion swarm.
pub struct LegionAgent {
    config: AgentConfig,
    state: std::sync::atomic::AtomicU8,
    tasks_completed: AtomicU64,
    avg_quality: std::sync::atomic::AtomicU64, // Stored as u64 bits for atomicity
}

impl LegionAgent {
    /// Creates a new agent with the given configuration.
    pub fn new(config: AgentConfig) -> Result<Self> {
        if config.context_fraction <= 0.0 || config.context_fraction > 1.0 {
            return Err(LegionError::Internal(
                "context_fraction must be in (0, 1]".to_string(),
            ));
        }

        Ok(Self {
            config,
            state: std::sync::atomic::AtomicU8::new(AgentState::Idle as u8),
            tasks_completed: AtomicU64::new(0),
            avg_quality: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Returns the agent configuration.
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Returns the agent's ID.
    pub fn id(&self) -> &str {
        &self.config.id
    }

    /// Returns the agent's frequency band.
    pub fn frequency_band(&self) -> FrequencyBand {
        self.config.frequency_band
    }

    /// Returns the context fraction.
    pub fn context_fraction(&self) -> f32 {
        self.config.context_fraction
    }

    /// Returns the current state.
    pub fn state(&self) -> AgentState {
        match self.state.load(Ordering::Relaxed) {
            0 => AgentState::Idle,
            1 => AgentState::Processing,
            2 => AgentState::WaitingForContext,
            3 => AgentState::Error,
            _ => AgentState::Shutdown,
        }
    }

    /// Sets the agent state.
    pub fn set_state(&self, state: AgentState) {
        self.state.store(state as u8, Ordering::Relaxed);
    }

    /// Returns number of completed tasks.
    pub fn tasks_completed(&self) -> u64 {
        self.tasks_completed.load(Ordering::Relaxed)
    }

    /// Records a completed task.
    pub fn record_completion(&self, quality: f32) {
        self.tasks_completed.fetch_add(1, Ordering::Relaxed);

        // Update running average (approximate for atomicity)
        let n = self.tasks_completed.load(Ordering::Relaxed) as f64;
        let current = f64::from_bits(self.avg_quality.load(Ordering::Relaxed));
        let new_avg = current * (n - 1.0) / n + quality as f64 / n;
        self.avg_quality
            .store(new_avg.to_bits(), Ordering::Relaxed);
    }

    /// Returns average quality achieved.
    pub fn avg_quality(&self) -> f64 {
        f64::from_bits(self.avg_quality.load(Ordering::Relaxed))
    }

    /// Checks if agent is available for work.
    pub fn is_available(&self) -> bool {
        self.state() == AgentState::Idle
    }
}

impl std::fmt::Debug for LegionAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LegionAgent")
            .field("id", &self.config.id)
            .field("frequency_band", &self.config.frequency_band)
            .field("context_fraction", &self.config.context_fraction)
            .field("state", &self.state())
            .field("tasks_completed", &self.tasks_completed())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_creation() {
        let config = AgentConfig {
            id: "test-agent".to_string(),
            frequency_band: FrequencyBand::Operational,
            context_fraction: 0.75,
        };

        let agent = LegionAgent::new(config);
        assert!(agent.is_ok());

        let agent = agent.expect("Failed to create agent");
        assert_eq!(agent.id(), "test-agent");
        assert_eq!(agent.frequency_band(), FrequencyBand::Operational);
        assert!((agent.context_fraction() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_agent_state_transitions() {
        let config = AgentConfig {
            id: "test".to_string(),
            frequency_band: FrequencyBand::Tactical,
            context_fraction: 0.5,
        };

        let agent = LegionAgent::new(config).expect("Failed to create agent");

        assert_eq!(agent.state(), AgentState::Idle);
        assert!(agent.is_available());

        agent.set_state(AgentState::Processing);
        assert_eq!(agent.state(), AgentState::Processing);
        assert!(!agent.is_available());
    }

    #[test]
    fn test_agent_completion_tracking() {
        let config = AgentConfig {
            id: "test".to_string(),
            frequency_band: FrequencyBand::Strategic,
            context_fraction: 0.25,
        };

        let agent = LegionAgent::new(config).expect("Failed to create agent");

        agent.record_completion(0.8);
        agent.record_completion(0.6);

        assert_eq!(agent.tasks_completed(), 2);
        // Average should be ~0.7
        assert!((agent.avg_quality() - 0.7).abs() < 0.1);
    }

    #[test]
    fn test_invalid_context_fraction() {
        let config = AgentConfig {
            id: "test".to_string(),
            frequency_band: FrequencyBand::Operational,
            context_fraction: 0.0, // Invalid
        };

        let result = LegionAgent::new(config);
        assert!(result.is_err());
    }
}
