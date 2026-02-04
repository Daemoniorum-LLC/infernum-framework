//! Task orchestration for Legion.
//!
//! Routes tasks to appropriate agents based on quality requirements
//! and agent availability.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::quality::{FrequencyBand, QualityTarget};
use crate::LegionConfig;

/// How to route a task to agents.
#[derive(Debug, Clone)]
pub enum TaskRouting {
    /// Route to a single agent at the specified band.
    Single(FrequencyBand),
    /// Route to multiple agents, take first response.
    RaceFirst(Vec<FrequencyBand>),
    /// Route to multiple agents, use consensus.
    Consensus(Vec<FrequencyBand>),
    /// Route to all available agents.
    Broadcast,
    /// Route based on quality target.
    Adaptive(QualityTarget),
}

impl Default for TaskRouting {
    fn default() -> Self {
        TaskRouting::Adaptive(QualityTarget::balanced())
    }
}

/// A task to be executed by agents.
#[derive(Debug, Clone)]
pub struct Task {
    /// Unique task identifier.
    pub id: String,
    /// Task input/prompt.
    pub input: String,
    /// Routing strategy.
    pub routing: TaskRouting,
    /// Quality target.
    pub quality_target: QualityTarget,
    /// Timeout for this task.
    pub timeout: Duration,
}

impl Task {
    /// Creates a new task.
    pub fn new(id: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            input: input.into(),
            routing: TaskRouting::default(),
            quality_target: QualityTarget::balanced(),
            timeout: Duration::from_secs(30),
        }
    }

    /// Sets the routing strategy.
    pub fn with_routing(mut self, routing: TaskRouting) -> Self {
        self.routing = routing;
        self
    }

    /// Sets the quality target.
    pub fn with_quality(mut self, target: QualityTarget) -> Self {
        self.quality_target = target;
        self
    }

    /// Sets the timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Statistics for orchestrator.
#[derive(Debug, Clone, Default)]
pub struct OrchestratorStats {
    /// Total tasks routed.
    pub tasks_routed: u64,
    /// Tasks completed successfully.
    pub tasks_completed: u64,
    /// Tasks that timed out.
    pub tasks_timed_out: u64,
    /// Tasks that failed.
    pub tasks_failed: u64,
    /// Average routing latency in microseconds.
    pub avg_routing_latency_us: u64,
}

/// Orchestrates task routing to agents.
pub struct Orchestrator {
    config: LegionConfig,
    stats: parking_lot::Mutex<OrchestratorStats>,
    task_counter: AtomicU64,
}

impl Orchestrator {
    /// Creates a new orchestrator.
    pub fn new(config: LegionConfig) -> Self {
        Self {
            config,
            stats: parking_lot::Mutex::new(OrchestratorStats::default()),
            task_counter: AtomicU64::new(0),
        }
    }

    /// Returns the configuration.
    pub fn config(&self) -> &LegionConfig {
        &self.config
    }

    /// Returns current statistics.
    pub fn stats(&self) -> OrchestratorStats {
        self.stats.lock().clone()
    }

    /// Generates a unique task ID.
    pub fn generate_task_id(&self) -> String {
        let counter = self.task_counter.fetch_add(1, Ordering::Relaxed);
        format!("task-{}", counter)
    }

    /// Determines which frequency bands to use for a quality target.
    pub fn bands_for_quality(&self, target: &QualityTarget) -> Vec<FrequencyBand> {
        // If preferred band is specified, use it
        if let Some(band) = target.preferred_band {
            return vec![band];
        }

        // Otherwise, select based on target quality
        let bands = if target.target >= 0.9 {
            // High quality: use Reflective and Verification
            vec![FrequencyBand::Reflective, FrequencyBand::Verification]
        } else if target.target >= 0.7 {
            // Balanced: use Operational and Tactical
            vec![FrequencyBand::Operational, FrequencyBand::Tactical]
        } else {
            // Fast: use Strategic
            vec![FrequencyBand::Strategic]
        };

        bands
    }

    /// Selects the best routing for a task.
    pub fn select_routing(&self, task: &Task) -> TaskRouting {
        match &task.routing {
            TaskRouting::Adaptive(target) => {
                let bands = self.bands_for_quality(target);
                if bands.len() == 1 {
                    TaskRouting::Single(bands[0])
                } else {
                    TaskRouting::Consensus(bands)
                }
            }
            other => other.clone(),
        }
    }

    /// Records a successful task completion.
    pub fn record_completion(&self) {
        let mut stats = self.stats.lock();
        stats.tasks_completed += 1;
        stats.tasks_routed += 1;
    }

    /// Records a task failure.
    pub fn record_failure(&self) {
        let mut stats = self.stats.lock();
        stats.tasks_failed += 1;
        stats.tasks_routed += 1;
    }

    /// Records a task timeout.
    pub fn record_timeout(&self) {
        let mut stats = self.stats.lock();
        stats.tasks_timed_out += 1;
        stats.tasks_routed += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_creation() {
        let config = LegionConfig::default();
        let orchestrator = Orchestrator::new(config);

        let stats = orchestrator.stats();
        assert_eq!(stats.tasks_routed, 0);
    }

    #[test]
    fn test_task_id_generation() {
        let config = LegionConfig::default();
        let orchestrator = Orchestrator::new(config);

        let id1 = orchestrator.generate_task_id();
        let id2 = orchestrator.generate_task_id();

        assert_ne!(id1, id2);
        assert!(id1.starts_with("task-"));
    }

    #[test]
    fn test_bands_for_high_quality() {
        let config = LegionConfig::default();
        let orchestrator = Orchestrator::new(config);

        let target = QualityTarget::quality();
        let bands = orchestrator.bands_for_quality(&target);

        assert!(bands.contains(&FrequencyBand::Reflective));
    }

    #[test]
    fn test_bands_for_fast_quality() {
        let config = LegionConfig::default();
        let orchestrator = Orchestrator::new(config);

        let target = QualityTarget::fast();
        let bands = orchestrator.bands_for_quality(&target);

        assert!(bands.contains(&FrequencyBand::Strategic));
    }

    #[test]
    fn test_task_creation() {
        let task = Task::new("test-1", "Hello, world!")
            .with_quality(QualityTarget::balanced())
            .with_timeout(Duration::from_secs(10));

        assert_eq!(task.id, "test-1");
        assert_eq!(task.input, "Hello, world!");
        assert_eq!(task.timeout, Duration::from_secs(10));
    }

    #[test]
    fn test_stats_recording() {
        let config = LegionConfig::default();
        let orchestrator = Orchestrator::new(config);

        orchestrator.record_completion();
        orchestrator.record_completion();
        orchestrator.record_failure();

        let stats = orchestrator.stats();
        assert_eq!(stats.tasks_completed, 2);
        assert_eq!(stats.tasks_failed, 1);
        assert_eq!(stats.tasks_routed, 3);
    }
}
