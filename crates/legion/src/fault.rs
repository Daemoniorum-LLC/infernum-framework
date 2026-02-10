//! # Fault Tolerance: Graceful Degradation in the Collective
//!
//! The Legion degrades gracefully when agents fail. Like a hologram,
//! partial information still reconstructs the whole - just at lower quality.
//!
//! ## The Holographic Principle
//!
//! Traditional systems: One failure → cascade failure
//! Legion systems: Agent failure → quality reduction → system continues
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    FAULT TOLERANCE                                   │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                      │
//! │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                  │
//! │   │Agent 1  │ │Agent 2  │ │Agent 3✗ │ │Agent 4  │                  │
//! │   │   OK    │ │   OK    │ │  FAIL   │ │   OK    │                  │
//! │   └────┬────┘ └────┬────┘ └─────────┘ └────┬────┘                  │
//! │        │           │                       │                        │
//! │        └───────────┼───────────────────────┘                        │
//! │                    ▼                                                 │
//! │   ┌─────────────────────────────────────────────────────────────┐   │
//! │   │        Quality: 75% (3/4 agents contributing)                │   │
//! │   │        Consensus still valid, just lower confidence          │   │
//! │   └─────────────────────────────────────────────────────────────┘   │
//! │                                                                      │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Insight
//!
//! Quality degrades linearly with agent loss (thanks to holographic encoding).
//! The system never "breaks" - it just gets less confident.
//!
//! *"We are diminished, but not defeated."*

use crate::{FrequencyBand, QualityCurve};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for fault tolerance.
#[derive(Debug, Clone)]
pub struct FaultConfig {
    /// Timeout before agent is considered failed.
    pub agent_timeout: Duration,

    /// Maximum retry attempts.
    pub max_retries: u32,

    /// Minimum agents for operation.
    pub min_agents: usize,

    /// Quality curve for degradation prediction.
    pub quality_curve: QualityCurve,

    /// Enable automatic agent respawn.
    pub auto_respawn: bool,

    /// Respawn delay.
    pub respawn_delay: Duration,

    /// Heartbeat interval for health checks.
    pub heartbeat_interval: Duration,

    /// Consecutive failures before quarantine.
    pub quarantine_threshold: u32,
}

impl Default for FaultConfig {
    fn default() -> Self {
        Self {
            agent_timeout: Duration::from_secs(5),
            max_retries: 3,
            min_agents: 2,
            quality_curve: QualityCurve::SPECTRAL,
            auto_respawn: true,
            respawn_delay: Duration::from_millis(100),
            heartbeat_interval: Duration::from_secs(1),
            quarantine_threshold: 3,
        }
    }
}

// ============================================================================
// Agent Health
// ============================================================================

/// Health status of an agent.
#[derive(Debug, Clone)]
pub enum AgentHealth {
    /// Agent is healthy and responding.
    Healthy,

    /// Agent is responding slowly.
    Degraded {
        /// Recent latency.
        latency: Duration,
        /// When degradation started.
        since: Instant,
    },

    /// Agent is not responding.
    Unresponsive {
        /// When last heartbeat was received.
        last_seen: Instant,
        /// Retry attempts made.
        retries: u32,
    },

    /// Agent has failed completely.
    Failed {
        /// Failure reason.
        reason: FailureReason,
        /// When failure occurred.
        failed_at: Instant,
    },

    /// Agent is quarantined (repeated failures).
    Quarantined {
        /// How many failures.
        failure_count: u32,
        /// When quarantine started.
        since: Instant,
    },
}

impl AgentHealth {
    /// Is agent available for work?
    pub fn is_available(&self) -> bool {
        matches!(self, AgentHealth::Healthy | AgentHealth::Degraded { .. })
    }

    /// Is agent completely failed?
    pub fn is_failed(&self) -> bool {
        matches!(
            self,
            AgentHealth::Failed { .. } | AgentHealth::Quarantined { .. }
        )
    }

    /// Duration in current state.
    pub fn duration_in_state(&self) -> Duration {
        match self {
            AgentHealth::Degraded { since, .. } => since.elapsed(),
            AgentHealth::Unresponsive { last_seen, .. } => last_seen.elapsed(),
            AgentHealth::Failed { failed_at, .. } => failed_at.elapsed(),
            AgentHealth::Quarantined { since, .. } => since.elapsed(),
            AgentHealth::Healthy => Duration::ZERO,
        }
    }
}

impl PartialEq for AgentHealth {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

/// Reason for agent failure.
#[derive(Debug, Clone, PartialEq)]
pub enum FailureReason {
    /// Timeout waiting for response.
    Timeout,
    /// Agent crashed.
    Crash {
        /// Error message.
        error: String,
    },
    /// Resource exhaustion.
    ResourceExhaustion {
        /// Which resource.
        resource: String,
    },
    /// Communication failure.
    CommunicationError,
    /// Manual shutdown.
    ManualShutdown,
    /// Unknown failure.
    Unknown,
}

// ============================================================================
// Health Monitor
// ============================================================================

/// Monitors health of all agents in the Legion.
#[derive(Debug)]
pub struct HealthMonitor {
    /// Configuration.
    config: FaultConfig,

    /// Agent health status.
    agent_health: HashMap<u64, AgentHealth>,

    /// Last heartbeat times.
    last_heartbeat: HashMap<u64, Instant>,

    /// Failure counts per agent.
    failure_counts: HashMap<u64, u32>,

    /// Quarantined agents.
    quarantined: HashSet<u64>,

    /// Statistics.
    stats: HealthStats,
}

impl HealthMonitor {
    /// Create new monitor.
    pub fn new(config: FaultConfig) -> Self {
        Self {
            config,
            agent_health: HashMap::new(),
            last_heartbeat: HashMap::new(),
            failure_counts: HashMap::new(),
            quarantined: HashSet::new(),
            stats: HealthStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_new() -> Self {
        Self::new(FaultConfig::default())
    }

    // ========================================================================
    // Registration
    // ========================================================================

    /// Register an agent for monitoring.
    pub fn register(&mut self, agent_id: u64) {
        self.agent_health.insert(agent_id, AgentHealth::Healthy);
        self.last_heartbeat.insert(agent_id, Instant::now());
        self.failure_counts.insert(agent_id, 0);
    }

    /// Unregister an agent.
    pub fn unregister(&mut self, agent_id: u64) {
        self.agent_health.remove(&agent_id);
        self.last_heartbeat.remove(&agent_id);
        self.failure_counts.remove(&agent_id);
        self.quarantined.remove(&agent_id);
    }

    // ========================================================================
    // Heartbeat
    // ========================================================================

    /// Record a heartbeat from an agent.
    pub fn heartbeat(&mut self, agent_id: u64) {
        self.last_heartbeat.insert(agent_id, Instant::now());

        // Recover from degraded/unresponsive if applicable
        if let Some(health) = self.agent_health.get_mut(&agent_id) {
            if matches!(
                health,
                AgentHealth::Degraded { .. } | AgentHealth::Unresponsive { .. }
            ) {
                *health = AgentHealth::Healthy;
                self.stats.recoveries += 1;
            }
        }
    }

    /// Record a heartbeat with latency.
    pub fn heartbeat_with_latency(&mut self, agent_id: u64, latency: Duration) {
        self.last_heartbeat.insert(agent_id, Instant::now());

        // Check if latency indicates degradation
        let threshold = self.config.heartbeat_interval / 2;
        if latency > threshold {
            if let Some(health) = self.agent_health.get_mut(&agent_id) {
                if matches!(health, AgentHealth::Healthy) {
                    *health = AgentHealth::Degraded {
                        latency,
                        since: Instant::now(),
                    };
                }
            }
        } else {
            // Good latency - mark healthy
            if let Some(health) = self.agent_health.get_mut(&agent_id) {
                if matches!(health, AgentHealth::Degraded { .. }) {
                    *health = AgentHealth::Healthy;
                    self.stats.recoveries += 1;
                }
            }
        }
    }

    // ========================================================================
    // Health Checks
    // ========================================================================

    /// Run health check on all agents.
    pub fn check_all(&mut self) -> HealthCheckResult {
        let now = Instant::now();
        let mut status_changes = Vec::new();

        let agent_ids: Vec<u64> = self.last_heartbeat.keys().copied().collect();

        for agent_id in agent_ids {
            // Skip quarantined agents
            if self.quarantined.contains(&agent_id) {
                continue;
            }

            let elapsed = self
                .last_heartbeat
                .get(&agent_id)
                .map(|t| now.duration_since(*t))
                .unwrap_or(Duration::MAX);

            let current_health = self.agent_health.get(&agent_id).cloned();

            // Determine new health status
            let new_health = if elapsed > self.config.agent_timeout {
                // Agent is unresponsive
                let retries = self.failure_counts.get(&agent_id).copied().unwrap_or(0);

                if retries >= self.config.max_retries {
                    AgentHealth::Failed {
                        reason: FailureReason::Timeout,
                        failed_at: now,
                    }
                } else {
                    AgentHealth::Unresponsive {
                        last_seen: self.last_heartbeat.get(&agent_id).copied().unwrap_or(now),
                        retries,
                    }
                }
            } else if elapsed > self.config.heartbeat_interval * 2 {
                // Agent is degraded
                AgentHealth::Degraded {
                    latency: elapsed,
                    since: now,
                }
            } else {
                // Agent is healthy
                AgentHealth::Healthy
            };

            // Track status change
            if current_health.as_ref() != Some(&new_health) {
                status_changes.push(HealthStatusChange {
                    agent_id,
                    from: current_health.clone().unwrap_or(AgentHealth::Healthy),
                    to: new_health.clone(),
                    timestamp: now,
                });

                // Update failure count
                if matches!(
                    new_health,
                    AgentHealth::Unresponsive { .. } | AgentHealth::Failed { .. }
                ) {
                    *self.failure_counts.entry(agent_id).or_insert(0) += 1;
                    self.stats.failures += 1;
                }

                // Check for quarantine
                if let Some(&count) = self.failure_counts.get(&agent_id) {
                    if count >= self.config.quarantine_threshold {
                        self.quarantine(agent_id);
                    }
                }
            }

            self.agent_health.insert(agent_id, new_health);
        }

        let healthy_count = self.healthy_count();
        let total_count = self.agent_health.len();

        HealthCheckResult {
            healthy_count,
            total_count,
            degraded_count: self.degraded_count(),
            failed_count: self.failed_count(),
            status_changes,
            quality: self.calculate_quality(healthy_count, total_count),
        }
    }

    /// Check health of specific agent.
    pub fn check_agent(&self, agent_id: u64) -> Option<&AgentHealth> {
        self.agent_health.get(&agent_id)
    }

    // ========================================================================
    // Failure Handling
    // ========================================================================

    /// Mark agent as failed.
    pub fn mark_failed(&mut self, agent_id: u64, reason: FailureReason) {
        self.agent_health.insert(
            agent_id,
            AgentHealth::Failed {
                reason,
                failed_at: Instant::now(),
            },
        );

        *self.failure_counts.entry(agent_id).or_insert(0) += 1;
        self.stats.failures += 1;

        // Check quarantine threshold
        if let Some(&count) = self.failure_counts.get(&agent_id) {
            if count >= self.config.quarantine_threshold {
                self.quarantine(agent_id);
            }
        }
    }

    /// Quarantine an agent.
    pub fn quarantine(&mut self, agent_id: u64) {
        self.quarantined.insert(agent_id);
        self.agent_health.insert(
            agent_id,
            AgentHealth::Quarantined {
                failure_count: self.failure_counts.get(&agent_id).copied().unwrap_or(0),
                since: Instant::now(),
            },
        );
        self.stats.quarantined += 1;
    }

    /// Release agent from quarantine.
    pub fn release_quarantine(&mut self, agent_id: u64) {
        self.quarantined.remove(&agent_id);
        self.agent_health.insert(agent_id, AgentHealth::Healthy);
        self.failure_counts.insert(agent_id, 0);
    }

    // ========================================================================
    // Quality Calculation
    // ========================================================================

    /// Calculate quality based on healthy agents.
    fn calculate_quality(&self, healthy: usize, total: usize) -> f32 {
        if total == 0 {
            return 0.0;
        }

        // Use quality curve to predict output quality
        self.config
            .quality_curve
            .predict(healthy as u16, total as u16)
    }

    /// Current quality level.
    pub fn quality(&self) -> f32 {
        let healthy = self.healthy_count();
        let total = self.agent_health.len();
        self.calculate_quality(healthy, total)
    }

    // ========================================================================
    // Counting
    // ========================================================================

    /// Count healthy agents.
    pub fn healthy_count(&self) -> usize {
        self.agent_health
            .values()
            .filter(|h| matches!(h, AgentHealth::Healthy))
            .count()
    }

    /// Count degraded agents.
    pub fn degraded_count(&self) -> usize {
        self.agent_health
            .values()
            .filter(|h| matches!(h, AgentHealth::Degraded { .. }))
            .count()
    }

    /// Count failed agents.
    pub fn failed_count(&self) -> usize {
        self.agent_health.values().filter(|h| h.is_failed()).count()
    }

    /// Count available agents (healthy + degraded).
    pub fn available_count(&self) -> usize {
        self.agent_health
            .values()
            .filter(|h| h.is_available())
            .count()
    }

    /// Is minimum agents available?
    pub fn has_minimum(&self) -> bool {
        self.available_count() >= self.config.min_agents
    }

    // ========================================================================
    // Introspection
    // ========================================================================

    /// Get all agent health statuses.
    pub fn all_health(&self) -> &HashMap<u64, AgentHealth> {
        &self.agent_health
    }

    /// Get statistics.
    pub fn stats(&self) -> &HealthStats {
        &self.stats
    }

    /// Get quarantined agents.
    pub fn quarantined_agents(&self) -> &HashSet<u64> {
        &self.quarantined
    }

    /// Get configuration.
    pub fn config(&self) -> &FaultConfig {
        &self.config
    }
}

// ============================================================================
// Recovery Manager
// ============================================================================

/// Manages recovery and respawning of failed agents.
#[derive(Debug)]
pub struct RecoveryManager {
    /// Configuration.
    config: FaultConfig,

    /// Pending respawns.
    pending_respawns: Vec<RespawnRequest>,

    /// Active recovery operations.
    active_recoveries: HashMap<u64, RecoveryOperation>,

    /// Statistics.
    stats: RecoveryStats,
}

impl RecoveryManager {
    /// Create new recovery manager.
    pub fn new(config: FaultConfig) -> Self {
        Self {
            config,
            pending_respawns: Vec::new(),
            active_recoveries: HashMap::new(),
            stats: RecoveryStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_new() -> Self {
        Self::new(FaultConfig::default())
    }

    /// Request agent respawn.
    pub fn request_respawn(&mut self, agent_id: u64, band: FrequencyBand) {
        if !self.config.auto_respawn {
            return;
        }

        self.pending_respawns.push(RespawnRequest {
            original_id: agent_id,
            band,
            requested_at: Instant::now(),
            priority: band.priority(),
        });

        self.stats.respawns_requested += 1;
    }

    /// Get pending respawns (sorted by priority).
    pub fn pending_respawns(&mut self) -> Vec<RespawnRequest> {
        self.pending_respawns
            .sort_by(|a, b| b.priority.cmp(&a.priority));
        std::mem::take(&mut self.pending_respawns)
    }

    /// Mark respawn as complete.
    pub fn respawn_complete(&mut self, original_id: u64, _new_id: u64) {
        self.active_recoveries.remove(&original_id);
        self.stats.respawns_completed += 1;
    }

    /// Start recovery operation.
    pub fn start_recovery(&mut self, agent_id: u64, operation: RecoveryOperation) {
        self.active_recoveries.insert(agent_id, operation);
    }

    /// Get active recoveries.
    pub fn active_recoveries(&self) -> &HashMap<u64, RecoveryOperation> {
        &self.active_recoveries
    }

    /// Get statistics.
    pub fn stats(&self) -> &RecoveryStats {
        &self.stats
    }

    /// Get configuration.
    pub fn config(&self) -> &FaultConfig {
        &self.config
    }
}

/// Request to respawn an agent.
#[derive(Debug, Clone)]
pub struct RespawnRequest {
    /// Original failed agent ID.
    pub original_id: u64,

    /// Frequency band to respawn.
    pub band: FrequencyBand,

    /// When requested.
    pub requested_at: Instant,

    /// Priority (higher = more urgent).
    pub priority: u8,
}

/// Active recovery operation.
#[derive(Debug, Clone)]
pub struct RecoveryOperation {
    /// Agent being recovered.
    pub agent_id: u64,

    /// Recovery type.
    pub recovery_type: RecoveryType,

    /// Started at.
    pub started_at: Instant,

    /// Current progress (0-1).
    pub progress: f32,
}

/// Type of recovery.
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryType {
    /// Full respawn.
    Respawn,
    /// State restoration from checkpoint.
    StateRestore,
    /// Partial recovery (continue with degraded state).
    Partial,
    /// Redistribute load to other agents.
    LoadRedistribute,
}

// ============================================================================
// Graceful Degradation
// ============================================================================

/// Manager for graceful degradation.
#[derive(Debug)]
pub struct DegradationManager {
    /// Configuration.
    config: FaultConfig,

    /// Current degradation level.
    degradation_level: f32,

    /// Degradation history.
    history: Vec<DegradationEvent>,
}

impl DegradationManager {
    /// Create new manager.
    pub fn new(config: FaultConfig) -> Self {
        Self {
            config,
            degradation_level: 0.0,
            history: Vec::new(),
        }
    }

    /// Create with default configuration.
    pub fn default_new() -> Self {
        Self::new(FaultConfig::default())
    }

    /// Update degradation based on health.
    pub fn update_from_health(&mut self, health_result: &HealthCheckResult) {
        let new_level = 1.0 - health_result.quality;

        if (new_level - self.degradation_level).abs() > 0.05 {
            self.history.push(DegradationEvent {
                timestamp: Instant::now(),
                previous_level: self.degradation_level,
                new_level,
                cause: if new_level > self.degradation_level {
                    DegradationCause::AgentFailure
                } else {
                    DegradationCause::AgentRecovery
                },
            });
        }

        self.degradation_level = new_level;
    }

    /// Current degradation level (0 = healthy, 1 = fully degraded).
    pub fn level(&self) -> f32 {
        self.degradation_level
    }

    /// Is system in degraded state?
    pub fn is_degraded(&self) -> bool {
        self.degradation_level > 0.1
    }

    /// Is system critically degraded?
    pub fn is_critical(&self) -> bool {
        self.degradation_level > 0.5
    }

    /// Adjusted quality (what we can provide).
    pub fn adjusted_quality(&self) -> f32 {
        1.0 - self.degradation_level
    }

    /// Get degradation history.
    pub fn history(&self) -> &[DegradationEvent] {
        &self.history
    }

    /// Get configuration.
    pub fn config(&self) -> &FaultConfig {
        &self.config
    }

    /// Clear history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}

/// Degradation event.
#[derive(Debug, Clone)]
pub struct DegradationEvent {
    /// When it happened.
    pub timestamp: Instant,

    /// Previous level.
    pub previous_level: f32,

    /// New level.
    pub new_level: f32,

    /// What caused it.
    pub cause: DegradationCause,
}

/// Cause of degradation.
#[derive(Debug, Clone, PartialEq)]
pub enum DegradationCause {
    /// Agent failed.
    AgentFailure,
    /// Agent recovered.
    AgentRecovery,
    /// Load increased.
    LoadIncrease,
    /// Resource exhaustion.
    ResourceExhaustion,
}

// ============================================================================
// Result Types
// ============================================================================

/// Result of health check.
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    /// Number of healthy agents.
    pub healthy_count: usize,

    /// Total agents.
    pub total_count: usize,

    /// Degraded agents.
    pub degraded_count: usize,

    /// Failed agents.
    pub failed_count: usize,

    /// Status changes since last check.
    pub status_changes: Vec<HealthStatusChange>,

    /// Overall quality (0-1).
    pub quality: f32,
}

impl HealthCheckResult {
    /// Is system healthy?
    pub fn is_healthy(&self) -> bool {
        self.failed_count == 0 && self.quality > 0.9
    }

    /// Availability percentage.
    pub fn availability(&self) -> f32 {
        if self.total_count > 0 {
            (self.healthy_count + self.degraded_count) as f32 / self.total_count as f32
        } else {
            0.0
        }
    }
}

/// Change in health status.
#[derive(Debug, Clone)]
pub struct HealthStatusChange {
    /// Agent ID.
    pub agent_id: u64,

    /// Previous status.
    pub from: AgentHealth,

    /// New status.
    pub to: AgentHealth,

    /// When it happened.
    pub timestamp: Instant,
}

// ============================================================================
// Statistics
// ============================================================================

/// Health monitor statistics.
#[derive(Debug, Clone, Default)]
pub struct HealthStats {
    /// Total failures detected.
    pub failures: usize,

    /// Successful recoveries.
    pub recoveries: usize,

    /// Agents quarantined.
    pub quarantined: usize,
}

/// Recovery statistics.
#[derive(Debug, Clone, Default)]
pub struct RecoveryStats {
    /// Respawns requested.
    pub respawns_requested: usize,

    /// Respawns completed.
    pub respawns_completed: usize,

    /// State restorations.
    pub state_restorations: usize,
}

impl RecoveryStats {
    /// Respawn success rate.
    pub fn success_rate(&self) -> f32 {
        if self.respawns_requested > 0 {
            self.respawns_completed as f32 / self.respawns_requested as f32
        } else {
            1.0
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_monitor_creation() {
        let monitor = HealthMonitor::default_new();
        assert_eq!(monitor.healthy_count(), 0);
    }

    #[test]
    fn test_agent_registration() {
        let mut monitor = HealthMonitor::default_new();

        monitor.register(1);
        monitor.register(2);
        monitor.register(3);

        assert_eq!(monitor.healthy_count(), 3);
        assert!(monitor.has_minimum());
    }

    #[test]
    fn test_heartbeat() {
        let mut monitor = HealthMonitor::default_new();
        monitor.register(1);

        monitor.heartbeat(1);
        assert!(matches!(monitor.check_agent(1), Some(AgentHealth::Healthy)));
    }

    #[test]
    fn test_failure_detection() {
        let mut monitor = HealthMonitor::default_new();
        monitor.register(1);

        monitor.mark_failed(1, FailureReason::Timeout);

        assert!(matches!(
            monitor.check_agent(1),
            Some(AgentHealth::Failed { .. })
        ));
        assert_eq!(monitor.failed_count(), 1);
    }

    #[test]
    fn test_quarantine() {
        let mut config = FaultConfig::default();
        config.quarantine_threshold = 2;
        let mut monitor = HealthMonitor::new(config);

        monitor.register(1);

        // Fail twice - both failures should count toward quarantine
        monitor.mark_failed(1, FailureReason::Timeout);
        // Manually reset to healthy without resetting failure count
        monitor.agent_health.insert(1, AgentHealth::Healthy);
        monitor.mark_failed(1, FailureReason::Timeout);

        assert!(monitor.quarantined_agents().contains(&1));
    }

    #[test]
    fn test_quality_calculation() {
        let mut monitor = HealthMonitor::default_new();

        for i in 1..=8 {
            monitor.register(i);
        }

        // All healthy
        assert!(monitor.quality() > 0.95);

        // Fail half
        for i in 1..=4 {
            monitor.mark_failed(i, FailureReason::Timeout);
        }

        // Quality should decrease
        assert!(monitor.quality() < 0.9);
    }

    #[test]
    fn test_recovery_request() {
        let mut recovery = RecoveryManager::default_new();

        recovery.request_respawn(1, FrequencyBand::Strategic);
        recovery.request_respawn(2, FrequencyBand::Tactical);

        let pending = recovery.pending_respawns();
        assert_eq!(pending.len(), 2);
    }

    #[test]
    fn test_degradation_manager() {
        let mut degradation = DegradationManager::default_new();

        let health = HealthCheckResult {
            healthy_count: 6,
            total_count: 8,
            degraded_count: 0,
            failed_count: 2,
            status_changes: vec![],
            quality: 0.75,
        };

        degradation.update_from_health(&health);

        assert!(degradation.is_degraded());
        assert!(!degradation.is_critical());
        assert!((degradation.adjusted_quality() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_health_status_available() {
        assert!(AgentHealth::Healthy.is_available());
        assert!(AgentHealth::Degraded {
            latency: Duration::from_millis(100),
            since: Instant::now()
        }
        .is_available());
        assert!(!AgentHealth::Failed {
            reason: FailureReason::Timeout,
            failed_at: Instant::now()
        }
        .is_available());
    }

    #[test]
    fn test_recovery_operation() {
        let mut recovery = RecoveryManager::default_new();

        let op = RecoveryOperation {
            agent_id: 1,
            recovery_type: RecoveryType::Respawn,
            started_at: Instant::now(),
            progress: 0.0,
        };

        recovery.start_recovery(1, op);
        assert!(recovery.active_recoveries().contains_key(&1));

        recovery.respawn_complete(1, 2);
        assert!(!recovery.active_recoveries().contains_key(&1));
        assert_eq!(recovery.stats().respawns_completed, 1);
    }

    #[test]
    fn test_degradation_history() {
        let mut degradation = DegradationManager::default_new();

        // Significant quality drop
        let health1 = HealthCheckResult {
            healthy_count: 4,
            total_count: 8,
            degraded_count: 0,
            failed_count: 4,
            status_changes: vec![],
            quality: 0.5,
        };
        degradation.update_from_health(&health1);

        // Recovery
        let health2 = HealthCheckResult {
            healthy_count: 8,
            total_count: 8,
            degraded_count: 0,
            failed_count: 0,
            status_changes: vec![],
            quality: 1.0,
        };
        degradation.update_from_health(&health2);

        assert_eq!(degradation.history().len(), 2);
        assert_eq!(
            degradation.history()[0].cause,
            DegradationCause::AgentFailure
        );
        assert_eq!(
            degradation.history()[1].cause,
            DegradationCause::AgentRecovery
        );
    }
}
