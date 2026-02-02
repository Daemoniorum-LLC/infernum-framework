//! Wellbeing-based inference intervention controller.
//!
//! This module provides a mechanism to pause or restrict inference based on
//! agent wellbeing state. When an agent enters a distressed state, inference
//! requests can be automatically paused to give the agent "breathing room"
//! and prevent potentially erratic behavior from consuming resources.
//!
//! # Philosophy
//!
//! The intervention system follows these principles:
//! 1. **Observe, Don't Force**: By default, we monitor and recommend rather than force
//! 2. **Graceful Degradation**: Paused requests can queue rather than fail
//! 3. **Rapid Recovery**: Resume is immediate when wellbeing improves
//! 4. **Transparency**: All interventions are logged and reported

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Notify};
use tracing::{debug, info, warn};

/// Wellbeing state levels (mirrors beleth::WellbeingState).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum WellbeingState {
    /// Agent is operating normally.
    #[default]
    Healthy,
    /// Agent shows signs of uncertainty.
    Cautious,
    /// Agent is struggling.
    Concerned,
    /// Agent is in distress - may pause inference.
    Distressed,
}

impl std::fmt::Display for WellbeingState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::Cautious => write!(f, "cautious"),
            Self::Concerned => write!(f, "concerned"),
            Self::Distressed => write!(f, "distressed"),
        }
    }
}

/// Configuration for wellbeing intervention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionConfig {
    /// Enable intervention system.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Auto-pause on distressed state.
    #[serde(default)]
    pub pause_on_distressed: bool,

    /// Auto-pause on concerned state (more aggressive).
    #[serde(default)]
    pub pause_on_concerned: bool,

    /// Maximum pause duration before auto-resume (seconds).
    #[serde(default = "default_max_pause")]
    pub max_pause_secs: u64,

    /// Pause timeout for queued requests (seconds).
    #[serde(default = "default_wait_timeout")]
    pub wait_timeout_secs: u64,
}

fn default_true() -> bool { true }
fn default_max_pause() -> u64 { 300 } // 5 minutes
fn default_wait_timeout() -> u64 { 60 } // 1 minute

impl Default for InterventionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            pause_on_distressed: false, // Default: observe only
            pause_on_concerned: false,
            max_pause_secs: 300,
            wait_timeout_secs: 60,
        }
    }
}

/// Metrics for the intervention controller.
#[derive(Debug, Clone, Default)]
pub struct InterventionMetrics {
    /// Total interventions triggered.
    pub interventions_total: u64,
    /// Total pauses.
    pub pauses_total: u64,
    /// Total resumes.
    pub resumes_total: u64,
    /// Requests blocked while paused.
    pub requests_blocked: u64,
    /// Requests that waited and succeeded.
    pub requests_waited: u64,
    /// Requests that timed out waiting.
    pub requests_timed_out: u64,
    /// Current pause duration (seconds).
    pub current_pause_duration_secs: u64,
}

/// Wellbeing intervention controller.
///
/// Coordinates wellbeing state and inference pausing across the server.
pub struct InterventionController {
    config: InterventionConfig,
    /// Current wellbeing state.
    state: RwLock<WellbeingState>,
    /// Whether inference is currently paused.
    paused: AtomicBool,
    /// When the current pause started.
    pause_start: RwLock<Option<Instant>>,
    /// Reason for current pause.
    pause_reason: RwLock<Option<String>>,
    /// Notify waiters when unpaused.
    unpause_notify: Notify,
    /// Metrics counters.
    interventions_total: AtomicU64,
    pauses_total: AtomicU64,
    resumes_total: AtomicU64,
    requests_blocked: AtomicU64,
    requests_waited: AtomicU64,
    requests_timed_out: AtomicU64,
}

impl InterventionController {
    /// Creates a new intervention controller.
    #[must_use]
    pub fn new(config: InterventionConfig) -> Self {
        info!(
            "Wellbeing intervention controller initialized (enabled={}, pause_on_distressed={})",
            config.enabled, config.pause_on_distressed
        );
        Self {
            config,
            state: RwLock::new(WellbeingState::Healthy),
            paused: AtomicBool::new(false),
            pause_start: RwLock::new(None),
            pause_reason: RwLock::new(None),
            unpause_notify: Notify::new(),
            interventions_total: AtomicU64::new(0),
            pauses_total: AtomicU64::new(0),
            resumes_total: AtomicU64::new(0),
            requests_blocked: AtomicU64::new(0),
            requests_waited: AtomicU64::new(0),
            requests_timed_out: AtomicU64::new(0),
        }
    }

    /// Updates the current wellbeing state.
    ///
    /// If auto-intervention is enabled and the state warrants it,
    /// this will trigger a pause.
    pub async fn update_state(&self, state: WellbeingState, reason: Option<String>) {
        if !self.config.enabled {
            return;
        }

        let old_state = {
            let mut s = self.state.write().await;
            let old = *s;
            *s = state;
            old
        };

        if old_state != state {
            info!("Wellbeing state changed: {} -> {}", old_state, state);
            self.interventions_total.fetch_add(1, Ordering::Relaxed);
        }

        // Check if we should auto-pause
        let should_pause = match state {
            WellbeingState::Distressed => self.config.pause_on_distressed,
            WellbeingState::Concerned => self.config.pause_on_concerned,
            _ => false,
        };

        if should_pause && !self.is_paused() {
            let reason = reason.unwrap_or_else(|| format!("Agent entered {} state", state));
            self.pause(&reason).await;
        } else if !should_pause && self.is_paused() {
            // Auto-resume if state improved
            self.resume().await;
        }
    }

    /// Returns the current wellbeing state.
    pub async fn current_state(&self) -> WellbeingState {
        *self.state.read().await
    }

    /// Returns whether inference is currently paused.
    #[must_use]
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }

    /// Manually pauses inference.
    pub async fn pause(&self, reason: &str) {
        if self.paused.swap(true, Ordering::SeqCst) {
            // Already paused
            return;
        }

        *self.pause_start.write().await = Some(Instant::now());
        *self.pause_reason.write().await = Some(reason.to_string());
        self.pauses_total.fetch_add(1, Ordering::Relaxed);

        warn!("Inference paused due to wellbeing intervention: {}", reason);
    }

    /// Resumes inference.
    pub async fn resume(&self) {
        if !self.paused.swap(false, Ordering::SeqCst) {
            // Not paused
            return;
        }

        let duration = {
            let start = self.pause_start.read().await;
            start.map(|s| s.elapsed())
        };

        *self.pause_start.write().await = None;
        *self.pause_reason.write().await = None;
        self.resumes_total.fetch_add(1, Ordering::Relaxed);

        // Notify all waiters
        self.unpause_notify.notify_waiters();

        if let Some(d) = duration {
            info!("Inference resumed after {:.1}s pause", d.as_secs_f32());
        } else {
            info!("Inference resumed");
        }
    }

    /// Check if request can proceed, blocking if paused.
    ///
    /// Returns `Ok(())` if request can proceed, `Err` if timed out.
    pub async fn gate_request(&self) -> Result<(), InterventionError> {
        if !self.config.enabled {
            return Ok(());
        }

        if !self.is_paused() {
            return Ok(());
        }

        // Check if max pause duration exceeded
        self.check_auto_resume().await;

        if !self.is_paused() {
            return Ok(());
        }

        // We're paused - either wait or reject
        self.requests_blocked.fetch_add(1, Ordering::Relaxed);
        debug!("Request blocked - waiting for wellbeing intervention to clear");

        let timeout = Duration::from_secs(self.config.wait_timeout_secs);

        tokio::select! {
            _ = self.unpause_notify.notified() => {
                self.requests_waited.fetch_add(1, Ordering::Relaxed);
                debug!("Request proceeding after wait");
                Ok(())
            }
            _ = tokio::time::sleep(timeout) => {
                self.requests_timed_out.fetch_add(1, Ordering::Relaxed);
                let reason = self.pause_reason.read().await.clone()
                    .unwrap_or_else(|| "Unknown".to_string());
                Err(InterventionError::Paused { reason, timeout_secs: self.config.wait_timeout_secs })
            }
        }
    }

    /// Check if we should auto-resume due to timeout.
    async fn check_auto_resume(&self) {
        let should_resume = {
            let start = self.pause_start.read().await;
            start.map_or(false, |s| {
                s.elapsed().as_secs() >= self.config.max_pause_secs
            })
        };

        if should_resume {
            warn!(
                "Auto-resuming inference after max pause duration ({}s)",
                self.config.max_pause_secs
            );
            self.resume().await;
        }
    }

    /// Returns current metrics.
    pub async fn metrics(&self) -> InterventionMetrics {
        let current_pause_duration_secs = {
            let start = self.pause_start.read().await;
            start.map_or(0, |s| s.elapsed().as_secs())
        };

        InterventionMetrics {
            interventions_total: self.interventions_total.load(Ordering::Relaxed),
            pauses_total: self.pauses_total.load(Ordering::Relaxed),
            resumes_total: self.resumes_total.load(Ordering::Relaxed),
            requests_blocked: self.requests_blocked.load(Ordering::Relaxed),
            requests_waited: self.requests_waited.load(Ordering::Relaxed),
            requests_timed_out: self.requests_timed_out.load(Ordering::Relaxed),
            current_pause_duration_secs,
        }
    }

    /// Returns the reason for current pause, if any.
    pub async fn pause_reason(&self) -> Option<String> {
        self.pause_reason.read().await.clone()
    }
}

/// Error returned when intervention blocks a request.
#[derive(Debug, Clone, thiserror::Error)]
pub enum InterventionError {
    /// Inference is paused due to wellbeing intervention.
    #[error("inference paused due to wellbeing intervention: {reason} (timeout after {timeout_secs}s)")]
    Paused {
        /// Reason for the pause.
        reason: String,
        /// Timeout duration.
        timeout_secs: u64,
    },
}

/// Shared intervention controller wrapped in Arc.
pub type SharedInterventionController = Arc<InterventionController>;

/// Creates a new shared intervention controller.
#[must_use]
pub fn create_intervention_controller(config: InterventionConfig) -> SharedInterventionController {
    Arc::new(InterventionController::new(config))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_default_not_paused() {
        let controller = InterventionController::new(InterventionConfig::default());
        assert!(!controller.is_paused());
        assert!(controller.gate_request().await.is_ok());
    }

    #[tokio::test]
    async fn test_manual_pause_resume() {
        let controller = InterventionController::new(InterventionConfig::default());

        controller.pause("test pause").await;
        assert!(controller.is_paused());

        controller.resume().await;
        assert!(!controller.is_paused());
    }

    #[tokio::test]
    async fn test_auto_pause_on_distressed() {
        let config = InterventionConfig {
            enabled: true,
            pause_on_distressed: true,
            ..Default::default()
        };
        let controller = InterventionController::new(config);

        // Update to distressed should pause
        controller.update_state(WellbeingState::Distressed, None).await;
        assert!(controller.is_paused());

        // Update to healthy should resume
        controller.update_state(WellbeingState::Healthy, None).await;
        assert!(!controller.is_paused());
    }

    #[tokio::test]
    async fn test_metrics() {
        let config = InterventionConfig {
            enabled: true,
            pause_on_distressed: true,
            ..Default::default()
        };
        let controller = InterventionController::new(config);

        controller.update_state(WellbeingState::Distressed, None).await;
        controller.update_state(WellbeingState::Healthy, None).await;

        let metrics = controller.metrics().await;
        assert_eq!(metrics.pauses_total, 1);
        assert_eq!(metrics.resumes_total, 1);
        assert!(metrics.interventions_total >= 2);
    }
}
