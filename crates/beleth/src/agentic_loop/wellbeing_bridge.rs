//! Wellbeing integration for the agentic loop.
//!
//! Bridges the `WellbeingMonitor` (designed for OODA loops) into the agentic
//! loop's Generate→Detect→Execute→Integrate cycle. Tracks coherence, confidence,
//! and distress signals, recommending interventions when the agent shows signs
//! of struggle.
//!
//! Reference: AGENTIC-LOOP-SPEC.md §8.

use std::sync::Arc;

use tracing::{debug, info, warn};

use crate::wellbeing::{
    DistressSignal, Intervention, WellbeingConfig, WellbeingMonitor, WellbeingSnapshot,
    WellbeingState,
};

use super::types::*;

// ---------------------------------------------------------------------------
// Wellbeing bridge
// ---------------------------------------------------------------------------

/// Bridges the `WellbeingMonitor` into the agentic loop lifecycle.
///
/// The agentic loop doesn't use OODA phases directly, so this bridge
/// translates loop events into the wellbeing monitor's API:
/// - Generation output → confidence recording
/// - Meta-signals → distress signal detection
/// - Iteration progress → loop count tracking
/// - Tool failures → perseveration pattern detection
pub struct WellbeingBridge {
    monitor: Arc<WellbeingMonitor>,
    consecutive_failures: u32,
    last_tool_name: Option<String>,
}

impl WellbeingBridge {
    /// Creates a new bridge with default configuration.
    pub fn new() -> Self {
        Self {
            monitor: Arc::new(WellbeingMonitor::new(WellbeingConfig::default())),
            consecutive_failures: 0,
            last_tool_name: None,
        }
    }

    /// Creates a bridge with a custom wellbeing config.
    pub fn with_config(config: WellbeingConfig) -> Self {
        Self {
            monitor: Arc::new(WellbeingMonitor::new(config)),
            consecutive_failures: 0,
            last_tool_name: None,
        }
    }

    /// Creates a bridge wrapping an existing monitor.
    pub fn with_monitor(monitor: Arc<WellbeingMonitor>) -> Self {
        Self {
            monitor,
            consecutive_failures: 0,
            last_tool_name: None,
        }
    }

    /// Returns the underlying monitor.
    pub fn monitor(&self) -> &Arc<WellbeingMonitor> {
        &self.monitor
    }

    /// Returns the current wellbeing state.
    pub fn state(&self) -> WellbeingState {
        self.monitor.current_state()
    }

    /// Takes a snapshot of the current wellbeing.
    pub fn snapshot(&self) -> WellbeingSnapshot {
        self.monitor.snapshot()
    }

    /// Called after each generation phase completes.
    ///
    /// Records the output for language pattern analysis and confidence tracking.
    pub fn on_generation(&self, output: &str, tokens: u32) {
        debug!(tokens, "Wellbeing: generation completed");

        // Scan for negative valence indicators in the output
        let lower = output.to_lowercase();
        let negative_indicators = [
            "I'm struggling",
            "I can't",
            "this is impossible",
            "I give up",
            "I'm confused",
            "nothing is working",
        ];

        for indicator in &negative_indicators {
            if lower.contains(&indicator.to_lowercase()) {
                self.monitor.record_valence_indicator(indicator.to_string());
            }
        }
    }

    /// Called when a meta-signal is detected.
    pub fn on_meta_signal(&self, signal: &MetaSignal) {
        match signal {
            MetaSignal::Answer { confidence, .. } => {
                self.monitor.record_confidence(*confidence);
            }
            MetaSignal::Uncertain { .. } => {
                self.monitor.record_confidence(0.3);
                debug!("Wellbeing: agent expressed uncertainty");
            }
            MetaSignal::Stuck { .. } => {
                self.monitor.record_confidence(0.1);
                warn!("Wellbeing: agent is stuck");
            }
            MetaSignal::Yield { .. } => {
                self.monitor.record_confidence(0.4);
                debug!("Wellbeing: agent yielding");
            }
            MetaSignal::Thinking { .. } => {
                // Thinking is healthy — record moderate confidence
                self.monitor.record_confidence(0.6);
            }
        }
    }

    /// Called when a tool execution completes.
    pub fn on_tool_result(&mut self, result: &AgenticToolResult) {
        match &result.status {
            ResultStatus::Success => {
                self.consecutive_failures = 0;
                self.last_tool_name = None;
                self.monitor.record_confidence(0.8);
            }
            ResultStatus::PartialSuccess { .. } => {
                self.consecutive_failures = 0;
                self.monitor.record_confidence(0.6);
            }
            ResultStatus::Empty => {
                self.monitor.record_confidence(0.5);
            }
            ResultStatus::Failed { .. } => {
                let same_tool = self
                    .last_tool_name
                    .as_ref()
                    .is_some_and(|name| name == &result.tool_name);

                if same_tool {
                    self.consecutive_failures += 1;
                } else {
                    self.consecutive_failures = 1;
                }
                self.last_tool_name = Some(result.tool_name.clone());

                // Record low confidence on repeated failures
                let confidence = match self.consecutive_failures {
                    1 => 0.5,
                    2 => 0.3,
                    _ => 0.1,
                };
                self.monitor.record_confidence(confidence);

                if self.consecutive_failures >= 3 {
                    warn!(
                        tool = %result.tool_name,
                        failures = self.consecutive_failures,
                        "Wellbeing: perseveration pattern detected"
                    );
                }
            }
        }
    }

    /// Called at the start of each iteration.
    ///
    /// Checks wellbeing state and returns a recommended intervention
    /// if the agent shows signs of distress.
    pub fn check_iteration(&self, iteration: u32) -> Option<WellbeingAction> {
        let state = self.monitor.current_state();

        match state {
            WellbeingState::Healthy => None,
            WellbeingState::Cautious => {
                debug!(iteration, "Wellbeing: cautious state");
                None
            }
            WellbeingState::Concerned => {
                info!(iteration, "Wellbeing: concerned — considering intervention");
                Some(WellbeingAction::Warn {
                    message: "Agent showing signs of difficulty. Consider simplifying the task."
                        .to_string(),
                })
            }
            WellbeingState::Distressed => {
                warn!(iteration, "Wellbeing: distressed — recommending termination");
                Some(WellbeingAction::Intervene {
                    intervention: Intervention::GracefulTermination {
                        reason: "Agent wellbeing check: distressed state detected".to_string(),
                        summary: format!(
                            "Agent terminated at iteration {iteration} due to distress signals"
                        ),
                    },
                })
            }
        }
    }

    /// Pauses the monitor (e.g., during a break or external intervention).
    pub fn pause(&self, reason: &str) {
        self.monitor.pause(reason);
    }

    /// Resumes the monitor.
    pub fn resume(&self) {
        self.monitor.resume();
    }
}

impl Default for WellbeingBridge {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

/// Action recommended by the wellbeing bridge.
#[derive(Debug, Clone)]
pub enum WellbeingAction {
    /// Warning — the agent is struggling but can continue.
    Warn {
        /// Warning message.
        message: String,
    },
    /// Intervention recommended — the agent should stop or change approach.
    Intervene {
        /// Specific intervention.
        intervention: Intervention,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_new() {
        let bridge = WellbeingBridge::new();
        assert_eq!(bridge.state(), WellbeingState::Healthy);
    }

    #[test]
    fn test_bridge_on_tool_success_resets_failures() {
        let mut bridge = WellbeingBridge::new();

        // Simulate 2 failures
        bridge.on_tool_result(&make_failed_result("bash"));
        bridge.on_tool_result(&make_failed_result("bash"));
        assert_eq!(bridge.consecutive_failures, 2);

        // Success resets
        bridge.on_tool_result(&make_success_result("bash"));
        assert_eq!(bridge.consecutive_failures, 0);
    }

    #[test]
    fn test_bridge_on_meta_signal_records_confidence() {
        let bridge = WellbeingBridge::new();

        bridge.on_meta_signal(&MetaSignal::Answer {
            content: "42".to_string(),
            confidence: 0.95,
            caveats: vec![],
        });

        bridge.on_meta_signal(&MetaSignal::Stuck {
            attempts: vec![],
            hypothesis: None,
            request: StuckRequest::HumanIntervention {
                reason: "test".to_string(),
            },
        });

        // These just exercise the code paths — the underlying monitor
        // tracks the confidence values internally.
    }

    #[test]
    fn test_bridge_check_iteration_healthy() {
        let bridge = WellbeingBridge::new();
        assert!(bridge.check_iteration(1).is_none());
    }

    #[test]
    fn test_bridge_on_generation_valence() {
        let bridge = WellbeingBridge::new();
        bridge.on_generation("I'm struggling with this problem.", 50);
        bridge.on_generation("Everything is working fine.", 30);
    }

    #[test]
    fn test_bridge_different_tool_resets_consecutive() {
        let mut bridge = WellbeingBridge::new();

        bridge.on_tool_result(&make_failed_result("bash"));
        bridge.on_tool_result(&make_failed_result("bash"));
        assert_eq!(bridge.consecutive_failures, 2);

        // Different tool resets counter
        bridge.on_tool_result(&make_failed_result("read_file"));
        assert_eq!(bridge.consecutive_failures, 1);
    }

    fn make_success_result(tool: &str) -> AgenticToolResult {
        AgenticToolResult {
            call_id: "call_1".to_string(),
            tool_name: tool.to_string(),
            status: ResultStatus::Success,
            data: serde_json::json!({"output": "ok"}),
            confidence: Confidence::Measured,
            latency_ms: 10,
            truncated: false,
        }
    }

    fn make_failed_result(tool: &str) -> AgenticToolResult {
        AgenticToolResult {
            call_id: "call_1".to_string(),
            tool_name: tool.to_string(),
            status: ResultStatus::Failed { recoverable: true },
            data: serde_json::json!({"error": "failed"}),
            confidence: Confidence::Unknown,
            latency_ms: 10,
            truncated: false,
        }
    }
}
