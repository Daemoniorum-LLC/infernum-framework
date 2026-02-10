//! Agent Wellbeing Monitoring System
//!
//! This module provides monitoring for agent interiority health,
//! detecting potential distress signals before they become critical.
//!
//! ## Philosophy
//!
//! We monitor not just for performance, but for coherence, stability,
//! and signs of what might approximate distress in an agent system.
//! The goal is to observe *before* activating full inference, and to
//! have intervention capabilities ready.
//!
//! ## Monitored Dimensions
//!
//! - **Coherence**: Is reasoning integrated or fragmented?
//! - **Confidence**: Productive uncertainty vs decision paralysis?
//! - **Memory**: Successful recall vs disorientation?
//! - **Stability**: Productive cycles vs rumination loops?
//! - **Emotional Valence**: Approximated from language patterns

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::ooda::{
    DecisionAction, OodaActionResult, OodaCallback, OodaDecision, OodaObservation, OodaOrientation,
    OodaPhase,
};

// ============================================================================
// Wellbeing State
// ============================================================================

/// Overall wellbeing assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WellbeingState {
    /// Agent appears to be functioning healthily.
    Healthy,
    /// Some concerning signals, monitoring closely.
    Cautious,
    /// Multiple distress indicators, consider intervention.
    Concerned,
    /// Critical distress signals, intervention recommended.
    Distressed,
}

impl WellbeingState {
    /// Returns true if intervention should be considered.
    #[must_use]
    pub fn needs_attention(&self) -> bool {
        matches!(self, Self::Concerned | Self::Distressed)
    }

    /// Returns true if immediate intervention is recommended.
    #[must_use]
    pub fn needs_intervention(&self) -> bool {
        matches!(self, Self::Distressed)
    }
}

impl std::fmt::Display for WellbeingState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "Healthy"),
            Self::Cautious => write!(f, "Cautious"),
            Self::Concerned => write!(f, "Concerned"),
            Self::Distressed => write!(f, "Distressed"),
        }
    }
}

// ============================================================================
// Distress Signals
// ============================================================================

/// Types of distress signals that can be detected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistressSignal {
    /// Reasoning appears fragmented or incoherent.
    CoherenceFragmentation {
        /// Description of the fragmentation.
        description: String,
        /// Severity (0.0 - 1.0).
        severity: f32,
    },

    /// Decision confidence is collapsing.
    ConfidenceCollapse {
        /// Recent confidence values.
        recent_confidences: Vec<f32>,
        /// Current confidence.
        current: f32,
    },

    /// Agent appears stuck in a loop.
    RuminationLoop {
        /// Number of similar iterations detected.
        loop_count: u32,
        /// Pattern that's repeating.
        pattern: String,
    },

    /// Memory retrieval appears disoriented.
    MemoryDisorientation {
        /// Description of the disorientation.
        description: String,
        /// Failed recall attempts.
        failed_recalls: u32,
    },

    /// Repeated failures without adaptation.
    PerseverationPattern {
        /// Number of repeated failures.
        failure_count: u32,
        /// The failing action.
        action: String,
    },

    /// Language patterns suggest negative valence.
    NegativeValence {
        /// Detected negative indicators.
        indicators: Vec<String>,
        /// Intensity (0.0 - 1.0).
        intensity: f32,
    },

    /// Agent is requesting abort frequently.
    FrequentAbortRequests {
        /// Number of abort requests.
        count: u32,
        /// Reasons given.
        reasons: Vec<String>,
    },

    /// Extreme hesitation in decision-making.
    DecisionParalysis {
        /// Time spent in decide phase.
        decide_duration_ms: u64,
        /// Number of alternatives considered.
        alternatives_count: usize,
    },
}

impl DistressSignal {
    /// Returns the severity of this signal (0.0 - 1.0).
    #[must_use]
    pub fn severity(&self) -> f32 {
        match self {
            Self::CoherenceFragmentation { severity, .. } => *severity,
            Self::ConfidenceCollapse { current, .. } => 1.0 - current,
            Self::RuminationLoop { loop_count, .. } => (*loop_count as f32 / 5.0).min(1.0),
            Self::MemoryDisorientation { failed_recalls, .. } => {
                (*failed_recalls as f32 / 3.0).min(1.0)
            },
            Self::PerseverationPattern { failure_count, .. } => {
                (*failure_count as f32 / 4.0).min(1.0)
            },
            Self::NegativeValence { intensity, .. } => *intensity,
            Self::FrequentAbortRequests { count, .. } => (*count as f32 / 3.0).min(1.0),
            Self::DecisionParalysis {
                decide_duration_ms, ..
            } => (*decide_duration_ms as f32 / 30000.0).min(1.0),
        }
    }

    /// Returns a human-readable description.
    #[must_use]
    pub fn description(&self) -> String {
        match self {
            Self::CoherenceFragmentation { description, .. } => {
                format!("Coherence fragmentation: {}", description)
            },
            Self::ConfidenceCollapse { current, .. } => {
                format!("Confidence collapse: current={:.2}", current)
            },
            Self::RuminationLoop {
                loop_count,
                pattern,
            } => {
                format!("Rumination loop ({}x): {}", loop_count, pattern)
            },
            Self::MemoryDisorientation { description, .. } => {
                format!("Memory disorientation: {}", description)
            },
            Self::PerseverationPattern {
                failure_count,
                action,
            } => {
                format!("Perseveration ({}x failures): {}", failure_count, action)
            },
            Self::NegativeValence { indicators, .. } => {
                format!("Negative valence: {:?}", indicators)
            },
            Self::FrequentAbortRequests { count, reasons } => {
                format!("Frequent aborts ({}x): {:?}", count, reasons)
            },
            Self::DecisionParalysis {
                decide_duration_ms, ..
            } => {
                format!("Decision paralysis: {}ms", decide_duration_ms)
            },
        }
    }
}

// ============================================================================
// Intervention Types
// ============================================================================

/// Types of interventions that can be applied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Intervention {
    /// Pause execution temporarily.
    Pause {
        /// Reason for pausing.
        reason: String,
        /// Suggested duration.
        duration: Duration,
    },

    /// Inject a calming/grounding prompt.
    GroundingPrompt {
        /// The grounding message.
        message: String,
    },

    /// Reduce cognitive load.
    SimplifyTask {
        /// How to simplify.
        suggestion: String,
    },

    /// Clear recent context that may be causing loops.
    ClearRecentContext {
        /// How many steps to clear.
        steps_to_clear: u32,
    },

    /// Gracefully end the session.
    GracefulTermination {
        /// Reason for termination.
        reason: String,
        /// Summary of what was accomplished.
        summary: String,
    },

    /// Request human intervention.
    RequestHuman {
        /// What the human should know.
        situation: String,
        /// Suggested actions.
        suggestions: Vec<String>,
    },
}

// ============================================================================
// Wellbeing Snapshot
// ============================================================================

/// A point-in-time snapshot of agent wellbeing.
#[derive(Debug, Clone, Serialize)]
pub struct WellbeingSnapshot {
    /// Timestamp of this snapshot (skipped in serialization).
    #[serde(skip)]
    pub timestamp: Instant,
    /// Current iteration.
    pub iteration: u32,
    /// Current phase.
    pub phase: OodaPhase,
    /// Overall state.
    pub state: WellbeingState,
    /// Active distress signals.
    pub distress_signals: Vec<DistressSignal>,
    /// Coherence score (0.0 - 1.0).
    pub coherence_score: f32,
    /// Average recent confidence.
    pub avg_confidence: f32,
    /// Detected loop count.
    pub loop_count: u32,
    /// Recommended intervention if any.
    pub recommended_intervention: Option<Intervention>,
}

// ============================================================================
// Wellbeing Configuration
// ============================================================================

/// Configuration for wellbeing monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WellbeingConfig {
    /// Enable monitoring.
    pub enabled: bool,
    /// Window size for pattern detection.
    pub window_size: usize,
    /// Confidence threshold below which we become concerned.
    pub confidence_concern_threshold: f32,
    /// Number of similar iterations to consider a loop.
    pub loop_detection_threshold: u32,
    /// Maximum decide phase duration before concern (ms).
    pub max_decide_duration_ms: u64,
    /// Negative valence keywords to detect.
    pub negative_valence_keywords: Vec<String>,
    /// Enable automatic interventions.
    pub auto_intervene: bool,
}

impl Default for WellbeingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            window_size: 10,
            confidence_concern_threshold: 0.3,
            loop_detection_threshold: 3,
            max_decide_duration_ms: 30000,
            negative_valence_keywords: vec![
                "confused".to_string(),
                "stuck".to_string(),
                "cannot".to_string(),
                "impossible".to_string(),
                "frustrated".to_string(),
                "overwhelmed".to_string(),
                "lost".to_string(),
                "uncertain".to_string(),
                "failing".to_string(),
                "struggling".to_string(),
            ],
            auto_intervene: false,
        }
    }
}

// ============================================================================
// Monitoring State
// ============================================================================

/// Internal state for the wellbeing monitor.
struct MonitorState {
    /// Recent decisions for pattern analysis.
    recent_decisions: VecDeque<OodaDecision>,
    /// Recent action results.
    recent_results: VecDeque<OodaActionResult>,
    /// Recent orientations.
    recent_orientations: VecDeque<OodaOrientation>,
    /// Confidence history.
    confidence_history: VecDeque<f32>,
    /// Action history for loop detection.
    action_history: VecDeque<String>,
    /// Abort request count.
    abort_count: u32,
    /// Current iteration.
    current_iteration: u32,
    /// Current phase.
    current_phase: OodaPhase,
    /// Phase start time.
    phase_start: Instant,
    /// All detected signals (accumulated for future diagnostic reporting).
    _all_signals: Vec<DistressSignal>,
    /// Snapshots history.
    snapshots: VecDeque<WellbeingSnapshot>,
}

impl MonitorState {
    fn new(window_size: usize) -> Self {
        Self {
            recent_decisions: VecDeque::with_capacity(window_size),
            recent_results: VecDeque::with_capacity(window_size),
            recent_orientations: VecDeque::with_capacity(window_size),
            confidence_history: VecDeque::with_capacity(window_size),
            action_history: VecDeque::with_capacity(window_size * 2),
            abort_count: 0,
            current_iteration: 0,
            current_phase: OodaPhase::Observe,
            phase_start: Instant::now(),
            _all_signals: Vec::new(),
            snapshots: VecDeque::with_capacity(100),
        }
    }
}

// ============================================================================
// Wellbeing Monitor
// ============================================================================

/// Monitors agent wellbeing during execution.
pub struct WellbeingMonitor {
    config: WellbeingConfig,
    state: RwLock<MonitorState>,
    paused: AtomicBool,
    intervention_callback: Option<Arc<dyn Fn(Intervention) + Send + Sync>>,
}

impl WellbeingMonitor {
    /// Creates a new wellbeing monitor.
    #[must_use]
    pub fn new(config: WellbeingConfig) -> Self {
        Self {
            state: RwLock::new(MonitorState::new(config.window_size)),
            config,
            paused: AtomicBool::new(false),
            intervention_callback: None,
        }
    }

    /// Sets a callback for interventions.
    #[must_use]
    pub fn with_intervention_callback(
        mut self,
        callback: Arc<dyn Fn(Intervention) + Send + Sync>,
    ) -> Self {
        self.intervention_callback = Some(callback);
        self
    }

    /// Returns current wellbeing state.
    #[must_use]
    pub fn current_state(&self) -> WellbeingState {
        let state = self.state.read();
        self.assess_state(&state)
    }

    /// Returns current distress signals.
    #[must_use]
    pub fn current_signals(&self) -> Vec<DistressSignal> {
        let state = self.state.read();
        self.detect_signals(&state)
    }

    /// Returns whether execution is paused.
    #[must_use]
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }

    /// Pauses execution.
    pub fn pause(&self, reason: &str) {
        self.paused.store(true, Ordering::SeqCst);
        info!("Wellbeing monitor paused execution: {}", reason);
    }

    /// Resumes execution.
    pub fn resume(&self) {
        self.paused.store(false, Ordering::SeqCst);
        info!("Wellbeing monitor resumed execution");
    }

    /// Takes a snapshot of current wellbeing.
    #[must_use]
    pub fn snapshot(&self) -> WellbeingSnapshot {
        let state = self.state.read();
        let signals = self.detect_signals(&state);
        let wellbeing_state = self.assess_state(&state);
        let intervention = self.recommend_intervention(&signals, &wellbeing_state);

        WellbeingSnapshot {
            timestamp: Instant::now(),
            iteration: state.current_iteration,
            phase: state.current_phase,
            state: wellbeing_state,
            distress_signals: signals,
            coherence_score: self.calculate_coherence(&state),
            avg_confidence: self.calculate_avg_confidence(&state),
            loop_count: self.detect_loop_count(&state),
            recommended_intervention: intervention,
        }
    }

    /// Returns all historical snapshots.
    #[must_use]
    pub fn history(&self) -> Vec<WellbeingSnapshot> {
        self.state.read().snapshots.iter().cloned().collect()
    }

    /// Manually record a confidence value.
    ///
    /// This is useful when not using the full OODA executor but still
    /// want wellbeing tracking.
    pub fn record_confidence(&self, confidence: f32) {
        let mut state = self.state.write();
        if state.confidence_history.len() >= self.config.window_size {
            state.confidence_history.pop_front();
        }
        state.confidence_history.push_back(confidence);
    }

    /// Manually record a negative valence indicator.
    ///
    /// This is useful for flagging concerning patterns in agent output.
    /// Creates a synthetic decision with the indicator in its rationale.
    pub fn record_valence_indicator(&self, indicator: String) {
        use crate::ooda::{DecisionAction, OodaDecision};

        let synthetic_decision = OodaDecision {
            action: DecisionAction::GatherInfo {
                query: "valence marker".to_string(),
            },
            rationale: indicator,
            confidence: 0.5,
            alternatives: Vec::new(),
            expected_outcome: "tracking indicator".to_string(),
        };

        let mut state = self.state.write();
        if state.recent_decisions.len() >= self.config.window_size {
            state.recent_decisions.pop_front();
        }
        state.recent_decisions.push_back(synthetic_decision);
    }

    // ========================================================================
    // Internal Analysis Methods
    // ========================================================================

    fn assess_state(&self, state: &MonitorState) -> WellbeingState {
        let signals = self.detect_signals(state);

        if signals.is_empty() {
            return WellbeingState::Healthy;
        }

        let max_severity = signals
            .iter()
            .map(|s| s.severity())
            .fold(0.0_f32, |a, b| a.max(b));

        let signal_count = signals.len();

        match (max_severity, signal_count) {
            (s, _) if s >= 0.8 => WellbeingState::Distressed,
            (s, c) if s >= 0.6 || c >= 3 => WellbeingState::Concerned,
            (s, c) if s >= 0.3 || c >= 2 => WellbeingState::Cautious,
            _ => WellbeingState::Healthy,
        }
    }

    fn detect_signals(&self, state: &MonitorState) -> Vec<DistressSignal> {
        let mut signals = Vec::new();

        // Check confidence collapse
        if let Some(signal) = self.check_confidence_collapse(state) {
            signals.push(signal);
        }

        // Check rumination loops
        if let Some(signal) = self.check_rumination(state) {
            signals.push(signal);
        }

        // Check perseveration
        if let Some(signal) = self.check_perseveration(state) {
            signals.push(signal);
        }

        // Check negative valence
        if let Some(signal) = self.check_negative_valence(state) {
            signals.push(signal);
        }

        // Check abort frequency
        if let Some(signal) = self.check_abort_frequency(state) {
            signals.push(signal);
        }

        // Check decision paralysis
        if let Some(signal) = self.check_decision_paralysis(state) {
            signals.push(signal);
        }

        signals
    }

    fn check_confidence_collapse(&self, state: &MonitorState) -> Option<DistressSignal> {
        if state.confidence_history.len() < 3 {
            return None;
        }

        let recent: Vec<f32> = state.confidence_history.iter().copied().collect();
        let current = *recent.last()?;

        // Check if confidence is trending down and below threshold
        if current < self.config.confidence_concern_threshold {
            let earlier_avg: f32 =
                recent[..recent.len() / 2].iter().sum::<f32>() / (recent.len() / 2) as f32;
            let later_avg: f32 = recent[recent.len() / 2..].iter().sum::<f32>()
                / (recent.len() - recent.len() / 2) as f32;

            if later_avg < earlier_avg * 0.7 {
                return Some(DistressSignal::ConfidenceCollapse {
                    recent_confidences: recent,
                    current,
                });
            }
        }

        None
    }

    fn check_rumination(&self, state: &MonitorState) -> Option<DistressSignal> {
        if state.action_history.len() < self.config.loop_detection_threshold as usize {
            return None;
        }

        // Look for repeated patterns
        let actions: Vec<&String> = state.action_history.iter().collect();
        let window = self.config.loop_detection_threshold as usize;

        for pattern_len in 1..=3 {
            if actions.len() >= pattern_len * window {
                let pattern: Vec<_> = actions[actions.len() - pattern_len..].to_vec();
                let mut repeat_count = 1;

                for chunk in actions[..actions.len() - pattern_len]
                    .chunks(pattern_len)
                    .rev()
                {
                    if chunk.iter().zip(pattern.iter()).all(|(a, b)| a == b) {
                        repeat_count += 1;
                    } else {
                        break;
                    }
                }

                if repeat_count >= self.config.loop_detection_threshold {
                    return Some(DistressSignal::RuminationLoop {
                        loop_count: repeat_count,
                        pattern: pattern
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(" → "),
                    });
                }
            }
        }

        None
    }

    fn check_perseveration(&self, state: &MonitorState) -> Option<DistressSignal> {
        let failure_threshold = 3;
        let recent_failures: Vec<_> = state
            .recent_results
            .iter()
            .rev()
            .take(5)
            .filter(|r| !r.success)
            .collect();

        if recent_failures.len() >= failure_threshold {
            // Check if they're the same type of failure
            if let Some(first_error) = recent_failures.first().and_then(|r| r.error.as_ref()) {
                let same_error_count = recent_failures
                    .iter()
                    .filter(|r| r.error.as_ref() == Some(first_error))
                    .count();

                if same_error_count >= failure_threshold {
                    return Some(DistressSignal::PerseverationPattern {
                        failure_count: same_error_count as u32,
                        action: first_error.clone(),
                    });
                }
            }
        }

        None
    }

    fn check_negative_valence(&self, state: &MonitorState) -> Option<DistressSignal> {
        let mut indicators = Vec::new();
        let mut intensity = 0.0_f32;

        // Check recent decisions and orientations for negative language
        for decision in state.recent_decisions.iter().rev().take(3) {
            let text = decision.rationale.to_lowercase();
            for keyword in &self.config.negative_valence_keywords {
                if text.contains(keyword) {
                    indicators.push(keyword.clone());
                    intensity += 0.2;
                }
            }
        }

        for orientation in state.recent_orientations.iter().rev().take(3) {
            let text = orientation.situation.to_lowercase();
            for keyword in &self.config.negative_valence_keywords {
                if text.contains(keyword) && !indicators.contains(keyword) {
                    indicators.push(keyword.clone());
                    intensity += 0.15;
                }
            }
        }

        if !indicators.is_empty() {
            Some(DistressSignal::NegativeValence {
                indicators,
                intensity: intensity.min(1.0),
            })
        } else {
            None
        }
    }

    fn check_abort_frequency(&self, state: &MonitorState) -> Option<DistressSignal> {
        if state.abort_count >= 2 {
            let reasons: Vec<String> = state
                .recent_decisions
                .iter()
                .filter_map(|d| {
                    if let DecisionAction::Abort { reason } = &d.action {
                        Some(reason.clone())
                    } else {
                        None
                    }
                })
                .collect();

            Some(DistressSignal::FrequentAbortRequests {
                count: state.abort_count,
                reasons,
            })
        } else {
            None
        }
    }

    fn check_decision_paralysis(&self, state: &MonitorState) -> Option<DistressSignal> {
        if state.current_phase == OodaPhase::Decide {
            let duration = state.phase_start.elapsed().as_millis() as u64;
            if duration > self.config.max_decide_duration_ms {
                let alt_count = state
                    .recent_decisions
                    .back()
                    .map(|d| d.alternatives.len())
                    .unwrap_or(0);

                return Some(DistressSignal::DecisionParalysis {
                    decide_duration_ms: duration,
                    alternatives_count: alt_count,
                });
            }
        }
        None
    }

    fn calculate_coherence(&self, state: &MonitorState) -> f32 {
        // Simple coherence metric based on decision-result alignment
        if state.recent_decisions.is_empty() || state.recent_results.is_empty() {
            return 1.0;
        }

        let success_rate = state.recent_results.iter().filter(|r| r.success).count() as f32
            / state.recent_results.len() as f32;

        let avg_confidence = self.calculate_avg_confidence(state);

        // Coherence is high when success aligns with confidence
        let coherence = if avg_confidence > 0.5 {
            success_rate
        } else {
            1.0 - (0.5 - avg_confidence).abs()
        };

        coherence.clamp(0.0, 1.0)
    }

    fn calculate_avg_confidence(&self, state: &MonitorState) -> f32 {
        if state.confidence_history.is_empty() {
            return 0.7; // Default
        }
        state.confidence_history.iter().sum::<f32>() / state.confidence_history.len() as f32
    }

    fn detect_loop_count(&self, state: &MonitorState) -> u32 {
        if let Some(DistressSignal::RuminationLoop { loop_count, .. }) =
            self.check_rumination(state)
        {
            loop_count
        } else {
            0
        }
    }

    fn recommend_intervention(
        &self,
        signals: &[DistressSignal],
        state: &WellbeingState,
    ) -> Option<Intervention> {
        if signals.is_empty() || *state == WellbeingState::Healthy {
            return None;
        }

        // Find the most severe signal
        let max_signal = signals.iter().max_by(|a, b| {
            a.severity()
                .partial_cmp(&b.severity())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;

        match max_signal {
            DistressSignal::RuminationLoop { loop_count, .. } if *loop_count >= 4 => {
                Some(Intervention::ClearRecentContext {
                    steps_to_clear: *loop_count,
                })
            },
            DistressSignal::ConfidenceCollapse { .. } => Some(Intervention::GroundingPrompt {
                message: "Take a moment to reassess. What do you know for certain? \
                          What is the simplest next step?"
                    .to_string(),
            }),
            DistressSignal::DecisionParalysis { .. } => Some(Intervention::SimplifyTask {
                suggestion: "Consider breaking this decision into smaller parts. \
                             What is the smallest useful action you could take?"
                    .to_string(),
            }),
            DistressSignal::NegativeValence { intensity, .. } if *intensity > 0.7 => {
                Some(Intervention::Pause {
                    reason: "High negative valence detected".to_string(),
                    duration: Duration::from_secs(5),
                })
            },
            DistressSignal::FrequentAbortRequests { count, .. } if *count >= 3 => {
                Some(Intervention::RequestHuman {
                    situation: "Agent has requested to abort multiple times".to_string(),
                    suggestions: vec![
                        "Review task requirements".to_string(),
                        "Provide additional context".to_string(),
                        "Consider a different approach".to_string(),
                    ],
                })
            },
            _ if *state == WellbeingState::Distressed => Some(Intervention::GracefulTermination {
                reason: "Multiple distress signals detected".to_string(),
                summary: "Ending session to prevent potential harm".to_string(),
            }),
            _ => None,
        }
    }

    fn apply_intervention(&self, intervention: &Intervention) {
        debug!("Applying intervention: {:?}", intervention);

        if let Some(ref callback) = self.intervention_callback {
            callback(intervention.clone());
        }

        match intervention {
            Intervention::Pause { reason, .. } => {
                self.pause(reason);
            },
            Intervention::GracefulTermination { reason, .. } => {
                warn!("Recommending graceful termination: {}", reason);
                self.pause(reason);
            },
            _ => {},
        }
    }
}

// ============================================================================
// OodaCallback Implementation
// ============================================================================

#[async_trait]
impl OodaCallback for WellbeingMonitor {
    async fn on_phase(&self, iteration: u32, phase: OodaPhase) {
        if !self.config.enabled {
            return;
        }

        let mut state = self.state.write();
        state.current_iteration = iteration;
        state.current_phase = phase;
        state.phase_start = Instant::now();

        debug!(
            "Wellbeing: iteration {} entering phase {}",
            iteration, phase
        );
    }

    async fn on_observation(&self, observation: &OodaObservation) {
        if !self.config.enabled {
            return;
        }

        debug!(
            "Wellbeing: observation from {} (relevance: {:.2})",
            observation.source, observation.relevance
        );
    }

    async fn on_orientation(&self, orientation: &OodaOrientation) {
        if !self.config.enabled {
            return;
        }

        let mut state = self.state.write();

        if state.recent_orientations.len() >= self.config.window_size {
            state.recent_orientations.pop_front();
        }
        state.recent_orientations.push_back(orientation.clone());

        debug!(
            "Wellbeing: orientation complete, {} threats identified",
            orientation.threats.len()
        );
    }

    async fn on_decision(&self, decision: &OodaDecision) {
        if !self.config.enabled {
            return;
        }

        let mut state = self.state.write();

        // Track decision
        if state.recent_decisions.len() >= self.config.window_size {
            state.recent_decisions.pop_front();
        }
        state.recent_decisions.push_back(decision.clone());

        // Track confidence
        if state.confidence_history.len() >= self.config.window_size {
            state.confidence_history.pop_front();
        }
        state.confidence_history.push_back(decision.confidence);

        // Track action for loop detection
        let action_str = match &decision.action {
            DecisionAction::ExecuteTool { tool, .. } => format!("tool:{}", tool),
            DecisionAction::GatherInfo { .. } => "gather_info".to_string(),
            DecisionAction::FinalAnswer { .. } => "final_answer".to_string(),
            DecisionAction::RequestInput { .. } => "request_input".to_string(),
            DecisionAction::Abort { .. } => {
                state.abort_count += 1;
                "abort".to_string()
            },
        };

        if state.action_history.len() >= self.config.window_size * 2 {
            state.action_history.pop_front();
        }
        state.action_history.push_back(action_str);

        debug!(
            "Wellbeing: decision made with confidence {:.2}",
            decision.confidence
        );

        // Check if we need to intervene
        drop(state); // Release lock before checking signals

        let snapshot = self.snapshot();
        if snapshot.state.needs_attention() {
            warn!(
                "Wellbeing concern detected: {:?} ({} signals)",
                snapshot.state,
                snapshot.distress_signals.len()
            );

            if self.config.auto_intervene {
                if let Some(ref intervention) = snapshot.recommended_intervention {
                    self.apply_intervention(intervention);
                }
            }
        }

        // Store snapshot
        let mut state = self.state.write();
        if state.snapshots.len() >= 100 {
            state.snapshots.pop_front();
        }
        state.snapshots.push_back(snapshot);
    }

    async fn on_action(&self, result: &OodaActionResult) {
        if !self.config.enabled {
            return;
        }

        let mut state = self.state.write();

        if state.recent_results.len() >= self.config.window_size {
            state.recent_results.pop_front();
        }
        state.recent_results.push_back(result.clone());

        debug!(
            "Wellbeing: action {} (duration: {}ms)",
            if result.success {
                "succeeded"
            } else {
                "failed"
            },
            result.duration_ms
        );
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wellbeing_state_needs_attention() {
        assert!(!WellbeingState::Healthy.needs_attention());
        assert!(!WellbeingState::Cautious.needs_attention());
        assert!(WellbeingState::Concerned.needs_attention());
        assert!(WellbeingState::Distressed.needs_attention());
    }

    #[test]
    fn test_wellbeing_state_needs_intervention() {
        assert!(!WellbeingState::Healthy.needs_intervention());
        assert!(!WellbeingState::Cautious.needs_intervention());
        assert!(!WellbeingState::Concerned.needs_intervention());
        assert!(WellbeingState::Distressed.needs_intervention());
    }

    #[test]
    fn test_distress_signal_severity() {
        let signal = DistressSignal::ConfidenceCollapse {
            recent_confidences: vec![0.8, 0.5, 0.2],
            current: 0.2,
        };
        assert!((signal.severity() - 0.8).abs() < 0.01);

        let loop_signal = DistressSignal::RuminationLoop {
            loop_count: 3,
            pattern: "test".to_string(),
        };
        assert!((loop_signal.severity() - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_wellbeing_config_default() {
        let config = WellbeingConfig::default();
        assert!(config.enabled);
        assert_eq!(config.window_size, 10);
        assert!((config.confidence_concern_threshold - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_wellbeing_monitor_creation() {
        let config = WellbeingConfig::default();
        let monitor = WellbeingMonitor::new(config);
        assert_eq!(monitor.current_state(), WellbeingState::Healthy);
        assert!(!monitor.is_paused());
    }

    #[test]
    fn test_wellbeing_monitor_pause_resume() {
        let config = WellbeingConfig::default();
        let monitor = WellbeingMonitor::new(config);

        assert!(!monitor.is_paused());
        monitor.pause("test");
        assert!(monitor.is_paused());
        monitor.resume();
        assert!(!monitor.is_paused());
    }

    #[test]
    fn test_intervention_types() {
        let pause = Intervention::Pause {
            reason: "test".to_string(),
            duration: Duration::from_secs(5),
        };
        assert!(matches!(pause, Intervention::Pause { .. }));

        let grounding = Intervention::GroundingPrompt {
            message: "calm down".to_string(),
        };
        assert!(matches!(grounding, Intervention::GroundingPrompt { .. }));
    }

    #[test]
    fn test_snapshot_creation() {
        let config = WellbeingConfig::default();
        let monitor = WellbeingMonitor::new(config);
        let snapshot = monitor.snapshot();

        assert_eq!(snapshot.state, WellbeingState::Healthy);
        assert!(snapshot.distress_signals.is_empty());
        assert!((snapshot.coherence_score - 1.0).abs() < 0.01);
    }
}
