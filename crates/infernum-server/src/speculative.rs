//! Speculative decoding integration for accelerated inference.
//!
//! This module provides HTTP API integration for speculative decoding,
//! coordinating with the batch scheduler to enable 2-3x inference speedup.
//!
//! # Overview
//!
//! Speculative decoding uses a smaller "draft" model to generate candidate tokens,
//! which are then verified by the main model in a single forward pass. This can
//! significantly speed up inference for models with similar vocabularies.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                     Speculative Decoding Pipeline                       │
//! │  ┌──────────────────────────────────────────────────────────────────┐  │
//! │  │ 1. Draft Generation                                              │  │
//! │  │    Draft model generates K tokens quickly                        │  │
//! │  │    [t1, t2, t3, t4, t5]                                          │  │
//! │  └──────────────────────────────────────────────────────────────────┘  │
//! │                                 │                                       │
//! │                                 ▼                                       │
//! │  ┌──────────────────────────────────────────────────────────────────┐  │
//! │  │ 2. Batch Verification                                            │  │
//! │  │    Main model evaluates all K+1 positions in one forward pass    │  │
//! │  │    [p1, p2, p3, p4, p5, p6]  (logits for each position)          │  │
//! │  └──────────────────────────────────────────────────────────────────┘  │
//! │                                 │                                       │
//! │                                 ▼                                       │
//! │  ┌──────────────────────────────────────────────────────────────────┐  │
//! │  │ 3. Accept/Reject                                                 │  │
//! │  │    Compare probabilities, accept matching tokens                 │  │
//! │  │    [t1✓, t2✓, t3✓, t4✗] → Accept 3, resample at position 4      │  │
//! │  └──────────────────────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Configuration
//!
//! ```rust,ignore
//! use infernum_server::speculative::{SpeculativeConfig, SpeculativeMode};
//!
//! let config = SpeculativeConfig::new("meta-llama/Llama-3.2-1B-Instruct")
//!     .with_num_tokens(5)
//!     .with_acceptance_threshold(0.9)
//!     .with_mode(SpeculativeMode::Auto);
//! ```
//!
//! # API Usage
//!
//! Enable speculative decoding via request parameter:
//!
//! ```json
//! {
//!   "model": "meta-llama/Llama-3.2-8B-Instruct",
//!   "messages": [...],
//!   "speculative": {
//!     "enabled": true,
//!     "draft_model": "meta-llama/Llama-3.2-1B-Instruct",
//!     "num_tokens": 5
//!   }
//! }
//! ```

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::batching::{BatchPriority, BatchScheduler, SamplingParams};

// ============================================================================
// Configuration
// ============================================================================

/// Speculative decoding mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpeculativeMode {
    /// Disabled - no speculative decoding.
    #[default]
    Disabled,
    /// Enabled - always use speculative decoding.
    Enabled,
    /// Auto - enable for deterministic sampling (temperature <= 0.1).
    Auto,
}

impl SpeculativeMode {
    /// Checks if speculative decoding should be used.
    pub fn should_speculate(&self, temperature: f32) -> bool {
        match self {
            Self::Disabled => false,
            Self::Enabled => true,
            Self::Auto => temperature <= 0.1,
        }
    }
}

impl fmt::Display for SpeculativeMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disabled => write!(f, "disabled"),
            Self::Enabled => write!(f, "enabled"),
            Self::Auto => write!(f, "auto"),
        }
    }
}

/// Configuration for speculative decoding in the HTTP API layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeConfig {
    /// Draft model identifier.
    pub draft_model: String,

    /// Number of speculative tokens to generate per round.
    pub num_tokens: u32,

    /// Acceptance threshold (0.0 - 1.0).
    /// Higher values = stricter matching, fewer accepted tokens.
    pub acceptance_threshold: f32,

    /// Speculative decoding mode.
    pub mode: SpeculativeMode,

    /// Maximum draft model batch size.
    pub max_draft_batch_size: u32,

    /// Enable adaptive speculation (adjust num_tokens based on acceptance rate).
    pub adaptive: bool,

    /// Minimum tokens to speculate (when adaptive).
    pub min_tokens: u32,

    /// Maximum tokens to speculate (when adaptive).
    pub max_tokens: u32,

    /// Target acceptance rate for adaptive speculation.
    pub target_acceptance_rate: f32,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            draft_model: String::new(),
            num_tokens: 5,
            acceptance_threshold: 0.9,
            mode: SpeculativeMode::Disabled,
            max_draft_batch_size: 32,
            adaptive: false,
            min_tokens: 2,
            max_tokens: 8,
            target_acceptance_rate: 0.7,
        }
    }
}

impl SpeculativeConfig {
    /// Creates a new speculative config with the given draft model.
    pub fn new(draft_model: impl Into<String>) -> Self {
        Self {
            draft_model: draft_model.into(),
            mode: SpeculativeMode::Enabled,
            ..Default::default()
        }
    }

    /// Builder method for num_tokens.
    pub fn with_num_tokens(mut self, num: u32) -> Self {
        self.num_tokens = num;
        self
    }

    /// Builder method for acceptance_threshold.
    pub fn with_acceptance_threshold(mut self, threshold: f32) -> Self {
        self.acceptance_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Builder method for mode.
    pub fn with_mode(mut self, mode: SpeculativeMode) -> Self {
        self.mode = mode;
        self
    }

    /// Builder method for adaptive speculation.
    pub fn with_adaptive(mut self, adaptive: bool) -> Self {
        self.adaptive = adaptive;
        self
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), SpeculativeError> {
        if self.mode != SpeculativeMode::Disabled && self.draft_model.is_empty() {
            return Err(SpeculativeError::Configuration(
                "draft_model is required when speculative decoding is enabled".to_string(),
            ));
        }

        if self.num_tokens == 0 {
            return Err(SpeculativeError::Configuration(
                "num_tokens must be greater than 0".to_string(),
            ));
        }

        if self.num_tokens > 16 {
            return Err(SpeculativeError::Configuration(
                "num_tokens cannot exceed 16".to_string(),
            ));
        }

        if self.min_tokens > self.max_tokens {
            return Err(SpeculativeError::Configuration(
                "min_tokens cannot exceed max_tokens".to_string(),
            ));
        }

        Ok(())
    }
}

// ============================================================================
// Per-Request Speculative Parameters
// ============================================================================

/// Per-request speculative decoding parameters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpeculativeParams {
    /// Enable speculative decoding for this request.
    #[serde(default)]
    pub enabled: bool,

    /// Override draft model (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub draft_model: Option<String>,

    /// Override number of speculative tokens (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_tokens: Option<u32>,

    /// Override acceptance threshold (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acceptance_threshold: Option<f32>,
}

impl SpeculativeParams {
    /// Creates new params with speculative enabled.
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Default::default()
        }
    }

    /// Creates new params with speculative disabled.
    pub fn disabled() -> Self {
        Self::default()
    }

    /// Builder method for draft_model override.
    pub fn with_draft_model(mut self, model: impl Into<String>) -> Self {
        self.draft_model = Some(model.into());
        self
    }

    /// Builder method for num_tokens override.
    pub fn with_num_tokens(mut self, num: u32) -> Self {
        self.num_tokens = Some(num);
        self
    }
}

// ============================================================================
// Draft Token
// ============================================================================

/// A token generated by the draft model.
#[derive(Debug, Clone)]
pub struct DraftToken {
    /// Token ID.
    pub token_id: u32,

    /// Draft model's probability for this token.
    pub draft_prob: f32,

    /// Position in the sequence.
    pub position: usize,

    /// Generation timestamp.
    pub generated_at: Instant,
}

impl DraftToken {
    /// Creates a new draft token.
    pub fn new(token_id: u32, draft_prob: f32, position: usize) -> Self {
        Self {
            token_id,
            draft_prob,
            position,
            generated_at: Instant::now(),
        }
    }
}

// ============================================================================
// Verification Result
// ============================================================================

/// Result of verifying draft tokens against the main model.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Number of tokens accepted.
    pub accepted: usize,

    /// Number of tokens rejected.
    pub rejected: usize,

    /// The next token to emit (sampled from main model).
    pub next_token: Option<u32>,

    /// Main model probabilities for each draft token.
    pub main_probs: Vec<f32>,

    /// Whether EOS was reached.
    pub eos_reached: bool,

    /// Verification duration.
    pub duration: Duration,
}

impl VerificationResult {
    /// Returns the acceptance rate for this verification round.
    pub fn acceptance_rate(&self) -> f32 {
        let total = self.accepted + self.rejected;
        if total == 0 {
            0.0
        } else {
            self.accepted as f32 / total as f32
        }
    }

    /// Returns total tokens verified.
    pub fn total_verified(&self) -> usize {
        self.accepted + self.rejected
    }
}

// ============================================================================
// Speculative Statistics
// ============================================================================

/// Statistics from speculative decoding.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// Total speculation rounds.
    pub rounds: u64,

    /// Total draft tokens generated.
    pub draft_tokens: u64,

    /// Total tokens accepted.
    pub accepted_tokens: u64,

    /// Total tokens rejected.
    pub rejected_tokens: u64,

    /// Total draft generation time.
    pub draft_time_ns: u64,

    /// Total verification time.
    pub verify_time_ns: u64,

    /// EOS reached via speculation count.
    pub eos_reached: u64,
}

impl SpeculativeStats {
    /// Returns the overall acceptance rate.
    pub fn acceptance_rate(&self) -> f32 {
        if self.draft_tokens == 0 {
            0.0
        } else {
            self.accepted_tokens as f32 / self.draft_tokens as f32
        }
    }

    /// Returns average tokens accepted per round.
    pub fn avg_accepted_per_round(&self) -> f32 {
        if self.rounds == 0 {
            0.0
        } else {
            self.accepted_tokens as f32 / self.rounds as f32
        }
    }

    /// Returns average draft generation time.
    pub fn avg_draft_time(&self) -> Duration {
        if self.rounds == 0 {
            Duration::ZERO
        } else {
            Duration::from_nanos(self.draft_time_ns / self.rounds)
        }
    }

    /// Returns average verification time.
    pub fn avg_verify_time(&self) -> Duration {
        if self.rounds == 0 {
            Duration::ZERO
        } else {
            Duration::from_nanos(self.verify_time_ns / self.rounds)
        }
    }

    /// Returns speedup factor estimate.
    /// Assumes sequential decoding would take N forward passes for N tokens.
    pub fn estimated_speedup(&self) -> f32 {
        if self.rounds == 0 {
            1.0
        } else {
            // Total tokens generated = accepted + rejected (resampled) + next tokens
            let total_tokens = self.accepted_tokens + self.rounds;
            // Rounds = number of main model forward passes
            total_tokens as f32 / self.rounds as f32
        }
    }
}

// ============================================================================
// Speculative State
// ============================================================================

/// Current state of speculative decoding for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculativeState {
    /// Not using speculative decoding.
    Disabled,
    /// Generating draft tokens.
    Drafting,
    /// Verifying draft tokens with main model.
    Verifying,
    /// Waiting for scheduler.
    Queued,
    /// Completed.
    Completed,
    /// Failed.
    Failed,
}

impl fmt::Display for SpeculativeState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disabled => write!(f, "disabled"),
            Self::Drafting => write!(f, "drafting"),
            Self::Verifying => write!(f, "verifying"),
            Self::Queued => write!(f, "queued"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

// ============================================================================
// Speculative Request
// ============================================================================

/// A request for speculative generation.
#[derive(Debug)]
pub struct SpeculativeRequest {
    /// Request ID.
    pub request_id: String,

    /// Main model ID.
    pub main_model: String,

    /// Draft model ID.
    pub draft_model: String,

    /// Prompt token IDs.
    pub prompt_tokens: Vec<u32>,

    /// Maximum tokens to generate.
    pub max_tokens: u32,

    /// Sampling parameters.
    pub sampling_params: SamplingParams,

    /// Number of speculative tokens per round.
    pub num_spec_tokens: u32,

    /// Acceptance threshold.
    pub acceptance_threshold: f32,

    /// Request priority.
    pub priority: BatchPriority,

    /// Current state.
    pub state: SpeculativeState,

    /// Accumulated statistics.
    pub stats: SpeculativeStats,

    /// Generated tokens so far.
    pub generated_tokens: Vec<u32>,

    /// EOS token ID.
    pub eos_token: u32,
}

impl SpeculativeRequest {
    /// Creates a new speculative request.
    pub fn new(
        request_id: impl Into<String>,
        main_model: impl Into<String>,
        draft_model: impl Into<String>,
        prompt_tokens: Vec<u32>,
        max_tokens: u32,
        eos_token: u32,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            main_model: main_model.into(),
            draft_model: draft_model.into(),
            prompt_tokens,
            max_tokens,
            sampling_params: SamplingParams::default(),
            num_spec_tokens: 5,
            acceptance_threshold: 0.9,
            priority: BatchPriority::Normal,
            state: SpeculativeState::Queued,
            stats: SpeculativeStats::default(),
            generated_tokens: Vec::new(),
            eos_token,
        }
    }

    /// Builder method for sampling params.
    pub fn with_sampling(mut self, params: SamplingParams) -> Self {
        self.sampling_params = params;
        self
    }

    /// Builder method for priority.
    pub fn with_priority(mut self, priority: BatchPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Returns true if generation is complete.
    pub fn is_complete(&self) -> bool {
        self.state == SpeculativeState::Completed
            || self.state == SpeculativeState::Failed
            || self.generated_tokens.len() >= self.max_tokens as usize
    }

    /// Records a speculation round.
    pub fn record_round(&mut self, result: &VerificationResult, draft_time: Duration) {
        self.stats.rounds += 1;
        self.stats.draft_tokens += (result.accepted + result.rejected) as u64;
        self.stats.accepted_tokens += result.accepted as u64;
        self.stats.rejected_tokens += result.rejected as u64;
        self.stats.draft_time_ns += draft_time.as_nanos() as u64;
        self.stats.verify_time_ns += result.duration.as_nanos() as u64;
        if result.eos_reached {
            self.stats.eos_reached += 1;
        }
    }
}

// ============================================================================
// Speculative Scheduler
// ============================================================================

/// Scheduler for speculative decoding that coordinates draft and main model.
pub struct SpeculativeScheduler {
    /// Configuration.
    config: SpeculativeConfig,

    /// Main model batch scheduler.
    main_scheduler: Arc<BatchScheduler>,

    /// Draft model batch scheduler (optional - may use same scheduler).
    draft_scheduler: Option<Arc<BatchScheduler>>,

    /// Metrics.
    metrics: Arc<SpeculativeMetrics>,

    /// Active speculative requests.
    active_requests: RwLock<Vec<SpeculativeRequest>>,

    /// Adaptive speculation state.
    adaptive_tokens: RwLock<u32>,
}

impl SpeculativeScheduler {
    /// Creates a new speculative scheduler.
    pub fn new(config: SpeculativeConfig, main_scheduler: Arc<BatchScheduler>) -> Self {
        let initial_tokens = config.num_tokens;
        Self {
            config,
            main_scheduler,
            draft_scheduler: None,
            metrics: Arc::new(SpeculativeMetrics::new()),
            active_requests: RwLock::new(Vec::new()),
            adaptive_tokens: RwLock::new(initial_tokens),
        }
    }

    /// Sets a separate draft scheduler.
    pub fn with_draft_scheduler(mut self, scheduler: Arc<BatchScheduler>) -> Self {
        self.draft_scheduler = Some(scheduler);
        self
    }

    /// Returns the configuration.
    pub fn config(&self) -> &SpeculativeConfig {
        &self.config
    }

    /// Returns current metrics.
    pub fn metrics(&self) -> Arc<SpeculativeMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Returns the main model scheduler.
    #[must_use]
    pub fn main_scheduler(&self) -> &Arc<BatchScheduler> {
        &self.main_scheduler
    }

    /// Returns current adaptive token count.
    pub fn current_tokens(&self) -> u32 {
        if self.config.adaptive {
            *self.adaptive_tokens.read()
        } else {
            self.config.num_tokens
        }
    }

    /// Submits a request for speculative generation.
    pub fn submit(&self, request: SpeculativeRequest) -> Result<(), SpeculativeError> {
        if self.config.mode == SpeculativeMode::Disabled {
            return Err(SpeculativeError::Disabled);
        }

        self.metrics.requests_total.fetch_add(1, Ordering::Relaxed);
        self.active_requests.write().push(request);
        Ok(())
    }

    /// Returns number of active speculative requests.
    pub fn active_count(&self) -> usize {
        self.active_requests.read().len()
    }

    /// Checks if speculative decoding should be used for given parameters.
    pub fn should_speculate(&self, temperature: f32) -> bool {
        self.config.mode.should_speculate(temperature)
    }

    /// Updates adaptive token count based on recent acceptance rate.
    pub fn update_adaptive(&self, acceptance_rate: f32) {
        if !self.config.adaptive {
            return;
        }

        let mut tokens = self.adaptive_tokens.write();
        let target = self.config.target_acceptance_rate;

        if acceptance_rate > target + 0.1 {
            // High acceptance - increase speculation
            *tokens = (*tokens + 1).min(self.config.max_tokens);
        } else if acceptance_rate < target - 0.1 {
            // Low acceptance - decrease speculation
            *tokens = tokens.saturating_sub(1).max(self.config.min_tokens);
        }
    }

    /// Removes a completed request.
    pub fn remove(&self, request_id: &str) -> Option<SpeculativeRequest> {
        let mut requests = self.active_requests.write();
        if let Some(idx) = requests.iter().position(|r| r.request_id == request_id) {
            let request = requests.remove(idx);
            if request.state == SpeculativeState::Completed {
                self.metrics
                    .requests_success
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                self.metrics.requests_failed.fetch_add(1, Ordering::Relaxed);
            }
            Some(request)
        } else {
            None
        }
    }

    /// Gets aggregated statistics.
    pub fn get_stats(&self) -> SpeculativeStats {
        let requests = self.active_requests.read();
        let mut total = SpeculativeStats::default();

        for request in requests.iter() {
            total.rounds += request.stats.rounds;
            total.draft_tokens += request.stats.draft_tokens;
            total.accepted_tokens += request.stats.accepted_tokens;
            total.rejected_tokens += request.stats.rejected_tokens;
            total.draft_time_ns += request.stats.draft_time_ns;
            total.verify_time_ns += request.stats.verify_time_ns;
            total.eos_reached += request.stats.eos_reached;
        }

        total
    }
}

impl fmt::Debug for SpeculativeScheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpeculativeScheduler")
            .field("config", &self.config)
            .field("active_requests", &self.active_requests.read().len())
            .field("current_tokens", &self.current_tokens())
            .finish()
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Errors from speculative decoding.
#[derive(Debug, Clone)]
pub enum SpeculativeError {
    /// Speculative decoding is disabled.
    Disabled,
    /// Configuration error.
    Configuration(String),
    /// Draft model error.
    DraftModel(String),
    /// Verification error.
    Verification(String),
    /// Scheduler error.
    Scheduler(String),
    /// Request not found.
    NotFound(String),
}

impl fmt::Display for SpeculativeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disabled => write!(f, "Speculative decoding is disabled"),
            Self::Configuration(msg) => write!(f, "Configuration error: {}", msg),
            Self::DraftModel(msg) => write!(f, "Draft model error: {}", msg),
            Self::Verification(msg) => write!(f, "Verification error: {}", msg),
            Self::Scheduler(msg) => write!(f, "Scheduler error: {}", msg),
            Self::NotFound(msg) => write!(f, "Not found: {}", msg),
        }
    }
}

impl std::error::Error for SpeculativeError {}

// ============================================================================
// Metrics
// ============================================================================

/// Prometheus metrics for speculative decoding.
#[derive(Debug)]
pub struct SpeculativeMetrics {
    /// Total speculative requests.
    requests_total: AtomicU64,
    /// Successful speculative requests.
    requests_success: AtomicU64,
    /// Failed speculative requests.
    requests_failed: AtomicU64,
    /// Total speculation rounds.
    rounds_total: AtomicU64,
    /// Total draft tokens generated.
    draft_tokens_total: AtomicU64,
    /// Total tokens accepted.
    accepted_tokens_total: AtomicU64,
    /// Total tokens rejected.
    rejected_tokens_total: AtomicU64,
    /// Total draft generation time (nanoseconds).
    draft_time_ns: AtomicU64,
    /// Total verification time (nanoseconds).
    verify_time_ns: AtomicU64,
    /// Started at.
    started_at: Instant,
}

impl SpeculativeMetrics {
    /// Creates new metrics.
    pub fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            requests_success: AtomicU64::new(0),
            requests_failed: AtomicU64::new(0),
            rounds_total: AtomicU64::new(0),
            draft_tokens_total: AtomicU64::new(0),
            accepted_tokens_total: AtomicU64::new(0),
            rejected_tokens_total: AtomicU64::new(0),
            draft_time_ns: AtomicU64::new(0),
            verify_time_ns: AtomicU64::new(0),
            started_at: Instant::now(),
        }
    }

    /// Records a speculation round.
    pub fn record_round(&self, result: &VerificationResult, draft_time: Duration) {
        self.rounds_total.fetch_add(1, Ordering::Relaxed);
        self.draft_tokens_total.fetch_add(
            (result.accepted + result.rejected) as u64,
            Ordering::Relaxed,
        );
        self.accepted_tokens_total
            .fetch_add(result.accepted as u64, Ordering::Relaxed);
        self.rejected_tokens_total
            .fetch_add(result.rejected as u64, Ordering::Relaxed);
        self.draft_time_ns
            .fetch_add(draft_time.as_nanos() as u64, Ordering::Relaxed);
        self.verify_time_ns
            .fetch_add(result.duration.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Returns total requests.
    pub fn requests_total(&self) -> u64 {
        self.requests_total.load(Ordering::Relaxed)
    }

    /// Returns successful requests.
    pub fn requests_success(&self) -> u64 {
        self.requests_success.load(Ordering::Relaxed)
    }

    /// Returns failed requests.
    pub fn requests_failed(&self) -> u64 {
        self.requests_failed.load(Ordering::Relaxed)
    }

    /// Returns total rounds.
    pub fn rounds_total(&self) -> u64 {
        self.rounds_total.load(Ordering::Relaxed)
    }

    /// Returns acceptance rate.
    pub fn acceptance_rate(&self) -> f64 {
        let draft = self.draft_tokens_total.load(Ordering::Relaxed);
        let accepted = self.accepted_tokens_total.load(Ordering::Relaxed);
        if draft == 0 {
            0.0
        } else {
            accepted as f64 / draft as f64
        }
    }

    /// Returns estimated speedup.
    pub fn estimated_speedup(&self) -> f64 {
        let rounds = self.rounds_total.load(Ordering::Relaxed);
        let accepted = self.accepted_tokens_total.load(Ordering::Relaxed);
        if rounds == 0 {
            1.0
        } else {
            (accepted + rounds) as f64 / rounds as f64
        }
    }

    /// Returns uptime.
    pub fn uptime(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Renders metrics in Prometheus format.
    pub fn prometheus(&self) -> String {
        let mut output = String::new();

        output.push_str("# HELP infernum_speculative_requests_total Total speculative requests\n");
        output.push_str("# TYPE infernum_speculative_requests_total counter\n");
        output.push_str(&format!(
            "infernum_speculative_requests_total {}\n",
            self.requests_total()
        ));

        output.push_str(
            "# HELP infernum_speculative_requests_success Successful speculative requests\n",
        );
        output.push_str("# TYPE infernum_speculative_requests_success counter\n");
        output.push_str(&format!(
            "infernum_speculative_requests_success {}\n",
            self.requests_success()
        ));

        output
            .push_str("# HELP infernum_speculative_requests_failed Failed speculative requests\n");
        output.push_str("# TYPE infernum_speculative_requests_failed counter\n");
        output.push_str(&format!(
            "infernum_speculative_requests_failed {}\n",
            self.requests_failed()
        ));

        output.push_str("# HELP infernum_speculative_rounds_total Total speculation rounds\n");
        output.push_str("# TYPE infernum_speculative_rounds_total counter\n");
        output.push_str(&format!(
            "infernum_speculative_rounds_total {}\n",
            self.rounds_total()
        ));

        output.push_str(
            "# HELP infernum_speculative_draft_tokens_total Total draft tokens generated\n",
        );
        output.push_str("# TYPE infernum_speculative_draft_tokens_total counter\n");
        output.push_str(&format!(
            "infernum_speculative_draft_tokens_total {}\n",
            self.draft_tokens_total.load(Ordering::Relaxed)
        ));

        output.push_str(
            "# HELP infernum_speculative_accepted_tokens_total Total tokens accepted from draft\n",
        );
        output.push_str("# TYPE infernum_speculative_accepted_tokens_total counter\n");
        output.push_str(&format!(
            "infernum_speculative_accepted_tokens_total {}\n",
            self.accepted_tokens_total.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP infernum_speculative_acceptance_rate Token acceptance rate\n");
        output.push_str("# TYPE infernum_speculative_acceptance_rate gauge\n");
        output.push_str(&format!(
            "infernum_speculative_acceptance_rate {:.4}\n",
            self.acceptance_rate()
        ));

        output.push_str("# HELP infernum_speculative_speedup Estimated speedup factor\n");
        output.push_str("# TYPE infernum_speculative_speedup gauge\n");
        output.push_str(&format!(
            "infernum_speculative_speedup {:.2}\n",
            self.estimated_speedup()
        ));

        let draft_time_ms = self.draft_time_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        output.push_str(
            "# HELP infernum_speculative_draft_time_seconds Total draft generation time\n",
        );
        output.push_str("# TYPE infernum_speculative_draft_time_seconds counter\n");
        output.push_str(&format!(
            "infernum_speculative_draft_time_seconds {:.6}\n",
            draft_time_ms / 1000.0
        ));

        let verify_time_ms = self.verify_time_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        output
            .push_str("# HELP infernum_speculative_verify_time_seconds Total verification time\n");
        output.push_str("# TYPE infernum_speculative_verify_time_seconds counter\n");
        output.push_str(&format!(
            "infernum_speculative_verify_time_seconds {:.6}\n",
            verify_time_ms / 1000.0
        ));

        output
    }
}

impl Default for SpeculativeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Header Constants
// ============================================================================

/// Header for enabling speculative decoding.
pub const SPECULATIVE_HEADER: &str = "X-Speculative";

/// Header for speculative draft model override.
pub const SPECULATIVE_DRAFT_HEADER: &str = "X-Speculative-Draft-Model";

/// Header for speculative token count override.
pub const SPECULATIVE_TOKENS_HEADER: &str = "X-Speculative-Tokens";

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_mode_should_speculate() {
        assert!(!SpeculativeMode::Disabled.should_speculate(0.0));
        assert!(!SpeculativeMode::Disabled.should_speculate(1.0));

        assert!(SpeculativeMode::Enabled.should_speculate(0.0));
        assert!(SpeculativeMode::Enabled.should_speculate(1.0));

        assert!(SpeculativeMode::Auto.should_speculate(0.0));
        assert!(SpeculativeMode::Auto.should_speculate(0.1));
        assert!(!SpeculativeMode::Auto.should_speculate(0.5));
        assert!(!SpeculativeMode::Auto.should_speculate(1.0));
    }

    #[test]
    fn test_speculative_mode_display() {
        assert_eq!(SpeculativeMode::Disabled.to_string(), "disabled");
        assert_eq!(SpeculativeMode::Enabled.to_string(), "enabled");
        assert_eq!(SpeculativeMode::Auto.to_string(), "auto");
    }

    #[test]
    fn test_speculative_config_default() {
        let config = SpeculativeConfig::default();

        assert!(config.draft_model.is_empty());
        assert_eq!(config.num_tokens, 5);
        assert!((config.acceptance_threshold - 0.9).abs() < 0.001);
        assert_eq!(config.mode, SpeculativeMode::Disabled);
        assert!(!config.adaptive);
    }

    #[test]
    fn test_speculative_config_builder() {
        let config = SpeculativeConfig::new("llama-1b")
            .with_num_tokens(7)
            .with_acceptance_threshold(0.85)
            .with_mode(SpeculativeMode::Auto)
            .with_adaptive(true);

        assert_eq!(config.draft_model, "llama-1b");
        assert_eq!(config.num_tokens, 7);
        assert!((config.acceptance_threshold - 0.85).abs() < 0.001);
        assert_eq!(config.mode, SpeculativeMode::Auto);
        assert!(config.adaptive);
    }

    #[test]
    fn test_speculative_config_validate_ok() {
        let config = SpeculativeConfig::new("llama-1b");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_speculative_config_validate_no_draft() {
        let config = SpeculativeConfig::default().with_mode(SpeculativeMode::Enabled);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_speculative_config_validate_zero_tokens() {
        let mut config = SpeculativeConfig::new("llama-1b");
        config.num_tokens = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_speculative_config_validate_too_many_tokens() {
        let mut config = SpeculativeConfig::new("llama-1b");
        config.num_tokens = 20;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_speculative_params_default() {
        let params = SpeculativeParams::default();
        assert!(!params.enabled);
        assert!(params.draft_model.is_none());
        assert!(params.num_tokens.is_none());
    }

    #[test]
    fn test_speculative_params_enabled() {
        let params = SpeculativeParams::enabled();
        assert!(params.enabled);
    }

    #[test]
    fn test_speculative_params_builder() {
        let params = SpeculativeParams::enabled()
            .with_draft_model("llama-1b")
            .with_num_tokens(3);

        assert!(params.enabled);
        assert_eq!(params.draft_model, Some("llama-1b".to_string()));
        assert_eq!(params.num_tokens, Some(3));
    }

    #[test]
    fn test_draft_token_new() {
        let token = DraftToken::new(42, 0.95, 5);

        assert_eq!(token.token_id, 42);
        assert!((token.draft_prob - 0.95).abs() < 0.001);
        assert_eq!(token.position, 5);
    }

    #[test]
    fn test_verification_result_acceptance_rate() {
        let result = VerificationResult {
            accepted: 3,
            rejected: 1,
            next_token: Some(42),
            main_probs: vec![0.9, 0.85, 0.8, 0.3],
            eos_reached: false,
            duration: Duration::from_millis(10),
        };

        assert!((result.acceptance_rate() - 0.75).abs() < 0.001);
        assert_eq!(result.total_verified(), 4);
    }

    #[test]
    fn test_verification_result_empty() {
        let result = VerificationResult {
            accepted: 0,
            rejected: 0,
            next_token: None,
            main_probs: vec![],
            eos_reached: false,
            duration: Duration::ZERO,
        };

        assert_eq!(result.acceptance_rate(), 0.0);
        assert_eq!(result.total_verified(), 0);
    }

    #[test]
    fn test_speculative_stats_default() {
        let stats = SpeculativeStats::default();

        assert_eq!(stats.rounds, 0);
        assert_eq!(stats.draft_tokens, 0);
        assert_eq!(stats.accepted_tokens, 0);
        assert_eq!(stats.acceptance_rate(), 0.0);
    }

    #[test]
    fn test_speculative_stats_acceptance_rate() {
        let mut stats = SpeculativeStats::default();
        stats.draft_tokens = 100;
        stats.accepted_tokens = 75;

        assert!((stats.acceptance_rate() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_speculative_stats_avg_accepted() {
        let mut stats = SpeculativeStats::default();
        stats.rounds = 10;
        stats.accepted_tokens = 35;

        assert!((stats.avg_accepted_per_round() - 3.5).abs() < 0.001);
    }

    #[test]
    fn test_speculative_stats_estimated_speedup() {
        let mut stats = SpeculativeStats::default();
        stats.rounds = 10;
        stats.accepted_tokens = 30;
        // Total tokens = 30 accepted + 10 resampled = 40
        // Speedup = 40 / 10 = 4x
        assert!((stats.estimated_speedup() - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_speculative_state_display() {
        assert_eq!(SpeculativeState::Disabled.to_string(), "disabled");
        assert_eq!(SpeculativeState::Drafting.to_string(), "drafting");
        assert_eq!(SpeculativeState::Verifying.to_string(), "verifying");
        assert_eq!(SpeculativeState::Queued.to_string(), "queued");
        assert_eq!(SpeculativeState::Completed.to_string(), "completed");
        assert_eq!(SpeculativeState::Failed.to_string(), "failed");
    }

    #[test]
    fn test_speculative_request_new() {
        let request =
            SpeculativeRequest::new("req-1", "llama-8b", "llama-1b", vec![1, 2, 3], 100, 2);

        assert_eq!(request.request_id, "req-1");
        assert_eq!(request.main_model, "llama-8b");
        assert_eq!(request.draft_model, "llama-1b");
        assert_eq!(request.prompt_tokens, vec![1, 2, 3]);
        assert_eq!(request.max_tokens, 100);
        assert_eq!(request.eos_token, 2);
        assert_eq!(request.state, SpeculativeState::Queued);
    }

    #[test]
    fn test_speculative_request_is_complete() {
        let mut request =
            SpeculativeRequest::new("req-1", "llama-8b", "llama-1b", vec![1, 2, 3], 10, 2);

        assert!(!request.is_complete());

        request.state = SpeculativeState::Completed;
        assert!(request.is_complete());

        request.state = SpeculativeState::Queued;
        request.generated_tokens = vec![1; 10];
        assert!(request.is_complete());
    }

    #[test]
    fn test_speculative_request_record_round() {
        let mut request =
            SpeculativeRequest::new("req-1", "llama-8b", "llama-1b", vec![1, 2, 3], 100, 2);

        let result = VerificationResult {
            accepted: 3,
            rejected: 2,
            next_token: Some(42),
            main_probs: vec![0.9, 0.85, 0.8, 0.3, 0.2],
            eos_reached: false,
            duration: Duration::from_millis(10),
        };

        request.record_round(&result, Duration::from_millis(5));

        assert_eq!(request.stats.rounds, 1);
        assert_eq!(request.stats.draft_tokens, 5);
        assert_eq!(request.stats.accepted_tokens, 3);
        assert_eq!(request.stats.rejected_tokens, 2);
    }

    #[test]
    fn test_speculative_error_display() {
        assert_eq!(
            SpeculativeError::Disabled.to_string(),
            "Speculative decoding is disabled"
        );

        let err = SpeculativeError::Configuration("test".to_string());
        assert!(err.to_string().contains("Configuration error"));

        let err = SpeculativeError::DraftModel("not found".to_string());
        assert!(err.to_string().contains("Draft model error"));
    }

    #[test]
    fn test_speculative_metrics_new() {
        let metrics = SpeculativeMetrics::new();

        assert_eq!(metrics.requests_total(), 0);
        assert_eq!(metrics.rounds_total(), 0);
        assert_eq!(metrics.acceptance_rate(), 0.0);
        assert_eq!(metrics.estimated_speedup(), 1.0);
    }

    #[test]
    fn test_speculative_metrics_record_round() {
        let metrics = SpeculativeMetrics::new();

        let result = VerificationResult {
            accepted: 4,
            rejected: 1,
            next_token: Some(42),
            main_probs: vec![0.9, 0.85, 0.8, 0.7, 0.3],
            eos_reached: false,
            duration: Duration::from_millis(10),
        };

        metrics.record_round(&result, Duration::from_millis(5));

        assert_eq!(metrics.rounds_total(), 1);
        assert_eq!(metrics.draft_tokens_total.load(Ordering::Relaxed), 5);
        assert_eq!(metrics.accepted_tokens_total.load(Ordering::Relaxed), 4);
        assert!((metrics.acceptance_rate() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_speculative_metrics_prometheus() {
        let metrics = SpeculativeMetrics::new();

        let result = VerificationResult {
            accepted: 4,
            rejected: 1,
            next_token: Some(42),
            main_probs: vec![],
            eos_reached: false,
            duration: Duration::from_millis(10),
        };

        metrics.record_round(&result, Duration::from_millis(5));

        let output = metrics.prometheus();

        assert!(output.contains("infernum_speculative_requests_total"));
        assert!(output.contains("infernum_speculative_rounds_total 1"));
        assert!(output.contains("infernum_speculative_acceptance_rate"));
        assert!(output.contains("infernum_speculative_speedup"));
    }

    #[test]
    fn test_speculative_scheduler_new() {
        use crate::batching::BatchConfig;

        let batch_scheduler = Arc::new(BatchScheduler::new(BatchConfig::default()));
        let config = SpeculativeConfig::new("llama-1b");
        let scheduler = SpeculativeScheduler::new(config, batch_scheduler);

        assert_eq!(scheduler.current_tokens(), 5);
        assert_eq!(scheduler.active_count(), 0);
    }

    #[test]
    fn test_speculative_scheduler_should_speculate() {
        use crate::batching::BatchConfig;

        let batch_scheduler = Arc::new(BatchScheduler::new(BatchConfig::default()));

        let config = SpeculativeConfig::new("llama-1b").with_mode(SpeculativeMode::Auto);
        let scheduler = SpeculativeScheduler::new(config, batch_scheduler);

        assert!(scheduler.should_speculate(0.0));
        assert!(!scheduler.should_speculate(0.5));
    }

    #[test]
    fn test_speculative_scheduler_adaptive_update() {
        use crate::batching::BatchConfig;

        let batch_scheduler = Arc::new(BatchScheduler::new(BatchConfig::default()));

        let mut config = SpeculativeConfig::new("llama-1b");
        config.adaptive = true;
        config.num_tokens = 5;
        config.min_tokens = 2;
        config.max_tokens = 8;
        config.target_acceptance_rate = 0.7;

        let scheduler = SpeculativeScheduler::new(config, batch_scheduler);

        // High acceptance - should increase
        scheduler.update_adaptive(0.9);
        assert_eq!(scheduler.current_tokens(), 6);

        // Low acceptance - should decrease
        scheduler.update_adaptive(0.5);
        assert_eq!(scheduler.current_tokens(), 5);
    }

    #[test]
    fn test_speculative_scheduler_submit_disabled() {
        use crate::batching::BatchConfig;

        let batch_scheduler = Arc::new(BatchScheduler::new(BatchConfig::default()));
        let config = SpeculativeConfig::default(); // mode = Disabled
        let scheduler = SpeculativeScheduler::new(config, batch_scheduler);

        let request =
            SpeculativeRequest::new("req-1", "llama-8b", "llama-1b", vec![1, 2, 3], 100, 2);

        let result = scheduler.submit(request);
        assert!(matches!(result, Err(SpeculativeError::Disabled)));
    }

    #[test]
    fn test_header_constants() {
        assert_eq!(SPECULATIVE_HEADER, "X-Speculative");
        assert_eq!(SPECULATIVE_DRAFT_HEADER, "X-Speculative-Draft-Model");
        assert_eq!(SPECULATIVE_TOKENS_HEADER, "X-Speculative-Tokens");
    }
}
