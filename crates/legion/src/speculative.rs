//! # Speculative Legion Decoding
//!
//! Multi-agent speculative decoding with wave interference selection.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    SPECULATIVE LEGION                                │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                      │
//! │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
//! │   │ Draft Agent │  │ Draft Agent │  │ Draft Agent │                │
//! │   │ (Strategic) │  │ (Tactical)  │  │(Operational)│                │
//! │   │  Qwen-0.5B  │  │  Qwen-0.5B  │  │  Qwen-0.5B  │                │
//! │   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                │
//! │          │                │                │                        │
//! │          ▼                ▼                ▼                        │
//! │   ┌────────────────────────────────────────────────────────────┐   │
//! │   │                    LEGION FIELD                             │   │
//! │   │  Draft tokens superimpose → interference pattern emerges    │   │
//! │   │                                                             │   │
//! │   │  "The quick" ─┬─ "brown fox" ─── high coherence (0.9)      │   │
//! │   │               └─ "red car"   ─── low coherence (0.3)       │   │
//! │   └────────────────────────────────────────────────────────────┘   │
//! │                              │                                      │
//! │                              ▼                                      │
//! │   ┌────────────────────────────────────────────────────────────┐   │
//! │   │              VERIFICATION AGENT (Reflective)                │   │
//! │   │                      Qwen-7B                                │   │
//! │   │                                                             │   │
//! │   │  Verify top-K coherent paths in parallel                   │   │
//! │   │  Accept: "The quick brown fox" (draft matches verify)      │   │
//! │   │  Reject: Continue from last accepted token                 │   │
//! │   └────────────────────────────────────────────────────────────┘   │
//! │                                                                      │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Concepts
//!
//! - **Draft Agents**: Small models running in parallel, each emphasizing different
//!   frequency bands (Strategic, Tactical, Operational)
//! - **Legion Field**: Superposition substrate where drafts interfere to reveal consensus
//! - **Verification**: Large model validates top-K coherent paths
//! - **Coherence**: Wave interference metric indicating agreement between agents
//!
//! ## Example
//!
//! ```ignore
//! use legion::speculative::{SpeculativeLegion, SpeculativeLegionConfig};
//!
//! let config = SpeculativeLegionConfig {
//!     draft_agent_count: 3,
//!     lookahead: 4,
//!     coherence_threshold: 0.6,
//!     max_verify_paths: 3,
//!     ..Default::default()
//! };
//!
//! let legion = SpeculativeLegion::new(config);
//!
//! // Generate drafts from multiple perspectives
//! let drafts = legion.draft("The quick brown").await;
//!
//! // Superimpose and find coherent paths
//! let paths = legion.rank_paths(drafts);
//!
//! // Verify top paths against oracle
//! let result = legion.verify(paths).await;
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};

use crate::field::{FieldConfig, LegionField, LegionPattern, Resonance};
use crate::quality::FrequencyBand;

/// Token ID type (matches common tokenizer output).
pub type TokenId = u32;

/// Configuration for speculative Legion decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeLegionConfig {
    /// Number of draft agents.
    pub draft_agent_count: usize,

    /// Lookahead tokens per draft agent.
    pub lookahead: usize,

    /// Minimum coherence for path consideration.
    pub coherence_threshold: f32,

    /// Maximum paths to verify.
    pub max_verify_paths: usize,

    /// Quality curve for acceptance prediction.
    pub quality_curve: QualityCurve,

    /// Adaptive lookahead based on acceptance rate.
    pub adaptive_lookahead: bool,

    /// Minimum lookahead (for adaptive mode).
    pub min_lookahead: usize,

    /// Maximum lookahead (for adaptive mode).
    pub max_lookahead: usize,

    /// Target acceptance rate for adaptive lookahead.
    pub target_acceptance_rate: f32,
}

impl Default for SpeculativeLegionConfig {
    fn default() -> Self {
        Self {
            draft_agent_count: 3,
            lookahead: 4,
            coherence_threshold: 0.6,
            max_verify_paths: 3,
            quality_curve: QualityCurve::Spectral,
            adaptive_lookahead: true,
            min_lookahead: 2,
            max_lookahead: 8,
            target_acceptance_rate: 0.7,
        }
    }
}

impl SpeculativeLegionConfig {
    /// Creates a configuration optimized for high throughput.
    pub fn high_throughput() -> Self {
        Self {
            draft_agent_count: 4,
            lookahead: 6,
            coherence_threshold: 0.5,
            max_verify_paths: 4,
            adaptive_lookahead: true,
            min_lookahead: 4,
            max_lookahead: 12,
            target_acceptance_rate: 0.6,
            ..Default::default()
        }
    }

    /// Creates a configuration optimized for low latency.
    pub fn low_latency() -> Self {
        Self {
            draft_agent_count: 2,
            lookahead: 3,
            coherence_threshold: 0.7,
            max_verify_paths: 2,
            adaptive_lookahead: false,
            ..Default::default()
        }
    }
}

/// Quality curve for speculative decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityCurve {
    /// Spectral (holographic) encoding - 60% quality from first fragment.
    Spectral,
    /// LRDF (low-rank decomposition) - faster convergence.
    Lrdf,
    /// Linear scaling.
    Linear,
}

impl QualityCurve {
    /// Predicts quality at a given fraction (k/n).
    pub fn quality_at(&self, fraction: f32) -> f32 {
        match self {
            Self::Spectral => {
                // Q(k/n) = 0.60 + 0.30*(k/n) + 0.08*(k/n)² + 0.02*(k/n)³
                0.60 + 0.30 * fraction + 0.08 * fraction.powi(2) + 0.02 * fraction.powi(3)
            }
            Self::Lrdf => {
                // Q(k/n) = 0.30 + 0.50*(k/n) + 0.15*(k/n)² + 0.05*(k/n)³
                0.30 + 0.50 * fraction + 0.15 * fraction.powi(2) + 0.05 * fraction.powi(3)
            }
            Self::Linear => fraction,
        }
    }
}

/// A drafted token sequence from one agent.
#[derive(Debug, Clone)]
pub struct DraftSequence {
    /// Agent that produced this draft.
    pub agent_id: String,

    /// Frequency band of the agent.
    pub band: FrequencyBand,

    /// Drafted tokens.
    pub tokens: Vec<TokenId>,

    /// Log probabilities from draft model.
    pub log_probs: Vec<f32>,

    /// Encoded pattern for field superposition.
    pub pattern: LegionPattern,

    /// Generation timestamp.
    pub timestamp: Instant,
}

impl DraftSequence {
    /// Creates a new draft sequence.
    pub fn new(
        agent_id: impl Into<String>,
        band: FrequencyBand,
        tokens: Vec<TokenId>,
        log_probs: Vec<f32>,
    ) -> Self {
        // Encode tokens as spectral pattern
        let pattern = Self::encode_tokens(&tokens, &log_probs);

        Self {
            agent_id: agent_id.into(),
            band,
            tokens,
            log_probs,
            pattern,
            timestamp: Instant::now(),
        }
    }

    /// Encodes tokens and log probs into a Legion pattern.
    fn encode_tokens(tokens: &[TokenId], log_probs: &[f32]) -> LegionPattern {
        // Convert tokens to floating point representation
        // Each token contributes to the pattern based on its position and probability
        const PATTERN_SIZE: usize = 64;
        let mut coefficients = vec![0.0f32; PATTERN_SIZE * PATTERN_SIZE];

        for (i, (token, log_prob)) in tokens.iter().zip(log_probs.iter()).enumerate() {
            // Position in pattern based on token value
            let pos = *token as usize % coefficients.len();

            // Weight by log probability (convert to linear probability)
            let prob = log_prob.exp().min(1.0);

            // Add to pattern with position-based decay
            let decay = 1.0 / (1.0 + i as f32 * 0.1);
            coefficients[pos] += prob * decay;
        }

        LegionPattern::from_coefficients(coefficients, PATTERN_SIZE, PATTERN_SIZE)
    }

    /// Returns the length of the draft.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns true if the draft is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Calculates average log probability.
    pub fn avg_log_prob(&self) -> f32 {
        if self.log_probs.is_empty() {
            0.0
        } else {
            self.log_probs.iter().sum::<f32>() / self.log_probs.len() as f32
        }
    }
}

/// A ranked path after interference analysis.
#[derive(Debug, Clone)]
pub struct RankedPath {
    /// Index of the original draft.
    pub draft_index: usize,

    /// The draft sequence.
    pub draft: DraftSequence,

    /// Coherence score from interference.
    pub coherence: f32,

    /// Resonance analysis result.
    pub resonance: Resonance,
}

/// Result of speculative verification.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Accepted token prefix.
    pub accepted_tokens: Vec<TokenId>,

    /// Number of accepted tokens.
    pub accepted_count: usize,

    /// Which draft path was accepted.
    pub accepted_path: Option<usize>,

    /// Coherence score of accepted path.
    pub coherence: f32,

    /// Verification statistics.
    pub stats: VerificationStats,
}

impl VerificationResult {
    /// Creates an empty result (no tokens accepted).
    pub fn empty() -> Self {
        Self {
            accepted_tokens: Vec::new(),
            accepted_count: 0,
            accepted_path: None,
            coherence: 0.0,
            stats: VerificationStats::default(),
        }
    }

    /// Returns true if any tokens were accepted.
    pub fn has_accepted(&self) -> bool {
        self.accepted_count > 0
    }
}

/// Statistics from verification.
#[derive(Debug, Clone, Default)]
pub struct VerificationStats {
    /// Total drafts evaluated.
    pub drafts_evaluated: usize,

    /// Total tokens in drafts.
    pub total_draft_tokens: usize,

    /// Tokens accepted.
    pub tokens_accepted: usize,

    /// Tokens rejected.
    pub tokens_rejected: usize,

    /// Verification time.
    pub verification_time: Duration,
}

impl VerificationStats {
    /// Returns acceptance rate.
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_draft_tokens == 0 {
            0.0
        } else {
            self.tokens_accepted as f32 / self.total_draft_tokens as f32
        }
    }
}

/// Cumulative statistics for speculative decoding session.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// Total drafts generated.
    pub drafts_generated: usize,

    /// Total tokens drafted.
    pub tokens_drafted: usize,

    /// Total tokens verified.
    pub tokens_verified: usize,

    /// Total tokens accepted.
    pub tokens_accepted: usize,

    /// Total verification calls.
    pub verification_calls: usize,

    /// Total verification time.
    pub total_verification_time: Duration,

    /// Current lookahead (if adaptive).
    pub current_lookahead: usize,

    /// Running acceptance rate.
    pub acceptance_rate: f32,
}

impl SpeculativeStats {
    /// Updates stats with a verification result.
    pub fn update(&mut self, result: &VerificationResult) {
        self.tokens_verified += result.stats.total_draft_tokens;
        self.tokens_accepted += result.stats.tokens_accepted;
        self.verification_calls += 1;
        self.total_verification_time += result.stats.verification_time;

        // Update running acceptance rate with exponential moving average
        let new_rate = result.stats.acceptance_rate();
        self.acceptance_rate = 0.9 * self.acceptance_rate + 0.1 * new_rate;
    }

    /// Updates draft generation stats.
    pub fn record_drafts(&mut self, drafts: &[DraftSequence]) {
        self.drafts_generated += drafts.len();
        self.tokens_drafted += drafts.iter().map(|d| d.len()).sum::<usize>();
    }
}

/// Generates draft sequences for a specific frequency band.
pub struct DraftGenerator {
    /// Generator ID.
    pub id: String,

    /// Frequency band this generator operates at.
    pub band: FrequencyBand,

    /// Configuration.
    pub config: DraftGeneratorConfig,
}

/// Configuration for a draft generator.
#[derive(Debug, Clone)]
pub struct DraftGeneratorConfig {
    /// Default lookahead.
    pub lookahead: usize,

    /// Temperature for sampling.
    pub temperature: f32,

    /// Top-p for nucleus sampling.
    pub top_p: f32,
}

impl Default for DraftGeneratorConfig {
    fn default() -> Self {
        Self {
            lookahead: 4,
            temperature: 0.8,
            top_p: 0.9,
        }
    }
}

impl DraftGenerator {
    /// Creates a new draft generator.
    pub fn new(id: impl Into<String>, band: FrequencyBand) -> Self {
        Self {
            id: id.into(),
            band,
            config: DraftGeneratorConfig::default(),
        }
    }

    /// Creates a new draft generator with config.
    pub fn with_config(id: impl Into<String>, band: FrequencyBand, config: DraftGeneratorConfig) -> Self {
        Self {
            id: id.into(),
            band,
            config,
        }
    }

    /// Generates a draft sequence.
    ///
    /// This is a placeholder that should be wired to actual model inference.
    /// Returns simulated drafts for testing.
    pub fn generate(
        &self,
        _context_tokens: &[TokenId],
        lookahead: usize,
    ) -> DraftSequence {
        // Placeholder: In real implementation, this calls the draft model
        // For now, return a simulated draft
        let tokens: Vec<TokenId> = (0..lookahead).map(|i| i as TokenId).collect();
        let log_probs: Vec<f32> = (0..lookahead).map(|_| -0.5).collect();

        DraftSequence::new(&self.id, self.band, tokens, log_probs)
    }
}

/// Pool of draft generators organized by frequency band.
pub struct DraftPool {
    /// Generators by band.
    generators: HashMap<FrequencyBand, DraftGenerator>,

    /// Configuration (retained for future use in adaptive draft generation).
    _config: SpeculativeLegionConfig,
}

impl DraftPool {
    /// Creates a new draft pool.
    pub fn new(config: SpeculativeLegionConfig) -> Self {
        let mut generators = HashMap::new();

        // Create generators for each band based on config
        let bands = Self::select_bands(config.draft_agent_count);
        for (i, band) in bands.into_iter().enumerate() {
            let generator = DraftGenerator::new(format!("draft-{}", i), band);
            generators.insert(band, generator);
        }

        Self { generators, _config: config }
    }

    /// Selects frequency bands for the given number of agents.
    fn select_bands(count: usize) -> Vec<FrequencyBand> {
        match count {
            1 => vec![FrequencyBand::Operational],
            2 => vec![FrequencyBand::Strategic, FrequencyBand::Operational],
            3 => vec![FrequencyBand::Strategic, FrequencyBand::Tactical, FrequencyBand::Operational],
            4 => vec![
                FrequencyBand::Strategic,
                FrequencyBand::Tactical,
                FrequencyBand::Operational,
                FrequencyBand::Verification,
            ],
            _ => {
                // For 5+, include all standard bands
                vec![
                    FrequencyBand::Anima,
                    FrequencyBand::Strategic,
                    FrequencyBand::Tactical,
                    FrequencyBand::Operational,
                    FrequencyBand::Verification,
                    FrequencyBand::Reflective,
                ].into_iter().take(count).collect()
            }
        }
    }

    /// Generates drafts from all generators.
    pub fn generate_all(&self, context_tokens: &[TokenId], lookahead: usize) -> Vec<DraftSequence> {
        self.generators
            .values()
            .map(|gen| gen.generate(context_tokens, lookahead))
            .collect()
    }

    /// Returns the number of generators.
    pub fn generator_count(&self) -> usize {
        self.generators.len()
    }
}

/// The main Speculative Legion coordinator.
pub struct SpeculativeLegion {
    /// Configuration.
    config: SpeculativeLegionConfig,

    /// Draft pool.
    draft_pool: DraftPool,

    /// Legion field for interference.
    field: Arc<RwLock<LegionField>>,

    /// Statistics.
    stats: Arc<Mutex<SpeculativeStats>>,
}

impl SpeculativeLegion {
    /// Creates a new speculative Legion.
    pub fn new(config: SpeculativeLegionConfig) -> Self {
        let draft_pool = DraftPool::new(config.clone());
        let field = Arc::new(RwLock::new(LegionField::new(FieldConfig::default())));

        let mut stats = SpeculativeStats::default();
        stats.current_lookahead = config.lookahead;

        Self {
            config,
            draft_pool,
            field,
            stats: Arc::new(Mutex::new(stats)),
        }
    }

    /// Returns the configuration.
    pub fn config(&self) -> &SpeculativeLegionConfig {
        &self.config
    }

    /// Returns current statistics.
    pub fn stats(&self) -> SpeculativeStats {
        self.stats.lock().clone()
    }

    /// Generates drafts from all agents.
    pub fn draft(&self, context_tokens: &[TokenId]) -> Vec<DraftSequence> {
        let lookahead = self.current_lookahead();
        let drafts = self.draft_pool.generate_all(context_tokens, lookahead);

        // Update stats
        self.stats.lock().record_drafts(&drafts);

        drafts
    }

    /// Returns current lookahead (adaptive or fixed).
    fn current_lookahead(&self) -> usize {
        if self.config.adaptive_lookahead {
            self.stats.lock().current_lookahead
        } else {
            self.config.lookahead
        }
    }

    /// Ranks draft paths by coherence using field interference.
    pub fn rank_paths(&self, drafts: Vec<DraftSequence>) -> Vec<RankedPath> {
        let field = self.field.write();

        // Clear field for new ranking
        field.decay();

        // Superimpose all drafts into field
        for draft in &drafts {
            let weight = draft.band.emphasis() * draft.avg_log_prob().exp();
            let mut weighted_pattern = draft.pattern.clone();
            for coeff in &mut weighted_pattern.coefficients {
                *coeff *= weight;
            }
            field.superimpose(&weighted_pattern, draft.band);
        }

        // Compute coherence for each draft
        let mut ranked: Vec<RankedPath> = drafts
            .into_iter()
            .enumerate()
            .map(|(idx, draft)| {
                let resonance = field.interfere(&draft.pattern);
                let coherence = resonance.confidence();

                RankedPath {
                    draft_index: idx,
                    draft,
                    coherence,
                    resonance,
                }
            })
            .collect();

        // Sort by coherence (descending)
        ranked.sort_by(|a, b| b.coherence.partial_cmp(&a.coherence).unwrap_or(std::cmp::Ordering::Equal));

        // Filter by threshold
        ranked.retain(|p| p.coherence >= self.config.coherence_threshold);

        // Take top K
        ranked.truncate(self.config.max_verify_paths);

        ranked
    }

    /// Verifies ranked paths against oracle model.
    ///
    /// This is a placeholder that should be wired to actual verification.
    /// Returns simulated verification for testing.
    pub fn verify(
        &self,
        ranked_paths: Vec<RankedPath>,
        _oracle_tokens: &[TokenId],
    ) -> VerificationResult {
        let start = Instant::now();

        if ranked_paths.is_empty() {
            return VerificationResult::empty();
        }

        // Get the best path
        let best = &ranked_paths[0];

        // Placeholder: In real implementation, compare against oracle output
        // For now, accept based on coherence
        let acceptance_probability = best.coherence;
        let accept_count = (best.draft.len() as f32 * acceptance_probability) as usize;

        let accepted_tokens = best.draft.tokens[..accept_count].to_vec();

        let stats = VerificationStats {
            drafts_evaluated: ranked_paths.len(),
            total_draft_tokens: ranked_paths.iter().map(|p| p.draft.len()).sum(),
            tokens_accepted: accept_count,
            tokens_rejected: best.draft.len() - accept_count,
            verification_time: start.elapsed(),
        };

        let result = VerificationResult {
            accepted_tokens: accepted_tokens.clone(),
            accepted_count: accept_count,
            accepted_path: Some(0),
            coherence: best.coherence,
            stats,
        };

        // Update session stats
        let mut session_stats = self.stats.lock();
        session_stats.update(&result);

        // Adjust lookahead if adaptive
        if self.config.adaptive_lookahead {
            self.adjust_lookahead(&mut session_stats);
        }

        result
    }

    /// Adjusts lookahead based on acceptance rate.
    fn adjust_lookahead(&self, stats: &mut SpeculativeStats) {
        let current = stats.current_lookahead;
        let rate = stats.acceptance_rate;
        let target = self.config.target_acceptance_rate;

        let new_lookahead = if rate > target + 0.1 {
            // High acceptance - increase lookahead
            (current + 1).min(self.config.max_lookahead)
        } else if rate < target - 0.1 {
            // Low acceptance - decrease lookahead
            (current - 1).max(self.config.min_lookahead)
        } else {
            current
        };

        stats.current_lookahead = new_lookahead;
    }

    /// Full speculative generation step.
    ///
    /// 1. Generate drafts from all agents
    /// 2. Rank paths by coherence
    /// 3. Verify top paths
    /// 4. Return accepted tokens
    pub fn speculative_step(
        &self,
        context_tokens: &[TokenId],
        oracle_tokens: &[TokenId],
    ) -> VerificationResult {
        // Generate drafts
        let drafts = self.draft(context_tokens);

        // Rank by coherence
        let ranked = self.rank_paths(drafts);

        // Verify and return
        self.verify(ranked, oracle_tokens)
    }

    /// Checks if drafts show sufficient diversity.
    pub fn drafts_are_diverse(&self, drafts: &[DraftSequence]) -> bool {
        if drafts.len() < 2 {
            return false;
        }

        // Calculate pairwise similarity
        let mut total_similarity = 0.0;
        let mut comparisons = 0;

        for i in 0..drafts.len() {
            for j in (i + 1)..drafts.len() {
                total_similarity += drafts[i].pattern.similarity(&drafts[j].pattern);
                comparisons += 1;
            }
        }

        let avg_similarity = total_similarity / comparisons as f32;

        // Diverse if average similarity is below threshold
        avg_similarity < 0.8
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = SpeculativeLegionConfig::default();
        assert_eq!(config.draft_agent_count, 3);
        assert_eq!(config.lookahead, 4);
        assert!((config.coherence_threshold - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_config_high_throughput() {
        let config = SpeculativeLegionConfig::high_throughput();
        assert_eq!(config.draft_agent_count, 4);
        assert_eq!(config.lookahead, 6);
        assert!(config.max_lookahead > config.lookahead);
    }

    #[test]
    fn test_config_low_latency() {
        let config = SpeculativeLegionConfig::low_latency();
        assert_eq!(config.draft_agent_count, 2);
        assert_eq!(config.lookahead, 3);
        assert!(!config.adaptive_lookahead);
    }

    #[test]
    fn test_quality_curve_spectral() {
        let curve = QualityCurve::Spectral;

        // First fragment should give ~60% quality
        let q0 = curve.quality_at(0.0);
        assert!((q0 - 0.60).abs() < 0.01);

        // Full quality at 1.0
        let q1 = curve.quality_at(1.0);
        assert!((q1 - 1.0).abs() < 0.01);

        // Monotonically increasing
        let q_half = curve.quality_at(0.5);
        assert!(q_half > q0);
        assert!(q_half < q1);
    }

    #[test]
    fn test_draft_sequence_creation() {
        let tokens = vec![1, 2, 3, 4];
        let log_probs = vec![-0.5, -0.6, -0.7, -0.8];

        let draft = DraftSequence::new("agent-0", FrequencyBand::Strategic, tokens.clone(), log_probs.clone());

        assert_eq!(draft.agent_id, "agent-0");
        assert_eq!(draft.band, FrequencyBand::Strategic);
        assert_eq!(draft.tokens, tokens);
        assert_eq!(draft.len(), 4);
        assert!(!draft.is_empty());
    }

    #[test]
    fn test_draft_sequence_avg_log_prob() {
        let tokens = vec![1, 2, 3, 4];
        let log_probs = vec![-1.0, -2.0, -3.0, -4.0];

        let draft = DraftSequence::new("agent-0", FrequencyBand::Operational, tokens, log_probs);

        let avg = draft.avg_log_prob();
        assert!((avg - (-2.5)).abs() < 0.001);
    }

    #[test]
    fn test_draft_pool_creation() {
        let config = SpeculativeLegionConfig::default();
        let pool = DraftPool::new(config);

        assert_eq!(pool.generator_count(), 3);
    }

    #[test]
    fn test_draft_pool_generation() {
        let config = SpeculativeLegionConfig::default();
        let pool = DraftPool::new(config);

        let context = vec![100, 200, 300];
        let drafts = pool.generate_all(&context, 4);

        assert_eq!(drafts.len(), 3);
        for draft in &drafts {
            assert_eq!(draft.len(), 4);
        }
    }

    #[test]
    fn test_speculative_legion_creation() {
        let config = SpeculativeLegionConfig::default();
        let legion = SpeculativeLegion::new(config);

        assert_eq!(legion.config().draft_agent_count, 3);
    }

    #[test]
    fn test_speculative_legion_draft() {
        let config = SpeculativeLegionConfig::default();
        let legion = SpeculativeLegion::new(config);

        let context = vec![100, 200, 300];
        let drafts = legion.draft(&context);

        assert_eq!(drafts.len(), 3);

        let stats = legion.stats();
        assert_eq!(stats.drafts_generated, 3);
    }

    #[test]
    fn test_speculative_legion_rank_paths() {
        let config = SpeculativeLegionConfig::default();
        let legion = SpeculativeLegion::new(config);

        let context = vec![100, 200, 300];
        let drafts = legion.draft(&context);

        let ranked = legion.rank_paths(drafts);

        // Should return up to max_verify_paths
        assert!(ranked.len() <= legion.config().max_verify_paths);

        // Should be sorted by coherence (descending)
        for i in 1..ranked.len() {
            assert!(ranked[i - 1].coherence >= ranked[i].coherence);
        }
    }

    #[test]
    fn test_speculative_legion_verify() {
        let config = SpeculativeLegionConfig::default();
        let legion = SpeculativeLegion::new(config);

        let context = vec![100, 200, 300];
        let drafts = legion.draft(&context);
        let ranked = legion.rank_paths(drafts);

        let oracle = vec![1, 2, 3, 4];
        let result = legion.verify(ranked, &oracle);

        assert!(result.stats.drafts_evaluated > 0);
    }

    #[test]
    fn test_speculative_step() {
        let config = SpeculativeLegionConfig::default();
        let legion = SpeculativeLegion::new(config);

        let context = vec![100, 200, 300];
        let oracle = vec![1, 2, 3, 4];

        let result = legion.speculative_step(&context, &oracle);

        assert!(result.stats.drafts_evaluated > 0);
    }

    #[test]
    fn test_verification_stats_acceptance_rate() {
        let stats = VerificationStats {
            drafts_evaluated: 3,
            total_draft_tokens: 10,
            tokens_accepted: 7,
            tokens_rejected: 3,
            verification_time: Duration::from_millis(10),
        };

        let rate = stats.acceptance_rate();
        assert!((rate - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_speculative_stats_update() {
        let mut stats = SpeculativeStats::default();

        let result = VerificationResult {
            accepted_tokens: vec![1, 2, 3],
            accepted_count: 3,
            accepted_path: Some(0),
            coherence: 0.8,
            stats: VerificationStats {
                drafts_evaluated: 3,
                total_draft_tokens: 12,
                tokens_accepted: 9,
                tokens_rejected: 3,
                verification_time: Duration::from_millis(10),
            },
        };

        stats.update(&result);

        assert_eq!(stats.tokens_verified, 12);
        assert_eq!(stats.tokens_accepted, 9);
        assert_eq!(stats.verification_calls, 1);
    }

    #[test]
    fn test_diversity_check() {
        let config = SpeculativeLegionConfig::default();
        let legion = SpeculativeLegion::new(config);

        // Create diverse drafts (different bands)
        let draft1 = DraftSequence::new("a", FrequencyBand::Strategic, vec![1, 2, 3], vec![-0.5, -0.5, -0.5]);
        let draft2 = DraftSequence::new("b", FrequencyBand::Operational, vec![4, 5, 6], vec![-0.5, -0.5, -0.5]);

        let drafts = vec![draft1, draft2];
        let diverse = legion.drafts_are_diverse(&drafts);

        // Should be diverse (different tokens)
        assert!(diverse);
    }

    #[test]
    fn test_empty_verification_result() {
        let result = VerificationResult::empty();

        assert!(result.accepted_tokens.is_empty());
        assert_eq!(result.accepted_count, 0);
        assert!(!result.has_accepted());
    }
}
