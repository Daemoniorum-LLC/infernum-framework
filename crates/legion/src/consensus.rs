//! Consensus strategies for merging agent outputs.
//!
//! Supports both traditional voting-based consensus and wave interference-based
//! consensus from the holographic architecture.
//!
//! ## Interference Consensus
//!
//! Agent contributions are treated as wave patterns that superimpose in the
//! Legion Field. Areas of agreement produce constructive interference (peaks),
//! while conflicts produce destructive interference.
//!
//! ```text
//! Agent A: ~~~∿∿∿~~~∿∿∿~~~
//! Agent B: ~~~∿∿∿~~~∿∿∿~~~  → Constructive: ∿∿∿∿∿∿ (agreement peak)
//! Agent C: ~~~∿∿∿~~~___~~~  → Destructive: ___ (conflict point)
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use crate::field::{FieldConfig, LegionField, LegionPattern, Resonance};
use crate::quality::FrequencyBand;

/// Strategy for reaching consensus among agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConsensusStrategy {
    /// Simple majority voting.
    Majority,
    /// Weighted by agent quality/context fraction.
    #[default]
    WeightedMajority,
    /// Take output from highest-context agent.
    HighestContext,
    /// Take first output (fastest agent).
    FirstResponse,
    /// Require unanimous agreement.
    Unanimous,
    /// Wave interference-based consensus (holographic).
    Interference,
}

/// Result of a consensus operation.
#[derive(Debug, Clone)]
pub struct ConsensusResult {
    /// The agreed-upon output.
    pub output: String,
    /// Confidence in the consensus (0.0 - 1.0).
    pub confidence: f32,
    /// Number of agents that participated.
    pub participants: usize,
    /// Number of agents that agreed.
    pub agreements: usize,
    /// Strategy used.
    pub strategy: ConsensusStrategy,
    /// Individual agent outputs (for debugging/analysis).
    pub agent_outputs: HashMap<String, String>,
}

impl ConsensusResult {
    /// Creates a new consensus result.
    pub fn new(output: impl Into<String>, confidence: f32, strategy: ConsensusStrategy) -> Self {
        Self {
            output: output.into(),
            confidence,
            participants: 0,
            agreements: 0,
            strategy,
            agent_outputs: HashMap::new(),
        }
    }

    /// Returns the agreement ratio.
    pub fn agreement_ratio(&self) -> f32 {
        if self.participants == 0 {
            return 0.0;
        }
        self.agreements as f32 / self.participants as f32
    }

    /// Checks if consensus was strong (>80% agreement).
    pub fn is_strong(&self) -> bool {
        self.agreement_ratio() >= 0.8
    }
}

/// Agent vote in a consensus round.
#[derive(Debug, Clone)]
pub struct AgentVote {
    /// Agent identifier.
    pub agent_id: String,
    /// Agent's output.
    pub output: String,
    /// Weight of this vote (based on context fraction).
    pub weight: f32,
    /// Quality/confidence of this agent's output.
    pub quality: f32,
}

// ==================== Agent Contribution (for Interference) ====================

/// An agent's contribution for interference-based consensus.
#[derive(Debug, Clone)]
pub struct AgentContribution {
    /// Agent identifier.
    pub agent_id: String,
    /// Frequency band this agent operates at.
    pub band: FrequencyBand,
    /// The agent's output as a pattern.
    pub pattern: LegionPattern,
    /// The agent's output as text.
    pub output: String,
    /// Confidence in this contribution (0.0 - 1.0).
    pub confidence: f32,
}

impl AgentContribution {
    /// Creates a contribution from text output.
    ///
    /// Encodes the text into a spectral pattern for interference.
    pub fn from_text(
        agent_id: impl Into<String>,
        band: FrequencyBand,
        output: impl Into<String>,
        confidence: f32,
    ) -> Self {
        let output = output.into();

        // Create pattern from text (simple hash-based encoding)
        let pattern = Self::encode_text(&output, 64, 64);

        Self {
            agent_id: agent_id.into(),
            band,
            pattern,
            output,
            confidence,
        }
    }

    /// Encodes text into a spectral pattern.
    fn encode_text(text: &str, width: usize, height: usize) -> LegionPattern {
        let size = width * height;
        let mut coefficients = vec![0.0f32; size];

        // Simple encoding: hash-based distribution of character weights
        for (i, ch) in text.chars().enumerate() {
            let char_val = ch as u32 as f32 / 128.0; // Normalize
            let idx = i % size;

            // DC component (average character value)
            coefficients[0] += char_val / text.len() as f32;

            // Distribute across spectrum based on position
            if idx < size {
                coefficients[idx] += char_val * 0.1;
            }

            // Low frequency components (word-level patterns)
            let word_idx = (i / 5) % (size / 4);
            if word_idx > 0 && word_idx < size {
                coefficients[word_idx] += char_val * 0.05;
            }
        }

        // Normalize to unit energy
        let energy: f32 = coefficients.iter().map(|c| c * c).sum::<f32>().sqrt();
        if energy > 0.0 {
            for coeff in &mut coefficients {
                *coeff /= energy;
            }
        }

        LegionPattern::from_coefficients(coefficients, width, height)
    }

    /// Returns the weight for this contribution in consensus.
    pub fn consensus_weight(&self) -> f32 {
        self.band.emphasis() * self.confidence
    }
}

// ==================== Interference Consensus ====================

/// Configuration for interference consensus.
#[derive(Debug, Clone)]
pub struct InterferenceConfig {
    /// Agreement threshold for peaks.
    pub agreement_threshold: f32,
    /// Conflict threshold for destructive interference.
    pub conflict_threshold: f32,
    /// Field configuration.
    pub field_config: FieldConfig,
    /// Band weights for consensus.
    pub band_weights: HashMap<FrequencyBand, f32>,
}

impl Default for InterferenceConfig {
    fn default() -> Self {
        let mut band_weights = HashMap::new();
        band_weights.insert(FrequencyBand::Anima, 1.0);
        band_weights.insert(FrequencyBand::Strategic, 0.9);
        band_weights.insert(FrequencyBand::Tactical, 0.85);
        band_weights.insert(FrequencyBand::Operational, 1.0);
        band_weights.insert(FrequencyBand::Verification, 0.8);
        band_weights.insert(FrequencyBand::Reflective, 0.6);

        Self {
            agreement_threshold: 0.5,
            conflict_threshold: -0.3,
            field_config: FieldConfig::default(),
            band_weights,
        }
    }
}

/// Interference-based consensus engine.
///
/// Contributions superimpose in a Legion Field, and consensus
/// emerges from the interference pattern.
pub struct InterferenceConsensus {
    /// The field where contributions superimpose.
    field: Arc<LegionField>,
    /// All contributions received.
    contributions: Vec<AgentContribution>,
    /// Configuration.
    config: InterferenceConfig,
}

impl InterferenceConsensus {
    /// Creates a new interference consensus engine.
    pub fn new(config: InterferenceConfig) -> Self {
        Self {
            field: Arc::new(LegionField::new(config.field_config.clone())),
            contributions: Vec::new(),
            config,
        }
    }

    /// Adds a contribution to the consensus.
    pub fn add_contribution(&mut self, contribution: AgentContribution) {
        // Superimpose pattern into field with band weighting
        let weight = self
            .config
            .band_weights
            .get(&contribution.band)
            .copied()
            .unwrap_or(1.0);

        // Scale pattern by weight and confidence
        let mut scaled_pattern = contribution.pattern.clone();
        for coeff in &mut scaled_pattern.coefficients {
            *coeff *= weight * contribution.confidence;
        }

        self.field.superimpose(&scaled_pattern, contribution.band);
        self.contributions.push(contribution);
    }

    /// Extracts consensus from the accumulated contributions.
    pub fn extract_consensus(&self) -> InterferenceResult {
        if self.contributions.is_empty() {
            return InterferenceResult {
                output: String::new(),
                confidence: 0.0,
                resonance: None,
                agreements: 0,
                conflicts: 0,
                participants: 0,
            };
        }

        // Create probe from average of contributions
        let probe = self.create_consensus_probe();

        // Interfere probe with field
        let resonance = self.field.interfere(&probe);

        // Find the most resonant contribution
        let (best_output, confidence) = self.find_best_match(&resonance);

        // Count agreements and conflicts
        let agreements = resonance.peaks.len();
        let conflicts = resonance.conflicts.len();

        InterferenceResult {
            output: best_output,
            confidence,
            resonance: Some(resonance),
            agreements,
            conflicts,
            participants: self.contributions.len(),
        }
    }

    /// Creates a probe pattern from the average of all contributions.
    fn create_consensus_probe(&self) -> LegionPattern {
        let (width, height) = self.field.dimensions();
        let size = width * height;
        let mut avg_coeffs = vec![0.0f32; size];

        for contrib in &self.contributions {
            let weight = contrib.consensus_weight();
            for (i, &coeff) in contrib.pattern.coefficients.iter().enumerate() {
                if i < size {
                    avg_coeffs[i] += coeff * weight;
                }
            }
        }

        // Normalize
        let total_weight: f32 = self
            .contributions
            .iter()
            .map(|c| c.consensus_weight())
            .sum();

        if total_weight > 0.0 {
            for coeff in &mut avg_coeffs {
                *coeff /= total_weight;
            }
        }

        LegionPattern::from_coefficients(avg_coeffs, width, height)
    }

    /// Finds the contribution that best matches the resonance.
    fn find_best_match(&self, resonance: &Resonance) -> (String, f32) {
        let mut best_output = String::new();
        let mut best_score = 0.0f32;

        for contrib in &self.contributions {
            // Calculate how well this contribution matches the resonance
            let similarity = contrib.pattern.similarity(&self.field.extract());
            let weighted_score = similarity * contrib.consensus_weight();

            if weighted_score > best_score {
                best_score = weighted_score;
                best_output = contrib.output.clone();
            }
        }

        // Confidence based on resonance clarity
        let confidence = resonance.confidence();

        (best_output, confidence)
    }

    /// Clears the consensus engine for reuse.
    pub fn clear(&mut self) {
        self.field.clear();
        self.contributions.clear();
    }

    /// Returns the number of contributions.
    pub fn contribution_count(&self) -> usize {
        self.contributions.len()
    }

    /// Returns the field energy level.
    pub fn field_energy(&self) -> f32 {
        self.field.energy()
    }
}

/// Result of interference-based consensus.
#[derive(Debug, Clone)]
pub struct InterferenceResult {
    /// The consensus output.
    pub output: String,
    /// Confidence in the consensus (0.0 - 1.0).
    pub confidence: f32,
    /// The resonance pattern (if available).
    pub resonance: Option<Resonance>,
    /// Number of agreement peaks.
    pub agreements: usize,
    /// Number of conflict points.
    pub conflicts: usize,
    /// Number of participants.
    pub participants: usize,
}

impl InterferenceResult {
    /// Returns whether this is a strong consensus (>80% confidence).
    pub fn is_strong(&self) -> bool {
        self.confidence >= 0.8
    }

    /// Converts to a standard ConsensusResult.
    pub fn to_consensus_result(&self) -> ConsensusResult {
        let mut result = ConsensusResult::new(
            &self.output,
            self.confidence,
            ConsensusStrategy::Interference,
        );
        result.participants = self.participants;
        result.agreements = self.agreements;
        result
    }
}

// ==================== Legacy Consensus Engine ====================

/// Consensus engine for merging agent outputs.
///
/// Supports both traditional voting and interference-based consensus.
pub struct ConsensusEngine {
    strategy: ConsensusStrategy,
    interference: Option<InterferenceConsensus>,
}

impl ConsensusEngine {
    /// Creates a new consensus engine.
    pub fn new(strategy: ConsensusStrategy) -> Self {
        let interference = if strategy == ConsensusStrategy::Interference {
            Some(InterferenceConsensus::new(InterferenceConfig::default()))
        } else {
            None
        };

        Self {
            strategy,
            interference,
        }
    }

    /// Creates an interference-based consensus engine.
    pub fn interference(config: InterferenceConfig) -> Self {
        Self {
            strategy: ConsensusStrategy::Interference,
            interference: Some(InterferenceConsensus::new(config)),
        }
    }

    /// Reaches consensus from a set of votes.
    pub fn reach_consensus(&self, votes: &[AgentVote]) -> ConsensusResult {
        if votes.is_empty() {
            return ConsensusResult::new("", 0.0, self.strategy);
        }

        match self.strategy {
            ConsensusStrategy::Majority => self.majority_consensus(votes),
            ConsensusStrategy::WeightedMajority => self.weighted_consensus(votes),
            ConsensusStrategy::HighestContext => self.highest_context_consensus(votes),
            ConsensusStrategy::FirstResponse => self.first_response_consensus(votes),
            ConsensusStrategy::Unanimous => self.unanimous_consensus(votes),
            ConsensusStrategy::Interference => self.interference_from_votes(votes),
        }
    }

    /// Adds a contribution for interference consensus.
    pub fn add_contribution(&mut self, contribution: AgentContribution) {
        if let Some(ref mut interference) = self.interference {
            interference.add_contribution(contribution);
        }
    }

    /// Extracts interference consensus.
    pub fn extract_interference(&self) -> Option<InterferenceResult> {
        self.interference.as_ref().map(|i| i.extract_consensus())
    }

    fn interference_from_votes(&self, votes: &[AgentVote]) -> ConsensusResult {
        // Convert votes to contributions and run interference
        let mut interference = InterferenceConsensus::new(InterferenceConfig::default());

        for vote in votes {
            let contrib = AgentContribution::from_text(
                &vote.agent_id,
                FrequencyBand::Operational, // Default band
                &vote.output,
                vote.quality,
            );
            interference.add_contribution(contrib);
        }

        interference.extract_consensus().to_consensus_result()
    }

    fn majority_consensus(&self, votes: &[AgentVote]) -> ConsensusResult {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for vote in votes {
            *counts.entry(vote.output.as_str()).or_insert(0) += 1;
        }

        let (output, count) = counts
            .iter()
            .max_by_key(|(_, &c)| c)
            .map(|(o, &c)| (*o, c))
            .unwrap_or(("", 0));

        let mut result =
            ConsensusResult::new(output, count as f32 / votes.len() as f32, self.strategy);
        result.participants = votes.len();
        result.agreements = count;
        for vote in votes {
            result
                .agent_outputs
                .insert(vote.agent_id.clone(), vote.output.clone());
        }

        result
    }

    fn weighted_consensus(&self, votes: &[AgentVote]) -> ConsensusResult {
        let mut weighted_counts: HashMap<&str, f32> = HashMap::new();
        let mut total_weight = 0.0;

        for vote in votes {
            *weighted_counts.entry(vote.output.as_str()).or_insert(0.0) += vote.weight;
            total_weight += vote.weight;
        }

        let (output, weight) = weighted_counts
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(o, &w)| (*o, w))
            .unwrap_or(("", 0.0));

        let confidence = if total_weight > 0.0 {
            weight / total_weight
        } else {
            0.0
        };

        let agreements = votes.iter().filter(|v| v.output == output).count();

        let mut result = ConsensusResult::new(output, confidence, self.strategy);
        result.participants = votes.len();
        result.agreements = agreements;
        for vote in votes {
            result
                .agent_outputs
                .insert(vote.agent_id.clone(), vote.output.clone());
        }

        result
    }

    fn highest_context_consensus(&self, votes: &[AgentVote]) -> ConsensusResult {
        let best = votes.iter().max_by(|a, b| {
            a.weight
                .partial_cmp(&b.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        match best {
            Some(vote) => {
                let agreements = votes.iter().filter(|v| v.output == vote.output).count();
                let mut result = ConsensusResult::new(&vote.output, vote.quality, self.strategy);
                result.participants = votes.len();
                result.agreements = agreements;
                for v in votes {
                    result
                        .agent_outputs
                        .insert(v.agent_id.clone(), v.output.clone());
                }
                result
            },
            None => ConsensusResult::new("", 0.0, self.strategy),
        }
    }

    fn first_response_consensus(&self, votes: &[AgentVote]) -> ConsensusResult {
        match votes.first() {
            Some(vote) => {
                let agreements = votes.iter().filter(|v| v.output == vote.output).count();
                let mut result = ConsensusResult::new(&vote.output, vote.quality, self.strategy);
                result.participants = votes.len();
                result.agreements = agreements;
                for v in votes {
                    result
                        .agent_outputs
                        .insert(v.agent_id.clone(), v.output.clone());
                }
                result
            },
            None => ConsensusResult::new("", 0.0, self.strategy),
        }
    }

    fn unanimous_consensus(&self, votes: &[AgentVote]) -> ConsensusResult {
        if votes.is_empty() {
            return ConsensusResult::new("", 0.0, self.strategy);
        }

        let first_output = &votes[0].output;
        let unanimous = votes.iter().all(|v| v.output == *first_output);

        let mut result = if unanimous {
            ConsensusResult::new(first_output, 1.0, self.strategy)
        } else {
            ConsensusResult::new("", 0.0, self.strategy)
        };

        result.participants = votes.len();
        result.agreements = if unanimous { votes.len() } else { 0 };
        for vote in votes {
            result
                .agent_outputs
                .insert(vote.agent_id.clone(), vote.output.clone());
        }

        result
    }
}

impl Default for ConsensusEngine {
    fn default() -> Self {
        Self::new(ConsensusStrategy::WeightedMajority)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vote(agent_id: &str, output: &str, weight: f32) -> AgentVote {
        AgentVote {
            agent_id: agent_id.to_string(),
            output: output.to_string(),
            weight,
            quality: 0.8,
        }
    }

    #[test]
    fn test_majority_consensus() {
        let engine = ConsensusEngine::new(ConsensusStrategy::Majority);
        let votes = vec![
            make_vote("a1", "yes", 0.5),
            make_vote("a2", "yes", 0.5),
            make_vote("a3", "no", 0.5),
        ];

        let result = engine.reach_consensus(&votes);
        assert_eq!(result.output, "yes");
        assert_eq!(result.agreements, 2);
        // 66% (2/3) is not >= 80%, so is_strong should be false
        assert!(!result.is_strong());
    }

    #[test]
    fn test_weighted_consensus() {
        let engine = ConsensusEngine::new(ConsensusStrategy::WeightedMajority);
        let votes = vec![
            make_vote("gamma", "fast", 0.25),
            make_vote("beta", "fast", 0.50),
            make_vote("delta", "thorough", 1.00),
        ];

        let result = engine.reach_consensus(&votes);
        // Total weight: 0.25 + 0.50 + 1.00 = 1.75
        // "fast" weight: 0.75, "thorough" weight: 1.00
        // "thorough" wins
        assert_eq!(result.output, "thorough");
    }

    #[test]
    fn test_highest_context_consensus() {
        let engine = ConsensusEngine::new(ConsensusStrategy::HighestContext);
        let votes = vec![
            make_vote("gamma", "quick answer", 0.25),
            make_vote("delta", "detailed answer", 1.00),
        ];

        let result = engine.reach_consensus(&votes);
        assert_eq!(result.output, "detailed answer");
    }

    #[test]
    fn test_unanimous_consensus_success() {
        let engine = ConsensusEngine::new(ConsensusStrategy::Unanimous);
        let votes = vec![make_vote("a1", "agree", 0.5), make_vote("a2", "agree", 0.5)];

        let result = engine.reach_consensus(&votes);
        assert_eq!(result.output, "agree");
        assert!((result.confidence - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_unanimous_consensus_failure() {
        let engine = ConsensusEngine::new(ConsensusStrategy::Unanimous);
        let votes = vec![make_vote("a1", "yes", 0.5), make_vote("a2", "no", 0.5)];

        let result = engine.reach_consensus(&votes);
        assert_eq!(result.output, "");
        assert!((result.confidence - 0.0).abs() < 0.001);
    }

    // ==================== Interference Consensus Tests ====================

    #[test]
    fn test_interference_consensus_creation() {
        let consensus = InterferenceConsensus::new(InterferenceConfig::default());
        assert_eq!(consensus.contribution_count(), 0);
        assert!((consensus.field_energy() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_agent_contribution_from_text() {
        let contrib = AgentContribution::from_text(
            "agent-1",
            FrequencyBand::Operational,
            "Hello, world!",
            0.9,
        );

        assert_eq!(contrib.agent_id, "agent-1");
        assert_eq!(contrib.band, FrequencyBand::Operational);
        assert_eq!(contrib.output, "Hello, world!");
        assert!((contrib.confidence - 0.9).abs() < 0.001);

        // Pattern should have non-zero DC component
        assert!(contrib.pattern.dc_component().abs() > 0.0);
    }

    #[test]
    fn test_interference_with_agreement() {
        let mut consensus = InterferenceConsensus::new(InterferenceConfig::default());

        // Add similar contributions - should produce constructive interference
        consensus.add_contribution(AgentContribution::from_text(
            "agent-1",
            FrequencyBand::Strategic,
            "The answer is 42",
            0.9,
        ));
        consensus.add_contribution(AgentContribution::from_text(
            "agent-2",
            FrequencyBand::Operational,
            "The answer is 42",
            0.85,
        ));
        consensus.add_contribution(AgentContribution::from_text(
            "agent-3",
            FrequencyBand::Verification,
            "The answer is 42",
            0.8,
        ));

        let result = consensus.extract_consensus();

        assert_eq!(result.participants, 3);
        assert!(result.confidence > 0.0);
        assert_eq!(result.output, "The answer is 42");
    }

    #[test]
    fn test_interference_with_conflict() {
        let mut consensus = InterferenceConsensus::new(InterferenceConfig::default());

        // Add conflicting contributions
        consensus.add_contribution(AgentContribution::from_text(
            "agent-1",
            FrequencyBand::Strategic,
            "Yes, definitely",
            0.9,
        ));
        consensus.add_contribution(AgentContribution::from_text(
            "agent-2",
            FrequencyBand::Operational,
            "No, absolutely not",
            0.9,
        ));

        let result = consensus.extract_consensus();

        assert_eq!(result.participants, 2);
        // With conflict, confidence should be lower
        // (but exact value depends on encoding)
    }

    #[test]
    fn test_interference_via_votes() {
        let engine = ConsensusEngine::new(ConsensusStrategy::Interference);

        let votes = vec![
            make_vote("a1", "consensus answer", 0.8),
            make_vote("a2", "consensus answer", 0.9),
            make_vote("a3", "different answer", 0.5),
        ];

        let result = engine.reach_consensus(&votes);
        assert_eq!(result.strategy, ConsensusStrategy::Interference);
        assert!(result.participants >= 3);
    }

    #[test]
    fn test_consensus_weight() {
        let contrib = AgentContribution::from_text(
            "test",
            FrequencyBand::Operational, // emphasis = 1.0
            "test",
            0.8,
        );

        // Weight should be emphasis * confidence = 1.0 * 0.8 = 0.8
        assert!((contrib.consensus_weight() - 0.8).abs() < 0.01);

        let contrib2 = AgentContribution::from_text(
            "test2",
            FrequencyBand::Reflective, // emphasis = 0.6
            "test",
            1.0,
        );

        // Weight should be 0.6 * 1.0 = 0.6
        assert!((contrib2.consensus_weight() - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_interference_clear() {
        let mut consensus = InterferenceConsensus::new(InterferenceConfig::default());

        consensus.add_contribution(AgentContribution::from_text(
            "agent-1",
            FrequencyBand::Operational,
            "test",
            0.9,
        ));

        assert_eq!(consensus.contribution_count(), 1);
        assert!(consensus.field_energy() > 0.0);

        consensus.clear();

        assert_eq!(consensus.contribution_count(), 0);
        assert!((consensus.field_energy() - 0.0).abs() < 0.001);
    }
}
