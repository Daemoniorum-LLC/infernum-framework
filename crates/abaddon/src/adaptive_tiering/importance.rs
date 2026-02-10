//! Tensor importance scoring for allocation prioritization.
//!
//! Scores tensors based on their impact on inference quality. Higher importance
//! tensors are prioritized for VRAM placement and higher precision.
//!
//! # Importance Factors
//!
//! - **Layer position**: Edge layers (first/last few) are more critical
//! - **Tensor type**: Attention > LayerNorm > MLP
//! - **Access frequency**: Embeddings/lm_head used every token
//! - **Quality sensitivity**: From HoloTensor QualityCurve if available
//! - **Size efficiency**: Smaller tensors are cheaper to keep in VRAM

use super::types::{TensorInfo, TensorType};

/// Factors contributing to tensor importance.
#[derive(Debug, Clone, Copy)]
pub struct ImportanceFactors {
    /// Position factor: edge layers are more critical.
    /// Range: [0.5, 1.0] where edges = 1.0, middle = 0.5
    pub layer_position: f32,

    /// Type factor: attention > layernorm > mlp.
    /// Range: [0.5, 1.0]
    pub tensor_type: f32,

    /// Access frequency: tensors used every token vs occasionally.
    /// embed/lm_head: 1.0, layer weights: 0.8
    pub access_frequency: f32,

    /// Quality sensitivity from HoloTensor QualityCurve.
    /// Higher values = more sensitive to precision loss.
    /// Range: [0.0, 1.0]
    pub quality_sensitivity: f32,

    /// Size efficiency: smaller tensors are cheaper to keep in VRAM.
    /// Normalized inverse of tensor size.
    /// Range: [0.0, 1.0]
    pub size_efficiency: f32,
}

impl ImportanceFactors {
    /// Computes weighted importance score from factors.
    ///
    /// Weights:
    /// - tensor_type: 30% (most important - attention matters more than MLP)
    /// - layer_position: 25% (edge layers matter more)
    /// - access_frequency: 20% (frequently used tensors should be hot)
    /// - quality_sensitivity: 15% (respect HoloTensor hints)
    /// - size_efficiency: 10% (minor tiebreaker)
    pub fn compute_score(&self) -> f32 {
        0.30 * self.tensor_type
            + 0.25 * self.layer_position
            + 0.20 * self.access_frequency
            + 0.15 * self.quality_sensitivity
            + 0.10 * self.size_efficiency
    }
}

/// Scores tensors for allocation prioritization.
pub struct ImportanceScorer {
    /// Total number of layers in the model.
    num_layers: usize,
    /// Maximum tensor size for normalization.
    max_tensor_size: u64,
    /// Number of edge layers to consider "critical" on each side.
    edge_layer_count: usize,
}

impl ImportanceScorer {
    /// Creates a new importance scorer.
    ///
    /// # Arguments
    /// * `num_layers` - Total transformer layers in the model
    /// * `max_tensor_size` - Largest tensor size (for size normalization)
    pub fn new(num_layers: usize, max_tensor_size: u64) -> Self {
        Self {
            num_layers,
            max_tensor_size,
            edge_layer_count: 3.min(num_layers / 4).max(1),
        }
    }

    /// Scores a tensor's importance for allocation prioritization.
    ///
    /// Returns a score in [0.0, 1.0] where higher = more important.
    pub fn score(&self, tensor: &TensorInfo) -> f32 {
        let factors = self.compute_factors(tensor, None);
        factors.compute_score().clamp(0.0, 1.0)
    }

    /// Scores a tensor with optional quality sensitivity from HoloTensor metadata.
    pub fn score_with_quality(&self, tensor: &TensorInfo, quality_sensitivity: f32) -> f32 {
        let factors = self.compute_factors(tensor, Some(quality_sensitivity));
        factors.compute_score().clamp(0.0, 1.0)
    }

    /// Computes individual importance factors for a tensor.
    pub fn compute_factors(
        &self,
        tensor: &TensorInfo,
        quality_sensitivity: Option<f32>,
    ) -> ImportanceFactors {
        ImportanceFactors {
            layer_position: self.layer_position_factor(tensor.layer_index),
            tensor_type: self.tensor_type_factor(tensor.tensor_type),
            access_frequency: self.access_frequency_factor(tensor.tensor_type),
            quality_sensitivity: quality_sensitivity.unwrap_or(0.5),
            size_efficiency: self.size_efficiency_factor(tensor.size_bytes),
        }
    }

    /// Computes layer position factor.
    ///
    /// Edge layers (first and last few) get higher scores because:
    /// - First layers process raw embeddings (critical for understanding)
    /// - Last layers produce final logits (critical for output quality)
    fn layer_position_factor(&self, layer_index: Option<usize>) -> f32 {
        match layer_index {
            None => 1.0, // Embeddings/lm_head are always critical
            Some(idx) => {
                if self.num_layers == 0 {
                    return 0.5;
                }

                // Distance from nearest edge (0 = at edge, increases toward middle)
                let dist_from_start = idx;
                let dist_from_end = self.num_layers.saturating_sub(idx + 1);
                let edge_distance = dist_from_start.min(dist_from_end);

                // Layers within edge_layer_count get full score
                if edge_distance < self.edge_layer_count {
                    // Gradual falloff within edge zone
                    1.0 - (edge_distance as f32 / self.edge_layer_count as f32) * 0.2
                } else {
                    // Middle layers get base score with slight position preference
                    let middle_position = idx as f32 / self.num_layers.max(1) as f32;
                    // Slight preference for layers closer to edges even in middle zone
                    0.5 + 0.1 * (1.0 - (middle_position - 0.5).abs() * 2.0)
                }
            },
        }
    }

    /// Computes tensor type factor based on inference impact.
    fn tensor_type_factor(&self, tensor_type: TensorType) -> f32 {
        tensor_type.base_importance()
    }

    /// Computes access frequency factor.
    ///
    /// Embeddings and lm_head are accessed every token.
    /// Layer weights are accessed once per layer per token.
    fn access_frequency_factor(&self, tensor_type: TensorType) -> f32 {
        match tensor_type {
            TensorType::Embedding | TensorType::LmHead => 1.0,
            _ => 0.8, // Layer weights accessed once per forward pass
        }
    }

    /// Computes size efficiency factor.
    ///
    /// Smaller tensors are more efficient to keep in VRAM.
    fn size_efficiency_factor(&self, size_bytes: u64) -> f32 {
        if self.max_tensor_size == 0 {
            return 0.5;
        }
        // Inverse relationship: smaller = higher efficiency
        let normalized = size_bytes as f32 / self.max_tensor_size as f32;
        1.0 - normalized.clamp(0.0, 1.0)
    }
}

/// Computes importance from a HoloTensor quality curve.
///
/// Tensors with steeper early quality curves (where early fragments contribute
/// more to quality) are more sensitive to precision loss.
///
/// # Arguments
/// * `q_at_25_pct` - Quality at 25% of fragments loaded
/// * `q_at_50_pct` - Quality at 50% of fragments loaded
///
/// # Returns
/// Importance factor in [0.0, 1.0] where higher = more quality sensitive.
///
/// This function is used when integrating with HoloTensor metadata to derive
/// per-tensor importance from the quality curve stored in HCT files.
#[allow(dead_code)] // Used when HoloTensor metadata integration is available
pub fn importance_from_quality_curve(q_at_25_pct: f32, q_at_50_pct: f32) -> f32 {
    // Steep early curve = high importance
    // If 25% fragments give nearly as much quality as 50%, curve is steep
    let steepness = q_at_25_pct / q_at_50_pct.max(0.01);
    // Normalize to [0, 1] - steepness ranges from ~0.5 (linear) to ~1.0 (very steep)
    (steepness - 0.5).clamp(0.0, 1.0) * 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scorer() -> ImportanceScorer {
        ImportanceScorer::new(48, 100_000_000) // 48 layers, 100MB max tensor
    }

    #[test]
    fn test_embeddings_highest_priority() {
        let scorer = make_scorer();
        let embed = TensorInfo::from_name("model.embed_tokens.weight", 50_000_000);
        let middle_attn =
            TensorInfo::from_name("model.layers.24.self_attn.q_proj.weight", 50_000_000);
        let middle_mlp = TensorInfo::from_name("model.layers.24.mlp.gate_proj.weight", 50_000_000);

        let embed_score = scorer.score(&embed);
        let attn_score = scorer.score(&middle_attn);
        let mlp_score = scorer.score(&middle_mlp);

        assert!(
            embed_score > attn_score,
            "embeddings should be higher than attention"
        );
        assert!(
            attn_score > mlp_score,
            "attention should be higher than mlp"
        );
    }

    #[test]
    fn test_edge_layers_higher_than_middle() {
        let scorer = make_scorer();
        let layer_0 = TensorInfo::from_name("model.layers.0.self_attn.q_proj.weight", 50_000_000);
        let layer_24 = TensorInfo::from_name("model.layers.24.self_attn.q_proj.weight", 50_000_000);
        let layer_47 = TensorInfo::from_name("model.layers.47.self_attn.q_proj.weight", 50_000_000);

        let score_0 = scorer.score(&layer_0);
        let score_24 = scorer.score(&layer_24);
        let score_47 = scorer.score(&layer_47);

        assert!(
            score_0 > score_24,
            "layer 0 should be higher than middle layer"
        );
        assert!(
            score_47 > score_24,
            "last layer should be higher than middle layer"
        );
        // First and last should be similar
        assert!(
            (score_0 - score_47).abs() < 0.1,
            "first and last layers should have similar scores"
        );
    }

    #[test]
    fn test_smaller_tensors_slightly_preferred() {
        let scorer = make_scorer();
        let small = TensorInfo::from_name("model.layers.24.self_attn.q_proj.weight", 10_000_000);
        let large = TensorInfo::from_name("model.layers.24.self_attn.k_proj.weight", 90_000_000);

        let small_score = scorer.score(&small);
        let large_score = scorer.score(&large);

        // Size is minor factor (10%), so difference should be small but present
        assert!(
            small_score > large_score,
            "smaller tensor should have slightly higher score"
        );
        assert!(
            small_score - large_score < 0.15,
            "size difference should be minor factor"
        );
    }

    #[test]
    fn test_importance_factors_sum_reasonably() {
        let scorer = make_scorer();
        let tensor = TensorInfo::from_name("model.layers.10.mlp.down_proj.weight", 50_000_000);
        let factors = scorer.compute_factors(&tensor, None);

        // All factors should be in valid range
        assert!(factors.layer_position >= 0.0 && factors.layer_position <= 1.0);
        assert!(factors.tensor_type >= 0.0 && factors.tensor_type <= 1.0);
        assert!(factors.access_frequency >= 0.0 && factors.access_frequency <= 1.0);
        assert!(factors.quality_sensitivity >= 0.0 && factors.quality_sensitivity <= 1.0);
        assert!(factors.size_efficiency >= 0.0 && factors.size_efficiency <= 1.0);

        // Final score should be in range
        let score = factors.compute_score();
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_quality_curve_importance() {
        // Steep curve: 25% fragments give 90% quality
        let steep = importance_from_quality_curve(0.9, 0.95);
        // Linear curve: 25% fragments give ~50% quality
        let linear = importance_from_quality_curve(0.5, 0.9);

        assert!(
            steep > linear,
            "steep curve should indicate higher importance"
        );
    }

    #[test]
    fn test_score_invariants() {
        let scorer = make_scorer();

        // Property: All scores in [0, 1]
        let tensors = vec![
            TensorInfo::from_name("model.embed_tokens.weight", 100_000_000),
            TensorInfo::from_name("lm_head.weight", 100_000_000),
            TensorInfo::from_name("model.layers.0.self_attn.q_proj.weight", 50_000_000),
            TensorInfo::from_name("model.layers.47.mlp.gate_proj.weight", 50_000_000),
            TensorInfo::from_name("model.layers.24.input_layernorm.weight", 1_000_000),
        ];

        for tensor in tensors {
            let score = scorer.score(&tensor);
            assert!(
                score >= 0.0 && score <= 1.0,
                "score for {} should be in [0, 1], got {}",
                tensor.name,
                score
            );
        }
    }

    #[test]
    fn test_attention_vs_mlp_same_position() {
        let scorer = make_scorer();
        let attn = TensorInfo::from_name("model.layers.10.self_attn.q_proj.weight", 50_000_000);
        let mlp = TensorInfo::from_name("model.layers.10.mlp.gate_proj.weight", 50_000_000);

        let attn_score = scorer.score(&attn);
        let mlp_score = scorer.score(&mlp);

        assert!(
            attn_score > mlp_score,
            "attention should be prioritized over MLP at same layer"
        );
    }
}
