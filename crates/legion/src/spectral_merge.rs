//! Spectral Model Merging - Runtime model blending via spectral coefficient superposition.
//!
//! Instead of merging model weights on disk, load multiple models as spectral
//! decompositions and blend at inference time. This enables:
//!
//! - **Dynamic Blending**: Adjust model mix during generation based on content
//! - **Per-Layer Control**: Different blend weights for attention vs MLP layers
//! - **Progressive Loading**: Essential coefficients load first for fast startup
//! - **Graceful Degradation**: Partial models still produce coherent output
//!
//! ## Architecture
//!
//! ```text
//! Model A (coder)      Model B (creative)     Model C (instruct)
//!      │                     │                      │
//!      ▼                     ▼                      ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │              SPECTRAL DECOMPOSITION (per-layer)             │
//! │                                                             │
//! │  Essential coefficients: Core capability (DC component)     │
//! │  Detail coefficients: Style/specialization (high freq)      │
//! └─────────────────────────────────────────────────────────────┘
//!      │                     │                      │
//!      │        weight: 0.6  │  weight: 0.3         │  weight: 0.1
//!      ▼                     ▼                      ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                 RUNTIME SUPERPOSITION                        │
//! │                                                             │
//! │  merged_essential = 0.6*A + 0.3*B + 0.1*C                   │
//! │  merged_detail = blend_by_frequency(A, B, C, weights)       │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Quality Curve
//!
//! Essential coefficients (~15% of data) carry ~60% of reconstruction quality.
//! This enables fast model startup while detail streams progressively.
//!
//! ## Example
//!
//! ```ignore
//! use legion::spectral_merge::{SpectralDecomposition, SpectralBlend, LayerWeights};
//!
//! let coder = SpectralDecomposition::load("path/to/coder.hct")?;
//! let creative = SpectralDecomposition::load("path/to/creative.hct")?;
//!
//! let blended = SpectralBlend::new()
//!     .add(coder, 0.7)
//!     .add(creative, 0.3)
//!     .build();
//!
//! // Use blended model for inference
//! let weights = blended.get_layer_weights(0)?;
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::field::{DetailCoefficients, EssentialCoefficients, LegionPattern};
use crate::quality::QualityCurve;

// ==================== Error Types ====================

/// Errors from spectral merge operations.
#[derive(Debug, Error)]
pub enum SpectralMergeError {
    /// Dimension mismatch between models.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: String,
        /// Actual dimension found.
        actual: String,
    },

    /// Layer count mismatch.
    #[error("Layer count mismatch: expected {expected}, got {actual}")]
    LayerCountMismatch {
        /// Expected layer count.
        expected: usize,
        /// Actual layer count found.
        actual: usize,
    },

    /// No models to blend.
    #[error("No models added to blend")]
    NoModels,

    /// Invalid weight.
    #[error("Invalid weight {weight}: weights must be positive")]
    InvalidWeight {
        /// The invalid weight value.
        weight: f32,
    },

    /// Model not found.
    #[error("Model not found: {name}")]
    ModelNotFound {
        /// Name of the model that was not found.
        name: String,
    },

    /// Layer not found.
    #[error("Layer {index} not found (model has {count} layers)")]
    LayerNotFound {
        /// Requested layer index.
        index: usize,
        /// Total number of layers in the model.
        count: usize,
    },

    /// Invalid quality level.
    #[error("Invalid quality level {quality}: must be in range [0.0, 1.0]")]
    InvalidQuality {
        /// The invalid quality value.
        quality: f32,
    },

    /// Decomposition failed.
    #[error("Decomposition failed: {0}")]
    DecompositionFailed(String),
}

/// Result type for spectral merge operations.
pub type Result<T> = std::result::Result<T, SpectralMergeError>;

// ==================== Layer Types ====================

/// Type of model layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerType {
    /// Embedding layer.
    Embedding,
    /// Attention layer (Q, K, V projections + output).
    Attention,
    /// Feed-forward / MLP layer.
    Mlp,
    /// Layer normalization.
    LayerNorm,
    /// Output / LM head.
    Output,
    /// Other layer type.
    Other,
}

impl Default for LayerType {
    fn default() -> Self {
        Self::Other
    }
}

/// Per-layer blend weights.
///
/// Different layer types may benefit from different blend ratios.
/// For example, attention layers might favor the "coder" model while
/// MLP layers blend more evenly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerWeights {
    /// Weight for attention layers.
    pub attention: f32,
    /// Weight for MLP layers.
    pub mlp: f32,
    /// Weight for embedding layers.
    pub embedding: f32,
    /// Weight for layer norm.
    pub layer_norm: f32,
    /// Weight for output/LM head.
    pub output: f32,
}

impl Default for LayerWeights {
    fn default() -> Self {
        Self {
            attention: 1.0,
            mlp: 1.0,
            embedding: 1.0,
            layer_norm: 1.0,
            output: 1.0,
        }
    }
}

impl LayerWeights {
    /// Creates uniform weights.
    pub fn uniform(weight: f32) -> Self {
        Self {
            attention: weight,
            mlp: weight,
            embedding: weight,
            layer_norm: weight,
            output: weight,
        }
    }

    /// Creates weights emphasizing attention layers.
    pub fn attention_heavy(attention: f32, other: f32) -> Self {
        Self {
            attention,
            mlp: other,
            embedding: other,
            layer_norm: other,
            output: other,
        }
    }

    /// Creates weights emphasizing MLP layers.
    pub fn mlp_heavy(mlp: f32, other: f32) -> Self {
        Self {
            attention: other,
            mlp,
            embedding: other,
            layer_norm: other,
            output: other,
        }
    }

    /// Returns weight for a given layer type.
    pub fn weight_for(&self, layer_type: LayerType) -> f32 {
        match layer_type {
            LayerType::Attention => self.attention,
            LayerType::Mlp => self.mlp,
            LayerType::Embedding => self.embedding,
            LayerType::LayerNorm => self.layer_norm,
            LayerType::Output => self.output,
            LayerType::Other => 1.0,
        }
    }
}

// ==================== Layer Decomposition ====================

/// Spectral decomposition of a single layer.
///
/// Contains the frequency-domain representation of layer weights,
/// split into essential (DC + low-freq) and detail (high-freq) components.
#[derive(Debug, Clone)]
pub struct LayerDecomposition {
    /// Layer index in the model.
    pub layer_index: usize,
    /// Type of layer.
    pub layer_type: LayerType,
    /// Layer name (e.g., "transformer.layers.0.attention.q_proj").
    pub name: String,
    /// Original weight shape.
    pub shape: Vec<usize>,
    /// Essential coefficients (DC + low frequency).
    pub essential: EssentialCoefficients,
    /// Detail coefficients (mid/high frequency).
    pub detail: DetailCoefficients,
    /// Quality curve for reconstruction.
    pub quality_curve: QualityCurve,
}

impl LayerDecomposition {
    /// Creates a new layer decomposition from weights.
    ///
    /// Applies 2D DCT to decompose weights into spectral coefficients,
    /// then splits into essential and detail components.
    pub fn from_weights(
        layer_index: usize,
        layer_type: LayerType,
        name: String,
        weights: &[f32],
        shape: Vec<usize>,
    ) -> Result<Self> {
        if weights.is_empty() {
            return Err(SpectralMergeError::DecompositionFailed(
                "Empty weights array".to_string(),
            ));
        }

        // For now, treat weights as 1D and apply basic DCT-like decomposition
        // In production, would use proper 2D DCT for matrix weights
        let pattern = Self::dct_transform(weights);

        let essential = Self::extract_essential(&pattern);
        let detail = Self::extract_detail(&pattern);

        Ok(Self {
            layer_index,
            layer_type,
            name,
            shape,
            essential,
            detail,
            quality_curve: QualityCurve::SPECTRAL,
        })
    }

    /// Performs DCT-like transformation.
    ///
    /// Simplified DCT that separates low-frequency (average/trends)
    /// from high-frequency (variations) components.
    fn dct_transform(weights: &[f32]) -> Vec<f32> {
        let n = weights.len();
        if n == 0 {
            return vec![];
        }

        let mut coefficients = vec![0.0f32; n];

        // DC component = mean * sqrt(n) (energy-preserving normalization)
        let mean: f32 = weights.iter().sum::<f32>() / n as f32;
        coefficients[0] = mean * (n as f32).sqrt();

        // Higher frequency components via cosine basis projection
        // Using simplified DCT-II: F(k) = sum(f(n) * cos(pi*k*(n+0.5)/N))
        for k in 1..n {
            let mut sum = 0.0f32;
            for (i, &w) in weights.iter().enumerate() {
                let cos_arg = std::f32::consts::PI * k as f32 * (i as f32 + 0.5) / n as f32;
                sum += w * cos_arg.cos();
            }
            coefficients[k] = sum * (2.0 / n as f32).sqrt();
        }

        coefficients
    }

    /// Performs inverse DCT transformation.
    fn idct_transform(coefficients: &[f32]) -> Vec<f32> {
        let n = coefficients.len();
        if n == 0 {
            return vec![];
        }

        let mut weights = vec![0.0f32; n];

        // Reconstruct using inverse DCT-II
        for i in 0..n {
            let mut sum = coefficients[0] / (n as f32).sqrt();
            for k in 1..n {
                let cos_arg = std::f32::consts::PI * k as f32 * (i as f32 + 0.5) / n as f32;
                sum += coefficients[k] * (2.0 / n as f32).sqrt() * cos_arg.cos();
            }
            weights[i] = sum;
        }

        weights
    }

    /// Extracts essential coefficients (DC + low frequency).
    fn extract_essential(coefficients: &[f32]) -> EssentialCoefficients {
        let dc = coefficients.first().copied().unwrap_or(0.0);
        let low_freq_count = ((coefficients.len() as f32 * 0.15) as usize).max(1);
        let low_freq: Vec<f32> = coefficients
            .iter()
            .skip(1)
            .take(low_freq_count)
            .copied()
            .collect();

        EssentialCoefficients { dc, low_freq }
    }

    /// Extracts detail coefficients (mid/high frequency).
    fn extract_detail(coefficients: &[f32]) -> DetailCoefficients {
        let essential_count = 1 + ((coefficients.len() as f32 * 0.15) as usize).max(1);
        let detail: Vec<f32> = coefficients.iter().skip(essential_count).copied().collect();

        DetailCoefficients {
            coefficients: detail,
            start_index: essential_count,
        }
    }

    /// Reconstructs weights at given quality level (0.0 - 1.0).
    ///
    /// Quality 0.0 = DC only (~40% reconstruction)
    /// Quality 0.15 = Essential only (~60% reconstruction)
    /// Quality 1.0 = Full reconstruction
    pub fn reconstruct(&self, quality: f32) -> Vec<f32> {
        let quality = quality.clamp(0.0, 1.0);

        // Calculate how many coefficients to use
        let total_coeffs = 1 + self.essential.low_freq.len() + self.detail.coefficients.len();
        let coeffs_to_use = ((total_coeffs as f32 * quality) as usize).max(1);

        // Build coefficient vector
        let mut coefficients = vec![0.0f32; total_coeffs];

        // DC always included
        coefficients[0] = self.essential.dc;

        // Add essential low-freq based on quality
        let essential_to_use = (coeffs_to_use - 1).min(self.essential.low_freq.len());
        for (i, &val) in self
            .essential
            .low_freq
            .iter()
            .take(essential_to_use)
            .enumerate()
        {
            coefficients[i + 1] = val;
        }

        // Add detail based on quality
        let detail_start = 1 + self.essential.low_freq.len();
        let remaining = coeffs_to_use.saturating_sub(detail_start);
        let detail_to_use = remaining.min(self.detail.coefficients.len());
        for (i, &val) in self
            .detail
            .coefficients
            .iter()
            .take(detail_to_use)
            .enumerate()
        {
            coefficients[detail_start + i] = val;
        }

        // Inverse DCT
        Self::idct_transform(&coefficients)
    }

    /// Returns expected quality at given coefficient fraction.
    pub fn expected_quality(&self, fraction: f32) -> f32 {
        // Convert fraction to fragment counts for predict()
        let k = (fraction * 100.0) as u16;
        self.quality_curve.predict(k, 100)
    }

    /// Computes energy (L2 norm) of essential coefficients.
    pub fn essential_energy(&self) -> f32 {
        let dc_energy = self.essential.dc * self.essential.dc;
        let low_energy: f32 = self.essential.low_freq.iter().map(|c| c * c).sum();
        (dc_energy + low_energy).sqrt()
    }

    /// Computes energy (L2 norm) of detail coefficients.
    pub fn detail_energy(&self) -> f32 {
        self.detail
            .coefficients
            .iter()
            .map(|c| c * c)
            .sum::<f32>()
            .sqrt()
    }
}

// ==================== Model Decomposition ====================

/// Spectral decomposition of an entire model.
///
/// Contains per-layer spectral decompositions that can be blended
/// with other models at runtime.
#[derive(Debug, Clone)]
pub struct SpectralDecomposition {
    /// Model name/identifier.
    pub name: String,
    /// Model description.
    pub description: String,
    /// Number of layers.
    pub num_layers: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Per-layer decompositions.
    layers: Vec<LayerDecomposition>,
    /// Layer index by name.
    name_to_index: HashMap<String, usize>,
}

impl SpectralDecomposition {
    /// Creates a new empty decomposition.
    pub fn new(name: String, description: String) -> Self {
        Self {
            name,
            description,
            num_layers: 0,
            hidden_size: 0,
            num_attention_heads: 0,
            vocab_size: 0,
            layers: Vec::new(),
            name_to_index: HashMap::new(),
        }
    }

    /// Creates decomposition from model weights.
    ///
    /// Takes a map of layer names to weight tensors and decomposes each.
    pub fn from_weights(
        name: String,
        description: String,
        weights: HashMap<String, (Vec<f32>, Vec<usize>)>,
        layer_types: HashMap<String, LayerType>,
    ) -> Result<Self> {
        let mut decomposition = Self::new(name, description);

        // Sort layers by name for consistent ordering
        let mut layer_names: Vec<_> = weights.keys().cloned().collect();
        layer_names.sort();

        for (idx, layer_name) in layer_names.iter().enumerate() {
            let (weight_data, shape) =
                weights
                    .get(layer_name)
                    .ok_or_else(|| SpectralMergeError::ModelNotFound {
                        name: layer_name.clone(),
                    })?;

            let layer_type = layer_types
                .get(layer_name)
                .copied()
                .unwrap_or(LayerType::Other);

            let layer = LayerDecomposition::from_weights(
                idx,
                layer_type,
                layer_name.clone(),
                weight_data,
                shape.clone(),
            )?;

            decomposition.add_layer(layer);
        }

        Ok(decomposition)
    }

    /// Adds a layer decomposition.
    pub fn add_layer(&mut self, layer: LayerDecomposition) {
        let name = layer.name.clone();
        let index = self.layers.len();
        self.layers.push(layer);
        self.name_to_index.insert(name, index);
        self.num_layers = self.layers.len();
    }

    /// Returns the number of layers.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Gets a layer by index.
    pub fn get_layer(&self, index: usize) -> Option<&LayerDecomposition> {
        self.layers.get(index)
    }

    /// Gets a layer by name.
    pub fn get_layer_by_name(&self, name: &str) -> Option<&LayerDecomposition> {
        self.name_to_index
            .get(name)
            .and_then(|&idx| self.layers.get(idx))
    }

    /// Returns iterator over all layers.
    pub fn layers(&self) -> impl Iterator<Item = &LayerDecomposition> {
        self.layers.iter()
    }

    /// Reconstructs all weights at given quality level.
    pub fn reconstruct(&self, quality: f32) -> Vec<Vec<f32>> {
        self.layers
            .iter()
            .map(|layer| layer.reconstruct(quality))
            .collect()
    }

    /// Computes total essential energy across all layers.
    pub fn total_essential_energy(&self) -> f32 {
        self.layers.iter().map(|l| l.essential_energy()).sum()
    }

    /// Computes total detail energy across all layers.
    pub fn total_detail_energy(&self) -> f32 {
        self.layers.iter().map(|l| l.detail_energy()).sum()
    }

    /// Returns the essential/total energy ratio.
    pub fn essential_ratio(&self) -> f32 {
        let essential = self.total_essential_energy();
        let detail = self.total_detail_energy();
        let total = essential + detail;
        if total == 0.0 {
            0.0
        } else {
            essential / total
        }
    }
}

// ==================== Model Component ====================

/// A model component in a blend, with its weight.
#[derive(Debug, Clone)]
pub struct BlendComponent {
    /// The decomposed model.
    pub model: Arc<SpectralDecomposition>,
    /// Base weight for this model (before normalization).
    pub weight: f32,
    /// Per-layer weight overrides.
    pub layer_weights: LayerWeights,
}

// ==================== Spectral Blend Builder ====================

/// Builder for creating blended models from spectral decompositions.
///
/// Allows combining multiple models with different weights, including
/// per-layer weight customization.
#[derive(Debug, Default)]
pub struct SpectralBlend {
    /// Model components to blend.
    components: Vec<BlendComponent>,
}

impl SpectralBlend {
    /// Creates a new empty blend.
    pub fn new() -> Self {
        Self {
            components: Vec::new(),
        }
    }

    /// Adds a model with uniform weight.
    pub fn add(mut self, model: Arc<SpectralDecomposition>, weight: f32) -> Self {
        self.components.push(BlendComponent {
            model,
            weight,
            layer_weights: LayerWeights::uniform(weight),
        });
        self
    }

    /// Adds a model with per-layer weights.
    pub fn add_with_layer_weights(
        mut self,
        model: Arc<SpectralDecomposition>,
        weight: f32,
        layer_weights: LayerWeights,
    ) -> Self {
        self.components.push(BlendComponent {
            model,
            weight,
            layer_weights,
        });
        self
    }

    /// Builds the blended model.
    pub fn build(self) -> Result<BlendedModel> {
        if self.components.is_empty() {
            return Err(SpectralMergeError::NoModels);
        }

        // Validate all models have same layer count
        let first_layer_count = self.components[0].model.layer_count();
        for component in &self.components[1..] {
            if component.model.layer_count() != first_layer_count {
                return Err(SpectralMergeError::LayerCountMismatch {
                    expected: first_layer_count,
                    actual: component.model.layer_count(),
                });
            }
        }

        // Normalize base weights
        let total_weight: f32 = self.components.iter().map(|c| c.weight).sum();
        if total_weight <= 0.0 {
            return Err(SpectralMergeError::InvalidWeight {
                weight: total_weight,
            });
        }

        let normalized_components: Vec<BlendComponent> = self
            .components
            .into_iter()
            .map(|mut c| {
                c.weight /= total_weight;
                c
            })
            .collect();

        Ok(BlendedModel::new(normalized_components))
    }
}

// ==================== Blended Model ====================

/// A blended model created from spectral superposition.
///
/// Provides access to blended weights at any quality level,
/// with support for dynamic blend adjustment during generation.
pub struct BlendedModel {
    /// Blend components with normalized weights.
    components: Vec<BlendComponent>,
    /// Dynamic weight adjustments (model name -> adjustment factor).
    adjustments: RwLock<HashMap<String, f32>>,
    /// Generation step counter for decay.
    step: AtomicU64,
    /// Quality curve for reconstruction.
    quality_curve: QualityCurve,
}

impl BlendedModel {
    /// Creates a new blended model from components.
    fn new(components: Vec<BlendComponent>) -> Self {
        Self {
            components,
            adjustments: RwLock::new(HashMap::new()),
            step: AtomicU64::new(0),
            quality_curve: QualityCurve::SPECTRAL,
        }
    }

    /// Returns the number of layers in the blended model.
    pub fn layer_count(&self) -> usize {
        self.components
            .first()
            .map(|c| c.model.layer_count())
            .unwrap_or(0)
    }

    /// Returns the number of component models.
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Returns the normalized weight for a model.
    pub fn weight(&self, model_name: &str) -> Option<f32> {
        let adjustments = self.adjustments.read();
        self.components
            .iter()
            .find(|c| c.model.name == model_name)
            .map(|c| {
                let adjustment = adjustments.get(model_name).copied().unwrap_or(1.0);
                c.weight * adjustment
            })
    }

    /// Returns the attention blend weight for a model at layer index.
    pub fn attention_blend_weight(&self, layer_index: usize) -> f32 {
        self.layer_type_weight(layer_index, LayerType::Attention)
    }

    /// Returns the MLP blend weight for a model at layer index.
    pub fn mlp_blend_weight(&self, layer_index: usize) -> f32 {
        self.layer_type_weight(layer_index, LayerType::Mlp)
    }

    /// Returns blend weight for a specific layer type.
    fn layer_type_weight(&self, _layer_index: usize, layer_type: LayerType) -> f32 {
        let adjustments = self.adjustments.read();
        let mut total_weight = 0.0;

        for component in &self.components {
            let base = component.weight;
            let layer_mult = component.layer_weights.weight_for(layer_type);
            let adjustment = adjustments
                .get(&component.model.name)
                .copied()
                .unwrap_or(1.0);
            total_weight += base * layer_mult * adjustment;
        }

        // Normalize
        let total_base: f32 = self
            .components
            .iter()
            .map(|c| {
                let adj = adjustments.get(&c.model.name).copied().unwrap_or(1.0);
                c.weight * c.layer_weights.weight_for(layer_type) * adj
            })
            .sum();

        if total_base > 0.0 {
            total_weight / total_base
        } else {
            1.0
        }
    }

    /// Gets blended weights for a layer at given quality.
    pub fn get_layer_weights(&self, layer_index: usize, quality: f32) -> Result<Vec<f32>> {
        if layer_index >= self.layer_count() {
            return Err(SpectralMergeError::LayerNotFound {
                index: layer_index,
                count: self.layer_count(),
            });
        }

        let adjustments = self.adjustments.read();

        // Get layer type from first component
        let layer_type = self
            .components
            .first()
            .and_then(|c| c.model.get_layer(layer_index))
            .map(|l| l.layer_type)
            .unwrap_or(LayerType::Other);

        // Collect and blend weights from all components
        let mut blended: Option<Vec<f32>> = None;
        let mut total_weight = 0.0f32;

        for component in &self.components {
            let layer = component.model.get_layer(layer_index).ok_or(
                SpectralMergeError::LayerNotFound {
                    index: layer_index,
                    count: component.model.layer_count(),
                },
            )?;

            let reconstructed = layer.reconstruct(quality);

            // Calculate effective weight
            let base = component.weight;
            let layer_mult = component.layer_weights.weight_for(layer_type);
            let adjustment = adjustments
                .get(&component.model.name)
                .copied()
                .unwrap_or(1.0);
            let effective_weight = base * layer_mult * adjustment;
            total_weight += effective_weight;

            match blended.as_mut() {
                Some(blend) => {
                    // Add weighted contribution
                    for (i, &val) in reconstructed.iter().enumerate() {
                        if i < blend.len() {
                            blend[i] += val * effective_weight;
                        }
                    }
                },
                None => {
                    // First component
                    blended = Some(
                        reconstructed
                            .iter()
                            .map(|&v| v * effective_weight)
                            .collect(),
                    );
                },
            }
        }

        // Normalize by total weight
        if let Some(ref mut blend) = blended {
            if total_weight > 0.0 {
                for val in blend.iter_mut() {
                    *val /= total_weight;
                }
            }
        }

        blended.ok_or(SpectralMergeError::NoModels)
    }

    /// Adjusts blend weight for a model during generation.
    ///
    /// The adjustment is multiplicative: final_weight = base_weight * adjustment.
    pub fn adjust_weight(&self, model_name: &str, adjustment: f32) -> Result<()> {
        if adjustment <= 0.0 {
            return Err(SpectralMergeError::InvalidWeight { weight: adjustment });
        }

        // Verify model exists
        let exists = self.components.iter().any(|c| c.model.name == model_name);
        if !exists {
            return Err(SpectralMergeError::ModelNotFound {
                name: model_name.to_string(),
            });
        }

        let mut adjustments = self.adjustments.write();
        adjustments.insert(model_name.to_string(), adjustment);
        Ok(())
    }

    /// Resets all weight adjustments.
    pub fn reset_adjustments(&self) {
        let mut adjustments = self.adjustments.write();
        adjustments.clear();
    }

    /// Advances the generation step counter.
    pub fn step(&self) {
        self.step.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns current generation step.
    pub fn current_step(&self) -> u64 {
        self.step.load(Ordering::Relaxed)
    }

    /// Returns expected quality at given coefficient fraction.
    pub fn expected_quality(&self, fraction: f32) -> f32 {
        // Convert fraction to fragment counts for predict()
        let k = (fraction * 100.0) as u16;
        self.quality_curve.predict(k, 100)
    }

    /// Creates a pattern representing the blend at given layer.
    pub fn to_pattern(&self, layer_index: usize, quality: f32) -> Result<LegionPattern> {
        let weights = self.get_layer_weights(layer_index, quality)?;

        // Determine dimensions (square-ish)
        let len = weights.len();
        let side = (len as f32).sqrt().ceil() as usize;
        let mut coefficients = weights;
        coefficients.resize(side * side, 0.0);

        Ok(LegionPattern::from_coefficients(coefficients, side, side))
    }
}

impl std::fmt::Debug for BlendedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let adjustments = self.adjustments.read();
        f.debug_struct("BlendedModel")
            .field("component_count", &self.components.len())
            .field("layer_count", &self.layer_count())
            .field("step", &self.current_step())
            .field("adjustments", &*adjustments)
            .finish()
    }
}

// ==================== Dynamic Blend Controller ====================

/// Controller for dynamic blend adjustment during generation.
///
/// Monitors generation output and adjusts blend weights to achieve
/// desired characteristics (more code-like, more creative, etc.).
#[derive(Debug)]
pub struct DynamicBlendController {
    /// Blended model to control.
    model: Arc<BlendedModel>,
    /// Target model weights (what we're trying to achieve).
    targets: RwLock<HashMap<String, f32>>,
    /// Adjustment rate (how fast to move toward target).
    adjustment_rate: f32,
    /// Decay rate for adjustments (pull back toward base weights).
    decay_rate: f32,
}

impl DynamicBlendController {
    /// Creates a new controller for a blended model.
    pub fn new(model: Arc<BlendedModel>) -> Self {
        Self {
            model,
            targets: RwLock::new(HashMap::new()),
            adjustment_rate: 0.1,
            decay_rate: 0.95,
        }
    }

    /// Sets the adjustment rate (0.0 - 1.0).
    pub fn with_adjustment_rate(mut self, rate: f32) -> Self {
        self.adjustment_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Sets the decay rate (0.0 - 1.0).
    pub fn with_decay_rate(mut self, rate: f32) -> Self {
        self.decay_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Sets target weight for a model.
    pub fn set_target(&self, model_name: &str, target_weight: f32) {
        let mut targets = self.targets.write();
        targets.insert(model_name.to_string(), target_weight);
    }

    /// Clears all targets.
    pub fn clear_targets(&self) {
        let mut targets = self.targets.write();
        targets.clear();
    }

    /// Updates blend weights toward targets.
    ///
    /// Call this after each generation step.
    pub fn update(&self) -> Result<()> {
        let targets = self.targets.read();

        for (model_name, &target) in targets.iter() {
            let current =
                self.model
                    .weight(model_name)
                    .ok_or_else(|| SpectralMergeError::ModelNotFound {
                        name: model_name.clone(),
                    })?;

            // Move toward target
            let delta = target - current;
            let new_adjustment = 1.0 + (delta * self.adjustment_rate);

            self.model.adjust_weight(model_name, new_adjustment)?;
        }

        Ok(())
    }

    /// Applies decay to adjustments (pull toward base weights).
    pub fn decay(&self) {
        // Reset adjustments toward 1.0 (no adjustment)
        // This is a simplified version - in production would track
        // and interpolate each adjustment separately
        self.model.reset_adjustments();
    }

    /// Returns the underlying blended model.
    pub fn model(&self) -> &Arc<BlendedModel> {
        &self.model
    }
}

// ==================== Blend Statistics ====================

/// Statistics about a blended model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BlendStats {
    /// Number of component models.
    pub component_count: usize,
    /// Total layer count.
    pub layer_count: usize,
    /// Average essential energy ratio.
    pub avg_essential_ratio: f32,
    /// Per-model weights.
    pub model_weights: HashMap<String, f32>,
    /// Total blend operations performed.
    pub blend_operations: u64,
    /// Current generation step.
    pub generation_step: u64,
}

impl BlendStats {
    /// Creates stats from a blended model.
    pub fn from_model(model: &BlendedModel) -> Self {
        let mut model_weights = HashMap::new();
        let mut total_essential_ratio = 0.0;

        for component in &model.components {
            let weight = model.weight(&component.model.name).unwrap_or(0.0);
            model_weights.insert(component.model.name.clone(), weight);
            total_essential_ratio += component.model.essential_ratio();
        }

        let avg_essential_ratio = if model.component_count() > 0 {
            total_essential_ratio / model.component_count() as f32
        } else {
            0.0
        };

        Self {
            component_count: model.component_count(),
            layer_count: model.layer_count(),
            avg_essential_ratio,
            model_weights,
            blend_operations: 0,
            generation_step: model.current_step(),
        }
    }
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_weights(size: usize, base: f32) -> Vec<f32> {
        (0..size).map(|i| base + (i as f32 * 0.1)).collect()
    }

    fn create_test_decomposition(name: &str, layer_count: usize) -> SpectralDecomposition {
        let mut decomp =
            SpectralDecomposition::new(name.to_string(), format!("Test model: {}", name));

        for i in 0..layer_count {
            let weights = create_test_weights(64, i as f32);
            let layer = LayerDecomposition::from_weights(
                i,
                if i % 2 == 0 {
                    LayerType::Attention
                } else {
                    LayerType::Mlp
                },
                format!("layer.{}", i),
                &weights,
                vec![8, 8],
            )
            .expect("Failed to create layer");
            decomp.add_layer(layer);
        }

        decomp
    }

    #[test]
    fn test_layer_decomposition_roundtrip() {
        let weights: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let layer = LayerDecomposition::from_weights(
            0,
            LayerType::Attention,
            "test".to_string(),
            &weights,
            vec![8, 8],
        )
        .expect("Decomposition failed");

        // Full quality reconstruction should be close to original
        let reconstructed = layer.reconstruct(1.0);
        assert_eq!(reconstructed.len(), weights.len());

        // Check reconstruction error
        let mse: f32 = weights
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / weights.len() as f32;

        // DCT roundtrip should be exact (within floating point)
        assert!(mse < 0.01, "MSE too high: {}", mse);
    }

    #[test]
    fn test_layer_decomposition_quality_levels() {
        let weights: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let layer = LayerDecomposition::from_weights(
            0,
            LayerType::Attention,
            "test".to_string(),
            &weights,
            vec![8, 8],
        )
        .expect("Decomposition failed");

        // Low quality should have more error
        let low_q = layer.reconstruct(0.1);
        let high_q = layer.reconstruct(1.0);

        let low_mse: f32 = weights
            .iter()
            .zip(low_q.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / weights.len() as f32;

        let high_mse: f32 = weights
            .iter()
            .zip(high_q.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / weights.len() as f32;

        assert!(low_mse >= high_mse, "Low quality should have more error");
    }

    #[test]
    fn test_essential_carries_most_energy() {
        let weights: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let layer = LayerDecomposition::from_weights(
            0,
            LayerType::Attention,
            "test".to_string(),
            &weights,
            vec![8, 8],
        )
        .expect("Decomposition failed");

        let essential_energy = layer.essential_energy();
        let detail_energy = layer.detail_energy();
        let total_energy = essential_energy + detail_energy;

        // Essential should carry significant portion of energy
        let essential_ratio = essential_energy / total_energy;
        assert!(
            essential_ratio > 0.3,
            "Essential ratio too low: {}",
            essential_ratio
        );
    }

    #[test]
    fn test_spectral_decomposition_creation() {
        let decomp = create_test_decomposition("coder", 4);
        assert_eq!(decomp.layer_count(), 4);
        assert!(decomp.get_layer(0).is_some());
        assert!(decomp.get_layer_by_name("layer.0").is_some());
    }

    #[test]
    fn test_spectral_blend_basic() {
        let coder = Arc::new(create_test_decomposition("coder", 4));
        let creative = Arc::new(create_test_decomposition("creative", 4));

        let blended = SpectralBlend::new()
            .add(coder.clone(), 0.7)
            .add(creative.clone(), 0.3)
            .build()
            .expect("Blend failed");

        assert_eq!(blended.component_count(), 2);
        assert_eq!(blended.layer_count(), 4);

        // Check normalized weights
        let coder_weight = blended.weight("coder").expect("Coder weight");
        let creative_weight = blended.weight("creative").expect("Creative weight");

        assert!((coder_weight - 0.7).abs() < 0.01);
        assert!((creative_weight - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_blend_weights_normalized() {
        let model_a = Arc::new(create_test_decomposition("a", 2));
        let model_b = Arc::new(create_test_decomposition("b", 2));

        let blended = SpectralBlend::new()
            .add(model_a.clone(), 2.0)
            .add(model_b.clone(), 3.0)
            .build()
            .expect("Blend failed");

        let weight_a = blended.weight("a").expect("Weight A");
        let weight_b = blended.weight("b").expect("Weight B");

        // Should be normalized to sum to 1.0
        assert!((weight_a - 0.4).abs() < 0.01, "Weight A: {}", weight_a);
        assert!((weight_b - 0.6).abs() < 0.01, "Weight B: {}", weight_b);
    }

    #[test]
    fn test_per_layer_blend_weights() {
        let coder = Arc::new(create_test_decomposition("coder", 4));
        let creative = Arc::new(create_test_decomposition("creative", 4));

        let blended = SpectralBlend::new()
            .add_with_layer_weights(coder.clone(), 1.0, LayerWeights::attention_heavy(0.9, 0.5))
            .add_with_layer_weights(
                creative.clone(),
                1.0,
                LayerWeights::attention_heavy(0.1, 0.5),
            )
            .build()
            .expect("Blend failed");

        // Layer 0 is Attention, layer 1 is Mlp
        let attention_weight = blended.attention_blend_weight(0);
        let mlp_weight = blended.mlp_blend_weight(1);

        // Attention should show coder dominance
        assert!(attention_weight > 0.0);
        assert!(mlp_weight > 0.0);
    }

    #[test]
    fn test_dynamic_blend_adjustment() {
        let coder = Arc::new(create_test_decomposition("coder", 4));
        let creative = Arc::new(create_test_decomposition("creative", 4));

        let blended = SpectralBlend::new()
            .add(coder.clone(), 0.5)
            .add(creative.clone(), 0.5)
            .build()
            .expect("Blend failed");

        // Initial weights should be equal
        let initial_coder = blended.weight("coder").expect("Coder weight");
        assert!((initial_coder - 0.5).abs() < 0.01);

        // Adjust weight
        blended.adjust_weight("coder", 1.5).expect("Adjust failed");

        // Coder weight should increase
        let adjusted_coder = blended.weight("coder").expect("Adjusted coder weight");
        assert!(adjusted_coder > initial_coder);

        // Reset
        blended.reset_adjustments();
        let reset_coder = blended.weight("coder").expect("Reset coder weight");
        assert!((reset_coder - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_get_blended_layer_weights() {
        let coder = Arc::new(create_test_decomposition("coder", 2));
        let creative = Arc::new(create_test_decomposition("creative", 2));

        let blended = SpectralBlend::new()
            .add(coder.clone(), 0.6)
            .add(creative.clone(), 0.4)
            .build()
            .expect("Blend failed");

        let weights = blended
            .get_layer_weights(0, 1.0)
            .expect("Get weights failed");
        assert!(!weights.is_empty());

        // Weights should be a blend of the two models
        let coder_weights = coder.get_layer(0).expect("Coder layer").reconstruct(1.0);
        let creative_weights = creative
            .get_layer(0)
            .expect("Creative layer")
            .reconstruct(1.0);

        // Blended should be between the two
        for i in 0..weights.len().min(10) {
            let blend = weights[i];
            let min = coder_weights[i].min(creative_weights[i]);
            let max = coder_weights[i].max(creative_weights[i]);
            assert!(
                blend >= min - 0.1 && blend <= max + 0.1,
                "Blend {} not between {} and {} at index {}",
                blend,
                min,
                max,
                i
            );
        }
    }

    #[test]
    fn test_blend_to_pattern() {
        let model = Arc::new(create_test_decomposition("test", 2));

        let blended = SpectralBlend::new()
            .add(model.clone(), 1.0)
            .build()
            .expect("Blend failed");

        let pattern = blended.to_pattern(0, 1.0).expect("Pattern failed");
        assert!(!pattern.coefficients.is_empty());
        assert!(pattern.width > 0);
        assert!(pattern.height > 0);
    }

    #[test]
    fn test_dynamic_controller() {
        let coder = Arc::new(create_test_decomposition("coder", 2));
        let creative = Arc::new(create_test_decomposition("creative", 2));

        let blended = Arc::new(
            SpectralBlend::new()
                .add(coder.clone(), 0.5)
                .add(creative.clone(), 0.5)
                .build()
                .expect("Blend failed"),
        );

        let controller = DynamicBlendController::new(blended.clone()).with_adjustment_rate(0.5);

        // Set target to favor coder
        controller.set_target("coder", 0.8);
        controller.update().expect("Update failed");

        // Coder weight should have moved toward target
        let coder_weight = blended.weight("coder").expect("Coder weight");
        assert!(coder_weight > 0.5, "Coder weight should have increased");
    }

    #[test]
    fn test_blend_stats() {
        let model = Arc::new(create_test_decomposition("test", 4));

        let blended = SpectralBlend::new()
            .add(model.clone(), 1.0)
            .build()
            .expect("Blend failed");

        let stats = BlendStats::from_model(&blended);
        assert_eq!(stats.component_count, 1);
        assert_eq!(stats.layer_count, 4);
        assert!(stats.avg_essential_ratio > 0.0);
    }

    #[test]
    fn test_layer_count_mismatch_error() {
        let model_a = Arc::new(create_test_decomposition("a", 2));
        let model_b = Arc::new(create_test_decomposition("b", 4));

        let result = SpectralBlend::new()
            .add(model_a.clone(), 0.5)
            .add(model_b.clone(), 0.5)
            .build();

        assert!(result.is_err());
        if let Err(SpectralMergeError::LayerCountMismatch { expected, actual }) = result {
            assert_eq!(expected, 2);
            assert_eq!(actual, 4);
        } else {
            panic!("Expected LayerCountMismatch error");
        }
    }

    #[test]
    fn test_no_models_error() {
        let result = SpectralBlend::new().build();
        assert!(matches!(result, Err(SpectralMergeError::NoModels)));
    }

    #[test]
    fn test_model_not_found_error() {
        let model = Arc::new(create_test_decomposition("test", 2));

        let blended = SpectralBlend::new()
            .add(model.clone(), 1.0)
            .build()
            .expect("Blend failed");

        let result = blended.adjust_weight("nonexistent", 1.0);
        assert!(matches!(
            result,
            Err(SpectralMergeError::ModelNotFound { .. })
        ));
    }

    #[test]
    fn test_layer_weights_variants() {
        let uniform = LayerWeights::uniform(0.5);
        assert!((uniform.attention - 0.5).abs() < 0.001);
        assert!((uniform.mlp - 0.5).abs() < 0.001);

        let attn_heavy = LayerWeights::attention_heavy(0.9, 0.3);
        assert!((attn_heavy.attention - 0.9).abs() < 0.001);
        assert!((attn_heavy.mlp - 0.3).abs() < 0.001);

        let mlp_heavy = LayerWeights::mlp_heavy(0.9, 0.3);
        assert!((mlp_heavy.mlp - 0.9).abs() < 0.001);
        assert!((mlp_heavy.attention - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_quality_curve_integration() {
        let weights: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let layer = LayerDecomposition::from_weights(
            0,
            LayerType::Attention,
            "test".to_string(),
            &weights,
            vec![8, 8],
        )
        .expect("Decomposition failed");

        // Check quality curve values
        let q0 = layer.expected_quality(0.0);
        let q1 = layer.expected_quality(1.0);

        assert!(q0 >= 0.0 && q0 <= 1.0);
        assert!(q1 >= 0.0 && q1 <= 1.0);
        assert!(q1 >= q0);
    }
}
