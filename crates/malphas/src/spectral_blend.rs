//! Spectral Model Blending - Runtime model blending for the inference pipeline.
//!
//! This module provides integration between Legion's spectral model merging
//! and the Malphas inference pipeline, enabling runtime blending of multiple
//! models for flexible inference characteristics.
//!
//! ## Features
//!
//! - **Runtime Blending**: Combine models at inference time without disk merging
//! - **Per-Layer Control**: Different blend weights for attention vs MLP layers
//! - **Dynamic Adjustment**: Adjust blend weights mid-generation based on content
//! - **Progressive Quality**: Essential coefficients load first for fast startup
//!
//! ## Example
//!
//! ```ignore
//! use malphas::spectral_blend::{SpectralBlendEngine, SpectralBlendConfig};
//! use infernum_legion::{SpectralDecomposition, SpectralBlend, LayerWeights};
//!
//! let coder = SpectralDecomposition::load("path/to/coder.hct")?;
//! let creative = SpectralDecomposition::load("path/to/creative.hct")?;
//!
//! let engine = SpectralBlendEngine::builder()
//!     .add_model(coder, 0.7)
//!     .add_model(creative, 0.3)
//!     .with_quality(0.9)
//!     .build()?;
//!
//! let response = engine.generate("Write a poem about coding").await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use infernum_legion::{
    BlendStats, BlendedModel, DynamicBlendController, LayerWeights, SpectralBlend,
    SpectralDecomposition, SpectralMergeError,
};

// ==================== Error Types ====================

/// Errors from spectral blend engine operations.
#[derive(Debug, Error)]
pub enum SpectralBlendEngineError {
    /// Spectral merge error.
    #[error("Spectral merge error: {0}")]
    MergeError(#[from] SpectralMergeError),

    /// No models configured.
    #[error("No models configured for blending")]
    NoModels,

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Model loading failed.
    #[error("Failed to load model {name}: {reason}")]
    LoadFailed {
        /// Model name.
        name: String,
        /// Failure reason.
        reason: String,
    },

    /// Generation failed.
    #[error("Generation failed: {0}")]
    GenerationFailed(String),
}

/// Result type for spectral blend engine operations.
pub type Result<T> = std::result::Result<T, SpectralBlendEngineError>;

// ==================== Configuration ====================

/// Configuration for a model in the blend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlendModelConfig {
    /// Model name/identifier.
    pub name: String,
    /// Base weight for this model.
    pub weight: f32,
    /// Per-layer weight overrides.
    pub layer_weights: LayerWeights,
    /// Quality level for this model (0.0 - 1.0).
    pub quality: f32,
}

impl BlendModelConfig {
    /// Creates a new model config with uniform weights.
    pub fn new(name: impl Into<String>, weight: f32) -> Self {
        Self {
            name: name.into(),
            weight,
            layer_weights: LayerWeights::uniform(1.0),
            quality: 1.0,
        }
    }

    /// Sets per-layer weights.
    pub fn with_layer_weights(mut self, weights: LayerWeights) -> Self {
        self.layer_weights = weights;
        self
    }

    /// Sets quality level.
    pub fn with_quality(mut self, quality: f32) -> Self {
        self.quality = quality.clamp(0.0, 1.0);
        self
    }
}

/// Configuration for the spectral blend engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralBlendConfig {
    /// Target quality level for reconstruction (0.0 - 1.0).
    pub quality: f32,
    /// Enable dynamic blend adjustment.
    pub dynamic_adjustment: bool,
    /// Adjustment rate for dynamic blending (0.0 - 1.0).
    pub adjustment_rate: f32,
    /// Decay rate for adjustments (0.0 - 1.0).
    pub decay_rate: f32,
    /// Enable progressive loading (essential first).
    pub progressive_loading: bool,
    /// Minimum quality for generation start.
    pub min_quality: f32,
}

impl Default for SpectralBlendConfig {
    fn default() -> Self {
        Self {
            quality: 0.9,
            dynamic_adjustment: true,
            adjustment_rate: 0.1,
            decay_rate: 0.95,
            progressive_loading: true,
            min_quality: 0.6,
        }
    }
}

impl SpectralBlendConfig {
    /// Creates a high-quality configuration.
    pub fn high_quality() -> Self {
        Self {
            quality: 1.0,
            dynamic_adjustment: true,
            adjustment_rate: 0.05,
            decay_rate: 0.98,
            progressive_loading: false,
            min_quality: 0.9,
        }
    }

    /// Creates a fast startup configuration.
    pub fn fast_startup() -> Self {
        Self {
            quality: 0.7,
            dynamic_adjustment: true,
            adjustment_rate: 0.15,
            decay_rate: 0.9,
            progressive_loading: true,
            min_quality: 0.4,
        }
    }

    /// Creates a balanced configuration.
    pub fn balanced() -> Self {
        Self::default()
    }
}

// ==================== Blend Engine ====================

/// Engine for spectral model blending in the inference pipeline.
///
/// Provides high-level API for creating and using blended models,
/// with support for dynamic adjustment during generation.
pub struct SpectralBlendEngine {
    /// The blended model.
    blended: Arc<BlendedModel>,
    /// Dynamic controller (if enabled).
    controller: Option<DynamicBlendController>,
    /// Configuration.
    config: SpectralBlendConfig,
    /// Model decompositions for reference.
    decompositions: HashMap<String, Arc<SpectralDecomposition>>,
    /// Current quality level.
    current_quality: RwLock<f32>,
    /// Statistics.
    stats: RwLock<SpectralBlendEngineStats>,
}

impl SpectralBlendEngine {
    /// Creates a new builder for the blend engine.
    pub fn builder() -> SpectralBlendEngineBuilder {
        SpectralBlendEngineBuilder::new()
    }

    /// Returns the blended model.
    pub fn blended(&self) -> &Arc<BlendedModel> {
        &self.blended
    }

    /// Returns the configuration.
    pub fn config(&self) -> &SpectralBlendConfig {
        &self.config
    }

    /// Returns current quality level.
    pub fn current_quality(&self) -> f32 {
        *self.current_quality.read()
    }

    /// Returns statistics.
    pub fn stats(&self) -> SpectralBlendEngineStats {
        self.stats.read().clone()
    }

    /// Returns the number of component models.
    pub fn model_count(&self) -> usize {
        self.decompositions.len()
    }

    /// Returns the number of layers.
    pub fn layer_count(&self) -> usize {
        self.blended.layer_count()
    }

    /// Gets the weight for a specific model.
    pub fn model_weight(&self, name: &str) -> Option<f32> {
        self.blended.weight(name)
    }

    /// Gets blended weights for a layer at current quality.
    pub fn get_layer_weights(&self, layer_index: usize) -> Result<Vec<f32>> {
        let quality = self.current_quality();
        self.blended
            .get_layer_weights(layer_index, quality)
            .map_err(SpectralBlendEngineError::MergeError)
    }

    /// Gets blended weights for a layer at specified quality.
    pub fn get_layer_weights_at_quality(
        &self,
        layer_index: usize,
        quality: f32,
    ) -> Result<Vec<f32>> {
        self.blended
            .get_layer_weights(layer_index, quality)
            .map_err(SpectralBlendEngineError::MergeError)
    }

    /// Adjusts the weight of a model dynamically.
    pub fn adjust_weight(&self, model_name: &str, adjustment: f32) -> Result<()> {
        self.blended
            .adjust_weight(model_name, adjustment)
            .map_err(SpectralBlendEngineError::MergeError)?;

        // Update stats
        let mut stats = self.stats.write();
        stats.adjustments += 1;

        Ok(())
    }

    /// Resets all weight adjustments.
    pub fn reset_adjustments(&self) {
        self.blended.reset_adjustments();
    }

    /// Sets target weights for dynamic adjustment.
    pub fn set_targets(&self, targets: HashMap<String, f32>) {
        if let Some(ref controller) = self.controller {
            for (name, weight) in targets {
                controller.set_target(&name, weight);
            }
        }
    }

    /// Clears dynamic targets.
    pub fn clear_targets(&self) {
        if let Some(ref controller) = self.controller {
            controller.clear_targets();
        }
    }

    /// Updates dynamic blend (call after each generation step).
    pub fn update(&self) -> Result<()> {
        if let Some(ref controller) = self.controller {
            controller
                .update()
                .map_err(SpectralBlendEngineError::MergeError)?;
        }

        self.blended.step();

        // Update stats
        let mut stats = self.stats.write();
        stats.generation_steps += 1;

        Ok(())
    }

    /// Increases quality level (for progressive loading).
    pub fn increase_quality(&self, delta: f32) {
        let mut quality = self.current_quality.write();
        *quality = (*quality + delta).min(self.config.quality);
    }

    /// Sets quality level.
    pub fn set_quality(&self, quality: f32) {
        let mut q = self.current_quality.write();
        *q = quality.clamp(0.0, 1.0);
    }

    /// Returns blend statistics.
    pub fn blend_stats(&self) -> BlendStats {
        BlendStats::from_model(&self.blended)
    }

    /// Checks if a model exists in the blend.
    pub fn has_model(&self, name: &str) -> bool {
        self.decompositions.contains_key(name)
    }

    /// Returns all model names.
    pub fn model_names(&self) -> Vec<String> {
        self.decompositions.keys().cloned().collect()
    }
}

impl std::fmt::Debug for SpectralBlendEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpectralBlendEngine")
            .field("model_count", &self.model_count())
            .field("layer_count", &self.layer_count())
            .field("current_quality", &self.current_quality())
            .field("config", &self.config)
            .finish()
    }
}

// ==================== Engine Builder ====================

/// Builder for creating a spectral blend engine.
#[derive(Debug, Default)]
pub struct SpectralBlendEngineBuilder {
    /// Model decompositions with configs.
    models: Vec<(Arc<SpectralDecomposition>, BlendModelConfig)>,
    /// Engine configuration.
    config: SpectralBlendConfig,
}

impl SpectralBlendEngineBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a model with uniform weight.
    pub fn add_model(mut self, model: Arc<SpectralDecomposition>, weight: f32) -> Self {
        let config = BlendModelConfig::new(&model.name, weight);
        self.models.push((model, config));
        self
    }

    /// Adds a model with full configuration.
    pub fn add_model_with_config(
        mut self,
        model: Arc<SpectralDecomposition>,
        config: BlendModelConfig,
    ) -> Self {
        self.models.push((model, config));
        self
    }

    /// Adds a model with per-layer weights.
    pub fn add_model_with_layers(
        mut self,
        model: Arc<SpectralDecomposition>,
        weight: f32,
        layer_weights: LayerWeights,
    ) -> Self {
        let config = BlendModelConfig::new(&model.name, weight).with_layer_weights(layer_weights);
        self.models.push((model, config));
        self
    }

    /// Sets the target quality.
    pub fn with_quality(mut self, quality: f32) -> Self {
        self.config.quality = quality.clamp(0.0, 1.0);
        self
    }

    /// Sets whether to enable dynamic adjustment.
    pub fn with_dynamic_adjustment(mut self, enable: bool) -> Self {
        self.config.dynamic_adjustment = enable;
        self
    }

    /// Sets the adjustment rate.
    pub fn with_adjustment_rate(mut self, rate: f32) -> Self {
        self.config.adjustment_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Sets the full configuration.
    pub fn with_config(mut self, config: SpectralBlendConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets progressive loading.
    pub fn with_progressive_loading(mut self, enable: bool) -> Self {
        self.config.progressive_loading = enable;
        self
    }

    /// Builds the engine.
    pub fn build(self) -> Result<SpectralBlendEngine> {
        if self.models.is_empty() {
            return Err(SpectralBlendEngineError::NoModels);
        }

        // Build the spectral blend
        let mut blend_builder = SpectralBlend::new();
        let mut decompositions = HashMap::new();

        for (model, config) in &self.models {
            blend_builder = blend_builder.add_with_layer_weights(
                model.clone(),
                config.weight,
                config.layer_weights.clone(),
            );
            decompositions.insert(model.name.clone(), model.clone());
        }

        let blended = Arc::new(blend_builder.build()?);

        // Create dynamic controller if enabled
        let controller = if self.config.dynamic_adjustment {
            Some(
                DynamicBlendController::new(blended.clone())
                    .with_adjustment_rate(self.config.adjustment_rate)
                    .with_decay_rate(self.config.decay_rate),
            )
        } else {
            None
        };

        // Set initial quality
        let initial_quality = if self.config.progressive_loading {
            self.config.min_quality
        } else {
            self.config.quality
        };

        Ok(SpectralBlendEngine {
            blended,
            controller,
            config: self.config,
            decompositions,
            current_quality: RwLock::new(initial_quality),
            stats: RwLock::new(SpectralBlendEngineStats::default()),
        })
    }
}

// ==================== Statistics ====================

/// Statistics for the spectral blend engine.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpectralBlendEngineStats {
    /// Number of generation steps.
    pub generation_steps: u64,
    /// Number of weight adjustments.
    pub adjustments: u64,
    /// Number of layer weight retrievals.
    pub layer_retrievals: u64,
    /// Total blending time (microseconds).
    pub blend_time_us: u64,
}

// ==================== Blend Presets ====================

/// Preset blend configurations for common use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlendPreset {
    /// Code-focused blend (70% coder, 30% general).
    CodeFocused,
    /// Creative writing blend (30% code, 70% creative).
    CreativeFocused,
    /// Balanced blend (50/50).
    Balanced,
    /// Instruction-following blend (attention from instruct).
    InstructFollowing,
    /// Custom blend (use with `with_*` methods).
    Custom,
}

impl BlendPreset {
    /// Returns layer weights for this preset.
    pub fn layer_weights(&self) -> (LayerWeights, LayerWeights) {
        match self {
            BlendPreset::CodeFocused => (LayerWeights::uniform(0.7), LayerWeights::uniform(0.3)),
            BlendPreset::CreativeFocused => {
                (LayerWeights::uniform(0.3), LayerWeights::uniform(0.7))
            },
            BlendPreset::Balanced => (LayerWeights::uniform(0.5), LayerWeights::uniform(0.5)),
            BlendPreset::InstructFollowing => (
                LayerWeights::attention_heavy(0.3, 0.5),
                LayerWeights::attention_heavy(0.7, 0.5),
            ),
            BlendPreset::Custom => (LayerWeights::default(), LayerWeights::default()),
        }
    }

    /// Returns base weights for this preset.
    pub fn base_weights(&self) -> (f32, f32) {
        match self {
            BlendPreset::CodeFocused => (0.7, 0.3),
            BlendPreset::CreativeFocused => (0.3, 0.7),
            BlendPreset::Balanced => (0.5, 0.5),
            BlendPreset::InstructFollowing => (0.5, 0.5),
            BlendPreset::Custom => (0.5, 0.5),
        }
    }
}

// ==================== Blend Manager ====================

/// Manager for multiple blend configurations.
///
/// Allows switching between different blend presets during runtime.
pub struct BlendManager {
    /// Active engine.
    active: RwLock<Option<Arc<SpectralBlendEngine>>>,
    /// Available model decompositions.
    models: RwLock<HashMap<String, Arc<SpectralDecomposition>>>,
    /// Named blend configurations.
    presets: RwLock<HashMap<String, SpectralBlendConfig>>,
}

impl BlendManager {
    /// Creates a new blend manager.
    pub fn new() -> Self {
        Self {
            active: RwLock::new(None),
            models: RwLock::new(HashMap::new()),
            presets: RwLock::new(HashMap::new()),
        }
    }

    /// Registers a model decomposition.
    pub fn register_model(&self, model: Arc<SpectralDecomposition>) {
        let mut models = self.models.write();
        models.insert(model.name.clone(), model);
    }

    /// Registers a named preset.
    pub fn register_preset(&self, name: impl Into<String>, config: SpectralBlendConfig) {
        let mut presets = self.presets.write();
        presets.insert(name.into(), config);
    }

    /// Creates a blend engine from registered models.
    pub fn create_blend(
        &self,
        model_weights: Vec<(String, f32)>,
        config: SpectralBlendConfig,
    ) -> Result<Arc<SpectralBlendEngine>> {
        let models = self.models.read();

        let mut builder = SpectralBlendEngine::builder().with_config(config);

        for (name, weight) in model_weights {
            let model = models
                .get(&name)
                .ok_or_else(|| SpectralBlendEngineError::LoadFailed {
                    name: name.clone(),
                    reason: "Model not registered".to_string(),
                })?;
            builder = builder.add_model(model.clone(), weight);
        }

        let engine = Arc::new(builder.build()?);

        // Set as active
        let mut active = self.active.write();
        *active = Some(engine.clone());

        Ok(engine)
    }

    /// Creates a blend using a named preset.
    pub fn create_blend_with_preset(
        &self,
        model_weights: Vec<(String, f32)>,
        preset_name: &str,
    ) -> Result<Arc<SpectralBlendEngine>> {
        let presets = self.presets.read();
        let config = presets.get(preset_name).cloned().unwrap_or_default();

        drop(presets);
        self.create_blend(model_weights, config)
    }

    /// Returns the active blend engine.
    pub fn active(&self) -> Option<Arc<SpectralBlendEngine>> {
        self.active.read().clone()
    }

    /// Returns all registered model names.
    pub fn model_names(&self) -> Vec<String> {
        self.models.read().keys().cloned().collect()
    }

    /// Returns all preset names.
    pub fn preset_names(&self) -> Vec<String> {
        self.presets.read().keys().cloned().collect()
    }
}

impl Default for BlendManager {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for BlendManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlendManager")
            .field("model_count", &self.models.read().len())
            .field("preset_count", &self.presets.read().len())
            .field("has_active", &self.active.read().is_some())
            .finish()
    }
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;
    use infernum_legion::{LayerDecomposition, LayerType};

    fn create_test_decomposition(name: &str, layer_count: usize) -> SpectralDecomposition {
        let mut decomp =
            SpectralDecomposition::new(name.to_string(), format!("Test model: {}", name));

        for i in 0..layer_count {
            let weights: Vec<f32> = (0..64).map(|j| (i * 10 + j) as f32 * 0.1).collect();
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
    fn test_blend_engine_creation() {
        let coder = Arc::new(create_test_decomposition("coder", 4));
        let creative = Arc::new(create_test_decomposition("creative", 4));

        let engine = SpectralBlendEngine::builder()
            .add_model(coder, 0.7)
            .add_model(creative, 0.3)
            .build()
            .expect("Build failed");

        assert_eq!(engine.model_count(), 2);
        assert_eq!(engine.layer_count(), 4);
    }

    #[test]
    fn test_blend_engine_weights() {
        let coder = Arc::new(create_test_decomposition("coder", 2));
        let creative = Arc::new(create_test_decomposition("creative", 2));

        let engine = SpectralBlendEngine::builder()
            .add_model(coder, 0.6)
            .add_model(creative, 0.4)
            .build()
            .expect("Build failed");

        let coder_weight = engine.model_weight("coder").expect("Coder weight");
        let creative_weight = engine.model_weight("creative").expect("Creative weight");

        assert!((coder_weight - 0.6).abs() < 0.01);
        assert!((creative_weight - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_blend_engine_layer_weights() {
        let model = Arc::new(create_test_decomposition("test", 2));

        let engine = SpectralBlendEngine::builder()
            .add_model(model, 1.0)
            .with_quality(1.0)
            .build()
            .expect("Build failed");

        let weights = engine.get_layer_weights(0).expect("Get weights failed");
        assert!(!weights.is_empty());
    }

    #[test]
    fn test_blend_engine_quality() {
        let model = Arc::new(create_test_decomposition("test", 2));

        let engine = SpectralBlendEngine::builder()
            .add_model(model, 1.0)
            .with_quality(0.8)
            .with_progressive_loading(false) // Disable progressive to test direct quality
            .build()
            .expect("Build failed");

        assert!((engine.current_quality() - 0.8).abs() < 0.01);

        engine.set_quality(0.5);
        assert!((engine.current_quality() - 0.5).abs() < 0.01);

        engine.increase_quality(0.2);
        assert!((engine.current_quality() - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_blend_engine_progressive_loading() {
        let model = Arc::new(create_test_decomposition("test", 2));

        let engine = SpectralBlendEngine::builder()
            .add_model(model, 1.0)
            .with_progressive_loading(true)
            .with_config(SpectralBlendConfig {
                quality: 0.9,
                min_quality: 0.4,
                progressive_loading: true,
                ..Default::default()
            })
            .build()
            .expect("Build failed");

        // Should start at min_quality
        assert!((engine.current_quality() - 0.4).abs() < 0.01);

        // Increase quality
        engine.increase_quality(0.3);
        assert!((engine.current_quality() - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_blend_engine_adjustment() {
        let coder = Arc::new(create_test_decomposition("coder", 2));
        let creative = Arc::new(create_test_decomposition("creative", 2));

        let engine = SpectralBlendEngine::builder()
            .add_model(coder, 0.5)
            .add_model(creative, 0.5)
            .build()
            .expect("Build failed");

        // Adjust weight
        engine.adjust_weight("coder", 1.5).expect("Adjust failed");

        let coder_weight = engine.model_weight("coder").expect("Coder weight");
        assert!(coder_weight > 0.5);

        // Reset
        engine.reset_adjustments();
        let reset_weight = engine.model_weight("coder").expect("Reset weight");
        assert!((reset_weight - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_blend_engine_update() {
        let model = Arc::new(create_test_decomposition("test", 2));

        let engine = SpectralBlendEngine::builder()
            .add_model(model, 1.0)
            .with_dynamic_adjustment(true)
            .build()
            .expect("Build failed");

        engine.update().expect("Update failed");
        engine.update().expect("Update failed");

        let stats = engine.stats();
        assert_eq!(stats.generation_steps, 2);
    }

    #[test]
    fn test_blend_config_presets() {
        let default = SpectralBlendConfig::default();
        assert!((default.quality - 0.9).abs() < 0.01);

        let high = SpectralBlendConfig::high_quality();
        assert!((high.quality - 1.0).abs() < 0.01);

        let fast = SpectralBlendConfig::fast_startup();
        assert!(fast.progressive_loading);
    }

    #[test]
    fn test_blend_preset_weights() {
        let (w1, w2) = BlendPreset::CodeFocused.base_weights();
        assert!((w1 - 0.7).abs() < 0.01);
        assert!((w2 - 0.3).abs() < 0.01);

        let (w1, w2) = BlendPreset::Balanced.base_weights();
        assert!((w1 - 0.5).abs() < 0.01);
        assert!((w2 - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_blend_manager_creation() {
        let manager = BlendManager::new();
        assert!(manager.active().is_none());
        assert!(manager.model_names().is_empty());
    }

    #[test]
    fn test_blend_manager_register_model() {
        let manager = BlendManager::new();
        let model = Arc::new(create_test_decomposition("test", 2));

        manager.register_model(model);
        assert_eq!(manager.model_names().len(), 1);
        assert!(manager.model_names().contains(&"test".to_string()));
    }

    #[test]
    fn test_blend_manager_create_blend() {
        let manager = BlendManager::new();

        let coder = Arc::new(create_test_decomposition("coder", 2));
        let creative = Arc::new(create_test_decomposition("creative", 2));

        manager.register_model(coder);
        manager.register_model(creative);

        let engine = manager
            .create_blend(
                vec![("coder".to_string(), 0.6), ("creative".to_string(), 0.4)],
                SpectralBlendConfig::default(),
            )
            .expect("Create blend failed");

        assert_eq!(engine.model_count(), 2);
        assert!(manager.active().is_some());
    }

    #[test]
    fn test_blend_manager_preset() {
        let manager = BlendManager::new();

        manager.register_preset("fast", SpectralBlendConfig::fast_startup());
        assert!(manager.preset_names().contains(&"fast".to_string()));
    }

    #[test]
    fn test_no_models_error() {
        let result = SpectralBlendEngine::builder().build();
        assert!(matches!(result, Err(SpectralBlendEngineError::NoModels)));
    }

    #[test]
    fn test_model_not_found_error() {
        let manager = BlendManager::new();

        let result = manager.create_blend(
            vec![("nonexistent".to_string(), 1.0)],
            SpectralBlendConfig::default(),
        );

        assert!(matches!(
            result,
            Err(SpectralBlendEngineError::LoadFailed { .. })
        ));
    }

    #[test]
    fn test_blend_model_config() {
        let config = BlendModelConfig::new("test", 0.5)
            .with_quality(0.8)
            .with_layer_weights(LayerWeights::attention_heavy(0.9, 0.5));

        assert_eq!(config.name, "test");
        assert!((config.weight - 0.5).abs() < 0.01);
        assert!((config.quality - 0.8).abs() < 0.01);
        assert!((config.layer_weights.attention - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_engine_model_names() {
        let coder = Arc::new(create_test_decomposition("coder", 2));
        let creative = Arc::new(create_test_decomposition("creative", 2));

        let engine = SpectralBlendEngine::builder()
            .add_model(coder, 0.5)
            .add_model(creative, 0.5)
            .build()
            .expect("Build failed");

        let names = engine.model_names();
        assert!(names.contains(&"coder".to_string()));
        assert!(names.contains(&"creative".to_string()));
    }

    #[test]
    fn test_engine_has_model() {
        let model = Arc::new(create_test_decomposition("test", 2));

        let engine = SpectralBlendEngine::builder()
            .add_model(model, 1.0)
            .build()
            .expect("Build failed");

        assert!(engine.has_model("test"));
        assert!(!engine.has_model("nonexistent"));
    }

    #[test]
    fn test_blend_stats() {
        let model = Arc::new(create_test_decomposition("test", 2));

        let engine = SpectralBlendEngine::builder()
            .add_model(model, 1.0)
            .build()
            .expect("Build failed");

        let stats = engine.blend_stats();
        assert_eq!(stats.component_count, 1);
        assert_eq!(stats.layer_count, 2);
    }
}
