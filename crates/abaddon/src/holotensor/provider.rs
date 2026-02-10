//! Progressive Weight Provider for Holotensor Inference
//!
//! Provides model weights with progressive quality reconstruction, coordinating
//! the memory manager and streaming system for zero-latency weight delivery.
//!
//! ## Core Innovation
//!
//! The provider exploits holographic properties to enable immediate inference
//! at reduced quality (70%) while streaming improves quality to target (95%+).
//!
//! ```text
//! Quality Timeline During Generation:
//!
//! 100% ─────────────────────────────────────────●──●──●──●──●
//!                                           ●
//!                                       ●
//!  95% ────────────────────────────●────────────────────────── Target
//!                             ●
//!                         ●
//!  85% ────────────────●────────────────────────────────────── Good
//!                  ●
//!              ●
//!  70% ─────●────────────────────────────────────────────────── Min (start)
//!       │                                                    │
//!       Token 1                                           Token N
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use haagenti::compressive::CompressiveSpectralDecoder;
use haagenti::holotensor::{
    HoloFragment, HoloTensorHeader, HolographicEncoding, LrdfDecoder, QualityCurve,
};

use super::memory::{FragmentId, HoloMemoryManager};
use super::streaming::{StreamManager, StreamPriority, StreamRequest};
use super::{HoloInferenceConfig, HoloInferenceError, HoloModelMetadata, Result};

/// Weight types in transformer architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum WeightType {
    /// Query projection (attention).
    QProj = 0,
    /// Key projection (attention).
    KProj = 1,
    /// Value projection (attention).
    VProj = 2,
    /// Output projection (attention).
    OProj = 3,
    /// Gate projection (MLP).
    GateProj = 4,
    /// Up projection (MLP).
    UpProj = 5,
    /// Down projection (MLP).
    DownProj = 6,
    /// Layer normalization.
    LayerNorm = 7,
    /// Embedding.
    Embed = 8,
    /// LM head.
    LmHead = 9,
}

impl WeightType {
    /// Get importance weight for priority calculation.
    pub fn importance(&self) -> f32 {
        match self {
            WeightType::QProj => 1.0,
            WeightType::KProj => 0.8,
            WeightType::VProj => 0.8,
            WeightType::OProj => 0.9,
            WeightType::GateProj => 0.85,
            WeightType::UpProj => 0.85,
            WeightType::DownProj => 0.85,
            WeightType::LayerNorm => 0.5, // Small, less critical
            WeightType::Embed => 1.0,
            WeightType::LmHead => 1.0,
        }
    }
}

/// Quality metrics for a layer.
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Layer index.
    pub layer: usize,
    /// Current quality (0.0 - 1.0).
    pub current_quality: f32,
    /// Target quality.
    pub target_quality: f32,
    /// Fragments loaded.
    pub fragments_loaded: usize,
    /// Total fragments.
    pub total_fragments: usize,
    /// Whether quality target is reached.
    pub target_reached: bool,
    /// Time spent at reduced quality (ms).
    pub reduced_quality_time_ms: u64,
    /// Quality curve for this encoding type.
    quality_curve: QualityCurve,
}

impl QualityMetrics {
    /// Create new metrics for a layer with default LRDF encoding.
    pub fn new(layer: usize, total_fragments: usize, target_quality: f32) -> Self {
        Self::with_encoding(
            layer,
            total_fragments,
            target_quality,
            HolographicEncoding::LowRankDistributed,
        )
    }

    /// Create new metrics for a layer with specific encoding type.
    pub fn with_encoding(
        layer: usize,
        total_fragments: usize,
        target_quality: f32,
        encoding: HolographicEncoding,
    ) -> Self {
        Self {
            layer,
            current_quality: 0.0,
            target_quality,
            fragments_loaded: 0,
            total_fragments,
            target_reached: false,
            reduced_quality_time_ms: 0,
            quality_curve: encoding.default_quality_curve(),
        }
    }

    /// Create new metrics with a custom quality curve.
    pub fn with_curve(
        layer: usize,
        total_fragments: usize,
        target_quality: f32,
        quality_curve: QualityCurve,
    ) -> Self {
        Self {
            layer,
            current_quality: 0.0,
            target_quality,
            fragments_loaded: 0,
            total_fragments,
            target_reached: false,
            reduced_quality_time_ms: 0,
            quality_curve,
        }
    }

    /// Calculate quality from fragment count using the configured quality curve.
    ///
    /// Uses Haagenti's polynomial QualityCurve instead of simple sqrt(k/N).
    /// This provides more accurate quality prediction based on the encoding type.
    pub fn quality_from_fragments(&self, loaded: usize, total: usize) -> f32 {
        self.quality_curve.predict(loaded as u16, total as u16)
    }

    /// Calculate quality using default LRDF curve (for backwards compatibility).
    pub fn quality_from_fragments_default(loaded: usize, total: usize) -> f32 {
        HolographicEncoding::LowRankDistributed
            .default_quality_curve()
            .predict(loaded as u16, total as u16)
    }

    /// Update quality from fragment count.
    pub fn update_from_fragments(&mut self, loaded: usize) {
        self.fragments_loaded = loaded;
        self.current_quality = self.quality_from_fragments(loaded, self.total_fragments);
        self.target_reached = self.current_quality >= self.target_quality;
    }

    /// Get the minimum fragments needed to reach target quality.
    pub fn fragments_for_target(&self) -> usize {
        self.quality_curve
            .fragments_for_quality(self.target_quality, self.total_fragments as u16)
            as usize
    }

    /// Get the quality curve being used.
    pub fn quality_curve(&self) -> &QualityCurve {
        &self.quality_curve
    }
}

/// Reconstructed layer weights.
#[derive(Debug)]
pub struct LayerWeights {
    /// Layer index.
    pub layer: usize,
    /// Weight data (f32 tensor).
    pub data: Vec<f32>,
    /// Shape (rows, cols).
    pub shape: (usize, usize),
    /// Weight type.
    pub weight_type: WeightType,
    /// Current reconstruction quality.
    pub quality: f32,
    /// Is this a partial reconstruction?
    pub is_partial: bool,
}

impl LayerWeights {
    /// Create new layer weights.
    pub fn new(
        layer: usize,
        data: Vec<f32>,
        shape: (usize, usize),
        weight_type: WeightType,
        quality: f32,
    ) -> Self {
        Self {
            layer,
            data,
            shape,
            weight_type,
            quality,
            is_partial: quality < 0.999,
        }
    }
}

/// Progressive weight provider state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderState {
    /// Not initialized.
    Uninitialized,
    /// Loading initial fragments.
    Loading,
    /// Ready for inference at minimum quality.
    Ready,
    /// Streaming to improve quality.
    Streaming,
    /// Target quality reached.
    Complete,
    /// Error state.
    Error,
}

/// Progressive weight provider for holotensor inference.
///
/// Coordinates memory management and streaming to provide weights with
/// progressive quality improvement during inference.
pub struct ProgressiveWeightProvider {
    config: HoloInferenceConfig,
    memory: Arc<HoloMemoryManager>,
    stream: Arc<StreamManager>,

    /// Model metadata.
    metadata: Option<HoloModelMetadata>,

    /// Fragment data storage (RAM cache).
    fragment_cache: RwLock<HashMap<FragmentId, HoloFragment>>,

    /// Tensor headers per layer/weight.
    headers: RwLock<HashMap<(usize, WeightType), HoloTensorHeader>>,

    /// Quality metrics per layer.
    quality: RwLock<HashMap<usize, QualityMetrics>>,

    /// Current provider state.
    state: RwLock<ProviderState>,

    /// Current layer being processed.
    current_layer: AtomicUsize,

    /// Total layers.
    num_layers: AtomicUsize,

    /// Tokens generated (for adaptive streaming).
    tokens_generated: AtomicUsize,

    /// Time when inference started.
    inference_start: RwLock<Option<Instant>>,

    /// Layers that have reached target quality.
    layers_complete: RwLock<Vec<usize>>,
}

impl ProgressiveWeightProvider {
    /// Create new progressive weight provider.
    pub fn new(
        config: HoloInferenceConfig,
        memory: Arc<HoloMemoryManager>,
        stream: Arc<StreamManager>,
    ) -> Self {
        Self {
            config,
            memory,
            stream,
            metadata: None,
            fragment_cache: RwLock::new(HashMap::new()),
            headers: RwLock::new(HashMap::new()),
            quality: RwLock::new(HashMap::new()),
            state: RwLock::new(ProviderState::Uninitialized),
            current_layer: AtomicUsize::new(0),
            num_layers: AtomicUsize::new(0),
            tokens_generated: AtomicUsize::new(0),
            inference_start: RwLock::new(None),
            layers_complete: RwLock::new(Vec::new()),
        }
    }

    /// Set model metadata.
    pub fn set_metadata(&mut self, metadata: HoloModelMetadata) {
        self.num_layers
            .store(metadata.num_layers, Ordering::Relaxed);
        self.metadata = Some(metadata);
    }

    /// Get current state.
    pub fn state(&self) -> ProviderState {
        *self.state.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Set state.
    fn set_state(&self, state: ProviderState) {
        if let Ok(mut s) = self.state.write() {
            *s = state;
        }
    }

    /// Register a tensor header.
    pub fn register_header(
        &self,
        layer: usize,
        weight_type: WeightType,
        header: HoloTensorHeader,
    ) -> Result<()> {
        let mut headers = self
            .headers
            .write()
            .map_err(|_| HoloInferenceError::Haagenti("headers lock poisoned".to_string()))?;
        headers.insert((layer, weight_type), header);

        // Initialize quality metrics
        let num_fragments = self.config.num_fragments as usize;
        let mut quality = self
            .quality
            .write()
            .map_err(|_| HoloInferenceError::Haagenti("quality lock poisoned".to_string()))?;

        if !quality.contains_key(&layer) {
            quality.insert(
                layer,
                QualityMetrics::new(layer, num_fragments, self.config.target_quality),
            );
        }

        Ok(())
    }

    /// Add a fragment to the cache.
    pub fn add_fragment(&self, id: FragmentId, fragment: HoloFragment) -> Result<()> {
        let mut cache = self.fragment_cache.write().map_err(|_| {
            HoloInferenceError::Haagenti("fragment cache lock poisoned".to_string())
        })?;
        cache.insert(id, fragment);

        // Update quality metrics
        self.update_layer_quality(id.layer)?;

        Ok(())
    }

    /// Update quality metrics for a layer.
    fn update_layer_quality(&self, layer: usize) -> Result<()> {
        let cache = self.fragment_cache.read().map_err(|_| {
            HoloInferenceError::Haagenti("fragment cache lock poisoned".to_string())
        })?;

        // Count fragments for this layer
        let loaded = cache.keys().filter(|id| id.layer == layer).count();

        let mut quality = self
            .quality
            .write()
            .map_err(|_| HoloInferenceError::Haagenti("quality lock poisoned".to_string()))?;

        if let Some(metrics) = quality.get_mut(&layer) {
            metrics.update_from_fragments(loaded);

            // Track completion
            if metrics.target_reached {
                if let Ok(mut complete) = self.layers_complete.write() {
                    if !complete.contains(&layer) {
                        complete.push(layer);
                    }
                }
            }
        }

        Ok(())
    }

    /// Get quality metrics for a layer.
    pub fn get_quality(&self, layer: usize) -> Option<QualityMetrics> {
        self.quality.read().ok()?.get(&layer).cloned()
    }

    /// Get overall quality (average across all layers).
    pub fn overall_quality(&self) -> f32 {
        let quality = match self.quality.read() {
            Ok(q) => q,
            Err(_) => return 0.0,
        };

        if quality.is_empty() {
            return 0.0;
        }

        let sum: f32 = quality.values().map(|m| m.current_quality).sum();
        sum / quality.len() as f32
    }

    /// Check if minimum quality is reached for inference.
    pub fn ready_for_inference(&self) -> bool {
        self.overall_quality() >= self.config.min_quality
    }

    /// Check if target quality is reached.
    pub fn target_reached(&self) -> bool {
        self.overall_quality() >= self.config.target_quality
    }

    /// Get weights for a layer with progressive quality.
    ///
    /// This is the key method that provides weights to inference.
    /// It reconstructs from available fragments, even if partial.
    pub fn get_weights(&self, layer: usize, weight_type: WeightType) -> Result<LayerWeights> {
        // Get header
        let headers = self
            .headers
            .read()
            .map_err(|_| HoloInferenceError::Haagenti("headers lock poisoned".to_string()))?;

        let header = headers.get(&(layer, weight_type)).ok_or_else(|| {
            HoloInferenceError::FragmentNotFound {
                layer,
                fragment_index: 0,
            }
        })?;

        // Get fragments for this layer/weight
        let cache = self.fragment_cache.read().map_err(|_| {
            HoloInferenceError::Haagenti("fragment cache lock poisoned".to_string())
        })?;

        let mut fragments: Vec<&HoloFragment> = cache
            .iter()
            .filter(|(id, _)| id.layer == layer && id.weight_type == weight_type as u8)
            .map(|(_, frag)| frag)
            .collect();

        if fragments.is_empty() {
            return Err(HoloInferenceError::FragmentNotFound {
                layer,
                fragment_index: 0,
            });
        }

        // Sort by fragment index (lower = higher singular values = more important)
        fragments.sort_by_key(|f| f.index);

        // Get shape from header
        let shape = &header.shape;
        let rows = shape.get(0).copied().unwrap_or(1) as usize;
        let cols = shape.get(1).copied().unwrap_or(1) as usize;

        // Reconstruct using the appropriate decoder based on encoding type
        let data = match header.encoding {
            HolographicEncoding::Spectral => {
                // Use CompressiveSpectralDecoder for HCT3 format
                let mut decoder = CompressiveSpectralDecoder::new();

                for fragment in &fragments {
                    if fragment.index == 0 {
                        decoder.add_essentials(fragment).map_err(|e| {
                            HoloInferenceError::Haagenti(format!("Add essentials error: {}", e))
                        })?;
                    } else {
                        decoder.add_detail(fragment).map_err(|e| {
                            HoloInferenceError::Haagenti(format!("Add detail error: {}", e))
                        })?;
                    }
                }
                decoder.reconstruct().map_err(|e| {
                    HoloInferenceError::Haagenti(format!("Spectral reconstruct error: {}", e))
                })?
            },
            HolographicEncoding::LowRankDistributed | _ => {
                // Use LrdfDecoder for LRDF format
                // CRITICAL: Use header.total_fragments, NOT config.num_fragments!
                // Passthrough tensors have total_fragments=1, not the config default.
                let mut decoder = LrdfDecoder::new(rows, cols, header.total_fragments);
                for fragment in &fragments {
                    decoder.add_fragment(fragment).map_err(|e| {
                        HoloInferenceError::Haagenti(format!("Decoder error: {}", e))
                    })?;
                }
                decoder.reconstruct()
            },
        };

        // Calculate quality using the correct curve for the encoding
        // Use header.total_fragments, NOT config.num_fragments!
        let quality = header
            .encoding
            .default_quality_curve()
            .predict(fragments.len() as u16, header.total_fragments);

        // Update touch for LRU
        for (id, _) in cache.iter() {
            if id.layer == layer && id.weight_type == weight_type as u8 {
                self.memory.touch(id);
            }
        }

        // Request streaming for remaining fragments
        // Use header.total_fragments to check actual fragment count
        if self.config.enable_streaming && fragments.len() < header.total_fragments as usize {
            self.request_remaining_fragments(
                layer,
                weight_type,
                fragments.len(),
                header.total_fragments,
            )?;
        }

        Ok(LayerWeights::new(
            layer,
            data,
            (rows, cols),
            weight_type,
            quality,
        ))
    }

    /// Request streaming for remaining fragments.
    fn request_remaining_fragments(
        &self,
        layer: usize,
        weight_type: WeightType,
        already_loaded: usize,
        total_fragments: u16,
    ) -> Result<()> {
        let total = total_fragments as usize;

        for i in already_loaded..total {
            let id = FragmentId::new(layer, weight_type as u8, i as u16);
            if self.stream.needs_streaming(&id) {
                let priority = if i < total / 2 {
                    StreamPriority::High
                } else {
                    StreamPriority::Normal
                };
                self.stream.submit(StreamRequest::new(id, priority))?;
            }
        }

        Ok(())
    }

    /// Notify that we're about to process a layer.
    ///
    /// Triggers prefetching for upcoming layers.
    pub fn notify_layer_start(&self, layer: usize) -> Result<()> {
        self.current_layer.store(layer, Ordering::Relaxed);
        self.stream.set_current_layer(layer);

        // Prefetch next 2 layers
        if self.config.enable_streaming {
            let num_layers = self.num_layers.load(Ordering::Relaxed);
            if layer + 1 < num_layers {
                self.stream.prefetch_layers(
                    layer + 1,
                    2.min(num_layers - layer - 1),
                    self.config.num_fragments,
                )?;
            }
        }

        Ok(())
    }

    /// Notify that a layer is complete.
    pub fn notify_layer_complete(&self, _layer: usize) {
        // Layer processing done, could trigger cleanup or reprioritization
    }

    /// Notify token generation (for adaptive streaming).
    pub fn notify_token_generated(&self) {
        self.tokens_generated.fetch_add(1, Ordering::Relaxed);
    }

    /// Start inference session.
    pub fn start_inference(&self) {
        if let Ok(mut start) = self.inference_start.write() {
            *start = Some(Instant::now());
        }
        self.set_state(ProviderState::Ready);
        self.stream.resume();
    }

    /// End inference session.
    pub fn end_inference(&self) {
        self.stream.pause();
        if self.target_reached() {
            self.set_state(ProviderState::Complete);
        }
    }

    /// Get inference duration.
    pub fn inference_duration(&self) -> Option<Duration> {
        self.inference_start
            .read()
            .ok()?
            .map(|start| start.elapsed())
    }

    /// Get tokens generated.
    pub fn tokens_generated(&self) -> usize {
        self.tokens_generated.load(Ordering::Relaxed)
    }

    /// Calculate tokens per second.
    pub fn tokens_per_second(&self) -> f64 {
        let tokens = self.tokens_generated() as f64;
        let duration = self
            .inference_duration()
            .map(|d| d.as_secs_f64())
            .unwrap_or(1.0);
        tokens / duration
    }

    /// Get layers that have reached target quality.
    pub fn completed_layers(&self) -> Vec<usize> {
        self.layers_complete
            .read()
            .map(|v| v.clone())
            .unwrap_or_default()
    }

    /// Get current layer being processed.
    pub fn get_current_layer(&self) -> usize {
        self.current_layer.load(Ordering::Relaxed)
    }

    /// Set current layer (for testing).
    #[cfg(test)]
    pub fn set_current_layer_for_test(&self, layer: usize) {
        self.current_layer.store(layer, Ordering::Relaxed);
    }

    /// Calculate priority for a layer based on distance from current.
    pub fn layer_priority(&self, layer: usize) -> StreamPriority {
        let current = self.current_layer.load(Ordering::Relaxed);
        let distance = if layer >= current {
            layer - current
        } else {
            usize::MAX // Already processed
        };

        match distance {
            0 => StreamPriority::Critical,
            1 => StreamPriority::High,
            2..=4 => StreamPriority::Normal,
            _ => StreamPriority::Low,
        }
    }

    /// Get streaming statistics.
    pub fn stream_stats(&self) -> super::streaming::StreamStats {
        self.stream.stats()
    }

    /// Get memory statistics.
    pub fn memory_stats(&self) -> super::memory::MemoryStats {
        self.memory.stats()
    }

    /// Clear all cached data.
    pub fn clear(&self) {
        if let Ok(mut cache) = self.fragment_cache.write() {
            cache.clear();
        }
        if let Ok(mut headers) = self.headers.write() {
            headers.clear();
        }
        if let Ok(mut quality) = self.quality.write() {
            quality.clear();
        }
        if let Ok(mut complete) = self.layers_complete.write() {
            complete.clear();
        }
        self.memory.clear();
        self.stream.clear();
        self.tokens_generated.store(0, Ordering::Relaxed);
        self.current_layer.store(0, Ordering::Relaxed);
        self.set_state(ProviderState::Uninitialized);
    }
}

/// Builder for ProgressiveWeightProvider.
pub struct ProviderBuilder {
    config: HoloInferenceConfig,
    memory_config: super::memory::MemoryConfig,
    max_streams: usize,
}

impl ProviderBuilder {
    /// Create new builder with default config.
    pub fn new() -> Self {
        Self {
            config: HoloInferenceConfig::default(),
            memory_config: super::memory::MemoryConfig::default(),
            max_streams: 4,
        }
    }

    /// Set inference config.
    pub fn with_config(mut self, config: HoloInferenceConfig) -> Self {
        self.config = config;
        self
    }

    /// Set VRAM budget.
    pub fn with_vram_budget(mut self, bytes: usize) -> Self {
        self.config.vram_budget = bytes;
        self.memory_config.vram_budget = bytes;
        self
    }

    /// Set RAM budget.
    pub fn with_ram_budget(mut self, bytes: usize) -> Self {
        self.config.ram_budget = bytes;
        self.memory_config.ram_budget = bytes;
        self
    }

    /// Set minimum quality.
    pub fn with_min_quality(mut self, quality: f32) -> Self {
        self.config.min_quality = quality.clamp(0.1, 1.0);
        self
    }

    /// Set target quality.
    pub fn with_target_quality(mut self, quality: f32) -> Self {
        self.config.target_quality = quality.clamp(self.config.min_quality, 1.0);
        self
    }

    /// Set max concurrent streams.
    pub fn with_max_streams(mut self, count: usize) -> Self {
        self.max_streams = count;
        self
    }

    /// Build the provider.
    pub fn build(self) -> ProgressiveWeightProvider {
        let memory = Arc::new(HoloMemoryManager::new(self.memory_config));
        let stream = Arc::new(StreamManager::new(Arc::clone(&memory), self.max_streams));

        ProgressiveWeightProvider::new(self.config, memory, stream)
    }
}

impl Default for ProviderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_provider() -> ProgressiveWeightProvider {
        ProviderBuilder::new()
            .with_vram_budget(1024 * 1024 * 1024) // 1GB
            .with_ram_budget(4 * 1024 * 1024 * 1024) // 4GB
            .with_min_quality(0.5)
            .with_target_quality(0.9)
            .build()
    }

    #[test]
    fn test_quality_from_fragments() {
        // LowRankDistributed uses polynomial curve: 0.3 + 0.5*x + 0.15*x^2 + 0.05*x^3
        // First fragment (x=0.1) gives ~35% quality
        let q1 = QualityMetrics::quality_from_fragments_default(1, 10);
        assert!((q1 - 0.352).abs() < 0.01, "q1={}", q1);

        // Half fragments (x=0.5) gives ~59% quality
        let q5 = QualityMetrics::quality_from_fragments_default(5, 10);
        assert!((q5 - 0.594).abs() < 0.01, "q5={}", q5);

        // All fragments gives 100% (clamped to 1.0)
        let q10 = QualityMetrics::quality_from_fragments_default(10, 10);
        assert!((q10 - 1.0).abs() < 0.01, "q10={}", q10);
    }

    #[test]
    fn test_provider_state() {
        let provider = create_test_provider();
        assert_eq!(provider.state(), ProviderState::Uninitialized);

        provider.start_inference();
        assert_eq!(provider.state(), ProviderState::Ready);

        provider.end_inference();
    }

    #[test]
    fn test_layer_priority() {
        let provider = create_test_provider();
        provider.current_layer.store(5, Ordering::Relaxed);

        assert_eq!(provider.layer_priority(5), StreamPriority::Critical);
        assert_eq!(provider.layer_priority(6), StreamPriority::High);
        assert_eq!(provider.layer_priority(7), StreamPriority::Normal);
        assert_eq!(provider.layer_priority(10), StreamPriority::Low);
    }

    #[test]
    fn test_weight_type_importance() {
        assert!(WeightType::QProj.importance() > WeightType::LayerNorm.importance());
        assert_eq!(WeightType::Embed.importance(), 1.0);
    }

    #[test]
    fn test_builder() {
        let provider = ProviderBuilder::new()
            .with_vram_budget(24 * 1024 * 1024 * 1024)
            .with_min_quality(0.7)
            .with_target_quality(0.95)
            .with_max_streams(8)
            .build();

        assert_eq!(provider.config.vram_budget, 24 * 1024 * 1024 * 1024);
        assert_eq!(provider.config.min_quality, 0.7);
        assert_eq!(provider.config.target_quality, 0.95);
    }

    #[test]
    fn test_token_tracking() {
        let provider = create_test_provider();
        provider.start_inference();

        for _ in 0..10 {
            provider.notify_token_generated();
        }

        assert_eq!(provider.tokens_generated(), 10);
    }
}
