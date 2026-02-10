//! HoloTensor Inference Module
//!
//! Bridges haagenti's holotensor format with abaddon's inference engine,
//! enabling progressive weight loading with quality-aware scheduling.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    HoloMemoryManager                        │
//! │  Tracks fragment locations across memory tiers              │
//! │  (VRAM ← RAM ← NVMe ← Network)                             │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    StreamManager                            │
//! │  Prioritized fragment streaming with bandwidth management   │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │               ProgressiveWeightProvider                     │
//! │  Serves layer weights at requested quality levels           │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};
use thiserror::Error;

// Re-export haagenti types we depend on
pub use haagenti::holotensor::{HolographicEncoding, QualityCurve};

// Submodules
pub mod arena;
pub mod converter;
pub mod memory;
pub mod provider;
pub mod streaming;
pub mod tiered_loading;

#[cfg(test)]
mod tests;

// ==================== Error Types ====================

/// Errors from holotensor inference operations.
#[derive(Debug, Error)]
pub enum HoloInferenceError {
    /// Memory allocation failed.
    #[error("Memory allocation failed: {0}")]
    MemoryAllocation(String),

    /// Memory allocation failed (structured).
    #[error("Memory allocation failed in {tier:?}: {message}")]
    MemoryAlloc {
        /// Memory tier where allocation failed.
        tier: memory::MemoryTier,
        /// Error message.
        message: String,
    },

    /// Fragment loading failed.
    #[error("Fragment loading failed: {0}")]
    FragmentLoad(String),

    /// Fragment not found.
    #[error("Fragment not found: layer {layer}, index {fragment_index}")]
    FragmentNotFound {
        /// Layer index.
        layer: usize,
        /// Fragment index within the layer.
        fragment_index: u16,
    },

    /// Quality target cannot be achieved.
    #[error("Cannot achieve quality {target}: only {available} fragments available")]
    InsufficientQuality {
        /// Target quality level (0.0-1.0).
        target: f32,
        /// Available fragments.
        available: u16,
    },

    /// Quality target not reached during verification.
    #[error("Quality not reached: target {target}, current {current}")]
    QualityNotReached {
        /// Target quality level.
        target: f32,
        /// Current quality level.
        current: f32,
    },

    /// Insufficient memory for operation.
    #[error("Insufficient memory in {tier:?}: need {required} bytes, have {available} bytes")]
    InsufficientMemory {
        /// Memory tier.
        tier: memory::MemoryTier,
        /// Required bytes.
        required: usize,
        /// Available bytes.
        available: usize,
    },

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Conversion error.
    #[error("Conversion error: {0}")]
    Conversion(String),

    /// CUDA error.
    #[error("CUDA error: {0}")]
    Cuda(String),

    /// Device error.
    #[error("Device error: {0}")]
    DeviceError(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Haagenti error.
    #[error("Haagenti error: {0}")]
    Haagenti(String),
}

/// Result type for holotensor operations.
pub type Result<T> = std::result::Result<T, HoloInferenceError>;

// ==================== Memory Tier ====================

/// Memory tier for fragment storage.
///
/// Ordered by access latency (fastest first).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum MemoryTier {
    /// GPU VRAM - fastest, most limited.
    Vram = 0,
    /// System RAM - fast, larger capacity.
    Ram = 1,
    /// NVMe SSD - slower, large capacity.
    Nvme = 2,
    /// Network storage - slowest, unlimited.
    Network = 3,
}

impl MemoryTier {
    /// Typical latency in nanoseconds for this tier.
    pub fn typical_latency_ns(&self) -> u64 {
        match self {
            MemoryTier::Vram => 100,           // ~100ns
            MemoryTier::Ram => 100_000,        // ~100µs
            MemoryTier::Nvme => 10_000_000,    // ~10ms
            MemoryTier::Network => 50_000_000, // ~50ms
        }
    }

    /// Typical bandwidth in GB/s for this tier.
    pub fn typical_bandwidth_gbps(&self) -> f64 {
        match self {
            MemoryTier::Vram => 900.0,  // HBM3
            MemoryTier::Ram => 50.0,    // DDR5
            MemoryTier::Nvme => 7.0,    // Gen4 NVMe
            MemoryTier::Network => 1.0, // 10GbE
        }
    }
}

// ==================== Fragment Location ====================

/// Location of a fragment in the memory hierarchy.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FragmentLocation {
    fragment_id: u16,
    tier: MemoryTier,
    is_loading: bool,
    offset: Option<u64>,
}

impl FragmentLocation {
    /// Creates a new fragment location.
    pub fn new(fragment_id: u16, tier: MemoryTier) -> Self {
        Self {
            fragment_id,
            tier,
            is_loading: false,
            offset: None,
        }
    }

    /// Returns the fragment ID.
    pub fn fragment_id(&self) -> u16 {
        self.fragment_id
    }

    /// Returns the current memory tier.
    pub fn tier(&self) -> MemoryTier {
        self.tier
    }

    /// Returns whether this fragment is currently being loaded.
    pub fn is_loading(&self) -> bool {
        self.is_loading
    }

    /// Promotes this fragment to a faster tier.
    pub fn promote_to(&mut self, tier: MemoryTier) {
        if tier < self.tier {
            self.tier = tier;
        }
    }

    /// Marks this fragment as loading.
    pub fn set_loading(&mut self, loading: bool) {
        self.is_loading = loading;
    }
}

// ==================== Quality Metrics ====================

/// Tracks quality metrics for progressive loading.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct QualityMetrics {
    current_quality: f32,
    target_quality: f32,
    fragments_loaded: u16,
    total_fragments: u16,
    quality_history: Vec<(std::time::Instant, f32)>,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            current_quality: 0.0,
            target_quality: 1.0,
            fragments_loaded: 0,
            total_fragments: 8,
            quality_history: Vec::new(),
        }
    }
}

impl QualityMetrics {
    /// Creates metrics with a specific target quality.
    pub fn with_target(target: f32) -> Self {
        Self {
            target_quality: target,
            ..Default::default()
        }
    }

    /// Returns current quality level.
    pub fn current_quality(&self) -> f32 {
        self.current_quality
    }

    /// Returns target quality level.
    pub fn target_quality(&self) -> f32 {
        self.target_quality
    }

    /// Returns number of fragments loaded.
    pub fn fragments_loaded(&self) -> u16 {
        self.fragments_loaded
    }

    /// Returns the gap between current and target quality.
    pub fn quality_gap(&self) -> f32 {
        (self.target_quality - self.current_quality).max(0.0)
    }

    /// Records a fragment being loaded.
    pub fn record_fragment_loaded(&mut self, new_quality: f32) {
        self.fragments_loaded += 1;
        self.current_quality = new_quality;
        self.quality_history
            .push((std::time::Instant::now(), new_quality));
    }

    /// Returns whether target quality has been reached.
    pub fn target_reached(&self) -> bool {
        self.current_quality >= self.target_quality
    }
}

// ==================== Memory Config ====================

/// Configuration for memory management.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MemoryConfig {
    vram_budget_bytes: u64,
    ram_budget_bytes: u64,
    nvme_cache_path: Option<PathBuf>,
    enable_network_streaming: bool,
}

impl MemoryConfig {
    /// Creates a builder for memory configuration.
    pub fn builder() -> MemoryConfigBuilder {
        MemoryConfigBuilder::default()
    }

    /// Auto-detects system memory and returns reasonable defaults.
    ///
    /// Uses `SystemMemoryInfo` for RAM detection and estimates VRAM.
    /// For accurate VRAM, use `auto_detect_with_vram(vram_bytes)`.
    pub fn auto_detect() -> Self {
        let recommended = crate::system_memory::RecommendedConfig::detect(None);

        Self {
            vram_budget_bytes: recommended.vram_bytes,
            ram_budget_bytes: recommended.ram_bytes,
            nvme_cache_path: None,
            enable_network_streaming: false,
        }
    }

    /// Auto-detects with explicit VRAM size.
    ///
    /// Use this when you have GPU detection results from Arbiter.
    pub fn auto_detect_with_vram(vram_bytes: u64) -> Self {
        let recommended = crate::system_memory::RecommendedConfig::with_gpu(vram_bytes);

        Self {
            vram_budget_bytes: (vram_bytes as f64 * 0.9) as u64, // 90% usable
            ram_budget_bytes: recommended.ram_bytes,
            nvme_cache_path: None,
            enable_network_streaming: recommended.use_nvme_cache,
        }
    }

    /// Returns VRAM budget in bytes.
    pub fn vram_budget_bytes(&self) -> u64 {
        self.vram_budget_bytes
    }

    /// Returns RAM budget in bytes.
    pub fn ram_budget_bytes(&self) -> u64 {
        self.ram_budget_bytes
    }
}

/// Builder for MemoryConfig.
#[derive(Debug, Default)]
pub struct MemoryConfigBuilder {
    vram_budget_mb: Option<u64>,
    ram_budget_mb: Option<u64>,
    nvme_cache_path: Option<PathBuf>,
    enable_network_streaming: bool,
}

impl MemoryConfigBuilder {
    /// Sets VRAM budget in megabytes.
    pub fn vram_budget_mb(mut self, mb: u64) -> Self {
        self.vram_budget_mb = Some(mb);
        self
    }

    /// Sets RAM budget in megabytes.
    pub fn ram_budget_mb(mut self, mb: u64) -> Self {
        self.ram_budget_mb = Some(mb);
        self
    }

    /// Sets NVMe cache path.
    pub fn nvme_cache_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.nvme_cache_path = Some(path.into());
        self
    }

    /// Enables network streaming.
    pub fn enable_network_streaming(mut self, enable: bool) -> Self {
        self.enable_network_streaming = enable;
        self
    }

    /// Builds the configuration.
    pub fn build(self) -> MemoryConfig {
        MemoryConfig {
            vram_budget_bytes: self.vram_budget_mb.unwrap_or(20_000) * 1024 * 1024,
            ram_budget_bytes: self.ram_budget_mb.unwrap_or(64_000) * 1024 * 1024,
            nvme_cache_path: self.nvme_cache_path,
            enable_network_streaming: self.enable_network_streaming,
        }
    }
}

// ==================== Stream Priority ====================

/// Priority for fragment streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum StreamPriority {
    /// Background improvement - lowest priority.
    Background = 0,
    /// Normal inference priority.
    Normal = 1,
    /// High priority - user-facing generation.
    High = 2,
    /// Critical - blocking on this fragment.
    Critical = 3,
}

impl Default for StreamPriority {
    fn default() -> Self {
        Self::Normal
    }
}

// ==================== Stream Stats ====================

/// Statistics for fragment streaming.
#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    /// Total bytes transferred.
    pub bytes_transferred: u64,
    /// Fragments loaded.
    pub fragments_loaded: u64,
    /// Average latency in milliseconds.
    pub avg_latency_ms: f64,
    /// Current throughput in MB/s.
    pub throughput_mbps: f64,
}

// ==================== Stream Manager ====================

/// Manages fragment streaming with prioritization.
pub struct StreamManager {
    max_concurrent: usize,
    active_count: AtomicUsize,
    stats: Mutex<StreamStats>,
}

impl StreamManager {
    /// Creates a new stream manager.
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            max_concurrent,
            active_count: AtomicUsize::new(0),
            stats: Mutex::new(StreamStats::default()),
        }
    }

    /// Returns maximum concurrent streams.
    pub fn max_concurrent_streams(&self) -> usize {
        self.max_concurrent
    }

    /// Returns number of active streams.
    pub fn active_streams(&self) -> usize {
        self.active_count.load(Ordering::Relaxed)
    }

    /// Returns current statistics.
    pub fn stats(&self) -> StreamStats {
        self.stats.lock().clone()
    }
}

// ==================== HoloInferenceConfig ====================

/// Configuration for holographic inference.
#[derive(Debug, Clone)]
pub struct HoloInferenceConfig {
    /// Initial quality to start inference at (0.0-1.0).
    pub initial_quality: f32,
    /// Target quality to improve to (0.0-1.0).
    pub target_quality: f32,
    /// Minimum acceptable quality (0.0-1.0).
    pub min_quality: f32,
    /// Enable background quality improvement.
    pub background_improvement: bool,
    /// Enable async streaming from RAM to VRAM.
    pub enable_streaming: bool,
    /// Number of fragments per tensor.
    pub num_fragments: u16,
    /// VRAM budget in bytes.
    pub vram_budget: usize,
    /// RAM budget in bytes.
    pub ram_budget: usize,
    /// Memory configuration.
    pub memory_config: MemoryConfig,
}

impl Default for HoloInferenceConfig {
    fn default() -> Self {
        Self {
            initial_quality: 0.4,
            target_quality: 0.95,
            min_quality: 0.7,
            background_improvement: true,
            enable_streaming: true,
            num_fragments: 32,
            vram_budget: 22 * 1024 * 1024 * 1024, // 22GB
            ram_budget: 64 * 1024 * 1024 * 1024,  // 64GB
            memory_config: MemoryConfig::auto_detect(),
        }
    }
}

impl HoloInferenceConfig {
    /// Creates a builder.
    pub fn builder() -> HoloInferenceConfigBuilder {
        HoloInferenceConfigBuilder::default()
    }

    /// Returns initial quality target.
    pub fn initial_quality(&self) -> f32 {
        self.initial_quality
    }

    /// Returns final quality target.
    pub fn target_quality(&self) -> f32 {
        self.target_quality
    }

    /// Returns minimum quality threshold.
    pub fn min_quality(&self) -> f32 {
        self.min_quality
    }

    /// Returns whether background improvement is enabled.
    pub fn background_improvement_enabled(&self) -> bool {
        self.background_improvement
    }

    /// Returns whether streaming is enabled.
    pub fn streaming_enabled(&self) -> bool {
        self.enable_streaming
    }
}

/// Builder for HoloInferenceConfig.
#[derive(Debug, Default)]
pub struct HoloInferenceConfigBuilder {
    initial_quality: Option<f32>,
    target_quality: Option<f32>,
    min_quality: Option<f32>,
    background_improvement: Option<bool>,
    enable_streaming: Option<bool>,
    num_fragments: Option<u16>,
    vram_budget: Option<usize>,
    ram_budget: Option<usize>,
    memory_config: Option<MemoryConfig>,
}

impl HoloInferenceConfigBuilder {
    /// Sets initial quality.
    pub fn initial_quality(mut self, q: f32) -> Self {
        self.initial_quality = Some(q);
        self
    }

    /// Sets target quality.
    pub fn target_quality(mut self, q: f32) -> Self {
        self.target_quality = Some(q);
        self
    }

    /// Sets minimum quality.
    pub fn min_quality(mut self, q: f32) -> Self {
        self.min_quality = Some(q);
        self
    }

    /// Enables/disables background improvement.
    pub fn enable_background_improvement(mut self, enable: bool) -> Self {
        self.background_improvement = Some(enable);
        self
    }

    /// Enables/disables streaming.
    pub fn enable_streaming(mut self, enable: bool) -> Self {
        self.enable_streaming = Some(enable);
        self
    }

    /// Sets number of fragments.
    pub fn num_fragments(mut self, n: u16) -> Self {
        self.num_fragments = Some(n);
        self
    }

    /// Sets VRAM budget in bytes.
    pub fn vram_budget(mut self, bytes: usize) -> Self {
        self.vram_budget = Some(bytes);
        self
    }

    /// Sets RAM budget in bytes.
    pub fn ram_budget(mut self, bytes: usize) -> Self {
        self.ram_budget = Some(bytes);
        self
    }

    /// Sets memory configuration.
    pub fn memory_config(mut self, config: MemoryConfig) -> Self {
        self.memory_config = Some(config);
        self
    }

    /// Builds the configuration.
    pub fn build(self) -> HoloInferenceConfig {
        HoloInferenceConfig {
            initial_quality: self.initial_quality.unwrap_or(0.4),
            target_quality: self.target_quality.unwrap_or(0.95),
            min_quality: self.min_quality.unwrap_or(0.7),
            background_improvement: self.background_improvement.unwrap_or(true),
            enable_streaming: self.enable_streaming.unwrap_or(true),
            num_fragments: self.num_fragments.unwrap_or(32),
            vram_budget: self.vram_budget.unwrap_or(22 * 1024 * 1024 * 1024),
            ram_budget: self.ram_budget.unwrap_or(64 * 1024 * 1024 * 1024),
            memory_config: self.memory_config.unwrap_or_else(MemoryConfig::auto_detect),
        }
    }
}

// ==================== HoloInferenceStats ====================

/// Statistics for holographic inference.
#[derive(Debug, Clone, Default)]
pub struct HoloInferenceStats {
    /// Total inference calls.
    pub total_inferences: u64,
    /// Average quality at inference time.
    pub avg_inference_quality: f64,
    /// Fragments promoted (loaded to faster tier).
    pub fragments_promoted: u64,
    /// Cache hits (fragment already in target tier).
    pub cache_hits: u64,
    /// Cache misses.
    pub cache_misses: u64,
}

// ==================== HoloModelMetadata ====================

/// Metadata for a holotensor-encoded model.
#[derive(Debug, Clone)]
pub struct HoloModelMetadata {
    /// Model identifier.
    pub model_id: String,
    /// Total parameter count.
    pub total_parameters: u64,
    /// Number of fragments per tensor.
    pub total_fragments: u16,
    /// Encoding scheme used.
    pub encoding: HolographicEncoding,
    /// Number of transformer layers.
    pub layers: usize,
    /// Number of transformer layers (alias for layers).
    pub num_layers: usize,
    /// Hidden size.
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of KV heads (for GQA).
    pub num_kv_heads: usize,
    /// Original model size in bytes (for conversion metadata).
    pub original_size: u64,
    /// HCT compressed size in bytes.
    pub hct_size: u64,
    /// Verified quality after conversion.
    pub verified_quality: f32,
}

// ==================== HoloMemoryManager ====================

/// Manages fragment placement across memory tiers.
pub struct HoloMemoryManager {
    config: MemoryConfig,
    vram_used: AtomicU64,
    ram_used: AtomicU64,
    fragment_locations: RwLock<HashMap<(usize, u16), FragmentLocation>>, // (layer, fragment_id)
}

impl HoloMemoryManager {
    /// Creates a new memory manager.
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            config,
            vram_used: AtomicU64::new(0),
            ram_used: AtomicU64::new(0),
            fragment_locations: RwLock::new(HashMap::new()),
        }
    }

    /// Returns available VRAM in bytes.
    pub fn available_vram(&self) -> u64 {
        let used = self.vram_used.load(Ordering::Relaxed);
        self.config.vram_budget_bytes.saturating_sub(used)
    }

    /// Returns available RAM in bytes.
    pub fn available_ram(&self) -> u64 {
        let used = self.ram_used.load(Ordering::Relaxed);
        self.config.ram_budget_bytes.saturating_sub(used)
    }

    /// Returns VRAM used in bytes.
    pub fn vram_used(&self) -> u64 {
        self.vram_used.load(Ordering::Relaxed)
    }

    /// Returns RAM used in bytes.
    pub fn ram_used(&self) -> u64 {
        self.ram_used.load(Ordering::Relaxed)
    }

    /// Registers a fragment location.
    pub fn register_fragment(&self, layer: usize, fragment_id: u16, tier: MemoryTier) {
        let mut locations = self.fragment_locations.write();
        locations.insert(
            (layer, fragment_id),
            FragmentLocation::new(fragment_id, tier),
        );
    }

    /// Gets location of a fragment.
    pub fn get_fragment_location(
        &self,
        layer: usize,
        fragment_id: u16,
    ) -> Option<FragmentLocation> {
        let locations = self.fragment_locations.read();
        locations.get(&(layer, fragment_id)).cloned()
    }
}

// Re-export converter types (the full GPU-capable implementation)
pub use converter::{
    validate_hct_directory, ConversionConfig, ConversionMetadata, ConversionPhase,
    ConversionProgress, ConvertedTensor, HoloModelConverter, TensorInfo, ValidationReport,
};

// Re-export tiered loading types for progressive inference
pub use tiered_loading::{
    LayerWeightInfo, PlacementDecision, TieredConfig, TieredHoloLoader, TieredStats,
};

// ==================== ProgressiveWeightProvider ====================

/// Provides weights with progressive quality loading.
///
/// This is the main interface for inference engines to request
/// layer weights at specific quality levels.
pub struct ProgressiveWeightProvider {
    metadata: HoloModelMetadata,
    memory_manager: Arc<HoloMemoryManager>,
    stream_manager: Arc<StreamManager>,
    layer_quality: RwLock<HashMap<usize, QualityMetrics>>,
}

impl ProgressiveWeightProvider {
    /// Creates a new weight provider.
    pub fn new(
        metadata: HoloModelMetadata,
        memory_manager: Arc<HoloMemoryManager>,
        stream_manager: Arc<StreamManager>,
    ) -> Self {
        Self {
            metadata,
            memory_manager,
            stream_manager,
            layer_quality: RwLock::new(HashMap::new()),
        }
    }

    /// Returns model metadata.
    pub fn metadata(&self) -> &HoloModelMetadata {
        &self.metadata
    }

    /// Returns current quality for a layer.
    pub fn layer_quality(&self, layer: usize) -> f32 {
        self.layer_quality
            .read()
            .get(&layer)
            .map(|m| m.current_quality())
            .unwrap_or(0.0)
    }

    /// Sets target quality for a layer.
    pub fn set_layer_target(&self, layer: usize, target: f32) {
        let mut qualities = self.layer_quality.write();
        qualities
            .entry(layer)
            .or_insert_with(QualityMetrics::default)
            .target_quality = target;
    }

    /// Returns memory manager.
    pub fn memory_manager(&self) -> &Arc<HoloMemoryManager> {
        &self.memory_manager
    }

    /// Returns stream manager.
    pub fn stream_manager(&self) -> &Arc<StreamManager> {
        &self.stream_manager
    }
}

// ==================== Re-exports ====================

// Re-export LayerWeights from cuda_inference for convenience
#[cfg(feature = "cuda")]
pub use crate::cuda_inference::weight_store::LayerWeights;

/// Stub for non-CUDA builds.
#[cfg(not(feature = "cuda"))]
pub struct LayerWeights {
    /// Layer index.
    pub index: usize,
}
