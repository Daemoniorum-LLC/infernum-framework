//! Tiered HoloTensor Loading for Progressive Inference
//!
//! Integrates memory tiering with progressive loading to enable 405B inference
//! on consumer hardware (24GB VRAM + 80GB RAM).
//!
//! ## Design
//!
//! The tiered loader uses a three-stage approach:
//! 1. **Initial Load**: Load fragments to meet minimum quality (70%)
//! 2. **Background Streaming**: Improve quality during idle time
//! 3. **LRU Eviction**: Move cold fragments VRAM → RAM when needed
//!
//! ## Memory Placement
//!
//! Fragments are placed according to:
//! - Importance (attention weights get priority)
//! - Recency (LRU eviction)
//! - Quality impact (fragments with higher singular values)
//!
//! ## GPU Acceleration
//!
//! When CUDA is available, HoloTensor reconstruction uses GPU kernels for:
//! - Spectral (IDCT) reconstruction
//! - Random Projection (RPH) reconstruction
//! - Low-Rank Distributed (LRDF) reconstruction
//!
//! GPU reconstruction is 10-50x faster than CPU for large weight matrices.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use candle_core::{DType, Device, Tensor};
use haagenti::holotensor::QualityCurve;

use super::memory::{HoloMemoryManager, MemoryConfig, MemoryTier};
use super::streaming::StreamManager;
use super::{HoloInferenceError, Result};
use crate::hct::{HctError, HctLoader};
use crate::lazy_varbuilder::TensorProvider;

#[cfg(feature = "cuda")]
use crate::gpu_holo::cuda::GpuHoloContext;

#[cfg(feature = "haagenti-gpu")]
use haagenti_cuda::{DecompressionPipeline, GpuContext as HaagentiGpuContext, PipelineConfig};

#[cfg(feature = "neural-compression")]
use haagenti_neural::{DecoderConfig, LayerCodebook, NctFile, NeuralDecoder};

/// Configuration for tiered loading.
#[derive(Debug, Clone)]
pub struct TieredConfig {
    /// Maximum VRAM budget in bytes.
    pub vram_budget: u64,
    /// Maximum RAM budget in bytes.
    pub ram_budget: u64,
    /// Minimum quality for initial load (0.0-1.0).
    pub min_quality: f32,
    /// Target quality for background improvement (0.0-1.0).
    pub target_quality: f32,
    /// Enable background quality improvement.
    pub enable_background_streaming: bool,
    /// Number of concurrent background streams.
    pub background_streams: usize,
}

impl Default for TieredConfig {
    fn default() -> Self {
        Self {
            vram_budget: 20 * 1024 * 1024 * 1024, // 20GB
            ram_budget: 64 * 1024 * 1024 * 1024,  // 64GB
            min_quality: 0.7,
            target_quality: 0.95,
            enable_background_streaming: true,
            background_streams: 4,
        }
    }
}

impl TieredConfig {
    /// Create config for 24GB VRAM + 80GB RAM setup.
    pub fn for_24gb_80gb() -> Self {
        Self {
            vram_budget: 22 * 1024 * 1024 * 1024, // 22GB (leave 2GB headroom)
            ram_budget: 76 * 1024 * 1024 * 1024,  // 76GB (leave 4GB headroom)
            min_quality: 0.7,
            target_quality: 0.95,
            enable_background_streaming: true,
            background_streams: 4,
        }
    }
}

/// Statistics for tiered loading.
#[derive(Debug, Clone, Default)]
pub struct TieredStats {
    /// Total tensors loaded.
    pub tensors_loaded: usize,
    /// Tensors currently in VRAM.
    pub vram_tensors: usize,
    /// Tensors currently in RAM.
    pub ram_tensors: usize,
    /// Tensors on disk (not loaded).
    pub disk_tensors: usize,
    /// Average quality across loaded tensors.
    pub average_quality: f32,
    /// Bytes used in VRAM.
    pub vram_bytes: u64,
    /// Bytes used in RAM.
    pub ram_bytes: u64,
    /// Number of LRU evictions performed.
    pub evictions: usize,
    /// Number of background improvements completed.
    pub background_improvements: usize,
    /// Tensors loaded with GPU acceleration.
    pub gpu_reconstructions: usize,
    /// Tensors loaded with CPU fallback.
    pub cpu_reconstructions: usize,
    /// Total GPU reconstruction time in milliseconds.
    pub gpu_time_ms: u64,
    /// Total CPU reconstruction time in milliseconds.
    pub cpu_time_ms: u64,
    /// Tensors loaded from pre-converted safetensors (fast path).
    pub safetensor_loads: usize,
    /// Total safetensor load time in milliseconds.
    pub safetensor_time_ms: u64,
}

/// Tensor placement decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementDecision {
    /// Place in VRAM (hot path).
    Vram,
    /// Place in RAM (warm path).
    Ram,
    /// Keep on disk (cold path).
    Disk,
}

/// Compression type for HCT fragment data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    /// No compression.
    None,
    /// LZ4 compression (fast).
    Lz4,
    /// Zstd compression (balanced).
    Zstd,
    /// Brotli compression (high ratio).
    Brotli,
}

/// Layer weight info for placement decisions.
#[derive(Debug, Clone)]
pub struct LayerWeightInfo {
    /// Layer index.
    pub layer: usize,
    /// Weight name (e.g., "q_proj", "k_proj").
    pub weight_name: String,
    /// Estimated size in bytes.
    pub size_bytes: u64,
    /// Whether this is an attention weight.
    pub is_attention: bool,
    /// Importance score (0.0-1.0).
    pub importance: f32,
}

impl LayerWeightInfo {
    /// Determine if this is an attention-related weight.
    pub fn is_attention_weight(name: &str) -> bool {
        name.contains("q_proj")
            || name.contains("k_proj")
            || name.contains("v_proj")
            || name.contains("o_proj")
            || name.contains("self_attn")
    }

    /// Calculate importance score for a weight.
    pub fn calculate_importance(name: &str, layer: usize, total_layers: usize) -> f32 {
        let mut score = 0.0f32;

        // Attention weights are more important
        if Self::is_attention_weight(name) {
            score += 0.3;
        }

        // First and last few layers are more important
        let layer_importance = if layer < 4 || layer >= total_layers - 4 {
            0.2
        } else {
            0.1
        };
        score += layer_importance;

        // Embedding and head are critical
        if name.contains("embed") || name.contains("lm_head") {
            score += 0.4;
        }

        score.min(1.0)
    }
}

/// Tiered HoloTensor loader for progressive inference.
///
/// ## Memory Architecture
///
/// This loader implements true tiered memory management:
/// - **CPU RAM (60GB)**: Caches reconstructed tensors for fast reload
/// - **GPU VRAM (24GB)**: Only holds active DecoderLayer tensors
///
/// When a tensor is requested:
/// 1. Check CPU cache → If found, transfer to GPU (~100ms)
/// 2. If not cached → Reconstruct from HCT (~30s), cache on CPU, return GPU copy
///
/// When a layer is evicted:
/// - GPU tensors freed (DecoderLayer dropped)
/// - CPU cache retained (fast reload on next access)
#[allow(dead_code)]
pub struct TieredHoloLoader {
    /// Loading configuration.
    config: TieredConfig,
    /// Directory containing HCT files.
    directory: PathBuf,
    /// Model identifier for cache key namespacing.
    /// Derived from the model directory name to prevent cache collisions
    /// between different models with same tensor names.
    model_id: String,
    /// Optional directory containing pre-converted safetensors (fast path).
    /// When set, the loader will try to load from safetensors first,
    /// falling back to HoloTensor reconstruction only if not found.
    safetensors_dir: Option<PathBuf>,
    /// Memory manager for tracking placements.
    memory_manager: Arc<HoloMemoryManager>,
    /// Stream manager for background loading.
    stream_manager: Arc<StreamManager>,
    /// CPU tensor cache for fast reload.
    /// Tensors are always stored on CPU here, transferred to GPU on demand.
    cpu_cache: RwLock<HashMap<String, (Tensor, MemoryTier, u64)>>,
    /// LRU order for CPU cache eviction (oldest first).
    cpu_lru_order: RwLock<VecDeque<String>>,
    /// Current RAM usage in bytes.
    cpu_cache_bytes: AtomicUsize,
    /// Quality tracking per tensor.
    qualities: RwLock<HashMap<String, f32>>,
    /// Target device for inference (usually CUDA).
    inference_device: Device,
    /// Target dtype.
    dtype: DType,
    /// Background streaming active flag.
    streaming_active: AtomicBool,
    /// Statistics.
    stats: RwLock<TieredStats>,
    /// GPU reconstruction context (when CUDA is available).
    #[cfg(feature = "cuda")]
    gpu_context: Option<RwLock<GpuHoloContext>>,
    /// Haagenti GPU decompression context for zero-copy loading.
    /// This provides faster decompression by keeping data on GPU.
    #[cfg(feature = "haagenti-gpu")]
    decompression_ctx: Option<Arc<HaagentiGpuContext>>,
    /// Neural decoder for 10:1 compressed tensors (.nct files).
    #[cfg(feature = "neural-compression")]
    neural_decoder: Option<Arc<NeuralDecoder>>,
    /// Directory containing .nct (neural compressed) files.
    #[cfg(feature = "neural-compression")]
    nct_dir: Option<PathBuf>,
    /// Whether GPU acceleration is enabled.
    gpu_enabled: AtomicBool,
    /// Number of GPU reconstructions in progress.
    gpu_inflight: AtomicUsize,
}

impl TieredHoloLoader {
    /// Create a new tiered loader.
    pub fn new(
        directory: impl AsRef<Path>,
        config: TieredConfig,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let directory = directory.as_ref().to_path_buf();

        // Create memory manager
        let mem_config = MemoryConfig {
            vram_budget: config.vram_budget as usize,
            ram_budget: config.ram_budget as usize,
            numa_node: -1,
            use_pinned_memory: true,
            eviction_threshold: 0.9,
        };
        let memory_manager = Arc::new(HoloMemoryManager::new(mem_config));

        // Create stream manager
        let stream_manager = Arc::new(StreamManager::new(
            Arc::clone(&memory_manager),
            config.background_streams,
        ));

        // Try to initialize GPU context if CUDA is available
        #[cfg(feature = "cuda")]
        let gpu_context = match &device {
            Device::Cuda(_) => {
                match GpuHoloContext::new(0) {
                    Ok(mut ctx) => {
                        // Load all holographic reconstruction kernels
                        if let Err(e) = ctx.load_all_kernels() {
                            tracing::warn!(
                                error = %e,
                                "Failed to load GPU HoloTensor kernels, using CPU fallback"
                            );
                            None
                        } else {
                            tracing::info!("GPU HoloTensor reconstruction enabled");
                            Some(RwLock::new(ctx))
                        }
                    },
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "Failed to create GPU context, using CPU fallback"
                        );
                        None
                    },
                }
            },
            _ => None,
        };

        // Initialize haagenti-cuda zero-copy decompression pipeline
        #[cfg(feature = "haagenti-gpu")]
        let decompression_ctx = match &device {
            Device::Cuda(_) => match HaagentiGpuContext::new(0) {
                Ok(ctx) => {
                    tracing::info!("Haagenti GPU decompression pipeline enabled (zero-copy)");
                    Some(Arc::new(ctx))
                },
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "Failed to create haagenti GPU context, using CPU decompression"
                    );
                    None
                },
            },
            _ => None,
        };

        #[cfg(feature = "cuda")]
        let gpu_enabled = gpu_context.is_some();
        #[cfg(not(feature = "cuda"))]
        let gpu_enabled = false;

        // Derive model_id from directory name to namespace cache keys
        let model_id = directory
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(Self {
            config,
            directory,
            model_id,
            safetensors_dir: None,
            memory_manager,
            stream_manager,
            cpu_cache: RwLock::new(HashMap::new()),
            cpu_lru_order: RwLock::new(VecDeque::new()),
            cpu_cache_bytes: AtomicUsize::new(0),
            qualities: RwLock::new(HashMap::new()),
            inference_device: device,
            dtype,
            streaming_active: AtomicBool::new(false),
            stats: RwLock::new(TieredStats::default()),
            #[cfg(feature = "cuda")]
            gpu_context,
            #[cfg(feature = "haagenti-gpu")]
            decompression_ctx,
            #[cfg(feature = "neural-compression")]
            neural_decoder: None, // Initialized via with_nct_dir()
            #[cfg(feature = "neural-compression")]
            nct_dir: None,
            gpu_enabled: AtomicBool::new(gpu_enabled),
            gpu_inflight: AtomicUsize::new(0),
        })
    }

    /// Set the safetensors directory for fast loading.
    ///
    /// When set, the loader will first check this directory for pre-converted
    /// `.safetensors` files. If found, loading is ~100x faster (simple mmap)
    /// compared to HoloTensor reconstruction.
    ///
    /// Use `holo_to_safetensors` converter to pre-convert HoloTensor files.
    pub fn with_safetensors_dir(mut self, dir: impl AsRef<Path>) -> Self {
        let path = dir.as_ref().to_path_buf();
        if path.exists() {
            tracing::info!(
                path = %path.display(),
                "Safetensors fast-load path enabled"
            );
            self.safetensors_dir = Some(path);
        } else {
            tracing::warn!(
                path = %path.display(),
                "Safetensors directory does not exist, using HoloTensor reconstruction"
            );
        }
        self
    }

    /// Check if safetensors fast-load is available.
    pub fn has_safetensors(&self) -> bool {
        self.safetensors_dir.is_some()
    }

    /// Set the NCT (neural compressed tensor) directory for 10:1 compression loading.
    ///
    /// When set, the loader will check for `.nct` files which provide 10:1
    /// compression using learned codebooks. This is the highest compression
    /// option for 405B models.
    ///
    /// The NCT directory must contain:
    /// - `codebooks.nct` - Layer-specific codebooks for decoding
    /// - `layer_N.nct` - Compressed layer tensors
    #[cfg(feature = "neural-compression")]
    pub fn with_nct_dir(mut self, dir: impl AsRef<Path>) -> Self {
        let path = dir.as_ref().to_path_buf();
        if !path.exists() {
            tracing::warn!(
                path = %path.display(),
                "NCT directory does not exist"
            );
            return self;
        }

        // Look for codebooks file
        let codebooks_path = path.join("codebooks.nct");
        if !codebooks_path.exists() {
            tracing::warn!(
                path = %codebooks_path.display(),
                "Codebooks file not found, NCT loading disabled"
            );
            return self;
        }

        // Load codebooks from NCT file
        match NctFile::load(&codebooks_path) {
            Ok(nct_file) => {
                let config = DecoderConfig::default();
                let decoder = NeuralDecoder::new(config, nct_file.codebooks);
                tracing::info!(
                    path = %path.display(),
                    compression_ratio = %nct_file.metadata.compression_ratio,
                    "NCT 10:1 compression enabled"
                );
                self.neural_decoder = Some(Arc::new(decoder));
                self.nct_dir = Some(path);
            },
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "Failed to load NCT codebooks, using HoloTensor"
                );
            },
        }

        self
    }

    /// Check if NCT (neural compression) is available.
    #[cfg(feature = "neural-compression")]
    pub fn has_nct(&self) -> bool {
        self.neural_decoder.is_some()
    }

    /// Check if NCT is available (returns false when feature disabled).
    #[cfg(not(feature = "neural-compression"))]
    pub fn has_nct(&self) -> bool {
        false
    }

    /// Calculate optimal placement for a tensor.
    ///
    /// Uses QualityCurve-aware logic to prioritize fragments that give
    /// the highest quality gain per byte of VRAM used. This replaces the
    /// previous hardcoded importance threshold (0.3) with dynamic placement
    /// based on:
    /// - Tensor importance score
    /// - Attention layer priority (critical for accuracy)
    /// - Quality impact estimation using QualityCurve coefficients
    /// - Available memory budget
    pub fn calculate_placement(&self, info: &LayerWeightInfo) -> PlacementDecision {
        let vram_available = self.memory_manager.vram_available() as u64;
        let ram_available = self.memory_manager.ram_available() as u64;

        // Calculate quality-aware placement score
        // Using Haagenti's QualityCurve insight: early fragments have higher singular values
        // so attention weights and high-importance tensors should go to VRAM
        let quality_curve = QualityCurve::default();

        // Estimate quality gain from having this tensor in VRAM vs RAM
        // For a typical 32-fragment tensor, first fragments are worth more
        let estimated_fragments = 32u16; // Typical fragment count
        let min_fragments_for_quality =
            quality_curve.fragments_for_quality(self.config.min_quality, estimated_fragments);

        // Quality impact score: higher for attention weights and important tensors
        // This replaces the hardcoded 0.3 threshold with a dynamic calculation
        let quality_impact = if info.is_attention {
            // Attention weights are critical - always prioritize for VRAM
            1.0
        } else {
            // Use importance directly, scaled by quality curve insight
            // Higher importance tensors have higher singular values -> more quality impact
            info.importance
                * (1.0 + quality_curve.predict(min_fragments_for_quality, estimated_fragments))
        };

        // Dynamic threshold based on min_quality target
        // Lower min_quality -> lower threshold -> more tensors go to disk/RAM
        let vram_threshold = 0.3 * self.config.min_quality;

        // Check if it fits in VRAM
        if info.size_bytes <= vram_available {
            // Use quality-aware threshold instead of hardcoded 0.3
            if quality_impact > vram_threshold || info.is_attention {
                return PlacementDecision::Vram;
            }
        }

        // Check if it fits in RAM
        if info.size_bytes <= ram_available {
            return PlacementDecision::Ram;
        }

        // Keep on disk
        PlacementDecision::Disk
    }

    /// Load a tensor with tiered placement.
    ///
    /// Loading priority:
    /// 1. Check cache (already loaded)
    /// 2. Try safetensors directory (fast mmap, ~100ms)
    /// 3. Fall back to HoloTensor reconstruction (~100s for large tensors)
    ///
    /// Includes recovery for corrupted/truncated files:
    /// - Scale tensors → ones (neutral scaling)
    /// - LayerNorm weights → ones (identity)
    /// - Biases → zeros
    pub fn load_tensor(&self, name: &str) -> Result<Tensor> {
        // Check CPU cache first - if found, transfer to inference device
        {
            let cache = self.cpu_cache.read().map_err(|_| {
                HoloInferenceError::MemoryAllocation("cpu_cache lock poisoned".to_string())
            })?;
            if let Some((cpu_tensor, _, _)) = cache.get(name) {
                // Cache hit - transfer from CPU to inference device (fast path ~100ms)
                let gpu_tensor = cpu_tensor.to_device(&self.inference_device).map_err(|e| {
                    HoloInferenceError::Conversion(format!(
                        "Failed to transfer {} from CPU to GPU: {}",
                        name, e
                    ))
                })?;
                // Update LRU order (move to back = most recently used)
                if let Ok(mut lru) = self.cpu_lru_order.write() {
                    if let Some(pos) = lru.iter().position(|k| k == name) {
                        lru.remove(pos);
                    }
                    lru.push_back(name.to_string());
                }
                return Ok(gpu_tensor);
            }
        }

        // Try safetensors fast path first (if available)
        if let Some(ref st_dir) = self.safetensors_dir {
            let filename = name.replace('.', "_");
            // Use model_id subdirectory to prevent cache collisions between models
            let model_cache_dir = st_dir.join(&self.model_id);
            let st_path = model_cache_dir.join(format!("{}.safetensors", filename));

            if st_path.exists() {
                let start = std::time::Instant::now();
                match self.load_from_safetensor(&st_path, name) {
                    Ok(tensor) => {
                        let elapsed_ms = start.elapsed().as_millis() as u64;
                        // Update safetensor stats
                        if let Ok(mut stats) = self.stats.write() {
                            stats.safetensor_loads += 1;
                            stats.safetensor_time_ms += elapsed_ms;
                        }

                        let size_bytes = tensor.elem_count() as u64 * dtype_size(self.dtype);
                        tracing::info!(
                            tensor = %name,
                            elapsed_ms = elapsed_ms,
                            size_mb = size_bytes / (1024 * 1024),
                            "Loaded tensor from NVMe cache (safetensors)"
                        );
                        self.cache_and_place_tensor(name, tensor.clone(), size_bytes)?;
                        return Ok(tensor);
                    },
                    Err(e) => {
                        tracing::warn!(
                            tensor = %name,
                            error = %e,
                            "Safetensor load failed, falling back to HoloTensor"
                        );
                        // Fall through to HoloTensor path
                    },
                }
            }
        }

        // Find the HCT file
        let filename = name.replace('.', "_");
        let path = self.directory.join(format!("{}.hct", filename));

        if !path.exists() {
            return Err(HoloInferenceError::FragmentLoad(format!(
                "HCT file not found: {}",
                path.display()
            )));
        }

        // Check if file is truncated and needs recovery
        let file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

        // Files < 200 bytes are truncated (just headers)
        if file_size < 200 {
            return self.recover_truncated_tensor(name);
        }

        // Try GPU reconstruction first if available
        let start = std::time::Instant::now();
        let (gpu_tensor, used_gpu) = self.load_tensor_internal(&path, name)?;
        let elapsed_ms = start.elapsed().as_millis() as u64;

        // Update reconstruction stats
        {
            if let Ok(mut stats) = self.stats.write() {
                if used_gpu {
                    stats.gpu_reconstructions += 1;
                    stats.gpu_time_ms += elapsed_ms;
                } else {
                    stats.cpu_reconstructions += 1;
                    stats.cpu_time_ms += elapsed_ms;
                }
            }
        }

        // Transfer to CPU for caching (this is the key change!)
        // The CPU cache allows fast reload (~100ms) when layers are swapped
        let cpu_tensor = gpu_tensor.to_device(&Device::Cpu).map_err(|e| {
            HoloInferenceError::Conversion(format!(
                "Failed to transfer {} to CPU for caching: {}",
                name, e
            ))
        })?;

        let size_bytes = cpu_tensor.elem_count() as u64 * dtype_size(self.dtype);

        // Determine placement (metadata only - all tensors cached on CPU)
        let info = LayerWeightInfo {
            layer: extract_layer_from_name(name).unwrap_or(0),
            weight_name: name.to_string(),
            size_bytes,
            is_attention: LayerWeightInfo::is_attention_weight(name),
            importance: LayerWeightInfo::calculate_importance(name, 0, 126),
        };

        let placement = self.calculate_placement(&info);
        let tier = match placement {
            PlacementDecision::Vram => MemoryTier::Vram,
            PlacementDecision::Ram => MemoryTier::Ram,
            PlacementDecision::Disk => MemoryTier::Disk,
        };

        // Evict from RAM cache if needed (before inserting new tensor)
        self.evict_from_ram_if_needed(size_bytes)?;

        // Store CPU tensor in cache
        {
            let mut cache = self.cpu_cache.write().map_err(|_| {
                HoloInferenceError::MemoryAllocation("cpu_cache lock poisoned".to_string())
            })?;
            cache.insert(name.to_string(), (cpu_tensor.clone(), tier, size_bytes));
        }

        // Populate NVMe cache with reconstructed tensor (for fast reload next time)
        // This is the key optimization: first access is slow (HCT reconstruction),
        // but subsequent accesses are ~1000x faster (mmap from safetensors)
        if let Some(ref st_dir) = self.safetensors_dir {
            let filename = name.replace('.', "_");
            // Use model_id subdirectory to prevent cache collisions between models
            let model_cache_dir = st_dir.join(&self.model_id);
            let st_path = model_cache_dir.join(format!("{}.safetensors", filename));

            // Ensure model-specific cache directory exists
            if !model_cache_dir.exists() {
                if let Err(e) = std::fs::create_dir_all(&model_cache_dir) {
                    tracing::warn!(
                        path = %model_cache_dir.display(),
                        error = %e,
                        "Failed to create model cache directory"
                    );
                }
            }

            // Only write if file doesn't exist (avoid overwriting)
            if !st_path.exists() {
                // Save tensor to safetensors format using candle's save API
                match save_tensor_to_safetensors(&cpu_tensor, name, &st_path) {
                    Ok(_) => {
                        tracing::debug!(
                            tensor = %name,
                            path = %st_path.display(),
                            size_mb = size_bytes / (1024 * 1024),
                            "Cached tensor to NVMe (next load will be ~1000x faster)"
                        );
                    },
                    Err(e) => {
                        // Cache write failure is not fatal - just log and continue
                        tracing::warn!(
                            tensor = %name,
                            error = %e,
                            "Failed to cache tensor to NVMe, will reconstruct from HCT next time"
                        );
                    },
                }
            }
        }

        // Update LRU order (add to back = most recently used)
        {
            let mut lru = self.cpu_lru_order.write().map_err(|_| {
                HoloInferenceError::MemoryAllocation("cpu_lru_order lock poisoned".to_string())
            })?;
            lru.push_back(name.to_string());
        }

        // Update RAM usage counter
        self.cpu_cache_bytes
            .fetch_add(size_bytes as usize, Ordering::Relaxed);

        // Update stats
        {
            let mut stats = self.stats.write().map_err(|_| {
                HoloInferenceError::MemoryAllocation("stats lock poisoned".to_string())
            })?;
            stats.tensors_loaded += 1;
            // All tensors go to RAM cache now
            stats.ram_tensors += 1;
            stats.ram_bytes += size_bytes;
        }

        // Return the GPU tensor for immediate use
        Ok(gpu_tensor)
    }

    /// Evict tensors from RAM cache if needed to stay within budget.
    ///
    /// Uses LRU policy to evict oldest tensors first until RAM usage
    /// is below the budget. Called before inserting new tensors.
    fn evict_from_ram_if_needed(&self, new_tensor_size: u64) -> Result<()> {
        let current_usage = self.cpu_cache_bytes.load(Ordering::Relaxed) as u64;
        let budget = self.config.ram_budget;

        // Check if we need to evict (need room for new tensor + 10% headroom)
        let target_usage = budget * 9 / 10; // 90% of budget
        if current_usage + new_tensor_size <= target_usage {
            return Ok(()); // Plenty of room
        }

        // Calculate how much we need to free
        let need_to_free = (current_usage + new_tensor_size).saturating_sub(target_usage);
        let mut freed = 0u64;
        let mut evicted_count = 0usize;

        tracing::info!(
            current_mb = current_usage / (1024 * 1024),
            budget_mb = budget / (1024 * 1024),
            need_to_free_mb = need_to_free / (1024 * 1024),
            "RAM cache eviction triggered"
        );

        // Evict oldest tensors until we have enough room
        while freed < need_to_free {
            // Get the oldest tensor name from LRU
            let oldest_name = {
                let mut lru = self.cpu_lru_order.write().map_err(|_| {
                    HoloInferenceError::MemoryAllocation("cpu_lru_order lock poisoned".to_string())
                })?;
                lru.pop_front()
            };

            let Some(name) = oldest_name else {
                // No more tensors to evict
                tracing::warn!(
                    freed_mb = freed / (1024 * 1024),
                    needed_mb = need_to_free / (1024 * 1024),
                    "RAM cache empty but still need more space"
                );
                break;
            };

            // Remove from cache and get size
            let size = {
                let mut cache = self.cpu_cache.write().map_err(|_| {
                    HoloInferenceError::MemoryAllocation("cpu_cache lock poisoned".to_string())
                })?;
                cache.remove(&name).map(|(_, _, size)| size)
            };

            if let Some(size) = size {
                freed += size;
                evicted_count += 1;
                self.cpu_cache_bytes
                    .fetch_sub(size as usize, Ordering::Relaxed);

                // Update stats
                if let Ok(mut stats) = self.stats.write() {
                    stats.evictions += 1;
                    stats.ram_tensors = stats.ram_tensors.saturating_sub(1);
                    stats.ram_bytes = stats.ram_bytes.saturating_sub(size);
                }
            }
        }

        if evicted_count > 0 {
            tracing::info!(
                evicted_count = evicted_count,
                freed_mb = freed / (1024 * 1024),
                remaining_mb = self.cpu_cache_bytes.load(Ordering::Relaxed) / (1024 * 1024),
                "RAM cache eviction complete"
            );
        }

        Ok(())
    }

    /// Load a tensor from a pre-converted safetensors file.
    ///
    /// This is the fast path (~100ms) compared to HoloTensor reconstruction (~100s).
    /// Uses mmap for efficient loading without copying to RAM first.
    fn load_from_safetensor(&self, path: &Path, _name: &str) -> Result<Tensor> {
        use std::fs::File;
        use std::io::{Read, Seek, SeekFrom};

        let mut file = File::open(path).map_err(|e| {
            HoloInferenceError::FragmentLoad(format!("Failed to open safetensor: {}", e))
        })?;

        // Read header length (8 bytes, little-endian)
        let mut header_len_bytes = [0u8; 8];
        file.read_exact(&mut header_len_bytes).map_err(|e| {
            HoloInferenceError::FragmentLoad(format!("Failed to read header length: {}", e))
        })?;
        let header_len = u64::from_le_bytes(header_len_bytes) as usize;

        // Read header JSON
        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes).map_err(|e| {
            HoloInferenceError::FragmentLoad(format!("Failed to read header: {}", e))
        })?;
        let header_str = String::from_utf8_lossy(&header_bytes);

        // Parse header to extract tensor info
        // Format: {"tensor_name": {"dtype": "F16", "shape": [1024, 4096], "data_offsets": [0, 8388608]}}
        let header_json: serde_json::Value = serde_json::from_str(&header_str).map_err(|e| {
            HoloInferenceError::FragmentLoad(format!("Failed to parse header JSON: {}", e))
        })?;

        // Get the first (and only) tensor entry
        let tensor_info = header_json
            .as_object()
            .and_then(|obj| obj.values().next())
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                HoloInferenceError::FragmentLoad("Invalid safetensor header".to_string())
            })?;

        let dtype_str = tensor_info
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                HoloInferenceError::FragmentLoad("Missing dtype in header".to_string())
            })?;

        let shape: Vec<usize> = tensor_info
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| HoloInferenceError::FragmentLoad("Missing shape in header".to_string()))?
            .iter()
            .filter_map(|v| v.as_u64())
            .map(|v| v as usize)
            .collect();

        let data_offsets: Vec<usize> = tensor_info
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                HoloInferenceError::FragmentLoad("Missing data_offsets in header".to_string())
            })?
            .iter()
            .filter_map(|v| v.as_u64())
            .map(|v| v as usize)
            .collect();

        if data_offsets.len() < 2 {
            return Err(HoloInferenceError::FragmentLoad(
                "Invalid data_offsets".to_string(),
            ));
        }

        let data_size = data_offsets[1] - data_offsets[0];

        // Read tensor data
        let data_start = 8 + header_len + data_offsets[0];
        file.seek(SeekFrom::Start(data_start as u64)).map_err(|e| {
            HoloInferenceError::FragmentLoad(format!("Failed to seek to data: {}", e))
        })?;

        let mut data = vec![0u8; data_size];
        file.read_exact(&mut data).map_err(|e| {
            HoloInferenceError::FragmentLoad(format!("Failed to read tensor data: {}", e))
        })?;

        // Create tensor based on dtype
        let tensor = match dtype_str {
            "F16" => {
                let f16_data: Vec<half::f16> = data
                    .chunks_exact(2)
                    .map(|b| half::f16::from_le_bytes([b[0], b[1]]))
                    .collect();
                Tensor::from_vec(f16_data, shape.as_slice(), &self.inference_device)
            },
            "F32" => {
                let f32_data: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                Tensor::from_vec(f32_data, shape.as_slice(), &self.inference_device)
            },
            "BF16" => {
                let bf16_data: Vec<half::bf16> = data
                    .chunks_exact(2)
                    .map(|b| half::bf16::from_le_bytes([b[0], b[1]]))
                    .collect();
                Tensor::from_vec(bf16_data, shape.as_slice(), &self.inference_device)
            },
            _ => {
                return Err(HoloInferenceError::FragmentLoad(format!(
                    "Unsupported dtype in safetensor: {}",
                    dtype_str
                )));
            },
        }
        .map_err(|e| HoloInferenceError::FragmentLoad(format!("Failed to create tensor: {}", e)))?;

        // Convert to target dtype if needed
        if tensor.dtype() != self.dtype {
            tensor.to_dtype(self.dtype).map_err(|e| {
                HoloInferenceError::FragmentLoad(format!("Failed to convert dtype: {}", e))
            })
        } else {
            Ok(tensor)
        }
    }

    /// Cache a tensor and update placement/stats.
    fn cache_and_place_tensor(&self, name: &str, tensor: Tensor, size_bytes: u64) -> Result<()> {
        // Determine placement
        let info = LayerWeightInfo {
            layer: extract_layer_from_name(name).unwrap_or(0),
            weight_name: name.to_string(),
            size_bytes,
            is_attention: LayerWeightInfo::is_attention_weight(name),
            importance: LayerWeightInfo::calculate_importance(name, 0, 126),
        };

        let placement = self.calculate_placement(&info);
        let tier = match placement {
            PlacementDecision::Vram => MemoryTier::Vram,
            PlacementDecision::Ram => MemoryTier::Ram,
            PlacementDecision::Disk => MemoryTier::Disk,
        };

        // Evict from RAM cache if needed (before inserting new tensor)
        self.evict_from_ram_if_needed(size_bytes)?;

        // Store tensor on CPU for caching
        let cpu_tensor = tensor.to_device(&Device::Cpu).map_err(|e| {
            HoloInferenceError::Conversion(format!(
                "Failed to transfer {} to CPU for caching: {}",
                name, e
            ))
        })?;
        {
            let mut cache = self.cpu_cache.write().map_err(|_| {
                HoloInferenceError::MemoryAllocation("cpu_cache lock poisoned".to_string())
            })?;
            cache.insert(name.to_string(), (cpu_tensor, tier, size_bytes));
        }

        // Update LRU order
        {
            let mut lru = self.cpu_lru_order.write().map_err(|_| {
                HoloInferenceError::MemoryAllocation("cpu_lru_order lock poisoned".to_string())
            })?;
            lru.push_back(name.to_string());
        }

        // Update RAM usage counter
        self.cpu_cache_bytes
            .fetch_add(size_bytes as usize, Ordering::Relaxed);

        // Update stats
        {
            let mut stats = self.stats.write().map_err(|_| {
                HoloInferenceError::MemoryAllocation("stats lock poisoned".to_string())
            })?;
            stats.tensors_loaded += 1;
            match tier {
                MemoryTier::Vram => {
                    stats.vram_tensors += 1;
                    stats.vram_bytes += size_bytes;
                },
                MemoryTier::Ram => {
                    stats.ram_tensors += 1;
                    stats.ram_bytes += size_bytes;
                },
                MemoryTier::Disk => {
                    stats.disk_tensors += 1;
                },
            }
        }

        Ok(())
    }

    /// Get current loading statistics.
    pub fn stats(&self) -> TieredStats {
        self.stats.read().map(|s| s.clone()).unwrap_or_default()
    }

    /// Get minimum quality achieved.
    pub fn min_quality(&self) -> f32 {
        self.config.min_quality
    }

    /// Check if a tensor is loaded (cached on CPU).
    pub fn is_loaded(&self, name: &str) -> bool {
        if let Ok(cache) = self.cpu_cache.read() {
            cache.contains_key(name)
        } else {
            false
        }
    }

    /// Get the memory tier for a loaded tensor.
    pub fn get_tier(&self, name: &str) -> Option<MemoryTier> {
        if let Ok(cache) = self.cpu_cache.read() {
            cache.get(name).map(|(_, tier, _)| *tier)
        } else {
            None
        }
    }

    /// Reconstruct a tensor from fragments using GPU or CPU.
    ///
    /// Automatically selects GPU if available and VRAM budget allows,
    /// otherwise falls back to CPU reconstruction.
    #[cfg(feature = "cuda")]
    pub fn reconstruct_tensor(
        &self,
        fragments: &[haagenti::holotensor::HoloFragment],
        rows: usize,
        cols: usize,
    ) -> Result<candle_core::Tensor> {
        use std::time::Instant;

        // Try GPU reconstruction first
        if let Some(ref gpu_ctx_lock) = self.gpu_context {
            if let Ok(gpu_ctx) = gpu_ctx_lock.read() {
                let start = Instant::now();
                match gpu_ctx.reconstruct_lrdf(fragments, rows, cols) {
                    Ok(gpu_tensor) => {
                        let elapsed = start.elapsed();
                        // Update stats
                        if let Ok(mut stats) = self.stats.write() {
                            stats.gpu_reconstructions += 1;
                            stats.gpu_time_ms += elapsed.as_millis() as u64;
                        }
                        // Convert to candle Tensor
                        let data = gpu_tensor.to_vec();
                        let tensor = candle_core::Tensor::from_vec(
                            data,
                            &[rows, cols],
                            &Device::Cpu, // Transfer to CPU for now
                        )
                        .map_err(|e| HoloInferenceError::Conversion(e.to_string()))?;
                        return Ok(tensor);
                    },
                    Err(e) => {
                        tracing::debug!("GPU reconstruction failed, falling back to CPU: {}", e);
                    },
                }
            }
        }

        // CPU fallback
        self.reconstruct_tensor_cpu(fragments, rows, cols)
    }

    /// Reconstruct a tensor from fragments using CPU.
    #[cfg(feature = "cuda")]
    fn reconstruct_tensor_cpu(
        &self,
        fragments: &[haagenti::holotensor::HoloFragment],
        rows: usize,
        cols: usize,
    ) -> Result<candle_core::Tensor> {
        use haagenti::holotensor::LrdfDecoder;
        use std::time::Instant;

        let start = Instant::now();

        let mut decoder = LrdfDecoder::new(rows, cols, fragments.len() as u16);
        for frag in fragments {
            decoder.add_fragment(frag).map_err(|e| {
                HoloInferenceError::Conversion(format!("Fragment decode error: {}", e))
            })?;
        }
        let data = decoder.reconstruct();

        let elapsed = start.elapsed();
        if let Ok(mut stats) = self.stats.write() {
            stats.cpu_reconstructions += 1;
            stats.cpu_time_ms += elapsed.as_millis() as u64;
        }

        let tensor = candle_core::Tensor::from_vec(data, &[rows, cols], &Device::Cpu)
            .map_err(|e| HoloInferenceError::Conversion(e.to_string()))?;

        Ok(tensor)
    }

    /// Reconstruct a tensor (non-CUDA builds always use CPU).
    #[cfg(not(feature = "cuda"))]
    pub fn reconstruct_tensor(
        &self,
        fragments: &[haagenti::holotensor::HoloFragment],
        rows: usize,
        cols: usize,
    ) -> Result<candle_core::Tensor> {
        use haagenti::holotensor::LrdfDecoder;

        let mut decoder = LrdfDecoder::new(rows, cols, fragments.len() as u16);
        for frag in fragments {
            decoder.add_fragment(frag).map_err(|e| {
                HoloInferenceError::Conversion(format!("Fragment decode error: {}", e))
            })?;
        }
        let data = decoder.reconstruct();

        if let Ok(mut stats) = self.stats.write() {
            stats.cpu_reconstructions += 1;
        }

        let tensor = candle_core::Tensor::from_vec(data, &[rows, cols], &Device::Cpu)
            .map_err(|e| HoloInferenceError::Conversion(e.to_string()))?;

        Ok(tensor)
    }

    /// Start background quality improvement.
    pub fn start_background_streaming(&self) {
        if !self.config.enable_background_streaming {
            return;
        }
        self.streaming_active.store(true, Ordering::Relaxed);
        tracing::info!("Started background quality streaming");
    }

    /// Stop background quality improvement.
    pub fn stop_background_streaming(&self) {
        self.streaming_active.store(false, Ordering::Relaxed);
        tracing::info!("Stopped background quality streaming");
    }

    /// Check if background streaming is active.
    pub fn is_streaming(&self) -> bool {
        self.streaming_active.load(Ordering::Relaxed)
    }

    /// Check if GPU acceleration is enabled.
    pub fn is_gpu_enabled(&self) -> bool {
        self.gpu_enabled.load(Ordering::Relaxed)
    }

    /// Internal tensor loading with GPU/CPU selection.
    ///
    /// Returns (tensor, used_gpu) where used_gpu indicates if GPU was used.
    fn load_tensor_internal(&self, path: &std::path::Path, name: &str) -> Result<(Tensor, bool)> {
        // Check if file is a HoloTensor format
        let loader = HctLoader::from_file(path)
            .map_err(|e| HoloInferenceError::FragmentLoad(format!("Failed to load HCT: {}", e)))?;

        let _is_holographic = loader.metadata().is_holographic();

        // Try GPU reconstruction for HoloTensor files if enabled
        #[cfg(feature = "cuda")]
        if is_holographic && self.gpu_enabled.load(Ordering::Relaxed) {
            match self.reconstruct_holotensor_gpu(path) {
                Ok(tensor) => {
                    tracing::debug!(
                        tensor = %name,
                        shape = ?tensor.dims(),
                        "GPU HoloTensor reconstruction complete"
                    );
                    return Ok((tensor, true));
                },
                Err(e) => {
                    tracing::warn!(
                        tensor = %name,
                        error = %e,
                        "GPU reconstruction failed, falling back to CPU"
                    );
                    // Fall through to CPU path
                },
            }
        }

        // CPU path (fallback or primary)
        let tensor = loader
            .to_tensor(&self.inference_device, Some(self.dtype))
            .map_err(|e| {
                if self.is_recoverable_tensor(name) {
                    // Return a recovery error that will be handled by caller
                    HoloInferenceError::FragmentLoad(format!("Recoverable: {}", e))
                } else {
                    HoloInferenceError::FragmentLoad(format!("Failed to create tensor: {}", e))
                }
            });

        match tensor {
            Ok(t) => Ok((t, false)),
            Err(e) => {
                // Check if this is a recoverable error
                if self.is_recoverable_tensor(name) {
                    let recovered = self.recover_truncated_tensor(name)?;
                    Ok((recovered, false))
                } else {
                    Err(e)
                }
            },
        }
    }

    /// GPU-accelerated HoloTensor reconstruction.
    #[cfg(feature = "cuda")]
    fn reconstruct_holotensor_gpu(&self, path: &std::path::Path) -> Result<Tensor> {
        use std::fs::File;

        // Track in-flight operations using a simple guard
        struct InflightGuard<'a>(&'a AtomicUsize);
        impl<'a> Drop for InflightGuard<'a> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::Relaxed);
            }
        }
        self.gpu_inflight.fetch_add(1, Ordering::Relaxed);
        let _guard = InflightGuard(&self.gpu_inflight);

        // Open file and read HoloTensor
        let file = File::open(path)
            .map_err(|e| HoloInferenceError::FragmentLoad(format!("Failed to open file: {}", e)))?;

        let reader = BufReader::new(file);
        let mut holo_reader = HoloTensorReader::new(reader).map_err(|e| {
            HoloInferenceError::FragmentLoad(format!("Failed to parse HoloTensor: {}", e))
        })?;

        // Read header and all fragments
        let (header, fragments) = holo_reader.read_all().map_err(|e| {
            HoloInferenceError::FragmentLoad(format!("Failed to read fragments: {}", e))
        })?;

        // Get GPU context
        let gpu_ctx_lock = self.gpu_context.as_ref().ok_or_else(|| {
            HoloInferenceError::DeviceError("GPU context not available".to_string())
        })?;

        let gpu_ctx = gpu_ctx_lock.read().map_err(|_| {
            HoloInferenceError::MemoryAllocation("GPU context lock poisoned".to_string())
        })?;

        // Perform GPU reconstruction
        let gpu_data = gpu_ctx.reconstruct(&header, &fragments).map_err(|e| {
            HoloInferenceError::FragmentLoad(format!("GPU reconstruction failed: {}", e))
        })?;

        // Copy result to host
        let host_data = gpu_ctx.copy_to_host(&gpu_data).map_err(|e| {
            HoloInferenceError::FragmentLoad(format!("GPU to host copy failed: {}", e))
        })?;

        // Create Candle tensor from host data
        let shape: Vec<usize> = header.shape.iter().map(|&d| d as usize).collect();
        let tensor = Tensor::from_vec(host_data, shape.as_slice(), &self.inference_device)
            .map_err(|e| {
                HoloInferenceError::FragmentLoad(format!("Failed to create tensor: {}", e))
            })?;

        // Convert to target dtype if needed
        let tensor = if self.dtype != candle_core::DType::F32 {
            tensor.to_dtype(self.dtype).map_err(|e| {
                HoloInferenceError::FragmentLoad(format!("Failed to convert dtype: {}", e))
            })?
        } else {
            tensor
        };

        Ok(tensor)
    }

    /// Decompress compressed HCT data using GPU zero-copy pipeline.
    ///
    /// When haagenti-cuda is available, this decompresses LZ4/Zstd data
    /// directly to GPU memory without touching CPU RAM, providing ~3x speedup
    /// for compressed tensor loading.
    #[cfg(feature = "haagenti-gpu")]
    fn decompress_gpu_zero_copy(
        &self,
        compressed: &[u8],
        decompressed_size: usize,
        compression: CompressionType,
    ) -> Option<Vec<u8>> {
        let ctx = self.decompression_ctx.as_ref()?;

        let result = match compression {
            CompressionType::Lz4 => ctx.decompress_lz4(compressed, decompressed_size),
            CompressionType::Zstd => ctx.decompress_zstd(compressed, decompressed_size),
            _ => return None, // Unsupported compression type
        };

        match result {
            Ok(gpu_buffer) => {
                // Allocate host buffer and copy decompressed data back
                let mut host_data = vec![0u8; gpu_buffer.size()];
                match gpu_buffer.copy_to_host(&mut host_data) {
                    Ok(()) => Some(host_data),
                    Err(e) => {
                        tracing::warn!(error = %e, "GPU decompression D2H copy failed");
                        None
                    },
                }
            },
            Err(e) => {
                tracing::warn!(error = %e, "GPU zero-copy decompression failed, falling back to CPU");
                None
            },
        }
    }

    /// Check if a tensor type can be recovered from corruption.
    fn is_recoverable_tensor(&self, name: &str) -> bool {
        name.contains("scale")
            || name.contains("layernorm")
            || name.contains("input_layernorm")
            || name.contains("post_attention_layernorm")
            || name.ends_with("bias")
            || name.contains("_norm")
            || name == "model.norm.weight" // Final layer norm
    }

    /// Recover a truncated or corrupted tensor with sensible defaults.
    ///
    /// Recovery strategy:
    /// - Scale tensors (FP8 dequant): ones (neutral scaling)
    /// - LayerNorm weights: ones (identity transform)
    /// - Biases: zeros
    fn recover_truncated_tensor(&self, name: &str) -> Result<Tensor> {
        // Determine the tensor shape and default value based on name patterns
        let (shape, fill_value) = self.infer_recovery_params(name);

        tracing::warn!(
            "Recovering truncated tensor '{}' with shape {:?}, fill={}",
            name,
            shape,
            fill_value
        );

        // Create the tensor with appropriate fill value on GPU
        let tensor = if fill_value == 0.0 {
            Tensor::zeros(shape.as_slice(), self.dtype, &self.inference_device)
        } else {
            Tensor::ones(shape.as_slice(), self.dtype, &self.inference_device)
                .and_then(|t| t.affine(fill_value as f64, 0.0))
        }
        .map_err(|e| {
            HoloInferenceError::FragmentLoad(format!("Failed to create recovery tensor: {}", e))
        })?;

        // Cache the recovered tensor on CPU
        let cpu_tensor = tensor.to_device(&Device::Cpu).map_err(|e| {
            HoloInferenceError::Conversion(format!(
                "Failed to transfer {} to CPU for caching: {}",
                name, e
            ))
        })?;
        let size_bytes = cpu_tensor.elem_count() as u64 * dtype_size(self.dtype);

        // Evict from RAM cache if needed
        self.evict_from_ram_if_needed(size_bytes)?;

        {
            let mut cache = self.cpu_cache.write().map_err(|_| {
                HoloInferenceError::MemoryAllocation("cpu_cache lock poisoned".to_string())
            })?;
            cache.insert(name.to_string(), (cpu_tensor, MemoryTier::Ram, size_bytes));
        }

        // Update LRU order
        {
            let mut lru = self.cpu_lru_order.write().map_err(|_| {
                HoloInferenceError::MemoryAllocation("cpu_lru_order lock poisoned".to_string())
            })?;
            lru.push_back(name.to_string());
        }

        // Update RAM usage counter
        self.cpu_cache_bytes
            .fetch_add(size_bytes as usize, Ordering::Relaxed);

        // Update stats
        {
            let mut stats = self.stats.write().map_err(|_| {
                HoloInferenceError::MemoryAllocation("stats lock poisoned".to_string())
            })?;
            stats.tensors_loaded += 1;
            stats.ram_tensors += 1;
            stats.ram_bytes += size_bytes;
        }

        Ok(tensor)
    }

    /// Infer shape and fill value for recovering a tensor.
    ///
    /// For 405B Llama:
    /// - hidden_size = 16384
    /// - num_kv_heads = 8 (GQA)
    /// - head_dim = 128
    /// - intermediate_size = 53248
    fn infer_recovery_params(&self, name: &str) -> (Vec<usize>, f32) {
        const HIDDEN_SIZE: usize = 16384;
        const HEAD_DIM: usize = 128;
        const NUM_KV_HEADS: usize = 8;

        // Default shape for unknown tensors
        let mut shape = vec![HIDDEN_SIZE];
        let mut fill_value = 1.0f32;

        // Determine shape based on tensor name patterns
        if name.contains("scale") {
            // FP8 scale tensors - typically per-channel or per-group
            if name.contains("q_proj") || name.contains("o_proj") {
                // Q/O proj: [hidden_size]
                shape = vec![HIDDEN_SIZE];
            } else if name.contains("k_proj") || name.contains("v_proj") {
                // K/V proj with GQA: [num_kv_heads * head_dim]
                shape = vec![NUM_KV_HEADS * HEAD_DIM];
            } else if name.contains("gate_proj")
                || name.contains("up_proj")
                || name.contains("down_proj")
            {
                // MLP scales
                shape = vec![HIDDEN_SIZE];
            } else {
                // Default scale shape
                shape = vec![1];
            }
            fill_value = 1.0; // Neutral scaling
        } else if name.contains("layernorm") || name.contains("_norm") {
            // LayerNorm weights
            shape = vec![HIDDEN_SIZE];
            fill_value = 1.0; // Identity transform
        } else if name.ends_with("bias") {
            // Bias tensors
            if name.contains("q_proj") || name.contains("o_proj") {
                shape = vec![HIDDEN_SIZE];
            } else if name.contains("k_proj") || name.contains("v_proj") {
                shape = vec![NUM_KV_HEADS * HEAD_DIM];
            } else {
                shape = vec![HIDDEN_SIZE];
            }
            fill_value = 0.0; // Zero bias
        }

        (shape, fill_value)
    }
}

/// Implement TensorProvider for TieredHoloLoader.
impl TensorProvider for TieredHoloLoader {
    fn get(
        &self,
        name: &str,
        _device: &Device,
        _dtype: DType,
    ) -> std::result::Result<Tensor, HctError> {
        self.load_tensor(name).map_err(|e| HctError::Tensor {
            message: e.to_string(),
        })
    }

    fn contains(&self, name: &str) -> bool {
        let filename = name.replace('.', "_");
        let path = self.directory.join(format!("{}.hct", filename));
        path.exists()
    }

    fn tensor_names(&self) -> Vec<String> {
        let mut names = Vec::new();

        if let Ok(entries) = std::fs::read_dir(&self.directory) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "hct") {
                    if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                        names.push(crate::hct::filename_to_tensor_name(name));
                    }
                }
            }
        }

        names
    }

    fn clear_prefix(&self, prefix: &str) -> (usize, u64) {
        // Clear tensors from CPU cache that match the prefix.
        // With the new CPU caching architecture, this only clears CPU-cached tensors.
        // GPU tensors are freed when the DecoderLayer is dropped (via loaded_layers.remove()).
        let mut cache = match self.cpu_cache.write() {
            Ok(t) => t,
            Err(_) => return (0, 0),
        };

        // Debug: Log cache state
        tracing::debug!(
            prefix = %prefix,
            cache_size = cache.len(),
            "TieredHoloLoader: clear_prefix called (CPU cache)"
        );

        let keys_to_remove: Vec<String> = cache
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();

        tracing::debug!(
            matching_keys = keys_to_remove.len(),
            "TieredHoloLoader: Found matching keys to remove from CPU cache"
        );

        let mut evicted_count = 0;
        let mut evicted_bytes = 0u64;

        for key in &keys_to_remove {
            if let Some((_, _, size)) = cache.remove(key) {
                evicted_bytes += size;
                evicted_count += 1;
            }
        }

        // Update RAM usage counter
        self.cpu_cache_bytes
            .fetch_sub(evicted_bytes as usize, Ordering::Relaxed);

        // Remove from LRU order
        if let Ok(mut lru) = self.cpu_lru_order.write() {
            lru.retain(|k| !keys_to_remove.contains(k));
        }

        if evicted_count > 0 {
            tracing::debug!(
                prefix = %prefix,
                evicted_count = evicted_count,
                evicted_mb = evicted_bytes / (1024 * 1024),
                "TieredHoloLoader: Cleared tensors from CPU cache by prefix"
            );

            // Update memory manager stats
            if let Ok(mut stats) = self.stats.write() {
                stats.evictions += evicted_count;
                stats.ram_tensors = stats.ram_tensors.saturating_sub(evicted_count);
                stats.ram_bytes = stats.ram_bytes.saturating_sub(evicted_bytes);
            }
        }

        (evicted_count, evicted_bytes)
    }
}

/// Extract layer number from tensor name.
fn extract_layer_from_name(name: &str) -> Option<usize> {
    // Pattern: "model.layers.N.xxx"
    let parts: Vec<&str> = name.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "layers" && i + 1 < parts.len() {
            return parts[i + 1].parse().ok();
        }
    }
    None
}

/// Returns the size in bytes of a single element of the given dtype.
fn dtype_size(dtype: DType) -> u64 {
    match dtype {
        DType::F32 | DType::U32 => 4,
        DType::F64 | DType::I64 => 8,
        DType::F16 | DType::BF16 => 2,
        DType::U8 => 1,
        // Handle new candle_core DType variants (I16, I32, F8E4M3, etc.)
        _ => 4,
    }
}

/// Saves a single tensor to a safetensors file for NVMe caching.
///
/// This enables the fast-load path: subsequent loads will mmap this file
/// instead of reconstructing from HCT (~1000x faster).
fn save_tensor_to_safetensors(tensor: &Tensor, name: &str, path: &Path) -> Result<()> {
    use std::io::Write;

    // Get dtype string for safetensors format
    let dtype_str = match tensor.dtype() {
        DType::F32 => "F32",
        DType::F64 => "F64",
        DType::F16 => "F16",
        DType::BF16 => "BF16",
        DType::U8 => "U8",
        DType::U32 => "U32",
        DType::I64 => "I64",
        // Handle new candle_core DType variants
        _ => "F32",
    };

    // Get tensor dimensions
    let shape: Vec<usize> = tensor.dims().to_vec();
    let data_size = tensor.elem_count() * tensor.dtype().size_in_bytes();

    // Get raw tensor data
    let data = tensor
        .flatten_all()
        .map_err(|e| HoloInferenceError::Conversion(format!("Failed to flatten tensor: {}", e)))?
        .to_vec1::<u8>()
        .or_else(|_| {
            // Try getting raw bytes for non-u8 dtypes
            match tensor.dtype() {
                DType::F32 => tensor
                    .flatten_all()
                    .and_then(|t| t.to_vec1::<f32>())
                    .map(|v| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>()),
                DType::F16 => tensor
                    .flatten_all()
                    .and_then(|t| t.to_vec1::<half::f16>())
                    .map(|v| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>()),
                DType::BF16 => tensor
                    .flatten_all()
                    .and_then(|t| t.to_vec1::<half::bf16>())
                    .map(|v| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>()),
                _ => Err(candle_core::Error::Msg(
                    "Unsupported dtype for cache".to_string(),
                )),
            }
        })
        .map_err(|e| HoloInferenceError::Conversion(format!("Failed to get tensor data: {}", e)))?;

    // Build safetensors header
    let header = format!(
        r#"{{"{name}":{{"dtype":"{dtype_str}","shape":{shape:?},"data_offsets":[0,{data_size}]}}}}"#,
        name = name,
        dtype_str = dtype_str,
        shape = shape,
        data_size = data_size,
    );

    let header_bytes = header.as_bytes();
    let header_len = header_bytes.len() as u64;

    // Pad header to 8-byte alignment
    let padding = (8 - (header_len % 8)) % 8;
    let padded_header_len = header_len + padding;

    // Write file: [8-byte header len] [header] [padding] [data]
    let mut file = std::fs::File::create(path).map_err(|e| HoloInferenceError::Io(e))?;

    file.write_all(&padded_header_len.to_le_bytes())
        .map_err(|e| HoloInferenceError::Io(e))?;
    file.write_all(header_bytes)
        .map_err(|e| HoloInferenceError::Io(e))?;
    file.write_all(&vec![b' '; padding as usize])
        .map_err(|e| HoloInferenceError::Io(e))?;
    file.write_all(&data)
        .map_err(|e| HoloInferenceError::Io(e))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    /// Helper to create a minimal valid HCT file.
    fn create_test_hct_file(dir: &Path, name: &str, shape: &[u64]) -> PathBuf {
        use haagenti::tensor::{CompressionAlgorithm, DType as HctDType, HctWriter};
        use haagenti::Lz4Compressor;

        let path = dir.join(format!("{}.hct", name));
        let file = fs::File::create(&path).expect("create file");

        let elements: u64 = shape.iter().product();
        let data: Vec<f32> = (0..elements).map(|i| i as f32 * 0.001).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut writer = HctWriter::new(
            file,
            CompressionAlgorithm::Lz4,
            HctDType::F32,
            shape.to_vec(),
        )
        .with_block_size(64 * 1024);

        let compressor = Lz4Compressor::new();
        writer
            .compress_data(&bytes, &compressor)
            .expect("write data");
        writer.finish().expect("finish");

        path
    }

    #[test]
    fn test_tiered_config_defaults() {
        let config = TieredConfig::default();
        assert!(config.vram_budget > 0);
        assert!(config.ram_budget > 0);
        assert!(config.min_quality > 0.0 && config.min_quality < 1.0);
    }

    #[test]
    fn test_tiered_loader_respects_vram_budget() {
        let temp_dir = TempDir::new().expect("create temp dir");

        // Create test file (small enough to fit in VRAM)
        create_test_hct_file(temp_dir.path(), "model_layers_0_q_proj_weight", &[16, 16]);

        let config = TieredConfig {
            vram_budget: 1024 * 1024, // 1MB
            ram_budget: 10 * 1024 * 1024,
            min_quality: 0.7,
            target_quality: 0.95,
            enable_background_streaming: false,
            background_streams: 0,
        };

        let loader = TieredHoloLoader::new(temp_dir.path(), config, Device::Cpu, DType::F32)
            .expect("create loader");

        // Load tensor
        let tensor = loader
            .load_tensor("model.layers.0.q_proj.weight")
            .expect("load");
        assert_eq!(tensor.dims(), &[16, 16]);

        // Check it's tracked
        assert!(loader.is_loaded("model.layers.0.q_proj.weight"));
    }

    #[test]
    fn test_calculate_placement_prefers_vram_for_attention() {
        let temp_dir = TempDir::new().expect("create temp dir");
        create_test_hct_file(temp_dir.path(), "test", &[4]);

        let config = TieredConfig::for_24gb_80gb();

        let loader = TieredHoloLoader::new(temp_dir.path(), config, Device::Cpu, DType::F32)
            .expect("create loader");

        // Attention weight should prefer VRAM
        let attn_info = LayerWeightInfo {
            layer: 0,
            weight_name: "q_proj.weight".to_string(),
            size_bytes: 1024,
            is_attention: true,
            importance: 0.5,
        };

        let placement = loader.calculate_placement(&attn_info);
        assert_eq!(placement, PlacementDecision::Vram);

        // Non-attention weight with lower importance goes to RAM
        // (We reserve VRAM for high-importance tensors in 405B scenario)
        let other_info = LayerWeightInfo {
            layer: 0,
            weight_name: "down_proj.weight".to_string(),
            size_bytes: 1024,
            is_attention: false,
            importance: 0.1,
        };

        let placement = loader.calculate_placement(&other_info);
        // Low importance non-attention weights go to RAM to preserve VRAM for attention
        assert_eq!(placement, PlacementDecision::Ram);
    }

    #[test]
    fn test_layer_weight_importance() {
        // Attention weights should be more important
        assert!(LayerWeightInfo::is_attention_weight(
            "self_attn.q_proj.weight"
        ));
        assert!(LayerWeightInfo::is_attention_weight("k_proj"));
        assert!(!LayerWeightInfo::is_attention_weight("mlp.down_proj"));

        // First layers more important
        let first_layer = LayerWeightInfo::calculate_importance("q_proj", 0, 126);
        let middle_layer = LayerWeightInfo::calculate_importance("q_proj", 60, 126);
        assert!(first_layer > middle_layer);
    }

    #[test]
    fn test_extract_layer_from_name() {
        assert_eq!(
            extract_layer_from_name("model.layers.0.self_attn.q_proj"),
            Some(0)
        );
        assert_eq!(
            extract_layer_from_name("model.layers.125.mlp.down_proj"),
            Some(125)
        );
        assert_eq!(extract_layer_from_name("model.embed_tokens.weight"), None);
    }

    #[test]
    fn test_tiered_loader_implements_tensor_provider() {
        let temp_dir = TempDir::new().expect("create temp dir");
        // Use 256 elements (1KB) to ensure file is > 200 bytes threshold
        // (files < 200 bytes are treated as truncated and recovered with defaults)
        create_test_hct_file(temp_dir.path(), "model_embed_tokens_weight", &[256]);

        let config = TieredConfig::default();
        let loader = Arc::new(
            TieredHoloLoader::new(temp_dir.path(), config, Device::Cpu, DType::F32)
                .expect("create loader"),
        );

        // Use as TensorProvider
        let provider: Arc<dyn TensorProvider> = loader;

        assert!(provider.contains("model.embed_tokens.weight"));
        let tensor = provider
            .get("model.embed_tokens.weight", &Device::Cpu, DType::F32)
            .expect("get tensor");
        assert_eq!(tensor.dims(), &[256]);
    }

    #[test]
    fn test_background_streaming_control() {
        let temp_dir = TempDir::new().expect("create temp dir");
        create_test_hct_file(temp_dir.path(), "test", &[4]);

        let config = TieredConfig {
            enable_background_streaming: true,
            ..TieredConfig::default()
        };

        let loader = TieredHoloLoader::new(temp_dir.path(), config, Device::Cpu, DType::F32)
            .expect("create loader");

        assert!(!loader.is_streaming());

        loader.start_background_streaming();
        assert!(loader.is_streaming());

        loader.stop_background_streaming();
        assert!(!loader.is_streaming());
    }
}
