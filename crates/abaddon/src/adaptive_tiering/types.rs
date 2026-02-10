//! Core types for the adaptive memory tiering system.

use std::collections::HashMap;

/// Reconstruction path for a tensor.
///
/// Determines which code path to use for HoloTensor decompression.
/// Classification happens once at profile time, not per-access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReconstructionPath {
    /// 2D+ tensors with sufficient size benefit from GPU reconstruction.
    /// Threshold: shape.len() >= 2 && total_elements > 4096
    GpuFast,

    /// 1D tensors (bias, layernorm) or small tensors go directly to CPU.
    /// No GPU attempt, no fallback warning.
    CpuDirect,

    /// Tensors stored as raw safetensors (no HoloTensor reconstruction needed).
    DirectLoad,
}

impl ReconstructionPath {
    /// Minimum element count for GPU reconstruction to be worthwhile.
    const GPU_ELEMENT_THRESHOLD: usize = 4096;

    /// Classify a tensor's reconstruction path based on its shape.
    ///
    /// # Arguments
    /// * `shape` - Tensor dimensions (e.g., [4096, 4096] for a 2D weight matrix)
    /// * `is_safetensor` - Whether this tensor is stored as raw safetensor (no HCT)
    pub fn classify(shape: &[u64], is_safetensor: bool) -> Self {
        // Raw safetensors bypass reconstruction entirely
        if is_safetensor {
            return ReconstructionPath::DirectLoad;
        }

        // 1D tensors always use CPU (bias, layernorm, embeddings sometimes)
        if shape.len() < 2 {
            return ReconstructionPath::CpuDirect;
        }

        // Small tensors don't benefit from GPU overhead
        let total_elements: u64 = shape.iter().product();
        if total_elements < Self::GPU_ELEMENT_THRESHOLD as u64 {
            return ReconstructionPath::CpuDirect;
        }

        // 2D+ tensors with sufficient size use GPU
        ReconstructionPath::GpuFast
    }
}

/// Loading backend selection for the adaptive tiering system.
///
/// The backend has a larger impact on throughput than allocation strategy.
/// Wrapping `TieredHoloLoader` adds overhead even for preloaded tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoadingBackend {
    /// Fast path: decompress all tensors upfront, place per allocation plan.
    /// Uses `hct_sequential` loader. Expected: 2+ tk/s for 14B.
    Eager,

    /// Medium path: eager load to VRAM + RAM, LRU cache for overflow.
    /// For models that mostly fit but have occasional swaps.
    EagerWithRamCache,

    /// Slow path: stream quality fragments on demand with layer swapping.
    /// Uses `TieredHoloLoader`. Only for 405B+ that need true layer swapping.
    /// Expected: <1 tk/s (I/O bound).
    Progressive,
}

impl LoadingBackend {
    /// Select the optimal loading backend based on allocation plan.
    ///
    /// Decision logic:
    /// - If NVMe usage is 0 (model fits in VRAM + RAM) → Eager with tiered placement
    /// - If NVMe usage > 0 (model needs disk) → Progressive (405B+ mode)
    ///
    /// The key insight: Eager loading with VRAM/RAM tiering is FAST because:
    /// - All tensors are decompressed upfront (no streaming overhead)
    /// - Hot tensors stay in VRAM, warm tensors in RAM
    /// - RAM→VRAM transfers are fast (~10GB/s PCIe)
    /// - No HCT reconstruction during inference
    ///
    /// Progressive is only for models that don't fit in VRAM+RAM (405B+).
    pub fn select(plan: &AllocationPlan) -> Self {
        // If everything fits in VRAM + RAM, use eager loading with tiered placement
        // This is the fast path: decompress all to RAM, preload hot to VRAM
        if plan.nvme_usage == 0 {
            // swap_count represents RAM→VRAM transfers, which is fine for eager mode
            // The tiered placement handles hot (VRAM) vs warm (RAM) tensors
            LoadingBackend::Eager
        }
        // Only use progressive for truly massive models that need disk I/O
        else {
            LoadingBackend::Progressive
        }
    }
}

/// Memory tier for tensor placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryTier {
    /// GPU VRAM - fastest access, limited capacity (~24GB typical)
    Vram,
    /// System RAM - fast access, larger capacity (~64-128GB)
    Ram,
    /// NVMe/Disk - slowest access, virtually unlimited
    Nvme,
}

impl MemoryTier {
    /// Returns relative access latency (lower is faster).
    ///
    /// Approximate latencies:
    /// - VRAM: ~0.1ms
    /// - RAM: ~1ms
    /// - NVMe: ~10ms
    pub fn latency_factor(&self) -> f32 {
        match self {
            MemoryTier::Vram => 1.0,
            MemoryTier::Ram => 10.0,
            MemoryTier::Nvme => 100.0,
        }
    }

    /// Returns tier priority (higher = better for performance).
    pub fn priority(&self) -> u8 {
        match self {
            MemoryTier::Vram => 3,
            MemoryTier::Ram => 2,
            MemoryTier::Nvme => 1,
        }
    }
}

/// Precision level for tensor storage.
///
/// Lower precision reduces memory footprint but may impact inference quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorPrecision {
    /// 16-bit brain float - baseline quality
    BF16,
    /// 8-bit floating point - ~1% quality degradation
    FP8,
    /// 8-bit integer - ~2% quality degradation
    INT8,
    /// 4-bit integer - ~5% quality degradation
    INT4,
}

impl TensorPrecision {
    /// Returns bits per element.
    pub fn bits(&self) -> u32 {
        match self {
            TensorPrecision::BF16 => 16,
            TensorPrecision::FP8 => 8,
            TensorPrecision::INT8 => 8,
            TensorPrecision::INT4 => 4,
        }
    }

    /// Returns size divisor relative to BF16 (baseline = 1.0).
    ///
    /// - BF16: 1.0x (16 bits)
    /// - FP8/INT8: 2.0x reduction (8 bits)
    /// - INT4: 4.0x reduction (4 bits)
    pub fn size_divisor(&self) -> f32 {
        16.0 / self.bits() as f32
    }

    /// Calculates storage size for a tensor with this precision.
    ///
    /// # Arguments
    /// * `bf16_size` - Size in bytes at BF16 precision
    pub fn storage_size(&self, bf16_size: u64) -> u64 {
        (bf16_size as f64 / self.size_divisor() as f64).ceil() as u64
    }

    /// Returns estimated quality retention factor [0.0, 1.0].
    ///
    /// Higher values mean less quality degradation.
    pub fn quality_factor(&self) -> f32 {
        match self {
            TensorPrecision::BF16 => 1.0,
            TensorPrecision::FP8 => 0.99,
            TensorPrecision::INT8 => 0.98,
            TensorPrecision::INT4 => 0.95,
        }
    }

    /// Returns all precision levels in order of quality (best first).
    pub fn all_by_quality() -> &'static [TensorPrecision] {
        &[
            TensorPrecision::BF16,
            TensorPrecision::FP8,
            TensorPrecision::INT8,
            TensorPrecision::INT4,
        ]
    }
}

/// Allocation decision for a single tensor.
#[derive(Debug, Clone)]
pub struct TensorAllocation {
    /// Which memory tier to place this tensor.
    pub tier: MemoryTier,
    /// What precision to use.
    pub precision: TensorPrecision,
    /// Priority for eviction decisions [0.0, 1.0].
    /// Higher priority tensors are evicted last.
    pub priority: f32,
    /// Whether this tensor should be prefetched.
    pub prefetch: bool,
    /// Calculated storage size in bytes (with precision applied).
    pub storage_size: u64,
}

/// Complete allocation plan for a model.
#[derive(Debug, Clone)]
pub struct AllocationPlan {
    /// Per-tensor allocation decisions (tensor_name -> allocation).
    pub allocations: HashMap<String, TensorAllocation>,
    /// Total VRAM usage in bytes.
    pub vram_usage: u64,
    /// Total RAM usage in bytes.
    pub ram_usage: u64,
    /// Total NVMe usage in bytes.
    pub nvme_usage: u64,
    /// Number of tensors requiring runtime swapping.
    pub swap_count: usize,
    /// Estimated overall quality score [0.0, 1.0].
    pub quality_score: f32,
}

impl AllocationPlan {
    /// Creates an empty allocation plan.
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            vram_usage: 0,
            ram_usage: 0,
            nvme_usage: 0,
            swap_count: 0,
            quality_score: 1.0,
        }
    }

    /// Returns total memory usage across all tiers.
    pub fn total_usage(&self) -> u64 {
        self.vram_usage + self.ram_usage + self.nvme_usage
    }

    /// Returns whether this plan requires any runtime swapping.
    pub fn requires_swapping(&self) -> bool {
        self.swap_count > 0
    }

    /// Returns tensors allocated to a specific tier.
    pub fn tensors_in_tier(
        &self,
        tier: MemoryTier,
    ) -> impl Iterator<Item = (&String, &TensorAllocation)> {
        self.allocations.iter().filter(move |(_, a)| a.tier == tier)
    }
}

impl Default for AllocationPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a single tensor for allocation planning.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name (e.g., "model.layers.0.self_attn.q_proj.weight").
    pub name: String,
    /// Size in bytes at BF16 precision.
    pub size_bytes: u64,
    /// Tensor shape (dimensions).
    pub shape: Vec<u64>,
    /// Layer index (None for embeddings, lm_head).
    pub layer_index: Option<usize>,
    /// Tensor type classification.
    pub tensor_type: TensorType,
    /// Reconstruction path (GPU vs CPU) determined at profile time.
    pub reconstruction_path: ReconstructionPath,
}

impl TensorInfo {
    /// Creates tensor info from a name and size (shape unknown).
    ///
    /// Automatically parses the tensor type and layer index from the name.
    /// Uses `CpuDirect` reconstruction path since shape is unknown.
    pub fn from_name(name: impl Into<String>, size_bytes: u64) -> Self {
        Self::from_name_with_shape(name, size_bytes, Vec::new())
    }

    /// Creates tensor info from a name, size, and shape.
    ///
    /// Automatically parses the tensor type and layer index from the name.
    /// Classifies reconstruction path based on shape.
    pub fn from_name_with_shape(name: impl Into<String>, size_bytes: u64, shape: Vec<u64>) -> Self {
        let name = name.into();
        let tensor_type = TensorType::from_name(&name);
        let layer_index = Self::parse_layer_index(&name);
        let reconstruction_path = ReconstructionPath::classify(&shape, false);

        Self {
            name,
            size_bytes,
            shape,
            layer_index,
            tensor_type,
            reconstruction_path,
        }
    }

    /// Parses layer index from tensor name (e.g., "layers.5." -> Some(5)).
    fn parse_layer_index(name: &str) -> Option<usize> {
        // Match patterns like "layers.5." or "layer.5." or "blocks.5."
        let patterns = ["layers.", "layer.", "blocks.", "block."];

        for pattern in patterns {
            if let Some(start) = name.find(pattern) {
                let after_pattern = &name[start + pattern.len()..];
                if let Some(end) = after_pattern.find('.') {
                    if let Ok(idx) = after_pattern[..end].parse::<usize>() {
                        return Some(idx);
                    }
                }
            }
        }
        None
    }
}

/// Classification of tensor types for importance scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorType {
    /// Token embeddings - always critical.
    Embedding,
    /// Output projection (lm_head) - always critical.
    LmHead,
    /// Attention projections (Q, K, V, O).
    Attention,
    /// Layer normalization.
    LayerNorm,
    /// MLP gate projection.
    MlpGate,
    /// MLP up projection.
    MlpUp,
    /// MLP down projection.
    MlpDown,
    /// Unknown tensor type.
    Other,
}

impl TensorType {
    /// Parses tensor type from name.
    pub fn from_name(name: &str) -> Self {
        let name_lower = name.to_lowercase();

        if name_lower.contains("embed") || name_lower.contains("wte") || name_lower.contains("wpe")
        {
            TensorType::Embedding
        } else if name_lower.contains("lm_head")
            || name_lower.contains("output")
                && name_lower.contains("weight")
                && !name_lower.contains("layer")
        {
            TensorType::LmHead
        } else if name_lower.contains("q_proj")
            || name_lower.contains("k_proj")
            || name_lower.contains("v_proj")
            || name_lower.contains("o_proj")
            || name_lower.contains("self_attn")
            || name_lower.contains("attention")
        {
            TensorType::Attention
        } else if name_lower.contains("layernorm")
            || name_lower.contains("ln_")
            || name_lower.contains("_norm")
            || name_lower.contains("rmsnorm")
        {
            TensorType::LayerNorm
        } else if name_lower.contains("gate_proj") || name_lower.contains("w1") {
            TensorType::MlpGate
        } else if name_lower.contains("up_proj") || name_lower.contains("w3") {
            TensorType::MlpUp
        } else if name_lower.contains("down_proj") || name_lower.contains("w2") {
            TensorType::MlpDown
        } else {
            TensorType::Other
        }
    }

    /// Returns base importance factor for this tensor type.
    ///
    /// Higher values indicate tensors more critical for inference quality.
    pub fn base_importance(&self) -> f32 {
        match self {
            TensorType::Embedding => 1.0,
            TensorType::LmHead => 1.0,
            TensorType::Attention => 0.9,
            TensorType::LayerNorm => 0.85,
            TensorType::MlpDown => 0.7,
            TensorType::MlpGate => 0.6,
            TensorType::MlpUp => 0.6,
            TensorType::Other => 0.5,
        }
    }
}

/// Model architecture type for architecture-specific heuristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArchitectureType {
    /// Llama-style architecture (Llama, Mistral, Qwen, etc.).
    Llama,
    /// GPT-style architecture.
    Gpt,
    /// Unknown architecture.
    Unknown,
}

impl ArchitectureType {
    /// Detects architecture from model metadata or tensor names.
    pub fn detect(tensor_names: &[String]) -> Self {
        let has_gate = tensor_names.iter().any(|n| n.contains("gate_proj"));
        let has_rotary = tensor_names.iter().any(|n| n.contains("rotary"));
        let has_mlp_c = tensor_names.iter().any(|n| n.contains("mlp.c_"));

        if has_gate || has_rotary {
            ArchitectureType::Llama
        } else if has_mlp_c {
            ArchitectureType::Gpt
        } else {
            ArchitectureType::Unknown
        }
    }
}

/// Model profile containing all information needed for allocation planning.
#[derive(Debug, Clone)]
pub struct ModelProfile {
    /// All tensors in the model.
    pub tensors: Vec<TensorInfo>,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Detected architecture type.
    pub architecture: ArchitectureType,
    /// Total model size in bytes (BF16).
    pub total_size_bytes: u64,
}

/// Error building model profile.
#[derive(Debug, thiserror::Error)]
pub enum ProfileError {
    /// IO error reading directory.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// No tensors found in directory.
    #[error("no tensors found in directory: {0}")]
    NoTensors(String),

    /// HCT loading error.
    #[error("HCT error: {0}")]
    Hct(String),
}

impl ModelProfile {
    /// Creates a new model profile from tensor information.
    pub fn new(tensors: Vec<TensorInfo>) -> Self {
        let num_layers = tensors
            .iter()
            .filter_map(|t| t.layer_index)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);

        let tensor_names: Vec<_> = tensors.iter().map(|t| t.name.clone()).collect();
        let architecture = ArchitectureType::detect(&tensor_names);
        let total_size_bytes = tensors.iter().map(|t| t.size_bytes).sum();

        Self {
            tensors,
            num_layers,
            architecture,
            total_size_bytes,
        }
    }

    /// Creates a model profile by scanning an HCT directory.
    ///
    /// Reads metadata from all `.hct` files to determine tensor names and sizes.
    ///
    /// # Arguments
    /// * `directory` - Path to the HCT model directory
    ///
    /// # Errors
    /// Returns error if directory cannot be read or contains no tensors.
    pub fn from_hct_directory(
        directory: impl AsRef<std::path::Path>,
    ) -> Result<Self, ProfileError> {
        use crate::hct::{filename_to_tensor_name, HctLoader};

        let directory = directory.as_ref();
        let mut tensors = Vec::new();

        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();

            // Only process .hct files
            if path.extension().and_then(|e| e.to_str()) != Some("hct") {
                continue;
            }

            // Get tensor name from filename
            let filename = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");
            let tensor_name = filename_to_tensor_name(filename);

            // Load metadata (fast - only reads header)
            match HctLoader::from_file(&path) {
                Ok(loader) => {
                    let metadata = loader.metadata();
                    // original_size is uncompressed size in bytes (at original dtype)
                    // For BF16 allocation planning, we use this directly
                    tensors.push(TensorInfo::from_name_with_shape(
                        &tensor_name,
                        metadata.original_size,
                        metadata.shape.clone(),
                    ));
                },
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "Failed to load HCT metadata, skipping tensor"
                    );
                },
            }
        }

        if tensors.is_empty() {
            return Err(ProfileError::NoTensors(directory.display().to_string()));
        }

        tracing::info!(
            tensor_count = tensors.len(),
            directory = %directory.display(),
            "Built model profile from HCT directory"
        );

        Ok(Self::new(tensors))
    }

    /// Returns model size in gigabytes.
    pub fn size_gb(&self) -> f64 {
        self.total_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Returns a summary of reconstruction paths in this profile.
    ///
    /// Useful for logging and debugging tensor classification.
    pub fn reconstruction_summary(&self) -> ReconstructionSummary {
        let mut gpu_count = 0;
        let mut cpu_count = 0;
        let mut direct_count = 0;

        for tensor in &self.tensors {
            match tensor.reconstruction_path {
                ReconstructionPath::GpuFast => gpu_count += 1,
                ReconstructionPath::CpuDirect => cpu_count += 1,
                ReconstructionPath::DirectLoad => direct_count += 1,
            }
        }

        ReconstructionSummary {
            gpu_fast_count: gpu_count,
            cpu_direct_count: cpu_count,
            direct_load_count: direct_count,
            total_count: self.tensors.len(),
        }
    }
}

/// Summary of reconstruction paths in a model profile.
#[derive(Debug, Clone)]
pub struct ReconstructionSummary {
    /// Number of tensors using GPU fast path.
    pub gpu_fast_count: usize,
    /// Number of tensors using CPU direct path.
    pub cpu_direct_count: usize,
    /// Number of tensors using direct load (safetensors).
    pub direct_load_count: usize,
    /// Total tensor count.
    pub total_count: usize,
}

impl std::fmt::Display for ReconstructionSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Reconstruction paths: {} GPU, {} CPU, {} direct ({} total)",
            self.gpu_fast_count, self.cpu_direct_count, self.direct_load_count, self.total_count
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_precision_size_divisor() {
        assert!((TensorPrecision::BF16.size_divisor() - 1.0).abs() < f32::EPSILON);
        assert!((TensorPrecision::INT8.size_divisor() - 2.0).abs() < f32::EPSILON);
        assert!((TensorPrecision::INT4.size_divisor() - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_tensor_precision_storage_size() {
        let bf16_size = 1000u64;
        assert_eq!(TensorPrecision::BF16.storage_size(bf16_size), 1000);
        assert_eq!(TensorPrecision::INT8.storage_size(bf16_size), 500);
        assert_eq!(TensorPrecision::INT4.storage_size(bf16_size), 250);
    }

    #[test]
    fn test_tensor_type_from_name() {
        assert_eq!(
            TensorType::from_name("model.embed_tokens.weight"),
            TensorType::Embedding
        );
        assert_eq!(TensorType::from_name("lm_head.weight"), TensorType::LmHead);
        assert_eq!(
            TensorType::from_name("model.layers.0.self_attn.q_proj.weight"),
            TensorType::Attention
        );
        assert_eq!(
            TensorType::from_name("model.layers.0.mlp.gate_proj.weight"),
            TensorType::MlpGate
        );
        assert_eq!(
            TensorType::from_name("model.layers.0.input_layernorm.weight"),
            TensorType::LayerNorm
        );
    }

    #[test]
    fn test_tensor_info_parse_layer() {
        let info = TensorInfo::from_name("model.layers.5.self_attn.q_proj.weight", 1000);
        assert_eq!(info.layer_index, Some(5));
        assert_eq!(info.tensor_type, TensorType::Attention);

        let embed = TensorInfo::from_name("model.embed_tokens.weight", 1000);
        assert_eq!(embed.layer_index, None);
        assert_eq!(embed.tensor_type, TensorType::Embedding);
    }

    #[test]
    fn test_model_profile_layer_count() {
        let tensors = vec![
            TensorInfo::from_name("model.embed_tokens.weight", 1000),
            TensorInfo::from_name("model.layers.0.self_attn.q_proj.weight", 1000),
            TensorInfo::from_name("model.layers.1.self_attn.q_proj.weight", 1000),
            TensorInfo::from_name("model.layers.2.self_attn.q_proj.weight", 1000),
            TensorInfo::from_name("lm_head.weight", 1000),
        ];
        let profile = ModelProfile::new(tensors);
        assert_eq!(profile.num_layers, 3);
        assert_eq!(profile.total_size_bytes, 5000);
    }

    #[test]
    fn test_memory_tier_priority() {
        assert!(MemoryTier::Vram.priority() > MemoryTier::Ram.priority());
        assert!(MemoryTier::Ram.priority() > MemoryTier::Nvme.priority());
    }

    #[test]
    fn test_reconstruction_path_classify_1d() {
        // 1D tensors (bias, layernorm) should use CPU
        assert_eq!(
            ReconstructionPath::classify(&[4096], false),
            ReconstructionPath::CpuDirect
        );
        assert_eq!(
            ReconstructionPath::classify(&[128], false),
            ReconstructionPath::CpuDirect
        );
    }

    #[test]
    fn test_reconstruction_path_classify_small_2d() {
        // Small 2D tensors should use CPU (below 4096 element threshold)
        assert_eq!(
            ReconstructionPath::classify(&[63, 64], false), // 4032 elements (< 4096)
            ReconstructionPath::CpuDirect
        );
        assert_eq!(
            ReconstructionPath::classify(&[32, 32], false), // 1024 elements
            ReconstructionPath::CpuDirect
        );
        // 4096 exactly is at threshold, uses GPU
        assert_eq!(
            ReconstructionPath::classify(&[64, 64], false), // 4096 elements exactly
            ReconstructionPath::GpuFast
        );
    }

    #[test]
    fn test_reconstruction_path_classify_large_2d() {
        // Large 2D tensors should use GPU
        assert_eq!(
            ReconstructionPath::classify(&[4096, 4096], false),
            ReconstructionPath::GpuFast
        );
        assert_eq!(
            ReconstructionPath::classify(&[128, 4096], false),
            ReconstructionPath::GpuFast
        );
    }

    #[test]
    fn test_reconstruction_path_safetensor() {
        // Safetensors bypass reconstruction
        assert_eq!(
            ReconstructionPath::classify(&[4096, 4096], true),
            ReconstructionPath::DirectLoad
        );
    }

    #[test]
    fn test_loading_backend_eager_no_swap() {
        // Model fits entirely in VRAM+RAM → Eager
        let plan = AllocationPlan {
            allocations: HashMap::from([(
                "t1".to_string(),
                TensorAllocation {
                    tier: MemoryTier::Vram,
                    precision: TensorPrecision::BF16,
                    priority: 1.0,
                    prefetch: false,
                    storage_size: 1000,
                },
            )]),
            vram_usage: 1000,
            ram_usage: 0,
            nvme_usage: 0,
            swap_count: 0,
            quality_score: 1.0,
        };
        assert_eq!(LoadingBackend::select(&plan), LoadingBackend::Eager);
    }

    #[test]
    fn test_loading_backend_progressive() {
        // Many swaps needed → Progressive
        let mut allocations = HashMap::new();
        for i in 0..100 {
            allocations.insert(
                format!("t{i}"),
                TensorAllocation {
                    tier: MemoryTier::Nvme,
                    precision: TensorPrecision::BF16,
                    priority: 0.5,
                    prefetch: false,
                    storage_size: 1000,
                },
            );
        }
        let plan = AllocationPlan {
            allocations,
            vram_usage: 1000,
            ram_usage: 1000,
            nvme_usage: 100000,
            swap_count: 50, // 50% of tensors need swapping
            quality_score: 0.9,
        };
        assert_eq!(LoadingBackend::select(&plan), LoadingBackend::Progressive);
    }

    #[test]
    fn test_tensor_info_with_shape() {
        let info = TensorInfo::from_name_with_shape(
            "model.layers.0.self_attn.q_proj.weight",
            1000,
            vec![4096, 4096],
        );
        assert_eq!(info.shape, vec![4096, 4096]);
        assert_eq!(info.reconstruction_path, ReconstructionPath::GpuFast);
    }

    #[test]
    fn test_tensor_info_1d_shape() {
        let info = TensorInfo::from_name_with_shape(
            "model.layers.0.input_layernorm.weight",
            1000,
            vec![4096],
        );
        assert_eq!(info.shape, vec![4096]);
        assert_eq!(info.reconstruction_path, ReconstructionPath::CpuDirect);
    }

    #[test]
    fn test_reconstruction_summary() {
        let tensors = vec![
            TensorInfo::from_name_with_shape("t1", 1000, vec![4096, 4096]), // GpuFast
            TensorInfo::from_name_with_shape("t2", 1000, vec![4096, 4096]), // GpuFast
            TensorInfo::from_name_with_shape("t3", 1000, vec![4096]),       // CpuDirect
            TensorInfo::from_name_with_shape("t4", 1000, vec![32, 32]),     // CpuDirect (small)
        ];
        let profile = ModelProfile::new(tensors);
        let summary = profile.reconstruction_summary();

        assert_eq!(summary.gpu_fast_count, 2);
        assert_eq!(summary.cpu_direct_count, 2);
        assert_eq!(summary.direct_load_count, 0);
        assert_eq!(summary.total_count, 4);
    }
}
