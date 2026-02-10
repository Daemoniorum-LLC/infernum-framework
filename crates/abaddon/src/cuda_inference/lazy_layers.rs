//! Lazy layer loading for memory-efficient inference.
//!
//! Enables running models larger than available VRAM by loading layers
//! on-demand and evicting cold layers when memory pressure rises.
//!
//! ## Design
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │ ALWAYS LOADED (embedding, norm, lm_head)           │
//! ├─────────────────────────────────────────────────────┤
//! │ LAYER WINDOW (managed by LazyLayerStore)           │
//! │   Loaded layers kept in HashMap                    │
//! │   LRU eviction when budget exceeded                │
//! └─────────────────────────────────────────────────────┘
//! ```

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use cudarc::driver::CudaDevice;
use tracing::{debug, info, warn};

use super::arch::{ModelConfig, WeightNameMap};
use super::weight_store::LayerWeights;
use super::InferenceError;

/// Trait for loading individual layers on demand.
pub trait LayerLoader: Send + Sync {
    /// Load a single layer's weights to GPU.
    fn load_layer(
        &self,
        idx: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<LayerWeights, InferenceError>;

    /// Estimate VRAM size for a layer in bytes.
    fn layer_vram_size(&self, idx: usize) -> u64;

    /// Total number of layers in the model.
    fn num_layers(&self) -> usize;

    /// Model configuration.
    fn config(&self) -> &ModelConfig;
}

/// Statistics for lazy layer loading.
#[derive(Debug, Clone, Default)]
pub struct LazyLayerStats {
    /// Number of layers currently loaded in VRAM.
    pub layers_loaded: usize,
    /// Current VRAM usage by layers in bytes.
    pub vram_used: u64,
    /// VRAM budget for layers in bytes.
    pub vram_budget: u64,
    /// Total layer loads (cache misses).
    pub total_loads: u64,
    /// Total layer evictions.
    pub total_evictions: u64,
    /// Cache hit rate (0.0 - 1.0).
    pub hit_rate: f64,
}

/// Lazy layer storage with LRU eviction.
///
/// Manages a subset of model layers in VRAM, loading on-demand
/// and evicting least-recently-used layers when budget is exceeded.
pub struct LazyLayerStore {
    /// Currently loaded layers.
    loaded: HashMap<usize, LayerWeights>,

    /// Loader for on-demand layer loading.
    loader: Arc<dyn LayerLoader>,

    /// LRU order (front = oldest, back = newest).
    lru: VecDeque<usize>,

    /// VRAM budget for layers (excludes embed/norm/lm_head).
    vram_budget: u64,

    /// Current VRAM usage by loaded layers.
    current_vram: u64,

    /// CUDA device for loading.
    device: Arc<CudaDevice>,

    /// Statistics.
    total_loads: u64,
    total_evictions: u64,
    total_accesses: u64,
    cache_hits: u64,
}

impl LazyLayerStore {
    /// Create a new lazy layer store.
    ///
    /// # Arguments
    ///
    /// * `loader` - Layer loader for on-demand loading
    /// * `device` - CUDA device
    /// * `vram_budget` - Maximum VRAM for layer storage in bytes
    pub fn new(loader: Arc<dyn LayerLoader>, device: Arc<CudaDevice>, vram_budget: u64) -> Self {
        info!(
            "LazyLayerStore created: {} layers, {:.2} GB budget",
            loader.num_layers(),
            vram_budget as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        Self {
            loaded: HashMap::new(),
            loader,
            lru: VecDeque::new(),
            vram_budget,
            current_vram: 0,
            device,
            total_loads: 0,
            total_evictions: 0,
            total_accesses: 0,
            cache_hits: 0,
        }
    }

    /// Get a layer, loading it if necessary.
    ///
    /// Evicts LRU layers if VRAM budget would be exceeded.
    pub fn get_layer(&mut self, idx: usize) -> Result<&LayerWeights, InferenceError> {
        self.total_accesses += 1;

        if self.loaded.contains_key(&idx) {
            // Cache hit - move to back of LRU
            self.cache_hits += 1;
            self.touch_lru(idx);
            return Ok(self.loaded.get(&idx).unwrap());
        }

        // Cache miss - need to load
        self.load_layer_internal(idx)?;
        Ok(self.loaded.get(&idx).unwrap())
    }

    /// Check if a layer is currently loaded.
    pub fn is_loaded(&self, idx: usize) -> bool {
        self.loaded.contains_key(&idx)
    }

    /// Prefetch layers (load them proactively).
    ///
    /// Useful for loading layer i+1 while processing layer i.
    pub fn prefetch(&mut self, indices: &[usize]) -> Result<(), InferenceError> {
        for &idx in indices {
            if !self.loaded.contains_key(&idx) {
                self.load_layer_internal(idx)?;
            }
        }
        Ok(())
    }

    /// Explicitly evict a layer to free VRAM.
    pub fn evict(&mut self, idx: usize) {
        if let Some(layer) = self.loaded.remove(&idx) {
            let layer_size = self.loader.layer_vram_size(idx);
            self.current_vram = self.current_vram.saturating_sub(layer_size);
            self.total_evictions += 1;

            // Remove from LRU
            self.lru.retain(|&x| x != idx);

            debug!(
                "Evicted layer {}, freed {:.2} MB, now {:.2} MB used",
                idx,
                layer_size as f64 / (1024.0 * 1024.0),
                self.current_vram as f64 / (1024.0 * 1024.0)
            );

            // Drop the layer weights (frees GPU memory)
            drop(layer);
        }
    }

    /// Evict all layers to free VRAM.
    pub fn evict_all(&mut self) {
        let indices: Vec<_> = self.loaded.keys().copied().collect();
        for idx in indices {
            self.evict(idx);
        }
    }

    /// Get the number of layers in the model.
    pub fn num_layers(&self) -> usize {
        self.loader.num_layers()
    }

    /// Get current statistics.
    pub fn stats(&self) -> LazyLayerStats {
        LazyLayerStats {
            layers_loaded: self.loaded.len(),
            vram_used: self.current_vram,
            vram_budget: self.vram_budget,
            total_loads: self.total_loads,
            total_evictions: self.total_evictions,
            hit_rate: if self.total_accesses > 0 {
                self.cache_hits as f64 / self.total_accesses as f64
            } else {
                0.0
            },
        }
    }

    /// Get the model configuration.
    pub fn config(&self) -> &ModelConfig {
        self.loader.config()
    }

    /// Get the CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    // Internal: Load a layer, evicting if necessary
    fn load_layer_internal(&mut self, idx: usize) -> Result<(), InferenceError> {
        let layer_size = self.loader.layer_vram_size(idx);

        // Evict layers until we have room
        while self.current_vram + layer_size > self.vram_budget && !self.lru.is_empty() {
            if let Some(evict_idx) = self.lru.pop_front() {
                self.evict(evict_idx);
            }
        }

        // Check if we have room now
        if self.current_vram + layer_size > self.vram_budget {
            return Err(InferenceError::Memory(format!(
                "Cannot fit layer {} ({:.2} MB) in VRAM budget ({:.2} MB available of {:.2} MB)",
                idx,
                layer_size as f64 / (1024.0 * 1024.0),
                (self.vram_budget - self.current_vram) as f64 / (1024.0 * 1024.0),
                self.vram_budget as f64 / (1024.0 * 1024.0)
            )));
        }

        // Load the layer
        debug!(
            "Loading layer {} ({:.2} MB)",
            idx,
            layer_size as f64 / (1024.0 * 1024.0)
        );

        let layer = self.loader.load_layer(idx, &self.device)?;
        self.loaded.insert(idx, layer);
        self.current_vram += layer_size;
        self.lru.push_back(idx);
        self.total_loads += 1;

        debug!(
            "Layer {} loaded, {}/{} layers in VRAM ({:.2}/{:.2} MB)",
            idx,
            self.loaded.len(),
            self.loader.num_layers(),
            self.current_vram as f64 / (1024.0 * 1024.0),
            self.vram_budget as f64 / (1024.0 * 1024.0)
        );

        Ok(())
    }

    // Internal: Move layer to back of LRU (most recently used)
    fn touch_lru(&mut self, idx: usize) {
        self.lru.retain(|&x| x != idx);
        self.lru.push_back(idx);
    }
}

impl std::fmt::Debug for LazyLayerStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyLayerStore")
            .field("loaded_layers", &self.loaded.len())
            .field("total_layers", &self.loader.num_layers())
            .field("vram_used_mb", &(self.current_vram / (1024 * 1024)))
            .field("vram_budget_mb", &(self.vram_budget / (1024 * 1024)))
            .finish()
    }
}

// ============================================================================
// HoloTensor Layer Loader
// ============================================================================

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use cudarc::driver::CudaSlice;
use haagenti::holotensor::{HoloTensorDecoder, HoloTensorReader};

use super::tensor::{GpuDType, GpuTensor};
use super::weight_store::{QuantFormat, QuantizedWeight, RMSNormWeights};
use crate::gpu_holo::cuda::GpuHoloContext;

/// Weight names within a single layer.
const LAYER_WEIGHT_NAMES: &[&str] = &[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "input_norm",
    "post_attn_norm",
];

/// HoloTensor layer loader for on-demand layer loading.
///
/// Indexes HoloTensor files by layer and loads individual layers
/// using GPU holographic reconstruction.
pub struct HoloLayerLoader {
    /// HoloTensor files grouped by layer index.
    /// layer_idx -> weight_name -> file_path
    layer_files: HashMap<usize, HashMap<String, PathBuf>>,

    /// Shared weights (embed_tokens, final_norm, lm_head).
    /// These are loaded once and kept in memory.
    shared_files: HashMap<String, PathBuf>,

    /// Model configuration.
    config: ModelConfig,

    /// GPU holographic reconstruction context.
    /// Wrapped in Mutex for interior mutability (reconstruct is &mut self).
    gpu_ctx: Mutex<Option<GpuHoloContext>>,

    /// Estimated VRAM per layer (calculated once from config).
    layer_vram_estimate: u64,
}

impl HoloLayerLoader {
    /// Create a new HoloTensor layer loader.
    ///
    /// Scans the given directory for HoloTensor files and indexes them by layer.
    pub fn new(
        hct_dir: impl AsRef<Path>,
        config: ModelConfig,
        device: Arc<CudaDevice>,
    ) -> Result<Self, InferenceError> {
        let hct_dir = hct_dir.as_ref();

        // Find all .hct files
        let mut hct_files = Vec::new();
        for entry in std::fs::read_dir(hct_dir)
            .map_err(|e| InferenceError::ModelLoad(format!("Cannot read dir: {}", e)))?
        {
            let entry =
                entry.map_err(|e| InferenceError::ModelLoad(format!("Dir entry error: {}", e)))?;
            let path = entry.path();
            if path.extension().map(|e| e == "hct").unwrap_or(false) {
                hct_files.push(path);
            }
        }

        if hct_files.is_empty() {
            return Err(InferenceError::ModelLoad(format!(
                "No .hct files found in {}",
                hct_dir.display()
            )));
        }

        // Index files by layer
        let mut layer_files: HashMap<usize, HashMap<String, PathBuf>> = HashMap::new();
        let mut shared_files: HashMap<String, PathBuf> = HashMap::new();

        for path in hct_files {
            let filename = path
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| InferenceError::ModelLoad("Invalid filename".to_string()))?;

            let hf_name = filename_to_hf_name(filename);

            // Check if this is a layer weight
            if let Some(layer_idx) = extract_layer_index(&hf_name) {
                // Extract weight name (e.g., "q_proj" from "layers.0.q_proj")
                let weight_name = extract_weight_name(&hf_name);
                layer_files
                    .entry(layer_idx)
                    .or_default()
                    .insert(weight_name, path);
            } else {
                // Shared weight (embed_tokens, final_norm, lm_head)
                shared_files.insert(hf_name, path);
            }
        }

        info!(
            "HoloLayerLoader indexed {} layers, {} shared weights",
            layer_files.len(),
            shared_files.len()
        );

        // Estimate VRAM per layer based on model config
        let layer_vram_estimate = estimate_layer_vram(&config);

        // Initialize GPU context (will load kernels lazily on first use)
        let mut gpu_ctx = GpuHoloContext::with_device(device, 0);

        // Pre-load kernels
        gpu_ctx
            .load_all_kernels()
            .map_err(|e| InferenceError::ModelLoad(format!("Failed to load GPU kernels: {}", e)))?;
        gpu_ctx.load_fused_kernel().map_err(|e| {
            InferenceError::ModelLoad(format!("Failed to load fused kernel: {}", e))
        })?;

        Ok(Self {
            layer_files,
            shared_files,
            config,
            gpu_ctx: Mutex::new(Some(gpu_ctx)),
            layer_vram_estimate,
        })
    }

    /// Get paths for shared weights (embed_tokens, final_norm, lm_head).
    pub fn shared_files(&self) -> &HashMap<String, PathBuf> {
        &self.shared_files
    }

    /// Load a single HoloTensor file to GPU.
    fn load_holotensor_file(
        &self,
        path: &Path,
        device: &Arc<CudaDevice>,
    ) -> Result<(GpuTensor, Vec<usize>), InferenceError> {
        let file = File::open(path).map_err(|e| {
            InferenceError::ModelLoad(format!("Cannot open {}: {}", path.display(), e))
        })?;
        let reader = BufReader::new(file);
        let mut holo_reader = HoloTensorReader::new(reader)
            .map_err(|e| InferenceError::ModelLoad(format!("Failed to parse HoloTensor: {}", e)))?;

        let (header, fragments) = holo_reader
            .read_all()
            .map_err(|e| InferenceError::ModelLoad(format!("Failed to read fragments: {}", e)))?;

        let shape: Vec<usize> = header.shape.iter().map(|&d| d as usize).collect();

        // 1D tensors use CPU reconstruction, 2D+ use GPU
        let gpu_tensor = if shape.len() == 1 {
            // CPU reconstruction for 1D tensors (norms, biases)
            let mut decoder = HoloTensorDecoder::new(header);
            for fragment in &fragments {
                decoder.add_fragment(fragment.clone()).map_err(|e| {
                    InferenceError::ModelLoad(format!("Failed to add fragment: {}", e))
                })?;
            }
            let cpu_data = decoder.reconstruct().map_err(|e| {
                InferenceError::ModelLoad(format!("CPU reconstruction failed: {}", e))
            })?;

            // Convert f32 to f16 and upload
            let f16_data: Vec<half::f16> =
                cpu_data.iter().map(|&f| half::f16::from_f32(f)).collect();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f16_data.as_ptr() as *const u8, f16_data.len() * 2)
            };

            let gpu_data: CudaSlice<u8> = device
                .htod_sync_copy(bytes)
                .map_err(|e| InferenceError::Memory(format!("Failed to upload tensor: {}", e)))?;

            GpuTensor::from_cuda_slice(gpu_data, shape.clone(), GpuDType::F16, Arc::clone(device))?
        } else {
            // GPU reconstruction for 2D+ tensors
            let mut gpu_ctx_guard = self.gpu_ctx.lock().unwrap();
            let gpu_ctx = gpu_ctx_guard
                .as_mut()
                .ok_or_else(|| InferenceError::Device("GPU context not initialized".to_string()))?;

            let gpu_data_f32: CudaSlice<f32> =
                gpu_ctx.reconstruct(&header, &fragments).map_err(|e| {
                    InferenceError::ModelLoad(format!("GPU reconstruction failed: {}", e))
                })?;

            let gpu_data_f16 = gpu_ctx.convert_f32_to_f16(&gpu_data_f32).map_err(|e| {
                InferenceError::ModelLoad(format!("F32->F16 conversion failed: {}", e))
            })?;

            GpuTensor::from_cuda_slice_f16(gpu_data_f16, shape.clone(), Arc::clone(device))?
        };

        Ok((gpu_tensor, shape))
    }
}

impl LayerLoader for HoloLayerLoader {
    fn load_layer(
        &self,
        idx: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<LayerWeights, InferenceError> {
        let layer_map = self.layer_files.get(&idx).ok_or_else(|| {
            InferenceError::ModelLoad(format!("Layer {} not found in indexed files", idx))
        })?;

        // Load each weight for this layer
        let mut weights: HashMap<String, (GpuTensor, Vec<usize>)> = HashMap::new();

        for name in LAYER_WEIGHT_NAMES {
            let path = layer_map.get(*name).ok_or_else(|| {
                InferenceError::ModelLoad(format!("Missing {} for layer {}", name, idx))
            })?;

            let (tensor, shape) = self.load_holotensor_file(path, device)?;
            debug!(
                "Layer {} weight '{}': shape {:?}, file {:?}",
                idx,
                name,
                shape,
                path.file_name()
            );
            weights.insert(name.to_string(), (tensor, shape));
        }

        // Build LayerWeights from loaded tensors
        fn to_quantized(
            weights: &mut HashMap<String, (GpuTensor, Vec<usize>)>,
            name: &str,
        ) -> QuantizedWeight {
            let (data, shape) = weights.remove(name).unwrap();
            let shape_2d = if shape.len() >= 2 {
                (shape[0], shape[1])
            } else {
                (shape[0], 1)
            };
            QuantizedWeight {
                data,
                format: QuantFormat::F16,
                scales: None,
                zero_points: None,
                g_idx: None,
                shape: shape_2d,
                block_size: 128,
                // HoloTensor preserves PyTorch format [out, in] so we need transposed GEMM
                transposed: true,
            }
        }

        fn to_norm(
            weights: &mut HashMap<String, (GpuTensor, Vec<usize>)>,
            name: &str,
        ) -> RMSNormWeights {
            let (data, _) = weights.remove(name).unwrap();
            RMSNormWeights { weight: data }
        }

        Ok(LayerWeights {
            index: idx,
            q_proj: to_quantized(&mut weights, "q_proj"),
            k_proj: to_quantized(&mut weights, "k_proj"),
            v_proj: to_quantized(&mut weights, "v_proj"),
            o_proj: to_quantized(&mut weights, "o_proj"),
            gate_proj: to_quantized(&mut weights, "gate_proj"),
            up_proj: to_quantized(&mut weights, "up_proj"),
            down_proj: to_quantized(&mut weights, "down_proj"),
            input_norm: to_norm(&mut weights, "input_norm"),
            post_attn_norm: to_norm(&mut weights, "post_attn_norm"),
        })
    }

    fn layer_vram_size(&self, _idx: usize) -> u64 {
        self.layer_vram_estimate
    }

    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

/// Convert underscore-separated filename to dot-separated HuggingFace name.
fn filename_to_hf_name(filename: &str) -> String {
    const UNDERSCORE_COMPONENTS: &[&str] = &[
        "embed_tokens",
        "input_layernorm",
        "post_attention_layernorm",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "self_attn",
        "lm_head",
        "final_norm",
        "input_norm",
        "post_attn_norm",
    ];

    let mut result = filename.to_string();

    // Protect underscore components
    let mut protected = Vec::new();
    for (i, comp) in UNDERSCORE_COMPONENTS.iter().enumerate() {
        let placeholder = format!("XPLACEHOLDER{}X", i);
        if result.contains(comp) {
            result = result.replace(comp, &placeholder);
            protected.push((placeholder, comp.to_string()));
        }
    }

    // Convert underscores to dots
    result = result.replace('_', ".");

    // Restore protected components
    for (placeholder, original) in protected.into_iter().rev() {
        result = result.replace(&placeholder, &original);
    }

    // Remove .weight suffix if present
    if result.ends_with(".weight") {
        result = result[..result.len() - 7].to_string();
    }

    result
}

/// Extract layer index from HuggingFace weight name.
/// Returns None for non-layer weights (embed_tokens, final_norm, etc.)
fn extract_layer_index(hf_name: &str) -> Option<usize> {
    // Pattern: "model.layers.{idx}...." or "layers.{idx}...."
    let parts: Vec<&str> = hf_name.split('.').collect();

    for (i, part) in parts.iter().enumerate() {
        if *part == "layers" && i + 1 < parts.len() {
            if let Ok(idx) = parts[i + 1].parse::<usize>() {
                return Some(idx);
            }
        }
    }
    None
}

/// Extract weight name from HuggingFace name.
/// E.g., "model.layers.0.self_attn.q_proj" -> "q_proj"
fn extract_weight_name(hf_name: &str) -> String {
    // Map various naming conventions to our internal names
    if hf_name.contains("q_proj") {
        return "q_proj".to_string();
    }
    if hf_name.contains("k_proj") {
        return "k_proj".to_string();
    }
    if hf_name.contains("v_proj") {
        return "v_proj".to_string();
    }
    if hf_name.contains("o_proj") {
        return "o_proj".to_string();
    }
    if hf_name.contains("gate_proj") {
        return "gate_proj".to_string();
    }
    if hf_name.contains("up_proj") {
        return "up_proj".to_string();
    }
    if hf_name.contains("down_proj") {
        return "down_proj".to_string();
    }
    if hf_name.contains("input_layernorm") || hf_name.contains("input_norm") {
        return "input_norm".to_string();
    }
    if hf_name.contains("post_attention_layernorm") || hf_name.contains("post_attn_norm") {
        return "post_attn_norm".to_string();
    }

    // Fallback: last component
    hf_name.split('.').last().unwrap_or(hf_name).to_string()
}

/// Estimate VRAM usage for a single layer based on model config.
fn estimate_layer_vram(config: &ModelConfig) -> u64 {
    let hidden = config.hidden_size as u64;
    let intermediate = config.intermediate_size as u64;
    let num_heads = config.num_attention_heads as u64;
    let num_kv_heads = config.num_kv_heads as u64;
    let head_dim = config.head_dim as u64;

    // Attention projections (F16 = 2 bytes)
    let q_size = hidden * (num_heads * head_dim) * 2;
    let k_size = hidden * (num_kv_heads * head_dim) * 2;
    let v_size = hidden * (num_kv_heads * head_dim) * 2;
    let o_size = (num_heads * head_dim) * hidden * 2;

    // MLP projections
    let gate_size = hidden * intermediate * 2;
    let up_size = hidden * intermediate * 2;
    let down_size = intermediate * hidden * 2;

    // Norms (1D, small)
    let norm_size = hidden * 2 * 2; // input_norm + post_attn_norm

    q_size + k_size + v_size + o_size + gate_size + up_size + down_size + norm_size
}

#[cfg(test)]
mod tests {
    use super::super::arch::{Activation, ModelArch};
    use super::*;

    /// Creates a minimal mock ModelConfig for testing.
    fn mock_config(num_layers: usize) -> ModelConfig {
        ModelConfig {
            arch: ModelArch::Llama,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_layers,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            vocab_size: 32000,
            max_seq_len: 4096,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_scaling: None,
            attention_bias: false,
            mlp_bias: false,
            hidden_act: Activation::SiLU,
            tie_word_embeddings: false,
            sliding_window: None,
            bos_token_id: 1,
            eos_token_id: 2,
            pad_token_id: None,
        }
    }

    /// Mock layer loader for testing.
    struct MockLayerLoader {
        num_layers: usize,
        layer_size: u64,
        config: ModelConfig,
    }

    impl MockLayerLoader {
        fn new(num_layers: usize, layer_size_mb: u64) -> Self {
            Self {
                num_layers,
                layer_size: layer_size_mb * 1024 * 1024,
                config: mock_config(num_layers),
            }
        }
    }

    impl LayerLoader for MockLayerLoader {
        fn load_layer(
            &self,
            _idx: usize,
            _device: &Arc<CudaDevice>,
        ) -> Result<LayerWeights, InferenceError> {
            // Mock - would need actual device to test
            Err(InferenceError::Device(
                "Mock loader - no actual device".into(),
            ))
        }

        fn layer_vram_size(&self, _idx: usize) -> u64 {
            self.layer_size
        }

        fn num_layers(&self) -> usize {
            self.num_layers
        }

        fn config(&self) -> &ModelConfig {
            &self.config
        }
    }

    #[test]
    fn test_layer_size_calculation() {
        let loader = MockLayerLoader::new(28, 500); // 28 layers, 500MB each
        assert_eq!(loader.num_layers(), 28);
        assert_eq!(loader.layer_vram_size(0), 500 * 1024 * 1024);
    }
}
