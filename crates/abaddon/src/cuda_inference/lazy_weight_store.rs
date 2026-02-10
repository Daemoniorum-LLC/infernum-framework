//! Lazy-loading GPU weight storage for memory-efficient inference.
//!
//! Unlike `WeightStore` which loads all layers upfront, `LazyWeightStore`
//! loads layers on-demand using the `LazyLayerStore` mechanism.
//!
//! This enables running models larger than available VRAM by keeping only
//! a subset of layers loaded at any time.

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use tracing::info;

use super::arch::{ModelArch, ModelConfig};
use super::lazy_layers::{HoloLayerLoader, LayerLoader, LazyLayerStats, LazyLayerStore};
use super::tensor::{GpuDType, GpuTensor};
use super::weight_store::{LayerWeights, RMSNormWeights};
use super::InferenceError;

/// Default VRAM budget for layers: 80% of free memory.
const DEFAULT_VRAM_BUDGET_RATIO: f64 = 0.80;

/// Configuration for lazy weight loading.
#[derive(Debug, Clone)]
pub struct LazyWeightConfig {
    /// Maximum VRAM budget for layer storage in bytes.
    /// If None, uses 80% of available VRAM minus overhead.
    pub vram_budget: Option<u64>,

    /// CUDA device ID.
    pub device_id: usize,

    /// Model architecture (or None to auto-detect).
    pub arch: Option<ModelArch>,
}

impl Default for LazyWeightConfig {
    fn default() -> Self {
        Self {
            vram_budget: None,
            device_id: 0,
            arch: None,
        }
    }
}

impl LazyWeightConfig {
    /// Create config with explicit VRAM budget.
    pub fn with_vram_budget(vram_gb: f64) -> Self {
        Self {
            vram_budget: Some((vram_gb * 1024.0 * 1024.0 * 1024.0) as u64),
            ..Default::default()
        }
    }

    /// Create config for 24GB GPU.
    ///
    /// Uses 14GB budget for layers, leaving ~10GB for:
    /// - Shared weights (embed_tokens + lm_head): ~4GB for 70B
    /// - Generator working buffers: ~2GB
    /// - KV cache: ~640MB
    /// - HoloTensor decompression temp buffers
    /// - CUDA memory fragmentation headroom
    pub fn for_24gb_gpu() -> Self {
        Self {
            vram_budget: Some(14 * 1024 * 1024 * 1024), // 14GB for layers
            ..Default::default()
        }
    }
}

/// GPU-resident model weights with lazy layer loading.
///
/// Always keeps embedding, final_norm, and lm_head loaded.
/// Loads transformer layers on-demand with LRU eviction.
pub struct LazyWeightStore {
    /// Model configuration.
    pub config: ModelConfig,

    /// CUDA device.
    device: Arc<CudaDevice>,

    /// Token embeddings [vocab_size, hidden_size].
    pub embed_tokens: GpuTensor,

    /// Lazy layer storage.
    layer_store: LazyLayerStore,

    /// Final layer norm.
    pub final_norm: RMSNormWeights,

    /// LM head projection [hidden_size, vocab_size].
    /// None if tied to embed_tokens.
    pub lm_head: Option<GpuTensor>,

    /// Memory used by shared weights (embed, norm, lm_head).
    shared_memory: usize,
}

impl LazyWeightStore {
    /// Load HoloTensor model with lazy layer loading.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Directory containing .hct files and config.json
    /// * `config` - Lazy loading configuration
    pub fn load_holotensor(
        model_dir: impl AsRef<Path>,
        lazy_config: LazyWeightConfig,
    ) -> Result<Self, InferenceError> {
        let model_dir = model_dir.as_ref();

        // Initialize CUDA
        // Note: CudaDevice::new() already returns Arc<CudaDevice>
        let device = CudaDevice::new(lazy_config.device_id)
            .map_err(|e| InferenceError::Device(e.to_string()))?;

        // Detect architecture and load config
        let arch = match lazy_config.arch {
            Some(a) => a,
            None => Self::detect_arch(model_dir)?,
        };

        let model_config = Self::load_model_config(model_dir, arch)?;

        info!(
            "Loading HoloTensor model: {} layers, hidden={}, intermediate={}",
            model_config.num_layers, model_config.hidden_size, model_config.intermediate_size
        );

        // Create HoloLayerLoader
        let loader = Arc::new(HoloLayerLoader::new(
            model_dir,
            model_config.clone(),
            Arc::clone(&device),
        )?);

        // Load shared weights (embed_tokens, final_norm, lm_head)
        let shared_files = loader.shared_files();
        let (embed_tokens, final_norm, lm_head, shared_memory) =
            Self::load_shared_weights(shared_files, &device, &loader)?;

        // Calculate VRAM budget for layers
        let vram_budget = lazy_config.vram_budget.unwrap_or_else(|| {
            // Query available VRAM (simplified - assume 24GB)
            // In production, query device.total_memory()
            let total_vram = 24 * 1024 * 1024 * 1024u64; // 24GB
            let overhead = shared_memory as u64 + 6 * 1024 * 1024 * 1024; // shared + 6GB buffers
            ((total_vram.saturating_sub(overhead)) as f64 * DEFAULT_VRAM_BUDGET_RATIO) as u64
        });

        info!(
            "VRAM budget for layers: {:.2} GB ({} layers available)",
            vram_budget as f64 / (1024.0 * 1024.0 * 1024.0),
            vram_budget / loader.layer_vram_size(0)
        );

        // Create lazy layer store
        let layer_store = LazyLayerStore::new(loader, Arc::clone(&device), vram_budget);

        Ok(Self {
            config: model_config,
            device,
            embed_tokens,
            layer_store,
            final_norm,
            lm_head,
            shared_memory,
        })
    }

    /// Get a layer, loading it if necessary.
    ///
    /// This may evict other layers if VRAM budget is exceeded.
    pub fn get_layer(&mut self, idx: usize) -> Result<&LayerWeights, InferenceError> {
        self.layer_store.get_layer(idx)
    }

    /// Check if a layer is currently loaded.
    pub fn is_layer_loaded(&self, idx: usize) -> bool {
        self.layer_store.is_loaded(idx)
    }

    /// Prefetch layers (load them proactively).
    pub fn prefetch_layers(&mut self, indices: &[usize]) -> Result<(), InferenceError> {
        self.layer_store.prefetch(indices)
    }

    /// Evict a specific layer to free VRAM.
    pub fn evict_layer(&mut self, idx: usize) {
        self.layer_store.evict(idx)
    }

    /// Evict all layers to free VRAM.
    pub fn evict_all_layers(&mut self) {
        self.layer_store.evict_all()
    }

    /// Get total number of layers.
    pub fn num_layers(&self) -> usize {
        self.layer_store.num_layers()
    }

    /// Get lazy loading statistics.
    pub fn stats(&self) -> LazyLayerStats {
        self.layer_store.stats()
    }

    /// Get CUDA device reference.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get total memory used by shared weights.
    pub fn shared_memory(&self) -> usize {
        self.shared_memory
    }

    // Internal: Detect model architecture from config files
    fn detect_arch(model_dir: &Path) -> Result<ModelArch, InferenceError> {
        // Try config.json first
        let config_path = model_dir.join("config.json");
        if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
                InferenceError::ModelLoad(format!("Cannot read config.json: {}", e))
            })?;

            // Parse and detect architecture
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&config_str) {
                if let Some(arch_str) = json.get("model_type").and_then(|v| v.as_str()) {
                    return match arch_str.to_lowercase().as_str() {
                        "llama" => Ok(ModelArch::Llama),
                        "mistral" => Ok(ModelArch::Mistral),
                        "qwen2" | "qwen" => Ok(ModelArch::Qwen),
                        "phi" | "phi-msft" | "phi3" => Ok(ModelArch::Phi),
                        _ => Err(InferenceError::UnsupportedArch(arch_str.to_string())),
                    };
                }
            }
        }

        // Default to Llama
        Ok(ModelArch::Llama)
    }

    // Internal: Load model configuration using existing parser
    fn load_model_config(model_dir: &Path, arch: ModelArch) -> Result<ModelConfig, InferenceError> {
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| InferenceError::ModelLoad(format!("Cannot read config.json: {}", e)))?;

        // Use the existing from_json method for robust parsing
        ModelConfig::from_json(&config_str, arch)
    }

    // Internal: Load shared weights (embed_tokens, final_norm, lm_head)
    fn load_shared_weights(
        shared_files: &std::collections::HashMap<String, std::path::PathBuf>,
        device: &Arc<CudaDevice>,
        loader: &HoloLayerLoader,
    ) -> Result<(GpuTensor, RMSNormWeights, Option<GpuTensor>, usize), InferenceError> {
        use std::fs::File;
        use std::io::BufReader;

        use cudarc::driver::CudaSlice;
        use haagenti::holotensor::{HoloTensorDecoder, HoloTensorReader};

        let mut total_memory = 0usize;

        // Helper to load a HoloTensor file
        let load_file = |name: &str| -> Result<GpuTensor, InferenceError> {
            // Try various naming conventions
            let key = shared_files
                .keys()
                .find(|k| k.contains(name))
                .ok_or_else(|| {
                    InferenceError::ModelLoad(format!("Missing shared weight: {}", name))
                })?;

            let path = shared_files.get(key).unwrap();

            let file = File::open(path).map_err(|e| {
                InferenceError::ModelLoad(format!("Cannot open {}: {}", path.display(), e))
            })?;
            let reader = BufReader::new(file);
            let mut holo_reader = HoloTensorReader::new(reader).map_err(|e| {
                InferenceError::ModelLoad(format!("Failed to parse HoloTensor: {}", e))
            })?;

            let (header, fragments) = holo_reader.read_all().map_err(|e| {
                InferenceError::ModelLoad(format!("Failed to read fragments: {}", e))
            })?;

            let shape: Vec<usize> = header.shape.iter().map(|&d| d as usize).collect();

            // CPU reconstruction for all shared weights (simpler, works for any dimension)
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

            GpuTensor::from_cuda_slice(gpu_data, shape, GpuDType::F16, Arc::clone(device))
        };

        // Load embed_tokens
        let embed_tokens = load_file("embed_tokens")?;
        total_memory += embed_tokens.size_bytes();
        info!(
            "Loaded embed_tokens: {} MB",
            embed_tokens.size_bytes() / (1024 * 1024)
        );

        // Load final_norm (try various names)
        let final_norm_tensor = load_file("norm").or_else(|_| load_file("final_norm"))?;
        total_memory += final_norm_tensor.size_bytes();
        let final_norm = RMSNormWeights {
            weight: final_norm_tensor,
        };

        // Load lm_head (optional - may be tied to embed_tokens)
        let lm_head = match load_file("lm_head") {
            Ok(tensor) => {
                total_memory += tensor.size_bytes();
                info!("Loaded lm_head: {} MB", tensor.size_bytes() / (1024 * 1024));
                Some(tensor)
            },
            Err(_) => {
                info!("No lm_head found - assuming tied embeddings");
                None
            },
        };

        info!(
            "Total shared weight memory: {} MB",
            total_memory / (1024 * 1024)
        );

        Ok((embed_tokens, final_norm, lm_head, total_memory))
    }
}

impl std::fmt::Debug for LazyWeightStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats();
        f.debug_struct("LazyWeightStore")
            .field("arch", &self.config.arch)
            .field("num_layers", &self.config.num_layers)
            .field("layers_loaded", &stats.layers_loaded)
            .field("vram_used_mb", &(stats.vram_used / (1024 * 1024)))
            .field("vram_budget_mb", &(stats.vram_budget / (1024 * 1024)))
            .field("hit_rate", &format!("{:.1}%", stats.hit_rate * 100.0))
            .finish()
    }
}
