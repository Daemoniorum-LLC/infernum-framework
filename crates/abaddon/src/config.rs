//! Configuration types for the Abaddon inference engine.

use std::path::PathBuf;

use infernum_core::{DeviceType, ModelSource, QuantizationType};
use serde::{Deserialize, Serialize};

/// Configuration for the inference engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Model source.
    pub model: ModelSource,

    /// Device to run inference on.
    pub device: DeviceType,

    /// Memory configuration.
    pub memory: MemoryConfig,

    /// Quantization to apply.
    pub quantization: Option<QuantizationType>,

    /// Maximum batch size.
    pub max_batch_size: u32,

    /// Maximum sequence length.
    pub max_seq_len: u32,

    /// Enable speculative decoding.
    pub speculative: Option<SpeculativeConfig>,

    /// Path to store downloaded models.
    pub cache_dir: Option<PathBuf>,

    /// HoloTensor configuration for progressive quality inference.
    /// When set, enables 70B+ models on 24GB VRAM.
    pub holotensor: Option<HoloTensorConfig>,
}

/// Configuration for HoloTensor progressive quality inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoloTensorConfig {
    /// VRAM budget in bytes. Default: auto-detect with 2GB headroom.
    pub vram_budget: usize,
    /// RAM budget in bytes. Default: 64GB.
    pub ram_budget: usize,
    /// Minimum quality to start inference (0.0-1.0). Default: 0.7.
    pub min_quality: f32,
    /// Target quality to improve to during generation (0.0-1.0). Default: 0.95.
    pub target_quality: f32,
    /// Enable async streaming from RAM to VRAM. Default: true.
    pub enable_streaming: bool,
    /// Prefetch distance in layers. Default: 2.
    pub prefetch_layers: usize,
}

impl Default for HoloTensorConfig {
    fn default() -> Self {
        Self {
            vram_budget: 22 * 1024 * 1024 * 1024, // 22GB (2GB headroom on 24GB)
            ram_budget: 64 * 1024 * 1024 * 1024,  // 64GB
            min_quality: 0.7,
            target_quality: 0.95,
            enable_streaming: true,
            prefetch_layers: 2,
        }
    }
}

impl HoloTensorConfig {
    /// Creates a configuration optimized for RTX 4500 Ada (24GB VRAM).
    #[must_use]
    pub fn for_rtx_4500() -> Self {
        Self {
            vram_budget: 22 * 1024 * 1024 * 1024,
            ram_budget: 64 * 1024 * 1024 * 1024,
            min_quality: 0.7,
            target_quality: 0.95,
            enable_streaming: true,
            prefetch_layers: 2,
        }
    }

    /// Creates a configuration for maximum quality (slower startup).
    #[must_use]
    pub fn high_quality() -> Self {
        Self {
            min_quality: 0.85,
            target_quality: 0.98,
            ..Self::default()
        }
    }

    /// Creates a configuration for fastest startup (lower initial quality).
    #[must_use]
    pub fn fast_startup() -> Self {
        Self {
            min_quality: 0.6,
            target_quality: 0.90,
            prefetch_layers: 4,
            ..Self::default()
        }
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model: ModelSource::HuggingFace {
                repo_id: String::new(),
                revision: None,
            },
            device: DeviceType::Cpu,
            memory: MemoryConfig::default(),
            quantization: None,
            max_batch_size: 32,
            max_seq_len: 4096,
            speculative: None,
            cache_dir: None,
            holotensor: None,
        }
    }
}

impl EngineConfig {
    /// Creates a new configuration builder.
    #[must_use]
    pub fn builder() -> EngineConfigBuilder {
        EngineConfigBuilder::default()
    }
}

/// Builder for `EngineConfig`.
#[derive(Debug, Default)]
pub struct EngineConfigBuilder {
    model: Option<ModelSource>,
    device: Option<DeviceType>,
    memory: Option<MemoryConfig>,
    quantization: Option<QuantizationType>,
    max_batch_size: Option<u32>,
    max_seq_len: Option<u32>,
    speculative: Option<SpeculativeConfig>,
    cache_dir: Option<PathBuf>,
    holotensor: Option<HoloTensorConfig>,
}

impl EngineConfigBuilder {
    /// Sets the model from a HuggingFace repository ID or local path.
    ///
    /// Auto-detects the source type:
    /// - If path exists and contains .hct files → HoloTensor
    /// - If path exists and is a directory → LocalPath
    /// - If path exists and is a .gguf file → Gguf
    /// - Otherwise → HuggingFace repo ID
    #[must_use]
    pub fn model(mut self, model_id: impl Into<String>) -> Self {
        let model_id = model_id.into();
        let path = std::path::Path::new(&model_id);

        // Check if it's a local path
        if path.exists() {
            if path.is_dir() {
                // Check if it's an HCT directory (contains .hct files)
                if let Ok(entries) = std::fs::read_dir(path) {
                    let has_hct = entries
                        .filter_map(|e| e.ok())
                        .any(|e| e.path().extension().map_or(false, |ext| ext == "hct"));
                    if has_hct {
                        tracing::info!("Auto-detected HoloTensor model directory");
                        self.model = Some(ModelSource::holotensor(path));
                        return self;
                    }
                }
                // Regular local directory
                tracing::info!("Auto-detected local model directory");
                self.model = Some(ModelSource::local(path));
            } else if path.extension().map_or(false, |ext| ext == "gguf") {
                tracing::info!("Auto-detected GGUF model file");
                self.model = Some(ModelSource::gguf(path));
            } else {
                // Regular file, treat as local
                self.model = Some(ModelSource::local(path));
            }
        } else {
            // Not a local path, treat as HuggingFace repo ID
            self.model = Some(ModelSource::huggingface(model_id));
        }
        self
    }

    /// Sets the model source directly.
    #[must_use]
    pub fn model_source(mut self, source: ModelSource) -> Self {
        self.model = Some(source);
        self
    }

    /// Sets the device.
    #[must_use]
    pub fn device(mut self, device: DeviceType) -> Self {
        self.device = Some(device);
        self
    }

    /// Sets to use CUDA with the specified device.
    #[must_use]
    pub fn cuda(mut self, device_id: usize) -> Self {
        self.device = Some(DeviceType::Cuda { device_id });
        self
    }

    /// Sets to use Metal.
    #[must_use]
    pub fn metal(mut self) -> Self {
        self.device = Some(DeviceType::Metal { device_id: 0 });
        self
    }

    /// Sets the memory configuration.
    #[must_use]
    pub fn memory(mut self, config: MemoryConfig) -> Self {
        self.memory = Some(config);
        self
    }

    /// Sets the quantization.
    #[must_use]
    pub fn quantization(mut self, quant: QuantizationType) -> Self {
        self.quantization = Some(quant);
        self
    }

    /// Sets the maximum batch size.
    #[must_use]
    pub fn max_batch_size(mut self, size: u32) -> Self {
        self.max_batch_size = Some(size);
        self
    }

    /// Sets the maximum sequence length.
    #[must_use]
    pub fn max_seq_len(mut self, len: u32) -> Self {
        self.max_seq_len = Some(len);
        self
    }

    /// Enables speculative decoding.
    #[must_use]
    pub fn speculative(mut self, config: SpeculativeConfig) -> Self {
        self.speculative = Some(config);
        self
    }

    /// Sets the cache directory.
    #[must_use]
    pub fn cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(path.into());
        self
    }

    /// Enables HoloTensor progressive quality inference.
    ///
    /// This enables 70B+ models on 24GB VRAM via holographic compression.
    #[must_use]
    pub fn holotensor(mut self, config: HoloTensorConfig) -> Self {
        self.holotensor = Some(config);
        self
    }

    /// Enables HoloTensor with RTX 4500 Ada optimized settings.
    #[must_use]
    pub fn holotensor_rtx4500(self) -> Self {
        self.holotensor(HoloTensorConfig::for_rtx_4500())
    }

    /// Sets a HoloTensor model from a local HCT directory.
    ///
    /// This is a convenience method that sets both the model source and
    /// enables HoloTensor with default settings.
    #[must_use]
    pub fn holotensor_model(mut self, path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        self.model = Some(ModelSource::holotensor(&path));
        self.holotensor = Some(HoloTensorConfig::for_rtx_4500());
        self
    }

    /// Builds the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if required fields are missing.
    pub fn build(self) -> Result<EngineConfig, String> {
        let model = self.model.ok_or("model is required")?;

        // Auto-detect HoloTensor config if model is HoloTensor source
        let holotensor = if model.is_holotensor() && self.holotensor.is_none() {
            Some(HoloTensorConfig::default())
        } else {
            self.holotensor
        };

        // Auto-detect best device if not specified
        let device = self.device.unwrap_or_else(|| {
            let best = crate::device::best_device();
            tracing::info!(device = ?best, "Auto-detected compute device");
            best
        });

        Ok(EngineConfig {
            model,
            device,
            memory: self.memory.unwrap_or_default(),
            quantization: self.quantization,
            max_batch_size: self.max_batch_size.unwrap_or(32),
            max_seq_len: self.max_seq_len.unwrap_or(4096),
            speculative: self.speculative,
            cache_dir: self.cache_dir,
            holotensor,
        })
    }
}

/// Memory configuration for the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum GPU memory to use in bytes (0 = auto-detect).
    pub gpu_memory_limit: usize,

    /// Fraction of memory to use for KV cache (0.0-1.0).
    pub kv_cache_fraction: f32,

    /// Enable memory-mapped model loading.
    pub mmap_enabled: bool,

    /// Offload layers to CPU when GPU memory is exhausted.
    pub cpu_offload: bool,

    /// Number of layers to keep on GPU (None = all).
    pub gpu_layers: Option<u32>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            gpu_memory_limit: 0,
            kv_cache_fraction: 0.9,
            mmap_enabled: true,
            cpu_offload: false,
            gpu_layers: None,
        }
    }
}

impl MemoryConfig {
    /// Creates a configuration optimized for low memory usage.
    #[must_use]
    pub fn low_memory() -> Self {
        Self {
            gpu_memory_limit: 0,
            kv_cache_fraction: 0.5,
            mmap_enabled: true,
            cpu_offload: true,
            gpu_layers: Some(20),
        }
    }

    /// Creates a configuration for maximum throughput.
    #[must_use]
    pub fn high_throughput() -> Self {
        Self {
            gpu_memory_limit: 0,
            kv_cache_fraction: 0.95,
            mmap_enabled: true,
            cpu_offload: false,
            gpu_layers: None,
        }
    }

    /// Creates a configuration optimized for RTX 4000 series / Ada Lovelace GPUs.
    ///
    /// Optimized for:
    /// - 24GB VRAM (RTX 4090, RTX 4500 Ada)
    /// - BF16 tensor core operations
    /// - Large KV cache for long contexts
    #[must_use]
    pub fn rtx_4000_series() -> Self {
        Self {
            gpu_memory_limit: 22 * 1024 * 1024 * 1024, // Reserve 2GB for overhead
            kv_cache_fraction: 0.92,                   // More cache for 128K contexts
            mmap_enabled: true,
            cpu_offload: false, // Plenty of VRAM
            gpu_layers: None,   // All layers on GPU
        }
    }

    /// Creates a configuration for professional workstation GPUs.
    ///
    /// Optimized for:
    /// - Large VRAM (24-48GB) typical of RTX A5000/A6000, RTX 4500/5000 Ada
    /// - Sustained workloads with ECC memory
    /// - Maximum throughput for batch processing
    #[must_use]
    pub fn workstation_gpu() -> Self {
        Self {
            gpu_memory_limit: 0,     // Auto-detect full VRAM
            kv_cache_fraction: 0.90, // Slightly conservative for stability
            mmap_enabled: true,
            cpu_offload: false,
            gpu_layers: None,
        }
    }

    /// Creates a configuration for 70B+ parameter models on high-VRAM GPUs.
    ///
    /// Optimized for:
    /// - Models like Llama-70B, Mixtral, etc.
    /// - Maximizing model fit in VRAM
    /// - Reducing KV cache to fit more layers
    #[must_use]
    pub fn large_model() -> Self {
        Self {
            gpu_memory_limit: 0,
            kv_cache_fraction: 0.5, // Smaller cache to fit model weights
            mmap_enabled: true,
            cpu_offload: true,    // Offload some layers if needed
            gpu_layers: Some(60), // Keep most layers on GPU
        }
    }
}

/// Configuration for speculative decoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeConfig {
    /// Draft model to use.
    pub draft_model: ModelSource,

    /// Number of speculative tokens to generate.
    pub num_speculative_tokens: u32,

    /// Acceptance threshold.
    pub acceptance_threshold: f32,
}

impl SpeculativeConfig {
    /// Creates a new speculative decoding configuration.
    #[must_use]
    pub fn new(draft_model: ModelSource) -> Self {
        Self {
            draft_model,
            num_speculative_tokens: 5,
            acceptance_threshold: 0.9,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // EngineConfig tests
    // ==========================================================================

    #[test]
    fn test_engine_config_default() {
        let config = EngineConfig::default();
        assert!(matches!(config.model, ModelSource::HuggingFace { .. }));
        assert_eq!(config.device, DeviceType::Cpu);
        assert!(config.quantization.is_none());
        assert_eq!(config.max_batch_size, 32);
        assert_eq!(config.max_seq_len, 4096);
        assert!(config.speculative.is_none());
        assert!(config.cache_dir.is_none());
    }

    #[test]
    fn test_engine_config_builder_basic() {
        let config = EngineConfig::builder()
            .model("meta-llama/Llama-3.2-3B")
            .build()
            .expect("build");

        match &config.model {
            ModelSource::HuggingFace { repo_id, .. } => {
                assert_eq!(repo_id, "meta-llama/Llama-3.2-3B");
            },
            _ => panic!("Expected HuggingFace source"),
        }
    }

    #[test]
    fn test_engine_config_builder_no_model_fails() {
        let result = EngineConfig::builder().build();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("model is required"));
    }

    #[test]
    fn test_engine_config_builder_full() {
        let config = EngineConfig::builder()
            .model("test-model")
            .cuda(0)
            .max_batch_size(64)
            .max_seq_len(8192)
            .quantization(QuantizationType::GgufQ4KM)
            .cache_dir("/tmp/models")
            .build()
            .expect("build");

        assert!(matches!(config.device, DeviceType::Cuda { device_id: 0 }));
        assert_eq!(config.max_batch_size, 64);
        assert_eq!(config.max_seq_len, 8192);
        assert_eq!(config.quantization, Some(QuantizationType::GgufQ4KM));
        assert_eq!(config.cache_dir, Some(PathBuf::from("/tmp/models")));
    }

    #[test]
    fn test_engine_config_builder_metal() {
        let config = EngineConfig::builder()
            .model("test")
            .metal()
            .build()
            .expect("build");

        assert!(matches!(config.device, DeviceType::Metal { device_id: 0 }));
    }

    #[test]
    fn test_engine_config_builder_model_source() {
        let source = ModelSource::local("/path/to/model");
        let config = EngineConfig::builder()
            .model_source(source)
            .build()
            .expect("build");

        assert!(matches!(config.model, ModelSource::LocalPath { .. }));
    }

    #[test]
    fn test_engine_config_serialization() {
        let config = EngineConfig::builder()
            .model("test-model")
            .max_batch_size(16)
            .build()
            .expect("build");

        let json = serde_json::to_string(&config).expect("serialize");
        assert!(json.contains("test-model"));
        assert!(json.contains("16"));

        let parsed: EngineConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.max_batch_size, 16);
    }

    #[test]
    fn test_engine_config_clone() {
        let config = EngineConfig::builder()
            .model("clone-test")
            .build()
            .expect("build");

        let cloned = config.clone();
        assert_eq!(cloned.max_batch_size, config.max_batch_size);
    }

    // ==========================================================================
    // MemoryConfig tests
    // ==========================================================================

    #[test]
    fn test_memory_config_default() {
        let config = MemoryConfig::default();
        assert_eq!(config.gpu_memory_limit, 0);
        assert!((config.kv_cache_fraction - 0.9).abs() < 0.01);
        assert!(config.mmap_enabled);
        assert!(!config.cpu_offload);
        assert!(config.gpu_layers.is_none());
    }

    #[test]
    fn test_memory_config_low_memory() {
        let config = MemoryConfig::low_memory();
        assert!((config.kv_cache_fraction - 0.5).abs() < 0.01);
        assert!(config.cpu_offload);
        assert_eq!(config.gpu_layers, Some(20));
    }

    #[test]
    fn test_memory_config_high_throughput() {
        let config = MemoryConfig::high_throughput();
        assert!((config.kv_cache_fraction - 0.95).abs() < 0.01);
        assert!(!config.cpu_offload);
        assert!(config.gpu_layers.is_none());
    }

    #[test]
    fn test_memory_config_rtx_4000() {
        let config = MemoryConfig::rtx_4000_series();
        assert_eq!(config.gpu_memory_limit, 22 * 1024 * 1024 * 1024);
        assert!((config.kv_cache_fraction - 0.92).abs() < 0.01);
        assert!(!config.cpu_offload);
    }

    #[test]
    fn test_memory_config_workstation() {
        let config = MemoryConfig::workstation_gpu();
        assert_eq!(config.gpu_memory_limit, 0); // Auto-detect
        assert!((config.kv_cache_fraction - 0.90).abs() < 0.01);
    }

    #[test]
    fn test_memory_config_large_model() {
        let config = MemoryConfig::large_model();
        assert!((config.kv_cache_fraction - 0.5).abs() < 0.01);
        assert!(config.cpu_offload);
        assert_eq!(config.gpu_layers, Some(60));
    }

    #[test]
    fn test_memory_config_serialization() {
        let config = MemoryConfig {
            gpu_memory_limit: 1000,
            kv_cache_fraction: 0.8,
            mmap_enabled: false,
            cpu_offload: true,
            gpu_layers: Some(10),
        };

        let json = serde_json::to_string(&config).expect("serialize");
        let parsed: MemoryConfig = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.gpu_memory_limit, 1000);
        assert!((parsed.kv_cache_fraction - 0.8).abs() < 0.01);
        assert!(!parsed.mmap_enabled);
        assert!(parsed.cpu_offload);
        assert_eq!(parsed.gpu_layers, Some(10));
    }

    #[test]
    fn test_memory_config_clone() {
        let config = MemoryConfig::low_memory();
        let cloned = config.clone();
        assert_eq!(cloned.gpu_layers, config.gpu_layers);
        assert!((cloned.kv_cache_fraction - config.kv_cache_fraction).abs() < 0.01);
    }

    // ==========================================================================
    // SpeculativeConfig tests
    // ==========================================================================

    #[test]
    fn test_speculative_config_new() {
        let draft_model = ModelSource::huggingface("draft-model");
        let config = SpeculativeConfig::new(draft_model);

        assert_eq!(config.num_speculative_tokens, 5);
        assert!((config.acceptance_threshold - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_speculative_config_serialization() {
        let config = SpeculativeConfig {
            draft_model: ModelSource::huggingface("small-model"),
            num_speculative_tokens: 8,
            acceptance_threshold: 0.85,
        };

        let json = serde_json::to_string(&config).expect("serialize");
        let parsed: SpeculativeConfig = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.num_speculative_tokens, 8);
        assert!((parsed.acceptance_threshold - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_speculative_config_clone() {
        let config = SpeculativeConfig::new(ModelSource::huggingface("model"));
        let cloned = config.clone();
        assert_eq!(cloned.num_speculative_tokens, config.num_speculative_tokens);
    }

    // ==========================================================================
    // EngineConfigBuilder edge cases
    // ==========================================================================

    #[test]
    fn test_builder_memory_config() {
        let memory = MemoryConfig::low_memory();
        let config = EngineConfig::builder()
            .model("test")
            .memory(memory.clone())
            .build()
            .expect("build");

        assert!(config.memory.cpu_offload);
    }

    #[test]
    fn test_builder_speculative_config() {
        let spec = SpeculativeConfig::new(ModelSource::huggingface("draft"));
        let config = EngineConfig::builder()
            .model("main")
            .speculative(spec)
            .build()
            .expect("build");

        assert!(config.speculative.is_some());
    }

    #[test]
    fn test_builder_device_overwrite() {
        // Device can be overwritten
        let config = EngineConfig::builder()
            .model("test")
            .cuda(0)
            .metal() // Overwrites cuda
            .build()
            .expect("build");

        assert!(matches!(config.device, DeviceType::Metal { .. }));
    }
}
