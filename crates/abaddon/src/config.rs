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
}

impl EngineConfigBuilder {
    /// Sets the model from a HuggingFace repository ID.
    #[must_use]
    pub fn model(mut self, repo_id: impl Into<String>) -> Self {
        self.model = Some(ModelSource::huggingface(repo_id));
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

    /// Builds the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if required fields are missing.
    pub fn build(self) -> Result<EngineConfig, String> {
        let model = self.model.ok_or("model is required")?;

        Ok(EngineConfig {
            model,
            device: self.device.unwrap_or(DeviceType::Cpu),
            memory: self.memory.unwrap_or_default(),
            quantization: self.quantization,
            max_batch_size: self.max_batch_size.unwrap_or(32),
            max_seq_len: self.max_seq_len.unwrap_or(4096),
            speculative: self.speculative,
            cache_dir: self.cache_dir,
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
