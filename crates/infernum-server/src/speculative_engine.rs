//! Speculative decoding engine integration for the HTTP server.
//!
//! This module provides server integration for speculative decoding,
//! wrapping `Speculative405B` from abaddon with server-compatible interfaces.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                     SpeculativeEngine                               │
//! │  ┌──────────────────────────────────────────────────────────────┐  │
//! │  │ Draft Model (1B-8B)                                          │  │
//! │  │ - Fully loaded in VRAM (Qwen2 or Llama)                      │  │
//! │  │ - Fast token generation                                      │  │
//! │  └──────────────────────────────────────────────────────────────┘  │
//! │                                                                     │
//! │  ┌──────────────────────────────────────────────────────────────┐  │
//! │  │ Target Model (70B-405B)                                      │  │
//! │  │ - TieredHoloLoader with NVMe cache                           │  │
//! │  │ - Layer streaming for large models (LazyLlama)               │  │
//! │  └──────────────────────────────────────────────────────────────┘  │
//! │                                                                     │
//! │  Speculative405B coordinates draft → verify → accept/reject        │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use parking_lot::Mutex;

use abaddon::holotensor::tiered_loading::{TieredConfig, TieredHoloLoader};
use abaddon::lazy_varbuilder::LazyVarBuilder;
use abaddon::loader::{ModelConfig, WeightFiles};
use abaddon::models::lazy_llama::LazyLlama;
use abaddon::models::llama::{Llama, LlamaConfig};
use abaddon::models::qwen2::{Qwen2, Qwen2Config};
use abaddon::models::ArchitectureType;
use abaddon::speculative_405b::{Speculative405B, Speculative405BConfig, Speculative405BStats};
use abaddon::Tokenizer;

/// Error type for speculative engine operations.
#[derive(Debug, thiserror::Error)]
pub enum SpeculativeEngineError {
    /// Model loading error.
    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Inference error.
    #[error("Inference error: {0}")]
    Inference(String),

    /// Tokenizer error.
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    /// Device error.
    #[error("Device error: {0}")]
    Device(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Configuration for the speculative engine.
#[derive(Debug, Clone)]
pub struct SpeculativeEngineConfig {
    /// Path to the draft model (small, 1B-8B).
    pub draft_model_path: String,

    /// Path to the target model (large, 70B-405B).
    pub target_model_path: String,

    /// Number of draft tokens per speculation round.
    pub num_draft_tokens: usize,

    /// Acceptance threshold (0.0-1.0). Higher = stricter verification.
    pub acceptance_threshold: f32,

    /// VRAM budget in bytes (for target model layer loading).
    pub vram_budget: u64,

    /// RAM budget in bytes (for target model layer caching).
    pub ram_budget: u64,

    /// NVMe cache directory (optional).
    pub cache_dir: Option<String>,

    /// CUDA device ID.
    pub device_id: usize,

    /// Whether to use greedy decoding for drafts.
    pub greedy_draft: bool,

    /// Min quality for HoloTensor loading.
    pub min_quality: f32,

    /// Target quality for HoloTensor loading.
    pub target_quality: f32,

    /// Maximum number of decoder layers to keep in memory for target model.
    /// For 405B: ~32 layers fit in 64GB RAM. For 70B: all layers can fit.
    pub max_loaded_layers: Option<usize>,
}

impl Default for SpeculativeEngineConfig {
    fn default() -> Self {
        Self {
            draft_model_path: String::new(),
            target_model_path: String::new(),
            num_draft_tokens: 5,
            acceptance_threshold: 0.1,
            vram_budget: 8 * 1024 * 1024 * 1024, // 8GB
            ram_budget: 64 * 1024 * 1024 * 1024, // 64GB
            cache_dir: None,
            device_id: 0,
            greedy_draft: true,
            min_quality: 0.7,
            target_quality: 0.95,
            max_loaded_layers: Some(32), // Default to 32 layers (~fits in 64GB)
        }
    }
}

/// Draft model type - either Qwen2 or Llama.
enum DraftModelKind {
    Qwen2(Qwen2),
    Llama(Llama),
}

/// Wrapper to implement DraftModel for the enum.
struct DraftModelWrapper {
    kind: DraftModelKind,
}

impl abaddon::speculative_405b::DraftModel for DraftModelWrapper {
    fn forward(
        &mut self,
        input_ids: &candle_core::Tensor,
        pos: usize,
    ) -> candle_core::Result<candle_core::Tensor> {
        match &mut self.kind {
            DraftModelKind::Qwen2(model) => model.forward(input_ids, pos),
            DraftModelKind::Llama(model) => model.forward(input_ids, pos),
        }
    }

    fn clear_cache(&mut self) {
        match &mut self.kind {
            DraftModelKind::Qwen2(model) => model.clear_cache(),
            DraftModelKind::Llama(model) => model.clear_cache(),
        }
    }

    fn device(&self) -> &Device {
        match &self.kind {
            DraftModelKind::Qwen2(model) => model.device(),
            DraftModelKind::Llama(model) => model.device(),
        }
    }

    fn dtype(&self) -> DType {
        match &self.kind {
            DraftModelKind::Qwen2(model) => model.dtype(),
            DraftModelKind::Llama(model) => model.dtype(),
        }
    }
}

/// Speculative decoding engine for the HTTP server.
pub struct SpeculativeEngine {
    /// The speculative decoder (type-erased for server use).
    inner: Arc<Mutex<Speculative405B<DraftModelWrapper, LazyLlama>>>,
    /// Tokenizer for text processing.
    tokenizer: Arc<Tokenizer>,
    /// Configuration.
    config: SpeculativeEngineConfig,
    /// Model identifier (for API responses).
    model_id: String,
    /// EOS token ID.
    eos_token_id: u32,
}

impl SpeculativeEngine {
    /// Creates a new speculative engine by loading both models.
    pub async fn new(config: SpeculativeEngineConfig) -> Result<Self, SpeculativeEngineError> {
        let start = std::time::Instant::now();

        // Create CUDA device
        let device = Device::new_cuda(config.device_id).map_err(|e| {
            SpeculativeEngineError::Device(format!("Failed to create CUDA device: {}", e))
        })?;
        let dtype = DType::BF16;

        tracing::info!(
            draft = %config.draft_model_path,
            target = %config.target_model_path,
            device = config.device_id,
            "Loading speculative decoding models"
        );

        // Load draft model (small, fully in VRAM)
        tracing::info!(path = %config.draft_model_path, "Loading draft model");
        let draft_start = std::time::Instant::now();
        let (draft_model, draft_tokenizer, draft_eos) =
            Self::load_draft_model(&config.draft_model_path, &device, dtype)?;
        tracing::info!(
            elapsed_ms = draft_start.elapsed().as_millis(),
            "Draft model loaded"
        );

        // Load target model (large, with lazy layer loading)
        tracing::info!(path = %config.target_model_path, "Loading target model");
        let target_start = std::time::Instant::now();
        let target_model = Self::load_target_model(&config, &device, dtype)?;
        tracing::info!(
            elapsed_ms = target_start.elapsed().as_millis(),
            "Target model initialized (lazy loading enabled)"
        );

        // Create speculative decoder
        let spec_config = Speculative405BConfig {
            num_draft_tokens: config.num_draft_tokens,
            acceptance_threshold: config.acceptance_threshold,
            greedy_draft: config.greedy_draft,
            draft_temperature: 0.7,
            target_temperature: 0.7,
        };

        let speculative = Speculative405B::new(draft_model, target_model, spec_config);

        let model_id = Path::new(&config.target_model_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("speculative-model")
            .to_string();

        tracing::info!(
            elapsed_ms = start.elapsed().as_millis(),
            model_id = %model_id,
            "Speculative engine initialized"
        );

        Ok(Self {
            inner: Arc::new(Mutex::new(speculative)),
            tokenizer: Arc::new(draft_tokenizer),
            config,
            model_id,
            eos_token_id: draft_eos,
        })
    }

    /// Loads the draft model (small model, fully in VRAM).
    fn load_draft_model(
        path: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<(DraftModelWrapper, Tokenizer, u32), SpeculativeEngineError> {
        let path = Path::new(path);

        // Load model config
        let config_path = path.join("config.json");
        let config_content = std::fs::read_to_string(&config_path).map_err(|e| {
            SpeculativeEngineError::ModelLoad(format!("Failed to read config.json: {}", e))
        })?;
        let model_config: ModelConfig = serde_json::from_str(&config_content).map_err(|e| {
            SpeculativeEngineError::ModelLoad(format!("Failed to parse config.json: {}", e))
        })?;

        // Load tokenizer
        let tokenizer_path = path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            SpeculativeEngineError::Tokenizer(format!("Failed to load tokenizer: {}", e))
        })?;

        // Get EOS token ID
        let eos_token_id = model_config.eos_token_ids().first().copied().unwrap_or(2);

        // Find weight files
        let weights = Self::find_weight_files(path)?;

        // Load weights into VarBuilder
        let vb = Self::load_weights(&weights, device, dtype)?;

        // Detect architecture and load model
        let arch_type = ArchitectureType::detect(
            model_config.model_type.as_deref(),
            model_config.architectures.as_deref(),
        );

        let draft_model = match arch_type {
            ArchitectureType::Qwen2 => {
                let qwen2_config = Qwen2Config {
                    hidden_size: model_config.hidden_size.unwrap_or(3584),
                    intermediate_size: model_config.intermediate_size.unwrap_or(18944),
                    vocab_size: model_config.vocab_size.unwrap_or(151936),
                    num_hidden_layers: model_config.num_hidden_layers.unwrap_or(28),
                    num_attention_heads: model_config.num_attention_heads.unwrap_or(28),
                    num_key_value_heads: model_config.num_key_value_heads,
                    rms_norm_eps: model_config.rms_norm_eps.unwrap_or(1e-6),
                    rope_theta: model_config.rope_theta.unwrap_or(1000000.0),
                    max_position_embeddings: model_config.max_position_embeddings.unwrap_or(32768),
                    tie_word_embeddings: model_config.tie_word_embeddings.unwrap_or(false),
                    bos_token_id: model_config.bos_token_id,
                    eos_token_id: Some(eos_token_id),
                    use_sliding_window: false,
                    sliding_window: None,
                };

                let model = Qwen2::load(qwen2_config, vb).map_err(|e| {
                    SpeculativeEngineError::ModelLoad(format!("Failed to load Qwen2: {}", e))
                })?;
                DraftModelKind::Qwen2(model)
            },
            ArchitectureType::Llama | ArchitectureType::Unknown => {
                let llama_config = LlamaConfig {
                    hidden_size: model_config.hidden_size.unwrap_or(4096),
                    intermediate_size: model_config.intermediate_size.unwrap_or(11008),
                    vocab_size: model_config.vocab_size.unwrap_or(32000),
                    num_hidden_layers: model_config.num_hidden_layers.unwrap_or(32),
                    num_attention_heads: model_config.num_attention_heads.unwrap_or(32),
                    num_key_value_heads: model_config.num_key_value_heads,
                    rms_norm_eps: model_config.rms_norm_eps.unwrap_or(1e-5),
                    rope_theta: model_config.rope_theta.unwrap_or(10000.0),
                    max_position_embeddings: model_config.max_position_embeddings.unwrap_or(4096),
                    tie_word_embeddings: model_config.tie_word_embeddings.unwrap_or(false),
                    bos_token_id: model_config.bos_token_id,
                    eos_token_id: Some(eos_token_id),
                    rope_scaling: model_config.rope_scaling.clone(),
                };

                let model = Llama::load(llama_config, vb).map_err(|e| {
                    SpeculativeEngineError::ModelLoad(format!("Failed to load Llama: {}", e))
                })?;
                DraftModelKind::Llama(model)
            },
        };

        Ok((
            DraftModelWrapper { kind: draft_model },
            tokenizer,
            eos_token_id,
        ))
    }

    /// Loads the target model (large model, with lazy layer loading).
    fn load_target_model(
        config: &SpeculativeEngineConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<LazyLlama, SpeculativeEngineError> {
        // Parse holo:// URL or use as plain path
        let model_path = if config.target_model_path.starts_with("holo://") {
            // Parse URL: holo:///path/to/model?min=0.7&target=0.95
            let url = url::Url::parse(&config.target_model_path).map_err(|e| {
                SpeculativeEngineError::ModelLoad(format!("Failed to parse holo URL: {}", e))
            })?;
            url.path().to_string()
        } else {
            config.target_model_path.clone()
        };

        let path = Path::new(&model_path);

        // Load model config
        let config_path = path.join("config.json");
        let config_content = std::fs::read_to_string(&config_path).map_err(|e| {
            SpeculativeEngineError::ModelLoad(format!("Failed to read target config.json: {}", e))
        })?;
        let model_config: ModelConfig = serde_json::from_str(&config_content).map_err(|e| {
            SpeculativeEngineError::ModelLoad(format!("Failed to parse target config.json: {}", e))
        })?;

        let eos_token_id = model_config.eos_token_ids().first().copied().unwrap_or(2);

        // Configure tiered loading for large models
        let tiered_config = TieredConfig {
            vram_budget: config.vram_budget,
            ram_budget: config.ram_budget,
            min_quality: config.min_quality,
            target_quality: config.target_quality,
            enable_background_streaming: true,
            background_streams: 4,
        };

        // Create tiered loader (signature: directory, config, device, dtype)
        let mut loader = TieredHoloLoader::new(path, tiered_config, device.clone(), dtype)
            .map_err(|e| {
                SpeculativeEngineError::ModelLoad(format!("Failed to create tiered loader: {}", e))
            })?;

        // Enable NVMe cache if configured
        if let Some(ref cache_dir) = config.cache_dir {
            loader = loader.with_safetensors_dir(PathBuf::from(cache_dir));
            tracing::info!(cache_dir = %cache_dir, "NVMe cache enabled for target model");
        }

        // Create lazy VarBuilder (TieredHoloLoader implements TensorProvider)
        let lazy_vb = LazyVarBuilder::new(Arc::new(loader), device.clone(), dtype);

        // Create Llama config for lazy model
        let llama_config = LlamaConfig {
            hidden_size: model_config.hidden_size.unwrap_or(8192),
            intermediate_size: model_config.intermediate_size.unwrap_or(28672),
            vocab_size: model_config.vocab_size.unwrap_or(128256),
            num_hidden_layers: model_config.num_hidden_layers.unwrap_or(80),
            num_attention_heads: model_config.num_attention_heads.unwrap_or(64),
            num_key_value_heads: model_config.num_key_value_heads,
            rms_norm_eps: model_config.rms_norm_eps.unwrap_or(1e-5),
            rope_theta: model_config.rope_theta.unwrap_or(500000.0),
            max_position_embeddings: model_config.max_position_embeddings.unwrap_or(131072),
            tie_word_embeddings: model_config.tie_word_embeddings.unwrap_or(false),
            bos_token_id: model_config.bos_token_id,
            eos_token_id: Some(eos_token_id),
            rope_scaling: model_config.rope_scaling.clone(),
        };

        // Max layers to keep in memory (limit based on available RAM)
        // For 405B: ~32 layers fit in 64GB RAM, for 70B: all layers can fit
        let max_loaded_layers = config.max_loaded_layers.unwrap_or(32);

        // Load lazy model
        let lazy_llama =
            LazyLlama::load(llama_config, lazy_vb, max_loaded_layers).map_err(|e| {
                SpeculativeEngineError::ModelLoad(format!("Failed to load LazyLlama: {}", e))
            })?;

        Ok(lazy_llama)
    }

    /// Finds weight files in a model directory.
    fn find_weight_files(path: &Path) -> Result<WeightFiles, SpeculativeEngineError> {
        // Check for single safetensors file
        let single_st = path.join("model.safetensors");
        if single_st.exists() {
            return Ok(WeightFiles::SingleSafetensors(single_st));
        }

        // Check for sharded safetensors
        let index_path = path.join("model.safetensors.index.json");
        if index_path.exists() {
            let index_content = std::fs::read_to_string(&index_path)?;
            let index: serde_json::Value = serde_json::from_str(&index_content).map_err(|e| {
                SpeculativeEngineError::ModelLoad(format!("Failed to parse shard index: {}", e))
            })?;

            let weight_map = index.get("weight_map").ok_or_else(|| {
                SpeculativeEngineError::ModelLoad("Missing weight_map in index".to_string())
            })?;

            let mut shard_files: std::collections::HashSet<String> =
                std::collections::HashSet::new();
            if let Some(map) = weight_map.as_object() {
                for file in map.values() {
                    if let Some(f) = file.as_str() {
                        shard_files.insert(f.to_string());
                    }
                }
            }

            let shards: Vec<PathBuf> = shard_files.into_iter().map(|f| path.join(f)).collect();

            return Ok(WeightFiles::ShardedSafetensors {
                index: index_path,
                shards,
            });
        }

        Err(SpeculativeEngineError::ModelLoad(
            "No safetensors weights found".to_string(),
        ))
    }

    /// Loads weights into a VarBuilder.
    fn load_weights(
        weights: &WeightFiles,
        device: &Device,
        dtype: DType,
    ) -> Result<VarBuilder<'static>, SpeculativeEngineError> {
        match weights {
            WeightFiles::SingleSafetensors(path) => {
                let data = std::fs::read(path)?;
                VarBuilder::from_buffered_safetensors(data, dtype, device).map_err(|e| {
                    SpeculativeEngineError::ModelLoad(format!("Failed to load weights: {}", e))
                })
            },
            WeightFiles::ShardedSafetensors { shards, .. } => {
                // SAFETY: Files are read-only and paths are controlled
                unsafe {
                    VarBuilder::from_mmaped_safetensors(shards, dtype, device).map_err(|e| {
                        SpeculativeEngineError::ModelLoad(format!("Failed to mmap shards: {}", e))
                    })
                }
            },
            _ => Err(SpeculativeEngineError::ModelLoad(
                "Unsupported weight format".to_string(),
            )),
        }
    }

    /// Generates tokens using speculative decoding.
    pub fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<String, SpeculativeEngineError> {
        // Tokenize prompt (abaddon Tokenizer returns Vec<u32> directly)
        let prompt_tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| SpeculativeEngineError::Tokenizer(format!("Failed to tokenize: {}", e)))?;

        // Generate tokens
        let generated_tokens = self.generate_tokens(&prompt_tokens, max_tokens)?;

        // Decode tokens
        let text = self
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| SpeculativeEngineError::Tokenizer(format!("Failed to decode: {}", e)))?;

        Ok(text)
    }

    /// Generates tokens using speculative decoding.
    pub fn generate_tokens(
        &self,
        prompt_tokens: &[u32],
        max_tokens: usize,
    ) -> Result<Vec<u32>, SpeculativeEngineError> {
        let speculative = self.inner.lock();

        speculative
            .generate(prompt_tokens, max_tokens, self.eos_token_id)
            .map_err(|e| SpeculativeEngineError::Inference(format!("Generation failed: {}", e)))
    }

    /// Returns current statistics.
    pub fn stats(&self) -> Speculative405BStats {
        self.inner.lock().stats()
    }

    /// Resets statistics.
    pub fn reset_stats(&self) {
        self.inner.lock().reset_stats();
    }

    /// Returns the model identifier.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Returns the configuration.
    pub fn config(&self) -> &SpeculativeEngineConfig {
        &self.config
    }

    /// Returns the tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Returns the EOS token ID.
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }
}

/// Builder for creating a SpeculativeEngine.
pub struct SpeculativeEngineBuilder {
    config: SpeculativeEngineConfig,
}

impl SpeculativeEngineBuilder {
    /// Creates a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: SpeculativeEngineConfig::default(),
        }
    }

    /// Sets the draft model path.
    pub fn draft_model(mut self, path: impl Into<String>) -> Self {
        self.config.draft_model_path = path.into();
        self
    }

    /// Sets the target model path.
    pub fn target_model(mut self, path: impl Into<String>) -> Self {
        self.config.target_model_path = path.into();
        self
    }

    /// Sets the number of draft tokens per round.
    pub fn num_draft_tokens(mut self, n: usize) -> Self {
        self.config.num_draft_tokens = n.clamp(1, 16);
        self
    }

    /// Sets the acceptance threshold.
    pub fn acceptance_threshold(mut self, threshold: f32) -> Self {
        self.config.acceptance_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Sets the VRAM budget.
    pub fn vram_budget(mut self, bytes: u64) -> Self {
        self.config.vram_budget = bytes;
        self
    }

    /// Sets the RAM budget.
    pub fn ram_budget(mut self, bytes: u64) -> Self {
        self.config.ram_budget = bytes;
        self
    }

    /// Sets the cache directory.
    pub fn cache_dir(mut self, dir: impl Into<String>) -> Self {
        self.config.cache_dir = Some(dir.into());
        self
    }

    /// Sets the CUDA device ID.
    pub fn device_id(mut self, id: usize) -> Self {
        self.config.device_id = id;
        self
    }

    /// Sets whether to use greedy decoding for drafts.
    pub fn greedy_draft(mut self, greedy: bool) -> Self {
        self.config.greedy_draft = greedy;
        self
    }

    /// Sets the min quality for HoloTensor loading.
    pub fn min_quality(mut self, quality: f32) -> Self {
        self.config.min_quality = quality.clamp(0.1, 1.0);
        self
    }

    /// Sets the target quality for HoloTensor loading.
    pub fn target_quality(mut self, quality: f32) -> Self {
        self.config.target_quality = quality.clamp(0.1, 1.0);
        self
    }

    /// Sets the maximum number of decoder layers to keep loaded.
    /// For 24GB VRAM with layer streaming, 7 layers is typical for 70B models.
    pub fn max_loaded_layers(mut self, layers: usize) -> Self {
        self.config.max_loaded_layers = Some(layers);
        self
    }

    /// Builds the speculative engine.
    pub async fn build(self) -> Result<SpeculativeEngine, SpeculativeEngineError> {
        // Validate configuration
        if self.config.draft_model_path.is_empty() {
            return Err(SpeculativeEngineError::Config(
                "Draft model path is required".to_string(),
            ));
        }
        if self.config.target_model_path.is_empty() {
            return Err(SpeculativeEngineError::Config(
                "Target model path is required".to_string(),
            ));
        }

        SpeculativeEngine::new(self.config).await
    }
}

impl Default for SpeculativeEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = SpeculativeEngineConfig::default();
        assert_eq!(config.num_draft_tokens, 5);
        assert!((config.acceptance_threshold - 0.1).abs() < f32::EPSILON);
        assert_eq!(config.vram_budget, 8 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_builder_validation() {
        let builder = SpeculativeEngineBuilder::new();
        // Can't test async build without models, but we can verify config
        assert!(builder.config.draft_model_path.is_empty());
        assert!(builder.config.target_model_path.is_empty());
    }

    #[test]
    fn test_builder_chaining() {
        let builder = SpeculativeEngineBuilder::new()
            .draft_model("/models/draft")
            .target_model("/models/target")
            .num_draft_tokens(8)
            .acceptance_threshold(0.2)
            .vram_budget(16 * 1024 * 1024 * 1024)
            .cache_dir("/cache");

        assert_eq!(builder.config.draft_model_path, "/models/draft");
        assert_eq!(builder.config.target_model_path, "/models/target");
        assert_eq!(builder.config.num_draft_tokens, 8);
        assert!((builder.config.acceptance_threshold - 0.2).abs() < f32::EPSILON);
        assert_eq!(builder.config.vram_budget, 16 * 1024 * 1024 * 1024);
        assert_eq!(builder.config.cache_dir, Some("/cache".to_string()));
    }
}
