//! Core inference engine implementation.

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use infernum_core::{
    EmbedRequest, EmbedResponse, GenerateRequest, GenerateResponse, Message, ModelMetadata, Result,
    Role, SamplingParams, TokenStream,
};
use parking_lot::Mutex;
use tokio::sync::mpsc;

#[cfg(feature = "cuda")]
use crate::backend::to_candle_dtype;
use crate::config::EngineConfig;
#[cfg(feature = "cuda")]
use crate::cuda_inference::{Generator as CudaGenerator, WeightStore as CudaWeightStore};
use crate::loader::{ModelConfig, ModelFiles, ModelLoader, WeightFiles};
use crate::models::llama::{Llama, LlamaConfig};
use crate::models::qwen2::{Qwen2, Qwen2Config};
use crate::models::{ArchitectureType, ModelKind};
use crate::sampler::Sampler;
use crate::speculative::SpeculativeDecoder;
use crate::tokenizer::Tokenizer;

/// Trait defining the inference engine interface.
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    /// Generates text from the given request.
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse>;

    /// Generates text for a batch of requests.
    ///
    /// The default implementation processes requests concurrently using async parallelism.
    /// Implementations may override this to provide true batched GPU inference.
    async fn generate_batch(
        &self,
        requests: Vec<GenerateRequest>,
    ) -> Vec<Result<GenerateResponse>> {
        use futures::future::join_all;

        join_all(requests.into_iter().map(|req| self.generate(req))).await
    }

    /// Generates text with streaming output.
    async fn generate_stream(&self, request: GenerateRequest) -> Result<TokenStream>;

    /// Generates embeddings from the given request.
    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse>;

    /// Returns metadata about the loaded model.
    fn model_info(&self) -> &ModelMetadata;

    /// Returns true if the engine is ready for inference.
    fn is_ready(&self) -> bool;
}

/// Loaded model state (wrapped in Arc for streaming support).
struct LoadedModel {
    model: Mutex<ModelKind>,
    tokenizer: Tokenizer,
    /// EOS token ID for generation termination.
    eos_token_id: u32,
}

/// The main inference engine.
pub struct Engine {
    config: EngineConfig,
    metadata: ModelMetadata,
    loaded: Option<Arc<LoadedModel>>,
    device: Device,
    #[allow(dead_code)] // Used for model quantization in future
    dtype: DType,
    /// Optional speculative decoder for accelerated inference.
    speculative_decoder: Option<Arc<SpeculativeDecoder>>,
    /// Optional CUDA-optimized generator for HoloTensor models.
    /// When available, bypasses Candle for 5-6x faster inference.
    #[cfg(feature = "cuda")]
    cuda_generator: Option<Mutex<CudaGenerator>>,
}

impl Engine {
    /// Creates a new engine with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded.
    pub async fn new(config: EngineConfig) -> Result<Self> {
        tracing::info!("Initializing Abaddon inference engine");
        tracing::debug!(?config, "Engine configuration");

        // Determine device
        let (device, device_info) = Self::select_device(&config)?;
        tracing::info!(device = ?device, "Using compute device");

        // Print device info prominently for user visibility
        eprintln!("\x1b[1mCompute Backend:\x1b[0m {}", device_info);

        // Determine dtype based on device capabilities
        let dtype = Self::select_dtype(&device, &config)?;

        let dtype_name = match dtype {
            DType::F32 => "F32",
            DType::F16 => "F16 (tensor cores)",
            DType::BF16 => "BF16 (tensor cores)",
            _ => "other",
        };
        tracing::info!(dtype = dtype_name, "Selected inference precision");

        // Load model files
        let loader = ModelLoader::default_cache()?;
        let files = loader.resolve(&config.model)?;

        // Load model config
        let model_config = ModelConfig::from_file(&files.config)?;
        tracing::debug!(?model_config, "Loaded model configuration");

        // Build metadata
        let metadata = Self::build_metadata(&config, &model_config)?;

        // Load the model
        let loaded = Self::load_model(&files, &model_config, &device, dtype)?;

        // Load speculative decoder if configured
        let speculative_decoder = if let Some(spec_config) = &config.speculative {
            tracing::info!("Loading draft model for speculative decoding");
            match Self::load_speculative_decoder(spec_config, &loader, &device, dtype) {
                Ok(decoder) => {
                    tracing::info!(
                        num_speculative_tokens = spec_config.num_speculative_tokens,
                        acceptance_threshold = spec_config.acceptance_threshold,
                        "Speculative decoding enabled"
                    );
                    Some(Arc::new(decoder))
                },
                Err(e) => {
                    tracing::warn!(
                        "Failed to load draft model, disabling speculative decoding: {}",
                        e
                    );
                    None
                },
            }
        } else {
            None
        };

        // Try to initialize CUDA optimized path for HoloTensor models
        #[cfg(feature = "cuda")]
        let cuda_generator = if let WeightFiles::HoloTensor { directory, .. } = &files.weights {
            if matches!(device, Device::Cuda(_)) {
                tracing::info!(
                    "Detected HoloTensor model on CUDA, attempting optimized inference path"
                );
                let max_seq_len = model_config.max_position_embeddings.unwrap_or(4096);
                match Self::init_cuda_generator(directory, max_seq_len) {
                    Some(gen) => {
                        eprintln!(
                            "\x1b[32m✓ CUDA optimized inference enabled (5-6x faster)\x1b[0m"
                        );
                        Some(Mutex::new(gen))
                    },
                    None => {
                        tracing::info!("Falling back to Candle inference path");
                        None
                    },
                }
            } else {
                None
            }
        } else {
            None
        };

        #[cfg(feature = "cuda")]
        let cuda_optimized = cuda_generator.is_some();
        #[cfg(not(feature = "cuda"))]
        let cuda_optimized = false;

        tracing::info!(
            model = %metadata.id,
            layers = model_config.num_hidden_layers,
            speculative = speculative_decoder.is_some(),
            cuda_optimized = cuda_optimized,
            "Engine initialized successfully"
        );

        Ok(Self {
            config,
            metadata,
            loaded: Some(Arc::new(loaded)),
            device,
            dtype,
            speculative_decoder,
            #[cfg(feature = "cuda")]
            cuda_generator,
        })
    }

    /// Tries to initialize the optimized CUDA inference path for HCT models.
    ///
    /// This bypasses Candle's tensor operations for 5-6x faster inference using
    /// fused CUDA kernels, pre-allocated buffers, and Flash Attention.
    ///
    /// Returns the CUDA generator if initialization succeeded.
    #[cfg(feature = "cuda")]
    fn init_cuda_generator(
        hct_directory: &std::path::Path,
        max_seq_len: usize,
    ) -> Option<CudaGenerator> {
        tracing::info!(
            directory = %hct_directory.display(),
            "Attempting to initialize optimized CUDA inference path"
        );

        // Try to load weights using cuda_inference
        let weights = match CudaWeightStore::load_hct(hct_directory, None, 0) {
            Ok(w) => {
                tracing::info!(
                    memory_mb = w.memory_used as f64 / 1024.0 / 1024.0,
                    num_layers = w.layers.len(),
                    "CUDA weights loaded successfully"
                );
                w
            },
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "Failed to load CUDA weights, falling back to Candle path"
                );
                return None;
            },
        };

        // Create the optimized generator
        match CudaGenerator::new(weights, max_seq_len) {
            Ok(generator) => {
                tracing::info!("CUDA generator initialized successfully");
                Some(generator)
            },
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "Failed to create CUDA generator, falling back to Candle path"
                );
                None
            },
        }
    }

    /// Loads the speculative decoder with the draft model.
    fn load_speculative_decoder(
        spec_config: &crate::config::SpeculativeConfig,
        loader: &ModelLoader,
        device: &Device,
        dtype: DType,
    ) -> Result<SpeculativeDecoder> {
        // Load draft model files
        let draft_files = loader.resolve(&spec_config.draft_model)?;
        let draft_model_config = ModelConfig::from_file(&draft_files.config)?;

        // Load draft model
        let draft_loaded = Self::load_model(&draft_files, &draft_model_config, device, dtype)?;

        Ok(SpeculativeDecoder::new(
            draft_loaded.model.into_inner(),
            draft_loaded.tokenizer,
            spec_config.clone(),
            device.clone(),
            dtype,
        ))
    }

    /// Selects the compute device based on configuration and availability.
    /// Returns both the device and a human-readable description.
    fn select_device(config: &EngineConfig) -> Result<(Device, String)> {
        use infernum_core::DeviceType;

        match &config.device {
            DeviceType::Cpu => Ok((Device::Cpu, "CPU".to_string())),
            DeviceType::Cuda { device_id } => {
                #[cfg(feature = "cuda")]
                {
                    let device = Device::new_cuda(*device_id).map_err(|e| {
                        infernum_core::Error::Backend {
                            backend: "cuda".to_string(),
                            message: e.to_string(),
                        }
                    })?;
                    Ok((device, format!("CUDA (GPU {})", device_id)))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = device_id; // Silence unused warning when cuda feature disabled
                    eprintln!("\x1b[33mWarning:\x1b[0m CUDA requested but not compiled in, falling back to CPU");
                    eprintln!("         Rebuild with: cargo build --features cuda");
                    tracing::warn!("CUDA requested but not compiled in, falling back to CPU");
                    Ok((Device::Cpu, "CPU (CUDA unavailable)".to_string()))
                }
            },
            DeviceType::Metal { device_id } => {
                #[cfg(feature = "metal")]
                {
                    let device = Device::new_metal(*device_id).map_err(|e| {
                        infernum_core::Error::Backend {
                            backend: "metal".to_string(),
                            message: e.to_string(),
                        }
                    })?;
                    Ok((device, "Metal (Apple GPU)".to_string()))
                }
                #[cfg(not(feature = "metal"))]
                {
                    let _ = device_id; // Silence unused warning when metal feature disabled
                    eprintln!("\x1b[33mWarning:\x1b[0m Metal requested but not compiled in, falling back to CPU");
                    eprintln!("         Rebuild with: cargo build --features metal");
                    tracing::warn!("Metal requested but not compiled in, falling back to CPU");
                    Ok((Device::Cpu, "CPU (Metal unavailable)".to_string()))
                }
            },
            DeviceType::WebGpu => {
                eprintln!("\x1b[33mWarning:\x1b[0m WebGPU not yet supported, falling back to CPU");
                tracing::warn!("WebGPU not yet supported, falling back to CPU");
                Ok((Device::Cpu, "CPU (WebGPU not yet supported)".to_string()))
            },
        }
    }

    /// Selects the optimal dtype based on device capabilities.
    ///
    /// For CUDA devices with tensor cores (compute >= 7.0), uses FP16 or BF16.
    /// For older CUDA devices, falls back to F32.
    /// Metal and CPU use F16 by default.
    fn select_dtype(device: &Device, config: &EngineConfig) -> Result<DType> {
        // If user explicitly requested a quantization, that takes precedence
        if config.quantization.is_some() {
            // Quantized models typically use F16 for activation
            return Ok(DType::F16);
        }

        match device {
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    use crate::backend::cuda::CudaDevice;

                    // Query GPU capabilities
                    if let infernum_core::DeviceType::Cuda { device_id } = &config.device {
                        if let Ok(cuda_dev) = CudaDevice::new(*device_id) {
                            let caps = cuda_dev.capabilities();
                            let dtype = caps.recommended_dtype();

                            eprintln!(
                                "\x1b[1mGPU:\x1b[0m Compute {}.{} | {} GB VRAM | Tensor Cores: {} | BF16: {}",
                                caps.compute_major,
                                caps.compute_minor,
                                caps.total_memory / (1024 * 1024 * 1024),
                                if caps.has_tensor_cores { "Yes" } else { "No" },
                                if caps.has_bf16 { "Yes" } else { "No" }
                            );

                            return Ok(to_candle_dtype(dtype));
                        }
                    }
                    // Fallback for capability query failure - assume modern GPU
                    Ok(DType::F16)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Ok(DType::F32)
                }
            },
            Device::Metal(_) => {
                // Apple Silicon GPUs handle F16 well
                Ok(DType::F16)
            },
            Device::Cpu => {
                // CPU can use F16 with modern SIMD (AVX-512, NEON)
                // but F32 is safer and often faster
                #[cfg(any(feature = "mkl", feature = "accelerate"))]
                {
                    Ok(DType::F32) // MKL/Accelerate optimized for F32
                }
                #[cfg(not(any(feature = "mkl", feature = "accelerate")))]
                {
                    Ok(DType::F16) // Native Candle can use F16
                }
            },
        }
    }

    /// Loads the model from files.
    fn load_model(
        files: &ModelFiles,
        model_config: &ModelConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<LoadedModel> {
        tracing::info!("Loading model weights...");
        let start = Instant::now();

        // Detect architecture from config
        let arch_type = ArchitectureType::detect(
            model_config.model_type.as_deref(),
            model_config.architectures.as_deref(),
        );

        tracing::info!(
            architecture = arch_type.name(),
            "Detected model architecture"
        );

        // Check if this is a HoloTensor progressive load (for 405B+ models)
        if let WeightFiles::HoloTensor {
            directory,
            min_quality,
            target_quality,
            vram_budget,
            ram_budget,
        } = &files.weights
        {
            return Self::load_model_lazy(
                files,
                model_config,
                device,
                dtype,
                directory,
                *min_quality,
                *target_quality,
                *vram_budget,
                *ram_budget,
                arch_type,
            );
        }

        // Standard loading path - load all weights into VarBuilder
        let vb = Self::load_weights(&files.weights, device, dtype)?;

        // Get EOS token ID
        let eos_token_id = model_config.eos_token_ids().first().copied().unwrap_or(2);

        // Load the appropriate model based on architecture
        let model = match arch_type {
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

                let qwen2 =
                    Qwen2::load(qwen2_config, vb).map_err(|e| infernum_core::Error::ModelLoad {
                        message: format!("Failed to load Qwen2 model: {}", e),
                    })?;
                ModelKind::Qwen2(qwen2)
            },
            ArchitectureType::Llama | ArchitectureType::Unknown => {
                // Default to Llama for unknown architectures (most compatible)
                if arch_type == ArchitectureType::Unknown {
                    tracing::warn!(
                        "Unknown architecture, defaulting to Llama. \
                         Model may not work correctly."
                    );
                }

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

                let llama =
                    Llama::load(llama_config, vb).map_err(|e| infernum_core::Error::ModelLoad {
                        message: format!("Failed to load Llama model: {}", e),
                    })?;
                ModelKind::Llama(llama)
            },
        };

        // Load tokenizer
        let tokenizer = if let Some(tokenizer_path) = &files.tokenizer {
            Tokenizer::from_file(tokenizer_path)?
        } else {
            return Err(infernum_core::Error::ModelLoad {
                message: "No tokenizer found for model".to_string(),
            });
        };

        let elapsed = start.elapsed();
        tracing::info!(
            elapsed_ms = elapsed.as_millis(),
            "Model loaded successfully"
        );

        Ok(LoadedModel {
            model: Mutex::new(model),
            tokenizer,
            eos_token_id,
        })
    }

    /// Loads a model using lazy layer-by-layer loading for 405B+ models.
    ///
    /// This uses `LazyLlama` or `LazyQwen2` (based on architecture) which loads
    /// decoder layers on-demand during inference, enabling 405B inference on
    /// systems with limited memory (24GB VRAM + 80GB RAM).
    #[allow(clippy::too_many_arguments)]
    fn load_model_lazy(
        files: &ModelFiles,
        model_config: &ModelConfig,
        device: &Device,
        dtype: DType,
        directory: &std::path::Path,
        min_quality: f32,
        target_quality: f32,
        vram_budget: u64,
        ram_budget: u64,
        arch_type: ArchitectureType,
    ) -> Result<LoadedModel> {
        use crate::holotensor::tiered_loading::{TieredConfig, TieredHoloLoader};
        use crate::lazy_varbuilder::LazyVarBuilder;
        use crate::models::{LazyLlama, LazyQwen2, LlamaConfig, Qwen2Config};

        // Check if adaptive tiering is enabled (experimental)
        let use_adaptive = std::env::var("INFERNUM_ADAPTIVE_TIERING")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        if use_adaptive {
            return Self::load_lazy_model_adaptive(
                files,
                model_config,
                device,
                dtype,
                directory,
                vram_budget,
                ram_budget,
                arch_type,
            );
        }

        tracing::info!(
            directory = %directory.display(),
            min_quality = %min_quality,
            target_quality = %target_quality,
            vram_gb = vram_budget / (1024 * 1024 * 1024),
            ram_gb = ram_budget / (1024 * 1024 * 1024),
            "Loading model with lazy layer loading (405B mode)"
        );

        let start = Instant::now();

        // Configure tiered loading
        let tiered_config = TieredConfig {
            vram_budget,
            ram_budget,
            min_quality,
            target_quality,
            enable_background_streaming: true,
            background_streams: 4,
        };

        // Create the tiered loader
        let mut loader = TieredHoloLoader::new(directory, tiered_config, device.clone(), dtype)
            .map_err(|e| infernum_core::Error::ModelLoad {
                message: format!("Failed to create tiered loader: {}", e),
            })?;

        // Enable NVMe cache for decompressed tensors if configured
        // This provides ~1000x speedup on subsequent layer loads (100ms vs 100s)
        if let Ok(cache_dir) = std::env::var("INFERNUM_CACHE_DIR") {
            let cache_path = std::path::Path::new(&cache_dir);
            if cache_path.exists() || std::fs::create_dir_all(cache_path).is_ok() {
                tracing::info!(
                    cache_dir = %cache_dir,
                    "NVMe cache enabled - subsequent layer loads will be ~1000x faster"
                );
                loader = loader.with_safetensors_dir(cache_path);
            } else {
                tracing::warn!(
                    cache_dir = %cache_dir,
                    "Could not create NVMe cache directory, falling back to HCT reconstruction"
                );
            }
        }

        // Start background streaming for quality improvement
        loader.start_background_streaming();

        // Create lazy VarBuilder backed by tiered loader with VRAM-appropriate cache
        let provider: std::sync::Arc<dyn crate::lazy_varbuilder::TensorProvider> =
            std::sync::Arc::new(loader);

        // CRITICAL: Disable LazyVarBuilder cache entirely!
        // TieredHoloLoader already caches tensors in its HashMap.
        // If LazyVarBuilder also caches, we get TRIPLE copies of each tensor:
        //   1. TieredHoloLoader's HashMap (original)
        //   2. LazyVarBuilder's LRU cache (clone)
        //   3. LazyLlama's loaded_layers (clone from LazyVarBuilder)
        // This causes OOM: 7 layers × 2GB × 3 copies = 42GB on 24GB VRAM!
        // Setting max_memory_bytes=0 disables the cache.
        let cache_config = crate::lazy_varbuilder::CacheConfig {
            max_memory_bytes: 0, // Disable cache - TieredHoloLoader already caches
            max_entries: 0,
        };
        tracing::info!("LazyVarBuilder cache disabled - TieredHoloLoader provides caching");
        let lazy_vb = LazyVarBuilder::with_cache_config(
            std::sync::Arc::clone(&provider),
            device.clone(),
            dtype,
            cache_config,
        );

        // Calculate max layers to keep in VRAM based on model dimensions
        // Each layer has: attention (q,k,v,o projections) + MLP (gate, up, down)
        let hidden = model_config.hidden_size.unwrap_or(8192) as u64;
        let intermediate = model_config.intermediate_size.unwrap_or(28672) as u64;
        let dtype_bytes: u64 = match dtype {
            DType::BF16 | DType::F16 => 2,
            DType::F32 => 4,
            _ => 2, // Default to 2 for other types
        };

        // Layer size = attention weights + MLP weights (approximate, ignores GQA savings)
        // Attention: 4 * hidden^2 (q,k,v,o projections)
        // MLP: 3 * hidden * intermediate (gate, up, down projections)
        let layer_size_bytes = (4 * hidden * hidden + 3 * hidden * intermediate) * dtype_bytes;
        let layer_size_gb = (layer_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)).ceil() as u64;

        // Use VRAM budget (not RAM!) with headroom for fixed allocations
        let vram_budget_gb = vram_budget / (1024 * 1024 * 1024);

        // Calculate headroom for non-layer tensors:
        // - Embeddings: vocab_size * hidden * dtype_bytes (~2GB for 70B)
        // - lm_head: same as embeddings (~2GB for 70B, unless tied)
        // - KV cache: grows with sequence length (~1-4GB typically)
        // - CUDA overhead, activations, gradients (~2-4GB)
        let vocab = model_config.vocab_size.unwrap_or(128256) as u64;
        let embedding_size_gb = (vocab * hidden * dtype_bytes) as f64 / (1024.0 * 1024.0 * 1024.0);
        let lm_head_size_gb = if model_config.tie_word_embeddings.unwrap_or(false) {
            0.0 // Tied embeddings share memory
        } else {
            embedding_size_gb
        };
        let fixed_overhead_gb = 4u64; // KV cache, activations, CUDA overhead
        let headroom_gb = (embedding_size_gb + lm_head_size_gb).ceil() as u64 + fixed_overhead_gb;

        let available_vram_gb = vram_budget_gb.saturating_sub(headroom_gb);

        // Calculate max layers, minimum 1 for basic operation (layer-by-layer streaming)
        let max_loaded_layers = if layer_size_gb > 0 && available_vram_gb > 0 {
            (available_vram_gb / layer_size_gb).max(1) as usize
        } else {
            // Not enough VRAM for even 1 layer - use streaming mode with 1 layer
            tracing::warn!(
                vram_budget_gb = vram_budget_gb,
                headroom_gb = headroom_gb,
                layer_size_gb = layer_size_gb,
                "Insufficient VRAM for layer caching, using single-layer streaming mode"
            );
            1
        };

        tracing::info!(
            hidden_size = hidden,
            intermediate_size = intermediate,
            layer_size_gb = layer_size_gb,
            vram_budget_gb = vram_budget_gb,
            headroom_gb = headroom_gb,
            available_vram_gb = available_vram_gb,
            max_loaded_layers = max_loaded_layers,
            "Calculated layer budget from VRAM"
        );

        // Get EOS token ID for config
        let eos_token_id_config = model_config.eos_token_ids().first().copied();

        // Load the appropriate lazy model based on architecture
        let model = match arch_type {
            ArchitectureType::Qwen2 => {
                tracing::info!("Loading LazyQwen2 model (Qwen2 architecture detected)");

                // Build Qwen2 config
                let qwen2_config = Qwen2Config {
                    hidden_size: model_config.hidden_size.unwrap_or(5120), // 14B default
                    intermediate_size: model_config.intermediate_size.unwrap_or(13824), // 14B default
                    vocab_size: model_config.vocab_size.unwrap_or(152064),
                    num_hidden_layers: model_config.num_hidden_layers.unwrap_or(48), // 14B has 48 layers
                    num_attention_heads: model_config.num_attention_heads.unwrap_or(40),
                    num_key_value_heads: model_config.num_key_value_heads,
                    rms_norm_eps: model_config.rms_norm_eps.unwrap_or(1e-6), // Qwen2 uses 1e-6
                    rope_theta: model_config.rope_theta.unwrap_or(1000000.0), // Qwen2 uses 1M
                    max_position_embeddings: model_config.max_position_embeddings.unwrap_or(32768),
                    tie_word_embeddings: model_config.tie_word_embeddings.unwrap_or(false),
                    bos_token_id: model_config.bos_token_id,
                    eos_token_id: eos_token_id_config,
                    use_sliding_window: false,
                    sliding_window: None,
                };

                let lazy_qwen2 = LazyQwen2::load(qwen2_config, lazy_vb, max_loaded_layers)
                    .map_err(|e| infernum_core::Error::ModelLoad {
                        message: format!("Failed to load LazyQwen2: {}", e),
                    })?;

                ModelKind::LazyQwen2(lazy_qwen2)
            },
            ArchitectureType::Llama | ArchitectureType::Unknown => {
                if arch_type == ArchitectureType::Unknown {
                    tracing::warn!(
                        "Unknown architecture, defaulting to LazyLlama. Model may not work correctly."
                    );
                } else {
                    tracing::info!("Loading LazyLlama model (Llama architecture detected)");
                }

                // Build Llama config
                let llama_config = LlamaConfig {
                    hidden_size: model_config.hidden_size.unwrap_or(16384), // 405B default
                    intermediate_size: model_config.intermediate_size.unwrap_or(53248), // 405B default
                    vocab_size: model_config.vocab_size.unwrap_or(128256),
                    num_hidden_layers: model_config.num_hidden_layers.unwrap_or(126), // 405B has 126 layers
                    num_attention_heads: model_config.num_attention_heads.unwrap_or(128),
                    num_key_value_heads: model_config.num_key_value_heads,
                    rms_norm_eps: model_config.rms_norm_eps.unwrap_or(1e-5),
                    rope_theta: model_config.rope_theta.unwrap_or(500000.0),
                    max_position_embeddings: model_config.max_position_embeddings.unwrap_or(131072),
                    tie_word_embeddings: model_config.tie_word_embeddings.unwrap_or(false),
                    bos_token_id: model_config.bos_token_id,
                    eos_token_id: eos_token_id_config,
                    rope_scaling: model_config.rope_scaling.clone(),
                };

                let lazy_llama = LazyLlama::load(llama_config, lazy_vb, max_loaded_layers)
                    .map_err(|e| infernum_core::Error::ModelLoad {
                        message: format!("Failed to load LazyLlama: {}", e),
                    })?;

                ModelKind::LazyLlama(lazy_llama)
            },
        };

        // Load tokenizer
        let tokenizer = if let Some(tokenizer_path) = &files.tokenizer {
            Tokenizer::from_file(tokenizer_path)?
        } else {
            return Err(infernum_core::Error::ModelLoad {
                message: "No tokenizer found for model".to_string(),
            });
        };

        // Get EOS token with architecture-appropriate default
        let eos_token_id =
            model_config
                .eos_token_ids()
                .first()
                .copied()
                .unwrap_or(match arch_type {
                    ArchitectureType::Qwen2 => 151643, // Qwen2 EOS token
                    _ => 128001,                       // Llama 3 EOS token
                });

        let elapsed = start.elapsed();
        tracing::info!(
            elapsed_ms = elapsed.as_millis(),
            architecture = arch_type.name(),
            "Lazy model base loaded (layers will load on-demand during inference)"
        );

        Ok(LoadedModel {
            model: Mutex::new(model),
            tokenizer,
            eos_token_id,
        })
    }

    /// Loads a model using adaptive memory tiering (experimental).
    ///
    /// Uses intelligent tensor placement based on importance scoring and
    /// mixed precision to minimize or eliminate layer swapping.
    #[allow(clippy::too_many_arguments)]
    fn load_lazy_model_adaptive(
        files: &ModelFiles,
        model_config: &ModelConfig,
        device: &Device,
        dtype: DType,
        directory: &std::path::Path,
        vram_budget: u64,
        ram_budget: u64,
        arch_type: ArchitectureType,
    ) -> Result<LoadedModel> {
        use crate::adaptive_tiering::{
            AdaptiveLoader, AdaptiveTieringConfig, AllocationPlanner, LoadingBackend, ModelProfile,
        };
        use crate::holotensor::tiered_loading::{TieredConfig, TieredHoloLoader};
        use crate::lazy_varbuilder::LazyVarBuilder;
        use crate::models::{LazyLlama, LazyQwen2, LlamaConfig, Qwen2Config};

        let start = Instant::now();

        tracing::info!(
            directory = %directory.display(),
            vram_gb = vram_budget / (1024 * 1024 * 1024),
            ram_gb = ram_budget / (1024 * 1024 * 1024),
            "Loading model with adaptive memory tiering"
        );

        // Build model profile from HCT directory
        let profile = ModelProfile::from_hct_directory(directory).map_err(|e| {
            infernum_core::Error::ModelLoad {
                message: format!("Failed to build model profile: {}", e),
            }
        })?;

        // Log reconstruction path distribution
        let reconstruction_summary = profile.reconstruction_summary();
        tracing::info!(
            model_size_gb = format!("{:.2}", profile.size_gb()),
            num_layers = profile.num_layers,
            tensor_count = profile.tensors.len(),
            gpu_path_count = reconstruction_summary.gpu_fast_count,
            cpu_path_count = reconstruction_summary.cpu_direct_count,
            "Model profile built"
        );

        // Configure and run allocation planner
        let adaptive_config = AdaptiveTieringConfig {
            vram_budget,
            ram_budget,
            quality_target: 0.95,
            enable_mixed_precision: true,
            ..AdaptiveTieringConfig::default()
        };

        let planner = AllocationPlanner::new(adaptive_config);
        let plan = planner
            .plan(&profile)
            .map_err(|e| infernum_core::Error::ModelLoad {
                message: format!("Failed to plan allocation: {}", e),
            })?;

        // Select loading backend based on allocation plan
        let backend = LoadingBackend::select(&plan);
        tracing::info!(
            vram_usage_gb = format!("{:.2}", plan.vram_usage as f64 / 1e9),
            ram_usage_gb = format!("{:.2}", plan.ram_usage as f64 / 1e9),
            nvme_usage_gb = format!("{:.2}", plan.nvme_usage as f64 / 1e9),
            swap_count = plan.swap_count,
            quality_score = format!("{:.3}", plan.quality_score),
            backend = ?backend,
            "Adaptive allocation plan computed"
        );

        // Route to appropriate loading backend
        match backend {
            LoadingBackend::Eager | LoadingBackend::EagerWithRamCache => {
                return Self::load_model_eager_with_placement(
                    files,
                    model_config,
                    device,
                    dtype,
                    directory,
                    &profile,
                    &plan,
                    arch_type,
                    start,
                );
            },
            LoadingBackend::Progressive => {
                // Continue with TieredHoloLoader for 405B+ models
                tracing::info!(
                    "Using progressive loading (405B mode) - model requires layer swapping"
                );
            },
        }

        // Create underlying tiered loader (only for Progressive backend)
        let tiered_config = TieredConfig {
            vram_budget,
            ram_budget,
            min_quality: 0.7,
            target_quality: 0.95,
            enable_background_streaming: true,
            background_streams: 4,
        };

        let mut tiered_loader =
            TieredHoloLoader::new(directory, tiered_config, device.clone(), dtype).map_err(
                |e| infernum_core::Error::ModelLoad {
                    message: format!("Failed to create tiered loader: {}", e),
                },
            )?;

        // Enable NVMe cache if configured
        if let Ok(cache_dir) = std::env::var("INFERNUM_CACHE_DIR") {
            let cache_path = std::path::Path::new(&cache_dir);
            if cache_path.exists() || std::fs::create_dir_all(cache_path).is_ok() {
                tracing::info!(cache_dir = %cache_dir, "NVMe cache enabled");
                tiered_loader = tiered_loader.with_safetensors_dir(cache_path);
            }
        }

        tiered_loader.start_background_streaming();

        // Wrap with adaptive loader for intelligent caching
        let adaptive_loader = AdaptiveLoader::new(plan.clone(), tiered_loader, device.clone());

        // Preload VRAM tensors for faster first inference
        tracing::info!("Preloading VRAM tensors...");
        if let Err(e) = adaptive_loader.preload_vram_tensors() {
            tracing::warn!(error = %e, "Failed to preload VRAM tensors, continuing with lazy loading");
        }

        let provider: std::sync::Arc<dyn crate::lazy_varbuilder::TensorProvider> =
            std::sync::Arc::new(adaptive_loader);

        // Disable LazyVarBuilder cache - AdaptiveLoader handles caching
        let cache_config = crate::lazy_varbuilder::CacheConfig {
            max_memory_bytes: 0,
            max_entries: 0,
        };

        let lazy_vb = LazyVarBuilder::with_cache_config(
            std::sync::Arc::clone(&provider),
            device.clone(),
            dtype,
            cache_config,
        );

        // With adaptive tiering, we can fit more layers in VRAM
        let max_loaded_layers = if plan.swap_count == 0 {
            profile.num_layers // All layers fit in VRAM
        } else {
            // Use existing calculation but take advantage of better packing
            let layer_count = profile.num_layers.max(1);
            let per_layer_vram = plan.vram_usage / layer_count as u64;
            (vram_budget / per_layer_vram.max(1)).min(layer_count as u64) as usize
        };

        tracing::info!(
            max_loaded_layers = max_loaded_layers,
            "Adaptive loading: max layers in VRAM"
        );

        // Get EOS token ID for config
        let eos_token_id_config = model_config.eos_token_ids().first().copied();

        // Load the appropriate lazy model based on architecture
        let model = match arch_type {
            ArchitectureType::Qwen2 => {
                tracing::info!("Loading LazyQwen2 model (Qwen2 architecture detected)");

                let qwen2_config = Qwen2Config {
                    hidden_size: model_config.hidden_size.unwrap_or(5120),
                    intermediate_size: model_config.intermediate_size.unwrap_or(13824),
                    vocab_size: model_config.vocab_size.unwrap_or(152064),
                    num_hidden_layers: model_config.num_hidden_layers.unwrap_or(48),
                    num_attention_heads: model_config.num_attention_heads.unwrap_or(40),
                    num_key_value_heads: model_config.num_key_value_heads,
                    rms_norm_eps: model_config.rms_norm_eps.unwrap_or(1e-6),
                    rope_theta: model_config.rope_theta.unwrap_or(1000000.0),
                    max_position_embeddings: model_config.max_position_embeddings.unwrap_or(32768),
                    tie_word_embeddings: model_config.tie_word_embeddings.unwrap_or(false),
                    bos_token_id: model_config.bos_token_id,
                    eos_token_id: eos_token_id_config,
                    use_sliding_window: false,
                    sliding_window: None,
                };

                let lazy_qwen2 = LazyQwen2::load(qwen2_config, lazy_vb, max_loaded_layers)
                    .map_err(|e| infernum_core::Error::ModelLoad {
                        message: format!("Failed to load LazyQwen2: {}", e),
                    })?;

                ModelKind::LazyQwen2(lazy_qwen2)
            },
            ArchitectureType::Llama | ArchitectureType::Unknown => {
                if arch_type == ArchitectureType::Unknown {
                    tracing::warn!(
                        "Unknown architecture, defaulting to LazyLlama. Model may not work correctly."
                    );
                } else {
                    tracing::info!("Loading LazyLlama model (Llama architecture detected)");
                }

                let llama_config = LlamaConfig {
                    hidden_size: model_config.hidden_size.unwrap_or(16384),
                    intermediate_size: model_config.intermediate_size.unwrap_or(53248),
                    vocab_size: model_config.vocab_size.unwrap_or(128256),
                    num_hidden_layers: model_config.num_hidden_layers.unwrap_or(126),
                    num_attention_heads: model_config.num_attention_heads.unwrap_or(128),
                    num_key_value_heads: model_config.num_key_value_heads,
                    rms_norm_eps: model_config.rms_norm_eps.unwrap_or(1e-5),
                    rope_theta: model_config.rope_theta.unwrap_or(500000.0),
                    max_position_embeddings: model_config.max_position_embeddings.unwrap_or(131072),
                    tie_word_embeddings: model_config.tie_word_embeddings.unwrap_or(false),
                    bos_token_id: model_config.bos_token_id,
                    eos_token_id: eos_token_id_config,
                    rope_scaling: model_config.rope_scaling.clone(),
                };

                let lazy_llama = LazyLlama::load(llama_config, lazy_vb, max_loaded_layers)
                    .map_err(|e| infernum_core::Error::ModelLoad {
                        message: format!("Failed to load LazyLlama: {}", e),
                    })?;

                ModelKind::LazyLlama(lazy_llama)
            },
        };

        // Load tokenizer
        let tokenizer = if let Some(tokenizer_path) = &files.tokenizer {
            Tokenizer::from_file(tokenizer_path)?
        } else {
            return Err(infernum_core::Error::ModelLoad {
                message: "No tokenizer found for model".to_string(),
            });
        };

        // Get EOS token with architecture-appropriate default
        let eos_token_id =
            model_config
                .eos_token_ids()
                .first()
                .copied()
                .unwrap_or(match arch_type {
                    ArchitectureType::Qwen2 => 151643,
                    _ => 128001,
                });

        let elapsed = start.elapsed();
        tracing::info!(
            elapsed_ms = elapsed.as_millis(),
            architecture = arch_type.name(),
            swap_count = plan.swap_count,
            quality_score = format!("{:.3}", plan.quality_score),
            "Adaptive model loaded successfully"
        );

        Ok(LoadedModel {
            model: Mutex::new(model),
            tokenizer,
            eos_token_id,
        })
    }

    /// Loads a model using eager decompression with intelligent tensor placement.
    ///
    /// This is the fast path for models that fit in VRAM+RAM. Uses `hct_sequential`
    /// for parallel tensor decompression instead of progressive streaming.
    ///
    /// Achieves ~2+ tk/s for 14B models vs ~0.3 tk/s with progressive loading.
    #[allow(clippy::too_many_arguments)]
    fn load_model_eager_with_placement(
        files: &ModelFiles,
        model_config: &ModelConfig,
        device: &Device,
        dtype: DType,
        directory: &std::path::Path,
        profile: &crate::adaptive_tiering::ModelProfile,
        plan: &crate::adaptive_tiering::AllocationPlan,
        arch_type: ArchitectureType,
        start: Instant,
    ) -> Result<LoadedModel> {
        use crate::adaptive_tiering::{AdaptiveLoader, EagerTensorProvider};
        use crate::lazy_varbuilder::{CacheConfig, LazyVarBuilder};
        use crate::models::{LazyLlama, LazyQwen2, LlamaConfig, Qwen2Config};

        tracing::info!(
            directory = %directory.display(),
            tensor_count = profile.tensors.len(),
            vram_budget_gb = format!("{:.2}", plan.vram_usage as f64 / 1e9),
            ram_budget_gb = format!("{:.2}", plan.ram_usage as f64 / 1e9),
            "Using eager loading with tiered placement (fast path)"
        );

        // PHASE 1: Eager decompression to CPU (RAM)
        // Load all tensors to CPU first - this is fast parallel decompression
        // without the streaming overhead of TieredHoloLoader
        tracing::info!("Phase 1: Eager decompression to RAM...");
        let cpu_device = Device::Cpu;
        let tensors = crate::hct_sequential::load_hct_directory_parallel(
            directory,
            &cpu_device, // Load to CPU (RAM), not GPU
            dtype,
        )
        .map_err(|e| infernum_core::Error::ModelLoad {
            message: format!("Failed to load HCT tensors: {}", e),
        })?;

        let total_bytes: u64 = tensors
            .values()
            .map(|t| (t.elem_count() * t.dtype().size_in_bytes()) as u64)
            .sum();

        tracing::info!(
            tensor_count = tensors.len(),
            total_gb = format!("{:.2}", total_bytes as f64 / 1e9),
            "Phase 1 complete: All tensors loaded to RAM"
        );

        // PHASE 2: Create tiered provider
        // EagerTensorProvider wraps the pre-loaded tensors
        // AdaptiveLoader adds VRAM/RAM tiering based on the allocation plan
        tracing::info!("Phase 2: Creating tiered provider...");
        let eager_provider = EagerTensorProvider::new(tensors);
        let adaptive_loader = AdaptiveLoader::new(plan.clone(), eager_provider, device.clone());

        // PHASE 3: Preload hot tensors to VRAM
        // This moves the most important tensors to GPU memory upfront
        tracing::info!("Phase 3: Preloading hot tensors to VRAM...");
        if let Err(e) = adaptive_loader.preload_vram_tensors() {
            tracing::warn!(error = %e, "Failed to preload VRAM tensors, continuing with on-demand loading");
        }

        let vram_usage = adaptive_loader.vram_cache_usage();
        let ram_usage = adaptive_loader.ram_cache_usage();
        tracing::info!(
            vram_gb = format!("{:.2}", vram_usage as f64 / 1e9),
            ram_gb = format!("{:.2}", ram_usage as f64 / 1e9),
            "Phase 3 complete: Hot tensors in VRAM, warm tensors in RAM"
        );

        // Create LazyVarBuilder with the adaptive loader
        // Disable LazyVarBuilder's own cache - AdaptiveLoader handles caching
        let provider: std::sync::Arc<dyn crate::lazy_varbuilder::TensorProvider> =
            std::sync::Arc::new(adaptive_loader);
        let cache_config = CacheConfig {
            max_memory_bytes: 0,
            max_entries: 0,
        };
        let lazy_vb = LazyVarBuilder::with_cache_config(
            std::sync::Arc::clone(&provider),
            device.clone(),
            dtype,
            cache_config,
        );

        // All layers can be "loaded" since tensors are already in RAM
        let max_loaded_layers = profile.num_layers;

        // Get EOS token ID for config
        let eos_token_id_config = model_config.eos_token_ids().first().copied();

        // PHASE 4: Build the model using lazy loading
        tracing::info!("Phase 4: Building model with lazy tensor access...");
        let model = match arch_type {
            ArchitectureType::Qwen2 => {
                tracing::info!("Loading LazyQwen2 model with eager-tiered weights");

                let qwen2_config = Qwen2Config {
                    hidden_size: model_config.hidden_size.unwrap_or(5120),
                    intermediate_size: model_config.intermediate_size.unwrap_or(13824),
                    vocab_size: model_config.vocab_size.unwrap_or(152064),
                    num_hidden_layers: model_config.num_hidden_layers.unwrap_or(48),
                    num_attention_heads: model_config.num_attention_heads.unwrap_or(40),
                    num_key_value_heads: model_config.num_key_value_heads,
                    rms_norm_eps: model_config.rms_norm_eps.unwrap_or(1e-6),
                    rope_theta: model_config.rope_theta.unwrap_or(1000000.0),
                    max_position_embeddings: model_config.max_position_embeddings.unwrap_or(32768),
                    tie_word_embeddings: model_config.tie_word_embeddings.unwrap_or(false),
                    bos_token_id: model_config.bos_token_id,
                    eos_token_id: eos_token_id_config,
                    use_sliding_window: false,
                    sliding_window: None,
                };

                let lazy_qwen2 = LazyQwen2::load(qwen2_config, lazy_vb, max_loaded_layers)
                    .map_err(|e| infernum_core::Error::ModelLoad {
                        message: format!("Failed to load LazyQwen2: {}", e),
                    })?;

                ModelKind::LazyQwen2(lazy_qwen2)
            },
            ArchitectureType::Llama | ArchitectureType::Unknown => {
                if arch_type == ArchitectureType::Unknown {
                    tracing::warn!(
                        "Unknown architecture, defaulting to LazyLlama. Model may not work correctly."
                    );
                } else {
                    tracing::info!("Loading LazyLlama model with eager-tiered weights");
                }

                let llama_config = LlamaConfig {
                    hidden_size: model_config.hidden_size.unwrap_or(16384),
                    intermediate_size: model_config.intermediate_size.unwrap_or(53248),
                    vocab_size: model_config.vocab_size.unwrap_or(128256),
                    num_hidden_layers: model_config.num_hidden_layers.unwrap_or(126),
                    num_attention_heads: model_config.num_attention_heads.unwrap_or(128),
                    num_key_value_heads: model_config.num_key_value_heads,
                    rms_norm_eps: model_config.rms_norm_eps.unwrap_or(1e-5),
                    rope_theta: model_config.rope_theta.unwrap_or(500000.0),
                    max_position_embeddings: model_config.max_position_embeddings.unwrap_or(131072),
                    tie_word_embeddings: model_config.tie_word_embeddings.unwrap_or(false),
                    bos_token_id: model_config.bos_token_id,
                    eos_token_id: eos_token_id_config,
                    rope_scaling: model_config.rope_scaling.clone(),
                };

                let lazy_llama = LazyLlama::load(llama_config, lazy_vb, max_loaded_layers)
                    .map_err(|e| infernum_core::Error::ModelLoad {
                        message: format!("Failed to load LazyLlama: {}", e),
                    })?;

                ModelKind::LazyLlama(lazy_llama)
            },
        };

        // Load tokenizer
        let tokenizer = if let Some(tokenizer_path) = &files.tokenizer {
            Tokenizer::from_file(tokenizer_path)?
        } else {
            return Err(infernum_core::Error::ModelLoad {
                message: "No tokenizer found for model".to_string(),
            });
        };

        // Get EOS token with architecture-appropriate default
        let eos_token_id =
            model_config
                .eos_token_ids()
                .first()
                .copied()
                .unwrap_or(match arch_type {
                    ArchitectureType::Qwen2 => 151643,
                    _ => 128001,
                });

        let elapsed = start.elapsed();
        tracing::info!(
            elapsed_ms = elapsed.as_millis(),
            architecture = arch_type.name(),
            vram_gb = format!("{:.2}", vram_usage as f64 / 1e9),
            ram_gb = format!("{:.2}", ram_usage as f64 / 1e9),
            quality_score = format!("{:.3}", plan.quality_score),
            "Model loaded successfully with eager-tiered placement"
        );

        Ok(LoadedModel {
            model: Mutex::new(model),
            tokenizer,
            eos_token_id,
        })
    }

    /// Loads weights from files into a VarBuilder.
    fn load_weights(
        weights: &WeightFiles,
        device: &Device,
        dtype: DType,
    ) -> Result<VarBuilder<'static>> {
        match weights {
            WeightFiles::SingleSafetensors(path) => {
                let data = std::fs::read(path).map_err(|e| infernum_core::Error::ModelLoad {
                    message: format!("Failed to read weights: {}", e),
                })?;

                let vb =
                    VarBuilder::from_buffered_safetensors(data, dtype, device).map_err(|e| {
                        infernum_core::Error::ModelLoad {
                            message: format!("Failed to create VarBuilder: {}", e),
                        }
                    })?;
                Ok(vb)
            },
            WeightFiles::ShardedSafetensors { shards, .. } => {
                // Use memory-mapped loading for sharded files for better memory efficiency
                // SAFETY: The files are read-only and we control the paths
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(shards, dtype, device).map_err(|e| {
                        infernum_core::Error::ModelLoad {
                            message: format!("Failed to mmap shards: {}", e),
                        }
                    })?
                };
                Ok(vb)
            },
            WeightFiles::Gguf(path) => {
                // Load and validate the GGUF file
                let loader = crate::gguf::GgufLoader::from_file(path)?;
                let metadata = loader.metadata();

                tracing::info!(
                    architecture = %metadata.architecture,
                    quantization = %metadata.quantization_type,
                    layers = metadata.num_layers,
                    vocab_size = metadata.vocab_size,
                    "Loaded GGUF metadata"
                );

                // For now, GGUF quantized inference is experimental
                // Return an informative error with details about the model
                Err(infernum_core::Error::ModelLoad {
                    message: format!(
                        "GGUF quantized inference is experimental. \
                        Model: {} ({} architecture, {} layers, {} quantization). \
                        Please use safetensors format for production use, or enable \
                        experimental GGUF support with --experimental-gguf flag.",
                        metadata.name.as_deref().unwrap_or("unknown"),
                        metadata.architecture,
                        metadata.num_layers,
                        metadata.quantization_type
                    ),
                })
            },
            WeightFiles::PyTorch(_) | WeightFiles::ShardedPyTorch { .. } => {
                Err(infernum_core::Error::ModelLoad {
                    message: "PyTorch format not supported, please use safetensors".to_string(),
                })
            },
            WeightFiles::Hct { directory, files } => {
                // Load HCT compressed tensors using sequential loader to prevent OOM
                tracing::info!(
                    directory = %directory.display(),
                    file_count = files.len(),
                    "Loading HCT compressed weights (sequential)"
                );

                // Use sequential loading to prevent OOM on large models
                let tensors =
                    crate::hct_sequential::load_hct_directory_sequential(directory, device, dtype)?;

                tracing::info!(tensor_count = tensors.len(), "Loaded HCT tensors");

                // Create a VarBuilder from the tensor map
                let vb = VarBuilder::from_tensors(tensors, dtype, device);
                Ok(vb)
            },
            WeightFiles::HoloTensor {
                directory,
                min_quality,
                target_quality,
                vram_budget,
                ram_budget,
            } => {
                // Progressive loading with tiered memory management for 405B+ models
                tracing::info!(
                    directory = %directory.display(),
                    min_quality = %min_quality,
                    target_quality = %target_quality,
                    vram_budget_gb = vram_budget / (1024 * 1024 * 1024),
                    ram_budget_gb = ram_budget / (1024 * 1024 * 1024),
                    "Loading HoloTensor weights (progressive)"
                );

                use crate::holotensor::tiered_loading::{TieredConfig, TieredHoloLoader};
                use crate::lazy_varbuilder::LazyVarBuilder;

                // Configure tiered loading
                let tiered_config = TieredConfig {
                    vram_budget: *vram_budget,
                    ram_budget: *ram_budget,
                    min_quality: *min_quality,
                    target_quality: *target_quality,
                    enable_background_streaming: true,
                    background_streams: 4,
                };

                // Create the tiered loader
                let mut loader =
                    TieredHoloLoader::new(directory, tiered_config, device.clone(), dtype)
                        .map_err(|e| infernum_core::Error::ModelLoad {
                            message: format!("Failed to create tiered loader: {}", e),
                        })?;

                // Enable NVMe cache for decompressed tensors if configured
                // This provides ~1000x speedup on subsequent layer loads (100ms vs 100s)
                if let Ok(cache_dir) = std::env::var("INFERNUM_CACHE_DIR") {
                    let cache_path = std::path::Path::new(&cache_dir);
                    if cache_path.exists() || std::fs::create_dir_all(cache_path).is_ok() {
                        tracing::info!(
                            cache_dir = %cache_dir,
                            "NVMe cache enabled - subsequent layer loads will be ~1000x faster"
                        );
                        loader = loader.with_safetensors_dir(cache_path);
                    } else {
                        tracing::warn!(
                            cache_dir = %cache_dir,
                            "Could not create NVMe cache directory, falling back to HCT reconstruction"
                        );
                    }
                }

                // Start background streaming for quality improvement
                loader.start_background_streaming();

                // Create lazy VarBuilder backed by tiered loader
                let provider: std::sync::Arc<dyn crate::lazy_varbuilder::TensorProvider> =
                    std::sync::Arc::new(loader);

                // CRITICAL: Disable LazyVarBuilder cache - TieredHoloLoader already caches on CPU
                // Without this, we'd have duplicate caches:
                //   1. TieredHoloLoader.cpu_cache: CPU tensors (correct)
                //   2. LazyVarBuilder.cache: GPU tensors (60GB default - OOM risk!)
                let cache_config = crate::lazy_varbuilder::CacheConfig {
                    max_memory_bytes: 0, // Disable cache
                    max_entries: 0,
                };
                let lazy_vb = LazyVarBuilder::with_cache_config(
                    std::sync::Arc::clone(&provider),
                    device.clone(),
                    dtype,
                    cache_config,
                );

                tracing::info!(
                    min_quality = %min_quality,
                    "Progressive loading initialized, background streaming started"
                );

                // Return a VarBuilder-compatible wrapper
                // For now, we load all tensors eagerly since Llama/Qwen2 expect VarBuilder
                // Future: Modify model loading to use lazy access
                let tensor_names = provider.tensor_names();
                let mut tensors = std::collections::HashMap::new();

                for name in tensor_names {
                    match lazy_vb.get(&name) {
                        Ok(tensor) => {
                            tensors.insert(name.clone(), tensor);
                        },
                        Err(e) => {
                            tracing::warn!(name = %name, error = %e, "Failed to load tensor, skipping");
                        },
                    }
                }

                tracing::info!(tensor_count = tensors.len(), "Progressive loading complete");

                let vb = VarBuilder::from_tensors(tensors, dtype, device);
                Ok(vb)
            },
        }
    }

    /// Builds model metadata from config.
    fn build_metadata(config: &EngineConfig, model_config: &ModelConfig) -> Result<ModelMetadata> {
        use infernum_core::model::LlamaVersion;
        use infernum_core::{ModelArchitecture, ModelId};

        let id = match &config.model {
            infernum_core::ModelSource::HuggingFace { repo_id, .. } => repo_id.clone(),
            infernum_core::ModelSource::LocalPath { path } => path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("local-model")
                .to_string(),
            infernum_core::ModelSource::S3 { key, .. } => key.clone(),
            infernum_core::ModelSource::Gguf { path } => path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("gguf-model")
                .to_string(),
            infernum_core::ModelSource::HoloTensor { path, .. } => path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("holo-model")
                .to_string(),
        };

        // Detect Llama version from config
        let version = match model_config.architecture() {
            Some(arch) if arch.contains("Llama3") || arch.contains("llama-3") => LlamaVersion::V3,
            _ => LlamaVersion::V3_2, // Default to latest
        };

        Ok(
            ModelMetadata::builder(ModelId::new(&id), ModelArchitecture::Llama { version })
                .source(config.model.clone())
                .context_length(model_config.max_position_embeddings.unwrap_or(4096) as u32)
                .vocab_size(model_config.vocab_size.unwrap_or(32000) as u32)
                .hidden_size(model_config.hidden_size.unwrap_or(4096) as u32)
                .num_layers(model_config.num_hidden_layers.unwrap_or(32) as u32)
                .num_attention_heads(model_config.num_attention_heads.unwrap_or(32) as u32)
                .build(),
        )
    }

    /// Returns the engine configuration.
    #[must_use]
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Returns speculative decoding statistics, if speculative decoding is enabled.
    #[must_use]
    pub fn speculative_stats(&self) -> Option<crate::speculative::SpeculativeStats> {
        self.speculative_decoder.as_ref().map(|d| d.stats())
    }

    /// Returns true if speculative decoding is enabled.
    #[must_use]
    pub fn has_speculative_decoding(&self) -> bool {
        self.speculative_decoder.is_some()
    }

    /// Creates a shareable reference to this engine.
    #[must_use]
    pub fn into_shared(self) -> Arc<Self> {
        Arc::new(self)
    }

    /// Generates tokens from the model.
    ///
    /// If speculative decoding is enabled, uses the draft model to accelerate generation.
    /// Otherwise, falls back to standard token-by-token generation.
    fn generate_tokens(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        sampler: &mut Sampler,
    ) -> Result<(Vec<u32>, Vec<String>)> {
        let loaded = self
            .loaded
            .as_ref()
            .ok_or_else(|| infernum_core::Error::Internal {
                message: "Model not loaded".to_string(),
            })?;

        // Use speculative decoding if available
        if let Some(spec_decoder) = &self.speculative_decoder {
            let eos_token = loaded.eos_token_id;
            let mut model = loaded.model.lock();

            let result = spec_decoder.generate(
                &mut model,
                prompt_tokens,
                max_tokens,
                sampler.params(),
                eos_token,
            );

            // Log speculative stats
            let stats = spec_decoder.stats();
            tracing::debug!(
                acceptance_rate = %format!("{:.1}%", stats.acceptance_rate() * 100.0),
                avg_tokens_per_round = %format!("{:.2}", stats.avg_tokens_per_round()),
                rounds = stats.rounds,
                "Speculative decoding stats"
            );

            return result;
        }

        // Standard generation path
        let mut model = loaded.model.lock();
        model.clear_cache();

        // Convert prompt to tensor
        let input_ids = Tensor::new(prompt_tokens, &self.device).map_err(|e| {
            infernum_core::Error::Internal {
                message: format!("Failed to create input tensor: {}", e),
            }
        })?;
        let input_ids = input_ids
            .unsqueeze(0)
            .map_err(|e| infernum_core::Error::Internal {
                message: format!("Failed to unsqueeze: {}", e),
            })?;

        // Prefill: process the entire prompt
        let logits = model
            .forward(&input_ids, 0)
            .map_err(|e| infernum_core::Error::Internal {
                message: format!("Forward pass failed: {}", e),
            })?;

        // Get logits for last position
        let seq_len = prompt_tokens.len();
        let last_logits =
            logits
                .i((0, seq_len - 1, ..))
                .map_err(|e| infernum_core::Error::Internal {
                    message: format!("Failed to index logits: {}", e),
                })?;

        // Convert to F32 if needed before extracting to Vec
        let last_logits = last_logits.to_dtype(candle_core::DType::F32).map_err(|e| {
            infernum_core::Error::Internal {
                message: format!("Failed to convert logits to F32: {}", e),
            }
        })?;

        let logits_vec: Vec<f32> =
            last_logits
                .to_vec1()
                .map_err(|e| infernum_core::Error::Internal {
                    message: format!("Failed to convert logits: {}", e),
                })?;

        // Note: We don't initialize context with prompt tokens as this can
        // penalize important tokens like special formatting tokens.
        // The context builds naturally as we generate.

        // Sample first token
        let mut generated_tokens = Vec::new();
        let mut generated_text = Vec::new();
        let mut next_token = sampler.sample(&logits_vec);
        sampler.add_token(next_token);

        // Check for EOS
        let eos_token = loaded.eos_token_id;

        for _ in 0..max_tokens {
            if next_token == eos_token {
                break;
            }

            generated_tokens.push(next_token);

            // Decode token
            let token_text = loaded.tokenizer.decode_token(next_token)?;
            generated_text.push(token_text);

            // Check stop sequences
            let full_text: String = generated_text.join("");
            if sampler.is_stop_token(&full_text) {
                break;
            }

            // Generate next token
            let next_input = Tensor::new(&[next_token], &self.device).map_err(|e| {
                infernum_core::Error::Internal {
                    message: format!("Failed to create next input: {}", e),
                }
            })?;
            let next_input =
                next_input
                    .unsqueeze(0)
                    .map_err(|e| infernum_core::Error::Internal {
                        message: format!("Failed to unsqueeze: {}", e),
                    })?;

            let logits = model
                .forward(&next_input, seq_len + generated_tokens.len() - 1)
                .map_err(|e| infernum_core::Error::Internal {
                    message: format!("Forward pass failed: {}", e),
                })?;

            let last_logits = logits
                .i((0, 0, ..))
                .map_err(|e| infernum_core::Error::Internal {
                    message: format!("Failed to index logits: {}", e),
                })?;

            // Convert to F32 if needed
            let last_logits = last_logits.to_dtype(candle_core::DType::F32).map_err(|e| {
                infernum_core::Error::Internal {
                    message: format!("Failed to convert logits to F32: {}", e),
                }
            })?;

            let logits_vec: Vec<f32> =
                last_logits
                    .to_vec1()
                    .map_err(|e| infernum_core::Error::Internal {
                        message: format!("Failed to convert logits: {}", e),
                    })?;

            next_token = sampler.sample(&logits_vec);
            sampler.add_token(next_token);
        }

        Ok((generated_tokens, generated_text))
    }

    // ========================================================================
    // Model Warmup (Sprint 7 - Leviathan Feature Parity)
    // ========================================================================

    /// Warms up the model by running a small inference.
    ///
    /// This populates the KV cache and exercises the model's compute paths,
    /// reducing first-inference latency by 30-50%.
    pub async fn warmup(&self) -> Result<WarmupResult> {
        let start = Instant::now();

        tracing::info!("Starting model warmup");

        let loaded = self
            .loaded
            .as_ref()
            .ok_or_else(|| infernum_core::Error::Internal {
                message: "Model not loaded".to_string(),
            })?;

        // Run a minimal generation to warm up the model
        let warmup_prompt = "Hello";
        let input_ids = loaded.tokenizer.encode(warmup_prompt, true).map_err(|e| {
            infernum_core::Error::Internal {
                message: format!("Tokenization failed: {}", e),
            }
        })?;

        let input_tensor = Tensor::new(input_ids.as_slice(), &self.device)
            .map_err(|e| infernum_core::Error::Internal {
                message: format!("Failed to create tensor: {}", e),
            })?
            .unsqueeze(0)
            .map_err(|e| infernum_core::Error::Internal {
                message: format!("Failed to unsqueeze: {}", e),
            })?;

        // Run forward pass
        let mut model = loaded.model.lock();
        let _logits =
            model
                .forward(&input_tensor, 0)
                .map_err(|e| infernum_core::Error::Internal {
                    message: format!("Warmup forward pass failed: {}", e),
                })?;
        drop(model);

        // Clear KV cache after warmup
        let mut model = loaded.model.lock();
        model.clear_cache();
        drop(model);

        let duration = start.elapsed();
        tracing::info!(duration_ms = duration.as_millis(), "Model warmup complete");

        Ok(WarmupResult {
            success: true,
            duration_ms: duration.as_millis() as u64,
            model_id: self.metadata.id.to_string(),
        })
    }

    /// Warms up the model with a custom prompt.
    pub async fn warmup_with_prompt(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<WarmupResult> {
        let start = Instant::now();

        tracing::info!(
            prompt_len = prompt.len(),
            max_tokens,
            "Starting custom warmup"
        );

        let request = GenerateRequest::new(vec![Message {
            role: Role::User,
            content: prompt.to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }])
        .with_sampling(
            SamplingParams::default().with_max_tokens(max_tokens.min(10) as u32), // Cap at 10 tokens for warmup
        );

        // Run inference
        let _response = self.generate(request).await?;

        let duration = start.elapsed();
        tracing::info!(duration_ms = duration.as_millis(), "Custom warmup complete");

        Ok(WarmupResult {
            success: true,
            duration_ms: duration.as_millis() as u64,
            model_id: self.metadata.id.to_string(),
        })
    }

    // ========================================================================
    // Graceful Shutdown (Sprint 7 - Leviathan Feature Parity)
    // ========================================================================

    /// Gracefully unloads the model and releases resources.
    ///
    /// This should be called before dropping the engine to ensure proper
    /// cleanup of GPU memory and other resources.
    pub async fn shutdown(&mut self) -> Result<ShutdownResult> {
        let start = Instant::now();

        tracing::info!("Starting graceful shutdown");

        // Clear speculative decoder first
        if self.speculative_decoder.is_some() {
            tracing::debug!("Releasing speculative decoder");
            self.speculative_decoder = None;
        }

        // Unload the model
        if let Some(loaded) = self.loaded.take() {
            tracing::debug!("Releasing model resources");

            // Clear KV cache before dropping
            if let Ok(mut model) = Arc::try_unwrap(loaded) {
                let model_lock = model.model.get_mut();
                model_lock.clear_cache();
                tracing::debug!("KV cache cleared");
            }
            // If Arc::try_unwrap fails, there are other references - that's ok,
            // they will clean up when dropped
        }

        let duration = start.elapsed();
        tracing::info!(
            duration_ms = duration.as_millis(),
            "Graceful shutdown complete"
        );

        Ok(ShutdownResult {
            success: true,
            duration_ms: duration.as_millis() as u64,
            resources_released: vec![
                "model_weights".to_string(),
                "kv_cache".to_string(),
                "speculative_decoder".to_string(),
            ],
        })
    }

    /// Checks if the model is currently loaded.
    #[must_use]
    pub fn is_loaded(&self) -> bool {
        self.loaded.is_some()
    }

    /// Returns the current device.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Result of a warmup operation.
#[derive(Debug, Clone)]
pub struct WarmupResult {
    /// Whether warmup succeeded.
    pub success: bool,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Model ID that was warmed up.
    pub model_id: String,
}

/// Result of a shutdown operation.
#[derive(Debug, Clone)]
pub struct ShutdownResult {
    /// Whether shutdown succeeded.
    pub success: bool,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Resources that were released.
    pub resources_released: Vec<String>,
}

#[async_trait]
impl InferenceEngine for Engine {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        let start = Instant::now();
        tracing::debug!(request_id = %request.request_id, "Starting generation");

        let loaded = self
            .loaded
            .as_ref()
            .ok_or_else(|| infernum_core::Error::Internal {
                message: "Model not loaded".to_string(),
            })?;

        // Encode prompt
        let prompt_text = match &request.prompt {
            infernum_core::request::PromptInput::Text(s) => s.clone(),
            infernum_core::request::PromptInput::Messages(msgs) => {
                loaded.tokenizer.apply_chat_template(msgs, true)?
            },
            infernum_core::request::PromptInput::Tokens(_) => {
                return Err(infernum_core::Error::Internal {
                    message: "Pre-tokenized input not yet supported".to_string(),
                });
            },
        };

        let prompt_tokens = loaded.tokenizer.encode(&prompt_text, true)?;
        let prompt_token_count = prompt_tokens.len() as u32;

        tracing::debug!(prompt_tokens = prompt_token_count, "Encoded prompt");

        let time_to_first_token = start.elapsed();

        // Try CUDA optimized path first (5-6x faster)
        #[cfg(feature = "cuda")]
        if let Some(ref cuda_gen) = self.cuda_generator {
            tracing::debug!("Using CUDA optimized inference path");

            // Convert sampling params
            let cuda_params = crate::cuda_inference::SamplingParams {
                temperature: request.sampling.temperature,
                top_p: request.sampling.top_p,
                top_k: request.sampling.top_k as usize,
                repetition_penalty: request.sampling.repetition_penalty,
                repetition_context: 64,
                max_tokens: request.sampling.max_tokens as usize,
                stop_tokens: vec![loaded.eos_token_id],
                seed: request.sampling.seed.unwrap_or(42),
            };

            // Generate using CUDA path
            let mut generator = cuda_gen.lock();
            let generated_tokens = generator
                .generate(&prompt_tokens, Some(&cuda_params))
                .map_err(|e| infernum_core::Error::Internal {
                    message: format!("CUDA generation failed: {}", e),
                })?;

            // Decode tokens
            let generated_text: Vec<String> = generated_tokens
                .iter()
                .filter_map(|&t| loaded.tokenizer.decode_token(t).ok())
                .collect();

            let completion_token_count = generated_tokens.len() as u32;
            let total_time = start.elapsed();
            let text = generated_text.join("");

            let tokens_per_sec = completion_token_count as f64 / total_time.as_secs_f64();
            tracing::info!(
                tokens_per_sec = format!("{:.1}", tokens_per_sec),
                path = "cuda_optimized",
                "Generation complete"
            );

            return Ok(GenerateResponse {
                request_id: request.request_id,
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64,
                model: self.metadata.id.clone(),
                choices: vec![infernum_core::response::Choice {
                    index: 0,
                    text,
                    message: None,
                    finish_reason: Some(infernum_core::FinishReason::Stop),
                    logprobs: None,
                }],
                usage: infernum_core::Usage::new(prompt_token_count, completion_token_count),
                time_to_first_token_ms: Some(time_to_first_token.as_secs_f64() * 1000.0),
                total_time_ms: Some(total_time.as_secs_f64() * 1000.0),
            });
        }

        // Fall back to Candle path
        let mut sampler = Sampler::new(request.sampling.clone());
        let (generated_tokens, generated_text) =
            self.generate_tokens(&prompt_tokens, request.sampling.max_tokens, &mut sampler)?;

        let completion_token_count = generated_tokens.len() as u32;
        let total_time = start.elapsed();

        let text = generated_text.join("");

        tracing::debug!(
            request_id = %request.request_id,
            prompt_tokens = prompt_token_count,
            completion_tokens = completion_token_count,
            total_time_ms = total_time.as_millis(),
            "Generation complete"
        );

        Ok(GenerateResponse {
            request_id: request.request_id,
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
            model: self.metadata.id.clone(),
            choices: vec![infernum_core::response::Choice {
                index: 0,
                text,
                message: None,
                finish_reason: Some(infernum_core::FinishReason::Stop),
                logprobs: None,
            }],
            usage: infernum_core::Usage::new(prompt_token_count, completion_token_count),
            time_to_first_token_ms: Some(time_to_first_token.as_secs_f64() * 1000.0),
            total_time_ms: Some(total_time.as_secs_f64() * 1000.0),
        })
    }

    async fn generate_stream(&self, request: GenerateRequest) -> Result<TokenStream> {
        use futures::stream;
        use infernum_core::streaming::{StreamChoice, StreamChunk, StreamDelta};
        use infernum_core::{FinishReason, Usage};

        tracing::debug!(request_id = %request.request_id, "Starting streaming generation");

        let loaded =
            Arc::clone(
                self.loaded
                    .as_ref()
                    .ok_or_else(|| infernum_core::Error::Internal {
                        message: "Model not loaded".to_string(),
                    })?,
            );

        // Encode prompt before moving into the task
        let prompt_text = match &request.prompt {
            infernum_core::request::PromptInput::Text(s) => s.clone(),
            infernum_core::request::PromptInput::Messages(msgs) => {
                loaded.tokenizer.apply_chat_template(msgs, true)?
            },
            infernum_core::request::PromptInput::Tokens(_) => {
                return Err(infernum_core::Error::Internal {
                    message: "Pre-tokenized input not yet supported".to_string(),
                });
            },
        };

        let prompt_tokens = loaded.tokenizer.encode(&prompt_text, true)?;
        let prompt_token_count = prompt_tokens.len() as u32;

        // Create channel for streaming tokens
        let (tx, rx) = mpsc::channel::<Result<StreamChunk>>(32);

        let request_id = request.request_id.clone();
        let model_id = self.metadata.id.clone();
        let device = self.device.clone();
        let max_tokens = request.sampling.max_tokens;
        let sampling_params = request.sampling.clone();
        let eos_token = loaded.eos_token_id;

        // Spawn token generation task - uses Arc<LoadedModel> to share state
        tokio::task::spawn_blocking(move || {
            let mut sampler = Sampler::new(sampling_params);
            let mut model_guard = loaded.model.lock();
            model_guard.clear_cache();

            // Convert prompt to tensor
            let input_ids = match Tensor::new(prompt_tokens.as_slice(), &device) {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx.blocking_send(Err(infernum_core::Error::Internal {
                        message: format!("Failed to create input tensor: {}", e),
                    }));
                    return;
                },
            };

            let input_ids = match input_ids.unsqueeze(0) {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx.blocking_send(Err(infernum_core::Error::Internal {
                        message: format!("Failed to unsqueeze: {}", e),
                    }));
                    return;
                },
            };

            // Prefill
            let logits = match model_guard.forward(&input_ids, 0) {
                Ok(l) => l,
                Err(e) => {
                    let _ = tx.blocking_send(Err(infernum_core::Error::Internal {
                        message: format!("Forward pass failed: {}", e),
                    }));
                    return;
                },
            };

            let seq_len = prompt_tokens.len();
            let last_logits = match logits.i((0, seq_len - 1, ..)) {
                Ok(l) => l,
                Err(e) => {
                    let _ = tx.blocking_send(Err(infernum_core::Error::Internal {
                        message: format!("Failed to index logits: {}", e),
                    }));
                    return;
                },
            };

            // Convert to F32 if needed
            let last_logits = match last_logits.to_dtype(candle_core::DType::F32) {
                Ok(l) => l,
                Err(e) => {
                    let _ = tx.blocking_send(Err(infernum_core::Error::Internal {
                        message: format!("Failed to convert logits to F32: {}", e),
                    }));
                    return;
                },
            };

            let logits_vec: Vec<f32> = match last_logits.to_vec1() {
                Ok(v) => v,
                Err(e) => {
                    let _ = tx.blocking_send(Err(infernum_core::Error::Internal {
                        message: format!("Failed to convert logits: {}", e),
                    }));
                    return;
                },
            };

            let mut next_token = sampler.sample(&logits_vec);
            let mut generated_count: u32 = 0;
            let mut full_text = String::new();

            for _ in 0..max_tokens {
                if next_token == eos_token {
                    break;
                }

                generated_count += 1;

                // Decode token
                let token_text = match loaded.tokenizer.decode_token(next_token) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(e));
                        return;
                    },
                };

                full_text.push_str(&token_text);

                // Send streaming chunk
                let chunk = StreamChunk {
                    request_id: request_id.clone(),
                    model: model_id.clone(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: StreamDelta::text(&token_text),
                        finish_reason: None,
                    }],
                    usage: None,
                };

                if tx.blocking_send(Ok(chunk)).is_err() {
                    // Receiver dropped, stop generation
                    return;
                }

                // Check stop sequences
                if sampler.is_stop_token(&full_text) {
                    break;
                }

                // Generate next token
                let next_input = match Tensor::new(&[next_token], &device) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(infernum_core::Error::Internal {
                            message: format!("Failed to create next input: {}", e),
                        }));
                        return;
                    },
                };

                let next_input = match next_input.unsqueeze(0) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(infernum_core::Error::Internal {
                            message: format!("Failed to unsqueeze: {}", e),
                        }));
                        return;
                    },
                };

                let logits = match model_guard
                    .forward(&next_input, seq_len + generated_count as usize - 1)
                {
                    Ok(l) => l,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(infernum_core::Error::Internal {
                            message: format!("Forward pass failed: {}", e),
                        }));
                        return;
                    },
                };

                let last_logits = match logits.i((0, 0, ..)) {
                    Ok(l) => l,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(infernum_core::Error::Internal {
                            message: format!("Failed to index logits: {}", e),
                        }));
                        return;
                    },
                };

                // Convert to F32 if needed
                let last_logits = match last_logits.to_dtype(candle_core::DType::F32) {
                    Ok(l) => l,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(infernum_core::Error::Internal {
                            message: format!("Failed to convert logits to F32: {}", e),
                        }));
                        return;
                    },
                };

                let logits_vec: Vec<f32> = match last_logits.to_vec1() {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(infernum_core::Error::Internal {
                            message: format!("Failed to convert logits: {}", e),
                        }));
                        return;
                    },
                };

                next_token = sampler.sample(&logits_vec);
            }

            // Send final chunk with usage info
            let final_chunk = StreamChunk {
                request_id: request_id.clone(),
                model: model_id.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: StreamDelta::empty(),
                    finish_reason: Some(FinishReason::Stop),
                }],
                usage: Some(Usage::new(prompt_token_count, generated_count)),
            };

            let _ = tx.blocking_send(Ok(final_chunk));
        });

        // Convert receiver to stream
        let stream = stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });

        Ok(TokenStream::new(stream))
    }

    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse> {
        use infernum_core::response::{Embedding, EmbeddingData};

        tracing::debug!(request_id = %request.request_id, "Starting embedding generation");

        let loaded = self
            .loaded
            .as_ref()
            .ok_or_else(|| infernum_core::Error::Internal {
                message: "Model not loaded".to_string(),
            })?;

        // Get input texts
        let texts = request.input.as_texts();
        let mut embeddings = Vec::with_capacity(texts.len());
        let mut total_tokens = 0u32;

        for (idx, text) in texts.iter().enumerate() {
            // Tokenize the input
            let tokens = loaded.tokenizer.encode(text, true)?;
            let token_count = tokens.len();
            total_tokens += token_count as u32;

            // Convert to tensor
            let input_ids = Tensor::new(tokens.as_slice(), &self.device)
                .map_err(|e| infernum_core::Error::Internal {
                    message: format!("Failed to create input tensor: {}", e),
                })?
                .unsqueeze(0)
                .map_err(|e| infernum_core::Error::Internal {
                    message: format!("Failed to unsqueeze tensor: {}", e),
                })?;

            // Extract embeddings
            let embedding_tensor = {
                let mut model = loaded.model.lock();
                model.extract_embeddings(&input_ids).map_err(|e| {
                    infernum_core::Error::Internal {
                        message: format!("Failed to extract embeddings: {}", e),
                    }
                })?
            };

            // Convert to vector
            let embedding_vec: Vec<f32> = embedding_tensor
                .squeeze(0)
                .map_err(|e| infernum_core::Error::Internal {
                    message: format!("Failed to squeeze embedding: {}", e),
                })?
                .to_vec1()
                .map_err(|e| infernum_core::Error::Internal {
                    message: format!("Failed to convert embedding to vector: {}", e),
                })?;

            // Apply dimension reduction if requested
            let final_embedding = if let Some(dims) = request.dimensions {
                let dims = dims as usize;
                if dims < embedding_vec.len() {
                    embedding_vec[..dims].to_vec()
                } else {
                    embedding_vec
                }
            } else {
                embedding_vec
            };

            embeddings.push(Embedding {
                index: idx as u32,
                embedding: EmbeddingData::Float(final_embedding),
            });

            tracing::debug!(
                request_id = %request.request_id,
                text_idx = idx,
                tokens = token_count,
                "Generated embedding"
            );
        }

        tracing::debug!(
            request_id = %request.request_id,
            num_embeddings = embeddings.len(),
            total_tokens = total_tokens,
            "Embedding generation complete"
        );

        Ok(EmbedResponse {
            request_id: request.request_id,
            model: self.metadata.id.clone(),
            data: embeddings,
            usage: infernum_core::Usage::new(total_tokens, 0),
        })
    }

    fn model_info(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn is_ready(&self) -> bool {
        self.loaded.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    // Note: These tests require model files to be present
    // They are disabled by default
    #[ignore]
    #[tokio::test]
    async fn test_engine_creation() {
        let config = EngineConfig::builder()
            .model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            .build()
            .unwrap();

        let engine = Engine::new(config).await.unwrap();
        assert!(engine.is_ready());
    }

    /// Mock engine for testing the batch interface.
    struct MockEngine {
        call_count: AtomicU32,
        metadata: ModelMetadata,
    }

    impl MockEngine {
        fn new() -> Self {
            use infernum_core::{ModelArchitecture, ModelSource};

            Self {
                call_count: AtomicU32::new(0),
                metadata: ModelMetadata::builder("mock-model", ModelArchitecture::Bert)
                    .source(ModelSource::LocalPath {
                        path: std::path::PathBuf::from("/mock"),
                    })
                    .context_length(2048)
                    .vocab_size(32000)
                    .hidden_size(768)
                    .num_layers(12)
                    .num_attention_heads(12)
                    .build(),
            }
        }

        fn call_count(&self) -> u32 {
            self.call_count.load(Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl InferenceEngine for MockEngine {
        async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
            self.call_count.fetch_add(1, Ordering::Relaxed);

            // Simulate some async work
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;

            Ok(GenerateResponse {
                request_id: request.request_id,
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64,
                model: self.metadata.id.clone(),
                choices: vec![infernum_core::response::Choice {
                    index: 0,
                    text: "Mock response".to_string(),
                    message: None,
                    finish_reason: Some(infernum_core::FinishReason::Stop),
                    logprobs: None,
                }],
                usage: infernum_core::Usage::new(10, 5),
                time_to_first_token_ms: Some(1.0),
                total_time_ms: Some(5.0),
            })
        }

        async fn generate_stream(&self, _request: GenerateRequest) -> Result<TokenStream> {
            unimplemented!("Mock doesn't support streaming")
        }

        async fn embed(&self, _request: EmbedRequest) -> Result<EmbedResponse> {
            unimplemented!("Mock doesn't support embedding")
        }

        fn model_info(&self) -> &ModelMetadata {
            &self.metadata
        }

        fn is_ready(&self) -> bool {
            true
        }
    }

    #[tokio::test]
    async fn test_generate_batch_default_implementation() {
        let engine = MockEngine::new();

        // Create multiple requests
        let requests: Vec<GenerateRequest> = (0..5)
            .map(|i| GenerateRequest::new(format!("Test prompt {}", i)))
            .collect();

        // Call batch generate (uses default trait implementation)
        let results = engine.generate_batch(requests).await;

        // All 5 requests should have been processed
        assert_eq!(results.len(), 5);
        assert_eq!(engine.call_count(), 5);

        // All results should be Ok
        for result in results {
            assert!(result.is_ok());
            let response = result.unwrap();
            assert_eq!(response.choices.len(), 1);
            assert_eq!(response.choices[0].text, "Mock response");
        }
    }

    #[tokio::test]
    async fn test_generate_batch_empty() {
        let engine = MockEngine::new();

        let results = engine.generate_batch(vec![]).await;

        assert!(results.is_empty());
        assert_eq!(engine.call_count(), 0);
    }

    #[tokio::test]
    async fn test_generate_batch_single_request() {
        let engine = MockEngine::new();

        let requests = vec![GenerateRequest::new("Single request")];
        let results = engine.generate_batch(requests).await;

        assert_eq!(results.len(), 1);
        assert_eq!(engine.call_count(), 1);
        assert!(results[0].is_ok());
    }
}
