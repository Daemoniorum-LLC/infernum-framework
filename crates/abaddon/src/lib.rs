//! # Abaddon
//!
//! *"The Destroyer renders judgment"*
//!
//! Abaddon is the core inference engine for the Infernum ecosystem.
//! It provides blazingly fast LLM inference with support for multiple backends
//! and advanced optimizations.
//!
//! ## Features
//!
//! - **Multi-Backend Support**: CUDA, Metal, WebGPU, and CPU backends
//! - **PagedAttention**: Efficient KV-cache memory management
//! - **FlashAttention**: Fused attention kernels for speedup
//! - **Continuous Batching**: Dynamic request batching
//! - **Speculative Decoding**: Draft model acceleration
//! - **In-Situ Quantization**: Runtime INT4/INT8 conversion
//!
//! ## Example
//!
//! ```ignore
//! use abaddon::{Engine, EngineConfig};
//! use infernum_core::{GenerateRequest, SamplingParams};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = EngineConfig::builder()
//!         .model("meta-llama/Llama-3.2-3B-Instruct")
//!         .device(DeviceType::Cuda { device_id: 0 })
//!         .build()?;
//!
//!     let engine = Engine::new(config).await?;
//!
//!     let request = GenerateRequest::new("Hello, world!")
//!         .with_sampling(SamplingParams::balanced().with_max_tokens(100));
//!
//!     let response = engine.generate(request).await?;
//!     println!("{}", response.choices[0].text);
//!
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![deny(clippy::unwrap_used)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod arbiter_integration;
pub mod attention_cache;
pub mod backend;
pub mod config;
pub mod device;
pub mod engine;
pub mod flash_attention;
pub mod gguf;
pub mod gpu_dequant;
pub mod gpu_fused;
pub mod gpu_holo;
pub mod gpu_lz4;
pub mod hct;
pub mod hct_sequential;
pub mod holotensor;
pub mod lazy_varbuilder;
pub mod kv_cache;
pub mod kv_cache_quant;
pub mod kv_cache_quant_cuda;
pub mod loader;
pub mod models;
pub mod quantize;
pub mod sampler;
pub mod speculative;
pub mod speculative_405b;
pub mod system_memory;
pub mod tokenizer;

#[cfg(feature = "cuda")]
pub mod cuda_inference;
#[cfg(feature = "cuda")]
pub mod cuda_svd;
#[cfg(feature = "cuda")]
pub mod gpu_dtype;
#[cfg(feature = "cuda")]
pub mod gpu_lrdf;

pub use arbiter_integration::{ArbiterCoordinator, ArbiterCoordinatorError, QualityLevel};
pub use config::{EngineConfig, EngineConfigBuilder, HoloTensorConfig, MemoryConfig, SpeculativeConfig};
pub use device::{DeviceInfo, best_device, cuda_available, enumerate_devices, print_devices};
pub use system_memory::{SystemMemoryInfo, RecommendedConfig, MemoryPressure as SystemMemoryPressure};
pub use engine::{Engine, InferenceEngine, ShutdownResult, WarmupResult};
pub use gguf::{GgufLoader, GgufMetadata, QuantizedModelConfig};
pub use gpu_dequant::GpuDequantContext;
#[cfg(feature = "cuda")]
pub use gpu_dequant::GpuDequantError;
#[cfg(feature = "cuda")]
pub use gpu_dequant::INT4_BLOCK_SIZE;
pub use gpu_fused::GpuFusedContext;
#[cfg(feature = "cuda")]
pub use gpu_fused::GpuFusedError;
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::GpuHoloContext;
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::StreamingHoloContext;
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::GpuHoloError;
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::ProgressiveHoloLoader;
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::KernelConfig;
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::HoloStreamPool;
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::StreamingHoloStats;
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::PinnedMemoryPool;
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::PinnedPoolStats;
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::MultiGpuHoloContext;
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::MultiGpuStats;
// Phase 7: Fault Tolerance
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::{ValidationResult, FaultToleranceConfig, FaultTolerantDecoder, FaultToleranceStats};
// Phase 7: Distributed Loading
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::{FragmentSource, DistributedLoadConfig, DistributedLoader, DistributedLoadStats, MemoryFragmentSource};
// Phase 7: Adaptive Quality
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::{QualityPolicy, AdaptiveQualityConfig, LayerQualityTarget, AdaptiveQualityController, AdaptiveQualityStats};
// Phase 7: Hot Reload
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::{HotReloadController, HotReloadStats};
pub use gpu_lz4::GpuLz4Context;
#[cfg(feature = "cuda")]
pub use gpu_lz4::GpuLz4Error;
pub use gpu_lz4::cuda::{CudaStreamPool, StreamingLz4Context, StreamingStats};
#[cfg(feature = "cuda")]
pub use gpu_lrdf::cuda::{GpuLrdfEncoder, GpuHoloFragment};
pub use hct::{HctLoader, HctMetadata, HctError, load_hct_directory, load_hct_directory_gpu, load_hct_directory_gpu_progressive, filename_to_tensor_name};
#[cfg(feature = "cuda")]
pub use hct::ProgressiveLoadResult;
pub use hct_sequential::{
    SequentialHctLoader, SequentialLoadConfig, MemoryBudget, FallbackStrategy,
    LoadedTensor, LoadProgress, load_hct_directory_sequential, load_hct_directory_sequential_budgeted,
    load_hct_directory_parallel,
};
#[cfg(feature = "haagenti-gpu")]
pub use hct_sequential::{
    load_hct_directory_gpu as load_hct_directory_gpu_fast,
    load_hct_directory_gpu_with_stats, GpuDecompressStats,
};
pub use lazy_varbuilder::{
    LazyVarBuilder, TensorProvider, DirectoryTensorProvider, CacheConfig,
};
pub use kv_cache::KVCache;
pub use loader::ModelLoader;
pub use sampler::Sampler;
pub use speculative::{SpeculativeDecoder, SpeculativeStats};
pub use tokenizer::Tokenizer;
pub use flash_attention::{FlashAttention, FlashAttentionConfig, AttentionVariant};
pub use quantize::{
    Quantizer, QuantizeConfig, QuantizeFormat, QuantizedTensor, QuantizeStats, QuantizeError,
    ModelQuantizer, DEFAULT_BLOCK_SIZE,
    // Runtime quantization (on-the-fly weight quantization during model load)
    RuntimeQuantConfig, RuntimeQuantizedWeight, RuntimeQuantizedStore,
};

// KV Cache Quantization (legacy)
pub use kv_cache_quant::{QuantizedKvCache, KvCacheQuantConfig};
pub use kv_cache_quant_cuda::{
    DynamicQuantConfig, OptimizedQuantizedKvCache, QuantGranularity,
};
#[cfg(feature = "cuda")]
pub use kv_cache_quant_cuda::cuda::{CudaQuantizedKvCache, Int8AttentionContext, Int8AttentionError};

// Model-agnostic attention cache system
pub use attention_cache::{
    KvCache, KvCacheConfig, CacheType, StandardCache, QuantizedCache, QuantizationGranularity,
    attention_with_cache, repeat_kv, create_causal_mask, AttentionConfig,
};
#[cfg(feature = "cuda")]
pub use attention_cache::CudaQuantizedCache;

// HoloTensor inference (progressive VRAM/RAM hybrid)
pub use holotensor::{
    HoloInferenceConfig, HoloInferenceError, HoloInferenceStats, HoloModelMetadata,
    HoloMemoryManager, MemoryConfig as HoloMemoryConfig, MemoryTier, FragmentLocation,
    StreamManager, StreamPriority, StreamStats,
    ProgressiveWeightProvider, LayerWeights, QualityMetrics,
    HoloModelConverter, ConversionConfig, ConversionProgress,
    // Tiered loading for 405B+ models
    TieredConfig, TieredHoloLoader, TieredStats, PlacementDecision, LayerWeightInfo,
};

// Re-exports from infernum-core
pub use infernum_core::{
    EmbedRequest, EmbedResponse, GenerateRequest, GenerateResponse, ModelArchitecture,
    ModelMetadata, ModelSource, SamplingParams, TokenStream,
};
