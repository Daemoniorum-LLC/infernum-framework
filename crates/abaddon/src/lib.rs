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
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod adaptive_tiering;
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
pub mod kv_cache;
pub mod kv_cache_quant;
pub mod kv_cache_quant_cuda;
pub mod lazy_varbuilder;
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
pub mod gpu_dtype;
#[cfg(feature = "cuda")]
pub mod gpu_lrdf;

// llama.cpp backend for production inference (50-100x faster than Candle)
// Module is always available for BackendType; engine requires llama-cpp feature
pub mod llama_cpp_engine;

pub use arbiter_integration::{ArbiterCoordinator, ArbiterCoordinatorError, QualityLevel};
pub use config::{
    EngineConfig, EngineConfigBuilder, HoloTensorConfig, MemoryConfig, SpeculativeConfig,
};
pub use device::{best_device, cuda_available, enumerate_devices, print_devices, DeviceInfo};
pub use engine::{Engine, InferenceEngine, ShutdownResult, WarmupResult};
pub use gguf::{GgufLoader, GgufMetadata, QuantizedModelConfig};
pub use gpu_dequant::GpuDequantContext;
#[cfg(feature = "cuda")]
pub use gpu_dequant::GpuDequantError;
pub use gpu_dequant::INT4_BLOCK_SIZE;
pub use gpu_fused::GpuFusedContext;
#[cfg(feature = "cuda")]
pub use gpu_fused::GpuFusedError;
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::GpuHoloContext;
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::GpuHoloError;
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::HoloStreamPool;
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::KernelConfig;
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::MultiGpuHoloContext;
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::MultiGpuStats;
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::PinnedMemoryPool;
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::PinnedPoolStats;
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::ProgressiveHoloLoader;
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::StreamingHoloContext;
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::StreamingHoloStats;
pub use system_memory::{
    MemoryPressure as SystemMemoryPressure, RecommendedConfig, SystemMemoryInfo,
};
// Phase 7: Fault Tolerance
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::{
    FaultToleranceConfig, FaultToleranceStats, FaultTolerantDecoder, ValidationResult,
};
// Phase 7: Distributed Loading
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::{
    DistributedLoadConfig, DistributedLoadStats, DistributedLoader, FragmentSource,
    MemoryFragmentSource,
};
// Phase 7: Adaptive Quality
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::{
    AdaptiveQualityConfig, AdaptiveQualityController, AdaptiveQualityStats, LayerQualityTarget,
    QualityPolicy,
};
// Phase 7: Hot Reload
pub use flash_attention::{AttentionVariant, FlashAttention, FlashAttentionConfig};
#[cfg(feature = "cuda")]
pub use gpu_holo::cuda::{HotReloadController, HotReloadStats};
#[cfg(feature = "cuda")]
pub use gpu_lrdf::cuda::{GpuHoloFragment, GpuLrdfEncoder};
pub use gpu_lz4::cuda::{CudaStreamPool, StreamingLz4Context, StreamingStats};
pub use gpu_lz4::GpuLz4Context;
#[cfg(feature = "cuda")]
pub use gpu_lz4::GpuLz4Error;
#[cfg(feature = "cuda")]
pub use hct::ProgressiveLoadResult;
pub use hct::{
    filename_to_tensor_name, load_hct_directory, load_hct_directory_gpu,
    load_hct_directory_gpu_progressive, HctError, HctLoader, HctMetadata,
};
#[cfg(feature = "haagenti-gpu")]
pub use hct_sequential::{
    load_hct_directory_gpu as load_hct_directory_gpu_fast, load_hct_directory_gpu_with_stats,
    GpuDecompressStats,
};
pub use hct_sequential::{
    load_hct_directory_parallel, load_hct_directory_sequential,
    load_hct_directory_sequential_budgeted, FallbackStrategy, LoadProgress, LoadedTensor,
    MemoryBudget, SequentialHctLoader, SequentialLoadConfig,
};
pub use kv_cache::KVCache;
pub use lazy_varbuilder::{CacheConfig, DirectoryTensorProvider, LazyVarBuilder, TensorProvider};
pub use loader::ModelLoader;
pub use quantize::{
    ModelQuantizer,
    QuantizeConfig,
    QuantizeError,
    QuantizeFormat,
    QuantizeStats,
    QuantizedTensor,
    Quantizer,
    // Runtime quantization (on-the-fly weight quantization during model load)
    RuntimeQuantConfig,
    RuntimeQuantizedStore,
    RuntimeQuantizedWeight,
    DEFAULT_BLOCK_SIZE,
};
pub use sampler::Sampler;
pub use speculative::{SpeculativeDecoder, SpeculativeStats};
pub use tokenizer::Tokenizer;

// KV Cache Quantization (legacy)
pub use kv_cache_quant::{KvCacheQuantConfig, QuantizedKvCache};
#[cfg(feature = "cuda")]
pub use kv_cache_quant_cuda::cuda::{
    CudaQuantizedKvCache, Int8AttentionContext, Int8AttentionError,
};
pub use kv_cache_quant_cuda::{DynamicQuantConfig, OptimizedQuantizedKvCache, QuantGranularity};

// Model-agnostic attention cache system
#[cfg(feature = "cuda")]
pub use attention_cache::CudaQuantizedCache;
pub use attention_cache::{
    attention_with_cache, create_causal_mask, repeat_kv, AttentionConfig, CacheType, KvCache,
    KvCacheConfig, QuantizationGranularity, QuantizedCache, StandardCache,
};

// HoloTensor inference (progressive VRAM/RAM hybrid)
pub use holotensor::{
    ConversionConfig,
    ConversionProgress,
    FragmentLocation,
    HoloInferenceConfig,
    HoloInferenceError,
    HoloInferenceStats,
    HoloMemoryManager,
    HoloModelConverter,
    HoloModelMetadata,
    LayerWeightInfo,
    LayerWeights,
    MemoryConfig as HoloMemoryConfig,
    MemoryTier,
    PlacementDecision,
    ProgressiveWeightProvider,
    QualityMetrics,
    StreamManager,
    StreamPriority,
    StreamStats,
    // Tiered loading for 405B+ models
    TieredConfig,
    TieredHoloLoader,
    TieredStats,
};

// Adaptive memory tiering (intelligent VRAM/RAM/NVMe allocation)
pub use adaptive_tiering::{
    AdaptiveLoader,
    AdaptiveLoaderError,
    AdaptiveLoaderStats,
    AdaptiveTieringConfig,
    AllocationPlan,
    AllocationPlanner,
    EagerTensorProvider,
    ImportanceScorer,
    LoadingBackend,
    MemoryTier as AdaptiveMemoryTier, // Renamed to avoid conflict with holotensor::MemoryTier
    ModelProfile,
    ProfileError,
    ReconstructionPath,
    ReconstructionSummary,
    TensorAllocation,
    TensorInfo,
    TensorPrecision,
    TensorType,
};

// Re-exports from infernum-core
pub use infernum_core::{
    EmbedRequest, EmbedResponse, GenerateRequest, GenerateResponse, ModelArchitecture,
    ModelMetadata, ModelSource, SamplingParams, TokenStream,
};

// llama.cpp engine (production inference, 50-100x faster than Candle)
#[cfg(feature = "llama-cpp")]
pub use llama_cpp_engine::{ChatTemplate, LlamaCppConfig, LlamaCppConfigBuilder, LlamaCppEngine};

// Backend selection and GPU split mode (always available for CLI use)
pub use llama_cpp_engine::{BackendType, GpuSplitMode};
