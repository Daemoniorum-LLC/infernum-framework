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
#![allow(clippy::module_name_repetitions)]

pub mod backend;
pub mod config;
pub mod engine;
pub mod gguf;
pub mod kv_cache;
pub mod loader;
pub mod models;
pub mod sampler;
pub mod tokenizer;

pub use config::{EngineConfig, EngineConfigBuilder, MemoryConfig};
pub use engine::{Engine, InferenceEngine};
pub use gguf::{GgufLoader, GgufMetadata, QuantizedModelConfig};
pub use kv_cache::KVCache;
pub use loader::ModelLoader;
pub use sampler::Sampler;
pub use tokenizer::Tokenizer;

// Re-exports from infernum-core
pub use infernum_core::{
    EmbedRequest, EmbedResponse, GenerateRequest, GenerateResponse, ModelArchitecture,
    ModelMetadata, ModelSource, SamplingParams, TokenStream,
};
