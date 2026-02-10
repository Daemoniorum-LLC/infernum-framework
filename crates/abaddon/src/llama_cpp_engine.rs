//! llama.cpp inference engine backend.
//!
//! This module provides a high-performance inference backend using llama.cpp
//! via the `llama_cpp` crate. It implements the `InferenceEngine` trait and
//! provides 50-100x speedup over Candle for GGUF models.
//!
//! # Usage
//!
//! ```ignore
//! use abaddon::llama_cpp_engine::{LlamaCppEngine, LlamaCppConfig};
//!
//! let config = LlamaCppConfig::builder()
//!     .model_path("/path/to/model.gguf")
//!     .n_gpu_layers(-1)  // All layers on GPU
//!     .build()?;
//!
//! let engine = LlamaCppEngine::load(config).await?;
//! let response = engine.generate(request).await?;
//! ```
//!
//! # Performance
//!
//! On RTX 4090 with Qwen2.5-7B:
//! - Candle backend: ~0.5 tk/s (4-8% GPU utilization)
//! - llama.cpp backend: 30-50 tk/s (80-95% GPU utilization)
//!
//! # Sampling Parameters
//!
//! The llama.cpp backend supports the following sampling parameters:
//! - `temperature` - Controls randomness (0.0 = greedy, higher = more random)
//! - `top_p` - Nucleus sampling threshold
//! - `top_k` - Keep only top k tokens
//! - `min_p` - Minimum probability threshold
//! - `repetition_penalty` - Penalize repeated tokens
//! - `frequency_penalty` - Penalize frequent tokens
//! - `presence_penalty` - Penalize tokens that have appeared
//! - `stop_sequences` - Strings that halt generation
//!
//! **Note:** `seed` is not currently supported by the llama_cpp crate's sampler API.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use tokio::sync::mpsc;

use infernum_core::model::{LlamaVersion, MistralVariant, PhiVersion, QwenVersion};
use infernum_core::response::{Choice, EmbedResponse, GenerateResponse};
use infernum_core::streaming::{StreamChoice, StreamChunk, StreamDelta};
use infernum_core::{
    EmbedRequest, FinishReason, GenerateRequest, Message, ModelArchitecture, ModelId,
    ModelMetadata, ModelSource, PromptInput, QuantizationType, Result, Role, SamplingParams,
    TokenStream, Usage,
};

use crate::engine::InferenceEngine;
use crate::gguf::GgufLoader;

/// Available inference backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendType {
    /// Automatically select backend based on model format.
    /// - `.gguf` files → LlamaCpp
    /// - HCT directories → Candle
    /// - Safetensors → LlamaCpp (for better performance)
    #[default]
    Auto,

    /// llama.cpp backend via llama-cpp-rs (production, high performance).
    /// Best for GGUF models, achieves 30-60 tk/s on 7B models.
    LlamaCpp,

    /// Candle native Rust backend (research, custom architectures).
    /// Best for HCT/HoloTensor models and experimental features.
    Candle,
}

impl BackendType {
    /// Parse backend type from string (for CLI).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "auto" => Some(BackendType::Auto),
            "llama-cpp" | "llamacpp" | "llama_cpp" | "gguf" => Some(BackendType::LlamaCpp),
            "candle" | "hct" | "holotensor" => Some(BackendType::Candle),
            _ => None,
        }
    }

    /// Alias for parse() to maintain compatibility.
    pub fn from_str(s: &str) -> Option<Self> {
        Self::parse(s)
    }

    /// Detect backend from model path.
    pub fn detect_from_path(path: &std::path::Path) -> Self {
        // Check file extension
        if let Some(ext) = path.extension() {
            let ext_lower = ext.to_string_lossy().to_lowercase();
            if ext_lower == "gguf" {
                return BackendType::LlamaCpp;
            }
        }

        // Check if it's an HCT directory
        if path.is_dir() {
            // Look for HCT marker files
            if path.join("metadata.json").exists() || path.join("layer_0").exists() {
                return BackendType::Candle;
            }
        }

        // Check for safetensors (prefer llama.cpp for speed)
        if path.is_dir() {
            if path.join("model.safetensors").exists()
                || path.join("model.safetensors.index.json").exists()
            {
                // Safetensors can be used with either backend
                // Default to LlamaCpp for better performance if available
                #[cfg(feature = "llama-cpp")]
                return BackendType::LlamaCpp;
                #[cfg(not(feature = "llama-cpp"))]
                return BackendType::Candle;
            }
        }

        // Default to Candle for unknown formats
        BackendType::Candle
    }

    /// Get human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            BackendType::Auto => "auto",
            BackendType::LlamaCpp => "llama-cpp",
            BackendType::Candle => "candle",
        }
    }
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for BackendType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        BackendType::parse(s).ok_or_else(|| {
            format!(
                "Invalid backend type '{}'. Valid options: auto, llama-cpp, candle",
                s
            )
        })
    }
}

/// Chat template format for different model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatTemplate {
    /// ChatML format used by Qwen, Yi, and others.
    /// Format: `<|im_start|>role\ncontent<|im_end|>`
    #[default]
    ChatML,

    /// Llama 3.x format.
    /// Format: `<|start_header_id|>role<|end_header_id|>\ncontent<|eot_id|>`
    Llama3,

    /// Mistral/Mixtral instruction format.
    /// Format: `[INST] content [/INST]`
    Mistral,

    /// Phi-3 format (similar to ChatML with slight variations).
    Phi3,

    /// Raw format - no special tokens, just concatenate content.
    Raw,
}

impl ChatTemplate {
    /// Detect the appropriate chat template from model architecture string.
    pub fn from_architecture(arch: &str) -> Self {
        let lower = arch.to_lowercase();
        if lower.contains("llama") {
            // Llama 3.x uses different template than Llama 2
            ChatTemplate::Llama3
        } else if lower.contains("qwen") || lower.contains("yi") {
            ChatTemplate::ChatML
        } else if lower.contains("mistral") || lower.contains("mixtral") {
            ChatTemplate::Mistral
        } else if lower.contains("phi") {
            ChatTemplate::Phi3
        } else {
            // Default to ChatML as it's widely supported
            ChatTemplate::ChatML
        }
    }

    /// Format a system message.
    pub fn format_system(&self, content: &str) -> String {
        match self {
            ChatTemplate::ChatML => {
                format!("<|im_start|>system\n{content}<|im_end|>\n")
            },
            ChatTemplate::Llama3 => {
                format!("<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>")
            },
            ChatTemplate::Mistral => {
                // Mistral doesn't have a dedicated system token, prepend to first user message
                String::new()
            },
            ChatTemplate::Phi3 => {
                format!("<|system|>\n{content}<|end|>\n")
            },
            ChatTemplate::Raw => {
                format!("System: {content}\n\n")
            },
        }
    }

    /// Format a user message.
    pub fn format_user(&self, content: &str) -> String {
        match self {
            ChatTemplate::ChatML => {
                format!("<|im_start|>user\n{content}<|im_end|>\n")
            },
            ChatTemplate::Llama3 => {
                format!("<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>")
            },
            ChatTemplate::Mistral => {
                format!("[INST] {content} [/INST]")
            },
            ChatTemplate::Phi3 => {
                format!("<|user|>\n{content}<|end|>\n")
            },
            ChatTemplate::Raw => {
                format!("User: {content}\n\n")
            },
        }
    }

    /// Format an assistant message.
    pub fn format_assistant(&self, content: &str) -> String {
        match self {
            ChatTemplate::ChatML => {
                format!("<|im_start|>assistant\n{content}<|im_end|>\n")
            },
            ChatTemplate::Llama3 => {
                format!("<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>")
            },
            ChatTemplate::Mistral => {
                format!("{content}</s>")
            },
            ChatTemplate::Phi3 => {
                format!("<|assistant|>\n{content}<|end|>\n")
            },
            ChatTemplate::Raw => {
                format!("Assistant: {content}\n\n")
            },
        }
    }

    /// Format the assistant generation prefix (added at the end to prompt generation).
    pub fn assistant_prefix(&self) -> &'static str {
        match self {
            ChatTemplate::ChatML => "<|im_start|>assistant\n",
            ChatTemplate::Llama3 => "<|start_header_id|>assistant<|end_header_id|>\n\n",
            ChatTemplate::Mistral => "",
            ChatTemplate::Phi3 => "<|assistant|>\n",
            ChatTemplate::Raw => "Assistant: ",
        }
    }

    /// Get the beginning-of-sequence token if applicable.
    pub fn bos_token(&self) -> &'static str {
        match self {
            ChatTemplate::Llama3 => "<|begin_of_text|>",
            _ => "",
        }
    }
}

/// GPU split mode for multi-GPU inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuSplitMode {
    /// Single GPU only (default).
    #[default]
    None,
    /// Split layers and KV cache across GPUs.
    /// Good for models that fit in combined VRAM.
    Layer,
    /// Split rows across GPUs (tensor parallelism).
    /// Better for very large models.
    Row,
}

#[cfg(feature = "llama-cpp")]
impl From<GpuSplitMode> for llama_cpp::SplitMode {
    fn from(mode: GpuSplitMode) -> Self {
        match mode {
            GpuSplitMode::None => llama_cpp::SplitMode::None,
            GpuSplitMode::Layer => llama_cpp::SplitMode::Layer,
            GpuSplitMode::Row => llama_cpp::SplitMode::Row,
        }
    }
}

/// Configuration for the llama.cpp engine.
#[derive(Debug, Clone)]
pub struct LlamaCppConfig {
    /// Path to GGUF model file.
    pub model_path: PathBuf,

    /// Number of GPU layers to offload.
    /// - `-1` = all layers (default)
    /// - `0` = CPU only
    /// - `n` = first n layers on GPU
    pub n_gpu_layers: i32,

    /// Context window size. `0` uses model default.
    pub context_size: usize,

    /// Batch size for prompt processing.
    pub batch_size: usize,

    /// Number of threads for CPU operations.
    pub n_threads: usize,

    /// Enable memory mapping for model loading.
    pub use_mmap: bool,

    /// Enable memory locking to prevent swapping.
    pub use_mlock: bool,

    /// Main GPU device ID for multi-GPU setups.
    pub main_gpu: i32,

    /// GPU split mode for multi-GPU inference.
    ///
    /// - `None`: Single GPU (default)
    /// - `Layer`: Split layers and KV cache across GPUs
    /// - `Row`: Split rows across GPUs (tensor parallelism)
    pub split_mode: GpuSplitMode,

    /// Flash attention is enabled by default in llama.cpp when supported.
    /// This option is provided for documentation purposes; llama.cpp manages
    /// flash attention automatically based on GPU capabilities.
    ///
    /// Note: The underlying llama_cpp crate doesn't expose a flash_attention toggle.
    /// llama.cpp enables it automatically when beneficial.
    _flash_attention_note: (),
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            n_gpu_layers: -1,
            context_size: 0,
            batch_size: 512,
            n_threads: num_cpus::get(),
            use_mmap: true,
            use_mlock: false,
            main_gpu: 0,
            split_mode: GpuSplitMode::default(),
            _flash_attention_note: (),
        }
    }
}

impl LlamaCppConfig {
    /// Create a new builder for `LlamaCppConfig`.
    pub fn builder() -> LlamaCppConfigBuilder {
        LlamaCppConfigBuilder::default()
    }
}

/// Builder for `LlamaCppConfig`.
#[derive(Debug, Default)]
pub struct LlamaCppConfigBuilder {
    config: LlamaCppConfig,
}

impl LlamaCppConfigBuilder {
    /// Set the model path.
    pub fn model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.model_path = path.into();
        self
    }

    /// Set the number of GPU layers.
    pub fn n_gpu_layers(mut self, n: i32) -> Self {
        self.config.n_gpu_layers = n;
        self
    }

    /// Set the context size.
    pub fn context_size(mut self, size: usize) -> Self {
        self.config.context_size = size;
        self
    }

    /// Set the batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    /// Set the number of threads.
    pub fn n_threads(mut self, n: usize) -> Self {
        self.config.n_threads = n;
        self
    }

    /// Enable or disable memory mapping.
    pub fn use_mmap(mut self, use_mmap: bool) -> Self {
        self.config.use_mmap = use_mmap;
        self
    }

    /// Enable or disable memory locking.
    pub fn use_mlock(mut self, use_mlock: bool) -> Self {
        self.config.use_mlock = use_mlock;
        self
    }

    /// Set the main GPU device ID.
    pub fn main_gpu(mut self, id: i32) -> Self {
        self.config.main_gpu = id;
        self
    }

    /// Set the GPU split mode for multi-GPU inference.
    ///
    /// - `GpuSplitMode::None`: Single GPU (default)
    /// - `GpuSplitMode::Layer`: Split layers and KV cache across GPUs
    /// - `GpuSplitMode::Row`: Split rows across GPUs (tensor parallelism)
    pub fn split_mode(mut self, mode: GpuSplitMode) -> Self {
        self.config.split_mode = mode;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> Result<LlamaCppConfig> {
        if self.config.model_path.as_os_str().is_empty() {
            return Err(infernum_core::Error::InvalidConfig {
                message: "model_path is required".to_string(),
            });
        }
        Ok(self.config)
    }
}

/// llama.cpp inference engine.
///
/// This engine provides high-performance inference using llama.cpp's optimized
/// CUDA kernels. It implements `InferenceEngine` and can be used as a drop-in
/// replacement for the Candle-based `Engine`.
#[cfg(feature = "llama-cpp")]
pub struct LlamaCppEngine {
    /// Shared model handle.
    model: Arc<llama_cpp::LlamaModel>,

    /// Configuration.
    config: LlamaCppConfig,

    /// Model metadata.
    metadata: ModelMetadata,

    /// Model ID for responses.
    model_id: ModelId,

    /// Chat template for formatting messages.
    chat_template: ChatTemplate,

    /// Whether the engine is ready.
    ready: bool,
}

#[cfg(feature = "llama-cpp")]
impl LlamaCppEngine {
    /// Load a GGUF model.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded.
    pub async fn load(config: LlamaCppConfig) -> Result<Self> {
        let model_path = config.model_path.clone();
        let n_gpu_layers = config.n_gpu_layers;
        let use_mmap = config.use_mmap;
        let use_mlock = config.use_mlock;
        let main_gpu = config.main_gpu;
        let split_mode = config.split_mode;

        tracing::info!(
            path = %model_path.display(),
            n_gpu_layers = n_gpu_layers,
            split_mode = ?split_mode,
            "Loading GGUF model with llama.cpp"
        );

        let start = Instant::now();

        // Load model on blocking thread (can take seconds for large models)
        let model = tokio::task::spawn_blocking(move || {
            use llama_cpp::LlamaModel;

            // LlamaParams has public fields, not builder methods
            let mut params = llama_cpp::LlamaParams::default();
            // Use 999 as "all layers" since u32::MAX causes issues with some CUDA backends
            params.n_gpu_layers = if n_gpu_layers < 0 {
                999
            } else {
                n_gpu_layers as u32
            };
            params.use_mmap = use_mmap;
            params.use_mlock = use_mlock;
            params.main_gpu = main_gpu as u32;
            params.split_mode = split_mode.into();

            LlamaModel::load_from_file(&model_path, params).map_err(|e| {
                infernum_core::Error::ModelLoad {
                    message: format!("llama.cpp model load failed: {e}"),
                }
            })
        })
        .await
        .map_err(|e| infernum_core::Error::ModelLoad {
            message: format!("Task join error: {e}"),
        })??;

        let load_time = start.elapsed();
        tracing::info!(
            load_time_ms = load_time.as_millis(),
            "llama.cpp model loaded"
        );

        // Extract metadata using existing GgufLoader
        let gguf = GgufLoader::from_file(&config.model_path)?;
        let gguf_meta = gguf.metadata();

        // Map architecture string to enum
        let architecture = match gguf_meta.architecture.to_lowercase().as_str() {
            "qwen2" | "qwen" => ModelArchitecture::Qwen {
                version: QwenVersion::V2_5,
            },
            "llama" => ModelArchitecture::Llama {
                version: LlamaVersion::V3_2,
            },
            "mistral" => ModelArchitecture::Mistral {
                variant: MistralVariant::Mistral7B,
            },
            "phi" | "phi3" => ModelArchitecture::Phi {
                version: PhiVersion::V3,
            },
            _ => ModelArchitecture::Llama {
                version: LlamaVersion::V3_2,
            },
        };

        // Map quantization string to enum
        let quantization = parse_quantization_type(&gguf_meta.quantization_type);

        let model_id = ModelId(
            gguf_meta
                .name
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
        );

        let metadata = ModelMetadata::builder(model_id.clone(), architecture)
            .source(ModelSource::Gguf {
                path: config.model_path.clone(),
            })
            .context_length(gguf_meta.context_length as u32)
            .vocab_size(gguf_meta.vocab_size as u32)
            .hidden_size(gguf_meta.hidden_size as u32)
            .num_layers(gguf_meta.num_layers as u32)
            .num_attention_heads(gguf_meta.num_attention_heads as u32)
            .num_kv_heads(gguf_meta.num_kv_heads as u32)
            .quantization(quantization)
            .build();

        // Detect chat template from architecture
        let chat_template = ChatTemplate::from_architecture(&gguf_meta.architecture);

        tracing::info!(
            model = %metadata.id.0,
            architecture = ?metadata.architecture,
            context_length = metadata.context_length,
            quantization = ?metadata.quantization,
            chat_template = ?chat_template,
            "Model metadata extracted"
        );

        Ok(Self {
            model: Arc::new(model),
            config,
            metadata,
            model_id,
            chat_template,
            ready: true,
        })
    }

    /// Format prompt from PromptInput.
    ///
    /// # Errors
    ///
    /// Returns an error if the input is pre-tokenized (PromptInput::Tokens),
    /// which is not supported by the llama.cpp backend.
    fn format_prompt(&self, prompt: &PromptInput) -> Result<String> {
        match prompt {
            PromptInput::Text(text) => {
                // Wrap raw text in chat template format for instruction-following models
                // This ensures proper tokenization and generation behavior
                let messages = vec![Message {
                    role: Role::User,
                    content: text.clone(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                }];
                Ok(Self::format_chat_messages(&messages, self.chat_template))
            },
            PromptInput::Messages(messages) => {
                Ok(Self::format_chat_messages(messages, self.chat_template))
            },
            PromptInput::Tokens(_) => Err(infernum_core::Error::InvalidConfig {
                message: "Pre-tokenized input (PromptInput::Tokens) is not supported by the \
                             llama.cpp backend. Use PromptInput::Text or PromptInput::Messages."
                    .to_string(),
            }),
        }
    }

    /// Format chat messages into a prompt string using the specified template.
    pub fn format_chat_messages(messages: &[Message], template: ChatTemplate) -> String {
        let mut prompt = String::new();

        // Add BOS token if applicable
        prompt.push_str(template.bos_token());

        // For Mistral, we need to handle system messages specially
        let mut system_content: Option<String> = None;

        for msg in messages {
            match msg.role {
                Role::System => {
                    if template == ChatTemplate::Mistral {
                        // Store system message to prepend to first user message
                        system_content = Some(msg.content.clone());
                    } else {
                        prompt.push_str(&template.format_system(&msg.content));
                    }
                },
                Role::User => {
                    // For Mistral, prepend system content to first user message
                    let content = if let Some(sys) = system_content.take() {
                        format!("{}\n\n{}", sys, msg.content)
                    } else {
                        msg.content.clone()
                    };
                    prompt.push_str(&template.format_user(&content));
                },
                Role::Assistant => {
                    prompt.push_str(&template.format_assistant(&msg.content));
                },
                Role::Tool => {
                    // Tool results formatted as user messages with tool context
                    let content = format!("[Tool Result]\n{}", msg.content);
                    prompt.push_str(&template.format_user(&content));
                },
            }
        }

        // Add assistant prefix for generation
        prompt.push_str(template.assistant_prefix());
        prompt
    }

    /// Get the current chat template.
    pub fn chat_template(&self) -> ChatTemplate {
        self.chat_template
    }
}

#[cfg(feature = "llama-cpp")]
#[async_trait]
impl InferenceEngine for LlamaCppEngine {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        let model = self.model.clone();
        let context_size = if self.config.context_size > 0 {
            self.config.context_size
        } else {
            4096
        };
        let sampling = request.sampling.clone();
        let model_id = self.model_id.clone();
        let request_id = request.request_id.clone();

        // Format prompt before entering spawn_blocking (needs &self)
        let prompt = self.format_prompt(&request.prompt)?;
        tracing::debug!(prompt = %prompt, "Formatted prompt for llama.cpp");

        // Run blocking llama.cpp inference on thread pool
        tokio::task::spawn_blocking(move || {
            use llama_cpp::SessionParams;

            // SessionParams has public fields
            let mut session_params = SessionParams::default();
            session_params.n_ctx = context_size as u32;

            let mut session = model.create_session(session_params).map_err(|e| {
                infernum_core::Error::Internal {
                    message: format!("Failed to create session: {e}"),
                }
            })?;

            // Process prompt
            session
                .advance_context(&prompt)
                .map_err(|e| infernum_core::Error::Internal {
                    message: format!("Failed to process prompt: {e}"),
                })?;

            // Generate tokens
            let mut output = String::new();
            let max_tokens = sampling.max_tokens as usize;
            let mut tokens_generated = 0usize;
            let start = Instant::now();
            let mut time_to_first_token: Option<std::time::Duration> = None;
            let mut hit_stop_sequence = false;

            // Build sampler with user's sampling parameters
            let sampler = build_sampler(&sampling);
            let completions = session
                .start_completing_with(sampler, max_tokens)
                .map_err(|e| infernum_core::Error::Internal {
                    message: format!("Failed to start completion: {e}"),
                })?;

            for token in completions.into_strings() {
                if time_to_first_token.is_none() {
                    time_to_first_token = Some(start.elapsed());
                }
                output.push_str(&token);
                tokens_generated += 1;

                // Check for stop sequences
                if !sampling.stop_sequences.is_empty() {
                    if check_stop_sequences(&output, &sampling.stop_sequences).is_some() {
                        hit_stop_sequence = true;
                        trim_stop_sequence(&mut output, &sampling.stop_sequences);
                        break;
                    }
                }
            }

            let elapsed = start.elapsed();
            let tokens_per_sec = if elapsed.as_secs_f64() > 0.0 {
                tokens_generated as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            };

            tracing::debug!(
                tokens = tokens_generated,
                elapsed_ms = elapsed.as_millis(),
                tokens_per_sec = format!("{tokens_per_sec:.2}"),
                ttft_ms = time_to_first_token.map(|d| d.as_millis()),
                hit_stop = hit_stop_sequence,
                "Generation complete"
            );

            let finish_reason = if hit_stop_sequence {
                FinishReason::Stop
            } else if tokens_generated >= max_tokens {
                FinishReason::Length
            } else {
                FinishReason::Stop
            };

            let created = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);

            // Use model tokenizer for accurate prompt token count
            let prompt_tokens = model
                .tokenize_bytes(prompt.as_bytes(), false, false)
                .map(|tokens| tokens.len() as u32)
                .unwrap_or_else(|_| (prompt.len() / 4) as u32); // Fallback to heuristic
            let completion_tokens = tokens_generated as u32;

            let response = GenerateResponse {
                request_id,
                created,
                model: model_id,
                choices: vec![Choice {
                    index: 0,
                    text: output,
                    message: None,
                    finish_reason: Some(finish_reason),
                    logprobs: None,
                }],
                usage: Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                },
                time_to_first_token_ms: time_to_first_token.map(|d| d.as_secs_f64() * 1000.0),
                total_time_ms: Some(elapsed.as_millis() as f64),
            };

            Ok(response)
        })
        .await
        .map_err(|e| infernum_core::Error::Internal {
            message: format!("Task join error: {e}"),
        })?
    }

    async fn generate_stream(&self, request: GenerateRequest) -> Result<TokenStream> {
        let (tx, rx) = mpsc::channel::<Result<StreamChunk>>(32);
        let model = self.model.clone();
        let context_size = if self.config.context_size > 0 {
            self.config.context_size
        } else {
            4096
        };
        let sampling = request.sampling.clone();
        let model_id = self.model_id.clone();
        let request_id = request.request_id.clone();

        // Format prompt before entering spawn_blocking (needs &self)
        let prompt = self.format_prompt(&request.prompt)?;

        // Spawn blocking task that sends tokens through channel
        tokio::task::spawn_blocking(move || {
            use llama_cpp::SessionParams;

            let mut session_params = SessionParams::default();
            session_params.n_ctx = context_size as u32;

            let result = (|| -> Result<()> {
                let mut session = model.create_session(session_params).map_err(|e| {
                    infernum_core::Error::Internal {
                        message: format!("Failed to create session: {e}"),
                    }
                })?;

                session
                    .advance_context(&prompt)
                    .map_err(|e| infernum_core::Error::Internal {
                        message: format!("Failed to process prompt: {e}"),
                    })?;

                let max_tokens = sampling.max_tokens as usize;
                let stop_sequences = sampling.stop_sequences.clone();

                // Build sampler with user's sampling parameters
                let sampler = build_sampler(&sampling);
                let completions =
                    session
                        .start_completing_with(sampler, max_tokens)
                        .map_err(|e| infernum_core::Error::Internal {
                            message: format!("Failed to start completion: {e}"),
                        })?;

                // Track token count and accumulated text for stop sequence detection
                let mut tokens_generated = 0u32;
                // Use model tokenizer for accurate prompt token count
                let prompt_tokens = model
                    .tokenize_bytes(prompt.as_bytes(), false, false)
                    .map(|tokens| tokens.len() as u32)
                    .unwrap_or_else(|_| (prompt.len() / 4) as u32); // Fallback to heuristic
                let mut accumulated_text = String::new();
                let mut hit_stop_sequence = false;

                for token_text in completions.into_strings() {
                    tokens_generated += 1;
                    accumulated_text.push_str(&token_text);

                    // Check for stop sequences
                    if !stop_sequences.is_empty() {
                        if let Some(stop) = check_stop_sequences(&accumulated_text, &stop_sequences)
                        {
                            hit_stop_sequence = true;
                            // Send the text before the stop sequence
                            let text_before_stop =
                                token_text.strip_suffix(stop).unwrap_or(&token_text);
                            if !text_before_stop.is_empty() {
                                let chunk = StreamChunk {
                                    request_id: request_id.clone(),
                                    model: model_id.clone(),
                                    choices: vec![StreamChoice {
                                        index: 0,
                                        delta: StreamDelta::text(text_before_stop.to_string()),
                                        finish_reason: None,
                                    }],
                                    usage: None,
                                };
                                let _ = tx.blocking_send(Ok(chunk));
                            }
                            break;
                        }
                    }

                    let chunk = StreamChunk {
                        request_id: request_id.clone(),
                        model: model_id.clone(),
                        choices: vec![StreamChoice {
                            index: 0,
                            delta: StreamDelta::text(token_text),
                            finish_reason: None,
                        }],
                        usage: None,
                    };
                    if tx.blocking_send(Ok(chunk)).is_err() {
                        break; // Receiver dropped
                    }
                }

                // Determine finish reason
                let finish_reason = if hit_stop_sequence {
                    FinishReason::Stop
                } else if tokens_generated as usize >= max_tokens {
                    FinishReason::Length
                } else {
                    FinishReason::Stop
                };

                // Send final chunk with finish reason and usage stats
                let final_chunk = StreamChunk {
                    request_id: request_id.clone(),
                    model: model_id.clone(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: StreamDelta::empty(),
                        finish_reason: Some(finish_reason),
                    }],
                    usage: Some(Usage {
                        prompt_tokens,
                        completion_tokens: tokens_generated,
                        total_tokens: prompt_tokens + tokens_generated,
                    }),
                };
                let _ = tx.blocking_send(Ok(final_chunk));

                Ok(())
            })();

            if let Err(e) = result {
                tracing::error!("Streaming generation failed: {e}");
            }
        });

        // Convert receiver to stream
        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(TokenStream::new(stream))
    }

    async fn generate_batch(
        &self,
        requests: Vec<GenerateRequest>,
    ) -> Vec<Result<GenerateResponse>> {
        use futures::future::join_all;

        let batch_size = requests.len();
        if batch_size == 0 {
            return Vec::new();
        }

        tracing::debug!(
            batch_size = batch_size,
            "Processing batch with concurrent sessions (llama.cpp doesn't expose true GPU batching)"
        );

        // llama.cpp sessions run on worker threads, so concurrent execution is efficient.
        // True continuous batching would require lower-level llama_cpp_sys access.
        // Each request gets its own session, which llama.cpp handles efficiently
        // by using separate worker threads for token generation.
        let start = Instant::now();

        let results = join_all(requests.into_iter().map(|req| self.generate(req))).await;

        let elapsed = start.elapsed();
        let success_count = results.iter().filter(|r| r.is_ok()).count();
        let total_tokens: u32 = results
            .iter()
            .filter_map(|r| r.as_ref().ok())
            .map(|resp| resp.usage.completion_tokens)
            .sum();

        tracing::info!(
            batch_size = batch_size,
            success_count = success_count,
            total_tokens = total_tokens,
            elapsed_ms = elapsed.as_millis(),
            tokens_per_sec = format!("{:.2}", total_tokens as f64 / elapsed.as_secs_f64()),
            "Batch inference complete"
        );

        results
    }

    async fn embed(&self, _request: EmbedRequest) -> Result<EmbedResponse> {
        // Note: llama_cpp embedding support varies by model and version.
        // For now, return an error indicating this is not fully implemented.
        // Full implementation would use session.get_embeddings() after advance_context.
        //
        // TODO(#TBD): Implement proper embedding support once llama_cpp API is verified
        Err(infernum_core::Error::Internal {
            message: "Embedding support for llama.cpp backend is not yet implemented. \
                     Use the Candle backend for embeddings."
                .to_string(),
        })
    }

    fn model_info(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn is_ready(&self) -> bool {
        self.ready
    }
}

/// Parse GGUF quantization string to QuantizationType enum.
fn parse_quantization_type(quant_str: &str) -> QuantizationType {
    let lower = quant_str.to_lowercase();
    if lower.contains("q4_k_m") || lower.contains("q4km") {
        QuantizationType::GgufQ4KM
    } else if lower.contains("q5_k_m") || lower.contains("q5km") {
        QuantizationType::GgufQ5KM
    } else if lower.contains("q4_0") || lower.contains("q4.0") {
        QuantizationType::GgufQ4_0
    } else if lower.contains("q8_0") || lower.contains("q8.0") {
        QuantizationType::GgufQ8_0
    } else if lower.contains("f16") || lower.contains("fp16") {
        QuantizationType::None
    } else {
        // Default to Q4_K_M as it's most common
        QuantizationType::GgufQ4KM
    }
}

/// Build a llama.cpp sampler from SamplingParams.
///
/// Converts Infernum's sampling parameters to llama_cpp's SamplerStage pipeline.
/// The stages are applied in order: grammar → repetition penalty → temperature → top_k → top_p → min_p.
#[cfg(feature = "llama-cpp")]
fn build_sampler(params: &SamplingParams) -> llama_cpp::standard_sampler::StandardSampler {
    use llama_cpp::grammar::LlamaGrammar;
    use llama_cpp::standard_sampler::{SamplerStage, StandardSampler};

    // Warn about unsupported seed parameter
    if params.seed.is_some() {
        tracing::warn!(
            "seed parameter is not supported by llama.cpp sampler API - output will not be reproducible"
        );
    }

    // Greedy sampling when temperature is 0 (but still apply grammar if present)
    if params.temperature == 0.0 && params.grammar.is_none() {
        return StandardSampler::new_greedy();
    }

    // Build sampler stages in recommended order
    let mut stages = Vec::new();

    // 0. Grammar constraint (applied first to filter invalid tokens)
    if let Some(grammar_constraint) = &params.grammar {
        let gbnf = grammar_constraint.to_gbnf();
        match gbnf.parse::<LlamaGrammar>() {
            Ok(grammar) => {
                // Grammar starts at the end of context (None = current position)
                stages.push(SamplerStage::from_grammar(grammar, None));
                tracing::debug!("Grammar constraint enabled");
            },
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "Failed to parse grammar constraint, continuing without it"
                );
            },
        }
    }

    // For greedy with grammar, still return early but with grammar applied
    if params.temperature == 0.0 {
        // Create a minimal sampler that just does greedy selection after grammar filtering
        stages.push(SamplerStage::Temperature(0.0));
        return StandardSampler::new_softmax(stages, 1);
    }

    // 1. Repetition penalty (applied first to raw logits, after grammar)
    if params.repetition_penalty != 1.0
        || params.frequency_penalty != 0.0
        || params.presence_penalty != 0.0
    {
        stages.push(SamplerStage::RepetitionPenalty {
            repetition_penalty: params.repetition_penalty,
            frequency_penalty: params.frequency_penalty,
            presence_penalty: params.presence_penalty,
            last_n: 64, // Context window for penalty
        });
    }

    // 2. Temperature (scales logits)
    if params.temperature != 1.0 {
        stages.push(SamplerStage::Temperature(params.temperature));
    }

    // 3. Top-K (reduces candidate tokens)
    if params.top_k > 0 {
        stages.push(SamplerStage::TopK(params.top_k as i32));
    }

    // 4. Top-P / nucleus sampling
    if params.top_p < 1.0 {
        stages.push(SamplerStage::TopP(params.top_p));
    }

    // 5. Min-P (relative probability threshold)
    if params.min_p > 0.0 {
        stages.push(SamplerStage::MinP(params.min_p));
    }

    // min_keep ensures at least 1 token survives filtering
    StandardSampler::new_softmax(stages, 1)
}

/// Check if the generated text ends with any of the stop sequences.
///
/// Returns the matching stop sequence if found, or None.
fn check_stop_sequences<'a>(text: &str, stop_sequences: &'a [String]) -> Option<&'a str> {
    for stop in stop_sequences {
        if text.ends_with(stop) {
            return Some(stop);
        }
    }
    None
}

/// Trim stop sequence from the end of text if present.
fn trim_stop_sequence(text: &mut String, stop_sequences: &[String]) -> bool {
    for stop in stop_sequences {
        if text.ends_with(stop) {
            text.truncate(text.len() - stop.len());
            return true;
        }
    }
    false
}

/// Stub implementation when llama-cpp feature is disabled.
#[cfg(not(feature = "llama-cpp"))]
pub struct LlamaCppEngine {
    _private: (),
}

#[cfg(not(feature = "llama-cpp"))]
impl LlamaCppEngine {
    /// Load a GGUF model.
    ///
    /// # Errors
    ///
    /// Always returns an error when llama-cpp feature is disabled.
    pub async fn load(_config: LlamaCppConfig) -> Result<Self> {
        Err(infernum_core::Error::InvalidConfig {
            message: "llama-cpp feature is not enabled. Rebuild with --features llama-cpp-cuda"
                .to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = LlamaCppConfig::builder()
            .model_path("/path/to/model.gguf")
            .n_gpu_layers(-1)
            .context_size(4096)
            .build()
            .expect("config should build");

        assert_eq!(config.model_path, PathBuf::from("/path/to/model.gguf"));
        assert_eq!(config.n_gpu_layers, -1);
        assert_eq!(config.context_size, 4096);
    }

    #[test]
    fn test_config_builder_requires_path() {
        let result = LlamaCppConfig::builder().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_default_config() {
        let config = LlamaCppConfig::default();
        assert_eq!(config.n_gpu_layers, -1);
        assert_eq!(config.batch_size, 512);
        assert!(config.use_mmap);
        assert!(!config.use_mlock);
        assert_eq!(config.split_mode, GpuSplitMode::None);
    }

    #[test]
    fn test_gpu_split_mode_default() {
        assert_eq!(GpuSplitMode::default(), GpuSplitMode::None);
    }

    #[test]
    fn test_config_builder_with_split_mode() {
        let config = LlamaCppConfig::builder()
            .model_path("/path/to/model.gguf")
            .split_mode(GpuSplitMode::Layer)
            .build()
            .expect("config should build");

        assert_eq!(config.split_mode, GpuSplitMode::Layer);
    }

    #[test]
    fn test_config_builder_with_row_split() {
        let config = LlamaCppConfig::builder()
            .model_path("/path/to/model.gguf")
            .split_mode(GpuSplitMode::Row)
            .main_gpu(1)
            .build()
            .expect("config should build");

        assert_eq!(config.split_mode, GpuSplitMode::Row);
        assert_eq!(config.main_gpu, 1);
    }

    #[test]
    fn test_parse_quantization() {
        assert_eq!(
            parse_quantization_type("Q4_K_M"),
            QuantizationType::GgufQ4KM
        );
        assert_eq!(
            parse_quantization_type("q5_k_m"),
            QuantizationType::GgufQ5KM
        );
        assert_eq!(parse_quantization_type("Q8_0"), QuantizationType::GgufQ8_0);
        assert_eq!(parse_quantization_type("F16"), QuantizationType::None);
    }

    #[cfg(feature = "llama-cpp")]
    #[test]
    fn test_format_chat_messages_chatml() {
        let messages = vec![Message::system("You are helpful."), Message::user("Hello!")];

        let prompt = LlamaCppEngine::format_chat_messages(&messages, ChatTemplate::ChatML);
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("Hello!"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[cfg(feature = "llama-cpp")]
    #[test]
    fn test_format_chat_messages_llama3() {
        let messages = vec![Message::system("You are helpful."), Message::user("Hello!")];

        let prompt = LlamaCppEngine::format_chat_messages(&messages, ChatTemplate::Llama3);
        assert!(prompt.contains("<|begin_of_text|>"));
        assert!(prompt.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(prompt.contains("Hello!"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[cfg(feature = "llama-cpp")]
    #[test]
    fn test_format_chat_messages_mistral() {
        let messages = vec![Message::system("You are helpful."), Message::user("Hello!")];

        let prompt = LlamaCppEngine::format_chat_messages(&messages, ChatTemplate::Mistral);
        // Mistral prepends system to first user message
        assert!(prompt.contains("[INST] You are helpful."));
        assert!(prompt.contains("Hello!"));
        assert!(prompt.contains("[/INST]"));
    }

    #[test]
    fn test_chat_template_from_architecture() {
        assert_eq!(
            ChatTemplate::from_architecture("llama"),
            ChatTemplate::Llama3
        );
        assert_eq!(
            ChatTemplate::from_architecture("Llama-3.2"),
            ChatTemplate::Llama3
        );
        assert_eq!(
            ChatTemplate::from_architecture("qwen2"),
            ChatTemplate::ChatML
        );
        assert_eq!(
            ChatTemplate::from_architecture("Qwen"),
            ChatTemplate::ChatML
        );
        assert_eq!(
            ChatTemplate::from_architecture("mistral"),
            ChatTemplate::Mistral
        );
        assert_eq!(ChatTemplate::from_architecture("phi3"), ChatTemplate::Phi3);
        assert_eq!(
            ChatTemplate::from_architecture("unknown"),
            ChatTemplate::ChatML
        );
    }

    #[test]
    fn test_backend_type_from_str() {
        assert_eq!(BackendType::from_str("auto"), Some(BackendType::Auto));
        assert_eq!(
            BackendType::from_str("llama-cpp"),
            Some(BackendType::LlamaCpp)
        );
        assert_eq!(
            BackendType::from_str("llamacpp"),
            Some(BackendType::LlamaCpp)
        );
        assert_eq!(
            BackendType::from_str("llama_cpp"),
            Some(BackendType::LlamaCpp)
        );
        assert_eq!(BackendType::from_str("gguf"), Some(BackendType::LlamaCpp));
        assert_eq!(BackendType::from_str("candle"), Some(BackendType::Candle));
        assert_eq!(BackendType::from_str("hct"), Some(BackendType::Candle));
        assert_eq!(
            BackendType::from_str("holotensor"),
            Some(BackendType::Candle)
        );
        assert_eq!(BackendType::from_str("invalid"), None);
    }

    #[test]
    fn test_backend_type_detect_from_path() {
        use std::path::Path;

        // GGUF files should use LlamaCpp
        assert_eq!(
            BackendType::detect_from_path(Path::new("/path/to/model.gguf")),
            BackendType::LlamaCpp
        );
        assert_eq!(
            BackendType::detect_from_path(Path::new("model.GGUF")),
            BackendType::LlamaCpp
        );

        // Unknown paths default to Candle
        assert_eq!(
            BackendType::detect_from_path(Path::new("/some/random/path")),
            BackendType::Candle
        );
    }

    #[test]
    fn test_backend_type_display() {
        assert_eq!(BackendType::Auto.to_string(), "auto");
        assert_eq!(BackendType::LlamaCpp.to_string(), "llama-cpp");
        assert_eq!(BackendType::Candle.to_string(), "candle");
    }

    #[test]
    fn test_backend_type_parse() {
        // Test the FromStr trait implementation
        let auto: std::result::Result<BackendType, String> = "auto".parse();
        assert!(auto.is_ok());
        assert_eq!(auto.unwrap(), BackendType::Auto);

        let llama: std::result::Result<BackendType, String> = "llama-cpp".parse();
        assert!(llama.is_ok());
        assert_eq!(llama.unwrap(), BackendType::LlamaCpp);

        let invalid: std::result::Result<BackendType, String> = "invalid".parse();
        assert!(invalid.is_err());
    }

    #[test]
    fn test_check_stop_sequences() {
        let stops = vec!["END".to_string(), "\n\n".to_string()];

        // Should match when text ends with stop sequence
        assert_eq!(check_stop_sequences("Hello worldEND", &stops), Some("END"));
        assert_eq!(
            check_stop_sequences("Hello world\n\n", &stops),
            Some("\n\n")
        );

        // Should not match partial or no match
        assert_eq!(check_stop_sequences("Hello world", &stops), None);
        assert_eq!(check_stop_sequences("Hello EN", &stops), None);
        assert_eq!(check_stop_sequences("ENDHello", &stops), None);

        // Empty stop sequences
        assert_eq!(check_stop_sequences("Hello", &[]), None);
    }

    #[test]
    fn test_trim_stop_sequence() {
        let stops = vec!["END".to_string(), "STOP".to_string()];

        let mut text = "Hello worldEND".to_string();
        assert!(trim_stop_sequence(&mut text, &stops));
        assert_eq!(text, "Hello world");

        let mut text = "Hello worldSTOP".to_string();
        assert!(trim_stop_sequence(&mut text, &stops));
        assert_eq!(text, "Hello world");

        let mut text = "Hello world".to_string();
        assert!(!trim_stop_sequence(&mut text, &stops));
        assert_eq!(text, "Hello world");
    }

    #[cfg(feature = "llama-cpp")]
    #[test]
    fn test_build_sampler_greedy() {
        let params = SamplingParams::greedy();
        // Should not panic - greedy returns a special sampler
        let _sampler = build_sampler(&params);
    }

    #[cfg(feature = "llama-cpp")]
    #[test]
    fn test_build_sampler_with_all_params() {
        let params = SamplingParams::default()
            .with_temperature(0.7)
            .with_top_p(0.9)
            .with_top_k(40)
            .with_min_p(0.1)
            .with_repetition_penalty(1.1)
            .with_presence_penalty(0.1)
            .with_frequency_penalty(0.1);

        // Should not panic when building with all parameters
        let _sampler = build_sampler(&params);
    }

    #[cfg(feature = "llama-cpp")]
    #[test]
    fn test_build_sampler_default() {
        let params = SamplingParams::default();
        // Default params (temp=1.0, top_p=1.0, etc.) should produce minimal stages
        let _sampler = build_sampler(&params);
    }

    #[cfg(feature = "llama-cpp")]
    #[test]
    fn test_build_sampler_with_json_mode() {
        use infernum_core::GrammarConstraint;

        let params = SamplingParams::default()
            .with_grammar(GrammarConstraint::Json)
            .with_temperature(0.7);

        // Should not panic when building with grammar
        let _sampler = build_sampler(&params);
    }

    #[cfg(feature = "llama-cpp")]
    #[test]
    fn test_build_sampler_greedy_with_grammar() {
        use infernum_core::GrammarConstraint;

        let params = SamplingParams::greedy().with_grammar(GrammarConstraint::Json);

        // Greedy with grammar should still work (creates minimal sampler with grammar)
        let _sampler = build_sampler(&params);
    }

    #[cfg(feature = "llama-cpp")]
    #[test]
    fn test_build_sampler_with_custom_gbnf() {
        use infernum_core::GrammarConstraint;

        let grammar = r#"root ::= "yes" | "no""#;
        let params = SamplingParams::default().with_grammar(GrammarConstraint::gbnf(grammar));

        // Should not panic
        let _sampler = build_sampler(&params);
    }
}
