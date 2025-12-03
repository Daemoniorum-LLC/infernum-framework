//! Core inference engine implementation.

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use infernum_core::{
    EmbedRequest, EmbedResponse, GenerateRequest, GenerateResponse, ModelMetadata, Result,
    TokenStream,
};
use parking_lot::Mutex;
use tokio::sync::mpsc;

use crate::config::EngineConfig;
use crate::loader::{ModelConfig, ModelFiles, ModelLoader, WeightFiles};
use crate::models::llama::{Llama, LlamaConfig};
use crate::sampler::Sampler;
use crate::tokenizer::Tokenizer;

/// Trait defining the inference engine interface.
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    /// Generates text from the given request.
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse>;

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
    model: Mutex<Llama>,
    tokenizer: Tokenizer,
    config: LlamaConfig,
}

/// The main inference engine.
pub struct Engine {
    config: EngineConfig,
    metadata: ModelMetadata,
    loaded: Option<Arc<LoadedModel>>,
    device: Device,
    #[allow(dead_code)] // Used for model quantization in future
    dtype: DType,
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

        // Determine dtype
        let dtype = DType::F16; // Default to F16 for efficiency

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

        tracing::info!(
            model = %metadata.id,
            layers = model_config.num_hidden_layers,
            "Engine initialized successfully"
        );

        Ok(Self {
            config,
            metadata,
            loaded: Some(Arc::new(loaded)),
            device,
            dtype,
        })
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

    /// Loads the model from files.
    fn load_model(
        files: &ModelFiles,
        model_config: &ModelConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<LoadedModel> {
        tracing::info!("Loading model weights...");
        let start = Instant::now();

        // Build Llama config
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
            eos_token_id: model_config.eos_token_ids().first().copied(),
        };

        // Load weights
        let vb = Self::load_weights(&files.weights, device, dtype)?;

        // Build model
        let model =
            Llama::load(llama_config.clone(), vb).map_err(|e| infernum_core::Error::ModelLoad {
                message: format!("Failed to load Llama model: {}", e),
            })?;

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
            config: llama_config,
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
            WeightFiles::Gguf(_path) => Err(infernum_core::Error::ModelLoad {
                message: "GGUF loading not yet implemented".to_string(),
            }),
            WeightFiles::PyTorch(_) | WeightFiles::ShardedPyTorch { .. } => {
                Err(infernum_core::Error::ModelLoad {
                    message: "PyTorch format not supported, please use safetensors".to_string(),
                })
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

    /// Creates a shareable reference to this engine.
    #[must_use]
    pub fn into_shared(self) -> Arc<Self> {
        Arc::new(self)
    }

    /// Generates tokens from the model.
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

        let logits_vec: Vec<f32> =
            last_logits
                .to_vec1()
                .map_err(|e| infernum_core::Error::Internal {
                    message: format!("Failed to convert logits: {}", e),
                })?;

        // Sample first token
        let mut generated_tokens = Vec::new();
        let mut generated_text = Vec::new();
        let mut next_token = sampler.sample(&logits_vec);

        // Check for EOS
        let eos_token = loaded.config.eos_token_id.unwrap_or(2);

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

            let logits_vec: Vec<f32> =
                last_logits
                    .to_vec1()
                    .map_err(|e| infernum_core::Error::Internal {
                        message: format!("Failed to convert logits: {}", e),
                    })?;

            next_token = sampler.sample(&logits_vec);
        }

        Ok((generated_tokens, generated_text))
    }
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

        // Generate
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
            model: self.metadata.id.clone(),
            choices: vec![infernum_core::response::Choice {
                index: 0,
                text,
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
        let eos_token = loaded.config.eos_token_id.unwrap_or(2);

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
        tracing::debug!(request_id = %request.request_id, "Embedding generation not yet implemented");

        // TODO: Implement embedding extraction
        Err(infernum_core::Error::Internal {
            message: "Embedding generation not yet implemented for decoder-only models".to_string(),
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
}
