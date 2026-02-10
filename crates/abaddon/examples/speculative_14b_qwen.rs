//! Speculative Decoding: Qwen 0.5B (draft) + Qwen 14B HoloTensor (target)
//!
//! Uses Qwen2.5-0.5B-Instruct (~1GB) as draft with same tokenizer as 14B.
//! Uses Qwen2.5-14B HoloTensor as target model with progressive loading.
//!
//! Usage:
//!   cargo run --release --example speculative_14b_qwen --features cuda

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use safetensors::SafeTensors;
use tokenizers::Tokenizer;

use abaddon::holotensor::tiered_loading::{TieredConfig, TieredHoloLoader};
use abaddon::lazy_varbuilder::{LazyVarBuilder, TensorProvider};
use abaddon::models::lazy_qwen2::LazyQwen2;
use abaddon::models::qwen2::{Qwen2, Qwen2Config};
use abaddon::speculative_405b::{DraftModel, Speculative405B, Speculative405BConfig, TargetModel};

/// Wrapper to make Qwen2 implement DraftModel trait.
/// Uses Mutex to satisfy Sync requirement.
struct Qwen2Draft {
    model: Mutex<Qwen2>,
    device: Device,
    dtype: DType,
}

impl Qwen2Draft {
    fn new(model: Qwen2, device: Device, dtype: DType) -> Self {
        Self {
            model: Mutex::new(model),
            device,
            dtype,
        }
    }
}

impl DraftModel for Qwen2Draft {
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> candle_core::Result<Tensor> {
        let mut model = self.model.lock().unwrap();
        model.forward(input_ids, pos)
    }

    fn clear_cache(&mut self) {
        let mut model = self.model.lock().unwrap();
        model.clear_cache();
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Load SafeTensors model weights from HuggingFace
fn load_safetensors_weights(
    model_id: &str,
    device: &Device,
    dtype: DType,
) -> Result<std::collections::HashMap<String, Tensor>> {
    println!("Loading SafeTensors from HuggingFace: {}", model_id);

    let api = Api::new()?;
    let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

    // Try single file first, then sharded
    let tensors_result = repo.get("model.safetensors");

    let mut tensors = std::collections::HashMap::new();

    if let Ok(path) = tensors_result {
        println!("Loading single safetensors file");
        let data = std::fs::read(&path)?;
        let safetensors = SafeTensors::deserialize(&data)?;

        for name in safetensors.names() {
            let tensor_view = safetensors.tensor(name)?;
            let shape: Vec<usize> = tensor_view.shape().to_vec();
            let tensor_dtype = match tensor_view.dtype() {
                safetensors::Dtype::F16 => DType::F16,
                safetensors::Dtype::BF16 => DType::BF16,
                safetensors::Dtype::F32 => DType::F32,
                _ => DType::F32,
            };

            let tensor =
                Tensor::from_raw_buffer(tensor_view.data(), tensor_dtype, &shape, &Device::Cpu)?;

            let tensor = tensor.to_device(device)?.to_dtype(dtype)?;
            tensors.insert(name.to_string(), tensor);
        }
    } else {
        // Sharded model - get index
        let index_path = repo.get("model.safetensors.index.json")?;
        let index_str = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_str)?;

        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("Missing weight_map"))?;

        let mut shard_files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str())
            .map(|s| s.to_string())
            .collect();
        shard_files.sort();
        shard_files.dedup();

        println!("Found {} shard files", shard_files.len());

        for (i, shard_name) in shard_files.iter().enumerate() {
            println!(
                "Loading shard {}/{}: {}",
                i + 1,
                shard_files.len(),
                shard_name
            );
            let shard_path = repo.get(shard_name)?;
            let data = std::fs::read(&shard_path)?;
            let safetensors = SafeTensors::deserialize(&data)?;

            for name in safetensors.names() {
                let tensor_view = safetensors.tensor(name)?;
                let shape: Vec<usize> = tensor_view.shape().to_vec();
                let tensor_dtype = match tensor_view.dtype() {
                    safetensors::Dtype::F16 => DType::F16,
                    safetensors::Dtype::BF16 => DType::BF16,
                    safetensors::Dtype::F32 => DType::F32,
                    _ => DType::F32,
                };

                let tensor = Tensor::from_raw_buffer(
                    tensor_view.data(),
                    tensor_dtype,
                    &shape,
                    &Device::Cpu,
                )?;

                let tensor = tensor.to_device(device)?.to_dtype(dtype)?;
                tensors.insert(name.to_string(), tensor);
            }
        }
    }

    println!("Loaded {} tensors total\n", tensors.len());
    Ok(tensors)
}

fn main() -> Result<()> {
    println!("========================================================================");
    println!("  Speculative Decoding: Qwen 0.5B (draft) + Qwen 14B HoloTensor (target)");
    println!("========================================================================\n");

    let hct_dir =
        PathBuf::from("/home/crook/.cache/infernum/models/hct/Qwen--Qwen2.5-14B-HoloTensor");

    if !hct_dir.exists() {
        println!(
            "ERROR: 14B HoloTensor model not found at: {}",
            hct_dir.display()
        );
        return Ok(());
    }

    if !candle_core::utils::cuda_is_available() {
        println!("ERROR: CUDA required for speculative decoding");
        return Ok(());
    }

    let device = Device::new_cuda(0)?;
    let dtype = DType::BF16;

    println!("Device: CUDA:0");
    println!("DType: BF16\n");

    // ============================================================
    // Load Draft Model (Qwen2.5-0.5B-Instruct) - ~1GB VRAM
    // ============================================================
    println!("=== Loading Draft Model: Qwen2.5-0.5B-Instruct ===");
    let draft_start = Instant::now();

    let api = Api::new()?;
    let draft_model_id = "Qwen/Qwen2.5-0.5B-Instruct";
    let draft_repo = api.repo(Repo::new(draft_model_id.to_string(), RepoType::Model));

    let config_path = draft_repo.get("config.json")?;
    let config_str = std::fs::read_to_string(&config_path)?;
    let draft_config: Qwen2Config = serde_json::from_str(&config_str)?;

    println!(
        "Draft config: {} layers, {} hidden, {} heads",
        draft_config.num_hidden_layers, draft_config.hidden_size, draft_config.num_attention_heads
    );

    let tensors = load_safetensors_weights(draft_model_id, &device, dtype)?;
    let vb = VarBuilder::from_tensors(tensors, dtype, &device);
    let draft_model = Qwen2::load(draft_config.clone(), vb)?;
    let draft = Qwen2Draft::new(draft_model, device.clone(), dtype);

    println!("Draft model loaded in {:?}\n", draft_start.elapsed());

    // ============================================================
    // Load 14B Target Model (LazyQwen2 with tiered HoloTensor)
    // ============================================================
    println!("=== Loading Target Model: Qwen 14B (Tiered HoloTensor) ===");
    let target_start = Instant::now();

    // With 0.5B draft (~1GB), we have plenty for 14B
    let config = TieredConfig {
        vram_budget: 20 * 1024 * 1024 * 1024, // 20GB for 14B
        ram_budget: 32 * 1024 * 1024 * 1024,  // 32GB RAM cache
        min_quality: 1.0,                     // Full quality - no compression loss
        target_quality: 1.0,
        enable_background_streaming: false,
        background_streams: 0,
    };

    let loader = TieredHoloLoader::new(&hct_dir, config, device.clone(), dtype)?;
    let loader = Arc::new(loader);

    println!("TieredHoloLoader created");
    println!("VRAM budget: 20 GB (draft uses ~1GB)");
    println!("min_quality: 1.0 (full quality)");
    println!(
        "GPU acceleration: {}\n",
        if loader.is_gpu_enabled() {
            "enabled"
        } else {
            "disabled"
        }
    );

    let provider: Arc<dyn TensorProvider> = Arc::clone(&loader) as Arc<dyn TensorProvider>;
    let lazy_vb = LazyVarBuilder::new(Arc::clone(&provider), device.clone(), dtype);

    // Qwen2.5-14B config
    let model_config = Qwen2Config {
        hidden_size: 5120,
        intermediate_size: 13824,
        vocab_size: 152064,
        num_hidden_layers: 48,
        num_attention_heads: 40,
        num_key_value_heads: Some(8),
        rms_norm_eps: 1e-6,
        rope_theta: 1000000.0,
        max_position_embeddings: 32768,
        tie_word_embeddings: false,
        sliding_window: None,
        use_sliding_window: false,
        bos_token_id: Some(151643),
        eos_token_id: Some(151645),
    };

    // Keep more layers in VRAM since 14B is smaller
    let max_loaded_layers = 24; // Half the layers
    let target = LazyQwen2::load(model_config.clone(), lazy_vb, max_loaded_layers)?;

    println!("14B model shell created in {:?}", target_start.elapsed());
    let stats = target.stats();
    println!(
        "Initial layers loaded: {}/{}",
        stats.loaded_layers, stats.total_layers
    );
    println!("Max concurrent layers: {}\n", max_loaded_layers);

    // ============================================================
    // Setup Speculative Decoding
    // ============================================================
    println!("=== Setting Up Speculative Decoder ===");

    let spec_config = Speculative405BConfig {
        num_draft_tokens: 5, // More draft tokens for faster model
        acceptance_threshold: 0.1,
        draft_temperature: 0.7,
        target_temperature: 0.7,
        greedy_draft: true,
    };

    println!("Configuration:");
    println!("  Draft tokens per round: {}", spec_config.num_draft_tokens);
    println!(
        "  Acceptance threshold: {}",
        spec_config.acceptance_threshold
    );

    let speculative = Speculative405B::new(draft, target, spec_config);

    // ============================================================
    // Load Tokenizer (shared between draft and target!)
    // ============================================================
    let tokenizer_path = draft_repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // ============================================================
    // Test Generation
    // ============================================================
    println!("\n=== Starting Generation ===");

    let prompt = "The future of artificial intelligence will";
    let max_tokens = 50;
    let eos_token = model_config.eos_token_id.unwrap_or(151645) as u32;

    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

    println!("Prompt: \"{}\"", prompt);
    println!("Prompt tokens: {}", prompt_tokens.len());
    println!("Max new tokens: {}", max_tokens);
    println!("\nGenerating...\n");

    let gen_start = Instant::now();
    let generated_tokens = speculative.generate(&prompt_tokens, max_tokens, eos_token)?;
    let gen_elapsed = gen_start.elapsed();

    let generated_text = tokenizer
        .decode(&generated_tokens, false)
        .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))?;

    // ============================================================
    // Print Results
    // ============================================================
    println!("========================================================================");
    println!("SPECULATIVE DECODING RESULTS");
    println!("========================================================================");

    let stats = speculative.stats();
    let tokens_per_sec = generated_tokens.len() as f64 / gen_elapsed.as_secs_f64();

    println!("\nPerformance:");
    println!(
        "  Generated: {} tokens in {:.2}s",
        generated_tokens.len(),
        gen_elapsed.as_secs_f64()
    );
    println!("  Speed: {:.1} tokens/sec", tokens_per_sec);

    println!("\nSpeculation Stats:");
    println!("  Rounds: {}", stats.rounds);
    println!("  Draft tokens proposed: {}", stats.draft_tokens);
    println!(
        "  Accepted: {} ({:.1}%)",
        stats.accepted_tokens,
        stats.acceptance_rate() * 100.0
    );
    println!("  Rejected: {}", stats.rejected_tokens);
    println!("  Tokens per round: {:.2}", stats.tokens_per_round());
    println!("  Effective speedup: {:.1}x", stats.speedup());

    println!("\nTiming Breakdown:");
    println!(
        "  Draft time: {} ms ({:.1} ms/forward)",
        stats.draft_time_ms,
        stats.draft_time_ms as f64 / stats.draft_forward_passes.max(1) as f64
    );
    println!(
        "  Verify time: {} ms ({:.1} ms/forward)",
        stats.verify_time_ms,
        stats.verify_time_ms as f64 / stats.target_forward_passes.max(1) as f64
    );

    println!("\n========================================================================");
    println!("Generated Text:");
    println!("------------------------------------------------------------------------");
    println!("{}{}", prompt, generated_text);
    println!("========================================================================");

    let loader_stats = loader.stats();
    println!("\nTiered Loader Stats:");
    println!("  Tensors loaded: {}", loader_stats.tensors_loaded);
    println!(
        "  GPU reconstructions: {} ({} ms)",
        loader_stats.gpu_reconstructions, loader_stats.gpu_time_ms
    );
    println!(
        "  CPU fallbacks: {} ({} ms)",
        loader_stats.cpu_reconstructions, loader_stats.cpu_time_ms
    );

    println!("\n=== Complete ===");
    Ok(())
}
