//! Speculative Decoding for 405B Inference
//!
//! Uses a small draft model to generate candidate tokens quickly,
//! then verifies them with the 405B model in a single forward pass.
//!
//! ## Expected Performance
//!
//! | Mode | Time-to-first-token | Decode Speed |
//! |------|---------------------|--------------|
//! | Standard | ~10s (safetensors) | 2-3 tok/s |
//! | Speculative | ~10s (safetensors) | 8-12 tok/s |
//!
//! ## Usage
//!
//! ```bash
//! # First, convert HoloTensor to safetensors (one-time, ~3 hours)
//! cargo run --example holo_to_safetensors --release --features cuda -- \
//!     --input /tmp/llama405b-holo \
//!     --output /tmp/llama405b-safetensors
//!
//! # Then run speculative decoding
//! cargo run --example speculative_405b --release --features cuda
//! ```

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use abaddon::hct_sequential::load_hct_directory_parallel;
use abaddon::holotensor::tiered_loading::{TieredConfig, TieredHoloLoader};
use abaddon::lazy_varbuilder::{LazyVarBuilder, TensorProvider};
use abaddon::models::lazy_llama::LazyLlama;
use abaddon::models::qwen2::{Qwen2, Qwen2Config};
use abaddon::models::LlamaConfig;
use abaddon::speculative_405b::{DraftModel, Speculative405B, Speculative405BConfig};

/// Wrapper to make Qwen2 implement DraftModel trait.
struct Qwen2Draft {
    model: Qwen2,
    device: Device,
    dtype: DType,
}

impl Qwen2Draft {
    fn new(model: Qwen2, device: Device, dtype: DType) -> Self {
        Self {
            model,
            device,
            dtype,
        }
    }
}

impl DraftModel for Qwen2Draft {
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> candle_core::Result<Tensor> {
        self.model.forward(input_ids, pos)
    }

    fn clear_cache(&mut self) {
        self.model.clear_cache();
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

fn main() -> Result<()> {
    println!("=== Speculative Decoding for 405B ===\n");

    // Paths
    let draft_model_dir = Path::new(
        "/home/crook/dev2/workspace/nyx/infernum/infernum-complete/test_models/qwen2.5-7b-int4-v3",
    );
    let hct_dir = Path::new("/tmp/llama405b-holo");
    let safetensors_dir = Path::new("/tmp/llama405b-safetensors");

    // Check draft model
    if !draft_model_dir.exists() {
        println!("Draft model not found: {}", draft_model_dir.display());
        println!("Please ensure a small draft model is available.");
        println!("\nFor testing, you can use Qwen2.5 7B INT4 or similar.");
        return Ok(());
    }

    // Check 405B model
    if !hct_dir.exists() && !safetensors_dir.exists() {
        println!("405B model not found.");
        println!("Expected HCT: {}", hct_dir.display());
        println!("Expected safetensors: {}", safetensors_dir.display());
        return Ok(());
    }

    // Setup device
    let has_cuda = candle_core::utils::cuda_is_available();
    if !has_cuda {
        println!("CUDA required for speculative decoding with 405B");
        return Ok(());
    }

    let device = Device::new_cuda(0)?;
    let dtype = DType::F16; // Use F16 for both models

    println!("Device: {:?}", device);
    println!("DType: {:?}\n", dtype);

    // ============================================================
    // Load Draft Model (fits entirely in VRAM)
    // ============================================================
    println!("--- Loading Draft Model (Qwen2.5 7B) ---");

    let draft_start = Instant::now();

    // Load config
    let config_path = draft_model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let draft_config: Qwen2Config = serde_json::from_str(&config_str)?;

    println!(
        "Draft config: {} layers, {} hidden, {} heads",
        draft_config.num_hidden_layers, draft_config.hidden_size, draft_config.num_attention_heads
    );

    // Load weights
    let tensors = load_hct_directory_parallel(draft_model_dir, &device, dtype)?;
    println!("Loaded {} tensors", tensors.len());

    // Build model
    let vb = VarBuilder::from_tensors(tensors, dtype, &device);
    let draft_model = Qwen2::load_with_flash_attention(draft_config.clone(), vb)?;
    let draft = Qwen2Draft::new(draft_model, device.clone(), dtype);

    println!("Draft model loaded in {:?}\n", draft_start.elapsed());

    // ============================================================
    // Load 405B Target Model (lazy loading with safetensors)
    // ============================================================
    println!("--- Loading 405B Target Model (LazyLlama) ---");

    let target_start = Instant::now();

    // Configure tiered loader
    let config = TieredConfig {
        vram_budget: 20 * 1024 * 1024 * 1024, // 20GB (leave room for draft)
        ram_budget: 60 * 1024 * 1024 * 1024,  // 60GB
        min_quality: 0.95,
        target_quality: 0.95,
        enable_background_streaming: false,
        background_streams: 0,
    };

    // Create loader with safetensors fast-load if available
    let mut loader = TieredHoloLoader::new(hct_dir, config, device.clone(), dtype)?;

    if safetensors_dir.exists() {
        println!("Safetensors fast-load: ENABLED");
        loader = loader.with_safetensors_dir(safetensors_dir);
    } else {
        println!("Safetensors not found - will use HoloTensor reconstruction (slow)");
        println!("Run 'holo_to_safetensors' to pre-convert for faster loading");
    }

    let loader = Arc::new(loader);
    println!("TieredHoloLoader created");
    println!(
        "GPU acceleration: {}\n",
        if loader.is_gpu_enabled() {
            "enabled"
        } else {
            "disabled"
        }
    );

    // Create LazyVarBuilder
    let provider: Arc<dyn TensorProvider> = Arc::clone(&loader) as Arc<dyn TensorProvider>;
    let lazy_vb = LazyVarBuilder::new(Arc::clone(&provider), device.clone(), dtype);

    // Llama 405B config
    let model_config = LlamaConfig {
        hidden_size: 16384,
        intermediate_size: 53248,
        vocab_size: 128256,
        num_hidden_layers: 126,
        num_attention_heads: 128,
        num_key_value_heads: Some(8),
        rms_norm_eps: 1e-5,
        rope_theta: 500000.0,
        max_position_embeddings: 131072,
        tie_word_embeddings: false,
        bos_token_id: Some(128000),
        eos_token_id: Some(128001),
        rope_scaling: None,
    };

    // Load LazyLlama with limited layers initially
    let max_loaded_layers = 20; // Start with 20 layers loaded
    let target = LazyLlama::load(model_config.clone(), lazy_vb, max_loaded_layers)?;

    println!("405B model created in {:?}", target_start.elapsed());
    let stats = target.stats();
    println!(
        "Layers loaded: {}/{}\n",
        stats.loaded_layers, stats.total_layers
    );

    // ============================================================
    // Setup Speculative Decoding
    // ============================================================
    println!("--- Setting Up Speculative Decoder ---");

    let spec_config = Speculative405BConfig {
        num_draft_tokens: 5,       // Generate 5 draft tokens per round
        acceptance_threshold: 0.1, // Low threshold (405B is high quality)
        draft_temperature: 0.7,
        target_temperature: 0.7,
        greedy_draft: true, // Use greedy for speed
    };

    println!("Draft tokens per round: {}", spec_config.num_draft_tokens);
    println!("Greedy draft: {}", spec_config.greedy_draft);

    let speculative = Speculative405B::new(draft, target, spec_config);

    // ============================================================
    // Test Generation
    // ============================================================
    println!("\n--- Test Generation ---");

    // Simple prompt
    let prompt = "The future of artificial intelligence will";
    let max_tokens = 50;
    let eos_token = model_config.eos_token_id.unwrap_or(128001) as u32;

    // Tokenize (using draft tokenizer)
    let tokenizer_path = draft_model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

    println!("Prompt: \"{}\"", prompt);
    println!("Prompt tokens: {}", prompt_tokens.len());
    println!("Max new tokens: {}", max_tokens);
    println!();

    // Generate
    let gen_start = Instant::now();
    let generated_tokens = speculative.generate(&prompt_tokens, max_tokens, eos_token)?;
    let gen_elapsed = gen_start.elapsed();

    // Decode
    let generated_text = tokenizer
        .decode(&generated_tokens, false)
        .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))?;

    // Print results
    println!("\n{}", "=".repeat(60));
    println!("SPECULATIVE DECODING RESULTS:");
    println!("{}", "=".repeat(60));

    let stats = speculative.stats();
    let tokens_per_sec = generated_tokens.len() as f64 / gen_elapsed.as_secs_f64();

    println!(
        "Generated: {} tokens in {:.2}s ({:.1} tok/s)",
        generated_tokens.len(),
        gen_elapsed.as_secs_f64(),
        tokens_per_sec
    );

    println!("\nSpeculation stats:");
    println!("  Rounds: {}", stats.rounds);
    println!("  Draft tokens: {}", stats.draft_tokens);
    println!(
        "  Accepted: {} ({:.1}%)",
        stats.accepted_tokens,
        stats.acceptance_rate() * 100.0
    );
    println!("  Rejected: {}", stats.rejected_tokens);
    println!("  Tokens per round: {:.2}", stats.tokens_per_round());
    println!("  Effective speedup: {:.1}x", stats.speedup());

    println!("\nTiming:");
    println!(
        "  Draft time: {} ms ({:.1} ms/token)",
        stats.draft_time_ms,
        stats.draft_time_ms as f64 / stats.draft_forward_passes.max(1) as f64
    );
    println!(
        "  Verify time: {} ms ({:.1} ms/pass)",
        stats.verify_time_ms,
        stats.verify_time_ms as f64 / stats.target_forward_passes.max(1) as f64
    );

    println!("\n{}", "=".repeat(60));
    println!("Generated text:");
    println!("{}{}", prompt, generated_text);
    println!("{}", "=".repeat(60));

    // Print tiered loader stats
    println!("\n--- Tiered Loader Statistics ---");
    let loader_stats = loader.stats();
    println!("  Tensors loaded: {}", loader_stats.tensors_loaded);

    if loader_stats.safetensor_loads > 0 {
        println!(
            "  Safetensor loads: {} ({} ms)",
            loader_stats.safetensor_loads, loader_stats.safetensor_time_ms
        );
    }
    if loader_stats.gpu_reconstructions > 0 {
        println!(
            "  GPU reconstructions: {} ({} ms)",
            loader_stats.gpu_reconstructions, loader_stats.gpu_time_ms
        );
    }

    println!("\n=== Test Complete ===");
    Ok(())
}
