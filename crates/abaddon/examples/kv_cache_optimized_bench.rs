//! Optimized KV Cache Benchmark
//!
//! Compares three KV cache strategies:
//! 1. Standard BF16 (baseline)
//! 2. Full INT8 quantization (basic)
//! 3. Optimized INT8 with dynamic quantization and unquantized window (new)
//!
//! The optimized cache keeps recent tokens in BF16 for speed while
//! quantizing older tokens to INT8 for memory savings.
//!
//! Usage:
//!   cargo run --release -p abaddon --example kv_cache_optimized_bench --features cuda
//!   cargo run --release -p abaddon --example kv_cache_optimized_bench --features cuda -- --tokens=2000
//!   cargo run --release -p abaddon --example kv_cache_optimized_bench --features cuda -- --window=256

use std::path::Path;
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use abaddon::hct_sequential::load_hct_directory_parallel;
use abaddon::kv_cache_quant_cuda::{DynamicQuantConfig, QuantGranularity};
use abaddon::models::qwen2::{Qwen2, Qwen2Config};

#[derive(Debug, Clone, Copy, PartialEq)]
enum CacheMode {
    Standard,
    Quantized,
    Optimized,
}

impl std::fmt::Display for CacheMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Standard => write!(f, "Standard BF16"),
            Self::Quantized => write!(f, "Basic INT8 Quantized"),
            Self::Optimized => write!(f, "Optimized INT8 (dynamic)"),
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let target_tokens = args
        .iter()
        .find(|a| a.starts_with("--tokens="))
        .and_then(|a| a.strip_prefix("--tokens="))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1000);

    let unquant_window = args
        .iter()
        .find(|a| a.starts_with("--window="))
        .and_then(|a| a.strip_prefix("--window="))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(128);

    let mode = if args.contains(&"--standard".to_string()) {
        CacheMode::Standard
    } else if args.contains(&"--basic".to_string()) {
        CacheMode::Quantized
    } else {
        CacheMode::Optimized
    };

    println!("=== Optimized KV Cache Benchmark ===\n");

    let model_dir = Path::new(
        "/home/crook/dev2/workspace/nyx/infernum/infernum-complete/test_models/qwen2.5-7b-int4-v3",
    );
    let config_path = model_dir.join("config.json");
    let tokenizer_path = model_dir.join("tokenizer.json");

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };

    println!("Device: {:?}, DType: {:?}", device, dtype);
    println!("KV Cache Mode: {}", mode);
    if mode == CacheMode::Optimized {
        println!("Unquantized Window: {} tokens", unquant_window);
    }

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Load config
    let config_str = std::fs::read_to_string(&config_path)?;
    let config: Qwen2Config = serde_json::from_str(&config_str)?;

    // Load weights
    println!("\nLoading INT4 weights...");
    let start = Instant::now();
    let tensors = load_hct_directory_parallel(model_dir, &device, dtype)?;
    println!(
        "  Loaded {} tensors in {:.1}s",
        tensors.len(),
        start.elapsed().as_secs_f64()
    );

    // Build model with appropriate cache type
    println!("\nBuilding model with {} cache...", mode);
    let vb = VarBuilder::from_tensors(tensors, dtype, &device);
    let mut model = match mode {
        CacheMode::Standard => Qwen2::load_with_flash_attention(config.clone(), vb)?,
        CacheMode::Quantized => Qwen2::load_with_flash_and_quantized_cache(config.clone(), vb)?,
        CacheMode::Optimized => {
            let quant_config = DynamicQuantConfig {
                enabled: true,
                memory_threshold: 0.8,
                unquantized_window: unquant_window,
                granularity: QuantGranularity::PerToken,
            };
            Qwen2::load_with_flash_and_optimized_cache(config.clone(), vb, quant_config)?
        },
    };
    println!("  Model ready!");

    // Create a long prompt
    let base_text = r#"The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain.

The field of AI research was born at a workshop at Dartmouth College in 1956. Attendees Allen Newell, Herbert Simon, John McCarthy, Marvin Minsky and Arthur Samuel became the founders and leaders of AI research. They and their students produced programs that the press described as astonishing: computers were learning checkers strategies, solving word problems in algebra, proving logical theorems and speaking English.

"#;

    // Repeat to get target tokens (each repeat adds ~200 tokens)
    let repeats = (target_tokens / 200).max(1);
    let long_prompt = base_text.repeat(repeats);

    // Tokenize
    let encoding = tokenizer
        .encode(long_prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_len = prompt_tokens.len();

    println!("\n{}", "=".repeat(60));
    println!("Prompt length: {} tokens", prompt_len);
    println!("Generating: 100 tokens");
    println!("{}", "=".repeat(60));

    let max_new_tokens = 100;
    let eos_token_id = config.eos_token_id.unwrap_or(151645);

    // Prefill phase
    println!("\nPrefill phase ({} tokens)...", prompt_len);
    let prefill_start = Instant::now();

    let input = Tensor::new(&prompt_tokens[..], &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;

    let prefill_time = prefill_start.elapsed();
    let prefill_tps = prompt_len as f64 / prefill_time.as_secs_f64();
    println!(
        "  Prefill: {:.2}s ({:.1} tokens/s)",
        prefill_time.as_secs_f64(),
        prefill_tps
    );

    // Get first token
    let seq_len = logits.dim(1)?;
    let last_logits = logits.i((0, seq_len - 1, ..))?;
    let mut next_token = last_logits.argmax(0)?.to_scalar::<u32>()?;

    // Decode phase
    println!("\nDecode phase ({} tokens)...", max_new_tokens);
    let decode_start = Instant::now();
    let mut generated = vec![next_token];
    let mut current_pos = prompt_len;

    for _ in 1..max_new_tokens {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, current_pos)?;

        let last_logits = logits.i((0, 0, ..))?;
        next_token = last_logits.argmax(0)?.to_scalar::<u32>()?;

        if next_token == eos_token_id {
            println!("  [EOS reached]");
            break;
        }

        generated.push(next_token);
        current_pos += 1;
    }

    let decode_time = decode_start.elapsed();
    let decode_tps = generated.len() as f64 / decode_time.as_secs_f64();

    // Calculate memory estimates
    let num_layers = config.num_hidden_layers;
    let num_kv_heads = config
        .num_key_value_heads
        .unwrap_or(config.num_attention_heads);
    let head_dim = config.hidden_size / config.num_attention_heads;
    let total_seq_len = prompt_len + generated.len();

    // Standard: K+V in BF16 (2 bytes each)
    let standard_cache_bytes = 2 * num_layers * num_kv_heads * total_seq_len * head_dim * 2;

    // Basic quantized: K+V in U8 (1 byte) + scales in BF16
    let basic_quant_bytes = 2 * num_layers * num_kv_heads * total_seq_len * head_dim * 1  // K+V
        + 2 * num_layers * num_kv_heads * total_seq_len * 2; // scales

    // Optimized: Recent window in BF16 + older in U8
    let quantized_len = total_seq_len.saturating_sub(unquant_window);
    let recent_len = total_seq_len.min(unquant_window);
    let optimized_bytes =
        // Quantized portion (U8 + scales)
        2 * num_layers * num_kv_heads * quantized_len * head_dim * 1  // K+V
        + 2 * num_layers * num_kv_heads * quantized_len * 2  // scales
        // Recent portion (BF16)
        + 2 * num_layers * num_kv_heads * recent_len * head_dim * 2; // K+V

    println!("\n{}", "=".repeat(60));
    println!("RESULTS ({}):", mode);
    println!(
        "  Prefill ({} tokens): {:.2}s ({:.1} tok/s)",
        prompt_len,
        prefill_time.as_secs_f64(),
        prefill_tps
    );
    println!(
        "  Decode ({} tokens):  {:.2}s ({:.1} tok/s)",
        generated.len(),
        decode_time.as_secs_f64(),
        decode_tps
    );
    println!("{}", "=".repeat(60));

    println!("\nMEMORY ANALYSIS:");
    println!("  Total sequence length: {} tokens", total_seq_len);
    println!(
        "  Standard KV cache:     {:.1} MB",
        standard_cache_bytes as f64 / 1024.0 / 1024.0
    );
    println!(
        "  Basic quantized:       {:.1} MB ({:.2}x savings)",
        basic_quant_bytes as f64 / 1024.0 / 1024.0,
        standard_cache_bytes as f64 / basic_quant_bytes as f64
    );
    if mode == CacheMode::Optimized {
        println!(
            "  Optimized (window={}): {:.1} MB ({:.2}x savings)",
            unquant_window,
            optimized_bytes as f64 / 1024.0 / 1024.0,
            standard_cache_bytes as f64 / optimized_bytes as f64
        );
        println!(
            "    - Quantized tokens:  {} ({:.1}%)",
            quantized_len,
            quantized_len as f64 / total_seq_len as f64 * 100.0
        );
        println!(
            "    - Recent (BF16):     {} ({:.1}%)",
            recent_len,
            recent_len as f64 / total_seq_len as f64 * 100.0
        );
    }
    println!("{}", "=".repeat(60));

    // Show generated text
    let generated_text = tokenizer
        .decode(&generated, false)
        .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))?;
    println!("\nGenerated text:\n{}", generated_text);

    // Comparison summary
    println!("\n{}", "=".repeat(60));
    println!("RUN OTHER MODES FOR COMPARISON:");
    println!("  --standard : Standard BF16 cache (fastest, most memory)");
    println!("  --basic    : Basic INT8 quantized (2x memory savings, slower)");
    println!("  (default)  : Optimized INT8 (best balance)");
    println!("  --window=N : Set unquantized window size (default: 128)");
    println!("{}", "=".repeat(60));

    Ok(())
}
