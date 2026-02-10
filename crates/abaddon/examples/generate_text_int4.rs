//! Generate actual text with INT4 model.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use abaddon::hct_sequential::load_hct_directory_parallel;
use abaddon::models::qwen2::{Qwen2, Qwen2Config};

fn main() -> anyhow::Result<()> {
    println!("=== INT4 Text Generation ===\n");

    let model_dir = Path::new(
        "/home/crook/dev2/workspace/nyx/infernum/infernum-complete/test_models/qwen2.5-7b-int4-v3",
    );
    let config_path = model_dir.join("config.json");
    let tokenizer_path = model_dir.join("tokenizer.json");

    // Check files exist
    if !model_dir.exists() {
        anyhow::bail!("Model directory not found: {}", model_dir.display());
    }
    if !tokenizer_path.exists() {
        anyhow::bail!("Tokenizer not found: {}", tokenizer_path.display());
    }

    // Use CUDA if available, otherwise CPU
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    // Use BF16 for GPU (better dynamic range than F16, half bandwidth of F32)
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };

    println!("Device: {:?}, DType: {:?}", device, dtype);

    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Load config
    println!("Loading config...");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config: Qwen2Config = serde_json::from_str(&config_str)?;

    println!("  Model: Qwen2.5-7B-INT4");
    println!(
        "  Hidden: {}, Layers: {}, Heads: {}/{}",
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_kv_heads()
    );

    // Load weights
    println!("\nLoading INT4 weights...");
    let start = Instant::now();
    let tensors = load_hct_directory_parallel(model_dir, &device, dtype)?;
    println!(
        "  Loaded {} tensors in {:.1}s",
        tensors.len(),
        start.elapsed().as_secs_f64()
    );

    // Build model with Flash Attention for faster inference
    println!("\nBuilding model...");
    let vb = VarBuilder::from_tensors(tensors, dtype, &device);
    let use_flash_attn = std::env::var("NO_FLASH_ATTN").is_err();
    let mut model = if use_flash_attn {
        println!("  Using Flash Attention (set NO_FLASH_ATTN=1 to disable)");
        Qwen2::load_with_flash_attention(config.clone(), vb)?
    } else {
        println!("  Using standard attention");
        Qwen2::load(config.clone(), vb)?
    };
    println!("  Model ready!");

    // Generation parameters
    let prompt = "The meaning of life is";
    let max_tokens = 50;

    println!("\n{}", "=".repeat(50));
    println!("Prompt: \"{}\"", prompt);
    println!("Max tokens: {}, Decoding: greedy", max_tokens);
    println!("{}", "=".repeat(50));

    // Tokenize prompt
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let mut tokens: Vec<u32> = encoding.get_ids().to_vec();

    println!("\nInput tokens: {:?}", tokens);
    print!("\nGenerating: {}", prompt);
    std::io::stdout().flush()?;

    let gen_start = Instant::now();
    let mut generated_tokens = Vec::new();

    // Get EOS token
    let eos_token_id = config.eos_token_id.unwrap_or(151645);

    for i in 0..max_tokens {
        // For the first iteration, process full prompt
        // For subsequent iterations, only process the last token with correct position
        let (input_tokens, start_pos) = if i == 0 {
            (tokens.clone(), 0)
        } else {
            (vec![tokens[tokens.len() - 1]], tokens.len() - 1)
        };

        let input = Tensor::new(&input_tokens[..], &device)?.unsqueeze(0)?;

        // Forward pass with correct position
        let logits = model.forward(&input, start_pos)?;

        // Get logits for last token
        let seq_len = logits.dim(1)?;
        let last_logits = logits.i((0, seq_len - 1, ..))?;

        // Greedy decoding (argmax)
        let next_token = last_logits.argmax(0)?.to_scalar::<u32>()?;

        // Check for EOS
        if next_token == eos_token_id {
            println!(" [EOS]");
            break;
        }

        // Decode and print token
        if let Some(text) = tokenizer.decode(&[next_token], false).ok() {
            print!("{}", text);
            std::io::stdout().flush()?;
        }

        tokens.push(next_token);
        generated_tokens.push(next_token);

        // Show progress every 10 tokens
        if (i + 1) % 10 == 0 {
            let elapsed = gen_start.elapsed().as_secs_f64();
            let tps = (i + 1) as f64 / elapsed;
            eprint!(" [{:.1} tok/s]", tps);
        }
    }

    let gen_time = gen_start.elapsed();
    let tokens_per_sec = generated_tokens.len() as f64 / gen_time.as_secs_f64();

    println!("\n");
    println!("{}", "=".repeat(50));
    println!(
        "Generated {} tokens in {:.2}s ({:.2} tok/s)",
        generated_tokens.len(),
        gen_time.as_secs_f64(),
        tokens_per_sec
    );

    // Decode full output
    let full_output = tokenizer
        .decode(&tokens, false)
        .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))?;

    println!("\nFull output:");
    println!("{}", "-".repeat(50));
    println!("{}", full_output);
    println!("{}", "-".repeat(50));

    println!("\n🎉 INT4 text generation completed successfully!");

    Ok(())
}
