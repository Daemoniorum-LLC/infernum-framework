//! Text generation with HCT model
use std::path::Path;
use std::time::Instant;

use abaddon::hct_sequential::load_hct_directory_sequential;
use abaddon::models::qwen2::{Qwen2, Qwen2Config};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <hct_model_dir> [prompt]", args[0]);
        std::process::exit(1);
    }

    let model_dir = Path::new(&args[1]);
    let prompt = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("The capital of France is");
    let max_tokens = 50;

    println!("=== HCT Text Generation ===");
    println!("Model: {}", model_dir.display());
    println!("Prompt: \"{}\"", prompt);
    println!("Max tokens: {}", max_tokens);
    println!();

    let device = Device::Cpu;
    let dtype = DType::F32;

    // Config for Qwen2.5-7B-Instruct
    let config = Qwen2Config {
        hidden_size: 3584,
        intermediate_size: 18944,
        vocab_size: 152064,
        num_hidden_layers: 28,
        num_attention_heads: 28,
        num_key_value_heads: Some(4),
        rms_norm_eps: 1e-6,
        rope_theta: 1000000.0,
        max_position_embeddings: 32768,
        tie_word_embeddings: false,
        sliding_window: None,
        use_sliding_window: false,
        bos_token_id: Some(151643),
        eos_token_id: Some(151645),
    };

    // Load tokenizer
    let tokenizer_path = "/home/crook/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/tokenizer.json";
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    println!("Tokenizer loaded");

    // Load HCT weights
    println!("\n--- Loading HCT Weights ---");
    let start = Instant::now();
    let tensors = load_hct_directory_sequential(model_dir, &device, dtype)?;
    println!(
        "Loaded {} tensors in {:.2}s",
        tensors.len(),
        start.elapsed().as_secs_f32()
    );

    // Build model
    println!("\n--- Building Model ---");
    let vb = VarBuilder::from_tensors(tensors, dtype, &device);
    let start = Instant::now();
    let mut model = Qwen2::load(config.clone(), vb)?;
    println!("Model built in {:.2}s", start.elapsed().as_secs_f32());

    // Tokenize prompt
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
    println!("\nInput tokens ({} total): {:?}", tokens.len(), tokens);

    let eos_token_id: u32 = 151645;

    // Generate
    println!("\n--- Generating ---");
    print!("Output: ");
    use std::io::Write;
    std::io::stdout().flush()?;

    let gen_start = Instant::now();
    let initial_len = tokens.len();

    // First pass: process entire prompt
    let input = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?; // seqlen_offset = 0 for first pass

    // Get first generated token
    let seq_len = logits.dim(1)?;
    let last_logits = logits.i((0, seq_len - 1, ..))?;
    let next_token = last_logits.argmax(0)?.to_scalar::<u32>()?;

    if next_token != eos_token_id {
        tokens.push(next_token);
        if let Ok(text) = tokenizer.decode(&[next_token], false) {
            print!("{}", text);
            std::io::stdout().flush()?;
        }
    }

    // Subsequent passes: feed only the last token, use KV cache
    for _ in 1..max_tokens {
        let last_token = *tokens.last().unwrap();
        let input = Tensor::new(&[last_token], &device)?.unsqueeze(0)?;

        // seqlen_offset = number of tokens before this one (KV cache position)
        let seqlen_offset = tokens.len() - 1;
        let logits = model.forward(&input, seqlen_offset)?;

        let last_logits = logits.i((0, 0, ..))?; // Only 1 token output
        let next_token = last_logits.argmax(0)?.to_scalar::<u32>()?;

        if next_token == eos_token_id {
            print!("[EOS]");
            std::io::stdout().flush()?;
            break;
        }

        tokens.push(next_token);

        if let Ok(text) = tokenizer.decode(&[next_token], false) {
            print!("{}", text);
            std::io::stdout().flush()?;
        }
    }

    let gen_time = gen_start.elapsed().as_secs_f32();
    let tokens_generated = tokens.len() - initial_len;
    println!(
        "\n\nGenerated {} tokens in {:.2}s ({:.2} tok/s)",
        tokens_generated,
        gen_time,
        tokens_generated as f32 / gen_time
    );

    // Decode full output
    let output = tokenizer
        .decode(&tokens, false)
        .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))?;
    println!("\n=== Full Output ===");
    println!("{}", output);

    Ok(())
}
