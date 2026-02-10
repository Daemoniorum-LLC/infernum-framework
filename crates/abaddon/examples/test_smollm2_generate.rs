//! SmolLM2 HCT text generation test
use std::io::Write;
use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use abaddon::hct_sequential::load_hct_directory_sequential;
use abaddon::models::{Llama, LlamaConfig};
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== SmolLM2-135M HCT Text Generation ===\n");

    let hct_dir = Path::new("/tmp/smollm2-hct-test");
    let tokenizer_path = "/home/crook/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac/tokenizer.json";

    let device = Device::Cpu;
    let dtype = DType::F32;

    // SmolLM2-135M config
    let config = LlamaConfig {
        hidden_size: 576,
        intermediate_size: 1536,
        vocab_size: 49152,
        num_hidden_layers: 30,
        num_attention_heads: 9,
        num_key_value_heads: Some(3),
        rms_norm_eps: 1e-5,
        rope_theta: 100000.0,
        max_position_embeddings: 8192,
        tie_word_embeddings: true,
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        rope_scaling: None,
    };

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    println!("Tokenizer loaded");

    // Load HCT weights
    println!("Loading HCT weights...");
    let tensors = load_hct_directory_sequential(hct_dir, &device, dtype)?;
    println!("Loaded {} tensors\n", tensors.len());

    // Build model
    let vb = VarBuilder::from_tensors(tensors, dtype, &device);
    let mut model = Llama::load(config.clone(), vb)?;
    println!("Model built\n");

    let prompt = "The capital of France is";
    println!("Prompt: \"{}\"\n", prompt);

    // Tokenize
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
    println!("Input tokens: {:?}\n", tokens);

    // Generate
    print!("Generated: ");
    std::io::stdout().flush()?;

    let max_new_tokens = 30;
    let eos_token_id = 2u32;

    for i in 0..max_new_tokens {
        let input = if i == 0 {
            Tensor::new(&tokens[..], &device)?.unsqueeze(0)?
        } else {
            Tensor::new(&[tokens[tokens.len() - 1]], &device)?.unsqueeze(0)?
        };

        let start_pos = if i == 0 { 0 } else { tokens.len() - 1 };
        let logits = model.forward(&input, start_pos)?;

        let seq_len = logits.dims()[1];
        let last_logits = logits.i((.., seq_len - 1, ..))?;

        // Simple greedy decoding
        let next_token = last_logits
            .argmax(candle_core::D::Minus1)?
            .flatten_all()?
            .to_vec1::<u32>()?[0];

        if next_token == eos_token_id {
            break;
        }

        tokens.push(next_token);

        // Decode and print
        if let Ok(text) = tokenizer.decode(&[next_token], false) {
            print!("{}", text);
            std::io::stdout().flush()?;
        }
    }
    println!("\n");

    // Decode full output
    let output_text = tokenizer
        .decode(&tokens, false)
        .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))?;
    println!("Full output: {}", output_text);

    Ok(())
}
