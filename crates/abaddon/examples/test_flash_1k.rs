//! Test Flash Attention with 1K token prompt.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use abaddon::hct_sequential::load_hct_directory_parallel;
use abaddon::models::qwen2::{Qwen2, Qwen2Config};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let use_flash = !args.contains(&"--no-flash".to_string());

    println!("=== Flash Attention 1K Token Benchmark ===\n");

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
    println!(
        "Flash Attention: {}",
        if use_flash { "ENABLED" } else { "DISABLED" }
    );

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

    // Build model
    println!("\nBuilding model...");
    let vb = VarBuilder::from_tensors(tensors, dtype, &device);
    let mut model = if use_flash {
        Qwen2::load_with_flash_attention(config.clone(), vb)?
    } else {
        Qwen2::load(config.clone(), vb)?
    };
    println!("  Model ready!");

    // Create a ~1K token prompt by repeating text
    let base_text = r#"The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain.

The field of AI research was born at a workshop at Dartmouth College in 1956. Attendees Allen Newell, Herbert Simon, John McCarthy, Marvin Minsky and Arthur Samuel became the founders and leaders of AI research. They and their students produced programs that the press described as astonishing: computers were learning checkers strategies, solving word problems in algebra, proving logical theorems and speaking English. By the middle of the 1960s, research in the U.S. was heavily funded by the Department of Defense and laboratories had been established around the world.

AI's founders were optimistic about the future: Herbert Simon predicted, "machines will be capable, within twenty years, of doing any work a man can do". Marvin Minsky agreed, writing, "within a generation ... the problem of creating 'artificial intelligence' will substantially be solved". They failed to recognize the difficulty of some of the remaining tasks. Progress slowed and in 1974, in response to the criticism of Sir James Lighthill and ongoing pressure from the US Congress to fund more productive projects, both the U.S. and British governments cut off exploratory research in AI. The next few years would later be called an "AI winter", a period when obtaining funding for AI projects was difficult.

"#;

    // Get target token count from args (default 1K)
    let target_tokens = args
        .iter()
        .find(|a| a.starts_with("--tokens="))
        .and_then(|a| a.strip_prefix("--tokens="))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1000);

    // Repeat to get target tokens (each repeat adds ~375 tokens)
    let repeats = (target_tokens / 375).max(1);
    let long_prompt = base_text.repeat(repeats);

    // Tokenize
    let encoding = tokenizer
        .encode(long_prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_len = prompt_tokens.len();

    println!("\n{}", "=".repeat(60));
    println!("Prompt length: {} tokens", prompt_len);
    println!("Generating: 50 tokens");
    println!("{}", "=".repeat(60));

    let mut tokens = prompt_tokens.clone();
    let max_new_tokens = 50;
    let eos_token_id = config.eos_token_id.unwrap_or(151645);

    // Prefill (process entire prompt)
    println!("\nPrefill phase ({} tokens)...", prompt_len);
    let prefill_start = Instant::now();

    let input = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;

    let prefill_time = prefill_start.elapsed();
    println!(
        "  Prefill: {:.2}s ({:.1} tokens/s)",
        prefill_time.as_secs_f64(),
        prompt_len as f64 / prefill_time.as_secs_f64()
    );

    // Get first generated token
    let seq_len = logits.dim(1)?;
    let last_logits = logits.i((0, seq_len - 1, ..))?;
    let next_token = last_logits.argmax(0)?.to_scalar::<u32>()?;
    tokens.push(next_token);

    // Decode phase (generate remaining tokens)
    println!("\nDecode phase ({} tokens)...", max_new_tokens - 1);
    let decode_start = Instant::now();
    let mut generated = vec![next_token];

    for i in 1..max_new_tokens {
        let input = Tensor::new(&[tokens[tokens.len() - 1]], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, tokens.len() - 1)?;

        let last_logits = logits.i((0, 0, ..))?;
        let next_token = last_logits.argmax(0)?.to_scalar::<u32>()?;

        if next_token == eos_token_id {
            println!("  [EOS at token {}]", i);
            break;
        }

        tokens.push(next_token);
        generated.push(next_token);
    }

    let decode_time = decode_start.elapsed();
    let decode_tps = (generated.len() - 1) as f64 / decode_time.as_secs_f64();

    println!(
        "  Decode: {:.2}s ({:.1} tokens/s)",
        decode_time.as_secs_f64(),
        decode_tps
    );

    // Summary
    let total_time = prefill_time + decode_time;
    println!("\n{}", "=".repeat(60));
    println!(
        "RESULTS ({}):",
        if use_flash {
            "Flash Attention"
        } else {
            "Standard Attention"
        }
    );
    println!(
        "  Prefill ({} tokens): {:.2}s ({:.1} tok/s)",
        prompt_len,
        prefill_time.as_secs_f64(),
        prompt_len as f64 / prefill_time.as_secs_f64()
    );
    println!(
        "  Decode ({} tokens):  {:.2}s ({:.1} tok/s)",
        generated.len(),
        decode_time.as_secs_f64(),
        decode_tps
    );
    println!("  Total time: {:.2}s", total_time.as_secs_f64());
    println!("{}", "=".repeat(60));

    // Show generated text
    let generated_text = tokenizer
        .decode(&generated, false)
        .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))?;
    println!("\nGenerated text:\n{}", generated_text);

    Ok(())
}
