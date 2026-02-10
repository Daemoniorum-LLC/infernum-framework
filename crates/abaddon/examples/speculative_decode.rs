//! Speculative decoding benchmark for INT4 models.
//!
//! This example demonstrates speculative decoding concepts, but note that
//! **same-model speculative decoding is NOT faster** than standard decoding.
//!
//! The math:
//! - Standard: N forward passes for N tokens
//! - Speculative (K=5, 85% acceptance): N/4.26 × 6 = ~1.4N forward passes (SLOWER)
//!
//! Speculative decoding only provides speedup when:
//! 1. Using a SEPARATE smaller draft model (e.g., 0.5B draft → 7B target)
//! 2. Using self-speculative techniques (early exit layers)
//! 3. Using Medusa-style parallel prediction heads
//!
//! For single-model scenarios, better optimizations are:
//! - Flash Attention (36-163% speedup, already implemented)
//! - KV cache quantization (INT8 cache for longer context)
//! - Continuous batching (for multi-user scenarios)
//!
//! Usage:
//!   cargo run --release -p abaddon --example speculative_decode --features cuda
//!   cargo run --release -p abaddon --example speculative_decode --features cuda -- --no-spec
//!   cargo run --release -p abaddon --example speculative_decode --features cuda -- --draft=3
//!
//! The key insight: verify multiple draft tokens in one forward pass (like prefill)
//! instead of generating them one by one.

use std::path::Path;
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use abaddon::hct_sequential::load_hct_directory_parallel;
use abaddon::models::qwen2::{Qwen2, Qwen2Config};

/// Simple speculative decoder using greedy sampling.
struct SpeculativeDecoder {
    /// Number of draft tokens to generate per round.
    num_draft_tokens: usize,
}

impl SpeculativeDecoder {
    fn new(num_draft_tokens: usize) -> Self {
        Self { num_draft_tokens }
    }

    /// Generate tokens using speculative decoding.
    ///
    /// Algorithm:
    /// 1. Generate K draft tokens using single-token forward passes
    /// 2. Verify all K tokens in one batched forward pass
    /// 3. Accept matching tokens, resample on first mismatch
    fn generate(
        &self,
        model: &mut Qwen2,
        _tokenizer: &Tokenizer,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        eos_token_id: u32,
        device: &Device,
    ) -> anyhow::Result<SpeculativeResult> {
        let mut stats = SpeculativeStats::default();
        let start = Instant::now();

        // We need to manage without relying on KV cache for this simple implementation
        // Full context will be passed each time (not optimal but correct)

        let prefill_time = start.elapsed();
        stats.prefill_time_ms = prefill_time.as_millis() as u64;

        let decode_start = Instant::now();
        let mut generated = Vec::new();

        while generated.len() < max_new_tokens {
            stats.rounds += 1;

            // Build current full context
            let full_context: Vec<u32> = prompt_tokens
                .iter()
                .chain(generated.iter())
                .copied()
                .collect();

            // Step 1: Generate draft tokens greedily (one at a time)
            let mut draft_tokens = Vec::with_capacity(self.num_draft_tokens);

            // First draft token from current context
            model.clear_cache();
            let input = Tensor::new(&full_context[..], device)?.unsqueeze(0)?;
            let logits = model.forward(&input, 0)?;
            let seq_len = logits.dim(1)?;
            let mut draft_logits = logits.i((0, seq_len - 1, ..))?;

            for i in 0..self.num_draft_tokens {
                let draft_token = draft_logits.argmax(0)?.to_scalar::<u32>()?;

                if draft_token == eos_token_id {
                    draft_tokens.push(draft_token);
                    break;
                }

                draft_tokens.push(draft_token);

                // Generate next draft token
                if i < self.num_draft_tokens - 1 {
                    let pos = full_context.len() + i;
                    let input = Tensor::new(&[draft_token], device)?.unsqueeze(0)?;
                    let logits = model.forward(&input, pos)?;
                    draft_logits = logits.i((0, 0, ..))?;
                }
            }

            if draft_tokens.is_empty() {
                break;
            }

            stats.draft_tokens += draft_tokens.len() as u64;

            // Step 2: Verify all draft tokens in one forward pass
            // Clear cache and process full context + all draft tokens together
            model.clear_cache();

            let verify_context: Vec<u32> = full_context
                .iter()
                .chain(draft_tokens.iter())
                .copied()
                .collect();

            let verify_input = Tensor::new(&verify_context[..], device)?.unsqueeze(0)?;
            let verify_logits = model.forward(&verify_input, 0)?;

            // Step 3: Accept/reject draft tokens
            // Position i in full_context predicts token at position i+1
            // So verify_logits[:, full_context.len()-1, :] predicts draft_tokens[0]
            let base_pos = full_context.len() - 1;
            let mut accepted = 0;

            for (i, &draft_token) in draft_tokens.iter().enumerate() {
                let pos_logits = verify_logits.i((0, base_pos + i, ..))?;
                let verified_token = pos_logits.argmax(0)?.to_scalar::<u32>()?;

                if verified_token == draft_token {
                    accepted += 1;
                    if draft_token == eos_token_id {
                        stats.accepted_tokens += accepted as u64;
                        stats.decode_time_ms = decode_start.elapsed().as_millis() as u64;
                        return Ok(SpeculativeResult {
                            tokens: generated,
                            stats,
                        });
                    }
                    generated.push(draft_token);
                } else {
                    // Reject - use verified token instead
                    stats.rejected_tokens += (draft_tokens.len() - accepted) as u64;
                    if verified_token == eos_token_id {
                        stats.accepted_tokens += accepted as u64;
                        stats.decode_time_ms = decode_start.elapsed().as_millis() as u64;
                        return Ok(SpeculativeResult {
                            tokens: generated,
                            stats,
                        });
                    }
                    generated.push(verified_token);
                    break;
                }
            }

            stats.accepted_tokens += accepted as u64;

            // If all accepted, get next token from last position
            if accepted == draft_tokens.len() && !draft_tokens.is_empty() {
                let next_pos = base_pos + draft_tokens.len();
                if next_pos < verify_logits.dim(1)? {
                    let next_logits = verify_logits.i((0, next_pos, ..))?;
                    let next_token = next_logits.argmax(0)?.to_scalar::<u32>()?;
                    if next_token == eos_token_id {
                        break;
                    }
                    generated.push(next_token);
                }
            }
        }

        stats.decode_time_ms = decode_start.elapsed().as_millis() as u64;

        Ok(SpeculativeResult {
            tokens: generated,
            stats,
        })
    }
}

/// Standard greedy decoding for comparison.
fn generate_standard(
    model: &mut Qwen2,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    eos_token_id: u32,
    device: &Device,
) -> anyhow::Result<StandardResult> {
    let start = Instant::now();

    // Prefill
    let input = Tensor::new(&prompt_tokens[..], device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;
    let seq_len = logits.dim(1)?;
    let last_logits = logits.i((0, seq_len - 1, ..))?;

    let prefill_time = start.elapsed();

    let decode_start = Instant::now();
    let mut generated = Vec::new();
    let mut tokens = prompt_tokens.to_vec();
    let mut next_token = last_logits.argmax(0)?.to_scalar::<u32>()?;

    while generated.len() < max_new_tokens && next_token != eos_token_id {
        generated.push(next_token);
        tokens.push(next_token);

        let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, tokens.len() - 1)?;
        let last_logits = logits.i((0, 0, ..))?;
        next_token = last_logits.argmax(0)?.to_scalar::<u32>()?;
    }

    let decode_time = decode_start.elapsed();

    Ok(StandardResult {
        tokens: generated,
        prefill_time_ms: prefill_time.as_millis() as u64,
        decode_time_ms: decode_time.as_millis() as u64,
    })
}

#[derive(Debug, Default)]
struct SpeculativeStats {
    rounds: u64,
    draft_tokens: u64,
    accepted_tokens: u64,
    rejected_tokens: u64,
    prefill_time_ms: u64,
    decode_time_ms: u64,
}

impl SpeculativeStats {
    fn acceptance_rate(&self) -> f32 {
        if self.draft_tokens == 0 {
            0.0
        } else {
            self.accepted_tokens as f32 / self.draft_tokens as f32
        }
    }

    fn tokens_per_round(&self) -> f32 {
        if self.rounds == 0 {
            0.0
        } else {
            self.accepted_tokens as f32 / self.rounds as f32
        }
    }
}

struct SpeculativeResult {
    tokens: Vec<u32>,
    stats: SpeculativeStats,
}

struct StandardResult {
    tokens: Vec<u32>,
    prefill_time_ms: u64,
    decode_time_ms: u64,
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let use_speculative = !args.contains(&"--no-spec".to_string());
    let num_draft = args
        .iter()
        .find(|a| a.starts_with("--draft="))
        .and_then(|a| a.strip_prefix("--draft="))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(5);

    println!("=== Speculative Decoding Benchmark ===\n");

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
        "Mode: {}",
        if use_speculative {
            format!("Speculative (K={})", num_draft)
        } else {
            "Standard".to_string()
        }
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
    let mut model = Qwen2::load_with_flash_attention(config.clone(), vb)?;
    println!("  Model ready!");

    // Test prompt
    let prompt = "The future of artificial intelligence is";
    let max_tokens = 100;
    let eos_token_id = config.eos_token_id.unwrap_or(151645);

    println!("\n{}", "=".repeat(60));
    println!("Prompt: \"{}\"", prompt);
    println!("Max tokens: {}", max_tokens);
    println!("{}", "=".repeat(60));

    // Tokenize
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
    println!("\nPrompt tokens: {}", prompt_tokens.len());

    if use_speculative {
        // Speculative decoding
        let decoder = SpeculativeDecoder::new(num_draft);
        let result = decoder.generate(
            &mut model,
            &tokenizer,
            &prompt_tokens,
            max_tokens,
            eos_token_id,
            &device,
        )?;

        let decode_tps = result.tokens.len() as f64 / (result.stats.decode_time_ms as f64 / 1000.0);

        // Calculate overhead analysis
        let tokens_per_round = result.stats.tokens_per_round();
        let forward_passes_per_round = num_draft as f32 + 1.0 + 1.0; // K drafts + 1 verify + 1 context
        let effective_passes = result.stats.rounds as f32 * forward_passes_per_round;
        let standard_passes = result.tokens.len() as f32;
        let overhead_ratio = effective_passes / standard_passes;

        println!("\n{}", "=".repeat(60));
        println!("SPECULATIVE DECODING RESULTS:");
        println!("  Rounds: {}", result.stats.rounds);
        println!("  Draft tokens: {}", result.stats.draft_tokens);
        println!(
            "  Accepted: {} ({:.1}%)",
            result.stats.accepted_tokens,
            result.stats.acceptance_rate() * 100.0
        );
        println!("  Rejected: {}", result.stats.rejected_tokens);
        println!("  Tokens per round: {:.2}", tokens_per_round);
        println!(
            "  Decode: {}ms ({:.1} tok/s)",
            result.stats.decode_time_ms, decode_tps
        );
        println!("  Generated: {} tokens", result.tokens.len());
        println!("{}", "=".repeat(60));
        println!("\nOVERHEAD ANALYSIS:");
        println!(
            "  Standard would use: {:.0} forward passes",
            standard_passes
        );
        println!(
            "  Speculative used:   {:.0} forward passes",
            effective_passes
        );
        println!(
            "  Overhead ratio:     {:.2}x (>1.0 = slower)",
            overhead_ratio
        );
        println!(
            "\n  NOTE: Same-model speculative is {}",
            if overhead_ratio > 1.0 {
                "SLOWER (expected)"
            } else {
                "faster (unexpected)"
            }
        );
        println!("  For actual speedup, use a separate smaller draft model.");
        println!("{}", "=".repeat(60));

        let text = tokenizer
            .decode(&result.tokens, false)
            .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))?;
        println!("\nGenerated text:\n{}{}", prompt, text);
    } else {
        // Standard decoding
        let result = generate_standard(
            &mut model,
            &prompt_tokens,
            max_tokens,
            eos_token_id,
            &device,
        )?;

        let decode_tps = result.tokens.len() as f64 / (result.decode_time_ms as f64 / 1000.0);

        println!("\n{}", "=".repeat(60));
        println!("STANDARD DECODING RESULTS:");
        println!("  Prefill: {}ms", result.prefill_time_ms);
        println!(
            "  Decode: {}ms ({:.1} tok/s)",
            result.decode_time_ms, decode_tps
        );
        println!("  Generated: {} tokens", result.tokens.len());
        println!("{}", "=".repeat(60));

        let text = tokenizer
            .decode(&result.tokens, false)
            .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))?;
        println!("\nGenerated text:\n{}{}", prompt, text);
    }

    Ok(())
}
