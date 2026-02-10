//! Test original safetensors vs 45% compressed - baseline comparison

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use safetensors::SafeTensors;

use abaddon::hct_sequential::load_hct_directory_sequential;
use abaddon::models::{Llama, LlamaConfig};
use anyhow::Result;

fn get_config() -> LlamaConfig {
    LlamaConfig {
        hidden_size: 2048,
        intermediate_size: 8192,
        vocab_size: 128256,
        num_hidden_layers: 16,
        num_attention_heads: 32,
        num_key_value_heads: Some(8),
        rms_norm_eps: 1e-5,
        rope_theta: 500000.0,
        max_position_embeddings: 131072,
        tie_word_embeddings: true,
        bos_token_id: Some(128000),
        eos_token_id: Some(128001),
        rope_scaling: None,
    }
}

fn load_safetensors(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let file_content = std::fs::read(path)?;
    let st = SafeTensors::deserialize(&file_content)?;

    let mut tensors = HashMap::new();

    for name in st.names() {
        let st_tensor = st.tensor(name)?;
        let shape: Vec<usize> = st_tensor.shape().to_vec();
        let data = st_tensor.data();

        let tensor = match st_tensor.dtype() {
            safetensors::Dtype::BF16 => {
                let halfs: Vec<half::bf16> = data
                    .chunks_exact(2)
                    .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                let floats: Vec<f32> = halfs.iter().map(|h| h.to_f32()).collect();
                Tensor::from_vec(floats, shape.as_slice(), device)?
            },
            safetensors::Dtype::F32 => {
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Tensor::from_vec(floats, shape.as_slice(), device)?
            },
            safetensors::Dtype::F16 => {
                let halfs: Vec<half::f16> = data
                    .chunks_exact(2)
                    .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                let floats: Vec<f32> = halfs.iter().map(|h| h.to_f32()).collect();
                Tensor::from_vec(floats, shape.as_slice(), device)?
            },
            _ => continue,
        };

        tensors.insert(name.to_string(), tensor);
    }

    Ok(tensors)
}

fn load_hybrid(
    hct_dir: &Path,
    safetensors_path: &Path,
    device: &Device,
    dtype: DType,
) -> Result<HashMap<String, Tensor>> {
    let mut tensors = load_hct_directory_sequential(hct_dir, device, dtype)?;
    let file_content = std::fs::read(safetensors_path)?;
    let st = SafeTensors::deserialize(&file_content)?;

    for name in st.names() {
        let tensor_name = name.to_string();
        if !tensors.contains_key(&tensor_name) {
            let st_tensor = st.tensor(&tensor_name)?;
            let shape: Vec<usize> = st_tensor.shape().to_vec();
            let data = st_tensor.data();

            let tensor = match st_tensor.dtype() {
                safetensors::Dtype::BF16 => {
                    let halfs: Vec<half::bf16> = data
                        .chunks_exact(2)
                        .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]))
                        .collect();
                    let floats: Vec<f32> = halfs.iter().map(|h| h.to_f32()).collect();
                    Tensor::from_vec(floats, shape.as_slice(), device)?
                },
                safetensors::Dtype::F32 => {
                    let floats: Vec<f32> = data
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();
                    Tensor::from_vec(floats, shape.as_slice(), device)?
                },
                safetensors::Dtype::F16 => {
                    let halfs: Vec<half::f16> = data
                        .chunks_exact(2)
                        .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]))
                        .collect();
                    let floats: Vec<f32> = halfs.iter().map(|h| h.to_f32()).collect();
                    Tensor::from_vec(floats, shape.as_slice(), device)?
                },
                _ => continue,
            };

            tensors.insert(tensor_name, tensor);
        }
    }

    Ok(tensors)
}

fn run_inference(
    tensors: HashMap<String, Tensor>,
    name: &str,
    device: &Device,
    dtype: DType,
) -> Result<(Vec<u32>, Vec<f32>)> {
    println!("\n=== {} ===", name);

    let vb = VarBuilder::from_tensors(tensors, dtype, device);
    let mut model = Llama::load(get_config(), vb)?;

    let test_tokens = vec![128000u32, 9906u32, 11u32, 358u32]; // BOS + "Hello, I"
    let input_ids = Tensor::new(&test_tokens[..], device)?.unsqueeze(0)?;

    let start = Instant::now();
    let logits = model.forward(&input_ids, 0)?;
    println!("  Forward pass: {:.3}s", start.elapsed().as_secs_f64());

    let last_logits = logits.i((0, logits.dim(1)? - 1, ..))?;
    let logits_vec: Vec<f32> = last_logits.to_vec1()?;

    let mut indexed: Vec<(u32, f32)> = logits_vec
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_tokens: Vec<u32> = indexed.iter().take(10).map(|(t, _)| *t).collect();

    println!("  Top 5 predictions:");
    for i in 0..5 {
        let (token, score) = indexed[i];
        println!("    {}. Token {} (score: {:.4})", i + 1, token, score);
    }

    Ok((top_tokens, logits_vec))
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

fn main() -> Result<()> {
    println!("=== Baseline Quality Test ===\n");

    let safetensors_path = Path::new("/home/crook/models/llama-3.2-1b/model.safetensors");
    let compressed_dir = Path::new("/home/crook/models/llama-3.2-1b-hct-45pct");

    let device = Device::Cpu;
    let dtype = DType::F32;

    // Load original model from safetensors
    println!("Loading original safetensors...");
    let start = Instant::now();
    let original_tensors = load_safetensors(safetensors_path, &device)?;
    println!(
        "Loaded {} tensors in {:.2}s",
        original_tensors.len(),
        start.elapsed().as_secs_f64()
    );
    let (original_tokens, original_logits) =
        run_inference(original_tensors, "Original (safetensors)", &device, dtype)?;

    // Load 45% compressed hybrid
    println!("\nLoading 45% compressed hybrid...");
    let start = Instant::now();
    let compressed_tensors = load_hybrid(compressed_dir, safetensors_path, &device, dtype)?;
    println!(
        "Loaded {} tensors in {:.2}s",
        compressed_tensors.len(),
        start.elapsed().as_secs_f64()
    );
    let (compressed_tokens, compressed_logits) =
        run_inference(compressed_tensors, "45% Retention (hybrid)", &device, dtype)?;

    // Compare
    println!("\n=== Quality Comparison ===");

    let mut matching = 0;
    for i in 0..10.min(original_tokens.len()).min(compressed_tokens.len()) {
        if original_tokens[i] == compressed_tokens[i] {
            matching += 1;
        }
    }

    let similarity = cosine_similarity(&original_logits, &compressed_logits);

    println!("  Top-10 token agreement: {}/10", matching);
    println!(
        "  Top-1 match: {}",
        if original_tokens[0] == compressed_tokens[0] {
            "✓"
        } else {
            "✗"
        }
    );
    println!("  Logit cosine similarity: {:.6}", similarity);

    println!("\n=== Summary ===");
    if similarity > 0.95 {
        println!(
            "  Result: EXCELLENT - High similarity ({:.2}%)",
            similarity * 100.0
        );
    } else if similarity > 0.90 {
        println!(
            "  Result: GOOD - Moderate similarity ({:.2}%)",
            similarity * 100.0
        );
    } else if similarity > 0.80 {
        println!(
            "  Result: ACCEPTABLE - Lower similarity ({:.2}%)",
            similarity * 100.0
        );
    } else {
        println!(
            "  Result: POOR - Low similarity ({:.2}%)",
            similarity * 100.0
        );
    }

    Ok(())
}
