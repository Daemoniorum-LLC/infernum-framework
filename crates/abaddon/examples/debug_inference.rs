//! Debug inference step by step to find where quality degrades

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, IndexOp, Module, Tensor};
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

fn main() -> Result<()> {
    let safetensors_path = Path::new("/home/crook/models/llama-3.2-1b/model.safetensors");
    let hct_dir = Path::new("/home/crook/models/llama-3.2-1b-hct-45pct");

    let device = Device::Cpu;
    let dtype = DType::F32;
    let config = get_config();

    // Load tensors
    println!("Loading original safetensors...");
    let original_tensors = load_safetensors(safetensors_path, &device)?;
    println!("Loading HCT hybrid...");
    let compressed_tensors = load_hybrid(hct_dir, safetensors_path, &device, dtype)?;

    // Compare specific critical tensors
    println!("\n=== Critical Tensor Comparison ===");

    let critical_tensors = [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.15.mlp.down_proj.weight", // Last layer
        "model.norm.weight",
    ];

    for name in &critical_tensors {
        let orig = original_tensors.get(*name);
        let comp = compressed_tensors.get(*name);

        match (orig, comp) {
            (Some(o), Some(c)) => {
                let o_flat: Vec<f32> = o.flatten_all()?.to_vec1()?;
                let c_flat: Vec<f32> = c.flatten_all()?.to_vec1()?;

                let sim = cosine_similarity(&o_flat, &c_flat);
                let o_mean: f32 = o_flat.iter().sum::<f32>() / o_flat.len() as f32;
                let c_mean: f32 = c_flat.iter().sum::<f32>() / c_flat.len() as f32;
                let o_std: f32 = (o_flat.iter().map(|x| (x - o_mean).powi(2)).sum::<f32>()
                    / o_flat.len() as f32)
                    .sqrt();
                let c_std: f32 = (c_flat.iter().map(|x| (x - c_mean).powi(2)).sum::<f32>()
                    / c_flat.len() as f32)
                    .sqrt();

                println!("{}", name);
                println!("  Shapes: {:?} vs {:?}", o.dims(), c.dims());
                println!("  Cosine: {:.6}", sim);
                println!("  Original: mean={:.6}, std={:.6}", o_mean, o_std);
                println!("  HCT:      mean={:.6}, std={:.6}", c_mean, c_std);

                // Check for NaN/Inf
                let o_nan = o_flat.iter().any(|x| x.is_nan());
                let o_inf = o_flat.iter().any(|x| x.is_infinite());
                let c_nan = c_flat.iter().any(|x| x.is_nan());
                let c_inf = c_flat.iter().any(|x| x.is_infinite());
                if o_nan || o_inf || c_nan || c_inf {
                    println!(
                        "  WARNING: NaN/Inf! orig_nan={}, orig_inf={}, hct_nan={}, hct_inf={}",
                        o_nan, o_inf, c_nan, c_inf
                    );
                }
            },
            _ => {
                println!(
                    "{}: MISSING (orig={}, comp={})",
                    name,
                    orig.is_some(),
                    comp.is_some()
                );
            },
        }
    }

    // Build models
    println!("\n=== Building models ===");
    let orig_vb = VarBuilder::from_tensors(original_tensors, dtype, &device);
    let comp_vb = VarBuilder::from_tensors(compressed_tensors, dtype, &device);

    let mut orig_model = Llama::load(config.clone(), orig_vb)?;
    let mut comp_model = Llama::load(config.clone(), comp_vb)?;

    // Run inference and compare at each step
    println!("\n=== Step-by-step comparison ===");
    let test_tokens = vec![128000u32, 9906u32, 11u32, 358u32];
    let input_ids = Tensor::new(&test_tokens[..], &device)?.unsqueeze(0)?;

    // Get logits
    let orig_logits = orig_model.forward(&input_ids, 0)?;
    let comp_logits = comp_model.forward(&input_ids, 0)?;

    // Compare final logits
    let orig_last = orig_logits.i((0, orig_logits.dim(1)? - 1, ..))?;
    let comp_last = comp_logits.i((0, comp_logits.dim(1)? - 1, ..))?;

    let orig_vec: Vec<f32> = orig_last.to_vec1()?;
    let comp_vec: Vec<f32> = comp_last.to_vec1()?;

    println!("\nFinal logits comparison:");
    println!(
        "  Original: min={:.4}, max={:.4}, mean={:.4}",
        orig_vec.iter().cloned().fold(f32::INFINITY, f32::min),
        orig_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        orig_vec.iter().sum::<f32>() / orig_vec.len() as f32
    );
    println!(
        "  HCT:      min={:.4}, max={:.4}, mean={:.4}",
        comp_vec.iter().cloned().fold(f32::INFINITY, f32::min),
        comp_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        comp_vec.iter().sum::<f32>() / comp_vec.len() as f32
    );

    let sim = cosine_similarity(&orig_vec, &comp_vec);
    println!("  Cosine similarity: {:.6}", sim);

    // Check for NaN
    let orig_nan = orig_vec.iter().any(|x| x.is_nan());
    let comp_nan = comp_vec.iter().any(|x| x.is_nan());
    println!("  NaN in orig: {}, NaN in comp: {}", orig_nan, comp_nan);

    // Top predictions comparison
    println!("\nTop 5 predictions:");
    let mut orig_indexed: Vec<(u32, f32)> = orig_vec
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    let mut comp_indexed: Vec<(u32, f32)> = comp_vec
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();

    orig_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    comp_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  Original:");
    for i in 0..5 {
        println!(
            "    {}. Token {} ({:.4})",
            i + 1,
            orig_indexed[i].0,
            orig_indexed[i].1
        );
    }
    println!("  HCT:");
    for i in 0..5 {
        println!(
            "    {}. Token {} ({:.4})",
            i + 1,
            comp_indexed[i].0,
            comp_indexed[i].1
        );
    }

    // Correlation analysis
    println!("\nCorrelation analysis:");
    let mut sorted_orig: Vec<(usize, f32)> =
        orig_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    let mut sorted_comp: Vec<(usize, f32)> =
        comp_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    sorted_orig.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    sorted_comp.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let orig_ranks: Vec<usize> = sorted_orig.iter().map(|(i, _)| *i).collect();
    let comp_ranks: Vec<usize> = sorted_comp.iter().map(|(i, _)| *i).collect();

    // Spearman correlation approximation
    let n = orig_ranks.len() as f32;
    let rank_diffs: f32 = orig_ranks
        .iter()
        .zip(comp_ranks.iter())
        .map(|(&a, &b)| ((a as i64 - b as i64).pow(2)) as f32)
        .sum();
    let spearman = 1.0 - (6.0 * rank_diffs) / (n * (n * n - 1.0));
    println!("  Spearman rank correlation: {:.6}", spearman);

    Ok(())
}
