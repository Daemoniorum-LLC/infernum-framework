//! Test error accumulation through single transformer layer

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{Linear, Module, RmsNorm, VarBuilder};
use safetensors::SafeTensors;

use abaddon::hct_sequential::load_hct_directory_sequential;
use anyhow::Result;

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
                _ => continue,
            };

            tensors.insert(tensor_name, tensor);
        }
    }

    Ok(tensors)
}

/// Simple MLP forward pass for testing
fn mlp_forward(
    x: &Tensor,
    gate_proj: &Tensor,
    up_proj: &Tensor,
    down_proj: &Tensor,
) -> Result<Tensor> {
    // gate = silu(x @ gate_proj.T)
    // up = x @ up_proj.T
    // out = (gate * up) @ down_proj.T

    let gate = x.matmul(&gate_proj.t()?)?;
    let gate = candle_nn::ops::silu(&gate)?;

    let up = x.matmul(&up_proj.t()?)?;

    let hidden = (gate * up)?;
    let out = hidden.matmul(&down_proj.t()?)?;

    Ok(out)
}

fn main() -> Result<()> {
    let safetensors_path = Path::new("/home/crook/models/llama-3.2-1b/model.safetensors");
    let hct_dir = Path::new("/home/crook/models/llama-3.2-1b-hct-45pct");

    let device = Device::Cpu;
    let dtype = DType::F32;

    println!("Loading tensors...");
    let orig_tensors = load_safetensors(safetensors_path, &device)?;
    let hct_tensors = load_hybrid(hct_dir, safetensors_path, &device, dtype)?;

    // Get embedding for initial hidden state
    let embed_orig = orig_tensors.get("model.embed_tokens.weight").unwrap();
    let embed_hct = hct_tensors.get("model.embed_tokens.weight").unwrap();

    let test_token = Tensor::new(&[9906u32], &device)?; // "Hello"

    let hidden_orig = embed_orig.index_select(&test_token, 0)?; // [1, 2048]
    let hidden_hct = embed_hct.index_select(&test_token, 0)?;

    let hidden_orig_flat: Vec<f32> = hidden_orig.flatten_all()?.to_vec1()?;
    let hidden_hct_flat: Vec<f32> = hidden_hct.flatten_all()?.to_vec1()?;

    println!("\n=== Initial Hidden State ===");
    println!(
        "Cosine similarity: {:.6}",
        cosine_similarity(&hidden_orig_flat, &hidden_hct_flat)
    );

    // Test MLP layer 0
    println!("\n=== MLP Layer 0 Test ===");

    let gate_orig = orig_tensors
        .get("model.layers.0.mlp.gate_proj.weight")
        .unwrap();
    let up_orig = orig_tensors
        .get("model.layers.0.mlp.up_proj.weight")
        .unwrap();
    let down_orig = orig_tensors
        .get("model.layers.0.mlp.down_proj.weight")
        .unwrap();

    let gate_hct = hct_tensors
        .get("model.layers.0.mlp.gate_proj.weight")
        .unwrap();
    let up_hct = hct_tensors
        .get("model.layers.0.mlp.up_proj.weight")
        .unwrap();
    let down_hct = hct_tensors
        .get("model.layers.0.mlp.down_proj.weight")
        .unwrap();

    // Test with original hidden state
    let mlp_out_orig = mlp_forward(&hidden_orig, gate_orig, up_orig, down_orig)?;
    let mlp_out_hct = mlp_forward(&hidden_orig, gate_hct, up_hct, down_hct)?;

    let mlp_orig_flat: Vec<f32> = mlp_out_orig.flatten_all()?.to_vec1()?;
    let mlp_hct_flat: Vec<f32> = mlp_out_hct.flatten_all()?.to_vec1()?;

    println!(
        "MLP output cosine (orig hidden, orig vs hct weights): {:.6}",
        cosine_similarity(&mlp_orig_flat, &mlp_hct_flat)
    );
    println!("MLP output stats:");
    println!(
        "  Original: mean={:.6}, std={:.6}",
        mlp_orig_flat.iter().sum::<f32>() / mlp_orig_flat.len() as f32,
        (mlp_orig_flat.iter().map(|x| x.powi(2)).sum::<f32>() / mlp_orig_flat.len() as f32).sqrt()
    );
    println!(
        "  HCT:      mean={:.6}, std={:.6}",
        mlp_hct_flat.iter().sum::<f32>() / mlp_hct_flat.len() as f32,
        (mlp_hct_flat.iter().map(|x| x.powi(2)).sum::<f32>() / mlp_hct_flat.len() as f32).sqrt()
    );

    // Test with HCT hidden state
    let mlp_out_both_hct = mlp_forward(&hidden_hct, gate_hct, up_hct, down_hct)?;
    let mlp_both_hct_flat: Vec<f32> = mlp_out_both_hct.flatten_all()?.to_vec1()?;

    println!(
        "\nMLP output cosine (hct hidden + hct weights vs orig): {:.6}",
        cosine_similarity(&mlp_orig_flat, &mlp_both_hct_flat)
    );

    // Test cumulative error through multiple layers
    println!("\n=== Cumulative Error Through Layers ===");

    let mut hidden_o = hidden_orig.clone();
    let mut hidden_h = hidden_hct.clone();

    for layer_idx in 0..16 {
        let gate_o = orig_tensors
            .get(&format!("model.layers.{}.mlp.gate_proj.weight", layer_idx))
            .unwrap();
        let up_o = orig_tensors
            .get(&format!("model.layers.{}.mlp.up_proj.weight", layer_idx))
            .unwrap();
        let down_o = orig_tensors
            .get(&format!("model.layers.{}.mlp.down_proj.weight", layer_idx))
            .unwrap();

        let gate_h = hct_tensors
            .get(&format!("model.layers.{}.mlp.gate_proj.weight", layer_idx))
            .unwrap();
        let up_h = hct_tensors
            .get(&format!("model.layers.{}.mlp.up_proj.weight", layer_idx))
            .unwrap();
        let down_h = hct_tensors
            .get(&format!("model.layers.{}.mlp.down_proj.weight", layer_idx))
            .unwrap();

        // Simple: just pass through MLP (skip attention for now)
        let mlp_out_o = mlp_forward(&hidden_o, gate_o, up_o, down_o)?;
        let mlp_out_h = mlp_forward(&hidden_h, gate_h, up_h, down_h)?;

        // Residual connection
        hidden_o = (&hidden_o + &mlp_out_o)?;
        hidden_h = (&hidden_h + &mlp_out_h)?;

        let ho_flat: Vec<f32> = hidden_o.flatten_all()?.to_vec1()?;
        let hh_flat: Vec<f32> = hidden_h.flatten_all()?.to_vec1()?;

        let sim = cosine_similarity(&ho_flat, &hh_flat);

        println!("After layer {} MLP: cosine={:.6}", layer_idx, sim);
    }

    Ok(())
}
