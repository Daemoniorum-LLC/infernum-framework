//! Test attention mechanism specifically to find where quality degrades

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor, D};
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

/// Simple RoPE implementation for testing
fn apply_rope(x: &Tensor, seq_len: usize, head_dim: usize) -> Result<Tensor> {
    // Simplified RoPE - just rotate by position
    // For testing purposes, skip RoPE complexity
    Ok(x.clone())
}

/// Simple attention forward pass (no KV cache, single head for testing)
fn attention_forward(
    hidden: &Tensor, // [batch, seq, hidden]
    q_proj: &Tensor, // [hidden, hidden]
    k_proj: &Tensor, // [kv_dim, hidden]
    v_proj: &Tensor, // [kv_dim, hidden]
    o_proj: &Tensor, // [hidden, hidden]
    num_heads: usize,
    num_kv_heads: usize,
) -> Result<Tensor> {
    let (batch, seq_len, hidden_size) = hidden.dims3()?;
    let head_dim = hidden_size / num_heads;

    // Q, K, V projections (use broadcast_matmul for batched input)
    let q = hidden.broadcast_matmul(&q_proj.t()?)?; // [batch, seq, hidden]
    let k = hidden.broadcast_matmul(&k_proj.t()?)?; // [batch, seq, kv_dim]
    let v = hidden.broadcast_matmul(&v_proj.t()?)?; // [batch, seq, kv_dim]

    // Reshape for multi-head attention
    let q = q.reshape((batch, seq_len, num_heads, head_dim))?;
    let k = k.reshape((batch, seq_len, num_kv_heads, head_dim))?;
    let v = v.reshape((batch, seq_len, num_kv_heads, head_dim))?;

    // Transpose to [batch, heads, seq, head_dim]
    let q = q.transpose(1, 2)?.contiguous()?;
    let k = k.transpose(1, 2)?.contiguous()?;
    let v = v.transpose(1, 2)?.contiguous()?;

    // Expand K, V for GQA (repeat KV heads to match Q heads)
    let repeat_factor = num_heads / num_kv_heads;
    let k = k.repeat(&[1, repeat_factor, 1, 1])?;
    let v = v.repeat(&[1, repeat_factor, 1, 1])?;

    // Scaled dot-product attention
    let scale = (head_dim as f64).sqrt();
    let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?; // [batch, heads, seq, seq]
    let scores = (scores / scale)?;

    // Causal mask (lower triangular) - add large negative to positions to mask
    // Create [seq, seq] mask manually
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask_data[i * seq_len + j] = -1e10;
            }
        }
    }
    let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), hidden.device())?;
    let mask = mask.broadcast_as((batch, num_heads, seq_len, seq_len))?;
    let scores = scores.broadcast_add(&mask)?;

    // Softmax
    let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;

    // Apply attention to values
    let attn_output = attn_weights.matmul(&v)?; // [batch, heads, seq, head_dim]

    // Reshape back
    let attn_output = attn_output.transpose(1, 2)?.contiguous()?; // [batch, seq, heads, head_dim]
    let attn_output = attn_output.reshape((batch, seq_len, hidden_size))?;

    // Output projection
    let output = attn_output.broadcast_matmul(&o_proj.t()?)?;

    Ok(output)
}

fn main() -> Result<()> {
    let safetensors_path = Path::new("/home/crook/models/llama-3.2-1b/model.safetensors");
    let hct_dir = Path::new("/home/crook/models/llama-3.2-1b-hct-45pct");

    let device = Device::Cpu;
    let dtype = DType::F32;

    println!("Loading tensors...");
    let orig_tensors = load_safetensors(safetensors_path, &device)?;
    let hct_tensors = load_hybrid(hct_dir, safetensors_path, &device, dtype)?;

    // Llama 3.2 1B config
    let hidden_size = 2048;
    let num_heads = 32;
    let num_kv_heads = 8;
    let head_dim = hidden_size / num_heads;

    // Get initial hidden state from embedding
    let embed_orig = orig_tensors.get("model.embed_tokens.weight").unwrap();
    let test_tokens = Tensor::new(&[128000u32, 9906u32, 11u32, 358u32], &device)?; // BOS + "Hello, I"

    let hidden_orig = embed_orig.index_select(&test_tokens, 0)?; // [4, 2048]
    let hidden_orig = hidden_orig.unsqueeze(0)?; // [1, 4, 2048]

    let embed_hct = hct_tensors.get("model.embed_tokens.weight").unwrap();
    let hidden_hct = embed_hct.index_select(&test_tokens, 0)?.unsqueeze(0)?;

    println!("\n=== Attention Projection Tests ===");

    // Test individual projections
    for layer_idx in [0, 7, 15] {
        println!("\n--- Layer {} ---", layer_idx);

        let q_orig = orig_tensors
            .get(&format!(
                "model.layers.{}.self_attn.q_proj.weight",
                layer_idx
            ))
            .unwrap();
        let k_orig = orig_tensors
            .get(&format!(
                "model.layers.{}.self_attn.k_proj.weight",
                layer_idx
            ))
            .unwrap();
        let v_orig = orig_tensors
            .get(&format!(
                "model.layers.{}.self_attn.v_proj.weight",
                layer_idx
            ))
            .unwrap();
        let o_orig = orig_tensors
            .get(&format!(
                "model.layers.{}.self_attn.o_proj.weight",
                layer_idx
            ))
            .unwrap();

        let q_hct = hct_tensors
            .get(&format!(
                "model.layers.{}.self_attn.q_proj.weight",
                layer_idx
            ))
            .unwrap();
        let k_hct = hct_tensors
            .get(&format!(
                "model.layers.{}.self_attn.k_proj.weight",
                layer_idx
            ))
            .unwrap();
        let v_hct = hct_tensors
            .get(&format!(
                "model.layers.{}.self_attn.v_proj.weight",
                layer_idx
            ))
            .unwrap();
        let o_hct = hct_tensors
            .get(&format!(
                "model.layers.{}.self_attn.o_proj.weight",
                layer_idx
            ))
            .unwrap();

        // Check weight similarities
        let q_o_flat: Vec<f32> = q_orig.flatten_all()?.to_vec1()?;
        let q_h_flat: Vec<f32> = q_hct.flatten_all()?.to_vec1()?;
        println!(
            "Q weight cosine: {:.6}",
            cosine_similarity(&q_o_flat, &q_h_flat)
        );

        let k_o_flat: Vec<f32> = k_orig.flatten_all()?.to_vec1()?;
        let k_h_flat: Vec<f32> = k_hct.flatten_all()?.to_vec1()?;
        println!(
            "K weight cosine: {:.6}",
            cosine_similarity(&k_o_flat, &k_h_flat)
        );

        let v_o_flat: Vec<f32> = v_orig.flatten_all()?.to_vec1()?;
        let v_h_flat: Vec<f32> = v_hct.flatten_all()?.to_vec1()?;
        println!(
            "V weight cosine: {:.6}",
            cosine_similarity(&v_o_flat, &v_h_flat)
        );

        let o_o_flat: Vec<f32> = o_orig.flatten_all()?.to_vec1()?;
        let o_h_flat: Vec<f32> = o_hct.flatten_all()?.to_vec1()?;
        println!(
            "O weight cosine: {:.6}",
            cosine_similarity(&o_o_flat, &o_h_flat)
        );

        // Test Q projection output (using original hidden)
        // For batched matmul with [batch, seq, hidden] @ [hidden, out]
        let q_out_orig = hidden_orig.broadcast_matmul(&q_orig.t()?)?;
        let q_out_hct = hidden_orig.broadcast_matmul(&q_hct.t()?)?;
        let q_out_o_flat: Vec<f32> = q_out_orig.flatten_all()?.to_vec1()?;
        let q_out_h_flat: Vec<f32> = q_out_hct.flatten_all()?.to_vec1()?;
        println!(
            "Q projection output cosine: {:.6}",
            cosine_similarity(&q_out_o_flat, &q_out_h_flat)
        );

        // Test full attention output (using original hidden)
        let attn_out_orig = attention_forward(
            &hidden_orig,
            q_orig,
            k_orig,
            v_orig,
            o_orig,
            num_heads,
            num_kv_heads,
        )?;
        let attn_out_hct = attention_forward(
            &hidden_orig,
            q_hct,
            k_hct,
            v_hct,
            o_hct,
            num_heads,
            num_kv_heads,
        )?;

        let attn_o_flat: Vec<f32> = attn_out_orig.flatten_all()?.to_vec1()?;
        let attn_h_flat: Vec<f32> = attn_out_hct.flatten_all()?.to_vec1()?;
        println!(
            "Full attention output cosine: {:.6}",
            cosine_similarity(&attn_o_flat, &attn_h_flat)
        );
    }

    // Test cumulative attention through layers
    println!("\n=== Cumulative Error Through Attention Layers ===");

    let mut hidden_o = hidden_orig.clone();
    let mut hidden_h = hidden_hct.clone();

    for layer_idx in 0..16 {
        let q_o = orig_tensors
            .get(&format!(
                "model.layers.{}.self_attn.q_proj.weight",
                layer_idx
            ))
            .unwrap();
        let k_o = orig_tensors
            .get(&format!(
                "model.layers.{}.self_attn.k_proj.weight",
                layer_idx
            ))
            .unwrap();
        let v_o = orig_tensors
            .get(&format!(
                "model.layers.{}.self_attn.v_proj.weight",
                layer_idx
            ))
            .unwrap();
        let o_o = orig_tensors
            .get(&format!(
                "model.layers.{}.self_attn.o_proj.weight",
                layer_idx
            ))
            .unwrap();

        let q_h = hct_tensors
            .get(&format!(
                "model.layers.{}.self_attn.q_proj.weight",
                layer_idx
            ))
            .unwrap();
        let k_h = hct_tensors
            .get(&format!(
                "model.layers.{}.self_attn.k_proj.weight",
                layer_idx
            ))
            .unwrap();
        let v_h = hct_tensors
            .get(&format!(
                "model.layers.{}.self_attn.v_proj.weight",
                layer_idx
            ))
            .unwrap();
        let o_h = hct_tensors
            .get(&format!(
                "model.layers.{}.self_attn.o_proj.weight",
                layer_idx
            ))
            .unwrap();

        // Attention forward
        let attn_out_o = attention_forward(&hidden_o, q_o, k_o, v_o, o_o, num_heads, num_kv_heads)?;
        let attn_out_h = attention_forward(&hidden_h, q_h, k_h, v_h, o_h, num_heads, num_kv_heads)?;

        // Residual connection
        hidden_o = (&hidden_o + &attn_out_o)?;
        hidden_h = (&hidden_h + &attn_out_h)?;

        let ho_flat: Vec<f32> = hidden_o.flatten_all()?.to_vec1()?;
        let hh_flat: Vec<f32> = hidden_h.flatten_all()?.to_vec1()?;

        let sim = cosine_similarity(&ho_flat, &hh_flat);
        println!("After layer {} attention: cosine={:.6}", layer_idx, sim);
    }

    // Test attention + MLP combined
    println!("\n=== Cumulative Error Through Full Transformer Layers ===");

    hidden_o = hidden_orig.clone();
    hidden_h = hidden_hct.clone();

    for layer_idx in 0..16 {
        // Attention
        let q_o = orig_tensors
            .get(&format!(
                "model.layers.{}.self_attn.q_proj.weight",
                layer_idx
            ))
            .unwrap();
        let k_o = orig_tensors
            .get(&format!(
                "model.layers.{}.self_attn.k_proj.weight",
                layer_idx
            ))
            .unwrap();
        let v_o = orig_tensors
            .get(&format!(
                "model.layers.{}.self_attn.v_proj.weight",
                layer_idx
            ))
            .unwrap();
        let o_o = orig_tensors
            .get(&format!(
                "model.layers.{}.self_attn.o_proj.weight",
                layer_idx
            ))
            .unwrap();

        let q_h = hct_tensors
            .get(&format!(
                "model.layers.{}.self_attn.q_proj.weight",
                layer_idx
            ))
            .unwrap();
        let k_h = hct_tensors
            .get(&format!(
                "model.layers.{}.self_attn.k_proj.weight",
                layer_idx
            ))
            .unwrap();
        let v_h = hct_tensors
            .get(&format!(
                "model.layers.{}.self_attn.v_proj.weight",
                layer_idx
            ))
            .unwrap();
        let o_h = hct_tensors
            .get(&format!(
                "model.layers.{}.self_attn.o_proj.weight",
                layer_idx
            ))
            .unwrap();

        let attn_out_o = attention_forward(&hidden_o, q_o, k_o, v_o, o_o, num_heads, num_kv_heads)?;
        let attn_out_h = attention_forward(&hidden_h, q_h, k_h, v_h, o_h, num_heads, num_kv_heads)?;

        hidden_o = (&hidden_o + &attn_out_o)?;
        hidden_h = (&hidden_h + &attn_out_h)?;

        // MLP
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

        let gate_out_o = hidden_o.broadcast_matmul(&gate_o.t()?)?;
        let gate_out_o = candle_nn::ops::silu(&gate_out_o)?;
        let up_out_o = hidden_o.broadcast_matmul(&up_o.t()?)?;
        let mlp_out_o = (gate_out_o * up_out_o)?.broadcast_matmul(&down_o.t()?)?;

        let gate_out_h = hidden_h.broadcast_matmul(&gate_h.t()?)?;
        let gate_out_h = candle_nn::ops::silu(&gate_out_h)?;
        let up_out_h = hidden_h.broadcast_matmul(&up_h.t()?)?;
        let mlp_out_h = (gate_out_h * up_out_h)?.broadcast_matmul(&down_h.t()?)?;

        hidden_o = (&hidden_o + &mlp_out_o)?;
        hidden_h = (&hidden_h + &mlp_out_h)?;

        let ho_flat: Vec<f32> = hidden_o.flatten_all()?.to_vec1()?;
        let hh_flat: Vec<f32> = hidden_h.flatten_all()?.to_vec1()?;

        let sim = cosine_similarity(&ho_flat, &hh_flat);
        println!("After layer {} (attn+mlp): cosine={:.6}", layer_idx, sim);
    }

    Ok(())
}
