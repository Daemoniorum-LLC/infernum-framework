//! Compare single tensor: original safetensors vs HCT compressed

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use safetensors::SafeTensors;

use abaddon::hct_sequential::load_hct_directory_sequential;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

fn main() -> anyhow::Result<()> {
    let safetensors_path = Path::new("/home/crook/models/llama-3.2-1b/model.safetensors");
    let hct_dir = Path::new("/home/crook/models/llama-3.2-1b-hct-45pct");

    let device = Device::Cpu;

    // Load original safetensors
    println!("Loading original safetensors...");
    let file_content = std::fs::read(safetensors_path)?;
    let st = SafeTensors::deserialize(&file_content)?;

    // Load HCT tensors
    println!("Loading HCT tensors...");
    let hct_tensors = load_hct_directory_sequential(hct_dir, &device, DType::F32)?;
    println!("Loaded {} HCT tensors", hct_tensors.len());

    // Compare a few specific tensors
    let test_names = [
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.8.mlp.gate_proj.weight",
    ];

    for name in &test_names {
        println!("\n=== {} ===", name);

        // Get original tensor
        let st_tensor = st.tensor(name)?;
        let shape: Vec<usize> = st_tensor.shape().to_vec();
        let data = st_tensor.data();

        // Convert BF16 to F32
        let original: Vec<f32> = data
            .chunks_exact(2)
            .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
            .collect();

        println!("  Original shape: {:?}", shape);
        println!(
            "  Original range: [{:.6}, {:.6}]",
            original.iter().cloned().fold(f32::INFINITY, f32::min),
            original.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );

        // Get HCT tensor
        if let Some(hct_tensor) = hct_tensors.get(*name) {
            let hct_flat = hct_tensor.flatten_all()?;
            let hct_values: Vec<f32> = hct_flat.to_vec1()?;

            println!("  HCT shape: {:?}", hct_tensor.dims());
            println!(
                "  HCT range: [{:.6}, {:.6}]",
                hct_values.iter().cloned().fold(f32::INFINITY, f32::min),
                hct_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );

            // Check size match
            if original.len() != hct_values.len() {
                println!(
                    "  SIZE MISMATCH: {} vs {}",
                    original.len(),
                    hct_values.len()
                );
                continue;
            }

            // Compute similarity
            let similarity = cosine_similarity(&original, &hct_values);
            println!("  Cosine similarity: {:.6}", similarity);

            // Sample values
            println!("  Sample values (first 5):");
            for i in 0..5 {
                println!(
                    "    [{:4}] Original: {:12.6} HCT: {:12.6} diff: {:12.6}",
                    i,
                    original[i],
                    hct_values[i],
                    (original[i] - hct_values[i]).abs()
                );
            }

            // Check for transposition
            // If shape is [A, B], check if values at (0, j) match either row-major or col-major
            if shape.len() == 2 {
                let rows = shape[0];
                let cols = shape[1];
                println!("\n  Checking for transpose (shape [{}, {}]):", rows, cols);

                // Row-major: element at (i,j) is at index i*cols+j
                // Col-major: element at (i,j) is at index j*rows+i

                // Sample row-major vs col-major comparison
                let mut row_major_matches = 0;
                let mut col_major_matches = 0;
                let samples = 100.min(rows).min(cols);

                for test_idx in 0..samples {
                    let i = test_idx % rows;
                    let j = test_idx % cols;

                    let row_major_idx = i * cols + j;
                    let col_major_idx = j * rows + i;

                    if (original[row_major_idx] - hct_values[row_major_idx]).abs() < 0.01 {
                        row_major_matches += 1;
                    }
                    if col_major_idx < original.len() && col_major_idx < hct_values.len() {
                        if (original[row_major_idx] - hct_values[col_major_idx]).abs() < 0.01 {
                            col_major_matches += 1;
                        }
                    }
                }
                println!("    Row-major matches: {}/{}", row_major_matches, samples);
                println!("    Col-major matches: {}/{}", col_major_matches, samples);
            }
        } else {
            println!("  NOT FOUND in HCT!");
        }
    }

    Ok(())
}
