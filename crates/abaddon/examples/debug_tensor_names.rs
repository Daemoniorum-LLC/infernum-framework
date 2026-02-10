//! Debug tensor names and shapes between HCT and safetensors

use std::collections::{HashMap, HashSet};
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use safetensors::SafeTensors;

use abaddon::hct_sequential::load_hct_directory_sequential;
use anyhow::Result;

fn main() -> Result<()> {
    let safetensors_path = Path::new("/home/crook/models/llama-3.2-1b/model.safetensors");
    let hct_dir = Path::new("/home/crook/models/llama-3.2-1b-hct-45pct");

    let device = Device::Cpu;

    // Load safetensors names
    println!("=== Safetensors Tensor Names ===");
    let file_content = std::fs::read(safetensors_path)?;
    let st = SafeTensors::deserialize(&file_content)?;

    let st_names: HashSet<String> = st.names().into_iter().map(|s| s.to_string()).collect();
    println!("Total safetensors tensors: {}", st_names.len());

    let mut st_names_sorted: Vec<_> = st_names.iter().collect();
    st_names_sorted.sort();
    for (i, name) in st_names_sorted.iter().take(10).enumerate() {
        let tensor = st.tensor(name)?;
        println!("  {:3}. {} {:?}", i + 1, name, tensor.shape());
    }
    println!("  ... and {} more", st_names.len().saturating_sub(10));

    // Load HCT names
    println!("\n=== HCT Tensor Names ===");
    let hct_tensors = load_hct_directory_sequential(hct_dir, &device, DType::F32)?;

    let hct_names: HashSet<String> = hct_tensors.keys().cloned().collect();
    println!("Total HCT tensors: {}", hct_names.len());

    let mut hct_names_sorted: Vec<_> = hct_names.iter().collect();
    hct_names_sorted.sort();
    for (i, name) in hct_names_sorted.iter().take(10).enumerate() {
        let tensor = hct_tensors.get(*name).unwrap();
        println!("  {:3}. {} {:?}", i + 1, name, tensor.dims());
    }
    println!("  ... and {} more", hct_names.len().saturating_sub(10));

    // Find common tensors
    let common: HashSet<_> = st_names.intersection(&hct_names).collect();
    println!("\n=== Overlap ===");
    println!("Common tensors: {}", common.len());

    // Find tensors only in safetensors (missing from HCT)
    let only_in_st: HashSet<_> = st_names.difference(&hct_names).collect();
    println!("\n=== Only in Safetensors (missing from HCT) ===");
    println!("Count: {}", only_in_st.len());
    let mut missing_sorted: Vec<_> = only_in_st.iter().collect();
    missing_sorted.sort();
    for name in &missing_sorted {
        let tensor = st.tensor(name)?;
        println!("  {} {:?}", name, tensor.shape());
    }

    // Find tensors only in HCT (not in safetensors - shouldn't happen)
    let only_in_hct: HashSet<_> = hct_names.difference(&st_names).collect();
    if !only_in_hct.is_empty() {
        println!("\n=== Only in HCT (unexpected) ===");
        println!("Count: {}", only_in_hct.len());
        for name in &only_in_hct {
            println!("  {}", name);
        }
    }

    // Verify shapes match for common tensors
    println!("\n=== Shape Verification (first 5 common tensors) ===");
    let mut common_sorted: Vec<_> = common.iter().collect();
    common_sorted.sort();
    for name in common_sorted.iter().take(5) {
        let st_tensor = st.tensor(name)?;
        let hct_tensor = hct_tensors.get(**name).unwrap();

        let st_shape: Vec<usize> = st_tensor.shape().to_vec();
        let hct_shape: Vec<usize> = hct_tensor.dims().to_vec();

        let match_status = if st_shape == hct_shape {
            "OK"
        } else {
            "MISMATCH"
        };
        println!(
            "  {} {} {:?} vs {:?}",
            match_status, name, st_shape, hct_shape
        );
    }

    // Test hybrid loading
    println!("\n=== Testing Hybrid Load ===");
    let mut hybrid_tensors = hct_tensors;
    let mut supplemented = 0;

    for name in st.names() {
        if !hybrid_tensors.contains_key(name) {
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
                    Tensor::from_vec(floats, shape.as_slice(), &device)?
                },
                safetensors::Dtype::F32 => {
                    let floats: Vec<f32> = data
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();
                    Tensor::from_vec(floats, shape.as_slice(), &device)?
                },
                safetensors::Dtype::F16 => {
                    let halfs: Vec<half::f16> = data
                        .chunks_exact(2)
                        .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]))
                        .collect();
                    let floats: Vec<f32> = halfs.iter().map(|h| h.to_f32()).collect();
                    Tensor::from_vec(floats, shape.as_slice(), &device)?
                },
                _ => continue,
            };

            hybrid_tensors.insert(name.to_string(), tensor);
            supplemented += 1;
        }
    }

    println!("Supplemented {} tensors from safetensors", supplemented);
    println!("Final hybrid tensor count: {}", hybrid_tensors.len());
    println!("Expected: {}", st_names.len());

    if hybrid_tensors.len() == st_names.len() {
        println!("OK: All tensors present in hybrid load");
    } else {
        println!("WARNING: Tensor count mismatch!");
    }

    Ok(())
}
