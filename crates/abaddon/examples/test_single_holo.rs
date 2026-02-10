//! Test loading a single HoloTensor file on CPU

use abaddon::hct::HctLoader;
use candle_core::{DType, Device};
use std::env;

fn main() -> anyhow::Result<()> {
    println!("Testing single HoloTensor file load on CPU...\n");

    let file_path = env::args().nth(1).unwrap_or_else(|| {
        "/tmp/llama405b-holo/model_layers_0_input_layernorm_weight.hct".to_string()
    });

    println!("Loading: {}", file_path);
    let start = std::time::Instant::now();

    let loader = HctLoader::from_file(&file_path)?;
    println!("  Format: {:?}", loader.format());
    println!("  Is holographic: {}", loader.is_holographic());
    println!("  Shape: {:?}", loader.metadata().shape);
    println!("  DType: {:?}", loader.metadata().dtype);
    println!("  Original size: {} bytes", loader.metadata().original_size);
    println!(
        "  Compressed size: {} bytes",
        loader.metadata().compressed_size
    );

    println!("\nAttempting CPU reconstruction...");

    match loader.to_tensor(&Device::Cpu, Some(DType::F32)) {
        Ok(tensor) => {
            println!("  ✓ Tensor reconstructed successfully!");
            println!("  Shape: {:?}", tensor.dims());

            // Print some values
            let flat = tensor.flatten_all()?;
            let values: Vec<f32> = flat.to_vec1()?;
            println!("  First 5 values: {:?}", &values[..5.min(values.len())]);

            // Check for NaN/Inf
            let nan_count = values.iter().filter(|v| v.is_nan()).count();
            let inf_count = values.iter().filter(|v| v.is_infinite()).count();
            println!("  NaN count: {}", nan_count);
            println!("  Inf count: {}", inf_count);
        },
        Err(e) => {
            println!("  ✗ Failed to reconstruct: {}", e);
        },
    }

    println!("\nTotal time: {:?}", start.elapsed());

    Ok(())
}
