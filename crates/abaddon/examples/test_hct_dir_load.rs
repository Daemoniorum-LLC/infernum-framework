//! Test loading HCT directory with TieredHoloLoader
//!
//! Usage:
//! ```bash
//! cargo run --release -p abaddon --example test_hct_dir_load -- \
//!     --model /home/crook/models/llama-3.1-70b-hct-test
//! ```

use std::path::PathBuf;
use std::time::Instant;

use abaddon::holotensor::tiered_loading::{TieredConfig, TieredHoloLoader};
use abaddon::lazy_varbuilder::TensorProvider;
use candle_core::{DType, Device};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_dir = PathBuf::from("/home/crook/models/llama-3.1-70b-hct-test");

    // Parse --model argument
    for i in 1..args.len() {
        if args[i] == "--model" || args[i] == "-m" {
            if i + 1 < args.len() {
                model_dir = PathBuf::from(&args[i + 1]);
            }
        }
    }

    println!("=== HCT Directory Load Test ===\n");
    println!("Directory: {}\n", model_dir.display());

    // Create tiered loader config
    let config = TieredConfig {
        vram_budget: 8 * 1024 * 1024 * 1024, // 8GB
        ram_budget: 32 * 1024 * 1024 * 1024, // 32GB
        min_quality: 0.7,
        target_quality: 0.95,
        enable_background_streaming: false,
        background_streams: 0,
    };

    // Create loader
    let device = Device::Cpu; // Use CPU for this test
    let loader = TieredHoloLoader::new(&model_dir, config, device.clone(), DType::F32)?;

    // List available tensors
    let tensor_names = loader.tensor_names();
    println!("Found {} tensors\n", tensor_names.len());

    // Load first 5 tensors as a test
    let test_count = 5.min(tensor_names.len());
    println!("Loading first {} tensors...\n", test_count);

    let mut total_bytes = 0u64;
    let overall_start = Instant::now();

    for name in tensor_names.iter().take(test_count) {
        print!("  {}... ", name);
        let start = Instant::now();

        match loader.load_tensor(name) {
            Ok(tensor) => {
                let elapsed_ms = start.elapsed().as_millis();
                let shape = tensor.dims();
                let elem_count = tensor.elem_count();
                let bytes = elem_count as u64 * 4; // f32 = 4 bytes
                total_bytes += bytes;

                println!(
                    "OK ({:?}, {:.1} MB, {} ms)",
                    shape,
                    bytes as f64 / 1e6,
                    elapsed_ms
                );

                // Verify first few values
                if let Ok(data) = tensor.flatten_all()?.to_vec1::<f32>() {
                    let non_zero = data.iter().filter(|x| x.abs() > 1e-10).count();
                    let nz_pct = 100.0 * non_zero as f64 / data.len() as f64;
                    println!("    First 5 values: {:?}", &data[..5.min(data.len())]);
                    println!("    Non-zero: {} ({:.1}%)", non_zero, nz_pct);
                }
            },
            Err(e) => {
                println!("FAILED: {}", e);
            },
        }
        println!();
    }

    let overall_elapsed = overall_start.elapsed().as_secs_f64();
    println!("=== Summary ===");
    println!("Tensors loaded: {}", test_count);
    println!("Total bytes: {:.2} MB", total_bytes as f64 / 1e6);
    println!("Total time: {:.2} s", overall_elapsed);
    println!(
        "Throughput: {:.1} MB/s",
        total_bytes as f64 / 1e6 / overall_elapsed
    );

    // Show loader stats
    let stats = loader.stats();
    println!("\nLoader stats:");
    println!("  VRAM tensors: {}", stats.vram_tensors);
    println!("  RAM tensors: {}", stats.ram_tensors);
    println!("  GPU reconstructions: {}", stats.gpu_reconstructions);
    println!("  CPU reconstructions: {}", stats.cpu_reconstructions);

    Ok(())
}
