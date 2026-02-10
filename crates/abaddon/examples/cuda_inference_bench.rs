//! End-to-end CUDA inference benchmark.
//!
//! This tests the full inference pipeline with a real model.

use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use abaddon::cuda_inference::{ComputeEngine, WeightStore};
    use cudarc::driver::CudaDevice;

    println!("=== CUDA Inference Benchmark ===\n");

    // Initialize CUDA
    let device = CudaDevice::new(0)?;
    let device = Arc::new(device);

    println!("CUDA Device: {:?}", device.name());

    // Get model path from command line or use default
    let args: Vec<String> = std::env::args().collect();
    let model_dir = if args.len() > 1 && args[1] == "--model" && args.len() > 2 {
        args[2].clone()
    } else {
        "/home/crook/dev2/workspace/nyx/infernum/infernum-complete/test_models/smollm2-135m-int4"
            .to_string()
    };

    if !std::path::Path::new(&model_dir).exists() {
        println!("Test model not found at: {}", model_dir);
        println!("Please run the quantization script first or use --model <path>");
        return Ok(());
    }

    println!("Loading model from: {}", model_dir);
    let start = Instant::now();

    // Load weights (auto-detect architecture from config.json)
    let weights = WeightStore::load_hct(&model_dir, None, 0)?;
    let load_time = start.elapsed();

    println!("Model loaded in {:?}", load_time);
    println!("Config: {:?}", weights.config);
    println!(
        "Memory used: {:.2} MB",
        weights.memory_used as f64 / 1024.0 / 1024.0
    );

    // Create compute engine
    let max_seq_len = 512;
    let mut engine = ComputeEngine::new(weights.config.clone(), max_seq_len, Arc::clone(&device))?;

    println!("\nCompute engine initialized");

    // Benchmark forward passes
    println!("\n--- Forward Pass Benchmarks ---\n");

    // Test different sequence lengths
    let test_sequences = [
        (1, "Single token (decode)"),
        (8, "8 tokens"),
        (32, "32 tokens"),
        (128, "128 tokens (typical prompt)"),
    ];

    for (seq_len, label) in test_sequences {
        // Create dummy input tokens
        let input_ids: Vec<u32> = (0..seq_len).map(|i| (i % 1000) as u32).collect();

        // Calculate how many iterations we can fit in the max_seq_len
        let iterations = std::cmp::min(
            if seq_len <= 32 { 50 } else { 10 },
            max_seq_len / seq_len - 1,
        )
        .max(3);

        // Fresh engine for this test
        let mut engine =
            ComputeEngine::new(weights.config.clone(), max_seq_len, Arc::clone(&device))?;

        // Warmup
        let _ = engine.forward(&input_ids, &weights, 0);

        // Create fresh engine for benchmark
        let mut engine =
            ComputeEngine::new(weights.config.clone(), max_seq_len, Arc::clone(&device))?;

        let start = Instant::now();
        for i in 0..iterations {
            // Simulate autoregressive generation by advancing position
            let _ = engine.forward(&input_ids, &weights, i * seq_len)?;
        }
        let elapsed = start.elapsed();

        let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        let tokens_per_sec = seq_len as f64 / (avg_ms / 1000.0);

        println!(
            "{:25} | {:7.2} ms | {:8.0} tok/s",
            label, avg_ms, tokens_per_sec
        );
    }

    // Decode throughput (single token generation)
    println!("\n--- Decode Throughput (Single Token) ---\n");

    let input_ids = vec![1u32]; // Single token
    let iterations = 200; // Limit to fit in max_seq_len

    // Fresh engine
    let mut engine = ComputeEngine::new(weights.config.clone(), max_seq_len, Arc::clone(&device))?;

    // Warmup
    for i in 0..10 {
        let _ = engine.forward(&input_ids, &weights, i);
    }

    // Fresh engine for benchmark
    let mut engine = ComputeEngine::new(weights.config.clone(), max_seq_len, Arc::clone(&device))?;

    let start = Instant::now();
    for i in 0..iterations {
        let _ = engine.forward(&input_ids, &weights, i)?;
    }
    let elapsed = start.elapsed();

    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let tok_per_sec = 1000.0 / avg_ms;

    println!("Decode latency:    {:.2} ms/token", avg_ms);
    println!("Decode throughput: {:.0} tok/s", tok_per_sec);

    println!("\n=== Benchmark Complete ===");

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("This example requires the 'cuda' feature.");
    println!("Run with: cargo run --example cuda_inference_bench --features cuda");
}
