//! Profile CUDA inference to identify bottlenecks.

use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use abaddon::cuda_inference::{ComputeEngine, ModelArch, WeightStore};
    use cudarc::driver::CudaDevice;

    println!("=== CUDA Profiling ===\n");

    let device = CudaDevice::new(0)?;
    let device = Arc::new(device);
    println!("GPU: {:?}", device.name());

    // Get GPU info
    let (free, total) = cudarc::driver::result::mem_get_info()?;
    println!(
        "GPU Memory: {:.1} GB free / {:.1} GB total\n",
        free as f64 / 1024.0 / 1024.0 / 1024.0,
        total as f64 / 1024.0 / 1024.0 / 1024.0
    );

    let model_dir =
        "/home/crook/dev2/workspace/nyx/infernum/infernum-complete/test_models/smollm2-135m-int4";

    // Profile weight loading
    println!("--- Weight Loading ---");
    let start = Instant::now();
    let weights = WeightStore::load_hct(model_dir, Some(ModelArch::Llama), 0)?;
    println!("Load time: {:?}", start.elapsed());
    println!("Layers: {}", weights.config.num_layers);
    println!("Hidden size: {}", weights.config.hidden_size);

    // Profile engine creation
    println!("\n--- Engine Creation ---");
    let max_seq_len = 256;
    let start = Instant::now();
    let mut engine = ComputeEngine::new(weights.config.clone(), max_seq_len, Arc::clone(&device))?;
    println!("Engine creation: {:?}", start.elapsed());

    // Profile first forward (includes kernel JIT)
    println!("\n--- First Forward (includes JIT) ---");
    let input_ids = vec![1u32];
    let start = Instant::now();
    let _ = engine.forward(&input_ids, &weights, 0)?;
    device.synchronize()?; // Ensure GPU work is done
    println!("First forward: {:?}", start.elapsed());

    // Profile subsequent forwards (cached kernels)
    println!("\n--- Cached Forwards (single token) ---");
    let mut times = Vec::new();
    for i in 1..21 {
        let start = Instant::now();
        let _ = engine.forward(&input_ids, &weights, i)?;
        device.synchronize()?;
        times.push(start.elapsed());
    }

    let avg = times.iter().map(|t| t.as_micros()).sum::<u128>() / times.len() as u128;
    let min = times.iter().map(|t| t.as_micros()).min().unwrap();
    let max = times.iter().map(|t| t.as_micros()).max().unwrap();

    println!("Avg: {} µs ({:.1} ms)", avg, avg as f64 / 1000.0);
    println!("Min: {} µs", min);
    println!("Max: {} µs", max);
    println!("Throughput: {:.0} tok/s", 1_000_000.0 / avg as f64);

    // Profile prefill
    println!("\n--- Prefill (32 tokens) ---");
    let mut engine = ComputeEngine::new(weights.config.clone(), max_seq_len, Arc::clone(&device))?;
    let input_ids: Vec<u32> = (0..32).collect();

    let start = Instant::now();
    let _ = engine.forward(&input_ids, &weights, 0)?;
    device.synchronize()?;
    let prefill_time = start.elapsed();
    println!("Prefill time: {:?}", prefill_time);
    println!("Throughput: {:.0} tok/s", 32.0 / prefill_time.as_secs_f64());

    // Profile larger prefill
    println!("\n--- Prefill (128 tokens) ---");
    let mut engine = ComputeEngine::new(weights.config.clone(), max_seq_len, Arc::clone(&device))?;
    let input_ids: Vec<u32> = (0..128).collect();

    let start = Instant::now();
    let _ = engine.forward(&input_ids, &weights, 0)?;
    device.synchronize()?;
    let prefill_time = start.elapsed();
    println!("Prefill time: {:?}", prefill_time);
    println!(
        "Throughput: {:.0} tok/s",
        128.0 / prefill_time.as_secs_f64()
    );

    println!("\n=== Profile Complete ===");
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("Requires 'cuda' feature");
}
