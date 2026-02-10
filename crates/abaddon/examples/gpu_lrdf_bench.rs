//! Benchmark GPU vs CPU LRDF encoding.
//!
//! Run with: cargo run --release --example gpu_lrdf_bench --features cuda

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU vs CPU LRDF Encoder Benchmark ===\n");

    // Test matrix dimensions
    let rows = 4096;
    let cols = 4096;
    let num_fragments = 32;
    let max_rank = 64;
    let seed = 42u64;

    // Generate test data
    println!("Generating {rows}x{cols} test matrix...");
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32 * 0.01).sin() + (i as f32 * 0.007).cos()) / 2.0)
        .collect();
    println!(
        "  Matrix size: {:.2} MB\n",
        (rows * cols * 4) as f64 / (1024.0 * 1024.0)
    );

    // CPU benchmark
    println!("CPU LRDF Encoding:");
    {
        use haagenti::holotensor::LrdfEncoder;

        let encoder = LrdfEncoder::new(num_fragments).with_max_rank(max_rank);

        let start = Instant::now();
        let fragments = encoder.encode_2d(&data, rows, cols)?;
        let cpu_time = start.elapsed();

        println!("  Time: {:.2}ms", cpu_time.as_millis());
        println!("  Fragments: {}", fragments.len());
        println!(
            "  Total fragment bytes: {} KB",
            fragments.iter().map(|f| f.data.len()).sum::<usize>() / 1024
        );
    }

    // GPU benchmark
    println!("\nGPU LRDF Encoding:");
    #[cfg(feature = "cuda")]
    {
        use abaddon::gpu_lrdf::cuda::GpuLrdfEncoder;
        use cudarc::driver::CudaDevice;

        let device = CudaDevice::new(0)?;
        let encoder = GpuLrdfEncoder::new(device, num_fragments, 42)?.with_max_rank(max_rank);

        let start = Instant::now();
        let fragments = encoder.encode_2d(&data, rows, cols)?;
        let gpu_time = start.elapsed();

        println!("  Time: {:.2}ms", gpu_time.as_millis());
        println!("  Fragments: {}", fragments.len());
        println!(
            "  Total fragment bytes: {} KB",
            fragments.iter().map(|f| f.data.len()).sum::<usize>() / 1024
        );
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("  (CUDA not enabled)");
    }

    // Verify GPU fragments decode correctly
    println!("\nVerifying GPU fragment compatibility...");
    #[cfg(feature = "cuda")]
    {
        use abaddon::gpu_lrdf::cuda::GpuLrdfEncoder;
        use cudarc::driver::CudaDevice;
        use haagenti::holotensor::LrdfDecoder;

        let device = CudaDevice::new(0)?;
        let encoder = GpuLrdfEncoder::new(device, num_fragments, 42)?.with_max_rank(max_rank);

        let gpu_fragments = encoder.encode_2d(&data, rows, cols)?;
        let holo_fragments: Vec<_> = gpu_fragments.iter().map(|f| f.to_haagenti()).collect();

        // Decode using haagenti's decoder
        let mut decoder = LrdfDecoder::new(rows, cols, num_fragments);
        for frag in &holo_fragments {
            decoder.add_fragment(frag)?;
        }
        let reconstructed = decoder.reconstruct();

        // Calculate quality
        let dot: f64 = data
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        let norm_a: f64 = data.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let norm_b: f64 = reconstructed
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let similarity = dot / (norm_a * norm_b);

        println!("  Cosine similarity: {:.4}", similarity);
        println!(
            "  Status: {}",
            if similarity > 0.9 { "PASS" } else { "FAIL" }
        );
    }

    println!("\nDone!");
    Ok(())
}
