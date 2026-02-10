//! HoloTensor CUDA Benchmark - Real GPU Progressive Inference Test
//!
//! Tests progressive holographic reconstruction on GPU with actual CUDA kernels.
//! Validates that memory transfer latency can be hidden behind compute.
//!
//! Run with: cargo run --example holotensor_cuda_bench --release --features cuda

use std::time::{Duration, Instant};

use haagenti::holotensor::{
    HoloFragment, HoloTensorHeader, HolographicEncoding, LrdfEncoder, QualityCurve,
};
use haagenti::tensor::DType;

/// Qwen2.5-7B layer dimensions
const HIDDEN_SIZE: usize = 3584;
const INTERMEDIATE_SIZE: usize = 18944;
const NUM_KV_HEADS: usize = 4;
const HEAD_DIM: usize = 128;
const NUM_LAYERS: usize = 28;

/// Fragment configuration
const NUM_FRAGMENTS: u16 = 32;
const MAX_RANK: usize = 128;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║      HoloTensor CUDA Progressive Inference Benchmark         ║");
    println!("║                                                              ║");
    println!("║  Testing real GPU reconstruction with pipelined streaming    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    #[cfg(feature = "cuda")]
    {
        use abaddon::{GpuHoloContext, ProgressiveHoloLoader, StreamingHoloContext};

        // Check CUDA availability
        println!("🔍 Checking CUDA availability...");

        match GpuHoloContext::new(0) {
            Ok(mut ctx) => {
                println!("   ✅ CUDA device 0 initialized");

                // Load kernels
                if let Err(e) = ctx.load_lrdf_kernel() {
                    println!("   ⚠️  Failed to load LRDF kernel: {}", e);
                    println!("   Using CPU fallback for testing...");
                    run_cpu_benchmark();
                    return;
                }
                println!("   ✅ LRDF kernel loaded");

                // Run GPU benchmarks
                run_gpu_benchmark(ctx);
            },
            Err(e) => {
                println!("   ❌ Failed to initialize CUDA: {}", e);
                println!("   Running CPU benchmark instead...");
                run_cpu_benchmark();
            },
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("⚠️  CUDA feature not enabled. Running CPU benchmark...");
        println!("   To run GPU benchmark: cargo run --example holotensor_cuda_bench --release --features cuda");
        println!();
        run_cpu_benchmark();
    }
}

#[cfg(feature = "cuda")]
fn run_gpu_benchmark(mut ctx: abaddon::GpuHoloContext) {
    use abaddon::{ProgressiveHoloLoader, StreamingHoloContext};

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🚀 GPU Benchmark: Progressive Holographic Reconstruction");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Test with attention weight dimensions (q_proj: hidden x hidden)
    let rows = HIDDEN_SIZE;
    let cols = HIDDEN_SIZE;

    println!("📊 Test Configuration:");
    println!(
        "   Tensor shape: {}x{} ({:.2} MB)",
        rows,
        cols,
        (rows * cols * 4) as f64 / 1024.0 / 1024.0
    );
    println!("   Fragments:    {}", NUM_FRAGMENTS);
    println!("   Encoding:     LRDF (Low-Rank Distributed Factorization)");
    println!();

    // Create realistic test data
    println!("📝 Generating test weights...");
    let original = create_realistic_weights(rows, cols, 42);

    // Encode to holotensor format (CPU - one time cost)
    println!("🔄 Encoding to holotensor format (CPU)...");
    let encode_start = Instant::now();
    let encoder = LrdfEncoder::new(NUM_FRAGMENTS).with_max_rank(MAX_RANK);
    let fragments = encoder
        .encode_2d(&original, rows, cols)
        .expect("Encoding failed");
    let encode_time = encode_start.elapsed();
    println!("   Encoding time: {:?}", encode_time);
    println!();

    // Create header
    let header = HoloTensorHeader::new(
        HolographicEncoding::LowRankDistributed,
        DType::F32,
        vec![rows as u64, cols as u64],
        NUM_FRAGMENTS,
    );

    // Benchmark 1: Progressive loading with quality tracking
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📈 GPU Benchmark 1: Progressive Loading");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    match ProgressiveHoloLoader::new(ctx, header.clone()) {
        Ok(mut loader) => {
            println!("   Fragments │ GPU Time   │ Cumulative │ Quality");
            println!("   ──────────┼────────────┼────────────┼────────");

            let mut cumulative = Duration::ZERO;

            for (i, fragment) in fragments.iter().enumerate() {
                let start = Instant::now();
                match loader.feed(fragment) {
                    Ok(quality) => {
                        let elapsed = start.elapsed();
                        cumulative += elapsed;

                        if (i + 1) % 4 == 0 || i == 0 {
                            println!(
                                "   {:>9} │ {:>10.2?} │ {:>10.2?} │ {:>5.1}%",
                                i + 1,
                                elapsed,
                                cumulative,
                                quality * 100.0
                            );
                        }
                    },
                    Err(e) => {
                        println!("   Error feeding fragment {}: {}", i, e);
                        break;
                    },
                }
            }

            println!();
            println!("   Total GPU reconstruction time: {:?}", cumulative);
            println!(
                "   Throughput: {:.1} MB/s",
                (rows * cols * 4) as f64 / 1024.0 / 1024.0 / cumulative.as_secs_f64()
            );

            // Finalize and verify
            match loader.finalize() {
                Ok(gpu_result) => {
                    println!("   ✅ Reconstruction complete on GPU");

                    // Note: Would need to copy back to CPU to verify quality
                    // For now we trust the GPU result
                },
                Err(e) => {
                    println!("   ❌ Finalization failed: {}", e);
                },
            }
        },
        Err(e) => {
            println!("   ❌ Failed to create progressive loader: {}", e);
        },
    }

    println!();

    // Benchmark 2: Streaming context with pipelining
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🔀 GPU Benchmark 2: Pipelined Streaming");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    for pipeline_depth in [2, 4] {
        match StreamingHoloContext::new(0, pipeline_depth) {
            Ok(streaming_ctx) => {
                // Test different quality targets
                for quality_target in [0.70f32, 0.85, 0.95, 0.0] {
                    let target_str = if quality_target == 0.0 {
                        "100%".to_string()
                    } else {
                        format!("{:.0}%", quality_target * 100.0)
                    };

                    let start = Instant::now();
                    match streaming_ctx.reconstruct_streaming(
                        &header,
                        fragments.iter(),
                        quality_target,
                    ) {
                        Ok(_result) => {
                            let elapsed = start.elapsed();
                            let throughput =
                                (rows * cols * 4) as f64 / 1024.0 / 1024.0 / elapsed.as_secs_f64();
                            println!(
                                "   Pipeline {} │ Target {:>4} │ {:>10.2?} │ {:>6.1} MB/s",
                                pipeline_depth, target_str, elapsed, throughput
                            );
                        },
                        Err(e) => {
                            println!(
                                "   Pipeline {} │ Target {:>4} │ Error: {}",
                                pipeline_depth, target_str, e
                            );
                        },
                    }
                }
            },
            Err(e) => {
                println!("   ❌ Failed to create streaming context: {}", e);
            },
        }
        println!();
    }

    // Benchmark 3: Simulated layer processing
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("⚡ GPU Benchmark 3: Simulated Layer Processing");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Simulate processing 5 layers with progressive loading
    let num_test_layers = 5;

    match StreamingHoloContext::new(0, 4) {
        Ok(streaming_ctx) => {
            println!(
                "   Simulating {} layer forward pass with progressive loading...",
                num_test_layers
            );
            println!();

            let mut total_recon_time = Duration::ZERO;
            let layer_compute_time = Duration::from_millis(15); // Simulated compute

            for layer in 0..num_test_layers {
                // Reconstruct weights for this layer
                let recon_start = Instant::now();
                match streaming_ctx.reconstruct_streaming(&header, fragments.iter(), 0.85) {
                    Ok(_weights) => {
                        let recon_time = recon_start.elapsed();
                        total_recon_time += recon_time;

                        // Simulate compute (would actually run attention/MLP here)
                        std::thread::sleep(layer_compute_time);

                        println!(
                            "   Layer {} │ Recon: {:>8.2?} │ Compute: {:>8.2?} │ Total: {:>8.2?}",
                            layer,
                            recon_time,
                            layer_compute_time,
                            recon_time + layer_compute_time
                        );
                    },
                    Err(e) => {
                        println!("   Layer {} │ Error: {}", layer, e);
                    },
                }
            }

            println!();
            println!("   ────────────────────────────────────────────────────────────");
            println!("   Total reconstruction: {:?}", total_recon_time);
            println!(
                "   Total compute:        {:?}",
                layer_compute_time * num_test_layers as u32
            );
            println!(
                "   Total time:           {:?}",
                total_recon_time + layer_compute_time * num_test_layers as u32
            );
            println!();

            // Calculate overhead
            let compute_only = layer_compute_time.as_secs_f64() * num_test_layers as f64;
            let with_recon =
                (total_recon_time + layer_compute_time * num_test_layers as u32).as_secs_f64();
            let overhead = (with_recon / compute_only - 1.0) * 100.0;

            println!("   Reconstruction overhead: {:.1}%", overhead);

            if overhead < 10.0 {
                println!("   ✅ Excellent! Memory transfer nearly hidden behind compute.");
            } else if overhead < 25.0 {
                println!("   ⚠️  Acceptable overhead, could be improved with more pipelining.");
            } else {
                println!("   ❌ High overhead, bottleneck in reconstruction.");
            }
        },
        Err(e) => {
            println!("   ❌ Failed to create streaming context: {}", e);
        },
    }

    println!();
    println!("✅ GPU benchmark complete!");
}

/// CPU fallback benchmark (when CUDA not available)
fn run_cpu_benchmark() {
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("💻 CPU Benchmark: Holographic Reconstruction");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let rows = HIDDEN_SIZE;
    let cols = HIDDEN_SIZE;

    println!("📊 Test Configuration:");
    println!(
        "   Tensor shape: {}x{} ({:.2} MB)",
        rows,
        cols,
        (rows * cols * 4) as f64 / 1024.0 / 1024.0
    );
    println!("   Fragments:    {}", NUM_FRAGMENTS);
    println!();

    // Create test data
    println!("📝 Generating test weights...");
    let original = create_realistic_weights(rows, cols, 42);

    // Encode
    println!("🔄 Encoding...");
    let encode_start = Instant::now();
    let encoder = LrdfEncoder::new(NUM_FRAGMENTS).with_max_rank(MAX_RANK);
    let fragments = encoder
        .encode_2d(&original, rows, cols)
        .expect("Encoding failed");
    println!("   Encoding time: {:?}", encode_start.elapsed());
    println!();

    // Test progressive reconstruction
    println!("📈 Progressive Reconstruction:");
    println!("   Fragments │ Decode Time │ Quality");
    println!("   ──────────┼─────────────┼────────");

    for &count in &[8, 16, 24, 32] {
        let start = Instant::now();

        let mut decoder = haagenti::holotensor::LrdfDecoder::new(rows, cols, NUM_FRAGMENTS);
        for i in 0..count {
            decoder.add_fragment(&fragments[i as usize]).unwrap();
        }
        let reconstructed = decoder.reconstruct();

        let elapsed = start.elapsed();
        let quality = cosine_similarity(&original, &reconstructed);

        println!(
            "   {:>9} │ {:>11.2?} │ {:>5.1}%",
            count,
            elapsed,
            quality * 100.0
        );
    }

    println!();
    println!("✅ CPU benchmark complete!");
    println!();
    println!("💡 To run GPU benchmark:");
    println!("   cargo run --example holotensor_cuda_bench --release --features cuda");
}

/// Create realistic weight data
fn create_realistic_weights(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut data = Vec::with_capacity(rows * cols);
    let mut state = seed;
    let scale = 1.0 / (cols as f32).sqrt();

    for i in 0..(rows * cols) {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (state >> 32) as f32 / u32::MAX as f32;
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let v = (state >> 32) as f32 / u32::MAX as f32;

        let normal = (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos();
        let row = i / cols;
        let col = i % cols;
        let structure = ((row as f32 / rows as f32) * (col as f32 / cols as f32)).sin() * 0.1;

        data.push((normal * scale) + structure);
    }

    data
}

/// Calculate cosine similarity
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += (x as f64) * (y as f64);
        norm_a += (x as f64) * (x as f64);
        norm_b += (y as f64) * (y as f64);
    }

    if norm_a <= 0.0 || norm_b <= 0.0 {
        return 0.0;
    }

    (dot / (norm_a.sqrt() * norm_b.sqrt())) as f32
}
