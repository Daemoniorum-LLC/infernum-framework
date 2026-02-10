//! Benchmarks for holographic tensor reconstruction.
//!
//! These benchmarks measure the performance of:
//! - CPU encoding/decoding (via haagenti)
//! - GPU reconstruction (spectral, RPH, LRDF)
//! - Streaming pipeline throughput
//! - Multi-GPU scaling
//! - Memory coalescing optimizations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// ============================================================================
// CPU ENCODING BENCHMARKS (haagenti)
// ============================================================================

fn cpu_encoding_benchmark(c: &mut Criterion) {
    use haagenti::holotensor::{HoloTensorEncoder, HolographicEncoding};

    let mut group = c.benchmark_group("holo_cpu_encoding");

    // Test different tensor sizes
    let sizes = [
        (256, 256, "256x256"),
        (512, 512, "512x512"),
        (1024, 1024, "1024x1024"),
        (2048, 2048, "2048x2048"),
    ];

    for (width, height, label) in sizes {
        let data: Vec<f32> = (0..width * height)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();

        group.throughput(Throughput::Bytes((width * height * 4) as u64));

        // Spectral encoding
        group.bench_with_input(
            BenchmarkId::new("spectral_8frags", label),
            &data,
            |b, data| {
                let encoder =
                    HoloTensorEncoder::new(HolographicEncoding::Spectral).with_fragments(8);
                b.iter(|| encoder.encode_2d(black_box(data), width, height))
            },
        );

        // RPH encoding
        group.bench_with_input(BenchmarkId::new("rph_8frags", label), &data, |b, data| {
            let encoder =
                HoloTensorEncoder::new(HolographicEncoding::RandomProjection).with_fragments(8);
            b.iter(|| encoder.encode_2d(black_box(data), width, height))
        });

        // LRDF encoding
        group.bench_with_input(BenchmarkId::new("lrdf_8frags", label), &data, |b, data| {
            let encoder =
                HoloTensorEncoder::new(HolographicEncoding::LowRankDistributed).with_fragments(8);
            b.iter(|| encoder.encode_2d(black_box(data), width, height))
        });
    }

    group.finish();
}

fn cpu_decoding_benchmark(c: &mut Criterion) {
    use haagenti::holotensor::{HoloTensorDecoder, HoloTensorEncoder, HolographicEncoding};

    let mut group = c.benchmark_group("holo_cpu_decoding");

    // Test different tensor sizes
    let sizes = [
        (256, 256, "256x256"),
        (512, 512, "512x512"),
        (1024, 1024, "1024x1024"),
    ];

    for (width, height, label) in sizes {
        let data: Vec<f32> = (0..width * height)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();

        // Pre-encode for decoding benchmarks
        let encoder = HoloTensorEncoder::new(HolographicEncoding::Spectral).with_fragments(8);
        let (header, fragments) = encoder.encode_2d(&data, width, height).unwrap();

        group.throughput(Throughput::Bytes((width * height * 4) as u64));

        // Full reconstruction (all fragments)
        group.bench_with_input(
            BenchmarkId::new("spectral_full", label),
            &(&header, &fragments),
            |b, (header, fragments)| {
                b.iter(|| {
                    let mut decoder = HoloTensorDecoder::new((*header).clone());
                    for frag in *fragments {
                        decoder.add_fragment(frag.clone()).unwrap();
                    }
                    decoder.reconstruct()
                })
            },
        );

        // Partial reconstruction (50% fragments)
        let half_frags: Vec<_> = fragments.iter().take(4).cloned().collect();
        group.bench_with_input(
            BenchmarkId::new("spectral_50pct", label),
            &(&header, &half_frags),
            |b, (header, fragments)| {
                b.iter(|| {
                    let mut decoder = HoloTensorDecoder::new((*header).clone());
                    for frag in *fragments {
                        decoder.add_fragment(frag.clone()).unwrap();
                    }
                    decoder.reconstruct()
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// GPU RECONSTRUCTION BENCHMARKS
// ============================================================================

#[cfg(feature = "cuda")]
fn gpu_reconstruction_benchmark(c: &mut Criterion) {
    use abaddon::gpu_holo::cuda::GpuHoloContext;
    use haagenti::holotensor::{HoloTensorEncoder, HolographicEncoding};

    // Initialize GPU context
    let mut ctx = match GpuHoloContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("CUDA not available, skipping GPU benchmarks");
            return;
        },
    };

    if ctx.load_all_kernels().is_err() {
        eprintln!("Failed to load GPU kernels, skipping GPU benchmarks");
        return;
    }

    let mut group = c.benchmark_group("holo_gpu_reconstruction");

    // Test different tensor sizes
    let sizes = [
        (512, 512, "512x512"),
        (1024, 1024, "1024x1024"),
        (2048, 2048, "2048x2048"),
        (4096, 4096, "4096x4096"),
    ];

    for (width, height, label) in sizes {
        let data: Vec<f32> = (0..width * height)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();

        group.throughput(Throughput::Bytes((width * height * 4) as u64));

        // Spectral reconstruction
        let encoder = HoloTensorEncoder::new(HolographicEncoding::Spectral).with_fragments(8);
        let (header, fragments) = encoder.encode_2d(&data, width, height).unwrap();

        group.bench_with_input(
            BenchmarkId::new("gpu_spectral", label),
            &(&header, &fragments),
            |b, (header, fragments)| {
                b.iter(|| ctx.reconstruct(black_box(*header), black_box(*fragments)))
            },
        );

        // RPH reconstruction
        let encoder =
            HoloTensorEncoder::new(HolographicEncoding::RandomProjection).with_fragments(8);
        let (header, fragments) = encoder.encode_2d(&data, width, height).unwrap();

        group.bench_with_input(
            BenchmarkId::new("gpu_rph", label),
            &(&header, &fragments),
            |b, (header, fragments)| {
                b.iter(|| ctx.reconstruct(black_box(*header), black_box(*fragments)))
            },
        );

        // LRDF reconstruction
        let encoder =
            HoloTensorEncoder::new(HolographicEncoding::LowRankDistributed).with_fragments(8);
        let (header, fragments) = encoder.encode_2d(&data, width, height).unwrap();

        group.bench_with_input(
            BenchmarkId::new("gpu_lrdf", label),
            &(&header, &fragments),
            |b, (header, fragments)| {
                b.iter(|| ctx.reconstruct(black_box(*header), black_box(*fragments)))
            },
        );
    }

    group.finish();
}

#[cfg(not(feature = "cuda"))]
fn gpu_reconstruction_benchmark(_c: &mut Criterion) {
    // Skip GPU benchmarks when CUDA is not available
}

// ============================================================================
// STREAMING PIPELINE BENCHMARKS
// ============================================================================

#[cfg(feature = "cuda")]
fn streaming_pipeline_benchmark(c: &mut Criterion) {
    use abaddon::gpu_holo::cuda::StreamingHoloContext;
    use haagenti::holotensor::{HoloTensorEncoder, HolographicEncoding};

    // Initialize streaming context
    let ctx = match StreamingHoloContext::new(0, 4) {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("CUDA not available, skipping streaming benchmarks");
            return;
        },
    };

    let mut group = c.benchmark_group("holo_streaming");

    // Test streaming with different quality targets
    let quality_targets = [0.5, 0.75, 0.9, 0.95, 0.99];
    let size = 1024;
    let data: Vec<f32> = (0..size * size).map(|i| (i as f32 * 0.001).sin()).collect();

    let encoder = HoloTensorEncoder::new(HolographicEncoding::Spectral).with_fragments(16);
    let (header, fragments) = encoder.encode_2d(&data, size, size).unwrap();

    for target in quality_targets {
        let label = format!("q{:.0}pct", target * 100.0);
        group.bench_with_input(
            BenchmarkId::new("streaming_spectral", &label),
            &(&header, &fragments, target),
            |b, (header, fragments, target)| {
                b.iter(|| ctx.reconstruct_streaming(*header, fragments.iter(), **target))
            },
        );
    }

    // Benchmark with callback
    group.bench_function("streaming_with_callback", |b| {
        b.iter(|| {
            ctx.reconstruct_with_callback(&header, fragments.iter(), |_frags, quality| {
                quality < 0.95 // Continue until 95% quality
            })
        })
    });

    group.finish();
}

#[cfg(not(feature = "cuda"))]
fn streaming_pipeline_benchmark(_c: &mut Criterion) {
    // Skip streaming benchmarks when CUDA is not available
}

// ============================================================================
// MEMORY COALESCING BENCHMARKS
// ============================================================================

#[cfg(feature = "cuda")]
fn coalesced_memory_benchmark(c: &mut Criterion) {
    use abaddon::gpu_holo::cuda::GpuHoloContext;

    // Initialize GPU context with coalesced kernels
    let mut ctx = match GpuHoloContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("CUDA not available, skipping coalescing benchmarks");
            return;
        },
    };

    if ctx.load_coalesced_kernels().is_err() {
        eprintln!("Failed to load coalesced kernels, skipping benchmarks");
        return;
    }
    if ctx.load_fused_kernel().is_err() {
        eprintln!("Failed to load fused kernel, skipping benchmarks");
        return;
    }

    let mut group = c.benchmark_group("holo_coalescing");

    // Test F32 to F16 conversion (common operation)
    let sizes = [
        (1 << 18, "256K"), // 256K elements
        (1 << 20, "1M"),   // 1M elements
        (1 << 22, "4M"),   // 4M elements
        (1 << 24, "16M"),  // 16M elements
    ];

    for (size, label) in sizes {
        let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();

        // Copy to GPU
        let d_input = ctx.device().htod_copy(data.clone()).unwrap();

        group.throughput(Throughput::Bytes((size * 4) as u64));

        // Standard F32 to F16
        group.bench_with_input(
            BenchmarkId::new("f32_to_f16_standard", label),
            &d_input,
            |b, input| b.iter(|| ctx.convert_f32_to_f16(black_box(input))),
        );

        // Coalesced F32 to F16
        group.bench_with_input(
            BenchmarkId::new("f32_to_f16_coalesced", label),
            &d_input,
            |b, input| b.iter(|| ctx.convert_f32_to_f16_coalesced(black_box(input))),
        );
    }

    group.finish();
}

#[cfg(not(feature = "cuda"))]
fn coalesced_memory_benchmark(_c: &mut Criterion) {
    // Skip coalescing benchmarks when CUDA is not available
}

// ============================================================================
// QUALITY CURVE BENCHMARKS
// ============================================================================

fn quality_curve_benchmark(c: &mut Criterion) {
    use haagenti::holotensor::QualityCurve;

    let mut group = c.benchmark_group("quality_curve");

    // Create sample quality curves
    let spectral_curve = QualityCurve {
        coefficients: [0.0, 0.3, 0.5, 0.2],
        min_fragments: 1,
        sufficient_fragments: 6,
    };

    // Benchmark quality prediction
    group.bench_function("predict_quality", |b| {
        b.iter(|| spectral_curve.predict(black_box(4), black_box(8)))
    });

    // Benchmark finding fragments for quality target
    group.bench_function("fragments_for_quality_0.95", |b| {
        b.iter(|| spectral_curve.fragments_for_quality(black_box(0.95), black_box(8)))
    });

    // Benchmark batch predictions (common when planning downloads)
    group.bench_function("predict_all_fragment_counts", |b| {
        b.iter(|| {
            let n = 16u16;
            (1..=n)
                .map(|k| spectral_curve.predict(black_box(k), black_box(n)))
                .collect::<Vec<_>>()
        })
    });

    group.finish();
}

// ============================================================================
// FILE I/O BENCHMARKS
// ============================================================================

fn file_io_benchmark(c: &mut Criterion) {
    use haagenti::holotensor::{HoloTensorEncoder, HolographicEncoding};
    use std::io::Cursor;

    let mut group = c.benchmark_group("holo_file_io");

    // Test writing to memory buffer (simulates file I/O without disk variability)
    let sizes = [
        (256, 256, "256x256"),
        (512, 512, "512x512"),
        (1024, 1024, "1024x1024"),
    ];

    for (width, height, label) in sizes {
        let data: Vec<f32> = (0..width * height)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();

        let encoder = HoloTensorEncoder::new(HolographicEncoding::Spectral).with_fragments(8);
        let (header, fragments) = encoder.encode_2d(&data, width, height).unwrap();

        group.throughput(Throughput::Bytes((width * height * 4) as u64));

        // Write to buffer
        group.bench_with_input(
            BenchmarkId::new("write_to_buffer", label),
            &(&header, &fragments),
            |b, (header, fragments)| {
                b.iter(|| {
                    let mut buffer = Cursor::new(Vec::new());
                    haagenti::holotensor::HoloTensorWriter::new(&mut buffer)
                        .write_header(*header)
                        .unwrap()
                        .write_fragments(*fragments)
                        .unwrap()
                        .finish()
                })
            },
        );

        // Read from buffer (pre-write to buffer first)
        let mut write_buffer = Cursor::new(Vec::new());
        haagenti::holotensor::HoloTensorWriter::new(&mut write_buffer)
            .write_header(&header)
            .unwrap()
            .write_fragments(&fragments)
            .unwrap()
            .finish()
            .unwrap();
        let read_data = write_buffer.into_inner();

        group.bench_with_input(
            BenchmarkId::new("read_from_buffer", label),
            &read_data,
            |b, data| {
                b.iter(|| {
                    let cursor = Cursor::new(black_box(data.as_slice()));
                    haagenti::holotensor::HoloTensorReader::open(cursor).and_then(|r| r.read_all())
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// FRAGMENT COUNT SCALING BENCHMARKS
// ============================================================================

fn fragment_scaling_benchmark(c: &mut Criterion) {
    use haagenti::holotensor::{HoloTensorEncoder, HolographicEncoding};

    let mut group = c.benchmark_group("holo_fragment_scaling");

    // Fixed size, varying fragment counts
    let size = 512;
    let data: Vec<f32> = (0..size * size).map(|i| (i as f32 * 0.001).sin()).collect();

    let fragment_counts = [4, 8, 16, 32, 64];

    for num_frags in fragment_counts {
        let label = format!("{}frags", num_frags);

        // Encoding with different fragment counts
        group.bench_with_input(
            BenchmarkId::new("encode_spectral", &label),
            &num_frags,
            |b, &frags| {
                let encoder =
                    HoloTensorEncoder::new(HolographicEncoding::Spectral).with_fragments(frags);
                b.iter(|| encoder.encode_2d(black_box(&data), size, size))
            },
        );
    }

    group.finish();
}

// ============================================================================
// CRITERION CONFIGURATION
// ============================================================================

criterion_group!(
    benches,
    cpu_encoding_benchmark,
    cpu_decoding_benchmark,
    gpu_reconstruction_benchmark,
    streaming_pipeline_benchmark,
    coalesced_memory_benchmark,
    quality_curve_benchmark,
    file_io_benchmark,
    fragment_scaling_benchmark,
);
criterion_main!(benches);
