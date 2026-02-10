//! Benchmarks comparing safetensors vs HCT compressed weight loading.
//!
//! These benchmarks measure the real-world performance impact of using
//! compressed HCT weights vs standard safetensors.

use std::collections::HashMap;
use std::fs::{self, File};
use std::path::PathBuf;

use candle_core::{DType, Device};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use safetensors::serialize;
use safetensors::tensor::TensorView;

use haagenti::tensor::{CompressionAlgorithm, DType as HctDType, HctWriter};
use haagenti::{Lz4Compressor, ZstdCompressor};
use haagenti_core::CompressionLevel;

use abaddon::hct::{load_hct_directory, HctLoader};

// ============================================================================
// TEST DATA GENERATION
// ============================================================================

/// Generates synthetic INT8-like weight data (simulating quantized weights).
/// Uses a pattern similar to real quantized neural network weights.
fn generate_quantized_weights(size: usize) -> Vec<u8> {
    // Quantized weights typically have a Gaussian-like distribution centered around 0
    // with most values in the range [-8, 8] for INT4 or [-128, 128] for INT8
    let mut data = Vec::with_capacity(size);
    let mut rng_state: u64 = 0xDEADBEEF;

    for _ in 0..size {
        // Simple PRNG for reproducible benchmarks
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);

        // Generate a value with Gaussian-like distribution using Box-Muller approximation
        // Most values cluster near zero with occasional outliers
        let uniform = (rng_state >> 33) as f32 / (1u64 << 31) as f32;
        let gaussian = ((uniform * 2.0 - 1.0) * 127.0).clamp(-127.0, 127.0) as i8;
        data.push(gaussian as u8);
    }

    data
}

/// Generates F32 weight data (simulating floating point weights).
fn generate_f32_weights(count: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(count);
    let mut rng_state: u64 = 0xCAFEBABE;

    for _ in 0..count {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        // Values typically in range [-1, 1] with most near zero
        let uniform = (rng_state >> 33) as f32 / (1u64 << 31) as f32;
        let value = (uniform * 2.0 - 1.0) * 0.1;
        data.push(value);
    }

    data
}

// ============================================================================
// TEST FILE CREATION
// ============================================================================

struct TestFiles {
    temp_dir: PathBuf,
    safetensors_path: PathBuf,
    hct_lz4_dir: PathBuf,
    hct_zstd_dir: PathBuf,
}

impl TestFiles {
    fn new(_tensor_shape: &[usize]) -> Self {
        let temp_dir = std::env::temp_dir().join(format!("weight_bench_{}", std::process::id()));
        fs::create_dir_all(&temp_dir).expect("create temp dir");

        let hct_lz4_dir = temp_dir.join("hct_lz4");
        let hct_zstd_dir = temp_dir.join("hct_zstd");
        fs::create_dir_all(&hct_lz4_dir).expect("create lz4 dir");
        fs::create_dir_all(&hct_zstd_dir).expect("create zstd dir");

        let safetensors_path = temp_dir.join("model.safetensors");

        Self {
            temp_dir,
            safetensors_path,
            hct_lz4_dir,
            hct_zstd_dir,
        }
    }

    fn create_f32_files(&self, tensors: &[(&str, &[usize], &[f32])]) {
        // Create safetensors file
        let mut tensor_map: HashMap<String, TensorView<'_>> = HashMap::new();

        for (name, shape, data) in tensors {
            // Convert f32 to bytes - leak to static lifetime for safetensors API
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            let bytes_static: &'static [u8] = Box::leak(bytes.into_boxed_slice());

            let view = TensorView::new(safetensors::Dtype::F32, shape.to_vec(), bytes_static)
                .expect("create tensor view");

            tensor_map.insert(name.to_string(), view);
        }

        let serialized = serialize(&tensor_map, &None).expect("serialize safetensors");
        fs::write(&self.safetensors_path, &serialized).expect("write safetensors");

        // Create HCT files (LZ4 and Zstd)
        for (name, shape, data) in tensors {
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

            let shape_u64: Vec<u64> = shape.iter().map(|&s| s as u64).collect();

            // LZ4
            let lz4_path = self.hct_lz4_dir.join(format!("{}.hct", name));
            let file = File::create(&lz4_path).expect("create lz4 file");
            let compressor = Lz4Compressor::with_level(CompressionLevel::Default);
            let mut writer = HctWriter::new(
                file,
                CompressionAlgorithm::Lz4,
                HctDType::F32,
                shape_u64.clone(),
            );
            writer
                .compress_data(&bytes, &compressor)
                .expect("compress lz4");
            writer.finish().expect("finish lz4");

            // Zstd
            let zstd_path = self.hct_zstd_dir.join(format!("{}.hct", name));
            let file = File::create(&zstd_path).expect("create zstd file");
            let compressor = ZstdCompressor::with_level(CompressionLevel::Default);
            let mut writer =
                HctWriter::new(file, CompressionAlgorithm::Zstd, HctDType::F32, shape_u64);
            writer
                .compress_data(&bytes, &compressor)
                .expect("compress zstd");
            writer.finish().expect("finish zstd");
        }
    }
}

impl Drop for TestFiles {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.temp_dir);
    }
}

// ============================================================================
// BENCHMARKS
// ============================================================================

fn weight_loading_benchmark(c: &mut Criterion) {
    let device = Device::Cpu;

    // Test with different tensor sizes (simulating model weights)
    let sizes: [(usize, &str); 3] = [
        (1024 * 1024, "1M_params"),       // ~4MB F32
        (16 * 1024 * 1024, "16M_params"), // ~64MB F32
        (64 * 1024 * 1024, "64M_params"), // ~256MB F32
    ];

    for (size, label) in sizes {
        let mut group = c.benchmark_group(format!("weight_loading/{}", label));
        group.throughput(Throughput::Bytes((size * 4) as u64)); // F32 = 4 bytes

        // Generate test data
        let weights = generate_f32_weights(size);
        let shape = vec![size];

        // Create test files
        let files = TestFiles::new(&shape);
        files.create_f32_files(&[("weights", &shape, &weights)]);

        // Report file sizes
        let st_size = fs::metadata(&files.safetensors_path)
            .map(|m| m.len())
            .unwrap_or(0);
        let lz4_size: u64 = fs::read_dir(&files.hct_lz4_dir)
            .map(|d| {
                d.filter_map(|e| e.ok())
                    .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
                    .sum()
            })
            .unwrap_or(0);
        let zstd_size: u64 = fs::read_dir(&files.hct_zstd_dir)
            .map(|d| {
                d.filter_map(|e| e.ok())
                    .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
                    .sum()
            })
            .unwrap_or(0);

        eprintln!("\n=== {} ===", label);
        eprintln!("SafeTensors: {:.2} MB", st_size as f64 / 1_000_000.0);
        eprintln!(
            "HCT (LZ4):   {:.2} MB ({:.2}x)",
            lz4_size as f64 / 1_000_000.0,
            st_size as f64 / lz4_size as f64
        );
        eprintln!(
            "HCT (Zstd):  {:.2} MB ({:.2}x)",
            zstd_size as f64 / 1_000_000.0,
            st_size as f64 / zstd_size as f64
        );

        // Benchmark safetensors loading
        group.bench_function("safetensors", |b| {
            b.iter(|| {
                let data = fs::read(&files.safetensors_path).expect("read safetensors");
                let tensors = safetensors::SafeTensors::deserialize(&data).expect("deserialize");
                let tensor_data = tensors.tensor("weights").expect("get weights");
                black_box(tensor_data.data().len())
            })
        });

        // Benchmark HCT LZ4 loading
        group.bench_function("hct_lz4", |b| {
            b.iter(|| {
                let tensors = load_hct_directory(&files.hct_lz4_dir, &device, DType::F32)
                    .expect("load hct lz4");
                black_box(tensors.len())
            })
        });

        // Benchmark HCT Zstd loading
        group.bench_function("hct_zstd", |b| {
            b.iter(|| {
                let tensors = load_hct_directory(&files.hct_zstd_dir, &device, DType::F32)
                    .expect("load hct zstd");
                black_box(tensors.len())
            })
        });

        group.finish();
    }
}

/// Benchmark decompression throughput specifically
fn decompression_throughput_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompression_throughput");

    // 16MB of quantized data
    let size = 16 * 1024 * 1024;
    let data = generate_quantized_weights(size);
    let shape = vec![size];

    group.throughput(Throughput::Bytes(size as u64));

    // Create temp files
    let temp_dir = std::env::temp_dir().join(format!("decompress_bench_{}", std::process::id()));
    fs::create_dir_all(&temp_dir).expect("create temp dir");

    let lz4_path = temp_dir.join("data.hct");
    let zstd_path = temp_dir.join("data_zstd.hct");

    // Create LZ4 file
    {
        let file = File::create(&lz4_path).expect("create file");
        let compressor = Lz4Compressor::with_level(CompressionLevel::Default);
        let shape_u64: Vec<u64> = shape.iter().map(|&s| s as u64).collect();
        let mut writer = HctWriter::new(file, CompressionAlgorithm::Lz4, HctDType::I8, shape_u64);
        writer.compress_data(&data, &compressor).expect("compress");
        writer.finish().expect("finish");
    }

    // Create Zstd file
    {
        let file = File::create(&zstd_path).expect("create file");
        let compressor = ZstdCompressor::with_level(CompressionLevel::Default);
        let shape_u64: Vec<u64> = shape.iter().map(|&s| s as u64).collect();
        let mut writer = HctWriter::new(file, CompressionAlgorithm::Zstd, HctDType::I8, shape_u64);
        writer.compress_data(&data, &compressor).expect("compress");
        writer.finish().expect("finish");
    }

    let lz4_size = fs::metadata(&lz4_path).map(|m| m.len()).unwrap_or(0);
    let zstd_size = fs::metadata(&zstd_path).map(|m| m.len()).unwrap_or(0);

    eprintln!("\n=== Decompression Throughput (16MB INT8 data) ===");
    eprintln!("Original:  {:.2} MB", size as f64 / 1_000_000.0);
    eprintln!(
        "LZ4:       {:.2} MB ({:.2}x)",
        lz4_size as f64 / 1_000_000.0,
        size as f64 / lz4_size as f64
    );
    eprintln!(
        "Zstd:      {:.2} MB ({:.2}x)",
        zstd_size as f64 / 1_000_000.0,
        size as f64 / zstd_size as f64
    );

    group.bench_function("lz4_decompress", |b| {
        b.iter(|| {
            let loader = HctLoader::from_file(&lz4_path).expect("load");
            let decompressed = loader.decompress_all().expect("decompress");
            black_box(decompressed.len())
        })
    });

    group.bench_function("zstd_decompress", |b| {
        b.iter(|| {
            let loader = HctLoader::from_file(&zstd_path).expect("load");
            let decompressed = loader.decompress_all().expect("decompress");
            black_box(decompressed.len())
        })
    });

    group.finish();

    // Cleanup
    let _ = fs::remove_dir_all(&temp_dir);
}

/// Benchmark memory-mapped vs buffered loading comparison
fn loading_strategies_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("loading_strategies");

    // 32MB tensor (simulating a typical layer)
    let size = 32 * 1024 * 1024;
    let weights = generate_f32_weights(size);
    let shape = vec![4096, 8192]; // 4096 x 8192 matrix

    let temp_dir = std::env::temp_dir().join(format!("strategy_bench_{}", std::process::id()));
    fs::create_dir_all(&temp_dir).expect("create dir");

    // Create safetensors file
    let st_path = temp_dir.join("model.safetensors");
    let bytes: Vec<u8> = weights.iter().flat_map(|f| f.to_le_bytes()).collect();
    let bytes_static: &'static [u8] = Box::leak(bytes.clone().into_boxed_slice());
    let view = TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes_static).expect("view");
    let mut tensor_map = HashMap::new();
    tensor_map.insert("layer".to_string(), view);
    let serialized = serialize(&tensor_map, &None).expect("serialize");
    fs::write(&st_path, &serialized).expect("write");

    group.throughput(Throughput::Bytes((size * 4) as u64));

    // Buffered read (current approach)
    group.bench_function("safetensors_buffered", |b| {
        b.iter(|| {
            let data = fs::read(&st_path).expect("read");
            let tensors = safetensors::SafeTensors::deserialize(&data).expect("parse");
            black_box(tensors.tensor("layer").expect("get").data().len())
        })
    });

    // Memory-mapped read
    group.bench_function("safetensors_mmap", |b| {
        b.iter(|| {
            let file = File::open(&st_path).expect("open");
            let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
            let tensors = safetensors::SafeTensors::deserialize(&mmap).expect("parse");
            black_box(tensors.tensor("layer").expect("get").data().len())
        })
    });

    group.finish();

    // Cleanup
    let _ = fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// CRITERION CONFIGURATION
// ============================================================================

criterion_group!(
    benches,
    weight_loading_benchmark,
    decompression_throughput_benchmark,
    loading_strategies_benchmark,
);
criterion_main!(benches);
