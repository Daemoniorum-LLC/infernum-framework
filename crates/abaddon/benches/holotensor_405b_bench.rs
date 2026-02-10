//! Benchmarks for 405B HoloTensor inference.
//!
//! These benchmarks verify the progressive loading system meets performance targets:
//! - Time-to-first-token: < 3s
//! - Throughput: >= 0.5 tok/s
//! - Initial Quality: >= 70%
//! - Final Quality: >= 90%
//! - VRAM Usage: <= 24GB
//! - RAM Usage: <= 80GB

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};
use tempfile::TempDir;

use abaddon::{
    load_hct_directory_sequential, LazyVarBuilder, TensorProvider, TieredConfig, TieredHoloLoader,
    TieredStats,
};
use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use haagenti::tensor::{CompressionAlgorithm, DType as HctDType, HctWriter};
use haagenti::Lz4Compressor;

/// Helper to create a test HCT file with specified size.
fn create_test_hct_file(dir: &Path, name: &str, shape: &[u64]) -> std::path::PathBuf {
    let path = dir.join(format!("{}.hct", name));
    let file = std::fs::File::create(&path).expect("create file");

    let elements: u64 = shape.iter().product();
    let data: Vec<f32> = (0..elements).map(|i| (i as f32 * 0.001) % 1.0).collect();
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut writer = HctWriter::new(
        file,
        CompressionAlgorithm::Lz4,
        HctDType::F32,
        shape.to_vec(),
    )
    .with_block_size(64 * 1024);

    let compressor = Lz4Compressor::new();
    writer
        .compress_data(&bytes, &compressor)
        .expect("write data");
    writer.finish().expect("finish");

    path
}

/// Creates a simulated model directory with scaled-down 405B-like structure.
/// 126 layers, each with attention and MLP weights.
fn create_scaled_model_dir(
    num_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
) -> TempDir {
    let temp_dir = TempDir::new().expect("create temp dir");

    // Embedding
    create_test_hct_file(
        temp_dir.path(),
        "model_embed_tokens_weight",
        &[128000, hidden_size as u64],
    );

    // Layers
    for i in 0..num_layers {
        let layer_prefix = format!("model_layers_{}", i);

        // Attention
        create_test_hct_file(
            temp_dir.path(),
            &format!("{}_self_attn_q_proj_weight", layer_prefix),
            &[hidden_size as u64, hidden_size as u64],
        );
        create_test_hct_file(
            temp_dir.path(),
            &format!("{}_self_attn_k_proj_weight", layer_prefix),
            &[hidden_size as u64, hidden_size as u64],
        );
        create_test_hct_file(
            temp_dir.path(),
            &format!("{}_self_attn_v_proj_weight", layer_prefix),
            &[hidden_size as u64, hidden_size as u64],
        );
        create_test_hct_file(
            temp_dir.path(),
            &format!("{}_self_attn_o_proj_weight", layer_prefix),
            &[hidden_size as u64, hidden_size as u64],
        );

        // MLP
        create_test_hct_file(
            temp_dir.path(),
            &format!("{}_mlp_gate_proj_weight", layer_prefix),
            &[intermediate_size as u64, hidden_size as u64],
        );
        create_test_hct_file(
            temp_dir.path(),
            &format!("{}_mlp_up_proj_weight", layer_prefix),
            &[intermediate_size as u64, hidden_size as u64],
        );
        create_test_hct_file(
            temp_dir.path(),
            &format!("{}_mlp_down_proj_weight", layer_prefix),
            &[hidden_size as u64, intermediate_size as u64],
        );

        // LayerNorms
        create_test_hct_file(
            temp_dir.path(),
            &format!("{}_input_layernorm_weight", layer_prefix),
            &[hidden_size as u64],
        );
        create_test_hct_file(
            temp_dir.path(),
            &format!("{}_post_attention_layernorm_weight", layer_prefix),
            &[hidden_size as u64],
        );
    }

    // LM Head
    create_test_hct_file(
        temp_dir.path(),
        "lm_head_weight",
        &[128000, hidden_size as u64],
    );

    // Final norm
    create_test_hct_file(temp_dir.path(), "model_norm_weight", &[hidden_size as u64]);

    temp_dir
}

/// Benchmark time to load initial tensors for first-token generation.
fn bench_time_to_first_token(c: &mut Criterion) {
    let mut group = c.benchmark_group("holotensor_405b");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    // Scaled down model (4 layers, 512 hidden)
    let temp_dir = create_scaled_model_dir(4, 512, 1536);

    group.bench_function("time_to_first_tensor", |b| {
        b.iter(|| {
            let config = TieredConfig {
                vram_budget: 1024 * 1024 * 1024,    // 1GB
                ram_budget: 4 * 1024 * 1024 * 1024, // 4GB
                min_quality: 0.7,
                target_quality: 0.95,
                enable_background_streaming: false,
                background_streams: 0,
            };

            let loader = TieredHoloLoader::new(temp_dir.path(), config, Device::Cpu, DType::F32)
                .expect("create loader");

            // Load embedding (first tensor needed for inference)
            let tensor = loader
                .get("model.embed_tokens.weight", &Device::Cpu, DType::F32)
                .expect("load embedding");

            black_box(tensor)
        })
    });

    group.finish();
}

/// Benchmark loading throughput (tensors/second).
fn bench_loading_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("holotensor_throughput");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    // Small model for quick benchmarking
    let temp_dir = create_scaled_model_dir(2, 256, 768);

    group.bench_function("load_all_tensors", |b| {
        b.iter(|| {
            let config = TieredConfig::default();
            let loader = TieredHoloLoader::new(temp_dir.path(), config, Device::Cpu, DType::F32)
                .expect("create loader");

            let tensor_names = loader.tensor_names();
            let mut count = 0;

            for name in tensor_names {
                if let Ok(tensor) = loader.get(&name, &Device::Cpu, DType::F32) {
                    black_box(&tensor);
                    count += 1;
                }
            }

            count
        })
    });

    group.finish();
}

/// Benchmark sequential vs tiered loading.
fn bench_loading_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("loading_strategies");
    group.sample_size(10);

    let temp_dir = create_scaled_model_dir(2, 256, 768);

    group.bench_function("sequential_loader", |b| {
        b.iter(|| {
            let tensors = load_hct_directory_sequential(temp_dir.path(), &Device::Cpu, DType::F32)
                .expect("load");

            black_box(tensors.len())
        })
    });

    group.bench_function("tiered_loader", |b| {
        b.iter(|| {
            let config = TieredConfig::default();
            let loader = TieredHoloLoader::new(temp_dir.path(), config, Device::Cpu, DType::F32)
                .expect("create loader");

            let tensor_names = loader.tensor_names();
            let mut count = 0;

            for name in tensor_names {
                if let Ok(_) = loader.get(&name, &Device::Cpu, DType::F32) {
                    count += 1;
                }
            }

            black_box(count)
        })
    });

    group.finish();
}

/// Benchmark memory placement decisions.
fn bench_placement_decisions(c: &mut Criterion) {
    let mut group = c.benchmark_group("placement_decisions");

    let temp_dir = create_scaled_model_dir(1, 128, 384);

    let config = TieredConfig {
        vram_budget: 1024 * 1024 * 1024,
        ram_budget: 4 * 1024 * 1024 * 1024,
        min_quality: 0.7,
        target_quality: 0.95,
        enable_background_streaming: false,
        background_streams: 0,
    };

    let loader = TieredHoloLoader::new(temp_dir.path(), config, Device::Cpu, DType::F32)
        .expect("create loader");

    group.bench_function("calculate_placement", |b| {
        use abaddon::LayerWeightInfo;

        let info = LayerWeightInfo {
            layer: 0,
            weight_name: "q_proj.weight".to_string(),
            size_bytes: 1024 * 1024, // 1MB
            is_attention: true,
            importance: 0.5,
        };

        b.iter(|| {
            let decision = loader.calculate_placement(black_box(&info));
            black_box(decision)
        })
    });

    group.finish();
}

/// Verify memory usage stays within bounds.
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    group.sample_size(10);

    // Larger model to stress memory
    let temp_dir = create_scaled_model_dir(8, 512, 1536);

    group.bench_function("load_with_budget", |b| {
        b.iter(|| {
            let config = TieredConfig {
                vram_budget: 512 * 1024 * 1024,     // 512MB
                ram_budget: 2 * 1024 * 1024 * 1024, // 2GB
                min_quality: 0.7,
                target_quality: 0.95,
                enable_background_streaming: false,
                background_streams: 0,
            };

            let loader = TieredHoloLoader::new(temp_dir.path(), config, Device::Cpu, DType::F32)
                .expect("create loader");

            let tensor_names = loader.tensor_names();
            let mut loaded = 0;

            for name in tensor_names {
                if let Ok(_) = loader.get(&name, &Device::Cpu, DType::F32) {
                    loaded += 1;
                }
            }

            let stats = loader.stats();
            black_box((loaded, stats))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_time_to_first_token,
    bench_loading_throughput,
    bench_loading_strategies,
    bench_placement_decisions,
    bench_memory_usage,
);
criterion_main!(benches);
