//! Integration tests for progressive inference loading.
//!
//! These tests verify that the progressive loading path works correctly
//! for HoloTensor models, enabling 405B inference on consumer hardware.

use std::collections::HashMap;
use std::path::Path;
use tempfile::TempDir;

use abaddon::loader::WeightFiles;
use abaddon::{LazyVarBuilder, TensorProvider, TieredConfig, TieredHoloLoader};
use candle_core::{DType, Device, Tensor};
use haagenti::tensor::{CompressionAlgorithm, DType as HctDType, HctWriter};
use haagenti::Lz4Compressor;

/// Helper to create a test HCT file.
fn create_test_hct_file(dir: &Path, name: &str, shape: &[u64]) -> std::path::PathBuf {
    let path = dir.join(format!("{}.hct", name));
    let file = std::fs::File::create(&path).expect("create file");

    let elements: u64 = shape.iter().product();
    let data: Vec<f32> = (0..elements).map(|i| i as f32 * 0.001).collect();
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

/// Helper to create a minimal model directory.
fn create_test_model_dir() -> TempDir {
    let temp_dir = TempDir::new().expect("create temp dir");

    // Create some model weights
    create_test_hct_file(temp_dir.path(), "model_embed_tokens_weight", &[128, 64]);
    create_test_hct_file(
        temp_dir.path(),
        "model_layers_0_self_attn_q_proj_weight",
        &[64, 64],
    );
    create_test_hct_file(
        temp_dir.path(),
        "model_layers_0_self_attn_k_proj_weight",
        &[64, 64],
    );
    create_test_hct_file(
        temp_dir.path(),
        "model_layers_0_self_attn_v_proj_weight",
        &[64, 64],
    );
    create_test_hct_file(
        temp_dir.path(),
        "model_layers_0_self_attn_o_proj_weight",
        &[64, 64],
    );
    create_test_hct_file(
        temp_dir.path(),
        "model_layers_0_mlp_down_proj_weight",
        &[64, 128],
    );
    create_test_hct_file(
        temp_dir.path(),
        "model_layers_0_mlp_up_proj_weight",
        &[128, 64],
    );
    create_test_hct_file(
        temp_dir.path(),
        "model_layers_0_input_layernorm_weight",
        &[64],
    );
    create_test_hct_file(temp_dir.path(), "lm_head_weight", &[128, 64]);

    temp_dir
}

#[test]
fn test_weight_files_holotensor_variant() {
    let temp_dir = create_test_model_dir();

    // Create the WeightFiles::HoloTensor variant
    let weights = WeightFiles::HoloTensor {
        directory: temp_dir.path().to_path_buf(),
        min_quality: 0.7,
        target_quality: 0.95,
        vram_budget: 20 * 1024 * 1024 * 1024, // 20GB
        ram_budget: 64 * 1024 * 1024 * 1024,  // 64GB
    };

    // Verify paths() works
    let paths = weights.paths();
    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0], temp_dir.path());
}

#[test]
fn test_tiered_loader_loads_all_tensors() {
    let temp_dir = create_test_model_dir();

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

    // Get list of tensors
    let tensor_names = loader.tensor_names();
    assert!(
        tensor_names.len() >= 9,
        "Expected at least 9 tensors, got {}",
        tensor_names.len()
    );

    // Load each tensor
    for name in tensor_names {
        let tensor = loader
            .get(&name, &Device::Cpu, DType::F32)
            .expect(&format!("load tensor: {}", name));

        // Verify tensor is valid
        assert!(
            tensor.elem_count() > 0,
            "Tensor {} should have elements",
            name
        );
    }
}

#[test]
fn test_tiered_loader_as_tensor_provider() {
    let temp_dir = create_test_model_dir();

    let config = TieredConfig::default();
    let loader = TieredHoloLoader::new(temp_dir.path(), config, Device::Cpu, DType::F32)
        .expect("create loader");

    // Use as TensorProvider
    let provider: std::sync::Arc<dyn TensorProvider> = std::sync::Arc::new(loader);

    // Verify contains works
    assert!(provider.contains("model.embed_tokens.weight"));
    assert!(provider.contains("model.layers.0.self_attn.q_proj.weight"));

    // Verify get works
    let tensor = provider
        .get("model.embed_tokens.weight", &Device::Cpu, DType::F32)
        .expect("get tensor");
    assert_eq!(tensor.dims(), &[128, 64]);
}

#[test]
fn test_lazy_varbuilder_with_tiered_loader() {
    let temp_dir = create_test_model_dir();

    let config = TieredConfig::default();
    let loader = TieredHoloLoader::new(temp_dir.path(), config, Device::Cpu, DType::F32)
        .expect("create loader");

    let provider: std::sync::Arc<dyn TensorProvider> = std::sync::Arc::new(loader);
    let lazy_vb = LazyVarBuilder::new(std::sync::Arc::clone(&provider), Device::Cpu, DType::F32);

    // Get a tensor via LazyVarBuilder
    let tensor = lazy_vb
        .get("model.embed_tokens.weight")
        .expect("get tensor");
    assert_eq!(tensor.dims(), &[128, 64]);

    // Get another tensor
    let tensor = lazy_vb
        .get("model.layers.0.self_attn.q_proj.weight")
        .expect("get tensor");
    assert_eq!(tensor.dims(), &[64, 64]);
}

#[test]
fn test_progressive_loading_collects_all_tensors() {
    let temp_dir = create_test_model_dir();

    let config = TieredConfig::default();
    let loader = TieredHoloLoader::new(temp_dir.path(), config, Device::Cpu, DType::F32)
        .expect("create loader");

    // Start background streaming
    loader.start_background_streaming();
    assert!(loader.is_streaming());

    // Create provider
    let provider: std::sync::Arc<dyn TensorProvider> = std::sync::Arc::new(loader);
    let lazy_vb = LazyVarBuilder::new(std::sync::Arc::clone(&provider), Device::Cpu, DType::F32);

    // Collect all tensors
    let tensor_names = provider.tensor_names();
    let mut tensors: HashMap<String, Tensor> = HashMap::new();

    for name in tensor_names {
        match lazy_vb.get(&name) {
            Ok(tensor) => {
                tensors.insert(name.clone(), tensor);
            },
            Err(e) => {
                eprintln!("Failed to load {}: {}", name, e);
            },
        }
    }

    // Verify we loaded the expected number of tensors
    assert!(
        tensors.len() >= 9,
        "Expected at least 9 tensors, got {}",
        tensors.len()
    );

    // Verify specific tensors exist
    assert!(tensors.contains_key("model.embed_tokens.weight"));
    assert!(tensors.contains_key("model.layers.0.self_attn.q_proj.weight"));
    assert!(tensors.contains_key("lm_head.weight"));
}
