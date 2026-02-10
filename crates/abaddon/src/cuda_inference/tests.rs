//! Integration tests for the cuda_inference module.
//!
//! These tests require a CUDA device to run. They're marked with `#[ignore]`
//! by default and can be run with `cargo test --features cuda -- --ignored`.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cudarc::driver::CudaDevice;

    use crate::cuda_inference::arch::{Activation, ModelArch, ModelConfig};
    use crate::cuda_inference::cublas::CublasHandle;
    use crate::cuda_inference::tensor::{GpuDType, GpuTensor};

    /// Helper to get CUDA device if available.
    fn get_cuda_device() -> Option<Arc<CudaDevice>> {
        CudaDevice::new(0).ok()
    }

    // ==================== GpuTensor Tests ====================

    #[test]
    #[ignore = "requires CUDA device"]
    fn test_gpu_tensor_zeros() {
        let device = get_cuda_device().expect("CUDA device required");

        let tensor = GpuTensor::zeros(vec![4, 128], GpuDType::F16, device.clone())
            .expect("Failed to create tensor");

        assert_eq!(tensor.shape(), &[4, 128]);
        assert_eq!(tensor.dtype(), GpuDType::F16);
        assert_eq!(tensor.numel(), 512);
        assert_eq!(tensor.size_bytes(), 1024); // 512 elements * 2 bytes
        assert_eq!(tensor.ndim(), 2);
    }

    #[test]
    #[ignore = "requires CUDA device"]
    fn test_gpu_tensor_strides() {
        let device = get_cuda_device().expect("CUDA device required");

        let tensor = GpuTensor::zeros(vec![2, 3, 4], GpuDType::F32, device.clone())
            .expect("Failed to create tensor");

        // Row-major strides: [12, 4, 1]
        assert_eq!(tensor.strides(), &[12, 4, 1]);
    }

    #[test]
    #[ignore = "requires CUDA device"]
    fn test_gpu_tensor_reshape() {
        let device = get_cuda_device().expect("CUDA device required");

        let tensor = GpuTensor::zeros(vec![4, 8], GpuDType::F16, device.clone())
            .expect("Failed to create tensor");

        // Reshape to [2, 16]
        let reshaped = tensor.reshape(vec![2, 16]).expect("Reshape failed");
        assert_eq!(reshaped.shape(), &[2, 16]);
        assert_eq!(reshaped.numel(), 32);

        // Invalid reshape should fail
        let result = tensor.reshape(vec![3, 10]);
        assert!(result.is_err());
    }

    #[test]
    #[ignore = "requires CUDA device"]
    fn test_gpu_tensor_slice() {
        let device = get_cuda_device().expect("CUDA device required");

        let tensor = GpuTensor::zeros(vec![8, 64], GpuDType::F16, device.clone())
            .expect("Failed to create tensor");

        // Slice first 4 rows
        let sliced = tensor.slice_dim0(0, 4).expect("Slice failed");
        assert_eq!(sliced.shape(), &[4, 64]);

        // Slice middle rows
        let sliced = tensor.slice_dim0(2, 6).expect("Slice failed");
        assert_eq!(sliced.shape(), &[4, 64]);

        // Invalid slice should fail
        let result = tensor.slice_dim0(5, 10);
        assert!(result.is_err());
    }

    #[test]
    #[ignore = "requires CUDA device"]
    fn test_gpu_tensor_host_copy() {
        let device = get_cuda_device().expect("CUDA device required");

        // Create tensor and copy data to it
        let mut tensor = GpuTensor::zeros(vec![4], GpuDType::F32, device.clone())
            .expect("Failed to create tensor");

        // Source data: [1.0, 2.0, 3.0, 4.0]
        let src_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        tensor
            .copy_from_host(&src_data)
            .expect("Copy to GPU failed");

        // Copy back and verify
        let mut dst_data = vec![0u8; 16];
        tensor
            .copy_to_host(&mut dst_data)
            .expect("Copy from GPU failed");

        assert_eq!(src_data, dst_data);
    }

    #[test]
    #[ignore = "requires CUDA device"]
    fn test_gpu_tensor_to_host() {
        let device = get_cuda_device().expect("CUDA device required");

        let mut tensor = GpuTensor::zeros(vec![4], GpuDType::F32, device.clone())
            .expect("Failed to create tensor");

        let src_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        tensor
            .copy_from_host(&src_data)
            .expect("Copy to GPU failed");

        // Use to_host convenience method
        let dst_data = tensor.to_host().expect("to_host failed");
        assert_eq!(src_data, dst_data);
    }

    #[test]
    #[ignore = "requires CUDA device"]
    fn test_gpu_tensor_clone() {
        let device = get_cuda_device().expect("CUDA device required");

        let mut tensor = GpuTensor::zeros(vec![8], GpuDType::F32, device.clone())
            .expect("Failed to create tensor");

        // Write some data
        let src: Vec<u8> = (0..8u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
        tensor.copy_from_host(&src).expect("Copy failed");

        // Clone the tensor
        let cloned = tensor.clone_tensor().expect("Clone failed");

        // Verify they have same data
        let mut original_data = vec![0u8; 32];
        let mut cloned_data = vec![0u8; 32];

        tensor
            .copy_to_host(&mut original_data)
            .expect("Copy failed");
        cloned.copy_to_host(&mut cloned_data).expect("Copy failed");

        assert_eq!(original_data, cloned_data);

        // Verify they have different device pointers
        assert_ne!(tensor.device_ptr(), cloned.device_ptr());
    }

    #[test]
    #[ignore = "requires CUDA device"]
    fn test_gpu_tensor_i32() {
        let device = get_cuda_device().expect("CUDA device required");

        let mut tensor = GpuTensor::zeros(vec![4], GpuDType::I32, device.clone())
            .expect("Failed to create I32 tensor");

        assert_eq!(tensor.dtype(), GpuDType::I32);
        assert_eq!(tensor.size_bytes(), 16); // 4 elements * 4 bytes

        // Write position indices
        let positions: Vec<u8> = [0i32, 1, 2, 3]
            .iter()
            .flat_map(|p| p.to_le_bytes())
            .collect();
        tensor.copy_from_host(&positions).expect("Copy failed");

        let result = tensor.to_host().expect("to_host failed");
        assert_eq!(positions, result);
    }

    // ==================== cuBLAS Tests ====================

    #[test]
    #[ignore = "requires CUDA device"]
    fn test_cublas_handle() {
        let device = get_cuda_device().expect("CUDA device required");

        let handle = CublasHandle::new(device.clone()).expect("Failed to create cuBLAS handle");

        handle.set_math_mode(true).expect("Failed to set math mode");
    }

    #[test]
    #[ignore = "requires CUDA device"]
    fn test_cublas_axpy() {
        let device = get_cuda_device().expect("CUDA device required");
        let handle = CublasHandle::new(device.clone()).expect("Failed to create cuBLAS handle");

        let x =
            GpuTensor::zeros(vec![128], GpuDType::F16, device.clone()).expect("Failed to create x");
        let mut y =
            GpuTensor::zeros(vec![128], GpuDType::F16, device.clone()).expect("Failed to create y");

        // This is a placeholder test - actual result depends on implementation
        handle.axpy(1.0, &x, &mut y).expect("axpy failed");
    }

    // ==================== Kernel Tests ====================

    #[test]
    #[ignore = "requires CUDA device"]
    fn test_rmsnorm_kernel() {
        use crate::cuda_inference::kernels::RMSNormKernel;

        let device = get_cuda_device().expect("CUDA device required");
        let mut kernel = RMSNormKernel::new(device.clone()).expect("Failed to create kernel");

        // Create test tensors
        let seq_len = 4;
        let hidden_size = 128;

        let mut input = GpuTensor::zeros(vec![seq_len, hidden_size], GpuDType::F16, device.clone())
            .expect("Failed to create input");

        let weight = GpuTensor::zeros(vec![hidden_size], GpuDType::F16, device.clone())
            .expect("Failed to create weight");

        let mut output =
            GpuTensor::zeros(vec![seq_len, hidden_size], GpuDType::F16, device.clone())
                .expect("Failed to create output");

        // Initialize weight to 1.0
        let ones: Vec<u8> = (0..hidden_size)
            .flat_map(|_| half::f16::from_f32(1.0).to_le_bytes())
            .collect();
        let mut weight_mut = weight;
        weight_mut.copy_from_host(&ones).expect("Copy failed");

        // Run kernel
        kernel
            .forward(&input, &weight_mut, &mut output, 1e-5)
            .expect("RMSNorm failed");
    }

    #[test]
    #[ignore = "requires CUDA device"]
    fn test_rope_kernel() {
        use crate::cuda_inference::kernels::RoPEKernel;

        let device = get_cuda_device().expect("CUDA device required");
        let mut kernel = RoPEKernel::new(device.clone()).expect("Failed to create kernel");

        // Create test tensors [batch, heads, head_dim]
        let batch = 4;
        let heads = 8;
        let head_dim = 64;

        let mut x = GpuTensor::zeros(vec![batch, heads, head_dim], GpuDType::F16, device.clone())
            .expect("Failed to create x");

        let mut positions = GpuTensor::zeros(vec![batch], GpuDType::I32, device.clone())
            .expect("Failed to create positions");

        // Set positions [0, 1, 2, 3]
        let pos_data: Vec<u8> = (0..batch as i32).flat_map(|p| p.to_le_bytes()).collect();
        positions.copy_from_host(&pos_data).expect("Copy failed");

        // Run kernel
        kernel
            .forward(&mut x, &positions, 10000.0, 1.0)
            .expect("RoPE failed");
    }

    #[test]
    #[ignore = "requires CUDA device"]
    fn test_activation_silu() {
        use crate::cuda_inference::kernels::{ActivationKernel, ActivationType};

        let device = get_cuda_device().expect("CUDA device required");
        let mut kernel = ActivationKernel::new(device.clone()).expect("Failed to create kernel");

        let n = 256;
        let mut x =
            GpuTensor::zeros(vec![n], GpuDType::F16, device.clone()).expect("Failed to create x");
        let mut out =
            GpuTensor::zeros(vec![n], GpuDType::F16, device.clone()).expect("Failed to create out");

        // Initialize with some values
        let data: Vec<u8> = (0..n)
            .map(|i| half::f16::from_f32((i as f32 - 128.0) / 64.0))
            .flat_map(|f| f.to_le_bytes())
            .collect();
        x.copy_from_host(&data).expect("Copy failed");

        // Run SiLU
        kernel
            .forward(&x, &mut out, ActivationType::SiLU)
            .expect("SiLU failed");
    }

    #[test]
    #[ignore = "requires CUDA device"]
    fn test_activation_silu_mul() {
        use crate::cuda_inference::kernels::ActivationKernel;

        let device = get_cuda_device().expect("CUDA device required");
        let mut kernel = ActivationKernel::new(device.clone()).expect("Failed to create kernel");

        let n = 256;
        let gate = GpuTensor::zeros(vec![n], GpuDType::F16, device.clone())
            .expect("Failed to create gate");
        let up =
            GpuTensor::zeros(vec![n], GpuDType::F16, device.clone()).expect("Failed to create up");
        let mut out =
            GpuTensor::zeros(vec![n], GpuDType::F16, device.clone()).expect("Failed to create out");

        // Run fused SiLU * up
        kernel
            .silu_mul(&gate, &up, &mut out)
            .expect("SiLU mul failed");
    }

    #[test]
    #[ignore = "requires CUDA device"]
    fn test_fused_gemm_f16() {
        use crate::cuda_inference::kernels::FusedGemmKernel;

        let device = get_cuda_device().expect("CUDA device required");
        let mut kernel = FusedGemmKernel::new(device.clone()).expect("Failed to create kernel");

        // Matrix sizes
        let m = 32; // batch/seq
        let k = 128; // in features
        let n = 256; // out features

        let a = GpuTensor::zeros(vec![m, k], GpuDType::F16, device.clone())
            .expect("Failed to create A");
        let b = GpuTensor::zeros(vec![k, n], GpuDType::F16, device.clone())
            .expect("Failed to create B");
        let mut c = GpuTensor::zeros(vec![m, n], GpuDType::F16, device.clone())
            .expect("Failed to create C");

        // Run F16 GEMM
        kernel.forward_f16(&a, &b, &mut c).expect("F16 GEMM failed");
    }

    #[test]
    #[ignore = "requires CUDA device"]
    fn test_flash_attention() {
        use crate::cuda_inference::kernels::FlashAttentionKernel;

        let device = get_cuda_device().expect("CUDA device required");
        let kernel = FlashAttentionKernel::new(device.clone()).expect("Failed to create kernel");

        // Attention dimensions (4D: [batch, heads, seq, head_dim])
        let batch = 1;
        let seq_len = 16;
        let num_heads = 8;
        let num_kv_heads = 2; // GQA
        let head_dim = 64;

        let q = GpuTensor::zeros(
            vec![batch, num_heads, seq_len, head_dim],
            GpuDType::F16,
            device.clone(),
        )
        .expect("Failed to create Q");

        let k = GpuTensor::zeros(
            vec![batch, num_kv_heads, seq_len, head_dim],
            GpuDType::F16,
            device.clone(),
        )
        .expect("Failed to create K");

        let v = GpuTensor::zeros(
            vec![batch, num_kv_heads, seq_len, head_dim],
            GpuDType::F16,
            device.clone(),
        )
        .expect("Failed to create V");

        let mut out = GpuTensor::zeros(
            vec![batch, num_heads, seq_len, head_dim],
            GpuDType::F16,
            device.clone(),
        )
        .expect("Failed to create output");

        // Run attention (causal)
        kernel
            .forward(&q, &k, &v, &mut out, true)
            .expect("Flash attention failed");
    }

    // ==================== ComputeEngine Tests ====================

    #[test]
    #[ignore = "requires CUDA device"]
    fn test_compute_engine_creation() {
        use crate::cuda_inference::compute::ComputeEngine;

        let device = get_cuda_device().expect("CUDA device required");

        let config = ModelConfig {
            arch: ModelArch::Llama,
            hidden_size: 256,
            intermediate_size: 512,
            num_layers: 2,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            vocab_size: 1000,
            max_seq_len: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling: None,
            attention_bias: false,
            mlp_bias: false,
            hidden_act: Activation::SiLU,
            tie_word_embeddings: true,
            sliding_window: None,
            bos_token_id: 1,
            eos_token_id: 2,
            pad_token_id: None,
        };

        let engine = ComputeEngine::new(config, 128, device.clone());
        assert!(
            engine.is_ok(),
            "Failed to create ComputeEngine: {:?}",
            engine.err()
        );

        let engine = engine.unwrap();
        assert_eq!(engine.config().hidden_size, 256);
        assert_eq!(engine.current_position(), 0);
    }

    #[test]
    #[ignore = "requires CUDA device"]
    fn test_kv_cache() {
        use crate::cuda_inference::kv_cache::KvCache;

        let device = get_cuda_device().expect("CUDA device required");

        let config = ModelConfig {
            arch: ModelArch::Llama,
            hidden_size: 256,
            intermediate_size: 512,
            num_layers: 2,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            vocab_size: 1000,
            max_seq_len: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling: None,
            attention_bias: false,
            mlp_bias: false,
            hidden_act: Activation::SiLU,
            tie_word_embeddings: true,
            sliding_window: None,
            bos_token_id: 1,
            eos_token_id: 2,
            pad_token_id: None,
        };

        let mut cache =
            KvCache::new(&config, 128, device.clone()).expect("Failed to create KV cache");

        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.max_seq_len(), 128);

        // Reset
        cache.reset();
        assert_eq!(cache.seq_len(), 0);
    }

    // ==================== Model Config Tests ====================

    #[test]
    fn test_model_config_llama() {
        let config = ModelConfig {
            arch: ModelArch::Llama,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            vocab_size: 32000,
            max_seq_len: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling: None,
            attention_bias: false,
            mlp_bias: false,
            hidden_act: Activation::SiLU,
            tie_word_embeddings: false,
            sliding_window: None,
            bos_token_id: 1,
            eos_token_id: 2,
            pad_token_id: None,
        };

        assert_eq!(config.arch, ModelArch::Llama);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_kv_heads, 8);
    }

    #[test]
    fn test_model_arch_equality() {
        assert_eq!(ModelArch::Llama, ModelArch::Llama);
        assert_ne!(ModelArch::Llama, ModelArch::Mistral);
        assert_ne!(ModelArch::Qwen, ModelArch::Phi);
    }

    #[test]
    fn test_gpu_dtype_sizes() {
        assert_eq!(GpuDType::F16.size_bytes(), 2);
        assert_eq!(GpuDType::BF16.size_bytes(), 2);
        assert_eq!(GpuDType::F32.size_bytes(), 4);
        assert_eq!(GpuDType::I32.size_bytes(), 4);
        assert_eq!(GpuDType::I8.size_bytes(), 1);
        assert_eq!(GpuDType::U8.size_bytes(), 1);
        assert_eq!(GpuDType::I4.size_bytes(), 1);
    }

    #[test]
    fn test_gpu_dtype_packed() {
        assert!(!GpuDType::F16.is_packed());
        assert!(!GpuDType::F32.is_packed());
        assert!(!GpuDType::I32.is_packed());
        assert!(GpuDType::I4.is_packed());

        assert_eq!(GpuDType::F16.pack_factor(), 1);
        assert_eq!(GpuDType::I4.pack_factor(), 2);
    }

    // ==================== RoPE Scaling Tests ====================

    #[test]
    fn test_rope_scaling_functions() {
        use crate::cuda_inference::kernels::rope::{linear_scaling, ntk_aware_scaling};

        // No extension
        let theta = ntk_aware_scaling(4096, 4096, 10000.0);
        assert!((theta - 10000.0).abs() < 0.01);

        // 2x extension
        let theta = ntk_aware_scaling(4096, 8192, 10000.0);
        assert!(theta > 10000.0);

        // Linear scaling
        let scale = linear_scaling(4096, 4096);
        assert!((scale - 1.0).abs() < 0.01);

        let scale = linear_scaling(4096, 8192);
        assert!((scale - 0.5).abs() < 0.01);
    }

    // ==================== Activation Type Tests ====================

    #[test]
    fn test_activation_type_equality() {
        use crate::cuda_inference::kernels::ActivationType;

        assert_eq!(ActivationType::SiLU, ActivationType::SiLU);
        assert_ne!(ActivationType::SiLU, ActivationType::GELUFast);
        assert_ne!(ActivationType::GELUFast, ActivationType::GELUTanh);
        assert_ne!(ActivationType::GELUTanh, ActivationType::ReLU);
    }

    // ==================== E2E Integration Tests ====================

    /// Test loading HCT model weights.
    ///
    /// This test requires:
    /// 1. A CUDA device
    /// 2. SmolLM2-135M converted to HCT format at test_models/smollm2-135m-int4/
    #[test]
    #[ignore = "requires CUDA device and HCT model files"]
    fn test_load_hct_weights() {
        use crate::cuda_inference::weight_store::WeightStore;

        let model_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_models/smollm2-135m-int4");

        if !model_dir.exists() {
            eprintln!(
                "Skipping test: model directory not found at {:?}",
                model_dir
            );
            return;
        }

        let weights =
            WeightStore::load_hct(&model_dir, None, 0).expect("Failed to load HCT weights");

        // Verify config was parsed correctly
        assert_eq!(weights.config.hidden_size, 576);
        assert_eq!(weights.config.num_layers, 30);
        assert_eq!(weights.config.num_attention_heads, 9);
        assert_eq!(weights.config.num_kv_heads, 3);
        assert_eq!(weights.config.vocab_size, 49152);

        // Verify layers were loaded
        assert_eq!(weights.layers.len(), 30);

        // Verify embedding shape
        assert_eq!(weights.embed_tokens.shape()[0], 49152);
        assert_eq!(weights.embed_tokens.shape()[1], 576);

        // Verify layer 0 weight shapes and formats
        eprintln!("Layer 0 shapes and formats:");
        eprintln!(
            "  q_proj: {:?}, format: {:?}",
            weights.layers[0].q_proj.shape, weights.layers[0].q_proj.format
        );
        eprintln!(
            "  k_proj: {:?}, format: {:?}",
            weights.layers[0].k_proj.shape, weights.layers[0].k_proj.format
        );
        eprintln!(
            "  v_proj: {:?}, format: {:?}",
            weights.layers[0].v_proj.shape, weights.layers[0].v_proj.format
        );
        eprintln!(
            "  o_proj: {:?}, format: {:?}",
            weights.layers[0].o_proj.shape, weights.layers[0].o_proj.format
        );
        eprintln!(
            "  gate_proj: {:?}, format: {:?}",
            weights.layers[0].gate_proj.shape, weights.layers[0].gate_proj.format
        );
        eprintln!(
            "  up_proj: {:?}, format: {:?}",
            weights.layers[0].up_proj.shape, weights.layers[0].up_proj.format
        );
        eprintln!(
            "  down_proj: {:?}, format: {:?}",
            weights.layers[0].down_proj.shape, weights.layers[0].down_proj.format
        );
        eprintln!(
            "  embed_tokens data shape: {:?}",
            weights.embed_tokens.shape()
        );

        // Verify o_proj shape is [hidden_size, hidden_size] = [576, 576]
        assert_eq!(weights.layers[0].o_proj.shape, (576, 576));

        // Verify weights are INT4
        use crate::cuda_inference::weight_store::QuantFormat;
        assert!(matches!(weights.layers[0].o_proj.format, QuantFormat::Int4));

        eprintln!(
            "Successfully loaded SmolLM2-135M: {:.2} MB GPU memory",
            weights.memory_used as f64 / 1024.0 / 1024.0
        );
    }

    /// Test creating ComputeEngine from loaded HCT weights.
    #[test]
    #[ignore = "requires CUDA device and HCT model files"]
    fn test_compute_engine_from_hct() {
        use crate::cuda_inference::compute::ComputeEngine;
        use crate::cuda_inference::weight_store::WeightStore;

        let model_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_models/smollm2-135m-int4");

        if !model_dir.exists() {
            eprintln!("Skipping test: model directory not found");
            return;
        }

        let weights = WeightStore::load_hct(&model_dir, None, 0).expect("Failed to load weights");

        let _engine = ComputeEngine::new(
            weights.config.clone(),
            512, // max_seq_len
            weights.device().clone(),
        )
        .expect("Failed to create ComputeEngine");

        eprintln!("ComputeEngine created from HCT weights successfully");
    }

    /// Test full forward pass with sample tokens.
    ///
    /// This is the main E2E integration test that:
    /// 1. Loads the model from HCT files
    /// 2. Creates a ComputeEngine
    /// 3. Runs a forward pass with sample input
    /// 4. Verifies output logits
    #[test]
    #[ignore = "requires CUDA device and HCT model files"]
    fn test_e2e_forward_pass() {
        use crate::cuda_inference::compute::ComputeEngine;
        use crate::cuda_inference::weight_store::WeightStore;

        let model_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_models/smollm2-135m-int4");

        if !model_dir.exists() {
            eprintln!("Skipping test: model directory not found");
            return;
        }

        // Load weights
        let weights = WeightStore::load_hct(&model_dir, None, 0).expect("Failed to load weights");

        eprintln!(
            "Loaded weights: {} layers, {} MB",
            weights.config.num_layers,
            weights.memory_used as f64 / 1024.0 / 1024.0
        );

        // Create compute engine
        let mut engine = ComputeEngine::new(
            weights.config.clone(),
            128, // max_seq_len for testing
            weights.device().clone(),
        )
        .expect("Failed to create engine");

        // Sample input tokens (e.g., "Hello" might be token 9906 in some tokenizers)
        // Using simple test tokens that are valid for vocab_size = 49152
        let input_tokens: Vec<u32> = vec![1, 7592, 2]; // BOS, "Hello", EOS

        // Run forward pass
        let start = std::time::Instant::now();
        let output = engine
            .forward(&input_tokens, &weights, 0)
            .expect("Forward pass failed");
        let elapsed = start.elapsed();

        // Verify output shape
        let output_shape = output.shape();
        assert_eq!(output_shape.len(), 2);
        assert_eq!(output_shape[0], input_tokens.len()); // seq_len
        assert_eq!(output_shape[1], 49152); // vocab_size

        eprintln!(
            "Forward pass successful: output shape {:?}, time: {:?}",
            output_shape, elapsed
        );

        // Get logits for sampling
        let logits = engine.get_logits().expect("Failed to get logits");

        // Verify logits shape (last token's logits)
        assert_eq!(logits.shape().len(), 1);
        assert_eq!(logits.shape()[0], 49152);

        eprintln!("E2E test passed!");
    }

    /// Performance benchmark for inference throughput.
    ///
    /// Measures:
    /// - Prefill latency (processing prompt)
    /// - Decode latency (per-token generation)
    /// - Tokens per second
    #[test]
    #[ignore = "requires CUDA device and HCT model files"]
    fn test_performance_benchmark() {
        use crate::cuda_inference::compute::ComputeEngine;
        use crate::cuda_inference::weight_store::WeightStore;

        let model_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_models/smollm2-135m-int4");

        if !model_dir.exists() {
            eprintln!("Skipping benchmark: model directory not found");
            return;
        }

        // Load weights
        let weights = WeightStore::load_hct(&model_dir, None, 0).expect("Failed to load weights");

        eprintln!("\n=== Performance Benchmark ===");
        eprintln!("Model: SmolLM2-135M (INT4)");
        eprintln!("Layers: {}", weights.config.num_layers);
        eprintln!("Hidden size: {}", weights.config.hidden_size);
        eprintln!(
            "GPU Memory: {:.2} MB",
            weights.memory_used as f64 / 1024.0 / 1024.0
        );
        eprintln!();

        // Create compute engine
        let mut engine = ComputeEngine::new(
            weights.config.clone(),
            512, // max_seq_len for benchmark
            weights.device().clone(),
        )
        .expect("Failed to create engine");

        // Warmup run
        let warmup_tokens: Vec<u32> = vec![1, 100, 200, 300];
        let _ = engine
            .prefill(&warmup_tokens, &weights)
            .expect("Warmup failed");

        // === Prefill Benchmark ===
        let prompt_lengths = [8, 16, 32, 64, 128];
        eprintln!("--- Prefill Latency ---");

        for &seq_len in &prompt_lengths {
            let tokens: Vec<u32> = (0..seq_len).map(|i| (i % 1000 + 1) as u32).collect();

            // Run multiple iterations for stable timing
            let iterations = 5;
            let mut total_time = std::time::Duration::ZERO;

            for _ in 0..iterations {
                engine.reset();
                let start = std::time::Instant::now();
                let _ = engine.prefill(&tokens, &weights).expect("Prefill failed");
                // Sync to ensure GPU work completes
                weights.device().synchronize().ok();
                total_time += start.elapsed();
            }

            let avg_time = total_time / iterations as u32;
            let tokens_per_sec = seq_len as f64 / avg_time.as_secs_f64();

            eprintln!(
                "  seq_len={:3}: {:6.2}ms ({:.0} tok/s)",
                seq_len,
                avg_time.as_secs_f64() * 1000.0,
                tokens_per_sec
            );
        }

        // === Decode Benchmark ===
        eprintln!("\n--- Decode Latency (single token) ---");

        // Prefill a prompt first
        let prompt: Vec<u32> = (0..32).map(|i| (i % 1000 + 1) as u32).collect();
        engine.reset();
        let _ = engine.prefill(&prompt, &weights).expect("Prefill failed");

        // Measure decode latency
        let decode_iterations = 20;
        let mut decode_times = Vec::with_capacity(decode_iterations);

        for i in 0..decode_iterations {
            let token = (i % 1000 + 500) as u32;

            let start = std::time::Instant::now();
            let _ = engine.decode(token, &weights).expect("Decode failed");
            weights.device().synchronize().ok();
            decode_times.push(start.elapsed());
        }

        // Calculate statistics
        decode_times.sort();
        let median_decode = decode_times[decode_iterations / 2];
        let min_decode = decode_times[0];
        let max_decode = decode_times[decode_iterations - 1];
        let avg_decode: std::time::Duration =
            decode_times.iter().sum::<std::time::Duration>() / decode_iterations as u32;

        let tokens_per_sec = 1.0 / avg_decode.as_secs_f64();

        eprintln!(
            "  Avg:    {:6.2}ms ({:.0} tok/s)",
            avg_decode.as_secs_f64() * 1000.0,
            tokens_per_sec
        );
        eprintln!("  Median: {:6.2}ms", median_decode.as_secs_f64() * 1000.0);
        eprintln!("  Min:    {:6.2}ms", min_decode.as_secs_f64() * 1000.0);
        eprintln!("  Max:    {:6.2}ms", max_decode.as_secs_f64() * 1000.0);

        eprintln!("\n=== Benchmark Complete ===");
    }
}
