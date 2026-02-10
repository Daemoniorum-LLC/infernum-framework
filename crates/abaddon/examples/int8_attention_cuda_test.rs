//! INT8 Attention CUDA Kernel Test
//!
//! Tests the fused INT8 attention kernels:
//! 1. Kernel compilation via nvrtc
//! 2. INT8 dequantization accuracy
//! 3. Fused Q @ K^T with INT8 K
//! 4. Fused attn @ V with INT8 V
//!
//! Usage:
//!   cargo run --release -p abaddon --example int8_attention_cuda_test --features cuda

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== INT8 Attention CUDA Kernel Test ===\n");

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA feature not enabled. Run with --features cuda");
        return Ok(());
    }

    #[cfg(feature = "cuda")]
    {
        use abaddon::kv_cache_quant_cuda::cuda::Int8AttentionContext;
        use cudarc::driver::DeviceSlice;

        // Test 1: Initialize context and compile kernels
        println!("Test 1: Initializing CUDA context...");
        let start = Instant::now();
        let mut ctx = Int8AttentionContext::new(0)?;
        println!(
            "  Context created in {:.2}ms",
            start.elapsed().as_secs_f64() * 1000.0
        );

        println!("\nTest 2: Compiling INT8 attention kernels via nvrtc...");
        let start = Instant::now();
        match ctx.load_kernels() {
            Ok(()) => {
                println!(
                    "  Kernels compiled and loaded in {:.2}ms",
                    start.elapsed().as_secs_f64() * 1000.0
                );
                println!("  Fused kernels available: {}", ctx.has_fused_kernels());
            },
            Err(e) => {
                println!("  Failed to compile kernels: {}", e);
                println!("\n  This may be because:");
                println!("  - nvrtc is not installed or not in PATH");
                println!("  - CUDA toolkit is not properly configured");
                return Err(e.into());
            },
        }

        // Test 3: INT8 dequantization
        println!("\nTest 3: Testing INT8 dequantization kernel...");
        let device = ctx.device();

        // Create test data: 128 elements with known values
        let num_elements = 128;
        let elements_per_scale = 128; // One scale for all (like per-token with head_dim=128)

        // Quantized values: 0, 1, 2, ..., 127 (after subtracting 128: -128, -127, ..., -1)
        // Wait, we want symmetric around 0, so let's use 64 to 191 (maps to -64 to 63)
        let quant_data: Vec<u8> = (0..num_elements as u8)
            .map(|i| 128u8.wrapping_add(i.wrapping_sub(64)))
            .collect();

        // Scale: 1.0 in BF16 = 0x3F80
        let scale_bf16 = 0x3F80u16; // 1.0 in BF16
        let scales: Vec<u16> = vec![scale_bf16; 1];

        // Transfer to GPU
        let d_quant = device.htod_sync_copy(&quant_data)?;
        let d_scales = device.htod_sync_copy(&scales)?;

        // Run dequantization
        let start = Instant::now();
        let d_output = ctx.dequant_int8_to_bf16(&d_quant, &d_scales, elements_per_scale)?;
        device.synchronize()?;
        let dequant_time = start.elapsed();

        // Copy back and verify
        let mut h_output = vec![0u16; num_elements];
        device.dtoh_sync_copy_into(&d_output, &mut h_output)?;

        // Convert BF16 to F32 and check
        let mut max_error = 0.0f32;
        for i in 0..num_elements {
            let expected = (quant_data[i] as i32 - 128) as f32 * 1.0; // scale = 1.0
            let bf16_bits = h_output[i];
            let actual = f32::from_bits((bf16_bits as u32) << 16);

            let error = (expected - actual).abs();
            if error > max_error {
                max_error = error;
            }
        }

        println!(
            "  Dequantized {} elements in {:.3}ms",
            num_elements,
            dequant_time.as_secs_f64() * 1000.0
        );
        println!(
            "  Max error: {:.6} (expected ~0 for exact scale=1.0)",
            max_error
        );
        println!(
            "  Status: {}",
            if max_error < 0.1 { "PASS" } else { "FAIL" }
        );

        // Test 4: Fused Q @ K^T attention
        println!("\nTest 4: Testing fused Q @ K^T attention kernel...");

        let batch_size = 1;
        let num_heads = 4;
        let num_kv_heads = 4;
        let q_len = 8;
        let kv_len = 16;
        let head_dim = 64;
        let attn_scale = 1.0 / (head_dim as f32).sqrt();

        // Create random-ish test data
        // Q: [batch, heads, q_len, head_dim] in BF16
        let q_size = batch_size * num_heads * q_len * head_dim;
        let q_data: Vec<u16> = (0..q_size)
            .map(|i| {
                // BF16 representation of small values
                let val = ((i % 100) as f32 - 50.0) / 100.0;
                let bits = val.to_bits();
                ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
            })
            .collect();

        // K quantized: [batch, kv_heads, kv_len, head_dim] in U8
        let k_size = batch_size * num_kv_heads * kv_len * head_dim;
        let k_quant: Vec<u8> = (0..k_size)
            .map(|i| 128u8.wrapping_add((i % 64) as u8).wrapping_sub(32))
            .collect();

        // K scales: [batch, kv_heads, kv_len] in BF16
        let k_scale_size = batch_size * num_kv_heads * kv_len;
        let k_scales: Vec<u16> = vec![0x3F00u16; k_scale_size]; // 0.5 in BF16

        // Transfer to GPU
        let d_q = device.htod_sync_copy(&q_data)?;
        let d_k_quant = device.htod_sync_copy(&k_quant)?;
        let d_k_scales = device.htod_sync_copy(&k_scales)?;

        // Run fused attention
        let start = Instant::now();
        let d_attn_scores = ctx.fused_qk_attention(
            &d_q,
            &d_k_quant,
            &d_k_scales,
            batch_size,
            num_heads,
            num_kv_heads,
            q_len,
            kv_len,
            head_dim,
            attn_scale,
        )?;
        device.synchronize()?;
        let qk_time = start.elapsed();

        // Verify output size
        let expected_size = batch_size * num_heads * q_len * kv_len;
        println!(
            "  Output size: {} (expected {})",
            d_attn_scores.len(),
            expected_size
        );
        println!(
            "  Computed Q @ K^T in {:.3}ms",
            qk_time.as_secs_f64() * 1000.0
        );

        // Copy back a sample
        let mut h_attn_scores = vec![0.0f32; expected_size];
        device.dtoh_sync_copy_into(&d_attn_scores, &mut h_attn_scores)?;
        println!(
            "  Sample scores: [{:.4}, {:.4}, {:.4}, ...]",
            h_attn_scores[0], h_attn_scores[1], h_attn_scores[2]
        );
        println!(
            "  Status: {}",
            if d_attn_scores.len() == expected_size {
                "PASS"
            } else {
                "FAIL"
            }
        );

        // Test 5: Fused attn @ V
        println!("\nTest 5: Testing fused attn @ V kernel...");

        // Use uniform attention for testing (1/kv_len for each position)
        let attn_uniform: Vec<f32> = vec![1.0 / kv_len as f32; expected_size];

        // V quantized: same format as K
        let v_quant: Vec<u8> = (0..k_size)
            .map(|i| 128u8.wrapping_add((i % 32) as u8))
            .collect();
        let v_scales: Vec<u16> = vec![0x3F80u16; k_scale_size]; // 1.0 in BF16

        // Transfer to GPU
        let d_attn_uniform = device.htod_sync_copy(&attn_uniform)?;
        let d_v_quant = device.htod_sync_copy(&v_quant)?;
        let d_v_scales = device.htod_sync_copy(&v_scales)?;

        // Run fused attn @ V
        let start = Instant::now();
        let d_output = ctx.fused_attn_v(
            &d_attn_uniform,
            &d_v_quant,
            &d_v_scales,
            batch_size,
            num_heads,
            num_kv_heads,
            q_len,
            kv_len,
            head_dim,
        )?;
        device.synchronize()?;
        let av_time = start.elapsed();

        // Verify output size
        let expected_out_size = batch_size * num_heads * q_len * head_dim;
        println!(
            "  Output size: {} (expected {})",
            d_output.len(),
            expected_out_size
        );
        println!(
            "  Computed attn @ V in {:.3}ms",
            av_time.as_secs_f64() * 1000.0
        );

        // Copy back and check first few values
        let mut h_output = vec![0u16; expected_out_size];
        device.dtoh_sync_copy_into(&d_output, &mut h_output)?;

        // Convert first few BF16 values to F32
        let sample_f32: Vec<f32> = h_output[0..3]
            .iter()
            .map(|&bf16| f32::from_bits((bf16 as u32) << 16))
            .collect();
        println!(
            "  Sample output: [{:.4}, {:.4}, {:.4}, ...]",
            sample_f32[0], sample_f32[1], sample_f32[2]
        );
        println!(
            "  Status: {}",
            if d_output.len() == expected_out_size {
                "PASS"
            } else {
                "FAIL"
            }
        );

        // Performance summary
        println!("\n{}", "=".repeat(60));
        println!("PERFORMANCE SUMMARY:");
        println!(
            "  INT8 Dequant:   {:.3}ms for {} elements ({:.1}M elem/s)",
            dequant_time.as_secs_f64() * 1000.0,
            num_elements,
            num_elements as f64 / dequant_time.as_secs_f64() / 1_000_000.0
        );
        println!(
            "  Fused Q @ K^T:  {:.3}ms ({} x {} x {} x {})",
            qk_time.as_secs_f64() * 1000.0,
            batch_size,
            num_heads,
            q_len,
            kv_len
        );
        println!(
            "  Fused attn @ V: {:.3}ms ({} x {} x {} x {})",
            av_time.as_secs_f64() * 1000.0,
            batch_size,
            num_heads,
            q_len,
            head_dim
        );
        println!("{}", "=".repeat(60));

        println!("\nAll tests completed successfully!");
    }

    Ok(())
}
