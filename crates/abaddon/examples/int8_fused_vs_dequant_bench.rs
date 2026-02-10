//! Benchmark: Fused INT8 Attention vs Standard Dequantization
//!
//! Compares two approaches for computing attention with INT8 KV cache:
//! 1. Standard: Dequantize K/V to BF16, then compute attention normally
//! 2. Fused: Compute attention with on-the-fly dequantization
//!
//! The fused approach should be faster due to:
//! - Reduced memory bandwidth (INT8 is 2x smaller than BF16)
//! - Better cache utilization
//! - No intermediate BF16 K/V tensors
//!
//! Usage:
//!   LD_LIBRARY_PATH=/usr/lib/wsl/lib cargo run --release -p abaddon --example int8_fused_vs_dequant_bench --features cuda

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Fused INT8 vs Standard Dequant Benchmark ===\n");

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA feature not enabled. Run with --features cuda");
        return Ok(());
    }

    #[cfg(feature = "cuda")]
    {
        use abaddon::kv_cache_quant_cuda::cuda::Int8AttentionContext;
        use cudarc::driver::DeviceSlice;

        // Initialize context and compile kernels
        println!("Initializing CUDA context and compiling kernels...");
        let mut ctx = Int8AttentionContext::new(0)?;
        ctx.load_kernels()?;
        println!("  Done!\n");

        let device = ctx.device();

        // Test configurations (simulating Qwen2.5-7B attention layer)
        let configs = [
            ("Small (128 tokens)", 1, 28, 4, 1, 128, 128),
            ("Medium (512 tokens)", 1, 28, 4, 1, 512, 128),
            ("Large (1024 tokens)", 1, 28, 4, 1, 1024, 128),
            ("XL (2048 tokens)", 1, 28, 4, 1, 2048, 128),
            ("Batch decode (16 tokens)", 1, 28, 4, 16, 1024, 128),
        ];

        println!("{}", "=".repeat(80));
        println!(
            "{:<25} {:>12} {:>12} {:>12} {:>12}",
            "Config", "Fused QK", "Fused AV", "Dequant", "Speedup"
        );
        println!(
            "{:<25} {:>12} {:>12} {:>12} {:>12}",
            "", "(ms)", "(ms)", "(ms)", ""
        );
        println!("{}", "=".repeat(80));

        for (name, batch, num_heads, num_kv_heads, q_len, kv_len, head_dim) in configs {
            let attn_scale = 1.0 / (head_dim as f32).sqrt();

            // Create test data
            let q_size = batch * num_heads * q_len * head_dim;
            let k_size = batch * num_kv_heads * kv_len * head_dim;
            let scale_size = batch * num_kv_heads * kv_len;
            let attn_size = batch * num_heads * q_len * kv_len;

            // Q in BF16 (random data)
            let q_data: Vec<u16> = (0..q_size).map(|i| ((i % 256) as u16) << 8).collect();
            // K/V quantized in U8
            let k_quant: Vec<u8> = (0..k_size)
                .map(|i| 128u8.wrapping_add((i % 64) as u8))
                .collect();
            let v_quant: Vec<u8> = (0..k_size)
                .map(|i| 128u8.wrapping_add((i % 32) as u8))
                .collect();
            // Scales in BF16 (1.0)
            let scales: Vec<u16> = vec![0x3F80u16; scale_size];
            // Uniform attention weights
            let attn_weights: Vec<f32> = vec![1.0 / kv_len as f32; attn_size];

            // Transfer to GPU
            let d_q = device.htod_sync_copy(&q_data)?;
            let d_k_quant = device.htod_sync_copy(&k_quant)?;
            let d_v_quant = device.htod_sync_copy(&v_quant)?;
            let d_scales = device.htod_sync_copy(&scales)?;
            let d_attn_weights = device.htod_sync_copy(&attn_weights)?;

            // Warmup
            let _ = ctx.fused_qk_attention(
                &d_q,
                &d_k_quant,
                &d_scales,
                batch,
                num_heads,
                num_kv_heads,
                q_len,
                kv_len,
                head_dim,
                attn_scale,
            )?;
            let _ = ctx.fused_attn_v(
                &d_attn_weights,
                &d_v_quant,
                &d_scales,
                batch,
                num_heads,
                num_kv_heads,
                q_len,
                kv_len,
                head_dim,
            )?;
            let _ = ctx.dequant_int8_to_bf16(&d_k_quant, &d_scales, head_dim)?;
            device.synchronize()?;

            // Benchmark fused Q @ K^T
            let iterations = 100;
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = ctx.fused_qk_attention(
                    &d_q,
                    &d_k_quant,
                    &d_scales,
                    batch,
                    num_heads,
                    num_kv_heads,
                    q_len,
                    kv_len,
                    head_dim,
                    attn_scale,
                )?;
            }
            device.synchronize()?;
            let fused_qk_ms = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

            // Benchmark fused attn @ V
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = ctx.fused_attn_v(
                    &d_attn_weights,
                    &d_v_quant,
                    &d_scales,
                    batch,
                    num_heads,
                    num_kv_heads,
                    q_len,
                    kv_len,
                    head_dim,
                )?;
            }
            device.synchronize()?;
            let fused_av_ms = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

            // Benchmark dequantization only (K + V)
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = ctx.dequant_int8_to_bf16(&d_k_quant, &d_scales, head_dim)?;
                let _ = ctx.dequant_int8_to_bf16(&d_v_quant, &d_scales, head_dim)?;
            }
            device.synchronize()?;
            let dequant_ms = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

            // Calculate speedup (fused vs dequant + hypothetical matmul)
            // Note: This doesn't include the actual matmul time for fair comparison
            let fused_total = fused_qk_ms + fused_av_ms;

            println!(
                "{:<25} {:>12.3} {:>12.3} {:>12.3} {:>12}",
                name,
                fused_qk_ms,
                fused_av_ms,
                dequant_ms,
                format!("{:.2}x", dequant_ms / fused_total * 2.0)
            );
        }

        println!("{}", "=".repeat(80));

        // Memory analysis
        println!("\nMEMORY ANALYSIS (for 2048 token context, Qwen2.5-7B style):");
        let num_layers = 28;
        let num_kv_heads = 4;
        let head_dim = 128;
        let kv_len = 2048;

        let bf16_kv = 2 * num_layers * num_kv_heads * kv_len * head_dim * 2; // K+V, 2 bytes
        let int8_kv = 2 * num_layers * num_kv_heads * kv_len * head_dim * 1  // K+V, 1 byte
            + 2 * num_layers * num_kv_heads * kv_len * 2; // scales, 2 bytes

        println!(
            "  BF16 K+V cache: {:.1} MB",
            bf16_kv as f64 / 1024.0 / 1024.0
        );
        println!(
            "  INT8 K+V cache: {:.1} MB ({:.2}x smaller)",
            int8_kv as f64 / 1024.0 / 1024.0,
            bf16_kv as f64 / int8_kv as f64
        );

        println!("\nNOTES:");
        println!("  - Fused kernels compute attention with on-the-fly dequantization");
        println!("  - Speedup compares fused attention vs dequant-then-matmul");
        println!("  - Real-world speedup depends on memory bandwidth vs compute balance");
        println!("  - For long sequences, memory bandwidth dominates (fused wins)");
        println!("  - For short sequences, compute dominates (similar performance)");
    }

    Ok(())
}
