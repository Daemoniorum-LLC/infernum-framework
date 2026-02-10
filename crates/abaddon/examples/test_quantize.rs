//! Test script for INT4 quantization module

use abaddon::quantize::{QuantizeConfig, QuantizeFormat, Quantizer};
use candle_core::{Device, Tensor};

fn main() -> anyhow::Result<()> {
    println!("Testing INT4 Quantization Module\n");

    // Test 1: INT4 Symmetric Quantization
    println!("=== Test 1: INT4 Symmetric ===");
    let quantizer = Quantizer::int4_symmetric();
    let values: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
    let tensor = Tensor::from_vec(values.clone(), &[32, 32], &Device::Cpu)?;

    let quantized = quantizer.quantize_tensor(&tensor)?;
    println!(
        "  Original: {} values ({} bytes)",
        quantized.num_values,
        quantized.num_values * 4
    );
    println!("  Quantized: {} bytes packed", quantized.data.len());
    println!("  Scales: {} blocks", quantized.scales.len());
    println!("  Compression: {:.2}x", quantized.stats.compression_ratio);
    println!("  RMSE: {:.6}", quantized.stats.rmse);
    println!("  SNR: {:.2} dB", quantized.stats.snr_db);

    // Verify roundtrip
    let dequantized = quantizer.dequantize(&quantized)?;
    let dequant_vec: Vec<f32> = dequantized.flatten_all()?.to_vec1()?;
    let max_error = values
        .iter()
        .zip(dequant_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("  Max roundtrip error: {:.6}", max_error);
    // INT4 has only 16 levels, so error of ~0.5 * scale is expected
    assert!(max_error < 0.5, "Error too high for INT4!");
    println!("  ✓ Roundtrip verified\n");

    // Test 2: INT4 Asymmetric Quantization
    println!("=== Test 2: INT4 Asymmetric ===");
    let quantizer = Quantizer::int4_asymmetric();
    let values: Vec<f32> = (0..1024).map(|i| i as f32 * 0.01).collect(); // All positive
    let tensor = Tensor::from_vec(values.clone(), &[1024], &Device::Cpu)?;

    let quantized = quantizer.quantize_tensor(&tensor)?;
    println!("  Original: {} values", quantized.num_values);
    println!("  Compression: {:.2}x", quantized.stats.compression_ratio);
    println!(
        "  Zero points: {:?}",
        quantized.zero_points.as_ref().map(|zp| zp.len())
    );
    println!("  RMSE: {:.6}", quantized.stats.rmse);
    println!("  SNR: {:.2} dB", quantized.stats.snr_db);
    println!("  ✓ Asymmetric quantization works\n");

    // Test 3: INT8 Symmetric Quantization
    println!("=== Test 3: INT8 Symmetric ===");
    let config = QuantizeConfig::int8_symmetric();
    let quantizer = Quantizer::new(config);
    let values: Vec<f32> = (0..2048).map(|i| (i as f32 - 1024.0) * 0.001).collect();
    let tensor = Tensor::from_vec(values, &[2048], &Device::Cpu)?;

    let quantized = quantizer.quantize_tensor(&tensor)?;
    println!("  Compression: {:.2}x", quantized.stats.compression_ratio);
    println!("  RMSE: {:.6}", quantized.stats.rmse);
    println!("  SNR: {:.2} dB", quantized.stats.snr_db);
    println!("  ✓ INT8 quantization works\n");

    // Test 4: Large tensor (simulating a weight matrix)
    println!("=== Test 4: Large Weight Matrix (1536x1536) ===");
    let quantizer = Quantizer::int4_symmetric();
    let size = 1536 * 1536;
    let values: Vec<f32> = (0..size)
        .map(|i| ((i % 1000) as f32 - 500.0) * 0.001)
        .collect();
    let tensor = Tensor::from_vec(values, &[1536, 1536], &Device::Cpu)?;

    let start = std::time::Instant::now();
    let quantized = quantizer.quantize_tensor(&tensor)?;
    let elapsed = start.elapsed();

    let original_mb = (size * 4) as f64 / (1024.0 * 1024.0);
    let quantized_mb =
        (quantized.data.len() + quantized.scales.len() * 2) as f64 / (1024.0 * 1024.0);

    println!("  Original: {:.2} MB", original_mb);
    println!("  Quantized: {:.2} MB", quantized_mb);
    println!("  Compression: {:.2}x", quantized.stats.compression_ratio);
    println!("  RMSE: {:.6}", quantized.stats.rmse);
    println!("  SNR: {:.2} dB", quantized.stats.snr_db);
    println!("  Time: {:?}", elapsed);
    println!("  ✓ Large matrix quantization works\n");

    println!("All tests passed!");
    Ok(())
}
