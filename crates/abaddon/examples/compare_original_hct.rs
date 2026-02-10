//! Compare original safetensors model with HCT V3 compressed version.
//!
//! This example loads tensors from both the original model and the compressed
//! HCT version, computing quality metrics to verify inference quality.
//!
//! Usage:
//!   cargo run --release --example compare_original_hct -- \
//!     /path/to/original/model /path/to/hct/model [num_tensors]

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

fn load_safetensors_tensor(path: &Path, tensor_name: &str) -> Result<Option<Tensor>> {
    // Find all safetensors files
    let st_files: Vec<_> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
        .map(|e| e.path())
        .collect();

    // Try each file until we find the tensor
    for st_path in st_files {
        let tensors = candle_core::safetensors::load(&st_path, &Device::Cpu)?;
        if let Some(tensor) = tensors.get(tensor_name) {
            return Ok(Some(tensor.clone()));
        }
    }

    Ok(None)
}

fn compute_quality_metrics(original: &Tensor, compressed: &Tensor) -> Result<QualityMetrics> {
    // Convert both to f32 for comparison
    let orig_f32 = original.to_dtype(DType::F32)?;
    let comp_f32 = compressed.to_dtype(DType::F32)?;

    // Element-wise difference
    let diff = (&orig_f32 - &comp_f32)?;
    let diff_abs = diff.abs()?;

    // Basic statistics
    let orig_mean = orig_f32.mean_all()?.to_scalar::<f32>()?;
    let comp_mean = comp_f32.mean_all()?.to_scalar::<f32>()?;
    let max_diff = diff_abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    let mean_diff = diff_abs.mean_all()?.to_scalar::<f32>()?;

    // MSE and RMSE
    let diff_sq = diff.sqr()?;
    let mse = diff_sq.mean_all()?.to_scalar::<f32>()?;
    let rmse = mse.sqrt();

    // Relative error (Frobenius norm ratio)
    let orig_norm = orig_f32.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    let diff_norm = diff_sq.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    let relative_error = if orig_norm > 1e-10 {
        diff_norm / orig_norm
    } else {
        0.0
    };

    // Cosine similarity
    let dot_product = (&orig_f32 * &comp_f32)?.sum_all()?.to_scalar::<f32>()?;
    let comp_norm = comp_f32.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    let cosine_similarity = if orig_norm > 1e-10 && comp_norm > 1e-10 {
        dot_product / (orig_norm * comp_norm)
    } else {
        1.0
    };

    // Signal-to-Noise Ratio (SNR) in dB
    let signal_power = orig_f32.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let noise_power = mse;
    let snr_db = if noise_power > 1e-10 {
        10.0 * (signal_power / noise_power).log10()
    } else {
        f32::INFINITY
    };

    Ok(QualityMetrics {
        original_mean: orig_mean,
        compressed_mean: comp_mean,
        max_difference: max_diff,
        mean_difference: mean_diff,
        mse,
        rmse,
        relative_error,
        cosine_similarity,
        snr_db,
        quality_score: 1.0 - relative_error,
    })
}

#[derive(Debug)]
struct QualityMetrics {
    original_mean: f32,
    compressed_mean: f32,
    max_difference: f32,
    mean_difference: f32,
    mse: f32,
    rmse: f32,
    relative_error: f32,
    cosine_similarity: f32,
    snr_db: f32,
    quality_score: f32,
}

impl QualityMetrics {
    fn print(&self, tensor_name: &str) {
        println!("\n  Tensor: {}", tensor_name);
        println!("  ──────────────────────────────────────────");
        println!("  Original mean:     {:.6}", self.original_mean);
        println!("  Compressed mean:   {:.6}", self.compressed_mean);
        println!("  Max difference:    {:.6}", self.max_difference);
        println!("  Mean difference:   {:.6}", self.mean_difference);
        println!("  MSE:               {:.2e}", self.mse);
        println!("  RMSE:              {:.6}", self.rmse);
        println!("  Relative error:    {:.4}%", self.relative_error * 100.0);
        println!("  Cosine similarity: {:.6}", self.cosine_similarity);
        println!(
            "  SNR:               {:.1} dB",
            if self.snr_db.is_infinite() {
                "∞".to_string()
            } else {
                format!("{:.1}", self.snr_db)
            }
        );
        println!("  Quality score:     {:.2}%", self.quality_score * 100.0);

        // Status
        if self.cosine_similarity >= 0.99 {
            println!("  Status:            ✓ EXCELLENT (cosine >= 0.99)");
        } else if self.cosine_similarity >= 0.95 {
            println!("  Status:            ✓ GOOD (cosine >= 0.95)");
        } else if self.cosine_similarity >= 0.90 {
            println!("  Status:            ~ ACCEPTABLE (cosine >= 0.90)");
        } else {
            println!("  Status:            ✗ POOR (cosine < 0.90)");
        }
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!(
            "Usage: {} <original_model_dir> <hct_model_dir> [num_tensors]",
            args[0]
        );
        eprintln!();
        eprintln!("Example:");
        eprintln!(
            "  {} /home/crook/models/llama-3.1-70b /home/crook/models/llama-3.1-70b-hct-v3 10",
            args[0]
        );
        std::process::exit(1);
    }

    let original_dir = Path::new(&args[1]);
    let hct_dir = Path::new(&args[2]);
    let num_tensors: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         HCT V3 Quality Comparison Tool                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Original model: {}", original_dir.display());
    println!("HCT model:      {}", hct_dir.display());
    println!("Tensors to compare: {}", num_tensors);

    // Verify directories exist
    if !original_dir.exists() {
        anyhow::bail!(
            "Original model directory not found: {}",
            original_dir.display()
        );
    }
    if !hct_dir.exists() {
        anyhow::bail!("HCT model directory not found: {}", hct_dir.display());
    }

    // Load HCT tensors
    println!("\nLoading HCT tensors...");
    let hct_tensors = abaddon::load_hct_directory_sequential(hct_dir, &Device::Cpu, DType::F32)?;
    println!("Loaded {} HCT tensors", hct_tensors.len());

    // Select tensors to compare (prioritize important ones)
    let priority_tensors = [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
    ];

    let mut tensors_to_compare: Vec<&str> = Vec::new();

    // Add priority tensors that exist
    for name in &priority_tensors {
        if hct_tensors.contains_key(*name) && tensors_to_compare.len() < num_tensors {
            tensors_to_compare.push(name);
        }
    }

    // Fill remaining with other tensors
    for name in hct_tensors.keys() {
        if tensors_to_compare.len() >= num_tensors {
            break;
        }
        if !tensors_to_compare.contains(&name.as_str()) {
            tensors_to_compare.push(name);
        }
    }

    println!("\nComparing {} tensors...", tensors_to_compare.len());
    println!("══════════════════════════════════════════════════════════════════");

    let mut all_metrics: Vec<(String, QualityMetrics)> = Vec::new();
    let mut success_count = 0;
    let mut fail_count = 0;

    for tensor_name in &tensors_to_compare {
        // Get HCT tensor
        let hct_tensor = match hct_tensors.get(*tensor_name) {
            Some(t) => t,
            None => {
                println!("\n  {} - NOT FOUND in HCT", tensor_name);
                fail_count += 1;
                continue;
            },
        };

        // Load original tensor
        let original_tensor = match load_safetensors_tensor(original_dir, tensor_name)? {
            Some(t) => t,
            None => {
                println!("\n  {} - NOT FOUND in original", tensor_name);
                fail_count += 1;
                continue;
            },
        };

        // Compute quality metrics
        match compute_quality_metrics(&original_tensor, hct_tensor) {
            Ok(metrics) => {
                metrics.print(tensor_name);
                all_metrics.push((tensor_name.to_string(), metrics));
                success_count += 1;
            },
            Err(e) => {
                println!("\n  {} - ERROR: {}", tensor_name, e);
                fail_count += 1;
            },
        }
    }

    // Summary
    println!("\n══════════════════════════════════════════════════════════════════");
    println!("                          SUMMARY");
    println!("══════════════════════════════════════════════════════════════════");
    println!();
    println!(
        "Tensors compared: {} / {}",
        success_count,
        tensors_to_compare.len()
    );
    println!("Failed:           {}", fail_count);

    if !all_metrics.is_empty() {
        // Aggregate statistics
        let avg_cosine: f32 = all_metrics
            .iter()
            .map(|(_, m)| m.cosine_similarity)
            .sum::<f32>()
            / all_metrics.len() as f32;
        let min_cosine: f32 = all_metrics
            .iter()
            .map(|(_, m)| m.cosine_similarity)
            .fold(f32::INFINITY, f32::min);
        let max_cosine: f32 = all_metrics
            .iter()
            .map(|(_, m)| m.cosine_similarity)
            .fold(f32::NEG_INFINITY, f32::max);

        let avg_quality: f32 = all_metrics
            .iter()
            .map(|(_, m)| m.quality_score)
            .sum::<f32>()
            / all_metrics.len() as f32;
        let avg_snr: f32 = all_metrics
            .iter()
            .filter(|(_, m)| !m.snr_db.is_infinite())
            .map(|(_, m)| m.snr_db)
            .sum::<f32>()
            / all_metrics
                .iter()
                .filter(|(_, m)| !m.snr_db.is_infinite())
                .count()
                .max(1) as f32;

        let excellent = all_metrics
            .iter()
            .filter(|(_, m)| m.cosine_similarity >= 0.99)
            .count();
        let good = all_metrics
            .iter()
            .filter(|(_, m)| m.cosine_similarity >= 0.95 && m.cosine_similarity < 0.99)
            .count();
        let acceptable = all_metrics
            .iter()
            .filter(|(_, m)| m.cosine_similarity >= 0.90 && m.cosine_similarity < 0.95)
            .count();
        let poor = all_metrics
            .iter()
            .filter(|(_, m)| m.cosine_similarity < 0.90)
            .count();

        println!();
        println!("Cosine Similarity:");
        println!("  Average: {:.6}", avg_cosine);
        println!("  Min:     {:.6}", min_cosine);
        println!("  Max:     {:.6}", max_cosine);
        println!();
        println!("Quality Distribution:");
        println!("  Excellent (≥0.99): {}", excellent);
        println!("  Good (≥0.95):      {}", good);
        println!("  Acceptable (≥0.90): {}", acceptable);
        println!("  Poor (<0.90):      {}", poor);
        println!();
        println!("Average Quality Score: {:.2}%", avg_quality * 100.0);
        println!("Average SNR:           {:.1} dB", avg_snr);

        // Overall verdict
        println!();
        if min_cosine >= 0.95 {
            println!("╔══════════════════════════════════════════════════════════════╗");
            println!("║  VERDICT: ✓ EXCELLENT - All tensors have cosine ≥ 0.95      ║");
            println!("║  The compressed model should produce high-quality inference. ║");
            println!("╚══════════════════════════════════════════════════════════════╝");
        } else if min_cosine >= 0.90 {
            println!("╔══════════════════════════════════════════════════════════════╗");
            println!("║  VERDICT: ~ ACCEPTABLE - All tensors have cosine ≥ 0.90     ║");
            println!("║  The compressed model may have minor quality degradation.    ║");
            println!("╚══════════════════════════════════════════════════════════════╝");
        } else {
            println!("╔══════════════════════════════════════════════════════════════╗");
            println!("║  VERDICT: ✗ POOR - Some tensors have cosine < 0.90          ║");
            println!("║  Consider increasing retention or investigating issues.      ║");
            println!("╚══════════════════════════════════════════════════════════════╝");
        }

        // Find worst tensor
        if let Some((worst_name, worst_metrics)) = all_metrics.iter().min_by(|(_, a), (_, b)| {
            a.cosine_similarity
                .partial_cmp(&b.cosine_similarity)
                .unwrap()
        }) {
            println!();
            println!(
                "Worst tensor: {} (cosine={:.6})",
                worst_name, worst_metrics.cosine_similarity
            );
        }
    }

    Ok(())
}
