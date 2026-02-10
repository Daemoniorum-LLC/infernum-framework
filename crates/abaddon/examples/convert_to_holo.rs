//! Convert SafeTensors model to HoloTensor HCT format.
//!
//! Usage:
//!   cargo run --release --example convert_to_holo --features cuda -- \
//!     --model "Qwen/Qwen2.5-72B-Instruct" \
//!     --output /tmp/qwen72b-holo \
//!     --gpu
//!
//! For local models:
//!   cargo run --release --example convert_to_holo --features cuda -- \
//!     --model /path/to/model \
//!     --output /tmp/model-holo \
//!     --gpu
//!
//! For maximum throughput with pipeline mode (overlaps I/O with GPU):
//!   cargo run --release --example convert_to_holo --features cuda -- \
//!     --model /path/to/model \
//!     --output /tmp/model-holo \
//!     --gpu --pipeline --producers 8

use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

use abaddon::holotensor::{ConversionConfig, HoloModelConverter};

#[derive(Parser, Debug)]
#[command(name = "convert_to_holo")]
#[command(about = "Convert SafeTensors model to HoloTensor HCT format")]
struct Args {
    /// Model path or HuggingFace repo ID (e.g., "Qwen/Qwen2.5-72B-Instruct")
    #[arg(short, long)]
    model: String,

    /// Output directory for HCT files
    #[arg(short, long)]
    output: PathBuf,

    /// Use GPU for SVD computation (3-4x faster)
    #[arg(long, default_value = "false")]
    gpu: bool,

    /// Number of fragments per tensor (more = higher quality, larger files)
    #[arg(long, default_value = "32")]
    fragments: u16,

    /// Maximum SVD rank (higher = better quality, slower)
    #[arg(long, default_value = "128")]
    max_rank: usize,

    /// Skip quality verification (faster but no quality metrics)
    #[arg(long, default_value = "false")]
    skip_verify: bool,

    /// Number of CPU threads for parallel conversion (non-pipeline mode)
    #[arg(long, default_value = "4")]
    threads: usize,

    /// Fast mode (fewer fragments, lower rank, no verification)
    #[arg(long, default_value = "false")]
    fast: bool,

    /// Enable pipeline mode (overlaps I/O with GPU processing)
    /// Recommended for large models and high-thread-count CPUs
    #[arg(long, default_value = "false")]
    pipeline: bool,

    /// Number of producer threads for pipeline mode (default: CPU count / 2)
    /// Each producer loads and prepares files while GPU processes
    #[arg(long)]
    producers: Option<usize>,

    /// Lossless mode - stores all tensors without LRDF compression.
    /// Results in larger files but exact tensor reconstruction.
    /// Use for testing inference correctness before enabling compression.
    #[arg(long, default_value = "false")]
    lossless: bool,

    /// Use Spectral (DCT) encoding instead of LRDF (SVD).
    /// Spectral encoding uses 2D DCT which benefits from FFT-based IDCT
    /// for 40-80x faster reconstruction on large tensors (>4096 dims).
    /// Best for dense MLP weights. LRDF is better for attention matrices.
    #[arg(long, default_value = "false")]
    spectral: bool,

    /// DCT retention ratio for Spectral encoding (0.0-1.0).
    /// Higher = better quality, larger files. Default: 0.2 (keep 20% of coefficients).
    #[arg(long, default_value = "0.2")]
    retention: f32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         HoloTensor Model Converter (GPU-Accelerated)         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Build configuration
    let config = if args.lossless {
        println!("Mode: LOSSLESS (exact tensor reconstruction, larger files)");
        let mut config = ConversionConfig::lossless();
        config.num_threads = args.threads;
        config
    } else if args.spectral {
        println!("Mode: SPECTRAL (DCT encoding with FFT-accelerated reconstruction)");
        let mut config = if args.gpu {
            #[cfg(feature = "cuda")]
            {
                ConversionConfig::gpu()
            }
            #[cfg(not(feature = "cuda"))]
            {
                println!("Warning: CUDA not enabled, falling back to CPU");
                ConversionConfig::default()
            }
        } else {
            ConversionConfig::default()
        };
        config.encoding = abaddon::holotensor::HolographicEncoding::Spectral;
        config.num_fragments = args.fragments;
        config.verify_quality = !args.skip_verify;
        config.num_threads = args.threads;
        config.retention_ratio = args.retention;
        println!(
            "  Retention: {:.0}% of DCT coefficients",
            args.retention * 100.0
        );
        config
    } else if args.fast {
        println!("Mode: FAST (reduced quality for speed)");
        #[cfg(feature = "cuda")]
        if args.gpu {
            ConversionConfig::gpu_fast()
        } else {
            ConversionConfig::fast()
        }
        #[cfg(not(feature = "cuda"))]
        ConversionConfig::fast()
    } else {
        let mut config = if args.gpu {
            println!("Mode: GPU-ACCELERATED");
            #[cfg(feature = "cuda")]
            {
                ConversionConfig::gpu()
            }
            #[cfg(not(feature = "cuda"))]
            {
                println!("Warning: CUDA not enabled, falling back to CPU");
                ConversionConfig::default()
            }
        } else {
            println!("Mode: CPU (use --gpu for 3-4x speedup)");
            ConversionConfig::default()
        };

        config.num_fragments = args.fragments;
        config.max_rank = args.max_rank;
        config.verify_quality = !args.skip_verify;
        config.num_threads = args.threads;
        config
    };

    // Determine number of producers for pipeline mode
    let num_producers = args.producers.unwrap_or_else(|| {
        // Default: half of CPU count, minimum 4, maximum 16
        let cpus = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(8);
        (cpus / 2).clamp(4, 16)
    });

    println!("Model:      {}", args.model);
    println!("Output:     {}", args.output.display());
    println!("Encoding:   {:?}", config.encoding);
    println!("Fragments:  {}", config.num_fragments);
    if !args.spectral {
        println!("Max Rank:   {}", config.max_rank);
    }
    println!("Verify:     {}", config.verify_quality);
    println!("Lossless:   {}", config.lossless);
    println!("Threads:    {}", config.num_threads);
    #[cfg(feature = "cuda")]
    println!("GPU:        {}", config.use_gpu);
    if args.pipeline {
        println!("Pipeline:   {} producers → 1 GPU consumer", num_producers);
    }
    println!();

    // Create converter
    let converter = HoloModelConverter::new(config);

    println!("Starting conversion...");
    println!();

    let start = Instant::now();

    // Run conversion (pipeline mode if requested with GPU)
    #[cfg(feature = "cuda")]
    let result = if args.pipeline && args.gpu {
        converter
            .convert_model_pipeline(&args.model, &args.output, num_producers)
            .await
    } else {
        converter.convert_model(&args.model, &args.output).await
    };

    #[cfg(not(feature = "cuda"))]
    let result = {
        if args.pipeline {
            println!("Warning: Pipeline mode requires CUDA, using standard conversion");
        }
        converter.convert_model(&args.model, &args.output).await
    };

    match result {
        Ok(metadata) => {
            let elapsed = start.elapsed();

            println!();
            println!("╔══════════════════════════════════════════════════════════════╗");
            println!("║                    Conversion Complete!                       ║");
            println!("╚══════════════════════════════════════════════════════════════╝");
            println!();
            println!("Results:");
            println!("  Layers:           {}", metadata.num_layers);
            println!("  Fragments/tensor: {}", metadata.total_fragments);
            println!(
                "  Original size:    {:.2} GB",
                metadata.original_size as f64 / (1024.0 * 1024.0 * 1024.0)
            );
            println!(
                "  HCT size:         {:.2} GB",
                metadata.hct_size as f64 / (1024.0 * 1024.0 * 1024.0)
            );
            println!(
                "  Compression:      {:.2}x",
                metadata.original_size as f64 / metadata.hct_size as f64
            );
            println!("  Min quality:      {:.4}", metadata.verified_quality);
            println!(
                "  Time elapsed:     {:.1} minutes",
                elapsed.as_secs_f64() / 60.0
            );
            println!();
            println!("Output directory: {}", args.output.display());
            println!();
            println!("To run inference:");
            println!(
                "  cargo run --release -p infernum -- serve --model {} --holotensor",
                args.output.display()
            );
        },
        Err(e) => {
            eprintln!("Conversion failed: {}", e);
            std::process::exit(1);
        },
    }

    Ok(())
}
