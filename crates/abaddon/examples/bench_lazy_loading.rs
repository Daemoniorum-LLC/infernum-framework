//! Benchmark comparing eager vs lazy HoloTensor loading.
//!
//! Usage:
//!   cargo run -p abaddon --features cuda --example bench_lazy_loading -- /path/to/model

use std::env;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter("info,abaddon=debug")
        .init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model_dir>", args[0]);
        eprintln!(
            "Example: {} /home/crook/models/llama-3.1-70b-hct-holo",
            args[0]
        );
        std::process::exit(1);
    }

    let model_dir = &args[1];
    println!("\n=== Lazy HoloTensor Loading Benchmark ===\n");
    println!("Model: {}", model_dir);

    // Query GPU info
    println!("\n--- GPU Information ---");
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader",
        ])
        .output()?;
    println!("{}", String::from_utf8_lossy(&output.stdout));

    // Set CUDA library path for WSL
    std::env::set_var("LD_LIBRARY_PATH", "/usr/lib/wsl/lib:/usr/local/cuda/lib64");

    #[cfg(feature = "cuda")]
    {
        use abaddon::cuda_inference::{
            LazyGenerator, LazyWeightConfig, LazyWeightStore, SamplingParams,
        };

        // Test 1: Lazy Loading
        println!("\n--- Test 1: Lazy Loading ---");
        let lazy_start = Instant::now();

        let config = LazyWeightConfig::for_24gb_gpu();
        println!("VRAM budget for layers: {:?}", config.vram_budget);

        let weights_start = Instant::now();
        let weights = match LazyWeightStore::load_holotensor(model_dir, config) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("Failed to load weights: {}", e);
                return Err(e.into());
            },
        };
        let weights_time = weights_start.elapsed();
        println!("Weight indexing time: {:.2?}", weights_time);
        println!(
            "Shared memory loaded: {} MB",
            weights.shared_memory() / (1024 * 1024)
        );
        println!("Total layers: {}", weights.num_layers());

        // Create generator
        let gen_start = Instant::now();
        let mut generator = match LazyGenerator::new(weights, 2048) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Failed to create generator: {}", e);
                return Err(e.into());
            },
        };
        let gen_time = gen_start.elapsed();
        println!("Generator creation time: {:.2?}", gen_time);

        let total_init = lazy_start.elapsed();
        println!("Total initialization: {:.2?}", total_init);

        // Generate a few tokens
        println!("\n--- Generating tokens (lazy) ---");
        let input_ids: Vec<u32> = vec![128000, 2028, 374, 264, 1296]; // "This is a test"

        let params = SamplingParams {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            max_tokens: 20,
            ..Default::default()
        };

        let gen_start = Instant::now();
        match generator.generate(&input_ids, Some(&params)) {
            Ok(tokens) => {
                let gen_time = gen_start.elapsed();
                println!("Generated {} tokens in {:.2?}", tokens.len(), gen_time);
                println!("Tokens: {:?}", tokens);

                let stats = generator.layer_stats();
                println!("\nLayer loading stats:");
                println!("  Layers loaded: {}", stats.layers_loaded);
                println!("  VRAM used: {} MB", stats.vram_used / (1024 * 1024));
                println!("  Total loads: {}", stats.total_loads);
                println!("  Total evictions: {}", stats.total_evictions);
                println!("  Cache hit rate: {:.1}%", stats.hit_rate * 100.0);

                if tokens.len() > 0 {
                    let tokens_per_sec = tokens.len() as f64 / gen_time.as_secs_f64();
                    println!("\nThroughput: {:.2} tokens/sec", tokens_per_sec);
                }
            },
            Err(e) => {
                eprintln!("Generation failed: {}", e);
                return Err(e.into());
            },
        }

        // Report GPU memory after generation
        println!("\n--- GPU Memory After Generation ---");
        let output = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=memory.used,memory.free",
                "--format=csv,noheader",
            ])
            .output()?;
        println!("{}", String::from_utf8_lossy(&output.stdout));
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This benchmark requires the 'cuda' feature.");
        eprintln!("Run with: cargo run -p abaddon --features cuda --example bench_lazy_loading");
    }

    Ok(())
}
