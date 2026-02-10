//! Continuous batching throughput benchmark.

use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use abaddon::cuda_inference::{BatchScheduler, ModelArch, WeightStore};
    use cudarc::driver::CudaDevice;

    println!("=== Continuous Batching Benchmark ===\n");

    let device = CudaDevice::new(0)?;
    let device = Arc::new(device);
    println!("GPU: {:?}", device.name());

    let model_dir =
        "/home/crook/dev2/workspace/nyx/infernum/infernum-complete/test_models/smollm2-135m-int4";

    let weights = WeightStore::load_hct(model_dir, Some(ModelArch::Llama), 0)?;
    println!(
        "Model loaded: {} layers, {} hidden\n",
        weights.config.num_layers, weights.config.hidden_size
    );

    // Test different batch sizes
    let batch_sizes = [1, 2, 4, 8, 16];
    let max_seq_len = 256;
    let tokens_per_request = 32;

    println!("--- Batch Scheduler Performance ---\n");
    println!(
        "{:>10} | {:>12} | {:>12} | {:>12}",
        "Batch Size", "Schedule (µs)", "Prefill (µs)", "Free (µs)"
    );
    println!("{:-<10}-+-{:-<12}-+-{:-<12}-+-{:-<12}", "", "", "", "");

    for &batch_size in &batch_sizes {
        let mut scheduler = BatchScheduler::new(
            &weights.config,
            batch_size,
            max_seq_len,
            Arc::clone(&device),
        )?;

        // Measure scheduling time
        let start = Instant::now();
        for i in 0..batch_size {
            let input_ids: Vec<u32> = (0..tokens_per_request).map(|x| x as u32).collect();
            scheduler.add_request(input_ids, 64);
        }
        let add_time = start.elapsed();

        let start = Instant::now();
        scheduler.schedule()?;
        let schedule_time = start.elapsed();

        // Get prefill requests
        let start = Instant::now();
        let prefill_slots: Vec<usize> = scheduler
            .get_prefill_requests()
            .iter()
            .map(|(slot, _)| *slot)
            .collect();
        let prefill_time = start.elapsed();

        // Transition and complete
        for slot in prefill_slots {
            scheduler.transition_to_decode(slot)?;
            scheduler.add_token(slot, 1)?;
            scheduler.complete_request(slot)?;
        }

        let start = Instant::now();
        let _completed = scheduler.get_completed();
        let free_time = start.elapsed();

        println!(
            "{:>10} | {:>12} | {:>12} | {:>12}",
            batch_size,
            schedule_time.as_micros(),
            prefill_time.as_micros(),
            free_time.as_micros()
        );
    }

    // Measure batched inference throughput
    println!("\n--- Batched Inference Simulation ---\n");

    let batch_size = 8;
    let mut scheduler = BatchScheduler::new(
        &weights.config,
        batch_size,
        max_seq_len,
        Arc::clone(&device),
    )?;

    // Add 100 requests
    let num_requests = 100;
    for i in 0..num_requests {
        let input_ids: Vec<u32> = (0..16).map(|x| x as u32).collect();
        scheduler.add_request(input_ids, 32);
    }

    let start = Instant::now();
    let mut total_tokens = 0;
    let mut iterations = 0;

    while scheduler.has_work() && iterations < 1000 {
        scheduler.schedule()?;

        // Process prefill - collect slots first
        let prefill_slots: Vec<(usize, usize)> = scheduler
            .get_prefill_requests()
            .iter()
            .map(|(slot, req)| (*slot, req.prompt_len))
            .collect();

        for (slot, prompt_len) in prefill_slots {
            total_tokens += prompt_len;
            scheduler.transition_to_decode(slot)?;
        }

        // Process decode - collect slots first
        let decode_slots: Vec<usize> = scheduler
            .get_decode_requests()
            .iter()
            .map(|(slot, _)| *slot)
            .collect();

        for slot in decode_slots {
            total_tokens += 1;
            let complete = scheduler.add_token(slot, 1)?;
            if complete {
                scheduler.complete_request(slot)?;
            }
        }

        iterations += 1;
    }

    let elapsed = start.elapsed();
    let stats = scheduler.stats();

    println!("Total tokens: {}", total_tokens);
    println!("Iterations: {}", iterations);
    println!("Time: {:?}", elapsed);
    println!(
        "Throughput: {:.0} tok/s (scheduler overhead only)",
        total_tokens as f64 / elapsed.as_secs_f64()
    );
    println!("\nFinal stats: {:?}", stats);

    let completed = scheduler.get_completed();
    println!("Completed requests: {}", completed.len());

    println!("\n=== Benchmark Complete ===");
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("Requires 'cuda' feature");
}
