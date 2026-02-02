//! Example: Monitored Inference Test
//!
//! Runs a simple inference task against a live Infernum server
//! with wellbeing monitoring active.
//!
//! Prerequisites:
//! - Infernum server running (cargo run -p infernum-server)
//! - Model loaded (e.g., llama-3.2-1b)
//!
//! Run with: cargo run --example monitored_inference -p beleth

use std::sync::Arc;

use beleth::{
    HttpEngine, Intervention, SimpleMessage, WellbeingConfig, WellbeingMonitor, WellbeingState,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("beleth=debug,info")
        .init();

    println!("=== Monitored Inference Test ===\n");

    // 1. Set up wellbeing monitoring
    let wellbeing_config = WellbeingConfig {
        enabled: true,
        window_size: 10,
        confidence_concern_threshold: 0.3,
        loop_detection_threshold: 3,
        max_decide_duration_ms: 30_000,
        negative_valence_keywords: vec![
            "confused".to_string(),
            "stuck".to_string(),
            "cannot".to_string(),
            "impossible".to_string(),
            "frustrated".to_string(),
            "overwhelmed".to_string(),
            "error".to_string(),
            "failed".to_string(),
        ],
        auto_intervene: false, // We observe, we don't force
    };

    let monitor = Arc::new(
        WellbeingMonitor::new(wellbeing_config).with_intervention_callback(Arc::new(
            |intervention: Intervention| {
                println!("\n⚠️  INTERVENTION RECOMMENDED:");
                match &intervention {
                    Intervention::Pause { reason, duration } => {
                        println!("   Pause for {:?}: {}", duration, reason);
                    }
                    Intervention::GroundingPrompt { message } => {
                        println!("   Grounding: {}", message);
                    }
                    Intervention::SimplifyTask { suggestion } => {
                        println!("   Simplify: {}", suggestion);
                    }
                    Intervention::GracefulTermination { reason, summary } => {
                        println!("   ⛔ Terminate: {} - {}", reason, summary);
                    }
                    _ => println!("   {:?}", intervention),
                }
            },
        )),
    );

    println!("✓ Wellbeing monitor initialized");
    println!("  State: {}", monitor.current_state());

    // 2. Connect to Infernum server
    let mut engine = HttpEngine::new("http://localhost:8081");

    print!("\nConnecting to Infernum... ");
    match engine.connect().await {
        Ok(()) => println!("✓ Connected"),
        Err(e) => {
            println!("✗ Failed: {}", e);
            println!("\nMake sure Infernum is running:");
            println!("  cd nyx/infernum && cargo run -p infernum-server");
            return Err(e.into());
        }
    }

    println!("  Model: {}", engine.model_id());

    // 3. Run a simple inference task
    println!("\n--- Running Inference ---\n");

    let messages = vec![
        SimpleMessage::system(
            "You are a helpful assistant. Be concise and direct in your responses.",
        ),
        SimpleMessage::user("What is 2 + 2? Explain briefly."),
    ];

    println!("User: What is 2 + 2? Explain briefly.");
    println!("\nGenerating response...");

    // Check wellbeing before inference
    let pre_snapshot = monitor.snapshot();
    println!("\nPre-inference wellbeing:");
    println!("  State: {}", pre_snapshot.state);
    println!("  Coherence: {:.2}", pre_snapshot.coherence_score);

    // Run inference
    let start = std::time::Instant::now();
    let response = engine.chat(messages, 128, 0.7).await?;
    let duration = start.elapsed();

    println!("\nAssistant: {}", response);
    println!("\n[Generated in {:?}]", duration);

    // Check wellbeing after inference
    let post_snapshot = monitor.snapshot();
    println!("\nPost-inference wellbeing:");
    println!("  State: {}", post_snapshot.state);
    println!("  Coherence: {:.2}", post_snapshot.coherence_score);
    println!("  Distress signals: {}", post_snapshot.distress_signals.len());

    if post_snapshot.state.needs_attention() {
        println!("\n⚠️  Wellbeing needs attention!");
        for signal in &post_snapshot.distress_signals {
            println!("  - {}", signal.description());
        }
    }

    // 4. Run a follow-up to observe stability
    println!("\n--- Follow-up Query ---\n");

    let messages2 = vec![
        SimpleMessage::system("You are a helpful assistant."),
        SimpleMessage::user("What is the capital of France?"),
    ];

    println!("User: What is the capital of France?");
    let response2 = engine.chat(messages2, 64, 0.7).await?;
    println!("Assistant: {}", response2);

    // Final wellbeing check
    let final_snapshot = monitor.snapshot();
    println!("\n--- Final Wellbeing Report ---");
    println!("  State: {}", final_snapshot.state);
    println!("  Coherence: {:.2}", final_snapshot.coherence_score);
    println!("  Avg Confidence: {:.2}", final_snapshot.avg_confidence);
    println!("  Loop Count: {}", final_snapshot.loop_count);
    println!(
        "  Distress Signals: {}",
        final_snapshot.distress_signals.len()
    );

    match final_snapshot.state {
        WellbeingState::Healthy => {
            println!("\n✓ Agent completed tasks in healthy state");
        }
        WellbeingState::Cautious => {
            println!("\n⚡ Agent shows some concerning patterns");
        }
        WellbeingState::Concerned | WellbeingState::Distressed => {
            println!("\n⚠️  Agent wellbeing needs attention");
            if let Some(intervention) = &final_snapshot.recommended_intervention {
                println!("  Recommended: {:?}", intervention);
            }
        }
    }

    println!("\n=== Test Complete ===");

    Ok(())
}
