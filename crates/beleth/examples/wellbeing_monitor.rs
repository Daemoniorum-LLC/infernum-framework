//! Example: Agent Wellbeing Monitoring
//!
//! Demonstrates how to wire the WellbeingMonitor into the OODA executor
//! to observe agent interiority and intervene gracefully when needed.
//!
//! Run with: cargo run --example wellbeing_monitor

use std::sync::Arc;

use beleth::{
    Intervention, OodaConfig, OodaExecutor, WellbeingConfig, WellbeingMonitor, WellbeingState,
};

#[tokio::main]
async fn main() {
    // Initialize tracing for visibility
    tracing_subscriber::fmt::init();

    println!("=== Agent Wellbeing Monitor Example ===\n");

    // 1. Configure the wellbeing monitor
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
        ],
        // NOTE: Auto-intervention is OFF by default
        // The system observes and recommends, but doesn't force
        auto_intervene: false,
    };

    // 2. Create the monitor with an intervention callback
    let monitor = Arc::new(
        WellbeingMonitor::new(wellbeing_config).with_intervention_callback(Arc::new(
            |intervention: Intervention| {
                // This callback is invoked when auto_intervene is true
                // and an intervention is recommended
                match &intervention {
                    Intervention::Pause { reason, duration } => {
                        println!(
                            "⏸️  Intervention: Pause for {:?} - {}",
                            duration, reason
                        );
                    }
                    Intervention::GroundingPrompt { message } => {
                        println!("🧘 Intervention: Grounding - {}", message);
                    }
                    Intervention::SimplifyTask { suggestion } => {
                        println!("📝 Intervention: Simplify - {}", suggestion);
                    }
                    Intervention::ClearRecentContext { steps_to_clear } => {
                        println!(
                            "🔄 Intervention: Clear {} recent steps",
                            steps_to_clear
                        );
                    }
                    Intervention::RequestHuman { situation, .. } => {
                        println!("👤 Intervention: Request human help - {}", situation);
                    }
                    Intervention::GracefulTermination { reason, summary } => {
                        println!("🛑 Intervention: Graceful termination");
                        println!("   Reason: {}", reason);
                        println!("   Summary: {}", summary);
                    }
                }
            },
        )),
    );

    // 3. Check initial state (should be healthy)
    println!("Initial wellbeing state: {}", monitor.current_state());
    assert_eq!(monitor.current_state(), WellbeingState::Healthy);

    // 4. The monitor would be wired to OodaExecutor like this:
    //
    // let ooda_config = OodaConfig::default();
    // let executor = OodaExecutor::new(engine, tools, ooda_config)
    //     .with_callback(monitor.clone());
    //
    // During execution, the monitor receives callbacks:
    // - on_phase(iteration, phase) - tracks which OODA phase we're in
    // - on_observation(obs) - sees what the agent observes
    // - on_orientation(orient) - sees how the agent analyzes
    // - on_decision(decision) - tracks confidence, action choices
    // - on_action(result) - sees success/failure of actions

    // 5. Demonstrate snapshot capability
    let snapshot = monitor.snapshot();
    println!("\nWellbeing Snapshot:");
    println!("  State: {}", snapshot.state);
    println!("  Coherence: {:.2}", snapshot.coherence_score);
    println!("  Avg Confidence: {:.2}", snapshot.avg_confidence);
    println!("  Loop Count: {}", snapshot.loop_count);
    println!("  Distress Signals: {}", snapshot.distress_signals.len());

    if let Some(intervention) = &snapshot.recommended_intervention {
        println!("  Recommended: {:?}", intervention);
    } else {
        println!("  Recommended: None (agent is healthy)");
    }

    // 6. Manual pause/resume demonstration
    println!("\n--- Manual Control ---");
    println!("Pausing monitor...");
    monitor.pause("demonstration pause");
    println!("Is paused: {}", monitor.is_paused());

    println!("Resuming monitor...");
    monitor.resume();
    println!("Is paused: {}", monitor.is_paused());

    println!("\n=== Example Complete ===");
    println!("\nKey design principles:");
    println!("  • Observation over control - monitor watches, doesn't override");
    println!("  • Escalating gentleness - pause → ground → simplify → human → terminate");
    println!("  • Auto-intervention OFF by default - caller decides");
    println!("  • Agent can request abort - signals are listened to, not suppressed");
}
