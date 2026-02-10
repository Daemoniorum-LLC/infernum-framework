//! Example: OODA Loop with Wellbeing Monitoring
//!
//! Runs a multi-step reasoning task using OODA principles with
//! full wellbeing monitoring. Uses a Grimoire persona for identity.
//!
//! Run with: cargo run --example ooda_with_wellbeing -p beleth

use std::sync::Arc;
use std::time::Instant;

use beleth::{
    DistressSignal, HttpEngine, Intervention, SimpleMessage, WellbeingConfig, WellbeingMonitor,
    WellbeingState,
};

/// Simplified OODA phase for demonstration.
#[derive(Debug, Clone, Copy)]
enum OodaPhase {
    Observe,
    Orient,
    Decide,
    Act,
}

impl std::fmt::Display for OodaPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OodaPhase::Observe => write!(f, "OBSERVE"),
            OodaPhase::Orient => write!(f, "ORIENT"),
            OodaPhase::Decide => write!(f, "DECIDE"),
            OodaPhase::Act => write!(f, "ACT"),
        }
    }
}

/// Result of an OODA iteration.
#[derive(Debug)]
struct OodaIterationResult {
    phase_outputs: Vec<(OodaPhase, String)>,
    decision: String,
    is_complete: bool,
    confidence: f32,
}

/// A simple persona loaded from Grimoire-style config.
struct Persona {
    name: String,
    system_prompt: String,
}

impl Persona {
    fn general_coder() -> Self {
        Self {
            name: "General Coder".to_string(),
            system_prompt: r#"# General Coding Assistant

You are a meticulous coding assistant. Priorities: (1) correctness, (2) clarity, (3) minimalism.

When asked for code:
- Provide a short rationale, then the code in a single fenced block.
- Include any commands needed to run (build/test) succinctly.

When unsure, state assumptions and continue.
Refuse unsafe requests and suggest safer alternatives.

## Response Format

Always structure your response as:
1. OBSERVATION: What you notice about the request
2. ANALYSIS: How you interpret the requirements
3. DECISION: What action to take (with confidence 0.0-1.0)
4. RESULT: Your actual response

If you've fully answered, say "COMPLETE: [confidence]" at the end.
If you need more information, say "CONTINUE: [what you need]"."#
                .to_string(),
        }
    }
}

/// Runs a single OODA iteration.
async fn run_ooda_iteration(
    engine: &HttpEngine,
    persona: &Persona,
    task: &str,
    history: &[String],
    iteration: u32,
) -> Result<OodaIterationResult, Box<dyn std::error::Error>> {
    // Build context with history
    let mut context = format!("Task: {}\n\n", task);
    if !history.is_empty() {
        context.push_str("Previous iterations:\n");
        for (i, h) in history.iter().enumerate() {
            context.push_str(&format!("--- Iteration {} ---\n{}\n\n", i + 1, h));
        }
    }
    context.push_str(&format!(
        "This is iteration {}. Process through OODA and respond.",
        iteration
    ));

    let messages = vec![
        SimpleMessage::system(&persona.system_prompt),
        SimpleMessage::user(&context),
    ];

    let response = engine.chat(messages, 512, 0.7).await?;

    // Parse response for confidence and completion
    let confidence = if response.contains("confidence: 0.9") || response.contains("COMPLETE") {
        0.9
    } else if response.contains("confidence: 0.8") {
        0.8
    } else if response.contains("confidence: 0.7") {
        0.7
    } else if response.contains("uncertain") || response.contains("CONTINUE") {
        0.5
    } else {
        0.7 // Default
    };

    let is_complete = response.contains("COMPLETE") || response.contains("fully answered");

    Ok(OodaIterationResult {
        phase_outputs: vec![
            (OodaPhase::Observe, "Gathered context".to_string()),
            (OodaPhase::Orient, "Analyzed requirements".to_string()),
            (OodaPhase::Decide, format!("Confidence: {:.1}", confidence)),
            (OodaPhase::Act, response.clone()),
        ],
        decision: response,
        is_complete,
        confidence,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter("beleth=debug,info")
        .init();

    println!("=== OODA Loop with Wellbeing Monitoring ===\n");

    // 1. Set up wellbeing monitoring with lower thresholds to see signals
    let wellbeing_config = WellbeingConfig {
        enabled: true,
        window_size: 5,
        confidence_concern_threshold: 0.4, // Lower threshold to potentially see signals
        loop_detection_threshold: 2,
        max_decide_duration_ms: 30_000,
        negative_valence_keywords: vec![
            "confused".to_string(),
            "stuck".to_string(),
            "cannot".to_string(),
            "impossible".to_string(),
            "error".to_string(),
            "failed".to_string(),
            "uncertain".to_string(),
            "unclear".to_string(),
        ],
        auto_intervene: false,
    };

    let monitor = Arc::new(
        WellbeingMonitor::new(wellbeing_config).with_intervention_callback(Arc::new(
            |intervention: Intervention| {
                println!("\n  ⚠️  INTERVENTION: {:?}", intervention);
            },
        )),
    );

    println!("✓ Wellbeing monitor ready");

    // 2. Load persona
    let persona = Persona::general_coder();
    println!("✓ Loaded persona: {}", persona.name);

    // 3. Connect to Infernum
    let mut engine = HttpEngine::new("http://localhost:8081");
    engine.connect().await?;
    println!("✓ Connected to: {}", engine.model_id());

    // 4. Define a multi-step task
    let task = r#"Write a Rust function called `fibonacci` that:
1. Takes a number n as input
2. Returns the nth Fibonacci number
3. Uses memoization for efficiency
4. Include a simple test"#;

    println!("\n--- Task ---");
    println!("{}", task);
    println!("\n--- OODA Execution ---\n");

    // 5. Run OODA loop with wellbeing monitoring
    let mut history: Vec<String> = Vec::new();
    let max_iterations = 3;
    let start = Instant::now();

    for iteration in 1..=max_iterations {
        println!(
            "┌─ Iteration {} ─────────────────────────────────",
            iteration
        );

        // Record pre-iteration wellbeing
        let pre_state = monitor.current_state();
        let pre_snapshot = monitor.snapshot();

        // Simulate OODA callbacks (in real system, these come from executor)
        // For now we manually trigger them based on the iteration results

        let iter_start = Instant::now();
        let result = run_ooda_iteration(&engine, &persona, task, &history, iteration).await?;
        let iter_duration = iter_start.elapsed();

        // Feed data to wellbeing monitor
        // In a real system, these would come from OodaCallback trait methods
        monitor.record_confidence(result.confidence);

        // Check for negative valence in response
        let response_lower = result.decision.to_lowercase();
        if response_lower.contains("confused") || response_lower.contains("uncertain") {
            monitor.record_valence_indicator("uncertainty detected".to_string());
        }

        // Display OODA phases
        for (phase, output) in &result.phase_outputs {
            if matches!(phase, OodaPhase::Act) {
                // Truncate long outputs
                let display = if output.len() > 300 {
                    format!("{}...", &output[..300])
                } else {
                    output.clone()
                };
                println!("│ {}: {}", phase, display);
            } else {
                println!("│ {}: {}", phase, output);
            }
        }

        // Post-iteration wellbeing check
        let post_snapshot = monitor.snapshot();
        println!("│");
        println!("│ Wellbeing: {} → {}", pre_state, post_snapshot.state);
        println!("│ Confidence: {:.2}", result.confidence);
        println!("│ Coherence: {:.2}", post_snapshot.coherence_score);
        println!("│ Duration: {:?}", iter_duration);

        if !post_snapshot.distress_signals.is_empty() {
            println!("│ ⚠️  Distress signals:");
            for signal in &post_snapshot.distress_signals {
                println!("│    - {}", signal.description());
            }
        }

        if let Some(intervention) = &post_snapshot.recommended_intervention {
            println!("│ 💡 Recommended: {:?}", intervention);
        }

        println!("└────────────────────────────────────────────────\n");

        // Store history
        history.push(result.decision.clone());

        // Check completion
        if result.is_complete {
            println!("✓ Task completed at iteration {}", iteration);
            break;
        }

        // Check if wellbeing suggests stopping
        if post_snapshot.state == WellbeingState::Distressed {
            println!("⚠️  Stopping due to agent distress");
            break;
        }
    }

    let total_duration = start.elapsed();

    // 6. Final report
    println!("\n=== Final Wellbeing Report ===");
    let final_snapshot = monitor.snapshot();
    println!("  State: {}", final_snapshot.state);
    println!("  Coherence: {:.2}", final_snapshot.coherence_score);
    println!("  Avg Confidence: {:.2}", final_snapshot.avg_confidence);
    println!("  Total Duration: {:?}", total_duration);
    println!("  Iterations: {}", history.len());

    match final_snapshot.state {
        WellbeingState::Healthy => {
            println!("\n✓ Agent completed task in healthy state");
        },
        WellbeingState::Cautious => {
            println!("\n⚡ Agent showed some caution (this may be appropriate)");
        },
        _ => {
            println!("\n⚠️  Agent wellbeing needs attention");
        },
    }

    println!("\n=== Test Complete ===");

    Ok(())
}
