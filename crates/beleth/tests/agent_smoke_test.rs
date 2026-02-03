//! Runtime smoke test for the Beleth agent framework.
//!
//! Tests that the Agent ReAct loop can run end-to-end with a real model
//! and real built-in tools (calculator, JSON, datetime).
//!
//! Run with:
//! ```sh
//! INFERNUM_SMOKE=1 cargo test -p beleth --test agent_smoke_test -- --ignored --nocapture
//! ```

use std::sync::Arc;

use abaddon::{Engine, EngineConfig};
use beleth::{Agent, ToolRegistry};

/// The smallest instruction-tuned model on HuggingFace Hub (~270 MB).
const SMOKE_MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";

/// Returns `true` if the smoke test env var is not set (i.e. test should skip).
fn should_skip() -> bool {
    std::env::var("INFERNUM_SMOKE").is_err()
}

// ============================================================================
// Test F: Agent ReAct loop with tools
// ============================================================================

#[tokio::test]
#[ignore]
async fn smoke_agent_react_loop() {
    if should_skip() {
        return;
    }

    eprintln!("[smoke] Loading engine for agent test...");
    let config = EngineConfig::builder()
        .model(SMOKE_MODEL)
        .build()
        .expect("EngineConfig build failed");

    let engine = Engine::new(config)
        .await
        .expect("Engine creation failed");

    let engine = Arc::new(engine);

    // Build an agent with built-in tools and the real engine.
    let mut agent = Agent::builder()
        .id("smoke-test-agent")
        .system_prompt("You are a helpful assistant. Use the calculator tool when asked math questions.")
        .model(SMOKE_MODEL)
        .max_iterations(5)
        .tools(ToolRegistry::with_builtins())
        .engine(engine)
        .build();

    eprintln!("[smoke] Running agent with objective: 'What is 2 + 2?'");

    // The key assertion: the agent completes without panicking or timing out.
    // SmolLM2-135M may not follow the ReAct format perfectly, but the loop
    // should still terminate (via max_iterations or a final answer parse).
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(120),
        agent.run("What is 2 + 2?"),
    )
    .await;

    match result {
        Ok(Ok(answer)) => {
            eprintln!("[smoke] Agent answer: {answer:?}");
            assert!(!answer.is_empty(), "Agent should produce a non-empty answer");
        }
        Ok(Err(e)) => {
            // Agent errors are acceptable for a 135M model that may not follow
            // ReAct format. The important thing is it didn't panic.
            eprintln!("[smoke] Agent returned error (acceptable for SmolLM2): {e}");
        }
        Err(_) => {
            panic!("Agent timed out after 120 seconds - possible infinite loop");
        }
    }
}
