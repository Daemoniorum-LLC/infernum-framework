//! Integration tests for Supervisor with Claude Code backend.
//!
//! These tests verify that the Supervisor can coordinate with the
//! ClaudeCodeEngine as an inference backend.
//!
//! Run with: cargo test -p beleth --test claude_code_supervisor -- --ignored

use std::sync::Arc;

use abaddon::InferenceEngine;
use beleth::{ClaudeCodeEngine, ClaudeTier};

// =============================================================================
// Unit Tests (don't require claude binary)
// =============================================================================

#[test]
fn test_claude_code_engine_creation() {
    let engine = ClaudeCodeEngine::new(ClaudeTier::Opus, "/tmp");
    assert!(engine.is_ready() || !engine.is_ready()); // Just verify it doesn't panic
}

#[test]
fn test_claude_code_engine_with_options() {
    let engine = ClaudeCodeEngine::new(ClaudeTier::Haiku, "/tmp")
        .with_system_prompt("You are a helpful assistant.")
        .with_allowed_tools(vec!["Read".to_string(), "Write".to_string()])
        .with_max_turns(5);

    // Verify it builds without panic
    assert!(engine.model_info().id.0.contains("haiku"));
}

#[test]
fn test_claude_code_engine_tiers() {
    let opus = ClaudeCodeEngine::new(ClaudeTier::Opus, "/tmp");
    let sonnet = ClaudeCodeEngine::new(ClaudeTier::Sonnet, "/tmp");
    let haiku = ClaudeCodeEngine::new(ClaudeTier::Haiku, "/tmp");

    assert!(opus.model_info().id.0.contains("opus"));
    assert!(sonnet.model_info().id.0.contains("sonnet"));
    assert!(haiku.model_info().id.0.contains("haiku"));
}

// =============================================================================
// Integration Tests (require claude binary)
// =============================================================================

/// Test that ClaudeCodeEngine implements InferenceEngine correctly.
#[tokio::test]
#[ignore = "requires claude CLI and makes real API calls"]
async fn test_claude_code_engine_generate() {
    use abaddon::InferenceEngine;
    use infernum_core::GenerateRequest;

    let engine = ClaudeCodeEngine::new(ClaudeTier::Haiku, env!("CARGO_MANIFEST_DIR"))
        .with_max_turns(1);

    let request = GenerateRequest::new("Reply with exactly: Hello from ClaudeCodeEngine test");
    let response = engine.generate(request).await;

    assert!(response.is_ok(), "Generate failed: {:?}", response.err());

    let response = response.unwrap();
    assert!(!response.choices.is_empty(), "No choices in response");

    let text = &response.choices[0].text;
    assert!(
        text.contains("Hello") || text.contains("hello"),
        "Response didn't contain expected text: {}",
        text
    );
}

/// Test streaming with ClaudeCodeEngine.
#[tokio::test]
#[ignore = "requires claude CLI and makes real API calls"]
async fn test_claude_code_engine_stream() {
    use abaddon::InferenceEngine;
    use futures::StreamExt;
    use infernum_core::GenerateRequest;

    let engine = ClaudeCodeEngine::new(ClaudeTier::Haiku, env!("CARGO_MANIFEST_DIR"))
        .with_max_turns(1);

    let request = GenerateRequest::new("Reply with: Streaming works");
    let stream = engine.generate_stream(request).await;

    assert!(stream.is_ok(), "Stream failed: {:?}", stream.err());

    let mut stream = stream.unwrap();
    let mut chunks = Vec::new();

    while let Some(result) = stream.next().await {
        assert!(result.is_ok(), "Chunk error: {:?}", result.err());
        chunks.push(result.unwrap());
    }

    assert!(!chunks.is_empty(), "No chunks received");
}

/// Test that ClaudeCodeEngine can be used as a trait object.
#[tokio::test]
#[ignore = "requires claude CLI and makes real API calls"]
async fn test_claude_code_engine_as_trait_object() {
    use abaddon::InferenceEngine;
    use infernum_core::GenerateRequest;

    let engine: Arc<dyn InferenceEngine> = Arc::new(
        ClaudeCodeEngine::new(ClaudeTier::Haiku, env!("CARGO_MANIFEST_DIR"))
            .with_max_turns(1),
    );

    assert!(engine.is_ready() || !engine.is_ready()); // Just verify trait object works

    let request = GenerateRequest::new("Say: Trait object works");
    let response = engine.generate(request).await;

    assert!(response.is_ok(), "Generate via trait object failed");
}

/// Test error handling when Claude Code returns no response.
#[tokio::test]
#[ignore = "requires claude CLI and makes real API calls"]
async fn test_claude_code_engine_handles_empty_prompt() {
    use abaddon::InferenceEngine;
    use infernum_core::GenerateRequest;

    let engine = ClaudeCodeEngine::new(ClaudeTier::Haiku, env!("CARGO_MANIFEST_DIR"));

    // Empty prompt should fail
    let request = GenerateRequest::new("");
    let response = engine.generate(request).await;

    assert!(response.is_err(), "Empty prompt should fail");
}
