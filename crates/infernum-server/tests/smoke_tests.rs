//! Runtime smoke tests for the Infernum stack.
//!
//! These tests download a real model and run real inference. They are gated
//! behind the `INFERNUM_SMOKE` environment variable and `#[ignore]` to avoid
//! running in CI or casual `cargo test` invocations.
//!
//! Run with:
//! ```sh
//! INFERNUM_SMOKE=1 cargo test -p infernum-server --test smoke_tests -- --ignored --nocapture
//! ```

mod test_helpers;

use serde_json::json;
use test_helpers::{json_body, TestServer};

use abaddon::{Engine, EngineConfig, InferenceEngine};
use infernum_core::{GenerateRequest, SamplingParams};
use infernum_server::{Server, ServerConfig};

/// The smallest instruction-tuned model on HuggingFace Hub (~270 MB).
/// CPU-capable, fast to load, adequate for smoke-testing the full stack.
const SMOKE_MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";

/// Returns `true` if the smoke test env var is not set (i.e. test should skip).
fn should_skip() -> bool {
    std::env::var("INFERNUM_SMOKE").is_err()
}

/// Helper: create an engine config for the smoke model.
fn smoke_engine_config() -> EngineConfig {
    EngineConfig::builder()
        .model(SMOKE_MODEL)
        .build()
        .expect("EngineConfig build failed")
}

/// Helper: create a server config bound to a random port.
fn smoke_server_config() -> ServerConfig {
    ServerConfig::builder()
        .addr("127.0.0.1:0".parse().expect("valid addr"))
        .build()
}

// ============================================================================
// Test A: Server starts without model, health returns 200
// ============================================================================

#[tokio::test]
#[ignore]
async fn smoke_server_health_without_model() {
    if should_skip() {
        return;
    }

    let config = smoke_server_config();
    let server = Server::new(config);
    let test = TestServer::start(server.router()).await;

    // /health returns 200 even without a model loaded.
    let resp = test.get("/health").await;
    assert_eq!(resp.status(), 200, "health should be 200");
    let body = json_body(resp).await;
    assert_eq!(body["status"], "ok");

    // /ready returns 503 because no model is loaded.
    let resp = test.get("/ready").await;
    assert_eq!(resp.status(), 503, "ready should be 503 without model");
    let body = json_body(resp).await;
    assert_eq!(body["ready"], false);

    // POST /v1/chat/completions should fail (model not loaded).
    let resp = test
        .post_json(
            "/v1/chat/completions",
            &json!({
                "model": "test",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }),
        )
        .await;
    assert_ne!(
        resp.status(),
        200,
        "chat completions should fail without model"
    );
}

// ============================================================================
// Test B: Engine loads SmolLM2 and generates tokens
// ============================================================================

#[tokio::test]
#[ignore]
async fn smoke_engine_loads_and_generates() {
    if should_skip() {
        return;
    }

    eprintln!("[smoke] Loading engine with {SMOKE_MODEL}...");
    let engine = Engine::new(smoke_engine_config())
        .await
        .expect("Engine creation failed");

    assert!(engine.is_ready(), "Engine should be ready after creation");

    let request = GenerateRequest::new("Hello, how are you?")
        .with_sampling(SamplingParams::greedy().with_max_tokens(10));

    eprintln!("[smoke] Generating...");
    let response = engine.generate(request).await.expect("Generation failed");

    assert!(
        !response.choices.is_empty(),
        "Should have at least one choice"
    );
    assert!(
        !response.choices[0].text.is_empty(),
        "Generated text should not be empty"
    );
    assert!(
        response.usage.completion_tokens > 0,
        "Should have generated tokens"
    );
    assert!(
        response.usage.prompt_tokens > 0,
        "Should have prompt tokens"
    );

    eprintln!(
        "[smoke] Generated: {:?} ({} prompt + {} completion tokens)",
        response.choices[0].text,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    );
}

// ============================================================================
// Test C: Full HTTP chat completion round-trip
// ============================================================================

#[tokio::test]
#[ignore]
async fn smoke_http_chat_completion() {
    if should_skip() {
        return;
    }

    eprintln!("[smoke] Loading engine for HTTP test...");
    let engine = Engine::new(smoke_engine_config())
        .await
        .expect("Engine creation failed");

    let config = smoke_server_config();
    let server = Server::with_engine(config, engine);
    let test = TestServer::start(server.router()).await;

    // /ready should now return 200 with a model loaded.
    let resp = test.get("/ready").await;
    assert_eq!(resp.status(), 200, "ready should be 200 with model");

    // POST /v1/chat/completions
    let resp = test
        .post_json(
            "/v1/chat/completions",
            &json!({
                "model": SMOKE_MODEL,
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 10
            }),
        )
        .await;
    assert_eq!(resp.status(), 200, "chat completion should succeed");

    let body = json_body(resp).await;
    eprintln!("[smoke] HTTP response: {}", serde_json::to_string_pretty(&body).unwrap_or_default());

    // Verify OpenAI-compatible response structure.
    assert!(
        body["choices"].is_array(),
        "response should have choices array"
    );
    assert!(
        !body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .is_empty(),
        "response content should not be empty"
    );
    assert!(
        body["usage"]["completion_tokens"].as_u64().unwrap_or(0) > 0,
        "should have completion tokens"
    );
}

// ============================================================================
// Test D: Streaming SSE
// ============================================================================

#[tokio::test]
#[ignore]
async fn smoke_streaming_sse() {
    if should_skip() {
        return;
    }

    eprintln!("[smoke] Loading engine for streaming test...");
    let engine = Engine::new(smoke_engine_config())
        .await
        .expect("Engine creation failed");

    let config = smoke_server_config();
    let server = Server::with_engine(config, engine);
    let test = TestServer::start(server.router()).await;

    // POST with stream: true
    let resp = test
        .post_json(
            "/v1/chat/completions",
            &json!({
                "model": SMOKE_MODEL,
                "messages": [{"role": "user", "content": "Count to three"}],
                "max_tokens": 20,
                "stream": true
            }),
        )
        .await;
    assert_eq!(resp.status(), 200, "streaming request should succeed");

    // Read the entire SSE body as text and parse events.
    let body_text = resp.text().await.expect("Failed to read response body");
    eprintln!("[smoke] SSE body ({} bytes):\n{body_text}", body_text.len());

    // Parse SSE events: lines starting with "data: "
    let events: Vec<&str> = body_text
        .lines()
        .filter(|line| line.starts_with("data: "))
        .map(|line| line.strip_prefix("data: ").unwrap_or(line))
        .collect();

    assert!(
        !events.is_empty(),
        "Should have at least one SSE data event"
    );

    // Look for text content events and a done event.
    let mut has_content = false;
    let mut has_done = false;
    for event_str in &events {
        if let Ok(event) = serde_json::from_str::<serde_json::Value>(event_str) {
            // Server emits {"type":"text","content":"..."} for token deltas
            if event["type"] == "text" || event["type"] == "content_delta" {
                has_content = true;
            }
            if event["type"] == "done" {
                has_done = true;
            }
        }
    }

    assert!(has_content, "Should have received text content events");
    assert!(has_done, "Stream should end with a done event");
}

// ============================================================================
// Test E: Tool schema in request (parsing pipeline)
// ============================================================================

#[tokio::test]
#[ignore]
async fn smoke_tool_schema_in_request() {
    if should_skip() {
        return;
    }

    eprintln!("[smoke] Loading engine for tool schema test...");
    let engine = Engine::new(smoke_engine_config())
        .await
        .expect("Engine creation failed");

    let config = smoke_server_config();
    let server = Server::with_engine(config, engine);
    let test = TestServer::start(server.router()).await;

    // POST with tools array - exercises tool formatting + model output + detection pipeline.
    // SmolLM2-135M may not follow the tool calling format, so we only assert 200.
    let resp = test
        .post_json(
            "/v1/chat/completions",
            &json!({
                "model": SMOKE_MODEL,
                "messages": [{"role": "user", "content": "What is 15 * 7?"}],
                "max_tokens": 50,
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "description": "Evaluate a mathematical expression",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "The math expression to evaluate"
                                }
                            },
                            "required": ["expression"]
                        }
                    }
                }]
            }),
        )
        .await;
    assert_eq!(
        resp.status(),
        200,
        "chat completion with tools should succeed"
    );

    let body = json_body(resp).await;
    eprintln!(
        "[smoke] Tool response: {}",
        serde_json::to_string_pretty(&body).unwrap_or_default()
    );

    // The response should have the standard structure regardless of whether
    // the model actually called the tool.
    assert!(
        body["choices"].is_array(),
        "response should have choices array"
    );
    assert!(
        body["usage"]["total_tokens"].as_u64().unwrap_or(0) > 0,
        "should have used tokens"
    );
}
