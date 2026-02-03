//! Integration tests for test-utils crate.
//!
//! Tests the integration between mock implementations, fixtures,
//! assertions, and server helpers.

use test_utils::{
    // Mock
    MockConfig, MockInferenceEngine, MockError, MockVectorStore, MockDocument,
    // Fixtures
    sample_chat_request, sample_streaming_chat_request,
    sample_embedding_request, sample_batch_embedding_request,
    sample_chat_response, sample_error_response,
    invalid_request_missing_model, invalid_request_empty_messages, invalid_request_bad_temperature,
    TEST_API_KEY_ADMIN, TEST_API_KEY_INFERENCE, TEST_API_KEY_INVALID,
    TEST_MODEL_ID, TEST_EMBEDDING_MODEL_ID,
    // Assertions
    assert_error_code, assert_error_type, assert_has_choices, assert_has_usage,
    assert_header_present, assert_header_value, assert_response_time_ms,
    get_json_string, get_json_i64, get_json_f64, get_json_bool,
    assert_json_string, assert_json_i64,
    // Server
    TestServer, TestRequest, TestResponse,
};
use infernum_core::{GenerateRequest, SamplingParams};
use axum::{routing::get, routing::post, Json, Router};
use serde_json::json;

// ============================================================================
// MockConfig Tests
// ============================================================================

#[test]
fn test_mock_config_default() {
    let config = MockConfig::default();

    assert!(!config.default_response.is_empty());
    assert_eq!(config.latency_ms, 0);
    assert!(!config.should_fail);
    assert!(!config.error_message.is_empty());
}

#[test]
fn test_mock_config_clone() {
    let config = MockConfig {
        default_response: "Custom response".to_string(),
        latency_ms: 100,
        should_fail: true,
        error_message: "Custom error".to_string(),
    };

    let cloned = config.clone();
    assert_eq!(cloned.default_response, config.default_response);
    assert_eq!(cloned.latency_ms, config.latency_ms);
    assert_eq!(cloned.should_fail, config.should_fail);
}

#[test]
fn test_mock_config_debug() {
    let config = MockConfig::default();
    let debug = format!("{:?}", config);
    assert!(debug.contains("MockConfig"));
}

// ============================================================================
// MockInferenceEngine Tests
// ============================================================================

#[tokio::test]
async fn test_mock_engine_new() {
    let engine = MockInferenceEngine::new();
    assert_eq!(engine.call_count().await, 0);
    assert!(engine.last_request().await.is_none());
}

#[tokio::test]
async fn test_mock_engine_default() {
    let engine = MockInferenceEngine::default();
    assert_eq!(engine.call_count().await, 0);
}

#[tokio::test]
async fn test_mock_engine_generate() {
    let engine = MockInferenceEngine::new();
    let request = GenerateRequest::new("Hello, world!")
        .with_sampling(SamplingParams::default().with_max_tokens(100));

    let response = engine.generate(request.clone()).await.expect("generate");

    assert!(!response.choices.is_empty());
    assert!(!response.choices[0].text.is_empty());
    assert_eq!(response.usage.total_tokens, 30);
    assert_eq!(engine.call_count().await, 1);
    assert!(engine.last_request().await.is_some());
}

#[tokio::test]
async fn test_mock_engine_multiple_calls() {
    let engine = MockInferenceEngine::new();

    for i in 0..5 {
        let request = GenerateRequest::new(format!("Request {}", i))
            .with_sampling(SamplingParams::default());
        engine.generate(request).await.expect("generate");
    }

    assert_eq!(engine.call_count().await, 5);
}

#[tokio::test]
async fn test_mock_engine_custom_response() {
    let engine = MockInferenceEngine::new();
    engine.set_response("Custom test response").await;

    let request = GenerateRequest::new("Test")
        .with_sampling(SamplingParams::default());
    let response = engine.generate(request).await.expect("generate");

    assert_eq!(response.choices[0].text, "Custom test response");
}

#[tokio::test]
async fn test_mock_engine_latency() {
    let engine = MockInferenceEngine::new();
    engine.set_latency(50).await;

    let start = std::time::Instant::now();
    let request = GenerateRequest::new("Test")
        .with_sampling(SamplingParams::default());
    engine.generate(request).await.expect("generate");
    let elapsed = start.elapsed();

    assert!(elapsed.as_millis() >= 50);
}

#[tokio::test]
async fn test_mock_engine_failing() {
    let engine = MockInferenceEngine::failing("Connection refused");
    let request = GenerateRequest::new("Test")
        .with_sampling(SamplingParams::default());

    let result = engine.generate(request).await;

    assert!(result.is_err());
    match result.unwrap_err() {
        MockError::SimulatedError(msg) => assert!(msg.contains("Connection refused")),
        _ => panic!("Expected SimulatedError"),
    }
}

#[tokio::test]
async fn test_mock_engine_last_request_tracking() {
    let engine = MockInferenceEngine::new();

    let request1 = GenerateRequest::new("First request")
        .with_sampling(SamplingParams::default());
    engine.generate(request1).await.expect("generate 1");

    let request2 = GenerateRequest::new("Second request")
        .with_sampling(SamplingParams::default());
    engine.generate(request2).await.expect("generate 2");

    let last = engine.last_request().await.expect("last request");
    // Verify we have a last request (the actual prompt format depends on PromptInput)
    assert!(last.request_id.to_string().len() > 0);
}

#[tokio::test]
async fn test_mock_engine_clone() {
    let engine = MockInferenceEngine::new();
    engine.set_response("Cloned response").await;

    let cloned = engine.clone();

    let request = GenerateRequest::new("Test")
        .with_sampling(SamplingParams::default());
    let response = cloned.generate(request).await.expect("generate");

    assert_eq!(response.choices[0].text, "Cloned response");
}

// ============================================================================
// MockError Tests
// ============================================================================

#[test]
fn test_mock_error_simulated() {
    let error = MockError::SimulatedError("Test error".to_string());
    let display = format!("{}", error);
    assert!(display.contains("Simulated error"));
    assert!(display.contains("Test error"));
}

#[test]
fn test_mock_error_model_not_loaded() {
    let error = MockError::ModelNotLoaded;
    let display = format!("{}", error);
    assert!(display.contains("Model not loaded"));
}

#[test]
fn test_mock_error_context_exceeded() {
    let error = MockError::ContextExceeded;
    let display = format!("{}", error);
    assert!(display.contains("Context length exceeded"));
}

#[test]
fn test_mock_error_clone() {
    let error = MockError::SimulatedError("Clone test".to_string());
    let cloned = error.clone();
    assert_eq!(format!("{}", error), format!("{}", cloned));
}

// ============================================================================
// MockVectorStore Tests
// ============================================================================

#[tokio::test]
async fn test_vector_store_new() {
    let store = MockVectorStore::new();
    assert_eq!(store.query_count().await, 0);
}

#[tokio::test]
async fn test_vector_store_default() {
    let store = MockVectorStore::default();
    assert_eq!(store.query_count().await, 0);
}

#[tokio::test]
async fn test_vector_store_add_and_query() {
    let store = MockVectorStore::new();

    store.add_document("doc1", "First document content", 0.95).await;
    store.add_document("doc2", "Second document content", 0.85).await;
    store.add_document("doc3", "Third document content", 0.75).await;

    let results = store.query("test query", 2).await;

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, "doc1"); // Highest score
    assert_eq!(results[1].id, "doc2");
    assert_eq!(store.query_count().await, 1);
}

#[tokio::test]
async fn test_vector_store_top_k() {
    let store = MockVectorStore::new();

    for i in 0..10 {
        store.add_document(&format!("doc{}", i), &format!("Content {}", i), i as f32 * 0.1).await;
    }

    let results = store.query("test", 5).await;
    assert_eq!(results.len(), 5);

    // Verify sorted by score descending
    for i in 0..4 {
        assert!(results[i].score >= results[i + 1].score);
    }
}

#[tokio::test]
async fn test_vector_store_query_count() {
    let store = MockVectorStore::new();
    store.add_document("doc1", "Content", 0.9).await;

    for _ in 0..10 {
        store.query("test", 1).await;
    }

    assert_eq!(store.query_count().await, 10);
}

#[tokio::test]
async fn test_vector_store_clone() {
    let store = MockVectorStore::new();
    store.add_document("doc1", "Content 1", 0.9).await;

    let cloned = store.clone();
    cloned.add_document("doc2", "Content 2", 0.8).await;

    // Both should see the same data due to Arc
    let results = store.query("test", 10).await;
    assert_eq!(results.len(), 2);
}

#[test]
fn test_mock_document_debug() {
    let doc = MockDocument {
        id: "test-doc".to_string(),
        content: "Test content".to_string(),
        score: 0.95,
    };

    let debug = format!("{:?}", doc);
    assert!(debug.contains("test-doc"));
    assert!(debug.contains("Test content"));
}

// ============================================================================
// Fixtures Tests
// ============================================================================

#[test]
fn test_sample_chat_request() {
    let request = sample_chat_request();

    assert_eq!(request.get("model").and_then(|v| v.as_str()), Some("test-model"));
    assert!(request.get("messages").is_some());
    assert!(request.get("temperature").is_some());
    assert!(request.get("max_tokens").is_some());
}

#[test]
fn test_sample_streaming_chat_request() {
    let request = sample_streaming_chat_request();

    assert_eq!(request.get("stream").and_then(|v| v.as_bool()), Some(true));
    assert!(request.get("messages").is_some());
}

#[test]
fn test_sample_embedding_request() {
    let request = sample_embedding_request();

    assert_eq!(request.get("model").and_then(|v| v.as_str()), Some("test-embedding-model"));
    assert!(request.get("input").is_some());
}

#[test]
fn test_sample_batch_embedding_request() {
    let request = sample_batch_embedding_request();

    let input = request.get("input").and_then(|v| v.as_array());
    assert!(input.is_some());
    assert_eq!(input.unwrap().len(), 3);
}

#[test]
fn test_sample_chat_response() {
    let response = sample_chat_response();

    assert!(response.get("id").is_some());
    assert!(response.get("choices").is_some());
    assert!(response.get("usage").is_some());
}

#[test]
fn test_sample_error_response() {
    let response = sample_error_response("rate_limit_exceeded", "Too many requests");

    assert_error_code(&response, "rate_limit_exceeded");
    let error = response.get("error").expect("error");
    assert_eq!(error.get("message").and_then(|v| v.as_str()), Some("Too many requests"));
}

#[test]
fn test_invalid_requests() {
    let missing_model = invalid_request_missing_model();
    assert!(missing_model.get("model").is_none());

    let empty_messages = invalid_request_empty_messages();
    let messages = empty_messages.get("messages").and_then(|v| v.as_array());
    assert!(messages.is_some());
    assert!(messages.unwrap().is_empty());

    let bad_temp = invalid_request_bad_temperature();
    let temp = bad_temp.get("temperature").and_then(|v| v.as_f64());
    assert!(temp.is_some());
    assert!(temp.unwrap() > 2.0); // Invalid temperature
}

#[test]
fn test_api_key_constants() {
    assert!(TEST_API_KEY_ADMIN.starts_with("sk-adm-"));
    assert!(TEST_API_KEY_INFERENCE.starts_with("sk-inf-"));
    assert!(TEST_API_KEY_INVALID.contains("invalid"));
}

#[test]
fn test_model_id_constants() {
    assert_eq!(TEST_MODEL_ID, "test-model");
    assert_eq!(TEST_EMBEDDING_MODEL_ID, "test-embedding-model");
}

// ============================================================================
// Assertions Tests
// ============================================================================

#[test]
fn test_assert_error_code_passes() {
    let response = json!({
        "error": {
            "code": "invalid_api_key",
            "message": "Invalid API key"
        }
    });
    assert_error_code(&response, "invalid_api_key");
}

#[test]
#[should_panic(expected = "Expected error code")]
fn test_assert_error_code_fails() {
    let response = json!({
        "error": {
            "code": "wrong_code",
            "message": "Error"
        }
    });
    assert_error_code(&response, "expected_code");
}

#[test]
fn test_assert_error_type_passes() {
    let response = json!({
        "error": {
            "code": "test",
            "type": "invalid_request_error"
        }
    });
    assert_error_type(&response, "invalid_request_error");
}

#[test]
fn test_assert_has_choices_passes() {
    let response = json!({
        "choices": [{"index": 0, "text": "Hello"}]
    });
    assert_has_choices(&response);
}

#[test]
#[should_panic(expected = "at least one choice")]
fn test_assert_has_choices_fails_empty() {
    let response = json!({
        "choices": []
    });
    assert_has_choices(&response);
}

#[test]
fn test_assert_has_usage_passes() {
    let response = json!({
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    });
    assert_has_usage(&response);
}

#[test]
fn test_assert_header_present() {
    let headers = vec![
        ("Content-Type".to_string(), "application/json".to_string()),
        ("X-Request-Id".to_string(), "12345".to_string()),
    ];

    assert_header_present(&headers, "Content-Type");
    assert_header_present(&headers, "content-type"); // Case insensitive
    assert_header_present(&headers, "X-Request-Id");
}

#[test]
#[should_panic(expected = "to be present")]
fn test_assert_header_present_fails() {
    let headers = vec![
        ("Content-Type".to_string(), "application/json".to_string()),
    ];
    assert_header_present(&headers, "X-Missing");
}

#[test]
fn test_assert_header_value() {
    let headers = vec![
        ("Content-Type".to_string(), "application/json".to_string()),
    ];
    assert_header_value(&headers, "Content-Type", "application/json");
}

#[test]
fn test_assert_response_time_passes() {
    assert_response_time_ms(50, 100);
    assert_response_time_ms(100, 100);
}

#[test]
#[should_panic(expected = "exceeded maximum")]
fn test_assert_response_time_fails() {
    assert_response_time_ms(150, 100);
}

#[test]
fn test_get_json_helpers() {
    let json = json!({
        "name": "test",
        "count": 42,
        "ratio": 3.14,
        "active": true
    });

    assert_eq!(get_json_string(&json, "name"), Some("test".to_string()));
    assert_eq!(get_json_i64(&json, "count"), Some(42));
    assert_eq!(get_json_f64(&json, "ratio"), Some(3.14));
    assert_eq!(get_json_bool(&json, "active"), Some(true));
    assert_eq!(get_json_string(&json, "missing"), None);
}

#[test]
fn test_assert_json_string_passes() {
    let json = json!({"model": "claude-opus-4"});
    assert_json_string(&json, "model", "claude-opus-4");
}

#[test]
fn test_assert_json_i64_passes() {
    let json = json!({"tokens": 256});
    assert_json_i64(&json, "tokens", 256);
}

// ============================================================================
// TestRequest Builder Tests
// ============================================================================

#[test]
fn test_request_get() {
    let request = TestRequest::get("/v1/models");

    assert_eq!(request.method(), "GET");
    assert_eq!(request.url(), "/v1/models");
    assert!(request.headers().is_empty());
    assert!(request.body().is_none());
}

#[test]
fn test_request_post() {
    let request = TestRequest::post("/v1/chat/completions");

    assert_eq!(request.method(), "POST");
    assert_eq!(request.url(), "/v1/chat/completions");
}

#[test]
fn test_request_with_header() {
    let request = TestRequest::get("/test")
        .header("X-Custom", "value");

    assert_eq!(request.headers().len(), 1);
    assert!(request.headers().iter().any(|(k, v)| k == "X-Custom" && v == "value"));
}

#[test]
fn test_request_with_bearer_token() {
    let request = TestRequest::post("/test")
        .bearer_token("sk-test-token");

    let auth_header = request.headers()
        .iter()
        .find(|(k, _)| k == "Authorization")
        .map(|(_, v)| v.as_str());

    assert_eq!(auth_header, Some("Bearer sk-test-token"));
}

#[test]
fn test_request_with_json_body() {
    let body = json!({"model": "test", "messages": []});
    let request = TestRequest::post("/test")
        .json(body.clone());

    assert!(request.body().is_some());
    assert_eq!(request.body().unwrap(), &body);
}

#[test]
fn test_request_builder_chain() {
    let request = TestRequest::post("/v1/chat/completions")
        .bearer_token("sk-test")
        .header("Content-Type", "application/json")
        .header("X-Request-Id", "12345")
        .json(sample_chat_request());

    assert_eq!(request.method(), "POST");
    assert_eq!(request.headers().len(), 3);
    assert!(request.body().is_some());
}

// ============================================================================
// TestResponse Tests
// ============================================================================

#[test]
fn test_response_is_success() {
    for status in [200, 201, 204, 299] {
        let response = TestResponse {
            status,
            headers: Vec::new(),
            json: None,
            body: String::new(),
        };
        assert!(response.is_success(), "Status {} should be success", status);
    }

    for status in [400, 401, 403, 404, 500] {
        let response = TestResponse {
            status,
            headers: Vec::new(),
            json: None,
            body: String::new(),
        };
        assert!(!response.is_success(), "Status {} should not be success", status);
    }
}

#[test]
fn test_response_has_header() {
    let response = TestResponse {
        status: 200,
        headers: vec![
            ("Content-Type".to_string(), "application/json".to_string()),
            ("X-Request-Id".to_string(), "abc123".to_string()),
        ],
        json: None,
        body: String::new(),
    };

    assert!(response.has_header("Content-Type"));
    assert!(response.has_header("content-type")); // Case insensitive
    assert!(response.has_header("X-Request-Id"));
    assert!(!response.has_header("X-Missing"));
}

#[test]
fn test_response_header() {
    let response = TestResponse {
        status: 200,
        headers: vec![
            ("Content-Type".to_string(), "application/json".to_string()),
        ],
        json: None,
        body: String::new(),
    };

    assert_eq!(response.header("Content-Type"), Some("application/json"));
    assert_eq!(response.header("X-Missing"), None);
}

#[test]
fn test_response_json() {
    let json_body = json!({"status": "ok"});
    let response = TestResponse {
        status: 200,
        headers: Vec::new(),
        json: Some(json_body.clone()),
        body: json_body.to_string(),
    };

    assert_eq!(response.json(), &json_body);
}

#[test]
fn test_response_debug() {
    let response = TestResponse {
        status: 200,
        headers: vec![("X-Test".to_string(), "value".to_string())],
        json: Some(json!({"test": true})),
        body: "{\"test\": true}".to_string(),
    };

    let debug = format!("{:?}", response);
    assert!(debug.contains("200"));
    assert!(debug.contains("test"));
}

// ============================================================================
// TestServer Tests
// ============================================================================

async fn health_handler() -> Json<serde_json::Value> {
    Json(json!({"status": "healthy"}))
}

async fn echo_handler(Json(body): Json<serde_json::Value>) -> Json<serde_json::Value> {
    Json(json!({"echo": body}))
}

#[tokio::test]
async fn test_server_start() {
    let router = Router::new().route("/health", get(health_handler));
    let server = TestServer::start(router).await;

    assert!(!server.url().is_empty());
    assert!(server.url().starts_with("http://"));

    server.shutdown();
}

#[tokio::test]
async fn test_server_url_for() {
    let router = Router::new().route("/health", get(health_handler));
    let server = TestServer::start(router).await;

    let url = server.url_for("/api/v1/test");
    assert!(url.contains("/api/v1/test"));
    assert!(url.starts_with(&server.url()));

    // Test without leading slash
    let url2 = server.url_for("health");
    assert!(url2.contains("/health"));

    server.shutdown();
}

#[tokio::test]
async fn test_server_addr() {
    let router = Router::new().route("/health", get(health_handler));
    let server = TestServer::start(router).await;

    let addr = server.addr();
    assert_eq!(addr.ip().to_string(), "127.0.0.1");
    assert!(addr.port() > 0);

    server.shutdown();
}

// ============================================================================
// Integration Workflow Tests
// ============================================================================

#[tokio::test]
async fn test_mock_engine_with_fixtures() {
    let engine = MockInferenceEngine::new();

    // Use fixture to get a sample request
    let fixture = sample_chat_request();
    let messages: Vec<infernum_core::Message> = fixture
        .get("messages")
        .and_then(|m| m.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| {
                    let role = m.get("role")?.as_str()?;
                    let content = m.get("content")?.as_str()?;
                    Some(infernum_core::Message {
                        role: match role {
                            "system" => infernum_core::Role::System,
                            "user" => infernum_core::Role::User,
                            "assistant" => infernum_core::Role::Assistant,
                            _ => infernum_core::Role::User,
                        },
                        content: content.to_string(),
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let request = GenerateRequest::chat(messages)
        .with_sampling(SamplingParams::default().with_max_tokens(256));

    let response = engine.generate(request).await.expect("generate");

    // Use assertions to validate
    let response_json = serde_json::to_value(&response).expect("serialize");
    assert_has_choices(&response_json);
    assert_has_usage(&response_json);
}

#[tokio::test]
async fn test_vector_store_with_assertions() {
    let store = MockVectorStore::new();

    // Add documents
    store.add_document("relevant-1", "Rust programming guide", 0.95).await;
    store.add_document("relevant-2", "Rust async patterns", 0.88).await;
    store.add_document("less-relevant", "Python basics", 0.45).await;

    // Query
    let results = store.query("Rust programming", 2).await;

    // Use assertions
    assert_eq!(results.len(), 2);
    assert!(results[0].score >= results[1].score, "Results should be sorted by score");
    assert!(results[0].content.contains("Rust"), "Top result should be about Rust");
}

#[tokio::test]
async fn test_complete_testing_workflow() {
    // 1. Create mock engine with custom response
    let engine = MockInferenceEngine::new();
    engine.set_response("This is the AI response.").await;

    // 2. Create test request
    let request = GenerateRequest::new("What is 2+2?")
        .with_sampling(SamplingParams::default().with_max_tokens(100));

    // 3. Generate response
    let response = engine.generate(request).await.expect("generate");

    // 4. Verify response structure
    assert!(!response.choices.is_empty());
    assert_eq!(response.choices[0].text, "This is the AI response.");
    assert_eq!(response.usage.total_tokens, 30);

    // 5. Verify call was tracked
    assert_eq!(engine.call_count().await, 1);

    // 6. Use JSON assertions on serialized response
    let json = serde_json::to_value(&response).expect("serialize");
    assert_has_choices(&json);
    assert_has_usage(&json);
}

#[tokio::test]
async fn test_error_handling_workflow() {
    // Create failing engine
    let engine = MockInferenceEngine::failing("Service unavailable");

    let request = GenerateRequest::new("Test")
        .with_sampling(SamplingParams::default());

    let result = engine.generate(request).await;

    // Verify error
    assert!(result.is_err());
    let error = result.unwrap_err();
    assert!(format!("{}", error).contains("Service unavailable"));

    // Create sample error response
    let error_response = sample_error_response("service_unavailable", "Service unavailable");
    assert_error_code(&error_response, "service_unavailable");
}

#[tokio::test]
async fn test_concurrent_mock_access() {
    let engine = std::sync::Arc::new(MockInferenceEngine::new());

    let mut handles = Vec::new();
    for i in 0..10 {
        let engine_clone = engine.clone();
        let handle = tokio::spawn(async move {
            let request = GenerateRequest::new(format!("Request {}", i))
                .with_sampling(SamplingParams::default());
            engine_clone.generate(request).await
        });
        handles.push(handle);
    }

    for handle in handles {
        let result = handle.await.expect("join");
        assert!(result.is_ok());
    }

    assert_eq!(engine.call_count().await, 10);
}

#[tokio::test]
async fn test_vector_store_rag_workflow() {
    // Set up knowledge base
    let store = MockVectorStore::new();
    store.add_document("doc1", "The capital of France is Paris.", 0.95).await;
    store.add_document("doc2", "France is a country in Europe.", 0.85).await;
    store.add_document("doc3", "The Eiffel Tower is in Paris.", 0.80).await;
    store.add_document("doc4", "Tokyo is the capital of Japan.", 0.20).await;

    // Query for relevant context
    let query = "What is the capital of France?";
    let context = store.query(query, 3).await;

    // Top results should be about France/Paris
    assert!(context[0].score > 0.8);
    assert!(context[0].content.contains("France") || context[0].content.contains("Paris"));

    // Build prompt with context
    let context_text: String = context.iter()
        .map(|d| d.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    assert!(context_text.contains("Paris"));
    assert!(context_text.contains("France"));
}
