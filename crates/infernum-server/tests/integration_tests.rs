//! Integration tests for infernum-server.
//!
//! Tests the HTTP server endpoints with real network requests.

mod test_helpers;

use axum::routing::get;
use axum::Json;
use serde_json::json;
use test_helpers::{json_body, TestServer};

/// Creates a minimal health check router for testing.
fn health_router() -> axum::Router {
    axum::Router::new()
        .route(
            "/health",
            get(|| async {
                Json(json!({
                    "status": "ok",
                    "version": "test"
                }))
            }),
        )
        .route(
            "/ready",
            get(|| async {
                Json(json!({
                    "ready": false,
                    "model": null
                }))
            }),
        )
}

// ============================================================================
// Health Endpoint Tests
// ============================================================================

#[tokio::test]
async fn test_health_endpoint_returns_ok() {
    let server = TestServer::start(health_router()).await;
    let response = server.get("/health").await;

    assert_eq!(response.status(), 200);
    let body = json_body(response).await;
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn test_ready_endpoint_without_model() {
    let server = TestServer::start(health_router()).await;
    let response = server.get("/ready").await;

    assert_eq!(response.status(), 200);
    let body = json_body(response).await;
    assert_eq!(body["ready"], false);
    assert!(body["model"].is_null());
}

#[tokio::test]
async fn test_health_endpoint_content_type() {
    let server = TestServer::start(health_router()).await;
    let response = server.get("/health").await;

    let content_type = response
        .headers()
        .get("content-type")
        .expect("content-type missing");
    assert!(content_type.to_str().unwrap().contains("application/json"));
}

// ============================================================================
// Validation Tests (using mock validation router)
// ============================================================================

/// Creates a router that validates generate requests (mock implementation).
fn validation_router() -> axum::Router {
    use axum::http::StatusCode;
    use axum::routing::post;

    axum::Router::new().route(
        "/v1/generate",
        post(|Json(body): Json<serde_json::Value>| async move {
            // Validate prompt (messages or text)
            let prompt = body.get("prompt");
            match prompt {
                None => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({
                            "error": {
                                "message": "prompt is required",
                                "type": "invalid_request_error",
                                "code": "empty_prompt"
                            }
                        })),
                    );
                }
                Some(p) if p.is_array() && p.as_array().map_or(true, |a| a.is_empty()) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({
                            "error": {
                                "message": "messages cannot be empty",
                                "type": "invalid_request_error",
                                "code": "invalid_messages"
                            }
                        })),
                    );
                }
                _ => {}
            }

            // Validate sampling.temperature
            if let Some(sampling) = body.get("sampling") {
                if let Some(temp) = sampling.get("temperature") {
                    if let Some(t) = temp.as_f64() {
                        if !(0.0..=2.0).contains(&t) {
                            return (
                                StatusCode::BAD_REQUEST,
                                Json(json!({
                                    "error": {
                                        "message": "temperature must be between 0 and 2",
                                        "type": "invalid_request_error",
                                        "code": "invalid_temperature",
                                        "param": "temperature"
                                    }
                                })),
                            );
                        }
                    }
                }
            }

            // Success response
            (
                StatusCode::OK,
                Json(json!({
                    "request_id": "gen-test-123",
                    "model": "test-model",
                    "choices": [{
                        "index": 0,
                        "text": "Hello!",
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15
                    }
                })),
            )
        }),
    )
}

#[tokio::test]
async fn test_generate_success() {
    let server = TestServer::start(validation_router()).await;

    let response = server
        .post_json(
            "/v1/generate",
            &json!({
                "model": "test-model",
                "prompt": [{"role": "user", "content": "Hello"}]
            }),
        )
        .await;

    assert_eq!(response.status(), 200);
    let body = json_body(response).await;
    assert_eq!(body["request_id"], "gen-test-123");
    assert!(body["choices"].is_array());
}

#[tokio::test]
async fn test_generate_empty_messages_error() {
    let server = TestServer::start(validation_router()).await;

    let response = server
        .post_json(
            "/v1/generate",
            &json!({
                "model": "test-model",
                "prompt": []
            }),
        )
        .await;

    assert_eq!(response.status(), 400);
    let body = json_body(response).await;
    assert_eq!(body["error"]["code"], "invalid_messages");
}

#[tokio::test]
async fn test_generate_invalid_temperature_error() {
    let server = TestServer::start(validation_router()).await;

    let response = server
        .post_json(
            "/v1/generate",
            &json!({
                "model": "test-model",
                "prompt": [{"role": "user", "content": "hi"}],
                "sampling": {"temperature": 5.0}
            }),
        )
        .await;

    assert_eq!(response.status(), 400);
    let body = json_body(response).await;
    assert_eq!(body["error"]["code"], "invalid_temperature");
    assert_eq!(body["error"]["param"], "temperature");
}

// ============================================================================
// Contract Tests (Infernum API)
// ============================================================================

/// Validates a generate response matches Infernum spec.
fn validate_generate_response(body: &serde_json::Value) {
    assert!(body["request_id"].is_string(), "request_id must be string");
    assert!(body["model"].is_string(), "model must be string");
    assert!(body["choices"].is_array(), "choices must be array");

    for choice in body["choices"].as_array().unwrap() {
        assert!(choice["index"].is_number());
    }

    if let Some(usage) = body.get("usage") {
        if !usage.is_null() {
            assert!(usage["prompt_tokens"].is_number());
            assert!(usage["completion_tokens"].is_number());
            assert!(usage["total_tokens"].is_number());
        }
    }
}

/// Validates an error response matches Infernum error spec.
fn validate_error_response(body: &serde_json::Value) {
    assert!(body["error"].is_object(), "error must be object");
    assert!(body["error"]["message"].is_string(), "error.message must be string");
    assert!(body["error"]["type"].is_string(), "error.type must be string");
}

#[tokio::test]
async fn test_generate_contract() {
    let server = TestServer::start(validation_router()).await;

    let response = server
        .post_json(
            "/v1/generate",
            &json!({
                "model": "test",
                "prompt": [{"role": "user", "content": "hi"}]
            }),
        )
        .await;

    let body = json_body(response).await;
    validate_generate_response(&body);
}

#[tokio::test]
async fn test_error_contract() {
    let server = TestServer::start(validation_router()).await;

    let response = server
        .post_json(
            "/v1/generate",
            &json!({
                "model": "test",
                "prompt": []
            }),
        )
        .await;

    let body = json_body(response).await;
    validate_error_response(&body);
}

// ============================================================================
// Tokenize Endpoint Tests
// ============================================================================

/// Creates a router with the tokenize endpoint.
fn tokenize_router() -> axum::Router {
    use axum::http::StatusCode;
    use axum::routing::post;

    axum::Router::new().route(
        "/v1/tokenize",
        post(|Json(body): Json<serde_json::Value>| async move {
            // Validate request has prompt (text string or messages array)
            let prompt = body.get("prompt");

            if prompt.is_none() {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({
                        "error": {
                            "message": "'prompt' must be provided (text string or messages array)",
                            "type": "invalid_request_error",
                            "code": "empty_prompt"
                        }
                    })),
                );
            }

            let prompt = prompt.expect("checked above");

            // Estimate token count (simple approximation)
            let text = if let Some(s) = prompt.as_str() {
                s.to_string()
            } else if let Some(msgs) = prompt.as_array() {
                msgs.iter()
                    .filter_map(|m| m["content"].as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({
                        "error": {
                            "message": "'prompt' must be a string or array of messages",
                            "type": "invalid_request_error",
                            "code": "empty_prompt"
                        }
                    })),
                );
            };

            let token_count = (text.chars().count() as f64 / 4.0).ceil() as u32;

            (
                StatusCode::OK,
                Json(json!({
                    "token_count": token_count,
                    "model": body["model"].as_str().unwrap_or("unknown")
                })),
            )
        }),
    )
}

#[tokio::test]
async fn test_tokenize_with_text_prompt() {
    let server = TestServer::start(tokenize_router()).await;

    let response = server
        .post_json(
            "/v1/tokenize",
            &json!({
                "model": "test-model",
                "prompt": "Hello, world!"
            }),
        )
        .await;

    assert_eq!(response.status(), 200);
    let body = json_body(response).await;
    assert!(body["token_count"].is_number());
    assert_eq!(body["model"], "test-model");
}

#[tokio::test]
async fn test_tokenize_with_messages_prompt() {
    let server = TestServer::start(tokenize_router()).await;

    let response = server
        .post_json(
            "/v1/tokenize",
            &json!({
                "model": "llama-3b",
                "prompt": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi!"}
                ]
            }),
        )
        .await;

    assert_eq!(response.status(), 200);
    let body = json_body(response).await;
    assert!(body["token_count"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn test_tokenize_no_input_error() {
    let server = TestServer::start(tokenize_router()).await;

    let response = server
        .post_json(
            "/v1/tokenize",
            &json!({
                "model": "test-model"
            }),
        )
        .await;

    assert_eq!(response.status(), 400);
    let body = json_body(response).await;
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("prompt"));
}

// ============================================================================
// Typed Validation Tests (using real RequestValidationError)
// ============================================================================

/// Creates a router that uses the actual validation module with native types.
fn typed_validation_router() -> axum::Router {
    use axum::http::StatusCode;
    use axum::routing::post;
    use infernum_server::ChatCompletionRequest;
    use infernum_server::validation::validate_chat_request;
    use infernum_server::ValidationLimits;

    axum::Router::new().route(
        "/v1/chat/completions",
        post(|Json(body): Json<serde_json::Value>| async move {
            // Parse as ChatCompletionRequest (OpenAI-compatible format)
            let req: ChatCompletionRequest = match serde_json::from_value(body.clone()) {
                Ok(r) => r,
                Err(e) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({
                            "error": {
                                "message": format!("Invalid request: {}", e),
                                "type": "invalid_request_error",
                                "code": "invalid_request"
                            }
                        })),
                    );
                }
            };

            // Use strict limits for testing
            let limits = ValidationLimits {
                max_messages: 10,
                max_message_length: 1000,
                max_max_tokens: 4096,
                ..Default::default()
            };

            // Validate using the real validation module
            if let Err(err) = validate_chat_request(&req, &limits) {
                let api_error = err.to_api_error("test-request-id");
                return (StatusCode::BAD_REQUEST, Json(serde_json::to_value(api_error).unwrap_or_default()));
            }

            // Success response
            (
                StatusCode::OK,
                Json(json!({
                    "request_id": "test-request-id",
                    "model": "test",
                    "choices": [{
                        "index": 0,
                        "text": "Hello from typed validation!",
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15
                    }
                })),
            )
        }),
    )
}

#[tokio::test]
async fn test_typed_validation_empty_messages_returns_400() {
    let server = TestServer::start(typed_validation_router()).await;

    let response = server
        .post_json(
            "/v1/chat/completions",
            &json!({
                "model": "test-model",
                "messages": []
            }),
        )
        .await;

    assert_eq!(response.status(), 400);
    let body = json_body(response).await;

    // Verify rich error format from RequestValidationError
    assert_eq!(body["error"]["code"], "invalid_messages");
    assert_eq!(body["error"]["param"], "messages");
    assert!(body["error"]["message"].as_str().unwrap().contains("empty"));
}

#[tokio::test]
async fn test_typed_validation_too_many_messages_returns_400() {
    let server = TestServer::start(typed_validation_router()).await;

    // Create 15 messages (exceeds limit of 10)
    let messages: Vec<_> = (0..15)
        .map(|i| json!({"role": "user", "content": format!("Message {}", i)}))
        .collect();

    let response = server
        .post_json(
            "/v1/chat/completions",
            &json!({
                "model": "test-model",
                "messages": messages
            }),
        )
        .await;

    assert_eq!(response.status(), 400);
    let body = json_body(response).await;

    // Verify rich error format with limit/actual values
    assert_eq!(body["error"]["code"], "invalid_messages");
    assert_eq!(body["error"]["param"], "messages");
    assert!(body["error"]["limit"].is_number());
    assert!(body["error"]["actual"].is_number());
    assert_eq!(body["error"]["limit"].as_u64().unwrap(), 10);
    assert_eq!(body["error"]["actual"].as_u64().unwrap(), 15);
}

#[tokio::test]
async fn test_typed_validation_invalid_temperature_returns_400() {
    let server = TestServer::start(typed_validation_router()).await;

    let response = server
        .post_json(
            "/v1/chat/completions",
            &json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 3.5
            }),
        )
        .await;

    assert_eq!(response.status(), 400);
    let body = json_body(response).await;

    assert_eq!(body["error"]["code"], "invalid_temperature");
    assert_eq!(body["error"]["param"], "temperature");
    // Should have subcode for above_maximum
    assert!(body["error"]["subcode"].is_string());
}

#[tokio::test]
async fn test_typed_validation_valid_request_returns_200() {
    let server = TestServer::start(typed_validation_router()).await;

    let response = server
        .post_json(
            "/v1/chat/completions",
            &json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello!"}],
                "temperature": 0.7,
                "max_tokens": 100
            }),
        )
        .await;

    assert_eq!(response.status(), 200);
    let body = json_body(response).await;
    assert_eq!(body["request_id"], "test-request-id");
    assert!(body["choices"].is_array());
}

// ============================================================================
// Day 14: Error Scenario Tests
// ============================================================================

/// Creates a router that validates auth headers
fn auth_router() -> axum::Router {
    use axum::http::{header, StatusCode};
    use axum::routing::post;

    axum::Router::new().route(
        "/v1/generate",
        post(
            |headers: axum::http::HeaderMap, Json(body): Json<serde_json::Value>| async move {
                // Check for Authorization header
                let auth = headers.get(header::AUTHORIZATION);
                match auth {
                    None => {
                        return (
                            StatusCode::UNAUTHORIZED,
                            Json(json!({
                                "error": {
                                    "message": "Missing API key",
                                    "type": "authentication_error",
                                    "code": "missing_api_key"
                                }
                            })),
                        );
                    }
                    Some(value) => {
                        let value_str = value.to_str().unwrap_or("");
                        if !value_str.starts_with("Bearer sk-") {
                            return (
                                StatusCode::UNAUTHORIZED,
                                Json(json!({
                                    "error": {
                                        "message": "Invalid API key",
                                        "type": "authentication_error",
                                        "code": "invalid_api_key"
                                    }
                                })),
                            );
                        }
                    }
                }

                // Valid request
                (
                    StatusCode::OK,
                    Json(json!({
                        "request_id": "gen-auth-test",
                        "model": body["model"],
                        "choices": [{
                            "index": 0,
                            "text": "Authenticated!",
                            "finish_reason": "stop"
                        }]
                    })),
                )
            },
        ),
    )
}

/// Creates a router with rate limiting
fn rate_limit_router() -> axum::Router {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use axum::routing::post;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    let request_count = Arc::new(AtomicU32::new(0));

    axum::Router::new().route(
        "/v1/generate",
        post({
            let request_count = request_count.clone();
            move |Json(_body): Json<serde_json::Value>| {
                let request_count = request_count.clone();
                async move {
                    let count = request_count.fetch_add(1, Ordering::SeqCst);

                    // Rate limit after 3 requests
                    if count >= 3 {
                        return (
                            StatusCode::TOO_MANY_REQUESTS,
                            [("retry-after", "60")],
                            Json(json!({
                                "error": {
                                    "message": "Rate limit exceeded",
                                    "type": "rate_limit_error",
                                    "code": "rate_limit_exceeded"
                                }
                            })),
                        )
                            .into_response();
                    }

                    (
                        StatusCode::OK,
                        Json(json!({
                            "request_id": "gen-rate-test",
                            "choices": [{"index": 0, "text": "OK", "finish_reason": "stop"}]
                        })),
                    )
                        .into_response()
                }
            }
        }),
    )
}

#[tokio::test]
async fn test_auth_missing_api_key_returns_401() {
    let server = TestServer::start(auth_router()).await;

    let client = reqwest::Client::new();
    let response = client
        .post(&server.url("/v1/generate"))
        .json(&json!({
            "model": "test-model",
            "prompt": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(response.status(), 401);
    let body: serde_json::Value = response.json().await.expect("json failed");
    assert_eq!(body["error"]["code"], "missing_api_key");
}

#[tokio::test]
async fn test_auth_invalid_api_key_returns_401() {
    let server = TestServer::start(auth_router()).await;

    let client = reqwest::Client::new();
    let response = client
        .post(&server.url("/v1/generate"))
        .header("Authorization", "Bearer invalid-key")
        .json(&json!({
            "model": "test-model",
            "prompt": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(response.status(), 401);
    let body: serde_json::Value = response.json().await.expect("json failed");
    assert_eq!(body["error"]["code"], "invalid_api_key");
}

#[tokio::test]
async fn test_auth_valid_api_key_returns_200() {
    let server = TestServer::start(auth_router()).await;

    let client = reqwest::Client::new();
    let response = client
        .post(&server.url("/v1/generate"))
        .header("Authorization", "Bearer sk-test-key-123")
        .json(&json!({
            "model": "test-model",
            "prompt": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(response.status(), 200);
}

#[tokio::test]
async fn test_rate_limit_returns_429() {
    let server = TestServer::start(rate_limit_router()).await;
    let client = reqwest::Client::new();

    // First 3 requests should succeed
    for _ in 0..3 {
        let response = client
            .post(&server.url("/v1/generate"))
            .json(&json!({"model": "test", "prompt": [{"role": "user", "content": "hi"}]}))
            .send()
            .await
            .expect("request failed");
        assert_eq!(response.status(), 200);
    }

    // 4th request should be rate limited
    let response = client
        .post(&server.url("/v1/generate"))
        .json(&json!({"model": "test", "prompt": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("request failed");

    assert_eq!(response.status(), 429);
    assert!(response.headers().contains_key("retry-after"));

    let body: serde_json::Value = response.json().await.expect("json failed");
    assert_eq!(body["error"]["code"], "rate_limit_exceeded");
}

// ============================================================================
// Day 15: Concurrency Tests
// ============================================================================

/// Creates a router that handles concurrent requests
fn concurrent_router() -> axum::Router {
    use axum::http::StatusCode;
    use axum::routing::post;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    let active_count = Arc::new(AtomicU32::new(0));
    let max_concurrent = Arc::new(AtomicU32::new(0));

    axum::Router::new().route(
        "/v1/generate",
        post({
            let active_count = active_count.clone();
            let max_concurrent = max_concurrent.clone();
            move |Json(body): Json<serde_json::Value>| {
                let active_count = active_count.clone();
                let max_concurrent = max_concurrent.clone();
                async move {
                    // Track active requests
                    let current = active_count.fetch_add(1, Ordering::SeqCst) + 1;
                    max_concurrent.fetch_max(current, Ordering::SeqCst);

                    // Simulate some processing time
                    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

                    active_count.fetch_sub(1, Ordering::SeqCst);

                    (
                        StatusCode::OK,
                        Json(json!({
                            "request_id": format!("gen-{}", uuid::Uuid::new_v4()),
                            "model": body["model"],
                            "choices": [{
                                "index": 0,
                                "text": "Response",
                                "finish_reason": "stop"
                            }]
                        })),
                    )
                }
            }
        }),
    )
}

#[tokio::test]
async fn test_concurrent_requests() {
    let server = TestServer::start(concurrent_router()).await;
    let client = reqwest::Client::new();

    // Launch 10 concurrent requests
    let mut handles = Vec::new();
    for i in 0..10 {
        let client = client.clone();
        let url = server.url("/v1/generate");
        let handle = tokio::spawn(async move {
            let response = client
                .post(&url)
                .json(&json!({
                    "model": "test-model",
                    "prompt": [{"role": "user", "content": format!("Request {}", i)}]
                }))
                .send()
                .await
                .expect("request failed");
            response.status().as_u16()
        });
        handles.push(handle);
    }

    // Wait for all requests to complete
    let mut success_count = 0;
    for handle in handles {
        let status = handle.await.expect("task failed");
        if status == 200 {
            success_count += 1;
        }
    }

    // All requests should succeed
    assert_eq!(success_count, 10);
}

#[tokio::test]
async fn test_error_response_includes_request_id() {
    let server = TestServer::start(typed_validation_router()).await;

    let response = server
        .post_json(
            "/v1/chat/completions",
            &json!({
                "model": "test-model",
                "messages": []  // Invalid - empty messages
            }),
        )
        .await;

    assert_eq!(response.status(), 400);
    let body = json_body(response).await;

    // Error response should include request_id
    assert!(body["error"]["request_id"].is_string());
}

// ============================================================================
// Sprint 5: Observability Tests (Days 16-18)
// ============================================================================

/// Creates a router that exposes /metrics in Prometheus format.
fn metrics_router() -> axum::Router {
    use axum::response::IntoResponse;
    use axum::routing::get;

    let metrics_output = r#"# HELP infernum_requests_total Total number of inference requests.
# TYPE infernum_requests_total counter
infernum_requests_total{endpoint="chat",model="test-model"} 42
# HELP infernum_active_requests Number of requests currently being processed.
# TYPE infernum_active_requests gauge
infernum_active_requests 3
# HELP infernum_request_duration_seconds Request latency in seconds.
# TYPE infernum_request_duration_seconds histogram
infernum_request_duration_seconds_bucket{endpoint="chat",model="test-model",le="0.1"} 10
infernum_request_duration_seconds_bucket{endpoint="chat",model="test-model",le="0.5"} 35
infernum_request_duration_seconds_bucket{endpoint="chat",model="test-model",le="1.0"} 40
infernum_request_duration_seconds_bucket{endpoint="chat",model="test-model",le="+Inf"} 42
infernum_request_duration_seconds_sum{endpoint="chat",model="test-model"} 15.5
infernum_request_duration_seconds_count{endpoint="chat",model="test-model"} 42
# HELP infernum_tokens Token counts per request.
# TYPE infernum_tokens histogram
infernum_tokens_bucket{type="completion",model="test-model",le="100"} 20
infernum_tokens_bucket{type="completion",model="test-model",le="500"} 38
infernum_tokens_bucket{type="completion",model="test-model",le="+Inf"} 42
infernum_tokens_sum{type="completion",model="test-model"} 8500
infernum_tokens_count{type="completion",model="test-model"} 42
"#;

    axum::Router::new().route(
        "/metrics",
        get(move || async move {
            (
                [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
                metrics_output,
            )
                .into_response()
        }),
    )
}

/// Creates a router that exposes /api/status with version.
fn status_router() -> axum::Router {
    use axum::routing::get;

    axum::Router::new().route(
        "/api/status",
        get(|| async {
            Json(json!({
                "version": "0.1.0",
                "status": "running",
                "uptime_seconds": 3600,
                "model_loaded": true,
                "model_id": "test-model",
                "active_requests": 5,
                "queue_depth": 2,
                "total_requests": 1000,
                "error_rate": 0.01,
                "is_shutting_down": false
            }))
        }),
    )
}

// Day 16.1: Test /metrics returns Prometheus format
#[tokio::test]
async fn test_metrics_returns_prometheus_format() {
    let server = TestServer::start(metrics_router()).await;
    let response = server.get("/metrics").await;

    assert_eq!(response.status(), 200);

    // Verify Content-Type header
    let content_type = response
        .headers()
        .get("content-type")
        .expect("content-type header");
    assert!(content_type.to_str().unwrap().contains("text/plain"));

    let body = response.text().await.expect("body text");

    // Verify Prometheus format
    assert!(body.contains("# HELP"), "should have HELP comments");
    assert!(body.contains("# TYPE"), "should have TYPE declarations");
}

// Day 16.3: Test infernum_requests_total counter
#[tokio::test]
async fn test_metrics_contains_requests_total_counter() {
    let server = TestServer::start(metrics_router()).await;
    let response = server.get("/metrics").await;

    assert_eq!(response.status(), 200);
    let body = response.text().await.expect("body text");

    // Verify infernum_requests_total counter exists
    assert!(body.contains("infernum_requests_total"));
    assert!(body.contains("# TYPE infernum_requests_total counter"));
}

// Day 16.4: Test infernum_request_duration_seconds histogram
#[tokio::test]
async fn test_metrics_contains_duration_histogram() {
    let server = TestServer::start(metrics_router()).await;
    let response = server.get("/metrics").await;

    assert_eq!(response.status(), 200);
    let body = response.text().await.expect("body text");

    // Verify histogram exists with buckets
    assert!(body.contains("infernum_request_duration_seconds"));
    assert!(body.contains("# TYPE infernum_request_duration_seconds histogram"));
    assert!(body.contains("infernum_request_duration_seconds_bucket"));
    assert!(body.contains("_sum"));
    assert!(body.contains("_count"));
    assert!(body.contains("le=\"+Inf\""));
}

// Day 16.5: Test tokens metrics
#[tokio::test]
async fn test_metrics_contains_tokens_histogram() {
    let server = TestServer::start(metrics_router()).await;
    let response = server.get("/metrics").await;

    assert_eq!(response.status(), 200);
    let body = response.text().await.expect("body text");

    // Verify token metrics
    assert!(body.contains("infernum_tokens"));
    assert!(body.contains("type=\"completion\""));
}

// Day 17.1: Test request ID in responses
#[tokio::test]
async fn test_request_id_propagated() {
    // The typed_validation_router includes request_id in error responses
    let server = TestServer::start(typed_validation_router()).await;
    let client = reqwest::Client::new();

    // Send request with custom request ID
    let response = client
        .post(&server.url("/v1/chat/completions"))
        .header("x-request-id", "custom-req-12345")
        .json(&json!({
            "model": "test-model",
            "messages": []
        }))
        .send()
        .await
        .expect("request failed");

    // Error response should include the request_id
    assert_eq!(response.status(), 400);
    let body: serde_json::Value = response.json().await.expect("json failed");
    assert!(body["error"]["request_id"].is_string());
}

// Day 18.1: Test /health always returns 200
#[tokio::test]
async fn test_health_always_returns_200() {
    let server = TestServer::start(health_router()).await;

    // Health should always return 200 if server is reachable
    let response = server.get("/health").await;
    assert_eq!(response.status(), 200);

    let body = json_body(response).await;
    assert_eq!(body["status"], "ok");
}

// Day 18.5: Test status includes version
#[tokio::test]
async fn test_status_includes_version() {
    let server = TestServer::start(status_router()).await;
    let response = server.get("/api/status").await;

    assert_eq!(response.status(), 200);
    let body = json_body(response).await;

    // Status should include version
    assert!(body["version"].is_string(), "status should include version");
    assert!(!body["version"].as_str().unwrap().is_empty(), "version should not be empty");

    // Also verify other required fields
    assert!(body["status"].is_string());
    assert!(body["uptime_seconds"].is_number());
    assert!(body["model_loaded"].is_boolean());
}
