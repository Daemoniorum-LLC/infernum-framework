//! Test fixtures and sample data
//!
//! Provides pre-built test data for common testing scenarios.

use serde_json::json;

/// Sample chat completion request body
pub fn sample_chat_request() -> serde_json::Value {
    json!({
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "temperature": 0.7,
        "max_tokens": 256
    })
}

/// Sample chat completion request with streaming
pub fn sample_streaming_chat_request() -> serde_json::Value {
    json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Tell me a joke"}
        ],
        "stream": true,
        "max_tokens": 100
    })
}

/// Sample embedding request
pub fn sample_embedding_request() -> serde_json::Value {
    json!({
        "model": "test-embedding-model",
        "input": "Hello, world!"
    })
}

/// Sample embedding request with multiple inputs
pub fn sample_batch_embedding_request() -> serde_json::Value {
    json!({
        "model": "test-embedding-model",
        "input": [
            "First text to embed",
            "Second text to embed",
            "Third text to embed"
        ]
    })
}

/// Sample chat completion response
pub fn sample_chat_response() -> serde_json::Value {
    json!({
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! I'm doing well, thank you for asking. How can I help you today?"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 25,
            "completion_tokens": 20,
            "total_tokens": 45
        }
    })
}

/// Sample error response
pub fn sample_error_response(code: &str, message: &str) -> serde_json::Value {
    json!({
        "error": {
            "code": code,
            "message": message,
            "type": "invalid_request_error"
        }
    })
}

/// Invalid request - missing model
pub fn invalid_request_missing_model() -> serde_json::Value {
    json!({
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })
}

/// Invalid request - empty messages
pub fn invalid_request_empty_messages() -> serde_json::Value {
    json!({
        "model": "test-model",
        "messages": []
    })
}

/// Invalid request - temperature out of range
pub fn invalid_request_bad_temperature() -> serde_json::Value {
    json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 5.0
    })
}

/// Test API key (admin scope)
pub const TEST_API_KEY_ADMIN: &str = "sk-adm-test1234567890abcdefghij";

/// Test API key (inference scope)
pub const TEST_API_KEY_INFERENCE: &str = "sk-inf-test1234567890abcdefghij";

/// Test API key (invalid)
pub const TEST_API_KEY_INVALID: &str = "sk-invalid-key";

/// Test model ID
pub const TEST_MODEL_ID: &str = "test-model";

/// Test embedding model ID
pub const TEST_EMBEDDING_MODEL_ID: &str = "test-embedding-model";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_chat_request_valid_json() {
        let request = sample_chat_request();
        assert!(request.get("model").is_some());
        assert!(request.get("messages").is_some());
    }

    #[test]
    fn test_sample_chat_response_structure() {
        let response = sample_chat_response();
        assert!(response.get("choices").is_some());
        assert!(response.get("usage").is_some());
    }
}
