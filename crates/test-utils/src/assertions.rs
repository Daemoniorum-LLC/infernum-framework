//! Custom assertion helpers for testing
//!
//! Provides macros and functions for common test assertions.

use serde_json::Value;

/// Assert that a JSON response has an error with the given code
pub fn assert_error_code(response: &Value, expected_code: &str) {
    let error = response
        .get("error")
        .expect("Response should have 'error' field");
    let code = error
        .get("code")
        .and_then(|v| v.as_str())
        .expect("Error should have 'code' field");
    assert_eq!(
        code, expected_code,
        "Expected error code '{}', got '{}'",
        expected_code, code
    );
}

/// Assert that a JSON response has the expected HTTP-like status in error
pub fn assert_error_type(response: &Value, expected_type: &str) {
    let error = response
        .get("error")
        .expect("Response should have 'error' field");
    let error_type = error
        .get("type")
        .and_then(|v| v.as_str())
        .expect("Error should have 'type' field");
    assert_eq!(
        error_type, expected_type,
        "Expected error type '{}', got '{}'",
        expected_type, error_type
    );
}

/// Assert that a chat response has at least one choice
pub fn assert_has_choices(response: &Value) {
    let choices = response
        .get("choices")
        .and_then(|v| v.as_array())
        .expect("Response should have 'choices' array");
    assert!(
        !choices.is_empty(),
        "Response should have at least one choice"
    );
}

/// Assert that a chat response has usage information
pub fn assert_has_usage(response: &Value) {
    let usage = response
        .get("usage")
        .expect("Response should have 'usage' field");
    assert!(
        usage.get("prompt_tokens").is_some(),
        "Usage should have 'prompt_tokens'"
    );
    assert!(
        usage.get("completion_tokens").is_some(),
        "Usage should have 'completion_tokens'"
    );
    assert!(
        usage.get("total_tokens").is_some(),
        "Usage should have 'total_tokens'"
    );
}

/// Assert that a response contains a specific header
pub fn assert_header_present(headers: &[(String, String)], name: &str) {
    let found = headers.iter().any(|(k, _)| k.eq_ignore_ascii_case(name));
    assert!(found, "Expected header '{}' to be present", name);
}

/// Assert that a response header has a specific value
pub fn assert_header_value(headers: &[(String, String)], name: &str, expected: &str) {
    let value = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.as_str());

    match value {
        Some(v) => assert_eq!(
            v, expected,
            "Expected header '{}' to be '{}', got '{}'",
            name, expected, v
        ),
        None => panic!("Header '{}' not found", name),
    }
}

/// Assert response time is within acceptable range
pub fn assert_response_time_ms(elapsed_ms: u64, max_ms: u64) {
    assert!(
        elapsed_ms <= max_ms,
        "Response time {}ms exceeded maximum {}ms",
        elapsed_ms,
        max_ms
    );
}

/// Get a string field from JSON
pub fn get_json_string(json: &Value, path: &str) -> Option<String> {
    json.get(path).and_then(|v| v.as_str()).map(String::from)
}

/// Get an i64 field from JSON
pub fn get_json_i64(json: &Value, path: &str) -> Option<i64> {
    json.get(path).and_then(|v| v.as_i64())
}

/// Get an f64 field from JSON
pub fn get_json_f64(json: &Value, path: &str) -> Option<f64> {
    json.get(path).and_then(|v| v.as_f64())
}

/// Get a bool field from JSON
pub fn get_json_bool(json: &Value, path: &str) -> Option<bool> {
    json.get(path).and_then(|v| v.as_bool())
}

/// Assert JSON string field equals expected value
pub fn assert_json_string(json: &Value, path: &str, expected: &str) {
    let value = get_json_string(json, path);
    assert_eq!(
        value.as_deref(),
        Some(expected),
        "Expected field '{}' to equal '{}'",
        path,
        expected
    );
}

/// Assert JSON i64 field equals expected value
pub fn assert_json_i64(json: &Value, path: &str, expected: i64) {
    let value = get_json_i64(json, path);
    assert_eq!(
        value,
        Some(expected),
        "Expected field '{}' to equal {}",
        path,
        expected
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_assert_error_code() {
        let response = json!({
            "error": {
                "code": "invalid_request",
                "message": "Bad request",
                "type": "invalid_request_error"
            }
        });
        assert_error_code(&response, "invalid_request");
    }

    #[test]
    fn test_assert_has_choices() {
        let response = json!({
            "choices": [{"index": 0, "message": {"content": "Hello"}}]
        });
        assert_has_choices(&response);
    }

    #[test]
    fn test_assert_has_usage() {
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
    }

    #[test]
    fn test_assert_json_string() {
        let json = json!({"model": "gpt-4"});
        assert_json_string(&json, "model", "gpt-4");
    }

    #[test]
    fn test_assert_json_i64() {
        let json = json!({"tokens": 100});
        assert_json_i64(&json, "tokens", 100);
    }
}
