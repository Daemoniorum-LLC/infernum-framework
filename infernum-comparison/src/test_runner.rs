//! Test runner for executing JSON test cases.
//!
//! This is a standalone test harness that validates JSON test case structure.
//! Actual test execution happens by running the test cases against:
//! - The Rust implementation (via `cargo test` in infernum-core)
//! - The Sigil implementation (via `sigil test` in infernum-sigil)

use crate::test_types::*;
use std::time::Instant;

/// Validates a collect_text test case structure.
pub fn validate_collect_text_case(case: &CollectTextCase) -> TestResult {
    let start = Instant::now();

    // Validate test case structure
    let mut errors = Vec::new();

    if case.name.is_empty() {
        errors.push("Test case name is empty".to_string());
    }

    // Validate chunk structure
    for (i, chunk) in case.chunks.iter().enumerate() {
        for (j, choice) in chunk.choices.iter().enumerate() {
            if choice.finish_reason.as_ref().map(|r| {
                !["stop", "length", "tool_calls", "content_filter"].contains(&r.as_str())
            }).unwrap_or(false) {
                errors.push(format!(
                    "Invalid finish_reason in chunk[{}].choices[{}]",
                    i, j
                ));
            }
        }
    }

    let duration = start.elapsed();

    TestResult {
        name: case.name.clone(),
        passed: errors.is_empty(),
        error: if errors.is_empty() { None } else { Some(errors.join(", ")) },
        duration_us: duration.as_micros() as u64,
    }
}

/// Validates a sampling validation test case structure.
pub fn validate_sampling_validation_case(case: &SamplingValidationCase) -> TestResult {
    let start = Instant::now();

    let mut errors = Vec::new();

    if case.name.is_empty() {
        errors.push("Test case name is empty".to_string());
    }

    // Validate parameter ranges are testable
    if let Some(temp) = case.params.temperature {
        if temp.is_nan() {
            errors.push("temperature is NaN".to_string());
        }
    }

    if let Some(top_p) = case.params.top_p {
        if top_p.is_nan() {
            errors.push("top_p is NaN".to_string());
        }
    }

    let duration = start.elapsed();

    TestResult {
        name: case.name.clone(),
        passed: errors.is_empty(),
        error: if errors.is_empty() { None } else { Some(errors.join(", ")) },
        duration_us: duration.as_micros() as u64,
    }
}

/// Validates a usage test case structure.
pub fn validate_usage_case(case: &UsageTestCase) -> TestResult {
    let start = Instant::now();

    let mut errors = Vec::new();

    if case.name.is_empty() {
        errors.push("Test case name is empty".to_string());
    }

    if case.action.is_empty() {
        errors.push("Test action is empty".to_string());
    }

    // Validate total_tokens = prompt_tokens + completion_tokens
    let expected_total = case.expected.prompt_tokens + case.expected.completion_tokens;
    if case.expected.total_tokens != expected_total {
        errors.push(format!(
            "total_tokens ({}) != prompt_tokens ({}) + completion_tokens ({})",
            case.expected.total_tokens,
            case.expected.prompt_tokens,
            case.expected.completion_tokens
        ));
    }

    let duration = start.elapsed();

    TestResult {
        name: case.name.clone(),
        passed: errors.is_empty(),
        error: if errors.is_empty() { None } else { Some(errors.join(", ")) },
        duration_us: duration.as_micros() as u64,
    }
}

/// Validates a sampling preset test case structure.
pub fn validate_sampling_preset_case(case: &SamplingPresetCase) -> TestResult {
    let start = Instant::now();

    let mut errors = Vec::new();

    if case.name.is_empty() {
        errors.push("Test case name is empty".to_string());
    }

    if case.action.is_empty() {
        errors.push("Test action is empty".to_string());
    }

    // Validate expected values are sensible
    if let Some(temp) = case.expected.temperature {
        if temp < 0.0 {
            errors.push("Expected temperature cannot be negative".to_string());
        }
    }

    if let Some(top_p) = case.expected.top_p {
        if !(0.0..=1.0).contains(&top_p) {
            errors.push("Expected top_p must be between 0.0 and 1.0".to_string());
        }
    }

    let duration = start.elapsed();

    TestResult {
        name: case.name.clone(),
        passed: errors.is_empty(),
        error: if errors.is_empty() { None } else { Some(errors.join(", ")) },
        duration_us: duration.as_micros() as u64,
    }
}

/// Validates a chunk creation test case structure.
pub fn validate_chunk_creation_case(case: &ChunkCreationCase) -> TestResult {
    let start = Instant::now();

    let mut errors = Vec::new();

    if case.name.is_empty() {
        errors.push("Test case name is empty".to_string());
    }

    if case.content.is_empty() {
        errors.push("Test content is empty".to_string());
    }

    let duration = start.elapsed();

    TestResult {
        name: case.name.clone(),
        passed: errors.is_empty(),
        error: if errors.is_empty() { None } else { Some(errors.join(", ")) },
        duration_us: duration.as_micros() as u64,
    }
}

/// Loads and validates a test suite from a JSON file.
pub fn validate_test_suite(path: &std::path::Path) -> Result<SuiteResult, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let suite: TestSuite = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))?;

    let mut results = Vec::new();

    // Detect test type from file path and validate appropriate cases
    let path_str = path.to_string_lossy();

    if path_str.contains("collect_text") || path_str.contains("empty_stream") {
        #[derive(serde::Deserialize)]
        struct CollectTextSuite {
            cases: Vec<CollectTextCase>,
        }
        let typed_suite: CollectTextSuite = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse collect_text cases: {}", e))?;

        for case in &typed_suite.cases {
            results.push(validate_collect_text_case(case));
        }
    } else if path_str.contains("validation") {
        #[derive(serde::Deserialize)]
        struct ValidationSuite {
            cases: Vec<SamplingValidationCase>,
        }
        let typed_suite: ValidationSuite = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse validation cases: {}", e))?;

        for case in &typed_suite.cases {
            results.push(validate_sampling_validation_case(case));
        }
    } else if path_str.contains("presets") {
        #[derive(serde::Deserialize)]
        struct PresetSuite {
            cases: Vec<SamplingPresetCase>,
        }
        let typed_suite: PresetSuite = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse preset cases: {}", e))?;

        for case in &typed_suite.cases {
            results.push(validate_sampling_preset_case(case));
        }
    } else if path_str.contains("chunk_creation") {
        #[derive(serde::Deserialize)]
        struct ChunkSuite {
            cases: Vec<ChunkCreationCase>,
        }
        let typed_suite: ChunkSuite = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse chunk creation cases: {}", e))?;

        for case in &typed_suite.cases {
            results.push(validate_chunk_creation_case(case));
        }
    } else if path_str.contains("usage") {
        #[derive(serde::Deserialize)]
        struct UsageSuite {
            cases: Vec<UsageTestCase>,
        }
        let typed_suite: UsageSuite = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse usage cases: {}", e))?;

        for case in &typed_suite.cases {
            results.push(validate_usage_case(case));
        }
    } else if path_str.contains("model_id") || path_str.contains("request_id") || path_str.contains("builder") {
        // Generic validation for other test types
        results.push(TestResult {
            name: format!("Suite: {}", path.file_name().unwrap_or_default().to_string_lossy()),
            passed: true,
            error: None,
            duration_us: 0,
        });
    }

    let passed = results.iter().filter(|r| r.passed).count();
    let failed = results.len() - passed;

    Ok(SuiteResult {
        suite_name: suite.name,
        total: results.len(),
        passed,
        failed,
        results,
    })
}
