//! Test case types for JSON test files.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A test suite containing multiple test cases.
#[derive(Debug, Deserialize)]
pub struct TestSuite {
    pub name: String,
    pub description: Option<String>,
    pub cases: Vec<TestCase>,
}

/// A single test case.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum TestCase {
    CollectText(CollectTextCase),
    ChunkCreation(ChunkCreationCase),
    SamplingValidation(SamplingValidationCase),
    SamplingPreset(SamplingPresetCase),
    UsageTest(UsageTestCase),
}

/// Test case for collect_text functionality.
#[derive(Debug, Deserialize)]
pub struct CollectTextCase {
    pub name: String,
    pub description: Option<String>,
    pub chunks: Vec<ChunkInput>,
    pub expected_text: String,
}

/// Input chunk for streaming tests.
#[derive(Debug, Deserialize)]
pub struct ChunkInput {
    pub choices: Vec<ChoiceInput>,
}

/// Input choice for streaming tests.
#[derive(Debug, Deserialize)]
pub struct ChoiceInput {
    pub index: u32,
    pub delta: DeltaInput,
    pub finish_reason: Option<String>,
}

/// Input delta for streaming tests.
#[derive(Debug, Deserialize)]
pub struct DeltaInput {
    pub content: Option<String>,
    pub role: Option<String>,
}

/// Test case for chunk creation.
#[derive(Debug, Deserialize)]
pub struct ChunkCreationCase {
    pub name: String,
    pub request_id: Option<String>,
    pub model: Option<String>,
    pub content: String,
    pub expected_fields: HashMap<String, serde_json::Value>,
}

/// Test case for sampling parameter validation.
#[derive(Debug, Deserialize)]
pub struct SamplingValidationCase {
    pub name: String,
    pub params: SamplingParamsInput,
    pub expected_valid: bool,
    pub expected_error: Option<String>,
}

/// Input sampling parameters.
#[derive(Debug, Default, Deserialize)]
pub struct SamplingParamsInput {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub max_tokens: Option<u32>,
    pub stop: Option<Vec<String>>,
}

/// Test case for sampling presets.
#[derive(Debug, Deserialize)]
pub struct SamplingPresetCase {
    pub name: String,
    pub action: String,
    pub expected: SamplingPresetExpected,
}

/// Expected values for sampling preset tests.
#[derive(Debug, Default, Deserialize)]
pub struct SamplingPresetExpected {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub max_tokens: Option<u32>,
    pub stop_sequences: Option<Vec<String>>,
    pub seed: Option<u64>,
}

/// Test case for Usage type.
#[derive(Debug, Deserialize)]
pub struct UsageTestCase {
    pub name: String,
    pub action: String,
    pub expected: UsageExpected,
}

/// Expected usage values.
#[derive(Debug, Deserialize)]
pub struct UsageExpected {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Test result for a single case.
#[derive(Debug, Serialize)]
pub struct TestResult {
    pub name: String,
    pub passed: bool,
    pub error: Option<String>,
    pub duration_us: u64,
}

/// Summary of test results for a suite.
#[derive(Debug, Serialize)]
pub struct SuiteResult {
    pub suite_name: String,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub results: Vec<TestResult>,
}
