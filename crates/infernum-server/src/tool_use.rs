//! Tool Use Processing for Chat Completions
//!
//! This module handles:
//! - Formatting tools into model-specific prompts
//! - Detecting tool calls in model output
//! - Extracting text content from mixed tool/text responses
//!
//! # Supported Models
//!
//! Currently supports Qwen format. Llama and Mistral formats are planned.
//!
//! # Example
//!
//! ```rust,ignore
//! use infernum_server::tool_use::{ModelFamily, format_tools_for_prompt, detect_tool_calls};
//!
//! let family = ModelFamily::from_model_name("Qwen/Qwen2.5-7B-Instruct");
//! let prompt = format_tools_for_prompt(&tools, family);
//! let detected = detect_tool_calls(&model_output, family);
//! ```

use std::sync::OnceLock;

use regex::Regex;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::openai::{FunctionCall, Tool, ToolCall, ToolChoice};

/// Static regex for detecting tool calls (compiled once).
static TOOL_CALL_REGEX: OnceLock<Regex> = OnceLock::new();

/// Static regex for extracting text content (compiled once).
static TOOL_CALL_EXTRACT_REGEX: OnceLock<Regex> = OnceLock::new();

fn get_tool_call_regex() -> &'static Regex {
    TOOL_CALL_REGEX.get_or_init(|| {
        Regex::new(r"(?s)<tool_call>\s*(.*?)\s*</tool_call>")
            .expect("invalid tool_call regex")
    })
}

fn get_tool_call_extract_regex() -> &'static Regex {
    TOOL_CALL_EXTRACT_REGEX.get_or_init(|| {
        Regex::new(r"(?s)<tool_call>.*?</tool_call>")
            .expect("invalid tool_call_extract regex")
    })
}

/// Model family for tool format selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelFamily {
    /// Qwen models (Qwen2.5, etc.)
    Qwen,
    /// Llama models (Llama 3, etc.)
    Llama,
    /// Mistral models
    Mistral,
    /// Unknown model family - use generic format
    #[default]
    Unknown,
}

impl ModelFamily {
    /// Detect model family from model name.
    #[must_use]
    pub fn from_model_name(name: &str) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("qwen") {
            Self::Qwen
        } else if lower.contains("llama") {
            Self::Llama
        } else if lower.contains("mistral") || lower.contains("mixtral") {
            Self::Mistral
        } else {
            Self::Unknown
        }
    }
}

/// A detected tool call from model output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedToolCall {
    /// Unique ID for this tool call.
    pub id: String,
    /// Function name.
    pub name: String,
    /// Arguments as JSON string.
    pub arguments: String,
}

impl DetectedToolCall {
    /// Convert to OpenAI ToolCall format.
    #[must_use]
    pub fn to_tool_call(&self) -> ToolCall {
        ToolCall {
            id: self.id.clone(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: self.name.clone(),
                arguments: self.arguments.clone(),
            },
        }
    }
}

/// Result of processing model output for tool calls.
#[derive(Debug, Clone)]
pub struct ToolProcessingResult {
    /// Text content (if any) after removing tool calls.
    pub content: Option<String>,
    /// Detected tool calls.
    pub tool_calls: Vec<ToolCall>,
    /// Finish reason ("stop" or "tool_calls").
    pub finish_reason: String,
}

/// Format tools for inclusion in the model prompt.
///
/// Different model families require different formatting:
/// - Qwen: Uses markdown-style tool definitions with `<tool_call>` output format
/// - Llama: (planned) Uses different tags
/// - Mistral: (planned) Uses different tags
///
/// # Arguments
/// * `tools` - The tools to format
/// * `model_family` - The model family for format selection
///
/// # Returns
/// A string to append to the system prompt
#[must_use]
pub fn format_tools_for_prompt(tools: &[Tool], model_family: ModelFamily) -> String {
    if tools.is_empty() {
        return String::new();
    }

    match model_family {
        ModelFamily::Qwen | ModelFamily::Unknown => format_tools_qwen(tools),
        ModelFamily::Llama => format_tools_llama(tools),
        ModelFamily::Mistral => format_tools_mistral(tools),
    }
}

/// Format tools in Qwen style.
fn format_tools_qwen(tools: &[Tool]) -> String {
    let mut result = String::from("\n\n# Tools\n\nYou have access to the following tools:\n");

    for tool in tools {
        result.push_str("\n## ");
        result.push_str(&tool.function.name);
        result.push('\n');

        if let Some(desc) = &tool.function.description {
            result.push('\n');
            result.push_str(desc);
            result.push('\n');
        }

        if let Some(params) = &tool.function.parameters {
            result.push_str("\nParameters:\n");
            if let Some(props) = params.get("properties") {
                if let Some(obj) = props.as_object() {
                    let required: Vec<&str> = params
                        .get("required")
                        .and_then(|r| r.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str())
                                .collect()
                        })
                        .unwrap_or_default();

                    for (name, schema) in obj {
                        let type_str = schema.get("type").and_then(|t| t.as_str()).unwrap_or("any");
                        let is_required = required.contains(&name.as_str());
                        let req_str = if is_required { ", required" } else { "" };

                        result.push_str(&format!("- {name} ({type_str}{req_str})"));

                        if let Some(desc) = schema.get("description").and_then(|d| d.as_str()) {
                            result.push_str(": ");
                            result.push_str(desc);
                        }
                        result.push('\n');
                    }
                }
            }
        }
    }

    result.push_str("\nTo use a tool, respond with:\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"arg1\": \"value1\"}}\n</tool_call>\n");

    result
}

/// Format tools in Llama 3 style.
///
/// Llama 3 uses a JSON-based function calling format with `<|python_tag|>` for tool calls.
fn format_tools_llama(tools: &[Tool]) -> String {
    let mut result = String::from("\n\nYou have access to the following functions:\n\n");

    for tool in tools {
        // Build JSON schema for the function
        let func_json = serde_json::json!({
            "name": tool.function.name,
            "description": tool.function.description,
            "parameters": tool.function.parameters
        });

        result.push_str(&serde_json::to_string_pretty(&func_json).unwrap_or_default());
        result.push_str("\n\n");
    }

    result.push_str("To call a function, respond with a JSON object in the following format:\n");
    result.push_str("<|python_tag|>{\"name\": \"function_name\", \"arguments\": {\"arg1\": \"value1\"}}\n");
    result.push_str("\nOnly call functions when necessary to answer the user's request.\n");

    result
}

/// Format tools in Mistral style.
///
/// Mistral uses `[AVAILABLE_TOOLS]` for defining tools and `[TOOL_CALLS]` for responses.
fn format_tools_mistral(tools: &[Tool]) -> String {
    let mut result = String::from("\n\n[AVAILABLE_TOOLS]\n");

    // Format tools as JSON array
    let tools_json: Vec<serde_json::Value> = tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": t.function.name,
                    "description": t.function.description,
                    "parameters": t.function.parameters
                }
            })
        })
        .collect();

    result.push_str(&serde_json::to_string(&tools_json).unwrap_or_default());
    result.push_str("\n[/AVAILABLE_TOOLS]\n\n");
    result.push_str("When you need to call a tool, respond with:\n");
    result.push_str("[TOOL_CALLS] [{\"name\": \"function_name\", \"arguments\": {\"arg1\": \"value1\"}}]\n");

    result
}

/// Detect tool calls in model output.
///
/// Parses the model output for tool call patterns based on the model family.
///
/// # Arguments
/// * `output` - The raw model output text
/// * `model_family` - The model family for pattern selection
///
/// # Returns
/// A vector of detected tool calls (may be empty)
#[must_use]
pub fn detect_tool_calls(output: &str, model_family: ModelFamily) -> Vec<DetectedToolCall> {
    match model_family {
        ModelFamily::Qwen | ModelFamily::Unknown => detect_tool_calls_qwen(output),
        ModelFamily::Llama => detect_tool_calls_llama(output),
        ModelFamily::Mistral => detect_tool_calls_mistral(output),
    }
}

/// Detect Qwen-style tool calls using `<tool_call>...</tool_call>` tags.
fn detect_tool_calls_qwen(output: &str) -> Vec<DetectedToolCall> {
    let re = get_tool_call_regex();
    let mut calls = Vec::new();

    for cap in re.captures_iter(output) {
        if let Some(json_match) = cap.get(1) {
            let json_str = json_match.as_str();
            if let Ok(parsed) = serde_json::from_str::<ToolCallJson>(json_str) {
                // Use full UUID for safety (no slicing that could panic)
                let id = format!("call_{}", Uuid::new_v4().simple());
                calls.push(DetectedToolCall {
                    id,
                    name: parsed.name,
                    arguments: serde_json::to_string(&parsed.arguments).unwrap_or_default(),
                });
            }
        }
    }

    calls
}

/// Internal struct for parsing tool call JSON.
#[derive(Debug, Deserialize)]
struct ToolCallJson {
    name: String,
    arguments: serde_json::Value,
}

/// Static regex for Llama python_tag detection.
static LLAMA_TOOL_CALL_REGEX: OnceLock<Regex> = OnceLock::new();

fn get_llama_tool_call_regex() -> &'static Regex {
    LLAMA_TOOL_CALL_REGEX.get_or_init(|| {
        // Match JSON object after <|python_tag|>, allowing nested braces
        Regex::new(r#"<\|python_tag\|>\s*(\{(?:[^{}]|\{[^{}]*\})*\})"#)
            .expect("invalid llama tool_call regex")
    })
}

/// Detect Llama-style tool calls using `<|python_tag|>` markers.
fn detect_tool_calls_llama(output: &str) -> Vec<DetectedToolCall> {
    let re = get_llama_tool_call_regex();
    let mut calls = Vec::new();

    for cap in re.captures_iter(output) {
        if let Some(json_match) = cap.get(1) {
            let json_str = json_match.as_str();
            if let Ok(parsed) = serde_json::from_str::<ToolCallJson>(json_str) {
                let id = format!("call_{}", Uuid::new_v4().simple());
                calls.push(DetectedToolCall {
                    id,
                    name: parsed.name,
                    arguments: serde_json::to_string(&parsed.arguments).unwrap_or_default(),
                });
            }
        }
    }

    // Fallback: also check for Qwen-style tags (models sometimes use both)
    if calls.is_empty() {
        calls = detect_tool_calls_qwen(output);
    }

    calls
}

/// Static regex for Mistral TOOL_CALLS detection.
static MISTRAL_TOOL_CALL_REGEX: OnceLock<Regex> = OnceLock::new();

fn get_mistral_tool_call_regex() -> &'static Regex {
    MISTRAL_TOOL_CALL_REGEX.get_or_init(|| {
        Regex::new(r#"\[TOOL_CALLS\]\s*\[([^\]]+)\]"#)
            .expect("invalid mistral tool_call regex")
    })
}

/// Detect Mistral-style tool calls using `[TOOL_CALLS]` markers.
fn detect_tool_calls_mistral(output: &str) -> Vec<DetectedToolCall> {
    let re = get_mistral_tool_call_regex();
    let mut calls = Vec::new();

    for cap in re.captures_iter(output) {
        if let Some(json_match) = cap.get(1) {
            // Mistral wraps the calls in an array, so we need to parse it
            let json_str = format!("[{}]", json_match.as_str());
            if let Ok(parsed) = serde_json::from_str::<Vec<ToolCallJson>>(&json_str) {
                for tool_call in parsed {
                    let id = format!("call_{}", Uuid::new_v4().simple());
                    calls.push(DetectedToolCall {
                        id,
                        name: tool_call.name,
                        arguments: serde_json::to_string(&tool_call.arguments).unwrap_or_default(),
                    });
                }
            }
        }
    }

    // Fallback: also check for Qwen-style tags
    if calls.is_empty() {
        calls = detect_tool_calls_qwen(output);
    }

    calls
}

/// Extract text content from model output, removing tool call sections.
///
/// # Arguments
/// * `output` - The raw model output text
/// * `model_family` - The model family for pattern selection
///
/// # Returns
/// The text content with tool calls removed, or None if only tool calls
#[must_use]
pub fn extract_text_content(output: &str, model_family: ModelFamily) -> Option<String> {
    match model_family {
        ModelFamily::Qwen | ModelFamily::Unknown => extract_text_content_qwen(output),
        ModelFamily::Llama => extract_text_content_llama(output),
        ModelFamily::Mistral => extract_text_content_mistral(output),
    }
}

/// Extract text content for Qwen format.
fn extract_text_content_qwen(output: &str) -> Option<String> {
    let re = get_tool_call_extract_regex();
    let cleaned = re.replace_all(output, "");
    let trimmed = cleaned.trim();

    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Static regex for Llama text extraction.
static LLAMA_EXTRACT_REGEX: OnceLock<Regex> = OnceLock::new();

fn get_llama_extract_regex() -> &'static Regex {
    LLAMA_EXTRACT_REGEX.get_or_init(|| {
        // Match <|python_tag|> followed by JSON object with nested braces
        Regex::new(r#"<\|python_tag\|>\s*\{(?:[^{}]|\{[^{}]*\})*\}"#)
            .expect("invalid llama extract regex")
    })
}

/// Extract text content for Llama format.
fn extract_text_content_llama(output: &str) -> Option<String> {
    let re = get_llama_extract_regex();
    let cleaned = re.replace_all(output, "");
    // Also remove Qwen-style tags as fallback
    let qwen_re = get_tool_call_extract_regex();
    let cleaned = qwen_re.replace_all(&cleaned, "");
    let trimmed = cleaned.trim();

    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Static regex for Mistral text extraction.
static MISTRAL_EXTRACT_REGEX: OnceLock<Regex> = OnceLock::new();

fn get_mistral_extract_regex() -> &'static Regex {
    MISTRAL_EXTRACT_REGEX.get_or_init(|| {
        Regex::new(r#"\[TOOL_CALLS\]\s*\[[^\]]+\]"#)
            .expect("invalid mistral extract regex")
    })
}

/// Extract text content for Mistral format.
fn extract_text_content_mistral(output: &str) -> Option<String> {
    let re = get_mistral_extract_regex();
    let cleaned = re.replace_all(output, "");
    // Also remove Qwen-style tags as fallback
    let qwen_re = get_tool_call_extract_regex();
    let cleaned = qwen_re.replace_all(&cleaned, "");
    let trimmed = cleaned.trim();

    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Process model output for tool calls and content.
///
/// This is the main entry point for processing model output. It detects
/// tool calls, extracts text content, and determines the finish reason.
///
/// # Arguments
/// * `output` - The raw model output text
/// * `model_family` - The model family for format selection
///
/// # Returns
/// A `ToolProcessingResult` with content, tool calls, and finish reason
#[must_use]
pub fn process_model_output(output: &str, model_family: ModelFamily) -> ToolProcessingResult {
    let detected = detect_tool_calls(output, model_family);
    let content = extract_text_content(output, model_family);

    let tool_calls: Vec<ToolCall> = detected.iter().map(DetectedToolCall::to_tool_call).collect();

    let finish_reason = if tool_calls.is_empty() {
        "stop".to_string()
    } else {
        "tool_calls".to_string()
    };

    ToolProcessingResult {
        content,
        tool_calls,
        finish_reason,
    }
}

/// Validate that a tool exists in the provided tools list.
///
/// # Arguments
/// * `tool_name` - The name of the tool to validate
/// * `tools` - The list of available tools
///
/// # Returns
/// True if the tool exists, false otherwise
#[must_use]
pub fn validate_tool_exists(tool_name: &str, tools: &[Tool]) -> bool {
    tools.iter().any(|t| t.function.name == tool_name)
}

/// Check if tools should be included in the prompt based on tool_choice.
///
/// # Arguments
/// * `tool_choice` - The tool choice setting from the request
///
/// # Returns
/// True if tools should be formatted into the prompt
#[must_use]
pub fn should_include_tools(tool_choice: Option<&ToolChoice>) -> bool {
    match tool_choice {
        None => true, // Default is "auto"
        Some(ToolChoice::String(s)) => s != "none",
        Some(ToolChoice::Tool(_)) => true, // Specific tool requested
    }
}

/// Get the specific tool name if tool_choice forces a specific tool.
///
/// # Arguments
/// * `tool_choice` - The tool choice setting from the request
///
/// # Returns
/// Some(tool_name) if a specific tool is forced, None otherwise
#[must_use]
pub fn get_forced_tool(tool_choice: Option<&ToolChoice>) -> Option<&str> {
    match tool_choice {
        Some(ToolChoice::Tool(tc)) => Some(&tc.function.name),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_tool(name: &str, description: &str, params: serde_json::Value) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: crate::openai::FunctionDefinition {
                name: name.to_string(),
                description: Some(description.to_string()),
                parameters: Some(params),
                strict: None,
            },
        }
    }

    // === Model Family Tests ===

    #[test]
    fn test_model_family_qwen() {
        assert_eq!(
            ModelFamily::from_model_name("Qwen/Qwen2.5-7B-Instruct"),
            ModelFamily::Qwen
        );
        assert_eq!(
            ModelFamily::from_model_name("qwen2.5-coder"),
            ModelFamily::Qwen
        );
    }

    #[test]
    fn test_model_family_llama() {
        assert_eq!(
            ModelFamily::from_model_name("meta-llama/Llama-3.2-3B-Instruct"),
            ModelFamily::Llama
        );
        assert_eq!(
            ModelFamily::from_model_name("llama-3.1"),
            ModelFamily::Llama
        );
    }

    #[test]
    fn test_model_family_mistral() {
        assert_eq!(
            ModelFamily::from_model_name("mistralai/Mistral-7B-Instruct"),
            ModelFamily::Mistral
        );
        assert_eq!(
            ModelFamily::from_model_name("Mixtral-8x7B"),
            ModelFamily::Mistral
        );
    }

    #[test]
    fn test_model_family_unknown() {
        assert_eq!(
            ModelFamily::from_model_name("unknown-model"),
            ModelFamily::Unknown
        );
    }

    // === Tool Formatting Tests ===

    #[test]
    fn test_format_empty_tools() {
        let result = format_tools_for_prompt(&[], ModelFamily::Qwen);
        assert!(result.is_empty());
    }

    #[test]
    fn test_format_single_tool_qwen() {
        let tool = make_tool(
            "get_weather",
            "Get current weather for a location",
            json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }),
        );

        let result = format_tools_for_prompt(&[tool], ModelFamily::Qwen);

        assert!(result.contains("# Tools"));
        assert!(result.contains("## get_weather"));
        assert!(result.contains("Get current weather for a location"));
        assert!(result.contains("- location (string, required): City name"));
        assert!(result.contains("<tool_call>"));
    }

    #[test]
    fn test_format_multiple_tools() {
        let tools = vec![
            make_tool("tool_a", "First tool", json!({"type": "object", "properties": {}})),
            make_tool("tool_b", "Second tool", json!({"type": "object", "properties": {}})),
        ];

        let result = format_tools_for_prompt(&tools, ModelFamily::Qwen);

        assert!(result.contains("## tool_a"));
        assert!(result.contains("## tool_b"));
        assert!(result.contains("First tool"));
        assert!(result.contains("Second tool"));
    }

    // === Tool Detection Tests ===

    #[test]
    fn test_detect_qwen_tool_call() {
        let output = r#"I'll get the weather for you.
<tool_call>
{"name": "get_weather", "arguments": {"location": "Seattle"}}
</tool_call>"#;

        let calls = detect_tool_calls(output, ModelFamily::Qwen);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert!(calls[0].arguments.contains("Seattle"));
        assert!(calls[0].id.starts_with("call_"));
    }

    #[test]
    fn test_detect_multiple_tool_calls() {
        let output = r#"<tool_call>
{"name": "tool_a", "arguments": {}}
</tool_call>
Some text
<tool_call>
{"name": "tool_b", "arguments": {"x": 1}}
</tool_call>"#;

        let calls = detect_tool_calls(output, ModelFamily::Qwen);

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "tool_a");
        assert_eq!(calls[1].name, "tool_b");
    }

    #[test]
    fn test_detect_no_tool_calls() {
        let output = "Just a regular response without any tool calls.";
        let calls = detect_tool_calls(output, ModelFamily::Qwen);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_detect_malformed_tool_call() {
        let output = r#"<tool_call>
not valid json
</tool_call>"#;

        let calls = detect_tool_calls(output, ModelFamily::Qwen);
        assert!(calls.is_empty()); // Gracefully handle malformed JSON
    }

    // === Text Extraction Tests ===

    #[test]
    fn test_extract_text_with_tool_call() {
        let output = r#"Here is some text.
<tool_call>
{"name": "test", "arguments": {}}
</tool_call>
More text after."#;

        let content = extract_text_content(output, ModelFamily::Qwen);

        assert!(content.is_some());
        let text = content.unwrap();
        assert!(text.contains("Here is some text."));
        assert!(text.contains("More text after."));
        assert!(!text.contains("<tool_call>"));
    }

    #[test]
    fn test_extract_text_only_tool_call() {
        let output = r#"<tool_call>
{"name": "test", "arguments": {}}
</tool_call>"#;

        let content = extract_text_content(output, ModelFamily::Qwen);
        assert!(content.is_none());
    }

    #[test]
    fn test_extract_text_no_tool_call() {
        let output = "Just plain text.";
        let content = extract_text_content(output, ModelFamily::Qwen);

        assert!(content.is_some());
        assert_eq!(content.unwrap(), "Just plain text.");
    }

    // === Processing Tests ===

    #[test]
    fn test_process_model_output_with_tool_call() {
        let output = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "Seattle"}}
</tool_call>"#;

        let result = process_model_output(output, ModelFamily::Qwen);

        assert_eq!(result.finish_reason, "tool_calls");
        assert!(result.content.is_none());
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_process_model_output_no_tool_call() {
        let output = "This is a regular response.";

        let result = process_model_output(output, ModelFamily::Qwen);

        assert_eq!(result.finish_reason, "stop");
        assert!(result.content.is_some());
        assert!(result.tool_calls.is_empty());
    }

    #[test]
    fn test_process_model_output_mixed() {
        let output = r#"Let me help you with that.
<tool_call>
{"name": "search", "arguments": {"query": "test"}}
</tool_call>"#;

        let result = process_model_output(output, ModelFamily::Qwen);

        assert_eq!(result.finish_reason, "tool_calls");
        assert!(result.content.is_some());
        assert!(result.content.unwrap().contains("Let me help you"));
        assert_eq!(result.tool_calls.len(), 1);
    }

    // === Validation Tests ===

    #[test]
    fn test_validate_tool_exists() {
        let tools = vec![
            make_tool("tool_a", "A", json!({})),
            make_tool("tool_b", "B", json!({})),
        ];

        assert!(validate_tool_exists("tool_a", &tools));
        assert!(validate_tool_exists("tool_b", &tools));
        assert!(!validate_tool_exists("tool_c", &tools));
    }

    // === DetectedToolCall Conversion ===

    #[test]
    fn test_detected_to_tool_call() {
        let detected = DetectedToolCall {
            id: "call_abc123".to_string(),
            name: "test_function".to_string(),
            arguments: r#"{"key": "value"}"#.to_string(),
        };

        let tool_call = detected.to_tool_call();

        assert_eq!(tool_call.id, "call_abc123");
        assert_eq!(tool_call.call_type, "function");
        assert_eq!(tool_call.function.name, "test_function");
        assert_eq!(tool_call.function.arguments, r#"{"key": "value"}"#);
    }

    // === Tool Choice Tests ===

    #[test]
    fn test_should_include_tools_none_default() {
        // Default (None) should include tools
        assert!(should_include_tools(None));
    }

    #[test]
    fn test_should_include_tools_auto() {
        let choice = ToolChoice::String("auto".to_string());
        assert!(should_include_tools(Some(&choice)));
    }

    #[test]
    fn test_should_include_tools_none_string() {
        let choice = ToolChoice::String("none".to_string());
        assert!(!should_include_tools(Some(&choice)));
    }

    #[test]
    fn test_should_include_tools_required() {
        let choice = ToolChoice::String("required".to_string());
        assert!(should_include_tools(Some(&choice)));
    }

    #[test]
    fn test_should_include_tools_specific_tool() {
        use crate::openai::{ToolChoiceFunction, ToolChoiceFunctionName};
        let choice = ToolChoice::Tool(ToolChoiceFunction {
            choice_type: "function".to_string(),
            function: ToolChoiceFunctionName {
                name: "get_weather".to_string(),
            },
        });
        assert!(should_include_tools(Some(&choice)));
    }

    #[test]
    fn test_get_forced_tool_none() {
        assert!(get_forced_tool(None).is_none());
    }

    #[test]
    fn test_get_forced_tool_auto() {
        let choice = ToolChoice::String("auto".to_string());
        assert!(get_forced_tool(Some(&choice)).is_none());
    }

    #[test]
    fn test_get_forced_tool_specific() {
        use crate::openai::{ToolChoiceFunction, ToolChoiceFunctionName};
        let choice = ToolChoice::Tool(ToolChoiceFunction {
            choice_type: "function".to_string(),
            function: ToolChoiceFunctionName {
                name: "get_weather".to_string(),
            },
        });
        assert_eq!(get_forced_tool(Some(&choice)), Some("get_weather"));
    }

    // === Llama Format Tests ===

    #[test]
    fn test_format_single_tool_llama() {
        let tool = make_tool(
            "get_weather",
            "Get current weather for a location",
            json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }),
        );

        let result = format_tools_for_prompt(&[tool], ModelFamily::Llama);

        assert!(result.contains("get_weather"));
        assert!(result.contains("Get current weather"));
        assert!(result.contains("<|python_tag|>"));
    }

    #[test]
    fn test_detect_llama_tool_call() {
        let output = r#"I'll check the weather.
<|python_tag|>{"name": "get_weather", "arguments": {"location": "Seattle"}}"#;

        let calls = detect_tool_calls(output, ModelFamily::Llama);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert!(calls[0].arguments.contains("Seattle"));
    }

    #[test]
    fn test_extract_text_llama() {
        let output = r#"Here is some text.
<|python_tag|>{"name": "test", "arguments": {}}
More text after."#;

        let content = extract_text_content(output, ModelFamily::Llama);

        assert!(content.is_some());
        let text = content.unwrap();
        assert!(text.contains("Here is some text."));
        assert!(text.contains("More text after."));
        assert!(!text.contains("<|python_tag|>"));
    }

    // === Mistral Format Tests ===

    #[test]
    fn test_format_single_tool_mistral() {
        let tool = make_tool(
            "get_weather",
            "Get current weather for a location",
            json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }),
        );

        let result = format_tools_for_prompt(&[tool], ModelFamily::Mistral);

        assert!(result.contains("[AVAILABLE_TOOLS]"));
        assert!(result.contains("get_weather"));
        assert!(result.contains("[TOOL_CALLS]"));
    }

    #[test]
    fn test_detect_mistral_tool_call() {
        let output = r#"I'll check the weather.
[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "Seattle"}}]"#;

        let calls = detect_tool_calls(output, ModelFamily::Mistral);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert!(calls[0].arguments.contains("Seattle"));
    }

    #[test]
    fn test_extract_text_mistral() {
        let output = r#"Here is some text.
[TOOL_CALLS] [{"name": "test", "arguments": {}}]
More text after."#;

        let content = extract_text_content(output, ModelFamily::Mistral);

        assert!(content.is_some());
        let text = content.unwrap();
        assert!(text.contains("Here is some text."));
        assert!(text.contains("More text after."));
        assert!(!text.contains("[TOOL_CALLS]"));
    }

    #[test]
    fn test_llama_falls_back_to_qwen_format() {
        // Llama detection should fall back to Qwen format if <|python_tag|> not found
        let output = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "Seattle"}}
</tool_call>"#;

        let calls = detect_tool_calls(output, ModelFamily::Llama);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
    }

    #[test]
    fn test_mistral_falls_back_to_qwen_format() {
        // Mistral detection should fall back to Qwen format if [TOOL_CALLS] not found
        let output = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "Seattle"}}
</tool_call>"#;

        let calls = detect_tool_calls(output, ModelFamily::Mistral);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
    }
}
