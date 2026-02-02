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
    let marker = "<|python_tag|>";
    let mut calls = Vec::new();
    let mut search_start = 0;

    // Use deep JSON parsing instead of regex for nested structures
    while let Some(marker_pos) = output[search_start..].find(marker) {
        let abs_marker_pos = search_start + marker_pos;
        let json_start = abs_marker_pos + marker.len();
        let remaining = &output[json_start..];

        if let Some(json_str) = extract_json_object(remaining, 0) {
            if let Ok(parsed) = serde_json::from_str::<ToolCallJson>(&json_str) {
                let id = format!("call_{}", Uuid::new_v4().simple());
                calls.push(DetectedToolCall {
                    id,
                    name: parsed.name,
                    arguments: serde_json::to_string(&parsed.arguments).unwrap_or_default(),
                });
                search_start = json_start + json_str.len();
                continue;
            }
        }
        // Move past this marker if we couldn't parse
        search_start = json_start;
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

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 3: STREAMING TOOL DETECTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Tool call start markers by model family.
const QWEN_START_MARKERS: &[&str] = &["<tool_call>", "<tool_call", "<tool_", "<tool"];
const LLAMA_START_MARKERS: &[&str] = &["<|python_tag|>", "<|python_tag", "<|python_", "<|python", "<|"];
const MISTRAL_START_MARKERS: &[&str] = &["[TOOL_CALLS]", "[TOOL_CALLS", "[TOOL_", "[TOOL"];

/// Check if a buffer might contain the start of a tool call marker.
///
/// Used during streaming to decide whether to buffer content instead of
/// immediately sending it to the client.
///
/// # Arguments
/// * `buffer` - The current buffer content
/// * `model_family` - The model family for marker selection
///
/// # Returns
/// True if the buffer ends with a potential tool marker prefix
#[must_use]
pub fn buffer_might_contain_tool_start(buffer: &str, model_family: ModelFamily) -> bool {
    let markers = match model_family {
        ModelFamily::Qwen | ModelFamily::Unknown => QWEN_START_MARKERS,
        ModelFamily::Llama => LLAMA_START_MARKERS,
        ModelFamily::Mistral => MISTRAL_START_MARKERS,
    };

    // Check if buffer ends with any prefix of any marker
    for marker in markers {
        // Check all prefixes of the marker
        for prefix_len in 1..=marker.len() {
            let prefix = &marker[..prefix_len];
            if buffer.ends_with(prefix) {
                return true;
            }
        }
    }

    false
}

/// Check if buffered content is definitely not a tool call marker.
///
/// Used to release buffered content when we've accumulated enough
/// characters to know it's not a tool marker.
///
/// # Arguments
/// * `buffer` - The current buffer content
/// * `model_family` - The model family for marker selection
///
/// # Returns
/// True if we can be certain this isn't a tool call
#[must_use]
pub fn definitely_not_tool_call(buffer: &str, model_family: ModelFamily) -> bool {
    let full_marker = match model_family {
        ModelFamily::Qwen | ModelFamily::Unknown => "<tool_call>",
        ModelFamily::Llama => "<|python_tag|>",
        ModelFamily::Mistral => "[TOOL_CALLS]",
    };

    // If buffer contains the full marker, it's definitely a tool call
    if buffer.contains(full_marker) {
        return false;
    }

    // If buffer is longer than the marker and doesn't contain it,
    // check if it could still be starting one at the end
    if buffer.len() > full_marker.len() {
        // Get the tail that could potentially be a partial marker
        let tail_start = buffer.len().saturating_sub(full_marker.len());
        let tail = &buffer[tail_start..];

        // If tail doesn't start any valid marker prefix, we're safe
        !buffer_might_contain_tool_start(tail, model_family)
    } else {
        // Buffer is short - check if it matches any prefix
        !buffer_might_contain_tool_start(buffer, model_family)
    }
}

/// Result of attempting to extract a complete tool call from a buffer.
#[derive(Debug, Clone)]
pub struct StreamingExtractResult {
    /// Whether a complete tool call was found.
    pub found: bool,
    /// Text content before the tool call (if any).
    pub text_before: Option<String>,
    /// The extracted tool call (if found).
    pub call: Option<DetectedToolCall>,
    /// Remaining buffer content after the tool call.
    pub remaining: String,
}

/// Try to extract a complete tool call from a streaming buffer.
///
/// # Arguments
/// * `buffer` - The accumulated buffer content
/// * `model_family` - The model family for pattern selection
///
/// # Returns
/// Extraction result with tool call and remaining content
#[must_use]
pub fn try_extract_complete_tool_call(buffer: &str, model_family: ModelFamily) -> StreamingExtractResult {
    match model_family {
        ModelFamily::Qwen | ModelFamily::Unknown => try_extract_qwen(buffer),
        ModelFamily::Llama => try_extract_llama(buffer),
        ModelFamily::Mistral => try_extract_mistral(buffer),
    }
}

fn try_extract_qwen(buffer: &str) -> StreamingExtractResult {
    let start_tag = "<tool_call>";
    let end_tag = "</tool_call>";

    if let Some(start_idx) = buffer.find(start_tag) {
        if let Some(end_idx) = buffer.find(end_tag) {
            let json_start = start_idx + start_tag.len();
            let json_content = buffer[json_start..end_idx].trim();

            if let Ok(parsed) = serde_json::from_str::<ToolCallJson>(json_content) {
                let id = format!("call_{}", Uuid::new_v4().simple());
                let call = DetectedToolCall {
                    id,
                    name: parsed.name,
                    arguments: serde_json::to_string(&parsed.arguments).unwrap_or_default(),
                };

                let text_before = if start_idx > 0 {
                    Some(buffer[..start_idx].to_string())
                } else {
                    None
                };

                let remaining = buffer[end_idx + end_tag.len()..].to_string();

                return StreamingExtractResult {
                    found: true,
                    text_before,
                    call: Some(call),
                    remaining,
                };
            }
        }
    }

    StreamingExtractResult {
        found: false,
        text_before: None,
        call: None,
        remaining: buffer.to_string(),
    }
}

fn try_extract_llama(buffer: &str) -> StreamingExtractResult {
    let marker = "<|python_tag|>";

    if let Some(start_idx) = buffer.find(marker) {
        let json_start = start_idx + marker.len();
        let json_part = &buffer[json_start..];

        // Try to extract complete JSON object
        if let Some(json_str) = extract_json_object(json_part, 0) {
            if let Ok(parsed) = serde_json::from_str::<ToolCallJson>(&json_str) {
                let id = format!("call_{}", Uuid::new_v4().simple());
                let call = DetectedToolCall {
                    id,
                    name: parsed.name,
                    arguments: serde_json::to_string(&parsed.arguments).unwrap_or_default(),
                };

                let text_before = if start_idx > 0 {
                    Some(buffer[..start_idx].to_string())
                } else {
                    None
                };

                let remaining = buffer[json_start + json_str.len()..].to_string();

                return StreamingExtractResult {
                    found: true,
                    text_before,
                    call: Some(call),
                    remaining,
                };
            }
        }
    }

    // Fallback to Qwen style
    try_extract_qwen(buffer)
}

fn try_extract_mistral(buffer: &str) -> StreamingExtractResult {
    let marker = "[TOOL_CALLS]";

    if let Some(start_idx) = buffer.find(marker) {
        let after_marker = &buffer[start_idx + marker.len()..];

        // Look for JSON array
        if let Some(arr_start) = after_marker.find('[') {
            if let Some(arr_end) = find_matching_bracket(after_marker, arr_start) {
                let json_arr = &after_marker[arr_start..=arr_end];

                if let Ok(parsed) = serde_json::from_str::<Vec<ToolCallJson>>(json_arr) {
                    if let Some(first) = parsed.into_iter().next() {
                        let id = format!("call_{}", Uuid::new_v4().simple());
                        let call = DetectedToolCall {
                            id,
                            name: first.name,
                            arguments: serde_json::to_string(&first.arguments).unwrap_or_default(),
                        };

                        let text_before = if start_idx > 0 {
                            Some(buffer[..start_idx].to_string())
                        } else {
                            None
                        };

                        let remaining = after_marker[arr_end + 1..].to_string();

                        return StreamingExtractResult {
                            found: true,
                            text_before,
                            call: Some(call),
                            remaining,
                        };
                    }
                }
            }
        }
    }

    // Fallback to Qwen style
    try_extract_qwen(buffer)
}

/// Find the matching closing bracket for an opening bracket.
fn find_matching_bracket(s: &str, start: usize) -> Option<usize> {
    let bytes = s.as_bytes();
    let open_char = bytes.get(start)?;
    let close_char = match open_char {
        b'[' => b']',
        b'{' => b'}',
        _ => return None,
    };

    let mut depth = 0;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if escape_next {
            escape_next = false;
            continue;
        }

        if b == b'\\' && in_string {
            escape_next = true;
            continue;
        }

        if b == b'"' {
            in_string = !in_string;
            continue;
        }

        if in_string {
            continue;
        }

        if b == *open_char {
            depth += 1;
        } else if b == close_char {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }

    None
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 3: DEEP JSON PARSING
// ═══════════════════════════════════════════════════════════════════════════════

/// Extract a complete JSON object from a string starting at the given position.
///
/// This handles arbitrarily nested JSON, unlike regex-based approaches.
///
/// # Arguments
/// * `s` - The string to extract from
/// * `start` - The position to start searching from
///
/// # Returns
/// The extracted JSON string if a complete object is found
#[must_use]
pub fn extract_json_object(s: &str, start: usize) -> Option<String> {
    let bytes = s.as_bytes();

    // Find the opening brace
    let obj_start = bytes.iter().skip(start).position(|&b| b == b'{')? + start;

    // Find the matching closing brace
    let obj_end = find_matching_bracket(s, obj_start)?;

    Some(s[obj_start..=obj_end].to_string())
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 3: PARALLEL TOOL CALLS ENFORCEMENT
// ═══════════════════════════════════════════════════════════════════════════════

/// Options for processing model output.
#[derive(Debug, Clone, Default)]
pub struct ProcessingOptions {
    /// Whether parallel tool calls are allowed.
    /// When false, only the first tool call is returned.
    pub parallel_tool_calls: bool,
    /// Tools available for validation (for strict mode).
    pub tools: Option<Vec<Tool>>,
}

/// Enforce parallel_tool_calls setting on detected calls.
///
/// When `parallel` is false, only the first tool call is returned.
///
/// # Arguments
/// * `calls` - The detected tool calls
/// * `parallel` - Whether parallel calls are allowed
///
/// # Returns
/// Filtered list of tool calls
#[must_use]
pub fn enforce_parallel_tool_calls(calls: Vec<DetectedToolCall>, parallel: bool) -> Vec<DetectedToolCall> {
    if parallel || calls.is_empty() {
        calls
    } else {
        vec![calls.into_iter().next().unwrap()]
    }
}

/// Process model output with additional options.
///
/// This is an extended version of `process_model_output` that supports
/// parallel_tool_calls enforcement.
///
/// # Arguments
/// * `output` - The raw model output text
/// * `model_family` - The model family for format selection
/// * `options` - Processing options
///
/// # Returns
/// A `ToolProcessingResult` with content, tool calls, and finish reason
#[must_use]
pub fn process_model_output_with_options(
    output: &str,
    model_family: ModelFamily,
    options: ProcessingOptions,
) -> ToolProcessingResult {
    let detected = detect_tool_calls(output, model_family);
    let detected = enforce_parallel_tool_calls(detected, options.parallel_tool_calls);
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

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 3: STRICT MODE SCHEMA VALIDATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Validate tool arguments against a JSON schema.
///
/// This performs basic validation including:
/// - Required field checking
/// - Type validation (string, integer, number, boolean, array, object)
/// - Enum value validation
///
/// # Arguments
/// * `arguments` - JSON string of the arguments
/// * `schema` - The JSON schema to validate against
///
/// # Returns
/// Ok(()) if valid, Err(message) if invalid
pub fn validate_tool_arguments(arguments: &str, schema: &serde_json::Value) -> Result<(), String> {
    let args: serde_json::Value = serde_json::from_str(arguments)
        .map_err(|e| format!("Invalid JSON: {e}"))?;

    validate_value_against_schema(&args, schema, "")
}

fn validate_value_against_schema(
    value: &serde_json::Value,
    schema: &serde_json::Value,
    path: &str,
) -> Result<(), String> {
    // Check type
    if let Some(expected_type) = schema.get("type").and_then(|t| t.as_str()) {
        let actual_type = match value {
            serde_json::Value::Null => "null",
            serde_json::Value::Bool(_) => "boolean",
            serde_json::Value::Number(n) => {
                if n.is_i64() || n.is_u64() {
                    "integer"
                } else {
                    "number"
                }
            }
            serde_json::Value::String(_) => "string",
            serde_json::Value::Array(_) => "array",
            serde_json::Value::Object(_) => "object",
        };

        // Allow integer for number type
        let type_matches = actual_type == expected_type
            || (expected_type == "number" && actual_type == "integer");

        if !type_matches {
            return Err(format!(
                "Type mismatch at {}: expected {expected_type}, got {actual_type}",
                if path.is_empty() { "root" } else { path }
            ));
        }
    }

    // Check enum
    if let Some(enum_values) = schema.get("enum").and_then(|e| e.as_array()) {
        if !enum_values.contains(value) {
            return Err(format!(
                "Invalid enum value at {}: {:?} not in {:?}",
                if path.is_empty() { "root" } else { path },
                value,
                enum_values
            ));
        }
    }

    // Check object properties and required fields
    if let serde_json::Value::Object(obj) = value {
        // Check required fields
        if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
            for req in required {
                if let Some(field_name) = req.as_str() {
                    if !obj.contains_key(field_name) {
                        return Err(format!("Missing required field: {field_name}"));
                    }
                }
            }
        }

        // Validate each property
        if let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) {
            for (key, prop_value) in obj {
                if let Some(prop_schema) = properties.get(key) {
                    let prop_path = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{path}.{key}")
                    };
                    validate_value_against_schema(prop_value, prop_schema, &prop_path)?;
                }
            }
        }
    }

    // Check array items
    if let serde_json::Value::Array(arr) = value {
        if let Some(items_schema) = schema.get("items") {
            for (i, item) in arr.iter().enumerate() {
                let item_path = format!("{path}[{i}]");
                validate_value_against_schema(item, items_schema, &item_path)?;
            }
        }
    }

    Ok(())
}

/// Result of processing with strict mode validation.
#[derive(Debug, Clone)]
pub struct ToolValidationResult {
    /// The standard processing result.
    pub result: ToolProcessingResult,
    /// Validation errors (tool name -> error message).
    pub validation_errors: Vec<(String, String)>,
}

/// Process model output with strict mode validation.
///
/// # Arguments
/// * `output` - The raw model output text
/// * `model_family` - The model family for format selection
/// * `tools` - Available tools (for schema lookup)
///
/// # Returns
/// Processing result with validation errors
#[must_use]
pub fn process_model_output_with_validation(
    output: &str,
    model_family: ModelFamily,
    tools: &[Tool],
) -> ToolValidationResult {
    let detected = detect_tool_calls(output, model_family);
    let content = extract_text_content(output, model_family);

    let mut validation_errors = Vec::new();

    // Validate each tool call against its schema if strict mode is enabled
    for call in &detected {
        if let Some(tool) = tools.iter().find(|t| t.function.name == call.name) {
            if tool.function.strict == Some(true) {
                if let Some(schema) = &tool.function.parameters {
                    if let Err(e) = validate_tool_arguments(&call.arguments, schema) {
                        validation_errors.push((call.name.clone(), e));
                    }
                }
            }
        }
    }

    let tool_calls: Vec<ToolCall> = detected.iter().map(DetectedToolCall::to_tool_call).collect();

    let finish_reason = if tool_calls.is_empty() {
        "stop".to_string()
    } else {
        "tool_calls".to_string()
    };

    ToolValidationResult {
        result: ToolProcessingResult {
            content,
            tool_calls,
            finish_reason,
        },
        validation_errors,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 3: UNKNOWN TOOL VALIDATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of validating detected tool calls.
#[derive(Debug, Clone)]
pub struct DetectedCallsValidation {
    /// Tool calls that passed validation.
    pub valid_calls: Vec<DetectedToolCall>,
    /// Names of tools that were not found in the tools list.
    pub unknown_tools: Vec<String>,
}

/// Validate detected tool calls against the available tools list.
///
/// Unknown tools are reported but still returned for client handling.
///
/// # Arguments
/// * `detected` - The detected tool calls
/// * `tools` - The available tools
///
/// # Returns
/// Validation result with unknown tool names
#[must_use]
pub fn validate_detected_calls(detected: &[DetectedToolCall], tools: &[Tool]) -> DetectedCallsValidation {
    let mut unknown_tools = Vec::new();

    for call in detected {
        if !validate_tool_exists(&call.name, tools) {
            unknown_tools.push(call.name.clone());
        }
    }

    DetectedCallsValidation {
        valid_calls: detected.to_vec(),
        unknown_tools,
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

    // ═══════════════════════════════════════════════════════════════════════════
    // PHASE 3 TDD TESTS - RED PHASE
    // These tests specify Phase 3 behavior. They should FAIL until implemented.
    // ═══════════════════════════════════════════════════════════════════════════

    mod phase3_streaming {
        use super::*;

        /// Streaming buffer should detect when we're potentially inside a tool call marker.
        /// This is needed to avoid streaming partial tool call content to the client.
        #[test]
        fn test_buffer_detects_potential_tool_start_qwen() {
            // "<tool" could be start of "<tool_call>"
            assert!(buffer_might_contain_tool_start("<tool", ModelFamily::Qwen));
            assert!(buffer_might_contain_tool_start("<tool_", ModelFamily::Qwen));
            assert!(buffer_might_contain_tool_start("<tool_call", ModelFamily::Qwen));

            // Regular text should not trigger buffering
            assert!(!buffer_might_contain_tool_start("hello world", ModelFamily::Qwen));
            assert!(!buffer_might_contain_tool_start("the tool is ready", ModelFamily::Qwen));
        }

        #[test]
        fn test_buffer_detects_potential_tool_start_llama() {
            assert!(buffer_might_contain_tool_start("<|python", ModelFamily::Llama));
            assert!(buffer_might_contain_tool_start("<|python_tag", ModelFamily::Llama));
            assert!(!buffer_might_contain_tool_start("hello", ModelFamily::Llama));
        }

        #[test]
        fn test_buffer_detects_potential_tool_start_mistral() {
            assert!(buffer_might_contain_tool_start("[TOOL", ModelFamily::Mistral));
            assert!(buffer_might_contain_tool_start("[TOOL_CALLS", ModelFamily::Mistral));
            assert!(!buffer_might_contain_tool_start("hello", ModelFamily::Mistral));
        }

        /// Should extract a complete tool call from a buffer, returning remaining content.
        #[test]
        fn test_try_extract_complete_tool_call_qwen() {
            let buffer = r#"Some text<tool_call>
{"name": "test", "arguments": {"key": "value"}}
</tool_call>More text"#;

            let result = try_extract_complete_tool_call(buffer, ModelFamily::Qwen);

            assert!(result.found);
            assert_eq!(result.text_before, Some("Some text".to_string()));
            assert_eq!(result.call.as_ref().unwrap().name, "test");
            assert_eq!(result.remaining, "More text");
        }

        #[test]
        fn test_try_extract_incomplete_tool_call() {
            // Incomplete - no closing tag yet
            let buffer = r#"<tool_call>
{"name": "test", "arguments": {"key": "value"}}"#;

            let result = try_extract_complete_tool_call(buffer, ModelFamily::Qwen);

            assert!(!result.found);
            assert!(result.call.is_none());
        }

        #[test]
        fn test_definitely_not_tool_call() {
            // If we've buffered enough and it's clearly not a tool marker
            assert!(definitely_not_tool_call("<tooltip>hover</tooltip>", ModelFamily::Qwen));
            assert!(!definitely_not_tool_call("<tool_call>", ModelFamily::Qwen));
        }
    }

    mod phase3_parallel_tool_calls {
        use super::*;

        /// When parallel_tool_calls is false, only the first tool call should be returned.
        #[test]
        fn test_enforce_single_tool_call() {
            let calls = vec![
                DetectedToolCall {
                    id: "call_1".to_string(),
                    name: "tool_a".to_string(),
                    arguments: "{}".to_string(),
                },
                DetectedToolCall {
                    id: "call_2".to_string(),
                    name: "tool_b".to_string(),
                    arguments: "{}".to_string(),
                },
            ];

            // When parallel_tool_calls = false, should return only first call
            let enforced = enforce_parallel_tool_calls(calls.clone(), false);
            assert_eq!(enforced.len(), 1);
            assert_eq!(enforced[0].name, "tool_a");

            // When parallel_tool_calls = true (or default), return all
            let all = enforce_parallel_tool_calls(calls, true);
            assert_eq!(all.len(), 2);
        }

        /// Processing should respect parallel_tool_calls setting.
        #[test]
        fn test_process_model_output_respects_parallel() {
            let output = r#"<tool_call>
{"name": "tool_a", "arguments": {}}
</tool_call>
<tool_call>
{"name": "tool_b", "arguments": {}}
</tool_call>"#;

            // With parallel disabled, should only get first tool
            let result = process_model_output_with_options(
                output,
                ModelFamily::Qwen,
                ProcessingOptions { parallel_tool_calls: false, ..Default::default() }
            );
            assert_eq!(result.tool_calls.len(), 1);
            assert_eq!(result.tool_calls[0].function.name, "tool_a");
        }
    }

    mod phase3_strict_mode {
        use super::*;

        /// Strict mode should validate tool arguments against the JSON schema.
        #[test]
        fn test_validate_tool_arguments_valid() {
            let schema = json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            });

            let arguments = r#"{"location": "Seattle", "units": "celsius"}"#;
            let result = validate_tool_arguments(arguments, &schema);

            assert!(result.is_ok());
        }

        #[test]
        fn test_validate_tool_arguments_missing_required() {
            let schema = json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            });

            let arguments = r#"{}"#;  // Missing required "location"
            let result = validate_tool_arguments(arguments, &schema);

            assert!(result.is_err());
            assert!(result.unwrap_err().contains("location"));
        }

        #[test]
        fn test_validate_tool_arguments_wrong_type() {
            let schema = json!({
                "type": "object",
                "properties": {
                    "count": {"type": "integer"}
                }
            });

            let arguments = r#"{"count": "not a number"}"#;
            let result = validate_tool_arguments(arguments, &schema);

            assert!(result.is_err());
        }

        #[test]
        fn test_validate_tool_arguments_invalid_enum() {
            let schema = json!({
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["active", "inactive"]}
                }
            });

            let arguments = r#"{"status": "unknown"}"#;
            let result = validate_tool_arguments(arguments, &schema);

            assert!(result.is_err());
        }

        /// When strict=true on a tool, arguments should be validated.
        #[test]
        fn test_process_validates_strict_tools() {
            let tool = Tool {
                tool_type: "function".to_string(),
                function: crate::openai::FunctionDefinition {
                    name: "get_weather".to_string(),
                    description: Some("Get weather".to_string()),
                    parameters: Some(json!({
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    })),
                    strict: Some(true),  // Strict mode enabled
                },
            };

            let output = r#"<tool_call>
{"name": "get_weather", "arguments": {}}
</tool_call>"#;  // Missing required location

            let result = process_model_output_with_validation(
                output,
                ModelFamily::Qwen,
                &[tool],
            );

            // Should report validation errors
            assert!(!result.validation_errors.is_empty());
        }
    }

    mod phase3_deep_json_parsing {
        use super::*;

        /// Current regex fails with deeply nested JSON. This test ensures proper parsing.
        #[test]
        fn test_extract_deeply_nested_json() {
            // 3 levels of nesting - current regex fails
            let json_str = r#"{"name": "test", "arguments": {"outer": {"middle": {"inner": "value"}}}}"#;

            let result = extract_json_object(json_str, 0);

            assert!(result.is_some());
            let extracted = result.unwrap();
            assert!(extracted.contains("inner"));
            assert!(extracted.contains("value"));
        }

        #[test]
        fn test_extract_json_with_arrays() {
            let json_str = r#"{"items": [{"a": 1}, {"b": [1, 2, {"c": 3}]}]}"#;

            let result = extract_json_object(json_str, 0);

            assert!(result.is_some());
            let parsed: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
            assert!(parsed.get("items").is_some());
        }

        #[test]
        fn test_extract_json_with_escaped_quotes() {
            let json_str = r#"{"message": "He said \"hello\""}"#;

            let result = extract_json_object(json_str, 0);

            assert!(result.is_some());
        }

        #[test]
        fn test_detect_deeply_nested_tool_call() {
            // This should work with proper JSON parsing (currently fails with regex)
            let output = r#"<tool_call>
{"name": "complex_tool", "arguments": {"data": {"level1": {"level2": {"level3": "deep"}}}}}
</tool_call>"#;

            let calls = detect_tool_calls(output, ModelFamily::Qwen);

            assert_eq!(calls.len(), 1);
            let args: serde_json::Value = serde_json::from_str(&calls[0].arguments).unwrap();
            assert!(args["data"]["level1"]["level2"]["level3"].as_str() == Some("deep"));
        }

        #[test]
        fn test_llama_deeply_nested() {
            let output = r#"<|python_tag|>{"name": "test", "arguments": {"a": {"b": {"c": {"d": "deep"}}}}}"#;

            let calls = detect_tool_calls(output, ModelFamily::Llama);

            assert_eq!(calls.len(), 1);
            let args: serde_json::Value = serde_json::from_str(&calls[0].arguments).unwrap();
            assert_eq!(args["a"]["b"]["c"]["d"], "deep");
        }
    }

    mod phase3_unknown_tool_logging {
        use super::*;

        /// Detected tool calls should be validated against available tools.
        #[test]
        fn test_validate_detected_calls_known_tools() {
            let tools = vec![
                make_tool("tool_a", "A", json!({})),
                make_tool("tool_b", "B", json!({})),
            ];

            let detected = vec![
                DetectedToolCall {
                    id: "call_1".to_string(),
                    name: "tool_a".to_string(),
                    arguments: "{}".to_string(),
                },
            ];

            let result = validate_detected_calls(&detected, &tools);

            assert!(result.unknown_tools.is_empty());
            assert_eq!(result.valid_calls.len(), 1);
        }

        #[test]
        fn test_validate_detected_calls_unknown_tools() {
            let tools = vec![make_tool("tool_a", "A", json!({}))];

            let detected = vec![
                DetectedToolCall {
                    id: "call_1".to_string(),
                    name: "unknown_tool".to_string(),
                    arguments: "{}".to_string(),
                },
            ];

            let result = validate_detected_calls(&detected, &tools);

            // Unknown tool should be reported but still returned
            assert_eq!(result.unknown_tools.len(), 1);
            assert_eq!(result.unknown_tools[0], "unknown_tool");
            assert_eq!(result.valid_calls.len(), 1);  // Still returned for client handling
        }
    }
}
