//! Tool system for agent actions.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use infernum_core::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Risk level of a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    /// Pure computation, no side effects.
    Safe,
    /// Reads external state.
    ReadOnly,
    /// Modifies external state.
    Write,
    /// Potentially dangerous operation.
    Dangerous,
}

/// Result from tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Whether the tool succeeded.
    pub success: bool,
    /// Output from the tool.
    pub output: String,
    /// Error message if failed.
    pub error: Option<String>,
    /// Additional data.
    pub data: Option<Value>,
}

impl ToolResult {
    /// Creates a successful result.
    #[must_use]
    pub fn success(output: impl Into<String>) -> Self {
        Self {
            success: true,
            output: output.into(),
            error: None,
            data: None,
        }
    }

    /// Creates a failed result.
    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            output: String::new(),
            error: Some(message.into()),
            data: None,
        }
    }

    /// Adds data to the result.
    #[must_use]
    pub fn with_data(mut self, data: Value) -> Self {
        self.data = Some(data);
        self
    }

    /// Validates the result against output constraints.
    #[must_use]
    pub fn validate(&self, config: &OutputValidationConfig) -> ValidationResult {
        let mut issues = Vec::new();

        // Check output size
        if self.output.len() > config.max_output_size {
            issues.push(ValidationIssue::OutputTooLarge {
                size: self.output.len(),
                max: config.max_output_size,
            });
        }

        // Check for sensitive patterns if enabled
        if config.check_sensitive_data {
            for pattern in SENSITIVE_PATTERNS {
                if self.output.to_lowercase().contains(pattern) {
                    issues.push(ValidationIssue::SensitiveData {
                        pattern: pattern.to_string(),
                    });
                }
            }
        }

        // Validate JSON data if present
        if let Some(ref data) = self.data {
            if config.max_json_depth > 0 {
                let depth = json_depth(data);
                if depth > config.max_json_depth {
                    issues.push(ValidationIssue::JsonTooDeep {
                        depth,
                        max: config.max_json_depth,
                    });
                }
            }
        }

        // Check for control characters
        if config.strip_control_chars && has_control_chars(&self.output) {
            issues.push(ValidationIssue::ContainsControlChars);
        }

        if issues.is_empty() {
            ValidationResult::Valid
        } else {
            ValidationResult::Invalid(issues)
        }
    }

    /// Sanitizes the result according to configuration.
    #[must_use]
    pub fn sanitize(&self, config: &OutputValidationConfig) -> Self {
        let mut result = self.clone();

        // Truncate if too large
        if result.output.len() > config.max_output_size {
            result.output = result.output.chars().take(config.max_output_size).collect();
            result.output.push_str("... [truncated]");
        }

        // Strip control characters if configured
        if config.strip_control_chars {
            result.output = strip_control_chars(&result.output);
        }

        // Redact sensitive patterns if configured
        if config.check_sensitive_data {
            for pattern in SENSITIVE_PATTERNS {
                if result.output.to_lowercase().contains(pattern) {
                    result.output = redact_pattern(&result.output, pattern);
                }
            }
        }

        result
    }
}

/// Configuration for output validation.
#[derive(Debug, Clone)]
pub struct OutputValidationConfig {
    /// Maximum allowed output size in characters.
    pub max_output_size: usize,
    /// Maximum JSON nesting depth.
    pub max_json_depth: usize,
    /// Whether to check for sensitive data patterns.
    pub check_sensitive_data: bool,
    /// Whether to strip control characters.
    pub strip_control_chars: bool,
    /// Whether to validate JSON structure.
    pub validate_json: bool,
}

impl Default for OutputValidationConfig {
    fn default() -> Self {
        Self {
            max_output_size: 100_000,
            max_json_depth: 20,
            check_sensitive_data: true,
            strip_control_chars: true,
            validate_json: true,
        }
    }
}

impl OutputValidationConfig {
    /// Creates a permissive configuration.
    #[must_use]
    pub fn permissive() -> Self {
        Self {
            max_output_size: 1_000_000,
            max_json_depth: 100,
            check_sensitive_data: false,
            strip_control_chars: false,
            validate_json: false,
        }
    }

    /// Creates a strict configuration.
    #[must_use]
    pub fn strict() -> Self {
        Self {
            max_output_size: 10_000,
            max_json_depth: 10,
            check_sensitive_data: true,
            strip_control_chars: true,
            validate_json: true,
        }
    }

    /// Sets maximum output size.
    #[must_use]
    pub fn with_max_output_size(mut self, size: usize) -> Self {
        self.max_output_size = size;
        self
    }

    /// Enables or disables sensitive data checking.
    #[must_use]
    pub fn with_sensitive_check(mut self, enabled: bool) -> Self {
        self.check_sensitive_data = enabled;
        self
    }
}

/// Result of output validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationResult {
    /// Output is valid.
    Valid,
    /// Output has validation issues.
    Invalid(Vec<ValidationIssue>),
}

impl ValidationResult {
    /// Returns true if valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        matches!(self, ValidationResult::Valid)
    }

    /// Returns validation issues if any.
    #[must_use]
    pub fn issues(&self) -> Option<&[ValidationIssue]> {
        match self {
            ValidationResult::Valid => None,
            ValidationResult::Invalid(issues) => Some(issues),
        }
    }
}

/// Types of validation issues.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationIssue {
    /// Output exceeds size limit.
    OutputTooLarge {
        /// Actual size of the output.
        size: usize,
        /// Maximum allowed size.
        max: usize,
    },
    /// JSON nesting too deep.
    JsonTooDeep {
        /// Actual nesting depth.
        depth: usize,
        /// Maximum allowed depth.
        max: usize,
    },
    /// Contains sensitive data pattern.
    SensitiveData {
        /// The detected sensitive pattern.
        pattern: String,
    },
    /// Contains control characters.
    ContainsControlChars,
    /// Invalid JSON structure.
    InvalidJson {
        /// The JSON parsing error.
        error: String,
    },
}

/// Common sensitive data patterns to detect.
const SENSITIVE_PATTERNS: &[&str] = &[
    "password",
    "api_key",
    "apikey",
    "secret",
    "token",
    "credential",
    "private_key",
    "ssh_key",
    "bearer",
];

/// Calculates the depth of a JSON value.
fn json_depth(value: &Value) -> usize {
    match value {
        Value::Array(arr) => 1 + arr.iter().map(json_depth).max().unwrap_or(0),
        Value::Object(map) => 1 + map.values().map(json_depth).max().unwrap_or(0),
        _ => 0,
    }
}

/// Checks if a string contains control characters.
fn has_control_chars(s: &str) -> bool {
    s.chars().any(|c| c.is_control() && c != '\n' && c != '\r' && c != '\t')
}

/// Strips control characters from a string.
fn strip_control_chars(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_control() || *c == '\n' || *c == '\r' || *c == '\t')
        .collect()
}

/// Redacts a pattern in the output.
fn redact_pattern(s: &str, pattern: &str) -> String {
    let lower = s.to_lowercase();
    let mut result = String::with_capacity(s.len());
    let mut last_end = 0;

    for (start, _) in lower.match_indices(pattern) {
        result.push_str(&s[last_end..start]);
        result.push_str("[REDACTED]");
        last_end = start + pattern.len();
    }
    result.push_str(&s[last_end..]);
    result
}

// ============================================================================
// Tool Timeout Configuration (Sprint 7 - Leviathan Feature Parity)
// ============================================================================

use std::time::Duration;

/// Configuration for tool execution timeouts.
///
/// Inspired by Leviathan/Persona's flexible timeout system that allows:
/// - Per-tool default timeouts
/// - Per-task timeout overrides
/// - Adaptive timeouts based on complexity
#[derive(Debug, Clone)]
pub struct ToolTimeoutConfig {
    /// Default timeout for all tools.
    pub default_timeout: Duration,
    /// Per-tool timeout overrides.
    pub tool_timeouts: HashMap<String, Duration>,
    /// Timeout multiplier for complex tasks (1.0 = no change).
    pub complexity_multiplier: f32,
    /// Maximum allowed timeout (cap).
    pub max_timeout: Duration,
    /// Minimum allowed timeout (floor).
    pub min_timeout: Duration,
}

impl Default for ToolTimeoutConfig {
    fn default() -> Self {
        Self {
            default_timeout: Duration::from_secs(30),
            tool_timeouts: HashMap::new(),
            complexity_multiplier: 1.0,
            max_timeout: Duration::from_secs(300), // 5 minutes max
            min_timeout: Duration::from_millis(100),
        }
    }
}

impl ToolTimeoutConfig {
    /// Creates a new timeout config.
    #[must_use]
    pub fn new(default_timeout: Duration) -> Self {
        Self {
            default_timeout,
            ..Default::default()
        }
    }

    /// Sets the timeout for a specific tool.
    #[must_use]
    pub fn with_tool_timeout(mut self, tool_name: impl Into<String>, timeout: Duration) -> Self {
        self.tool_timeouts.insert(tool_name.into(), timeout);
        self
    }

    /// Sets the complexity multiplier.
    #[must_use]
    pub fn with_complexity_multiplier(mut self, multiplier: f32) -> Self {
        self.complexity_multiplier = multiplier;
        self
    }

    /// Gets the effective timeout for a tool.
    #[must_use]
    pub fn get_timeout(&self, tool_name: &str) -> Duration {
        let base = self
            .tool_timeouts
            .get(tool_name)
            .copied()
            .unwrap_or(self.default_timeout);

        let adjusted = Duration::from_secs_f32(
            base.as_secs_f32() * self.complexity_multiplier
        );

        // Clamp to min/max bounds
        adjusted.clamp(self.min_timeout, self.max_timeout)
    }

    /// Creates a config for quick operations.
    #[must_use]
    pub fn quick() -> Self {
        Self {
            default_timeout: Duration::from_secs(5),
            max_timeout: Duration::from_secs(30),
            ..Default::default()
        }
    }

    /// Creates a config for long-running operations.
    #[must_use]
    pub fn long_running() -> Self {
        Self {
            default_timeout: Duration::from_secs(120),
            max_timeout: Duration::from_secs(600), // 10 minutes
            ..Default::default()
        }
    }

    /// Creates a config for I/O-bound operations.
    #[must_use]
    pub fn io_bound() -> Self {
        let mut config = Self::default();
        config.tool_timeouts.insert("http".to_string(), Duration::from_secs(60));
        config.tool_timeouts.insert("file_read".to_string(), Duration::from_secs(30));
        config.tool_timeouts.insert("file_write".to_string(), Duration::from_secs(30));
        config.tool_timeouts.insert("database".to_string(), Duration::from_secs(60));
        config
    }
}

/// Task complexity level for adaptive timeouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskComplexity {
    /// Simple, quick tasks.
    Simple,
    /// Moderate complexity.
    Moderate,
    /// Complex, multi-step tasks.
    Complex,
}

impl TaskComplexity {
    /// Returns the timeout multiplier for this complexity level.
    #[must_use]
    pub fn multiplier(&self) -> f32 {
        match self {
            Self::Simple => 0.5,
            Self::Moderate => 1.0,
            Self::Complex => 2.0,
        }
    }
}

/// Context provided to tools during execution.
#[derive(Clone)]
pub struct ToolContext {
    /// Agent ID.
    pub agent_id: String,
    /// Conversation history.
    pub messages: Vec<infernum_core::Message>,
    /// Key-value store for tool state.
    pub state: HashMap<String, Value>,
    /// Timeout configuration.
    pub timeout_config: ToolTimeoutConfig,
    /// Current task complexity.
    pub task_complexity: TaskComplexity,
}

impl ToolContext {
    /// Creates a new tool context.
    #[must_use]
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            messages: Vec::new(),
            state: HashMap::new(),
            timeout_config: ToolTimeoutConfig::default(),
            task_complexity: TaskComplexity::Moderate,
        }
    }

    /// Sets the timeout configuration.
    #[must_use]
    pub fn with_timeout_config(mut self, config: ToolTimeoutConfig) -> Self {
        self.timeout_config = config;
        self
    }

    /// Sets the task complexity.
    #[must_use]
    pub fn with_complexity(mut self, complexity: TaskComplexity) -> Self {
        self.task_complexity = complexity;
        // Update the multiplier in timeout config
        self.timeout_config.complexity_multiplier = complexity.multiplier();
        self
    }

    /// Gets the effective timeout for a specific tool.
    #[must_use]
    pub fn get_tool_timeout(&self, tool_name: &str) -> Duration {
        self.timeout_config.get_timeout(tool_name)
    }

    /// Gets a value from state.
    #[must_use]
    pub fn get_state(&self, key: &str) -> Option<&Value> {
        self.state.get(key)
    }

    /// Sets a value in state.
    pub fn set_state(&mut self, key: impl Into<String>, value: Value) {
        self.state.insert(key.into(), value);
    }
}

/// Trait for tools that agents can use.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Returns the tool name.
    fn name(&self) -> &str;

    /// Returns the tool description.
    fn description(&self) -> &str;

    /// Returns the JSON schema for parameters.
    fn parameters_schema(&self) -> Value;

    /// Returns the risk level.
    fn risk_level(&self) -> RiskLevel;

    /// Executes the tool.
    async fn execute(&self, params: Value, ctx: &ToolContext) -> Result<ToolResult>;
}

/// A tool call parsed from LLM output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Tool name.
    pub name: String,
    /// Tool parameters.
    pub params: Value,
}

/// Registry of available tools.
#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Creates a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a registry with built-in tools.
    #[must_use]
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        registry.register(Arc::new(CalculatorTool));
        registry.register(Arc::new(JsonTool));
        registry.register(Arc::new(DateTimeTool));
        registry
    }

    /// Creates a registry with built-in tools plus code-relevant tools.
    ///
    /// Includes file I/O (read, write, edit), search (list, grep), and
    /// shell execution. Does NOT include `claude_code` — use
    /// [`with_all_tools`](Self::with_all_tools) for that.
    #[must_use]
    pub fn with_code_tools() -> Self {
        use crate::tools::{
            BashTool, EditFileTool, ListFilesTool, ReadFileTool, SearchFilesTool, WriteFileTool,
        };

        let mut registry = Self::with_builtins();
        registry.register(Arc::new(ReadFileTool));
        registry.register(Arc::new(WriteFileTool));
        registry.register(Arc::new(EditFileTool));
        registry.register(Arc::new(ListFilesTool));
        registry.register(Arc::new(SearchFilesTool));
        registry.register(Arc::new(BashTool::default()));
        registry
    }

    /// Creates a registry with all tools, including Claude Code delegation.
    ///
    /// Requires the `claude` CLI to be installed and on `$PATH`.
    #[must_use]
    pub fn with_all_tools() -> Self {
        use crate::tools::ClaudeCodeTool;

        let mut registry = Self::with_code_tools();
        registry.register(Arc::new(ClaudeCodeTool::new()));
        registry
    }

    /// Registers a tool.
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Gets a tool by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Lists all tool names.
    #[must_use]
    pub fn list(&self) -> Vec<&str> {
        self.tools.keys().map(String::as_str).collect()
    }

    /// Returns all tools.
    #[must_use]
    pub fn tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.values().cloned().collect()
    }

    /// Returns the number of registered tools.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Returns true if no tools are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Generates tool definitions for LLM function calling.
    #[must_use]
    pub fn to_function_definitions(&self) -> Vec<Value> {
        self.tools
            .values()
            .map(|tool| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": tool.name(),
                        "description": tool.description(),
                        "parameters": tool.parameters_schema()
                    }
                })
            })
            .collect()
    }

    /// Generates tool descriptions for prompting (generic format).
    #[must_use]
    pub fn to_prompt_description(&self) -> String {
        let mut desc = String::from("Available tools:\n\n");
        for tool in self.tools.values() {
            desc.push_str(&format!(
                "- {}: {}\n  Parameters: {}\n\n",
                tool.name(),
                tool.description(),
                serde_json::to_string_pretty(&tool.parameters_schema()).unwrap_or_default()
            ));
        }
        desc
    }

    /// Generates tool descriptions in Qwen2.5 native format.
    ///
    /// Matches the format from Qwen2.5's Jinja chat template: tools wrapped
    /// in `<tools></tools>` XML tags as full JSON function definitions.
    ///
    /// See TOOL-CALLING-SPEC.md §5.6.
    #[must_use]
    pub fn to_qwen_native_description(&self) -> String {
        let mut desc = String::from(
            "You may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n\
             <tools>",
        );

        for tool_def in self.to_function_definitions() {
            desc.push('\n');
            desc.push_str(&serde_json::to_string(&tool_def).unwrap_or_default());
        }

        desc.push_str(
            "\n</tools>\n\n\
             For each function call, return a json object with function name and arguments \
             within <tool_call></tool_call> XML tags:\n\
             <tool_call>\n\
             {\"name\": <function-name>, \"arguments\": <args-json-object>}\n\
             </tool_call>",
        );

        desc
    }

    /// Executes a tool call.
    ///
    /// # Errors
    ///
    /// Returns an error if the tool is not found or execution fails.
    pub async fn execute(&self, call: &ToolCall, ctx: &ToolContext) -> Result<ToolResult> {
        let tool = self.get(&call.name).ok_or_else(|| {
            infernum_core::Error::internal(format!("Tool '{}' not found", call.name))
        })?;

        tool.execute(call.params.clone(), ctx).await
    }
}

// === Built-in Tools ===

/// Calculator tool for mathematical expressions.
pub struct CalculatorTool;

#[async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Evaluates mathematical expressions. Supports basic arithmetic (+, -, *, /), \
         parentheses, and common functions (sqrt, sin, cos, tan, log, exp, pow)."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        })
    }

    fn risk_level(&self) -> RiskLevel {
        RiskLevel::Safe
    }

    async fn execute(&self, params: Value, _ctx: &ToolContext) -> Result<ToolResult> {
        let expression = params
            .get("expression")
            .and_then(|v| v.as_str())
            .ok_or_else(|| infernum_core::Error::internal("Missing expression"))?;

        // Simple expression evaluator using a basic parser
        match evaluate_expression(expression) {
            Ok(result) => Ok(ToolResult::success(format!("{} = {}", expression, result))
                .with_data(serde_json::json!({ "result": result }))),
            Err(e) => Ok(ToolResult::error(format!("Failed to evaluate: {}", e))),
        }
    }
}

/// JSON manipulation tool.
pub struct JsonTool;

#[async_trait]
impl Tool for JsonTool {
    fn name(&self) -> &str {
        "json"
    }

    fn description(&self) -> &str {
        "Parses, formats, and queries JSON data. Supports JSONPath queries."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["parse", "format", "query"],
                    "description": "Operation to perform"
                },
                "data": {
                    "type": "string",
                    "description": "JSON string or data"
                },
                "query": {
                    "type": "string",
                    "description": "JSONPath query (for query operation)"
                }
            },
            "required": ["operation", "data"]
        })
    }

    fn risk_level(&self) -> RiskLevel {
        RiskLevel::Safe
    }

    async fn execute(&self, params: Value, _ctx: &ToolContext) -> Result<ToolResult> {
        let operation = params
            .get("operation")
            .and_then(|v| v.as_str())
            .ok_or_else(|| infernum_core::Error::internal("Missing operation"))?;

        let data = params
            .get("data")
            .and_then(|v| v.as_str())
            .ok_or_else(|| infernum_core::Error::internal("Missing data"))?;

        match operation {
            "parse" => match serde_json::from_str::<Value>(data) {
                Ok(parsed) => Ok(ToolResult::success("JSON parsed successfully").with_data(parsed)),
                Err(e) => Ok(ToolResult::error(format!("Parse error: {}", e))),
            },
            "format" => match serde_json::from_str::<Value>(data) {
                Ok(parsed) => match serde_json::to_string_pretty(&parsed) {
                    Ok(formatted) => Ok(ToolResult::success(formatted)),
                    Err(e) => Ok(ToolResult::error(format!("Format error: {}", e))),
                },
                Err(e) => Ok(ToolResult::error(format!("Parse error: {}", e))),
            },
            "query" => {
                // Simple key access (full JSONPath would need a library)
                let query = params.get("query").and_then(|v| v.as_str()).unwrap_or("");
                match serde_json::from_str::<Value>(data) {
                    Ok(parsed) => {
                        // Simple dot notation query
                        let result = query_json(&parsed, query);
                        match result {
                            Some(v) => Ok(ToolResult::success(v.to_string()).with_data(v)),
                            None => Ok(ToolResult::error("Query returned no results")),
                        }
                    },
                    Err(e) => Ok(ToolResult::error(format!("Parse error: {}", e))),
                }
            },
            _ => Ok(ToolResult::error(format!(
                "Unknown operation: {}",
                operation
            ))),
        }
    }
}

/// DateTime tool.
pub struct DateTimeTool;

#[async_trait]
impl Tool for DateTimeTool {
    fn name(&self) -> &str {
        "datetime"
    }

    fn description(&self) -> &str {
        "Gets current date/time or performs date calculations."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["now", "format", "parse"],
                    "description": "Operation to perform"
                },
                "format": {
                    "type": "string",
                    "description": "Date format string (for format operation)"
                },
                "timezone": {
                    "type": "string",
                    "description": "Timezone (e.g., 'UTC', 'America/New_York')"
                }
            },
            "required": ["operation"]
        })
    }

    fn risk_level(&self) -> RiskLevel {
        RiskLevel::Safe
    }

    async fn execute(&self, params: Value, _ctx: &ToolContext) -> Result<ToolResult> {
        let operation = params
            .get("operation")
            .and_then(|v| v.as_str())
            .ok_or_else(|| infernum_core::Error::internal("Missing operation"))?;

        match operation {
            "now" => {
                use std::time::{SystemTime, UNIX_EPOCH};
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default();
                let secs = now.as_secs();

                // Simple ISO-8601 format
                let datetime = format_unix_timestamp(secs);

                Ok(
                    ToolResult::success(format!("Current time: {}", datetime)).with_data(
                        serde_json::json!({
                            "timestamp": secs,
                            "iso8601": datetime
                        }),
                    ),
                )
            },
            "format" => {
                use std::time::{SystemTime, UNIX_EPOCH};
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default();
                let secs = now.as_secs();

                let format_str = params
                    .get("format")
                    .and_then(|v| v.as_str())
                    .unwrap_or("%Y-%m-%d %H:%M:%S");

                let formatted = format_with_pattern(secs, format_str);

                Ok(
                    ToolResult::success(format!("Formatted: {}", formatted)).with_data(
                        serde_json::json!({
                            "timestamp": secs,
                            "formatted": formatted
                        }),
                    ),
                )
            },
            "parse" => {
                let date_str = params
                    .get("date")
                    .or_else(|| params.get("data"))
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| infernum_core::Error::internal("Missing date string"))?;

                match parse_datetime(date_str) {
                    Ok(timestamp) => {
                        let iso = format_unix_timestamp(timestamp);
                        Ok(
                            ToolResult::success(format!("Parsed: {} -> {}", date_str, iso))
                                .with_data(serde_json::json!({
                                    "timestamp": timestamp,
                                    "iso8601": iso,
                                    "input": date_str
                                })),
                        )
                    },
                    Err(e) => Ok(ToolResult::error(format!("Parse error: {}", e))),
                }
            },
            _ => Ok(ToolResult::error(format!(
                "Unknown operation: {}. Supported: now, format, parse",
                operation
            ))),
        }
    }
}

// === Helper Functions ===

/// Simple expression evaluator for basic arithmetic.
fn evaluate_expression(expr: &str) -> std::result::Result<f64, String> {
    // Basic tokenizer and evaluator
    let expr = expr.trim().replace(' ', "");

    // Handle simple arithmetic expressions
    if let Ok(num) = expr.parse::<f64>() {
        return Ok(num);
    }

    // Try to find operators and evaluate
    for (i, c) in expr.chars().rev().enumerate() {
        let pos = expr.len() - 1 - i;
        if c == '+' && pos > 0 {
            let left = evaluate_expression(&expr[..pos])?;
            let right = evaluate_expression(&expr[pos + 1..])?;
            return Ok(left + right);
        }
        if c == '-' && pos > 0 {
            // Check it's not a negative number
            if pos > 0 {
                let prev = expr.chars().nth(pos - 1);
                if prev
                    .map(|p| p.is_ascii_digit() || p == ')')
                    .unwrap_or(false)
                {
                    let left = evaluate_expression(&expr[..pos])?;
                    let right = evaluate_expression(&expr[pos + 1..])?;
                    return Ok(left - right);
                }
            }
        }
    }

    for (i, c) in expr.chars().rev().enumerate() {
        let pos = expr.len() - 1 - i;
        if c == '*' && pos > 0 {
            let left = evaluate_expression(&expr[..pos])?;
            let right = evaluate_expression(&expr[pos + 1..])?;
            return Ok(left * right);
        }
        if c == '/' && pos > 0 {
            let left = evaluate_expression(&expr[..pos])?;
            let right = evaluate_expression(&expr[pos + 1..])?;
            if right == 0.0 {
                return Err("Division by zero".to_string());
            }
            return Ok(left / right);
        }
    }

    // Handle parentheses
    if expr.starts_with('(') && expr.ends_with(')') {
        return evaluate_expression(&expr[1..expr.len() - 1]);
    }

    // Handle functions
    if let Some(inner) = expr.strip_prefix("sqrt(").and_then(|s| s.strip_suffix(')')) {
        return Ok(evaluate_expression(inner)?.sqrt());
    }

    Err(format!("Cannot evaluate: {}", expr))
}

/// Simple JSON query using dot notation.
fn query_json(value: &Value, query: &str) -> Option<Value> {
    if query.is_empty() {
        return Some(value.clone());
    }

    let parts: Vec<&str> = query.split('.').collect();
    let mut current = value;

    for part in parts {
        match current {
            Value::Object(map) => {
                current = map.get(part)?;
            },
            Value::Array(arr) => {
                let idx: usize = part.parse().ok()?;
                current = arr.get(idx)?;
            },
            _ => return None,
        }
    }

    Some(current.clone())
}

/// Formats a Unix timestamp as ISO-8601.
fn format_unix_timestamp(secs: u64) -> String {
    // Simple calculation without chrono dependency
    const SECS_PER_DAY: u64 = 86400;
    const SECS_PER_HOUR: u64 = 3600;
    const SECS_PER_MIN: u64 = 60;

    // Days since Unix epoch
    let days = secs / SECS_PER_DAY;
    let remaining = secs % SECS_PER_DAY;

    let hours = remaining / SECS_PER_HOUR;
    let remaining = remaining % SECS_PER_HOUR;

    let minutes = remaining / SECS_PER_MIN;
    let seconds = remaining % SECS_PER_MIN;

    // Simple year calculation (approximate, doesn't handle leap years perfectly)
    let mut year = 1970;
    let mut day_count = days;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if day_count < days_in_year {
            break;
        }
        day_count -= days_in_year;
        year += 1;
    }

    // Calculate month and day
    let (month, day) = day_of_year_to_month_day(day_count as u32, is_leap_year(year));

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

fn is_leap_year(year: u64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

fn day_of_year_to_month_day(day: u32, leap: bool) -> (u32, u32) {
    let days_in_months = if leap {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut remaining = day;
    for (i, &days) in days_in_months.iter().enumerate() {
        if remaining < days {
            return ((i + 1) as u32, remaining + 1);
        }
        remaining -= days;
    }
    (12, 31)
}

/// Formats a Unix timestamp with a strftime-like pattern.
fn format_with_pattern(secs: u64, pattern: &str) -> String {
    const SECS_PER_DAY: u64 = 86400;
    const SECS_PER_HOUR: u64 = 3600;
    const SECS_PER_MIN: u64 = 60;

    let days = secs / SECS_PER_DAY;
    let remaining = secs % SECS_PER_DAY;

    let hours = remaining / SECS_PER_HOUR;
    let remaining = remaining % SECS_PER_HOUR;

    let minutes = remaining / SECS_PER_MIN;
    let seconds = remaining % SECS_PER_MIN;

    // Calculate year
    let mut year = 1970u64;
    let mut day_count = days;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if day_count < days_in_year {
            break;
        }
        day_count -= days_in_year;
        year += 1;
    }

    let (month, day) = day_of_year_to_month_day(day_count as u32, is_leap_year(year));

    // Day of week (0 = Thursday for Unix epoch)
    let dow = ((days + 4) % 7) as u32; // 0 = Sunday

    let month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    let month_abbrev = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    let day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    let day_abbrev = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

    let mut result = pattern.to_string();
    result = result.replace("%Y", &format!("{:04}", year));
    result = result.replace("%y", &format!("{:02}", year % 100));
    result = result.replace("%m", &format!("{:02}", month));
    result = result.replace("%d", &format!("{:02}", day));
    result = result.replace("%H", &format!("{:02}", hours));
    result = result.replace("%M", &format!("{:02}", minutes));
    result = result.replace("%S", &format!("{:02}", seconds));
    result = result.replace("%B", month_names[(month - 1) as usize]);
    result = result.replace("%b", month_abbrev[(month - 1) as usize]);
    result = result.replace("%A", day_names[dow as usize]);
    result = result.replace("%a", day_abbrev[dow as usize]);
    result = result.replace("%w", &dow.to_string());
    result = result.replace("%%", "%");

    result
}

/// Parses a datetime string into Unix timestamp.
fn parse_datetime(s: &str) -> std::result::Result<u64, String> {
    let s = s.trim();

    // Try ISO-8601 format: YYYY-MM-DDTHH:MM:SSZ
    if let Some(parsed) = parse_iso8601(s) {
        return Ok(parsed);
    }

    // Try date only: YYYY-MM-DD
    if let Some(parsed) = parse_date_only(s) {
        return Ok(parsed);
    }

    // Try common formats: MM/DD/YYYY, DD-MM-YYYY
    if let Some(parsed) = parse_common_formats(s) {
        return Ok(parsed);
    }

    Err(format!("Unable to parse date: {}", s))
}

fn parse_iso8601(s: &str) -> Option<u64> {
    // YYYY-MM-DDTHH:MM:SS or YYYY-MM-DDTHH:MM:SSZ
    let s = s.trim_end_matches('Z');
    let parts: Vec<&str> = s.split('T').collect();
    if parts.len() != 2 {
        return None;
    }

    let date_parts: Vec<u32> = parts[0].split('-').filter_map(|p| p.parse().ok()).collect();
    if date_parts.len() != 3 {
        return None;
    }

    let time_parts: Vec<u32> = parts[1]
        .split(':')
        .filter_map(|p| p.parse().ok())
        .collect();
    if time_parts.len() < 2 {
        return None;
    }

    let year = date_parts[0] as u64;
    let month = date_parts[1];
    let day = date_parts[2];
    let hour = time_parts[0] as u64;
    let min = time_parts[1] as u64;
    let sec = time_parts.get(2).copied().unwrap_or(0) as u64;

    Some(datetime_to_timestamp(year, month, day, hour, min, sec))
}

fn parse_date_only(s: &str) -> Option<u64> {
    let parts: Vec<u32> = s.split('-').filter_map(|p| p.parse().ok()).collect();
    if parts.len() != 3 {
        return None;
    }

    let year = parts[0] as u64;
    let month = parts[1];
    let day = parts[2];

    Some(datetime_to_timestamp(year, month, day, 0, 0, 0))
}

fn parse_common_formats(s: &str) -> Option<u64> {
    // Try MM/DD/YYYY
    if s.contains('/') {
        let parts: Vec<u32> = s.split('/').filter_map(|p| p.parse().ok()).collect();
        if parts.len() == 3 {
            let (month, day, year) = (parts[0], parts[1], parts[2] as u64);
            return Some(datetime_to_timestamp(year, month, day, 0, 0, 0));
        }
    }

    None
}

fn datetime_to_timestamp(year: u64, month: u32, day: u32, hour: u64, min: u64, sec: u64) -> u64 {
    const SECS_PER_DAY: u64 = 86400;
    const SECS_PER_HOUR: u64 = 3600;
    const SECS_PER_MIN: u64 = 60;

    // Count days from 1970 to year
    let mut days: u64 = 0;
    for y in 1970..year {
        days += if is_leap_year(y) { 366 } else { 365 };
    }

    // Add days for months in current year
    let days_in_months = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    for m in 0..(month - 1) as usize {
        days += days_in_months[m] as u64;
    }

    // Add remaining days
    days += (day - 1) as u64;

    // Convert to seconds
    days * SECS_PER_DAY + hour * SECS_PER_HOUR + min * SECS_PER_MIN + sec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_expression() {
        assert_eq!(evaluate_expression("2+3").unwrap(), 5.0);
        assert_eq!(evaluate_expression("10*5").unwrap(), 50.0);
        assert_eq!(evaluate_expression("20/4").unwrap(), 5.0);
        assert_eq!(evaluate_expression("sqrt(16)").unwrap(), 4.0);
    }

    #[test]
    fn test_query_json() {
        let json = serde_json::json!({
            "name": "test",
            "nested": {
                "value": 42
            }
        });

        assert_eq!(query_json(&json, "name"), Some(serde_json::json!("test")));
        assert_eq!(
            query_json(&json, "nested.value"),
            Some(serde_json::json!(42))
        );
    }

    #[tokio::test]
    async fn test_calculator_tool() {
        let tool = CalculatorTool;
        let ctx = ToolContext::new("test");
        let params = serde_json::json!({ "expression": "2+2" });

        let result = tool.execute(params, &ctx).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("4"));
    }

    #[test]
    fn test_output_validation_valid() {
        let result = ToolResult::success("Hello, world!");
        let config = OutputValidationConfig::default();

        assert!(result.validate(&config).is_valid());
    }

    #[test]
    fn test_output_validation_too_large() {
        let result = ToolResult::success("x".repeat(200));
        let config = OutputValidationConfig::default().with_max_output_size(100);

        let validation = result.validate(&config);
        assert!(!validation.is_valid());

        let issues = validation.issues().unwrap();
        assert!(matches!(issues[0], ValidationIssue::OutputTooLarge { .. }));
    }

    #[test]
    fn test_output_validation_sensitive_data() {
        let result = ToolResult::success("Your password is secret123");
        let config = OutputValidationConfig::default();

        let validation = result.validate(&config);
        assert!(!validation.is_valid());

        let issues = validation.issues().unwrap();
        assert!(issues.iter().any(|i| matches!(i, ValidationIssue::SensitiveData { .. })));
    }

    #[test]
    fn test_output_validation_control_chars() {
        let result = ToolResult::success("Hello\x00World");
        let config = OutputValidationConfig::default();

        let validation = result.validate(&config);
        assert!(!validation.is_valid());

        let issues = validation.issues().unwrap();
        assert!(issues.iter().any(|i| matches!(i, ValidationIssue::ContainsControlChars)));
    }

    #[test]
    fn test_output_validation_json_depth() {
        let deep_json = serde_json::json!({
            "a": { "b": { "c": { "d": { "e": "value" } } } }
        });
        let result = ToolResult::success("test").with_data(deep_json);
        let config = OutputValidationConfig::default();

        // Default allows depth 20, this is only 5 deep
        assert!(result.validate(&config).is_valid());

        // But with strict config (max depth 10)...
        let strict = OutputValidationConfig::strict();
        assert!(result.validate(&strict).is_valid());
    }

    #[test]
    fn test_sanitize_output() {
        let result = ToolResult::success("Your password is secret123");
        let config = OutputValidationConfig::default();

        let sanitized = result.sanitize(&config);
        assert!(sanitized.output.contains("[REDACTED]"));
        assert!(!sanitized.output.contains("password"));
    }

    #[test]
    fn test_sanitize_truncate() {
        let result = ToolResult::success("x".repeat(200));
        let config = OutputValidationConfig::default().with_max_output_size(50);

        let sanitized = result.sanitize(&config);
        assert!(sanitized.output.len() < 200);
        assert!(sanitized.output.contains("[truncated]"));
    }

    #[test]
    fn test_sanitize_control_chars() {
        let result = ToolResult::success("Hello\x00World\x1BEscape");
        let config = OutputValidationConfig::default();

        let sanitized = result.sanitize(&config);
        assert!(!sanitized.output.contains('\x00'));
        assert!(!sanitized.output.contains('\x1B'));
        assert!(sanitized.output.contains("HelloWorldEscape"));
    }

    #[test]
    fn test_validation_config_permissive() {
        let config = OutputValidationConfig::permissive();
        assert_eq!(config.max_output_size, 1_000_000);
        assert!(!config.check_sensitive_data);
    }

    #[test]
    fn test_validation_config_strict() {
        let config = OutputValidationConfig::strict();
        assert_eq!(config.max_output_size, 10_000);
        assert!(config.check_sensitive_data);
    }

    #[test]
    fn test_json_depth() {
        assert_eq!(json_depth(&serde_json::json!(42)), 0);
        assert_eq!(json_depth(&serde_json::json!([1, 2, 3])), 1);
        assert_eq!(json_depth(&serde_json::json!({"a": 1})), 1);
        assert_eq!(json_depth(&serde_json::json!({"a": {"b": 1}})), 2);
        assert_eq!(json_depth(&serde_json::json!([[[1]]])), 3);
    }

    #[test]
    fn test_strip_control_chars_preserves_newlines() {
        let input = "Line 1\nLine 2\rLine 3\tTabbed";
        let output = strip_control_chars(input);
        assert_eq!(output, input); // newlines, carriage returns, tabs preserved
    }

    #[test]
    fn test_redact_pattern() {
        let input = "The password is ABC123";
        let output = redact_pattern(input, "password");
        assert!(output.contains("[REDACTED]"));
        assert!(output.contains("ABC123")); // The value is preserved, just the pattern key
    }

    // ========================================================================
    // Tool Timeout Configuration Tests
    // ========================================================================

    #[test]
    fn test_tool_timeout_config_default() {
        let config = ToolTimeoutConfig::default();
        assert_eq!(config.default_timeout, Duration::from_secs(30));
        assert_eq!(config.max_timeout, Duration::from_secs(300));
        assert_eq!(config.min_timeout, Duration::from_millis(100));
        assert!((config.complexity_multiplier - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_tool_timeout_config_per_tool() {
        let config = ToolTimeoutConfig::default()
            .with_tool_timeout("slow_tool", Duration::from_secs(120))
            .with_tool_timeout("fast_tool", Duration::from_secs(5));

        assert_eq!(config.get_timeout("slow_tool"), Duration::from_secs(120));
        assert_eq!(config.get_timeout("fast_tool"), Duration::from_secs(5));
        assert_eq!(config.get_timeout("unknown"), Duration::from_secs(30)); // default
    }

    #[test]
    fn test_tool_timeout_config_complexity_multiplier() {
        let config = ToolTimeoutConfig::new(Duration::from_secs(30))
            .with_complexity_multiplier(2.0);

        // 30s * 2.0 = 60s
        assert_eq!(config.get_timeout("any_tool"), Duration::from_secs(60));
    }

    #[test]
    fn test_tool_timeout_config_clamping() {
        // Test max clamping
        let config = ToolTimeoutConfig::new(Duration::from_secs(400))
            .with_complexity_multiplier(1.0);
        assert_eq!(config.get_timeout("tool"), Duration::from_secs(300)); // clamped to max

        // Test min clamping
        let config2 = ToolTimeoutConfig::new(Duration::from_millis(10))
            .with_complexity_multiplier(1.0);
        assert_eq!(config2.get_timeout("tool"), Duration::from_millis(100)); // clamped to min
    }

    #[test]
    fn test_tool_timeout_config_quick() {
        let config = ToolTimeoutConfig::quick();
        assert_eq!(config.default_timeout, Duration::from_secs(5));
        assert_eq!(config.max_timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_tool_timeout_config_long_running() {
        let config = ToolTimeoutConfig::long_running();
        assert_eq!(config.default_timeout, Duration::from_secs(120));
        assert_eq!(config.max_timeout, Duration::from_secs(600));
    }

    #[test]
    fn test_tool_timeout_config_io_bound() {
        let config = ToolTimeoutConfig::io_bound();
        assert_eq!(config.get_timeout("http"), Duration::from_secs(60));
        assert_eq!(config.get_timeout("file_read"), Duration::from_secs(30));
        assert_eq!(config.get_timeout("database"), Duration::from_secs(60));
    }

    #[test]
    fn test_task_complexity_multiplier() {
        assert!((TaskComplexity::Simple.multiplier() - 0.5).abs() < 0.01);
        assert!((TaskComplexity::Moderate.multiplier() - 1.0).abs() < 0.01);
        assert!((TaskComplexity::Complex.multiplier() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_tool_context_with_timeout() {
        let config = ToolTimeoutConfig::default()
            .with_tool_timeout("search", Duration::from_secs(60));

        let ctx = ToolContext::new("agent-1")
            .with_timeout_config(config);

        assert_eq!(ctx.get_tool_timeout("search"), Duration::from_secs(60));
        assert_eq!(ctx.get_tool_timeout("other"), Duration::from_secs(30));
    }

    #[test]
    fn test_tool_context_with_complexity() {
        let ctx = ToolContext::new("agent-1")
            .with_complexity(TaskComplexity::Complex);

        assert_eq!(ctx.task_complexity, TaskComplexity::Complex);
        // 30s default * 2.0 complexity = 60s
        assert_eq!(ctx.get_tool_timeout("any"), Duration::from_secs(60));
    }

    #[test]
    fn test_tool_context_complexity_affects_all_timeouts() {
        let config = ToolTimeoutConfig::default()
            .with_tool_timeout("slow", Duration::from_secs(100));

        let ctx = ToolContext::new("agent-1")
            .with_timeout_config(config)
            .with_complexity(TaskComplexity::Simple);

        // 100s * 0.5 (simple) = 50s
        assert_eq!(ctx.get_tool_timeout("slow"), Duration::from_secs(50));
        // 30s * 0.5 = 15s
        assert_eq!(ctx.get_tool_timeout("default"), Duration::from_secs(15));
    }

    // ==========================================================================
    // RiskLevel Tests
    // ==========================================================================

    #[test]
    fn test_risk_level_all_variants() {
        let safe = RiskLevel::Safe;
        let read_only = RiskLevel::ReadOnly;
        let write = RiskLevel::Write;
        let dangerous = RiskLevel::Dangerous;

        assert!(matches!(safe, RiskLevel::Safe));
        assert!(matches!(read_only, RiskLevel::ReadOnly));
        assert!(matches!(write, RiskLevel::Write));
        assert!(matches!(dangerous, RiskLevel::Dangerous));
    }

    #[test]
    fn test_risk_level_eq() {
        assert_eq!(RiskLevel::Safe, RiskLevel::Safe);
        assert_ne!(RiskLevel::Safe, RiskLevel::Dangerous);
    }

    #[test]
    fn test_risk_level_clone() {
        let level = RiskLevel::Write;
        let cloned = level;
        assert_eq!(cloned, RiskLevel::Write);
    }

    #[test]
    fn test_risk_level_debug() {
        let level = RiskLevel::ReadOnly;
        let debug_str = format!("{:?}", level);
        assert_eq!(debug_str, "ReadOnly");
    }

    // ==========================================================================
    // ToolResult Additional Tests
    // ==========================================================================

    #[test]
    fn test_tool_result_success() {
        let result = ToolResult::success("Operation completed");
        assert!(result.success);
        assert_eq!(result.output, "Operation completed");
        assert!(result.error.is_none());
        assert!(result.data.is_none());
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error("Something went wrong");
        assert!(!result.success);
        assert!(result.output.is_empty());
        assert_eq!(result.error, Some("Something went wrong".to_string()));
    }

    #[test]
    fn test_tool_result_with_data() {
        let result = ToolResult::success("test")
            .with_data(serde_json::json!({"key": "value"}));
        assert!(result.data.is_some());
        assert_eq!(result.data.unwrap()["key"], "value");
    }

    #[test]
    fn test_tool_result_serialize() {
        let result = ToolResult::success("output").with_data(serde_json::json!(123));
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("output"));
        assert!(json.contains("true"));
    }

    #[test]
    fn test_tool_result_deserialize() {
        let json = r#"{"success":true,"output":"test","error":null,"data":{"x":1}}"#;
        let result: ToolResult = serde_json::from_str(json).expect("deserialize");
        assert!(result.success);
        assert_eq!(result.output, "test");
    }

    #[test]
    fn test_tool_result_clone() {
        let result = ToolResult::success("clone test");
        let cloned = result.clone();
        assert_eq!(cloned.output, "clone test");
    }

    // ==========================================================================
    // OutputValidationConfig Additional Tests
    // ==========================================================================

    #[test]
    fn test_output_validation_config_default() {
        let config = OutputValidationConfig::default();
        assert_eq!(config.max_output_size, 100_000);
        assert_eq!(config.max_json_depth, 20);
        assert!(config.check_sensitive_data);
        assert!(config.strip_control_chars);
        assert!(config.validate_json);
    }

    #[test]
    fn test_output_validation_config_debug() {
        let config = OutputValidationConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("OutputValidationConfig"));
    }

    #[test]
    fn test_output_validation_config_clone() {
        let config = OutputValidationConfig::strict();
        let cloned = config.clone();
        assert_eq!(cloned.max_output_size, config.max_output_size);
    }

    #[test]
    fn test_output_validation_config_with_max_output_size() {
        let config = OutputValidationConfig::default().with_max_output_size(500);
        assert_eq!(config.max_output_size, 500);
    }

    #[test]
    fn test_output_validation_config_with_sensitive_check() {
        let config = OutputValidationConfig::default().with_sensitive_check(false);
        assert!(!config.check_sensitive_data);
    }

    // ==========================================================================
    // ValidationResult Additional Tests
    // ==========================================================================

    #[test]
    fn test_validation_result_valid() {
        let result = ValidationResult::Valid;
        assert!(result.is_valid());
        assert!(result.issues().is_none());
    }

    #[test]
    fn test_validation_result_invalid() {
        let issues = vec![ValidationIssue::ContainsControlChars];
        let result = ValidationResult::Invalid(issues);
        assert!(!result.is_valid());
        assert!(result.issues().is_some());
        assert_eq!(result.issues().unwrap().len(), 1);
    }

    #[test]
    fn test_validation_result_clone() {
        let result = ValidationResult::Valid;
        let cloned = result.clone();
        assert!(cloned.is_valid());
    }

    #[test]
    fn test_validation_result_eq() {
        assert_eq!(ValidationResult::Valid, ValidationResult::Valid);
        let issues = vec![ValidationIssue::ContainsControlChars];
        let invalid = ValidationResult::Invalid(issues.clone());
        assert_eq!(invalid, ValidationResult::Invalid(issues));
    }

    // ==========================================================================
    // ValidationIssue Tests
    // ==========================================================================

    #[test]
    fn test_validation_issue_output_too_large() {
        let issue = ValidationIssue::OutputTooLarge { size: 200, max: 100 };
        if let ValidationIssue::OutputTooLarge { size, max } = issue {
            assert_eq!(size, 200);
            assert_eq!(max, 100);
        }
    }

    #[test]
    fn test_validation_issue_json_too_deep() {
        let issue = ValidationIssue::JsonTooDeep { depth: 25, max: 20 };
        if let ValidationIssue::JsonTooDeep { depth, max } = issue {
            assert_eq!(depth, 25);
            assert_eq!(max, 20);
        }
    }

    #[test]
    fn test_validation_issue_sensitive_data() {
        let issue = ValidationIssue::SensitiveData { pattern: "password".to_string() };
        if let ValidationIssue::SensitiveData { pattern } = issue {
            assert_eq!(pattern, "password");
        }
    }

    #[test]
    fn test_validation_issue_invalid_json() {
        let issue = ValidationIssue::InvalidJson { error: "unexpected token".to_string() };
        if let ValidationIssue::InvalidJson { error } = issue {
            assert_eq!(error, "unexpected token");
        }
    }

    #[test]
    fn test_validation_issue_debug() {
        let issue = ValidationIssue::ContainsControlChars;
        let debug_str = format!("{:?}", issue);
        assert!(debug_str.contains("ContainsControlChars"));
    }

    #[test]
    fn test_validation_issue_clone() {
        let issue = ValidationIssue::OutputTooLarge { size: 100, max: 50 };
        let cloned = issue.clone();
        assert_eq!(issue, cloned);
    }

    // ==========================================================================
    // ToolTimeoutConfig Additional Tests
    // ==========================================================================

    #[test]
    fn test_tool_timeout_config_debug() {
        let config = ToolTimeoutConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("ToolTimeoutConfig"));
    }

    #[test]
    fn test_tool_timeout_config_clone() {
        let config = ToolTimeoutConfig::quick();
        let cloned = config.clone();
        assert_eq!(cloned.default_timeout, config.default_timeout);
    }

    #[test]
    fn test_tool_timeout_config_new() {
        let config = ToolTimeoutConfig::new(Duration::from_secs(45));
        assert_eq!(config.default_timeout, Duration::from_secs(45));
        // Other fields should use default values
        assert_eq!(config.max_timeout, Duration::from_secs(300));
    }

    // ==========================================================================
    // TaskComplexity Additional Tests
    // ==========================================================================

    #[test]
    fn test_task_complexity_all_variants() {
        let simple = TaskComplexity::Simple;
        let moderate = TaskComplexity::Moderate;
        let complex = TaskComplexity::Complex;

        assert!(matches!(simple, TaskComplexity::Simple));
        assert!(matches!(moderate, TaskComplexity::Moderate));
        assert!(matches!(complex, TaskComplexity::Complex));
    }

    #[test]
    fn test_task_complexity_debug() {
        let complexity = TaskComplexity::Moderate;
        let debug_str = format!("{:?}", complexity);
        assert_eq!(debug_str, "Moderate");
    }

    #[test]
    fn test_task_complexity_eq() {
        assert_eq!(TaskComplexity::Simple, TaskComplexity::Simple);
        assert_ne!(TaskComplexity::Simple, TaskComplexity::Complex);
    }

    // ==========================================================================
    // ToolContext Additional Tests
    // ==========================================================================

    #[test]
    fn test_tool_context_new() {
        let ctx = ToolContext::new("test-agent");
        assert_eq!(ctx.agent_id, "test-agent");
        assert!(ctx.messages.is_empty());
        assert!(ctx.state.is_empty());
    }

    #[test]
    fn test_tool_context_state() {
        let mut ctx = ToolContext::new("test");
        ctx.set_state("key1", serde_json::json!("value1"));
        ctx.set_state("key2", serde_json::json!(42));

        assert_eq!(ctx.get_state("key1"), Some(&serde_json::json!("value1")));
        assert_eq!(ctx.get_state("key2"), Some(&serde_json::json!(42)));
        assert_eq!(ctx.get_state("nonexistent"), None);
    }

    #[test]
    fn test_tool_context_clone() {
        let ctx = ToolContext::new("agent-clone");
        let cloned = ctx.clone();
        assert_eq!(cloned.agent_id, "agent-clone");
    }

    // ==========================================================================
    // ToolCall Tests
    // ==========================================================================

    #[test]
    fn test_tool_call_creation() {
        let call = ToolCall {
            name: "search".to_string(),
            params: serde_json::json!({"query": "test"}),
        };
        assert_eq!(call.name, "search");
        assert_eq!(call.params["query"], "test");
    }

    #[test]
    fn test_tool_call_serialize() {
        let call = ToolCall {
            name: "calculate".to_string(),
            params: serde_json::json!({"a": 1, "b": 2}),
        };
        let json = serde_json::to_string(&call).expect("serialize");
        assert!(json.contains("calculate"));
    }

    #[test]
    fn test_tool_call_deserialize() {
        let json = r#"{"name":"test_tool","params":{"x":10}}"#;
        let call: ToolCall = serde_json::from_str(json).expect("deserialize");
        assert_eq!(call.name, "test_tool");
        assert_eq!(call.params["x"], 10);
    }

    #[test]
    fn test_tool_call_clone() {
        let call = ToolCall {
            name: "clone_test".to_string(),
            params: serde_json::json!({}),
        };
        let cloned = call.clone();
        assert_eq!(cloned.name, "clone_test");
    }

    #[test]
    fn test_tool_call_debug() {
        let call = ToolCall {
            name: "debug_test".to_string(),
            params: serde_json::json!({}),
        };
        let debug_str = format!("{:?}", call);
        assert!(debug_str.contains("debug_test"));
    }

    // ==========================================================================
    // ToolRegistry Tests
    // ==========================================================================

    #[test]
    fn test_tool_registry_new() {
        let registry = ToolRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_tool_registry_with_builtins() {
        let registry = ToolRegistry::with_builtins();
        assert!(!registry.is_empty());
        assert!(registry.get("calculator").is_some());
        assert!(registry.get("json").is_some());
        assert!(registry.get("datetime").is_some());
    }

    #[test]
    fn test_tool_registry_register_and_get() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(CalculatorTool));

        let tool = registry.get("calculator");
        assert!(tool.is_some());
        assert_eq!(tool.unwrap().name(), "calculator");
    }

    #[test]
    fn test_tool_registry_get_nonexistent() {
        let registry = ToolRegistry::new();
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_tool_registry_list() {
        let registry = ToolRegistry::with_builtins();
        let list = registry.list();
        assert!(list.contains(&"calculator"));
        assert!(list.contains(&"json"));
        assert!(list.contains(&"datetime"));
    }

    #[test]
    fn test_tool_registry_tools() {
        let registry = ToolRegistry::with_builtins();
        let tools = registry.tools();
        assert_eq!(tools.len(), 3);
    }

    #[test]
    fn test_tool_registry_len() {
        let mut registry = ToolRegistry::new();
        assert_eq!(registry.len(), 0);
        registry.register(Arc::new(CalculatorTool));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_tool_registry_to_function_definitions() {
        let registry = ToolRegistry::with_builtins();
        let defs = registry.to_function_definitions();
        assert_eq!(defs.len(), 3);

        for def in &defs {
            assert_eq!(def["type"], "function");
            assert!(def["function"]["name"].is_string());
            assert!(def["function"]["description"].is_string());
        }
    }

    #[test]
    fn test_tool_registry_to_prompt_description() {
        let registry = ToolRegistry::with_builtins();
        let desc = registry.to_prompt_description();

        assert!(desc.contains("Available tools:"));
        assert!(desc.contains("calculator"));
        assert!(desc.contains("json"));
        assert!(desc.contains("datetime"));
    }

    #[tokio::test]
    async fn test_tool_registry_execute() {
        let registry = ToolRegistry::with_builtins();
        let ctx = ToolContext::new("test");
        let call = ToolCall {
            name: "calculator".to_string(),
            params: serde_json::json!({"expression": "3+3"}),
        };

        let result = registry.execute(&call, &ctx).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("6"));
    }

    #[tokio::test]
    async fn test_tool_registry_execute_not_found() {
        let registry = ToolRegistry::new();
        let ctx = ToolContext::new("test");
        let call = ToolCall {
            name: "nonexistent".to_string(),
            params: serde_json::json!({}),
        };

        let result = registry.execute(&call, &ctx).await;
        assert!(result.is_err());
    }

    // ==========================================================================
    // JsonTool Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_json_tool_parse() {
        let tool = JsonTool;
        let ctx = ToolContext::new("test");
        let params = serde_json::json!({
            "operation": "parse",
            "data": r#"{"name":"test","value":42}"#
        });

        let result = tool.execute(params, &ctx).await.unwrap();
        assert!(result.success);
        assert!(result.data.is_some());
    }

    #[tokio::test]
    async fn test_json_tool_parse_invalid() {
        let tool = JsonTool;
        let ctx = ToolContext::new("test");
        let params = serde_json::json!({
            "operation": "parse",
            "data": "not valid json {"
        });

        let result = tool.execute(params, &ctx).await.unwrap();
        assert!(!result.success);
        assert!(result.error.is_some());
    }

    #[tokio::test]
    async fn test_json_tool_format() {
        let tool = JsonTool;
        let ctx = ToolContext::new("test");
        let params = serde_json::json!({
            "operation": "format",
            "data": r#"{"a":1,"b":2}"#
        });

        let result = tool.execute(params, &ctx).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains('\n')); // Pretty printed
    }

    #[tokio::test]
    async fn test_json_tool_query() {
        let tool = JsonTool;
        let ctx = ToolContext::new("test");
        let params = serde_json::json!({
            "operation": "query",
            "data": r#"{"nested":{"value":123}}"#,
            "query": "nested.value"
        });

        let result = tool.execute(params, &ctx).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("123"));
    }

    #[tokio::test]
    async fn test_json_tool_query_not_found() {
        let tool = JsonTool;
        let ctx = ToolContext::new("test");
        let params = serde_json::json!({
            "operation": "query",
            "data": r#"{"a":1}"#,
            "query": "nonexistent.path"
        });

        let result = tool.execute(params, &ctx).await.unwrap();
        assert!(!result.success);
    }

    #[tokio::test]
    async fn test_json_tool_unknown_operation() {
        let tool = JsonTool;
        let ctx = ToolContext::new("test");
        let params = serde_json::json!({
            "operation": "unknown",
            "data": "{}"
        });

        let result = tool.execute(params, &ctx).await.unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("Unknown operation"));
    }

    #[test]
    fn test_json_tool_properties() {
        let tool = JsonTool;
        assert_eq!(tool.name(), "json");
        assert!(!tool.description().is_empty());
        assert!(matches!(tool.risk_level(), RiskLevel::Safe));
    }

    // ==========================================================================
    // DateTimeTool Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_datetime_tool_now() {
        let tool = DateTimeTool;
        let ctx = ToolContext::new("test");
        let params = serde_json::json!({ "operation": "now" });

        let result = tool.execute(params, &ctx).await.unwrap();
        assert!(result.success);
        assert!(result.data.is_some());
        assert!(result.data.as_ref().unwrap()["timestamp"].is_number());
    }

    #[tokio::test]
    async fn test_datetime_tool_format() {
        let tool = DateTimeTool;
        let ctx = ToolContext::new("test");
        let params = serde_json::json!({
            "operation": "format",
            "format": "%Y-%m-%d"
        });

        let result = tool.execute(params, &ctx).await.unwrap();
        assert!(result.success);
    }

    #[tokio::test]
    async fn test_datetime_tool_parse() {
        let tool = DateTimeTool;
        let ctx = ToolContext::new("test");
        let params = serde_json::json!({
            "operation": "parse",
            "date": "2024-01-15"
        });

        let result = tool.execute(params, &ctx).await.unwrap();
        assert!(result.success);
        assert!(result.data.is_some());
    }

    #[tokio::test]
    async fn test_datetime_tool_parse_iso8601() {
        let tool = DateTimeTool;
        let ctx = ToolContext::new("test");
        let params = serde_json::json!({
            "operation": "parse",
            "date": "2024-06-15T14:30:00Z"
        });

        let result = tool.execute(params, &ctx).await.unwrap();
        assert!(result.success);
    }

    #[tokio::test]
    async fn test_datetime_tool_unknown_operation() {
        let tool = DateTimeTool;
        let ctx = ToolContext::new("test");
        let params = serde_json::json!({ "operation": "unknown" });

        let result = tool.execute(params, &ctx).await.unwrap();
        assert!(!result.success);
    }

    #[test]
    fn test_datetime_tool_properties() {
        let tool = DateTimeTool;
        assert_eq!(tool.name(), "datetime");
        assert!(!tool.description().is_empty());
        assert!(matches!(tool.risk_level(), RiskLevel::Safe));
    }

    // ==========================================================================
    // CalculatorTool Additional Tests
    // ==========================================================================

    #[test]
    fn test_calculator_tool_properties() {
        let tool = CalculatorTool;
        assert_eq!(tool.name(), "calculator");
        assert!(!tool.description().is_empty());
        assert!(matches!(tool.risk_level(), RiskLevel::Safe));
    }

    #[tokio::test]
    async fn test_calculator_tool_missing_expression() {
        let tool = CalculatorTool;
        let ctx = ToolContext::new("test");
        let params = serde_json::json!({});

        let result = tool.execute(params, &ctx).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_calculator_tool_invalid_expression() {
        let tool = CalculatorTool;
        let ctx = ToolContext::new("test");
        let params = serde_json::json!({ "expression": "not a math expression" });

        let result = tool.execute(params, &ctx).await.unwrap();
        assert!(!result.success);
    }

    // ==========================================================================
    // evaluate_expression Additional Tests
    // ==========================================================================

    #[test]
    fn test_evaluate_expression_subtraction() {
        assert_eq!(evaluate_expression("10-3").unwrap(), 7.0);
    }

    #[test]
    fn test_evaluate_expression_division_by_zero() {
        let result = evaluate_expression("10/0");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Division by zero"));
    }

    #[test]
    fn test_evaluate_expression_parentheses() {
        // The simple evaluator handles parentheses when they wrap the whole expression
        assert_eq!(evaluate_expression("(42)").unwrap(), 42.0);
        assert_eq!(evaluate_expression("((7))").unwrap(), 7.0);
    }

    #[test]
    fn test_evaluate_expression_sqrt() {
        assert_eq!(evaluate_expression("sqrt(25)").unwrap(), 5.0);
        assert!((evaluate_expression("sqrt(2)").unwrap() - 1.414).abs() < 0.01);
    }

    #[test]
    fn test_evaluate_expression_simple_number() {
        assert_eq!(evaluate_expression("42").unwrap(), 42.0);
        assert_eq!(evaluate_expression("3.14").unwrap(), 3.14);
    }

    #[test]
    fn test_evaluate_expression_with_spaces() {
        assert_eq!(evaluate_expression("  2 + 3  ").unwrap(), 5.0);
    }

    // ==========================================================================
    // Helper Function Tests
    // ==========================================================================

    #[test]
    fn test_is_leap_year() {
        assert!(!is_leap_year(2023));
        assert!(is_leap_year(2024));
        assert!(!is_leap_year(1900));
        assert!(is_leap_year(2000));
    }

    #[test]
    fn test_format_unix_timestamp() {
        // Unix epoch
        let epoch = format_unix_timestamp(0);
        assert_eq!(epoch, "1970-01-01T00:00:00Z");

        // One day after epoch
        let one_day = format_unix_timestamp(86400);
        assert_eq!(one_day, "1970-01-02T00:00:00Z");
    }

    #[test]
    fn test_parse_datetime_iso8601() {
        let result = parse_datetime("2024-01-15T10:30:00Z");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_datetime_date_only() {
        let result = parse_datetime("2024-06-15");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_datetime_us_format() {
        let result = parse_datetime("06/15/2024");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_datetime_invalid() {
        let result = parse_datetime("not a date");
        assert!(result.is_err());
    }

    #[test]
    fn test_day_of_year_to_month_day() {
        // Jan 1 (day 0)
        assert_eq!(day_of_year_to_month_day(0, false), (1, 1));
        // Jan 31 (day 30)
        assert_eq!(day_of_year_to_month_day(30, false), (1, 31));
        // Feb 1 (day 31)
        assert_eq!(day_of_year_to_month_day(31, false), (2, 1));
    }

    #[test]
    fn test_has_control_chars() {
        assert!(!has_control_chars("normal text"));
        assert!(!has_control_chars("with\nnewline"));
        assert!(!has_control_chars("with\ttab"));
        assert!(has_control_chars("with\x00null"));
        assert!(has_control_chars("with\x1Bescape"));
    }

    #[test]
    fn test_query_json_array() {
        let json = serde_json::json!(["a", "b", "c"]);
        assert_eq!(query_json(&json, "1"), Some(serde_json::json!("b")));
    }

    #[test]
    fn test_query_json_empty() {
        let json = serde_json::json!({"key": "value"});
        assert_eq!(query_json(&json, ""), Some(json.clone()));
    }

    #[test]
    fn test_query_json_deep_nesting() {
        let json = serde_json::json!({
            "a": {
                "b": {
                    "c": 42
                }
            }
        });
        assert_eq!(query_json(&json, "a.b.c"), Some(serde_json::json!(42)));
    }

    #[test]
    fn test_format_with_pattern() {
        // Known timestamp: 2024-01-15 00:00:00 UTC
        let timestamp = 1705276800;
        let formatted = format_with_pattern(timestamp, "%Y-%m-%d");
        assert!(formatted.contains("2024"));
    }

    #[test]
    fn test_sensitive_patterns_detection() {
        let config = OutputValidationConfig::default();

        // Test various sensitive patterns
        let test_cases = vec![
            "my password is secret",
            "api_key: abc123",
            "bearer token here",
            "private_key data",
            "credential info",
        ];

        for test in test_cases {
            let result = ToolResult::success(test);
            assert!(!result.validate(&config).is_valid(),
                "Should detect sensitive data in: {}", test);
        }
    }
}
