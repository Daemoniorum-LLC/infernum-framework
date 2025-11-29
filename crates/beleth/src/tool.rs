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
}

impl ToolContext {
    /// Creates a new tool context.
    #[must_use]
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            messages: Vec::new(),
            state: HashMap::new(),
        }
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

    /// Generates tool descriptions for prompting.
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
            "parse" => {
                match serde_json::from_str::<Value>(data) {
                    Ok(parsed) => Ok(ToolResult::success("JSON parsed successfully")
                        .with_data(parsed)),
                    Err(e) => Ok(ToolResult::error(format!("Parse error: {}", e))),
                }
            }
            "format" => {
                match serde_json::from_str::<Value>(data) {
                    Ok(parsed) => {
                        match serde_json::to_string_pretty(&parsed) {
                            Ok(formatted) => Ok(ToolResult::success(formatted)),
                            Err(e) => Ok(ToolResult::error(format!("Format error: {}", e))),
                        }
                    }
                    Err(e) => Ok(ToolResult::error(format!("Parse error: {}", e))),
                }
            }
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
                    }
                    Err(e) => Ok(ToolResult::error(format!("Parse error: {}", e))),
                }
            }
            _ => Ok(ToolResult::error(format!("Unknown operation: {}", operation))),
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

                Ok(ToolResult::success(format!("Current time: {}", datetime))
                    .with_data(serde_json::json!({
                        "timestamp": secs,
                        "iso8601": datetime
                    })))
            }
            _ => Ok(ToolResult::error(format!("Operation '{}' not yet implemented", operation))),
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
                if prev.map(|p| p.is_ascii_digit() || p == ')').unwrap_or(false) {
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
            }
            Value::Array(arr) => {
                let idx: usize = part.parse().ok()?;
                current = arr.get(idx)?;
            }
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

        assert_eq!(
            query_json(&json, "name"),
            Some(serde_json::json!("test"))
        );
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
}
