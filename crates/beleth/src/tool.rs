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
pub struct ToolContext {
    /// Agent ID.
    pub agent_id: String,
    /// Conversation history.
    pub messages: Vec<infernum_core::Message>,
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
}

// === Built-in Tools ===

/// Calculator tool.
pub struct CalculatorTool;

#[async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Evaluates mathematical expressions"
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

        // Simple evaluation (in production, use a proper math parser)
        // For now, just return a placeholder
        Ok(ToolResult::success(format!(
            "Calculated: {} = [evaluation not implemented]",
            expression
        )))
    }
}
