//! Common types used across the Infernum ecosystem.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// Unique identifier for a model.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelId(pub String);

impl ModelId {
    /// Creates a new `ModelId` from a string.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for ModelId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for ModelId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Model family for format selection.
///
/// Different model families require different tool calling formats,
/// chat templates, and conversation reconstruction. This type is
/// shared across the Infernum runtime (server, agent, tokenizer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ModelFamily {
    /// Qwen models (Qwen2.5, etc.) — `<tools>` / `<tool_call>` / `<tool_response>` format.
    Qwen,
    /// Llama models (Llama 3, etc.) — `<|python_tag|>` format.
    Llama,
    /// Mistral models — `[TOOL_CALLS]` format.
    Mistral,
    /// Unknown model family — use generic format.
    #[default]
    Unknown,
}

impl ModelFamily {
    /// Detect model family from model name or path.
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

    /// Whether this model family has a native tool calling format.
    #[must_use]
    pub fn has_native_tool_format(&self) -> bool {
        !matches!(self, Self::Unknown)
    }
}

impl std::fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Qwen => write!(f, "Qwen"),
            Self::Llama => write!(f, "Llama"),
            Self::Mistral => write!(f, "Mistral"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Unique identifier for a request.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct RequestId(pub Uuid);

impl RequestId {
    /// Creates a new random `RequestId`.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Data type for tensor operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    /// 32-bit floating point.
    F32,
    /// 16-bit floating point.
    F16,
    /// Brain floating point (16-bit).
    BF16,
    /// 8-bit integer (quantized).
    I8,
    /// 4-bit integer (quantized).
    I4,
}

/// Quantization type for models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationType {
    /// No quantization (full precision).
    None,
    /// 8-bit integer quantization.
    Int8,
    /// 4-bit integer quantization.
    Int4,
    /// GPTQ quantization.
    GPTQ,
    /// AWQ quantization.
    AWQ,
    /// GGUF Q4_0 quantization.
    GgufQ4_0,
    /// GGUF Q4_K_M quantization.
    GgufQ4KM,
    /// GGUF Q5_K_M quantization.
    GgufQ5KM,
    /// GGUF Q8_0 quantization.
    GgufQ8_0,
}

/// Device type for computation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    /// CPU computation.
    Cpu,
    /// CUDA GPU computation.
    Cuda {
        /// GPU device index.
        device_id: usize,
    },
    /// Apple Metal GPU computation.
    Metal {
        /// GPU device index.
        device_id: usize,
    },
    /// WebGPU computation.
    WebGpu,
}

impl Default for DeviceType {
    fn default() -> Self {
        Self::Cpu
    }
}

/// Finish reason for generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Generation stopped due to reaching max tokens.
    Length,
    /// Generation stopped due to hitting a stop sequence.
    Stop,
    /// Generation stopped due to tool/function call.
    ToolCalls,
    /// Generation stopped due to content filter.
    ContentFilter,
}

/// Role in a conversation.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System message (instructions).
    System,
    /// User message.
    User,
    /// Assistant message.
    Assistant,
    /// Tool/function result.
    Tool,
}

/// Definition of a tool available to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name.
    pub name: String,
    /// Human-readable description of what the tool does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema describing the tool's parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    /// Whether to enforce strict schema adherence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl ToolDefinition {
    /// Creates a new tool definition with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            parameters: None,
            strict: None,
        }
    }

    /// Sets the description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Sets the parameters schema.
    #[must_use]
    pub fn with_parameters(mut self, parameters: Value) -> Self {
        self.parameters = Some(parameters);
        self
    }
}

/// A tool call emitted by the model.
///
/// Arguments are structured JSON values, not strings containing JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this call.
    pub id: String,
    /// Name of the tool being called.
    pub name: String,
    /// Structured arguments (parsed JSON, not a string).
    pub arguments: Value,
}

/// Controls how the model uses tools.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolControl {
    /// String mode: "none", "auto", or "required".
    Mode(ToolControlMode),
    /// Force a specific tool by name.
    Specific {
        /// Name of the tool to call.
        name: String,
    },
}

/// Tool control mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolControlMode {
    /// Model must not call tools.
    None,
    /// Model decides whether to call tools.
    Auto,
    /// Model must call at least one tool.
    Required,
}

impl Default for ToolControl {
    fn default() -> Self {
        Self::Mode(ToolControlMode::Auto)
    }
}

/// A message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role of the message sender.
    pub role: Role,
    /// Content of the message.
    pub content: String,
    /// Optional name for the sender.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool calls emitted by the assistant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Tool call ID (for tool responses).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    /// Creates a new system message.
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Creates a new user message.
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Creates a new assistant message.
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Creates an assistant message with tool calls.
    #[must_use]
    pub fn assistant_with_tool_calls(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            name: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        }
    }

    /// Creates a tool result message.
    #[must_use]
    pub fn tool_result(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            name: None,
            tool_calls: None,
            tool_call_id: Some(call_id.into()),
        }
    }
}

/// Token usage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,
    /// Number of tokens generated.
    pub completion_tokens: u32,
    /// Total tokens (prompt + completion).
    pub total_tokens: u32,
}

impl Usage {
    /// Creates a new `Usage` from prompt and completion token counts.
    #[must_use]
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // ModelId tests
    // ==========================================================================

    #[test]
    fn test_model_id_new() {
        let id = ModelId::new("llama-3.2-3b");
        assert_eq!(id.0, "llama-3.2-3b");
    }

    #[test]
    fn test_model_id_from_string() {
        let id: ModelId = "test-model".to_string().into();
        assert_eq!(id.0, "test-model");
    }

    #[test]
    fn test_model_id_from_str() {
        let id: ModelId = "my-model".into();
        assert_eq!(id.0, "my-model");
    }

    #[test]
    fn test_model_id_display() {
        let id = ModelId::new("display-model");
        assert_eq!(format!("{}", id), "display-model");
    }

    #[test]
    fn test_model_id_equality() {
        let id1 = ModelId::new("same");
        let id2 = ModelId::new("same");
        let id3 = ModelId::new("different");

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_model_id_clone() {
        let id = ModelId::new("clone-me");
        let cloned = id.clone();
        assert_eq!(id, cloned);
    }

    #[test]
    fn test_model_id_serialization() {
        let id = ModelId::new("serialized");
        let json = serde_json::to_string(&id).expect("serialize");
        assert!(json.contains("serialized"));

        let parsed: ModelId = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, id);
    }

    // ==========================================================================
    // RequestId tests
    // ==========================================================================

    #[test]
    fn test_request_id_new() {
        let id1 = RequestId::new();
        let id2 = RequestId::new();

        // UUIDs should be unique
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_request_id_default() {
        let id = RequestId::default();
        // Just verify it creates without panicking
        let _ = format!("{}", id);
    }

    #[test]
    fn test_request_id_display() {
        let id = RequestId::new();
        let display = format!("{}", id);
        // UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        assert!(display.len() == 36);
        assert!(display.contains('-'));
    }

    #[test]
    fn test_request_id_clone() {
        let id = RequestId::new();
        let cloned = id.clone();
        assert_eq!(id, cloned);
    }

    // ==========================================================================
    // DType tests
    // ==========================================================================

    #[test]
    fn test_dtype_variants() {
        assert_eq!(DType::F32, DType::F32);
        assert_ne!(DType::F32, DType::F16);
        assert_ne!(DType::F16, DType::BF16);
        assert_ne!(DType::I8, DType::I4);
    }

    #[test]
    fn test_dtype_serialization() {
        let dtype = DType::BF16;
        let json = serde_json::to_string(&dtype).expect("serialize");

        let parsed: DType = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, dtype);
    }

    // ==========================================================================
    // QuantizationType tests
    // ==========================================================================

    #[test]
    fn test_quantization_type_variants() {
        assert_eq!(QuantizationType::None, QuantizationType::None);
        assert_ne!(QuantizationType::None, QuantizationType::Int8);
        assert_ne!(QuantizationType::Int4, QuantizationType::GPTQ);
        assert_ne!(QuantizationType::AWQ, QuantizationType::GgufQ4_0);
    }

    #[test]
    fn test_quantization_serialization() {
        let quant = QuantizationType::GgufQ4KM;
        let json = serde_json::to_string(&quant).expect("serialize");

        let parsed: QuantizationType = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, quant);
    }

    // ==========================================================================
    // DeviceType tests
    // ==========================================================================

    #[test]
    fn test_device_type_cpu() {
        let device = DeviceType::Cpu;
        assert_eq!(device, DeviceType::Cpu);
    }

    #[test]
    fn test_device_type_cuda() {
        let device = DeviceType::Cuda { device_id: 0 };
        match device {
            DeviceType::Cuda { device_id } => assert_eq!(device_id, 0),
            _ => panic!("Expected Cuda"),
        }
    }

    #[test]
    fn test_device_type_metal() {
        let device = DeviceType::Metal { device_id: 1 };
        match device {
            DeviceType::Metal { device_id } => assert_eq!(device_id, 1),
            _ => panic!("Expected Metal"),
        }
    }

    #[test]
    fn test_device_type_default() {
        let device = DeviceType::default();
        assert_eq!(device, DeviceType::Cpu);
    }

    #[test]
    fn test_device_type_serialization() {
        let device = DeviceType::Cuda { device_id: 2 };
        let json = serde_json::to_string(&device).expect("serialize");
        assert!(json.contains("Cuda") || json.contains("cuda"));

        let parsed: DeviceType = serde_json::from_str(&json).expect("deserialize");
        match parsed {
            DeviceType::Cuda { device_id } => assert_eq!(device_id, 2),
            _ => panic!("Wrong device type"),
        }
    }

    // ==========================================================================
    // FinishReason tests
    // ==========================================================================

    #[test]
    fn test_finish_reason_variants() {
        assert_eq!(FinishReason::Length, FinishReason::Length);
        assert_ne!(FinishReason::Length, FinishReason::Stop);
        assert_ne!(FinishReason::ToolCalls, FinishReason::ContentFilter);
    }

    #[test]
    fn test_finish_reason_serialization() {
        let reason = FinishReason::Stop;
        let json = serde_json::to_string(&reason).expect("serialize");
        assert!(json.contains("stop"));

        let parsed: FinishReason = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, reason);
    }

    // ==========================================================================
    // Role tests
    // ==========================================================================

    #[test]
    fn test_role_variants() {
        assert_eq!(Role::System, Role::System);
        assert_ne!(Role::System, Role::User);
        assert_ne!(Role::User, Role::Assistant);
        assert_ne!(Role::Assistant, Role::Tool);
    }

    #[test]
    fn test_role_serialization() {
        let role = Role::User;
        let json = serde_json::to_string(&role).expect("serialize");
        assert!(json.contains("user"));

        let parsed: Role = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, role);
    }

    // ==========================================================================
    // Message tests
    // ==========================================================================

    #[test]
    fn test_message_system() {
        let msg = Message::system("You are a helpful assistant.");
        assert_eq!(msg.role, Role::System);
        assert_eq!(msg.content, "You are a helpful assistant.");
        assert!(msg.name.is_none());
        assert!(msg.tool_call_id.is_none());
    }

    #[test]
    fn test_message_user() {
        let msg = Message::user("Hello!");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content, "Hello!");
    }

    #[test]
    fn test_message_assistant() {
        let msg = Message::assistant("Hi there!");
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content, "Hi there!");
    }

    #[test]
    fn test_message_serialization() {
        let msg = Message::user("Test message");
        let json = serde_json::to_string(&msg).expect("serialize");
        assert!(json.contains("user"));
        assert!(json.contains("Test message"));

        let parsed: Message = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.role, Role::User);
        assert_eq!(parsed.content, "Test message");
    }

    #[test]
    fn test_message_with_name() {
        let msg = Message {
            role: Role::User,
            content: "Hello".to_string(),
            name: Some("Alice".to_string()),
            tool_calls: None,
            tool_call_id: None,
        };

        let json = serde_json::to_string(&msg).expect("serialize");
        assert!(json.contains("Alice"));

        let parsed: Message = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.name, Some("Alice".to_string()));
    }

    #[test]
    fn test_message_clone() {
        let msg = Message::user("Clone me");
        let cloned = msg.clone();
        assert_eq!(cloned.role, msg.role);
        assert_eq!(cloned.content, msg.content);
    }

    // ==========================================================================
    // Usage tests
    // ==========================================================================

    #[test]
    fn test_usage_new() {
        let usage = Usage::new(100, 50);
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn test_usage_default() {
        let usage = Usage::default();
        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
    }

    #[test]
    fn test_usage_serialization() {
        let usage = Usage::new(200, 100);
        let json = serde_json::to_string(&usage).expect("serialize");
        assert!(json.contains("200"));
        assert!(json.contains("100"));
        assert!(json.contains("300"));

        let parsed: Usage = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.prompt_tokens, 200);
        assert_eq!(parsed.completion_tokens, 100);
        assert_eq!(parsed.total_tokens, 300);
    }

    #[test]
    fn test_usage_clone() {
        let usage = Usage::new(50, 25);
        let cloned = usage.clone();
        assert_eq!(cloned.prompt_tokens, usage.prompt_tokens);
        assert_eq!(cloned.completion_tokens, usage.completion_tokens);
        assert_eq!(cloned.total_tokens, usage.total_tokens);
    }

    // ==========================================================================
    // ToolDefinition tests
    // ==========================================================================

    #[test]
    fn test_tool_definition_new() {
        let tool = ToolDefinition::new("read_file");
        assert_eq!(tool.name, "read_file");
        assert!(tool.description.is_none());
        assert!(tool.parameters.is_none());
        assert!(tool.strict.is_none());
    }

    #[test]
    fn test_tool_definition_builder() {
        let tool = ToolDefinition::new("read_file")
            .with_description("Read a file")
            .with_parameters(serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"]
            }));

        assert_eq!(tool.name, "read_file");
        assert_eq!(tool.description.as_deref(), Some("Read a file"));
        assert!(tool.parameters.is_some());
    }

    #[test]
    fn test_tool_definition_serialization() {
        let tool = ToolDefinition::new("bash")
            .with_description("Run a command");

        let json = serde_json::to_string(&tool).expect("serialize");
        assert!(json.contains("\"name\":\"bash\""));
        assert!(json.contains("\"description\":\"Run a command\""));
        assert!(!json.contains("parameters"));

        let parsed: ToolDefinition = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.name, "bash");
    }

    // ==========================================================================
    // ToolCall tests
    // ==========================================================================

    #[test]
    fn test_tool_call_structured_arguments() {
        let call = ToolCall {
            id: "call_1".into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "src/main.rs"}),
        };

        assert_eq!(call.name, "read_file");
        assert_eq!(call.arguments["path"], "src/main.rs");
    }

    #[test]
    fn test_tool_call_serialization() {
        let call = ToolCall {
            id: "call_abc".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command": "ls"}),
        };

        let json = serde_json::to_string(&call).expect("serialize");
        assert!(json.contains("\"arguments\":{"));
        assert!(!json.contains("\\\""));

        let parsed: ToolCall = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.arguments["command"], "ls");
    }

    // ==========================================================================
    // ToolControl tests
    // ==========================================================================

    #[test]
    fn test_tool_control_mode_serialization() {
        let auto = ToolControlMode::Auto;
        let json = serde_json::to_string(&auto).expect("serialize");
        assert_eq!(json, "\"auto\"");

        let none = ToolControlMode::None;
        let json = serde_json::to_string(&none).expect("serialize");
        assert_eq!(json, "\"none\"");

        let required = ToolControlMode::Required;
        let json = serde_json::to_string(&required).expect("serialize");
        assert_eq!(json, "\"required\"");
    }

    #[test]
    fn test_tool_control_specific_serialization() {
        let specific = ToolControl::Specific { name: "read_file".into() };
        let json = serde_json::to_string(&specific).expect("serialize");
        assert!(json.contains("\"name\":\"read_file\""));

        let parsed: ToolControl = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, specific);
    }

    #[test]
    fn test_tool_control_default() {
        let tc = ToolControl::default();
        assert_eq!(tc, ToolControl::Mode(ToolControlMode::Auto));
    }

    // ==========================================================================
    // Message with tool_calls tests
    // ==========================================================================

    #[test]
    fn test_message_assistant_with_tool_calls() {
        let msg = Message::assistant_with_tool_calls(
            "I'll read that file.",
            vec![ToolCall {
                id: "call_1".into(),
                name: "read_file".into(),
                arguments: serde_json::json!({"path": "main.rs"}),
            }],
        );

        assert_eq!(msg.role, Role::Assistant);
        assert!(msg.tool_calls.is_some());
        assert_eq!(msg.tool_calls.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_message_tool_result() {
        let msg = Message::tool_result("call_1", "fn main() {}");
        assert_eq!(msg.role, Role::Tool);
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(msg.content, "fn main() {}");
    }

    #[test]
    fn test_message_tool_calls_omitted_when_none() {
        let msg = Message::user("Hello");
        let json = serde_json::to_string(&msg).expect("serialize");
        assert!(!json.contains("tool_calls"));
    }

    #[test]
    fn test_message_tool_calls_serialization() {
        let msg = Message::assistant_with_tool_calls(
            "Reading file.",
            vec![ToolCall {
                id: "c1".into(),
                name: "read_file".into(),
                arguments: serde_json::json!({"path": "lib.rs"}),
            }],
        );

        let json = serde_json::to_string(&msg).expect("serialize");
        assert!(json.contains("\"tool_calls\""));
        assert!(json.contains("\"arguments\":{"));

        let parsed: Message = serde_json::from_str(&json).expect("deserialize");
        assert!(parsed.tool_calls.is_some());
        assert_eq!(parsed.tool_calls.unwrap()[0].name, "read_file");
    }
}
