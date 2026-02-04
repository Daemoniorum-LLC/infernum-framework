# Tool Calling Specification

**Version:** 1.4.0
**Status:** Phase 5 Complete, Active Design Debt (§11.4)
**Date:** 2026-02-04
**Prerequisite:** ROADMAP.md Phase 1.3

---

## 1. Overview

This specification defines tool calling behavior for Infernum's native runtime, including the `/v1/chat/completions` endpoint and the Beleth agent framework.

### 1.1 Scope

**In Scope:**
- Accepting tools in chat completion requests
- Formatting tools into **model-native** prompts (matching each model's training format)
- Detecting tool calls in model output
- Returning tool_calls in responses
- Formatting multi-turn tool conversations in model-native format
- Server-side tool execution with policy control (Phase 4)
- Agent-centric observability and coordination (Phase 5)
- Beleth model-aware tool calling across all execution strategies (ReAct, ToT, OODA)

**Out of Scope:**
- Custom tool runtime environments
- Tool marketplace/discovery (future consideration)

### 1.2 Architecture Decision

Tool calling supports **both patterns**:

1. **Client-side execution** (API, Phases 1-3)
2. **Server-side execution** (Agent runtime, Phases 4-5)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Tool Calling Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 1-3: Client-Side          Phase 4-5: Server-Side         │
│  ─────────────────────           ──────────────────────          │
│                                                                  │
│  Client        Server            Client        Server            │
│    │             │                 │             │               │
│    ├─ tools ────►│                 ├─ tools ────►│               │
│    │             │ detect          │             │ detect        │
│    │◄─ tool_calls┤                 │             │ execute ◄─┐   │
│    │             │                 │             │ (Beleth)   │   │
│    │ execute     │                 │             │            │   │
│    │ locally     │                 │             │ policy ────┤   │
│    │             │                 │             │ (Phase 4.3)│   │
│    ├─ result ───►│                 │◄─ result ──┤◄───────────┘   │
│    │◄─ response─┤                 │◄─ response─┤               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Request Specification

### 2.1 Tool Definition

```typescript
interface Tool {
  type: "function";
  function: {
    name: string;           // Function name (alphanumeric + underscore)
    description?: string;   // What the function does
    parameters?: object;    // JSON Schema for parameters
    strict?: boolean;       // Require exact schema match
  };
}
```

### 2.2 Tool Choice

**Specified Interface (Target):**

```typescript
type ToolChoice =
  | "none"                    // Never call tools
  | "auto"                    // Model decides (default)
  | "required"                // Must call at least one tool
  | { type: "function"; function: { name: string } };  // Force specific tool
```

**Rust Implementation (Target):**

```rust
/// Tool choice for chat completions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// String variants: "none", "auto", "required"
    Mode(ToolChoiceMode),
    /// Force a specific function
    Function(ToolChoiceFunction),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceMode {
    None,
    Auto,
    Required,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    #[serde(rename = "type")]
    pub choice_type: String,  // Always "function"
    pub function: ToolChoiceFunctionName,
}
```

> **Design Debt:** Current implementation uses `ToolChoice::String(String)` instead of typed `ToolChoiceMode` enum. See §11 Design Debt.

### 2.3 Request Example

```json
{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "messages": [
    {"role": "user", "content": "What's the weather in Seattle?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

---

## 3. Response Specification

### 3.1 Tool Call Response

When the model decides to call a tool:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"Seattle\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }],
  "usage": {...}
}
```

### 3.2 Regular Response (No Tool Call)

When the model responds without tools:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "I can help you with that..."
    },
    "finish_reason": "stop"
  }]
}
```

### 3.3 Finish Reasons

| Reason | Meaning |
|--------|---------|
| `stop` | Normal completion |
| `length` | Max tokens reached |
| `tool_calls` | Model wants to call tools |
| `content_filter` | Content filtered (if applicable) |

---

## 4. Tool Result Messages

### 4.1 Tool Message Format

After executing a tool, client sends result:

```json
{
  "messages": [
    {"role": "user", "content": "What's the weather in Seattle?"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"Seattle\"}"
        }
      }]
    },
    {
      "role": "tool",
      "tool_call_id": "call_abc123",
      "content": "{\"temperature\": 62, \"condition\": \"cloudy\"}"
    }
  ]
}
```

### 4.2 Tool Message Validation

- `tool_call_id` MUST match a previous tool_call id
- `content` SHOULD be JSON but MAY be plain text
- Multiple tool results MAY be sent for parallel calls

---

## 5. Model-Specific Formatting

### 5.1 Qwen Format

Qwen2.5 models use a native tool calling format defined in their training chat template
(extracted from `tokenizer_config.json` Jinja template). The format uses `<tools></tools>` XML
tags with full JSON function definitions and structured multi-turn conversation tags.

**System Message with Tools:**

```
<|im_start|>system
You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City name"}}, "required": ["location"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
<|im_end|>
```

**Key requirements:**
- Each tool is serialized as a complete `{"type": "function", "function": {...}}` JSON object
- Tools are wrapped in `<tools></tools>` XML tags, one JSON object per line
- Instruction text matches the native training format verbatim

### 5.2 Llama Format

Llama uses `<|python_tag|>` markers for tool calls.

### 5.3 Mistral Format

Mistral uses `[TOOL_CALLS]` markers for tool calls.

### 5.4 Detection Patterns

Tool calls are detected by parsing model output for:
- `<tool_call>...</tool_call>` tags (Qwen)
- `<|python_tag|>` markers (Llama)
- `[TOOL_CALLS]` markers (Mistral)
- `{"name": "...", "arguments": ...}` JSON patterns

### 5.5 Multi-Turn Conversation Format

When a tool calling conversation spans multiple turns, the conversation history
MUST be reconstructed in each model's native format. This is critical for models
to understand the conversation flow and produce correct tool calls.

#### 5.5.1 Qwen Multi-Turn Format

**Assistant message with tool call:**

The assistant's tool call output MUST be reconstructed with `<tool_call>` tags
in the message content, even when the original response is stored as a structured
`tool_calls` array in the API:

```
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"location": "Seattle"}}
</tool_call>
<|im_end|>
```

**Tool result message:**

Tool results MUST be wrapped in `<tool_response></tool_response>` tags and
sent as a `user` role message (Qwen2.5's training template uses `user` role
for tool responses):

```
<|im_start|>user
<tool_response>
{"temperature": 62, "condition": "cloudy"}
</tool_response>
<|im_end|>
```

**Complete multi-turn example:**

```
<|im_start|>system
You are a helpful assistant.
{tools_prompt}
<|im_end|>
<|im_start|>user
What's the weather in Seattle?
<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"location": "Seattle"}}
</tool_call>
<|im_end|>
<|im_start|>user
<tool_response>
{"temperature": 62, "condition": "cloudy"}
</tool_response>
<|im_end|>
<|im_start|>assistant
The weather in Seattle is currently 62°F and cloudy.
<|im_end|>
```

#### 5.5.2 Implementation Requirements

For each model family, the message-building pipeline MUST:

1. **Reconstruct assistant tool_calls**: When an assistant message has a `tool_calls`
   array, embed the calls in the content using the model's native tool call syntax.
2. **Format tool results**: When a `tool` role message is encountered, wrap the
   content in the model's native tool response syntax.
3. **Preserve text content**: If an assistant message has both `content` and
   `tool_calls`, the text content MUST be preserved alongside the reconstructed tags.

### 5.6 Beleth Agent Native Format

Beleth's agent execution strategies (ReAct, ToT, OODA) MUST use model-native
tool calling format when the model supports it, rather than generic text-based
`Action: / Action Input:` format.

**Rationale:** Models trained with native tool calling formats (e.g., Qwen2.5's
`<tool_call>` syntax) produce more reliable tool calls when prompted in their
training format. Generic ReAct-style prompting works but sacrifices the model's
built-in tool calling capability.

**Architecture:**

```
┌────────────────────────────────────────────────────┐
│                   Beleth Agent                      │
├────────────────────────────────────────────────────┤
│                                                     │
│  Model-Aware Tool Formatter                         │
│  ┌──────────────┐  ┌────────────┐  ┌──────────┐   │
│  │ Qwen Native  │  │ Llama      │  │ Generic  │   │
│  │ <tools>      │  │ <|python_  │  │ Action:  │   │
│  │ <tool_call>  │  │  tag|>     │  │ Action   │   │
│  │ <tool_resp>  │  │            │  │  Input:  │   │
│  └──────────────┘  └────────────┘  └──────────┘   │
│         │                │               │          │
│         └────────────────┼───────────────┘          │
│                          ▼                          │
│              build_system_prompt()                   │
│              parse_action()                          │
│                                                     │
│  ReAct  │  ToT  │  OODA  │  Hierarchical            │
└────────────────────────────────────────────────────┘
```

**Implementation approach:**
- `build_system_prompt()` uses `ModelFamily` to select native or generic format
- `parse_action()` detects both native format (`<tool_call>` tags) and generic
  format (`Action: / Action Input:`) for backwards compatibility
- Tool results are formatted in native `<tool_response>` tags when supported
- `ToolRegistry::to_prompt_description()` gains a model-family-aware variant

---

## 6. Error Handling

### 6.1 Invalid Tool Definition

```json
{
  "error": {
    "message": "Invalid tool definition: missing 'name' field",
    "type": "invalid_request_error",
    "code": "invalid_tools"
  }
}
```

### 6.2 Tool Not Found (for forced tool_choice)

```json
{
  "error": {
    "message": "Tool 'unknown_tool' not found in tools list",
    "type": "invalid_request_error",
    "code": "tool_not_found"
  }
}
```

### 6.3 Invalid Tool Call ID

```json
{
  "error": {
    "message": "tool_call_id 'call_xyz' does not match any previous tool call",
    "type": "invalid_request_error",
    "code": "invalid_tool_call_id"
  }
}
```

### 6.4 Policy Denial (Phase 4.3)

```json
{
  "error": {
    "message": "Tool 'dangerous_tool' denied by policy: requires elevated permissions",
    "type": "permission_denied",
    "code": "tool_policy_denied"
  }
}
```

---

## 7. Implementation Phases

### Phase 1: Format Detection (COMPLETE)
- [x] Accept tools in request
- [x] Format tools into Qwen prompt (`tool_use.rs`)
- [x] Detect tool calls in output (Qwen format)
- [x] Return tool_calls in response
- [x] Handle tool role messages

**Required Types:** `ModelFamily`
**Required Functions:** `detect_model_family`

### Phase 2: Tool Formatting (COMPLETE)
- [x] Llama tool format support (`<|python_tag|>` detection)
- [x] Mistral tool format support (`[TOOL_CALLS]` detection)
- [x] Deep JSON parsing (replaces regex for nested arguments)

**Required Types:** `Tool`, `FunctionDefinition`
**Required Functions:** `format_tools_for_prompt`, `should_include_tools`

### Phase 3: Tool Detection (COMPLETE)
- [x] Streaming with tool detection (`StreamingToolDetector`)
- [x] Agent-centric SSE format (`SseEvent`, `SseUsage`)
- [x] `parallel_tool_calls` enforcement (`process_model_output_with_options`)
- [x] `strict` mode schema validation (`validate_tool_arguments`)
- [x] Unknown tool detection and logging (`validate_detected_calls`)

**Required Types:** `DetectedToolCall`, `StreamingToolDetector`, `ToolDetectionEvent`
**Required Functions:** `detect_tool_calls`, `process_model_output`

### Phase 4: Agent Runtime (COMPLETE - Implementation)

#### Phase 4.1: Tool Choice Validation (COMPLETE)
- [x] Server-side `validate_tool_exists` enforcement
- [x] `tool_choice: "required"` enforcement
- [x] Forced tool extraction (`get_forced_tool`)

**Required Types:** `ToolChoice`, `ToolChoiceFunction`
**Required Functions:** `validate_tool_choice`, `validate_tool_exists`

#### Phase 4.2: Tool Execution (COMPLETE)
- [x] Server-side tool execution via Beleth (`ToolExecutor`)
- [x] Risk-based timeouts (`RiskTimeouts`)
- [x] Execution metrics tracking (`ToolExecutorMetrics`)
- [x] Parallel and sequential execution modes

**Required Types:** `ToolExecutor`, `ToolExecutorConfig`, `ToolExecutionResult`, `ToolExecutorError`
**Required Functions:** `execute`, `execute_parallel`, `execute_sequential`

#### Phase 4.3: Policy & Identity (COMPLETE)
- [x] Policy-based tool access control (`PolicyEngine`)
- [x] Agent identity with Ed25519 signing (`AgentIdentity`)
- [x] Cryptographic audit trail (`SignedEvent`)
- [x] Rate limiting and argument validation

**Required Types:** `ToolPolicy`, `PolicyEngine`, `PolicyDecision`, `DenyReason`, `AgentIdentity`, `SignedEvent`
**Required Functions:** `evaluate`, `sign_event`, `verify_event`

### Phase 5: Agent-Centric Infrastructure (COMPLETE - Implementation)

#### Phase 5.1: Agent Awareness (COMPLETE)
Provides agents with runtime visibility into tool reliability and cost.

- [x] Tool health monitoring (`ToolHealthStatus`, `HealthState`)
- [x] Cost estimation (`CostEstimate`)
- [x] Quota tracking (`QuotaStatus`)
- [x] Retry guidance (`RetryHints`, `ErrorCategory`)
- [x] Cache control (`CacheInfo`, `CacheControl`)
- [x] Aware execution with enriched results (`AwareExecutionResult`)

**Required Types:** `ToolHealthStatus`, `HealthState`, `CostEstimate`, `QuotaStatus`, `RetryHints`, `ErrorCategory`, `CacheInfo`, `CacheControl`, `AwareExecutionResult`
**Required Functions:** `get_tool_health`, `estimate_cost`, `retry_hints`, `execute_aware`

#### Phase 5.2: Agent Autonomy (COMPLETE)
Enables agents to discover tools semantically and recover from failures.

- [x] Capability-based tool discovery (`CapabilityRegistry`, `Capability`)
- [x] Semantic tool metadata (`ToolMetadata`)
- [x] Fallback chains for graceful degradation (`FallbackChain`)
- [x] Recovery hints for failure modes (`RecoveryHint`, `RecoveryHintRegistry`)
- [x] Alternative tool suggestions (`AlternativeSuggestion`)
- [x] Degraded mode options (`DegradedModeOptions`)

**Required Types:** `CapabilityRegistry`, `Capability`, `ToolMetadata`, `FallbackChain`, `RecoveryHint`, `RecoveryHintRegistry`, `AlternativeSuggestion`, `DegradedModeOptions`
**Required Functions:** `find_by_capability`, `find_by_category`, `get_alternatives`

#### Phase 5.3: Ops Observability (COMPLETE)
Production-grade metrics and tracing for tool operations.

- [x] Per-tool latency histograms (`ToolMetricsCollector`, `ToolMetricsSnapshot`)
- [x] Global metrics aggregation (`GlobalMetricsSnapshot`, `LatencyHistogramSnapshot`)
- [x] Prometheus export (`export_prometheus`)
- [x] Structured audit logging (`ToolExecutionLog`)
- [x] OpenTelemetry span attributes (`ToolSpanAttributes`)

**Required Types:** `ToolMetricsCollector`, `ToolMetricsSnapshot`, `GlobalMetricsSnapshot`, `LatencyHistogramSnapshot`, `ToolExecutionLog`, `ToolSpanAttributes`
**Required Functions:** `record_execution`, `export_prometheus`, `to_otel_attributes`

#### Phase 5.4: Multi-Agent Coordination (COMPLETE)
Primitives for coordinating tool access across multiple agents.

- [x] Agent registry (`AgentRegistry`, `AgentInfo`, `AgentStatus`)
- [x] Tool locks with timeouts (`ToolLockManager`, `ToolLockGuard`)
- [x] Resource quota management (`ResourceQuotaManager`, `ResourceAllocation`)
- [x] Coordination events (`CoordinationEvent`)

**Required Types:** `AgentRegistry`, `AgentInfo`, `AgentStatus`, `AgentHandle`, `ToolLockManager`, `ToolLockGuard`, `LockError`, `ResourceQuotaManager`, `ResourceAllocation`, `CoordinationEvent`
**Required Functions:** `register`, `acquire`, `allocate`

#### Phase 5.5: Performance & Reliability (COMPLETE)
Infrastructure for reliable tool execution at scale.

- [x] Connection pooling (`ToolConnectionPool`, `PoolConfig`, `PooledConnection`)
- [x] Graceful shutdown with request draining (`GracefulShutdown`, `ShutdownSignal`)
- [x] Tool cancellation (`ToolCancellation`, `CancellableExecution`)

**Required Types:** `ToolConnectionPool`, `PoolConfig`, `PooledConnection`, `PoolError`, `PoolStatistics`, `GracefulShutdown`, `ShutdownSignal`, `ShutdownResult`, `ActiveRequestGuard`, `ToolCancellation`, `CancellableExecution`, `CancellationEvent`, `ActiveExecution`
**Required Functions:** `acquire`, `initiate`, `register`, `cancel`

---

## 8. Integration Requirements

The Phase 4-5 modules are implemented but require integration into the main execution flow.

### 8.1 ToolExecutor Integration

```rust
// Current: Tools detected but executed client-side
// Target: Tools executed server-side through ToolExecutor

// In chat_completions handler:
let tool_calls = detect_tool_calls(&output, model_family);
for call in tool_calls {
    // Phase 4.2: Execute through Beleth
    let result = executor.execute(&call, &context).await?;

    // Phase 4.3: Policy check happens inside execute()
    // Phase 5.1: Health/cost info available via execute_aware()
}
```

### 8.2 Coordination Integration

```rust
// Phase 5.4: Multi-agent coordination
let coordinator = AgentCoordinator::new();
let handle = coordinator.register_agent("planner");

// Acquire exclusive tool access
let _guard = coordinator.locks.acquire("file_write", &handle).await?;
// Tool execution...
// Lock released on guard drop
```

### 8.3 Reliability Integration

```rust
// Phase 5.5: Connection pooling and shutdown
let pool = ToolConnectionPool::new();
let shutdown = GracefulShutdown::new();

// In request handler:
let _request_guard = shutdown.register_request()?;
let _conn = pool.acquire("http_tool").await?;
// Handle request...

// On SIGTERM:
shutdown.initiate().await;
pool.shutdown();
```

---

## 9. Test Requirements

Per Agent-TDD methodology, tests MUST validate the **specification**, not just the current implementation.

### 9.1 Specification Compliance Tests

```rust
// These tests should FAIL if implementation deviates from spec
mod spec_compliance {
    #[test] fn spec_tool_choice_typed_enum()      // Not stringly-typed
    #[test] fn spec_tool_choice_mode_variants()   // None, Auto, Required
    #[test] fn spec_policy_denial_error_format()  // Phase 4.3 error format
}
```

### 9.2 Phase-by-Phase Tests

```rust
// Phase 1: Format Detection
#[test] fn phase_1_model_family_detection()
#[test] fn phase_1_qwen_format_detected()

// Phase 2: Tool Formatting
#[test] fn phase_2_tools_formatted_for_prompt()
#[test] fn phase_2_multiple_tools_formatted()

// Phase 3: Tool Detection
#[test] fn phase_3_streaming_tool_detection()
#[test] fn phase_3_parallel_tool_calls()

// Phase 4: Agent Runtime
#[test] fn phase_4_1_tool_choice_validation()
#[test] fn phase_4_2_tool_execution()
#[test] fn phase_4_3_policy_enforcement()

// Phase 5: Agent Infrastructure
#[test] fn phase_5_1_agent_awareness()
#[test] fn phase_5_2_capability_discovery()
#[test] fn phase_5_3_metrics_collection()
#[test] fn phase_5_4_multi_agent_coordination()
#[test] fn phase_5_5_graceful_shutdown()
```

### 9.3 Integration Tests

```rust
#[test] fn integration_full_server_side_execution()
#[test] fn integration_policy_blocks_dangerous_tool()
#[test] fn integration_fallback_on_tool_failure()
#[test] fn integration_multi_agent_lock_contention()
```

---

## 10. Files

### Core Modules (Phases 1-3)
- `src/tool_use.rs` - Tool formatting and detection
- `src/api_types.rs` - API wire types

### Agent Runtime (Phase 4)
- `src/tool_executor.rs` - Server-side execution (Phase 4.2)
- `src/tool_policy.rs` - Policy engine (Phase 4.3)

### Agent Infrastructure (Phase 5)
- `src/agent_awareness.rs` - Health, cost, retry hints (Phase 5.1)
- `src/agent_autonomy.rs` - Capability discovery, fallbacks (Phase 5.2)
- `src/tool_metrics.rs` - Prometheus metrics, logging (Phase 5.3)
- `src/agent_coordination.rs` - Multi-agent primitives (Phase 5.4)
- `src/tool_reliability.rs` - Pooling, shutdown, cancellation (Phase 5.5)

### Validation
- `src/spec_compliance.rs` - Automated spec compliance checks
- `src/validation.rs` - Request validation

---

## 11. Design Debt

This section documents known deviations from the specification that require remediation.

### 11.1 ToolChoice Stringly-Typed ✅ RESOLVED

**Status:** Resolved 2026-02-02

**Spec says:**
```rust
pub enum ToolChoiceMode { None, Auto, Required }
```

**Implementation now has:**
```rust
pub enum ToolChoiceMode {
    None,
    Auto,
    Required,
}

pub enum ToolChoice {
    Mode(ToolChoiceMode),  // Typed! Compile-time safety
    Tool(ToolChoiceFunction),
}
```

**Impact:** Compile-time validation of tool choice modes. Typos like `"auot"` are rejected at deserialization.

**Resolution:** Refactored to use typed enum with `#[serde(rename_all = "lowercase")]`. 951 tests passing.

### 11.2 DetectedToolCall.arguments Type ✅ RESOLVED

**Status:** Resolved 2026-02-02

**Spec says:** `arguments` should be `serde_json::Value` for structured access.

**Implementation now has:**
```rust
pub struct DetectedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,  // Structured! Direct access without parsing
}

impl DetectedToolCall {
    /// Get arguments as JSON string for API wire format
    pub fn arguments_string(&self) -> String { ... }
}
```

**Impact:** Direct structured access (`call.arguments["location"]`) without parsing. 954 tests passing.

### 11.3 Integration Wiring ✅ RESOLVED

**Status:** Resolved 2026-02-02

**Implementation adds to `AppState`:**
```rust
pub struct AppState {
    // ... existing fields ...

    // Tool calling (Phase 4-5)
    pub tool_executor: RwLock<Option<Arc<ToolExecutor>>>,
    pub agent_coordinator: Arc<AgentCoordinator>,
    pub tool_pool: Arc<ToolConnectionPool>,
    pub tool_metrics: Arc<ToolMetricsCollector>,
}
```

**Initialization:**
- `ToolMetricsCollector` for observability
- `ToolConnectionPool` with configurable max connections and idle timeout
- `AgentCoordinator` for multi-agent orchestration
- `ToolExecutor` field for lazy initialization with registered tools

**Impact:** All Phase 4-5 infrastructure now accessible from request handlers. 954 tests passing.

### 11.4 Non-Native Tool Calling Format

**Status:** Open (discovered 2026-02-04)

**Gap discovered during runtime testing:** When Qwen2.5-7B-Instruct receives tool
definitions in markdown-style format (the previous §5.1 format) without an explicit
system message, it sometimes outputs raw JSON instead of `<tool_call>` tagged responses.
Root cause: the prompt format diverges from the model's training template.

**Three components affected:**

1. **Server tool prompt** (`tool_use.rs:format_tools_qwen`): Uses markdown-style
   `## tool_name / Parameters:` format instead of `<tools></tools>` XML with JSON
   function definitions.

2. **Server multi-turn messages** (`server.rs` message builder):
   - Tool results formatted as `[Tool Result for {id}]: {content}` instead of
     `<tool_response>` tags.
   - Assistant messages with `tool_calls` don't reconstruct `<tool_call>` tags
     in content — they just clone the (often empty) content string.

3. **Beleth agent prompt** (`agent.rs:build_system_prompt`): Uses generic
   `Action: / Action Input:` text protocol regardless of model. Models with
   native tool calling support (Qwen2.5, Llama 3, etc.) produce more reliable
   results when prompted in their training format.

**Resolution plan:**
- Fix server-side format (§5.1 updated, §5.5 added) — affects `tool_use.rs` and `server.rs`
- Add model-aware formatting to Beleth (§5.6 added) — affects `agent.rs` and `tool.rs`
- Both generic and native formats remain supported; model family determines selection

---

## 12. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-01 | Initial draft. Gap discovered during Observer v0.2.0-rc prep. |
| 0.2.0 | 2026-02-01 | Phase 1 implementation complete. Added `tool_use.rs` module. |
| 0.2.1 | 2026-02-01 | Code review fixes: OnceLock for regex, safe UUID generation, tool_choice handling, system message injection. Documented Phase 1 limitations. |
| 0.3.0 | 2026-02-01 | Phase 2-3 complete. Llama/Mistral formats, streaming tool detection, agent-centric SSE, parallel_tool_calls, strict mode, deep JSON parsing. 97 tests passing. |
| 1.0.0 | 2026-02-02 | Phase 4-5 complete (implementation). Added agent runtime (4.1-4.3), agent infrastructure (5.1-5.5). Documented design debt. Integration pending. 949 tests passing. |
| 1.1.0 | 2026-02-02 | §11.1 RESOLVED: ToolChoice refactored from `String(String)` to typed `Mode(ToolChoiceMode)`. Compile-time safety achieved. 951 tests passing. |
| 1.2.0 | 2026-02-02 | §11.2 RESOLVED: DetectedToolCall.arguments refactored from `String` to `serde_json::Value`. Structured access without parsing. Added `arguments_string()` for API compat. 954 tests passing. |
| 1.3.0 | 2026-02-02 | §11.3 RESOLVED: Tool infrastructure wired into AppState. ToolExecutor, AgentCoordinator, ToolConnectionPool, ToolMetricsCollector now accessible from request handlers. All design debt retired. 954 tests passing. |
| 1.4.0 | 2026-02-04 | §11.4 OPENED: Non-native tool calling format discovered during runtime testing. §1 reframed from OpenAI-compatible to Infernum native runtime. §5.1 updated to Qwen2.5 native training format. §5.5 added for multi-turn conversation format. §5.6 added for Beleth model-aware tool calling. |
