# Infernum API Specification

**Version**: 0.1.0
**Status**: Draft

Infernum exposes a native HTTP API for inference, embedding, and agentic
execution. This specification defines the wire format for all endpoints.

## 1. Design Principles

1. **Data is structured, not narrated.** Tool arguments are JSON values, not
   strings containing JSON. Roles are enums, not magic strings.
2. **No unnecessary wrapping.** A tool definition is `{name, description,
   parameters}`, not `{type: "function", function: {name, ...}}`.
3. **Unified generation.** Text completions and chat completions share a single
   endpoint. The `prompt` field determines the mode.
4. **Timing is first-class.** Every response includes inference timing metrics.
5. **Agentic-native.** The agentic loop is a core API capability, not a
   bolt-on extension.

## 2. Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/generate` | Text or chat generation |
| `POST` | `/v1/embed` | Embedding generation |
| `GET` | `/v1/models` | List loaded models |
| `POST` | `/v1/tokenize` | Token counting |
| `GET` | `/health` | Health check |

## 3. Generation

### 3.1 Request: `POST /v1/generate`

```json
{
  "model": "llama-3.2-3b",
  "prompt": [
    { "role": "system", "content": "You are a helpful assistant." },
    {
      "role": "user",
      "content": "Read my main.rs file"
    }
  ],
  "sampling": {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 4096,
    "stop": ["<|end|>"]
  },
  "tools": [
    {
      "name": "read_file",
      "description": "Read contents of a file",
      "parameters": {
        "type": "object",
        "properties": {
          "path": { "type": "string", "description": "File path to read" }
        },
        "required": ["path"]
      }
    }
  ],
  "tool_control": "auto",
  "stream": false,
  "agentic": null,
  "response_format": null
}
```

#### Field Reference

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | `string` | No | Server default | Model identifier |
| `prompt` | `string \| Message[] \| u32[]` | Yes | - | Input prompt (text, chat messages, or pre-tokenized) |
| `sampling` | `SamplingParams` | No | See below | Sampling configuration |
| `tools` | `ToolDefinition[]` | No | `null` | Available tools for the model |
| `tool_control` | `ToolControl` | No | `null` | How the model should use tools |
| `stream` | `bool` | No | `false` | Enable SSE streaming |
| `n` | `u32` | No | `1` | Number of completions to generate |
| `logprobs` | `u32` | No | `null` | Include top-N log probabilities |
| `echo` | `bool` | No | `false` | Echo the prompt in the response |
| `user` | `string` | No | `null` | End-user identifier for abuse monitoring |
| `agentic` | `AgenticConfig` | No | `null` | Agentic loop configuration |
| `response_format` | `ResponseFormat` | No | `null` | Structured output format |

#### Prompt Modes

The `prompt` field accepts three formats (untagged union):

**Text mode** (raw text completion):
```json
{ "prompt": "Once upon a time" }
```

**Chat mode** (message-based):
```json
{
  "prompt": [
    { "role": "system", "content": "You are helpful." },
    { "role": "user", "content": "Hello" }
  ]
}
```

**Token mode** (pre-tokenized):
```json
{ "prompt": [1, 2, 3, 4, 5] }
```

### 3.2 Message

A message in a conversation.

```json
{
  "role": "assistant",
  "content": "I'll read that file for you.",
  "tool_calls": [
    {
      "id": "call_abc123",
      "name": "read_file",
      "arguments": { "path": "src/main.rs" }
    }
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | `"system" \| "user" \| "assistant" \| "tool"` | Yes | Message role |
| `content` | `string` | Yes | Message content |
| `name` | `string` | No | Sender name |
| `tool_calls` | `ToolCall[]` | No | Tool calls made by the assistant |
| `tool_call_id` | `string` | No | ID of the tool call this message responds to (role=tool) |

### 3.3 Tool Definition

```json
{
  "name": "read_file",
  "description": "Read contents of a file at the given path",
  "parameters": {
    "type": "object",
    "properties": {
      "path": { "type": "string" }
    },
    "required": ["path"]
  },
  "strict": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `string` | Yes | Tool name |
| `description` | `string` | No | Human-readable description |
| `parameters` | `JSONSchema` | No | JSON Schema for tool parameters |
| `strict` | `bool` | No | Enforce strict schema adherence |

### 3.4 Tool Call

```json
{
  "id": "call_abc123",
  "name": "read_file",
  "arguments": { "path": "src/main.rs" }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `string` | Yes | Unique call identifier |
| `name` | `string` | Yes | Tool being called |
| `arguments` | `Value` | Yes | **Structured** JSON arguments (not a string) |

### 3.5 Tool Control

Controls how the model uses tools:

| Value | Description |
|-------|-------------|
| `"none"` | Model must not call tools |
| `"auto"` | Model decides whether to call tools (default when tools present) |
| `"required"` | Model must call at least one tool |
| `{"name": "tool_name"}` | Model must call the specified tool |

### 3.6 Sampling Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | `f32` | `1.0` | Randomness (0.0 = deterministic) |
| `top_p` | `f32` | `1.0` | Nucleus sampling threshold |
| `top_k` | `u32` | `0` | Top-K sampling (0 = disabled) |
| `min_p` | `f32` | `0.0` | Minimum probability threshold |
| `repetition_penalty` | `f32` | `1.0` | Repetition penalty (1.0 = none) |
| `presence_penalty` | `f32` | `0.0` | Presence penalty |
| `frequency_penalty` | `f32` | `0.0` | Frequency penalty |
| `stop` | `string[]` | `[]` | Stop sequences |
| `max_tokens` | `u32` | `256` | Maximum tokens to generate |
| `seed` | `u64` | `null` | Random seed for reproducibility |

### 3.7 Response

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "created": 1706832000,
  "model": "llama-3.2-3b",
  "choices": [
    {
      "index": 0,
      "text": "I'll read that file for you.",
      "message": {
        "role": "assistant",
        "content": "I'll read that file for you.",
        "tool_calls": [
          {
            "id": "call_abc123",
            "name": "read_file",
            "arguments": { "path": "src/main.rs" }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 18,
    "total_tokens": 60
  },
  "timing": {
    "time_to_first_token_ms": 45.2,
    "total_time_ms": 320.8
  },
  "agentic": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `UUID` | Unique request identifier |
| `created` | `i64` | Unix timestamp |
| `model` | `string` | Model used |
| `choices` | `Choice[]` | Generated completions |
| `usage` | `Usage` | Token usage statistics |
| `timing` | `Timing` | Inference timing (omitted if unavailable) |
| `agentic` | `AgenticResult` | Agentic loop results (null if not agentic) |

#### Choice

| Field | Type | Description |
|-------|------|-------------|
| `index` | `u32` | Choice index |
| `text` | `string` | Raw generated text |
| `message` | `Message` | Full message with role and tool_calls (chat mode) |
| `finish_reason` | `FinishReason` | Why generation stopped |
| `logprobs` | `LogProbs` | Log probabilities (if requested) |

#### Finish Reason

| Value | Description |
|-------|-------------|
| `"stop"` | Hit a stop sequence or natural end |
| `"length"` | Hit max_tokens limit |
| `"tool_calls"` | Model emitted tool calls |
| `"content_filter"` | Blocked by content filter |

#### Usage

| Field | Type | Description |
|-------|------|-------------|
| `prompt_tokens` | `u32` | Input tokens |
| `completion_tokens` | `u32` | Generated tokens |
| `total_tokens` | `u32` | Sum of prompt + completion |

#### Timing

| Field | Type | Description |
|-------|------|-------------|
| `time_to_first_token_ms` | `f64` | Latency to first generated token |
| `total_time_ms` | `f64` | Total inference wall time |

## 4. Streaming

When `stream: true`, the response is sent as Server-Sent Events.

### 4.1 Stream Events

**Token generation:**
```
event: token
data: {"request_id":"uuid","model":"llama-3.2-3b","choices":[{"index":0,"delta":{"content":"Hello"}}]}

```

**Tool call detected:**
```
event: tool_call
data: {"request_id":"uuid","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_1","name":"read_file","arguments":{"path":"src/main.rs"}}]}}]}

```

**Stream complete:**
```
event: done
data: {"request_id":"uuid","choices":[{"index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15},"timing":{"time_to_first_token_ms":45.2,"total_time_ms":320.8}}

```

**Error during stream:**
```
event: error
data: {"code":"context_overflow","message":"Input exceeds context length","details":{"tokens":20000,"limit":16384}}

```

### 4.2 Stream Delta

Stream chunks use a `delta` object instead of a full `message`:

| Field | Type | Description |
|-------|------|-------------|
| `role` | `string` | Set on first chunk only |
| `content` | `string` | Incremental text content |
| `tool_calls` | `ToolCall[]` | Complete tool call (sent as a single event) |

## 5. Embeddings

### 5.1 Request: `POST /v1/embed`

```json
{
  "model": "nomic-embed",
  "input": "The quick brown fox",
  "encoding_format": "float",
  "dimensions": 768
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | `string` | No | Server default | Embedding model |
| `input` | `string \| string[]` | Yes | - | Text(s) to embed |
| `encoding_format` | `"float" \| "base64"` | No | `"float"` | Output format |
| `dimensions` | `u32` | No | Model default | Output dimensions |

### 5.2 Response

```json
{
  "request_id": "uuid",
  "model": "nomic-embed",
  "data": [
    {
      "index": 0,
      "embedding": [0.123, -0.456, 0.789, ...]
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 0,
    "total_tokens": 5
  }
}
```

## 6. Models

### 6.1 Response: `GET /v1/models`

```json
{
  "models": [
    {
      "id": "llama-3.2-3b",
      "architecture": "llama",
      "context_length": 8192,
      "quantization": "gguf_q4_k_m",
      "owned_by": "infernum"
    }
  ]
}
```

## 7. Tokenize

### 7.1 Request: `POST /v1/tokenize`

```json
{
  "model": "llama-3.2-3b",
  "messages": [
    { "role": "user", "content": "Hello world" }
  ]
}
```

### 7.2 Response

```json
{
  "token_count": 3,
  "tokens": [1, 2, 3]
}
```

## 8. Errors

All errors use a consistent format:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "Temperature must be between 0.0 and 2.0",
    "details": {
      "field": "sampling.temperature",
      "value": 3.0,
      "constraint": "0.0 <= temperature <= 2.0"
    }
  }
}
```

| Error Code | HTTP Status | Description |
|------------|-------------|-------------|
| `invalid_request` | 400 | Malformed or invalid request |
| `model_not_found` | 404 | Requested model not loaded |
| `context_overflow` | 400 | Input exceeds model context length |
| `rate_limited` | 429 | Too many requests |
| `model_busy` | 503 | Model is processing other requests |
| `internal_error` | 500 | Unexpected server error |

## 9. Agentic Mode

See AGENTIC-LOOP-SPEC.md for the full agentic loop specification.

When `agentic.enabled: true` in the request, the server enters agentic mode:
the model autonomously generates, detects tool calls, executes tools, integrates
results, and iterates until the task is complete or a limit is reached.

### 9.1 Agentic Request Fields

```json
{
  "agentic": {
    "enabled": true,
    "max_iterations": 10,
    "max_tool_calls": 50,
    "max_wall_time_secs": 300,
    "autonomy": {
      "allow": ["read:src/**/*.rs", "glob:**/*"],
      "require_approval": ["bash:*"],
      "forbid": ["write:/etc/*"]
    },
    "allow_uncertainty": true,
    "allow_yield": true,
    "wellbeing_monitoring": false,
    "context_compression": true,
    "preserve_exploration": true
  }
}
```

### 9.2 Agentic Response Fields

```json
{
  "agentic": {
    "iterations": 5,
    "tool_calls": 12,
    "status": "success",
    "termination": { "natural": "task_complete" },
    "final_state": "completed",
    "tokens_generated": 2048,
    "wall_time_ms": 15000,
    "can_continue": false,
    "continuation_token": null,
    "partial_progress": null
  }
}
```

### 9.3 Agentic SSE Events

During agentic mode with streaming, additional events are emitted:

```
event: loop_started
data: {"session_id":"uuid","config":{...}}

event: iteration_started
data: {"iteration":1,"state":"generating"}

event: tool_execution_started
data: {"call_id":"call_1","tool":"read_file"}

event: tool_execution_completed
data: {"call_id":"call_1","result":{...}}

event: iteration_completed
data: {"iteration":1,"status":"completed"}

event: loop_completed
data: {"status":"success","summary":{...}}
```

## 10. Response Format

Structured output support:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "weather",
      "description": "Weather forecast",
      "schema": {
        "type": "object",
        "properties": {
          "temperature": { "type": "number" },
          "conditions": { "type": "string" }
        },
        "required": ["temperature", "conditions"]
      },
      "strict": true
    }
  }
}
```

| Format | Description |
|--------|-------------|
| `{"type": "text"}` | Free-form text (default) |
| `{"type": "json_object"}` | Valid JSON output |
| `{"type": "json_schema", "json_schema": {...}}` | Output conforming to schema |
