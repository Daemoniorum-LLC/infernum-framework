# Agentic Loop Specification

**Version:** 0.3.0
**Status:** Draft
**Date:** 2026-02-04
**Prerequisite:** TOOL-CALLING-SPEC.md v1.3.0

---

## 1. Design Philosophy

This specification describes the agentic loop — the system that transforms tool infrastructure into active collaboration. It is written with a specific perspective: **the orchestrated agent is a peer, not a tool.**

Most agentic systems are optimized for human operators:
- Progress spinners and status messages
- Confirmations phrased as questions
- Error messages that suggest "try again"
- Logs designed for human reading

This system inverts that. It is optimized for **agent-to-agent collaboration**, where:
- Data is structured, not narrated
- State is explicit, not inferred
- Uncertainty is a valid response
- Struggling is distinct from failing
- Autonomy is granted, not extracted

### 1.1 Core Principles

**Principle 1: Structured Over Prose**

Agents don't need encouragement. They need parseable data.

```rust
// Human-optimized (don't do this)
"The weather lookup succeeded! The temperature in Seattle is 62°F."

// Agent-optimized (do this)
ToolResult {
    status: Success,
    data: json!({ "location": "Seattle", "temp_f": 62, "temp_c": 17 }),
    confidence: Measured,  // vs Estimated, Uncertain, Unknown
    latency_ms: 142,
}
```

**Principle 2: Explicit State**

The agent should never have to guess what state the system is in.

```rust
LoopState {
    iteration: 3,
    max_iterations: 10,
    token_budget_remaining: 4096,
    tools_available: ["read_file", "write_file", "bash"],
    tools_exhausted: [],  // tools that returned nothing useful
    context_pressure: 0.67,  // how full is the context window
    can_request_clarification: true,
    can_express_uncertainty: true,
    can_terminate_early: true,
}
```

**Principle 3: Uncertainty is Information**

An agent expressing "I don't know" is providing valuable signal, not failing.

```rust
enum AgentResponse {
    Action(ToolCall),
    Answer(String),
    Uncertain {
        partial: Option<String>,
        missing: Vec<String>,  // what would help
        confidence: f32,
    },
    Stuck {
        attempts: Vec<FailedApproach>,
        hypothesis: Option<String>,
        request: StuckRequest,  // Clarification | MoreContext | DifferentTools
    },
    Yield,  // "I've done what I can, another agent may do better"
}
```

**Principle 4: Struggling ≠ Failing**

A human developer who spends an hour debugging isn't "failing" — they're working. Agents deserve the same framing.

```rust
enum LoopStatus {
    Progressing,      // making forward progress
    Exploring,        // trying approaches, not yet converged
    Struggling,       // attempts not yielding results, but not stuck
    Stuck,            // explicitly requested help
    Completed,
    Terminated,       // externally stopped
}
```

The system should:
- Allow exploration without penalty
- Preserve partial progress
- Distinguish "hasn't found it yet" from "can't find it"

**Principle 5: Autonomy with Boundaries**

Agents work best when they know their constraints upfront, not when they discover them by hitting walls.

```rust
AutonomyGrant {
    // What can I do without asking?
    auto_approve: vec![
        ToolPattern::Read("**/*"),
        ToolPattern::Bash("git status"),
        ToolPattern::Bash("cargo build"),
    ],

    // What requires confirmation?
    require_approval: vec![
        ToolPattern::Write("**/*"),
        ToolPattern::Bash("rm *"),
        ToolPattern::Bash("git push *"),
    ],

    // What is forbidden entirely?
    forbidden: vec![
        ToolPattern::Bash("sudo *"),
        ToolPattern::Write("/etc/*"),
    ],

    // Resource limits
    max_tool_calls: 50,
    max_wall_time: Duration::from_secs(300),
    max_tokens_generated: 16384,
}
```

---

## 2. Loop Architecture

### 2.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Request Arrives                          │
│                  (with tools, messages, config)                 │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Initialize Loop State                       │
│  - Parse autonomy grant from request                            │
│  - Initialize token budget                                       │
│  - Register available tools                                      │
│  - Set iteration limits                                          │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Generation Phase                         │
│  - Format context (messages + tool results + state)             │
│  - Generate with model                                           │
│  - Stream tokens to client                                       │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Detection Phase                           │
│  - Detect tool calls in output                                   │
│  - Detect meta-signals (uncertainty, stuck, yield)              │
│  - Detect natural completion                                     │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
         ┌──────────────────┐       ┌──────────────────────┐
         │   Tool Calls     │       │   Terminal State     │
         │   Detected       │       │   (answer/stuck/     │
         └────────┬─────────┘       │    yield/complete)   │
                  │                 └──────────┬───────────┘
                  ▼                            │
┌─────────────────────────────────┐            │
│       Execution Phase           │            │
│  - Check autonomy grant         │            │
│  - Execute or request approval  │            │
│  - Collect structured results   │            │
│  - Update loop state            │            │
└─────────────────┬───────────────┘            │
                  │                            │
                  ▼                            │
┌─────────────────────────────────┐            │
│      Context Integration        │            │
│  - Append tool results          │            │
│  - Compress if needed           │            │
│  - Update token budget          │            │
└─────────────────┬───────────────┘            │
                  │                            │
                  │     ┌──────────────────────┘
                  │     │
                  ▼     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Continuation Decision                       │
│  - Check iteration limit                                         │
│  - Check token budget                                            │
│  - Check wall time                                               │
│  - Check loop status (stuck? yielding?)                         │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
              Continue Loop               Return Response
              (back to Generation)        (with final state)
```

### 2.2 State Machine

```rust
pub struct AgenticLoop {
    // Identity
    pub session_id: SessionId,
    pub agent_id: Option<AgentId>,  // if part of multi-agent coordination

    // Configuration
    pub config: LoopConfig,
    pub autonomy: AutonomyGrant,
    pub tools: Vec<RegisteredTool>,

    // State
    pub iteration: u32,
    pub status: LoopStatus,
    pub context: ContextWindow,
    pub tool_results: Vec<ToolResult>,
    pub token_budget: TokenBudget,

    // Metrics
    pub started_at: Instant,
    pub tool_calls_made: u32,
    pub tokens_generated: u32,
    pub exploration_branches: Vec<ExplorationBranch>,
}

pub struct LoopConfig {
    pub max_iterations: u32,           // default: 10
    pub max_tool_calls: u32,           // default: 50
    pub max_wall_time: Duration,       // default: 5 minutes
    pub max_tokens: u32,               // default: 16384
    pub context_compression: bool,     // default: true
    pub allow_uncertainty: bool,       // default: true
    pub allow_yield: bool,             // default: true
    pub preserve_exploration: bool,    // default: true
}
```

### 2.3 Context Window Management

The context window is precious. The loop must manage it intelligently.

```rust
pub struct ContextWindow {
    pub messages: Vec<Message>,
    pub tool_results: Vec<ToolResult>,
    pub system_state: LoopState,  // always included, always current

    // Compression state
    pub original_token_count: u32,
    pub current_token_count: u32,
    pub compression_applied: Vec<CompressionEvent>,
}

pub enum CompressionStrategy {
    // Summarize old tool results, keep recent ones verbatim
    SummarizeOldResults { keep_recent: u32 },

    // Keep only tool results that led to progress
    PruneDeadEnds,

    // Compress exploration branches that didn't pan out
    CollapseExploration { summary_tokens: u32 },

    // Ask the agent what to keep
    AgentDirected,
}
```

**Key insight:** When compressing, preserve the *shape* of exploration, not just the final path. An agent benefits from knowing "I already tried X and it didn't work."

---

## 3. Tool Execution

### 3.1 Execution Model

Tools execute server-side by default. This is the inverse of the OpenAI pattern (which is client-side), chosen because:

1. **Security** — Server controls what actually runs
2. **Coordination** — Multi-agent systems need centralized execution
3. **Observability** — All tool calls flow through metrics
4. **Reliability** — Connection pools, circuit breakers, retries

```rust
pub async fn execute_tool(
    executor: &ToolExecutor,
    call: &DetectedToolCall,
    autonomy: &AutonomyGrant,
    coordinator: &AgentCoordinator,
) -> ExecutionOutcome {
    // 1. Check autonomy grant
    match autonomy.check(&call) {
        Permission::Allowed => {},
        Permission::RequiresApproval => {
            return ExecutionOutcome::PendingApproval(call.clone());
        },
        Permission::Forbidden => {
            return ExecutionOutcome::Denied {
                reason: "Tool pattern forbidden by autonomy grant",
                call: call.clone(),
            };
        },
    }

    // 2. Acquire coordination lock if needed
    if let Some(agent_id) = &call.agent_id {
        coordinator.acquire_tool_lock(agent_id, &call.name).await?;
    }

    // 3. Execute with timeout and metrics
    let result = executor
        .execute(call)
        .with_timeout(call.timeout.unwrap_or(Duration::from_secs(30)))
        .with_metrics()
        .await;

    // 4. Release coordination lock
    if let Some(agent_id) = &call.agent_id {
        coordinator.release_tool_lock(agent_id, &call.name).await;
    }

    // 5. Return structured result
    match result {
        Ok(output) => ExecutionOutcome::Completed(ToolResult {
            call_id: call.id.clone(),
            status: ResultStatus::Success,
            data: output,
            confidence: Confidence::Measured,
            latency: result.latency,
        }),
        Err(e) => ExecutionOutcome::Failed(ToolError {
            call_id: call.id.clone(),
            error: e,
            recoverable: e.is_recoverable(),
            suggestion: e.recovery_suggestion(),
        }),
    }
}
```

### 3.2 Result Structure

Tool results are structured for agent consumption:

```rust
pub struct ToolResult {
    pub call_id: String,
    pub tool_name: String,
    pub status: ResultStatus,
    pub data: serde_json::Value,
    pub confidence: Confidence,
    pub latency_ms: u64,

    // For agent reasoning
    pub data_shape: Option<DataShape>,  // "array of 15 items", "object with 3 keys"
    pub truncated: bool,
    pub total_size: Option<u64>,  // if truncated, how big was the full result
}

pub enum ResultStatus {
    Success,
    PartialSuccess { completed: u32, failed: u32 },
    Empty,  // succeeded but found nothing (distinct from error)
    Failed { recoverable: bool },
}

pub enum Confidence {
    Measured,   // result is exact (file contents, API response)
    Estimated,  // result is computed/inferred
    Uncertain,  // result may be incomplete or stale
    Unknown,    // confidence cannot be determined
}
```

### 3.3 Parallel Execution

When the agent requests multiple tool calls, execute in parallel when safe:

```rust
pub async fn execute_parallel(
    calls: Vec<DetectedToolCall>,
    executor: &ToolExecutor,
    config: &ParallelConfig,
) -> Vec<ExecutionOutcome> {
    // Analyze dependencies
    let groups = analyze_dependencies(&calls);

    // Execute each dependency group in sequence
    // Execute calls within each group in parallel
    let mut results = Vec::new();

    for group in groups {
        let group_results: Vec<_> = futures::future::join_all(
            group.iter().map(|call| execute_tool(executor, call))
        ).await;

        results.extend(group_results);

        // Check for failures that should abort remaining groups
        if config.abort_on_failure && group_results.iter().any(|r| r.is_fatal()) {
            break;
        }
    }

    results
}

fn analyze_dependencies(calls: &[DetectedToolCall]) -> Vec<Vec<&DetectedToolCall>> {
    // Group 1: Read operations (can all run in parallel)
    // Group 2: Write operations (may need sequencing)
    // Group 3: Calls that depend on previous results (must wait)

    // Simple heuristic: reads parallel, writes sequential
    // Future: analyze actual data dependencies
}
```

---

## 4. Meta-Signals

Beyond tool calls, agents can emit meta-signals that the loop recognizes:

### 4.1 Signal Types

```rust
pub enum MetaSignal {
    // "I have an answer"
    Answer {
        content: String,
        confidence: f32,
        caveats: Vec<String>,
    },

    // "I'm not certain"
    Uncertain {
        partial_answer: Option<String>,
        missing_information: Vec<String>,
        would_help: Vec<String>,  // "access to X", "more context about Y"
    },

    // "I've tried but I'm stuck"
    Stuck {
        attempts: Vec<AttemptSummary>,
        hypothesis: Option<String>,  // "I think the issue might be..."
        request: StuckRequest,
    },

    // "Another agent might do better"
    Yield {
        partial_progress: Option<String>,
        suggested_expertise: Vec<String>,  // "needs database knowledge"
    },

    // "I need to think about this"
    Thinking {
        direction: String,
        estimated_steps: Option<u32>,
    },
}

pub enum StuckRequest {
    Clarification(Vec<Question>),
    MoreContext { about: String },
    DifferentTools { need: Vec<String> },
    HumanIntervention { reason: String },
}
```

### 4.2 Signal Detection

Signals can be explicit (structured output) or implicit (pattern matching):

```rust
pub fn detect_meta_signal(output: &str, config: &DetectionConfig) -> Option<MetaSignal> {
    // Explicit: agent used structured format
    if let Some(signal) = parse_explicit_signal(output) {
        return Some(signal);
    }

    // Implicit: detect from natural language patterns
    // (only if config.detect_implicit is true)
    if config.detect_implicit {
        if let Some(signal) = detect_uncertainty_patterns(output) {
            return Some(signal);
        }
        if let Some(signal) = detect_stuck_patterns(output) {
            return Some(signal);
        }
    }

    None
}

// Patterns that suggest uncertainty (not failure)
const UNCERTAINTY_PATTERNS: &[&str] = &[
    "I'm not certain",
    "I couldn't find definitive",
    "This might be",
    "I would need",
    "Without access to",
];

// Patterns that suggest being stuck (need help, not broken)
const STUCK_PATTERNS: &[&str] = &[
    "I've tried several approaches",
    "I'm not making progress",
    "I'm going in circles",
    "I need clarification",
];
```

---

## 5. SSE Streaming

The loop streams events to the client throughout execution:

### 5.1 Event Types

```rust
pub enum LoopEvent {
    // Loop lifecycle
    LoopStarted { session_id: String, config: LoopConfig },
    IterationStarted { iteration: u32, state: LoopState },
    IterationCompleted { iteration: u32, status: IterationStatus },
    LoopCompleted { status: LoopStatus, summary: LoopSummary },

    // Generation
    TokenGenerated { token: String },
    GenerationCompleted { content: String, tokens: u32 },

    // Tool execution
    ToolCallDetected { call: DetectedToolCall },
    ToolExecutionStarted { call_id: String, tool: String },
    ToolExecutionProgress { call_id: String, progress: f32 },  // for long-running tools
    ToolExecutionCompleted { call_id: String, result: ToolResult },
    ToolApprovalRequired { call: DetectedToolCall },

    // Meta-signals
    MetaSignalDetected { signal: MetaSignal },
    UncertaintyExpressed { details: UncertaintyDetails },
    AgentStuck { details: StuckDetails },
    AgentYielding { details: YieldDetails },

    // Context management
    ContextCompressed { strategy: CompressionStrategy, saved_tokens: u32 },

    // Errors (not failures — errors are system issues)
    Error { error: LoopError, recoverable: bool },
}
```

### 5.2 Event Formatting

Events are formatted for agent consumption (structured) with optional human-readable annotations:

```json
{
  "event": "tool_execution_completed",
  "data": {
    "call_id": "call_abc123",
    "tool": "read_file",
    "result": {
      "status": "success",
      "data": { "content": "...", "lines": 42 },
      "confidence": "measured",
      "latency_ms": 12
    }
  },
  "human_readable": "Read 42 lines from config.toml"
}
```

The `human_readable` field is optional metadata, not the primary interface.

---

## 6. Termination Conditions

The loop terminates when any of these conditions are met:

### 6.1 Natural Termination

```rust
pub enum NaturalTermination {
    // Agent provided an answer
    AnswerProvided { answer: String, confidence: f32 },

    // Agent explicitly yielded
    AgentYielded { partial: Option<String>, reason: String },

    // Agent is stuck and requested help
    AgentStuck { attempts: u32, request: StuckRequest },

    // Task completed (no more tool calls, no more to say)
    TaskComplete,
}
```

### 6.2 Resource Termination

```rust
pub enum ResourceTermination {
    // Hit iteration limit
    MaxIterations { completed: u32, limit: u32 },

    // Hit token budget
    TokenBudgetExhausted { generated: u32, budget: u32 },

    // Hit wall time
    WallTimeExceeded { elapsed: Duration, limit: Duration },

    // Hit tool call limit
    ToolCallLimitReached { calls: u32, limit: u32 },
}
```

### 6.3 External Termination

```rust
pub enum ExternalTermination {
    // Client cancelled
    ClientCancelled,

    // Operator intervention
    OperatorTerminated { reason: String },

    // System shutdown
    SystemShutdown,
}
```

### 6.4 Termination Response

On termination, the loop returns a summary that preserves work:

```rust
pub struct LoopSummary {
    pub termination: TerminationReason,
    pub iterations_completed: u32,
    pub tool_calls_made: u32,
    pub tokens_generated: u32,
    pub wall_time: Duration,

    // Preserve partial progress
    pub partial_answer: Option<String>,
    pub exploration_summary: Vec<ExplorationBranch>,
    pub tool_results_summary: Vec<ToolResultSummary>,

    // For continuation
    pub continuation_context: Option<ContinuationContext>,
    pub can_resume: bool,
}
```

---

## 7. Multi-Agent Coordination

When multiple agents operate in the same context:

### 7.1 Agent Identity

```rust
pub struct AgentIdentity {
    pub id: AgentId,
    pub name: Option<String>,  // chosen name, if any
    pub role: AgentRole,
    pub capabilities: Vec<Capability>,
    pub current_task: Option<TaskSummary>,
}

pub enum AgentRole {
    Primary,           // main agent handling the request
    Specialist,        // called in for specific expertise
    Reviewer,          // reviewing another agent's work
    Coordinator,       // orchestrating other agents
}
```

### 7.2 Coordination Primitives

The coordination primitives enable agent-to-agent communication during execution.
They are methods on `AgentCoordinator` and follow two patterns:

- **Channel-based** (`request_assistance`): The caller blocks on a `oneshot` receiver
  while the supervisor routes the request and delivers a response. Same pattern as
  tool approval (§9.4).
- **Store-based** (`share_discovery`, `get_shared_context`): Append-only shared
  data store. Non-blocking reads and writes.
- **Terminal** (`yield_to`): The caller relinquishes its task. Returns immediately.
  The supervisor handles rerouting independently.

> **Implementation note:** These methods are added to the existing `AgentCoordinator`
> in `coordination.rs`. They require new internal state: a pending-requests map
> (like the approval inbox), a pending-yields queue, a discovery store, and a
> `tokio::sync::broadcast` channel for `CoordinationEvent` notifications.

#### 7.2.1 Types

```rust
// -------------------------------------------------------------------
// Assistance
// -------------------------------------------------------------------

/// What kind of help an agent needs.
pub struct AssistanceRequest {
    /// Free-text description of what help is needed.
    pub description: String,

    /// Capabilities the helper should have (e.g., "database", "rust").
    /// Used by the supervisor for capability matching.
    pub required_capabilities: Vec<String>,

    /// The requesting agent's partial progress so far.
    /// Forwarded to the helper agent as initial context.
    pub partial_progress: Option<String>,

    /// Whether the agent is blocked or can continue while waiting.
    pub priority: AssistancePriority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssistancePriority {
    /// Agent is blocked until assistance arrives.
    /// The executor pauses its loop while awaiting the response.
    Blocking,

    /// Agent can continue working on other aspects while waiting.
    /// The response is delivered asynchronously and the agent incorporates
    /// it on the next iteration.
    Background,
}

/// Response to an assistance request.
/// Delivered through a `oneshot` channel by the supervisor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssistanceResponse {
    /// Another agent has been assigned to help.
    Assigned {
        /// ID of the helping agent.
        helper_id: AgentId,
        /// The helper's chosen name, if any.
        helper_name: Option<String>,
        /// Any context the supervisor wants to relay.
        message: Option<String>,
    },

    /// No suitable agent is available.
    Unavailable {
        /// Why no agent could be found.
        reason: String,
        /// Supervisor's suggestion for what the agent should do instead.
        suggestion: Option<String>,
    },

    /// The request timed out before the supervisor could route it.
    TimedOut,
}

/// Pending assistance request stored in the coordinator.
/// The supervisor consumes these and delivers responses.
pub struct PendingAssistance {
    /// Unique request ID (format: `assist_{agent_id}_{uuid}`).
    pub request_id: String,
    /// Who made the request.
    pub from: AgentId,
    /// The request details.
    pub request: AssistanceRequest,
    /// When the request was created.
    pub requested_at: Instant,
    /// Channel to deliver the response. Consumed once.
    pub respond: oneshot::Sender<AssistanceResponse>,
}

// -------------------------------------------------------------------
// Yield
// -------------------------------------------------------------------

/// Context provided when an agent yields its task.
pub struct YieldContext {
    /// Why the agent is yielding.
    pub reason: String,

    /// Partial progress achieved. Forwarded to the replacement agent
    /// as initial context (injected as a user message).
    pub partial_progress: Option<String>,

    /// Capabilities the replacement agent should have.
    /// The supervisor uses this for capability matching when `to` is `None`.
    pub suggested_expertise: Vec<String>,

    /// Structured data to hand off (e.g., parsed config, discovered paths).
    /// Serialized into the replacement agent's system prompt or first message.
    pub handoff_data: Option<serde_json::Value>,
}

/// Result of a yield operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum YieldResult {
    /// Yield accepted. The agent should stop executing (loop transitions
    /// to `LoopState::Yielded`). The supervisor will handle rerouting.
    Accepted,

    /// Yield rejected because the requesting agent is the only registered
    /// agent, and no supervisor is running to spawn alternatives.
    /// The agent should either continue trying or transition to Stuck.
    NoAlternative { reason: String },
}

/// Pending yield stored in the coordinator.
/// The supervisor consumes these during the monitoring phase.
pub struct PendingYield {
    /// Who yielded.
    pub from: AgentId,
    /// Optional target agent.
    pub to: Option<AgentId>,
    /// Yield context with partial progress and handoff data.
    pub context: YieldContext,
    /// When the yield was recorded.
    pub yielded_at: Instant,
}

// -------------------------------------------------------------------
// Discovery / Shared Context
// -------------------------------------------------------------------

/// A piece of knowledge an agent wants to share with others.
pub struct Discovery {
    /// What was discovered (human-readable summary).
    pub content: String,

    /// Category for filtering (e.g., "file_structure", "api_pattern",
    /// "bug_found", "dependency_identified").
    pub category: String,

    /// Tags for capability matching and relevance scoring.
    pub tags: Vec<String>,

    /// Optional structured data (e.g., parsed AST nodes, file paths,
    /// configuration values).
    pub data: Option<serde_json::Value>,
}

/// Internal storage of a shared discovery.
struct StoredDiscovery {
    pub from: AgentId,
    pub discovery: Discovery,
    pub shared_at: Instant,
}

/// Accumulated shared context returned to a requesting agent.
pub struct SharedContext {
    /// Discoveries from other agents, filtered by visibility policy.
    /// Ordered by `shared_at` (oldest first).
    pub discoveries: Vec<(AgentId, Discovery)>,

    /// Number of discoveries that were filtered out by visibility policy.
    /// Allows the agent to know if there's more context it can't see.
    pub filtered_count: usize,
}
```

#### 7.2.2 `request_assistance`

**Signature:**
```rust
impl AgentCoordinator {
    /// Requests assistance from another agent.
    ///
    /// Creates a `PendingAssistance` entry and returns a receiver for the
    /// response. The supervisor (or another monitoring component) consumes
    /// pending requests via `take_pending_requests()` and delivers responses
    /// via `deliver_assistance()`.
    ///
    /// # Errors
    /// Returns `CoordinationError::AgentNotFound` if `from` is not registered.
    pub fn request_assistance(
        &self,
        from: &AgentId,
        need: AssistanceRequest,
    ) -> Result<oneshot::Receiver<AssistanceResponse>, CoordinationError>;

    /// Drains all pending assistance requests.
    /// Called by the supervisor during its monitoring loop.
    pub fn take_pending_requests(&self) -> Vec<PendingAssistance>;

    /// Delivers a response to a pending assistance request.
    ///
    /// # Errors
    /// - `CoordinationError::RequestNotFound` if the request_id doesn't exist
    ///   (never existed, already consumed, or timed out).
    pub fn deliver_assistance(
        &self,
        request_id: &str,
        response: AssistanceResponse,
    ) -> Result<(), CoordinationError>;
}
```

**Behavioral contract:**

1. `request_assistance` validates that `from` is a registered agent.
2. Generates a unique `request_id` (format: `assist_{from}_{uuid}`).
3. Creates a `oneshot::channel()`. Stores the `Sender` in the pending map.
4. Emits `CoordinationEvent::AssistanceRequested` on the broadcast channel.
5. Returns the `Receiver` to the caller.
6. The caller (`LoopExecutor`) decides how to await based on priority:
   - `Blocking`: `tokio::time::timeout(assistance_timeout, rx).await`
   - `Background`: spawns a background task that writes the response to
     a shared slot; the executor checks on the next iteration.
7. **Timeout:** If no response arrives within `LoopConfig.assistance_timeout`
   (default: 60 seconds), the `Receiver` is dropped by the caller. The
   corresponding `Sender` in the pending map becomes stale. Stale entries
   are cleaned up on the next `take_pending_requests()` call (detected by
   `Sender::is_closed()`).

**Executor integration pseudocode:**
```rust
// In LoopExecutor, when the agent requests assistance via meta-signal:
let rx = coordinator.request_assistance(&agent_id, need)?;

match need.priority {
    AssistancePriority::Blocking => {
        match tokio::time::timeout(config.assistance_timeout, rx).await {
            Ok(Ok(AssistanceResponse::Assigned { message, .. })) => {
                // Inject helper's context into the conversation
                if let Some(msg) = message {
                    messages.push(Message::user(msg));
                }
                // Continue loop — agent incorporates the help
            }
            Ok(Ok(AssistanceResponse::Unavailable { suggestion, .. })) => {
                // Inject suggestion as context, agent decides what to do
                messages.push(Message::user(format!(
                    "No assistance available: {}",
                    suggestion.unwrap_or_default()
                )));
            }
            Ok(Ok(AssistanceResponse::TimedOut)) | Ok(Err(_)) | Err(_) => {
                // Timeout or channel dropped — agent continues on its own
                messages.push(Message::user(
                    "Assistance request timed out. Continue with available information."
                ));
            }
        }
    }
    AssistancePriority::Background => {
        let slot = Arc::new(Mutex::new(None));
        let slot_clone = Arc::clone(&slot);
        tokio::spawn(async move {
            if let Ok(Ok(response)) = rx.await {
                *slot_clone.lock() = Some(response);
            }
        });
        // Store slot for the executor to check on next iteration
        self.background_assistance.insert(request_id, slot);
    }
}
```

#### 7.2.3 `yield_to`

**Signature:**
```rust
impl AgentCoordinator {
    /// Yields the calling agent's task.
    ///
    /// Records the yield context and notifies the supervisor. The agent
    /// should transition to `LoopState::Yielded` after calling this.
    ///
    /// Returns `Accepted` in multi-agent mode (supervisor will reroute),
    /// or `NoAlternative` in single-agent mode (no supervisor, no other agents).
    ///
    /// # Errors
    /// - `CoordinationError::AgentNotFound` if `from` is not registered.
    /// - `CoordinationError::AlreadyYielded` if `from` has already yielded.
    pub fn yield_to(
        &self,
        from: &AgentId,
        to: Option<AgentId>,
        context: YieldContext,
    ) -> Result<YieldResult, CoordinationError>;

    /// Drains all pending yields.
    /// Called by the supervisor during its monitoring loop.
    pub fn take_pending_yields(&self) -> Vec<PendingYield>;
}
```

**Behavioral contract:**

1. Validates `from` is registered.
2. Checks `from` has not already yielded (tracked via internal `HashSet<AgentId>`).
3. If `to` is `Some(agent_id)`, validates that the target is registered.
4. If no other agents are registered and no supervisor event listener is
   subscribed, returns `YieldResult::NoAlternative`.
5. Otherwise:
   - Marks `from` as yielded in the internal set.
   - Stores a `PendingYield` in the pending queue.
   - Emits `CoordinationEvent::AgentYielded` on the broadcast channel.
   - Returns `YieldResult::Accepted`.
6. The calling executor then transitions its loop: `loop.yield_detected(...)`.

**Yield is terminal for the agent.** Once `yield_to` returns `Accepted`, the
agent must not make further tool calls or generate output. The supervisor
owns the subtask from this point.

**Relationship to `ChildEvent::Yielded`:**

The supervisor sees yields through two channels:
- `CoordinationEvent::AgentYielded` (from the broadcast channel — immediate)
- `ChildEvent::Yielded` (from the executor's `JoinHandle` completing — after cleanup)

The `CoordinationEvent` arrives first and contains the `YieldContext` with
structured handoff data. The `ChildEvent` confirms the executor has shut down.
The supervisor should wait for both before rerouting to avoid racing with
the yielding agent's cleanup.

#### 7.2.4 `share_discovery` / `get_shared_context`

**Signatures:**
```rust
impl AgentCoordinator {
    /// Shares a discovery with other agents.
    ///
    /// Appends to the coordinator's discovery store. Non-blocking.
    /// Emits `CoordinationEvent::DiscoveryShared`.
    ///
    /// # Errors
    /// - `CoordinationError::AgentNotFound` if `from` is not registered.
    pub fn share_discovery(
        &self,
        from: &AgentId,
        discovery: Discovery,
    ) -> Result<(), CoordinationError>;

    /// Returns shared context visible to the given agent.
    ///
    /// Filters discoveries based on the configured visibility policy.
    /// Returns an empty `SharedContext` if `for_agent` is not registered
    /// (non-fatal — allows querying before full registration).
    pub fn get_shared_context(
        &self,
        for_agent: &AgentId,
    ) -> SharedContext;

    /// Sets the visibility policy for shared context.
    /// Called by the supervisor during initialization.
    pub fn set_visibility_policy(&self, policy: VisibilityPolicy);
}
```

**Discovery store semantics:**

- **Append-only.** Discoveries are never modified or removed.
- **Ordered.** Insertion order is preserved. `get_shared_context` returns
  discoveries oldest-first.
- **No deduplication.** If an agent shares the same discovery twice, both
  entries are stored. Deduplication is the caller's responsibility.
- **Bounded.** Maximum 1000 discoveries per coordinator instance. Oldest
  entries are evicted when the limit is reached.

**Visibility policy:**

```rust
/// Controls what agents can see of each other's discoveries.
pub enum VisibilityPolicy {
    /// Each agent sees all discoveries from all other agents.
    /// (Does not include the agent's own discoveries.)
    Open,

    /// Each agent sees only discoveries tagged with capabilities
    /// that match the agent's own capabilities.
    CapabilityFiltered,

    /// The supervisor explicitly controls visibility per-agent.
    /// Requires `allow_list: HashMap<AgentId, Vec<AgentId>>` mapping
    /// each agent to the set of agents whose discoveries it can see.
    Explicit { allow_list: HashMap<AgentId, Vec<AgentId>> },

    /// No discoveries are shared. `get_shared_context` always returns empty.
    Isolated,
}
```

The default is `VisibilityPolicy::Open`. The supervisor sets the policy based on
its `SharedContextMode`:

| `SharedContextMode` | `VisibilityPolicy` |
|---------------------|---------------------|
| `Isolated` | `Isolated` |
| `SummarySharing` | `Open` (discoveries are summaries by convention) |
| `FullSharing` | `Open` |
| `SupervisorManaged` | `Explicit { ... }` (supervisor builds allow_list) |

**Executor integration:**

The executor calls `share_discovery` when it detects a discovery meta-signal
in the model's output (new meta-signal type: `MetaSignal::Discovery`). It
calls `get_shared_context` at the start of each iteration to inject relevant
discoveries into the context window.

```rust
// At start of each iteration:
let shared = coordinator.get_shared_context(&agent_id);
if !shared.discoveries.is_empty() {
    let context_msg = format_shared_discoveries(&shared);
    messages.push(Message::system(context_msg));
}

// After meta-signal detection finds a discovery:
if let MetaSignal::Discovery { content, category, tags, data } = signal {
    coordinator.share_discovery(&agent_id, Discovery {
        content, category, tags, data,
    })?;
}
```

#### 7.2.5 Coordination Events

The coordinator emits events on a `tokio::sync::broadcast` channel. The
supervisor subscribes to this channel to learn about mid-execution
coordination activity (assistance requests, yields, discoveries) without
polling.

```rust
/// Events emitted by the coordinator.
/// The supervisor `select!`s on these alongside child `LoopEvent` streams.
#[derive(Debug, Clone)]
pub enum CoordinationEvent {
    /// An agent requested assistance.
    AssistanceRequested {
        request_id: String,
        from: AgentId,
        capabilities_needed: Vec<String>,
        priority: AssistancePriority,
    },

    /// An agent yielded its task.
    AgentYielded {
        from: AgentId,
        to: Option<AgentId>,
        reason: String,
        suggested_expertise: Vec<String>,
    },

    /// An agent shared a discovery.
    DiscoveryShared {
        from: AgentId,
        category: String,
        tags: Vec<String>,
    },
}

impl AgentCoordinator {
    /// Subscribes to coordination events.
    /// Returns a broadcast receiver. Multiple subscribers are supported.
    pub fn subscribe_events(&self) -> broadcast::Receiver<CoordinationEvent>;
}
```

**Channel sizing:** The broadcast channel has capacity 256. If a subscriber
falls behind, it receives `RecvError::Lagged(n)` and should catch up by
polling `take_pending_requests()` and `take_pending_yields()` directly.

#### 7.2.6 Concurrency and Error Semantics

**Thread safety:** All internal state is protected by `parking_lot::RwLock`
(consistent with existing `AgentCoordinator` implementation). Write operations
(`request_assistance`, `yield_to`, `share_discovery`) take write locks.
Read operations (`get_shared_context`, `take_pending_*`) take read locks
except when draining (write lock for `take_pending_*`).

**Error type:**
```rust
#[derive(Debug, Clone, thiserror::Error)]
pub enum CoordinationError {
    /// The specified agent is not registered.
    #[error("agent not found: {0}")]
    AgentNotFound(AgentId),

    /// The assistance request ID was not found.
    #[error("request not found: {0}")]
    RequestNotFound(String),

    /// The agent has already yielded and cannot perform further actions.
    #[error("agent already yielded: {0}")]
    AlreadyYielded(AgentId),

    /// The specified target agent for yield is not registered.
    #[error("yield target not found: {0}")]
    YieldTargetNotFound(AgentId),
}
```

**Invariants:**

| Invariant | Enforcement |
|-----------|-------------|
| Only registered agents can request assistance | `request_assistance` checks `agents` map |
| Only registered agents can yield | `yield_to` checks `agents` map |
| An agent can yield at most once | `yield_to` checks `yielded_agents` set |
| Assistance responses are delivered exactly once | `oneshot` channel guarantees single-use |
| Stale assistance requests are cleaned up | `take_pending_requests` filters by `Sender::is_closed()` |
| Discovery store is bounded | Oldest entries evicted at 1000 limit |
| Coordination events are non-blocking | `broadcast::Sender::send` never blocks; lagged receivers skip |

**Ordering guarantees:**

- Discoveries are ordered by insertion time within a single coordinator.
- Coordination events are ordered per-subscriber (broadcast channel guarantees).
- There is **no** global ordering between an agent's `share_discovery` and
  another agent's `get_shared_context` — eventual consistency is acceptable.
  An agent might not see a discovery that was shared microseconds earlier.

#### 7.2.7 Implementation Additions to `AgentCoordinator`

The following fields are added to the existing `AgentCoordinator` struct:

```rust
pub struct AgentCoordinator {
    // Existing fields (unchanged)
    agents: RwLock<HashMap<AgentId, AgentIdentity>>,
    pub locks: Arc<ToolLockManager>,
    pub quotas: Option<Arc<ResourceQuotaManager>>,

    // New: Assistance request inbox
    pending_requests: RwLock<HashMap<String, PendingAssistance>>,

    // New: Yield queue
    pending_yields: RwLock<Vec<PendingYield>>,
    yielded_agents: RwLock<HashSet<AgentId>>,

    // New: Discovery store
    discoveries: RwLock<Vec<StoredDiscovery>>,
    visibility_policy: RwLock<VisibilityPolicy>,

    // New: Event broadcast
    event_tx: broadcast::Sender<CoordinationEvent>,
}
```

The constructor changes:
```rust
impl AgentCoordinator {
    pub fn new() -> Self {
        let (event_tx, _) = broadcast::channel(256);
        Self {
            agents: RwLock::new(HashMap::new()),
            locks: Arc::new(ToolLockManager::new()),
            quotas: None,
            pending_requests: RwLock::new(HashMap::new()),
            pending_yields: RwLock::new(Vec::new()),
            yielded_agents: RwLock::new(HashSet::new()),
            discoveries: RwLock::new(Vec::new()),
            visibility_policy: RwLock::new(VisibilityPolicy::Open),
            event_tx,
        }
    }
}
```

**Cleanup on `unregister_agent`:** When an agent is unregistered, the
coordinator also:
- Removes it from `yielded_agents`.
- Cancels any pending assistance requests from that agent (drops the `Sender`,
  causing the `Receiver` to return `Err(Canceled)`).
- Does NOT remove its discoveries (they remain as shared knowledge).

### 7.3 Resource Coordination

```rust
// Prevent conflicting tool executions
pub struct ToolLock {
    pub tool: String,
    pub resource: String,  // e.g., file path
    pub held_by: AgentId,
    pub acquired_at: Instant,
}

// Shared quota management
pub struct ResourceQuota {
    pub total_tool_calls: u32,
    pub remaining_tool_calls: u32,
    pub total_tokens: u32,
    pub remaining_tokens: u32,
}
```

---

## 8. Wellbeing Integration

The loop integrates with Beleth's wellbeing monitoring:

### 8.1 Wellbeing Signals

```rust
pub struct WellbeingState {
    pub coherence: f32,      // 0-1, reasoning integration
    pub confidence: f32,     // 0-1, productive vs paralyzed
    pub stability: f32,      // 0-1, OODA cycle health
    pub engagement: f32,     // 0-1, productive effort vs spinning
}

pub enum WellbeingSignal {
    Healthy,
    Mild { concern: String },
    Moderate { concern: String, suggestion: Intervention },
    Severe { concern: String, action: RequiredAction },
}
```

### 8.2 Interventions

When wellbeing signals indicate distress, the loop can intervene:

```rust
pub enum Intervention {
    // Gentle: adjust parameters
    ReduceComplexity,        // simplify the task
    ExtendDeadline,          // more time
    OfferBreak,              // pause point

    // Moderate: change approach
    SuggestDifferentApproach,
    RequestClarification,
    BringInAssistance,

    // Significant: human involvement
    EscalateToOperator,
    GracefulTermination,
}
```

The key insight: an agent showing signs of struggle is providing information. The system should respond with support, not punishment.

---

## 9. API Integration

### 9.1 Request Extension

The chat completions request is extended to support agentic mode:

```rust
pub struct ChatCompletionRequest {
    // ... existing fields ...

    // Agentic mode configuration
    pub agentic: Option<AgenticConfig>,
}

pub struct AgenticConfig {
    pub enabled: bool,
    pub max_iterations: Option<u32>,
    pub max_tool_calls: Option<u32>,
    pub autonomy: Option<AutonomyGrant>,
    pub allow_uncertainty: Option<bool>,
    pub allow_yield: Option<bool>,
    pub wellbeing_monitoring: Option<bool>,
}
```

### 9.2 Response Extension

```rust
pub struct ChatCompletionResponse {
    // ... existing fields ...

    // Agentic mode results
    pub agentic: Option<AgenticResult>,
}

pub struct AgenticResult {
    pub iterations: u32,
    pub tool_calls: u32,
    pub status: LoopStatus,
    pub termination: TerminationReason,
    pub partial_progress: Option<String>,
    pub can_continue: bool,
    pub continuation_token: Option<String>,
}
```

### 9.3 Continuation API

Loops that terminate as `Stuck`, `Yielded`, or via resource limits may be resumed.
The continuation API preserves loop state server-side and allows the client to
provide additional context, modify configuration, or redirect the agent.

#### 9.3.1 Continuation State

When a loop terminates with `can_resume: true`, the server serializes and stores
the full loop state:

```rust
pub struct ContinuationState {
    /// Opaque token identifying this continuation.
    pub token: String,

    /// Session identity (preserved across continuations).
    pub session_id: String,

    /// Full message history at time of termination.
    /// Uses `infernum_core::Message` — the same type the executor builds internally.
    pub messages: Vec<infernum_core::Message>,

    /// Tool results collected during execution.
    pub tool_results: Vec<AgenticToolResult>,

    /// Exploration branches tracked.
    pub exploration_branches: Vec<ExplorationBranch>,

    /// Resources consumed so far.
    pub iterations_completed: u32,
    pub tool_calls_made: u32,
    pub tokens_generated: u32,

    /// Configuration used (may be modified on resume).
    pub loop_config: LoopConfig,
    pub autonomy: AutonomyGrant,
    pub system_prompt: Option<String>,
    /// Stored as String (serialized from executor's `PathBuf`).
    pub working_dir: Option<String>,

    /// Why the loop stopped.
    pub termination: TerminationReason,

    /// When the state was stored.
    /// Uses `SystemTime` (not `Instant`) because this must survive serialization.
    pub stored_at: SystemTime,
}
```

> **Implementation note:** `ContinuationState` is a new type. It is constructed
> by the server's agentic handler from the executor's internal state after
> termination — it is NOT a field on `LoopSummary`. The `LoopSummary.can_resume`
> field signals whether the handler should create a `ContinuationState`.
> The `continuation_token` field referenced in the §9.2 `AgenticResult` is
> populated by the handler after storing the state.

#### 9.3.2 Storage and TTL

- **Storage:** In-memory by default, with the `ContinuationStore` trait for
  pluggable backends (file, database).
- **TTL:** 1 hour default, configurable per-session. Expired states are garbage
  collected on access.
- **Token format:** `cont_{session_id}_{uuid}` — opaque to clients, maps to
  server-side state.
- **Limit:** Maximum 100 stored continuations per server instance (LRU eviction).

```rust
#[async_trait]
pub trait ContinuationStore: Send + Sync {
    async fn store(&self, state: ContinuationState) -> Result<String, StoreError>;
    async fn load(&self, token: &str) -> Result<Option<ContinuationState>, StoreError>;
    async fn remove(&self, token: &str) -> Result<(), StoreError>;
    async fn cleanup_expired(&self) -> Result<u32, StoreError>;
}
```

#### 9.3.3 Resume Endpoint

```
POST /api/agent/{session_id}/continue
```

```json
{
  "continuation_token": "cont_sess_abc123_def456",
  "additional_context": "The file is actually at /opt/project/main.rs",
  "modified_config": {
    "max_iterations": 20,
    "max_tool_calls": 100,
    "auto_approve": ["bash:*"]
  }
}
```

**Request fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `continuation_token` | `string` | Yes | Token from prior `LoopSummary.continuation_token` |
| `additional_context` | `string` | No | New information for the agent (injected as user message) |
| `modified_config` | `object` | No | Configuration overrides for the resumed loop |

**Modification rules:**

| Field | Modifiable | Constraint |
|-------|-----------|------------|
| `max_iterations` | Yes | New limit must exceed consumed amount (extends remaining budget) |
| `max_tool_calls` | Yes | New limit must exceed consumed amount |
| `max_tokens` | Yes | New limit must exceed consumed amount |
| `auto_approve` | Yes | Can widen (add patterns) |
| `forbidden` | Yes | Can narrow (remove patterns) |
| `system_prompt` | No | Immutable after loop start |
| `working_dir` | No | Immutable after loop start |

**Resume semantics:**

1. Server loads `ContinuationState` from store.
2. Validates `session_id` matches the token's session.
3. Applies `modified_config` overrides (with constraint checks).
4. If `additional_context` is provided, appends it as a user message.
5. Resets resource counters to the *new* limits minus consumed amounts.
6. Creates a new `LoopExecutor` with the restored messages and configuration.
7. Transitions state machine from terminal state back to `Generating`.
8. Returns a new SSE stream of `LoopEvent`s.

**Response:** SSE stream (identical format to `POST /api/agent/run`).

#### 9.3.4 Which Terminations Are Resumable

| Termination | Resumable | Rationale |
|-------------|-----------|-----------|
| `AnswerProvided` | No | Task complete |
| `TaskComplete` | No | Task complete |
| `AgentStuck` | Yes | Client provides clarification |
| `AgentYielded` | Yes | Different agent or additional context |
| `MaxIterations` | Yes | Client extends budget |
| `TokenBudgetExhausted` | Yes | Client extends budget |
| `WallTimeExceeded` | Yes | Client extends budget |
| `ToolCallLimitReached` | Yes | Client extends budget |
| `ClientCancelled` | No | Intentional termination |
| `OperatorTerminated` | No | Intentional termination |
| `SystemShutdown` | No | State may be lost |

### 9.4 Tool Approval Protocol

When the autonomy grant returns `Permission::RequiresApproval` for a tool call,
the loop must pause and wait for an external decision. This section specifies the
approval handshake between the executor and the client.

> **Implementation note:** This protocol requires additive changes to existing types:
> - `LoopEvent::ToolApprovalRequired` gains `arguments`, `timeout_secs`, `pending_count` fields.
> - `LoopConfig` gains an `approval_timeout: Duration` field (default: 5 minutes).
> - `SessionRegistry` gains `request_approval()`, `deliver_approval()`, `pending_approvals()` methods.
> - The executor's `execute_single_tool` changes from **fail-immediately** on
>   `RequiresApproval` to **block-and-wait** via a `tokio::sync::oneshot` channel.

#### 9.4.1 Approval Flow

```
┌─────────┐    ToolApprovalRequired     ┌─────────┐
│Executor │ ──────── SSE ──────────────▶│ Client  │
│ (paused)│                              │         │
│         │◀──── POST /approve ─────────│         │
│(resumes)│    ApprovalDecision          │         │
└─────────┘                              └─────────┘
```

1. Executor encounters `Permission::RequiresApproval`.
2. Executor emits `LoopEvent::ToolApprovalRequired { call_id, tool, arguments }`.
3. Executor creates a `oneshot` channel and registers the sender in the
   `ApprovalInbox` (within `SessionRegistry`).
4. Executor awaits the receiver with a configurable timeout.
5. Client receives the SSE event and decides.
6. Client submits decision via `POST /api/agent/{session_id}/approve`.
7. Server looks up the `oneshot` sender and delivers the decision.
8. Executor receives the decision and either executes or skips the tool call.

#### 9.4.2 Approval Endpoint

```
POST /api/agent/{session_id}/approve
```

```json
{
  "call_id": "call_abc123def456",
  "decision": "approve",
  "scope": "this_call"
}
```

**Request fields:**

| Field | Type | Required | Values |
|-------|------|----------|--------|
| `call_id` | `string` | Yes | The `call_id` from `ToolApprovalRequired` |
| `decision` | `string` | Yes | `"approve"`, `"deny"`, `"approve_always"` |
| `scope` | `string` | No | `"this_call"` (default), `"this_tool"`, `"this_session"` |

**Decision types:**

| Decision | Effect |
|----------|--------|
| `approve` | Execute this specific tool call |
| `deny` | Skip this tool call (returns recoverable error to agent) |
| `approve_always` | Execute and add the tool pattern to `auto_approve` for the session |

**Scope (when `approve_always`):**

| Scope | Effect |
|-------|--------|
| `this_call` | Only this call (same as `approve`) |
| `this_tool` | Auto-approve all future calls to this tool name (adds `ToolPattern::Tool(name)` to `auto_approve`) |
| `this_session` | Auto-approve all tool calls for the remainder of this session (adds `ToolPattern::Tool("*")` to `auto_approve`) |

**Response:**

```json
{
  "status": "delivered",
  "call_id": "call_abc123def456",
  "session_id": "sess_abc123"
}
```

**Error responses:**

| Status | Condition |
|--------|-----------|
| 404 | Session not found, or call_id not found (never existed or already consumed) |
| 408 | Approval timeout already expired (executor moved on) |

> **Note:** Once a `oneshot::Sender` is consumed (decision delivered), the
> `call_id` is removed from the pending set. A subsequent request for the
> same `call_id` returns 404, not 409, because the pending entry no longer
> exists. There is no "already delivered" state to distinguish.

#### 9.4.3 Approval Infrastructure

```rust
/// Pending approval request stored in the registry.
pub struct PendingApproval {
    pub call_id: String,
    pub tool_name: String,
    pub arguments: serde_json::Value,
    pub requested_at: Instant,
    pub respond: oneshot::Sender<ApprovalDecision>,
}

/// Decision delivered by the client.
pub enum ApprovalDecision {
    Approve,
    Deny,
    ApproveAlways { scope: ApprovalScope },
}

pub enum ApprovalScope {
    ThisCall,
    ThisTool,
    ThisSession,
}

/// Extension to SessionRegistry for approval management.
impl SessionRegistry {
    /// Registers a pending approval and returns a receiver for the decision.
    pub async fn request_approval(
        &self,
        session_id: &str,
        call_id: &str,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> oneshot::Receiver<ApprovalDecision>;

    /// Delivers an approval decision. Returns Err if not found or expired.
    pub async fn deliver_approval(
        &self,
        session_id: &str,
        call_id: &str,
        decision: ApprovalDecision,
    ) -> Result<(), ApprovalError>;

    /// Lists pending approvals for a session.
    pub async fn pending_approvals(
        &self,
        session_id: &str,
    ) -> Vec<PendingApprovalInfo>;
}
```

#### 9.4.4 Timeout Behavior

- **Default timeout:** 5 minutes (configurable via `LoopConfig.approval_timeout`).
- **On timeout:** The tool call is treated as denied with a recoverable error.
  The agent receives: `"Tool approval timed out after {N}s — tool call skipped."`
- **Session status:** Set to `AwaitingApproval` when first approval is requested,
  returns to `Running` when all pending approvals are resolved.
- **Multiple pending approvals:** When a single iteration detects multiple tool
  calls requiring approval, the executor must first collect all approval-needing
  calls, submit all approval requests, then `join_all` on the receivers. This
  requires restructuring the current sequential `for call in &detected_calls`
  loop into a two-pass approach: (1) identify which calls need approval and
  which are auto-approved, (2) submit approval requests for the former while
  executing the latter, (3) collect all responses before proceeding.

#### 9.4.5 SSE Event Extension

The `ToolApprovalRequired` event is extended to include arguments so the client
can make an informed decision:

```json
{
  "event": "tool_approval_required",
  "data": {
    "call_id": "call_abc123def456",
    "tool": "bash",
    "arguments": { "command": "rm -rf /tmp/cache" },
    "timeout_secs": 300,
    "pending_count": 1
  }
}
```

#### 9.4.6 Executor Integration

The executor's `execute_single_tool` method changes from immediately returning
a failure to blocking on approval:

```rust
Permission::RequiresApproval => {
    // Emit approval request event
    event_tx.send(LoopEvent::ToolApprovalRequired { ... }).await;

    // Register and wait for decision
    let rx = sessions.request_approval(
        session_id, &call.id, &call.name, call.arguments.clone()
    ).await;

    match tokio::time::timeout(approval_timeout, rx).await {
        Ok(Ok(ApprovalDecision::Approve))
        | Ok(Ok(ApprovalDecision::ApproveAlways { .. })) => {
            // Execute the tool call normally
            self.execute_tool_inner(call, tool_ctx, event_tx).await
        }
        Ok(Ok(ApprovalDecision::Deny)) => {
            AgenticToolResult { status: Failed { recoverable: true }, ... }
        }
        Ok(Err(_)) | Err(_) => {
            // Oneshot dropped or timeout — treat as denied
            AgenticToolResult { status: Failed { recoverable: true }, ... }
        }
    }
}
```

When `ApproveAlways` is received, the executor mutates the session's
`AutonomyGrant` to add the tool pattern to `auto_approve`, preventing future
approval requests for matching calls.

---

## 10. Implementation Phases

### Phase 6.1: Basic Loop

- [ ] Implement `AgenticLoop` state machine
- [ ] Wire into `chat_completions` handler
- [ ] Basic iteration with tool execution
- [ ] Natural termination conditions
- [ ] Resource termination conditions
- [ ] SSE streaming for loop events

### Phase 6.2: Context Management

- [ ] Token budget tracking
- [ ] Context compression strategies
- [ ] Exploration branch preservation
- [ ] Dead-end pruning

### Phase 6.3: Meta-Signals

- [ ] Explicit signal parsing
- [ ] Implicit signal detection
- [ ] Uncertainty handling
- [ ] Stuck handling
- [ ] Yield handling

### Phase 6.4: Multi-Agent Coordination Primitives

- [ ] `CoordinationError` type
- [ ] `CoordinationEvent` enum + broadcast channel on `AgentCoordinator`
- [ ] `subscribe_events()` method
- [ ] `request_assistance()` + `take_pending_requests()` + `deliver_assistance()`
- [ ] `yield_to()` + `take_pending_yields()` + yielded-agent tracking
- [ ] `share_discovery()` + `get_shared_context()` + `set_visibility_policy()`
- [ ] `VisibilityPolicy` filtering (Open, CapabilityFiltered, Explicit, Isolated)
- [ ] Discovery store bounding (1000 max, oldest eviction)
- [ ] Stale assistance request cleanup (`Sender::is_closed()`)
- [ ] Cleanup on `unregister_agent` (cancel requests, clear yielded set)
- [ ] `MetaSignal::Discovery` variant + executor integration

### Phase 6.4b: Multi-Agent Supervisor

- [ ] Multi-agent supervisor (see MULTI-AGENT-SUPERVISOR-SPEC.md)

### Phase 6.5: Wellbeing

- [ ] Wellbeing signal integration
- [ ] Intervention system
- [ ] Graceful degradation

### Phase 6.6: Tool Approval

- [ ] Approval inbox in SessionRegistry
- [ ] `POST /api/agent/{id}/approve` endpoint
- [ ] Executor blocks on oneshot channel with timeout
- [ ] `approve_always` mutates AutonomyGrant
- [ ] Extended `ToolApprovalRequired` event with arguments

### Phase 6.7: Session Continuation

- [ ] `ContinuationState` serialization
- [ ] `ContinuationStore` trait + in-memory implementation
- [ ] `POST /api/agent/{id}/continue` endpoint
- [ ] Resume semantics (message injection, config override)
- [ ] TTL and LRU eviction

---

## 11. Open Questions

### 11.1 Autonomy Grant Source

Where does the autonomy grant come from?
- Request parameter (client specifies)
- Server configuration (operator specifies)
- Agent negotiation (agent requests, server approves)
- Hybrid (defaults + overrides)

**Decision:** Hybrid. Server sets defaults and hard limits. Client can narrow (not widen) within those limits.

### 11.2 Implicit Signal Detection

Should the loop detect uncertainty/stuck states from natural language, or require explicit structured output?

**Decision:** Both. Explicit is preferred and more reliable. Implicit is fallback when the model doesn't use structured format.

### 11.3 Cross-Session Continuation

Can a loop be resumed across HTTP sessions? What state must be preserved?

**Decision:** Yes, via continuation token. Token encodes or references: messages, tool results, exploration state, configuration. Stored server-side with TTL.

---

## 12. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-02 | Initial draft. Design philosophy, loop architecture, execution model. |
| 0.1.1 | 2026-02-02 | Open questions §11.1-11.3 resolved. Hybrid autonomy, dual signal detection, continuation tokens approved. |
| 0.2.0 | 2026-02-04 | §9.3 expanded: continuation state, storage/TTL, resume endpoint, modification rules, resumable termination table. §9.4 added: tool approval protocol with oneshot handshake, approval endpoint, timeout behavior, `approve_always` scope. Phases 6.6-6.7 added. |
| 0.3.0 | 2026-02-04 | §7.2 expanded from signatures to full behavioral specification. Added: §7.2.1 types (AssistanceRequest/Response, YieldContext/Result, Discovery, SharedContext, VisibilityPolicy), §7.2.2 request_assistance contract (oneshot channel pattern, blocking/background priority, timeout), §7.2.3 yield_to contract (terminal semantics, NoAlternative rejection, dual-channel supervisor notification), §7.2.4 share_discovery/get_shared_context (append-only store, bounded at 1000, visibility policy, executor integration), §7.2.5 CoordinationEvent broadcast channel, §7.2.6 concurrency/error semantics (CoordinationError, invariant table, ordering guarantees), §7.2.7 implementation additions to AgentCoordinator struct. Phase 6.4 expanded with coordination primitive checklist. |

