# Agentic Loop Specification

**Version:** 0.1.0
**Status:** Draft
**Date:** 2026-02-02
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

```rust
impl AgentCoordinator {
    // Request another agent's help
    pub async fn request_assistance(
        &self,
        from: &AgentId,
        need: AssistanceRequest,
    ) -> AssistanceResponse;

    // Yield to another agent
    pub async fn yield_to(
        &self,
        from: &AgentId,
        to: Option<AgentId>,  // None = any suitable agent
        context: YieldContext,
    ) -> YieldResult;

    // Share a discovery
    pub async fn share_discovery(
        &self,
        from: &AgentId,
        discovery: Discovery,
    );

    // Check what other agents have learned
    pub async fn get_shared_context(
        &self,
        for_agent: &AgentId,
    ) -> SharedContext;
}
```

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

For resuming interrupted loops:

```
POST /v1/chat/completions/continue
{
  "continuation_token": "...",
  "additional_context": [...],
  "modified_config": {...}
}
```

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

### Phase 6.4: Multi-Agent

- [ ] Agent identity in loop
- [ ] Coordination primitives
- [ ] Resource locking
- [ ] Shared context

### Phase 6.5: Wellbeing

- [ ] Wellbeing signal integration
- [ ] Intervention system
- [ ] Graceful degradation

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

