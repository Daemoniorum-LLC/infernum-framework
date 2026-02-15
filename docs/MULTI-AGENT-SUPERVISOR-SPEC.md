# Multi-Agent Supervisor Specification

**Version:** 0.3.0
**Status:** Implementing (Phase 5 pending)
**Date:** 2026-02-15
**Prerequisite:** AGENTIC-LOOP-SPEC.md v0.2.0

---

## 1. Purpose

AGENTIC-LOOP-SPEC §7 defines coordination *primitives*: agent identity, tool
locks, and resource quotas. The current implementation (`coordination.rs`)
provides `AgentCoordinator` (identity registry), `ToolLockManager` (exclusive
tool access), and `ResourceQuotaManager` (shared budgets).

> **Prerequisite:** AGENTIC-LOOP-SPEC §7 also specifies communication methods
> (`request_assistance`, `yield_to`, `share_discovery`, `get_shared_context`)
> that are **not yet implemented** on `AgentCoordinator`. These must be added
> before the supervisor can be built. See §10 Phase 2.

These primitives enable cooperation but do not orchestrate it.

This specification defines the **Supervisor** — the component that spawns,
monitors, routes, and aggregates multiple `LoopExecutor` instances working
toward a shared objective.

### 1.1 Scope

**In scope:**
- Task decomposition and delegation
- Agent lifecycle management (spawn, monitor, terminate)
- Yield and assistance routing
- Result aggregation
- Failure recovery

**Out of scope:**
- The coordination primitives themselves (specified in AGENTIC-LOOP-SPEC §7)
- The individual loop executor behavior (specified in AGENTIC-LOOP-SPEC §2-6)
- Cross-server agent coordination (future specification)

---

## 2. Design Principles

**Principle 1: Supervisor is an Agent, Not a Scheduler**

The supervisor uses the same `LoopExecutor` infrastructure as child agents.
It generates plans, detects when children need help, and makes decisions.
It is not a static task queue.

**Principle 2: Children are Autonomous**

Child agents operate their own loops with their own autonomy grants. The
supervisor sets boundaries but does not micromanage iteration-level decisions.

**Principle 3: Failure is Routing Information**

When a child agent gets stuck or yields, the supervisor treats this as a
routing signal, not an error. Stuck agents provide hypotheses. Yielding agents
provide partial progress. Both inform the next assignment.

**Principle 4: Shared Context is Opt-In**

Children do not automatically see each other's full context. The supervisor
decides what to share and when, preventing context pollution between agents
working on unrelated subtasks.

---

## 3. Architecture

### 3.1 Component Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                         Supervisor                            │
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │   Planner    │  │   Router     │  │   Aggregator      │   │
│  │ (decompose)  │  │ (assign)     │  │ (collect results) │   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬──────────┘   │
│         │                 │                    │              │
│         ▼                 ▼                    ▼              │
│  ┌───────────────────────────────────────────────────────┐   │
│  │               Agent Coordinator                        │   │
│  │  (identities, tool locks, resource quotas, context)   │   │
│  └──────────────────────┬────────────────────────────────┘   │
└─────────────────────────┼────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Agent A  │   │ Agent B  │   │ Agent C  │
    │(Executor)│   │(Executor)│   │(Executor)│
    └──────────┘   └──────────┘   └──────────┘
```

### 3.2 Core Types

```rust
/// Configuration for the multi-agent supervisor.
pub struct SupervisorConfig {
    /// Maximum number of concurrent child agents.
    pub max_concurrent_agents: u32,

    /// Maximum total agents spawned (including sequential).
    pub max_total_agents: u32,

    /// Strategy for decomposing the top-level objective.
    pub decomposition: DecompositionStrategy,

    /// How to assign subtasks to agents.
    pub routing: RoutingStrategy,

    /// Global resource budget (split among children).
    pub resource_budget: ResourceBudget,

    /// Whether children can see each other's results.
    pub shared_context_mode: SharedContextMode,

    /// Timeout for the entire supervisor run.
    pub max_wall_time: Duration,
}

/// How to decompose the top-level objective into subtasks.
pub enum DecompositionStrategy {
    /// Use the supervisor's own LLM to plan subtasks.
    LlmPlanned {
        /// Maximum subtasks to generate.
        max_subtasks: u32,
    },

    /// Client provides explicit subtasks.
    ClientProvided {
        subtasks: Vec<Subtask>,
    },

    /// Single agent, no decomposition (supervisor acts as
    /// a resilient wrapper with retry/reassign on failure).
    SingleAgent,
}

/// How to route subtasks and handle yields.
pub enum RoutingStrategy {
    /// All subtasks run concurrently.
    Parallel,

    /// Subtasks run in sequence, each receiving prior results.
    Sequential,

    /// Supervisor decides based on dependencies.
    DependencyAware,
}

/// What child agents can see of each other's work.
pub enum SharedContextMode {
    /// Children see nothing of each other.
    Isolated,

    /// Children see summaries of completed subtask results.
    SummarySharing,

    /// Children see full results of completed subtasks.
    FullSharing,

    /// Supervisor decides per-subtask what to share.
    SupervisorManaged,
}

/// Estimated complexity of a subtask (used for budget allocation).
pub enum Complexity {
    Low,
    Medium,
    High,
}

/// A subtask to be assigned to a child agent.
pub struct Subtask {
    /// Unique identifier.
    pub id: String,

    /// Objective for the child agent.
    pub objective: String,

    /// Role assigned to the child (uses existing `AgentRole` from coordination.rs).
    /// `AgentRole` is `Copy`, so cloning is unnecessary.
    pub role: AgentRole,

    /// Dependencies: subtask IDs that must complete first.
    pub depends_on: Vec<String>,

    /// Estimated complexity (used for weighted budget allocation).
    pub estimated_complexity: Complexity,

    /// Resource allocation for this subtask.
    pub resources: Option<ResourceQuota>,

    /// Autonomy override for this subtask.
    pub autonomy: Option<AutonomyGrant>,

    /// System prompt override for this subtask.
    pub system_prompt: Option<String>,
}

/// Global resource budget for the supervisor run.
///
/// This is distinct from `ResourceQuota` in coordination.rs (which tracks
/// shared per-agent consumption). `ResourceBudget` is the top-level
/// allocation that the supervisor splits among children.
pub struct ResourceBudget {
    /// Total iterations across all agents.
    pub total_iterations: u32,

    /// Total tool calls across all agents.
    pub total_tool_calls: u32,

    /// Total tokens across all agents.
    pub total_tokens: u32,

    /// Wall time for the entire run.
    pub wall_time: Duration,
}
```

---

## 4. Supervisor Lifecycle

### 4.1 State Machine

```
Initialized → Planning → Dispatching → Monitoring → Aggregating → Completed
                                            │
                                            ├── Rerouting (on yield/stuck)
                                            └── Recovering (on failure)
```

```rust
pub enum SupervisorState {
    /// Not yet started.
    Initialized,

    /// Decomposing objective into subtasks.
    Planning,

    /// Assigning subtasks to child agents.
    Dispatching,

    /// Children are executing. Supervisor monitors events.
    Monitoring,

    /// A child yielded or got stuck. Supervisor is reassigning.
    Rerouting { reason: RerouteReason },

    /// A child failed. Supervisor is deciding how to recover.
    Recovering { failed_agent: AgentId, error: String },

    /// All subtasks complete. Aggregating results.
    Aggregating,

    /// Final answer produced.
    Completed,

    /// Supervisor terminated (timeout, cancellation, unrecoverable failure).
    Terminated { reason: String },
}

pub enum RerouteReason {
    /// Child agent yielded.
    AgentYielded {
        agent_id: AgentId,
        partial_progress: Option<String>,
        suggested_expertise: Vec<String>,
    },
    /// Child agent is stuck.
    AgentStuck {
        agent_id: AgentId,
        attempts: u32,
        request: StuckRequest,
    },
    /// Child hit resource limits.
    ResourceExhausted {
        agent_id: AgentId,
        resource: String,
    },
}
```

### 4.2 Lifecycle Phases

#### Phase 1: Planning

The supervisor decomposes the top-level objective into subtasks.

For `LlmPlanned`:
1. Supervisor runs its own `LoopExecutor` with a planning-specific system prompt.
2. The model output is parsed for a structured task list.
3. Dependencies between subtasks are identified.
4. Resource budgets are allocated proportionally.

```rust
/// Planning prompt template.
const PLANNING_PROMPT: &str = r#"
You are a task planner. Decompose the following objective into subtasks.

Objective: {objective}

Available tools: {tool_list}

Output a JSON array of subtasks:
[
  {
    "id": "subtask_1",
    "objective": "...",
    "role": "primary|specialist|reviewer",
    "depends_on": [],
    "estimated_complexity": "low|medium|high"
  }
]
"#;
```

For `ClientProvided`: Skip planning, use the provided subtask list directly.

For `SingleAgent`: Create one subtask equal to the full objective.

#### Phase 2: Dispatching

For each subtask (respecting dependency order and concurrency limits):

1. Allocate an `AgentId` via `AgentCoordinator`.
2. Create an `ExecutorConfig` from the subtask specification.
3. Split the resource budget proportionally.
4. Spawn a `LoopExecutor` in a tokio task.
5. Connect the executor's event channel to the supervisor's monitor.

```rust
/// Handle to a running child agent.
pub struct AgentHandle {
    pub agent_id: AgentId,
    pub subtask_id: String,
    pub handle: tokio::task::JoinHandle<Result<LoopSummary, LoopError>>,
    pub event_rx: mpsc::Receiver<LoopEvent>,
}

impl Supervisor {
    async fn dispatch_subtask(
        &self,
        subtask: &Subtask,
        coordinator: &AgentCoordinator,
        engine: Arc<dyn abaddon::InferenceEngine>,
        tools: Arc<ToolRegistry>,
    ) -> AgentHandle {
        // Generate a unique agent ID
        let agent_id: AgentId = format!("agent_{}", uuid::Uuid::new_v4().simple());

        // Register with the coordinator (synchronous — uses parking_lot::RwLock)
        let identity = AgentIdentity::new(&agent_id, subtask.role)
            .with_task(&subtask.objective);
        coordinator.register_agent(identity);

        let config = ExecutorConfig::new(&agent_id)
            .with_system_prompt(subtask.system_prompt.as_deref().unwrap_or(
                "You are a specialist agent working on a subtask."
            ))
            .with_autonomy(subtask.autonomy.clone().unwrap_or_default())
            .with_loop_config(self.config_for_subtask(subtask));

        let executor = LoopExecutor::new(
            Arc::clone(&engine),
            Arc::clone(&tools),
            config,
        );

        let (event_tx, event_rx) = mpsc::channel(128);
        let objective = subtask.objective.clone();

        let handle = tokio::spawn(async move {
            executor.run(&objective, event_tx).await
        });

        AgentHandle { agent_id, subtask_id: subtask.id.clone(), handle, event_rx }
    }
}
```

#### Phase 3: Monitoring

```rust
/// Events received from child agents (derived from LoopEvent + JoinHandle results).
pub enum ChildEvent {
    /// Loop event from a running child (relayed from its event channel).
    Event { agent_id: AgentId, event: LoopEvent },
    /// Child's LoopExecutor completed successfully.
    Completed { agent_id: AgentId, summary: LoopSummary },
    /// Child signaled it is stuck (from LoopSummary.termination).
    Stuck { agent_id: AgentId, request: StuckRequest, attempts: u32 },
    /// Child yielded (from LoopSummary.termination).
    Yielded { agent_id: AgentId, partial: Option<String>, expertise: Vec<String> },
    /// Child failed (LoopError from executor).
    Failed { agent_id: AgentId, error: String },
    /// Child emitted a tool approval request (bubbled up).
    ApprovalRequired { agent_id: AgentId, call_id: String, tool: String, args: serde_json::Value },
}
```

The supervisor monitors all child agents concurrently:

```rust
impl Supervisor {
    async fn monitor(&mut self, handles: &mut Vec<AgentHandle>) {
        loop {
            // Select across all child event channels
            let event = select_next_event(handles).await;

            match event {
                ChildEvent::Completed { agent_id, summary } => {
                    self.results.insert(agent_id, summary);
                    if self.all_subtasks_complete() {
                        break;
                    }
                    self.dispatch_ready_dependents().await;
                }
                ChildEvent::Stuck { agent_id, request, .. } => {
                    self.handle_stuck_child(agent_id, request).await;
                }
                ChildEvent::Yielded { agent_id, partial, expertise } => {
                    self.handle_yield(agent_id, partial, expertise).await;
                }
                ChildEvent::Failed { agent_id, error } => {
                    self.handle_failure(agent_id, error).await;
                }
                ChildEvent::ApprovalRequired { agent_id, call_id, tool, args } => {
                    // Bubble up to supervisor's own approval mechanism
                    self.forward_approval(agent_id, call_id, tool, args).await;
                }
                _ => {
                    // Relay other events to supervisor's event stream
                    self.relay_event(event).await;
                }
            }
        }
    }
}
```

#### Phase 4: Rerouting

When a child yields or gets stuck:

**Yield handling:**
1. Record the partial progress.
2. If `suggested_expertise` matches another pending or idle agent's role,
   route to that agent with the partial progress as context.
3. If no suitable agent exists, spawn a new one (within limits).
4. If agent limit reached, the supervisor attempts the subtask itself.

**Stuck handling:**
1. Examine the `StuckRequest`:
   - `Clarification`: Supervisor attempts to answer from its own context or
     other agents' results. If it cannot, bubbles up to the client.
   - `MoreContext`: Supervisor injects relevant context from other agents'
     completed results.
   - `DifferentTools`: If the requested tools are available but weren't in the
     child's tool set, respawn with expanded tool access.
   - `HumanIntervention`: Bubble up to the client via the supervisor's event
     stream.
2. The stuck agent's state is preserved via the continuation API (§9.3).
3. A new agent (or the same one resumed) receives the additional context.

**Resource exhaustion:**
1. If the global budget allows, extend the child's limits and resume.
2. If the global budget is exhausted, collect partial results and proceed
   to aggregation.

#### Phase 5: Aggregation

Once all subtasks are complete (or the supervisor decides to stop):

1. Collect all child `LoopSummary` results.
2. If `SharedContextMode::FullSharing` or `SupervisorManaged`, run a final
   aggregation step where the supervisor's own executor synthesizes a
   unified answer from all subtask results.
3. Return the aggregated `SupervisorSummary`.

```rust
pub struct SupervisorSummary {
    /// Aggregated answer from all subtasks.
    pub answer: Option<String>,

    /// Per-subtask results.
    pub subtask_results: Vec<SubtaskResult>,

    /// Why the supervisor stopped.
    pub termination: SupervisorTermination,

    /// Global metrics.
    pub total_agents_spawned: u32,
    pub total_iterations: u32,
    pub total_tool_calls: u32,
    pub total_tokens: u32,
    pub wall_time: Duration,

    /// Rerouting events that occurred.
    pub reroutes: Vec<RerouteEvent>,
}

pub struct SubtaskResult {
    pub subtask_id: String,
    pub agent_id: AgentId,
    pub status: SubtaskStatus,
    pub summary: Option<LoopSummary>,
    pub partial_progress: Option<String>,
}

pub enum SubtaskStatus {
    Completed,
    Partial,
    Failed { error: String },
    Skipped { reason: String },
}

pub enum SupervisorTermination {
    /// All subtasks completed.
    AllComplete,
    /// Sufficient subtasks completed for an answer.
    PartialComplete { completed: u32, total: u32 },
    /// Resource budget exhausted.
    BudgetExhausted,
    /// Wall time exceeded.
    Timeout,
    /// Client cancelled.
    Cancelled,
    /// Unrecoverable failure.
    Failed { error: String },
}
```

---

## 5. SSE Events

The supervisor emits its own events that wrap child events:

```rust
pub enum SupervisorEvent {
    /// Supervisor started, planning phase beginning.
    SupervisorStarted {
        session_id: String,
        config: SupervisorConfig,
    },

    /// Planning complete, subtasks identified.
    PlanCreated {
        subtasks: Vec<SubtaskInfo>,
        dependencies: Vec<(String, String)>,
    },

    /// Child agent spawned.
    AgentSpawned {
        agent_id: AgentId,
        subtask_id: String,
        role: AgentRole,
    },

    /// Child agent event (relayed).
    AgentEvent {
        agent_id: AgentId,
        subtask_id: String,
        event: LoopEvent,
    },

    /// Child agent completed.
    AgentCompleted {
        agent_id: AgentId,
        subtask_id: String,
        summary: LoopSummary,
    },

    /// Rerouting occurred.
    Rerouted {
        from_agent: AgentId,
        to_agent: AgentId,
        reason: RerouteReason,
    },

    /// Aggregation phase started.
    AggregationStarted {
        completed_subtasks: u32,
        total_subtasks: u32,
    },

    /// Supervisor completed.
    SupervisorCompleted {
        summary: SupervisorSummary,
    },

    /// Global resource budget running low.
    BudgetWarning {
        resource: String,
        remaining_pct: f32,
    },

    /// Error at supervisor level.
    SupervisorError {
        message: String,
        recoverable: bool,
    },
}
```

Clients subscribe to supervisor events via the same SSE mechanism:

```
GET /api/agent/sessions/{supervisor_session_id}/stream
```

Child agent events are nested within `AgentEvent`, allowing the client to
track individual agent progress while seeing the overall orchestration.

---

## 6. API Integration

### 6.1 Supervisor Run Endpoint

```
POST /api/agent/supervise
```

```json
{
  "objective": "Refactor the authentication module to use JWT tokens",
  "decomposition": "llm_planned",
  "max_agents": 3,
  "routing": "dependency_aware",
  "shared_context": "summary_sharing",
  "resource_budget": {
    "total_iterations": 50,
    "total_tool_calls": 200,
    "total_tokens": 65536,
    "max_wall_time_secs": 600
  },
  "auto_approve": ["read_file", "list_files", "search_files"],
  "forbidden": ["bash:rm *"],
  "working_dir": "/home/user/project"
}
```

**Response:** SSE stream of `SupervisorEvent`s.

### 6.2 Subtask Override Endpoint

Allow the client to modify subtasks during execution:

```
POST /api/agent/{supervisor_session_id}/subtasks/{subtask_id}
```

```json
{
  "action": "cancel" | "reprioritize" | "modify",
  "priority": 1,
  "additional_context": "Focus on the OAuth2 flow, not basic auth"
}
```

### 6.3 Agent Inspection Endpoint

Inspect a specific child agent within a supervisor run:

```
GET /api/agent/{supervisor_session_id}/agents/{agent_id}
```

Returns the child agent's current session state, including iteration,
tool calls, and event history.

---

## 7. Resource Management

### 7.1 Budget Allocation

The supervisor splits the global `ResourceBudget` among children:

```rust
impl Supervisor {
    fn allocate_budget(&self, subtask: &Subtask) -> LoopConfig {
        let n = self.subtask_count() as u32;
        let complexity_weight = match subtask.estimated_complexity {
            Complexity::Low => 0.5,
            Complexity::Medium => 1.0,
            Complexity::High => 2.0,
        };

        // Weighted fair share
        let total_weight: f64 = self.subtasks.iter()
            .map(|s| s.complexity_weight())
            .sum();
        let share = complexity_weight / total_weight;

        LoopConfig {
            max_iterations: (self.budget.total_iterations as f64 * share) as u32,
            max_tool_calls: (self.budget.total_tool_calls as f64 * share) as u32,
            max_tokens: (self.budget.total_tokens as f64 * share) as u32,
            // Each child gets the full supervisor wall time — the supervisor itself
            // enforces the global wall time limit and terminates children if exceeded.
            max_wall_time: self.config.max_wall_time,
            ..LoopConfig::default()
        }
    }
}
```

### 7.2 Dynamic Rebalancing

When a child completes under budget, its surplus is redistributed:

1. Calculate unused resources from the completed child.
2. Distribute proportionally to still-running children.
3. Running children are notified (but their current iteration is not
   interrupted).

When the global budget runs low:

1. Supervisor emits a `SupervisorEvent::BudgetWarning`.
2. Running children receive tighter limits on their next continuation.
3. If budget is exhausted, supervisor moves to aggregation with partial results.

---

## 8. Failure Recovery

### 8.1 Recovery Strategies

```rust
pub enum RecoveryStrategy {
    /// Retry the same subtask with the same agent.
    Retry { max_retries: u32 },

    /// Reassign to a new agent with different configuration.
    Reassign { with_additional_context: bool },

    /// Skip the subtask and proceed with partial results.
    Skip,

    /// Escalate to the client for intervention.
    Escalate,

    /// Terminate the entire supervisor run.
    Abort,
}

impl Supervisor {
    fn recovery_strategy(&self, failure: &AgentFailure) -> RecoveryStrategy {
        match failure {
            // Engine errors (model issues) — retry once, then reassign
            AgentFailure::EngineError { retries, .. } if *retries < 1 => {
                RecoveryStrategy::Retry { max_retries: 1 }
            }
            AgentFailure::EngineError { .. } => {
                RecoveryStrategy::Reassign { with_additional_context: false }
            }

            // Tool errors — reassign with context about what failed
            AgentFailure::ToolError { .. } => {
                RecoveryStrategy::Reassign { with_additional_context: true }
            }

            // Transition errors (bugs) — escalate
            AgentFailure::TransitionError { .. } => {
                RecoveryStrategy::Escalate
            }

            // Resource exhaustion — extend or skip
            AgentFailure::ResourceExhausted { .. } => {
                if self.global_budget_remaining() > 0 {
                    RecoveryStrategy::Retry { max_retries: 1 }
                } else {
                    RecoveryStrategy::Skip
                }
            }
        }
    }
}
```

### 8.2 Cascading Failure Prevention

- **Max retries per subtask:** 2 (configurable).
- **Max total failures:** If more than 50% of subtasks fail, the supervisor
  moves to aggregation with whatever results are available.
- **Circuit breaker:** If 3 consecutive children fail with the same error type,
  the supervisor pauses dispatching and emits an escalation event.

---

## 9. Wellbeing Integration

The supervisor monitors aggregate wellbeing across all children.

> **Note:** The current `WellbeingState` in `wellbeing.rs` is an enum
> (`Healthy`, `Cautious`, `Concerned`, `Distressed`), not a struct with
> numeric fields. The `WellbeingSnapshot` contains the state enum plus
> metrics from `WellbeingMonitor`. The aggregate below works with the
> snapshot's numeric fields (from PAD model scores), not the enum directly.

```rust
pub struct AggregateWellbeing {
    pub agents_total: usize,
    pub agents_healthy: usize,
    pub agents_cautious: usize,
    pub agents_concerned: usize,
    pub agents_distressed: usize,
}

impl Supervisor {
    fn aggregate_wellbeing(&self) -> AggregateWellbeing {
        let snapshots: Vec<WellbeingState> = self.children.iter()
            .filter_map(|c| c.wellbeing_state())
            .collect();

        AggregateWellbeing {
            agents_total: snapshots.len(),
            agents_healthy: snapshots.iter()
                .filter(|s| matches!(s, WellbeingState::Healthy)).count(),
            agents_cautious: snapshots.iter()
                .filter(|s| matches!(s, WellbeingState::Cautious)).count(),
            agents_concerned: snapshots.iter()
                .filter(|s| matches!(s, WellbeingState::Concerned)).count(),
            agents_distressed: snapshots.iter()
                .filter(|s| matches!(s, WellbeingState::Distressed)).count(),
        }
    }
}
```

**Supervisor-level interventions:**

| Signal | Response |
|--------|----------|
| Single child `Cautious` | Let it continue (caution ≠ failure) |
| Multiple children `Concerned` on same subtask type | Reduce complexity, split subtasks further |
| Child stuck | Reroute (see §4.4) |
| Any child `Distressed` | Pause that child, reassign subtask |
| Majority of children `Concerned` or worse | Pause, re-plan remaining work |
| All children `Concerned` or worse | Escalate to client |

---

## 10. Implementation Phases

### Phase 1: Single-Agent Supervisor ✅

- [x] `Supervisor` struct with `SingleAgent` decomposition
- [x] Spawn one `LoopExecutor`, monitor events, collect results
- [x] Retry on failure
- [x] `POST /api/agent/supervise` endpoint
- [x] `SupervisorEvent` SSE stream

### Phase 2: Multi-Agent Parallel ✅

**Prerequisite:** Add communication primitives to `AgentCoordinator`:
- [x] `request_assistance(from, to, context)` method
- [x] `yield_to(from, partial_progress, suggested_expertise)` method
- [x] `share_discovery(from, discovery)` method
- [x] `get_shared_context(agent_id)` method

Then:
- [x] `LlmPlanned` decomposition with planning prompt
- [x] Parallel dispatch with concurrency limits
- [x] `AgentCoordinator` integration for identity and tool locks
- [x] Resource budget allocation and tracking
- [x] Dependency-aware routing

### Phase 3: Rerouting and Recovery ✅

- [x] Yield handling with partial progress forwarding
- [x] Stuck handling with context injection
- [x] Recovery strategies (retry, reassign, skip, escalate)
- [x] Dynamic budget rebalancing
- [x] Cascading failure prevention (circuit breaker, 50% threshold)

### Phase 4: Shared Context and Aggregation ✅

- [x] `SharedContextMode` implementations (`None`, `SummarySharing`, `FullSharing`, `SupervisorManaged`)
- [x] Aggregation executor for synthesizing multi-agent results (`ResultAggregator::synthesize_answer`)
- [x] Subtask override endpoint (`POST /api/agent/{session_id}/subtasks/{subtask_id}`)
- [x] Agent inspection endpoint (`GET /api/agent/{session_id}/agents/{agent_id}`)

### Phase 5: Wellbeing and Polish

- [ ] Aggregate wellbeing monitoring
- [ ] Supervisor-level interventions
- [ ] Client-provided subtask decomposition
- [ ] Performance tuning for high agent counts

---

## 11. Open Questions

### 11.1 Planning Model

Should the supervisor use the same model as child agents, or a separate
(potentially more capable) model for planning?

**Recommendation:** Same model by default, with an optional `planning_model`
override in `SupervisorConfig`. Planning needs reasoning ability but generates
fewer tokens than execution.

### 11.2 Tool Lock Granularity

Should tool locks be file-level (`write_file:/path/to/file`), tool-level
(`bash`), or both?

**Recommendation:** Both. File tools lock at path granularity.
Shell tools lock at tool-name granularity (only one `bash` at a time by
default, configurable).

### 11.3 Cross-Supervisor Coordination

Can multiple supervisor instances coordinate? (e.g., a "super-supervisor")

**Decision:** Deferred. The current design supports one level of supervision.
Nested supervision is architecturally possible (a supervisor is just another
executor) but not specified or tested.

### 11.4 Agent Affinity

Should the supervisor prefer routing related subtasks to the same agent
(for context locality) or fresh agents (for independence)?

**Recommendation:** Context locality by default. An agent that completed
subtask A and is available should handle subtask B if B depends on A.
Fresh agents are used for independent subtasks.

---

## 12. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-04 | Initial draft. Architecture, lifecycle, API, resource management, failure recovery, wellbeing integration. |
| 0.2.0 | 2026-02-15 | Phase 1-3 complete. Single-agent supervisor, multi-agent parallel dispatch, rerouting/recovery with structured failure types, circuit breaker, 50% failure threshold. |
| 0.3.0 | 2026-02-15 | Phase 4 complete. SharedContextMode (incl. SupervisorManaged), result aggregation with answer synthesis, subtask override endpoint, agent inspection endpoint. |
