//! Core types for the agentic loop.
//!
//! These types model the agentic loop's state machine, configuration,
//! termination conditions, meta-signals, and autonomy grants.
//!
//! Reference: AGENTIC-LOOP-SPEC.md v0.1.0

use std::time::Duration;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Loop State Machine
// ---------------------------------------------------------------------------

/// States of the agentic loop state machine.
///
/// ```text
/// Initialized → Generating → Detecting → Executing → Integrating → (continue → Generating)
///                              ↓             ↓
///                           Completed      (back to Detecting on exec complete)
///                           Stuck
///                           Yielded
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LoopState {
    /// Loop created, not yet started.
    Initialized,
    /// Generating model output.
    Generating,
    /// Analyzing output for tool calls and meta-signals.
    Detecting,
    /// Executing detected tool calls.
    Executing,
    /// Integrating tool results into context.
    Integrating,
    /// Agent provided a final answer — terminal.
    Completed,
    /// Agent declared it is stuck — terminal.
    Stuck,
    /// Agent yielded to another agent — terminal.
    Yielded,
}

impl LoopState {
    /// Returns `true` if this is a terminal state (no further transitions allowed).
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Stuck | Self::Yielded)
    }
}

/// High-level status of the loop, separate from the mechanical state.
///
/// Describes *how* the agent is doing, not *where* it is in the state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LoopStatus {
    /// Making forward progress.
    Progressing,
    /// Trying approaches, not yet converged.
    Exploring,
    /// Attempts not yielding results, but not stuck.
    Struggling,
    /// Explicitly requested help.
    Stuck,
    /// Finished.
    Completed,
    /// Externally stopped.
    Terminated,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the agentic loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopConfig {
    /// Maximum iterations before resource termination. Default: 10.
    pub max_iterations: u32,
    /// Maximum total tool calls. Default: 50.
    pub max_tool_calls: u32,
    /// Maximum wall-clock time. Default: 5 minutes.
    pub max_wall_time: Duration,
    /// Maximum tokens the model may generate across all iterations. Default: 16384.
    pub max_tokens: u32,
    /// Enable context compression when budget is tight. Default: true.
    pub context_compression: bool,
    /// Allow the agent to express uncertainty. Default: true.
    pub allow_uncertainty: bool,
    /// Allow the agent to yield to another agent. Default: true.
    pub allow_yield: bool,
    /// Preserve exploration history on termination. Default: true.
    pub preserve_exploration: bool,
    /// Detect implicit meta-signals from natural language. Default: true.
    pub detect_implicit_signals: bool,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            max_tool_calls: 50,
            max_wall_time: Duration::from_secs(300),
            max_tokens: 16384,
            context_compression: true,
            allow_uncertainty: true,
            allow_yield: true,
            preserve_exploration: true,
            detect_implicit_signals: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Token Budget
// ---------------------------------------------------------------------------

/// Tracks token usage across the loop's lifetime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBudget {
    /// Maximum tokens allowed.
    pub limit: u32,
    /// Tokens generated so far.
    pub used: u32,
}

impl TokenBudget {
    /// Creates a new budget with the given limit.
    pub fn new(limit: u32) -> Self {
        Self { limit, used: 0 }
    }

    /// Returns remaining tokens.
    pub fn remaining(&self) -> u32 {
        self.limit.saturating_sub(self.used)
    }

    /// Returns true if the budget is exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.used >= self.limit
    }

    /// Records token usage. Returns `true` if still within budget.
    pub fn consume(&mut self, tokens: u32) -> bool {
        self.used = self.used.saturating_add(tokens);
        !self.is_exhausted()
    }
}

// ---------------------------------------------------------------------------
// Termination
// ---------------------------------------------------------------------------

/// Why the loop terminated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TerminationReason {
    /// Agent terminated naturally (answer, stuck, yield, task complete).
    Natural(NaturalTermination),
    /// Resource limit reached.
    Resource(ResourceTermination),
    /// External intervention.
    External(ExternalTermination),
}

/// Natural termination — the agent decided to stop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NaturalTermination {
    /// Agent provided a final answer.
    AnswerProvided {
        /// The answer content.
        answer: String,
        /// Agent's self-assessed confidence (0.0–1.0).
        confidence: f32,
    },
    /// Agent yielded to another agent.
    AgentYielded {
        /// Partial progress, if any.
        partial: Option<String>,
        /// Why the agent is yielding.
        reason: String,
    },
    /// Agent is stuck and requested help.
    AgentStuck {
        /// Number of approaches attempted.
        attempts: u32,
        /// What kind of help is needed.
        request: StuckRequest,
    },
    /// Task finished with no more tool calls or answers.
    TaskComplete,
}

/// Resource termination — a limit was reached.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceTermination {
    /// Hit iteration limit.
    MaxIterations {
        /// Iterations completed.
        completed: u32,
        /// Configured limit.
        limit: u32,
    },
    /// Hit token budget.
    TokenBudgetExhausted {
        /// Tokens generated.
        generated: u32,
        /// Configured budget.
        budget: u32,
    },
    /// Hit wall time.
    WallTimeExceeded {
        /// Elapsed time.
        elapsed: Duration,
        /// Configured limit.
        limit: Duration,
    },
    /// Hit tool call limit.
    ToolCallLimitReached {
        /// Tool calls made.
        calls: u32,
        /// Configured limit.
        limit: u32,
    },
}

/// External termination — someone else stopped us.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExternalTermination {
    /// Client cancelled the request.
    ClientCancelled,
    /// Operator intervention.
    OperatorTerminated {
        /// Reason for termination.
        reason: String,
    },
    /// System shutting down.
    SystemShutdown,
}

// ---------------------------------------------------------------------------
// Meta-Signals
// ---------------------------------------------------------------------------

/// Signals emitted by the agent beyond tool calls.
///
/// These communicate agent state to the loop orchestrator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetaSignal {
    /// "I have an answer."
    Answer {
        /// The answer content.
        content: String,
        /// Self-assessed confidence (0.0–1.0).
        confidence: f32,
        /// Caveats or assumptions.
        caveats: Vec<String>,
    },
    /// "I'm not certain."
    Uncertain {
        /// Partial answer, if any.
        partial_answer: Option<String>,
        /// Information the agent is missing.
        missing_information: Vec<String>,
        /// What would help the agent.
        would_help: Vec<String>,
    },
    /// "I've tried but I'm stuck."
    Stuck {
        /// Approaches the agent has tried.
        attempts: Vec<AttemptSummary>,
        /// Agent's hypothesis about the problem.
        hypothesis: Option<String>,
        /// What kind of help is needed.
        request: StuckRequest,
    },
    /// "Another agent might do better."
    Yield {
        /// Partial progress so far.
        partial_progress: Option<String>,
        /// What expertise would help.
        suggested_expertise: Vec<String>,
    },
    /// "I need to think about this."
    Thinking {
        /// What the agent is considering.
        direction: String,
        /// Estimated remaining steps.
        estimated_steps: Option<u32>,
    },
}

/// Summary of a failed approach the agent tried.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttemptSummary {
    /// Description of what was tried.
    pub description: String,
    /// Why it didn't work.
    pub outcome: String,
}

/// What the stuck agent is requesting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StuckRequest {
    /// Needs clarification — has specific questions.
    Clarification(Vec<String>),
    /// Needs more context about a topic.
    MoreContext {
        /// What the agent needs context about.
        about: String,
    },
    /// Needs different tools than what's available.
    DifferentTools {
        /// What capabilities are needed.
        need: Vec<String>,
    },
    /// Needs a human to intervene.
    HumanIntervention {
        /// Why human help is needed.
        reason: String,
    },
}

/// Configuration for meta-signal detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
    /// Whether to detect implicit signals from natural language patterns.
    pub detect_implicit: bool,
    /// Minimum confidence for implicit detection (0.0–1.0).
    pub implicit_threshold: f32,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            detect_implicit: true,
            implicit_threshold: 0.7,
        }
    }
}

// ---------------------------------------------------------------------------
// Autonomy
// ---------------------------------------------------------------------------

/// Defines what the agent can do without asking permission.
///
/// Principle 5: Agents work best when they know their constraints upfront.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AutonomyGrant {
    /// Patterns that are auto-approved (no confirmation needed).
    pub auto_approve: Vec<ToolPattern>,
    /// Patterns that require explicit approval before execution.
    pub require_approval: Vec<ToolPattern>,
    /// Patterns that are forbidden entirely.
    pub forbidden: Vec<ToolPattern>,
}

/// A pattern that matches tool calls.
///
/// Uses glob-style matching for file paths and command strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolPattern {
    /// Matches read-type tools with the given path glob.
    Read(String),
    /// Matches write-type tools with the given path glob.
    Write(String),
    /// Matches bash/shell tools with the given command glob.
    Bash(String),
    /// Matches any tool by name.
    Tool(String),
}

/// Result of checking a tool call against an autonomy grant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission {
    /// Tool is auto-approved.
    Allowed,
    /// Tool requires explicit approval.
    RequiresApproval,
    /// Tool is forbidden.
    Forbidden,
}

/// Outcome of attempting to execute a tool call.
#[derive(Debug, Clone)]
pub enum ExecutionOutcome {
    /// Tool executed successfully.
    Completed(AgenticToolResult),
    /// Tool call is forbidden by autonomy grant.
    Denied {
        /// Why it was denied.
        reason: String,
    },
    /// Tool call needs approval before proceeding.
    PendingApproval {
        /// The tool call awaiting approval.
        call_id: String,
    },
    /// Tool execution failed.
    Failed(ToolError),
}

// ---------------------------------------------------------------------------
// Tool Results (namespaced to avoid conflict with beleth::tool::ToolResult)
// ---------------------------------------------------------------------------

/// Structured tool result for agentic loop consumption.
///
/// Richer than `beleth::tool::ToolResult` — designed for agent reasoning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgenticToolResult {
    /// ID linking back to the tool call.
    pub call_id: String,
    /// Which tool was called.
    pub tool_name: String,
    /// Outcome status.
    pub status: ResultStatus,
    /// Structured result data.
    pub data: serde_json::Value,
    /// How confident is the result.
    pub confidence: Confidence,
    /// How long the tool took (milliseconds).
    pub latency_ms: u64,
    /// Whether the output was truncated.
    pub truncated: bool,
}

/// Status of a tool execution result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ResultStatus {
    /// Tool succeeded fully.
    Success,
    /// Tool partially succeeded.
    PartialSuccess {
        /// Number of operations that completed.
        completed: u32,
        /// Number of operations that failed.
        failed: u32,
    },
    /// Tool succeeded but found nothing (distinct from error).
    Empty,
    /// Tool failed.
    Failed {
        /// Whether the failure is recoverable (retry might help).
        recoverable: bool,
    },
}

/// How confident the tool result is.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Confidence {
    /// Result is exact (file contents, API response).
    Measured,
    /// Result is computed/inferred.
    Estimated,
    /// Result may be incomplete or stale.
    Uncertain,
    /// Confidence cannot be determined.
    Unknown,
}

/// Error from tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolError {
    /// ID linking back to the tool call.
    pub call_id: String,
    /// Error description.
    pub error: String,
    /// Whether retrying might succeed.
    pub recoverable: bool,
    /// Suggestion for the agent.
    pub suggestion: Option<String>,
}

// ---------------------------------------------------------------------------
// Exploration Tracking
// ---------------------------------------------------------------------------

/// A branch of exploration the agent pursued.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationBranch {
    /// Human/agent-readable description of this approach.
    pub description: String,
    /// Tool calls made during this exploration.
    pub tool_calls: Vec<String>,
    /// Whether this branch yielded useful results.
    pub productive: bool,
    /// Summary of findings, if any.
    pub findings: Option<String>,
}

// ---------------------------------------------------------------------------
// Loop Summary
// ---------------------------------------------------------------------------

/// Summary returned when the loop terminates, preserving partial work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopSummary {
    /// Why the loop terminated.
    pub termination: TerminationReason,
    /// Number of iterations completed.
    pub iterations_completed: u32,
    /// Total tool calls made.
    pub tool_calls_made: u32,
    /// Total tokens generated.
    pub tokens_generated: u32,
    /// Total wall-clock time.
    pub wall_time: Duration,
    /// Partial answer if the agent didn't finish.
    pub partial_answer: Option<String>,
    /// Exploration branches pursued.
    pub exploration_summary: Vec<ExplorationBranch>,
    /// Summary of all tool results.
    pub tool_results_summary: Vec<AgenticToolResult>,
    /// Whether the loop can be resumed.
    pub can_resume: bool,
}

// ---------------------------------------------------------------------------
// Context Window
// ---------------------------------------------------------------------------

/// Manages the context window for the agentic loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextWindow {
    /// Conversation messages.
    pub messages: Vec<ContextMessage>,
    /// Current loop state visible to the agent.
    pub system_state: LoopStateSnapshot,
    /// Original token count before any compression.
    pub original_token_count: u32,
    /// Current token count after compression.
    pub current_token_count: u32,
    /// Compression events applied.
    pub compressions_applied: Vec<CompressionEvent>,
}

/// A message in the agentic context window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextMessage {
    /// Role (system, user, assistant, tool).
    pub role: String,
    /// Content of the message.
    pub content: String,
    /// Tool call ID, if this is a tool result.
    pub tool_call_id: Option<String>,
}

/// Snapshot of loop state visible to the agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopStateSnapshot {
    /// Current iteration number.
    pub iteration: u32,
    /// Maximum iterations allowed.
    pub max_iterations: u32,
    /// Remaining token budget.
    pub token_budget_remaining: u32,
    /// Names of available tools.
    pub tools_available: Vec<String>,
    /// Context pressure (0.0–1.0), how full the context window is.
    pub context_pressure: f32,
}

/// How context was compressed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionStrategy {
    /// Summarize old tool results, keep recent ones verbatim.
    SummarizeOldResults {
        /// Number of recent results to keep verbatim.
        keep_recent: u32,
    },
    /// Keep only tool results that led to progress.
    PruneDeadEnds,
    /// Compress exploration branches that didn't pan out.
    CollapseExploration {
        /// Token budget for summaries.
        summary_tokens: u32,
    },
    /// Ask the agent what to keep.
    AgentDirected,
}

/// Record of a compression event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionEvent {
    /// Strategy used.
    pub strategy: CompressionStrategy,
    /// Tokens saved.
    pub tokens_saved: u32,
    /// Iteration at which compression occurred.
    pub at_iteration: u32,
}

// ---------------------------------------------------------------------------
// SSE Streaming Events
// ---------------------------------------------------------------------------

/// Events streamed to the client throughout loop execution.
///
/// Reference: AGENTIC-LOOP-SPEC.md §5.1
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", content = "data")]
#[serde(rename_all = "snake_case")]
pub enum LoopEvent {
    // -- Loop lifecycle --
    /// Loop has started.
    LoopStarted {
        /// Session identifier.
        session_id: String,
        /// Loop configuration.
        config: LoopConfig,
    },
    /// A new iteration has begun.
    IterationStarted {
        /// Iteration number.
        iteration: u32,
        /// Current loop status.
        status: LoopStatus,
    },
    /// An iteration has completed.
    IterationCompleted {
        /// Iteration number.
        iteration: u32,
        /// What happened in this iteration.
        outcome: IterationOutcome,
    },
    /// Loop has finished.
    LoopCompleted {
        /// Final summary.
        summary: LoopSummary,
    },

    // -- Generation --
    /// A token was generated.
    TokenGenerated {
        /// The token text.
        token: String,
    },
    /// Generation phase completed.
    GenerationCompleted {
        /// Full generated content.
        content: String,
        /// Token count.
        tokens: u32,
    },

    // -- Tool execution --
    /// Tool call detected in model output.
    ToolCallDetected {
        /// Tool call ID.
        call_id: String,
        /// Tool name.
        tool: String,
    },
    /// Tool execution started.
    ToolExecutionStarted {
        /// Tool call ID.
        call_id: String,
        /// Tool name.
        tool: String,
    },
    /// Tool execution completed.
    ToolExecutionCompleted {
        /// Tool call ID.
        call_id: String,
        /// Tool result.
        result: AgenticToolResult,
    },
    /// Tool requires approval before execution.
    ToolApprovalRequired {
        /// Tool call ID.
        call_id: String,
        /// Tool name.
        tool: String,
    },

    // -- Meta-signals --
    /// Meta-signal detected in model output.
    MetaSignalDetected {
        /// The detected signal.
        signal: MetaSignal,
    },

    // -- Context management --
    /// Context was compressed to fit budget.
    ContextCompressed {
        /// Strategy used.
        strategy: CompressionStrategy,
        /// Tokens saved.
        saved_tokens: u32,
    },

    // -- Errors --
    /// An error occurred (system issue, not agent failure).
    Error {
        /// Error message.
        message: String,
        /// Whether the error is recoverable.
        recoverable: bool,
    },
}

/// What happened during an iteration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IterationOutcome {
    /// Tool calls were executed.
    ToolCallsExecuted {
        /// Number of tool calls.
        count: u32,
    },
    /// Agent provided an answer.
    AnswerProvided,
    /// Agent is stuck.
    Stuck,
    /// Agent yielded.
    Yielded,
    /// Iteration was cut short by resource limit.
    ResourceLimitReached,
}

// ---------------------------------------------------------------------------
// Transition Errors
// ---------------------------------------------------------------------------

/// Errors from invalid state transitions.
#[derive(Debug, Clone, thiserror::Error)]
pub enum TransitionError {
    /// Attempted an invalid state transition.
    #[error("invalid transition from {from:?}: {reason}")]
    InvalidTransition {
        /// Current state.
        from: LoopState,
        /// Why the transition is invalid.
        reason: String,
    },
    /// A resource limit was reached.
    #[error("resource limit: {0}")]
    ResourceLimitReached(String),
    /// Loop has already terminated.
    #[error("loop already terminated")]
    AlreadyTerminated,
}
