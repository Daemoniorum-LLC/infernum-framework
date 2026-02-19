//! # Beleth
//!
//! *"The King commands legions"*
//!
//! Beleth is the agent framework for the Infernum ecosystem,
//! enabling autonomous task execution with tool use, planning, and memory.
//!
//! ## Features
//!
//! - **Tool System**: Extensible tool interface for agent actions
//! - **Planning**: Multiple planning strategies (ReAct, ToT, Hierarchical)
//! - **Memory**: Working, episodic, and semantic memory systems
//! - **Grimoire Integration**: Native support for Grimoire personas

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod agent;
pub mod agentic_loop;
pub mod claude_code_engine;
pub mod dynamic_context;
pub mod http_engine;
pub mod long_term_memory;
pub mod memory;
pub mod ooda;
pub mod planner;
pub mod react;
pub mod tool;
pub mod tools;
pub mod tracing_spans;
pub mod wellbeing;
pub mod wellbeing_persist;

#[cfg(test)]
mod proptest_suite;

pub use agent::{
    Agent, AgentAction, AgentBuilder, Persona, PersonaSource, PlanExecutionResult, PlanStepResult,
    StepResult, StepUsage,
};
pub use agentic_loop::{
    apply_config_override,
    build_resumed_messages,
    compute_aggregate_wellbeing,
    create_continuation_state,
    detect_meta_signal,
    rebalance_budget,
    supervisor_level_response,
    supervisor_wellbeing_response,
    // Multi-agent coordination
    AgentCoordinator,
    AgentId,
    AgentIdentity,
    AgentRole,
    // Multi-agent supervisor (§12)
    AgentWellbeingAction,
    AgentWellbeingState,
    AgenticLoop,
    AgenticToolResult,
    // Tool approval protocol (§9.4)
    ApprovalDecision,
    ApprovalError,
    ApprovalGate,
    ApprovalScope,
    // Coordination primitives (§7.2)
    AssistancePriority,
    AssistanceRequest,
    AssistanceResponse,
    AttemptSummary,
    AutonomyGrant,
    AutonomyGrantBuilder,
    BudgetAllocator,
    CircuitBreaker,
    Complexity,
    CompressionEvent,
    CompressionStrategy,
    ConcurrencyLimiter,
    Confidence,
    // Session continuation (§9.3)
    ConfigOverride,
    ContextMessage,
    ContextWindow,
    // Context management
    ContextWindowManager,
    ContinuationState,
    ContinuationStore,
    CoordinationError,
    CoordinationEvent,
    DecompositionStrategy,
    DependencyResolver,
    // Executor
    DetectedCall,
    DetectionConfig,
    Discovery,
    ExecutionOutcome,
    ExecutorConfig,
    ExplorationBranch,
    ExternalTermination,
    FailureType,
    InMemoryContinuationStore,
    IterationOutcome,
    LifecycleTracker,
    LoopConfig,
    LoopError,
    LoopEvent,
    LoopExecutor,
    LoopState,
    LoopStateSnapshot,
    LoopStatus,
    LoopSummary,
    MetaSignal,
    NaturalTermination,
    PendingApprovalInfo,
    PendingAssistance,
    PendingYield,
    Permission,
    QwenToolCallDetector,
    RerouteReason,
    RerouteResolver,
    ResourceBudget,
    ResourceConsumption,
    ResourceQuota,
    ResourceQuotaManager,
    ResourceTermination,
    ResultAggregator,
    ResultStatus,
    ResumeError,
    RoutingStrategy,
    SharedContext,
    SharedContextMode,
    StoreError,
    StuckRequest,
    Subtask,
    SubtaskResult,
    SubtaskStatus,
    Supervisor,
    SupervisorConfig,
    SupervisorError,
    SupervisorEvent,
    SupervisorSummary,
    SupervisorTermination,
    SupervisorWellbeingAction,
    TerminationReason,
    TokenBudget,
    ToolCallDetector,
    ToolError as AgenticToolError,
    ToolLock,
    ToolLockManager,
    ToolPattern,
    TransitionError,
    VisibilityPolicy,
    // Wellbeing bridge
    WellbeingAction,
    WellbeingAggregate,
    WellbeingBridge,
    WellbeingResponse,
    // Phase 5: Wellbeing tracker
    WellbeingTracker,
    YieldContext,
    YieldResult,
    // Phase 4: Result aggregation
    RerouteEvent,
};
pub use dynamic_context::{
    score_message_relevance, semantic_chunk, ChunkType, ContextComplexity, ContextConfig,
    DynamicContextManager, RelevanceFactors, SemanticChunk,
};
pub use claude_code_engine::{ClaudeCodeEngine, ClaudeTier};
pub use http_engine::{HttpEngine, HttpEngineError, SimpleMessage};
pub use long_term_memory::{ImportanceLevel, LongTermMemory, MemoryEntry, MemoryStats, MemoryType};
pub use memory::{
    AgentMemory, ConversationManager, ConversationStore, ConversationSummary,
    FileConversationStore, MemoryConversationStore, PersistentConversation,
    SerializableConversation, SerializableMessage, SummarizationStrategy,
};
pub use ooda::{
    DecisionAction, NoOpOodaCallback, OodaActionResult, OodaCallback, OodaCompletionReason,
    OodaConfig, OodaDecision, OodaExecutor, OodaObservation, OodaOrientation, OodaPhase,
    OodaResult, OodaStep,
};
pub use planner::{
    DefaultPlanner, HierarchicalTask, LLMPlanner, Plan, PlanStep, Planner, PlanningStrategy,
    ThoughtNode,
};
pub use react::{
    generate_observation_reasoning, parse_observation, parse_observation_with_validation,
    ActionType, CompletionReason, NoOpCallback, Observation, ReactAction, ReactCallback,
    ReactConfig, ReactExecutor, ReactResult, ReactStep,
};
pub use tool::{
    CalculatorTool, DateTimeTool, JsonTool, OutputValidationConfig, RiskLevel, TaskComplexity,
    Tool, ToolCall, ToolContext, ToolRegistry, ToolResult, ToolTimeoutConfig, ValidationIssue,
    ValidationResult,
};
pub use tools::{
    BashTool, ClaudeCodeTool, EditFileTool, ListFilesTool, ReadFileTool, SearchFilesTool,
    WriteFileTool,
};
pub use tracing_spans::{
    agent_execution_span, context_optimization_span, llm_call_span, memory_operation_span,
    memory_query_span, ooda_act_span, ooda_decide_span, ooda_iteration_span, ooda_observe_span,
    ooda_orient_span, persona_load_span, planning_span, react_step_span, semantic_chunking_span,
    tool_invocation_span, tool_validation_span, SpanExt,
};
pub use wellbeing::{
    DistressSignal, Intervention, WellbeingConfig, WellbeingMonitor, WellbeingSnapshot,
    WellbeingState,
};
pub use wellbeing_persist::{
    default_history_path, load_history, load_or_create_history, save_history, HistorySummary,
    PersistedHistory, PersistedSnapshot,
};
