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
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![deny(clippy::unwrap_used)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod agent;
pub mod agentic_loop;
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
pub use memory::{
    AgentMemory, ConversationManager, ConversationStore, ConversationSummary,
    FileConversationStore, MemoryConversationStore, PersistentConversation,
    SerializableConversation, SerializableMessage, SummarizationStrategy,
};
pub use planner::{DefaultPlanner, HierarchicalTask, LLMPlanner, Plan, PlanStep, Planner, PlanningStrategy, ThoughtNode};
pub use tool::{
    CalculatorTool, DateTimeTool, JsonTool, OutputValidationConfig, RiskLevel, TaskComplexity,
    Tool, ToolCall, ToolContext, ToolRegistry, ToolResult, ToolTimeoutConfig, ValidationIssue,
    ValidationResult,
};
pub use react::{
    generate_observation_reasoning, parse_observation, parse_observation_with_validation,
    ActionType, CompletionReason, NoOpCallback, Observation, ReactAction, ReactCallback,
    ReactConfig, ReactExecutor, ReactResult, ReactStep,
};
pub use ooda::{
    DecisionAction, NoOpOodaCallback, OodaActionResult, OodaCallback, OodaCompletionReason,
    OodaConfig, OodaDecision, OodaExecutor, OodaObservation, OodaOrientation, OodaPhase,
    OodaResult, OodaStep,
};
pub use long_term_memory::{
    ImportanceLevel, LongTermMemory, MemoryEntry, MemoryStats, MemoryType,
};
pub use dynamic_context::{
    ChunkType, ContextComplexity, ContextConfig, DynamicContextManager, RelevanceFactors,
    SemanticChunk, score_message_relevance, semantic_chunk,
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
pub use agentic_loop::{
    AgenticLoop, AgenticToolResult, AttemptSummary, AutonomyGrant, AutonomyGrantBuilder,
    Confidence, CompressionEvent, CompressionStrategy, ContextMessage, ContextWindow,
    DetectionConfig, ExecutionOutcome, ExplorationBranch, ExternalTermination,
    IterationOutcome, LoopConfig, LoopEvent, LoopState, LoopStateSnapshot, LoopStatus,
    LoopSummary, MetaSignal, NaturalTermination, Permission, ResourceTermination, ResultStatus,
    StuckRequest, TerminationReason, TokenBudget, ToolError as AgenticToolError, ToolPattern,
    TransitionError, detect_meta_signal,
    // Executor
    DetectedCall, ExecutorConfig, LoopError, LoopExecutor, QwenToolCallDetector,
    ToolCallDetector,
    // Context management
    ContextWindowManager,
    // Multi-agent coordination
    AgentCoordinator, AgentId, AgentIdentity, AgentRole, ResourceQuota,
    ResourceQuotaManager, ToolLock, ToolLockManager,
    // Coordination primitives (§7.2)
    AssistancePriority, AssistanceRequest, AssistanceResponse, CoordinationError,
    CoordinationEvent, Discovery, PendingAssistance, PendingYield, SharedContext,
    VisibilityPolicy, YieldContext, YieldResult,
    // Tool approval protocol (§9.4)
    ApprovalDecision, ApprovalError, ApprovalGate, ApprovalScope, PendingApprovalInfo,
    // Session continuation (§9.3)
    ConfigOverride, ContinuationState, ContinuationStore, InMemoryContinuationStore,
    ResumeError, StoreError, apply_config_override, build_resumed_messages,
    create_continuation_state,
    // Wellbeing bridge
    WellbeingAction, WellbeingBridge,
};
pub use http_engine::{HttpEngine, HttpEngineError, SimpleMessage};
pub use tools::{
    BashTool, ClaudeCodeTool, EditFileTool, ListFilesTool, ReadFileTool, SearchFilesTool,
    WriteFileTool,
};
