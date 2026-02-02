//! Tracing instrumentation for agent framework observability.
//!
//! Provides structured spans for:
//! - Agent execution lifecycle
//! - OODA loop phases
//! - Tool invocations
//! - Memory operations
//! - Context optimization
//!
//! # Usage
//!
//! ```rust,ignore
//! use beleth::tracing_spans::*;
//!
//! // Automatically creates a span for agent execution
//! let _span = agent_execution_span("solve math problem", "gpt-4");
//! ```

use tracing::{info_span, Span};

/// Creates a span for the overall agent execution.
///
/// # Arguments
/// * `objective` - The agent's task objective
/// * `model` - The LLM model being used
///
/// # Fields recorded
/// * `otel.name` - "agent.execute"
/// * `agent.objective` - Task description
/// * `agent.model` - Model identifier
/// * `agent.iteration` - Empty (set dynamically)
#[inline]
pub fn agent_execution_span(objective: &str, model: &str) -> Span {
    info_span!(
        "agent.execute",
        otel.name = "agent.execute",
        agent.objective = %objective,
        agent.model = %model,
        agent.iteration = tracing::field::Empty,
        agent.tool_count = tracing::field::Empty,
        agent.duration_ms = tracing::field::Empty,
    )
}

/// Creates a span for an OODA loop iteration.
///
/// # Arguments
/// * `iteration` - Current iteration number
/// * `phase` - OODA phase (observe, orient, decide, act)
#[inline]
pub fn ooda_iteration_span(iteration: u32, phase: &str) -> Span {
    info_span!(
        "ooda.iteration",
        otel.name = "ooda.iteration",
        ooda.iteration = iteration,
        ooda.phase = %phase,
        ooda.decision = tracing::field::Empty,
        ooda.confidence = tracing::field::Empty,
    )
}

/// Creates a span for the observe phase.
#[inline]
pub fn ooda_observe_span() -> Span {
    info_span!(
        "ooda.observe",
        otel.name = "ooda.observe",
        observe.context_size = tracing::field::Empty,
        observe.tools_available = tracing::field::Empty,
    )
}

/// Creates a span for the orient phase.
#[inline]
pub fn ooda_orient_span() -> Span {
    info_span!(
        "ooda.orient",
        otel.name = "ooda.orient",
        orient.complexity = tracing::field::Empty,
        orient.analysis = tracing::field::Empty,
    )
}

/// Creates a span for the decide phase.
#[inline]
pub fn ooda_decide_span() -> Span {
    info_span!(
        "ooda.decide",
        otel.name = "ooda.decide",
        decide.action_type = tracing::field::Empty,
        decide.tool_name = tracing::field::Empty,
        decide.confidence = tracing::field::Empty,
    )
}

/// Creates a span for the act phase.
#[inline]
pub fn ooda_act_span(action_type: &str) -> Span {
    info_span!(
        "ooda.act",
        otel.name = "ooda.act",
        act.action_type = %action_type,
        act.success = tracing::field::Empty,
        act.error = tracing::field::Empty,
        act.duration_ms = tracing::field::Empty,
    )
}

/// Creates a span for tool invocation.
///
/// # Arguments
/// * `tool_name` - Name of the tool being invoked
/// * `risk_level` - Tool's risk classification
#[inline]
pub fn tool_invocation_span(tool_name: &str, risk_level: &str) -> Span {
    info_span!(
        "tool.invoke",
        otel.name = "tool.invoke",
        tool.name = %tool_name,
        tool.risk_level = %risk_level,
        tool.input_size = tracing::field::Empty,
        tool.output_size = tracing::field::Empty,
        tool.success = tracing::field::Empty,
        tool.duration_ms = tracing::field::Empty,
        tool.error = tracing::field::Empty,
    )
}

/// Creates a span for tool validation.
#[inline]
pub fn tool_validation_span(tool_name: &str) -> Span {
    info_span!(
        "tool.validate",
        otel.name = "tool.validate",
        tool.name = %tool_name,
        validation.passed = tracing::field::Empty,
        validation.issues = tracing::field::Empty,
    )
}

/// Creates a span for memory operations.
///
/// # Arguments
/// * `operation` - The memory operation (store, retrieve, search, delete)
/// * `memory_type` - Type of memory being accessed
#[inline]
pub fn memory_operation_span(operation: &str, memory_type: &str) -> Span {
    info_span!(
        "memory.operation",
        otel.name = "memory.operation",
        memory.operation = %operation,
        memory.type = %memory_type,
        memory.entries_affected = tracing::field::Empty,
        memory.duration_ms = tracing::field::Empty,
    )
}

/// Creates a span for memory retrieval queries.
#[inline]
pub fn memory_query_span(query: &str) -> Span {
    info_span!(
        "memory.query",
        otel.name = "memory.query",
        memory.query = %query,
        memory.results = tracing::field::Empty,
        memory.cache_hit = tracing::field::Empty,
    )
}

/// Creates a span for context optimization.
///
/// # Arguments
/// * `message_count` - Number of messages being optimized
/// * `complexity` - Task complexity classification
#[inline]
pub fn context_optimization_span(message_count: usize, complexity: &str) -> Span {
    info_span!(
        "context.optimize",
        otel.name = "context.optimize",
        context.message_count = message_count,
        context.complexity = %complexity,
        context.tokens_before = tracing::field::Empty,
        context.tokens_after = tracing::field::Empty,
        context.reduction_pct = tracing::field::Empty,
    )
}

/// Creates a span for semantic chunking.
#[inline]
pub fn semantic_chunking_span(content_size: usize, max_chunk_size: usize) -> Span {
    info_span!(
        "context.chunk",
        otel.name = "context.chunk",
        chunk.content_size = content_size,
        chunk.max_size = max_chunk_size,
        chunk.count = tracing::field::Empty,
        chunk.duration_ms = tracing::field::Empty,
    )
}

/// Creates a span for LLM API calls.
///
/// # Arguments
/// * `model` - Model being called
/// * `purpose` - Purpose of the call (reasoning, tool_selection, summarization)
#[inline]
pub fn llm_call_span(model: &str, purpose: &str) -> Span {
    info_span!(
        "llm.call",
        otel.name = "llm.call",
        llm.model = %model,
        llm.purpose = %purpose,
        llm.prompt_tokens = tracing::field::Empty,
        llm.completion_tokens = tracing::field::Empty,
        llm.total_tokens = tracing::field::Empty,
        llm.duration_ms = tracing::field::Empty,
        llm.stream = tracing::field::Empty,
    )
}

/// Creates a span for planning operations.
#[inline]
pub fn planning_span(strategy: &str, task: &str) -> Span {
    info_span!(
        "planner.plan",
        otel.name = "planner.plan",
        planner.strategy = %strategy,
        planner.task = %task,
        planner.steps = tracing::field::Empty,
        planner.duration_ms = tracing::field::Empty,
    )
}

/// Creates a span for ReAct reasoning steps.
#[inline]
pub fn react_step_span(step_number: u32) -> Span {
    info_span!(
        "react.step",
        otel.name = "react.step",
        react.step = step_number,
        react.thought = tracing::field::Empty,
        react.action = tracing::field::Empty,
        react.observation = tracing::field::Empty,
    )
}

/// Creates a span for persona/grimoire loading.
#[inline]
pub fn persona_load_span(persona_name: &str) -> Span {
    info_span!(
        "persona.load",
        otel.name = "persona.load",
        persona.name = %persona_name,
        persona.source = tracing::field::Empty,
        persona.tools = tracing::field::Empty,
    )
}

/// Span extensions for recording values after span creation.
pub trait SpanExt {
    /// Record a success/failure status.
    fn record_success(&self, success: bool);

    /// Record duration in milliseconds.
    fn record_duration_ms(&self, duration_ms: u64);

    /// Record an error message.
    fn record_error(&self, error: &str);
}

impl SpanExt for Span {
    fn record_success(&self, success: bool) {
        self.record("success", success);
    }

    fn record_duration_ms(&self, duration_ms: u64) {
        self.record("duration_ms", duration_ms);
    }

    fn record_error(&self, error: &str) {
        self.record("error", error);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    #[test]
    fn test_span_creation() {
        // Initialize a no-op subscriber for testing
        let _ = tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer().with_test_writer())
            .try_init();

        // Verify spans can be created without panicking
        let _span = agent_execution_span("test objective", "test-model");
        let _span = ooda_iteration_span(1, "observe");
        let _span = ooda_observe_span();
        let _span = ooda_orient_span();
        let _span = ooda_decide_span();
        let _span = ooda_act_span("tool_use");
        let _span = tool_invocation_span("calculator", "low");
        let _span = tool_validation_span("calculator");
        let _span = memory_operation_span("store", "decision");
        let _span = memory_query_span("authentication");
        let _span = context_optimization_span(100, "moderate");
        let _span = semantic_chunking_span(5000, 2000);
        let _span = llm_call_span("gpt-4", "reasoning");
        let _span = planning_span("hierarchical", "implement feature");
        let _span = react_step_span(1);
        let _span = persona_load_span("code-reviewer");
    }

    #[test]
    fn test_span_extension() {
        let span = agent_execution_span("test", "model");
        span.record_success(true);
        span.record_duration_ms(150);
        span.record_error("test error");
    }
}
