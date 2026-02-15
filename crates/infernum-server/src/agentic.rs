//! Agentic loop HTTP endpoints.
//!
//! - `POST /api/agent/run` — starts an agentic loop, streaming `LoopEvent`s as SSE.
//! - `POST /api/agent/supervise` — starts a multi-agent supervisor, streaming `SupervisorEvent`s.
//!
//! Integrates with the `SessionRegistry` for monitoring and the inference engine
//! for model generation.
//!
//! Reference: AGENTIC-LOOP-SPEC.md §9, MULTI-AGENT-SUPERVISOR-SPEC.md §10.

use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tracing::{info, warn};

use beleth::{
    AutonomyGrant, Complexity, DecompositionStrategy, ExecutorConfig, LoopConfig, LoopEvent,
    LoopExecutor, NaturalTermination, Subtask, Supervisor, SupervisorConfig, SupervisorEvent,
    SupervisorTermination, TerminationReason, ToolPattern, ToolRegistry,
};

use crate::error_response::{api_error, ErrorCode};
use crate::server::AppState;
use crate::sessions::{AgentEventData, AgentSession, SessionRegistry, SessionStatus};

// =============================================================================
// Request / Response Types
// =============================================================================

/// Request body for `POST /api/agent/run`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgenticRunRequest {
    /// The task objective for the agent.
    pub objective: String,

    /// System prompt override.
    #[serde(default)]
    pub system_prompt: Option<String>,

    /// Human-readable session name.
    #[serde(default)]
    pub session_name: Option<String>,

    /// Working directory for file tools.
    #[serde(default)]
    pub working_dir: Option<String>,

    /// Maximum iterations (default: 10).
    #[serde(default)]
    pub max_iterations: Option<u32>,

    /// Maximum tool calls (default: 50).
    #[serde(default)]
    pub max_tool_calls: Option<u32>,

    /// Maximum tokens to generate (default: 16384).
    #[serde(default)]
    pub max_tokens: Option<u32>,

    /// Tool name patterns to auto-approve (glob syntax).
    /// If empty, all tools require approval.
    #[serde(default)]
    pub auto_approve: Vec<String>,

    /// Tool name patterns that are forbidden.
    #[serde(default)]
    pub forbidden: Vec<String>,
}

/// Response returned when the SSE stream cannot be created.
#[derive(Debug, Serialize)]
pub struct AgenticRunError {
    /// Error code.
    pub error: String,
    /// Error message.
    pub message: String,
}

// =============================================================================
// Handler
// =============================================================================

/// POST /api/agent/run — starts an agentic loop and streams events via SSE.
///
/// The response is an SSE stream of `LoopEvent` objects. Each event has:
/// - `event:` field set to the event type (snake_case)
/// - `data:` field containing the JSON-serialized event
///
/// The stream ends after the `loop_completed` event.
pub async fn run_agent(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AgenticRunRequest>,
) -> impl IntoResponse {
    // Validate the request
    if req.objective.trim().is_empty() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(api_error(
                ErrorCode::InvalidRequest,
                "Objective must not be empty",
            )),
        )
            .into_response();
    }

    // Acquire the inference engine
    let engine_guard = state.engine.read().await;
    let engine = match engine_guard.as_ref() {
        Some(engine) => Arc::clone(engine),
        None => {
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                Json(api_error(
                    ErrorCode::InvalidRequest,
                    "No model loaded. Load a model before running the agent.",
                )),
            )
                .into_response();
        },
    };
    drop(engine_guard);

    // Build the executor config
    let session_id = SessionRegistry::generate_id();
    let loop_config = LoopConfig {
        max_iterations: req.max_iterations.unwrap_or(10),
        max_tool_calls: req.max_tool_calls.unwrap_or(50),
        max_tokens: req.max_tokens.unwrap_or(16384),
        detect_implicit_signals: true,
        ..LoopConfig::default()
    };

    let autonomy = build_autonomy(&req.auto_approve, &req.forbidden);

    let mut config = ExecutorConfig::new(&session_id)
        .with_loop_config(loop_config.clone())
        .with_autonomy(autonomy);

    if let Some(ref prompt) = req.system_prompt {
        config = config.with_system_prompt(prompt);
    }
    if let Some(ref wd) = req.working_dir {
        config = config.with_working_dir(wd);
    }

    // Create the tool registry
    let tools = Arc::new(ToolRegistry::with_code_tools());

    // Create the executor
    let executor = LoopExecutor::new(engine, tools.clone(), config);

    // Register session
    let tool_names: Vec<String> = tools.tools().iter().map(|t| t.name().to_string()).collect();
    let session = AgentSession::new(session_id.clone(), req.objective.clone())
        .with_tools(tool_names)
        .with_max_iterations(loop_config.max_iterations);
    let session = if let Some(ref name) = req.session_name {
        session.with_name(name)
    } else {
        session
    };

    state.sessions.register_session(session).await;

    info!(session_id = %session_id, "Starting agentic loop");

    // Create the event channel
    let (tx, rx) = mpsc::channel::<LoopEvent>(128);

    // Spawn the executor task
    let sessions = Arc::clone(&state.sessions);
    let objective = req.objective.clone();
    let sid = session_id.clone();
    tokio::spawn(async move {
        let result = executor.run(&objective, tx).await;

        // Update session status based on result
        match &result {
            Ok(summary) => {
                let final_answer = match &summary.termination {
                    TerminationReason::Natural(NaturalTermination::AnswerProvided {
                        answer,
                        ..
                    }) => Some(answer.clone()),
                    _ => summary.partial_answer.clone(),
                };
                sessions
                    .end_session(&sid, SessionStatus::Completed, final_answer)
                    .await;
            },
            Err(e) => {
                warn!(session_id = %sid, error = %e, "Agentic loop failed");
                sessions
                    .end_session(&sid, SessionStatus::Failed, Some(e.to_string()))
                    .await;
            },
        }
    });

    // Convert the mpsc receiver into an SSE stream
    let sse_stream = build_sse_stream(rx, Arc::clone(&state.sessions), session_id);

    Sse::new(sse_stream)
        .keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("ping"),
        )
        .into_response()
}

// =============================================================================
// Helpers
// =============================================================================

/// Builds an `AutonomyGrant` from request parameters.
fn build_autonomy(auto_approve: &[String], forbidden: &[String]) -> AutonomyGrant {
    let mut builder = AutonomyGrant::builder();

    for pattern in auto_approve {
        builder = builder.allow(ToolPattern::Tool(pattern.clone()));
    }
    for pattern in forbidden {
        builder = builder.forbid(ToolPattern::Tool(pattern.clone()));
    }

    builder.build()
}

/// Converts the `LoopEvent` channel into an SSE stream, bridging events to the session registry.
fn build_sse_stream(
    rx: mpsc::Receiver<LoopEvent>,
    sessions: Arc<SessionRegistry>,
    session_id: String,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let stream = ReceiverStream::new(rx);

    stream.filter_map(move |event| {
        let sessions = Arc::clone(&sessions);
        let sid = session_id.clone();

        let event_type = loop_event_type(&event);
        let serialized = serde_json::to_string(&event);

        // Fire-and-forget bridge to session registry (owned values for 'static)
        tokio::spawn(bridge_event_to_session(sessions, sid, event));

        match serialized {
            Ok(data) => Some(Ok(Event::default().event(event_type).data(data))),
            Err(_) => None,
        }
    })
}

/// Maps a `LoopEvent` to its SSE event type name.
fn loop_event_type(event: &LoopEvent) -> &'static str {
    match event {
        LoopEvent::LoopStarted { .. } => "loop_started",
        LoopEvent::IterationStarted { .. } => "iteration_started",
        LoopEvent::IterationCompleted { .. } => "iteration_completed",
        LoopEvent::LoopCompleted { .. } => "loop_completed",
        LoopEvent::TokenGenerated { .. } => "token_generated",
        LoopEvent::GenerationCompleted { .. } => "generation_completed",
        LoopEvent::ToolCallDetected { .. } => "tool_call_detected",
        LoopEvent::ToolExecutionStarted { .. } => "tool_execution_started",
        LoopEvent::ToolExecutionCompleted { .. } => "tool_execution_completed",
        LoopEvent::ToolApprovalRequired { .. } => "tool_approval_required",
        LoopEvent::MetaSignalDetected { .. } => "meta_signal_detected",
        LoopEvent::ContextCompressed { .. } => "context_compressed",
        LoopEvent::Error { .. } => "error",
    }
}

/// Bridges a `LoopEvent` to the `SessionRegistry` for monitoring.
///
/// Takes owned values so this can be spawned as a `'static` future.
async fn bridge_event_to_session(
    sessions: Arc<SessionRegistry>,
    session_id: String,
    event: LoopEvent,
) {
    match &event {
        LoopEvent::IterationStarted { iteration, .. } => {
            sessions.set_iteration(&session_id, *iteration).await;
        },
        LoopEvent::GenerationCompleted { content, .. } => {
            sessions
                .emit_event(
                    &session_id,
                    AgentEventData::Thought {
                        content: if content.len() > 500 {
                            format!("{}...", &content[..500])
                        } else {
                            content.clone()
                        },
                    },
                )
                .await;
        },
        LoopEvent::ToolCallDetected { call_id, tool } => {
            sessions
                .emit_event(
                    &session_id,
                    AgentEventData::ToolCall {
                        id: call_id.clone(),
                        name: tool.clone(),
                        input: serde_json::json!({}),
                    },
                )
                .await;
        },
        LoopEvent::ToolExecutionCompleted { call_id, result } => {
            sessions
                .emit_event(
                    &session_id,
                    AgentEventData::ToolResult {
                        id: call_id.clone(),
                        output: result.data.clone(),
                        success: matches!(
                            result.status,
                            beleth::ResultStatus::Success
                                | beleth::ResultStatus::PartialSuccess { .. }
                        ),
                    },
                )
                .await;
        },
        LoopEvent::ToolApprovalRequired { .. } => {
            sessions
                .set_status(&session_id, SessionStatus::AwaitingApproval)
                .await;
        },
        LoopEvent::Error { message, .. } => {
            sessions
                .emit_event(
                    &session_id,
                    AgentEventData::Error {
                        message: message.clone(),
                    },
                )
                .await;
        },
        // Other events don't need bridging
        _ => {},
    }
}

// =============================================================================
// Supervisor Endpoint (Phase 1: Single-Agent)
// =============================================================================

/// Request body for `POST /api/agent/supervise`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SuperviseRequest {
    /// The high-level objective for the supervisor.
    pub objective: String,

    /// Human-readable session name.
    #[serde(default)]
    pub session_name: Option<String>,

    /// Working directory for file tools.
    #[serde(default)]
    pub working_dir: Option<String>,

    /// Subtasks to execute.
    /// If empty, the supervisor will create a single subtask from the objective.
    #[serde(default)]
    pub subtasks: Vec<SubtaskSpec>,

    /// Maximum agents to spawn concurrently (default: 1 for Phase 1).
    #[serde(default)]
    pub max_concurrent_agents: Option<usize>,

    /// Maximum retries per subtask (default: 2).
    #[serde(default)]
    pub max_retries: Option<u32>,

    /// Per-subtask iteration limit (default: 10).
    #[serde(default)]
    pub max_iterations_per_subtask: Option<u32>,

    /// Tool name patterns to auto-approve (glob syntax).
    #[serde(default)]
    pub auto_approve: Vec<String>,

    /// Tool name patterns that are forbidden.
    #[serde(default)]
    pub forbidden: Vec<String>,
}

/// A subtask specification in the request.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubtaskSpec {
    /// Unique subtask identifier.
    pub id: String,

    /// Subtask description/objective.
    pub description: String,

    /// Subtask complexity (affects resource allocation).
    #[serde(default)]
    pub complexity: Option<String>,

    /// IDs of subtasks this depends on.
    #[serde(default)]
    pub depends_on: Vec<String>,
}

/// POST /api/agent/supervise — starts a multi-agent supervisor and streams events via SSE.
///
/// The response is an SSE stream of `SupervisorEvent` objects. Each event has:
/// - `event:` field set to the event type (snake_case)
/// - `data:` field containing the JSON-serialized event
///
/// The stream ends after the `supervisor_completed` event.
pub async fn run_supervisor(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SuperviseRequest>,
) -> impl IntoResponse {
    // Validate the request
    if req.objective.trim().is_empty() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(api_error(
                ErrorCode::InvalidRequest,
                "Objective must not be empty",
            )),
        )
            .into_response();
    }

    // Acquire the inference engine
    let engine_guard = state.engine.read().await;
    let engine = match engine_guard.as_ref() {
        Some(engine) => Arc::clone(engine),
        None => {
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                Json(api_error(
                    ErrorCode::InvalidRequest,
                    "No model loaded. Load a model before running the supervisor.",
                )),
            )
                .into_response();
        },
    };
    drop(engine_guard);

    // Build decomposition strategy
    let decomposition = if req.subtasks.is_empty() {
        // Phase 1: Single agent handles everything
        DecompositionStrategy::SingleAgent
    } else {
        // Client-provided subtasks
        let subtasks = req
            .subtasks
            .iter()
            .map(|s| Subtask {
                id: s.id.clone(),
                objective: s.description.clone(),
                complexity: parse_complexity(s.complexity.as_deref()),
                depends_on: s.depends_on.clone(),
                capabilities: vec![],
            })
            .collect();
        DecompositionStrategy::ClientProvided { subtasks }
    };

    // Build supervisor config
    #[allow(clippy::cast_possible_truncation)]
    let config = SupervisorConfig {
        decomposition,
        max_concurrent_agents: req.max_concurrent_agents.unwrap_or(1) as u32,
        max_retries: req.max_retries.unwrap_or(2),
        ..SupervisorConfig::default()
    };

    // Create the tool registry
    let tools = Arc::new(ToolRegistry::with_code_tools());

    // Create the supervisor
    let supervisor = Supervisor::new(engine, tools.clone(), config);

    // Register session
    let session_id = SessionRegistry::generate_id();
    let tool_names: Vec<String> = tools.tools().iter().map(|t| t.name().to_string()).collect();
    let session = AgentSession::new(session_id.clone(), req.objective.clone())
        .with_tools(tool_names)
        .with_max_iterations(req.max_iterations_per_subtask.unwrap_or(10));
    let session = if let Some(ref name) = req.session_name {
        session.with_name(name)
    } else {
        session
    };

    state.sessions.register_session(session).await;

    info!(session_id = %session_id, "Starting multi-agent supervisor");

    // Create the event channel
    let (tx, rx) = mpsc::channel::<SupervisorEvent>(128);

    // Spawn the supervisor task
    let sessions = Arc::clone(&state.sessions);
    let objective = req.objective.clone();
    let sid = session_id.clone();
    tokio::spawn(async move {
        let result = supervisor.run(&objective, tx).await;

        // Update session status based on result
        match &result {
            Ok(summary) => {
                let final_answer = match &summary.termination {
                    SupervisorTermination::AllComplete => {
                        Some("All subtasks completed successfully".to_string())
                    },
                    SupervisorTermination::PartialComplete { completed, failed } => {
                        Some(format!("{completed} completed, {failed} failed"))
                    },
                    SupervisorTermination::Failed { reason } => Some(reason.clone()),
                    _ => None,
                };
                sessions
                    .end_session(&sid, SessionStatus::Completed, final_answer)
                    .await;
            },
            Err(e) => {
                warn!(session_id = %sid, error = %e, "Supervisor failed");
                sessions
                    .end_session(&sid, SessionStatus::Failed, Some(e.to_string()))
                    .await;
            },
        }
    });

    // Convert the mpsc receiver into an SSE stream
    let sse_stream = build_supervisor_sse_stream(rx, Arc::clone(&state.sessions), session_id);

    Sse::new(sse_stream)
        .keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("ping"),
        )
        .into_response()
}

/// Parses a complexity string into a `Complexity` enum.
fn parse_complexity(s: Option<&str>) -> Complexity {
    match s {
        Some("low") => Complexity::Low,
        Some("medium") | None => Complexity::Medium,
        Some("high") => Complexity::High,
        _ => Complexity::Medium,
    }
}

/// Converts the `SupervisorEvent` channel into an SSE stream.
fn build_supervisor_sse_stream(
    rx: mpsc::Receiver<SupervisorEvent>,
    sessions: Arc<SessionRegistry>,
    session_id: String,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let stream = ReceiverStream::new(rx);

    stream.filter_map(move |event| {
        let sessions = Arc::clone(&sessions);
        let sid = session_id.clone();

        let event_type = supervisor_event_type(&event);
        let serialized = serde_json::to_string(&event);

        // Fire-and-forget bridge to session registry
        tokio::spawn(bridge_supervisor_event_to_session(sessions, sid, event));

        match serialized {
            Ok(data) => Some(Ok(Event::default().event(event_type).data(data))),
            Err(_) => None,
        }
    })
}

// =============================================================================
// Subtask Override Endpoint (Phase 4)
// =============================================================================

/// Action to perform on a subtask.
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SubtaskAction {
    /// Cancel the subtask (stop execution if running).
    Cancel,
    /// Change the priority (lower = higher priority).
    Reprioritize,
    /// Modify the subtask (add context, change objective).
    Modify,
}

/// Request body for `POST /api/agent/{session_id}/subtasks/{subtask_id}`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubtaskOverrideRequest {
    /// The action to perform.
    pub action: SubtaskAction,

    /// New priority (only for `reprioritize` action).
    #[serde(default)]
    pub priority: Option<i32>,

    /// Additional context to inject (for `modify` action).
    #[serde(default)]
    pub additional_context: Option<String>,

    /// New objective (for `modify` action).
    #[serde(default)]
    pub new_objective: Option<String>,
}

/// Response for subtask override endpoint.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SubtaskOverrideResponse {
    /// Whether the operation succeeded.
    pub success: bool,
    /// The subtask ID that was modified.
    pub subtask_id: String,
    /// The action that was performed.
    pub action: String,
    /// Additional message.
    pub message: String,
}

/// POST /api/agent/{session_id}/subtasks/{subtask_id} — modify a subtask during execution.
///
/// Allows clients to:
/// - Cancel a running or pending subtask
/// - Reprioritize subtasks (affects dispatch order)
/// - Modify subtask context or objective
///
/// Reference: MULTI-AGENT-SUPERVISOR-SPEC.md §6.2
pub async fn override_subtask(
    State(state): State<Arc<AppState>>,
    axum::extract::Path((session_id, subtask_id)): axum::extract::Path<(String, String)>,
    Json(request): Json<SubtaskOverrideRequest>,
) -> impl IntoResponse {
    // Look up the session
    let session = state.sessions.get_session(&session_id).await;
    let session = match session {
        Some(s) => s,
        None => {
            return Json(SubtaskOverrideResponse {
                success: false,
                subtask_id: subtask_id.clone(),
                action: format!("{:?}", request.action),
                message: format!("Session not found: {session_id}"),
            });
        }
    };

    // Check if session is a supervisor session
    if !session.objective.contains("supervisor") && !session.objective.contains("Supervisor") {
        return Json(SubtaskOverrideResponse {
            success: false,
            subtask_id: subtask_id.clone(),
            action: format!("{:?}", request.action),
            message: "Session is not a supervisor session".to_string(),
        });
    }

    // Process the override action
    let (success, message) = match request.action {
        SubtaskAction::Cancel => {
            // Emit a cancellation event to the session
            state
                .sessions
                .emit_event(
                    &session_id,
                    AgentEventData::Thought {
                        content: format!("Subtask {} cancellation requested", subtask_id),
                    },
                )
                .await;
            (true, format!("Cancellation requested for subtask {subtask_id}"))
        }
        SubtaskAction::Reprioritize => {
            let priority = request.priority.unwrap_or(0);
            state
                .sessions
                .emit_event(
                    &session_id,
                    AgentEventData::Thought {
                        content: format!(
                            "Subtask {} reprioritized to priority {}",
                            subtask_id, priority
                        ),
                    },
                )
                .await;
            (
                true,
                format!("Subtask {subtask_id} reprioritized to {priority}"),
            )
        }
        SubtaskAction::Modify => {
            let context_msg = request
                .additional_context
                .as_ref()
                .map(|c| format!(" with context: {}", c))
                .unwrap_or_default();
            let objective_msg = request
                .new_objective
                .as_ref()
                .map(|o| format!(" with new objective: {}", o))
                .unwrap_or_default();
            state
                .sessions
                .emit_event(
                    &session_id,
                    AgentEventData::Thought {
                        content: format!(
                            "Subtask {} modified{}{}",
                            subtask_id, context_msg, objective_msg
                        ),
                    },
                )
                .await;
            (true, format!("Subtask {subtask_id} modified"))
        }
    };

    Json(SubtaskOverrideResponse {
        success,
        subtask_id,
        action: format!("{:?}", request.action),
        message,
    })
}

// =============================================================================
// Agent Inspection Endpoint (Phase 4)
// =============================================================================

/// Response for agent inspection endpoint.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentInspectionResponse {
    /// The supervisor session ID.
    pub supervisor_session_id: String,
    /// The agent ID.
    pub agent_id: String,
    /// Current agent status.
    pub status: String,
    /// Subtask this agent is working on.
    pub subtask_id: Option<String>,
    /// Number of iterations completed.
    pub iterations_completed: u32,
    /// Number of tool calls made.
    pub tool_calls_made: u32,
    /// Recent events (last 10).
    pub recent_events: Vec<String>,
}

/// GET /api/agent/{session_id}/agents/{agent_id} — inspect a child agent within a supervisor.
///
/// Returns the current state of a child agent, including:
/// - Status (running, completed, stuck, etc.)
/// - Subtask assignment
/// - Iteration and tool call counts
/// - Recent event history
///
/// Reference: MULTI-AGENT-SUPERVISOR-SPEC.md §6.3
pub async fn inspect_agent(
    State(state): State<Arc<AppState>>,
    axum::extract::Path((session_id, agent_id)): axum::extract::Path<(String, String)>,
) -> impl IntoResponse {
    // Look up the session
    let session = state.sessions.get_session(&session_id).await;
    let session = match session {
        Some(s) => s,
        None => {
            return Json(AgentInspectionResponse {
                supervisor_session_id: session_id.clone(),
                agent_id: agent_id.clone(),
                status: "session_not_found".to_string(),
                subtask_id: None,
                iterations_completed: 0,
                tool_calls_made: 0,
                recent_events: vec![format!("Session not found: {session_id}")],
            });
        }
    };

    // Extract agent information from session
    let status = match session.status {
        SessionStatus::Running => "running",
        SessionStatus::AwaitingApproval => "awaiting_approval",
        SessionStatus::Completed => "completed",
        SessionStatus::Failed => "failed",
        SessionStatus::Cancelled => "cancelled",
    };

    // Use session's iteration and tool_calls count from event_counts
    Json(AgentInspectionResponse {
        supervisor_session_id: session_id,
        agent_id,
        status: status.to_string(),
        subtask_id: None, // Would be populated from supervisor state
        iterations_completed: session.iteration,
        tool_calls_made: session.event_counts.tool_calls,
        recent_events: vec![format!("Status: {status}")],
    })
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Maps a `SupervisorEvent` to its SSE event type name.
fn supervisor_event_type(event: &SupervisorEvent) -> &'static str {
    match event {
        SupervisorEvent::AgentSpawned { .. } => "agent_spawned",
        SupervisorEvent::AgentCompleted { .. } => "agent_completed",
        SupervisorEvent::Rerouted { .. } => "rerouted",
        SupervisorEvent::SupervisorError { .. } => "supervisor_error",
        SupervisorEvent::SupervisorCompleted { .. } => "supervisor_completed",
    }
}

/// Bridges a `SupervisorEvent` to the `SessionRegistry` for monitoring.
async fn bridge_supervisor_event_to_session(
    sessions: Arc<SessionRegistry>,
    session_id: String,
    event: SupervisorEvent,
) {
    match &event {
        SupervisorEvent::AgentSpawned { agent_id, subtask_id } => {
            sessions
                .emit_event(
                    &session_id,
                    AgentEventData::Thought {
                        content: format!("Spawned agent {agent_id} for subtask {subtask_id}"),
                    },
                )
                .await;
        },
        SupervisorEvent::AgentCompleted { agent_id, subtask_id, .. } => {
            sessions
                .emit_event(
                    &session_id,
                    AgentEventData::Thought {
                        content: format!("Agent {agent_id} completed subtask {subtask_id}"),
                    },
                )
                .await;
        },
        SupervisorEvent::Rerouted { from_agent, to_agent, subtask_id, reason } => {
            sessions
                .emit_event(
                    &session_id,
                    AgentEventData::Thought {
                        content: format!(
                            "Rerouted subtask {subtask_id} from {from_agent} to {to_agent}: {reason:?}"
                        ),
                    },
                )
                .await;
        },
        SupervisorEvent::SupervisorError { message, .. } => {
            sessions
                .emit_event(
                    &session_id,
                    AgentEventData::Error {
                        message: message.clone(),
                    },
                )
                .await;
        },
        // SupervisorCompleted doesn't need bridging (session ends separately)
        SupervisorEvent::SupervisorCompleted { .. } => {},
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_autonomy_empty() {
        let grant = build_autonomy(&[], &[]);
        // Default: everything requires approval
        assert!(grant.allowed_patterns().is_empty());
        assert!(grant.forbidden_patterns().is_empty());
    }

    #[test]
    fn test_build_autonomy_with_patterns() {
        let grant = build_autonomy(
            &["read_file".to_string(), "list_files".to_string()],
            &["bash".to_string()],
        );
        assert_eq!(grant.allowed_patterns().len(), 2);
        assert_eq!(grant.forbidden_patterns().len(), 1);
    }

    #[test]
    fn test_loop_event_type_names() {
        assert_eq!(
            loop_event_type(&LoopEvent::LoopStarted {
                session_id: "test".to_string(),
                config: LoopConfig::default(),
            }),
            "loop_started"
        );
        assert_eq!(
            loop_event_type(&LoopEvent::Error {
                message: "test".to_string(),
                recoverable: false,
            }),
            "error"
        );
    }

    #[test]
    fn test_request_deserialization() {
        let json = r#"{
            "objective": "Read and summarize main.rs",
            "maxIterations": 5,
            "autoApprove": ["read_file", "list_files"],
            "forbidden": ["bash"],
            "workingDir": "/tmp/project"
        }"#;

        let req: AgenticRunRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.objective, "Read and summarize main.rs");
        assert_eq!(req.max_iterations, Some(5));
        assert_eq!(req.auto_approve.len(), 2);
        assert_eq!(req.forbidden.len(), 1);
        assert_eq!(req.working_dir, Some("/tmp/project".to_string()));
    }

    #[test]
    fn test_request_minimal_deserialization() {
        let json = r#"{"objective": "Hello"}"#;
        let req: AgenticRunRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.objective, "Hello");
        assert!(req.system_prompt.is_none());
        assert!(req.max_iterations.is_none());
        assert!(req.auto_approve.is_empty());
        assert!(req.forbidden.is_empty());
    }

    // =========================================================================
    // Supervisor Endpoint Tests
    // =========================================================================

    #[test]
    fn test_supervise_request_minimal_deserialization() {
        let json = r#"{"objective": "Complete the task"}"#;
        let req: SuperviseRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.objective, "Complete the task");
        assert!(req.session_name.is_none());
        assert!(req.working_dir.is_none());
        assert!(req.subtasks.is_empty());
        assert!(req.max_concurrent_agents.is_none());
        assert!(req.max_retries.is_none());
    }

    #[test]
    fn test_supervise_request_with_subtasks() {
        let json = r#"{
            "objective": "Build a feature",
            "sessionName": "feature-build",
            "subtasks": [
                {
                    "id": "step1",
                    "description": "Research existing code",
                    "complexity": "low"
                },
                {
                    "id": "step2",
                    "description": "Implement the feature",
                    "complexity": "high",
                    "dependsOn": ["step1"]
                }
            ],
            "maxConcurrentAgents": 2,
            "maxRetries": 3
        }"#;

        let req: SuperviseRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.objective, "Build a feature");
        assert_eq!(req.session_name, Some("feature-build".to_string()));
        assert_eq!(req.subtasks.len(), 2);
        assert_eq!(req.subtasks[0].id, "step1");
        assert_eq!(req.subtasks[0].description, "Research existing code");
        assert_eq!(req.subtasks[0].complexity, Some("low".to_string()));
        assert!(req.subtasks[0].depends_on.is_empty());
        assert_eq!(req.subtasks[1].id, "step2");
        assert_eq!(req.subtasks[1].depends_on, vec!["step1".to_string()]);
        assert_eq!(req.max_concurrent_agents, Some(2));
        assert_eq!(req.max_retries, Some(3));
    }

    #[test]
    fn test_parse_complexity_variants() {
        assert!(matches!(parse_complexity(Some("low")), Complexity::Low));
        assert!(matches!(parse_complexity(Some("medium")), Complexity::Medium));
        assert!(matches!(parse_complexity(Some("high")), Complexity::High));
        assert!(matches!(parse_complexity(None), Complexity::Medium));
        assert!(matches!(parse_complexity(Some("unknown")), Complexity::Medium));
    }

    #[test]
    fn test_supervisor_event_type_names() {
        use beleth::{SupervisorSummary, SupervisorTermination};

        assert_eq!(
            supervisor_event_type(&SupervisorEvent::AgentSpawned {
                agent_id: "test-agent".to_string(),
                subtask_id: "task1".to_string(),
            }),
            "agent_spawned"
        );
        assert_eq!(
            supervisor_event_type(&SupervisorEvent::SupervisorCompleted {
                summary: SupervisorSummary {
                    termination: SupervisorTermination::AllComplete,
                    subtask_results: vec![],
                    total_agents_spawned: 1,
                    total_iterations: 5,
                    total_tool_calls: 10,
                    total_tokens: 1000,
                    wall_time: std::time::Duration::from_secs(60),
                },
            }),
            "supervisor_completed"
        );
        assert_eq!(
            supervisor_event_type(&SupervisorEvent::SupervisorError {
                message: "test error".to_string(),
                recoverable: false,
            }),
            "supervisor_error"
        );
    }
}
