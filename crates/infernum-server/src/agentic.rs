//! Agentic loop HTTP endpoint.
//!
//! `POST /api/agent/run` — starts an agentic loop, streaming `LoopEvent`s as SSE.
//! Integrates with the `SessionRegistry` for monitoring and the inference engine
//! for model generation.
//!
//! Reference: AGENTIC-LOOP-SPEC.md §9.

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
    AutonomyGrant, ExecutorConfig, LoopConfig, LoopEvent, LoopExecutor, NaturalTermination,
    TerminationReason, ToolPattern, ToolRegistry,
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
}
