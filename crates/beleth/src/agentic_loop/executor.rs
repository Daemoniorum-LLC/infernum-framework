//! Async loop executor — drives the `AgenticLoop` state machine to completion.
//!
//! Orchestrates: Generate → Detect → Execute → Integrate → Continue
//!
//! The executor accepts an `InferenceEngine` for generation, a `ToolRegistry`
//! for tool execution, and an `AutonomyGrant` for permission enforcement.
//! It streams `LoopEvent`s through a channel for real-time observability.
//!
//! Reference: AGENTIC-LOOP-SPEC.md §2.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use infernum_core::{GenerateRequest, Message, SamplingParams};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::tool::{ToolCall as BelethToolCall, ToolContext, ToolRegistry};

use super::approval::{ApprovalDecision, ApprovalGate};
use super::meta_signal::detect_meta_signal;
use super::types::*;
use super::AgenticLoop;

// ---------------------------------------------------------------------------
// Tool call detection
// ---------------------------------------------------------------------------

/// A tool call detected in model output.
#[derive(Debug, Clone)]
pub struct DetectedCall {
    /// Unique identifier for this call.
    pub id: String,
    /// Name of the tool being called.
    pub name: String,
    /// Parsed arguments.
    pub arguments: serde_json::Value,
}

/// Trait for detecting tool calls in model output.
///
/// The default implementation handles Qwen-style `<tool_call>` tags.
/// Servers may provide a model-family-aware implementation.
#[async_trait]
pub trait ToolCallDetector: Send + Sync {
    /// Detect tool calls in the given model output text.
    fn detect(&self, output: &str) -> Vec<DetectedCall>;
}

/// Default detector that parses `<tool_call>{"name":..., "arguments":...}</tool_call>` tags.
#[derive(Debug, Clone, Default)]
pub struct QwenToolCallDetector;

impl QwenToolCallDetector {
    /// Creates a new Qwen-format tool call detector.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ToolCallDetector for QwenToolCallDetector {
    fn detect(&self, output: &str) -> Vec<DetectedCall> {
        parse_tool_call_tags(output)
    }
}

// ---------------------------------------------------------------------------
// Executor configuration
// ---------------------------------------------------------------------------

/// Configuration for the loop executor.
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Loop state machine configuration.
    pub loop_config: LoopConfig,
    /// Autonomy grant for permission checking.
    pub autonomy: AutonomyGrant,
    /// System prompt prepended to the context.
    pub system_prompt: Option<String>,
    /// Sampling parameters for generation.
    pub sampling: SamplingParams,
    /// Session identifier for event correlation.
    pub session_id: String,
    /// Working directory for file tools.
    pub working_dir: Option<PathBuf>,
}

impl ExecutorConfig {
    /// Creates a new executor config with the given session ID.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            loop_config: LoopConfig::default(),
            autonomy: AutonomyGrant::default(),
            system_prompt: None,
            sampling: SamplingParams::default().with_max_tokens(2048),
            session_id: session_id.into(),
            working_dir: None,
        }
    }

    /// Sets the loop configuration.
    pub fn with_loop_config(mut self, config: LoopConfig) -> Self {
        self.loop_config = config;
        self
    }

    /// Sets the autonomy grant.
    pub fn with_autonomy(mut self, autonomy: AutonomyGrant) -> Self {
        self.autonomy = autonomy;
        self
    }

    /// Sets the system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Sets the sampling parameters.
    pub fn with_sampling(mut self, sampling: SamplingParams) -> Self {
        self.sampling = sampling;
        self
    }

    /// Sets the working directory for file tools.
    pub fn with_working_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Loop executor
// ---------------------------------------------------------------------------

/// Errors from loop execution.
#[derive(Debug, thiserror::Error)]
pub enum LoopError {
    /// Error from the inference engine.
    #[error("engine error: {0}")]
    EngineError(String),
    /// Invalid state transition.
    #[error("transition error: {0}")]
    TransitionError(#[from] TransitionError),
    /// Tool execution error.
    #[error("tool error: {0}")]
    ToolError(String),
}

/// The async loop executor.
///
/// Drives the `AgenticLoop` state machine by generating model output,
/// detecting tool calls and meta-signals, executing tools through the
/// registry with autonomy enforcement, and integrating results back
/// into the conversation context.
///
/// When an [`ApprovalGate`] is attached, tools that require approval
/// will block until a decision is delivered (or the timeout expires),
/// rather than failing immediately. See AGENTIC-LOOP-SPEC §9.4.
pub struct LoopExecutor {
    engine: Arc<dyn abaddon::InferenceEngine>,
    tools: Arc<ToolRegistry>,
    detector: Arc<dyn ToolCallDetector>,
    config: ExecutorConfig,
    approval_gate: Option<Arc<ApprovalGate>>,
}

impl LoopExecutor {
    /// Creates a new executor.
    pub fn new(
        engine: Arc<dyn abaddon::InferenceEngine>,
        tools: Arc<ToolRegistry>,
        config: ExecutorConfig,
    ) -> Self {
        Self {
            engine,
            tools,
            detector: Arc::new(QwenToolCallDetector::new()),
            config,
            approval_gate: None,
        }
    }

    /// Sets a custom tool call detector.
    pub fn with_detector(mut self, detector: Arc<dyn ToolCallDetector>) -> Self {
        self.detector = detector;
        self
    }

    /// Attaches an approval gate for interactive tool approval.
    ///
    /// When set, tools requiring approval will block on the gate instead of
    /// failing immediately. Decisions are delivered through the gate by
    /// external systems (HTTP endpoints, CLI prompts).
    pub fn with_approval_gate(mut self, gate: Arc<ApprovalGate>) -> Self {
        self.approval_gate = Some(gate);
        self
    }

    /// Returns a reference to the approval gate, if attached.
    pub fn approval_gate(&self) -> Option<&Arc<ApprovalGate>> {
        self.approval_gate.as_ref()
    }

    /// Runs the agentic loop to completion.
    ///
    /// Streams `LoopEvent`s through `event_tx` for real-time observability.
    /// Returns the final `LoopSummary` when the loop terminates.
    pub async fn run(
        &self,
        objective: &str,
        event_tx: mpsc::Sender<LoopEvent>,
    ) -> Result<LoopSummary, LoopError> {
        let mut state_machine = AgenticLoop::new(self.config.loop_config.clone());
        let mut messages = self.build_initial_messages(objective);
        let detection_config = DetectionConfig {
            detect_implicit: self.config.loop_config.detect_implicit_signals,
            ..Default::default()
        };
        let tool_ctx = self.build_tool_context();

        // Initialized → Generating
        state_machine.start().map_err(|e| {
            warn!("Failed to start loop: {e}");
            e
        })?;

        let _ = event_tx
            .send(LoopEvent::LoopStarted {
                session_id: self.config.session_id.clone(),
                config: self.config.loop_config.clone(),
            })
            .await;

        loop {
            let iteration = state_machine.iteration();
            info!(iteration, "Agentic loop iteration starting");

            let _ = event_tx
                .send(LoopEvent::IterationStarted {
                    iteration,
                    status: state_machine.status(),
                })
                .await;

            // =================================================================
            // GENERATE
            // =================================================================
            let request = self.build_generate_request(&messages);
            let response = self.engine.generate(request).await.map_err(|e| {
                warn!("Engine generation failed: {e}");
                LoopError::EngineError(e.to_string())
            })?;

            let output = response
                .choices
                .first()
                .map(|c| c.text.clone())
                .unwrap_or_default();
            let tokens = response.usage.completion_tokens;

            debug!(tokens, output_len = output.len(), "Generation complete");

            // Generating → Detecting (may terminate on token budget)
            if let Err(e) = state_machine.generation_complete(&output, tokens) {
                let _ = event_tx
                    .send(LoopEvent::IterationCompleted {
                        iteration,
                        outcome: IterationOutcome::ResourceLimitReached,
                    })
                    .await;
                // If resource limit, break gracefully; otherwise propagate
                if matches!(e, TransitionError::ResourceLimitReached(_)) {
                    break;
                }
                return Err(e.into());
            }

            let _ = event_tx
                .send(LoopEvent::GenerationCompleted {
                    content: output.clone(),
                    tokens,
                })
                .await;

            // =================================================================
            // DETECT
            // =================================================================

            // Check for meta-signals first
            let mut terminal_signal = false;
            if let Some(signal) = detect_meta_signal(&output, &detection_config) {
                debug!(?signal, "Meta-signal detected");
                let _ = event_tx
                    .send(LoopEvent::MetaSignalDetected {
                        signal: signal.clone(),
                    })
                    .await;

                match signal {
                    MetaSignal::Answer {
                        content,
                        confidence,
                        caveats,
                    } => {
                        state_machine.answer_detected(content, confidence, caveats)?;
                        let _ = event_tx
                            .send(LoopEvent::IterationCompleted {
                                iteration,
                                outcome: IterationOutcome::AnswerProvided,
                            })
                            .await;
                        terminal_signal = true;
                    }
                    MetaSignal::Stuck {
                        attempts,
                        request,
                        ..
                    } => {
                        state_machine.stuck_detected(attempts, request)?;
                        let _ = event_tx
                            .send(LoopEvent::IterationCompleted {
                                iteration,
                                outcome: IterationOutcome::Stuck,
                            })
                            .await;
                        terminal_signal = true;
                    }
                    MetaSignal::Yield {
                        partial_progress,
                        suggested_expertise,
                    } => {
                        let reason = suggested_expertise
                            .first()
                            .cloned()
                            .unwrap_or_else(|| "Agent yielded".to_string());
                        state_machine.yield_detected(partial_progress, reason)?;
                        let _ = event_tx
                            .send(LoopEvent::IterationCompleted {
                                iteration,
                                outcome: IterationOutcome::Yielded,
                            })
                            .await;
                        terminal_signal = true;
                    }
                    MetaSignal::Uncertain { .. } | MetaSignal::Thinking { .. } => {
                        // Non-terminal: continue to tool call detection
                    }
                }
            }

            if terminal_signal {
                break;
            }

            // Check for tool calls
            let detected_calls = self.detector.detect(&output);

            if detected_calls.is_empty() {
                // No tool calls and no terminal signal → treat as implicit answer
                debug!("No tool calls or signals — treating as implicit answer");
                state_machine.answer_detected(output.clone(), 0.5, vec![])?;
                let _ = event_tx
                    .send(LoopEvent::IterationCompleted {
                        iteration,
                        outcome: IterationOutcome::AnswerProvided,
                    })
                    .await;
                break;
            }

            // =================================================================
            // EXECUTE
            // =================================================================

            // Detecting → Executing (may terminate on tool call limit)
            if let Err(e) = state_machine.tool_calls_detected(detected_calls.len() as u32) {
                let _ = event_tx
                    .send(LoopEvent::IterationCompleted {
                        iteration,
                        outcome: IterationOutcome::ResourceLimitReached,
                    })
                    .await;
                if matches!(e, TransitionError::ResourceLimitReached(_)) {
                    break;
                }
                return Err(e.into());
            }

            // Add assistant message to context
            messages.push(Message::assistant(&output));

            let mut agentic_results = Vec::new();

            for call in &detected_calls {
                let _ = event_tx
                    .send(LoopEvent::ToolCallDetected {
                        call_id: call.id.clone(),
                        tool: call.name.clone(),
                    })
                    .await;

                let result =
                    self.execute_single_tool(call, &tool_ctx, &event_tx).await;

                // Add tool result to conversation context
                let content = match &result.status {
                    ResultStatus::Success | ResultStatus::PartialSuccess { .. } => {
                        serde_json::to_string(&result.data).unwrap_or_default()
                    }
                    ResultStatus::Empty => "No results found.".to_string(),
                    ResultStatus::Failed { .. } => {
                        format!(
                            "Error: {}",
                            result
                                .data
                                .get("error")
                                .and_then(|e| e.as_str())
                                .unwrap_or("Tool execution failed")
                        )
                    }
                };
                messages.push(Message::tool_result(&call.id, &content));

                agentic_results.push(result);
            }

            // =================================================================
            // INTEGRATE
            // =================================================================

            // Executing → Integrating
            state_machine.execution_complete(agentic_results)?;

            let _ = event_tx
                .send(LoopEvent::IterationCompleted {
                    iteration,
                    outcome: IterationOutcome::ToolCallsExecuted {
                        count: detected_calls.len() as u32,
                    },
                })
                .await;

            // Integrating → Generating (or resource limit)
            match state_machine.continue_loop() {
                Ok(()) => {}
                Err(TransitionError::ResourceLimitReached(_)) => {
                    debug!("Resource limit reached, terminating loop");
                    break;
                }
                Err(e) => return Err(e.into()),
            }
        }

        let summary = state_machine.summary();
        info!(
            iterations = summary.iterations_completed,
            tool_calls = summary.tool_calls_made,
            "Agentic loop completed"
        );

        let _ = event_tx
            .send(LoopEvent::LoopCompleted {
                summary: summary.clone(),
            })
            .await;

        Ok(summary)
    }

    /// Computes the effective permission for a tool call.
    ///
    /// Checks runtime overrides from `ApproveAlways` decisions before falling
    /// back to the static `AutonomyGrant`. Forbidden always takes priority.
    fn effective_permission(&self, tool_name: &str, argument: Option<&str>) -> Permission {
        let static_perm = self.config.autonomy.check(tool_name, argument);

        // Forbidden cannot be overridden by runtime approvals
        if static_perm == Permission::Forbidden {
            return Permission::Forbidden;
        }

        // Runtime overrides from ApproveAlways only apply to RequiresApproval
        if static_perm == Permission::RequiresApproval {
            if let Some(gate) = &self.approval_gate {
                if gate.is_runtime_approved(tool_name) {
                    return Permission::Allowed;
                }
            }
        }

        static_perm
    }

    /// Execute a single tool call with autonomy checking and approval blocking.
    ///
    /// When an [`ApprovalGate`] is attached and the tool requires approval,
    /// the executor registers a pending request, emits a `ToolApprovalRequired`
    /// event, and blocks until a decision arrives or the configured timeout
    /// expires. Without a gate, the tool call fails immediately with a
    /// recoverable error (backward-compatible behavior).
    async fn execute_single_tool(
        &self,
        call: &DetectedCall,
        tool_ctx: &ToolContext,
        event_tx: &mpsc::Sender<LoopEvent>,
    ) -> AgenticToolResult {
        let argument_str = serde_json::to_string(&call.arguments).ok();
        let permission = self.effective_permission(&call.name, argument_str.as_deref());

        match permission {
            Permission::Forbidden => {
                warn!(tool = %call.name, "Tool call forbidden by autonomy grant");
                AgenticToolResult {
                    call_id: call.id.clone(),
                    tool_name: call.name.clone(),
                    status: ResultStatus::Failed { recoverable: false },
                    data: serde_json::json!({"error": "Tool call forbidden by autonomy grant"}),
                    confidence: Confidence::Measured,
                    latency_ms: 0,
                    truncated: false,
                }
            }
            Permission::RequiresApproval => {
                debug!(tool = %call.name, "Tool requires approval");
                let timeout = self.config.loop_config.approval_timeout;

                if let Some(gate) = &self.approval_gate {
                    // Register the pending request BEFORE emitting the event,
                    // so the entry exists by the time a client tries to deliver.
                    let rx = gate.request(
                        &call.id,
                        &call.name,
                        call.arguments.clone(),
                    );

                    let _ = event_tx
                        .send(LoopEvent::ToolApprovalRequired {
                            call_id: call.id.clone(),
                            tool: call.name.clone(),
                            arguments: call.arguments.clone(),
                            timeout_secs: timeout.as_secs(),
                            pending_count: gate.pending_count(),
                        })
                        .await;

                    // Block until decision or timeout
                    match tokio::time::timeout(timeout, rx).await {
                        Ok(Ok(
                            ApprovalDecision::Approve
                            | ApprovalDecision::ApproveAlways { .. },
                        )) => {
                            // Approved — execute the tool
                            self.execute_approved_tool(call, tool_ctx, event_tx).await
                        }
                        Ok(Ok(ApprovalDecision::Deny)) => {
                            debug!(tool = %call.name, "Tool call denied by operator");
                            AgenticToolResult {
                                call_id: call.id.clone(),
                                tool_name: call.name.clone(),
                                status: ResultStatus::Failed { recoverable: true },
                                data: serde_json::json!({"error": "Tool call denied by operator"}),
                                confidence: Confidence::Measured,
                                latency_ms: 0,
                                truncated: false,
                            }
                        }
                        Ok(Err(_)) | Err(_) => {
                            // Oneshot dropped or timeout expired
                            let secs = timeout.as_secs_f64();
                            let msg = format!(
                                "Tool approval timed out after {secs:.1}s — tool call skipped."
                            );
                            warn!(tool = %call.name, "{msg}");
                            AgenticToolResult {
                                call_id: call.id.clone(),
                                tool_name: call.name.clone(),
                                status: ResultStatus::Failed { recoverable: true },
                                data: serde_json::json!({"error": msg}),
                                confidence: Confidence::Measured,
                                latency_ms: 0,
                                truncated: false,
                            }
                        }
                    }
                } else {
                    // No approval gate — emit event and fail immediately
                    let _ = event_tx
                        .send(LoopEvent::ToolApprovalRequired {
                            call_id: call.id.clone(),
                            tool: call.name.clone(),
                            arguments: call.arguments.clone(),
                            timeout_secs: timeout.as_secs(),
                            pending_count: 0,
                        })
                        .await;
                    AgenticToolResult {
                        call_id: call.id.clone(),
                        tool_name: call.name.clone(),
                        status: ResultStatus::Failed { recoverable: true },
                        data: serde_json::json!({"error": "Tool requires approval (not granted)"}),
                        confidence: Confidence::Measured,
                        latency_ms: 0,
                        truncated: false,
                    }
                }
            }
            Permission::Allowed => {
                self.execute_approved_tool(call, tool_ctx, event_tx).await
            }
        }
    }

    /// Execute a tool that has already been approved (statically or via approval protocol).
    async fn execute_approved_tool(
        &self,
        call: &DetectedCall,
        tool_ctx: &ToolContext,
        event_tx: &mpsc::Sender<LoopEvent>,
    ) -> AgenticToolResult {
        let _ = event_tx
            .send(LoopEvent::ToolExecutionStarted {
                call_id: call.id.clone(),
                tool: call.name.clone(),
            })
            .await;

        let start = Instant::now();
        let beleth_call = BelethToolCall {
            name: call.name.clone(),
            params: call.arguments.clone(),
        };

        let tool_result = self.tools.execute(&beleth_call, tool_ctx).await;
        let latency_ms = start.elapsed().as_millis() as u64;

        let agentic_result = match tool_result {
            Ok(result) => {
                let status = if result.success {
                    ResultStatus::Success
                } else {
                    ResultStatus::Failed { recoverable: true }
                };
                let data = result.data.unwrap_or_else(|| {
                    if result.success {
                        serde_json::json!({"output": result.output})
                    } else {
                        serde_json::json!({"error": result.error.as_deref().unwrap_or("unknown error")})
                    }
                });
                AgenticToolResult {
                    call_id: call.id.clone(),
                    tool_name: call.name.clone(),
                    status,
                    data,
                    confidence: Confidence::Measured,
                    latency_ms,
                    truncated: false,
                }
            }
            Err(e) => {
                warn!(tool = %call.name, error = %e, "Tool execution failed");
                AgenticToolResult {
                    call_id: call.id.clone(),
                    tool_name: call.name.clone(),
                    status: ResultStatus::Failed { recoverable: true },
                    data: serde_json::json!({"error": e.to_string()}),
                    confidence: Confidence::Unknown,
                    latency_ms,
                    truncated: false,
                }
            }
        };

        let _ = event_tx
            .send(LoopEvent::ToolExecutionCompleted {
                call_id: call.id.clone(),
                result: agentic_result.clone(),
            })
            .await;

        agentic_result
    }

    fn build_initial_messages(&self, objective: &str) -> Vec<Message> {
        let system = self
            .config
            .system_prompt
            .as_deref()
            .unwrap_or("You are a helpful assistant.");
        let tool_desc = self.tools.to_qwen_native_description();
        let system_prompt = format!(
            "{system}\n\n{tool_desc}\n\n\
             You may express uncertainty with <uncertain>...</uncertain>, \
             signal you're stuck with <stuck>...</stuck>, \
             yield with <yield>...</yield>, \
             or provide a final answer with <answer confidence=\"0.9\">...</answer>."
        );

        vec![
            Message::system(system_prompt),
            Message::user(objective),
        ]
    }

    fn build_generate_request(&self, messages: &[Message]) -> GenerateRequest {
        GenerateRequest::chat(messages.to_vec()).with_sampling(self.config.sampling.clone())
    }

    fn build_tool_context(&self) -> ToolContext {
        let mut ctx = ToolContext::new(&self.config.session_id);
        if let Some(ref wd) = self.config.working_dir {
            ctx.state.insert(
                "working_dir".to_string(),
                serde_json::json!(wd.to_string_lossy().to_string()),
            );
        }
        ctx
    }
}

// ---------------------------------------------------------------------------
// Tool call parsing
// ---------------------------------------------------------------------------

/// Parse `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` tags.
fn parse_tool_call_tags(output: &str) -> Vec<DetectedCall> {
    let mut calls = Vec::new();
    let mut search_from = 0;
    let start_tag = "<tool_call>";
    let end_tag = "</tool_call>";

    while let Some(start) = output[search_from..].find(start_tag) {
        let abs_start = search_from + start + start_tag.len();
        if let Some(end) = output[abs_start..].find(end_tag) {
            let json_str = output[abs_start..abs_start + end].trim();
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
                let name = parsed
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string();
                let arguments = parsed
                    .get("arguments")
                    .cloned()
                    .unwrap_or(serde_json::json!({}));

                if !name.is_empty() {
                    let id = format!("call_{}", uuid::Uuid::new_v4().simple());
                    calls.push(DetectedCall {
                        id,
                        name,
                        arguments,
                    });
                }
            }
            search_from = abs_start + end + end_tag.len();
        } else {
            break;
        }
    }

    calls
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_tool_call() {
        let output = r#"Let me read that file.
<tool_call>
{"name": "read_file", "arguments": {"path": "src/main.rs"}}
</tool_call>"#;

        let calls = parse_tool_call_tags(output);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments["path"], "src/main.rs");
        assert!(calls[0].id.starts_with("call_"));
    }

    #[test]
    fn test_parse_multiple_tool_calls() {
        let output = r#"I'll read both files.
<tool_call>
{"name": "read_file", "arguments": {"path": "a.rs"}}
</tool_call>
<tool_call>
{"name": "read_file", "arguments": {"path": "b.rs"}}
</tool_call>"#;

        let calls = parse_tool_call_tags(output);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments["path"], "a.rs");
        assert_eq!(calls[1].arguments["path"], "b.rs");
    }

    #[test]
    fn test_parse_no_tool_calls() {
        let output = "Just a regular response with no tool calls.";
        let calls = parse_tool_call_tags(output);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_parse_malformed_json_ignored() {
        let output = "<tool_call>not valid json</tool_call>";
        let calls = parse_tool_call_tags(output);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_parse_empty_name_ignored() {
        let output = r#"<tool_call>{"name": "", "arguments": {}}</tool_call>"#;
        let calls = parse_tool_call_tags(output);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_executor_config_builder() {
        let config = ExecutorConfig::new("test-session")
            .with_system_prompt("You are a coder.")
            .with_loop_config(LoopConfig {
                max_iterations: 5,
                ..LoopConfig::default()
            })
            .with_autonomy(AutonomyGrant::default());

        assert_eq!(config.session_id, "test-session");
        assert_eq!(config.loop_config.max_iterations, 5);
        assert!(config.system_prompt.is_some());
    }

    #[test]
    fn test_qwen_detector_default() {
        let detector = QwenToolCallDetector::new();
        let output = r#"<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>"#;
        let calls = detector.detect(output);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
    }
}
