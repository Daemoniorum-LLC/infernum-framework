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
use infernum_core::sampling::GrammarConstraint;
use infernum_core::{GenerateRequest, Message, Role, SamplingParams};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::grammar::{compose_system_prompt, tool_call_grammar, FINAL_ANSWER_TOOL};
use crate::tool::{ToolCall as BelethToolCall, ToolContext, ToolRegistry};

use super::approval::{ApprovalDecision, ApprovalGate};
use super::continuation::{build_resumed_messages, create_continuation_state, ContinuationStore};
use super::meta_signal::detect_meta_signal;
use super::types::ContextMessage;
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
///
/// Falls back to a wrapper-agnostic scan when no `<tool_call>` tag is present
/// — see `parse_tool_call_tags` (private, this module) for why. Measured against a real
/// Qwen2.5-Coder-14B-Instruct model: the model reliably reasons to the
/// correct tool and arguments but unreliably picks `<tool_call>` over one of
/// the other tags this loop's own system prompt also offers
/// (`<answer>`, `<yield>`, `<stuck>`). See infernum-framework#70.
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

/// Confidence recorded for a `final_answer` call that omits the argument.
///
/// Matches the default the retired `<answer>` tag's parser used, so a run with
/// the grammar off and one with it on report comparable confidences.
const DEFAULT_ANSWER_CONFIDENCE: f32 = 0.8;

/// A terminal answer recovered from a `final_answer` call.
struct FinalAnswer {
    /// Id of the call it came from, so the loop can still report it detected.
    call_id: String,
    content: String,
    confidence: f32,
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

    /// Constrain generation to the tool-call envelope with a GBNF grammar.
    ///
    /// Defaults to `true`. When on, every tool-eligible turn carries the
    /// grammar from [`tool_call_grammar`]
    /// and the system prompt advertises only the envelopes that grammar
    /// permits.
    ///
    /// Turn it off for a backend with no grammar facility — the engine
    /// refuses a grammar it cannot honour rather than dropping it, so leaving
    /// this on against such a server produces a hard error, by design. See
    /// infernum-framework#17.
    pub tool_call_grammar: bool,
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
            tool_call_grammar: true,
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

    /// Enables or disables the tool-call grammar constraint.
    ///
    /// See [`tool_call_grammar`] for what
    /// the constraint costs as well as what it buys.
    pub fn with_tool_call_grammar(mut self, enabled: bool) -> Self {
        self.tool_call_grammar = enabled;
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
    /// Continuation could not be stored, loaded, or resumed.
    #[error("continuation error: {0}")]
    Continuation(String),
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
    /// Store for continuation state. Without one, a resumable termination
    /// still reports `can_resume` but yields no token — see
    /// `with_continuation_store`.
    continuation_store: Option<Arc<dyn ContinuationStore>>,
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
            continuation_store: None,
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

    /// Attaches a store so resumable terminations produce a usable token.
    ///
    /// Without one, [`LoopSummary::can_resume`] still reports whether the
    /// termination reason is *theoretically* resumable while
    /// `continuation_token` stays `None` — true but useless, and readers
    /// reliably hear "you can resume this". With a store attached the pair is
    /// coherent: a token is present exactly when resuming will work.
    #[must_use]
    pub fn with_continuation_store(mut self, store: Arc<dyn ContinuationStore>) -> Self {
        self.continuation_store = Some(store);
        self
    }

    /// Runs the agentic loop to completion as a fresh, single-turn session.
    ///
    /// Streams `LoopEvent`s through `event_tx` for real-time observability.
    /// Returns the final `LoopSummary` when the loop terminates.
    ///
    /// The conversation is discarded. For an interactive session that takes
    /// follow-up turns, use [`run_turn`](Self::run_turn), which threads the
    /// message history through.
    pub async fn run(
        &self,
        objective: &str,
        event_tx: mpsc::Sender<LoopEvent>,
    ) -> Result<LoopSummary, LoopError> {
        self.run_turn(objective, Vec::new(), event_tx)
            .await
            .map(|(summary, _)| summary)
    }

    /// Runs one turn of a multi-turn agentic session.
    ///
    /// `history` is the conversation so far — pass the `Vec<Message>` returned
    /// by the previous call, or an empty vector to start fresh. Returns the
    /// summary along with the full history including this turn, so a caller
    /// can feed it back for the next one.
    ///
    /// # Why this exists
    ///
    /// [`run`](Self::run) rebuilds the message list from scratch on every
    /// call, so invoking it twice produces two independent sessions with no
    /// memory of each other. That makes a persistent interactive agent
    /// impossible to build on it: every turn would be an amnesiac one-shot.
    ///
    /// Note this is *not* the same as resuming a terminated loop from a
    /// [`ContinuationState`](super::continuation::ContinuationState). This
    /// threads a live conversation; continuation restores a stored one. The
    /// continuation path still has no driver.
    pub async fn run_turn(
        &self,
        objective: &str,
        history: Vec<Message>,
        event_tx: mpsc::Sender<LoopEvent>,
    ) -> Result<(LoopSummary, Vec<Message>), LoopError> {
        let messages = if history.is_empty() {
            self.build_initial_messages(objective)
        } else {
            // Continuing: keep the established system prompt and prior turns,
            // and append the new instruction as a user message.
            let mut m = history;
            m.push(Message::user(objective));
            m
        };
        self.execute(messages, event_tx).await
    }

    /// Resumes a loop that terminated with a resumable reason.
    ///
    /// Loads the [`ContinuationState`](super::continuation::ContinuationState)
    /// stored under `token`, rebuilds the conversation as it stood at
    /// termination — including the tool results already collected — and
    /// continues from there. `additional_context` is
    /// appended as a user message, which is how a client answers a `Stuck`
    /// signal or redirects a `Yielded` one.
    ///
    /// This is *restoring a stored conversation*, which is not the same as
    /// [`run_turn`](Self::run_turn) threading a live one.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::Continuation`] if no store is attached, or if the
    /// token is unknown or expired.
    pub async fn resume(
        &self,
        token: &str,
        additional_context: Option<&str>,
        event_tx: mpsc::Sender<LoopEvent>,
    ) -> Result<LoopSummary, LoopError> {
        let store = self.continuation_store.as_ref().ok_or_else(|| {
            LoopError::Continuation(
                "no continuation store attached; call with_continuation_store()".to_string(),
            )
        })?;

        let state = store
            .load(token)
            .await
            .map_err(|e| LoopError::Continuation(format!("loading {token}: {e}")))?
            .ok_or_else(|| {
                LoopError::Continuation(format!("continuation not found or expired: {token}"))
            })?;

        let messages: Vec<Message> = build_resumed_messages(&state, additional_context)
            .iter()
            .map(context_message_to_message)
            .collect();

        // The token is single-use: a resumed loop stores a fresh one if it
        // also terminates resumably, so leaving the old one would let a client
        // silently rewind to a stale point.
        let _ = store.remove(token).await;

        self.execute(messages, event_tx).await.map(|(s, _)| s)
    }
}

/// Converts a live [`Message`] into the serializable [`ContextMessage`] form
/// used by stored continuations.
fn message_to_context_message(m: &Message) -> ContextMessage {
    ContextMessage {
        role: match m.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
        .to_string(),
        content: m.content.clone(),
        tool_call_id: m.tool_call_id.clone(),
    }
}

/// Inverse of [`message_to_context_message`].
///
/// An unrecognised role becomes `User` rather than failing the resume: a
/// stored conversation is not worth discarding over one odd role string, and
/// `User` is the least surprising place for unattributed content to land.
fn context_message_to_message(m: &ContextMessage) -> Message {
    let role = match m.role.as_str() {
        "system" => Role::System,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        _ => Role::User,
    };
    Message {
        role,
        content: m.content.clone(),
        name: None,
        tool_calls: None,
        tool_call_id: m.tool_call_id.clone(),
    }
}

impl LoopExecutor {
    /// Runs the loop over a fully-formed message list.
    ///
    /// Shared by [`run_turn`](Self::run_turn) and [`resume`](Self::resume):
    /// they differ only in how the opening conversation is assembled.
    async fn execute(
        &self,
        initial_messages: Vec<Message>,
        event_tx: mpsc::Sender<LoopEvent>,
    ) -> Result<(LoopSummary, Vec<Message>), LoopError> {
        let mut state_machine = AgenticLoop::new(self.config.loop_config.clone());
        let mut messages = initial_messages;
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

            // Record the assistant turn before any branch can exit the loop.
            //
            // This used to live after tool-call detection, which meant a
            // natural termination (answer/stuck/yield) broke out before the
            // final output was ever added. Invisible while `run` discarded
            // the history; visible the moment `run_turn` threads it, as the
            // agent would forget its own answers between turns.
            messages.push(Message::assistant(&output));

            // =================================================================
            // DETECT
            // =================================================================

            // Check for tool calls before honoring a terminal meta-signal.
            //
            // A model that wraps a well-formed call in `<answer>` or
            // `<yield>` instead of `<tool_call>` is still attempting a call,
            // not concluding — measured on Qwen2.5-Coder-14B-Instruct, where
            // this was the dominant failure mode (infernum-framework#70): the
            // tool name and arguments were correct in the large majority of
            // attempts, just wrapped in the wrong tag. Previously this branch
            // ran meta-signal detection first and broke out on Answer/Stuck/
            // Yield before `self.detector.detect` ever ran, so the detector's
            // own wrapper-leniency (see `parse_tool_call_tags`) could never
            // take effect here even after being taught to recognize these
            // wrappers — the call was discarded one step earlier. Computing
            // `detected_calls` first, and only honoring the terminal signal
            // when it's empty, fixes that without changing behavior for a
            // model that already uses `<tool_call>` correctly, or for a
            // genuine final answer/yield/stuck with no call embedded in it
            // (both remain terminal exactly as before).
            let mut detected_calls = self.detector.detect(&output);

            // Under the tool-call grammar there is no `<answer>` tag to
            // detect: the model finishes by CALLING `final_answer`, because a
            // grammar can remove an alternative but cannot re-rank one, and
            // leaving `<answer>` reachable meant Qwen2.5-Coder-14B opened it
            // 30/30 and never emitted a `<tool_call>` at all. See
            // `crate::grammar` for the measurements.
            //
            // Terminal only when it is the sole call in the turn: a model that
            // asks for one more tool *and* declares itself done has not
            // finished, and dropping the real call to honour the declaration
            // would lose work the model just asked for.
            if let Some(answer) = self.intercept_final_answer(&mut detected_calls) {
                // Report it as detected even though nothing executes it. The
                // call is well-formed and the executor consumed it, so a
                // consumer counting `<tool_call>` tags and subtracting
                // detections — which is how `toolcall-eval` derives its
                // malformed count — would otherwise score every completed run
                // as carrying one malformed call. The absence of a following
                // ToolExecutionStarted is what marks it as not dispatched.
                let _ = event_tx
                    .send(LoopEvent::ToolCallDetected {
                        call_id: answer.call_id,
                        tool: FINAL_ANSWER_TOOL.to_string(),
                    })
                    .await;
                state_machine.answer_detected(answer.content, answer.confidence, vec![])?;
                let _ = event_tx
                    .send(LoopEvent::IterationCompleted {
                        iteration,
                        outcome: IterationOutcome::AnswerProvided,
                    })
                    .await;
                break;
            }

            // Check for meta-signals
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
                        if detected_calls.is_empty() {
                            state_machine.answer_detected(content, confidence, caveats)?;
                            let _ = event_tx
                                .send(LoopEvent::IterationCompleted {
                                    iteration,
                                    outcome: IterationOutcome::AnswerProvided,
                                })
                                .await;
                            terminal_signal = true;
                        }
                    },
                    MetaSignal::Stuck {
                        attempts, request, ..
                    } => {
                        if detected_calls.is_empty() {
                            state_machine.stuck_detected(attempts, request)?;
                            let _ = event_tx
                                .send(LoopEvent::IterationCompleted {
                                    iteration,
                                    outcome: IterationOutcome::Stuck,
                                })
                                .await;
                            terminal_signal = true;
                        }
                    },
                    MetaSignal::Yield {
                        partial_progress,
                        suggested_expertise,
                    } => {
                        if detected_calls.is_empty() {
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
                    },
                    MetaSignal::Uncertain { .. } | MetaSignal::Thinking { .. } => {
                        // Non-terminal: continue to tool call detection
                    },
                }
            }

            if terminal_signal {
                break;
            }

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

            let mut agentic_results = Vec::new();

            for call in &detected_calls {
                let _ = event_tx
                    .send(LoopEvent::ToolCallDetected {
                        call_id: call.id.clone(),
                        tool: call.name.clone(),
                    })
                    .await;

                let result = self.execute_single_tool(call, &tool_ctx, &event_tx).await;

                // Add tool result to conversation context
                messages.push(Message::tool_result(
                    &call.id,
                    &tool_result_content(&result),
                ));

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
                Ok(()) => {},
                Err(TransitionError::ResourceLimitReached(_)) => {
                    debug!("Resource limit reached, terminating loop");
                    break;
                },
                Err(e) => return Err(e.into()),
            }
        }

        let mut summary = state_machine.summary();
        info!(
            iterations = summary.iterations_completed,
            tool_calls = summary.tool_calls_made,
            "Agentic loop completed"
        );

        // Store continuation state so `can_resume` and `continuation_token`
        // agree. A store failure degrades to "not resumable" rather than
        // failing the run: the work is done either way, and reporting a token
        // that cannot be loaded would be the defect this exists to fix.
        if summary.can_resume {
            if let Some(store) = &self.continuation_store {
                let state = create_continuation_state(
                    &self.config.session_id,
                    messages.iter().map(message_to_context_message).collect(),
                    summary.tool_results_summary.clone(),
                    summary.exploration_summary.clone(),
                    summary.iterations_completed,
                    summary.tool_calls_made,
                    summary.tokens_generated,
                    self.config.loop_config.clone(),
                    self.config.autonomy.clone(),
                    self.config.system_prompt.clone(),
                    self.config
                        .working_dir
                        .as_ref()
                        .map(|p| p.to_string_lossy().to_string()),
                    summary.termination.clone(),
                );
                match store.store(state).await {
                    Ok(token) => summary.continuation_token = Some(token),
                    Err(e) => {
                        warn!(error = %e, "failed to store continuation; reporting not resumable");
                        summary.can_resume = false;
                    },
                }
            } else {
                // No store: say so, rather than advertising a resume that
                // cannot happen.
                summary.can_resume = false;
            }
        }

        let _ = event_tx
            .send(LoopEvent::LoopCompleted {
                summary: summary.clone(),
            })
            .await;

        Ok((summary, messages))
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
            },
            Permission::RequiresApproval => {
                debug!(tool = %call.name, "Tool requires approval");
                let timeout = self.config.loop_config.approval_timeout;

                if let Some(gate) = &self.approval_gate {
                    // Register the pending request BEFORE emitting the event,
                    // so the entry exists by the time a client tries to deliver.
                    let rx = gate.request(&call.id, &call.name, call.arguments.clone());

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
                            ApprovalDecision::Approve | ApprovalDecision::ApproveAlways { .. },
                        )) => {
                            // Approved — execute the tool
                            self.execute_approved_tool(call, tool_ctx, event_tx).await
                        },
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
                        },
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
                        },
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
            },
            Permission::Allowed => self.execute_approved_tool(call, tool_ctx, event_tx).await,
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
                // Always include the tool's output in data so the model can see it
                let data = match result.data {
                    Some(mut d) => {
                        let needs_output = d.get("output").is_none() && d.get("content").is_none();
                        // A tool that reported an error AND attached data used
                        // to lose the error entirely: the branch below only
                        // synthesises `{"error": ..}` when there is no data at
                        // all, and `tool_result_content` reads the message from
                        // `data`. The model got "Tool execution failed" and
                        // nothing else. Inert for every tool in the current
                        // registry — `bash` was the only one taking this path,
                        // and no longer does — but the trap is worth closing.
                        let needs_error = result.error.is_some() && d.get("error").is_none();
                        if let Some(obj) = d.as_object_mut() {
                            if needs_output {
                                obj.insert("output".to_string(), serde_json::json!(result.output));
                            }
                            if needs_error {
                                obj.insert("error".to_string(), serde_json::json!(result.error));
                            }
                        }
                        d
                    },
                    None => {
                        if result.success {
                            serde_json::json!({"output": result.output})
                        } else {
                            serde_json::json!({"error": result.error.as_deref().unwrap_or("unknown error")})
                        }
                    },
                };
                AgenticToolResult {
                    call_id: call.id.clone(),
                    tool_name: call.name.clone(),
                    status,
                    data,
                    confidence: Confidence::Measured,
                    latency_ms,
                    truncated: false,
                }
            },
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
            },
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

        // Composed through the shared builder so the prompt always matches
        // the grammar the same turn will carry. A prompt offering tags the
        // grammar forbids sets the model against the sampler, which is what
        // infernum-framework#72 looks like from the outside.
        let system_prompt =
            compose_system_prompt(system, &self.tools, self.grammar_constraint().is_some());

        vec![Message::system(system_prompt), Message::user(objective)]
    }

    /// The grammar this turn should carry, if any.
    ///
    /// `None` when the caller disabled it, and `None` for an empty registry:
    /// there is no tool call to constrain, and a grammar permitting only
    /// `<answer>` would quietly turn the loop into a single-shot completion.
    /// Every turn of this loop is tool-eligible — the model may act or
    /// conclude at any point — so there is no narrower gate to apply.
    fn grammar_constraint(&self) -> Option<GrammarConstraint> {
        if !self.config.tool_call_grammar {
            return None;
        }
        tool_call_grammar(&self.tools)
    }

    /// Consumes a terminal `final_answer` call, if this turn has one.
    ///
    /// Returns the answer when `final_answer` is the only call detected. When
    /// it appears alongside real tool calls it is *removed* from the list and
    /// `None` is returned, so the turn dispatches the real work and continues.
    ///
    /// A registry that defines its own `final_answer` tool keeps it: the
    /// grammar does not synthesise one in that case (see
    /// [`FINAL_ANSWER_TOOL`]), so the call
    /// belongs to that tool and must dispatch normally.
    fn intercept_final_answer(&self, calls: &mut Vec<DetectedCall>) -> Option<FinalAnswer> {
        if self.grammar_constraint().is_none() || self.tools.get(FINAL_ANSWER_TOOL).is_some() {
            return None;
        }
        if !calls.iter().any(|c| c.name == FINAL_ANSWER_TOOL) {
            return None;
        }

        if calls.len() > 1 {
            calls.retain(|c| c.name != FINAL_ANSWER_TOOL);
            return None;
        }

        let call = calls.pop()?;
        Some(FinalAnswer {
            call_id: call.id.clone(),
            content: call
                .arguments
                .get("answer")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            confidence: call
                .arguments
                .get("confidence")
                .and_then(serde_json::Value::as_f64)
                .map_or(DEFAULT_ANSWER_CONFIDENCE, |c| (c as f32).clamp(0.0, 1.0)),
        })
    }

    fn build_generate_request(&self, messages: &[Message]) -> GenerateRequest {
        let mut sampling = self.config.sampling.clone();
        // An explicit grammar on the caller's SamplingParams wins: they asked
        // for something more specific than the loop's default envelope.
        if sampling.grammar.is_none() {
            sampling.grammar = self.grammar_constraint();
        }
        GenerateRequest::chat(messages.to_vec()).with_sampling(sampling)
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

/// Renders a tool result as the `tool` message the model will read.
///
/// Extracted so it can be tested directly: this is the last point at which
/// anything a tool learned can be dropped, and a drop here is invisible
/// everywhere else — the `AgenticToolResult` still carries the data, the
/// events still report it, and only the model goes without. See
/// infernum-framework#20.
fn tool_result_content(result: &AgenticToolResult) -> String {
    match &result.status {
        ResultStatus::Success | ResultStatus::PartialSuccess { .. } => {
            serde_json::to_string(&result.data).unwrap_or_default()
        },
        ResultStatus::Empty => "No results found.".to_string(),
        ResultStatus::Failed { .. } => match result.data.get("error").and_then(|e| e.as_str()) {
            Some(message) => format!("Error: {message}"),
            // Last line of defence. "Tool execution failed" on its own tells
            // the model nothing it can act on, and whatever the tool DID
            // learn is sitting right here in `data`. Prefer handing that over
            // to inventing a placeholder — a message that says nothing is how
            // infernum-framework#20 stayed invisible.
            None if !result.data.is_null() => {
                format!(
                    "Error: {}",
                    serde_json::to_string(&result.data).unwrap_or_default()
                )
            },
            None => "Error: Tool execution failed".to_string(),
        },
    }
}

// ---------------------------------------------------------------------------
// Tool call parsing
// ---------------------------------------------------------------------------

/// Parse `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` tags,
/// falling back to a wrapper-agnostic scan when that finds nothing.
///
/// # Why the fallback exists
///
/// Measured against a real Qwen2.5-Coder-14B-Instruct model
/// (infernum-framework#70 and its discriminator comment): across 126
/// captured completions, 89 contained a JSON object with the exact
/// `{"name": ..., "arguments": {...}}` shape the model was instructed to
/// use — right tool, right arguments — just outside the `<tool_call>` tag.
/// Most were wrapped in `<answer confidence="0.9">...</answer>`, some in a
/// fenced ` ```json ` block, some bare. A model that gets the call right and
/// only the tag wrong is attempting a call; scoring that as "no call
/// happened" was a measurement and detection defect, not evidence the model
/// can't do this.
///
/// The strict pass runs first and, if it finds anything, its result is
/// returned unchanged — a model that already emits `<tool_call>` correctly
/// is scored exactly as before, and the fallback only ever activates for the
/// previously-100%-failing case of zero detected calls. This keeps the
/// change narrowly scoped: it recovers a demonstrated failure mode without
/// widening what a *successful* `<tool_call>` emission means.
///
/// # What is deliberately still rejected
///
/// The fallback requires the literal keys `name` (non-empty string) and
/// `arguments` (a JSON object) — exactly the schema
/// [`ToolRegistry::to_qwen_native_description`](crate::tool::ToolRegistry::to_qwen_native_description)
/// instructs the model to use. It does **not** accept key variants such as
/// `function_name`/`function` or `params`/`parameters`: three of the 126
/// captured completions used `function_name` and remain unrecognized after
/// this change — a known, deliberate gap rather than an oversight, left
/// unhandled for lack of enough evidence to widen the schema without
/// increasing false-positive risk on unrelated JSON (e.g. an echoed tool
/// result). It also does not accept prose narrating an intent to call a tool
/// with no JSON attached, or a bare tag with no content — see the
/// `negative_control_*` tests below, each a verbatim transcript from that
/// measurement that must keep producing zero calls.
fn parse_tool_call_tags(output: &str) -> Vec<DetectedCall> {
    let strict = parse_strict_tool_call_tags(output);
    if !strict.is_empty() {
        return strict;
    }
    parse_lenient_call_json(output)
}

/// Strict pass: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`.
fn parse_strict_tool_call_tags(output: &str) -> Vec<DetectedCall> {
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

/// Lenient fallback: scans the whole text for a JSON object shaped exactly
/// like `{"name": "...", "arguments": {...}}`, regardless of what tag,
/// fence, or nothing surrounds it. Only reached when the strict pass above
/// finds nothing — see [`parse_tool_call_tags`] for why, and for what this
/// deliberately does not match.
fn parse_lenient_call_json(output: &str) -> Vec<DetectedCall> {
    find_json_objects(output)
        .into_iter()
        .filter_map(|candidate| {
            let name = candidate.get("name")?.as_str()?;
            if name.is_empty() {
                return None;
            }
            let arguments = candidate.get("arguments")?;
            if !arguments.is_object() {
                return None;
            }
            Some(DetectedCall {
                id: format!("call_{}", uuid::Uuid::new_v4().simple()),
                name: name.to_string(),
                arguments: arguments.clone(),
            })
        })
        .collect()
}

/// Finds every balanced top-level `{...}` object in `text` and returns the
/// ones that parse as JSON, in order of appearance.
///
/// Brace-matching rather than a regex: tool arguments are themselves nested
/// JSON objects, and a JSON-aware regex would not meaningfully simplify on
/// hand-rolled depth counting. `{`/`}` are single-byte ASCII, so byte offsets
/// from `char_indices` are valid UTF-8 slice boundaries here.
fn find_json_objects(text: &str) -> Vec<serde_json::Value> {
    let mut objects = Vec::new();
    let mut depth = 0usize;
    let mut start = None;

    for (i, ch) in text.char_indices() {
        match ch {
            '{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            },
            '}' if depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start.take() {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text[s..=i]) {
                            objects.push(v);
                        }
                    }
                }
            },
            _ => {},
        }
    }

    objects
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

    // -----------------------------------------------------------------------
    // Wrapper-agnostic fallback (infernum-framework#70)
    //
    // The real evidence — all 126 completions captured from a real
    // Qwen2.5-Coder-14B-Instruct model, with full provenance — lives in
    // `crates/beleth/tests/fixtures/issue_70_14b_completions.json`, and
    // every aggregate number quoted in #70/#71 is recomputed against that
    // committed file by `crates/beleth/tests/issue_70_regression.rs`. That
    // is the source of truth; nothing here duplicates it.
    //
    // The unit tests below are fast, in-crate sanity checks for specific
    // *shapes* of input (an answer-wrapped call, a fenced block, a bare
    // object, a stray unrelated object alongside a real tag). Two use text
    // copied verbatim from the fixture (noted on each); the rest are
    // hand-written to exercise a shape distinctly from what the fixture
    // happens to contain — labeled as such, not implied to be captured
    // evidence. An earlier version of this file blurred that line: one of
    // these was a paraphrased approximation of a captured sample presented
    // as if verbatim, and the full dataset wasn't committed at all, so
    // nothing here could actually be checked against source. Both are fixed.
    // -----------------------------------------------------------------------

    #[test]
    fn lenient_fallback_recovers_answer_wrapped_call() {
        // Verbatim: fixture id A/run1/read-cargo-toml#0.
        let output = "<answer confidence=\"0.9\">\n\
             {\n  \"name\": \"read_file\",\n  \"arguments\": {\n    \"path\": \"Cargo.toml\"\n  }\n}\n\
             </answer>";
        let calls = parse_tool_call_tags(output);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments["path"], "Cargo.toml");
    }

    #[test]
    fn lenient_fallback_recovers_fenced_json_call() {
        // Verbatim: fixture id C/run1/bash-cat-sentinel#0, including the
        // real generated fixture path (fixed here after an earlier version
        // of this test quietly shortened it — that made it a paraphrase
        // presented as a quote, which is exactly the kind of thing this
        // whole exercise is supposed to catch, not commit).
        let output = "```json\n{\n  \"name\": \"bash\",\n  \"arguments\": {\n    \"command\": \"cat /tmp/toolcall-eval-ca87b5d104f441bba930395b3320a9ac/secret.txt\"\n  }\n}\n```";
        let calls = parse_tool_call_tags(output);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
        assert_eq!(
            calls[0].arguments["command"],
            "cat /tmp/toolcall-eval-ca87b5d104f441bba930395b3320a9ac/secret.txt"
        );
    }

    #[test]
    fn lenient_fallback_recovers_bare_json_call() {
        // Synthetic — not from the #70 corpus. Exercises the "no wrapper at
        // all" shape, which the captured corpus happens not to contain in
        // isolation (bare JSON there always co-occurs with a fence or tag).
        let output =
            "I'll do that now.\n{\"name\": \"list_files\", \"arguments\": {\"path\": \"src\"}}";
        let calls = parse_tool_call_tags(output);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "list_files");
    }

    #[test]
    fn strict_tool_call_tag_takes_priority_over_stray_json() {
        // Synthetic. If a real <tool_call> is present, the fallback must
        // not also fire on unrelated JSON elsewhere in the same output.
        let output = r#"For reference the config looks like {"name": "unrelated", "arguments": {}}.
<tool_call>
{"name": "read_file", "arguments": {"path": "real.rs"}}
</tool_call>"#;
        let calls = parse_tool_call_tags(output);
        assert_eq!(
            calls.len(),
            1,
            "expected only the tagged call, not the stray JSON too"
        );
        assert_eq!(calls[0].name, "read_file");
    }

    /// Synthetic, documenting a deliberate boundary decision rather than
    /// captured evidence — the fixture's own 3 real `function_name_gap`
    /// completions are checked against this same detector in
    /// `tests/issue_70_regression.rs::every_non_recovered_completion_-
    /// still_produces_zero_calls`.
    #[test]
    fn function_name_key_variant_is_a_known_unhandled_gap() {
        let output = r#"<answer confidence="0.9">
{
  "function_name": "bash",
  "arguments": {
    "command": "pwd"
  }
}
</answer>"#;
        let calls = parse_tool_call_tags(output);
        assert!(
            calls.is_empty(),
            "function_name is intentionally not accepted yet; if this now passes, \
             the scope of the fix has changed and the PR description must be updated"
        );
    }

    // === final_answer interception (infernum-framework#17) ===

    /// A do-nothing engine. These tests exercise request construction and
    /// call interception; none of them generates.
    struct NullEngine {
        metadata: infernum_core::model::ModelMetadata,
    }

    impl Default for NullEngine {
        fn default() -> Self {
            Self {
                metadata: infernum_core::model::ModelMetadata::builder(
                    "test-model",
                    infernum_core::model::ModelArchitecture::Llama {
                        version: infernum_core::model::LlamaVersion::V3,
                    },
                )
                .source(infernum_core::model::ModelSource::huggingface("test-model"))
                .build(),
            }
        }
    }

    #[async_trait]
    impl abaddon::InferenceEngine for NullEngine {
        async fn generate(
            &self,
            _request: GenerateRequest,
        ) -> infernum_core::Result<infernum_core::GenerateResponse> {
            unreachable!("these tests never generate")
        }
        async fn generate_stream(
            &self,
            _request: GenerateRequest,
        ) -> infernum_core::Result<infernum_core::TokenStream> {
            unreachable!("these tests never generate")
        }
        async fn embed(
            &self,
            _request: infernum_core::EmbedRequest,
        ) -> infernum_core::Result<infernum_core::EmbedResponse> {
            unreachable!("these tests never embed")
        }
        fn model_info(&self) -> &infernum_core::model::ModelMetadata {
            &self.metadata
        }
        fn is_ready(&self) -> bool {
            true
        }
    }

    /// An executor over the code registry, with the given config.
    fn executor_with(config: ExecutorConfig) -> LoopExecutor {
        LoopExecutor::new(
            Arc::new(NullEngine::default()),
            Arc::new(ToolRegistry::with_code_tools()),
            config,
        )
    }

    fn executor_with_grammar(enabled: bool) -> LoopExecutor {
        executor_with(ExecutorConfig::new("test").with_tool_call_grammar(enabled))
    }

    fn call(name: &str, arguments: serde_json::Value) -> DetectedCall {
        DetectedCall {
            id: format!("call_{name}"),
            name: name.to_string(),
            arguments,
        }
    }

    #[test]
    fn a_lone_final_answer_call_terminates_the_turn() {
        let exec = executor_with_grammar(true);
        let mut calls = vec![call(
            "final_answer",
            serde_json::json!({"answer": "the flags were -s and --stream", "confidence": 0.95}),
        )];

        let answer = exec
            .intercept_final_answer(&mut calls)
            .expect("a lone final_answer must terminate");
        assert_eq!(answer.content, "the flags were -s and --stream");
        assert!((answer.confidence - 0.95).abs() < 1e-6);
        assert!(
            calls.is_empty(),
            "the call must be consumed, not dispatched"
        );
    }

    #[test]
    fn an_intercepted_final_answer_still_reports_its_call_id() {
        // Consumers derive a malformed-call count by subtracting detections
        // from raw `<tool_call>` tags. A terminal call that produced a tag but
        // no detection would score as malformed on every completed run — a
        // wrong number produced by the fix itself.
        let exec = executor_with_grammar(true);
        let mut calls = vec![call("final_answer", serde_json::json!({"answer": "no"}))];
        let id = calls[0].id.clone();
        let answer = exec.intercept_final_answer(&mut calls).unwrap();
        assert_eq!(answer.call_id, id);
    }

    #[test]
    fn final_answer_alongside_real_work_is_dropped_not_honoured() {
        // The model asked for one more tool AND declared itself done. It has
        // not finished; honouring the declaration would discard the call it
        // just asked for.
        let exec = executor_with_grammar(true);
        let mut calls = vec![
            call("read_file", serde_json::json!({"path": "Cargo.toml"})),
            call("final_answer", serde_json::json!({"answer": "done"})),
        ];

        assert!(exec.intercept_final_answer(&mut calls).is_none());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
    }

    #[test]
    fn final_answer_without_a_confidence_gets_the_documented_default() {
        let exec = executor_with_grammar(true);
        let mut calls = vec![call("final_answer", serde_json::json!({"answer": "no"}))];
        let answer = exec.intercept_final_answer(&mut calls).unwrap();
        assert!((answer.confidence - DEFAULT_ANSWER_CONFIDENCE).abs() < 1e-6);
    }

    #[test]
    fn final_answer_is_not_intercepted_with_the_grammar_off() {
        // With no grammar the model still has `<answer>`, and `final_answer`
        // is not in its prompt at all. Intercepting a call by that name would
        // be inventing a contract the model was never told about.
        let exec = executor_with_grammar(false);
        let mut calls = vec![call("final_answer", serde_json::json!({"answer": "x"}))];
        assert!(exec.intercept_final_answer(&mut calls).is_none());
        assert_eq!(calls.len(), 1, "the call must be left for normal dispatch");
    }

    #[test]
    fn the_grammar_reaches_the_request_beleth_actually_sends() {
        // `with_grammar` appeared zero times in crates/beleth/src before #17.
        // This is the end-to-end assertion that it no longer does: the
        // constraint is on the GenerateRequest the loop hands the engine.
        let exec = executor_with_grammar(true);
        let request = exec.build_generate_request(&[Message::user("hi")]);
        let grammar = request
            .sampling
            .grammar
            .as_ref()
            .expect("a tool-eligible turn must carry the tool-call grammar");
        assert!(grammar.to_gbnf().contains("call-final-answer"));

        let off = executor_with_grammar(false);
        assert!(
            off.build_generate_request(&[Message::user("hi")])
                .sampling
                .grammar
                .is_none(),
            "an opted-out executor must send nothing"
        );
    }

    // === What the model actually reads back (infernum-framework#20) ===

    fn agentic_result(status: ResultStatus, data: serde_json::Value) -> AgenticToolResult {
        AgenticToolResult {
            call_id: "call_1".to_string(),
            tool_name: "bash".to_string(),
            status,
            data,
            confidence: Confidence::Measured,
            latency_ms: 1,
            truncated: false,
        }
    }

    #[test]
    fn a_failed_tool_result_still_names_what_went_wrong() {
        // The `Failed` branch renders only `data["error"]`. A tool that fails
        // without putting its message there vanishes into a generic string —
        // which is what `bash` did, because it reported its message on
        // `ToolResult.error` while also setting `data`, and `data` is what
        // this branch reads.
        let content = tool_result_content(&agentic_result(
            ResultStatus::Failed { recoverable: true },
            serde_json::json!({"exit_code": 1, "command": "false", "output": "(no output)"}),
        ));

        assert!(
            !content.contains("Tool execution failed"),
            "the model must be told what happened, not handed a placeholder \
             while the real detail sits unread in `data`: {content:?}"
        );
        assert!(
            content.contains('1'),
            "the exit code was known and must survive to the model: {content:?}"
        );
    }

    #[test]
    fn a_successful_tool_result_serialises_its_data() {
        // Control: the success path was never broken, and the fix must not
        // change it.
        let content = tool_result_content(&agentic_result(
            ResultStatus::Success,
            serde_json::json!({"output": "hello"}),
        ));
        assert!(content.contains("hello"), "{content:?}");
    }

    #[test]
    fn an_explicit_caller_grammar_is_not_overwritten() {
        // A caller that set its own constraint asked for something more
        // specific than the loop's default envelope.
        let mine = infernum_core::sampling::GrammarConstraint::gbnf("root ::= \"x\"");
        let exec = executor_with(
            ExecutorConfig::new("test")
                .with_sampling(SamplingParams::default().with_grammar(mine.clone())),
        );
        assert_eq!(
            exec.build_generate_request(&[Message::user("hi")])
                .sampling
                .grammar,
            Some(mine)
        );
    }
}
