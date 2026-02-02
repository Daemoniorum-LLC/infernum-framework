//! OODA (Observe-Orient-Decide-Act) Execution Loop.
//!
//! This module provides an OODA executor inspired by Boyd's decision cycle,
//! which is used in Leviathan/Persona Framework for agent orchestration.
//!
//! ## Architecture
//!
//! ```text
//!                    ┌─────────────────────┐
//!                    │   Initial Context   │
//!                    └──────────┬──────────┘
//!                               │
//!            ┌──────────────────▼──────────────────┐
//!            │          OODA Loop                   │
//!            │  ┌────────────────────────────────┐ │
//!            │  │  1. Observe (gather info)      │ │
//!            │  │  2. Orient (analyze/synthesize)│ │
//!            │  │  3. Decide (choose action)     │ │
//!            │  │  4. Act (execute decision)     │ │
//!            │  └────────────────────────────────┘ │
//!            │              │                      │
//!            │     Continue │ or Done              │
//!            └──────────────┴──────────────────────┘
//!                               │
//!                    ┌──────────▼──────────┐
//!                    │    Final Result     │
//!                    └─────────────────────┘
//! ```
//!
//! ## Comparison with ReAct
//!
//! | Aspect | ReAct | OODA |
//! |--------|-------|------|
//! | Focus | Reasoning + Action | Situational awareness |
//! | Phases | Think → Act → Observe | Observe → Orient → Decide → Act |
//! | Strength | Explicit reasoning | Rapid adaptation |
//! | Use Case | Complex reasoning | Dynamic environments |

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use infernum_core::{GenerateRequest, Message, Result, Role, SamplingParams};
use serde::{Deserialize, Serialize};

use crate::tool::{ToolCall, ToolContext, ToolRegistry};
use abaddon::{Engine, InferenceEngine};
use infernum_core::GenerateResponse;

// ============================================================================
// Helper Functions
// ============================================================================

/// Extracts text from the first choice in a generate response.
fn extract_response_text(response: &GenerateResponse) -> String {
    response
        .choices
        .first()
        .map(|c| c.text.clone())
        .unwrap_or_default()
}

/// JSON structure for parsing decision responses.
#[derive(Debug, Deserialize)]
struct DecisionJson {
    action_type: String,
    #[serde(default)]
    tool_name: Option<String>,
    #[serde(default)]
    tool_params: Option<serde_json::Value>,
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    answer: Option<String>,
    #[serde(default)]
    question: Option<String>,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    rationale: Option<String>,
    #[serde(default)]
    confidence: Option<f32>,
    #[serde(default)]
    expected_outcome: Option<String>,
}

/// Parses a decision response, attempting JSON first with text fallback.
fn parse_decision_response(text: &str) -> Result<OodaDecision> {
    // Try to extract JSON from the response (may be wrapped in markdown)
    let json_text = extract_json_block(text);

    if let Ok(parsed) = serde_json::from_str::<DecisionJson>(&json_text) {
        let action = match parsed.action_type.as_str() {
            "execute_tool" => DecisionAction::ExecuteTool {
                tool: parsed.tool_name.unwrap_or_else(|| "unknown".to_string()),
                params: parsed.tool_params.unwrap_or(serde_json::json!({})),
            },
            "gather_info" => DecisionAction::GatherInfo {
                query: parsed.query.unwrap_or_else(|| "Continue analysis".to_string()),
            },
            "final_answer" => DecisionAction::FinalAnswer {
                answer: parsed.answer.unwrap_or_else(|| text.to_string()),
            },
            "request_input" => DecisionAction::RequestInput {
                question: parsed.question.unwrap_or_else(|| "Need more information".to_string()),
            },
            "abort" => DecisionAction::Abort {
                reason: parsed.reason.unwrap_or_else(|| "Task cannot be completed".to_string()),
            },
            _ => return fallback_decision_parse(text),
        };

        return Ok(OodaDecision {
            action,
            rationale: parsed.rationale.unwrap_or_else(|| text.to_string()),
            confidence: parsed.confidence.unwrap_or(0.7),
            alternatives: Vec::new(),
            expected_outcome: parsed.expected_outcome.unwrap_or_else(|| "Task progress".to_string()),
        });
    }

    // Fallback to keyword-based parsing
    fallback_decision_parse(text)
}

/// Extracts JSON block from text (handles markdown code blocks).
fn extract_json_block(text: &str) -> String {
    // Try to find JSON in markdown code block
    if let Some(start) = text.find("```json") {
        if let Some(end) = text[start + 7..].find("```") {
            return text[start + 7..start + 7 + end].trim().to_string();
        }
    }

    // Try to find JSON in generic code block
    if let Some(start) = text.find("```") {
        let after_start = start + 3;
        // Skip language identifier if present
        let json_start = text[after_start..].find('\n').map(|n| after_start + n + 1).unwrap_or(after_start);
        if let Some(end) = text[json_start..].find("```") {
            return text[json_start..json_start + end].trim().to_string();
        }
    }

    // Try to find raw JSON object
    if let Some(start) = text.find('{') {
        if let Some(end) = text.rfind('}') {
            if end > start {
                return text[start..=end].to_string();
            }
        }
    }

    text.to_string()
}

/// Fallback keyword-based decision parsing.
fn fallback_decision_parse(text: &str) -> Result<OodaDecision> {
    let text_lower = text.to_lowercase();

    let action = if text_lower.contains("final answer") || text_lower.contains("conclusion") {
        DecisionAction::FinalAnswer {
            answer: text.to_string(),
        }
    } else if text_lower.contains("abort") || text_lower.contains("cannot") || text_lower.contains("impossible") {
        DecisionAction::Abort {
            reason: text.to_string(),
        }
    } else if text_lower.contains("need input") || text_lower.contains("clarify") || text_lower.contains("ask user") {
        DecisionAction::RequestInput {
            question: text.to_string(),
        }
    } else {
        DecisionAction::GatherInfo {
            query: "Continue analysis".to_string(),
        }
    };

    Ok(OodaDecision {
        action,
        rationale: text.to_string(),
        confidence: 0.6, // Lower confidence for fallback parsing
        alternatives: Vec::new(),
        expected_outcome: "Task progress".to_string(),
    })
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the OODA executor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OodaConfig {
    /// Maximum iterations before stopping.
    pub max_iterations: u32,
    /// Maximum tokens per phase.
    pub max_tokens_per_phase: u32,
    /// Temperature for observation phase.
    pub observe_temperature: f32,
    /// Temperature for orientation phase.
    pub orient_temperature: f32,
    /// Temperature for decision phase.
    pub decide_temperature: f32,
    /// Timeout for tool execution.
    pub tool_timeout: Duration,
    /// Whether to enable parallel observation gathering.
    pub parallel_observe: bool,
    /// Minimum confidence to accept a decision (0.0 - 1.0).
    pub min_confidence: f32,
    /// Number of retries for failed actions.
    pub action_retry_count: u32,
}

impl Default for OodaConfig {
    fn default() -> Self {
        Self {
            max_iterations: 25,
            max_tokens_per_phase: 1024,
            observe_temperature: 0.3,
            orient_temperature: 0.5,
            decide_temperature: 0.7,
            tool_timeout: Duration::from_secs(30),
            parallel_observe: true,
            min_confidence: 0.7,
            action_retry_count: 2,
        }
    }
}

impl OodaConfig {
    /// Creates a config for fast, reactive tasks.
    #[must_use]
    pub fn fast() -> Self {
        Self {
            max_iterations: 10,
            max_tokens_per_phase: 512,
            observe_temperature: 0.2,
            orient_temperature: 0.3,
            decide_temperature: 0.5,
            min_confidence: 0.6,
            ..Default::default()
        }
    }

    /// Creates a config for thorough, deliberate tasks.
    #[must_use]
    pub fn thorough() -> Self {
        Self {
            max_iterations: 50,
            max_tokens_per_phase: 2048,
            observe_temperature: 0.4,
            orient_temperature: 0.6,
            decide_temperature: 0.8,
            min_confidence: 0.85,
            action_retry_count: 3,
            ..Default::default()
        }
    }
}

// ============================================================================
// OODA Phases
// ============================================================================

/// The current phase in the OODA loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OodaPhase {
    /// Gathering information from the environment.
    Observe,
    /// Analyzing and synthesizing observations.
    Orient,
    /// Determining the best course of action.
    Decide,
    /// Executing the decision.
    Act,
}

impl std::fmt::Display for OodaPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Observe => write!(f, "Observe"),
            Self::Orient => write!(f, "Orient"),
            Self::Decide => write!(f, "Decide"),
            Self::Act => write!(f, "Act"),
        }
    }
}

/// An observation gathered during the Observe phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OodaObservation {
    /// Source of the observation (tool name, environment, etc.).
    pub source: String,
    /// The observed data.
    pub data: String,
    /// Relevance score (0.0 - 1.0).
    pub relevance: f32,
    /// Whether this observation is from the current iteration.
    pub is_current: bool,
}

/// Orientation analysis from the Orient phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OodaOrientation {
    /// Current situation assessment.
    pub situation: String,
    /// Identified patterns or insights.
    pub patterns: Vec<String>,
    /// Potential threats or blockers.
    pub threats: Vec<String>,
    /// Opportunities identified.
    pub opportunities: Vec<String>,
    /// Mental model updates.
    pub model_updates: Vec<String>,
}

/// A decision from the Decide phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OodaDecision {
    /// The chosen action.
    pub action: DecisionAction,
    /// Rationale for the decision.
    pub rationale: String,
    /// Confidence in the decision (0.0 - 1.0).
    pub confidence: f32,
    /// Alternative actions considered.
    pub alternatives: Vec<String>,
    /// Expected outcome.
    pub expected_outcome: String,
}

/// Types of decisions the agent can make.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionAction {
    /// Execute a tool.
    ExecuteTool {
        /// Tool name.
        tool: String,
        /// Tool parameters.
        params: serde_json::Value,
    },
    /// Gather more information.
    GatherInfo {
        /// What information to gather.
        query: String,
    },
    /// Provide final answer.
    FinalAnswer {
        /// The final answer.
        answer: String,
    },
    /// Request human input.
    RequestInput {
        /// The question to ask.
        question: String,
    },
    /// Abort the task.
    Abort {
        /// Reason for aborting.
        reason: String,
    },
}

/// Result of an action from the Act phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OodaActionResult {
    /// Whether the action succeeded.
    pub success: bool,
    /// Output from the action.
    pub output: String,
    /// Error message if failed.
    pub error: Option<String>,
    /// Duration of the action.
    pub duration_ms: u64,
    /// Side effects observed.
    pub side_effects: Vec<String>,
}

// ============================================================================
// OODA Step and Result
// ============================================================================

/// A complete OODA cycle step.
#[derive(Debug, Clone, Serialize)]
pub struct OodaStep {
    /// Iteration number.
    pub iteration: u32,
    /// Observations gathered.
    pub observations: Vec<OodaObservation>,
    /// Orientation analysis.
    pub orientation: Option<OodaOrientation>,
    /// Decision made.
    pub decision: Option<OodaDecision>,
    /// Action result.
    pub action_result: Option<OodaActionResult>,
    /// Total duration of this step.
    pub duration_ms: u64,
}

/// Why the OODA loop completed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OodaCompletionReason {
    /// Task completed successfully with final answer.
    FinalAnswer,
    /// Maximum iterations reached.
    MaxIterations,
    /// Timeout exceeded.
    Timeout,
    /// Agent decided to abort.
    Aborted {
        /// The reason for aborting.
        reason: String,
    },
    /// Unrecoverable error.
    Error {
        /// The error message.
        message: String,
    },
    /// Human input requested.
    HumanInputRequired {
        /// The question to ask the human.
        question: String,
    },
}

/// Result of OODA execution.
#[derive(Debug, Clone, Serialize)]
pub struct OodaResult {
    /// Final answer if available.
    pub answer: Option<String>,
    /// Why execution completed.
    pub completion_reason: OodaCompletionReason,
    /// All steps executed.
    pub steps: Vec<OodaStep>,
    /// Total iterations.
    pub total_iterations: u32,
    /// Total duration in milliseconds.
    pub total_duration_ms: u64,
}

// ============================================================================
// OODA Executor
// ============================================================================

/// Callback for OODA execution events.
#[async_trait]
pub trait OodaCallback: Send + Sync {
    /// Called when entering a new phase.
    async fn on_phase(&self, iteration: u32, phase: OodaPhase);
    /// Called when an observation is gathered.
    async fn on_observation(&self, observation: &OodaObservation);
    /// Called when orientation is complete.
    async fn on_orientation(&self, orientation: &OodaOrientation);
    /// Called when a decision is made.
    async fn on_decision(&self, decision: &OodaDecision);
    /// Called when an action completes.
    async fn on_action(&self, result: &OodaActionResult);
}

/// No-op callback implementation.
pub struct NoOpOodaCallback;

#[async_trait]
impl OodaCallback for NoOpOodaCallback {
    async fn on_phase(&self, _iteration: u32, _phase: OodaPhase) {}
    async fn on_observation(&self, _observation: &OodaObservation) {}
    async fn on_orientation(&self, _orientation: &OodaOrientation) {}
    async fn on_decision(&self, _decision: &OodaDecision) {}
    async fn on_action(&self, _result: &OodaActionResult) {}
}

/// The OODA executor.
pub struct OodaExecutor {
    engine: Arc<Engine>,
    tools: Arc<ToolRegistry>,
    config: OodaConfig,
    callback: Arc<dyn OodaCallback>,
}

impl OodaExecutor {
    /// Creates a new OODA executor.
    pub fn new(engine: Arc<Engine>, tools: Arc<ToolRegistry>, config: OodaConfig) -> Self {
        Self {
            engine,
            tools,
            config,
            callback: Arc::new(NoOpOodaCallback),
        }
    }

    /// Sets a callback for execution events.
    #[must_use]
    pub fn with_callback(mut self, callback: Arc<dyn OodaCallback>) -> Self {
        self.callback = callback;
        self
    }

    /// Executes the OODA loop for the given task.
    pub async fn execute(&self, task: &str) -> Result<OodaResult> {
        let start = Instant::now();
        let mut steps = Vec::new();
        let mut messages = vec![
            Message {
                role: Role::System,
                content: self.system_prompt(),
                name: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: task.to_string(),
                name: None,
                tool_call_id: None,
            },
        ];

        for iteration in 1..=self.config.max_iterations {
            let step_start = Instant::now();

            // OBSERVE
            self.callback.on_phase(iteration, OodaPhase::Observe).await;
            let observations = self.observe(&messages).await?;
            for obs in &observations {
                self.callback.on_observation(obs).await;
            }

            // ORIENT
            self.callback.on_phase(iteration, OodaPhase::Orient).await;
            let orientation = self.orient(&messages, &observations).await?;
            self.callback.on_orientation(&orientation).await;

            // DECIDE
            self.callback.on_phase(iteration, OodaPhase::Decide).await;
            let decision = self.decide(&messages, &orientation).await?;
            self.callback.on_decision(&decision).await;

            // ACT
            self.callback.on_phase(iteration, OodaPhase::Act).await;
            let action_result = self.act(&decision).await?;
            self.callback.on_action(&action_result).await;

            // Record step
            let step = OodaStep {
                iteration,
                observations,
                orientation: Some(orientation),
                decision: Some(decision.clone()),
                action_result: Some(action_result.clone()),
                duration_ms: step_start.elapsed().as_millis() as u64,
            };
            steps.push(step);

            // Update context with action result
            messages.push(Message {
                role: Role::Assistant,
                content: format!(
                    "[OODA Iteration {}]\nDecision: {}\nResult: {}",
                    iteration, decision.rationale, action_result.output
                ),
                name: None,
                tool_call_id: None,
            });

            // Check for completion
            match &decision.action {
                DecisionAction::FinalAnswer { answer } => {
                    return Ok(OodaResult {
                        answer: Some(answer.clone()),
                        completion_reason: OodaCompletionReason::FinalAnswer,
                        steps,
                        total_iterations: iteration,
                        total_duration_ms: start.elapsed().as_millis() as u64,
                    });
                }
                DecisionAction::Abort { reason } => {
                    return Ok(OodaResult {
                        answer: None,
                        completion_reason: OodaCompletionReason::Aborted {
                            reason: reason.clone(),
                        },
                        steps,
                        total_iterations: iteration,
                        total_duration_ms: start.elapsed().as_millis() as u64,
                    });
                }
                DecisionAction::RequestInput { question } => {
                    return Ok(OodaResult {
                        answer: None,
                        completion_reason: OodaCompletionReason::HumanInputRequired {
                            question: question.clone(),
                        },
                        steps,
                        total_iterations: iteration,
                        total_duration_ms: start.elapsed().as_millis() as u64,
                    });
                }
                _ => {}
            }
        }

        Ok(OodaResult {
            answer: None,
            completion_reason: OodaCompletionReason::MaxIterations,
            steps,
            total_iterations: self.config.max_iterations,
            total_duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    fn system_prompt(&self) -> String {
        format!(
            "You are an autonomous agent using the OODA (Observe-Orient-Decide-Act) decision loop.\n\n\
             Available tools:\n{}\n\n\
             For each iteration:\n\
             1. OBSERVE: Gather relevant information\n\
             2. ORIENT: Analyze patterns, threats, and opportunities\n\
             3. DECIDE: Choose the best action with rationale\n\
             4. ACT: Execute the decision\n\n\
             Always provide structured JSON responses.",
            self.tools.list().join(", ")
        )
    }

    async fn observe(&self, messages: &[Message]) -> Result<Vec<OodaObservation>> {
        let prompt = "Analyze the current situation. What observations are relevant? \
                      List key facts, context, and any information gaps.";

        let mut observe_messages = messages.to_vec();
        observe_messages.push(Message {
            role: Role::User,
            content: prompt.to_string(),
            name: None,
            tool_call_id: None,
        });

        let request = GenerateRequest::new(observe_messages).with_sampling(
            SamplingParams::default()
                .with_temperature(self.config.observe_temperature)
                .with_max_tokens(self.config.max_tokens_per_phase),
        );

        let response = self.engine.generate(request).await?;

        // Parse observations from response
        Ok(vec![OodaObservation {
            source: "context".to_string(),
            data: extract_response_text(&response),
            relevance: 1.0,
            is_current: true,
        }])
    }

    async fn orient(
        &self,
        messages: &[Message],
        observations: &[OodaObservation],
    ) -> Result<OodaOrientation> {
        let obs_text: String = observations
            .iter()
            .map(|o| format!("- [{}]: {}", o.source, o.data))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Based on these observations:\n{}\n\n\
             Provide orientation analysis:\n\
             1. Current situation assessment\n\
             2. Patterns or insights\n\
             3. Potential threats or blockers\n\
             4. Opportunities\n\
             5. Mental model updates needed",
            obs_text
        );

        let mut orient_messages = messages.to_vec();
        orient_messages.push(Message {
            role: Role::User,
            content: prompt,
            name: None,
            tool_call_id: None,
        });

        let request = GenerateRequest::new(orient_messages).with_sampling(
            SamplingParams::default()
                .with_temperature(self.config.orient_temperature)
                .with_max_tokens(self.config.max_tokens_per_phase),
        );

        let response = self.engine.generate(request).await?;
        let text = extract_response_text(&response);

        Ok(OodaOrientation {
            situation: text,
            patterns: Vec::new(),
            threats: Vec::new(),
            opportunities: Vec::new(),
            model_updates: Vec::new(),
        })
    }

    async fn decide(
        &self,
        messages: &[Message],
        orientation: &OodaOrientation,
    ) -> Result<OodaDecision> {
        let tools_list = self.tools.list().join(", ");
        let prompt = format!(
            r#"Based on orientation:
{}

Decide the next action. Respond with JSON in this exact format:
{{
  "action_type": "execute_tool" | "gather_info" | "final_answer" | "request_input" | "abort",
  "tool_name": "tool name if execute_tool",
  "tool_params": {{}},
  "query": "query if gather_info",
  "answer": "answer if final_answer",
  "question": "question if request_input",
  "reason": "reason if abort",
  "rationale": "why this action",
  "confidence": 0.0-1.0,
  "expected_outcome": "what you expect"
}}

Available tools: {}"#,
            orientation.situation,
            tools_list
        );

        let mut decide_messages = messages.to_vec();
        decide_messages.push(Message {
            role: Role::User,
            content: prompt,
            name: None,
            tool_call_id: None,
        });

        let request = GenerateRequest::new(decide_messages).with_sampling(
            SamplingParams::default()
                .with_temperature(self.config.decide_temperature)
                .with_max_tokens(self.config.max_tokens_per_phase),
        );

        let response = self.engine.generate(request).await?;
        let text = extract_response_text(&response);

        // Parse JSON decision with fallback
        parse_decision_response(&text)
    }

    async fn act(&self, decision: &OodaDecision) -> Result<OodaActionResult> {
        let start = Instant::now();

        match &decision.action {
            DecisionAction::ExecuteTool { tool, params } => {
                let tool_call = ToolCall {
                    name: tool.clone(),
                    params: params.clone(),
                };
                let context = ToolContext::new("ooda-agent");

                match tokio::time::timeout(
                    self.config.tool_timeout,
                    self.tools.execute(&tool_call, &context),
                )
                .await
                {
                    Ok(Ok(result)) => Ok(OodaActionResult {
                        success: result.success,
                        output: result.output,
                        error: result.error,
                        duration_ms: start.elapsed().as_millis() as u64,
                        side_effects: Vec::new(),
                    }),
                    Ok(Err(e)) => Ok(OodaActionResult {
                        success: false,
                        output: String::new(),
                        error: Some(e.to_string()),
                        duration_ms: start.elapsed().as_millis() as u64,
                        side_effects: Vec::new(),
                    }),
                    Err(_) => Ok(OodaActionResult {
                        success: false,
                        output: String::new(),
                        error: Some("Tool execution timed out".to_string()),
                        duration_ms: start.elapsed().as_millis() as u64,
                        side_effects: Vec::new(),
                    }),
                }
            }
            DecisionAction::GatherInfo { query } => Ok(OodaActionResult {
                success: true,
                output: format!("Gathering info: {}", query),
                error: None,
                duration_ms: start.elapsed().as_millis() as u64,
                side_effects: Vec::new(),
            }),
            DecisionAction::FinalAnswer { answer } => Ok(OodaActionResult {
                success: true,
                output: answer.clone(),
                error: None,
                duration_ms: start.elapsed().as_millis() as u64,
                side_effects: Vec::new(),
            }),
            DecisionAction::RequestInput { question } => Ok(OodaActionResult {
                success: true,
                output: format!("Requesting input: {}", question),
                error: None,
                duration_ms: start.elapsed().as_millis() as u64,
                side_effects: Vec::new(),
            }),
            DecisionAction::Abort { reason } => Ok(OodaActionResult {
                success: false,
                output: format!("Aborted: {}", reason),
                error: Some(reason.clone()),
                duration_ms: start.elapsed().as_millis() as u64,
                side_effects: Vec::new(),
            }),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ooda_config_default() {
        let config = OodaConfig::default();
        assert_eq!(config.max_iterations, 25);
        assert_eq!(config.tool_timeout, Duration::from_secs(30));
        assert!(config.parallel_observe);
    }

    #[test]
    fn test_ooda_config_fast() {
        let config = OodaConfig::fast();
        assert_eq!(config.max_iterations, 10);
        assert!((config.min_confidence - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_ooda_config_thorough() {
        let config = OodaConfig::thorough();
        assert_eq!(config.max_iterations, 50);
        assert!((config.min_confidence - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_ooda_phase_display() {
        assert_eq!(format!("{}", OodaPhase::Observe), "Observe");
        assert_eq!(format!("{}", OodaPhase::Orient), "Orient");
        assert_eq!(format!("{}", OodaPhase::Decide), "Decide");
        assert_eq!(format!("{}", OodaPhase::Act), "Act");
    }

    #[test]
    fn test_ooda_observation_creation() {
        let obs = OodaObservation {
            source: "tool:search".to_string(),
            data: "Found 5 results".to_string(),
            relevance: 0.9,
            is_current: true,
        };
        assert_eq!(obs.source, "tool:search");
        assert!(obs.is_current);
    }

    #[test]
    fn test_ooda_orientation_creation() {
        let orient = OodaOrientation {
            situation: "Task in progress".to_string(),
            patterns: vec!["Pattern A".to_string()],
            threats: vec!["Threat B".to_string()],
            opportunities: vec!["Opportunity C".to_string()],
            model_updates: Vec::new(),
        };
        assert_eq!(orient.patterns.len(), 1);
        assert_eq!(orient.threats.len(), 1);
    }

    #[test]
    fn test_decision_action_variants() {
        let tool_action = DecisionAction::ExecuteTool {
            tool: "search".to_string(),
            params: serde_json::json!({"query": "test"}),
        };
        assert!(matches!(tool_action, DecisionAction::ExecuteTool { .. }));

        let final_action = DecisionAction::FinalAnswer {
            answer: "Done".to_string(),
        };
        assert!(matches!(final_action, DecisionAction::FinalAnswer { .. }));

        let abort_action = DecisionAction::Abort {
            reason: "Impossible".to_string(),
        };
        assert!(matches!(abort_action, DecisionAction::Abort { .. }));
    }

    #[test]
    fn test_ooda_step_creation() {
        let step = OodaStep {
            iteration: 1,
            observations: vec![OodaObservation {
                source: "test".to_string(),
                data: "data".to_string(),
                relevance: 1.0,
                is_current: true,
            }],
            orientation: None,
            decision: None,
            action_result: None,
            duration_ms: 100,
        };
        assert_eq!(step.iteration, 1);
        assert_eq!(step.observations.len(), 1);
    }

    #[test]
    fn test_ooda_completion_reasons() {
        let final_answer = OodaCompletionReason::FinalAnswer;
        assert!(matches!(final_answer, OodaCompletionReason::FinalAnswer));

        let aborted = OodaCompletionReason::Aborted {
            reason: "test".to_string(),
        };
        assert!(matches!(aborted, OodaCompletionReason::Aborted { .. }));

        let human_input = OodaCompletionReason::HumanInputRequired {
            question: "help?".to_string(),
        };
        assert!(matches!(
            human_input,
            OodaCompletionReason::HumanInputRequired { .. }
        ));
    }

    #[test]
    fn test_ooda_result_creation() {
        let result = OodaResult {
            answer: Some("42".to_string()),
            completion_reason: OodaCompletionReason::FinalAnswer,
            steps: Vec::new(),
            total_iterations: 3,
            total_duration_ms: 1500,
        };
        assert_eq!(result.answer, Some("42".to_string()));
        assert_eq!(result.total_iterations, 3);
    }

    #[test]
    fn test_ooda_action_result_creation() {
        let result = OodaActionResult {
            success: true,
            output: "Done".to_string(),
            error: None,
            duration_ms: 50,
            side_effects: vec!["created file".to_string()],
        };
        assert!(result.success);
        assert_eq!(result.side_effects.len(), 1);
    }

    #[test]
    fn test_extract_json_block_raw() {
        let text = r#"{"action_type": "final_answer", "answer": "42"}"#;
        let extracted = extract_json_block(text);
        assert!(extracted.contains("final_answer"));
    }

    #[test]
    fn test_extract_json_block_markdown() {
        let text = r#"Here's the decision:
```json
{"action_type": "gather_info", "query": "search"}
```
That's my choice."#;
        let extracted = extract_json_block(text);
        assert!(extracted.contains("gather_info"));
        assert!(!extracted.contains("```"));
    }

    #[test]
    fn test_extract_json_block_generic_code_block() {
        let text = r#"```
{"action_type": "abort", "reason": "impossible"}
```"#;
        let extracted = extract_json_block(text);
        assert!(extracted.contains("abort"));
    }

    #[test]
    fn test_parse_decision_json_final_answer() {
        let json = r#"{"action_type": "final_answer", "answer": "The answer is 42", "confidence": 0.95}"#;
        let decision = parse_decision_response(json).unwrap();
        assert!(matches!(decision.action, DecisionAction::FinalAnswer { .. }));
        assert!((decision.confidence - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_parse_decision_json_execute_tool() {
        let json = r#"{"action_type": "execute_tool", "tool_name": "search", "tool_params": {"query": "rust"}}"#;
        let decision = parse_decision_response(json).unwrap();
        match decision.action {
            DecisionAction::ExecuteTool { tool, params } => {
                assert_eq!(tool, "search");
                assert_eq!(params["query"], "rust");
            }
            _ => panic!("Expected ExecuteTool"),
        }
    }

    #[test]
    fn test_parse_decision_json_gather_info() {
        let json = r#"{"action_type": "gather_info", "query": "What files exist?"}"#;
        let decision = parse_decision_response(json).unwrap();
        match decision.action {
            DecisionAction::GatherInfo { query } => {
                assert_eq!(query, "What files exist?");
            }
            _ => panic!("Expected GatherInfo"),
        }
    }

    #[test]
    fn test_parse_decision_fallback() {
        let text = "I think the final answer is that we need more data.";
        let decision = parse_decision_response(text).unwrap();
        assert!(matches!(decision.action, DecisionAction::FinalAnswer { .. }));
        assert!((decision.confidence - 0.6).abs() < 0.01); // Fallback has lower confidence
    }

    #[test]
    fn test_parse_decision_fallback_abort() {
        let text = "This task is impossible to complete without more resources.";
        let decision = parse_decision_response(text).unwrap();
        assert!(matches!(decision.action, DecisionAction::Abort { .. }));
    }

    #[test]
    fn test_fallback_decision_parse_request_input() {
        let text = "I need to clarify what you mean by that.";
        let decision = fallback_decision_parse(text).unwrap();
        assert!(matches!(decision.action, DecisionAction::RequestInput { .. }));
    }
}
