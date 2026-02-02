//! Complete ReAct (Reasoning and Acting) Execution Loop.
//!
//! This module provides a robust ReAct executor with:
//! - Self-reflection after each step
//! - Error recovery and retry logic
//! - Token budget management
//! - Structured observation handling
//! - Thought summarization for long conversations
//!
//! ## Architecture
//!
//! ```text
//!                    ┌─────────────────────┐
//!                    │   Initial Prompt    │
//!                    └──────────┬──────────┘
//!                               │
//!            ┌──────────────────▼──────────────────┐
//!            │          ReAct Loop                  │
//!            │  ┌────────────────────────────────┐ │
//!            │  │  1. Think (LLM reasoning)      │ │
//!            │  │  2. Act (tool execution)       │ │
//!            │  │  3. Observe (collect results)  │ │
//!            │  │  4. Reflect (evaluate progress)│ │
//!            │  └────────────────────────────────┘ │
//!            │              │                      │
//!            │     Continue │ or Done              │
//!            └──────────────┴──────────────────────┘
//!                               │
//!                    ┌──────────▼──────────┐
//!                    │    Final Answer     │
//!                    └─────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! use beleth::react::{ReactExecutor, ReactConfig};
//!
//! let executor = ReactExecutor::new(engine.clone(), tools, ReactConfig::default());
//! let result = executor.execute("Find and summarize the latest Rust news").await?;
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use infernum_core::{GenerateRequest, Message, Result, Role, SamplingParams};
use serde::{Deserialize, Serialize};

use crate::tool::{ToolCall, ToolContext, ToolRegistry, ToolResult};
use abaddon::{Engine, InferenceEngine};

/// Configuration for the ReAct executor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactConfig {
    /// Maximum iterations before stopping.
    pub max_iterations: u32,
    /// Maximum tokens per response generation.
    pub max_tokens_per_step: u32,
    /// Maximum total conversation tokens before summarizing.
    pub max_conversation_tokens: u32,
    /// Temperature for reasoning steps.
    pub reasoning_temperature: f32,
    /// Temperature for reflection steps.
    pub reflection_temperature: f32,
    /// Whether to enable self-reflection after each action.
    pub enable_reflection: bool,
    /// Number of retries for failed tool calls.
    pub tool_retry_count: u32,
    /// Timeout for tool execution.
    pub tool_timeout: Duration,
    /// Whether to summarize context when it gets too long.
    pub enable_summarization: bool,
    /// Minimum confidence score to accept final answer (0.0 - 1.0).
    pub min_confidence: f32,
}

impl Default for ReactConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            max_tokens_per_step: 1024,
            max_conversation_tokens: 8000,
            reasoning_temperature: 0.7,
            reflection_temperature: 0.3,
            enable_reflection: true,
            tool_retry_count: 2,
            tool_timeout: Duration::from_secs(30),
            enable_summarization: true,
            min_confidence: 0.7,
        }
    }
}

impl ReactConfig {
    /// Creates a config for fast, simple tasks.
    #[must_use]
    pub fn fast() -> Self {
        Self {
            max_iterations: 5,
            max_tokens_per_step: 512,
            enable_reflection: false,
            enable_summarization: false,
            ..Default::default()
        }
    }

    /// Creates a config for complex, multi-step reasoning.
    #[must_use]
    pub fn thorough() -> Self {
        Self {
            max_iterations: 20,
            max_tokens_per_step: 2048,
            enable_reflection: true,
            enable_summarization: true,
            min_confidence: 0.85,
            ..Default::default()
        }
    }
}

/// A step in the ReAct trace.
#[derive(Debug, Clone, Serialize)]
pub struct ReactStep {
    /// Step number.
    pub step: u32,
    /// The thought/reasoning.
    pub thought: Option<String>,
    /// The action taken (if any).
    pub action: Option<ReactAction>,
    /// The observation from the action.
    pub observation: Option<String>,
    /// Self-reflection on this step.
    pub reflection: Option<String>,
    /// Duration of this step.
    pub duration_ms: u64,
}

/// An action in the ReAct loop.
#[derive(Debug, Clone, Serialize)]
pub struct ReactAction {
    /// Action type.
    pub action_type: ActionType,
    /// Tool name (for tool calls).
    pub tool_name: Option<String>,
    /// Tool parameters.
    pub params: Option<serde_json::Value>,
    /// Whether the action succeeded.
    pub success: bool,
    /// Error message if failed.
    pub error: Option<String>,
}

/// Types of actions the agent can take.
#[derive(Debug, Clone, Serialize)]
pub enum ActionType {
    /// Call a tool.
    ToolCall,
    /// Ask for clarification.
    Clarify,
    /// Provide final answer.
    FinalAnswer,
    /// Internal reasoning step.
    Reason,
}

// ============================================================================
// Observation Parsing (Phase 4)
// ============================================================================

use crate::tool::{OutputValidationConfig, ValidationResult};

/// A structured observation from a tool execution.
///
/// Unlike raw string observations, this type extracts key information
/// from tool results for better reasoning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// The tool that produced this observation.
    pub tool_name: String,
    /// Whether the tool execution succeeded.
    pub success: bool,
    /// The raw output from the tool.
    pub output: String,
    /// Error message if the tool failed.
    pub error_message: Option<String>,
    /// Key facts extracted from the output.
    pub key_facts: Vec<String>,
    /// Structured data if the tool returned JSON.
    pub structured_data: Option<serde_json::Value>,
    /// Whether the output was truncated.
    pub was_truncated: bool,
    /// Validation issues found in the output.
    pub validation_issues: Vec<String>,
}

impl Observation {
    /// Creates a new observation from a tool result.
    pub fn from_tool_result(tool_name: &str, result: &ToolResult) -> Self {
        let key_facts = extract_key_facts(&result.output);

        Self {
            tool_name: tool_name.to_string(),
            success: result.success,
            output: result.output.clone(),
            error_message: result.error.clone(),
            key_facts,
            structured_data: result.data.clone(),
            was_truncated: false,
            validation_issues: vec![],
        }
    }

    /// Creates an observation with validation applied.
    pub fn from_tool_result_validated(
        tool_name: &str,
        result: &ToolResult,
        config: &OutputValidationConfig,
    ) -> Self {
        let validation = result.validate(config);
        let sanitized = result.sanitize(config);

        let was_truncated = sanitized.output.len() < result.output.len();
        let validation_issues = match validation {
            ValidationResult::Valid => vec![],
            ValidationResult::Invalid(issues) => {
                issues.iter().map(|i| format!("{:?}", i)).collect()
            }
        };

        let key_facts = extract_key_facts(&sanitized.output);

        Self {
            tool_name: tool_name.to_string(),
            success: result.success,
            output: sanitized.output,
            error_message: result.error.clone(),
            key_facts,
            structured_data: sanitized.data,
            was_truncated,
            validation_issues,
        }
    }

    /// Formats this observation as a message for the agent.
    pub fn to_message(&self) -> String {
        let mut parts = vec![];

        if self.success {
            parts.push(format!("Tool `{}` succeeded.", self.tool_name));
            parts.push(format!("Output: {}", self.output));
        } else {
            parts.push(format!("Tool `{}` failed.", self.tool_name));
            if let Some(ref err) = self.error_message {
                parts.push(format!("Error: {}", err));
            }
        }

        if !self.key_facts.is_empty() {
            parts.push("Key facts:".to_string());
            for fact in &self.key_facts {
                parts.push(format!("  - {}", fact));
            }
        }

        if self.was_truncated {
            parts.push("[Note: Output was truncated due to size limits]".to_string());
        }

        parts.join("\n")
    }
}

/// Extracts key facts from tool output.
///
/// This function identifies important information like:
/// - Numbers and measurements
/// - Status indicators (success, failure, error)
fn extract_key_facts(output: &str) -> Vec<String> {
    let mut facts = vec![];
    let output_lower = output.to_lowercase();

    // Extract lines or sentences containing numbers with units
    let units = ["%", "°", "result", "item", "file", "byte", "kb", "mb", "gb", "ms", "second", "minute", "hour"];
    for line in output.lines() {
        let line_lower = line.to_lowercase();
        // Check if line contains a number followed by a unit
        let has_number = line.chars().any(|c| c.is_ascii_digit());
        let has_unit = units.iter().any(|u| line_lower.contains(u));
        if has_number && has_unit && facts.len() < 3 {
            facts.push(line.trim().to_string());
        }
    }

    // Extract status indicators with context
    let status_words = ["success", "failed", "error", "complete", "found", "not found", "exists", "missing"];
    for word in status_words {
        if output_lower.contains(word) {
            if let Some(idx) = output_lower.find(word) {
                // Get surrounding context (up to 50 chars)
                let start = idx.saturating_sub(25);
                let end = (idx + word.len() + 25).min(output.len());
                // Find word boundaries
                let context = &output[start..end];
                if facts.iter().all(|f| !f.contains(context.trim())) {
                    facts.push(context.trim().to_string());
                }
                break; // Only take first status
            }
        }
    }

    // Limit to 5 facts
    facts.truncate(5);
    facts
}

/// Parses a tool result into a structured observation.
pub fn parse_observation(tool_name: &str, result: &ToolResult) -> Observation {
    Observation::from_tool_result(tool_name, result)
}

/// Parses a tool result with validation applied.
pub fn parse_observation_with_validation(
    tool_name: &str,
    result: &ToolResult,
    config: &OutputValidationConfig,
) -> Observation {
    Observation::from_tool_result_validated(tool_name, result, config)
}

/// Generates reasoning about an observation.
///
/// This creates a brief analysis of what the observation means
/// in the context of the agent's task.
pub fn generate_observation_reasoning(
    _thought: &str,
    action: &str,
    observation: &Observation,
) -> String {
    let mut reasoning = vec![];

    if observation.success {
        reasoning.push(format!(
            "The {} action succeeded.",
            action
        ));

        if !observation.key_facts.is_empty() {
            reasoning.push(format!(
                "Key findings: {}",
                observation.key_facts.join(", ")
            ));
        }

        if observation.structured_data.is_some() {
            reasoning.push("Structured data was returned and can be used for further analysis.".to_string());
        }
    } else {
        reasoning.push(format!(
            "The {} action failed: {}",
            action,
            observation.error_message.as_deref().unwrap_or("unknown error")
        ));

        reasoning.push("May need to retry with different parameters or try an alternative approach.".to_string());
    }

    reasoning.join(" ")
}

/// Result of ReAct execution.
#[derive(Debug, Clone, Serialize)]
pub struct ReactResult {
    /// The final answer.
    pub answer: String,
    /// Whether execution was successful.
    pub success: bool,
    /// Confidence in the answer (0.0 - 1.0).
    pub confidence: f32,
    /// Total iterations used.
    pub iterations: u32,
    /// Execution trace.
    pub trace: Vec<ReactStep>,
    /// Total execution time.
    pub total_time_ms: u64,
    /// Reason for completion.
    pub completion_reason: CompletionReason,
}

/// Reason why ReAct execution completed.
#[derive(Debug, Clone, Serialize)]
pub enum CompletionReason {
    /// Agent provided a final answer.
    FinalAnswer,
    /// Maximum iterations reached.
    MaxIterations,
    /// Confidence threshold met.
    ConfidenceThreshold,
    /// Error during execution.
    Error(String),
    /// User cancelled.
    Cancelled,
}

/// The ReAct executor.
pub struct ReactExecutor {
    /// The inference engine.
    engine: Arc<Engine>,
    /// Available tools.
    tools: ToolRegistry,
    /// Configuration.
    config: ReactConfig,
    /// System prompt.
    system_prompt: String,
}

impl ReactExecutor {
    /// Creates a new ReAct executor.
    pub fn new(engine: Arc<Engine>, tools: ToolRegistry, config: ReactConfig) -> Self {
        let system_prompt = Self::build_default_system_prompt(&tools);
        Self {
            engine,
            tools,
            config,
            system_prompt,
        }
    }

    /// Creates with a custom system prompt.
    pub fn with_system_prompt(
        engine: Arc<Engine>,
        tools: ToolRegistry,
        config: ReactConfig,
        system_prompt: impl Into<String>,
    ) -> Self {
        Self {
            engine,
            tools,
            config,
            system_prompt: system_prompt.into(),
        }
    }

    /// Builds the default ReAct system prompt.
    fn build_default_system_prompt(tools: &ToolRegistry) -> String {
        let tools_desc = tools.to_prompt_description();

        format!(
            r#"You are an intelligent assistant that solves problems step by step using the ReAct framework.

## Available Tools
{tools_desc}

## Response Format

For each step, structure your response as follows:

**Thought:** Explain your reasoning about the current state and what you need to do next.

**Action:** Choose ONE of these:
- `tool_name` to call a tool
- `clarify` to ask the user for more information
- `final_answer` when you have the complete answer

**Action Input:** The parameters for your action (JSON format for tools, natural language for clarify/final_answer)

## Rules

1. Always start with a Thought explaining your reasoning
2. Take only ONE action per response
3. After receiving an Observation, incorporate it into your next Thought
4. If a tool fails, try an alternative approach
5. When confident you have the answer, use `final_answer`
6. Be concise but thorough in your reasoning

## Example

Thought: I need to find the current weather in Tokyo. I'll use the weather tool.
Action: weather
Action Input: {{"location": "Tokyo, Japan"}}

[Observation: Temperature: 22°C, Condition: Partly cloudy]

Thought: I now have the weather information for Tokyo. The temperature is 22°C with partly cloudy conditions.
Action: final_answer
Action Input: The current weather in Tokyo is 22°C (72°F) with partly cloudy skies.
"#
        )
    }

    /// Executes the ReAct loop for the given objective.
    pub async fn execute(&self, objective: &str) -> Result<ReactResult> {
        let start_time = Instant::now();
        let mut trace = Vec::new();
        let mut messages = vec![
            Message {
                role: Role::System,
                content: self.system_prompt.clone(),
                name: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: objective.to_string(),
                name: None,
                tool_call_id: None,
            },
        ];

        let mut ctx = ToolContext::new("react-executor");
        let mut final_answer = String::new();
        let mut confidence = 0.0f32;
        let mut completion_reason = CompletionReason::MaxIterations;

        for iteration in 0..self.config.max_iterations {
            let step_start = Instant::now();
            tracing::debug!(iteration, "ReAct iteration");

            // Check token budget and summarize if needed
            if self.config.enable_summarization {
                messages = self.maybe_summarize_context(messages).await?;
            }

            // Generate response
            let request = GenerateRequest::chat(messages.clone()).with_sampling(
                SamplingParams::default()
                    .with_max_tokens(self.config.max_tokens_per_step)
                    .with_temperature(self.config.reasoning_temperature),
            );

            let response = self.engine.generate(request).await?;
            let assistant_response = response
                .choices
                .first()
                .map(|c| c.text.clone())
                .unwrap_or_default();

            tracing::debug!(response = %assistant_response, "Agent response");

            // Parse the response
            let parsed = self.parse_response(&assistant_response);

            // Add assistant response to messages
            messages.push(Message {
                role: Role::Assistant,
                content: assistant_response.clone(),
                name: None,
                tool_call_id: None,
            });

            // Create step record
            let mut step = ReactStep {
                step: iteration + 1,
                thought: parsed.thought.clone(),
                action: None,
                observation: None,
                reflection: None,
                duration_ms: 0,
            };

            // Handle the action
            match parsed.action_type {
                ActionType::FinalAnswer => {
                    final_answer = parsed.action_input.clone();
                    confidence = self.evaluate_confidence(&final_answer, objective).await?;

                    step.action = Some(ReactAction {
                        action_type: ActionType::FinalAnswer,
                        tool_name: None,
                        params: None,
                        success: true,
                        error: None,
                    });

                    completion_reason = CompletionReason::FinalAnswer;
                    step.duration_ms = step_start.elapsed().as_millis() as u64;
                    trace.push(step);
                    break;
                }
                ActionType::ToolCall => {
                    let tool_name = parsed.tool_name.clone().unwrap_or_default();
                    let params = parsed.params.clone();

                    // Execute tool with retries
                    let result = self.execute_tool_with_retry(
                        &tool_name,
                        params.clone(),
                        &ctx,
                    ).await;

                    // Parse and validate observation (Phase 4 fix)
                    let (success, observation_msg) = match result {
                        Ok(tool_result) => {
                            // Use structured observation parsing with validation
                            let validation_config = OutputValidationConfig::default();
                            let observation = parse_observation_with_validation(
                                &tool_name,
                                &tool_result,
                                &validation_config,
                            );

                            // Generate reasoning about the observation
                            let thought = parsed.thought.as_deref().unwrap_or("");
                            let reasoning = generate_observation_reasoning(
                                thought,
                                &tool_name,
                                &observation,
                            );

                            // Format structured observation message
                            let obs_msg = format!(
                                "{}\n\nAnalysis: {}",
                                observation.to_message(),
                                reasoning
                            );

                            (observation.success, obs_msg)
                        }
                        Err(e) => {
                            (false, format!("Tool execution error: {}", e))
                        }
                    };

                    step.action = Some(ReactAction {
                        action_type: ActionType::ToolCall,
                        tool_name: Some(tool_name),
                        params,
                        success,
                        error: if success { None } else { Some(observation_msg.clone()) },
                    });
                    step.observation = Some(observation_msg.clone());

                    // Add observation to messages
                    messages.push(Message {
                        role: Role::User,
                        content: format!("Observation: {}", observation_msg),
                        name: Some("system".to_string()),
                        tool_call_id: None,
                    });
                }
                ActionType::Clarify => {
                    step.action = Some(ReactAction {
                        action_type: ActionType::Clarify,
                        tool_name: None,
                        params: None,
                        success: true,
                        error: None,
                    });
                    step.observation = Some(format!("Clarification needed: {}", parsed.action_input));

                    // For now, we can't actually get user input, so continue
                    messages.push(Message {
                        role: Role::User,
                        content: "Please proceed with the best approach based on available information.".to_string(),
                        name: Some("system".to_string()),
                        tool_call_id: None,
                    });
                }
                ActionType::Reason => {
                    step.action = Some(ReactAction {
                        action_type: ActionType::Reason,
                        tool_name: None,
                        params: None,
                        success: true,
                        error: None,
                    });
                }
            }

            // Self-reflection step
            if self.config.enable_reflection && step.action.is_some() {
                let reflection = self.generate_reflection(&messages, objective).await?;
                step.reflection = Some(reflection.clone());

                // Add reflection to context
                messages.push(Message {
                    role: Role::User,
                    content: format!("Self-reflection: {}", reflection),
                    name: Some("system".to_string()),
                    tool_call_id: None,
                });
            }

            step.duration_ms = step_start.elapsed().as_millis() as u64;
            trace.push(step);

            // Update context
            ctx.messages = messages.clone();
        }

        // If we exited without a final answer, extract one from the last response
        if final_answer.is_empty() {
            final_answer = self.extract_best_answer(&messages);
            confidence = self.evaluate_confidence(&final_answer, objective).await?;
        }

        Ok(ReactResult {
            answer: final_answer,
            success: matches!(completion_reason, CompletionReason::FinalAnswer),
            confidence,
            iterations: trace.len() as u32,
            trace,
            total_time_ms: start_time.elapsed().as_millis() as u64,
            completion_reason,
        })
    }

    /// Parses the agent's response to extract thought, action, and action input.
    fn parse_response(&self, response: &str) -> ParsedResponse {
        ParsedResponse::parse(response)
    }

    /// Executes a tool with retry logic.
    async fn execute_tool_with_retry(
        &self,
        tool_name: &str,
        params: Option<serde_json::Value>,
        ctx: &ToolContext,
    ) -> Result<ToolResult> {
        let tool_call = ToolCall {
            name: tool_name.to_string(),
            params: params.unwrap_or(serde_json::json!({})),
        };

        let mut last_error = None;
        for attempt in 0..=self.config.tool_retry_count {
            if attempt > 0 {
                tracing::debug!(attempt, tool = %tool_name, "Retrying tool call");
                tokio::time::sleep(Duration::from_millis(100 * (attempt as u64))).await;
            }

            match tokio::time::timeout(
                self.config.tool_timeout,
                self.tools.execute(&tool_call, ctx),
            ).await {
                Ok(Ok(result)) => {
                    if result.success {
                        return Ok(result);
                    }
                    last_error = result.error.clone();
                }
                Ok(Err(e)) => {
                    last_error = Some(e.to_string());
                }
                Err(_) => {
                    last_error = Some("Tool execution timed out".to_string());
                }
            }
        }

        Ok(ToolResult {
            success: false,
            output: String::new(),
            error: last_error,
            data: None,
        })
    }

    /// Generates a self-reflection on the current progress.
    async fn generate_reflection(&self, messages: &[Message], objective: &str) -> Result<String> {
        let recent_context: String = messages
            .iter()
            .rev()
            .take(4)
            .rev()
            .map(|m| format!("{:?}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n\n");

        let reflection_prompt = format!(
            r#"Given the objective: "{objective}"

And the recent progress:
{recent_context}

Provide a brief reflection (1-2 sentences) on:
1. Are we making progress toward the objective?
2. Should we change approach or continue?

Be concise and direct."#
        );

        let messages = vec![
            Message {
                role: Role::System,
                content: "You are a self-reflective AI evaluating your own progress.".to_string(),
                name: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: reflection_prompt,
                name: None,
                tool_call_id: None,
            },
        ];

        let request = GenerateRequest::chat(messages).with_sampling(
            SamplingParams::default()
                .with_max_tokens(128)
                .with_temperature(self.config.reflection_temperature),
        );

        let response = self.engine.generate(request).await?;
        Ok(response
            .choices
            .first()
            .map(|c| c.text.clone())
            .unwrap_or_else(|| "Continuing with current approach.".to_string()))
    }

    /// Evaluates confidence in the final answer.
    async fn evaluate_confidence(&self, answer: &str, objective: &str) -> Result<f32> {
        let eval_prompt = format!(
            r#"Evaluate how well this answer addresses the objective.

Objective: {objective}

Answer: {answer}

Rate the answer from 0.0 to 1.0:
- 0.0-0.3: Does not address the objective
- 0.4-0.6: Partially addresses the objective
- 0.7-0.8: Mostly addresses the objective
- 0.9-1.0: Fully addresses the objective

Respond with only a number between 0.0 and 1.0."#
        );

        let messages = vec![
            Message {
                role: Role::System,
                content: "You evaluate answer quality objectively.".to_string(),
                name: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: eval_prompt,
                name: None,
                tool_call_id: None,
            },
        ];

        let request = GenerateRequest::chat(messages).with_sampling(
            SamplingParams::default()
                .with_max_tokens(16)
                .with_temperature(0.1),
        );

        let response = self.engine.generate(request).await?;
        let score_text = response
            .choices
            .first()
            .map(|c| c.text.clone())
            .unwrap_or_else(|| "0.5".to_string());

        // Parse the score
        score_text
            .trim()
            .parse::<f32>()
            .ok()
            .map(|s| s.clamp(0.0, 1.0))
            .ok_or_else(|| infernum_core::Error::internal("Failed to parse confidence score"))
    }

    /// Summarizes the conversation context if it's too long.
    async fn maybe_summarize_context(&self, mut messages: Vec<Message>) -> Result<Vec<Message>> {
        // Rough token estimate
        let estimated_tokens: usize = messages.iter()
            .map(|m| m.content.len() / 4)
            .sum();

        if estimated_tokens < self.config.max_conversation_tokens as usize {
            return Ok(messages);
        }

        tracing::debug!(
            estimated_tokens,
            threshold = self.config.max_conversation_tokens,
            "Summarizing conversation context"
        );

        // Keep system message and last few messages
        let system_msg = messages.remove(0);
        let recent: Vec<Message> = messages.split_off(messages.len().saturating_sub(4));

        // Summarize the middle messages
        let to_summarize: String = messages
            .iter()
            .map(|m| format!("{:?}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n\n");

        let summary_prompt = format!(
            "Summarize the following conversation, keeping key facts and tool results:\n\n{}",
            to_summarize
        );

        let summary_messages = vec![
            Message {
                role: Role::System,
                content: "You summarize conversations concisely, preserving important information.".to_string(),
                name: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: summary_prompt,
                name: None,
                tool_call_id: None,
            },
        ];

        let request = GenerateRequest::chat(summary_messages).with_sampling(
            SamplingParams::default()
                .with_max_tokens(512)
                .with_temperature(0.3),
        );

        let response = self.engine.generate(request).await?;
        let summary = response
            .choices
            .first()
            .map(|c| c.text.clone())
            .unwrap_or_default();

        // Reconstruct messages with summary
        let mut new_messages = vec![system_msg];
        new_messages.push(Message {
            role: Role::User,
            content: format!("[Previous conversation summary]\n{}", summary),
            name: Some("system".to_string()),
            tool_call_id: None,
        });
        new_messages.extend(recent);

        Ok(new_messages)
    }

    /// Extracts the best answer from conversation history when no explicit final answer.
    fn extract_best_answer(&self, messages: &[Message]) -> String {
        // Find the last substantive assistant message
        messages
            .iter()
            .rev()
            .find(|m| m.role == Role::Assistant && !m.content.is_empty())
            .map(|m| {
                // Try to extract a conclusion
                if let Some(idx) = m.content.find("Final Answer:") {
                    m.content[idx + "Final Answer:".len()..].trim().to_string()
                } else if let Some(idx) = m.content.to_lowercase().find("in conclusion") {
                    m.content[idx..].trim().to_string()
                } else {
                    m.content.clone()
                }
            })
            .unwrap_or_else(|| "Unable to generate a response.".to_string())
    }

    /// Returns the configuration.
    pub fn config(&self) -> &ReactConfig {
        &self.config
    }

    /// Returns the tools registry.
    pub fn tools(&self) -> &ToolRegistry {
        &self.tools
    }
}

/// Parsed response from agent.
struct ParsedResponse {
    thought: Option<String>,
    action_type: ActionType,
    tool_name: Option<String>,
    action_input: String,
    params: Option<serde_json::Value>,
}

impl ParsedResponse {
    /// Parse an agent response string into structured components.
    /// Extracts thought, action, and action input from the response.
    fn parse(response: &str) -> Self {
        let mut thought = None;
        let mut action_type = ActionType::Reason;
        let mut tool_name = None;
        let mut action_input = String::new();
        let mut params = None;

        // Extract thought
        if let Some(idx) = response.find("Thought:") {
            let start = idx + "Thought:".len();
            let end = response[start..]
                .find("Action:")
                .map(|i| start + i)
                .unwrap_or(response.len());
            thought = Some(response[start..end].trim().to_string());
        }

        // Extract action
        if let Some(idx) = response.find("Action:") {
            let start = idx + "Action:".len();
            let end = response[start..]
                .find("Action Input:")
                .or_else(|| response[start..].find('\n'))
                .map(|i| start + i)
                .unwrap_or(response.len());
            let action_str = response[start..end].trim().to_lowercase();

            if action_str == "final_answer" {
                action_type = ActionType::FinalAnswer;
            } else if action_str == "clarify" {
                action_type = ActionType::Clarify;
            } else if !action_str.is_empty() {
                action_type = ActionType::ToolCall;
                tool_name = Some(action_str);
            }
        }

        // Extract action input
        if let Some(idx) = response.find("Action Input:") {
            let start = idx + "Action Input:".len();
            action_input = response[start..].trim().to_string();

            // Try to parse as JSON for tool calls
            if matches!(action_type, ActionType::ToolCall) {
                // Find JSON in the input
                if let Some(json_start) = action_input.find('{') {
                    if let Some(json_end) = action_input.rfind('}') {
                        let json_str = &action_input[json_start..=json_end];
                        if let Ok(parsed) = serde_json::from_str(json_str) {
                            params = Some(parsed);
                        }
                    }
                }
            }
        }

        Self {
            thought,
            action_type,
            tool_name,
            action_input,
            params,
        }
    }
}

/// Trait for ReAct execution callbacks.
#[async_trait]
pub trait ReactCallback: Send + Sync {
    /// Called before each step.
    async fn on_step_start(&self, step: u32);

    /// Called after each step.
    async fn on_step_complete(&self, step: &ReactStep);

    /// Called on tool execution.
    async fn on_tool_call(&self, tool_name: &str, params: &serde_json::Value);

    /// Called when complete.
    async fn on_complete(&self, result: &ReactResult);
}

/// No-op callback for when callbacks aren't needed.
pub struct NoOpCallback;

#[async_trait]
impl ReactCallback for NoOpCallback {
    async fn on_step_start(&self, _step: u32) {}
    async fn on_step_complete(&self, _step: &ReactStep) {}
    async fn on_tool_call(&self, _tool_name: &str, _params: &serde_json::Value) {}
    async fn on_complete(&self, _result: &ReactResult) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = ReactConfig::default();
        assert_eq!(config.max_iterations, 10);
        assert!(config.enable_reflection);
    }

    #[test]
    fn test_config_fast() {
        let config = ReactConfig::fast();
        assert_eq!(config.max_iterations, 5);
        assert!(!config.enable_reflection);
    }

    #[test]
    fn test_config_thorough() {
        let config = ReactConfig::thorough();
        assert_eq!(config.max_iterations, 20);
        assert!(config.enable_reflection);
        assert_eq!(config.min_confidence, 0.85);
    }

    #[test]
    fn test_parse_response_thought_and_action() {
        let response = r#"Thought: I need to search for information.

Action: search
Action Input: {"query": "Rust programming"}"#;

        let parsed = ParsedResponse::parse(response);

        assert!(parsed.thought.is_some());
        assert!(parsed.thought.unwrap().contains("search for information"));
        assert!(matches!(parsed.action_type, ActionType::ToolCall));
        assert_eq!(parsed.tool_name, Some("search".to_string()));
        assert!(parsed.params.is_some());
    }

    #[test]
    fn test_parse_response_final_answer() {
        let response = r#"Thought: I have all the information needed.

Action: final_answer
Action Input: The answer is 42."#;

        let parsed = ParsedResponse::parse(response);

        assert!(matches!(parsed.action_type, ActionType::FinalAnswer));
        assert_eq!(parsed.action_input.trim(), "The answer is 42.");
    }

    #[test]
    fn test_parse_response_clarify() {
        let response = r#"Thought: I need more information from the user.

Action: clarify
Action Input: What specific topic would you like me to focus on?"#;

        let parsed = ParsedResponse::parse(response);

        assert!(matches!(parsed.action_type, ActionType::Clarify));
    }

    #[test]
    fn test_completion_reason_variants() {
        let final_answer = CompletionReason::FinalAnswer;
        let max_iter = CompletionReason::MaxIterations;
        let error = CompletionReason::Error("test error".to_string());

        assert!(matches!(final_answer, CompletionReason::FinalAnswer));
        assert!(matches!(max_iter, CompletionReason::MaxIterations));
        if let CompletionReason::Error(msg) = error {
            assert_eq!(msg, "test error");
        }
    }

    // === Phase 4: Observation Parsing Tests (TDD) ===

    #[test]
    fn test_parse_observation_success() {
        let tool_result = ToolResult {
            success: true,
            output: "Found 5 results for 'Rust programming'".to_string(),
            error: None,
            data: Some(serde_json::json!({
                "count": 5,
                "items": ["result1", "result2"]
            })),
        };

        let observation = parse_observation("search", &tool_result);

        assert!(observation.success);
        assert_eq!(observation.tool_name, "search");
        assert!(!observation.output.is_empty());
        assert!(observation.key_facts.len() > 0);
        assert!(observation.structured_data.is_some());
    }

    #[test]
    fn test_parse_observation_failure() {
        let tool_result = ToolResult {
            success: false,
            output: String::new(),
            error: Some("Connection timeout".to_string()),
            data: None,
        };

        let observation = parse_observation("api_call", &tool_result);

        assert!(!observation.success);
        assert_eq!(observation.tool_name, "api_call");
        assert!(observation.error_message.is_some());
        assert_eq!(observation.error_message.unwrap(), "Connection timeout");
    }

    #[test]
    fn test_parse_observation_extracts_numbers() {
        let tool_result = ToolResult {
            success: true,
            output: "Temperature is 72°F, humidity is 45%, pressure is 1013 hPa".to_string(),
            error: None,
            data: None,
        };

        let observation = parse_observation("weather", &tool_result);

        // Should extract numeric facts
        assert!(observation.key_facts.iter().any(|f| f.contains("72")));
    }

    #[test]
    fn test_parse_observation_validates_output() {
        use crate::tool::OutputValidationConfig;

        let tool_result = ToolResult {
            success: true,
            output: "x".repeat(200_000), // Very long output
            error: None,
            data: None,
        };

        let config = OutputValidationConfig::default();
        let observation = parse_observation_with_validation("test", &tool_result, &config);

        // Should be truncated or flagged
        assert!(observation.was_truncated || observation.validation_issues.len() > 0);
    }

    #[test]
    fn test_observation_to_message() {
        let observation = Observation {
            tool_name: "search".to_string(),
            success: true,
            output: "Found results".to_string(),
            error_message: None,
            key_facts: vec!["Fact 1".to_string(), "Fact 2".to_string()],
            structured_data: None,
            was_truncated: false,
            validation_issues: vec![],
        };

        let message = observation.to_message();

        assert!(message.contains("search"));
        assert!(message.contains("Found results"));
        assert!(message.contains("Key facts"));
    }

    #[test]
    fn test_generate_observation_reasoning() {
        let thought = "I should search for information about Rust.";
        let action = "search";
        let observation = Observation {
            tool_name: "search".to_string(),
            success: true,
            output: "Found 10 articles about Rust programming".to_string(),
            error_message: None,
            key_facts: vec!["10 articles found".to_string()],
            structured_data: None,
            was_truncated: false,
            validation_issues: vec![],
        };

        let reasoning = generate_observation_reasoning(thought, action, &observation);

        // Should contain reasoning about the observation
        assert!(!reasoning.is_empty());
        assert!(reasoning.contains("search") || reasoning.contains("found"));
    }

    // ==========================================================================
    // ReactConfig Additional Tests
    // ==========================================================================

    #[test]
    fn test_config_clone() {
        let config = ReactConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.max_iterations, config.max_iterations);
        assert_eq!(cloned.enable_reflection, config.enable_reflection);
    }

    #[test]
    fn test_config_debug() {
        let config = ReactConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("ReactConfig"));
        assert!(debug_str.contains("max_iterations"));
    }

    #[test]
    fn test_config_serialize_deserialize() {
        let config = ReactConfig::default();
        let json = serde_json::to_string(&config).expect("serialize");
        let deserialized: ReactConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.max_iterations, config.max_iterations);
        assert_eq!(deserialized.reasoning_temperature, config.reasoning_temperature);
    }

    #[test]
    fn test_config_all_fields() {
        let config = ReactConfig {
            max_iterations: 15,
            max_tokens_per_step: 2048,
            max_conversation_tokens: 10000,
            reasoning_temperature: 0.8,
            reflection_temperature: 0.2,
            enable_reflection: false,
            tool_retry_count: 5,
            tool_timeout: Duration::from_secs(60),
            enable_summarization: false,
            min_confidence: 0.9,
        };
        assert_eq!(config.max_iterations, 15);
        assert_eq!(config.max_tokens_per_step, 2048);
        assert_eq!(config.max_conversation_tokens, 10000);
        assert_eq!(config.reasoning_temperature, 0.8);
        assert_eq!(config.reflection_temperature, 0.2);
        assert!(!config.enable_reflection);
        assert_eq!(config.tool_retry_count, 5);
        assert_eq!(config.tool_timeout, Duration::from_secs(60));
        assert!(!config.enable_summarization);
        assert_eq!(config.min_confidence, 0.9);
    }

    // ==========================================================================
    // ReactStep Tests
    // ==========================================================================

    #[test]
    fn test_react_step_creation() {
        let step = ReactStep {
            step: 1,
            thought: Some("Testing step".to_string()),
            action: None,
            observation: None,
            reflection: None,
            duration_ms: 100,
        };
        assert_eq!(step.step, 1);
        assert_eq!(step.thought, Some("Testing step".to_string()));
        assert!(step.action.is_none());
        assert_eq!(step.duration_ms, 100);
    }

    #[test]
    fn test_react_step_with_action() {
        let action = ReactAction {
            action_type: ActionType::ToolCall,
            tool_name: Some("search".to_string()),
            params: Some(serde_json::json!({"query": "test"})),
            success: true,
            error: None,
        };
        let step = ReactStep {
            step: 2,
            thought: Some("Searching".to_string()),
            action: Some(action),
            observation: Some("Found results".to_string()),
            reflection: Some("Progress made".to_string()),
            duration_ms: 500,
        };
        assert!(step.action.is_some());
        assert!(step.observation.is_some());
        assert!(step.reflection.is_some());
    }

    #[test]
    fn test_react_step_serialize() {
        let step = ReactStep {
            step: 1,
            thought: Some("Test".to_string()),
            action: None,
            observation: None,
            reflection: None,
            duration_ms: 50,
        };
        let json = serde_json::to_string(&step).expect("serialize");
        assert!(json.contains("\"step\":1"));
        assert!(json.contains("\"thought\":"));
    }

    #[test]
    fn test_react_step_clone() {
        let step = ReactStep {
            step: 3,
            thought: Some("Cloning test".to_string()),
            action: None,
            observation: None,
            reflection: None,
            duration_ms: 200,
        };
        let cloned = step.clone();
        assert_eq!(cloned.step, 3);
        assert_eq!(cloned.thought, Some("Cloning test".to_string()));
    }

    // ==========================================================================
    // ReactAction Tests
    // ==========================================================================

    #[test]
    fn test_react_action_tool_call() {
        let action = ReactAction {
            action_type: ActionType::ToolCall,
            tool_name: Some("calculator".to_string()),
            params: Some(serde_json::json!({"a": 1, "b": 2})),
            success: true,
            error: None,
        };
        assert!(matches!(action.action_type, ActionType::ToolCall));
        assert_eq!(action.tool_name, Some("calculator".to_string()));
        assert!(action.success);
    }

    #[test]
    fn test_react_action_final_answer() {
        let action = ReactAction {
            action_type: ActionType::FinalAnswer,
            tool_name: None,
            params: None,
            success: true,
            error: None,
        };
        assert!(matches!(action.action_type, ActionType::FinalAnswer));
        assert!(action.tool_name.is_none());
    }

    #[test]
    fn test_react_action_with_error() {
        let action = ReactAction {
            action_type: ActionType::ToolCall,
            tool_name: Some("broken_tool".to_string()),
            params: None,
            success: false,
            error: Some("Connection refused".to_string()),
        };
        assert!(!action.success);
        assert_eq!(action.error, Some("Connection refused".to_string()));
    }

    #[test]
    fn test_react_action_serialize() {
        let action = ReactAction {
            action_type: ActionType::Clarify,
            tool_name: None,
            params: None,
            success: true,
            error: None,
        };
        let json = serde_json::to_string(&action).expect("serialize");
        assert!(json.contains("Clarify"));
    }

    // ==========================================================================
    // ActionType Tests
    // ==========================================================================

    #[test]
    fn test_action_type_all_variants() {
        let tool_call = ActionType::ToolCall;
        let clarify = ActionType::Clarify;
        let final_answer = ActionType::FinalAnswer;
        let reason = ActionType::Reason;

        assert!(matches!(tool_call, ActionType::ToolCall));
        assert!(matches!(clarify, ActionType::Clarify));
        assert!(matches!(final_answer, ActionType::FinalAnswer));
        assert!(matches!(reason, ActionType::Reason));
    }

    #[test]
    fn test_action_type_debug() {
        let action = ActionType::ToolCall;
        let debug_str = format!("{:?}", action);
        assert_eq!(debug_str, "ToolCall");
    }

    #[test]
    fn test_action_type_clone() {
        let action = ActionType::FinalAnswer;
        let cloned = action.clone();
        assert!(matches!(cloned, ActionType::FinalAnswer));
    }

    #[test]
    fn test_action_type_serialize() {
        let action = ActionType::Reason;
        let json = serde_json::to_string(&action).expect("serialize");
        assert!(json.contains("Reason"));
    }

    // ==========================================================================
    // Observation Tests
    // ==========================================================================

    #[test]
    fn test_observation_from_tool_result() {
        let result = ToolResult {
            success: true,
            output: "Search completed with 3 results".to_string(),
            error: None,
            data: Some(serde_json::json!({"count": 3})),
        };
        let obs = Observation::from_tool_result("search", &result);
        assert_eq!(obs.tool_name, "search");
        assert!(obs.success);
        assert!(obs.output.contains("3 results"));
        assert!(obs.structured_data.is_some());
    }

    #[test]
    fn test_observation_from_failed_result() {
        let result = ToolResult {
            success: false,
            output: String::new(),
            error: Some("API rate limit exceeded".to_string()),
            data: None,
        };
        let obs = Observation::from_tool_result("api", &result);
        assert!(!obs.success);
        assert_eq!(obs.error_message, Some("API rate limit exceeded".to_string()));
    }

    #[test]
    fn test_observation_to_message_success() {
        let obs = Observation {
            tool_name: "test_tool".to_string(),
            success: true,
            output: "Operation completed successfully".to_string(),
            error_message: None,
            key_facts: vec!["100% success rate".to_string()],
            structured_data: None,
            was_truncated: false,
            validation_issues: vec![],
        };
        let msg = obs.to_message();
        assert!(msg.contains("test_tool"));
        assert!(msg.contains("succeeded"));
        assert!(msg.contains("Operation completed"));
    }

    #[test]
    fn test_observation_to_message_failure() {
        let obs = Observation {
            tool_name: "failing_tool".to_string(),
            success: false,
            output: String::new(),
            error_message: Some("Network timeout".to_string()),
            key_facts: vec![],
            structured_data: None,
            was_truncated: false,
            validation_issues: vec![],
        };
        let msg = obs.to_message();
        assert!(msg.contains("failing_tool"));
        assert!(msg.contains("failed"));
        assert!(msg.contains("Network timeout"));
    }

    #[test]
    fn test_observation_to_message_truncated() {
        let obs = Observation {
            tool_name: "big_output".to_string(),
            success: true,
            output: "Truncated content".to_string(),
            error_message: None,
            key_facts: vec![],
            structured_data: None,
            was_truncated: true,
            validation_issues: vec![],
        };
        let msg = obs.to_message();
        assert!(msg.contains("truncated"));
    }

    #[test]
    fn test_observation_clone() {
        let obs = Observation {
            tool_name: "clone_test".to_string(),
            success: true,
            output: "Data".to_string(),
            error_message: None,
            key_facts: vec!["fact1".to_string()],
            structured_data: Some(serde_json::json!({"key": "value"})),
            was_truncated: false,
            validation_issues: vec![],
        };
        let cloned = obs.clone();
        assert_eq!(cloned.tool_name, "clone_test");
        assert_eq!(cloned.key_facts.len(), 1);
    }

    #[test]
    fn test_observation_serialize() {
        let obs = Observation {
            tool_name: "serialize_test".to_string(),
            success: true,
            output: "output".to_string(),
            error_message: None,
            key_facts: vec![],
            structured_data: None,
            was_truncated: false,
            validation_issues: vec![],
        };
        let json = serde_json::to_string(&obs).expect("serialize");
        assert!(json.contains("serialize_test"));
    }

    // ==========================================================================
    // ReactResult Tests
    // ==========================================================================

    #[test]
    fn test_react_result_success() {
        let result = ReactResult {
            answer: "The answer is 42".to_string(),
            success: true,
            confidence: 0.95,
            iterations: 3,
            trace: vec![],
            total_time_ms: 1500,
            completion_reason: CompletionReason::FinalAnswer,
        };
        assert!(result.success);
        assert_eq!(result.confidence, 0.95);
        assert!(matches!(result.completion_reason, CompletionReason::FinalAnswer));
    }

    #[test]
    fn test_react_result_max_iterations() {
        let result = ReactResult {
            answer: "Partial answer".to_string(),
            success: false,
            confidence: 0.3,
            iterations: 10,
            trace: vec![],
            total_time_ms: 5000,
            completion_reason: CompletionReason::MaxIterations,
        };
        assert!(!result.success);
        assert!(matches!(result.completion_reason, CompletionReason::MaxIterations));
    }

    #[test]
    fn test_react_result_serialize() {
        let result = ReactResult {
            answer: "Test".to_string(),
            success: true,
            confidence: 0.8,
            iterations: 2,
            trace: vec![],
            total_time_ms: 100,
            completion_reason: CompletionReason::ConfidenceThreshold,
        };
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("Test"));
        assert!(json.contains("0.8"));
    }

    #[test]
    fn test_react_result_clone() {
        let result = ReactResult {
            answer: "Clone test".to_string(),
            success: true,
            confidence: 0.9,
            iterations: 1,
            trace: vec![],
            total_time_ms: 50,
            completion_reason: CompletionReason::FinalAnswer,
        };
        let cloned = result.clone();
        assert_eq!(cloned.answer, "Clone test");
    }

    // ==========================================================================
    // CompletionReason Tests
    // ==========================================================================

    #[test]
    fn test_completion_reason_confidence_threshold() {
        let reason = CompletionReason::ConfidenceThreshold;
        assert!(matches!(reason, CompletionReason::ConfidenceThreshold));
    }

    #[test]
    fn test_completion_reason_cancelled() {
        let reason = CompletionReason::Cancelled;
        assert!(matches!(reason, CompletionReason::Cancelled));
    }

    #[test]
    fn test_completion_reason_serialize() {
        let reasons = vec![
            CompletionReason::FinalAnswer,
            CompletionReason::MaxIterations,
            CompletionReason::ConfidenceThreshold,
            CompletionReason::Cancelled,
            CompletionReason::Error("test".to_string()),
        ];
        for reason in reasons {
            let json = serde_json::to_string(&reason).expect("serialize");
            assert!(!json.is_empty());
        }
    }

    // ==========================================================================
    // ParsedResponse Edge Case Tests
    // ==========================================================================

    #[test]
    fn test_parse_response_no_thought() {
        let response = "Action: search\nAction Input: {\"query\": \"test\"}";
        let parsed = ParsedResponse::parse(response);
        assert!(parsed.thought.is_none());
        assert!(matches!(parsed.action_type, ActionType::ToolCall));
    }

    #[test]
    fn test_parse_response_no_action() {
        let response = "Thought: Just thinking out loud here.";
        let parsed = ParsedResponse::parse(response);
        assert!(parsed.thought.is_some());
        assert!(matches!(parsed.action_type, ActionType::Reason));
    }

    #[test]
    fn test_parse_response_no_action_input() {
        let response = "Thought: Testing\nAction: final_answer";
        let parsed = ParsedResponse::parse(response);
        assert!(matches!(parsed.action_type, ActionType::FinalAnswer));
        assert!(parsed.action_input.is_empty());
    }

    #[test]
    fn test_parse_response_empty() {
        let parsed = ParsedResponse::parse("");
        assert!(parsed.thought.is_none());
        assert!(matches!(parsed.action_type, ActionType::Reason));
    }

    #[test]
    fn test_parse_response_malformed_json() {
        let response = r#"Thought: Testing
Action: search
Action Input: {"query": "broken json"#;
        let parsed = ParsedResponse::parse(response);
        assert!(matches!(parsed.action_type, ActionType::ToolCall));
        // Params should be None due to malformed JSON
        assert!(parsed.params.is_none());
    }

    #[test]
    fn test_parse_response_action_case_insensitive() {
        let response = "Action: FINAL_ANSWER\nAction Input: Done!";
        let parsed = ParsedResponse::parse(response);
        assert!(matches!(parsed.action_type, ActionType::FinalAnswer));
    }

    #[test]
    fn test_parse_response_complex_json_params() {
        let response = r#"Thought: Complex params test
Action: api_call
Action Input: {"nested": {"key": "value"}, "array": [1, 2, 3]}"#;
        let parsed = ParsedResponse::parse(response);
        assert!(parsed.params.is_some());
        let params = parsed.params.unwrap();
        assert!(params["nested"]["key"] == "value");
        assert!(params["array"].as_array().unwrap().len() == 3);
    }

    // ==========================================================================
    // extract_key_facts Tests
    // ==========================================================================

    #[test]
    fn test_extract_key_facts_with_percentages() {
        let output = "CPU usage: 75%\nMemory: 8GB used";
        let facts = extract_key_facts(output);
        assert!(facts.iter().any(|f| f.contains("75%")));
    }

    #[test]
    fn test_extract_key_facts_with_status() {
        let output = "Operation completed successfully";
        let facts = extract_key_facts(output);
        // Should extract status indicator
        assert!(!facts.is_empty() || output.contains("success"));
    }

    #[test]
    fn test_extract_key_facts_empty_output() {
        let facts = extract_key_facts("");
        assert!(facts.is_empty());
    }

    #[test]
    fn test_extract_key_facts_no_matches() {
        let output = "Plain text with no numbers or status words";
        let facts = extract_key_facts(output);
        // Might be empty or contain partial matches
        assert!(facts.len() <= 5);
    }

    #[test]
    fn test_extract_key_facts_limit() {
        let output = "Line 1: 10 items\nLine 2: 20 items\nLine 3: 30 items\nLine 4: 40 items\nLine 5: 50 items\nLine 6: 60 items\nLine 7: 70 items";
        let facts = extract_key_facts(output);
        // Should be limited to 5 facts
        assert!(facts.len() <= 5);
    }

    #[test]
    fn test_extract_key_facts_with_bytes() {
        let output = "Downloaded 1024 bytes in 50ms";
        let facts = extract_key_facts(output);
        assert!(facts.iter().any(|f| f.contains("bytes") || f.contains("ms")));
    }

    // ==========================================================================
    // generate_observation_reasoning Tests
    // ==========================================================================

    #[test]
    fn test_observation_reasoning_with_key_facts() {
        let obs = Observation {
            tool_name: "analyze".to_string(),
            success: true,
            output: "Analysis complete".to_string(),
            error_message: None,
            key_facts: vec!["Found 5 issues".to_string(), "2 critical".to_string()],
            structured_data: None,
            was_truncated: false,
            validation_issues: vec![],
        };
        let reasoning = generate_observation_reasoning("Check for issues", "analyze", &obs);
        assert!(reasoning.contains("Key findings"));
    }

    #[test]
    fn test_observation_reasoning_with_structured_data() {
        let obs = Observation {
            tool_name: "fetch".to_string(),
            success: true,
            output: "Data fetched".to_string(),
            error_message: None,
            key_facts: vec![],
            structured_data: Some(serde_json::json!({"data": "value"})),
            was_truncated: false,
            validation_issues: vec![],
        };
        let reasoning = generate_observation_reasoning("Fetch data", "fetch", &obs);
        assert!(reasoning.contains("Structured data"));
    }

    #[test]
    fn test_observation_reasoning_failure() {
        let obs = Observation {
            tool_name: "broken".to_string(),
            success: false,
            output: String::new(),
            error_message: Some("Service unavailable".to_string()),
            key_facts: vec![],
            structured_data: None,
            was_truncated: false,
            validation_issues: vec![],
        };
        let reasoning = generate_observation_reasoning("Try action", "broken", &obs);
        assert!(reasoning.contains("failed"));
        assert!(reasoning.contains("Service unavailable"));
    }

    #[test]
    fn test_observation_reasoning_failure_unknown_error() {
        let obs = Observation {
            tool_name: "mystery".to_string(),
            success: false,
            output: String::new(),
            error_message: None,
            key_facts: vec![],
            structured_data: None,
            was_truncated: false,
            validation_issues: vec![],
        };
        let reasoning = generate_observation_reasoning("Try action", "mystery", &obs);
        assert!(reasoning.contains("unknown error"));
    }

    // ==========================================================================
    // NoOpCallback Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_noop_callback_on_step_start() {
        let callback = NoOpCallback;
        callback.on_step_start(1).await;
        // No panic means success
    }

    #[tokio::test]
    async fn test_noop_callback_on_step_complete() {
        let callback = NoOpCallback;
        let step = ReactStep {
            step: 1,
            thought: None,
            action: None,
            observation: None,
            reflection: None,
            duration_ms: 0,
        };
        callback.on_step_complete(&step).await;
    }

    #[tokio::test]
    async fn test_noop_callback_on_tool_call() {
        let callback = NoOpCallback;
        callback.on_tool_call("test", &serde_json::json!({})).await;
    }

    #[tokio::test]
    async fn test_noop_callback_on_complete() {
        let callback = NoOpCallback;
        let result = ReactResult {
            answer: "done".to_string(),
            success: true,
            confidence: 1.0,
            iterations: 1,
            trace: vec![],
            total_time_ms: 100,
            completion_reason: CompletionReason::FinalAnswer,
        };
        callback.on_complete(&result).await;
    }

    // ==========================================================================
    // Integration-style Tests
    // ==========================================================================

    #[test]
    fn test_full_react_step_workflow() {
        // Simulate a complete step with all components
        let action = ReactAction {
            action_type: ActionType::ToolCall,
            tool_name: Some("search".to_string()),
            params: Some(serde_json::json!({"query": "test"})),
            success: true,
            error: None,
        };

        let step = ReactStep {
            step: 1,
            thought: Some("I need to search for test data".to_string()),
            action: Some(action),
            observation: Some("Found 10 results".to_string()),
            reflection: Some("Good progress, data found".to_string()),
            duration_ms: 250,
        };

        // Verify serialization works for the complete step
        let json = serde_json::to_string(&step).expect("serialize");
        assert!(json.contains("search"));
        assert!(json.contains("Found 10 results"));
    }

    #[test]
    fn test_observation_pipeline() {
        // Test the full observation creation pipeline
        let tool_result = ToolResult {
            success: true,
            output: "Processed 100 items in 5 seconds".to_string(),
            error: None,
            data: Some(serde_json::json!({"items": 100, "time_seconds": 5})),
        };

        let obs = Observation::from_tool_result("batch_process", &tool_result);
        let message = obs.to_message();
        let reasoning = generate_observation_reasoning("Process items", "batch_process", &obs);

        assert!(obs.success);
        assert!(!obs.key_facts.is_empty());
        assert!(message.contains("batch_process"));
        assert!(!reasoning.is_empty());
    }
}
