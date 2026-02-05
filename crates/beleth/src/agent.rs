//! Agent implementation with ReAct-style reasoning.

use std::sync::Arc;

use futures::StreamExt;
use infernum_core::{GenerateRequest, Message, ModelId, Result, Role, SamplingParams};

use crate::memory::AgentMemory;
use crate::planner::{Planner, PlanningStrategy};
use crate::tool::{ToolCall, ToolContext, ToolRegistry};

use abaddon::InferenceEngine;

/// Source for agent persona/system prompt.
#[derive(Debug, Clone)]
pub enum PersonaSource {
    /// Inline prompt text.
    Inline(String),
    /// Reference to Grimoire persona.
    Grimoire {
        /// Persona identifier.
        persona_id: String,
        /// Optional variant.
        variant: Option<String>,
    },
}

impl PersonaSource {
    /// Creates a new inline persona source.
    #[must_use]
    pub fn inline(prompt: impl Into<String>) -> Self {
        Self::Inline(prompt.into())
    }

    /// Creates a new Grimoire persona source.
    #[must_use]
    pub fn grimoire(persona_id: impl Into<String>) -> Self {
        Self::Grimoire {
            persona_id: persona_id.into(),
            variant: None,
        }
    }

    /// Creates a new Grimoire persona source with a variant.
    #[must_use]
    pub fn grimoire_with_variant(persona_id: impl Into<String>, variant: impl Into<String>) -> Self {
        Self::Grimoire {
            persona_id: persona_id.into(),
            variant: Some(variant.into()),
        }
    }

    /// Resolves the system prompt asynchronously using GrimoireLoader.
    ///
    /// This is the preferred method as it leverages caching and proper async I/O.
    pub async fn resolve(&self) -> String {
        match self {
            Self::Inline(s) => s.clone(),
            Self::Grimoire { persona_id, variant } => {
                let loader = grimoire_loader::GrimoireLoader::new();

                match loader.load(persona_id).await {
                    Ok(persona) => {
                        // If variant requested, try to get it from the persona's variants
                        if let Some(var) = variant {
                            persona.variants.get(var).cloned().unwrap_or_else(|| {
                                tracing::debug!(
                                    persona_id,
                                    variant = var,
                                    "Variant not found, using base system prompt"
                                );
                                persona.system_prompt.clone()
                            })
                        } else {
                            persona.system_prompt
                        }
                    }
                    Err(_) => {
                        // Try legacy filesystem path loading as fallback
                        Self::load_from_filesystem(persona_id, variant.as_deref())
                    }
                }
            }
        }
    }

    /// Fallback filesystem loading for backwards compatibility.
    fn load_from_filesystem(persona_id: &str, variant: Option<&str>) -> String {
        let base_path = grimoire_loader::default_grimoire_path();
        let prompt_path = if let Some(var) = variant {
            base_path.join(persona_id).join(format!("{}.md", var))
        } else {
            let dir_path = base_path.join(persona_id);
            if dir_path.is_dir() {
                dir_path.join("prompt.md")
            } else {
                base_path.join(format!("{}.md", persona_id))
            }
        };

        match std::fs::read_to_string(&prompt_path) {
            Ok(content) => content,
            Err(_) => {
                tracing::debug!(
                    persona_id,
                    path = %prompt_path.display(),
                    "Grimoire persona not found, using default prompt"
                );
                format!("You are {} - an AI assistant.", persona_id)
            }
        }
    }
}

/// Agent persona configuration.
#[derive(Debug, Clone)]
pub struct Persona {
    /// System prompt source.
    pub system: PersonaSource,
    /// Preferred model.
    pub model: Option<ModelId>,
    /// Maximum iterations.
    pub max_iterations: u32,
}

impl Default for Persona {
    fn default() -> Self {
        Self {
            system: PersonaSource::Inline("You are a helpful AI assistant.".to_string()),
            model: None,
            max_iterations: 10,
        }
    }
}

impl Persona {
    /// Creates a new persona with an inline system prompt.
    #[must_use]
    pub fn inline(prompt: impl Into<String>) -> Self {
        Self {
            system: PersonaSource::inline(prompt),
            ..Default::default()
        }
    }

    /// Creates a new persona from a Grimoire persona ID.
    #[must_use]
    pub fn from_grimoire(persona_id: impl Into<String>) -> Self {
        Self {
            system: PersonaSource::grimoire(persona_id),
            ..Default::default()
        }
    }

    /// Creates a new persona from a Grimoire persona ID with a variant.
    #[must_use]
    pub fn from_grimoire_variant(
        persona_id: impl Into<String>,
        variant: impl Into<String>,
    ) -> Self {
        Self {
            system: PersonaSource::grimoire_with_variant(persona_id, variant),
            ..Default::default()
        }
    }

    /// Sets the preferred model.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<ModelId>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the maximum iterations.
    #[must_use]
    pub fn with_max_iterations(mut self, max: u32) -> Self {
        self.max_iterations = max;
        self
    }

    /// Resolves the system prompt asynchronously.
    ///
    /// This is the preferred method for loading Grimoire personas as it
    /// uses proper async I/O and caching.
    pub async fn resolve_system_prompt(&self) -> String {
        self.system.resolve().await
    }
}

/// An autonomous agent with tool use and reasoning capabilities.
pub struct Agent {
    /// Agent identifier.
    pub id: String,
    /// Agent persona.
    pub persona: Persona,
    /// Available tools.
    pub tools: ToolRegistry,
    /// Agent memory.
    pub memory: AgentMemory,
    /// Planning strategy.
    pub planner: Arc<dyn Planner>,
    /// The inference engine.
    engine: Option<Arc<dyn InferenceEngine>>,
    /// Working directory for file tools (set on ToolContext automatically).
    working_dir: Option<std::path::PathBuf>,
}

impl Agent {
    /// Creates a new agent builder.
    #[must_use]
    pub fn builder() -> AgentBuilder {
        AgentBuilder::default()
    }

    /// Returns the system prompt.
    #[must_use]
    pub fn system_prompt(&self) -> String {
        match &self.persona.system {
            PersonaSource::Inline(s) => s.clone(),
            PersonaSource::Grimoire {
                persona_id,
                variant,
            } => {
                // Try to load from Grimoire filesystem (uses INFERNUM_GRIMOIRE_PATH env or default)
                let base_path = grimoire_loader::default_grimoire_path();
                let prompt_path = if let Some(var) = variant {
                    base_path.join(persona_id).join(format!("{}.md", var))
                } else {
                    let dir_path = base_path.join(persona_id);
                    if dir_path.is_dir() {
                        dir_path.join("prompt.md")
                    } else {
                        base_path.join(format!("{}.md", persona_id))
                    }
                };

                match std::fs::read_to_string(&prompt_path) {
                    Ok(content) => content,
                    Err(_) => {
                        // Provide a helpful fallback with guidance
                        tracing::debug!(
                            persona_id,
                            path = %prompt_path.display(),
                            "Grimoire persona not found, using default prompt"
                        );
                        format!("You are {} - an AI assistant.", persona_id)
                    },
                }
            },
        }
    }

    /// Sets the inference engine.
    pub fn set_engine(&mut self, engine: Arc<dyn InferenceEngine>) {
        self.engine = Some(engine);
    }

    /// Runs the agent with the given objective using ReAct-style reasoning.
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails.
    pub async fn run(&mut self, objective: &str) -> Result<String> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| infernum_core::Error::internal("No engine configured for agent"))?;

        tracing::info!(objective, agent_id = %self.id, "Starting agent execution");

        // Build system prompt with tool information
        let system_prompt = self.build_system_prompt();

        // Initialize conversation with system and user objective
        let mut messages = vec![
            Message {
                role: Role::System,
                content: system_prompt,
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: objective.to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        // Create tool context
        let mut ctx = ToolContext::new(&self.id);
        ctx.messages = messages.clone();
        if let Some(ref wd) = self.working_dir {
            ctx.set_state("working_dir", serde_json::json!(wd.to_string_lossy().as_ref()));
        }

        // ReAct loop
        let mut final_answer = String::new();
        for iteration in 0..self.persona.max_iterations {
            tracing::debug!(iteration, "ReAct iteration");

            // Generate response
            let request = GenerateRequest::chat(messages.clone()).with_sampling(
                SamplingParams::default()
                    .with_max_tokens(1024)
                    .with_temperature(0.7),
            );

            let response = engine.generate(request).await?;
            let assistant_response = response
                .choices
                .first()
                .map_or_else(String::new, |c| c.text.clone());

            tracing::debug!(response = %assistant_response, "Agent response");

            // Add assistant response to messages
            messages.push(Message {
                role: Role::Assistant,
                content: assistant_response.clone(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            });

            // Parse the response for actions
            let action = self.parse_action(&assistant_response);

            match action {
                AgentAction::Thought(thought) => {
                    tracing::debug!(thought, "Agent thinking");
                    // Continue to next iteration
                },
                AgentAction::ToolCall(tool_call) => {
                    tracing::info!(tool = %tool_call.name, "Executing tool");

                    // Execute the tool — catch errors as observations instead of crashing
                    let observation = match self.tools.execute(&tool_call, &ctx).await {
                        Ok(result) if result.success => {
                            format!("Observation: {}", result.output)
                        }
                        Ok(result) => {
                            format!(
                                "Observation: Tool error - {}",
                                result.error.unwrap_or_default()
                            )
                        }
                        Err(e) => {
                            tracing::warn!(tool = %tool_call.name, error = %e, "Tool execution failed");
                            format!("Observation: Tool execution failed - {}", e)
                        }
                    };

                    messages.push(Message {
                        role: Role::User,
                        content: observation.clone(),
                        name: Some("system".to_string()),
                        tool_calls: None,
                        tool_call_id: None,
                    });

                    tracing::debug!(observation = %observation, "Tool result");
                },
                AgentAction::FinalAnswer(answer) => {
                    tracing::info!("Agent reached final answer");
                    final_answer = answer;
                    break;
                },
                AgentAction::Continue => {
                    // No specific action, continue
                },
            }

            // Update context
            ctx.messages = messages.clone();
        }

        // Store conversation in memory
        for msg in &messages {
            self.memory.add_message(msg.clone());
        }

        if final_answer.is_empty() {
            // If no explicit final answer, use the last assistant response
            final_answer = messages
                .iter()
                .rev()
                .find(|m| m.role == Role::Assistant)
                .map_or_else(|| "No response generated.".to_string(), |m| m.content.clone());
        }

        Ok(final_answer)
    }

    /// Returns the model family based on the persona's model ID.
    fn model_family(&self) -> infernum_core::ModelFamily {
        self.persona
            .model
            .as_ref()
            .map(|m| infernum_core::ModelFamily::from_model_name(&m.0))
            .unwrap_or_default()
    }

    /// Builds the system prompt with tool information.
    ///
    /// Uses model-native format when the model supports it (per TOOL-CALLING-SPEC §5.6),
    /// falling back to generic Action: / Action Input: format for unknown models.
    fn build_system_prompt(&self) -> String {
        let base_prompt = self.system_prompt();

        match self.model_family() {
            infernum_core::ModelFamily::Qwen => {
                let tools_desc = self.tools.to_qwen_native_description();
                format!(
                    "{}\n\n{}\n\n\
                    When you have the final answer, respond with:\n\
                    Final Answer: <your_answer>\n\n\
                    Always think step by step.",
                    base_prompt, tools_desc
                )
            }
            _ => {
                let tools_desc = self.tools.to_prompt_description();
                format!(
                    "{}\n\n## Tools\n\n{}\n\n## Instructions\n\n\
                    When you need to use a tool, respond with:\n\
                    Action: <tool_name>\n\
                    Action Input: <json_parameters>\n\n\
                    After receiving the observation, continue reasoning.\n\n\
                    When you have the final answer, respond with:\n\
                    Final Answer: <your_answer>\n\n\
                    Always think step by step. Use Thought: to express your reasoning.",
                    base_prompt, tools_desc
                )
            }
        }
    }

    /// Parses the agent's response to extract actions.
    ///
    /// Detects both model-native `<tool_call>` tags (§5.6) and generic
    /// `Action: / Action Input:` format for backwards compatibility.
    fn parse_action(&self, response: &str) -> AgentAction {
        let response = response.trim();

        // Check for final answer
        if let Some(answer) = response.strip_prefix("Final Answer:").or_else(|| {
            response
                .lines()
                .find(|line| line.trim().starts_with("Final Answer:"))
                .and_then(|line| line.strip_prefix("Final Answer:"))
        }) {
            return AgentAction::FinalAnswer(answer.trim().to_string());
        }

        // Check for native <tool_call> tags (§5.6 — Qwen, Llama, etc.)
        if let Some(call) = self.parse_native_tool_call(response) {
            return AgentAction::ToolCall(call);
        }

        // Check for generic Action: / Action Input: format
        let mut action_name = None;
        let mut action_input = None;

        for line in response.lines() {
            let line = line.trim();
            if let Some(name) = line.strip_prefix("Action:") {
                action_name = Some(name.trim().to_string());
            } else if let Some(input) = line.strip_prefix("Action Input:") {
                action_input = Some(input.trim().to_string());
            }
        }

        // Also check for JSON block action input
        if action_input.is_none() && action_name.is_some() {
            // Look for JSON in the response
            if let Some(json_start) = response.find('{') {
                if let Some(json_end) = response.rfind('}') {
                    action_input = Some(response[json_start..=json_end].to_string());
                }
            }
        }

        if let (Some(name), Some(input)) = (action_name, action_input) {
            // Parse the JSON input
            let params = serde_json::from_str(&input).unwrap_or(serde_json::json!({}));
            return AgentAction::ToolCall(ToolCall { name, params });
        }

        // Check for thought
        if let Some(thought) = response.strip_prefix("Thought:").or_else(|| {
            response
                .lines()
                .find(|line| line.trim().starts_with("Thought:"))
                .and_then(|line| line.strip_prefix("Thought:"))
        }) {
            return AgentAction::Thought(thought.trim().to_string());
        }

        AgentAction::Continue
    }

    /// Parse a native `<tool_call>` tag from model output.
    ///
    /// Returns the first tool call found within `<tool_call></tool_call>` tags.
    /// Takes `&self` for method dispatch even though not currently needed —
    /// future model-family-specific parsing variants may use it.
    #[allow(clippy::unused_self)]
    fn parse_native_tool_call(&self, response: &str) -> Option<ToolCall> {
        let start_tag = "<tool_call>";
        let end_tag = "</tool_call>";

        let start = response.find(start_tag)?;
        let content_start = start + start_tag.len();
        let end = response[content_start..].find(end_tag)?;
        let json_str = response[content_start..content_start + end].trim();

        let parsed: serde_json::Value = serde_json::from_str(json_str).ok()?;
        let name = parsed.get("name")?.as_str()?.to_string();
        let params = parsed.get("arguments").cloned().unwrap_or(serde_json::json!({}));

        Some(ToolCall { name, params })
    }

    /// Adds a message to the agent's memory.
    pub fn add_message(&mut self, message: Message) {
        self.memory.add_message(message);
    }

    /// Clears the agent's working memory.
    pub fn clear_memory(&mut self) {
        self.memory.clear();
    }

    /// Runs the agent with a pre-generated plan.
    ///
    /// Executes each step of the plan in order, handling tool calls
    /// and collecting observations.
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails.
    pub async fn run_with_plan(
        &mut self,
        mut plan: crate::planner::Plan,
    ) -> Result<PlanExecutionResult> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| infernum_core::Error::internal("No engine configured for agent"))?;

        tracing::info!(
            plan_id = %plan.id,
            objective = %plan.objective,
            steps = plan.steps.len(),
            "Starting plan execution"
        );

        let mut ctx = ToolContext::new(&self.id);
        if let Some(ref wd) = self.working_dir {
            ctx.set_state("working_dir", serde_json::json!(wd.to_string_lossy().as_ref()));
        }
        let mut step_results = Vec::new();
        let mut final_output = String::new();

        while let Some(step) = plan.next_step() {
            tracing::debug!(
                step_id = %step.id,
                description = %step.description,
                "Executing plan step"
            );

            let step_result = if let Some(tool_name) = &step.tool {
                // Execute the tool specified in the plan
                let params = step.params.clone().unwrap_or(serde_json::json!({}));
                let tool_call = ToolCall {
                    name: tool_name.clone(),
                    params,
                };

                let result = self.tools.execute(&tool_call, &ctx).await?;
                let observation = if result.success {
                    result.output.clone()
                } else {
                    format!("Error: {}", result.error.unwrap_or_default())
                };

                // Add observation to context messages
                ctx.messages.push(Message {
                    role: Role::User,
                    content: format!("Step {}: {}\nResult: {}", step.id, step.description, observation),
                    name: Some("system".to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                });

                PlanStepResult {
                    step_id: step.id.clone(),
                    success: result.success,
                    output: observation,
                    tool_used: Some(tool_name.clone()),
                }
            } else {
                // No tool - generate reasoning with LLM
                let messages = vec![
                    Message {
                        role: Role::System,
                        content: self.build_system_prompt(),
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                    },
                    Message {
                        role: Role::User,
                        content: format!(
                            "Execute step {} of the plan: {}\n\nContext:\n{}",
                            step.id,
                            step.description,
                            ctx.messages
                                .iter()
                                .map(|m| m.content.as_str())
                                .collect::<Vec<_>>()
                                .join("\n---\n")
                        ),
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                    },
                ];

                let request = GenerateRequest::chat(messages).with_sampling(
                    SamplingParams::default()
                        .with_max_tokens(1024)
                        .with_temperature(0.7),
                );

                let response = engine.generate(request).await?;
                let output = response
                    .choices
                    .first()
                .map_or_else(String::new, |c| c.text.clone());

                ctx.messages.push(Message {
                    role: Role::Assistant,
                    content: output.clone(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                });

                PlanStepResult {
                    step_id: step.id.clone(),
                    success: true,
                    output,
                    tool_used: None,
                }
            };

            final_output = step_result.output.clone();
            step_results.push(step_result);
            plan.advance();
        }

        // Store in memory
        for msg in &ctx.messages {
            self.memory.add_message(msg.clone());
        }

        Ok(PlanExecutionResult {
            plan_id: plan.id,
            steps_executed: step_results.len(),
            step_results,
            final_output,
            success: plan.complete,
        })
    }

    /// Generates a plan for the given objective using the configured planner.
    ///
    /// # Errors
    ///
    /// Returns an error if planning fails.
    pub async fn generate_plan(&self, objective: &str) -> Result<crate::planner::Plan> {
        self.planner.plan(objective, &self.tools).await
    }

    /// Replans based on feedback.
    ///
    /// # Errors
    ///
    /// Returns an error if replanning fails.
    pub async fn replan(
        &self,
        plan: &crate::planner::Plan,
        feedback: &str,
    ) -> Result<crate::planner::Plan> {
        self.planner.replan(plan, feedback, &self.tools).await
    }

    /// Runs a single step of reasoning (for streaming/interactive use).
    pub async fn step(&mut self, input: &str) -> Result<StepResult> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| infernum_core::Error::internal("No engine configured for agent"))?;

        // Add user input to memory
        self.memory.add_message(Message::user(input));

        // Build messages from memory
        let mut messages = vec![Message {
            role: Role::System,
            content: self.build_system_prompt(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        messages.extend(self.memory.messages().iter().cloned());

        // Generate response
        let request = GenerateRequest::chat(messages).with_sampling(
            SamplingParams::default()
                .with_max_tokens(1024)
                .with_temperature(0.7),
        );

        let response = engine.generate(request).await?;
        let assistant_response = response
            .choices
            .first()
                .map_or_else(String::new, |c| c.text.clone());

        // Add to memory
        self.memory
            .add_message(Message::assistant(&assistant_response));

        // Parse action
        let action = self.parse_action(&assistant_response);

        Ok(StepResult {
            response: assistant_response,
            action,
            usage: StepUsage {
                prompt_tokens: response.usage.prompt_tokens,
                completion_tokens: response.usage.completion_tokens,
            },
        })
    }

    /// Runs a single step with streaming output.
    pub async fn step_streaming(
        &mut self,
        input: &str,
    ) -> Result<impl futures::Stream<Item = Result<String>>> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| infernum_core::Error::internal("No engine configured for agent"))?
            .clone();

        // Add user input to memory
        self.memory.add_message(Message::user(input));

        // Build messages from memory
        let mut messages = vec![Message {
            role: Role::System,
            content: self.build_system_prompt(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        messages.extend(self.memory.messages().iter().cloned());

        // Generate streaming response
        let request = GenerateRequest::chat(messages).with_sampling(
            SamplingParams::default()
                .with_max_tokens(1024)
                .with_temperature(0.7),
        );

        let token_stream = engine.generate_stream(request).await?;

        Ok(token_stream.map(|result| {
            result.map(|chunk| {
                chunk
                    .choices
                    .first()
                    .and_then(|c| c.delta.content.clone())
                    .unwrap_or_default()
            })
        }))
    }
}

/// Result from a single agent step.
#[derive(Debug)]
pub struct StepResult {
    /// The full response text.
    pub response: String,
    /// The parsed action.
    pub action: AgentAction,
    /// Token usage.
    pub usage: StepUsage,
}

/// Token usage for a step.
#[derive(Debug)]
pub struct StepUsage {
    /// Prompt tokens.
    pub prompt_tokens: u32,
    /// Completion tokens.
    pub completion_tokens: u32,
}

/// Action parsed from agent response.
#[derive(Debug, Clone)]
pub enum AgentAction {
    /// Agent is thinking/reasoning.
    Thought(String),
    /// Agent wants to call a tool.
    ToolCall(ToolCall),
    /// Agent has reached a final answer.
    FinalAnswer(String),
    /// No specific action, continue.
    Continue,
}

/// Result of executing a single plan step.
#[derive(Debug, Clone)]
pub struct PlanStepResult {
    /// The step ID.
    pub step_id: String,
    /// Whether the step succeeded.
    pub success: bool,
    /// Output from the step.
    pub output: String,
    /// Tool used (if any).
    pub tool_used: Option<String>,
}

/// Result of executing a complete plan.
#[derive(Debug)]
pub struct PlanExecutionResult {
    /// Plan ID that was executed.
    pub plan_id: String,
    /// Number of steps executed.
    pub steps_executed: usize,
    /// Results from each step.
    pub step_results: Vec<PlanStepResult>,
    /// Final output from the plan.
    pub final_output: String,
    /// Whether the plan completed successfully.
    pub success: bool,
}

/// Builder for creating agents.
#[derive(Default)]
pub struct AgentBuilder {
    id: Option<String>,
    persona: Option<Persona>,
    tools: Option<ToolRegistry>,
    planning_strategy: Option<PlanningStrategy>,
    engine: Option<Arc<dyn InferenceEngine>>,
    working_dir: Option<std::path::PathBuf>,
}

impl AgentBuilder {
    /// Sets the agent ID.
    #[must_use]
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Sets the persona from an inline prompt.
    #[must_use]
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        let mut persona = self.persona.unwrap_or_default();
        persona.system = PersonaSource::Inline(prompt.into());
        self.persona = Some(persona);
        self
    }

    /// Sets the persona from a Grimoire reference.
    #[must_use]
    pub fn grimoire_persona(mut self, persona_id: impl Into<String>) -> Self {
        let mut persona = self.persona.unwrap_or_default();
        persona.system = PersonaSource::Grimoire {
            persona_id: persona_id.into(),
            variant: None,
        };
        self.persona = Some(persona);
        self
    }

    /// Sets the preferred model.
    #[must_use]
    pub fn model(mut self, model: impl Into<ModelId>) -> Self {
        let mut persona = self.persona.unwrap_or_default();
        persona.model = Some(model.into());
        self.persona = Some(persona);
        self
    }

    /// Sets the maximum iterations.
    #[must_use]
    pub fn max_iterations(mut self, max: u32) -> Self {
        let mut persona = self.persona.unwrap_or_default();
        persona.max_iterations = max;
        self.persona = Some(persona);
        self
    }

    /// Sets the tool registry.
    #[must_use]
    pub fn tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Sets the planning strategy.
    #[must_use]
    pub fn planning_strategy(mut self, strategy: PlanningStrategy) -> Self {
        self.planning_strategy = Some(strategy);
        self
    }

    /// Sets the inference engine.
    #[must_use]
    pub fn engine(mut self, engine: Arc<dyn InferenceEngine>) -> Self {
        self.engine = Some(engine);
        self
    }

    /// Sets the working directory for file tools.
    ///
    /// This directory is set on the `ToolContext` automatically during
    /// agent execution. File tools validate that all paths stay within
    /// this boundary.
    #[must_use]
    pub fn working_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    /// Builds the agent.
    #[must_use]
    pub fn build(self) -> Agent {
        let strategy = self
            .planning_strategy
            .unwrap_or(PlanningStrategy::ReAct { max_iterations: 10 });

        Agent {
            id: self.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            persona: self.persona.unwrap_or_default(),
            tools: self.tools.unwrap_or_default(),
            memory: AgentMemory::new(),
            planner: Arc::new(crate::planner::DefaultPlanner::new(strategy)),
            engine: self.engine,
            working_dir: self.working_dir,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_action_final_answer() {
        let agent = Agent::builder().build();
        let response = "Thought: I've calculated the result.\nFinal Answer: The answer is 42.";

        match agent.parse_action(response) {
            AgentAction::FinalAnswer(answer) => {
                assert_eq!(answer, "The answer is 42.");
            },
            _ => panic!("Expected FinalAnswer"),
        }
    }

    #[test]
    fn test_parse_action_final_answer_multiline() {
        let agent = Agent::builder().build();
        // Parser extracts content from the Final Answer: line forward
        let response = r#"Thought: After considering all factors, I can now provide the final answer.

Final Answer: The solution involves three steps"#;

        match agent.parse_action(response) {
            AgentAction::FinalAnswer(answer) => {
                assert!(answer.contains("three steps"));
            },
            _ => panic!("Expected FinalAnswer"),
        }
    }

    #[test]
    fn test_parse_action_tool_call() {
        let agent = Agent::builder().build();
        let response = "Thought: I need to calculate something.\nAction: calculator\nAction Input: {\"expression\": \"2+2\"}";

        match agent.parse_action(response) {
            AgentAction::ToolCall(call) => {
                assert_eq!(call.name, "calculator");
                assert_eq!(call.params["expression"], "2+2");
            },
            _ => panic!("Expected ToolCall"),
        }
    }

    #[test]
    fn test_parse_action_tool_call_with_json_in_response() {
        let agent = Agent::builder().build();
        // When Action Input is empty, parser looks for JSON block in response
        let response = r#"Thought: I need to search for information.
Action: search
{"query": "Rust programming", "max_results": 5}"#;

        match agent.parse_action(response) {
            AgentAction::ToolCall(call) => {
                assert_eq!(call.name, "search");
                assert_eq!(call.params["query"], "Rust programming");
                assert_eq!(call.params["max_results"], 5);
            },
            _ => panic!("Expected ToolCall"),
        }
    }

    #[test]
    fn test_parse_action_thought() {
        let agent = Agent::builder().build();
        let response = "Thought: Let me think about this problem.";

        match agent.parse_action(response) {
            AgentAction::Thought(thought) => {
                assert_eq!(thought, "Let me think about this problem.");
            },
            _ => panic!("Expected Thought"),
        }
    }

    #[test]
    fn test_parse_action_continue() {
        let agent = Agent::builder().build();
        let response = "I'm processing the information...";

        match agent.parse_action(response) {
            AgentAction::Continue => {},
            _ => panic!("Expected Continue"),
        }
    }

    #[test]
    fn test_agent_builder_defaults() {
        let agent = Agent::builder().build();

        assert!(!agent.id.is_empty());
        assert_eq!(agent.persona.max_iterations, 10);
    }

    #[test]
    fn test_agent_builder_custom() {
        let agent = Agent::builder()
            .id("test-agent")
            .system_prompt("You are a test agent.")
            .max_iterations(5)
            .build();

        assert_eq!(agent.id, "test-agent");
        assert_eq!(agent.persona.max_iterations, 5);
        assert_eq!(agent.system_prompt(), "You are a test agent.");
    }

    #[test]
    fn test_agent_builder_grimoire_persona() {
        let agent = Agent::builder()
            .grimoire_persona("test-persona")
            .build();

        match &agent.persona.system {
            PersonaSource::Grimoire { persona_id, variant } => {
                assert_eq!(persona_id, "test-persona");
                assert!(variant.is_none());
            },
            _ => panic!("Expected Grimoire persona"),
        }
    }

    #[test]
    fn test_build_system_prompt() {
        let agent = Agent::builder()
            .system_prompt("You are a helpful assistant.")
            .build();

        let prompt = agent.build_system_prompt();

        assert!(prompt.contains("You are a helpful assistant."));
        assert!(prompt.contains("## Tools"));
        assert!(prompt.contains("## Instructions"));
        assert!(prompt.contains("Final Answer:"));
    }

    #[test]
    fn test_agent_memory_operations() {
        let mut agent = Agent::builder().build();

        assert!(agent.memory.messages().is_empty());

        agent.add_message(Message::user("Hello"));
        assert_eq!(agent.memory.messages().len(), 1);

        agent.add_message(Message::assistant("Hi there!"));
        assert_eq!(agent.memory.messages().len(), 2);

        agent.clear_memory();
        assert!(agent.memory.messages().is_empty());
    }

    #[test]
    fn test_persona_default() {
        let persona = Persona::default();

        assert_eq!(persona.max_iterations, 10);
        assert!(persona.model.is_none());
        match persona.system {
            PersonaSource::Inline(s) => {
                assert!(s.contains("helpful AI assistant"));
            },
            _ => panic!("Expected inline persona"),
        }
    }

    #[test]
    fn test_plan_step_result() {
        let result = PlanStepResult {
            step_id: "1".to_string(),
            success: true,
            output: "Step completed".to_string(),
            tool_used: Some("calculator".to_string()),
        };

        assert!(result.success);
        assert_eq!(result.tool_used, Some("calculator".to_string()));
    }

    #[test]
    fn test_agent_action_clone() {
        let action = AgentAction::ToolCall(ToolCall {
            name: "test".to_string(),
            params: serde_json::json!({"key": "value"}),
        });

        let cloned = action.clone();
        match cloned {
            AgentAction::ToolCall(call) => {
                assert_eq!(call.name, "test");
            },
            _ => panic!("Expected ToolCall"),
        }
    }

    // ==========================================================================
    // PersonaSource Tests
    // ==========================================================================

    #[test]
    fn test_persona_source_inline() {
        let source = PersonaSource::Inline("Custom prompt".to_string());
        match source {
            PersonaSource::Inline(s) => assert_eq!(s, "Custom prompt"),
            _ => panic!("Expected Inline"),
        }
    }

    #[test]
    fn test_persona_source_grimoire() {
        let source = PersonaSource::Grimoire {
            persona_id: "assistant".to_string(),
            variant: Some("friendly".to_string()),
        };
        match source {
            PersonaSource::Grimoire { persona_id, variant } => {
                assert_eq!(persona_id, "assistant");
                assert_eq!(variant, Some("friendly".to_string()));
            },
            _ => panic!("Expected Grimoire"),
        }
    }

    #[test]
    fn test_persona_source_grimoire_no_variant() {
        let source = PersonaSource::Grimoire {
            persona_id: "default".to_string(),
            variant: None,
        };
        match source {
            PersonaSource::Grimoire { persona_id, variant } => {
                assert_eq!(persona_id, "default");
                assert!(variant.is_none());
            },
            _ => panic!("Expected Grimoire"),
        }
    }

    #[test]
    fn test_persona_source_debug() {
        let source = PersonaSource::Inline("test".to_string());
        let debug_str = format!("{:?}", source);
        assert!(debug_str.contains("Inline"));
    }

    #[test]
    fn test_persona_source_clone() {
        let source = PersonaSource::Grimoire {
            persona_id: "test".to_string(),
            variant: None,
        };
        let cloned = source.clone();
        match cloned {
            PersonaSource::Grimoire { persona_id, .. } => {
                assert_eq!(persona_id, "test");
            },
            _ => panic!("Expected Grimoire"),
        }
    }

    // ==========================================================================
    // Persona Tests
    // ==========================================================================

    #[test]
    fn test_persona_with_all_fields() {
        let persona = Persona {
            system: PersonaSource::Inline("Expert assistant".to_string()),
            model: Some("gpt-4".into()),
            max_iterations: 15,
        };
        assert_eq!(persona.max_iterations, 15);
        assert!(persona.model.is_some());
    }

    #[test]
    fn test_persona_debug() {
        let persona = Persona::default();
        let debug_str = format!("{:?}", persona);
        assert!(debug_str.contains("Persona"));
        assert!(debug_str.contains("max_iterations"));
    }

    #[test]
    fn test_persona_clone() {
        let persona = Persona {
            system: PersonaSource::Inline("Clone test".to_string()),
            model: None,
            max_iterations: 5,
        };
        let cloned = persona.clone();
        assert_eq!(cloned.max_iterations, 5);
    }

    // ==========================================================================
    // AgentAction Tests
    // ==========================================================================

    #[test]
    fn test_agent_action_all_variants() {
        let thought = AgentAction::Thought("thinking".to_string());
        let tool = AgentAction::ToolCall(ToolCall {
            name: "test".to_string(),
            params: serde_json::json!({}),
        });
        let answer = AgentAction::FinalAnswer("done".to_string());
        let cont = AgentAction::Continue;

        assert!(matches!(thought, AgentAction::Thought(_)));
        assert!(matches!(tool, AgentAction::ToolCall(_)));
        assert!(matches!(answer, AgentAction::FinalAnswer(_)));
        assert!(matches!(cont, AgentAction::Continue));
    }

    #[test]
    fn test_agent_action_debug() {
        let action = AgentAction::Thought("debug test".to_string());
        let debug_str = format!("{:?}", action);
        assert!(debug_str.contains("Thought"));
        assert!(debug_str.contains("debug test"));
    }

    #[test]
    fn test_agent_action_clone_final_answer() {
        let action = AgentAction::FinalAnswer("The answer".to_string());
        let cloned = action.clone();
        match cloned {
            AgentAction::FinalAnswer(s) => assert_eq!(s, "The answer"),
            _ => panic!("Expected FinalAnswer"),
        }
    }

    #[test]
    fn test_agent_action_clone_continue() {
        let action = AgentAction::Continue;
        let cloned = action.clone();
        assert!(matches!(cloned, AgentAction::Continue));
    }

    // ==========================================================================
    // PlanStepResult Tests
    // ==========================================================================

    #[test]
    fn test_plan_step_result_all_fields() {
        let result = PlanStepResult {
            step_id: "step-1".to_string(),
            success: false,
            output: "Error occurred".to_string(),
            tool_used: None,
        };
        assert_eq!(result.step_id, "step-1");
        assert!(!result.success);
        assert!(result.tool_used.is_none());
    }

    #[test]
    fn test_plan_step_result_debug() {
        let result = PlanStepResult {
            step_id: "1".to_string(),
            success: true,
            output: "done".to_string(),
            tool_used: Some("calculator".to_string()),
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("PlanStepResult"));
        assert!(debug_str.contains("calculator"));
    }

    #[test]
    fn test_plan_step_result_clone() {
        let result = PlanStepResult {
            step_id: "clone-test".to_string(),
            success: true,
            output: "output".to_string(),
            tool_used: None,
        };
        let cloned = result.clone();
        assert_eq!(cloned.step_id, "clone-test");
    }

    // ==========================================================================
    // PlanExecutionResult Tests
    // ==========================================================================

    #[test]
    fn test_plan_execution_result_creation() {
        let result = PlanExecutionResult {
            plan_id: "plan-123".to_string(),
            steps_executed: 3,
            step_results: vec![
                PlanStepResult {
                    step_id: "1".to_string(),
                    success: true,
                    output: "step 1".to_string(),
                    tool_used: None,
                },
                PlanStepResult {
                    step_id: "2".to_string(),
                    success: true,
                    output: "step 2".to_string(),
                    tool_used: Some("search".to_string()),
                },
            ],
            final_output: "Complete".to_string(),
            success: true,
        };
        assert_eq!(result.plan_id, "plan-123");
        assert_eq!(result.steps_executed, 3);
        assert_eq!(result.step_results.len(), 2);
        assert!(result.success);
    }

    #[test]
    fn test_plan_execution_result_debug() {
        let result = PlanExecutionResult {
            plan_id: "debug-plan".to_string(),
            steps_executed: 1,
            step_results: vec![],
            final_output: "done".to_string(),
            success: true,
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("PlanExecutionResult"));
        assert!(debug_str.contains("debug-plan"));
    }

    // ==========================================================================
    // StepResult Tests
    // ==========================================================================

    #[test]
    fn test_step_result_creation() {
        let result = StepResult {
            response: "Agent response".to_string(),
            action: AgentAction::Continue,
            usage: StepUsage {
                prompt_tokens: 100,
                completion_tokens: 50,
            },
        };
        assert_eq!(result.response, "Agent response");
        assert!(matches!(result.action, AgentAction::Continue));
    }

    #[test]
    fn test_step_result_debug() {
        let result = StepResult {
            response: "test".to_string(),
            action: AgentAction::FinalAnswer("answer".to_string()),
            usage: StepUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
            },
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("StepResult"));
    }

    // ==========================================================================
    // StepUsage Tests
    // ==========================================================================

    #[test]
    fn test_step_usage_creation() {
        let usage = StepUsage {
            prompt_tokens: 500,
            completion_tokens: 200,
        };
        assert_eq!(usage.prompt_tokens, 500);
        assert_eq!(usage.completion_tokens, 200);
    }

    #[test]
    fn test_step_usage_debug() {
        let usage = StepUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
        };
        let debug_str = format!("{:?}", usage);
        assert!(debug_str.contains("StepUsage"));
        assert!(debug_str.contains("100"));
    }

    // ==========================================================================
    // AgentBuilder Additional Tests
    // ==========================================================================

    #[test]
    fn test_agent_builder_with_tools() {
        use crate::tool::ToolRegistry;

        let registry = ToolRegistry::with_builtins();
        let agent = Agent::builder()
            .tools(registry)
            .build();

        assert!(agent.tools.len() >= 3);
    }

    #[test]
    fn test_agent_builder_with_planning_strategy() {
        let agent = Agent::builder()
            .planning_strategy(PlanningStrategy::Hierarchical {
                max_depth: 3,
            })
            .build();

        // Agent should be built successfully with custom strategy
        assert!(!agent.id.is_empty());
    }

    #[test]
    fn test_agent_builder_with_model() {
        let agent = Agent::builder()
            .model("gpt-4-turbo")
            .build();

        assert_eq!(agent.persona.model, Some("gpt-4-turbo".into()));
    }

    #[test]
    fn test_agent_builder_chain_all() {
        let agent = Agent::builder()
            .id("full-agent")
            .system_prompt("Full test agent")
            .model("claude-3")
            .max_iterations(20)
            .tools(ToolRegistry::new())
            .planning_strategy(PlanningStrategy::SingleShot)
            .build();

        assert_eq!(agent.id, "full-agent");
        assert_eq!(agent.persona.max_iterations, 20);
        assert!(agent.persona.model.is_some());
    }

    // ==========================================================================
    // Agent Additional Tests
    // ==========================================================================

    #[test]
    fn test_agent_system_prompt_inline() {
        let agent = Agent::builder()
            .system_prompt("Custom inline prompt")
            .build();

        assert_eq!(agent.system_prompt(), "Custom inline prompt");
    }

    #[test]
    fn test_agent_system_prompt_grimoire_fallback() {
        // When Grimoire persona file doesn't exist, it should use a fallback
        let agent = Agent::builder()
            .grimoire_persona("nonexistent-persona")
            .build();

        let prompt = agent.system_prompt();
        // Should contain the persona_id in the fallback message
        assert!(prompt.contains("nonexistent-persona"));
    }

    // ==========================================================================
    // parse_action Edge Cases
    // ==========================================================================

    #[test]
    fn test_parse_action_empty_response() {
        let agent = Agent::builder().build();
        let action = agent.parse_action("");

        assert!(matches!(action, AgentAction::Continue));
    }

    #[test]
    fn test_parse_action_whitespace_only() {
        let agent = Agent::builder().build();
        let action = agent.parse_action("   \n\t  ");

        assert!(matches!(action, AgentAction::Continue));
    }

    #[test]
    fn test_parse_action_final_answer_at_start() {
        let agent = Agent::builder().build();
        let response = "Final Answer: Direct answer at the start";

        match agent.parse_action(response) {
            AgentAction::FinalAnswer(answer) => {
                assert_eq!(answer, "Direct answer at the start");
            },
            _ => panic!("Expected FinalAnswer"),
        }
    }

    #[test]
    fn test_parse_action_thought_in_middle() {
        let agent = Agent::builder().build();
        let response = "Some preamble\nThought: The actual thought\nMore text";

        match agent.parse_action(response) {
            AgentAction::Thought(thought) => {
                assert_eq!(thought, "The actual thought");
            },
            _ => panic!("Expected Thought"),
        }
    }

    #[test]
    fn test_parse_action_tool_call_empty_json() {
        let agent = Agent::builder().build();
        let response = "Action: empty_tool\nAction Input: {}";

        match agent.parse_action(response) {
            AgentAction::ToolCall(call) => {
                assert_eq!(call.name, "empty_tool");
                assert_eq!(call.params, serde_json::json!({}));
            },
            _ => panic!("Expected ToolCall"),
        }
    }

    #[test]
    fn test_parse_action_tool_call_invalid_json() {
        let agent = Agent::builder().build();
        let response = "Action: bad_json_tool\nAction Input: not valid json at all";

        match agent.parse_action(response) {
            AgentAction::ToolCall(call) => {
                assert_eq!(call.name, "bad_json_tool");
                // Invalid JSON should result in empty object
                assert_eq!(call.params, serde_json::json!({}));
            },
            _ => panic!("Expected ToolCall"),
        }
    }

    #[test]
    fn test_parse_action_prefers_final_answer() {
        let agent = Agent::builder().build();
        // When both Final Answer and Action are present, Final Answer should take precedence
        let response = "Thought: Done\nFinal Answer: Complete\nAction: should_not_run";

        match agent.parse_action(response) {
            AgentAction::FinalAnswer(answer) => {
                assert_eq!(answer, "Complete");
            },
            _ => panic!("Expected FinalAnswer to take precedence"),
        }
    }

    // ==========================================================================
    // Agent Memory Tests
    // ==========================================================================

    #[test]
    fn test_agent_add_multiple_messages() {
        let mut agent = Agent::builder().build();

        agent.add_message(Message::user("First"));
        agent.add_message(Message::assistant("Response 1"));
        agent.add_message(Message::user("Second"));
        agent.add_message(Message::assistant("Response 2"));

        assert_eq!(agent.memory.messages().len(), 4);
    }

    #[test]
    fn test_agent_clear_memory_is_complete() {
        let mut agent = Agent::builder().build();

        agent.add_message(Message::user("Test 1"));
        agent.add_message(Message::user("Test 2"));
        assert_eq!(agent.memory.messages().len(), 2);

        agent.clear_memory();
        assert!(agent.memory.messages().is_empty());

        // Should be able to add messages after clearing
        agent.add_message(Message::user("New message"));
        assert_eq!(agent.memory.messages().len(), 1);
    }

    // ==========================================================================
    // Build System Prompt Tests
    // ==========================================================================

    #[test]
    fn test_build_system_prompt_includes_tools() {
        use crate::tool::ToolRegistry;

        let registry = ToolRegistry::with_builtins();
        let agent = Agent::builder()
            .system_prompt("Base prompt")
            .tools(registry)
            .build();

        let prompt = agent.build_system_prompt();

        assert!(prompt.contains("Base prompt"));
        assert!(prompt.contains("## Tools"));
        assert!(prompt.contains("calculator"));
    }

    #[test]
    fn test_build_system_prompt_includes_instructions() {
        let agent = Agent::builder().build();
        let prompt = agent.build_system_prompt();

        assert!(prompt.contains("Action:"));
        assert!(prompt.contains("Action Input:"));
        assert!(prompt.contains("Final Answer:"));
        assert!(prompt.contains("Thought:"));
    }

    // ==========================================================================
    // AgentBuilder Default Tests
    // ==========================================================================

    #[test]
    fn test_agent_builder_default() {
        let builder = AgentBuilder::default();
        let agent = builder.build();

        // Default agent should have auto-generated ID
        assert!(!agent.id.is_empty());
        // Default persona with 10 iterations
        assert_eq!(agent.persona.max_iterations, 10);
        // Empty tools registry
        assert!(agent.tools.is_empty());
    }

    #[test]
    fn test_agent_builder_partial() {
        let agent = Agent::builder()
            .id("partial-agent")
            .build();

        assert_eq!(agent.id, "partial-agent");
        // Other fields should use defaults
        assert_eq!(agent.persona.max_iterations, 10);
    }

    // ==========================================================================
    // PersonaSource Builder Tests
    // ==========================================================================

    #[test]
    fn test_persona_source_inline_builder() {
        let source = PersonaSource::inline("Custom prompt");
        match source {
            PersonaSource::Inline(s) => assert_eq!(s, "Custom prompt"),
            _ => panic!("Expected Inline"),
        }
    }

    #[test]
    fn test_persona_source_grimoire_builder() {
        let source = PersonaSource::grimoire("code-reviewer");
        match source {
            PersonaSource::Grimoire { persona_id, variant } => {
                assert_eq!(persona_id, "code-reviewer");
                assert!(variant.is_none());
            },
            _ => panic!("Expected Grimoire"),
        }
    }

    #[test]
    fn test_persona_source_grimoire_with_variant_builder() {
        let source = PersonaSource::grimoire_with_variant("assistant", "friendly");
        match source {
            PersonaSource::Grimoire { persona_id, variant } => {
                assert_eq!(persona_id, "assistant");
                assert_eq!(variant, Some("friendly".to_string()));
            },
            _ => panic!("Expected Grimoire"),
        }
    }

    #[tokio::test]
    async fn test_persona_source_resolve_inline() {
        let source = PersonaSource::inline("Test prompt for resolution");
        let resolved = source.resolve().await;
        assert_eq!(resolved, "Test prompt for resolution");
    }

    #[tokio::test]
    async fn test_persona_source_resolve_grimoire_fallback() {
        // Non-existent persona should fall back to default message
        let source = PersonaSource::grimoire("nonexistent-persona-xyz");
        let resolved = source.resolve().await;
        assert!(resolved.contains("nonexistent-persona-xyz"));
    }

    // ==========================================================================
    // Persona Builder Tests
    // ==========================================================================

    #[test]
    fn test_persona_inline_constructor() {
        let persona = Persona::inline("Expert coding assistant");
        match &persona.system {
            PersonaSource::Inline(s) => assert_eq!(s, "Expert coding assistant"),
            _ => panic!("Expected Inline source"),
        }
        assert_eq!(persona.max_iterations, 10); // default
    }

    #[test]
    fn test_persona_from_grimoire() {
        let persona = Persona::from_grimoire("code-reviewer");
        match &persona.system {
            PersonaSource::Grimoire { persona_id, variant } => {
                assert_eq!(persona_id, "code-reviewer");
                assert!(variant.is_none());
            },
            _ => panic!("Expected Grimoire source"),
        }
    }

    #[test]
    fn test_persona_from_grimoire_variant() {
        let persona = Persona::from_grimoire_variant("assistant", "concise");
        match &persona.system {
            PersonaSource::Grimoire { persona_id, variant } => {
                assert_eq!(persona_id, "assistant");
                assert_eq!(variant.as_deref(), Some("concise"));
            },
            _ => panic!("Expected Grimoire source"),
        }
    }

    #[test]
    fn test_persona_builder_pattern() {
        let persona = Persona::from_grimoire("expert")
            .with_model("gpt-4")
            .with_max_iterations(20);

        assert_eq!(persona.model, Some("gpt-4".into()));
        assert_eq!(persona.max_iterations, 20);
    }

    #[tokio::test]
    async fn test_persona_resolve_system_prompt_inline() {
        let persona = Persona::inline("Resolved inline prompt");
        let prompt = persona.resolve_system_prompt().await;
        assert_eq!(prompt, "Resolved inline prompt");
    }

    #[tokio::test]
    async fn test_persona_resolve_system_prompt_grimoire_fallback() {
        let persona = Persona::from_grimoire("nonexistent-test-persona");
        let prompt = persona.resolve_system_prompt().await;
        // Should contain the persona_id in the fallback message
        assert!(prompt.contains("nonexistent-test-persona"));
    }

    // =========================================================================
    // Spec §5.6: Beleth Agent Native Format Tests
    //
    // These tests validate TOOL-CALLING-SPEC.md §5.6. When the agent's model
    // is Qwen, it should use native tool calling format instead of generic
    // Action: / Action Input: text protocol.
    // =========================================================================

    #[test]
    fn test_build_system_prompt_qwen_uses_native_format() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(crate::tool::CalculatorTool));

        let agent = Agent::builder()
            .model("Qwen/Qwen2.5-7B-Instruct")
            .tools(registry)
            .build();

        let prompt = agent.build_system_prompt();

        // §5.6: Qwen model must use <tools> XML tags with JSON definitions
        assert!(
            prompt.contains("<tools>"),
            "Qwen agent should use <tools> tag in system prompt"
        );
        assert!(
            prompt.contains("</tools>"),
            "Qwen agent should close <tools> tag"
        );
        assert!(
            prompt.contains("\"type\":\"function\""),
            "Qwen agent should use JSON function definitions"
        );

        // §5.6: Must use native preamble
        assert!(
            prompt.contains("You may call one or more functions"),
            "Qwen agent should use native preamble"
        );

        // §5.6: Must NOT use generic Action: format
        assert!(
            !prompt.contains("Action: <tool_name>"),
            "Qwen agent must not use generic Action: format"
        );
    }

    #[test]
    fn test_build_system_prompt_unknown_model_uses_generic_format() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(crate::tool::CalculatorTool));

        let agent = Agent::builder()
            .tools(registry)
            .build();

        let prompt = agent.build_system_prompt();

        // Unknown model (no model set) should fall back to generic format
        assert!(
            prompt.contains("Action: <tool_name>"),
            "Unknown model should use generic Action: format"
        );
    }

    #[test]
    fn test_parse_action_detects_native_tool_call_tags() {
        let agent = Agent::builder()
            .model("Qwen/Qwen2.5-7B-Instruct")
            .build();

        let response = r#"<tool_call>
{"name": "calculator", "arguments": {"expression": "2+2"}}
</tool_call>"#;

        match agent.parse_action(response) {
            AgentAction::ToolCall(call) => {
                assert_eq!(call.name, "calculator");
                assert_eq!(call.params["expression"], "2+2");
            }
            other => panic!("Expected ToolCall from native format, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_action_detects_native_with_text_before() {
        let agent = Agent::builder()
            .model("Qwen/Qwen2.5-7B-Instruct")
            .build();

        let response = r#"I'll calculate that for you.
<tool_call>
{"name": "calculator", "arguments": {"expression": "15*7"}}
</tool_call>"#;

        match agent.parse_action(response) {
            AgentAction::ToolCall(call) => {
                assert_eq!(call.name, "calculator");
                assert_eq!(call.params["expression"], "15*7");
            }
            other => panic!("Expected ToolCall, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_action_generic_still_works_with_qwen_model() {
        // §5.6: Backwards compatibility — generic format should still parse
        let agent = Agent::builder()
            .model("Qwen/Qwen2.5-7B-Instruct")
            .build();

        let response = "Thought: I need to calculate.\nAction: calculator\nAction Input: {\"expression\": \"3+3\"}";

        match agent.parse_action(response) {
            AgentAction::ToolCall(call) => {
                assert_eq!(call.name, "calculator");
            }
            other => panic!("Expected ToolCall from generic format, got {:?}", other),
        }
    }
}
