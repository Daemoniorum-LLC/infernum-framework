//! Agent implementation with ReAct-style reasoning.

use std::sync::Arc;

use futures::StreamExt;
use infernum_core::{GenerateRequest, Message, ModelId, Result, Role, SamplingParams};

use crate::memory::AgentMemory;
use crate::planner::{Planner, PlanningStrategy};
use crate::tool::{ToolCall, ToolContext, ToolRegistry};

use abaddon::{Engine, InferenceEngine};

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
    engine: Option<Arc<Engine>>,
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
    pub fn set_engine(&mut self, engine: Arc<Engine>) {
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
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: objective.to_string(),
                name: None,
                tool_call_id: None,
            },
        ];

        // Create tool context
        let mut ctx = ToolContext::new(&self.id);
        ctx.messages = messages.clone();

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
                .map(|c| c.text.clone())
                .unwrap_or_default();

            tracing::debug!(response = %assistant_response, "Agent response");

            // Add assistant response to messages
            messages.push(Message {
                role: Role::Assistant,
                content: assistant_response.clone(),
                name: None,
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

                    // Execute the tool
                    let result = self.tools.execute(&tool_call, &ctx).await?;

                    // Add observation to messages
                    let observation = if result.success {
                        format!("Observation: {}", result.output)
                    } else {
                        format!(
                            "Observation: Tool error - {}",
                            result.error.unwrap_or_default()
                        )
                    };

                    messages.push(Message {
                        role: Role::User,
                        content: observation.clone(),
                        name: Some("system".to_string()),
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
                .map(|m| m.content.clone())
                .unwrap_or_else(|| "No response generated.".to_string());
        }

        Ok(final_answer)
    }

    /// Builds the system prompt with tool information.
    fn build_system_prompt(&self) -> String {
        let base_prompt = self.system_prompt();
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

    /// Parses the agent's response to extract actions.
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

        // Check for action/tool call
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

    /// Adds a message to the agent's memory.
    pub fn add_message(&mut self, message: Message) {
        self.memory.add_message(message);
    }

    /// Clears the agent's working memory.
    pub fn clear_memory(&mut self) {
        self.memory.clear();
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
            .map(|c| c.text.clone())
            .unwrap_or_default();

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
#[derive(Debug)]
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

/// Builder for creating agents.
#[derive(Default)]
pub struct AgentBuilder {
    id: Option<String>,
    persona: Option<Persona>,
    tools: Option<ToolRegistry>,
    planning_strategy: Option<PlanningStrategy>,
    engine: Option<Arc<Engine>>,
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
    pub fn engine(mut self, engine: Arc<Engine>) -> Self {
        self.engine = Some(engine);
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
}
