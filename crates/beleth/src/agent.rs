//! Agent implementation.

use std::sync::Arc;

use infernum_core::{Message, ModelId, Result};

use crate::memory::AgentMemory;
use crate::planner::{Planner, PlanningStrategy};
use crate::tool::ToolRegistry;

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
            system: PersonaSource::Inline(
                "You are a helpful AI assistant.".to_string(),
            ),
            model: None,
            max_iterations: 10,
        }
    }
}

/// An autonomous agent.
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
            PersonaSource::Grimoire { persona_id, .. } => {
                // TODO: Load from Grimoire
                format!("Grimoire persona: {}", persona_id)
            }
        }
    }

    /// Runs the agent with the given objective.
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails.
    pub async fn run(&mut self, objective: &str) -> Result<String> {
        tracing::info!(objective, agent_id = %self.id, "Starting agent execution");

        // Add objective to memory
        self.memory.add_message(Message::user(objective));

        // Create plan
        let plan = self.planner.plan(objective, &self.tools).await?;
        tracing::debug!(?plan, "Generated plan");

        // Execute plan
        let mut result = String::new();
        for step in plan.steps {
            tracing::debug!(step = ?step, "Executing step");

            // TODO: Implement actual step execution with tool calling
            result.push_str(&format!("Executed: {}\n", step.description));
        }

        // Add result to memory
        self.memory.add_message(Message::assistant(&result));

        Ok(result)
    }

    /// Adds a message to the agent's memory.
    pub fn add_message(&mut self, message: Message) {
        self.memory.add_message(message);
    }

    /// Clears the agent's working memory.
    pub fn clear_memory(&mut self) {
        self.memory.clear();
    }
}

/// Builder for creating agents.
#[derive(Default)]
pub struct AgentBuilder {
    id: Option<String>,
    persona: Option<Persona>,
    tools: Option<ToolRegistry>,
    planning_strategy: Option<PlanningStrategy>,
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

    /// Builds the agent.
    #[must_use]
    pub fn build(self) -> Agent {
        let strategy = self.planning_strategy.unwrap_or(PlanningStrategy::ReAct {
            max_iterations: 10,
        });

        Agent {
            id: self.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            persona: self.persona.unwrap_or_default(),
            tools: self.tools.unwrap_or_default(),
            memory: AgentMemory::new(),
            planner: Arc::new(crate::planner::DefaultPlanner::new(strategy)),
        }
    }
}
