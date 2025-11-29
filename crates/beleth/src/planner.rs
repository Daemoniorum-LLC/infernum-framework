//! Planning strategies for agent execution.

use async_trait::async_trait;
use infernum_core::Result;

use crate::tool::ToolRegistry;

/// Strategy for planning.
#[derive(Debug, Clone)]
pub enum PlanningStrategy {
    /// Single-shot planning.
    SingleShot,
    /// ReAct-style interleaved reasoning and acting.
    ReAct {
        /// Maximum iterations.
        max_iterations: u32,
    },
    /// Tree of Thoughts with evaluation.
    TreeOfThoughts {
        /// Breadth of tree.
        breadth: u32,
        /// Depth of tree.
        depth: u32,
    },
    /// Hierarchical task decomposition.
    Hierarchical {
        /// Maximum decomposition depth.
        max_depth: u32,
    },
}

/// A step in a plan.
#[derive(Debug, Clone)]
pub struct PlanStep {
    /// Step identifier.
    pub id: String,
    /// Step description.
    pub description: String,
    /// Tool to use (if any).
    pub tool: Option<String>,
    /// Tool parameters.
    pub params: Option<serde_json::Value>,
    /// Dependencies on other steps.
    pub dependencies: Vec<String>,
}

/// A plan for achieving an objective.
#[derive(Debug, Clone)]
pub struct Plan {
    /// Plan identifier.
    pub id: String,
    /// Objective being achieved.
    pub objective: String,
    /// Steps in the plan.
    pub steps: Vec<PlanStep>,
}

/// Trait for planners.
#[async_trait]
pub trait Planner: Send + Sync {
    /// Generates a plan for the given objective.
    async fn plan(&self, objective: &str, tools: &ToolRegistry) -> Result<Plan>;

    /// Replans based on feedback.
    async fn replan(&self, plan: &Plan, feedback: &str, tools: &ToolRegistry) -> Result<Plan>;
}

/// Default planner implementation.
pub struct DefaultPlanner {
    strategy: PlanningStrategy,
}

impl DefaultPlanner {
    /// Creates a new planner with the given strategy.
    #[must_use]
    pub fn new(strategy: PlanningStrategy) -> Self {
        Self { strategy }
    }
}

#[async_trait]
impl Planner for DefaultPlanner {
    async fn plan(&self, objective: &str, tools: &ToolRegistry) -> Result<Plan> {
        tracing::debug!(objective, strategy = ?self.strategy, "Generating plan");

        // TODO: Implement actual planning with LLM
        // For now, return a simple single-step plan
        let step = PlanStep {
            id: "1".to_string(),
            description: format!("Execute objective: {}", objective),
            tool: None,
            params: None,
            dependencies: Vec::new(),
        };

        Ok(Plan {
            id: uuid::Uuid::new_v4().to_string(),
            objective: objective.to_string(),
            steps: vec![step],
        })
    }

    async fn replan(&self, plan: &Plan, feedback: &str, tools: &ToolRegistry) -> Result<Plan> {
        tracing::debug!(plan_id = %plan.id, feedback, "Replanning");

        // TODO: Implement actual replanning
        self.plan(&plan.objective, tools).await
    }
}
