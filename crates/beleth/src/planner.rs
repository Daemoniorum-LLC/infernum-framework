//! Planning strategies for agent execution.

use std::sync::Arc;

use async_trait::async_trait;
use infernum_core::{GenerateRequest, Message, Result, Role, SamplingParams};

use abaddon::{Engine, InferenceEngine};
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
    /// Estimated complexity (1-10).
    pub complexity: Option<u8>,
}

impl PlanStep {
    /// Creates a new plan step.
    #[must_use]
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            tool: None,
            params: None,
            dependencies: Vec::new(),
            complexity: None,
        }
    }

    /// Sets the tool for this step.
    #[must_use]
    pub fn with_tool(mut self, tool: impl Into<String>) -> Self {
        self.tool = Some(tool.into());
        self
    }

    /// Sets the parameters for this step.
    #[must_use]
    pub fn with_params(mut self, params: serde_json::Value) -> Self {
        self.params = Some(params);
        self
    }

    /// Adds a dependency.
    #[must_use]
    pub fn depends_on(mut self, step_id: impl Into<String>) -> Self {
        self.dependencies.push(step_id.into());
        self
    }
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
    /// Current step index being executed.
    pub current_step: usize,
    /// Whether the plan is complete.
    pub complete: bool,
}

impl Plan {
    /// Creates a new plan.
    #[must_use]
    pub fn new(objective: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            objective: objective.into(),
            steps: Vec::new(),
            current_step: 0,
            complete: false,
        }
    }

    /// Adds a step to the plan.
    pub fn add_step(&mut self, step: PlanStep) {
        self.steps.push(step);
    }

    /// Returns the next step to execute.
    #[must_use]
    pub fn next_step(&self) -> Option<&PlanStep> {
        self.steps.get(self.current_step)
    }

    /// Advances to the next step.
    pub fn advance(&mut self) {
        if self.current_step < self.steps.len() {
            self.current_step += 1;
        }
        if self.current_step >= self.steps.len() {
            self.complete = true;
        }
    }

    /// Returns remaining steps.
    #[must_use]
    pub fn remaining_steps(&self) -> &[PlanStep] {
        &self.steps[self.current_step..]
    }
}

/// Trait for planners.
#[async_trait]
pub trait Planner: Send + Sync {
    /// Generates a plan for the given objective.
    async fn plan(&self, objective: &str, tools: &ToolRegistry) -> Result<Plan>;

    /// Replans based on feedback.
    async fn replan(&self, plan: &Plan, feedback: &str, tools: &ToolRegistry) -> Result<Plan>;
}

/// Default planner implementation without LLM (fallback).
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
    async fn plan(&self, objective: &str, _tools: &ToolRegistry) -> Result<Plan> {
        tracing::debug!(objective, strategy = ?self.strategy, "Generating plan (fallback mode)");

        // Fallback: Return a simple single-step plan
        let mut plan = Plan::new(objective);
        plan.add_step(PlanStep::new("1", format!("Execute objective: {}", objective)));
        Ok(plan)
    }

    async fn replan(&self, plan: &Plan, _feedback: &str, tools: &ToolRegistry) -> Result<Plan> {
        tracing::debug!(plan_id = %plan.id, "Replanning (fallback mode)");
        self.plan(&plan.objective, tools).await
    }
}

/// LLM-powered planner implementation.
pub struct LLMPlanner {
    engine: Arc<Engine>,
    strategy: PlanningStrategy,
}

impl LLMPlanner {
    /// Creates a new LLM-powered planner.
    #[must_use]
    pub fn new(engine: Arc<Engine>, strategy: PlanningStrategy) -> Self {
        Self { engine, strategy }
    }

    /// Builds the planning prompt based on strategy.
    fn build_planning_prompt(&self, objective: &str, tools: &ToolRegistry) -> String {
        let tools_desc = tools.to_prompt_description();

        match &self.strategy {
            PlanningStrategy::SingleShot => {
                format!(
                    r#"You are a task planning assistant. Create a step-by-step plan to achieve the following objective.

## Objective
{objective}

## Available Tools
{tools_desc}

## Instructions
Create a detailed plan with numbered steps. For each step:
1. Provide a clear description of what needs to be done
2. If a tool should be used, specify the tool name and parameters
3. List any dependencies on previous steps

## Output Format
Respond with a JSON array of steps:
```json
[
  {{
    "id": "1",
    "description": "Step description",
    "tool": "tool_name or null",
    "params": {{}},
    "dependencies": []
  }}
]
```

Generate the plan now:"#
                )
            }
            PlanningStrategy::ReAct { max_iterations } => {
                format!(
                    r#"You are a ReAct-style planning assistant. Create a plan that interleaves reasoning and action.

## Objective
{objective}

## Available Tools
{tools_desc}

## Instructions
Create a plan with at most {max_iterations} steps. For each step, include:
- A thought explaining your reasoning
- An action to take (tool use or final answer)
- Expected observations

## Output Format
Respond with a JSON array of steps:
```json
[
  {{
    "id": "1",
    "description": "Thought: reasoning... Action: tool_name",
    "tool": "tool_name or null",
    "params": {{}},
    "dependencies": []
  }}
]
```

Generate the plan now:"#
                )
            }
            PlanningStrategy::Hierarchical { max_depth } => {
                format!(
                    r#"You are a hierarchical task decomposition planner. Break down the objective into subtasks.

## Objective
{objective}

## Available Tools
{tools_desc}

## Instructions
Decompose the objective into a hierarchy of tasks (max depth: {max_depth}).
- Start with high-level goals
- Break each into concrete, actionable subtasks
- Assign tools where appropriate

## Output Format
Respond with a JSON array of steps (use dependencies to show hierarchy):
```json
[
  {{
    "id": "1",
    "description": "High-level task",
    "tool": null,
    "params": {{}},
    "dependencies": []
  }},
  {{
    "id": "1.1",
    "description": "Subtask",
    "tool": "tool_name",
    "params": {{}},
    "dependencies": ["1"]
  }}
]
```

Generate the plan now:"#
                )
            }
            PlanningStrategy::TreeOfThoughts { breadth, depth } => {
                format!(
                    r#"You are a Tree of Thoughts planner. Explore multiple reasoning paths.

## Objective
{objective}

## Available Tools
{tools_desc}

## Instructions
Generate {breadth} alternative approaches, each with up to {depth} steps.
Evaluate each path and select the most promising one.

## Output Format
First show your thought tree, then output the selected plan:
```json
[
  {{
    "id": "1",
    "description": "Selected approach step",
    "tool": "tool_name or null",
    "params": {{}},
    "dependencies": []
  }}
]
```

Generate the plan now:"#
                )
            }
        }
    }

    /// Builds the replanning prompt.
    fn build_replan_prompt(
        &self,
        plan: &Plan,
        feedback: &str,
        tools: &ToolRegistry,
    ) -> String {
        let tools_desc = tools.to_prompt_description();
        let completed_steps: Vec<_> = plan.steps.iter()
            .take(plan.current_step)
            .map(|s| format!("- [DONE] {}: {}", s.id, s.description))
            .collect();
        let remaining_steps: Vec<_> = plan.steps.iter()
            .skip(plan.current_step)
            .map(|s| format!("- [TODO] {}: {}", s.id, s.description))
            .collect();

        format!(
            r#"You are a task planning assistant. The current plan needs to be revised based on feedback.

## Original Objective
{objective}

## Current Plan Status
### Completed Steps
{completed}

### Remaining Steps
{remaining}

## Feedback
{feedback}

## Available Tools
{tools_desc}

## Instructions
Based on the feedback, revise the remaining steps of the plan. You may:
- Modify existing steps
- Add new steps
- Remove steps that are no longer needed
- Reorder steps if necessary

## Output Format
Respond with a JSON array of the revised remaining steps:
```json
[
  {{
    "id": "new_id",
    "description": "Step description",
    "tool": "tool_name or null",
    "params": {{}},
    "dependencies": []
  }}
]
```

Generate the revised plan now:"#,
            objective = plan.objective,
            completed = if completed_steps.is_empty() { "None".to_string() } else { completed_steps.join("\n") },
            remaining = if remaining_steps.is_empty() { "None".to_string() } else { remaining_steps.join("\n") },
            feedback = feedback,
            tools_desc = tools_desc,
        )
    }

    /// Parses plan steps from LLM response.
    fn parse_plan_steps(&self, response: &str) -> Vec<PlanStep> {
        // Try to extract JSON from the response
        let json_str = self.extract_json(response);

        if let Some(json) = json_str {
            if let Ok(steps) = serde_json::from_str::<Vec<serde_json::Value>>(&json) {
                return steps
                    .into_iter()
                    .filter_map(|v| self.parse_step(&v))
                    .collect();
            }
        }

        // Fallback: try to parse numbered steps from text
        self.parse_text_steps(response)
    }

    /// Extracts JSON from response text.
    fn extract_json(&self, response: &str) -> Option<String> {
        // Look for JSON in code blocks
        if let Some(start) = response.find("```json") {
            let content = &response[start + 7..];
            if let Some(end) = content.find("```") {
                return Some(content[..end].trim().to_string());
            }
        }

        // Look for raw JSON array
        if let Some(start) = response.find('[') {
            let mut depth = 0;
            let mut end = start;
            for (i, c) in response[start..].char_indices() {
                match c {
                    '[' => depth += 1,
                    ']' => {
                        depth -= 1;
                        if depth == 0 {
                            end = start + i + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if depth == 0 {
                return Some(response[start..end].to_string());
            }
        }

        None
    }

    /// Parses a single step from JSON.
    fn parse_step(&self, value: &serde_json::Value) -> Option<PlanStep> {
        let id = value.get("id")?.as_str()?.to_string();
        let description = value.get("description")?.as_str()?.to_string();

        let mut step = PlanStep::new(id, description);

        if let Some(tool) = value.get("tool").and_then(|v| v.as_str()) {
            if !tool.is_empty() && tool != "null" {
                step.tool = Some(tool.to_string());
            }
        }

        if let Some(params) = value.get("params") {
            if !params.is_null() {
                step.params = Some(params.clone());
            }
        }

        if let Some(deps) = value.get("dependencies").and_then(|v| v.as_array()) {
            step.dependencies = deps
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
        }

        if let Some(complexity) = value.get("complexity").and_then(|v| v.as_u64()) {
            step.complexity = Some(complexity.min(10) as u8);
        }

        Some(step)
    }

    /// Fallback parser for text-based step descriptions.
    fn parse_text_steps(&self, response: &str) -> Vec<PlanStep> {
        let mut steps = Vec::new();
        let mut current_id = 1;

        for line in response.lines() {
            let line = line.trim();

            // Match patterns like "1.", "Step 1:", "- Step 1:"
            if line.starts_with(|c: char| c.is_ascii_digit()) ||
               line.starts_with("- ") ||
               line.to_lowercase().starts_with("step ") {

                // Extract description
                let description = line
                    .trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == '-' || c == ':' || c == ' ')
                    .trim_start_matches("step ")
                    .trim_start_matches(|c: char| c.is_ascii_digit())
                    .trim_start_matches(|c: char| c == '.' || c == ':' || c == ' ')
                    .trim();

                if !description.is_empty() {
                    let mut step = PlanStep::new(current_id.to_string(), description);

                    // Try to detect tool usage from common patterns
                    let desc_lower = description.to_lowercase();
                    if desc_lower.contains("calculate") || desc_lower.contains("compute") {
                        step.tool = Some("calculator".to_string());
                    } else if desc_lower.contains("search") || desc_lower.contains("find") {
                        step.tool = Some("search".to_string());
                    } else if desc_lower.contains("read") || desc_lower.contains("fetch") {
                        step.tool = Some("read".to_string());
                    }

                    steps.push(step);
                    current_id += 1;
                }
            }
        }

        // If no steps found, create a single step with the whole response
        if steps.is_empty() && !response.trim().is_empty() {
            steps.push(PlanStep::new("1", response.trim()));
        }

        steps
    }
}

#[async_trait]
impl Planner for LLMPlanner {
    async fn plan(&self, objective: &str, tools: &ToolRegistry) -> Result<Plan> {
        tracing::debug!(objective, strategy = ?self.strategy, "Generating LLM-based plan");

        let prompt = self.build_planning_prompt(objective, tools);

        let messages = vec![
            Message {
                role: Role::System,
                content: "You are an expert task planner. Create detailed, actionable plans.".to_string(),
                name: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: prompt,
                name: None,
                tool_call_id: None,
            },
        ];

        let request = GenerateRequest::chat(messages)
            .with_sampling(SamplingParams::default()
                .with_max_tokens(2048)
                .with_temperature(0.3)); // Lower temperature for more consistent planning

        let response = self.engine.generate(request).await?;
        let response_text = response.choices.first()
            .map(|c| c.text.clone())
            .unwrap_or_default();

        tracing::debug!(response = %response_text, "LLM planning response");

        let steps = self.parse_plan_steps(&response_text);

        let mut plan = Plan::new(objective);
        for step in steps {
            plan.add_step(step);
        }

        // Ensure at least one step
        if plan.steps.is_empty() {
            plan.add_step(PlanStep::new("1", format!("Execute: {}", objective)));
        }

        tracing::info!(
            plan_id = %plan.id,
            steps = plan.steps.len(),
            "Plan generated"
        );

        Ok(plan)
    }

    async fn replan(&self, plan: &Plan, feedback: &str, tools: &ToolRegistry) -> Result<Plan> {
        tracing::debug!(
            plan_id = %plan.id,
            feedback = %feedback,
            "Replanning with LLM"
        );

        let prompt = self.build_replan_prompt(plan, feedback, tools);

        let messages = vec![
            Message {
                role: Role::System,
                content: "You are an expert task planner. Revise plans based on feedback.".to_string(),
                name: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: prompt,
                name: None,
                tool_call_id: None,
            },
        ];

        let request = GenerateRequest::chat(messages)
            .with_sampling(SamplingParams::default()
                .with_max_tokens(2048)
                .with_temperature(0.3));

        let response = self.engine.generate(request).await?;
        let response_text = response.choices.first()
            .map(|c| c.text.clone())
            .unwrap_or_default();

        let new_steps = self.parse_plan_steps(&response_text);

        // Create new plan preserving completed steps
        let mut new_plan = Plan::new(&plan.objective);
        new_plan.id = plan.id.clone(); // Keep same plan ID for tracking

        // Add completed steps
        for step in plan.steps.iter().take(plan.current_step) {
            new_plan.add_step(step.clone());
        }

        // Add revised remaining steps
        for step in new_steps {
            new_plan.add_step(step);
        }

        // Advance to current position
        new_plan.current_step = plan.current_step;

        tracing::info!(
            plan_id = %new_plan.id,
            original_steps = plan.steps.len(),
            new_steps = new_plan.steps.len(),
            "Plan revised"
        );

        Ok(new_plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_step_builder() {
        let step = PlanStep::new("1", "Test step")
            .with_tool("calculator")
            .with_params(serde_json::json!({"expr": "2+2"}))
            .depends_on("0");

        assert_eq!(step.id, "1");
        assert_eq!(step.tool, Some("calculator".to_string()));
        assert_eq!(step.dependencies, vec!["0".to_string()]);
    }

    #[test]
    fn test_plan_navigation() {
        let mut plan = Plan::new("test");
        plan.add_step(PlanStep::new("1", "Step 1"));
        plan.add_step(PlanStep::new("2", "Step 2"));

        assert!(!plan.complete);
        assert_eq!(plan.next_step().unwrap().id, "1");

        plan.advance();
        assert_eq!(plan.next_step().unwrap().id, "2");

        plan.advance();
        assert!(plan.complete);
        assert!(plan.next_step().is_none());
    }

    #[test]
    fn test_parse_json_steps() {
        let strategy = PlanningStrategy::SingleShot;
        // Create a mock engine - in real tests we'd use a mock
        // For now just test the parsing logic
        let response = r#"
        Here's the plan:
        ```json
        [
            {"id": "1", "description": "First step", "tool": "calculator", "params": {"expr": "2+2"}, "dependencies": []},
            {"id": "2", "description": "Second step", "tool": null, "params": null, "dependencies": ["1"]}
        ]
        ```
        "#;

        // Create a default planner just to access parsing methods
        let planner = DefaultPlanner::new(strategy);

        // We can't directly test LLMPlanner parsing without an engine,
        // but we validate the JSON format is correct
        let json_start = response.find("```json").unwrap() + 7;
        let json_end = response[json_start..].find("```").unwrap() + json_start;
        let json_str = &response[json_start..json_end].trim();

        let parsed: Vec<serde_json::Value> = serde_json::from_str(json_str).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0]["id"], "1");
        assert_eq!(parsed[1]["dependencies"][0], "1");
    }
}
