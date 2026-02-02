//! Planning strategies for agent execution.

use std::sync::Arc;

use async_trait::async_trait;
use infernum_core::{GenerateRequest, Message, Result, Role, SamplingParams};

use crate::tool::ToolRegistry;
use abaddon::{Engine, InferenceEngine};

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
        plan.add_step(PlanStep::new(
            "1",
            format!("Execute objective: {}", objective),
        ));
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
            },
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
            },
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
            },
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
            },
        }
    }

    /// Builds the replanning prompt.
    fn build_replan_prompt(&self, plan: &Plan, feedback: &str, tools: &ToolRegistry) -> String {
        let tools_desc = tools.to_prompt_description();
        let completed_steps: Vec<_> = plan
            .steps
            .iter()
            .take(plan.current_step)
            .map(|s| format!("- [DONE] {}: {}", s.id, s.description))
            .collect();
        let remaining_steps: Vec<_> = plan
            .steps
            .iter()
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
            completed = if completed_steps.is_empty() {
                "None".to_string()
            } else {
                completed_steps.join("\n")
            },
            remaining = if remaining_steps.is_empty() {
                "None".to_string()
            } else {
                remaining_steps.join("\n")
            },
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
                    },
                    _ => {},
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
            if line.starts_with(|c: char| c.is_ascii_digit())
                || line.starts_with("- ")
                || line.to_lowercase().starts_with("step ")
            {
                // Extract description
                let description = line
                    .trim_start_matches(|c: char| {
                        c.is_ascii_digit() || c == '.' || c == '-' || c == ':' || c == ' '
                    })
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

    /// Implements Tree of Thoughts planning with full traversal.
    async fn plan_with_tot(
        &self,
        objective: &str,
        tools: &ToolRegistry,
        breadth: u32,
        depth: u32,
    ) -> Result<Plan> {
        tracing::info!(
            objective = %objective,
            breadth = breadth,
            depth = depth,
            "Starting Tree of Thoughts traversal"
        );

        let tools_desc = tools.to_prompt_description();

        // Step 1: Generate initial thought branches
        let branches = self.generate_thought_branches(objective, &tools_desc, breadth).await?;

        if branches.is_empty() {
            tracing::warn!("No initial branches generated, falling back to single plan");
            let mut plan = Plan::new(objective);
            plan.add_step(PlanStep::new("1", format!("Execute: {}", objective)));
            return Ok(plan);
        }

        tracing::debug!(branches = branches.len(), "Generated initial thought branches");

        // Step 2: Expand and evaluate each branch
        let mut evaluated_branches = Vec::new();
        for (i, branch) in branches.into_iter().enumerate() {
            let expanded = self.expand_branch(&branch, objective, &tools_desc, depth).await?;
            let score = self.evaluate_branch(&expanded, objective).await?;

            tracing::debug!(
                branch_id = i,
                score = score,
                steps = expanded.len(),
                "Evaluated branch"
            );

            evaluated_branches.push(EvaluatedBranch {
                thoughts: expanded,
                score,
            });
        }

        // Step 3: Select the best branch
        let best_branch = evaluated_branches
            .into_iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| infernum_core::Error::internal("No branches to evaluate"))?;

        tracing::info!(
            score = best_branch.score,
            thoughts = best_branch.thoughts.len(),
            "Selected best branch"
        );

        // Step 4: Convert thoughts to plan steps
        let mut plan = Plan::new(objective);
        for (i, thought) in best_branch.thoughts.into_iter().enumerate() {
            let mut step = PlanStep::new((i + 1).to_string(), &thought.content);
            step.tool = thought.suggested_tool;
            step.params = thought.tool_params;
            plan.add_step(step);
        }

        // Ensure at least one step
        if plan.steps.is_empty() {
            plan.add_step(PlanStep::new("1", format!("Execute: {}", objective)));
        }

        Ok(plan)
    }

    /// Generates initial thought branches for the objective.
    async fn generate_thought_branches(
        &self,
        objective: &str,
        tools_desc: &str,
        breadth: u32,
    ) -> Result<Vec<ThoughtNode>> {
        let prompt = format!(
            r#"You are exploring different approaches to solve a problem using Tree of Thoughts.

## Objective
{objective}

## Available Tools
{tools_desc}

## Instructions
Generate {breadth} DISTINCT approaches to solve this problem. Each approach should:
1. Start with a different strategy or perspective
2. Be clearly different from the other approaches
3. Consider the available tools

## Output Format
Respond with a JSON array of thoughts:
```json
[
  {{
    "approach": "Brief name for this approach",
    "reasoning": "Why this approach might work",
    "first_step": "What to do first in this approach",
    "tool": "suggested_tool or null"
  }}
]
```

Generate {breadth} distinct approaches now:"#
        );

        let messages = vec![
            Message {
                role: Role::System,
                content: "You are an expert problem solver exploring multiple solution paths.".to_string(),
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

        let request = GenerateRequest::chat(messages).with_sampling(
            SamplingParams::default()
                .with_max_tokens(1024)
                .with_temperature(0.8), // Higher temperature for diversity
        );

        let response = self.engine.generate(request).await?;
        let response_text = response
            .choices
            .first()
            .map(|c| c.text.clone())
            .unwrap_or_default();

        // Parse branches from response
        let mut branches = Vec::new();
        if let Some(json_str) = self.extract_json(&response_text) {
            if let Ok(parsed) = serde_json::from_str::<Vec<serde_json::Value>>(&json_str) {
                for value in parsed {
                    let approach = value.get("approach").and_then(|v| v.as_str()).unwrap_or("Unknown");
                    let reasoning = value.get("reasoning").and_then(|v| v.as_str()).unwrap_or("");
                    let first_step = value.get("first_step").and_then(|v| v.as_str()).unwrap_or("");
                    let tool = value.get("tool").and_then(|v| v.as_str()).filter(|s| *s != "null");

                    branches.push(ThoughtNode {
                        content: format!("{}: {}", approach, first_step),
                        reasoning: reasoning.to_string(),
                        suggested_tool: tool.map(String::from),
                        tool_params: None,
                        depth: 0,
                        children: Vec::new(),
                    });
                }
            }
        }

        // Fallback: create simple branches
        if branches.is_empty() {
            branches.push(ThoughtNode {
                content: format!("Approach 1: {}", objective),
                reasoning: "Direct approach".to_string(),
                suggested_tool: None,
                tool_params: None,
                depth: 0,
                children: Vec::new(),
            });
        }

        Ok(branches)
    }

    /// Expands a thought branch to the specified depth.
    async fn expand_branch(
        &self,
        initial_thought: &ThoughtNode,
        objective: &str,
        tools_desc: &str,
        max_depth: u32,
    ) -> Result<Vec<ThoughtNode>> {
        let mut thoughts = vec![initial_thought.clone()];
        let mut current_context = initial_thought.content.clone();

        for depth in 1..max_depth {
            let prompt = format!(
                r#"You are continuing a line of reasoning using Tree of Thoughts.

## Objective
{objective}

## Current Progress
{current_context}

## Available Tools
{tools_desc}

## Instructions
What is the next logical step in this approach? Consider:
1. What has been accomplished so far
2. What tools might help
3. How to make progress toward the objective

## Output Format
```json
{{
  "next_step": "What to do next",
  "reasoning": "Why this is the right next step",
  "tool": "suggested_tool or null",
  "params": {{}}
}}
```

Generate the next step:"#
            );

            let messages = vec![
                Message {
                    role: Role::System,
                    content: "You are following a specific solution path step by step.".to_string(),
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

            let request = GenerateRequest::chat(messages).with_sampling(
                SamplingParams::default()
                    .with_max_tokens(512)
                    .with_temperature(0.5),
            );

            let response = self.engine.generate(request).await?;
            let response_text = response
                .choices
                .first()
                .map(|c| c.text.clone())
                .unwrap_or_default();

            // Parse the next step
            if let Some(json_str) = self.extract_json(&response_text) {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json_str) {
                    let next_step = parsed.get("next_step").and_then(|v| v.as_str()).unwrap_or("Continue");
                    let reasoning = parsed.get("reasoning").and_then(|v| v.as_str()).unwrap_or("");
                    let tool = parsed.get("tool").and_then(|v| v.as_str()).filter(|s| *s != "null");
                    let params = parsed.get("params").cloned();

                    let thought = ThoughtNode {
                        content: next_step.to_string(),
                        reasoning: reasoning.to_string(),
                        suggested_tool: tool.map(String::from),
                        tool_params: params,
                        depth: depth as usize,
                        children: Vec::new(),
                    };

                    current_context = format!("{}\n{}: {}", current_context, depth, next_step);
                    thoughts.push(thought);
                }
            }
        }

        Ok(thoughts)
    }

    /// Evaluates a branch and returns a score (0.0 - 1.0).
    async fn evaluate_branch(&self, thoughts: &[ThoughtNode], objective: &str) -> Result<f32> {
        if thoughts.is_empty() {
            return Ok(0.0);
        }

        let thought_summary: String = thoughts
            .iter()
            .map(|t| format!("- {}", t.content))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            r#"You are evaluating a solution approach for effectiveness.

## Objective
{objective}

## Proposed Approach
{thought_summary}

## Evaluation Criteria
1. Completeness: Does it address the full objective?
2. Feasibility: Can each step be executed?
3. Efficiency: Is this a good use of available tools?
4. Clarity: Are the steps clear and actionable?

## Output Format
Respond with a JSON object:
```json
{{
  "score": 0.75,
  "rationale": "Brief explanation of the score"
}}
```

Score should be between 0.0 (poor) and 1.0 (excellent).

Evaluate this approach:"#
        );

        let messages = vec![
            Message {
                role: Role::System,
                content: "You are an objective evaluator of solution approaches.".to_string(),
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

        let request = GenerateRequest::chat(messages).with_sampling(
            SamplingParams::default()
                .with_max_tokens(256)
                .with_temperature(0.2), // Low temperature for consistent evaluation
        );

        let response = self.engine.generate(request).await?;
        let response_text = response
            .choices
            .first()
            .map(|c| c.text.clone())
            .unwrap_or_default();

        // Parse score from response
        if let Some(json_str) = self.extract_json(&response_text) {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json_str) {
                if let Some(score) = parsed.get("score").and_then(|v| v.as_f64()) {
                    return Ok((score as f32).clamp(0.0, 1.0));
                }
            }
        }

        // Fallback: heuristic score based on number of concrete steps
        let heuristic = (thoughts.len() as f32 / 5.0).min(1.0);
        Ok(heuristic)
    }

    /// Implements hierarchical task decomposition planning.
    async fn plan_with_hierarchical(
        &self,
        objective: &str,
        tools: &ToolRegistry,
        max_depth: u32,
    ) -> Result<Plan> {
        tracing::info!(
            objective = %objective,
            max_depth = max_depth,
            "Starting hierarchical task decomposition"
        );

        let tools_desc = tools.to_prompt_description();

        // Step 1: Initial decomposition of the objective
        let root_task = self.decompose_task(
            "1",
            objective,
            &tools_desc,
            0,
            max_depth as usize,
        ).await?;

        tracing::debug!(
            total_tasks = root_task.total_tasks(),
            "Initial decomposition complete"
        );

        // Step 2: Recursively decompose complex subtasks
        let refined_task = self.refine_decomposition(
            root_task,
            &tools_desc,
            max_depth as usize,
        ).await?;

        tracing::info!(
            total_tasks = refined_task.total_tasks(),
            "Hierarchical decomposition complete"
        );

        // Step 3: Flatten to plan
        let mut plan = Plan::new(objective);
        for step in refined_task.flatten() {
            plan.add_step(step);
        }

        // Ensure at least one step
        if plan.steps.is_empty() {
            plan.add_step(PlanStep::new("1", format!("Execute: {}", objective)));
        }

        Ok(plan)
    }

    /// Decomposes a task into subtasks using the LLM.
    async fn decompose_task(
        &self,
        task_id: &str,
        task_description: &str,
        tools_desc: &str,
        current_depth: usize,
        max_depth: usize,
    ) -> Result<HierarchicalTask> {
        // If at max depth, return atomic task
        if current_depth >= max_depth {
            return Ok(self.create_atomic_task(task_id, task_description, tools_desc).await?);
        }

        let prompt = format!(
            r#"You are a hierarchical task decomposition expert. Break down complex tasks into manageable subtasks.

## Task to Decompose
ID: {task_id}
Description: {task_description}

## Available Tools
{tools_desc}

## Instructions
Analyze this task and determine if it should be:
1. ATOMIC - A simple task that can be done in one action (complexity 1-3)
2. DECOMPOSABLE - A complex task that needs to be broken into 2-5 subtasks (complexity 4-10)

For ATOMIC tasks, specify the tool to use (if any).
For DECOMPOSABLE tasks, list the subtasks in logical order.

## Output Format
```json
{{
  "complexity": 5,
  "is_atomic": false,
  "tool": null,
  "params": null,
  "subtasks": [
    {{
      "description": "First subtask",
      "estimated_complexity": 3
    }},
    {{
      "description": "Second subtask",
      "estimated_complexity": 4
    }}
  ]
}}
```

For atomic tasks, set is_atomic=true and subtasks=[].

Decompose the task now:"#
        );

        let messages = vec![
            Message {
                role: Role::System,
                content: "You are an expert at breaking complex problems into manageable subtasks.".to_string(),
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

        let request = GenerateRequest::chat(messages).with_sampling(
            SamplingParams::default()
                .with_max_tokens(1024)
                .with_temperature(0.3),
        );

        let response = self.engine.generate(request).await?;
        let response_text = response
            .choices
            .first()
            .map(|c| c.text.clone())
            .unwrap_or_default();

        // Parse the decomposition
        self.parse_decomposition(task_id, task_description, &response_text, current_depth)
    }

    /// Creates an atomic task with tool detection.
    async fn create_atomic_task(
        &self,
        task_id: &str,
        task_description: &str,
        tools_desc: &str,
    ) -> Result<HierarchicalTask> {
        let prompt = format!(
            r#"You are assigning a tool to execute a simple task.

## Task
{task_description}

## Available Tools
{tools_desc}

## Instructions
Determine which tool (if any) should be used and what parameters.

## Output Format
```json
{{
  "tool": "tool_name or null",
  "params": {{}},
  "complexity": 2
}}
```

Assign the tool:"#
        );

        let messages = vec![
            Message {
                role: Role::System,
                content: "You are an expert at matching tasks to tools.".to_string(),
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

        let request = GenerateRequest::chat(messages).with_sampling(
            SamplingParams::default()
                .with_max_tokens(256)
                .with_temperature(0.2),
        );

        let response = self.engine.generate(request).await?;
        let response_text = response
            .choices
            .first()
            .map(|c| c.text.clone())
            .unwrap_or_default();

        let mut task = HierarchicalTask::new(task_id, task_description);
        task.is_atomic = true;

        if let Some(json_str) = self.extract_json(&response_text) {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json_str) {
                if let Some(tool) = parsed.get("tool").and_then(|v| v.as_str()).filter(|s| *s != "null") {
                    task.tool = Some(tool.to_string());
                }
                if let Some(params) = parsed.get("params") {
                    if let Some(obj) = params.as_object() {
                        if !obj.is_empty() {
                            task.params = Some(params.clone());
                        }
                    }
                }
                if let Some(complexity) = parsed.get("complexity").and_then(|v| v.as_u64()) {
                    task.complexity = (complexity.min(10)) as u8;
                }
            }
        }

        Ok(task)
    }

    /// Parses decomposition response from LLM.
    fn parse_decomposition(
        &self,
        task_id: &str,
        task_description: &str,
        response: &str,
        current_depth: usize,
    ) -> Result<HierarchicalTask> {
        let mut task = HierarchicalTask::new(task_id, task_description);
        task.depth = current_depth;

        if let Some(json_str) = self.extract_json(response) {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json_str) {
                // Parse complexity
                if let Some(complexity) = parsed.get("complexity").and_then(|v| v.as_u64()) {
                    task.complexity = (complexity.min(10)) as u8;
                }

                // Parse atomic status
                if let Some(is_atomic) = parsed.get("is_atomic").and_then(|v| v.as_bool()) {
                    task.is_atomic = is_atomic;
                }

                // Parse tool for atomic tasks
                if task.is_atomic {
                    if let Some(tool) = parsed.get("tool").and_then(|v| v.as_str()).filter(|s| *s != "null") {
                        task.tool = Some(tool.to_string());
                    }
                    if let Some(params) = parsed.get("params") {
                        if !params.is_null() {
                            task.params = Some(params.clone());
                        }
                    }
                }

                // Parse subtasks
                if let Some(subtasks) = parsed.get("subtasks").and_then(|v| v.as_array()) {
                    for (i, subtask_value) in subtasks.iter().enumerate() {
                        let subtask_id = format!("{}.{}", task_id, i + 1);
                        let description = subtask_value
                            .get("description")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Subtask");
                        let complexity = subtask_value
                            .get("estimated_complexity")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(5) as u8;

                        let mut subtask = HierarchicalTask::new(subtask_id, description);
                        subtask.complexity = complexity.min(10);
                        subtask.depth = current_depth + 1;

                        // Mark simple subtasks as atomic
                        if complexity <= 3 {
                            subtask.is_atomic = true;
                        }

                        task.subtasks.push(subtask);
                    }
                }

                return Ok(task);
            }
        }

        // Fallback: treat as atomic if parsing fails
        task.is_atomic = true;
        task.complexity = 3;
        Ok(task)
    }

    /// Recursively refines decomposition for complex subtasks.
    fn refine_decomposition<'a>(
        &'a self,
        mut task: HierarchicalTask,
        tools_desc: &'a str,
        max_depth: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<HierarchicalTask>> + Send + 'a>> {
        Box::pin(async move {
            // Don't refine atomic tasks or if we're at max depth
            if task.is_atomic || task.depth >= max_depth {
                return Ok(task);
            }

            let mut refined_subtasks = Vec::new();

            for subtask in task.subtasks {
                // If subtask is complex and not atomic, decompose it further
                if subtask.complexity > 3 && !subtask.is_atomic && subtask.depth < max_depth {
                    tracing::debug!(
                        task_id = %subtask.id,
                        complexity = subtask.complexity,
                        depth = subtask.depth,
                        "Refining complex subtask"
                    );

                    let decomposed = self.decompose_task(
                        &subtask.id,
                        &subtask.description,
                        tools_desc,
                        subtask.depth,
                        max_depth,
                    ).await?;

                    // Recursively refine the decomposed task
                    let refined = self.refine_decomposition(decomposed, tools_desc, max_depth).await?;
                    refined_subtasks.push(refined);
                } else {
                    // Keep atomic or simple tasks as-is
                    refined_subtasks.push(subtask);
                }
            }

            task.subtasks = refined_subtasks;
            Ok(task)
        })
    }
}

/// A node in the thought tree.
#[derive(Debug, Clone)]
pub struct ThoughtNode {
    /// The thought content.
    pub content: String,
    /// Reasoning behind this thought.
    pub reasoning: String,
    /// Suggested tool to use.
    pub suggested_tool: Option<String>,
    /// Tool parameters if applicable.
    pub tool_params: Option<serde_json::Value>,
    /// Depth in the tree.
    pub depth: usize,
    /// Child thoughts (for tree structure).
    pub children: Vec<ThoughtNode>,
}

/// An evaluated branch with its score.
#[derive(Debug)]
struct EvaluatedBranch {
    thoughts: Vec<ThoughtNode>,
    score: f32,
}

/// A hierarchical task in the decomposition tree.
#[derive(Debug, Clone)]
pub struct HierarchicalTask {
    /// Task identifier (hierarchical: "1", "1.1", "1.1.2").
    pub id: String,
    /// Task description.
    pub description: String,
    /// Estimated complexity (1-10).
    pub complexity: u8,
    /// Whether this task is atomic (cannot be decomposed further).
    pub is_atomic: bool,
    /// Tool to use for atomic tasks.
    pub tool: Option<String>,
    /// Tool parameters.
    pub params: Option<serde_json::Value>,
    /// Subtasks if not atomic.
    pub subtasks: Vec<HierarchicalTask>,
    /// Depth in the hierarchy.
    pub depth: usize,
}

impl HierarchicalTask {
    /// Creates a new hierarchical task.
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            complexity: 5,
            is_atomic: false,
            tool: None,
            params: None,
            subtasks: Vec::new(),
            depth: 0,
        }
    }

    /// Sets the complexity estimate.
    #[must_use]
    pub fn with_complexity(mut self, complexity: u8) -> Self {
        self.complexity = complexity.min(10);
        self
    }

    /// Marks the task as atomic with an optional tool.
    #[must_use]
    pub fn atomic(mut self, tool: Option<String>, params: Option<serde_json::Value>) -> Self {
        self.is_atomic = true;
        self.tool = tool;
        self.params = params;
        self
    }

    /// Adds a subtask.
    pub fn add_subtask(&mut self, subtask: HierarchicalTask) {
        self.subtasks.push(subtask);
    }

    /// Flattens the task hierarchy into a linear sequence of atomic tasks.
    pub fn flatten(&self) -> Vec<PlanStep> {
        let mut steps = Vec::new();
        self.flatten_recursive(&mut steps, &mut Vec::new());
        steps
    }

    fn flatten_recursive(&self, steps: &mut Vec<PlanStep>, parent_ids: &mut Vec<String>) {
        if self.is_atomic || self.subtasks.is_empty() {
            let mut step = PlanStep::new(&self.id, &self.description);
            step.tool = self.tool.clone();
            step.params = self.params.clone();
            step.complexity = Some(self.complexity);
            step.dependencies = parent_ids.clone();
            steps.push(step);
        } else {
            // Add parent as a milestone step
            let mut milestone = PlanStep::new(&self.id, format!("[Group] {}", &self.description));
            milestone.complexity = Some(self.complexity);
            milestone.dependencies = parent_ids.clone();
            steps.push(milestone);

            // Recursively flatten subtasks
            let mut prev_subtask_id: Option<String> = None;
            for subtask in &self.subtasks {
                let mut child_deps = vec![self.id.clone()];
                if let Some(prev_id) = &prev_subtask_id {
                    child_deps.push(prev_id.clone());
                }

                let start_len = steps.len();
                let child = subtask.clone();
                child.flatten_recursive(steps, &mut child_deps);

                // Track the last subtask for sequential dependency
                if steps.len() > start_len {
                    prev_subtask_id = steps.last().map(|s| s.id.clone());
                }
            }
        }
    }

    /// Returns total task count including all subtasks.
    pub fn total_tasks(&self) -> usize {
        1 + self.subtasks.iter().map(|s| s.total_tasks()).sum::<usize>()
    }
}

#[async_trait]
impl Planner for LLMPlanner {
    async fn plan(&self, objective: &str, tools: &ToolRegistry) -> Result<Plan> {
        tracing::debug!(objective, strategy = ?self.strategy, "Generating LLM-based plan");

        // Use Tree of Thoughts traversal for ToT strategy
        if let PlanningStrategy::TreeOfThoughts { breadth, depth } = &self.strategy {
            return self.plan_with_tot(objective, tools, *breadth, *depth).await;
        }

        // Use hierarchical decomposition for Hierarchical strategy
        if let PlanningStrategy::Hierarchical { max_depth } = &self.strategy {
            return self.plan_with_hierarchical(objective, tools, *max_depth).await;
        }

        let prompt = self.build_planning_prompt(objective, tools);

        let messages = vec![
            Message {
                role: Role::System,
                content: "You are an expert task planner. Create detailed, actionable plans."
                    .to_string(),
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

        let request = GenerateRequest::chat(messages).with_sampling(
            SamplingParams::default()
                .with_max_tokens(2048)
                .with_temperature(0.3),
        ); // Lower temperature for more consistent planning

        let response = self.engine.generate(request).await?;
        let response_text = response
            .choices
            .first()
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
                content: "You are an expert task planner. Revise plans based on feedback."
                    .to_string(),
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

        let request = GenerateRequest::chat(messages).with_sampling(
            SamplingParams::default()
                .with_max_tokens(2048)
                .with_temperature(0.3),
        );

        let response = self.engine.generate(request).await?;
        let response_text = response
            .choices
            .first()
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

    // ==========================================================================
    // PlanStep Tests
    // ==========================================================================

    #[test]
    fn test_plan_step_new() {
        let step = PlanStep::new("1", "Test step");
        assert_eq!(step.id, "1");
        assert_eq!(step.description, "Test step");
        assert!(step.tool.is_none());
        assert!(step.params.is_none());
        assert!(step.dependencies.is_empty());
        assert!(step.complexity.is_none());
    }

    #[test]
    fn test_plan_step_with_tool() {
        let step = PlanStep::new("1", "Calculate").with_tool("calculator");
        assert_eq!(step.tool, Some("calculator".to_string()));
    }

    #[test]
    fn test_plan_step_with_params() {
        let params = serde_json::json!({"expr": "2+2"});
        let step = PlanStep::new("1", "Calculate").with_params(params.clone());
        assert_eq!(step.params, Some(params));
    }

    #[test]
    fn test_plan_step_depends_on() {
        let step = PlanStep::new("2", "Second")
            .depends_on("1");
        assert_eq!(step.dependencies, vec!["1".to_string()]);
    }

    #[test]
    fn test_plan_step_multiple_dependencies() {
        let step = PlanStep::new("3", "Third")
            .depends_on("1")
            .depends_on("2");
        assert_eq!(step.dependencies.len(), 2);
        assert!(step.dependencies.contains(&"1".to_string()));
        assert!(step.dependencies.contains(&"2".to_string()));
    }

    #[test]
    fn test_plan_step_builder_chain() {
        let step = PlanStep::new("1", "Test step")
            .with_tool("calculator")
            .with_params(serde_json::json!({"expr": "2+2"}))
            .depends_on("0");

        assert_eq!(step.id, "1");
        assert_eq!(step.tool, Some("calculator".to_string()));
        assert_eq!(step.dependencies, vec!["0".to_string()]);
    }

    #[test]
    fn test_plan_step_clone() {
        let step = PlanStep::new("1", "Test").with_tool("calc");
        let cloned = step.clone();
        assert_eq!(cloned.id, step.id);
        assert_eq!(cloned.tool, step.tool);
    }

    #[test]
    fn test_plan_step_debug() {
        let step = PlanStep::new("1", "Test");
        let debug = format!("{:?}", step);
        assert!(debug.contains("PlanStep"));
        assert!(debug.contains("Test"));
    }

    // ==========================================================================
    // Plan Tests
    // ==========================================================================

    #[test]
    fn test_plan_new() {
        let plan = Plan::new("Test objective");
        assert!(!plan.id.is_empty());
        assert_eq!(plan.objective, "Test objective");
        assert!(plan.steps.is_empty());
        assert_eq!(plan.current_step, 0);
        assert!(!plan.complete);
    }

    #[test]
    fn test_plan_add_step() {
        let mut plan = Plan::new("Test");
        plan.add_step(PlanStep::new("1", "First"));
        plan.add_step(PlanStep::new("2", "Second"));
        assert_eq!(plan.steps.len(), 2);
    }

    #[test]
    fn test_plan_next_step() {
        let mut plan = Plan::new("Test");
        assert!(plan.next_step().is_none());

        plan.add_step(PlanStep::new("1", "First"));
        assert_eq!(plan.next_step().unwrap().id, "1");
    }

    #[test]
    fn test_plan_advance() {
        let mut plan = Plan::new("Test");
        plan.add_step(PlanStep::new("1", "First"));
        plan.add_step(PlanStep::new("2", "Second"));

        assert_eq!(plan.current_step, 0);
        plan.advance();
        assert_eq!(plan.current_step, 1);
        assert!(!plan.complete);

        plan.advance();
        assert_eq!(plan.current_step, 2);
        assert!(plan.complete);
    }

    #[test]
    fn test_plan_advance_empty() {
        let mut plan = Plan::new("Test");
        plan.advance();
        assert!(plan.complete);
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
    fn test_plan_remaining_steps() {
        let mut plan = Plan::new("Test plan");
        plan.add_step(PlanStep::new("1", "First step"));
        plan.add_step(PlanStep::new("2", "Second step"));
        plan.add_step(PlanStep::new("3", "Third step"));

        assert_eq!(plan.remaining_steps().len(), 3);

        plan.advance();
        assert_eq!(plan.remaining_steps().len(), 2);
        assert_eq!(plan.remaining_steps()[0].id, "2");

        plan.advance();
        assert_eq!(plan.remaining_steps().len(), 1);
        assert_eq!(plan.remaining_steps()[0].id, "3");

        plan.advance();
        assert!(plan.remaining_steps().is_empty());
    }

    #[test]
    fn test_plan_clone() {
        let mut plan = Plan::new("Test");
        plan.add_step(PlanStep::new("1", "First"));
        let cloned = plan.clone();
        assert_eq!(cloned.id, plan.id);
        assert_eq!(cloned.steps.len(), 1);
    }

    // ==========================================================================
    // PlanningStrategy Tests
    // ==========================================================================

    #[test]
    fn test_planning_strategy_single_shot() {
        let strategy = PlanningStrategy::SingleShot;
        assert!(matches!(strategy, PlanningStrategy::SingleShot));
    }

    #[test]
    fn test_planning_strategy_react() {
        let strategy = PlanningStrategy::ReAct { max_iterations: 10 };
        if let PlanningStrategy::ReAct { max_iterations } = strategy {
            assert_eq!(max_iterations, 10);
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_planning_strategy_tree_of_thoughts() {
        let strategy = PlanningStrategy::TreeOfThoughts { breadth: 3, depth: 5 };
        if let PlanningStrategy::TreeOfThoughts { breadth, depth } = strategy {
            assert_eq!(breadth, 3);
            assert_eq!(depth, 5);
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_planning_strategy_hierarchical() {
        let strategy = PlanningStrategy::Hierarchical { max_depth: 4 };
        if let PlanningStrategy::Hierarchical { max_depth } = strategy {
            assert_eq!(max_depth, 4);
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_planning_strategy_variants() {
        let single = PlanningStrategy::SingleShot;
        let react = PlanningStrategy::ReAct { max_iterations: 10 };
        let tot = PlanningStrategy::TreeOfThoughts { breadth: 3, depth: 4 };
        let hierarchical = PlanningStrategy::Hierarchical { max_depth: 3 };

        // Verify strategies can be pattern matched
        match single {
            PlanningStrategy::SingleShot => {}
            _ => panic!("Wrong variant"),
        }

        if let PlanningStrategy::ReAct { max_iterations } = react {
            assert_eq!(max_iterations, 10);
        }

        if let PlanningStrategy::TreeOfThoughts { breadth, depth } = tot {
            assert_eq!(breadth, 3);
            assert_eq!(depth, 4);
        }

        if let PlanningStrategy::Hierarchical { max_depth } = hierarchical {
            assert_eq!(max_depth, 3);
        }
    }

    #[test]
    fn test_planning_strategy_clone() {
        let strategy = PlanningStrategy::ReAct { max_iterations: 5 };
        let cloned = strategy.clone();
        if let PlanningStrategy::ReAct { max_iterations } = cloned {
            assert_eq!(max_iterations, 5);
        }
    }

    #[test]
    fn test_planning_strategy_debug() {
        let strategy = PlanningStrategy::SingleShot;
        let debug = format!("{:?}", strategy);
        assert!(debug.contains("SingleShot"));
    }

    // ==========================================================================
    // ThoughtNode Tests
    // ==========================================================================

    #[test]
    fn test_thought_node_creation() {
        let thought = ThoughtNode {
            content: "Analyze the data".to_string(),
            reasoning: "Need to understand the input".to_string(),
            suggested_tool: Some("analyzer".to_string()),
            tool_params: Some(serde_json::json!({"format": "json"})),
            depth: 1,
            children: vec![],
        };

        assert_eq!(thought.content, "Analyze the data");
        assert_eq!(thought.reasoning, "Need to understand the input");
        assert_eq!(thought.depth, 1);
        assert!(thought.suggested_tool.is_some());
        assert!(thought.children.is_empty());
    }

    #[test]
    fn test_thought_node_without_tool() {
        let thought = ThoughtNode {
            content: "Think about problem".to_string(),
            reasoning: "Reasoning step".to_string(),
            suggested_tool: None,
            tool_params: None,
            depth: 0,
            children: vec![],
        };

        assert!(thought.suggested_tool.is_none());
        assert!(thought.tool_params.is_none());
    }

    #[test]
    fn test_thought_node_with_children() {
        let child1 = ThoughtNode {
            content: "Sub-task 1".to_string(),
            reasoning: "First sub-task".to_string(),
            suggested_tool: None,
            tool_params: None,
            depth: 2,
            children: vec![],
        };

        let child2 = ThoughtNode {
            content: "Sub-task 2".to_string(),
            reasoning: "Second sub-task".to_string(),
            suggested_tool: Some("calculator".to_string()),
            tool_params: None,
            depth: 2,
            children: vec![],
        };

        let parent = ThoughtNode {
            content: "Main task".to_string(),
            reasoning: "Top-level reasoning".to_string(),
            suggested_tool: None,
            tool_params: None,
            depth: 1,
            children: vec![child1, child2],
        };

        assert_eq!(parent.children.len(), 2);
        assert_eq!(parent.children[0].content, "Sub-task 1");
        assert_eq!(parent.children[1].suggested_tool, Some("calculator".to_string()));
    }

    #[test]
    fn test_thought_node_clone() {
        let thought = ThoughtNode {
            content: "Test".to_string(),
            reasoning: "Reason".to_string(),
            suggested_tool: Some("tool".to_string()),
            tool_params: None,
            depth: 0,
            children: vec![],
        };

        let cloned = thought.clone();
        assert_eq!(cloned.content, thought.content);
        assert_eq!(cloned.suggested_tool, thought.suggested_tool);
    }

    #[test]
    fn test_thought_node_nested_depth() {
        let leaf = ThoughtNode {
            content: "Leaf".to_string(),
            reasoning: "".to_string(),
            suggested_tool: None,
            tool_params: None,
            depth: 3,
            children: vec![],
        };

        let middle = ThoughtNode {
            content: "Middle".to_string(),
            reasoning: "".to_string(),
            suggested_tool: None,
            tool_params: None,
            depth: 2,
            children: vec![leaf],
        };

        let root = ThoughtNode {
            content: "Root".to_string(),
            reasoning: "".to_string(),
            suggested_tool: None,
            tool_params: None,
            depth: 1,
            children: vec![middle],
        };

        assert_eq!(root.depth, 1);
        assert_eq!(root.children[0].depth, 2);
        assert_eq!(root.children[0].children[0].depth, 3);
    }

    // ==========================================================================
    // HierarchicalTask Tests
    // ==========================================================================

    #[test]
    fn test_hierarchical_task_new() {
        let task = HierarchicalTask::new("1", "Test task");
        assert_eq!(task.id, "1");
        assert_eq!(task.description, "Test task");
        assert_eq!(task.complexity, 5); // Default
        assert!(!task.is_atomic);
        assert!(task.tool.is_none());
        assert!(task.params.is_none());
        assert!(task.subtasks.is_empty());
        assert_eq!(task.depth, 0);
    }

    #[test]
    fn test_hierarchical_task_with_complexity() {
        let task = HierarchicalTask::new("1", "Test").with_complexity(8);
        assert_eq!(task.complexity, 8);
    }

    #[test]
    fn test_hierarchical_task_with_complexity_clamped() {
        let task = HierarchicalTask::new("1", "Test").with_complexity(15);
        assert_eq!(task.complexity, 10); // Clamped to max
    }

    #[test]
    fn test_hierarchical_task_atomic() {
        let task = HierarchicalTask::new("1", "Calculate")
            .atomic(Some("calculator".to_string()), Some(serde_json::json!({"x": 5})));

        assert!(task.is_atomic);
        assert_eq!(task.tool, Some("calculator".to_string()));
        assert!(task.params.is_some());
    }

    #[test]
    fn test_hierarchical_task_atomic_no_tool() {
        let task = HierarchicalTask::new("1", "Think").atomic(None, None);
        assert!(task.is_atomic);
        assert!(task.tool.is_none());
    }

    #[test]
    fn test_hierarchical_task_add_subtask() {
        let mut parent = HierarchicalTask::new("1", "Parent");
        let child = HierarchicalTask::new("1.1", "Child");
        parent.add_subtask(child);

        assert_eq!(parent.subtasks.len(), 1);
        assert_eq!(parent.subtasks[0].id, "1.1");
    }

    #[test]
    fn test_hierarchical_task_multiple_subtasks() {
        let mut parent = HierarchicalTask::new("1", "Parent");
        parent.add_subtask(HierarchicalTask::new("1.1", "First"));
        parent.add_subtask(HierarchicalTask::new("1.2", "Second"));
        parent.add_subtask(HierarchicalTask::new("1.3", "Third"));

        assert_eq!(parent.subtasks.len(), 3);
    }

    #[test]
    fn test_hierarchical_task_total_tasks_atomic() {
        let task = HierarchicalTask::new("1", "Atomic").atomic(None, None);
        assert_eq!(task.total_tasks(), 1);
    }

    #[test]
    fn test_hierarchical_task_total_tasks_with_subtasks() {
        let mut parent = HierarchicalTask::new("1", "Parent");
        parent.add_subtask(HierarchicalTask::new("1.1", "Child 1"));
        parent.add_subtask(HierarchicalTask::new("1.2", "Child 2"));

        assert_eq!(parent.total_tasks(), 3); // 1 parent + 2 children
    }

    #[test]
    fn test_hierarchical_task_total_tasks_nested() {
        let mut root = HierarchicalTask::new("1", "Root");
        let mut child = HierarchicalTask::new("1.1", "Child");
        child.add_subtask(HierarchicalTask::new("1.1.1", "Grandchild"));
        root.add_subtask(child);

        assert_eq!(root.total_tasks(), 3); // root + child + grandchild
    }

    #[test]
    fn test_hierarchical_task_flatten_atomic() {
        let task = HierarchicalTask::new("1", "Atomic task")
            .with_complexity(2)
            .atomic(Some("tool".to_string()), None);

        let steps = task.flatten();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].id, "1");
        assert_eq!(steps[0].tool, Some("tool".to_string()));
        assert_eq!(steps[0].complexity, Some(2));
    }

    #[test]
    fn test_hierarchical_task_flatten_with_subtasks() {
        let mut parent = HierarchicalTask::new("1", "Parent");
        parent.add_subtask(
            HierarchicalTask::new("1.1", "Child 1").atomic(None, None)
        );
        parent.add_subtask(
            HierarchicalTask::new("1.2", "Child 2").atomic(None, None)
        );

        let steps = parent.flatten();
        // 1 milestone + 2 atomic children
        assert_eq!(steps.len(), 3);
    }

    #[test]
    fn test_hierarchical_task_clone() {
        let mut task = HierarchicalTask::new("1", "Test").with_complexity(7);
        task.add_subtask(HierarchicalTask::new("1.1", "Child"));

        let cloned = task.clone();
        assert_eq!(cloned.id, task.id);
        assert_eq!(cloned.complexity, task.complexity);
        assert_eq!(cloned.subtasks.len(), 1);
    }

    // ==========================================================================
    // DefaultPlanner Tests
    // ==========================================================================

    #[test]
    fn test_default_planner_new() {
        let planner = DefaultPlanner::new(PlanningStrategy::SingleShot);
        // Just verify it constructs
        let _ = format!("{:?}", planner.strategy);
    }

    #[test]
    fn test_default_planner_with_react_strategy() {
        let planner = DefaultPlanner::new(PlanningStrategy::ReAct { max_iterations: 5 });
        if let PlanningStrategy::ReAct { max_iterations } = planner.strategy {
            assert_eq!(max_iterations, 5);
        }
    }

    #[tokio::test]
    async fn test_default_planner_plan() {
        let planner = DefaultPlanner::new(PlanningStrategy::SingleShot);
        let tools = ToolRegistry::new();

        let plan = planner.plan("Test objective", &tools).await.expect("plan");

        assert_eq!(plan.objective, "Test objective");
        assert!(!plan.steps.is_empty());
        assert!(plan.steps[0].description.contains("Test objective"));
    }

    #[tokio::test]
    async fn test_default_planner_replan() {
        let planner = DefaultPlanner::new(PlanningStrategy::SingleShot);
        let tools = ToolRegistry::new();

        let original = planner.plan("Original", &tools).await.expect("plan");
        let revised = planner.replan(&original, "feedback", &tools).await.expect("replan");

        // Replan creates a new plan for the same objective
        assert_eq!(revised.objective, original.objective);
    }

    // ==========================================================================
    // JSON Parsing Tests
    // ==========================================================================

    #[test]
    fn test_parse_json_steps() {
        let response = r#"
        Here's the plan:
        ```json
        [
            {"id": "1", "description": "First step", "tool": "calculator", "params": {"expr": "2+2"}, "dependencies": []},
            {"id": "2", "description": "Second step", "tool": null, "params": null, "dependencies": ["1"]}
        ]
        ```
        "#;

        // Validate the JSON format is correct
        let json_start = response.find("```json").unwrap() + 7;
        let json_end = response[json_start..].find("```").unwrap() + json_start;
        let json_str = &response[json_start..json_end].trim();

        let parsed: Vec<serde_json::Value> = serde_json::from_str(json_str).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0]["id"], "1");
        assert_eq!(parsed[0]["tool"], "calculator");
        assert_eq!(parsed[1]["dependencies"][0], "1");
    }

    #[test]
    fn test_parse_json_with_complexity() {
        let json = r#"[
            {"id": "1", "description": "Step", "tool": null, "params": null, "dependencies": [], "complexity": 5}
        ]"#;

        let parsed: Vec<serde_json::Value> = serde_json::from_str(json).unwrap();
        assert_eq!(parsed[0]["complexity"], 5);
    }

    #[test]
    fn test_parse_hierarchical_json() {
        let json = r#"{
            "complexity": 7,
            "is_atomic": false,
            "tool": null,
            "params": null,
            "subtasks": [
                {"description": "Subtask 1", "estimated_complexity": 3},
                {"description": "Subtask 2", "estimated_complexity": 4}
            ]
        }"#;

        let parsed: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(parsed["complexity"], 7);
        assert_eq!(parsed["is_atomic"], false);
        let subtasks = parsed["subtasks"].as_array().unwrap();
        assert_eq!(subtasks.len(), 2);
    }

    // ==========================================================================
    // Edge Cases Tests
    // ==========================================================================

    #[test]
    fn test_empty_plan() {
        let plan = Plan::new("Empty objective");
        assert!(plan.steps.is_empty());
        assert!(plan.next_step().is_none());
        assert!(plan.remaining_steps().is_empty());
    }

    #[test]
    fn test_plan_with_empty_objective() {
        let plan = Plan::new("");
        assert!(plan.objective.is_empty());
    }

    #[test]
    fn test_plan_step_with_empty_strings() {
        let step = PlanStep::new("", "");
        assert!(step.id.is_empty());
        assert!(step.description.is_empty());
    }

    #[test]
    fn test_hierarchical_task_deep_nesting() {
        let mut root = HierarchicalTask::new("1", "Root");
        root.depth = 0;

        let mut level1 = HierarchicalTask::new("1.1", "Level 1");
        level1.depth = 1;

        let mut level2 = HierarchicalTask::new("1.1.1", "Level 2");
        level2.depth = 2;

        let level3 = HierarchicalTask::new("1.1.1.1", "Level 3")
            .atomic(None, None);

        level2.add_subtask(level3);
        level1.add_subtask(level2);
        root.add_subtask(level1);

        assert_eq!(root.total_tasks(), 4);
    }

    #[test]
    fn test_plan_step_with_complex_params() {
        let params = serde_json::json!({
            "nested": {
                "array": [1, 2, 3],
                "object": {"key": "value"}
            },
            "number": 42,
            "boolean": true
        });

        let step = PlanStep::new("1", "Complex").with_params(params.clone());
        assert_eq!(step.params.unwrap()["nested"]["array"][0], 1);
    }

    #[test]
    fn test_thought_node_empty_children() {
        let thought = ThoughtNode {
            content: "Leaf".to_string(),
            reasoning: "End of branch".to_string(),
            suggested_tool: None,
            tool_params: None,
            depth: 5,
            children: vec![],
        };

        assert!(thought.children.is_empty());
        assert_eq!(thought.depth, 5);
    }
}
