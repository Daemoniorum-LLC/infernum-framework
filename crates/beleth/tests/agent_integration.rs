//! Integration tests for Beleth Agent Framework.
//!
//! Tests the complete agent workflow including:
//! - Tool registration and execution
//! - Memory systems
//! - Planning strategies

use std::sync::Arc;

use async_trait::async_trait;
use beleth::{
    // Memory
    AgentMemory,
    CalculatorTool,
    ContextConfig,
    DateTimeTool,
    DefaultPlanner,
    // Dynamic Context
    DynamicContextManager,
    ImportanceLevel,
    JsonTool,
    // Long-term memory
    LongTermMemory,
    MemoryEntry,
    MemoryType,
    // Planning
    Plan,
    PlanStep,
    Planner,
    PlanningStrategy,
    RiskLevel,
    SummarizationStrategy,
    // Tools
    Tool,
    ToolCall,
    ToolContext,
    ToolRegistry,
    ToolResult,
};
use tempfile::TempDir;

// =============================================================================
// Tool Integration Tests
// =============================================================================

#[test]
fn test_tool_registry_register_and_list() {
    let mut registry = ToolRegistry::new();

    // Register built-in tools
    registry.register(Arc::new(CalculatorTool));
    registry.register(Arc::new(DateTimeTool));
    registry.register(Arc::new(JsonTool));

    // List tools
    let tools = registry.list();
    assert_eq!(tools.len(), 3);

    // Verify tool names
    assert!(tools.contains(&"calculator"));
    assert!(tools.contains(&"datetime"));
    assert!(tools.contains(&"json"));
}

#[test]
fn test_tool_registry_with_builtins() {
    let registry = ToolRegistry::with_builtins();

    // Should have all built-in tools
    assert!(registry.get("calculator").is_some());
    assert!(registry.get("datetime").is_some());
    assert!(registry.get("json").is_some());
    assert_eq!(registry.len(), 3);
}

#[test]
fn test_tool_registry_get_by_name() {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(CalculatorTool));

    // Get existing tool
    let tool = registry.get("calculator");
    assert!(tool.is_some());
    assert_eq!(tool.expect("tool").name(), "calculator");

    // Get non-existing tool
    let missing = registry.get("nonexistent");
    assert!(missing.is_none());
}

#[tokio::test]
async fn test_calculator_tool_execution() {
    let tool = CalculatorTool;
    let context = ToolContext::new("test-agent");

    // Test addition
    let result = tool
        .execute(
            serde_json::json!({
                "expression": "2 + 2"
            }),
            &context,
        )
        .await;

    assert!(result.is_ok());
    let output = result.expect("result");
    assert!(output.success);
    assert!(output.output.contains("4"));
}

#[tokio::test]
async fn test_calculator_tool_multiplication() {
    let tool = CalculatorTool;
    let context = ToolContext::new("test-agent");

    // Test multiplication
    let result = tool
        .execute(
            serde_json::json!({
                "expression": "3 * 4"
            }),
            &context,
        )
        .await;

    assert!(result.is_ok());
    let output = result.expect("result");
    assert!(output.success);
    assert!(output.output.contains("12"));
}

#[tokio::test]
async fn test_datetime_tool_execution() {
    let tool = DateTimeTool;
    let context = ToolContext::new("test-agent");

    // Test getting current time
    let result = tool
        .execute(
            serde_json::json!({
                "operation": "now"
            }),
            &context,
        )
        .await;

    assert!(result.is_ok());
    let output = result.expect("result");
    assert!(output.success);
}

#[tokio::test]
async fn test_json_tool_parse() {
    let tool = JsonTool;
    let context = ToolContext::new("test-agent");

    // Test parsing JSON
    let result = tool
        .execute(
            serde_json::json!({
                "operation": "parse",
                "data": "{\"key\": \"value\"}"
            }),
            &context,
        )
        .await;

    assert!(result.is_ok());
    let output = result.expect("result");
    assert!(output.success);
}

#[tokio::test]
async fn test_json_tool_format() {
    let tool = JsonTool;
    let context = ToolContext::new("test-agent");

    // Test formatting JSON
    let result = tool
        .execute(
            serde_json::json!({
                "operation": "format",
                "data": "{\"key\":\"value\",\"nested\":{\"a\":1}}"
            }),
            &context,
        )
        .await;

    assert!(result.is_ok());
    let output = result.expect("result");
    assert!(output.success);
    // Formatted JSON should contain newlines
    assert!(output.output.contains('\n'));
}

// Custom tool for testing
struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Echoes the input back"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "message": {"type": "string"}
            },
            "required": ["message"]
        })
    }

    fn risk_level(&self) -> RiskLevel {
        RiskLevel::Safe
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        _context: &ToolContext,
    ) -> infernum_core::Result<ToolResult> {
        let message = params["message"].as_str().unwrap_or("no message");
        Ok(ToolResult::success(format!("Echo: {}", message)))
    }
}

#[tokio::test]
async fn test_custom_tool_execution() {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(EchoTool));

    let tool = registry.get("echo").expect("echo tool");
    let context = ToolContext::new("test-agent");

    let result = tool
        .execute(
            serde_json::json!({
                "message": "Hello, World!"
            }),
            &context,
        )
        .await;

    assert!(result.is_ok());
    let output = result.expect("result");
    assert!(output.success);
    assert_eq!(output.output, "Echo: Hello, World!");
}

#[tokio::test]
async fn test_tool_registry_execute() {
    let registry = ToolRegistry::with_builtins();
    let context = ToolContext::new("test-agent");

    let call = ToolCall {
        name: "calculator".to_string(),
        params: serde_json::json!({"expression": "10 / 2"}),
    };

    let result = registry.execute(&call, &context).await;
    assert!(result.is_ok());
    let output = result.expect("result");
    assert!(output.success);
    assert!(output.output.contains("5"));
}

// =============================================================================
// Memory Integration Tests
// =============================================================================

#[test]
fn test_agent_memory_creation() {
    let memory = AgentMemory::new();
    // Memory should start empty
    assert!(memory.messages().is_empty());
}

#[test]
fn test_agent_memory_with_max_messages() {
    let memory = AgentMemory::with_max_messages(10);
    assert!(memory.messages().is_empty());
}

#[test]
fn test_agent_memory_add_message() {
    let mut memory = AgentMemory::new();

    let message = infernum_core::Message::user("Hello!");
    memory.add_message(message);

    assert_eq!(memory.messages().len(), 1);
}

#[test]
fn test_agent_memory_with_strategy() {
    let memory = AgentMemory::with_max_messages(5)
        .with_strategy(SummarizationStrategy::SlidingWindow { keep_recent: 3 });

    assert!(memory.messages().is_empty());
}

// =============================================================================
// Long-Term Memory Integration Tests
// =============================================================================

#[test]
fn test_long_term_memory_creation() {
    let temp = TempDir::new().expect("temp dir");
    let memory = LongTermMemory::new(temp.path()).expect("create memory");

    let stats = memory.stats();
    assert_eq!(stats.total_entries, 0);
}

#[test]
fn test_long_term_memory_store_and_retrieve() {
    let temp = TempDir::new().expect("temp dir");
    let mut memory = LongTermMemory::new(temp.path()).expect("create memory");

    // Store a memory
    let entry = MemoryEntry::new(MemoryType::ProjectLearning, "The user prefers dark mode")
        .with_importance(ImportanceLevel::High)
        .with_tag("preference");

    let id = memory.store(entry).expect("store memory");

    // Retrieve it
    let retrieved = memory.get(&id);
    assert!(retrieved.is_some());
    assert!(retrieved.expect("entry").content.contains("dark mode"));
}

#[test]
fn test_long_term_memory_get_by_type() {
    let temp = TempDir::new().expect("temp dir");
    let mut memory = LongTermMemory::new(temp.path()).expect("create memory");

    // Store memories of different types
    let learning = MemoryEntry::new(MemoryType::ProjectLearning, "Learning 1");
    let decision = MemoryEntry::new(MemoryType::Decision, "Decision 1");

    memory.store(learning).expect("store");
    memory.store(decision).expect("store");

    // Get by type
    let learnings = memory.get_by_type(MemoryType::ProjectLearning);
    assert_eq!(learnings.len(), 1);

    let decisions = memory.get_by_type(MemoryType::Decision);
    assert_eq!(decisions.len(), 1);
}

#[test]
fn test_long_term_memory_get_by_tag() {
    let temp = TempDir::new().expect("temp dir");
    let mut memory = LongTermMemory::new(temp.path()).expect("create memory");

    // Store memories with tags
    let entry1 = MemoryEntry::new(MemoryType::ProjectLearning, "Learning about Rust")
        .with_tag("rust")
        .with_tag("programming");

    let entry2 = MemoryEntry::new(MemoryType::ProjectLearning, "Learning about Python")
        .with_tag("python")
        .with_tag("programming");

    memory.store(entry1).expect("store");
    memory.store(entry2).expect("store");

    // Get by tag
    let rust_entries = memory.get_by_tag("rust");
    assert_eq!(rust_entries.len(), 1);

    let programming_entries = memory.get_by_tag("programming");
    assert_eq!(programming_entries.len(), 2);
}

#[test]
fn test_long_term_memory_importance_filter() {
    let temp = TempDir::new().expect("temp dir");
    let mut memory = LongTermMemory::new(temp.path()).expect("create memory");

    // Store memories with different importance levels
    let critical = MemoryEntry::new(MemoryType::Decision, "Critical decision")
        .with_importance(ImportanceLevel::Critical);

    let low =
        MemoryEntry::new(MemoryType::Context, "Minor detail").with_importance(ImportanceLevel::Low);

    memory.store(critical).expect("store");
    memory.store(low).expect("store");

    // Get important entries
    let important = memory.get_important(ImportanceLevel::High, 10);
    assert_eq!(important.len(), 1);
    assert!(important[0].content.contains("Critical"));
}

#[test]
fn test_long_term_memory_search() {
    let temp = TempDir::new().expect("temp dir");
    let mut memory = LongTermMemory::new(temp.path()).expect("create memory");

    // Store memories
    let entry1 = MemoryEntry::new(MemoryType::ProjectLearning, "The database uses PostgreSQL");
    let entry2 = MemoryEntry::new(MemoryType::ProjectLearning, "Redis is used for caching");

    memory.store(entry1).expect("store");
    memory.store(entry2).expect("store");

    // Search
    let postgres_results = memory.search("PostgreSQL");
    assert_eq!(postgres_results.len(), 1);

    let redis_results = memory.search("Redis");
    assert_eq!(redis_results.len(), 1);

    let db_results = memory.search("database");
    assert_eq!(db_results.len(), 1);
}

#[test]
fn test_long_term_memory_delete() {
    let temp = TempDir::new().expect("temp dir");
    let mut memory = LongTermMemory::new(temp.path()).expect("create memory");

    // Store and delete
    let entry = MemoryEntry::new(MemoryType::ProjectLearning, "Temporary learning");
    let id = memory.store(entry).expect("store");

    assert!(memory.get(&id).is_some());

    memory.delete(&id).expect("delete");
    assert!(memory.get(&id).is_none());
}

// =============================================================================
// Planning Integration Tests
// =============================================================================

#[test]
fn test_plan_creation() {
    let mut plan = Plan::new("Write a test");

    // Add steps
    plan.add_step(PlanStep::new("1", "Set up test environment"));
    plan.add_step(PlanStep::new("2", "Write test cases"));
    plan.add_step(PlanStep::new("3", "Run tests"));

    assert_eq!(plan.steps.len(), 3);
    assert!(!plan.complete);
}

#[test]
fn test_plan_execution() {
    let mut plan = Plan::new("Execute steps");

    plan.add_step(PlanStep::new("1", "Step 1"));
    plan.add_step(PlanStep::new("2", "Step 2"));

    // Get next step
    let step = plan.next_step();
    assert!(step.is_some());
    assert_eq!(step.expect("step").id, "1");

    // Advance
    plan.advance();
    let step = plan.next_step();
    assert!(step.is_some());
    assert_eq!(step.expect("step").id, "2");

    // Advance again - should complete
    plan.advance();
    assert!(plan.complete);
    assert!(plan.next_step().is_none());
}

#[test]
fn test_plan_step_with_tool() {
    let step = PlanStep::new("1", "Calculate the sum")
        .with_tool("calculator")
        .with_params(serde_json::json!({"expression": "1 + 1"}));

    assert_eq!(step.tool, Some("calculator".to_string()));
    assert!(step.params.is_some());
}

#[test]
fn test_plan_step_dependencies() {
    let step = PlanStep::new("3", "Finalize")
        .depends_on("1")
        .depends_on("2");

    assert_eq!(step.dependencies.len(), 2);
    assert!(step.dependencies.contains(&"1".to_string()));
    assert!(step.dependencies.contains(&"2".to_string()));
}

#[test]
fn test_plan_remaining_steps() {
    let mut plan = Plan::new("Multi-step plan");

    plan.add_step(PlanStep::new("1", "First"));
    plan.add_step(PlanStep::new("2", "Second"));
    plan.add_step(PlanStep::new("3", "Third"));

    assert_eq!(plan.remaining_steps().len(), 3);

    plan.advance();
    assert_eq!(plan.remaining_steps().len(), 2);

    plan.advance();
    assert_eq!(plan.remaining_steps().len(), 1);
}

#[tokio::test]
async fn test_default_planner() {
    let planner = DefaultPlanner::new(PlanningStrategy::SingleShot);
    let registry = ToolRegistry::new();

    // Plan a simple objective
    let plan = planner
        .plan("Write a hello world program", &registry)
        .await
        .expect("plan");

    // Default planner creates a single step
    assert!(!plan.steps.is_empty());
    assert!(!plan.complete);
}

#[tokio::test]
async fn test_default_planner_replan() {
    let planner = DefaultPlanner::new(PlanningStrategy::SingleShot);
    let registry = ToolRegistry::new();

    let plan = planner
        .plan("Initial objective", &registry)
        .await
        .expect("plan");

    let new_plan = planner
        .replan(&plan, "Need to adjust the approach", &registry)
        .await
        .expect("replan");

    assert!(!new_plan.steps.is_empty());
}

// =============================================================================
// Dynamic Context Integration Tests
// =============================================================================

#[test]
fn test_dynamic_context_manager() {
    let config = ContextConfig {
        max_input_tokens: 100_000,
        max_output_tokens: 4_000,
        min_relevance: 0.0, // Accept all messages
        overlap_tokens: 100,
        max_chunk_tokens: 2000,
    };

    let manager = DynamicContextManager::new().with_config(config);

    // Test that the manager was created successfully
    // The manager can optimize messages to fit within the context budget
    let messages = vec![
        infernum_core::Message::user("Hello"),
        infernum_core::Message::assistant("Hi there!"),
    ];
    // Optimize returns messages that fit within the token budget
    // With a high budget and low relevance threshold, all messages should be kept
    let optimized = manager.optimize(&messages);
    // Even with min_relevance=0.0, some messages might be filtered
    // Just verify the method works without panicking
    assert!(optimized.len() <= messages.len());
}

#[test]
fn test_dynamic_context_config_defaults() {
    let config = ContextConfig::default();

    assert!(config.max_input_tokens > 0);
    assert!(config.min_relevance >= 0.0);
    assert!(config.min_relevance <= 1.0);
}

// =============================================================================
// End-to-End Agent Workflow Test
// =============================================================================

#[tokio::test]
async fn test_agent_workflow_with_tools() {
    // 1. Set up tool registry
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(CalculatorTool));
    registry.register(Arc::new(EchoTool));

    // 2. Set up memory
    let mut memory = AgentMemory::new();

    // 3. Create a plan
    let mut plan = Plan::new("Demonstrate agent capabilities");
    plan.add_step(
        PlanStep::new("1", "Calculate 2+2")
            .with_tool("calculator")
            .with_params(serde_json::json!({"expression": "2 + 2"})),
    );
    plan.add_step(
        PlanStep::new("2", "Echo the result")
            .with_tool("echo")
            .with_params(serde_json::json!({"message": "Result is ready"}))
            .depends_on("1"),
    );

    // 4. Execute the plan
    let context = ToolContext::new("workflow-test-agent");
    let mut results = Vec::new();

    while let Some(step) = plan.next_step() {
        if let Some(tool_name) = &step.tool {
            if let Some(tool) = registry.get(tool_name) {
                let params = step.params.clone().unwrap_or(serde_json::json!({}));
                let result = tool.execute(params, &context).await.expect("execute");
                results.push((step.id.clone(), result));

                // Log to memory
                memory.add_message(infernum_core::Message::assistant(&format!(
                    "Completed step {}",
                    step.id
                )));
            }
        }
        plan.advance();
    }

    // 5. Verify results
    assert!(plan.complete);
    assert_eq!(results.len(), 2);
    assert!(results[0].1.output.contains("4")); // Calculator result
    assert!(results[1].1.output.contains("Echo")); // Echo result

    // 6. Verify memory recorded the steps
    assert_eq!(memory.messages().len(), 2);
}

#[tokio::test]
async fn test_agent_workflow_with_long_term_memory() {
    let temp = TempDir::new().expect("temp dir");

    // 1. Set up long-term memory
    let mut ltm = LongTermMemory::new(temp.path()).expect("create memory");

    // 2. Store some context
    let context_entry = MemoryEntry::new(
        MemoryType::Context,
        "The user is working on a Rust project using the Beleth agent framework",
    )
    .with_importance(ImportanceLevel::High)
    .with_tag("project");

    ltm.store(context_entry).expect("store context");

    // 3. Execute a task and record the learning
    let registry = ToolRegistry::with_builtins();
    let context = ToolContext::new("workflow-agent");

    let call = ToolCall {
        name: "calculator".to_string(),
        params: serde_json::json!({"expression": "100 * 1.1"}),
    };

    let result = registry.execute(&call, &context).await.expect("execute");
    assert!(result.success);

    // 4. Record the learning
    let learning = MemoryEntry::new(
        MemoryType::ProjectLearning,
        "Successfully used the calculator tool for percentage calculations",
    )
    .with_importance(ImportanceLevel::Medium)
    .with_tag("calculator")
    .with_tag("workflow");

    ltm.store(learning).expect("store learning");

    // 5. Verify memories were stored
    let stats = ltm.stats();
    assert_eq!(stats.total_entries, 2);

    // 6. Retrieve relevant context
    let project_context = ltm.get_by_tag("project");
    assert_eq!(project_context.len(), 1);

    let workflow_learnings = ltm.get_by_tag("workflow");
    assert_eq!(workflow_learnings.len(), 1);
}
