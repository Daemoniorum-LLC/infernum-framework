//! End-to-end integration tests for the LoopExecutor.
//!
//! Tests the full Generate->Detect->Execute->Integrate cycle using a
//! ScriptedEngine mock and real tools.
//!
//! Each test validates a specific behavior of the agentic loop:
//! - Answer detection (explicit and implicit)
//! - Tool call execution cycles
//! - Autonomy enforcement (forbidden, requires-approval)
//! - Meta-signal detection (stuck, yield, thinking)
//! - Resource limits (max iterations, tool call limit)
//! - Event stream completeness

use std::sync::Arc;

use async_trait::async_trait;
use infernum_core::{
    EmbedRequest, EmbedResponse, GenerateRequest, GenerateResponse, ModelArchitecture, ModelId,
    ModelMetadata, ModelSource, RequestId, Result, TokenStream, Usage, model::LlamaVersion,
    response::Choice,
};
use parking_lot::Mutex;
use tokio::sync::mpsc;

use abaddon::InferenceEngine;
use beleth::{
    AutonomyGrant, ExecutorConfig, IterationOutcome, LoopConfig, LoopEvent,
    LoopExecutor, NaturalTermination, ResourceTermination, TerminationReason, ToolPattern,
    ToolRegistry,
};

// =============================================================================
// ScriptedEngine — mock InferenceEngine returning predetermined responses
// =============================================================================

struct ScriptedEngine {
    responses: Mutex<Vec<String>>,
    metadata: ModelMetadata,
    call_count: Mutex<usize>,
}

impl ScriptedEngine {
    fn new(responses: Vec<String>) -> Self {
        Self {
            responses: Mutex::new(responses),
            metadata: ModelMetadata::builder(
                "test-model",
                ModelArchitecture::Llama {
                    version: LlamaVersion::V3,
                },
            )
            .source(ModelSource::local("/tmp/test-model"))
            .build(),
            call_count: Mutex::new(0),
        }
    }

    fn make_response(text: &str) -> GenerateResponse {
        GenerateResponse {
            request_id: RequestId::new(),
            created: 0,
            model: ModelId::new("test-model"),
            choices: vec![Choice {
                index: 0,
                text: text.to_string(),
                message: None,
                finish_reason: None,
                logprobs: None,
            }],
            usage: Usage::new(10, 20),
            time_to_first_token_ms: None,
            total_time_ms: None,
        }
    }

    fn call_count(&self) -> usize {
        *self.call_count.lock()
    }
}

#[async_trait]
impl InferenceEngine for ScriptedEngine {
    async fn generate(&self, _request: GenerateRequest) -> Result<GenerateResponse> {
        let responses = self.responses.lock();
        let mut count = self.call_count.lock();
        let idx = *count;
        *count += 1;

        if idx >= responses.len() {
            // Safety net: return final answer to prevent infinite loop
            return Ok(Self::make_response(
                "<answer confidence=\"0.5\">Ran out of scripted responses</answer>",
            ));
        }

        Ok(Self::make_response(&responses[idx]))
    }

    async fn generate_stream(&self, _request: GenerateRequest) -> Result<TokenStream> {
        Ok(TokenStream::empty())
    }

    async fn embed(&self, _request: EmbedRequest) -> Result<EmbedResponse> {
        Err(infernum_core::Error::internal("Not supported in mock"))
    }

    fn model_info(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn is_ready(&self) -> bool {
        true
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Config with default autonomy (requires approval for all tools).
fn make_config(session_id: &str) -> ExecutorConfig {
    ExecutorConfig::new(session_id).with_system_prompt("You are a test assistant.")
}

/// Config with permissive autonomy (auto-approve all tools).
fn make_permissive_config(session_id: &str, dir: &std::path::Path) -> ExecutorConfig {
    ExecutorConfig::new(session_id)
        .with_system_prompt("You are a test assistant.")
        .with_working_dir(dir)
        .with_autonomy(
            AutonomyGrant::builder()
                .allow(ToolPattern::Tool("*".to_string()))
                .build(),
        )
}

/// Collects all events from the channel after the sender is dropped.
async fn collect_events(mut rx: mpsc::Receiver<LoopEvent>) -> Vec<LoopEvent> {
    let mut events = Vec::new();
    while let Some(event) = rx.recv().await {
        events.push(event);
    }
    events
}

fn event_name(e: &LoopEvent) -> &'static str {
    match e {
        LoopEvent::LoopStarted { .. } => "LoopStarted",
        LoopEvent::IterationStarted { .. } => "IterationStarted",
        LoopEvent::GenerationCompleted { .. } => "GenerationCompleted",
        LoopEvent::ToolCallDetected { .. } => "ToolCallDetected",
        LoopEvent::ToolExecutionStarted { .. } => "ToolExecutionStarted",
        LoopEvent::ToolExecutionCompleted { .. } => "ToolExecutionCompleted",
        LoopEvent::ToolApprovalRequired { .. } => "ToolApprovalRequired",
        LoopEvent::IterationCompleted { .. } => "IterationCompleted",
        LoopEvent::MetaSignalDetected { .. } => "MetaSignalDetected",
        LoopEvent::LoopCompleted { .. } => "LoopCompleted",
        LoopEvent::TokenGenerated { .. } => "TokenGenerated",
        LoopEvent::ContextCompressed { .. } => "ContextCompressed",
        LoopEvent::Error { .. } => "Error",
    }
}

// =============================================================================
// Tests
// =============================================================================

/// Explicit answer via <answer> tag — loop terminates in 1 iteration.
#[tokio::test]
async fn test_executor_explicit_answer() {
    let engine = Arc::new(ScriptedEngine::new(vec![
        r#"<answer confidence="0.95">The answer is 42.</answer>"#.to_string(),
    ]));

    let tools = Arc::new(ToolRegistry::with_builtins());
    let config = make_config("test-explicit-answer");
    let executor = LoopExecutor::new(engine.clone(), tools, config);
    let (tx, rx) = mpsc::channel(64);

    let summary = executor
        .run("What is the answer?", tx)
        .await
        .expect("run should succeed");

    assert_eq!(summary.iterations_completed, 1);
    assert_eq!(summary.tool_calls_made, 0);
    match &summary.termination {
        TerminationReason::Natural(NaturalTermination::AnswerProvided {
            confidence, ..
        }) => {
            assert!(*confidence > 0.9, "Expected high confidence, got {confidence}");
        }
        other => panic!("Expected AnswerProvided, got {other:?}"),
    }
    assert_eq!(engine.call_count(), 1);

    let events = collect_events(rx).await;
    assert!(events
        .iter()
        .any(|e| matches!(e, LoopEvent::LoopStarted { .. })));
    assert!(events
        .iter()
        .any(|e| matches!(e, LoopEvent::LoopCompleted { .. })));
    assert!(events
        .iter()
        .any(|e| matches!(e, LoopEvent::MetaSignalDetected { .. })));
}

/// Implicit answer — model outputs plain text with no tools or meta-signals.
#[tokio::test]
async fn test_executor_implicit_answer() {
    let engine = Arc::new(ScriptedEngine::new(vec![
        "Rust ownership is a memory management system that ensures memory safety without a garbage collector."
            .to_string(),
    ]));

    let tools = Arc::new(ToolRegistry::with_builtins());
    // Disable implicit signals so plain text doesn't accidentally match uncertainty patterns
    let config = make_config("test-implicit-answer").with_loop_config(LoopConfig {
        detect_implicit_signals: false,
        ..LoopConfig::default()
    });

    let executor = LoopExecutor::new(engine.clone(), tools, config);
    let (tx, _rx) = mpsc::channel(64);

    let summary = executor
        .run("Explain Rust ownership.", tx)
        .await
        .expect("run should succeed");

    assert_eq!(summary.iterations_completed, 1);
    assert_eq!(summary.tool_calls_made, 0);
    // No tools, no meta-signals → treated as implicit answer at 0.5 confidence
    match &summary.termination {
        TerminationReason::Natural(NaturalTermination::AnswerProvided {
            confidence, ..
        }) => {
            assert!(
                (*confidence - 0.5).abs() < 0.01,
                "Expected 0.5 confidence, got {confidence}"
            );
        }
        other => panic!("Expected AnswerProvided, got {other:?}"),
    }
}

/// Tool call cycle — model calls read_file, gets result, then answers.
#[tokio::test]
async fn test_executor_tool_call_and_answer() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file_path = dir.path().join("test.txt");
    std::fs::write(&file_path, "Hello from the test file!\n").expect("write seed file");
    let file_path_str = file_path.to_string_lossy().to_string();

    let engine = Arc::new(ScriptedEngine::new(vec![
        // Iteration 1: call read_file
        format!(
            "Let me read that file.\n\
             <tool_call>\n\
             {{\"name\": \"read_file\", \"arguments\": {{\"path\": \"{file_path_str}\"}}}}\n\
             </tool_call>"
        ),
        // Iteration 2: answer with the file contents
        r#"<answer confidence="0.9">The file contains: Hello from the test file!</answer>"#
            .to_string(),
    ]));

    let tools = Arc::new(ToolRegistry::with_code_tools());
    let config = make_permissive_config("test-tool-call", dir.path()).with_loop_config(LoopConfig {
        detect_implicit_signals: false,
        ..LoopConfig::default()
    });

    let executor = LoopExecutor::new(engine.clone(), tools, config);
    let (tx, rx) = mpsc::channel(64);

    let summary = executor
        .run("Read test.txt", tx)
        .await
        .expect("run should succeed");

    assert_eq!(summary.iterations_completed, 2);
    assert_eq!(summary.tool_calls_made, 1);
    assert!(matches!(
        summary.termination,
        TerminationReason::Natural(NaturalTermination::AnswerProvided { .. })
    ));
    assert_eq!(engine.call_count(), 2);

    // Check events include tool execution
    let events = collect_events(rx).await;
    assert!(events.iter().any(
        |e| matches!(e, LoopEvent::ToolCallDetected { tool, .. } if tool == "read_file")
    ));
    assert!(events
        .iter()
        .any(|e| matches!(e, LoopEvent::ToolExecutionStarted { .. })));
    assert!(events
        .iter()
        .any(|e| matches!(e, LoopEvent::ToolExecutionCompleted { .. })));
}

/// Multiple tool calls in one iteration.
#[tokio::test]
async fn test_executor_multiple_tool_calls_per_iteration() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file_a = dir.path().join("a.txt");
    let file_b = dir.path().join("b.txt");
    std::fs::write(&file_a, "Content A\n").expect("write a");
    std::fs::write(&file_b, "Content B\n").expect("write b");
    let path_a = file_a.to_string_lossy().to_string();
    let path_b = file_b.to_string_lossy().to_string();

    let engine = Arc::new(ScriptedEngine::new(vec![
        // Iteration 1: two tool calls
        format!(
            "I'll read both files.\n\
             <tool_call>\n\
             {{\"name\": \"read_file\", \"arguments\": {{\"path\": \"{path_a}\"}}}}\n\
             </tool_call>\n\
             <tool_call>\n\
             {{\"name\": \"read_file\", \"arguments\": {{\"path\": \"{path_b}\"}}}}\n\
             </tool_call>"
        ),
        // Iteration 2: answer
        r#"<answer confidence="0.85">File A: Content A, File B: Content B</answer>"#.to_string(),
    ]));

    let tools = Arc::new(ToolRegistry::with_code_tools());
    let config =
        make_permissive_config("test-multi-tool", dir.path()).with_loop_config(LoopConfig {
            detect_implicit_signals: false,
            ..LoopConfig::default()
        });

    let executor = LoopExecutor::new(engine.clone(), tools, config);
    let (tx, _rx) = mpsc::channel(64);

    let summary = executor
        .run("Read both files", tx)
        .await
        .expect("run should succeed");

    assert_eq!(summary.iterations_completed, 2);
    assert_eq!(summary.tool_calls_made, 2);
    assert!(matches!(
        summary.termination,
        TerminationReason::Natural(NaturalTermination::AnswerProvided { .. })
    ));
}

/// Write then read — verify tool results feed back into the conversation.
#[tokio::test]
async fn test_executor_write_then_read() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file_path = dir.path().join("created.txt");
    let path_str = file_path.to_string_lossy().to_string();

    let engine = Arc::new(ScriptedEngine::new(vec![
        // Iteration 1: write a file
        format!(
            "<tool_call>\n\
             {{\"name\": \"write_file\", \"arguments\": {{\"path\": \"{path_str}\", \"content\": \"Hello from executor!\"}}}}\n\
             </tool_call>"
        ),
        // Iteration 2: read it back
        format!(
            "<tool_call>\n\
             {{\"name\": \"read_file\", \"arguments\": {{\"path\": \"{path_str}\"}}}}\n\
             </tool_call>"
        ),
        // Iteration 3: answer
        r#"<answer confidence="0.95">File created and verified.</answer>"#.to_string(),
    ]));

    let tools = Arc::new(ToolRegistry::with_code_tools());
    let config =
        make_permissive_config("test-write-read", dir.path()).with_loop_config(LoopConfig {
            detect_implicit_signals: false,
            ..LoopConfig::default()
        });

    let executor = LoopExecutor::new(engine.clone(), tools, config);
    let (tx, _rx) = mpsc::channel(64);

    let summary = executor
        .run("Create and verify a file", tx)
        .await
        .expect("run should succeed");

    assert_eq!(summary.iterations_completed, 3);
    assert_eq!(summary.tool_calls_made, 2);
    assert!(matches!(
        summary.termination,
        TerminationReason::Natural(NaturalTermination::AnswerProvided { .. })
    ));

    // Verify the file was actually written to disk
    let content = std::fs::read_to_string(&file_path).expect("read created file");
    assert_eq!(content, "Hello from executor!");
}

/// Autonomy forbidden — model tries a forbidden tool, gets denied.
#[tokio::test]
async fn test_executor_forbidden_tool() {
    let engine = Arc::new(ScriptedEngine::new(vec![
        // Iteration 1: try to call bash (which is forbidden)
        r#"Let me run a command.
<tool_call>
{"name": "bash", "arguments": {"command": "echo hello"}}
</tool_call>"#
            .to_string(),
        // Iteration 2: answer after seeing the denial
        r#"<answer confidence="0.7">I cannot run bash commands.</answer>"#.to_string(),
    ]));

    let dir = tempfile::tempdir().expect("tempdir");
    let tools = Arc::new(ToolRegistry::with_code_tools());
    let config = ExecutorConfig::new("test-forbidden")
        .with_system_prompt("You are a test assistant.")
        .with_working_dir(dir.path())
        .with_autonomy(
            AutonomyGrant::builder()
                .allow(ToolPattern::Tool("read_file".to_string()))
                .forbid(ToolPattern::Tool("bash".to_string()))
                .build(),
        )
        .with_loop_config(LoopConfig {
            detect_implicit_signals: false,
            ..LoopConfig::default()
        });

    let executor = LoopExecutor::new(engine.clone(), tools, config);
    let (tx, rx) = mpsc::channel(64);

    let summary = executor
        .run("Run echo hello", tx)
        .await
        .expect("run should succeed");

    assert_eq!(summary.iterations_completed, 2);
    assert_eq!(summary.tool_calls_made, 1); // Still counted as detected
    assert!(matches!(
        summary.termination,
        TerminationReason::Natural(NaturalTermination::AnswerProvided { .. })
    ));

    // Check events: bash detected but no ToolExecutionStarted (forbidden)
    let events = collect_events(rx).await;
    assert!(events
        .iter()
        .any(|e| matches!(e, LoopEvent::ToolCallDetected { tool, .. } if tool == "bash")));
    let exec_started = events
        .iter()
        .filter(|e| matches!(e, LoopEvent::ToolExecutionStarted { .. }))
        .count();
    assert_eq!(
        exec_started, 0,
        "Forbidden tool should not emit ToolExecutionStarted"
    );
}

/// Default autonomy requires approval — tool denied with RequiresApproval.
#[tokio::test]
async fn test_executor_requires_approval() {
    let engine = Arc::new(ScriptedEngine::new(vec![
        // Iteration 1: try to call a tool (default = requires approval)
        r#"<tool_call>
{"name": "read_file", "arguments": {"path": "/tmp/test.txt"}}
</tool_call>"#
            .to_string(),
        // Iteration 2: answer after seeing the denial
        r#"<answer confidence="0.6">I need approval to use that tool.</answer>"#.to_string(),
    ]));

    let tools = Arc::new(ToolRegistry::with_code_tools());
    // Default autonomy grant = everything requires approval
    let config = make_config("test-approval").with_loop_config(LoopConfig {
        detect_implicit_signals: false,
        ..LoopConfig::default()
    });

    let executor = LoopExecutor::new(engine.clone(), tools, config);
    let (tx, rx) = mpsc::channel(64);

    let summary = executor
        .run("Read a file", tx)
        .await
        .expect("run should succeed");

    assert_eq!(summary.iterations_completed, 2);

    // Check that approval was required
    let events = collect_events(rx).await;
    let approval_required = events.iter().any(|e| {
        matches!(e, LoopEvent::ToolApprovalRequired { tool, .. } if tool == "read_file")
    });
    assert!(
        approval_required,
        "Should emit ToolApprovalRequired event"
    );
}

/// Stuck signal — model outputs <stuck> tag, loop terminates.
#[tokio::test]
async fn test_executor_stuck_signal() {
    let engine = Arc::new(ScriptedEngine::new(vec![
        r#"<stuck>
<attempt>Tried reading the documentation</attempt>
<hypothesis>The API might have changed</hypothesis>
<request>I need clarification on the new API format</request>
</stuck>"#
            .to_string(),
    ]));

    let tools = Arc::new(ToolRegistry::with_builtins());
    let config = make_config("test-stuck");
    let executor = LoopExecutor::new(engine.clone(), tools, config);
    let (tx, rx) = mpsc::channel(64);

    let summary = executor
        .run("Fix the API integration", tx)
        .await
        .expect("run should succeed");

    assert_eq!(summary.iterations_completed, 1);
    assert!(matches!(
        summary.termination,
        TerminationReason::Natural(NaturalTermination::AgentStuck { .. })
    ));
    // Stuck loops are resumable (can_resume = !is_terminal || Stuck | Yielded)
    assert!(summary.can_resume, "Stuck loops should be resumable");

    let events = collect_events(rx).await;
    assert!(events
        .iter()
        .any(|e| matches!(e, LoopEvent::MetaSignalDetected { .. })));
    assert!(events.iter().any(|e| matches!(
        e,
        LoopEvent::IterationCompleted {
            outcome: IterationOutcome::Stuck,
            ..
        }
    )));
}

/// Yield signal — model outputs <yield> tag, loop terminates.
#[tokio::test]
async fn test_executor_yield_signal() {
    let engine = Arc::new(ScriptedEngine::new(vec![r#"<yield>
<partial>I've identified the problem is in the auth module</partial>
<expertise>security specialist</expertise>
</yield>"#
        .to_string()]));

    let tools = Arc::new(ToolRegistry::with_builtins());
    let config = make_config("test-yield");

    let executor = LoopExecutor::new(engine.clone(), tools, config);
    let (tx, rx) = mpsc::channel(64);

    let summary = executor
        .run("Review the security config", tx)
        .await
        .expect("run should succeed");

    assert_eq!(summary.iterations_completed, 1);
    assert!(matches!(
        summary.termination,
        TerminationReason::Natural(NaturalTermination::AgentYielded { .. })
    ));
    assert!(summary.can_resume, "Yielded loops should be resumable");

    let events = collect_events(rx).await;
    assert!(events.iter().any(|e| matches!(
        e,
        LoopEvent::IterationCompleted {
            outcome: IterationOutcome::Yielded,
            ..
        }
    )));
}

/// Thinking signal is non-terminal — loop continues to tool detection.
#[tokio::test]
async fn test_executor_thinking_signal_continues() {
    let engine = Arc::new(ScriptedEngine::new(vec![
        // Output with thinking tag but no tools → implicit answer
        r#"<thinking direction="analyzing">Let me consider the options.</thinking>
The architecture uses a layered approach with clear separation of concerns."#
            .to_string(),
    ]));

    let tools = Arc::new(ToolRegistry::with_builtins());
    let config = make_config("test-thinking").with_loop_config(LoopConfig {
        detect_implicit_signals: false,
        ..LoopConfig::default()
    });

    let executor = LoopExecutor::new(engine.clone(), tools, config);
    let (tx, rx) = mpsc::channel(64);

    let summary = executor
        .run("Analyze the architecture", tx)
        .await
        .expect("run should succeed");

    // Thinking is non-terminal, no tool calls → implicit answer
    assert_eq!(summary.iterations_completed, 1);
    assert!(matches!(
        summary.termination,
        TerminationReason::Natural(NaturalTermination::AnswerProvided { .. })
    ));

    let events = collect_events(rx).await;
    assert!(events
        .iter()
        .any(|e| matches!(e, LoopEvent::MetaSignalDetected { .. })));
}

/// Max iterations — loop terminates when iteration limit is reached.
#[tokio::test]
async fn test_executor_max_iterations() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file_path = dir.path().join("loop.txt");
    std::fs::write(&file_path, "test content\n").expect("write");
    let path_str = file_path.to_string_lossy().to_string();

    // Model keeps making tool calls, never provides an answer
    let engine = Arc::new(ScriptedEngine::new(vec![
        format!(
            "<tool_call>\n\
             {{\"name\": \"read_file\", \"arguments\": {{\"path\": \"{path_str}\"}}}}\n\
             </tool_call>"
        ),
        format!(
            "<tool_call>\n\
             {{\"name\": \"read_file\", \"arguments\": {{\"path\": \"{path_str}\"}}}}\n\
             </tool_call>"
        ),
        format!(
            "<tool_call>\n\
             {{\"name\": \"read_file\", \"arguments\": {{\"path\": \"{path_str}\"}}}}\n\
             </tool_call>"
        ),
        // Extra: safety net in case limit doesn't trigger
        r#"<answer confidence="0.5">Done</answer>"#.to_string(),
    ]));

    let tools = Arc::new(ToolRegistry::with_code_tools());
    let config =
        make_permissive_config("test-max-iter", dir.path()).with_loop_config(LoopConfig {
            max_iterations: 2, // Only allow 2 iterations
            detect_implicit_signals: false,
            ..LoopConfig::default()
        });

    let executor = LoopExecutor::new(engine.clone(), tools, config);
    let (tx, _rx) = mpsc::channel(64);

    let summary = executor
        .run("Read the file repeatedly", tx)
        .await
        .expect("run should succeed");

    // Should complete 2 iterations, then hit the limit at continue_loop
    assert_eq!(summary.iterations_completed, 2);
    assert!(matches!(
        summary.termination,
        TerminationReason::Resource(ResourceTermination::MaxIterations {
            completed: 2,
            limit: 2
        })
    ));
}

/// Tool call limit — too many tool calls triggers resource termination.
#[tokio::test]
async fn test_executor_tool_call_limit() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file_path = dir.path().join("limit.txt");
    std::fs::write(&file_path, "content\n").expect("write");
    let path_str = file_path.to_string_lossy().to_string();

    // Model makes 3 tool calls per iteration; limit is 4 total
    let engine = Arc::new(ScriptedEngine::new(vec![
        // Iteration 1: 3 tool calls (under limit of 4)
        format!(
            "<tool_call>{{\"name\": \"read_file\", \"arguments\": {{\"path\": \"{path_str}\"}}}}</tool_call>\n\
             <tool_call>{{\"name\": \"read_file\", \"arguments\": {{\"path\": \"{path_str}\"}}}}</tool_call>\n\
             <tool_call>{{\"name\": \"read_file\", \"arguments\": {{\"path\": \"{path_str}\"}}}}</tool_call>"
        ),
        // Iteration 2: 3 more calls -> new_total = 6 > 4 -> terminated
        format!(
            "<tool_call>{{\"name\": \"read_file\", \"arguments\": {{\"path\": \"{path_str}\"}}}}</tool_call>\n\
             <tool_call>{{\"name\": \"read_file\", \"arguments\": {{\"path\": \"{path_str}\"}}}}</tool_call>\n\
             <tool_call>{{\"name\": \"read_file\", \"arguments\": {{\"path\": \"{path_str}\"}}}}</tool_call>"
        ),
    ]));

    let tools = Arc::new(ToolRegistry::with_code_tools());
    let config =
        make_permissive_config("test-tool-limit", dir.path()).with_loop_config(LoopConfig {
            max_tool_calls: 4,
            detect_implicit_signals: false,
            ..LoopConfig::default()
        });

    let executor = LoopExecutor::new(engine.clone(), tools, config);
    let (tx, _rx) = mpsc::channel(64);

    let summary = executor
        .run("Read a lot", tx)
        .await
        .expect("run should succeed");

    assert!(matches!(
        summary.termination,
        TerminationReason::Resource(ResourceTermination::ToolCallLimitReached { .. })
    ));
}

/// Event stream completeness — verify the expected event sequence.
#[tokio::test]
async fn test_executor_event_stream_completeness() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file_path = dir.path().join("events.txt");
    std::fs::write(&file_path, "event test\n").expect("write");
    let path_str = file_path.to_string_lossy().to_string();

    let engine = Arc::new(ScriptedEngine::new(vec![
        // Iteration 1: tool call
        format!(
            "<tool_call>\n\
             {{\"name\": \"read_file\", \"arguments\": {{\"path\": \"{path_str}\"}}}}\n\
             </tool_call>"
        ),
        // Iteration 2: answer
        r#"<answer confidence="0.8">Found: event test</answer>"#.to_string(),
    ]));

    let tools = Arc::new(ToolRegistry::with_code_tools());
    let config =
        make_permissive_config("test-events", dir.path()).with_loop_config(LoopConfig {
            detect_implicit_signals: false,
            ..LoopConfig::default()
        });

    let executor = LoopExecutor::new(engine.clone(), tools, config);
    let (tx, rx) = mpsc::channel(64);

    let _summary = executor
        .run("Read the file", tx)
        .await
        .expect("run should succeed");

    let events = collect_events(rx).await;
    let names: Vec<&str> = events.iter().map(event_name).collect();

    // Expected event sequence for a 2-iteration tool->answer cycle:
    // LoopStarted
    // IterationStarted (iter 1)
    // GenerationCompleted
    // ToolCallDetected
    // ToolExecutionStarted
    // ToolExecutionCompleted
    // IterationCompleted (ToolCallsExecuted)
    // IterationStarted (iter 2)
    // GenerationCompleted
    // MetaSignalDetected
    // IterationCompleted (AnswerProvided)
    // LoopCompleted

    // Verify first and last
    assert_eq!(names[0], "LoopStarted");
    assert_eq!(*names.last().expect("events"), "LoopCompleted");

    // Count specific events
    let gen_count = names.iter().filter(|&&n| n == "GenerationCompleted").count();
    assert_eq!(gen_count, 2, "Should have 2 generation completions");

    let iter_starts = names.iter().filter(|&&n| n == "IterationStarted").count();
    assert_eq!(iter_starts, 2, "Should have 2 iteration starts");

    let tool_detections = names.iter().filter(|&&n| n == "ToolCallDetected").count();
    assert_eq!(tool_detections, 1, "Should have 1 tool detection");

    let tool_execs = names
        .iter()
        .filter(|&&n| n == "ToolExecutionCompleted")
        .count();
    assert_eq!(tool_execs, 1, "Should have 1 tool execution completion");
}

/// Answer with caveats — caveats are extracted from the answer tag.
#[tokio::test]
async fn test_executor_answer_with_caveats() {
    let engine = Arc::new(ScriptedEngine::new(vec![
        r#"<answer confidence="0.7">The result is approximately 3.14.<caveat>Rounded to 2 decimal places</caveat></answer>"#
            .to_string(),
    ]));

    let tools = Arc::new(ToolRegistry::with_builtins());
    let config = make_config("test-caveats");
    let executor = LoopExecutor::new(engine.clone(), tools, config);
    let (tx, _rx) = mpsc::channel(64);

    let summary = executor
        .run("Calculate pi", tx)
        .await
        .expect("run should succeed");

    match &summary.termination {
        TerminationReason::Natural(NaturalTermination::AnswerProvided {
            confidence,
            answer,
        }) => {
            assert!(
                (*confidence - 0.7).abs() < 0.01,
                "Expected 0.7 confidence, got {confidence}"
            );
            assert!(
                answer.contains("3.14"),
                "Answer should contain the result: {answer}"
            );
        }
        other => panic!("Expected AnswerProvided, got {other:?}"),
    }
}

/// Tool error recovery — tool fails, model sees error and adapts.
#[tokio::test]
async fn test_executor_tool_error_recovery() {
    let dir = tempfile::tempdir().expect("tempdir");
    let missing = dir.path().join("nonexistent.txt");
    let missing_str = missing.to_string_lossy().to_string();

    let engine = Arc::new(ScriptedEngine::new(vec![
        // Iteration 1: try to read a file that doesn't exist
        format!(
            "<tool_call>\n\
             {{\"name\": \"read_file\", \"arguments\": {{\"path\": \"{missing_str}\"}}}}\n\
             </tool_call>"
        ),
        // Iteration 2: acknowledge the error
        r#"<answer confidence="0.8">The file does not exist.</answer>"#.to_string(),
    ]));

    let tools = Arc::new(ToolRegistry::with_code_tools());
    let config =
        make_permissive_config("test-error-recovery", dir.path()).with_loop_config(LoopConfig {
            detect_implicit_signals: false,
            ..LoopConfig::default()
        });

    let executor = LoopExecutor::new(engine.clone(), tools, config);
    let (tx, rx) = mpsc::channel(64);

    let summary = executor
        .run("Read nonexistent.txt", tx)
        .await
        .expect("run should succeed even on tool error");

    assert_eq!(summary.iterations_completed, 2);
    assert_eq!(summary.tool_calls_made, 1);
    assert!(matches!(
        summary.termination,
        TerminationReason::Natural(NaturalTermination::AnswerProvided { .. })
    ));

    // Verify the tool result was captured as failed
    let events = collect_events(rx).await;
    let tool_completed = events.iter().find(|e| matches!(e, LoopEvent::ToolExecutionCompleted { .. }));
    assert!(tool_completed.is_some(), "Should have ToolExecutionCompleted event");
}

/// Bash tool execution through the executor.
#[tokio::test]
async fn test_executor_bash_tool() {
    let dir = tempfile::tempdir().expect("tempdir");

    let engine = Arc::new(ScriptedEngine::new(vec![
        // Iteration 1: run a bash command
        r#"<tool_call>
{"name": "bash", "arguments": {"command": "echo hello-from-bash"}}
</tool_call>"#
            .to_string(),
        // Iteration 2: answer
        r#"<answer confidence="0.9">The command output: hello-from-bash</answer>"#.to_string(),
    ]));

    let tools = Arc::new(ToolRegistry::with_code_tools());
    let config =
        make_permissive_config("test-bash", dir.path()).with_loop_config(LoopConfig {
            detect_implicit_signals: false,
            ..LoopConfig::default()
        });

    let executor = LoopExecutor::new(engine.clone(), tools, config);
    let (tx, _rx) = mpsc::channel(64);

    let summary = executor
        .run("Run echo hello-from-bash", tx)
        .await
        .expect("run should succeed");

    assert_eq!(summary.iterations_completed, 2);
    assert_eq!(summary.tool_calls_made, 1);
    assert!(matches!(
        summary.termination,
        TerminationReason::Natural(NaturalTermination::AnswerProvided { .. })
    ));
}
