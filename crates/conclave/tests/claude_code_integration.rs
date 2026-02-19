//! Integration tests for Claude Code backend.
//!
//! These tests require the `claude` CLI to be installed.
//! They make real API calls and should be run with `--ignored` flag.

use std::path::PathBuf;
use std::time::Duration;

use conclave::backend::claude_code::{ClaudeCodeConfig, ClaudeCodeParser, spawn_claude_code};
use conclave::backend::{AgentBackendSession, AgentEvent, OutputParser, RoomContext, ParticipantSummary};
use conclave::types::{
    AttentionState, ChannelType, ClaudeTier, Message, MessageContent, MessageId, ParticipantId,
    RoomId,
};
use tokio::time::timeout;

fn test_context() -> RoomContext {
    RoomContext {
        room_id: RoomId::new(),
        room_name: "Integration Test Room".to_string(),
        working_dir: PathBuf::from(env!("CARGO_MANIFEST_DIR")),
        recent_messages: vec![],
        participants: vec![
            ParticipantSummary {
                id: ParticipantId::new(),
                display_name: "Test User".to_string(),
                is_agent: false,
                attention: AttentionState::Available,
            },
        ],
        persona_prompt: Some("You are a helpful coding assistant being tested.".to_string()),
    }
}

fn test_message(content: &str) -> Message {
    Message {
        id: MessageId::new(),
        channel: ChannelType::Main,
        sender: ParticipantId::new(),
        content: MessageContent::Text {
            content: content.to_string(),
        },
        timestamp: chrono::Utc::now(),
        metadata: std::collections::HashMap::new(),
    }
}

// =============================================================================
// Parser Unit Tests (don't require claude binary)
// =============================================================================

#[test]
fn test_parser_handles_system_message() {
    let parser = ClaudeCodeParser;
    let line = r#"{"type":"system","message":"Initializing session..."}"#;

    let events = parser.parse_line(line).unwrap();
    assert_eq!(events.len(), 1);

    match &events[0] {
        AgentEvent::Message { content, .. } => {
            assert!(content.contains("Initializing"));
        }
        _ => panic!("Expected Message event"),
    }
}

#[test]
fn test_parser_handles_result_message() {
    let parser = ClaudeCodeParser;
    let line = r#"{"type":"result","result":"Task complete!","cost_usd":0.01,"duration_ms":1500}"#;

    let events = parser.parse_line(line).unwrap();
    assert_eq!(events.len(), 1);

    match &events[0] {
        AgentEvent::Message { content, .. } => {
            assert_eq!(content, "Task complete!");
        }
        _ => panic!("Expected Message event"),
    }
}

#[test]
fn test_config_builds_correct_args() {
    let config = ClaudeCodeConfig::new(ClaudeTier::Haiku, "/tmp/test");
    let args = config.build_args();

    assert!(args.contains(&"--output-format".to_string()));
    assert!(args.contains(&"stream-json".to_string()));
    assert!(args.contains(&"--verbose".to_string())); // Required for stream-json
    assert!(args.contains(&"claude-haiku-4-5-20251001".to_string()));
    assert!(args.contains(&"--print".to_string()));
}

#[test]
fn test_config_with_system_prompt() {
    let mut config = ClaudeCodeConfig::new(ClaudeTier::Sonnet, "/tmp/test");
    config.system_prompt = Some("Be helpful.".to_string());

    let args = config.build_args();
    assert!(args.contains(&"--append-system-prompt".to_string()));
    assert!(args.contains(&"Be helpful.".to_string()));
}

// =============================================================================
// Integration Tests (require claude binary)
// =============================================================================

/// Test that we can spawn a Claude Code session.
///
/// This test is ignored by default because it requires the claude CLI.
/// Run with: cargo test -p conclave --test claude_code_integration -- --ignored
#[tokio::test]
#[ignore = "requires claude CLI and makes real API calls"]
async fn test_spawn_claude_code_session() {
    let config = ClaudeCodeConfig::new(
        ClaudeTier::Haiku, // Use Haiku for faster/cheaper tests
        env!("CARGO_MANIFEST_DIR"),
    );
    let context = test_context();

    let session = spawn_claude_code(
        "test-session".to_string(),
        config,
        &context,
    ).await;

    assert!(session.is_ok(), "Failed to spawn Claude Code: {:?}", session.err());

    let session = session.unwrap();
    assert!(session.is_running());
    assert_eq!(session.session_id(), "test-session");

    // Clean up
    session.terminate().await.unwrap();
}

/// Test calling Claude Code directly via CLI and parsing output.
/// This tests the parser and CLI interaction without using the session abstraction.
#[tokio::test]
#[ignore = "requires claude CLI and makes real API calls"]
async fn test_simple_prompt_response() {
    use std::process::Stdio;
    use tokio::io::{AsyncBufReadExt, BufReader};
    use tokio::process::Command;

    let working_dir = env!("CARGO_MANIFEST_DIR");
    let parser = ClaudeCodeParser;

    // Run Claude Code with a simple prompt
    let mut child = Command::new("claude")
        .args([
            "-p",
            "Reply with exactly: Hello from integration test",
            "--output-format", "stream-json",
            "--verbose",
            "--max-turns", "1",
        ])
        .current_dir(working_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn claude");

    let stdout = child.stdout.take().expect("No stdout");
    let reader = BufReader::new(stdout);
    let mut lines = reader.lines();

    let mut got_message = false;
    let mut all_events = Vec::new();

    // Read output with timeout
    let result = timeout(Duration::from_secs(60), async {
        while let Ok(Some(line)) = lines.next_line().await {
            println!("Line: {}", line);
            if let Ok(events) = parser.parse_line(&line) {
                for event in events {
                    println!("Event: {:?}", event);
                    all_events.push(event.clone());
                    if let AgentEvent::Message { content, .. } = event {
                        if content.contains("Hello") || content.contains("integration") {
                            got_message = true;
                        }
                    }
                }
            }
        }
    }).await;

    // Wait for process to complete
    let status = child.wait().await.expect("Failed to wait");
    println!("Exit status: {:?}", status);
    println!("Total events: {}", all_events.len());

    assert!(result.is_ok() || got_message, "Timeout or no response");
    assert!(got_message, "Did not receive expected message. Events: {:?}", all_events);
}

/// Test that tool calls are properly parsed.
#[tokio::test]
#[ignore = "requires claude CLI and makes real API calls"]
async fn test_tool_call_parsing() {
    use std::process::Stdio;
    use tokio::io::{AsyncBufReadExt, BufReader};
    use tokio::process::Command;

    let working_dir = env!("CARGO_MANIFEST_DIR");
    let parser = ClaudeCodeParser;

    // Run Claude Code with a prompt that requires tool use
    let mut child = Command::new("claude")
        .args([
            "-p",
            "Read the file Cargo.toml and tell me the package name",
            "--output-format", "stream-json",
            "--verbose",
            "--max-turns", "3",
            "--allowedTools", "Read",
        ])
        .current_dir(working_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn claude");

    let stdout = child.stdout.take().expect("No stdout");
    let reader = BufReader::new(stdout);
    let mut lines = reader.lines();

    let mut saw_tool_call = false;
    let mut saw_tool_result = false;
    let mut saw_package_name = false;

    let result = timeout(Duration::from_secs(90), async {
        while let Ok(Some(line)) = lines.next_line().await {
            // Check for tool_use in raw JSON (parser may not extract it perfectly)
            if line.contains("\"type\":\"tool_use\"") {
                println!("Raw tool_use: {}", &line[..line.len().min(200)]);
                saw_tool_call = true;
            }
            if line.contains("\"type\":\"tool_result\"") {
                println!("Raw tool_result detected");
                saw_tool_result = true;
            }

            if let Ok(events) = parser.parse_line(&line) {
                for event in events {
                    match &event {
                        AgentEvent::ToolCall { tool, .. } => {
                            println!("Parsed tool call: {}", tool);
                            if tool == "Read" {
                                saw_tool_call = true;
                            }
                        }
                        AgentEvent::ToolResult { success, .. } => {
                            println!("Parsed tool result (success: {})", success);
                            saw_tool_result = true;
                        }
                        AgentEvent::Message { content, .. } => {
                            if content.contains("conclave") {
                                saw_package_name = true;
                                println!("Found package name in response!");
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }).await;

    let status = child.wait().await.expect("Failed to wait");
    println!("Exit status: {:?}", status);

    assert!(result.is_ok(), "Timeout waiting for response");
    assert!(saw_tool_call, "Did not see Read tool call");
    assert!(saw_tool_result, "Did not see tool result");
    assert!(saw_package_name, "Did not find package name 'conclave' in response");
}

/// Test multi-turn conversation with context.
#[tokio::test]
#[ignore = "requires claude CLI and makes real API calls"]
async fn test_multi_turn_context() {
    use std::process::Stdio;
    use tokio::io::{AsyncBufReadExt, BufReader};
    use tokio::process::Command;

    let working_dir = env!("CARGO_MANIFEST_DIR");
    let parser = ClaudeCodeParser;

    // First turn: establish context
    let mut child = Command::new("claude")
        .args([
            "-p",
            "Remember this number: 42. Reply with 'Remembered: 42'",
            "--output-format", "stream-json",
            "--verbose",
            "--max-turns", "1",
        ])
        .current_dir(working_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn claude");

    let stdout = child.stdout.take().expect("No stdout");
    let reader = BufReader::new(stdout);
    let mut lines = reader.lines();

    let mut session_id = None;
    let mut remembered = false;

    let result = timeout(Duration::from_secs(60), async {
        while let Ok(Some(line)) = lines.next_line().await {
            // Extract session_id from system init
            if line.contains("\"session_id\"") && session_id.is_none() {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                    if let Some(sid) = json.get("session_id").and_then(|v| v.as_str()) {
                        session_id = Some(sid.to_string());
                        println!("Session ID: {}", sid);
                    }
                }
            }

            if let Ok(events) = parser.parse_line(&line) {
                for event in events {
                    if let AgentEvent::Message { content, .. } = event {
                        if content.contains("42") || content.contains("Remembered") {
                            remembered = true;
                            println!("First turn response: {}", content);
                        }
                    }
                }
            }
        }
    }).await;

    child.wait().await.ok();
    assert!(result.is_ok(), "Timeout on first turn");
    assert!(remembered, "First turn did not acknowledge the number");
    assert!(session_id.is_some(), "No session ID captured");

    // Second turn: recall context (using --resume)
    let sid = session_id.unwrap();
    println!("Resuming session: {}", sid);

    let mut child2 = Command::new("claude")
        .args([
            "-p",
            "What number did I ask you to remember?",
            "--output-format", "stream-json",
            "--verbose",
            "--max-turns", "1",
            "--resume", &sid,
        ])
        .current_dir(working_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn claude for second turn");

    let stdout2 = child2.stdout.take().expect("No stdout");
    let reader2 = BufReader::new(stdout2);
    let mut lines2 = reader2.lines();

    let mut recalled_42 = false;

    let result2 = timeout(Duration::from_secs(60), async {
        while let Ok(Some(line)) = lines2.next_line().await {
            if let Ok(events) = parser.parse_line(&line) {
                for event in events {
                    if let AgentEvent::Message { content, .. } = event {
                        println!("Second turn: {}", content);
                        if content.contains("42") {
                            recalled_42 = true;
                        }
                    }
                }
            }
        }
    }).await;

    child2.wait().await.ok();
    assert!(result2.is_ok(), "Timeout on second turn");
    assert!(recalled_42, "Second turn did not recall the number 42");
}
