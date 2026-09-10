//! End-to-end tests driving [`LoopExecutor`] through the OpenAI-compatible
//! HTTP backend against a mock server.
//!
//! These prove the agentic loop can run on an out-of-process inference server
//! **without any model artifact on disk** — no GGUF, no HuggingFace download.
//! The mock speaks the same OpenAI dialect as `llama-server` and vLLM, so a
//! passing run here means the wiring is correct and only model *quality*
//! remains unverified.
//!
//! Companion to `executor_e2e.rs`, which covers the same loop against an
//! in-process `ScriptedEngine`. The difference under test here is the
//! transport, not the loop.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use abaddon::openai_engine::{OpenAiConfig, OpenAiEngine};
use abaddon::InferenceEngine;
use beleth::{
    AutonomyGrant, ExecutorConfig, LoopConfig, LoopEvent, LoopExecutor, NaturalTermination,
    TerminationReason, ToolPattern, ToolRegistry,
};
use infernum_core::{GenerateRequest, Message, PromptInput};
use serde_json::json;
use tokio::sync::mpsc;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

// =============================================================================
// Scripted responder — returns canned completions in order
// =============================================================================

/// Serves a fixed sequence of assistant messages, one per request.
///
/// Mirrors `executor_e2e.rs`'s `ScriptedEngine`, but over HTTP. Requests past
/// the end of the script get a `stop` with empty content, which terminates the
/// loop rather than hanging it.
struct ScriptedResponder {
    replies: Vec<String>,
    calls: Arc<AtomicUsize>,
    /// Captures each request body so tests can assert on the wire format.
    seen: Arc<parking_lot::Mutex<Vec<serde_json::Value>>>,
}

impl Respond for ScriptedResponder {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        if let Ok(body) = serde_json::from_slice::<serde_json::Value>(&request.body) {
            self.seen.lock().push(body);
        }

        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        let content = self.replies.get(n).cloned().unwrap_or_default();

        ResponseTemplate::new(200).set_body_json(json!({
            "id": format!("chatcmpl-{n}"),
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": "mock-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }))
    }
}

/// Starts a mock server that answers `POST /v1/chat/completions` from `replies`.
///
/// Returns the server (which must be kept alive), the engine, and the shared
/// capture buffer of request bodies.
async fn scripted_server(
    replies: Vec<&str>,
) -> (
    MockServer,
    Arc<OpenAiEngine>,
    Arc<parking_lot::Mutex<Vec<serde_json::Value>>>,
) {
    let server = MockServer::start().await;
    let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ScriptedResponder {
            replies: replies.into_iter().map(String::from).collect(),
            calls: Arc::new(AtomicUsize::new(0)),
            seen: Arc::clone(&seen),
        })
        .mount(&server)
        .await;

    // Minimal /models so probe() succeeds, as a real server would.
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [{"id": "mock-model", "object": "model"}]
        })))
        .mount(&server)
        .await;

    let config = OpenAiConfig::builder(format!("{}/v1", server.uri()), "mock-model").build();
    let engine = Arc::new(
        OpenAiEngine::connect(config)
            .await
            .expect("engine should connect to mock server"),
    );

    (server, engine, seen)
}

/// Permissive config so tools run without an approval gate attached.
fn permissive_config(session_id: &str, dir: &std::path::Path) -> ExecutorConfig {
    ExecutorConfig::new(session_id)
        .with_working_dir(dir)
        .with_autonomy(
            AutonomyGrant::builder()
                .allow(ToolPattern::Tool("*".to_string()))
                .build(),
        )
        .with_loop_config(LoopConfig {
            detect_implicit_signals: false,
            ..LoopConfig::default()
        })
}

async fn collect_events(mut rx: mpsc::Receiver<LoopEvent>) -> Vec<LoopEvent> {
    let mut out = Vec::new();
    while let Some(e) = rx.recv().await {
        out.push(e);
    }
    out
}

// =============================================================================
// Engine-level tests
// =============================================================================

/// The engine reports not-ready until a successful probe, then ready.
#[tokio::test]
async fn connect_probes_models_and_marks_ready() {
    let (_server, engine, _) = scripted_server(vec![]).await;
    assert!(
        engine.is_ready(),
        "connect() should have probed and readied"
    );
}

/// A chat completion round-trips into a `GenerateResponse` whose `text` is
/// populated — the field beleth's executor actually reads.
#[tokio::test]
async fn generate_populates_choice_text() {
    let (_server, engine, _) = scripted_server(vec!["hello from the mock"]).await;

    let response = engine
        .generate(GenerateRequest::chat(vec![Message::user("hi")]))
        .await
        .expect("generate should succeed");

    assert_eq!(response.choices[0].text, "hello from the mock");
    assert_eq!(response.usage.total_tokens, 15);
}

/// Conversation history reaches the server as OpenAI-shaped messages.
#[tokio::test]
async fn messages_serialize_to_openai_wire_format() {
    let (_server, engine, seen) = scripted_server(vec!["ok"]).await;

    engine
        .generate(GenerateRequest::chat(vec![
            Message::system("you are terse"),
            Message::user("ping"),
        ]))
        .await
        .expect("generate should succeed");

    let body = seen.lock()[0].clone();
    assert_eq!(body["model"], "mock-model");
    assert_eq!(body["stream"], false);
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][0]["content"], "you are terse");
    assert_eq!(body["messages"][1]["role"], "user");
    assert_eq!(body["messages"][1]["content"], "ping");
}

/// A non-2xx status surfaces as an error rather than an empty completion.
#[tokio::test]
async fn server_error_surfaces_as_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("model not loaded"))
        .mount(&server)
        .await;

    let engine =
        OpenAiEngine::new(OpenAiConfig::builder(format!("{}/v1", server.uri()), "m").build())
            .expect("client builds");

    let err = engine
        .generate(GenerateRequest::chat(vec![Message::user("hi")]))
        .await
        .expect_err("500 should be an error");

    let msg = err.to_string();
    assert!(msg.contains("500"), "error should name the status: {msg}");
}

/// Streaming decodes SSE into ordered text deltas.
#[tokio::test]
async fn generate_stream_decodes_sse_deltas() {
    use futures::StreamExt;

    let server = MockServer::start().await;
    let sse = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;

    let engine =
        OpenAiEngine::new(OpenAiConfig::builder(format!("{}/v1", server.uri()), "m").build())
            .expect("client builds");

    let mut stream = engine
        .generate_stream(GenerateRequest::chat(vec![Message::user("hi")]))
        .await
        .expect("stream should open");

    let mut text = String::new();
    let mut saw_finish = false;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("chunk should decode");
        for choice in &chunk.choices {
            if let Some(c) = &choice.delta.content {
                text.push_str(c);
            }
            if choice.finish_reason.is_some() {
                saw_finish = true;
            }
        }
    }

    // The role-only delta contributes nothing; content arrives in order.
    assert_eq!(text, "Hello");
    assert!(saw_finish, "final chunk should carry a finish_reason");
}

/// Pre-tokenized input is refused explicitly rather than silently mangled.
#[tokio::test]
async fn tokens_prompt_is_rejected() {
    let (_server, engine, _) = scripted_server(vec!["unused"]).await;

    let mut request = GenerateRequest::chat(vec![]);
    request.prompt = PromptInput::Tokens(vec![1, 2, 3]);

    assert!(engine.generate(request).await.is_err());
}

// =============================================================================
// Full agentic loop over HTTP — the point of the exercise
// =============================================================================

/// The loop terminates on an explicit `<answer>` served over HTTP.
#[tokio::test]
async fn loop_completes_with_answer_over_http() {
    let (_server, engine, _) = scripted_server(vec![
        r#"<answer confidence="0.9">The answer is 42.</answer>"#,
    ])
    .await;

    let dir = tempfile::tempdir().expect("tempdir");
    let tools = Arc::new(ToolRegistry::with_code_tools());
    let executor = LoopExecutor::new(engine, tools, permissive_config("http-answer", dir.path()));

    let (tx, rx) = mpsc::channel(64);
    let summary = executor
        .run("What is the answer?", tx)
        .await
        .expect("loop should run");

    assert_eq!(summary.iterations_completed, 1);
    assert!(
        matches!(
            summary.termination,
            TerminationReason::Natural(NaturalTermination::AnswerProvided { .. })
        ),
        "expected AnswerProvided, got {:?}",
        summary.termination
    );

    let events = collect_events(rx).await;
    assert!(
        events
            .iter()
            .any(|e| matches!(e, LoopEvent::LoopCompleted { .. })),
        "should emit LoopCompleted"
    );
}

/// A full tool cycle: the model calls a real tool over HTTP, the executor runs
/// it, feeds the result back, and the model answers.
///
/// This is the end-to-end proof — real file I/O, real loop, real HTTP
/// transport, and no model artifact anywhere.
#[tokio::test]
async fn loop_executes_tool_then_answers_over_http() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file = dir.path().join("data.txt");
    std::fs::write(&file, "infernum").expect("write fixture");

    let (_server, engine, seen) = scripted_server(vec![
        // Iteration 1: read the file.
        &format!(
            "<tool_call>\n{{\"name\": \"read_file\", \"arguments\": {{\"path\": \"{}\"}}}}\n</tool_call>",
            file.display()
        ),
        // Iteration 2: answer using what the tool returned.
        r#"<answer confidence="0.95">The file says infernum.</answer>"#,
    ])
    .await;

    let tools = Arc::new(ToolRegistry::with_code_tools());
    let executor = LoopExecutor::new(engine, tools, permissive_config("http-tool", dir.path()));

    let (tx, rx) = mpsc::channel(128);
    let summary = executor
        .run("Read data.txt and tell me what it says", tx)
        .await
        .expect("loop should run");

    assert_eq!(summary.iterations_completed, 2, "expected two iterations");
    assert_eq!(summary.tool_calls_made, 1, "expected one tool call");
    assert!(matches!(
        summary.termination,
        TerminationReason::Natural(NaturalTermination::AnswerProvided { .. })
    ));

    let events = collect_events(rx).await;
    assert!(
        events
            .iter()
            .any(|e| matches!(e, LoopEvent::ToolCallDetected { tool, .. } if tool == "read_file")),
        "should emit ToolCallDetected for read_file"
    );

    // The tool's real output must have been fed back to the model: the second
    // request's message history should contain the file contents.
    let bodies = seen.lock();
    assert_eq!(bodies.len(), 2, "expected two round-trips");
    let second = bodies[1]["messages"].to_string();
    assert!(
        second.contains("infernum"),
        "tool result should be in the second request's history: {second}"
    );
}

/// Resource limits terminate the loop when the model never answers.
///
/// The model must keep making *tool calls* to stay alive: an iteration that
/// produces neither a tool call nor a meta-signal ends the loop immediately
/// (see `loop_ends_when_model_emits_bare_prose`), so bare prose would never
/// reach the budget.
#[tokio::test]
async fn loop_respects_max_iterations_over_http() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file = dir.path().join("loop.txt");
    std::fs::write(&file, "content").expect("write fixture");
    let call = format!(
        "<tool_call>\n{{\"name\": \"read_file\", \"arguments\": {{\"path\": \"{}\"}}}}\n</tool_call>",
        file.display()
    );

    let (_server, engine, _) = scripted_server(vec![&call, &call, &call, &call]).await;

    let config = permissive_config("http-budget", dir.path()).with_loop_config(LoopConfig {
        max_iterations: 2,
        detect_implicit_signals: false,
        ..LoopConfig::default()
    });

    let executor = LoopExecutor::new(engine, Arc::new(ToolRegistry::with_code_tools()), config);
    let (tx, _rx) = mpsc::channel(64);
    let summary = executor.run("Never finish", tx).await.expect("loop runs");

    assert_eq!(summary.iterations_completed, 2);
    assert!(
        matches!(summary.termination, TerminationReason::Resource(_)),
        "expected a resource termination, got {:?}",
        summary.termination
    );
}

/// An iteration yielding neither a tool call nor a meta-signal ends the loop.
///
/// Documents existing executor behaviour over the HTTP transport: with
/// implicit-signal detection off, bare prose still terminates after one
/// iteration rather than looping until the budget runs out.
#[tokio::test]
async fn loop_ends_when_model_emits_bare_prose() {
    let (_server, engine, _) =
        scripted_server(vec!["Still thinking about it.", "Still thinking about it."]).await;

    let dir = tempfile::tempdir().expect("tempdir");
    let config = permissive_config("http-prose", dir.path()).with_loop_config(LoopConfig {
        max_iterations: 5,
        detect_implicit_signals: false,
        ..LoopConfig::default()
    });

    let executor = LoopExecutor::new(engine, Arc::new(ToolRegistry::with_code_tools()), config);
    let (tx, _rx) = mpsc::channel(64);
    let summary = executor.run("Ponder", tx).await.expect("loop runs");

    assert_eq!(
        summary.iterations_completed, 1,
        "bare prose should end the loop after one iteration"
    );
}
