//! End-to-end integration test: Agent + Code Tools + Mock LLM.
//!
//! Proves the full pipeline: Agent.run() → scripted LLM responses →
//! parse_action → tool dispatch → observation → next iteration.
//!
//! Uses a `ScriptedEngine` that returns predetermined responses,
//! driving the agent through a multi-step file manipulation task
//! against a real temp directory.

use std::sync::Arc;

use async_trait::async_trait;
use infernum_core::{
    model::LlamaVersion, response::Choice, EmbedRequest, EmbedResponse, GenerateRequest,
    GenerateResponse, ModelArchitecture, ModelId, ModelMetadata, ModelSource, RequestId, Result,
    TokenStream, Usage,
};
use parking_lot::Mutex;

use abaddon::InferenceEngine;
use beleth::{Agent, ToolRegistry};

// =============================================================================
// ScriptedEngine — mock InferenceEngine returning predetermined responses
// =============================================================================

/// A mock inference engine that returns a sequence of scripted responses.
///
/// Each call to `generate()` pops the next response from the queue.
/// Panics if called more times than there are scripted responses.
struct ScriptedEngine {
    responses: Mutex<Vec<String>>,
    metadata: ModelMetadata,
    call_count: Mutex<usize>,
}

impl ScriptedEngine {
    /// Creates a new scripted engine with the given responses.
    ///
    /// Responses are consumed in order: first call gets `responses[0]`, etc.
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
                "Final Answer: Test completed (ran out of scripted responses)",
            ));
        }

        let text = responses[idx].clone();
        Ok(Self::make_response(&text))
    }

    async fn generate_stream(&self, _request: GenerateRequest) -> Result<TokenStream> {
        Ok(TokenStream::empty())
    }

    async fn embed(&self, _request: EmbedRequest) -> Result<EmbedResponse> {
        Err(infernum_core::Error::internal(
            "Embedding not supported in mock",
        ))
    }

    fn model_info(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn is_ready(&self) -> bool {
        true
    }
}

// =============================================================================
// Test Helpers
// =============================================================================

/// Creates a ToolRegistry with code tools and a working_dir set to the temp dir.
fn make_code_tools() -> ToolRegistry {
    ToolRegistry::with_code_tools()
}

// =============================================================================
// End-to-End Tests
// =============================================================================

/// Test: Agent reads a file, edits it, then reads it again to verify.
///
/// Scenario:
///   1. Create a temp file with known content
///   2. Agent calls read_file to see it
///   3. Agent calls edit_file to change a line
///   4. Agent calls read_file again to verify the change
///   5. Agent produces a final answer confirming the edit
#[tokio::test]
async fn test_agent_read_edit_verify_flow() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file_path = dir.path().join("hello.txt");
    std::fs::write(
        &file_path,
        "Hello, world!\nThis is a test file.\nGoodbye.\n",
    )
    .expect("write seed file");

    let file_path_str = file_path.to_string_lossy().to_string();

    // Script the LLM responses in ReAct format (generic Action:/Action Input:)
    let responses = vec![
        // Step 1: Read the file
        format!(
            "Thought: I need to read the file first to see its contents.\n\
             Action: read_file\n\
             Action Input: {{\"path\": \"{file_path_str}\"}}"
        ),
        // Step 2: Edit the file (replace "world" with "Infernum")
        format!(
            "Thought: I see the file. Let me edit line 1.\n\
             Action: edit_file\n\
             Action Input: {{\"path\": \"{file_path_str}\", \"old_string\": \"Hello, world!\", \"new_string\": \"Hello, Infernum!\"}}"
        ),
        // Step 3: Read again to verify
        format!(
            "Thought: Let me verify the edit took effect.\n\
             Action: read_file\n\
             Action Input: {{\"path\": \"{file_path_str}\"}}"
        ),
        // Step 4: Final answer
        "Final Answer: Successfully edited hello.txt — changed 'Hello, world!' to 'Hello, Infernum!' and verified the change.".to_string(),
    ];

    let engine = Arc::new(ScriptedEngine::new(responses));

    let mut agent = Agent::builder()
        .id("test-read-edit-verify")
        .system_prompt("You are a file editing assistant.")
        .max_iterations(10)
        .tools(make_code_tools())
        .engine(engine.clone())
        .working_dir(dir.path())
        .build();

    let result = agent
        .run("Edit hello.txt to say 'Hello, Infernum!' instead of 'Hello, world!'")
        .await;

    let answer = result.expect("agent run should succeed");
    assert!(
        answer.contains("Successfully edited"),
        "Expected success message, got: {answer}"
    );

    // Verify the file was actually modified on disk
    let content = std::fs::read_to_string(&file_path).expect("read modified file");
    assert!(
        content.contains("Hello, Infernum!"),
        "File should contain edited text, got: {content}"
    );
    assert!(
        !content.contains("Hello, world!"),
        "File should not contain original text, got: {content}"
    );

    // Verify engine was called exactly 4 times
    let count = *engine.call_count.lock();
    assert_eq!(
        count, 4,
        "Engine should have been called 4 times, got {count}"
    );
}

/// Test: Agent writes a new file and then lists the directory.
///
/// Scenario:
///   1. Agent calls write_file to create a new file
///   2. Agent calls list_files to see the directory contents
///   3. Agent produces a final answer confirming the file exists
#[tokio::test]
async fn test_agent_write_and_list_flow() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file_path = dir.path().join("output.txt");
    let file_path_str = file_path.to_string_lossy().to_string();
    let dir_path_str = dir.path().to_string_lossy().to_string();

    let responses = vec![
        // Step 1: Write a file
        format!(
            "Thought: I'll create the output file.\n\
             Action: write_file\n\
             Action Input: {{\"path\": \"{file_path_str}\", \"content\": \"Generated by Infernum agent\\nLine 2\\n\"}}"
        ),
        // Step 2: List directory
        format!(
            "Thought: Let me verify the file was created.\n\
             Action: list_files\n\
             Action Input: {{\"path\": \"{dir_path_str}\"}}"
        ),
        // Step 3: Final answer
        "Final Answer: Created output.txt with 2 lines of content and verified it exists in the directory.".to_string(),
    ];

    let engine = Arc::new(ScriptedEngine::new(responses));

    let mut agent = Agent::builder()
        .id("test-write-list")
        .system_prompt("You are a file management assistant.")
        .max_iterations(10)
        .tools(make_code_tools())
        .engine(engine)
        .working_dir(dir.path())
        .build();

    let result = agent
        .run("Create a file called output.txt with some content.")
        .await;

    let answer = result.expect("agent run should succeed");
    assert!(
        answer.contains("Created output.txt"),
        "Expected success message, got: {answer}"
    );

    // Verify the file exists and has correct content
    let content = std::fs::read_to_string(&file_path).expect("read created file");
    assert!(
        content.contains("Generated by Infernum agent"),
        "File should contain written text, got: {content}"
    );
}

/// Test: Agent uses search_files to find a pattern, then reads the matching file.
///
/// Scenario:
///   1. Create multiple files in a temp directory
///   2. Agent calls search_files to find files containing a keyword
///   3. Agent calls read_file on the match
///   4. Agent produces a final answer with the found content
#[tokio::test]
async fn test_agent_search_then_read_flow() {
    let dir = tempfile::tempdir().expect("tempdir");

    // Create several files, only one contains the target
    std::fs::write(dir.path().join("alpha.txt"), "nothing here\n").expect("write alpha");
    std::fs::write(dir.path().join("beta.txt"), "the SECRET_KEY is hidden\n").expect("write beta");
    std::fs::write(dir.path().join("gamma.txt"), "also nothing\n").expect("write gamma");

    let dir_path_str = dir.path().to_string_lossy().to_string();
    let beta_path_str = dir.path().join("beta.txt").to_string_lossy().to_string();

    let responses = vec![
        // Step 1: Search for the keyword
        format!(
            "Thought: I need to find which file contains SECRET_KEY.\n\
             Action: search_files\n\
             Action Input: {{\"pattern\": \"SECRET_KEY\", \"path\": \"{dir_path_str}\"}}"
        ),
        // Step 2: Read the file that matched
        format!(
            "Thought: Found it in beta.txt. Let me read the full file.\n\
             Action: read_file\n\
             Action Input: {{\"path\": \"{beta_path_str}\"}}"
        ),
        // Step 3: Final answer
        "Final Answer: Found SECRET_KEY in beta.txt. The file contains: 'the SECRET_KEY is hidden'.".to_string(),
    ];

    let engine = Arc::new(ScriptedEngine::new(responses));

    let mut agent = Agent::builder()
        .id("test-search-read")
        .system_prompt("You are a code search assistant.")
        .max_iterations(10)
        .tools(make_code_tools())
        .engine(engine)
        .working_dir(dir.path())
        .build();

    let result = agent
        .run("Find which file contains 'SECRET_KEY' and tell me what it says.")
        .await;

    let answer = result.expect("agent run should succeed");
    assert!(
        answer.contains("SECRET_KEY") && answer.contains("beta.txt"),
        "Expected answer mentioning SECRET_KEY and beta.txt, got: {answer}"
    );
}

/// Test: Agent runs a bash command and uses its output.
///
/// Scenario:
///   1. Agent calls bash to run `echo hello && echo world`
///   2. Agent produces a final answer with the command output
#[tokio::test]
async fn test_agent_bash_tool_flow() {
    let dir = tempfile::tempdir().expect("tempdir");

    let responses = vec![
        // Step 1: Run a bash command
        "Thought: Let me run a simple command.\n\
         Action: bash\n\
         Action Input: {\"command\": \"echo hello && echo world\"}"
            .to_string(),
        // Step 2: Final answer
        "Final Answer: The command output was: hello\\nworld".to_string(),
    ];

    let engine = Arc::new(ScriptedEngine::new(responses));

    let mut agent = Agent::builder()
        .id("test-bash")
        .system_prompt("You are a shell assistant.")
        .max_iterations(10)
        .tools(make_code_tools())
        .engine(engine)
        .working_dir(dir.path())
        .build();

    let result = agent.run("Run 'echo hello && echo world'").await;
    let answer = result.expect("agent run should succeed");
    assert!(
        answer.contains("hello"),
        "Expected answer about command output, got: {answer}"
    );
}

/// Test: Agent uses native <tool_call> format (Qwen-style).
///
/// Verifies that the parse_native_tool_call path works end-to-end.
#[tokio::test]
async fn test_agent_native_tool_call_format() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file_path = dir.path().join("native.txt");
    let file_path_str = file_path.to_string_lossy().to_string();

    let responses = vec![
        // Step 1: Write using native tool_call format
        format!(
            "<tool_call>\n\
             {{\"name\": \"write_file\", \"arguments\": {{\"path\": \"{file_path_str}\", \"content\": \"native format works!\"}}}}\n\
             </tool_call>"
        ),
        // Step 2: Final answer
        "Final Answer: File written using native tool call format.".to_string(),
    ];

    let engine = Arc::new(ScriptedEngine::new(responses));

    let mut agent = Agent::builder()
        .id("test-native-format")
        .system_prompt("You are a tool-calling assistant.")
        .max_iterations(10)
        .tools(make_code_tools())
        .engine(engine)
        .working_dir(dir.path())
        .build();

    let result = agent.run("Write a file using native format.").await;
    let answer = result.expect("agent run should succeed");
    assert!(
        answer.contains("native tool call format"),
        "Expected success message, got: {answer}"
    );

    // Verify the file was written
    let content = std::fs::read_to_string(&file_path).expect("read written file");
    assert_eq!(content, "native format works!");
}

/// Test: Agent hits max_iterations without a Final Answer.
///
/// Verifies graceful termination — the last assistant response is used.
#[tokio::test]
async fn test_agent_max_iterations_graceful_exit() {
    let responses = vec![
        "Thought: I'm thinking about this...".to_string(),
        "Thought: Still thinking...".to_string(),
        "Thought: Almost there...".to_string(),
    ];

    let engine = Arc::new(ScriptedEngine::new(responses));

    let dir = tempfile::tempdir().expect("tempdir");
    let mut agent = Agent::builder()
        .id("test-max-iter")
        .system_prompt("You think a lot.")
        .max_iterations(3)
        .tools(make_code_tools())
        .engine(engine)
        .working_dir(dir.path())
        .build();

    let result = agent.run("Think about things.").await;
    let answer = result.expect("should not error, just hit max iterations");
    // The agent returns the last assistant response when no Final Answer is given
    assert!(
        answer.contains("Almost there"),
        "Expected last response, got: {answer}"
    );
}

/// Test: Tool error is handled gracefully as an observation.
///
/// Agent tries to read a non-existent file, gets an error observation,
/// and then produces a final answer acknowledging the failure.
#[tokio::test]
async fn test_agent_tool_error_recovery() {
    let dir = tempfile::tempdir().expect("tempdir");
    let missing_path = dir
        .path()
        .join("nonexistent.txt")
        .to_string_lossy()
        .to_string();

    let responses = vec![
        // Step 1: Try to read a file that doesn't exist (within working dir)
        format!(
            "Thought: Let me read the file.\n\
             Action: read_file\n\
             Action Input: {{\"path\": \"{missing_path}\"}}"
        ),
        // Step 2: Acknowledge the error and give final answer
        "Final Answer: The file does not exist. I received an error when trying to read it."
            .to_string(),
    ];

    let engine = Arc::new(ScriptedEngine::new(responses));

    let mut agent = Agent::builder()
        .id("test-error-recovery")
        .system_prompt("You handle errors gracefully.")
        .max_iterations(5)
        .tools(make_code_tools())
        .engine(engine)
        .working_dir(dir.path())
        .build();

    let result = agent.run("Read nonexistent.txt").await;
    let answer = result.expect("agent should handle tool error gracefully");
    assert!(
        answer.contains("does not exist"),
        "Expected error acknowledgment, got: {answer}"
    );
}

/// Test: Multi-file workflow — create, search, edit, bash verify.
///
/// This is the most comprehensive test, exercising 4 different tools
/// in a single agent run.
#[tokio::test]
async fn test_agent_multi_tool_workflow() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src_path = dir.path().join("main.rs");
    let src_path_str = src_path.to_string_lossy().to_string();
    let dir_path_str = dir.path().to_string_lossy().to_string();

    let responses = vec![
        // Step 1: Write a Rust source file
        format!(
            "Thought: I'll create a simple Rust file.\n\
             Action: write_file\n\
             Action Input: {{\"path\": \"{src_path_str}\", \"content\": \"fn main() {{\\n    println!(\\\"old message\\\");\\n}}\\n\"}}"
        ),
        // Step 2: Search for the println
        format!(
            "Thought: Let me find where the print statement is.\n\
             Action: search_files\n\
             Action Input: {{\"pattern\": \"println\", \"path\": \"{dir_path_str}\"}}"
        ),
        // Step 3: Edit the println message
        format!(
            "Thought: Found it. Let me change the message.\n\
             Action: edit_file\n\
             Action Input: {{\"path\": \"{src_path_str}\", \"old_string\": \"old message\", \"new_string\": \"new message\"}}"
        ),
        // Step 4: Verify with bash (cat the file)
        format!(
            "Thought: Let me verify the edit with cat.\n\
             Action: bash\n\
             Action Input: {{\"command\": \"cat {src_path_str}\"}}"
        ),
        // Step 5: Final answer
        "Final Answer: Created main.rs, found the println, changed 'old message' to 'new message', and verified the change with cat.".to_string(),
    ];

    let engine = Arc::new(ScriptedEngine::new(responses));

    let mut agent = Agent::builder()
        .id("test-multi-tool")
        .system_prompt("You are a full-stack coding assistant.")
        .max_iterations(10)
        .tools(make_code_tools())
        .engine(engine.clone())
        .working_dir(dir.path())
        .build();

    let result = agent
        .run("Create a Rust file with a println, then change its message.")
        .await;

    let answer = result.expect("agent run should succeed");
    assert!(
        answer.contains("new message"),
        "Expected success message, got: {answer}"
    );

    // Verify final file state
    let content = std::fs::read_to_string(&src_path).expect("read final file");
    assert!(
        content.contains("new message"),
        "File should contain edited text, got: {content}"
    );
    assert!(
        !content.contains("old message"),
        "File should not contain old text, got: {content}"
    );

    // Verify engine call count: 5 scripted responses
    let count = *engine.call_count.lock();
    assert_eq!(
        count, 5,
        "Engine should have been called 5 times, got {count}"
    );
}
