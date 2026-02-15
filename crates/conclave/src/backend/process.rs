//! Process-based backend session infrastructure.
//!
//! Provides common functionality for backends that spawn external processes
//! (Claude Code, Codex CLI, etc).

use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, info, warn};

use super::{AgentBackendSession, AgentEvent, TerminationReason};
use crate::error::{ConclaveError, Result};
use crate::types::{AgentBackend, Message, MessageContent};

// =============================================================================
// Process Session
// =============================================================================

/// Safely converts a process ID to a libc pid_t.
///
/// Returns `None` if the PID exceeds the maximum value for `libc::pid_t`.
/// While this is extremely unlikely in practice (PIDs rarely exceed a few million),
/// this ensures we don't have undefined behavior from integer overflow.
#[cfg(unix)]
fn pid_to_libc(pid: u32) -> Option<libc::pid_t> {
    // libc::pid_t is typically i32 on Unix systems
    if pid <= libc::pid_t::MAX as u32 {
        Some(pid as libc::pid_t)
    } else {
        None
    }
}

/// A backend session that manages an external process.
///
/// This provides the core infrastructure for process-based backends:
/// - Spawning and managing the child process
/// - Stdin/stdout communication
/// - Graceful termination with SIGINT fallback to SIGKILL
/// - Output parsing and event emission
pub struct ProcessSession {
    /// Session identifier.
    session_id: String,
    /// Backend configuration.
    backend: AgentBackend,
    /// Whether the session is still running.
    running: Arc<AtomicBool>,
    /// Child process handle (protected by mutex for termination).
    child: Arc<Mutex<Option<Child>>>,
    /// Stdin writer for sending messages.
    stdin: Arc<Mutex<Option<tokio::process::ChildStdin>>>,
    /// Event sender.
    event_tx: mpsc::Sender<AgentEvent>,
    /// Event receiver (taken once).
    event_rx: std::sync::Mutex<Option<mpsc::Receiver<AgentEvent>>>,
    /// Output parser for this backend.
    parser: Arc<dyn OutputParser>,
}

impl ProcessSession {
    /// Creates a new process session.
    ///
    /// # Arguments
    /// * `session_id` - Unique session identifier
    /// * `backend` - Backend configuration
    /// * `command` - The command to spawn
    /// * `args` - Command arguments
    /// * `working_dir` - Working directory for the process
    /// * `env` - Environment variables to set
    /// * `parser` - Output parser for converting stdout to events
    pub async fn spawn(
        session_id: String,
        backend: AgentBackend,
        command: &str,
        args: &[&str],
        working_dir: &std::path::Path,
        env: Vec<(&str, &str)>,
        parser: Arc<dyn OutputParser>,
    ) -> Result<Self> {
        let (event_tx, event_rx) = mpsc::channel(256);

        // Build the command
        let mut cmd = Command::new(command);
        cmd.args(args)
            .current_dir(working_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        // Set environment variables
        for (key, value) in env {
            cmd.env(key, value);
        }

        info!(
            "Spawning process: {} {:?} in {}",
            command,
            args,
            working_dir.display()
        );

        // Spawn the process
        let mut child = cmd.spawn().map_err(|e| ConclaveError::BackendSpawnFailed {
            backend: format!("{:?}", backend),
            reason: e.to_string(),
        })?;

        // Take stdin for writing
        let stdin = child.stdin.take();

        // Take stdout for reading
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| ConclaveError::BackendSpawnFailed {
                backend: format!("{:?}", backend),
                reason: "Failed to capture stdout".to_string(),
            })?;

        // Take stderr for logging
        let stderr = child.stderr.take();

        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();
        let event_tx_clone = event_tx.clone();
        let parser_clone = parser.clone();
        let session_id_clone = session_id.clone();

        // Spawn stdout reader task
        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();

            while let Ok(Some(line)) = lines.next_line().await {
                if !running_clone.load(Ordering::SeqCst) {
                    break;
                }

                debug!("[{}] stdout: {}", session_id_clone, line);

                // Parse the line into events
                match parser_clone.parse_line(&line) {
                    Ok(events) => {
                        for event in events {
                            if event_tx_clone.send(event).await.is_err() {
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        warn!("[{}] Parse error: {}", session_id_clone, e);
                    }
                }
            }

            // Process ended
            running_clone.store(false, Ordering::SeqCst);
            let _ = event_tx_clone
                .send(AgentEvent::Terminated {
                    reason: TerminationReason::Completed,
                })
                .await;
        });

        // Spawn stderr reader task (just for logging)
        if let Some(stderr) = stderr {
            let session_id_clone = session_id.clone();
            tokio::spawn(async move {
                let reader = BufReader::new(stderr);
                let mut lines = reader.lines();

                while let Ok(Some(line)) = lines.next_line().await {
                    debug!("[{}] stderr: {}", session_id_clone, line);
                }
            });
        }

        Ok(Self {
            session_id,
            backend,
            running,
            child: Arc::new(Mutex::new(Some(child))),
            stdin: Arc::new(Mutex::new(stdin)),
            event_tx,
            event_rx: std::sync::Mutex::new(Some(event_rx)),
            parser,
        })
    }

    /// Sends raw text to the process stdin.
    pub async fn send_stdin(&self, text: &str) -> Result<()> {
        if !self.is_running() {
            return Err(ConclaveError::BackendTerminated {
                session_id: self.session_id.clone(),
            });
        }

        let mut stdin_guard = self.stdin.lock().await;
        if let Some(stdin) = stdin_guard.as_mut() {
            stdin
                .write_all(text.as_bytes())
                .await
                .map_err(|e| ConclaveError::BackendCommunicationFailed {
                    session_id: self.session_id.clone(),
                    reason: e.to_string(),
                })?;
            stdin
                .flush()
                .await
                .map_err(|e| ConclaveError::BackendCommunicationFailed {
                    session_id: self.session_id.clone(),
                    reason: e.to_string(),
                })?;
            Ok(())
        } else {
            Err(ConclaveError::BackendTerminated {
                session_id: self.session_id.clone(),
            })
        }
    }
}

#[async_trait]
impl AgentBackendSession for ProcessSession {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn backend(&self) -> &AgentBackend {
        &self.backend
    }

    async fn send_message(&self, message: &Message) -> Result<()> {
        // Format message for stdin
        let text = self.parser.format_message(message)?;
        self.send_stdin(&text).await
    }

    async fn interrupt(&self) -> Result<()> {
        if !self.is_running() {
            return Err(ConclaveError::BackendTerminated {
                session_id: self.session_id.clone(),
            });
        }

        // Send interrupt signal
        #[cfg(unix)]
        {
            let child_guard = self.child.lock().await;
            if let Some(child) = child_guard.as_ref() {
                if let Some(pid) = child.id() {
                    if let Some(libc_pid) = pid_to_libc(pid) {
                        info!("[{}] Sending SIGINT to pid {}", self.session_id, pid);
                        // SAFETY: We've validated that the PID fits within libc::pid_t bounds
                        unsafe {
                            libc::kill(libc_pid, libc::SIGINT);
                        }
                    } else {
                        warn!("[{}] PID {} exceeds libc::pid_t max, cannot send signal", self.session_id, pid);
                    }
                }
            }
        }

        #[cfg(not(unix))]
        {
            // On Windows, we can't send SIGINT, so just log
            warn!(
                "[{}] Interrupt not fully supported on this platform",
                self.session_id
            );
        }

        Ok(())
    }

    async fn terminate(&self) -> Result<()> {
        self.running.store(false, Ordering::SeqCst);

        let mut child_guard = self.child.lock().await;
        if let Some(mut child) = child_guard.take() {
            info!("[{}] Terminating process", self.session_id);

            // Try graceful shutdown first (SIGTERM on Unix)
            #[cfg(unix)]
            {
                if let Some(pid) = child.id() {
                    if let Some(libc_pid) = pid_to_libc(pid) {
                        // SAFETY: We've validated that the PID fits within libc::pid_t bounds
                        unsafe {
                            libc::kill(libc_pid, libc::SIGTERM);
                        }
                    } else {
                        warn!("[{}] PID {} exceeds libc::pid_t max, using kill()", self.session_id, pid);
                        let _ = child.kill().await;
                        return Ok(());
                    }
                }

                // Wait briefly for graceful shutdown
                tokio::select! {
                    _ = child.wait() => {
                        info!("[{}] Process terminated gracefully", self.session_id);
                    }
                    _ = tokio::time::sleep(std::time::Duration::from_secs(2)) => {
                        // Force kill if still running
                        warn!("[{}] Process didn't terminate, forcing kill", self.session_id);
                        let _ = child.kill().await;
                    }
                }
            }

            #[cfg(not(unix))]
            {
                let _ = child.kill().await;
            }
        }

        // Send termination event
        let _ = self
            .event_tx
            .send(AgentEvent::Terminated {
                reason: TerminationReason::Requested,
            })
            .await;

        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    fn take_event_receiver(&self) -> Option<mpsc::Receiver<AgentEvent>> {
        self.event_rx.lock().unwrap().take()
    }
}

// =============================================================================
// Output Parser Trait
// =============================================================================

/// Parses backend output and formats messages for input.
///
/// Different backends have different output formats (JSON streaming, plain text, etc).
/// This trait abstracts the parsing logic.
pub trait OutputParser: Send + Sync {
    /// Parses a line of output into zero or more events.
    fn parse_line(&self, line: &str) -> Result<Vec<AgentEvent>>;

    /// Formats a message for sending to the backend's stdin.
    fn format_message(&self, message: &Message) -> Result<String>;
}

// =============================================================================
// Plain Text Parser (Default)
// =============================================================================

/// Simple plain text parser for basic backends.
pub struct PlainTextParser;

impl OutputParser for PlainTextParser {
    fn parse_line(&self, line: &str) -> Result<Vec<AgentEvent>> {
        // Treat each non-empty line as a message
        if line.trim().is_empty() {
            return Ok(vec![]);
        }

        Ok(vec![AgentEvent::Message {
            content: line.to_string(),
            mentions: vec![],
        }])
    }

    fn format_message(&self, message: &Message) -> Result<String> {
        match &message.content {
            MessageContent::Text { content } => Ok(format!("{}\n", content)),
            MessageContent::ToolCall { tool, input, .. } => {
                Ok(format!("[Tool: {}] {}\n", tool, input))
            }
            MessageContent::ToolResult { output, .. } => Ok(format!("[Result] {}\n", output)),
            MessageContent::System { event } => Ok(format!("[System] {:?}\n", event)),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plain_text_parser_parse() {
        let parser = PlainTextParser;

        // Non-empty line produces event
        let events = parser.parse_line("Hello world").unwrap();
        assert_eq!(events.len(), 1);
        match &events[0] {
            AgentEvent::Message { content, .. } => assert_eq!(content, "Hello world"),
            _ => panic!("Expected Message event"),
        }

        // Empty line produces no events
        let events = parser.parse_line("").unwrap();
        assert!(events.is_empty());

        let events = parser.parse_line("   ").unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn test_plain_text_parser_format() {
        use crate::types::{ChannelType, MessageId, ParticipantId};
        use chrono::Utc;
        use std::collections::HashMap;

        let parser = PlainTextParser;

        let message = Message {
            id: MessageId::new(),
            channel: ChannelType::Main,
            sender: ParticipantId::new(),
            content: MessageContent::Text {
                content: "Hello".to_string(),
            },
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };

        let formatted = parser.format_message(&message).unwrap();
        assert_eq!(formatted, "Hello\n");
    }
}
