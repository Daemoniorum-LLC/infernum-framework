//! Session continuation — resuming terminated agentic loops.
//!
//! When a loop terminates as Stuck, Yielded, or via resource limits, the client
//! may resume it by providing additional context or extending resource budgets.
//! This module implements the continuation state, storage, and resume logic
//! described in AGENTIC-LOOP-SPEC §9.3.
//!
//! # Architecture
//!
//! - [`ContinuationState`] captures the full loop state at termination.
//! - [`ContinuationStore`] is a pluggable async storage backend.
//! - [`InMemoryContinuationStore`] provides a default implementation with TTL
//!   and LRU eviction.
//! - [`ConfigOverride`] specifies which configuration fields to modify on resume.
//! - [`apply_config_override`] validates and applies overrides.
//! - [`build_resumed_messages`] reconstructs the message history with optional
//!   additional context.
//!
//! # Token Format
//!
//! Continuation tokens are opaque strings: `cont_{session_id}_{uuid}`.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::types::{
    AgenticToolResult, AutonomyGrant, ContextMessage, ExplorationBranch, LoopConfig,
    NaturalTermination, TerminationReason, ToolPattern,
};

// ---------------------------------------------------------------------------
// ContinuationState
// ---------------------------------------------------------------------------

/// Full loop state captured at termination for possible resumption.
///
/// Serializable so it can be persisted to disk or database.
/// Uses [`SystemTime`] (not `Instant`) because this must survive serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuationState {
    /// Opaque token identifying this continuation.
    pub token: String,

    /// Session identity (preserved across continuations).
    pub session_id: String,

    /// Full message history at time of termination.
    pub messages: Vec<ContextMessage>,

    /// Tool results collected during execution.
    pub tool_results: Vec<AgenticToolResult>,

    /// Exploration branches tracked.
    pub exploration_branches: Vec<ExplorationBranch>,

    /// Iterations completed before termination.
    pub iterations_completed: u32,

    /// Total tool calls made before termination.
    pub tool_calls_made: u32,

    /// Total tokens generated before termination.
    pub tokens_generated: u32,

    /// Configuration used (may be modified on resume).
    pub loop_config: LoopConfig,

    /// Autonomy grant (may be widened on resume).
    pub autonomy: AutonomyGrant,

    /// System prompt (immutable after loop start).
    pub system_prompt: Option<String>,

    /// Working directory (immutable after loop start).
    pub working_dir: Option<String>,

    /// Why the loop stopped.
    pub termination: TerminationReason,

    /// When the state was stored.
    pub stored_at: SystemTime,
}

// ---------------------------------------------------------------------------
// Resumability
// ---------------------------------------------------------------------------

impl TerminationReason {
    /// Returns `true` if a loop that ended with this reason may be resumed.
    ///
    /// Resumable terminations:
    /// - `AgentStuck` — client provides clarification
    /// - `AgentYielded` — different agent or additional context
    /// - `MaxIterations` — client extends budget
    /// - `TokenBudgetExhausted` — client extends budget
    /// - `WallTimeExceeded` — client extends budget
    /// - `ToolCallLimitReached` — client extends budget
    ///
    /// Non-resumable:
    /// - `AnswerProvided` — task complete
    /// - `TaskComplete` — task complete
    /// - `ClientCancelled` — intentional termination
    /// - `OperatorTerminated` — intentional termination
    /// - `SystemShutdown` — state may be lost
    pub fn is_resumable(&self) -> bool {
        match self {
            Self::Natural(n) => matches!(
                n,
                NaturalTermination::AgentStuck { .. } | NaturalTermination::AgentYielded { .. }
            ),
            Self::Resource(_) => true,
            Self::External(_) => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Token Generation
// ---------------------------------------------------------------------------

/// Generates a continuation token: `cont_{session_id}_{uuid}`.
fn generate_token(session_id: &str) -> String {
    let uuid = Uuid::new_v4();
    format!("cont_{session_id}_{uuid}")
}

// ---------------------------------------------------------------------------
// Store Error
// ---------------------------------------------------------------------------

/// Errors from continuation store operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum StoreError {
    /// The continuation token was not found.
    #[error("continuation not found: {token}")]
    NotFound {
        /// The token that was not found.
        token: String,
    },

    /// The continuation has expired.
    #[error("continuation expired: {token}")]
    Expired {
        /// The expired token.
        token: String,
    },

    /// Storage backend error.
    #[error("store error: {message}")]
    Backend {
        /// Error description.
        message: String,
    },
}

// ---------------------------------------------------------------------------
// ContinuationStore trait
// ---------------------------------------------------------------------------

/// Pluggable async storage backend for continuation states.
///
/// The default implementation is [`InMemoryContinuationStore`].
/// Implementations may use files, databases, or distributed caches.
#[async_trait]
pub trait ContinuationStore: Send + Sync {
    /// Stores a continuation state and returns its token.
    async fn store(&self, state: ContinuationState) -> Result<String, StoreError>;

    /// Loads a continuation state by token. Returns `None` if expired or missing.
    async fn load(&self, token: &str) -> Result<Option<ContinuationState>, StoreError>;

    /// Removes a continuation state.
    async fn remove(&self, token: &str) -> Result<(), StoreError>;

    /// Garbage-collects expired states. Returns the number removed.
    async fn cleanup_expired(&self) -> Result<u32, StoreError>;
}

// ---------------------------------------------------------------------------
// InMemoryContinuationStore
// ---------------------------------------------------------------------------

/// In-memory entry with access timestamp for LRU.
#[derive(Debug, Clone)]
struct StoreEntry {
    state: ContinuationState,
    last_accessed: SystemTime,
}

/// In-memory continuation store with TTL and LRU eviction.
///
/// - **TTL**: States older than `ttl` are considered expired.
/// - **LRU eviction**: When `max_entries` is reached, the least-recently-accessed
///   entry is evicted to make room.
/// - Thread-safe via [`parking_lot::RwLock`].
#[derive(Debug)]
pub struct InMemoryContinuationStore {
    entries: RwLock<HashMap<String, StoreEntry>>,
    /// Time-to-live for stored states.
    ttl: Duration,
    /// Maximum number of stored entries (LRU eviction beyond this).
    max_entries: usize,
}

impl InMemoryContinuationStore {
    /// Creates a new store with the given TTL and capacity.
    pub fn new(ttl: Duration, max_entries: usize) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            ttl,
            max_entries,
        }
    }

    /// Creates a store with default settings (1 hour TTL, 100 max entries).
    pub fn with_defaults() -> Self {
        Self::new(Duration::from_secs(3600), 100)
    }

    /// Returns the number of entries currently stored (including expired).
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Returns `true` if the store has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Returns `true` if the entry is expired relative to `now`.
    fn is_expired(entry: &StoreEntry, now: SystemTime, ttl: Duration) -> bool {
        now.duration_since(entry.state.stored_at)
            .map(|age| age > ttl)
            .unwrap_or(false)
    }

    /// Evicts the least-recently-accessed entry. Returns the evicted token.
    fn evict_lru(entries: &mut HashMap<String, StoreEntry>) -> Option<String> {
        let oldest = entries
            .iter()
            .min_by_key(|(_, e)| e.last_accessed)
            .map(|(k, _)| k.clone());

        if let Some(ref key) = oldest {
            entries.remove(key);
        }
        oldest
    }
}

#[async_trait]
impl ContinuationStore for InMemoryContinuationStore {
    async fn store(&self, state: ContinuationState) -> Result<String, StoreError> {
        let token = state.token.clone();
        let now = SystemTime::now();

        let mut entries = self.entries.write();

        // Evict LRU if at capacity
        while entries.len() >= self.max_entries {
            Self::evict_lru(&mut entries);
        }

        entries.insert(
            token.clone(),
            StoreEntry {
                state,
                last_accessed: now,
            },
        );

        Ok(token)
    }

    async fn load(&self, token: &str) -> Result<Option<ContinuationState>, StoreError> {
        let now = SystemTime::now();
        let mut entries = self.entries.write();

        let Some(entry) = entries.get_mut(token) else {
            return Ok(None);
        };

        // Check TTL
        if Self::is_expired(entry, now, self.ttl) {
            entries.remove(token);
            return Ok(None);
        }

        // Refresh LRU position
        entry.last_accessed = now;
        Ok(Some(entry.state.clone()))
    }

    async fn remove(&self, token: &str) -> Result<(), StoreError> {
        self.entries.write().remove(token);
        Ok(())
    }

    async fn cleanup_expired(&self) -> Result<u32, StoreError> {
        let now = SystemTime::now();
        let ttl = self.ttl;
        let mut entries = self.entries.write();

        let before = entries.len();
        entries.retain(|_, e| !Self::is_expired(e, now, ttl));
        let removed = before - entries.len();

        // max_entries is bounded at construction time, so this won't overflow u32 in practice.
        #[allow(clippy::cast_possible_truncation)]
        Ok(removed as u32)
    }
}

// ---------------------------------------------------------------------------
// ConfigOverride
// ---------------------------------------------------------------------------

/// Configuration overrides for resuming a loop.
///
/// Only fields that are `Some` are applied. Fields that are `None` preserve
/// the original value from the continuation state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConfigOverride {
    /// New maximum iterations. Must exceed `iterations_completed`.
    pub max_iterations: Option<u32>,

    /// New maximum tool calls. Must exceed `tool_calls_made`.
    pub max_tool_calls: Option<u32>,

    /// New maximum tokens. Must exceed `tokens_generated`.
    pub max_tokens: Option<u32>,

    /// Additional auto-approve patterns to add (widens autonomy).
    pub auto_approve: Option<Vec<String>>,

    /// Forbidden patterns to remove (narrows restrictions).
    pub remove_forbidden: Option<Vec<String>>,
}

/// Errors from applying configuration overrides.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ResumeError {
    /// New resource limit is below the amount already consumed.
    #[error("new {resource} limit ({new_limit}) must exceed consumed amount ({consumed})")]
    LimitBelowConsumed {
        /// Which resource (`iterations`, `tool_calls`, `tokens`).
        resource: String,
        /// Amount already consumed.
        consumed: u32,
        /// The rejected new limit.
        new_limit: u32,
    },

    /// Attempted to modify an immutable field.
    #[error("{field} is immutable after loop start")]
    ImmutableField {
        /// The field that cannot be changed.
        field: String,
    },

    /// The termination reason does not allow resumption.
    #[error("termination reason is not resumable")]
    NotResumable,
}

// ---------------------------------------------------------------------------
// Resume Logic
// ---------------------------------------------------------------------------

/// Validates and applies configuration overrides to a continuation state.
///
/// # Errors
///
/// Returns [`ResumeError::LimitBelowConsumed`] if any new limit is at or below
/// the amount already consumed.
pub fn apply_config_override(
    state: &ContinuationState,
    overrides: &ConfigOverride,
) -> Result<(LoopConfig, AutonomyGrant), ResumeError> {
    let mut config = state.loop_config.clone();
    let mut autonomy = state.autonomy.clone();

    // Validate and apply iteration limit
    if let Some(new_max) = overrides.max_iterations {
        if new_max <= state.iterations_completed {
            return Err(ResumeError::LimitBelowConsumed {
                resource: "max_iterations".to_string(),
                consumed: state.iterations_completed,
                new_limit: new_max,
            });
        }
        config.max_iterations = new_max;
    }

    // Validate and apply tool call limit
    if let Some(new_max) = overrides.max_tool_calls {
        if new_max <= state.tool_calls_made {
            return Err(ResumeError::LimitBelowConsumed {
                resource: "max_tool_calls".to_string(),
                consumed: state.tool_calls_made,
                new_limit: new_max,
            });
        }
        config.max_tool_calls = new_max;
    }

    // Validate and apply token limit
    if let Some(new_max) = overrides.max_tokens {
        if new_max <= state.tokens_generated {
            return Err(ResumeError::LimitBelowConsumed {
                resource: "max_tokens".to_string(),
                consumed: state.tokens_generated,
                new_limit: new_max,
            });
        }
        config.max_tokens = new_max;
    }

    // Widen autonomy: add new auto-approve patterns
    if let Some(ref patterns) = overrides.auto_approve {
        for pattern in patterns {
            autonomy
                .auto_approve
                .push(ToolPattern::Tool(pattern.clone()));
        }
    }

    // Narrow forbidden: remove specified patterns
    if let Some(ref remove) = overrides.remove_forbidden {
        autonomy.forbidden.retain(|p| {
            let name = match p {
                ToolPattern::Tool(n)
                | ToolPattern::Read(n)
                | ToolPattern::Write(n)
                | ToolPattern::Bash(n) => n,
            };
            !remove.contains(name)
        });
    }

    Ok((config, autonomy))
}

/// Reconstructs the message history for a resumed loop, optionally appending
/// additional context as a user message.
pub fn build_resumed_messages(
    state: &ContinuationState,
    additional_context: Option<&str>,
) -> Vec<ContextMessage> {
    let mut messages = state.messages.clone();

    if let Some(context) = additional_context {
        messages.push(ContextMessage {
            role: "user".to_string(),
            content: context.to_string(),
            tool_call_id: None,
        });
    }

    messages
}

impl ContinuationState {
    /// Creates a new [`ContinuationState`] from loop execution data.
    ///
    /// Generates a token in the format `cont_{session_id}_{uuid}`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        session_id: &str,
        messages: Vec<ContextMessage>,
        tool_results: Vec<AgenticToolResult>,
        exploration_branches: Vec<ExplorationBranch>,
        iterations_completed: u32,
        tool_calls_made: u32,
        tokens_generated: u32,
        loop_config: LoopConfig,
        autonomy: AutonomyGrant,
        system_prompt: Option<String>,
        working_dir: Option<String>,
        termination: TerminationReason,
    ) -> Self {
        Self {
            token: generate_token(session_id),
            session_id: session_id.to_string(),
            messages,
            tool_results,
            exploration_branches,
            iterations_completed,
            tool_calls_made,
            tokens_generated,
            loop_config,
            autonomy,
            system_prompt,
            working_dir,
            termination,
            stored_at: SystemTime::now(),
        }
    }
}

/// Creates a [`ContinuationState`] from loop execution data.
///
/// Convenience alias for [`ContinuationState::new`].
#[allow(clippy::too_many_arguments)]
pub fn create_continuation_state(
    session_id: &str,
    messages: Vec<ContextMessage>,
    tool_results: Vec<AgenticToolResult>,
    exploration_branches: Vec<ExplorationBranch>,
    iterations_completed: u32,
    tool_calls_made: u32,
    tokens_generated: u32,
    loop_config: LoopConfig,
    autonomy: AutonomyGrant,
    system_prompt: Option<String>,
    working_dir: Option<String>,
    termination: TerminationReason,
) -> ContinuationState {
    ContinuationState::new(
        session_id,
        messages,
        tool_results,
        exploration_branches,
        iterations_completed,
        tool_calls_made,
        tokens_generated,
        loop_config,
        autonomy,
        system_prompt,
        working_dir,
        termination,
    )
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::super::types::{ExternalTermination, ResourceTermination};
    use super::*;
    use proptest::prelude::*;
    use std::time::Duration;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    fn make_messages(n: usize) -> Vec<ContextMessage> {
        (0..n)
            .map(|i| ContextMessage {
                role: if i % 2 == 0 {
                    "user".to_string()
                } else {
                    "assistant".to_string()
                },
                content: format!("message {i}"),
                tool_call_id: None,
            })
            .collect()
    }

    fn make_tool_result(call_id: &str, tool_name: &str) -> AgenticToolResult {
        AgenticToolResult {
            call_id: call_id.to_string(),
            tool_name: tool_name.to_string(),
            status: super::super::types::ResultStatus::Success,
            data: serde_json::json!({"output": "result"}),
            confidence: super::super::types::Confidence::Measured,
            latency_ms: 15,
            truncated: false,
        }
    }

    fn make_exploration(desc: &str, productive: bool) -> ExplorationBranch {
        ExplorationBranch {
            description: desc.to_string(),
            tool_calls: vec!["read_file".to_string()],
            productive,
            findings: if productive {
                Some("found it".to_string())
            } else {
                None
            },
        }
    }

    fn make_state(termination: TerminationReason) -> ContinuationState {
        ContinuationState {
            token: generate_token("test-session"),
            session_id: "test-session".to_string(),
            messages: make_messages(2),
            tool_results: vec![make_tool_result("call_1", "read_file")],
            exploration_branches: vec![make_exploration("Tried JSON config", false)],
            iterations_completed: 3,
            tool_calls_made: 5,
            tokens_generated: 1200,
            loop_config: LoopConfig::default(),
            autonomy: AutonomyGrant::default(),
            system_prompt: Some("You are a helper.".to_string()),
            working_dir: Some("/home/user/project".to_string()),
            termination,
            stored_at: SystemTime::now(),
        }
    }

    fn stuck_termination() -> TerminationReason {
        TerminationReason::Natural(NaturalTermination::AgentStuck {
            attempts: 3,
            request: super::super::types::StuckRequest::Clarification(vec![
                "Where is the config file?".to_string(),
            ]),
        })
    }

    fn max_iterations_termination() -> TerminationReason {
        TerminationReason::Resource(ResourceTermination::MaxIterations {
            completed: 10,
            limit: 10,
        })
    }

    // =======================================================================
    // §11.1 Store/Load Roundtrip
    // =======================================================================

    #[tokio::test]
    async fn test_store_load_roundtrip() {
        let store = InMemoryContinuationStore::with_defaults();
        let state = make_state(stuck_termination());
        let original_json = serde_json::to_string(&state).expect("serialize");

        let token = store.store(state).await.expect("store");
        let loaded = store.load(&token).await.expect("load").expect("found");
        let loaded_json = serde_json::to_string(&loaded).expect("serialize loaded");

        assert_eq!(loaded.session_id, "test-session");
        assert_eq!(loaded.iterations_completed, 3);
        assert_eq!(loaded.tool_calls_made, 5);
        assert_eq!(loaded.tokens_generated, 1200);
        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(loaded.tool_results.len(), 1);
        assert_eq!(loaded.exploration_branches.len(), 1);
        assert_eq!(original_json, loaded_json);
    }

    #[tokio::test]
    async fn test_no_silent_field_loss() {
        let store = InMemoryContinuationStore::with_defaults();
        let state = make_state(stuck_termination());

        let original_json = serde_json::to_string(&state).expect("serialize");
        let token = store.store(state).await.expect("store");
        let loaded = store.load(&token).await.expect("load").expect("found");
        let loaded_json = serde_json::to_string(&loaded).expect("serialize loaded");

        assert_eq!(
            original_json, loaded_json,
            "JSON roundtrip must be lossless"
        );
    }

    #[tokio::test]
    async fn test_store_with_tool_results_roundtrip() {
        let store = InMemoryContinuationStore::with_defaults();

        let state = ContinuationState {
            token: generate_token("sess-complex"),
            session_id: "sess-complex".to_string(),
            messages: vec![
                ContextMessage {
                    role: "user".to_string(),
                    content: "Find the config file".to_string(),
                    tool_call_id: None,
                },
                ContextMessage {
                    role: "assistant".to_string(),
                    content: "I'll search for it...".to_string(),
                    tool_call_id: None,
                },
            ],
            tool_results: vec![AgenticToolResult {
                call_id: "call_1".to_string(),
                tool_name: "read_file".to_string(),
                status: super::super::types::ResultStatus::Success,
                data: serde_json::json!({"path": "/etc/app.conf", "lines": 42}),
                confidence: super::super::types::Confidence::Measured,
                latency_ms: 15,
                truncated: false,
            }],
            exploration_branches: vec![ExplorationBranch {
                description: "Tried JSON config".to_string(),
                tool_calls: vec!["search_files".to_string()],
                productive: false,
                findings: None,
            }],
            iterations_completed: 3,
            tool_calls_made: 5,
            tokens_generated: 1200,
            loop_config: LoopConfig::default(),
            autonomy: AutonomyGrant::default(),
            system_prompt: None,
            working_dir: Some("/opt/project".to_string()),
            termination: stuck_termination(),
            stored_at: SystemTime::now(),
        };

        let token = store.store(state).await.expect("store");
        let loaded = store.load(&token).await.expect("load").expect("found");

        assert_eq!(loaded.tool_results.len(), 1);
        assert_eq!(loaded.tool_results[0].call_id, "call_1");
        assert_eq!(loaded.tool_results[0].tool_name, "read_file");
        assert_eq!(loaded.exploration_branches.len(), 1);
        assert_eq!(
            loaded.exploration_branches[0].description,
            "Tried JSON config"
        );
        assert_eq!(loaded.iterations_completed, 3);
        assert_eq!(loaded.tool_calls_made, 5);
        assert_eq!(loaded.tokens_generated, 1200);
    }

    // =======================================================================
    // §11.2 TTL and Eviction
    // =======================================================================

    #[tokio::test]
    async fn test_expired_state_returns_none() {
        let store = InMemoryContinuationStore::new(Duration::from_millis(100), 100);
        let state = make_state(stuck_termination());
        let token = store.store(state).await.expect("store");

        // Before TTL — should be present
        let loaded = store.load(&token).await.expect("load");
        assert!(loaded.is_some(), "state should exist before TTL");

        // Wait past TTL
        tokio::time::sleep(Duration::from_millis(150)).await;

        // After TTL — should be gone
        let loaded = store.load(&token).await.expect("load");
        assert!(loaded.is_none(), "state should be expired after TTL");
    }

    #[tokio::test]
    async fn test_lru_eviction_removes_oldest() {
        let store = InMemoryContinuationStore::new(Duration::from_secs(3600), 3);

        let mut tokens = Vec::new();
        for i in 0..3 {
            let mut state = make_state(stuck_termination());
            state.session_id = format!("sess_{i}");
            state.token = generate_token(&state.session_id);
            // Stagger stored_at so LRU ordering is deterministic
            state.stored_at = SystemTime::now() - Duration::from_secs(100 - i as u64);
            let token = store.store(state).await.expect("store");
            tokens.push(token);
        }

        // All 3 should be present
        for token in &tokens {
            assert!(
                store.load(token).await.expect("load").is_some(),
                "all initial states should be present"
            );
        }

        // Store a 4th — should evict the oldest (token[0] was stored first
        // with earliest stored_at, and LRU is based on last_accessed which
        // is set to now() on store/load. Since we loaded all three above,
        // they all got refreshed. The 4th insert evicts whichever was
        // last-accessed earliest. We'll verify that one got evicted.)
        let mut state4 = make_state(stuck_termination());
        state4.session_id = "sess_3".to_string();
        state4.token = generate_token("sess_3");
        let token4 = store.store(state4).await.expect("store");

        assert_eq!(store.len(), 3, "should still have max_entries");
        assert!(
            store.load(&token4).await.expect("load").is_some(),
            "newest entry should exist"
        );

        // At least one of the original three was evicted
        let remaining: usize = futures::future::join_all(tokens.iter().map(|t| store.load(t)))
            .await
            .into_iter()
            .filter(|r| r.as_ref().expect("load").is_some())
            .count();

        assert_eq!(remaining, 2, "one of the original three should be evicted");
    }

    #[tokio::test]
    async fn test_access_refreshes_lru_position() {
        let store = InMemoryContinuationStore::new(Duration::from_secs(3600), 3);

        // Store 3 entries with staggered times
        let mut tokens = Vec::new();
        for i in 0..3u64 {
            let mut state = make_state(stuck_termination());
            state.session_id = format!("sess_{i}");
            state.token = generate_token(&state.session_id);
            let token = store.store(state).await.expect("store");
            tokens.push(token);
            // Small sleep to ensure distinct timestamps
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        // Access token[0] to refresh its LRU position
        tokio::time::sleep(Duration::from_millis(5)).await;
        let _ = store.load(&tokens[0]).await.expect("load");

        // Now store a 4th entry — should evict token[1] (oldest unaccessed)
        let mut state4 = make_state(stuck_termination());
        state4.session_id = "sess_3".to_string();
        state4.token = generate_token("sess_3");
        let token4 = store.store(state4).await.expect("store");

        // token[0] should remain (was refreshed)
        assert!(
            store.load(&tokens[0]).await.expect("load").is_some(),
            "refreshed entry should survive eviction"
        );

        // token[1] should be evicted (oldest unaccessed)
        assert!(
            store.load(&tokens[1]).await.expect("load").is_none(),
            "oldest unaccessed entry should be evicted"
        );

        // token[2] and token4 should exist
        assert!(store.load(&tokens[2]).await.expect("load").is_some());
        assert!(store.load(&token4).await.expect("load").is_some());
    }

    #[tokio::test]
    async fn test_cleanup_removes_only_expired() {
        let store = InMemoryContinuationStore::new(Duration::from_millis(100), 100);

        // Store 3 entries
        let mut tokens = Vec::new();
        for i in 0..3 {
            let mut state = make_state(stuck_termination());
            state.session_id = format!("sess_{i}");
            state.token = generate_token(&state.session_id);
            let token = store.store(state).await.expect("store");
            tokens.push(token);
        }

        assert_eq!(store.len(), 3);

        // Wait for TTL to expire
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Store a fresh entry (not expired)
        let mut fresh = make_state(stuck_termination());
        fresh.session_id = "sess_fresh".to_string();
        fresh.token = generate_token("sess_fresh");
        let fresh_token = store.store(fresh).await.expect("store");

        // Cleanup should remove 3 expired, keep 1 fresh
        let removed = store.cleanup_expired().await.expect("cleanup");
        assert_eq!(removed, 3, "should remove exactly the expired entries");
        assert_eq!(store.len(), 1, "only fresh entry should remain");
        assert!(
            store.load(&fresh_token).await.expect("load").is_some(),
            "fresh entry should survive cleanup"
        );
    }

    // =======================================================================
    // §11.3 Resource Arithmetic
    // =======================================================================

    #[test]
    fn test_extend_budget_after_exhaustion() {
        let state = ContinuationState {
            iterations_completed: 10,
            tool_calls_made: 50,
            tokens_generated: 16384,
            loop_config: LoopConfig {
                max_iterations: 10,
                max_tool_calls: 50,
                max_tokens: 16384,
                ..LoopConfig::default()
            },
            termination: max_iterations_termination(),
            ..make_state(max_iterations_termination())
        };

        let overrides = ConfigOverride {
            max_iterations: Some(20),
            max_tool_calls: Some(100),
            max_tokens: Some(32768),
            ..ConfigOverride::default()
        };

        let (config, _) = apply_config_override(&state, &overrides).expect("apply");
        assert_eq!(config.max_iterations, 20);
        assert_eq!(config.max_tool_calls, 100);
        assert_eq!(config.max_tokens, 32768);
    }

    #[test]
    fn test_reject_reduction_below_consumed() {
        let state = ContinuationState {
            iterations_completed: 50,
            loop_config: LoopConfig {
                max_iterations: 50,
                ..LoopConfig::default()
            },
            termination: max_iterations_termination(),
            ..make_state(max_iterations_termination())
        };

        let overrides = ConfigOverride {
            max_iterations: Some(30),
            ..ConfigOverride::default()
        };

        let result = apply_config_override(&state, &overrides);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(
            matches!(err, ResumeError::LimitBelowConsumed { .. }),
            "should reject reduction below consumed: {err}"
        );
    }

    #[test]
    fn test_reject_limit_equal_to_consumed() {
        let state = ContinuationState {
            iterations_completed: 10,
            loop_config: LoopConfig {
                max_iterations: 10,
                ..LoopConfig::default()
            },
            termination: max_iterations_termination(),
            ..make_state(max_iterations_termination())
        };

        // Equal to consumed should also fail (no room to run)
        let overrides = ConfigOverride {
            max_iterations: Some(10),
            ..ConfigOverride::default()
        };

        let result = apply_config_override(&state, &overrides);
        assert!(
            result.is_err(),
            "limit equal to consumed should be rejected"
        );
    }

    // =======================================================================
    // §11.4 Resume Semantics
    // =======================================================================

    #[test]
    fn test_additional_context_appended_as_user_message() {
        let state = ContinuationState {
            messages: vec![
                ContextMessage {
                    role: "user".to_string(),
                    content: "Find the config file".to_string(),
                    tool_call_id: None,
                },
                ContextMessage {
                    role: "assistant".to_string(),
                    content: "I'll search...".to_string(),
                    tool_call_id: None,
                },
            ],
            ..make_state(stuck_termination())
        };

        let resumed = build_resumed_messages(&state, Some("The config is at /opt/app/config.yaml"));

        assert_eq!(resumed.len(), 3);
        assert_eq!(resumed[2].role, "user");
        assert_eq!(resumed[2].content, "The config is at /opt/app/config.yaml");
        assert!(resumed[2].tool_call_id.is_none());
    }

    #[test]
    fn test_no_additional_context_preserves_messages() {
        let state = make_state(stuck_termination());
        let original_len = state.messages.len();

        let resumed = build_resumed_messages(&state, None);
        assert_eq!(resumed.len(), original_len);
    }

    #[test]
    fn test_system_prompt_immutable() {
        // The system_prompt is not part of ConfigOverride. Verify that
        // apply_config_override does not provide any way to change it.
        let state = make_state(stuck_termination());
        let overrides = ConfigOverride::default();

        let (_, _) = apply_config_override(&state, &overrides).expect("apply");
        // system_prompt is preserved on the ContinuationState, not in
        // LoopConfig or AutonomyGrant — so the override cannot touch it.
        assert_eq!(state.system_prompt, Some("You are a helper.".to_string()));
    }

    #[test]
    fn test_working_dir_immutable() {
        // Same principle: working_dir is on ContinuationState, not in the
        // overridable config.
        let state = make_state(stuck_termination());
        let overrides = ConfigOverride::default();

        let (_, _) = apply_config_override(&state, &overrides).expect("apply");
        assert_eq!(state.working_dir, Some("/home/user/project".to_string()));
    }

    #[test]
    fn test_autonomy_widening() {
        let mut state = make_state(stuck_termination());
        state.autonomy.auto_approve = vec![ToolPattern::Tool("read_file".to_string())];
        state.autonomy.forbidden = vec![
            ToolPattern::Tool("bash".to_string()),
            ToolPattern::Tool("write_file".to_string()),
        ];

        let overrides = ConfigOverride {
            auto_approve: Some(vec!["list_files".to_string()]),
            remove_forbidden: Some(vec!["write_file".to_string()]),
            ..ConfigOverride::default()
        };

        let (_, autonomy) = apply_config_override(&state, &overrides).expect("apply");

        // Original pattern preserved + new one added
        assert_eq!(autonomy.auto_approve.len(), 2);

        // "write_file" removed from forbidden, "bash" remains
        assert_eq!(autonomy.forbidden.len(), 1);
        assert!(matches!(&autonomy.forbidden[0], ToolPattern::Tool(n) if n == "bash"));
    }

    // =======================================================================
    // §11.5 Resumability by Termination Type
    // =======================================================================

    #[test]
    fn test_answer_provided_not_resumable() {
        let term = TerminationReason::Natural(NaturalTermination::AnswerProvided {
            answer: "42".to_string(),
            confidence: 0.95,
        });
        assert!(!term.is_resumable());
    }

    #[test]
    fn test_task_complete_not_resumable() {
        let term = TerminationReason::Natural(NaturalTermination::TaskComplete);
        assert!(!term.is_resumable());
    }

    #[test]
    fn test_agent_stuck_is_resumable() {
        let term = stuck_termination();
        assert!(term.is_resumable());
    }

    #[test]
    fn test_agent_yielded_is_resumable() {
        let term = TerminationReason::Natural(NaturalTermination::AgentYielded {
            partial: Some("partial progress".to_string()),
            reason: "needs specialist".to_string(),
        });
        assert!(term.is_resumable());
    }

    #[test]
    fn test_max_iterations_resumable() {
        let term = max_iterations_termination();
        assert!(term.is_resumable());
    }

    #[test]
    fn test_token_budget_exhausted_resumable() {
        let term = TerminationReason::Resource(ResourceTermination::TokenBudgetExhausted {
            generated: 16384,
            budget: 16384,
        });
        assert!(term.is_resumable());
    }

    #[test]
    fn test_wall_time_exceeded_resumable() {
        let term = TerminationReason::Resource(ResourceTermination::WallTimeExceeded {
            elapsed: Duration::from_secs(300),
            limit: Duration::from_secs(300),
        });
        assert!(term.is_resumable());
    }

    #[test]
    fn test_tool_call_limit_resumable() {
        let term = TerminationReason::Resource(ResourceTermination::ToolCallLimitReached {
            calls: 50,
            limit: 50,
        });
        assert!(term.is_resumable());
    }

    #[test]
    fn test_client_cancelled_not_resumable() {
        let term = TerminationReason::External(ExternalTermination::ClientCancelled);
        assert!(!term.is_resumable());
    }

    #[test]
    fn test_operator_terminated_not_resumable() {
        let term = TerminationReason::External(ExternalTermination::OperatorTerminated {
            reason: "maintenance".to_string(),
        });
        assert!(!term.is_resumable());
    }

    #[test]
    fn test_system_shutdown_not_resumable() {
        let term = TerminationReason::External(ExternalTermination::SystemShutdown);
        assert!(!term.is_resumable());
    }

    // =======================================================================
    // §11.1 + §11.3 + §11.5 Property Tests
    // =======================================================================

    mod proptest_continuation {
        use super::*;

        fn arb_termination_reason() -> impl Strategy<Value = TerminationReason> {
            prop_oneof![
                // Natural
                (any::<u32>(), 0.0f32..1.0).prop_map(|(_, conf)| {
                    TerminationReason::Natural(NaturalTermination::AnswerProvided {
                        answer: "answer".to_string(),
                        confidence: conf,
                    })
                }),
                Just(TerminationReason::Natural(NaturalTermination::TaskComplete)),
                (1u32..10).prop_map(|attempts| {
                    TerminationReason::Natural(NaturalTermination::AgentStuck {
                        attempts,
                        request: super::super::super::types::StuckRequest::Clarification(vec![
                            "question".to_string(),
                        ]),
                    })
                }),
                Just(TerminationReason::Natural(
                    NaturalTermination::AgentYielded {
                        partial: None,
                        reason: "yield".to_string(),
                    }
                )),
                // Resource
                (1u32..100, 1u32..100).prop_map(|(c, l)| {
                    TerminationReason::Resource(ResourceTermination::MaxIterations {
                        completed: c,
                        limit: l,
                    })
                }),
                (1u32..16384, 1u32..16384).prop_map(|(g, b)| {
                    TerminationReason::Resource(ResourceTermination::TokenBudgetExhausted {
                        generated: g,
                        budget: b,
                    })
                }),
                (1u32..100, 1u32..100).prop_map(|(c, l)| {
                    TerminationReason::Resource(ResourceTermination::ToolCallLimitReached {
                        calls: c,
                        limit: l,
                    })
                }),
                // External
                Just(TerminationReason::External(
                    ExternalTermination::ClientCancelled
                )),
                Just(TerminationReason::External(
                    ExternalTermination::SystemShutdown
                )),
            ]
        }

        proptest! {
            #[test]
            fn prop_resumable_terminations(term in arb_termination_reason()) {
                let resumable = term.is_resumable();

                match &term {
                    TerminationReason::Natural(NaturalTermination::AgentStuck { .. })
                    | TerminationReason::Natural(NaturalTermination::AgentYielded { .. }) => {
                        prop_assert!(resumable, "Stuck/Yielded should be resumable");
                    }
                    TerminationReason::Natural(NaturalTermination::AnswerProvided { .. })
                    | TerminationReason::Natural(NaturalTermination::TaskComplete) => {
                        prop_assert!(!resumable, "Completed tasks should not be resumable");
                    }
                    TerminationReason::Resource(_) => {
                        prop_assert!(resumable, "Resource terminations should be resumable");
                    }
                    TerminationReason::External(_) => {
                        prop_assert!(!resumable, "External terminations should not be resumable");
                    }
                }
            }

            #[test]
            fn prop_new_limit_must_exceed_consumed(
                consumed in 0u32..1000,
                new_limit in 0u32..2000,
            ) {
                let state = ContinuationState {
                    iterations_completed: consumed,
                    loop_config: LoopConfig {
                        max_iterations: consumed.saturating_add(10),
                        ..LoopConfig::default()
                    },
                    ..make_state(max_iterations_termination())
                };

                let overrides = ConfigOverride {
                    max_iterations: Some(new_limit),
                    ..ConfigOverride::default()
                };

                let result = apply_config_override(&state, &overrides);

                if new_limit <= consumed {
                    prop_assert!(result.is_err(), "should reject limit <= consumed");
                } else {
                    prop_assert!(result.is_ok(), "should accept limit > consumed");
                    let (config, _) = result.unwrap();
                    prop_assert_eq!(config.max_iterations, new_limit);
                }
            }

            #[test]
            fn prop_remaining_budget_arithmetic(
                consumed_iters in 0u32..100,
                consumed_calls in 0u32..200,
                consumed_tokens in 0u32..8000,
            ) {
                let new_iters = consumed_iters + 1 + (consumed_iters % 100);
                let new_calls = consumed_calls + 1 + (consumed_calls % 200);
                let new_tokens = consumed_tokens + 1 + (consumed_tokens % 8000);

                let state = ContinuationState {
                    iterations_completed: consumed_iters,
                    tool_calls_made: consumed_calls,
                    tokens_generated: consumed_tokens,
                    loop_config: LoopConfig {
                        max_iterations: consumed_iters.saturating_add(10),
                        max_tool_calls: consumed_calls.saturating_add(10),
                        max_tokens: consumed_tokens.saturating_add(100),
                        ..LoopConfig::default()
                    },
                    ..make_state(max_iterations_termination())
                };

                let overrides = ConfigOverride {
                    max_iterations: Some(new_iters),
                    max_tool_calls: Some(new_calls),
                    max_tokens: Some(new_tokens),
                    ..ConfigOverride::default()
                };

                let (config, _) = apply_config_override(&state, &overrides).expect("apply");
                prop_assert_eq!(config.max_iterations, new_iters);
                prop_assert_eq!(config.max_tool_calls, new_calls);
                prop_assert_eq!(config.max_tokens, new_tokens);
            }
        }
    }

    // =======================================================================
    // Token generation
    // =======================================================================

    #[test]
    fn test_token_format() {
        let token = generate_token("my-session");
        assert!(
            token.starts_with("cont_my-session_"),
            "token should start with cont_{{session_id}}_: {token}"
        );
        // UUID part should be 36 chars (8-4-4-4-12)
        let uuid_part = &token["cont_my-session_".len()..];
        assert_eq!(uuid_part.len(), 36, "UUID should be 36 chars: {uuid_part}");
    }

    #[test]
    fn test_create_continuation_state() {
        let state = create_continuation_state(
            "test-sess",
            make_messages(2),
            vec![make_tool_result("c1", "read_file")],
            vec![make_exploration("branch1", true)],
            5,
            10,
            2000,
            LoopConfig::default(),
            AutonomyGrant::default(),
            Some("prompt".to_string()),
            Some("/work".to_string()),
            stuck_termination(),
        );

        assert!(state.token.starts_with("cont_test-sess_"));
        assert_eq!(state.session_id, "test-sess");
        assert_eq!(state.messages.len(), 2);
        assert_eq!(state.tool_results.len(), 1);
        assert_eq!(state.exploration_branches.len(), 1);
        assert_eq!(state.iterations_completed, 5);
        assert_eq!(state.tool_calls_made, 10);
        assert_eq!(state.tokens_generated, 2000);
        assert_eq!(state.system_prompt, Some("prompt".to_string()));
        assert_eq!(state.working_dir, Some("/work".to_string()));
    }

    // =======================================================================
    // InMemoryContinuationStore misc
    // =======================================================================

    #[tokio::test]
    async fn test_load_missing_token_returns_none() {
        let store = InMemoryContinuationStore::with_defaults();
        let result = store.load("nonexistent").await.expect("load");
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_remove_then_load_returns_none() {
        let store = InMemoryContinuationStore::with_defaults();
        let state = make_state(stuck_termination());
        let token = store.store(state).await.expect("store");

        store.remove(&token).await.expect("remove");

        let result = store.load(&token).await.expect("load");
        assert!(result.is_none());
    }

    #[test]
    fn test_store_is_empty_and_len() {
        let store = InMemoryContinuationStore::with_defaults();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }
}
