//! Tool approval protocol for the agentic loop.
//!
//! Implements the interactive approval workflow from AGENTIC-LOOP-SPEC.md §9.4.
//! When a tool call requires approval, the executor registers a pending request
//! through the [`ApprovalGate`], emits a `ToolApprovalRequired` event, and blocks
//! until a decision arrives or the timeout expires.
//!
//! External systems (HTTP endpoints, CLI prompts) deliver decisions through
//! the gate, which routes them to the waiting executor via oneshot channels.
//!
//! ## `ApproveAlways` Semantics
//!
//! When a client responds with `ApproveAlways`, the gate records a runtime
//! override so subsequent matching calls bypass the approval flow:
//!
//! - `ThisCall` — no persistent override (equivalent to `Approve`)
//! - `ThisTool` — auto-approve all future calls to the same tool name
//! - `ThisSession` — auto-approve all tool calls for the rest of the session

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Decision delivered by the client for a pending tool approval.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApprovalDecision {
    /// Execute this specific tool call.
    Approve,
    /// Skip this tool call (returns recoverable error to agent).
    Deny,
    /// Execute and add the tool pattern to auto-approve for the given scope.
    ApproveAlways {
        /// How broadly to auto-approve future calls.
        scope: ApprovalScope,
    },
}

/// Scope for [`ApprovalDecision::ApproveAlways`] decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ApprovalScope {
    /// Only this call (equivalent to [`ApprovalDecision::Approve`]).
    ThisCall,
    /// Auto-approve all future calls to this tool name.
    ThisTool,
    /// Auto-approve all tool calls for the remainder of the session.
    ThisSession,
}

/// Errors from approval operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ApprovalError {
    /// The `call_id` was not found in the pending set.
    ///
    /// This means the `call_id` never existed or was already consumed (oneshot).
    #[error("approval request not found: {call_id}")]
    NotFound {
        /// The `call_id` that was looked up.
        call_id: String,
    },
    /// The executor already moved on (receiver dropped due to timeout).
    #[error("approval expired: executor no longer waiting for {call_id}")]
    Expired {
        /// The `call_id` that expired.
        call_id: String,
    },
}

/// Information about a pending approval request (serializable, no channels).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingApprovalInfo {
    /// The tool call ID awaiting approval.
    pub call_id: String,
    /// Name of the tool.
    pub tool_name: String,
    /// Tool call arguments.
    pub arguments: serde_json::Value,
    /// How long the request has been pending.
    pub elapsed: Duration,
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// A pending approval request with its oneshot sender.
struct PendingApproval {
    call_id: String,
    tool_name: String,
    arguments: serde_json::Value,
    requested_at: Instant,
    respond: oneshot::Sender<ApprovalDecision>,
}

/// Runtime overrides from `ApproveAlways` decisions.
#[derive(Debug, Default)]
struct RuntimeOverrides {
    /// If true, all tools are auto-approved for this session.
    approve_all: bool,
    /// Set of tool names that are individually auto-approved.
    approved_tools: HashSet<String>,
}

// ---------------------------------------------------------------------------
// ApprovalGate
// ---------------------------------------------------------------------------

/// Manages pending tool approval requests for a session.
///
/// The gate is shared between the executor (which registers requests and waits)
/// and external systems (which deliver decisions). Thread-safe via
/// [`parking_lot::RwLock`].
///
/// # Lifecycle
///
/// 1. Executor encounters `Permission::RequiresApproval`
/// 2. Executor calls [`request()`](Self::request) → registers the pending entry
///    and gets a `oneshot::Receiver`
/// 3. Executor emits `ToolApprovalRequired` event to the client (entry already
///    exists, so the client can deliver immediately)
/// 4. Executor awaits the receiver with a configurable timeout
/// 5. Client calls [`deliver()`](Self::deliver) with an [`ApprovalDecision`]
/// 6. Gate routes the decision through the oneshot and removes the pending entry
pub struct ApprovalGate {
    /// Pending approval requests indexed by `call_id`.
    pending: RwLock<HashMap<String, PendingApproval>>,
    /// Runtime auto-approve overrides from `ApproveAlways` decisions.
    overrides: RwLock<RuntimeOverrides>,
}

impl ApprovalGate {
    /// Creates a new approval gate with no pending requests.
    pub fn new() -> Self {
        Self {
            pending: RwLock::new(HashMap::new()),
            overrides: RwLock::new(RuntimeOverrides::default()),
        }
    }

    /// Registers a pending approval request and returns a receiver for the decision.
    ///
    /// The executor calls this when a tool requires approval, then awaits the
    /// receiver (typically with a timeout).
    pub fn request(
        &self,
        call_id: &str,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> oneshot::Receiver<ApprovalDecision> {
        let (tx, rx) = oneshot::channel();
        let entry = PendingApproval {
            call_id: call_id.to_string(),
            tool_name: tool_name.to_string(),
            arguments,
            requested_at: Instant::now(),
            respond: tx,
        };
        self.pending.write().insert(call_id.to_string(), entry);
        rx
    }

    /// Delivers an approval decision for a pending request.
    ///
    /// The pending entry is consumed (removed) on delivery. A subsequent call
    /// with the same `call_id` returns [`ApprovalError::NotFound`].
    ///
    /// If the decision is [`ApprovalDecision::ApproveAlways`], the appropriate
    /// runtime override is recorded before sending the decision.
    ///
    /// # Errors
    ///
    /// - [`ApprovalError::NotFound`] if the `call_id` is not in the pending set.
    /// - [`ApprovalError::Expired`] if the receiver was dropped (executor timed out).
    pub fn deliver(&self, call_id: &str, decision: ApprovalDecision) -> Result<(), ApprovalError> {
        let entry =
            self.pending
                .write()
                .remove(call_id)
                .ok_or_else(|| ApprovalError::NotFound {
                    call_id: call_id.to_string(),
                })?;

        // Record runtime override before sending (so it's visible immediately)
        if let ApprovalDecision::ApproveAlways { scope } = &decision {
            self.apply_override(*scope, &entry.tool_name);
        }

        entry
            .respond
            .send(decision)
            .map_err(|_| ApprovalError::Expired {
                call_id: call_id.to_string(),
            })
    }

    /// Returns information about all pending approval requests.
    pub fn pending(&self) -> Vec<PendingApprovalInfo> {
        let map = self.pending.read();
        map.values()
            .map(|p| PendingApprovalInfo {
                call_id: p.call_id.clone(),
                tool_name: p.tool_name.clone(),
                arguments: p.arguments.clone(),
                elapsed: p.requested_at.elapsed(),
            })
            .collect()
    }

    /// Returns the number of pending approval requests.
    pub fn pending_count(&self) -> usize {
        self.pending.read().len()
    }

    /// Checks if a tool is auto-approved by runtime overrides.
    ///
    /// This is checked before the static [`AutonomyGrant`](super::AutonomyGrant)
    /// to allow `ApproveAlways` decisions to bypass future approval requirements.
    pub fn is_runtime_approved(&self, tool_name: &str) -> bool {
        let overrides = self.overrides.read();
        overrides.approve_all || overrides.approved_tools.contains(tool_name)
    }

    /// Removes pending requests where the receiver has been dropped.
    ///
    /// Returns the number of entries removed.
    pub fn cleanup_closed(&self) -> usize {
        let mut map = self.pending.write();
        let before = map.len();
        map.retain(|_, p| !p.respond.is_closed());
        before - map.len()
    }

    /// Removes pending requests older than the given duration.
    ///
    /// Returns the number of entries removed.
    pub fn cleanup_expired(&self, timeout: Duration) -> usize {
        let mut map = self.pending.write();
        let before = map.len();
        map.retain(|_, p| p.requested_at.elapsed() < timeout);
        before - map.len()
    }

    /// Resets all runtime overrides (useful for testing).
    pub fn reset_overrides(&self) {
        let mut overrides = self.overrides.write();
        overrides.approve_all = false;
        overrides.approved_tools.clear();
    }

    fn apply_override(&self, scope: ApprovalScope, tool_name: &str) {
        match scope {
            ApprovalScope::ThisCall => {
                // No persistent override — equivalent to a single Approve
            },
            ApprovalScope::ThisTool => {
                self.overrides
                    .write()
                    .approved_tools
                    .insert(tool_name.to_string());
            },
            ApprovalScope::ThisSession => {
                self.overrides.write().approve_all = true;
            },
        }
    }
}

impl Default for ApprovalGate {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ApprovalGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApprovalGate")
            .field("pending_count", &self.pending.read().len())
            .field("overrides", &*self.overrides.read())
            .finish()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // -----------------------------------------------------------------------
    // §10.1 Approval Handshake
    // -----------------------------------------------------------------------

    #[test]
    fn test_request_creates_pending_entry() {
        let gate = ApprovalGate::new();
        assert_eq!(gate.pending_count(), 0);

        let _rx = gate.request("call_1", "bash", serde_json::json!({"command": "ls"}));

        assert_eq!(gate.pending_count(), 1);
        let pending = gate.pending();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].call_id, "call_1");
        assert_eq!(pending[0].tool_name, "bash");
        assert_eq!(pending[0].arguments, serde_json::json!({"command": "ls"}));
    }

    #[tokio::test]
    async fn test_approve_delivers_to_receiver() {
        let gate = ApprovalGate::new();
        let rx = gate.request("call_1", "bash", serde_json::json!({}));

        gate.deliver("call_1", ApprovalDecision::Approve).unwrap();

        let decision = rx.await.unwrap();
        assert_eq!(decision, ApprovalDecision::Approve);
    }

    #[tokio::test]
    async fn test_deny_delivers_to_receiver() {
        let gate = ApprovalGate::new();
        let rx = gate.request("call_1", "bash", serde_json::json!({}));

        gate.deliver("call_1", ApprovalDecision::Deny).unwrap();

        let decision = rx.await.unwrap();
        assert_eq!(decision, ApprovalDecision::Deny);
    }

    #[tokio::test]
    async fn test_approve_always_delivers_to_receiver() {
        let gate = ApprovalGate::new();
        let rx = gate.request("call_1", "bash", serde_json::json!({}));

        let expected = ApprovalDecision::ApproveAlways {
            scope: ApprovalScope::ThisTool,
        };
        gate.deliver("call_1", expected.clone()).unwrap();

        let decision = rx.await.unwrap();
        assert_eq!(decision, expected);
    }

    #[test]
    fn test_deliver_unknown_call_id_returns_not_found() {
        let gate = ApprovalGate::new();

        let result = gate.deliver("nonexistent", ApprovalDecision::Approve);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ApprovalError::NotFound { call_id } if call_id == "nonexistent"
        ));
    }

    #[test]
    fn test_deliver_to_empty_gate_returns_not_found() {
        let gate = ApprovalGate::new();

        let result = gate.deliver("call_1", ApprovalDecision::Deny);
        assert!(matches!(
            result.unwrap_err(),
            ApprovalError::NotFound { .. }
        ));
    }

    // -----------------------------------------------------------------------
    // §10.2 Timeout Behavior
    // -----------------------------------------------------------------------

    #[test]
    fn test_dropped_receiver_causes_expired_error() {
        let gate = ApprovalGate::new();
        let rx = gate.request("call_1", "bash", serde_json::json!({}));

        // Executor drops the receiver (simulating timeout)
        drop(rx);

        // Pending entry still exists but sender will fail
        assert_eq!(gate.pending_count(), 1);

        let result = gate.deliver("call_1", ApprovalDecision::Approve);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ApprovalError::Expired { call_id } if call_id == "call_1"
        ));
    }

    #[tokio::test]
    async fn test_tokio_timeout_expires() {
        let gate = Arc::new(ApprovalGate::new());
        let rx = gate.request("call_1", "bash", serde_json::json!({}));

        // Simulate executor waiting with a short timeout — no one delivers
        let result = tokio::time::timeout(Duration::from_millis(50), rx).await;
        assert!(result.is_err(), "Should have timed out");
    }

    #[tokio::test]
    async fn test_decision_just_before_timeout_is_honored() {
        let gate = Arc::new(ApprovalGate::new());
        let rx = gate.request("call_1", "bash", serde_json::json!({}));

        // Deliver after a short delay but well before the generous timeout
        let gate_clone = Arc::clone(&gate);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            gate_clone
                .deliver("call_1", ApprovalDecision::Approve)
                .unwrap();
        });

        let result = tokio::time::timeout(Duration::from_millis(200), rx).await;
        assert!(result.is_ok(), "Should not have timed out");
        assert_eq!(result.unwrap().unwrap(), ApprovalDecision::Approve);
    }

    #[test]
    fn test_cleanup_expired_removes_old_entries() {
        let gate = ApprovalGate::new();
        let _rx = gate.request("call_1", "bash", serde_json::json!({}));
        assert_eq!(gate.pending_count(), 1);

        // Zero-duration timeout means everything is expired
        let removed = gate.cleanup_expired(Duration::ZERO);
        assert_eq!(removed, 1);
        assert_eq!(gate.pending_count(), 0);
    }

    #[test]
    fn test_cleanup_expired_keeps_fresh_entries() {
        let gate = ApprovalGate::new();
        let _rx = gate.request("call_1", "bash", serde_json::json!({}));

        let removed = gate.cleanup_expired(Duration::from_secs(3600));
        assert_eq!(removed, 0);
        assert_eq!(gate.pending_count(), 1);
    }

    #[test]
    fn test_cleanup_closed_removes_dropped_receivers() {
        let gate = ApprovalGate::new();
        let rx1 = gate.request("call_1", "bash", serde_json::json!({}));
        let _rx2 = gate.request("call_2", "read_file", serde_json::json!({}));
        assert_eq!(gate.pending_count(), 2);

        // Drop one receiver (simulating executor timeout)
        drop(rx1);

        let removed = gate.cleanup_closed();
        assert_eq!(removed, 1);
        assert_eq!(gate.pending_count(), 1);

        // Remaining entry should be call_2
        let pending = gate.pending();
        assert_eq!(pending[0].call_id, "call_2");
    }

    #[test]
    fn test_cleanup_closed_preserves_active_entries() {
        let gate = ApprovalGate::new();
        let _rx1 = gate.request("call_1", "bash", serde_json::json!({}));
        let _rx2 = gate.request("call_2", "read_file", serde_json::json!({}));

        let removed = gate.cleanup_closed();
        assert_eq!(removed, 0);
        assert_eq!(gate.pending_count(), 2);
    }

    // -----------------------------------------------------------------------
    // §10.3 ApproveAlways Semantics
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_approve_always_this_tool_adds_runtime_override() {
        let gate = ApprovalGate::new();
        assert!(!gate.is_runtime_approved("bash"));

        let rx = gate.request("call_1", "bash", serde_json::json!({}));
        gate.deliver(
            "call_1",
            ApprovalDecision::ApproveAlways {
                scope: ApprovalScope::ThisTool,
            },
        )
        .unwrap();
        let _ = rx.await;

        assert!(gate.is_runtime_approved("bash"));
    }

    #[tokio::test]
    async fn test_approve_always_this_tool_scoped_to_tool_name() {
        let gate = ApprovalGate::new();
        let rx = gate.request("call_1", "bash", serde_json::json!({}));

        gate.deliver(
            "call_1",
            ApprovalDecision::ApproveAlways {
                scope: ApprovalScope::ThisTool,
            },
        )
        .unwrap();
        let _ = rx.await;

        assert!(gate.is_runtime_approved("bash"));
        assert!(!gate.is_runtime_approved("write_file"));
        assert!(!gate.is_runtime_approved("read_file"));
        assert!(!gate.is_runtime_approved("edit_file"));
    }

    #[tokio::test]
    async fn test_approve_always_this_session_approves_all_tools() {
        let gate = ApprovalGate::new();
        let rx = gate.request("call_1", "bash", serde_json::json!({}));

        gate.deliver(
            "call_1",
            ApprovalDecision::ApproveAlways {
                scope: ApprovalScope::ThisSession,
            },
        )
        .unwrap();
        let _ = rx.await;

        assert!(gate.is_runtime_approved("bash"));
        assert!(gate.is_runtime_approved("write_file"));
        assert!(gate.is_runtime_approved("read_file"));
        assert!(gate.is_runtime_approved("any_tool"));
    }

    #[tokio::test]
    async fn test_approve_always_this_call_no_persistent_override() {
        let gate = ApprovalGate::new();
        let rx = gate.request("call_1", "bash", serde_json::json!({}));

        gate.deliver(
            "call_1",
            ApprovalDecision::ApproveAlways {
                scope: ApprovalScope::ThisCall,
            },
        )
        .unwrap();
        let _ = rx.await;

        // ThisCall should NOT add a runtime override
        assert!(!gate.is_runtime_approved("bash"));
    }

    #[tokio::test]
    async fn test_multiple_tool_overrides_accumulate() {
        let gate = ApprovalGate::new();

        // Approve "bash"
        let rx1 = gate.request("call_1", "bash", serde_json::json!({}));
        gate.deliver(
            "call_1",
            ApprovalDecision::ApproveAlways {
                scope: ApprovalScope::ThisTool,
            },
        )
        .unwrap();
        let _ = rx1.await;

        // Approve "write_file"
        let rx2 = gate.request("call_2", "write_file", serde_json::json!({}));
        gate.deliver(
            "call_2",
            ApprovalDecision::ApproveAlways {
                scope: ApprovalScope::ThisTool,
            },
        )
        .unwrap();
        let _ = rx2.await;

        assert!(gate.is_runtime_approved("bash"));
        assert!(gate.is_runtime_approved("write_file"));
        assert!(!gate.is_runtime_approved("read_file"));
    }

    #[test]
    fn test_reset_overrides_clears_all() {
        let gate = ApprovalGate::new();
        gate.overrides
            .write()
            .approved_tools
            .insert("bash".to_string());
        gate.overrides.write().approve_all = true;

        assert!(gate.is_runtime_approved("bash"));
        assert!(gate.is_runtime_approved("anything"));

        gate.reset_overrides();

        assert!(!gate.is_runtime_approved("bash"));
        assert!(!gate.is_runtime_approved("anything"));
    }

    #[test]
    fn test_no_runtime_overrides_initially() {
        let gate = ApprovalGate::new();
        assert!(!gate.is_runtime_approved("bash"));
        assert!(!gate.is_runtime_approved("read_file"));
        assert!(!gate.is_runtime_approved("write_file"));
    }

    // -----------------------------------------------------------------------
    // §10.4 Concurrent Approvals
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_multiple_pending_approvals_independent() {
        let gate = Arc::new(ApprovalGate::new());
        let rx_a = gate.request("call_a", "bash", serde_json::json!({}));
        let rx_b = gate.request("call_b", "write_file", serde_json::json!({}));
        assert_eq!(gate.pending_count(), 2);

        // Approve A, deny B
        gate.deliver("call_a", ApprovalDecision::Approve).unwrap();
        gate.deliver("call_b", ApprovalDecision::Deny).unwrap();

        assert_eq!(rx_a.await.unwrap(), ApprovalDecision::Approve);
        assert_eq!(rx_b.await.unwrap(), ApprovalDecision::Deny);
        assert_eq!(gate.pending_count(), 0);
    }

    #[tokio::test]
    async fn test_one_timeout_one_approved() {
        let gate = Arc::new(ApprovalGate::new());
        let rx_a = gate.request("call_a", "bash", serde_json::json!({}));
        let rx_b = gate.request("call_b", "read_file", serde_json::json!({}));

        // Approve A only
        gate.deliver("call_a", ApprovalDecision::Approve).unwrap();

        assert_eq!(rx_a.await.unwrap(), ApprovalDecision::Approve);

        // B times out
        let result_b = tokio::time::timeout(Duration::from_millis(50), rx_b).await;
        assert!(result_b.is_err(), "B should have timed out");

        // Delivering to B after timeout: entry still exists but receiver is dropped
        // The pending entry was NOT removed by the timeout (executor side cleanup needed)
        // Delivery should fail with Expired since the receiver was dropped by timeout
        let result = gate.deliver("call_b", ApprovalDecision::Approve);
        assert!(matches!(result.unwrap_err(), ApprovalError::Expired { .. }));
    }

    #[tokio::test]
    async fn test_concurrent_delivery_from_multiple_tasks() {
        let gate = Arc::new(ApprovalGate::new());
        let rx1 = gate.request("call_1", "bash", serde_json::json!({}));
        let rx2 = gate.request("call_2", "read_file", serde_json::json!({}));
        let rx3 = gate.request("call_3", "write_file", serde_json::json!({}));

        let g1 = Arc::clone(&gate);
        let g2 = Arc::clone(&gate);
        let g3 = Arc::clone(&gate);

        let (r1, r2, r3) = tokio::join!(
            tokio::spawn(async move { g1.deliver("call_1", ApprovalDecision::Approve) }),
            tokio::spawn(async move { g2.deliver("call_2", ApprovalDecision::Deny) }),
            tokio::spawn(async move {
                g3.deliver(
                    "call_3",
                    ApprovalDecision::ApproveAlways {
                        scope: ApprovalScope::ThisTool,
                    },
                )
            }),
        );

        assert!(r1.unwrap().is_ok());
        assert!(r2.unwrap().is_ok());
        assert!(r3.unwrap().is_ok());

        assert_eq!(rx1.await.unwrap(), ApprovalDecision::Approve);
        assert_eq!(rx2.await.unwrap(), ApprovalDecision::Deny);
        assert_eq!(
            rx3.await.unwrap(),
            ApprovalDecision::ApproveAlways {
                scope: ApprovalScope::ThisTool,
            }
        );

        // Only call_3 was ApproveAlways(ThisTool) for "write_file"
        assert!(gate.is_runtime_approved("write_file"));
        assert!(!gate.is_runtime_approved("bash"));
        assert!(!gate.is_runtime_approved("read_file"));
    }

    // -----------------------------------------------------------------------
    // §10.5 Oneshot Consumption
    // -----------------------------------------------------------------------

    #[test]
    fn test_oneshot_consumed_once() {
        let gate = ApprovalGate::new();
        let _rx = gate.request("call_1", "bash", serde_json::json!({}));

        // First delivery succeeds
        let first = gate.deliver("call_1", ApprovalDecision::Approve);
        assert!(first.is_ok());

        // Second delivery fails — entry was removed
        let second = gate.deliver("call_1", ApprovalDecision::Approve);
        assert!(second.is_err());
        assert!(matches!(
            second.unwrap_err(),
            ApprovalError::NotFound { call_id } if call_id == "call_1"
        ));
    }

    #[test]
    fn test_pending_list_reflects_state() {
        let gate = ApprovalGate::new();
        assert!(gate.pending().is_empty());

        let _rx1 = gate.request("call_1", "bash", serde_json::json!({}));
        assert_eq!(gate.pending().len(), 1);

        let _rx2 = gate.request("call_2", "read_file", serde_json::json!({}));
        assert_eq!(gate.pending().len(), 2);

        // Deliver one — it's consumed
        gate.deliver("call_1", ApprovalDecision::Approve).unwrap();
        assert_eq!(gate.pending().len(), 1);

        let remaining = gate.pending();
        assert_eq!(remaining[0].call_id, "call_2");
    }

    #[test]
    fn test_pending_info_contains_arguments() {
        let gate = ApprovalGate::new();
        let args = serde_json::json!({"command": "git status", "timeout_secs": 30});
        let _rx = gate.request("call_1", "bash", args.clone());

        let pending = gate.pending();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].arguments, args);
    }

    #[test]
    fn test_debug_format() {
        let gate = ApprovalGate::new();
        let debug = format!("{gate:?}");
        assert!(debug.contains("ApprovalGate"));
        assert!(debug.contains("pending_count"));
    }

    #[test]
    fn test_default_gate() {
        let gate = ApprovalGate::default();
        assert_eq!(gate.pending_count(), 0);
        assert!(!gate.is_runtime_approved("anything"));
    }

    // -----------------------------------------------------------------------
    // Proptest
    // -----------------------------------------------------------------------

    mod proptest_approval {
        use super::*;
        use proptest::prelude::*;

        fn approval_decision_strategy() -> impl Strategy<Value = ApprovalDecision> {
            prop_oneof![
                Just(ApprovalDecision::Approve),
                Just(ApprovalDecision::Deny),
                prop_oneof![
                    Just(ApprovalScope::ThisCall),
                    Just(ApprovalScope::ThisTool),
                    Just(ApprovalScope::ThisSession),
                ]
                .prop_map(|scope| ApprovalDecision::ApproveAlways { scope }),
            ]
        }

        fn tool_name_strategy() -> impl Strategy<Value = String> {
            prop_oneof![
                Just("bash".to_string()),
                Just("read_file".to_string()),
                Just("write_file".to_string()),
                Just("edit_file".to_string()),
                Just("list_files".to_string()),
                Just("search_files".to_string()),
                Just("claude_code".to_string()),
            ]
        }

        proptest! {
            /// Once delivered, the call_id is consumed — second delivery returns NotFound.
            #[test]
            fn prop_oneshot_consumed_once(
                decision in approval_decision_strategy(),
            ) {
                let gate = ApprovalGate::new();
                let _rx = gate.request("call_x", "bash", serde_json::json!({}));

                let first = gate.deliver("call_x", decision.clone());
                prop_assert!(first.is_ok());

                let second = gate.deliver("call_x", decision);
                prop_assert!(second.is_err());
                let is_not_found = matches!(
                    second.unwrap_err(),
                    ApprovalError::NotFound { .. }
                );
                prop_assert!(is_not_found, "expected NotFound after consumption");
            }

            /// ApproveAlways(ThisTool) makes the tool runtime-approved.
            #[test]
            fn prop_approve_always_this_tool_takes_effect(
                tool_name in tool_name_strategy(),
            ) {
                let gate = ApprovalGate::new();
                let _rx = gate.request("call_x", &tool_name, serde_json::json!({}));

                prop_assert!(!gate.is_runtime_approved(&tool_name));

                gate.deliver(
                    "call_x",
                    ApprovalDecision::ApproveAlways {
                        scope: ApprovalScope::ThisTool,
                    },
                )
                .unwrap();

                prop_assert!(gate.is_runtime_approved(&tool_name));
            }

            /// ApproveAlways(ThisTool) does NOT approve different tools.
            #[test]
            fn prop_approve_always_scoped_to_tool(
                approved in tool_name_strategy(),
                other in tool_name_strategy(),
            ) {
                prop_assume!(approved != other);

                let gate = ApprovalGate::new();
                let _rx = gate.request("call_x", &approved, serde_json::json!({}));

                gate.deliver(
                    "call_x",
                    ApprovalDecision::ApproveAlways {
                        scope: ApprovalScope::ThisTool,
                    },
                )
                .unwrap();

                prop_assert!(gate.is_runtime_approved(&approved));
                prop_assert!(!gate.is_runtime_approved(&other));
            }

            /// Deny decision is correctly delivered through the oneshot.
            #[test]
            fn prop_deny_delivers_correctly(
                tool_name in tool_name_strategy(),
            ) {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("tokio runtime");

                rt.block_on(async {
                    let gate = ApprovalGate::new();
                    let rx = gate.request("call_x", &tool_name, serde_json::json!({}));
                    gate.deliver("call_x", ApprovalDecision::Deny).unwrap();

                    let decision = rx.await.unwrap();
                    assert_eq!(decision, ApprovalDecision::Deny);
                });
            }

            /// Approve decision is correctly delivered through the oneshot.
            #[test]
            fn prop_approve_delivers_correctly(
                tool_name in tool_name_strategy(),
            ) {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("tokio runtime");

                rt.block_on(async {
                    let gate = ApprovalGate::new();
                    let rx = gate.request("call_x", &tool_name, serde_json::json!({}));
                    gate.deliver("call_x", ApprovalDecision::Approve).unwrap();

                    let decision = rx.await.unwrap();
                    assert_eq!(decision, ApprovalDecision::Approve);
                });
            }
        }
    }
}
