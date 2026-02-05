//! Multi-agent coordination for the agentic loop.
//!
//! Implements agent identity, tool locking, and resource quota management
//! from AGENTIC-LOOP-SPEC.md §7.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;

// ---------------------------------------------------------------------------
// Agent identity
// ---------------------------------------------------------------------------

/// Unique identifier for an agent within a coordination context.
pub type AgentId = String;

/// Role of an agent in a multi-agent system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentRole {
    /// Main agent handling the request.
    Primary,
    /// Called in for specific expertise.
    Specialist,
    /// Reviewing another agent's work.
    Reviewer,
    /// Orchestrating other agents.
    Coordinator,
}

/// Identity of an agent participating in coordination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentIdentity {
    /// Unique agent identifier.
    pub id: AgentId,
    /// Chosen name, if any.
    pub name: Option<String>,
    /// Role in the multi-agent system.
    pub role: AgentRole,
    /// Capabilities this agent offers.
    pub capabilities: Vec<String>,
    /// Current task summary.
    pub current_task: Option<String>,
}

impl AgentIdentity {
    /// Creates a new agent identity.
    pub fn new(id: impl Into<String>, role: AgentRole) -> Self {
        Self {
            id: id.into(),
            name: None,
            role,
            capabilities: Vec::new(),
            current_task: None,
        }
    }

    /// Sets the agent's chosen name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Adds a capability.
    pub fn with_capability(mut self, cap: impl Into<String>) -> Self {
        self.capabilities.push(cap.into());
        self
    }

    /// Sets the current task.
    pub fn with_task(mut self, task: impl Into<String>) -> Self {
        self.current_task = Some(task.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Tool locking
// ---------------------------------------------------------------------------

/// A lock on a tool+resource combination.
#[derive(Debug, Clone)]
pub struct ToolLock {
    /// Tool name.
    pub tool: String,
    /// Resource being locked (e.g., file path).
    pub resource: String,
    /// Agent holding the lock.
    pub held_by: AgentId,
    /// When the lock was acquired.
    pub acquired_at: Instant,
    /// Maximum lock duration before auto-release.
    pub max_duration: Duration,
}

/// Manages tool locks for multi-agent coordination.
///
/// Prevents conflicting tool executions (e.g., two agents writing to the
/// same file simultaneously).
pub struct ToolLockManager {
    locks: RwLock<HashMap<String, ToolLock>>,
    notify: Notify,
}

impl ToolLockManager {
    /// Creates a new tool lock manager.
    pub fn new() -> Self {
        Self {
            locks: RwLock::new(HashMap::new()),
            notify: Notify::new(),
        }
    }

    /// Attempts to acquire a lock on a tool+resource.
    ///
    /// Returns `true` if the lock was acquired, `false` if already held
    /// by another agent.
    pub fn try_acquire(
        &self,
        agent_id: &str,
        tool: &str,
        resource: &str,
        max_duration: Duration,
    ) -> bool {
        let key = lock_key(tool, resource);
        let mut locks = self.locks.write();

        // Check for expired locks
        if let Some(existing) = locks.get(&key) {
            if existing.acquired_at.elapsed() > existing.max_duration {
                // Auto-release expired lock
                locks.remove(&key);
            } else if existing.held_by != agent_id {
                return false;
            } else {
                return true; // Already held by this agent
            }
        }

        locks.insert(
            key,
            ToolLock {
                tool: tool.to_string(),
                resource: resource.to_string(),
                held_by: agent_id.to_string(),
                acquired_at: Instant::now(),
                max_duration,
            },
        );
        true
    }

    /// Releases a lock held by an agent.
    pub fn release(&self, agent_id: &str, tool: &str, resource: &str) {
        let key = lock_key(tool, resource);
        let mut locks = self.locks.write();

        if let Some(lock) = locks.get(&key) {
            if lock.held_by == agent_id {
                locks.remove(&key);
                self.notify.notify_waiters();
            }
        }
    }

    /// Releases all locks held by an agent.
    pub fn release_all(&self, agent_id: &str) {
        let mut locks = self.locks.write();
        locks.retain(|_, lock| lock.held_by != agent_id);
        self.notify.notify_waiters();
    }

    /// Returns locks held by an agent.
    pub fn locks_for(&self, agent_id: &str) -> Vec<ToolLock> {
        let locks = self.locks.read();
        locks
            .values()
            .filter(|l| l.held_by == agent_id)
            .cloned()
            .collect()
    }

    /// Returns the total number of active locks.
    pub fn active_lock_count(&self) -> usize {
        let locks = self.locks.read();
        locks
            .values()
            .filter(|l| l.acquired_at.elapsed() <= l.max_duration)
            .count()
    }

    /// Waits for a lock to become available.
    pub async fn wait_for_lock(&self) {
        self.notify.notified().await;
    }
}

impl Default for ToolLockManager {
    fn default() -> Self {
        Self::new()
    }
}

fn lock_key(tool: &str, resource: &str) -> String {
    format!("{tool}:{resource}")
}

// ---------------------------------------------------------------------------
// Resource quotas
// ---------------------------------------------------------------------------

/// Shared resource quota for multi-agent coordination.
///
/// When multiple agents share a resource pool, quotas prevent any single
/// agent from monopolizing resources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceQuota {
    /// Total tool calls allowed across all agents.
    pub total_tool_calls: u32,
    /// Remaining tool calls.
    pub remaining_tool_calls: u32,
    /// Total token budget across all agents.
    pub total_tokens: u32,
    /// Remaining tokens.
    pub remaining_tokens: u32,
}

/// Manages shared resource quotas.
pub struct ResourceQuotaManager {
    quota: RwLock<ResourceQuota>,
    per_agent: RwLock<HashMap<AgentId, AgentUsage>>,
}

/// Per-agent usage tracking.
#[derive(Debug, Clone, Default)]
struct AgentUsage {
    tool_calls: u32,
    tokens: u32,
}

impl ResourceQuotaManager {
    /// Creates a new quota manager with the given limits.
    pub fn new(total_tool_calls: u32, total_tokens: u32) -> Self {
        Self {
            quota: RwLock::new(ResourceQuota {
                total_tool_calls,
                remaining_tool_calls: total_tool_calls,
                total_tokens,
                remaining_tokens: total_tokens,
            }),
            per_agent: RwLock::new(HashMap::new()),
        }
    }

    /// Attempts to consume tool calls from the shared quota.
    ///
    /// Returns `true` if the quota allows it.
    pub fn try_consume_tool_calls(&self, agent_id: &str, count: u32) -> bool {
        let mut quota = self.quota.write();
        if quota.remaining_tool_calls >= count {
            quota.remaining_tool_calls -= count;

            let mut agents = self.per_agent.write();
            agents
                .entry(agent_id.to_string())
                .or_default()
                .tool_calls += count;
            true
        } else {
            false
        }
    }

    /// Attempts to consume tokens from the shared quota.
    ///
    /// Returns `true` if the quota allows it.
    pub fn try_consume_tokens(&self, agent_id: &str, count: u32) -> bool {
        let mut quota = self.quota.write();
        if quota.remaining_tokens >= count {
            quota.remaining_tokens -= count;

            let mut agents = self.per_agent.write();
            agents
                .entry(agent_id.to_string())
                .or_default()
                .tokens += count;
            true
        } else {
            false
        }
    }

    /// Returns the current quota state.
    pub fn current(&self) -> ResourceQuota {
        self.quota.read().clone()
    }

    /// Returns usage for a specific agent.
    pub fn agent_usage(&self, agent_id: &str) -> (u32, u32) {
        let agents = self.per_agent.read();
        agents
            .get(agent_id)
            .map(|u| (u.tool_calls, u.tokens))
            .unwrap_or((0, 0))
    }
}

// ---------------------------------------------------------------------------
// Agent coordinator
// ---------------------------------------------------------------------------

/// Coordinator for multi-agent interactions.
///
/// Combines identity management, tool locking, and resource quotas into
/// a single interface that the loop executor can use.
pub struct AgentCoordinator {
    /// Registered agents.
    agents: RwLock<HashMap<AgentId, AgentIdentity>>,
    /// Tool lock manager.
    pub locks: Arc<ToolLockManager>,
    /// Resource quota manager (optional — only present in multi-agent mode).
    pub quotas: Option<Arc<ResourceQuotaManager>>,
}

impl AgentCoordinator {
    /// Creates a new coordinator.
    pub fn new() -> Self {
        Self {
            agents: RwLock::new(HashMap::new()),
            locks: Arc::new(ToolLockManager::new()),
            quotas: None,
        }
    }

    /// Creates a coordinator with resource quotas.
    pub fn with_quotas(total_tool_calls: u32, total_tokens: u32) -> Self {
        Self {
            agents: RwLock::new(HashMap::new()),
            locks: Arc::new(ToolLockManager::new()),
            quotas: Some(Arc::new(ResourceQuotaManager::new(
                total_tool_calls,
                total_tokens,
            ))),
        }
    }

    /// Registers an agent.
    pub fn register_agent(&self, identity: AgentIdentity) {
        self.agents.write().insert(identity.id.clone(), identity);
    }

    /// Removes an agent and releases all its locks.
    pub fn unregister_agent(&self, agent_id: &str) {
        self.agents.write().remove(agent_id);
        self.locks.release_all(agent_id);
    }

    /// Returns registered agents.
    pub fn agents(&self) -> Vec<AgentIdentity> {
        self.agents.read().values().cloned().collect()
    }

    /// Returns an agent by ID.
    pub fn get_agent(&self, agent_id: &str) -> Option<AgentIdentity> {
        self.agents.read().get(agent_id).cloned()
    }
}

impl Default for AgentCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_identity_builder() {
        let identity = AgentIdentity::new("agent-1", AgentRole::Primary)
            .with_name("Atlas")
            .with_capability("code_review")
            .with_task("Review PR #42");

        assert_eq!(identity.id, "agent-1");
        assert_eq!(identity.name, Some("Atlas".to_string()));
        assert_eq!(identity.role, AgentRole::Primary);
        assert_eq!(identity.capabilities, vec!["code_review"]);
        assert_eq!(identity.current_task, Some("Review PR #42".to_string()));
    }

    #[test]
    fn test_tool_lock_acquire_and_release() {
        let mgr = ToolLockManager::new();

        // Agent 1 acquires a lock
        assert!(mgr.try_acquire("agent-1", "write_file", "/tmp/a.rs", Duration::from_secs(30)));

        // Agent 2 cannot acquire the same lock
        assert!(!mgr.try_acquire("agent-2", "write_file", "/tmp/a.rs", Duration::from_secs(30)));

        // Agent 1 can re-acquire (idempotent)
        assert!(mgr.try_acquire("agent-1", "write_file", "/tmp/a.rs", Duration::from_secs(30)));

        // Different resource is fine
        assert!(mgr.try_acquire("agent-2", "write_file", "/tmp/b.rs", Duration::from_secs(30)));

        assert_eq!(mgr.active_lock_count(), 2);

        // Release
        mgr.release("agent-1", "write_file", "/tmp/a.rs");
        assert!(mgr.try_acquire("agent-2", "write_file", "/tmp/a.rs", Duration::from_secs(30)));
    }

    #[test]
    fn test_tool_lock_release_all() {
        let mgr = ToolLockManager::new();
        mgr.try_acquire("agent-1", "write_file", "/tmp/a.rs", Duration::from_secs(30));
        mgr.try_acquire("agent-1", "write_file", "/tmp/b.rs", Duration::from_secs(30));
        mgr.try_acquire("agent-1", "bash", "git push", Duration::from_secs(30));

        assert_eq!(mgr.locks_for("agent-1").len(), 3);

        mgr.release_all("agent-1");
        assert_eq!(mgr.locks_for("agent-1").len(), 0);
        assert_eq!(mgr.active_lock_count(), 0);
    }

    #[test]
    fn test_resource_quota_consumption() {
        let mgr = ResourceQuotaManager::new(10, 1000);

        assert!(mgr.try_consume_tool_calls("agent-1", 3));
        assert!(mgr.try_consume_tool_calls("agent-2", 5));
        assert!(mgr.try_consume_tool_calls("agent-1", 2));
        assert!(!mgr.try_consume_tool_calls("agent-2", 1)); // Only 0 remaining

        let quota = mgr.current();
        assert_eq!(quota.remaining_tool_calls, 0);

        let (calls, _tokens) = mgr.agent_usage("agent-1");
        assert_eq!(calls, 5);
    }

    #[test]
    fn test_resource_quota_tokens() {
        let mgr = ResourceQuotaManager::new(100, 500);

        assert!(mgr.try_consume_tokens("agent-1", 200));
        assert!(mgr.try_consume_tokens("agent-2", 200));
        assert!(mgr.try_consume_tokens("agent-1", 100));
        assert!(!mgr.try_consume_tokens("agent-2", 1)); // 0 remaining

        let quota = mgr.current();
        assert_eq!(quota.remaining_tokens, 0);
    }

    #[test]
    fn test_coordinator_register_unregister() {
        let coord = AgentCoordinator::new();

        coord.register_agent(
            AgentIdentity::new("agent-1", AgentRole::Primary).with_name("Atlas"),
        );
        coord.register_agent(
            AgentIdentity::new("agent-2", AgentRole::Specialist),
        );

        assert_eq!(coord.agents().len(), 2);
        assert!(coord.get_agent("agent-1").is_some());

        coord.unregister_agent("agent-1");
        assert_eq!(coord.agents().len(), 1);
        assert!(coord.get_agent("agent-1").is_none());
    }

    #[test]
    fn test_coordinator_with_quotas() {
        let coord = AgentCoordinator::with_quotas(50, 10000);
        assert!(coord.quotas.is_some());

        let quotas = coord.quotas.as_ref().expect("quotas");
        assert!(quotas.try_consume_tool_calls("agent-1", 10));
        assert_eq!(quotas.current().remaining_tool_calls, 40);
    }
}
