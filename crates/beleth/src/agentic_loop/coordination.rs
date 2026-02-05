//! Multi-agent coordination for the agentic loop.
//!
//! Implements agent identity, tool locking, and resource quota management
//! from AGENTIC-LOOP-SPEC.md §7.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, oneshot, Notify};

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
// Coordination primitive types (AGENTIC-LOOP-SPEC §7.2)
// ---------------------------------------------------------------------------

/// Errors from coordination primitive operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CoordinationError {
    /// The specified agent is not registered.
    #[error("agent not found: {0}")]
    AgentNotFound(AgentId),

    /// The assistance request ID was not found.
    #[error("request not found: {0}")]
    RequestNotFound(String),

    /// The agent has already yielded and cannot perform further actions.
    #[error("agent already yielded: {0}")]
    AlreadyYielded(AgentId),

    /// The specified target agent for yield is not registered.
    #[error("yield target not found: {0}")]
    YieldTargetNotFound(AgentId),
}

/// Whether an assistance request blocks the requesting agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssistancePriority {
    /// Agent is blocked until assistance arrives.
    Blocking,
    /// Agent can continue working while waiting.
    Background,
}

/// What kind of help an agent needs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistanceRequest {
    /// Free-text description of what help is needed.
    pub description: String,
    /// Capabilities the helper should have.
    pub required_capabilities: Vec<String>,
    /// The requesting agent's partial progress so far.
    pub partial_progress: Option<String>,
    /// Whether the agent is blocked or can continue while waiting.
    pub priority: AssistancePriority,
}

/// Response to an assistance request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AssistanceResponse {
    /// Another agent has been assigned to help.
    Assigned {
        /// ID of the helping agent.
        helper_id: AgentId,
        /// The helper's chosen name, if any.
        helper_name: Option<String>,
        /// Any context the supervisor wants to relay.
        message: Option<String>,
    },
    /// No suitable agent is available.
    Unavailable {
        /// Why no agent could be found.
        reason: String,
        /// Supervisor's suggestion for what the agent should do instead.
        suggestion: Option<String>,
    },
    /// The request timed out.
    TimedOut,
}

/// Pending assistance request stored in the coordinator.
pub struct PendingAssistance {
    /// Unique request ID.
    pub request_id: String,
    /// Who made the request.
    pub from: AgentId,
    /// The request details.
    pub request: AssistanceRequest,
    /// When the request was created.
    pub requested_at: Instant,
    /// Channel to deliver the response. Consumed once.
    pub respond: oneshot::Sender<AssistanceResponse>,
}

/// Context provided when an agent yields its task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YieldContext {
    /// Why the agent is yielding.
    pub reason: String,
    /// Partial progress achieved.
    pub partial_progress: Option<String>,
    /// Capabilities the replacement agent should have.
    pub suggested_expertise: Vec<String>,
    /// Structured data to hand off.
    pub handoff_data: Option<serde_json::Value>,
}

/// Result of a yield operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum YieldResult {
    /// Yield accepted — the agent should transition to Yielded state.
    Accepted,
    /// No alternative agent available.
    NoAlternative {
        /// Why no alternative could be found.
        reason: String,
    },
}

/// Pending yield stored in the coordinator.
pub struct PendingYield {
    /// Who yielded.
    pub from: AgentId,
    /// Optional target agent.
    pub to: Option<AgentId>,
    /// Yield context with partial progress and handoff data.
    pub context: YieldContext,
    /// When the yield was recorded.
    pub yielded_at: Instant,
}

/// A piece of knowledge an agent wants to share with others.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Discovery {
    /// What was discovered.
    pub content: String,
    /// Category for filtering.
    pub category: String,
    /// Tags for capability matching.
    pub tags: Vec<String>,
    /// Optional structured data.
    pub data: Option<serde_json::Value>,
}

/// Internal storage of a shared discovery.
struct StoredDiscovery {
    from: AgentId,
    discovery: Discovery,
    #[allow(dead_code)]
    shared_at: Instant,
}

/// Accumulated shared context returned to a requesting agent.
#[derive(Debug, Clone)]
pub struct SharedContext {
    /// Discoveries from other agents, filtered by visibility policy.
    pub discoveries: Vec<(AgentId, Discovery)>,
    /// Number of discoveries filtered out by visibility policy.
    pub filtered_count: usize,
}

/// Controls what agents can see of each other's discoveries.
#[derive(Debug, Clone, Default)]
pub enum VisibilityPolicy {
    /// Each agent sees all discoveries from all other agents.
    #[default]
    Open,
    /// Only discoveries tagged with matching capabilities.
    CapabilityFiltered,
    /// Supervisor-controlled allow list.
    Explicit {
        /// Maps each agent to the agents whose discoveries it can see.
        allow_list: HashMap<AgentId, Vec<AgentId>>,
    },
    /// No discoveries shared.
    Isolated,
}

/// Events emitted by the coordinator on its broadcast channel.
#[derive(Debug, Clone)]
pub enum CoordinationEvent {
    /// An agent requested assistance.
    AssistanceRequested {
        /// Unique request ID.
        request_id: String,
        /// Who made the request.
        from: AgentId,
        /// Capabilities needed.
        capabilities_needed: Vec<String>,
        /// Priority level.
        priority: AssistancePriority,
    },
    /// An agent yielded its task.
    AgentYielded {
        /// Who yielded.
        from: AgentId,
        /// Optional target agent.
        to: Option<AgentId>,
        /// Why.
        reason: String,
        /// Suggested capabilities for replacement.
        suggested_expertise: Vec<String>,
    },
    /// An agent shared a discovery.
    DiscoveryShared {
        /// Who shared.
        from: AgentId,
        /// Discovery category.
        category: String,
        /// Discovery tags.
        tags: Vec<String>,
    },
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

/// Maximum number of discoveries stored per coordinator instance.
const MAX_DISCOVERIES: usize = 1000;

/// Coordinator for multi-agent interactions.
///
/// Combines identity management, tool locking, resource quotas, and
/// coordination primitives (assistance, yield, discovery sharing) into
/// a single interface that the loop executor and supervisor can use.
pub struct AgentCoordinator {
    /// Registered agents.
    agents: RwLock<HashMap<AgentId, AgentIdentity>>,
    /// Tool lock manager.
    pub locks: Arc<ToolLockManager>,
    /// Resource quota manager (optional — only present in multi-agent mode).
    pub quotas: Option<Arc<ResourceQuotaManager>>,

    // Coordination primitives (AGENTIC-LOOP-SPEC §7.2)

    /// Pending assistance requests (`request_id` → pending).
    pending_requests: RwLock<HashMap<String, PendingAssistance>>,
    /// Pending yield queue.
    pending_yields: RwLock<Vec<PendingYield>>,
    /// Agents that have already yielded (at-most-once enforcement).
    yielded_agents: RwLock<HashSet<AgentId>>,
    /// Discovery store (append-only, bounded at `MAX_DISCOVERIES`).
    discoveries: RwLock<Vec<StoredDiscovery>>,
    /// Visibility policy for shared context.
    visibility_policy: RwLock<VisibilityPolicy>,
    /// Event broadcast channel sender.
    event_tx: broadcast::Sender<CoordinationEvent>,
}

impl AgentCoordinator {
    /// Creates a new coordinator.
    pub fn new() -> Self {
        let (event_tx, _) = broadcast::channel(256);
        Self {
            agents: RwLock::new(HashMap::new()),
            locks: Arc::new(ToolLockManager::new()),
            quotas: None,
            pending_requests: RwLock::new(HashMap::new()),
            pending_yields: RwLock::new(Vec::new()),
            yielded_agents: RwLock::new(HashSet::new()),
            discoveries: RwLock::new(Vec::new()),
            visibility_policy: RwLock::new(VisibilityPolicy::default()),
            event_tx,
        }
    }

    /// Creates a coordinator with resource quotas.
    pub fn with_quotas(total_tool_calls: u32, total_tokens: u32) -> Self {
        let (event_tx, _) = broadcast::channel(256);
        Self {
            agents: RwLock::new(HashMap::new()),
            locks: Arc::new(ToolLockManager::new()),
            quotas: Some(Arc::new(ResourceQuotaManager::new(
                total_tool_calls,
                total_tokens,
            ))),
            pending_requests: RwLock::new(HashMap::new()),
            pending_yields: RwLock::new(Vec::new()),
            yielded_agents: RwLock::new(HashSet::new()),
            discoveries: RwLock::new(Vec::new()),
            visibility_policy: RwLock::new(VisibilityPolicy::default()),
            event_tx,
        }
    }

    /// Registers an agent.
    pub fn register_agent(&self, identity: AgentIdentity) {
        self.agents.write().insert(identity.id.clone(), identity);
    }

    /// Removes an agent and releases all its locks.
    ///
    /// Also clears yielded state and cancels pending assistance requests
    /// from this agent. Discoveries are preserved.
    pub fn unregister_agent(&self, agent_id: &str) {
        self.agents.write().remove(agent_id);
        self.locks.release_all(agent_id);
        self.yielded_agents.write().remove(agent_id);
        // Cancel pending assistance requests from this agent
        self.pending_requests
            .write()
            .retain(|_, req| req.from != agent_id);
    }

    /// Returns registered agents.
    pub fn agents(&self) -> Vec<AgentIdentity> {
        self.agents.read().values().cloned().collect()
    }

    /// Returns an agent by ID.
    pub fn get_agent(&self, agent_id: &str) -> Option<AgentIdentity> {
        self.agents.read().get(agent_id).cloned()
    }

    // -----------------------------------------------------------------------
    // Coordination primitives — §7.2.2 request_assistance
    // -----------------------------------------------------------------------

    /// Requests assistance from another agent.
    ///
    /// Creates a pending request and returns a receiver for the response.
    /// The supervisor consumes pending requests via `take_pending_requests()`
    /// and delivers responses via `deliver_assistance()`.
    ///
    /// # Errors
    ///
    /// Returns [`CoordinationError::AgentNotFound`] if `from` is not registered.
    pub fn request_assistance(
        &self,
        from: &AgentId,
        need: AssistanceRequest,
    ) -> Result<oneshot::Receiver<AssistanceResponse>, CoordinationError> {
        // Validate agent is registered
        if !self.agents.read().contains_key(from) {
            return Err(CoordinationError::AgentNotFound(from.clone()));
        }

        let request_id = format!("assist_{}_{}", from, uuid::Uuid::new_v4().simple());
        let (tx, rx) = oneshot::channel();

        self.pending_requests.write().insert(
            request_id.clone(),
            PendingAssistance {
                request_id: request_id.clone(),
                from: from.clone(),
                request: need.clone(),
                requested_at: Instant::now(),
                respond: tx,
            },
        );

        // Emit event (ignore send errors — no subscribers is fine)
        let _ = self.event_tx.send(CoordinationEvent::AssistanceRequested {
            request_id,
            from: from.clone(),
            capabilities_needed: need.required_capabilities,
            priority: need.priority,
        });

        Ok(rx)
    }

    /// Drains all pending assistance requests, filtering out stale ones.
    ///
    /// Stale requests (where the receiver has been dropped) are silently discarded.
    pub fn take_pending_requests(&self) -> Vec<PendingAssistance> {
        let mut map = self.pending_requests.write();
        let drained: HashMap<_, _> = std::mem::take(&mut *map);
        drained
            .into_values()
            .filter(|p| !p.respond.is_closed())
            .collect()
    }

    /// Delivers a response to a pending assistance request.
    ///
    /// # Errors
    ///
    /// Returns [`CoordinationError::RequestNotFound`] if the request ID is not
    /// in the pending map (already taken or never existed).
    pub fn deliver_assistance(
        &self,
        request_id: &str,
        response: AssistanceResponse,
    ) -> Result<(), CoordinationError> {
        let pending = self
            .pending_requests
            .write()
            .remove(request_id)
            .ok_or_else(|| CoordinationError::RequestNotFound(request_id.to_string()))?;

        // Send response through oneshot. If receiver dropped, that's fine.
        let _ = pending.respond.send(response);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Coordination primitives — §7.2.3 yield_to
    // -----------------------------------------------------------------------

    /// Yields the calling agent's task.
    ///
    /// Returns `Accepted` in multi-agent mode, or `NoAlternative` if
    /// no other agents are registered and no event subscribers exist.
    ///
    /// # Errors
    ///
    /// - [`CoordinationError::AgentNotFound`] if `from` is not registered.
    /// - [`CoordinationError::AlreadyYielded`] if the agent has already yielded.
    /// - [`CoordinationError::YieldTargetNotFound`] if `to` is specified but not registered.
    pub fn yield_to(
        &self,
        from: &AgentId,
        to: Option<AgentId>,
        context: YieldContext,
    ) -> Result<YieldResult, CoordinationError> {
        // Validate agent is registered
        if !self.agents.read().contains_key(from) {
            return Err(CoordinationError::AgentNotFound(from.clone()));
        }

        // Check not already yielded
        if self.yielded_agents.read().contains(from) {
            return Err(CoordinationError::AlreadyYielded(from.clone()));
        }

        // Validate target if specified
        if let Some(ref target) = to {
            if !self.agents.read().contains_key(target) {
                return Err(CoordinationError::YieldTargetNotFound(target.clone()));
            }
        }

        // Check if there's anyone to yield to
        let agents = self.agents.read();
        let other_agents_exist = agents.len() > 1
            || agents.keys().any(|id| id != from);
        let has_subscribers = self.event_tx.receiver_count() > 0;

        if !other_agents_exist && !has_subscribers {
            return Ok(YieldResult::NoAlternative {
                reason: "no other agents registered and no supervisor listening".to_string(),
            });
        }
        drop(agents);

        // Record the yield
        self.yielded_agents.write().insert(from.clone());
        self.pending_yields.write().push(PendingYield {
            from: from.clone(),
            to: to.clone(),
            context: context.clone(),
            yielded_at: Instant::now(),
        });

        // Emit event
        let _ = self.event_tx.send(CoordinationEvent::AgentYielded {
            from: from.clone(),
            to,
            reason: context.reason,
            suggested_expertise: context.suggested_expertise,
        });

        Ok(YieldResult::Accepted)
    }

    /// Drains all pending yields.
    pub fn take_pending_yields(&self) -> Vec<PendingYield> {
        let mut yields = self.pending_yields.write();
        std::mem::take(&mut *yields)
    }

    // -----------------------------------------------------------------------
    // Coordination primitives — §7.2.4 share_discovery / get_shared_context
    // -----------------------------------------------------------------------

    /// Shares a discovery with other agents.
    ///
    /// # Errors
    ///
    /// Returns [`CoordinationError::AgentNotFound`] if `from` is not registered.
    pub fn share_discovery(
        &self,
        from: &AgentId,
        discovery: Discovery,
    ) -> Result<(), CoordinationError> {
        if !self.agents.read().contains_key(from) {
            return Err(CoordinationError::AgentNotFound(from.clone()));
        }

        let category = discovery.category.clone();
        let tags = discovery.tags.clone();

        let mut store = self.discoveries.write();

        // Enforce bound: evict oldest if at capacity
        if store.len() >= MAX_DISCOVERIES {
            store.remove(0);
        }

        store.push(StoredDiscovery {
            from: from.clone(),
            discovery,
            shared_at: Instant::now(),
        });

        // Emit event
        let _ = self.event_tx.send(CoordinationEvent::DiscoveryShared {
            from: from.clone(),
            category,
            tags,
        });

        Ok(())
    }

    /// Returns shared context visible to the given agent.
    ///
    /// Returns empty context if the agent is not registered (non-fatal).
    pub fn get_shared_context(&self, for_agent: &AgentId) -> SharedContext {
        let store = self.discoveries.read();
        let policy = self.visibility_policy.read();
        let agents = self.agents.read();

        // Unregistered agents get empty context (non-fatal)
        let Some(agent_identity) = agents.get(for_agent) else {
            return SharedContext {
                discoveries: Vec::new(),
                filtered_count: 0,
            };
        };

        let agent_capabilities: Vec<String> = agent_identity.capabilities.clone();

        let mut discoveries = Vec::new();
        let mut filtered_count = 0;

        for stored in store.iter() {
            // Never include agent's own discoveries
            if stored.from == *for_agent {
                continue;
            }

            let visible = match &*policy {
                VisibilityPolicy::Open => true,
                VisibilityPolicy::Isolated => false,
                VisibilityPolicy::CapabilityFiltered => {
                    // Visible if any discovery tag matches any agent capability
                    stored
                        .discovery
                        .tags
                        .iter()
                        .any(|tag| agent_capabilities.contains(tag))
                }
                VisibilityPolicy::Explicit { allow_list } => allow_list
                    .get(for_agent)
                    .is_some_and(|allowed| allowed.contains(&stored.from)),
            };

            if visible {
                discoveries.push((stored.from.clone(), stored.discovery.clone()));
            } else {
                filtered_count += 1;
            }
        }

        SharedContext {
            discoveries,
            filtered_count,
        }
    }

    /// Sets the visibility policy for shared context.
    pub fn set_visibility_policy(&self, policy: VisibilityPolicy) {
        *self.visibility_policy.write() = policy;
    }

    // -----------------------------------------------------------------------
    // Coordination primitives — §7.2.5 Events
    // -----------------------------------------------------------------------

    /// Subscribes to coordination events.
    pub fn subscribe_events(&self) -> broadcast::Receiver<CoordinationEvent> {
        self.event_tx.subscribe()
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
    use proptest::prelude::*;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    fn blocking_assistance_request(desc: &str) -> AssistanceRequest {
        AssistanceRequest {
            description: desc.to_string(),
            required_capabilities: vec![],
            partial_progress: None,
            priority: AssistancePriority::Blocking,
        }
    }

    fn make_yield_context(reason: &str) -> YieldContext {
        YieldContext {
            reason: reason.to_string(),
            partial_progress: None,
            suggested_expertise: vec![],
            handoff_data: None,
        }
    }

    fn make_discovery(content: &str, category: &str) -> Discovery {
        Discovery {
            content: content.to_string(),
            category: category.to_string(),
            tags: vec![],
            data: None,
        }
    }

    // -----------------------------------------------------------------------
    // Proptest strategies
    // -----------------------------------------------------------------------

    fn agent_id_strategy() -> impl Strategy<Value = AgentId> {
        "[a-z][a-z0-9_]{2,20}".prop_map(|s| s)
    }

    fn assistance_priority_strategy() -> impl Strategy<Value = AssistancePriority> {
        prop_oneof![
            Just(AssistancePriority::Blocking),
            Just(AssistancePriority::Background),
        ]
    }

    fn assistance_request_strategy() -> impl Strategy<Value = AssistanceRequest> {
        (
            "[a-z ]{5,50}",
            prop::collection::vec("[a-z_]{3,15}", 0..3),
            proptest::option::of("[a-z ]{5,50}"),
            assistance_priority_strategy(),
        )
            .prop_map(|(desc, caps, progress, priority)| AssistanceRequest {
                description: desc,
                required_capabilities: caps,
                partial_progress: progress,
                priority,
            })
    }

    fn assistance_response_strategy() -> impl Strategy<Value = AssistanceResponse> {
        prop_oneof![
            ("[a-z_]{3,15}", proptest::option::of("[A-Z][a-z]{2,10}"), proptest::option::of("[a-z ]{5,30}"))
                .prop_map(|(id, name, msg)| AssistanceResponse::Assigned {
                    helper_id: id,
                    helper_name: name,
                    message: msg,
                }),
            ("[a-z ]{5,30}", proptest::option::of("[a-z ]{5,30}"))
                .prop_map(|(reason, suggestion)| AssistanceResponse::Unavailable {
                    reason,
                    suggestion,
                }),
            Just(AssistanceResponse::TimedOut),
        ]
    }

    fn yield_context_strategy() -> impl Strategy<Value = YieldContext> {
        (
            "[a-z ]{5,50}",
            proptest::option::of("[a-z ]{5,50}"),
            prop::collection::vec("[a-z_]{3,15}", 0..3),
        )
            .prop_map(|(reason, progress, expertise)| YieldContext {
                reason,
                partial_progress: progress,
                suggested_expertise: expertise,
                handoff_data: None,
            })
    }

    // -----------------------------------------------------------------------
    // Original tests
    // -----------------------------------------------------------------------

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

    // ===================================================================
    // §7.3 Assistance Request Protocol
    // ===================================================================

    proptest! {
        /// Property: Unregistered agents cannot request assistance.
        #[test]
        fn test_unregistered_agent_cannot_request(
            agent_id in agent_id_strategy(),
            request in assistance_request_strategy(),
        ) {
            let coordinator = AgentCoordinator::new();
            // Do NOT register the agent

            let result = coordinator.request_assistance(&agent_id, request);
            prop_assert!(matches!(result, Err(CoordinationError::AgentNotFound(_))));
        }

        /// Property: Every request gets a unique request_id.
        #[test]
        fn test_request_ids_unique(
            agent_count in 1usize..=5,
            requests_per in 1u8..=3,
        ) {
            let coordinator = AgentCoordinator::new();
            let agents: Vec<_> = (0..agent_count)
                .map(|i| AgentIdentity::new(format!("agent_{i}"), AgentRole::Primary))
                .collect();
            for agent in &agents {
                coordinator.register_agent(agent.clone());
            }

            for agent in &agents {
                for _ in 0..requests_per {
                    let _rx = coordinator.request_assistance(
                        &agent.id,
                        blocking_assistance_request("help"),
                    ).expect("registered agent should succeed");
                }
            }

            let pending = coordinator.take_pending_requests();
            let mut ids = HashSet::new();
            for req in &pending {
                prop_assert!(
                    ids.insert(req.request_id.clone()),
                    "Duplicate request_id: {}", req.request_id,
                );
            }
        }

        /// Property: Delivered responses arrive at the correct receiver.
        /// Uses deliver_assistance() without take (direct delivery pattern).
        #[test]
        fn test_response_delivered_to_correct_receiver(
            response in assistance_response_strategy(),
        ) {
            let coordinator = AgentCoordinator::new();
            coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));

            let mut rx = coordinator.request_assistance(
                &"agent_1".to_string(),
                blocking_assistance_request("need help with database"),
            ).expect("should succeed");

            // Get request_id without draining the map
            let request_id = {
                let map = coordinator.pending_requests.read();
                map.keys().next().expect("should have one pending").clone()
            };

            coordinator.deliver_assistance(&request_id, response.clone())
                .expect("deliver should succeed");

            let received = rx.try_recv().expect("should have response");
            prop_assert_eq!(received, response);
        }

        /// Property: Responses taken via PendingAssistance.respond also arrive correctly.
        /// This is the supervisor poll pattern: take → send directly.
        #[test]
        fn test_response_via_take_and_send(
            response in assistance_response_strategy(),
        ) {
            let coordinator = AgentCoordinator::new();
            coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));

            let mut rx = coordinator.request_assistance(
                &"agent_1".to_string(),
                blocking_assistance_request("need help"),
            ).expect("should succeed");

            let mut pending = coordinator.take_pending_requests();
            prop_assert_eq!(pending.len(), 1);

            // Supervisor sends response through the oneshot directly
            let assistance = pending.remove(0);
            let _ = assistance.respond.send(response.clone());

            let received = rx.try_recv().expect("should have response");
            prop_assert_eq!(received, response);
        }
    }

    #[tokio::test]
    async fn test_assistance_timeout_returns_none() {
        let coordinator = AgentCoordinator::new();
        coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));

        let rx = coordinator.request_assistance(
            &"agent_1".to_string(),
            blocking_assistance_request("need help"),
        )
        .expect("should succeed");

        // Nobody delivers a response — timeout
        let result = tokio::time::timeout(Duration::from_millis(50), rx).await;
        assert!(result.is_err()); // Elapsed
    }

    #[tokio::test]
    async fn test_stale_requests_cleaned_on_take() {
        let coordinator = AgentCoordinator::new();
        coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));

        let rx = coordinator.request_assistance(
            &"agent_1".to_string(),
            blocking_assistance_request("need help"),
        )
        .expect("should succeed");

        // Drop the receiver (simulates caller timeout/cancellation)
        drop(rx);

        // take_pending_requests should filter out stale entries
        let pending = coordinator.take_pending_requests();
        assert_eq!(pending.len(), 0); // Sender::is_closed() detected
    }

    #[tokio::test]
    async fn test_deliver_to_consumed_request_returns_not_found() {
        // take_pending_requests drains the map, so deliver_assistance
        // after take will return RequestNotFound.
        let coordinator = AgentCoordinator::new();
        coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));

        let _rx = coordinator.request_assistance(
            &"agent_1".to_string(),
            blocking_assistance_request("need help"),
        )
        .expect("should succeed");

        let pending = coordinator.take_pending_requests();
        assert_eq!(pending.len(), 1);
        let req_id = pending[0].request_id.clone();

        // After take, the map is empty — deliver should fail
        assert!(matches!(
            coordinator.deliver_assistance(&req_id, AssistanceResponse::TimedOut),
            Err(CoordinationError::RequestNotFound(_))
        ));
    }

    #[tokio::test]
    async fn test_deliver_then_deliver_again_fails() {
        let coordinator = AgentCoordinator::new();
        coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));

        let _rx = coordinator.request_assistance(
            &"agent_1".to_string(),
            blocking_assistance_request("need help"),
        )
        .expect("should succeed");

        // Peek at pending requests to get the ID without draining
        let pending_ids: Vec<String> = coordinator
            .pending_requests
            .read()
            .keys()
            .cloned()
            .collect();
        assert_eq!(pending_ids.len(), 1);
        let req_id = &pending_ids[0];

        // First delivery succeeds
        assert!(coordinator
            .deliver_assistance(req_id, AssistanceResponse::TimedOut)
            .is_ok());

        // Second delivery fails (oneshot consumed, removed from map)
        assert!(matches!(
            coordinator.deliver_assistance(req_id, AssistanceResponse::TimedOut),
            Err(CoordinationError::RequestNotFound(_))
        ));
    }

    // ===================================================================
    // §7.4 Yield Protocol
    // ===================================================================

    proptest! {
        /// Property: Unregistered agents cannot yield.
        #[test]
        fn test_unregistered_agent_cannot_yield(
            agent_id in agent_id_strategy(),
            context in yield_context_strategy(),
        ) {
            let coordinator = AgentCoordinator::new();

            let result = coordinator.yield_to(&agent_id, None, context);
            prop_assert!(matches!(result, Err(CoordinationError::AgentNotFound(_))));
        }

        /// Property: An agent can yield at most once.
        #[test]
        fn test_agent_yields_at_most_once(
            context1 in yield_context_strategy(),
            context2 in yield_context_strategy(),
        ) {
            let coordinator = AgentCoordinator::new();
            coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));
            // Need at least one other agent for Accepted
            coordinator.register_agent(AgentIdentity::new("agent_2", AgentRole::Specialist));

            let first = coordinator.yield_to(&"agent_1".to_string(), None, context1);
            prop_assert!(matches!(first, Ok(YieldResult::Accepted)));

            let second = coordinator.yield_to(&"agent_1".to_string(), None, context2);
            prop_assert!(matches!(second, Err(CoordinationError::AlreadyYielded(_))));
        }

        /// Property: Yield with invalid target returns error.
        #[test]
        fn test_yield_to_nonexistent_target(
            target_id in agent_id_strategy(),
            context in yield_context_strategy(),
        ) {
            let coordinator = AgentCoordinator::new();
            coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));
            // target_id is NOT registered (very unlikely to collide with "agent_1")

            let result = coordinator.yield_to(
                &"agent_1".to_string(),
                Some(target_id),
                context,
            );
            // Either YieldTargetNotFound (target unregistered) or NoAlternative
            // (if target_id happens to be "agent_1" — only 1 agent, no subscribers).
            let is_expected = matches!(
                result,
                Err(CoordinationError::YieldTargetNotFound(_))
                | Ok(YieldResult::NoAlternative { .. })
            );
            prop_assert!(is_expected, "Expected YieldTargetNotFound or NoAlternative, got: {:?}", result);
        }
    }

    #[tokio::test]
    async fn test_yield_no_alternative_single_agent() {
        let coordinator = AgentCoordinator::new();
        coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));
        // No other agents, no event subscribers

        let result = coordinator
            .yield_to(
                &"agent_1".to_string(),
                None,
                YieldContext {
                    reason: "Need database expertise".into(),
                    partial_progress: Some("Found the schema file".into()),
                    suggested_expertise: vec!["database".into()],
                    handoff_data: None,
                },
            )
            .expect("should not error");

        assert!(matches!(result, YieldResult::NoAlternative { .. }));
    }

    #[tokio::test]
    async fn test_yield_accepted_with_supervisor_listening() {
        let coordinator = AgentCoordinator::new();
        coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));
        coordinator.register_agent(AgentIdentity::new("agent_2", AgentRole::Specialist));

        // Subscribe to events (simulates supervisor)
        let mut event_rx = coordinator.subscribe_events();

        let result = coordinator
            .yield_to(
                &"agent_1".to_string(),
                None,
                YieldContext {
                    reason: "Need database expertise".into(),
                    partial_progress: Some("Found the schema file".into()),
                    suggested_expertise: vec!["database".into()],
                    handoff_data: None,
                },
            )
            .expect("should not error");

        assert!(matches!(result, YieldResult::Accepted));

        // Supervisor receives notification
        let event = event_rx.try_recv().expect("should have event");
        match event {
            CoordinationEvent::AgentYielded {
                from,
                suggested_expertise,
                ..
            } => {
                assert_eq!(from, "agent_1");
                assert!(suggested_expertise.contains(&"database".to_string()));
            }
            other => panic!("Expected AgentYielded, got {other:?}"),
        }

        // Pending yields queue has the entry
        let pending = coordinator.take_pending_yields();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].from, "agent_1");
        assert_eq!(
            pending[0].context.partial_progress,
            Some("Found the schema file".into())
        );
    }

    #[tokio::test]
    async fn test_yield_to_specific_target() {
        let coordinator = AgentCoordinator::new();
        coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));
        coordinator.register_agent(AgentIdentity::new("agent_2", AgentRole::Specialist));

        let result = coordinator
            .yield_to(
                &"agent_1".to_string(),
                Some("agent_2".to_string()),
                make_yield_context("Agent 2 is better suited"),
            )
            .expect("should not error");

        assert!(matches!(result, YieldResult::Accepted));

        let pending = coordinator.take_pending_yields();
        assert_eq!(pending[0].to, Some("agent_2".to_string()));
    }

    // ===================================================================
    // §7.5 Discovery Store and Shared Context
    // ===================================================================

    proptest! {
        /// Property: Agents never see their own discoveries.
        #[test]
        fn test_agent_does_not_see_own_discoveries(
            discovery_count in 1usize..=10,
        ) {
            let coordinator = AgentCoordinator::new();
            coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));

            for i in 0..discovery_count {
                coordinator
                    .share_discovery(
                        &"agent_1".to_string(),
                        make_discovery(&format!("disc {i}"), "test"),
                    )
                    .expect("should succeed");
            }

            let context = coordinator.get_shared_context(&"agent_1".to_string());
            prop_assert!(context.discoveries.is_empty());
            prop_assert_eq!(context.filtered_count, 0);
        }

        /// Property: Agent sees all other agents' discoveries under Open policy.
        #[test]
        fn test_open_policy_sees_all_others(
            agent_count in 2usize..=5,
            discoveries_per in 1u8..=3,
        ) {
            let coordinator = AgentCoordinator::new();
            coordinator.set_visibility_policy(VisibilityPolicy::Open);

            let agents: Vec<_> = (0..agent_count)
                .map(|i| format!("agent_{i}"))
                .collect();

            for a in &agents {
                coordinator.register_agent(AgentIdentity::new(a, AgentRole::Primary));
                for j in 0..discoveries_per {
                    coordinator
                        .share_discovery(
                            a,
                            make_discovery(&format!("discovery {j} from {a}"), "test"),
                        )
                        .expect("should succeed");
                }
            }

            // Each agent sees all discoveries EXCEPT its own
            for a in &agents {
                let context = coordinator.get_shared_context(a);
                let expected = (agent_count - 1) * discoveries_per as usize;
                prop_assert_eq!(context.discoveries.len(), expected);

                // None should be from the requesting agent
                for (from, _) in &context.discoveries {
                    prop_assert_ne!(from, a);
                }
            }
        }

        /// Property: Discovery store bounded at 1000 entries.
        #[test]
        fn test_discovery_store_bounded(
            discovery_count in 990u16..=1100,
        ) {
            let coordinator = AgentCoordinator::new();
            coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));
            coordinator.register_agent(AgentIdentity::new("reader", AgentRole::Primary));

            for i in 0..discovery_count {
                coordinator
                    .share_discovery(
                        &"agent_1".to_string(),
                        make_discovery(&format!("discovery {i}"), "test"),
                    )
                    .expect("should succeed");
            }

            let context = coordinator.get_shared_context(&"reader".to_string());
            prop_assert!(context.discoveries.len() <= 1000);

            // If over 1000 were added, oldest should have been evicted
            if discovery_count > 1000 {
                let first_content = &context.discoveries[0].1.content;
                let expected_start = discovery_count - 1000;
                prop_assert!(
                    first_content.contains(&expected_start.to_string()),
                    "Expected first discovery to contain {expected_start}, got {first_content}"
                );
            }
        }

        /// Property: Discoveries ordered oldest-first.
        #[test]
        fn test_discoveries_ordered_oldest_first(
            count in 2usize..=20,
        ) {
            let coordinator = AgentCoordinator::new();
            coordinator.register_agent(AgentIdentity::new("writer", AgentRole::Primary));
            coordinator.register_agent(AgentIdentity::new("reader", AgentRole::Primary));

            for i in 0..count {
                coordinator
                    .share_discovery(
                        &"writer".to_string(),
                        make_discovery(&format!("{i}_content"), "test"),
                    )
                    .expect("should succeed");
            }

            let context = coordinator.get_shared_context(&"reader".to_string());
            for (idx, (_, discovery)) in context.discoveries.iter().enumerate() {
                prop_assert!(
                    discovery.content.starts_with(&format!("{idx}_")),
                    "Expected discovery at index {idx} to start with '{idx}_', got '{}'",
                    discovery.content,
                );
            }
        }
    }

    #[tokio::test]
    async fn test_isolated_policy_returns_empty() {
        let coordinator = AgentCoordinator::new();
        coordinator.set_visibility_policy(VisibilityPolicy::Isolated);
        coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));
        coordinator.register_agent(AgentIdentity::new("agent_2", AgentRole::Primary));

        coordinator
            .share_discovery(
                &"agent_1".to_string(),
                make_discovery("important finding", "test"),
            )
            .expect("should succeed");

        let context = coordinator.get_shared_context(&"agent_2".to_string());
        assert!(context.discoveries.is_empty());
    }

    #[tokio::test]
    async fn test_capability_filtered_policy() {
        let coordinator = AgentCoordinator::new();
        coordinator.set_visibility_policy(VisibilityPolicy::CapabilityFiltered);

        coordinator.register_agent(
            AgentIdentity::new("db_agent", AgentRole::Specialist)
                .with_capability("database"),
        );
        coordinator.register_agent(
            AgentIdentity::new("ui_agent", AgentRole::Specialist)
                .with_capability("frontend"),
        );
        coordinator.register_agent(
            AgentIdentity::new("reader", AgentRole::Primary)
                .with_capability("database"),
        );

        // db_agent shares a discovery tagged "database"
        coordinator
            .share_discovery(
                &"db_agent".to_string(),
                Discovery {
                    content: "Schema has 12 tables".into(),
                    category: "database".into(),
                    tags: vec!["database".into(), "schema".into()],
                    data: None,
                },
            )
            .expect("should succeed");

        // ui_agent shares a discovery tagged "frontend"
        coordinator
            .share_discovery(
                &"ui_agent".to_string(),
                Discovery {
                    content: "Theme uses CSS grid".into(),
                    category: "frontend".into(),
                    tags: vec!["frontend".into(), "css".into()],
                    data: None,
                },
            )
            .expect("should succeed");

        // Reader has "database" capability — sees db discovery but not ui
        let context = coordinator.get_shared_context(&"reader".to_string());
        assert_eq!(context.discoveries.len(), 1);
        assert!(context.discoveries[0].1.content.contains("Schema"));
        assert_eq!(context.filtered_count, 1); // ui discovery was filtered
    }

    #[tokio::test]
    async fn test_explicit_policy_allow_list() {
        let coordinator = AgentCoordinator::new();

        let mut allow_list = HashMap::new();
        allow_list.insert("reader".to_string(), vec!["agent_a".to_string()]);

        coordinator.set_visibility_policy(VisibilityPolicy::Explicit { allow_list });
        coordinator.register_agent(AgentIdentity::new("agent_a", AgentRole::Primary));
        coordinator.register_agent(AgentIdentity::new("agent_b", AgentRole::Primary));
        coordinator.register_agent(AgentIdentity::new("reader", AgentRole::Primary));

        coordinator
            .share_discovery(
                &"agent_a".to_string(),
                make_discovery("visible", "test"),
            )
            .expect("should succeed");

        coordinator
            .share_discovery(
                &"agent_b".to_string(),
                make_discovery("hidden", "test"),
            )
            .expect("should succeed");

        let context = coordinator.get_shared_context(&"reader".to_string());
        assert_eq!(context.discoveries.len(), 1);
        assert_eq!(context.discoveries[0].1.content, "visible");
        assert_eq!(context.filtered_count, 1);
    }

    #[tokio::test]
    async fn test_unregistered_agent_cannot_share() {
        let coordinator = AgentCoordinator::new();

        let result = coordinator.share_discovery(
            &"ghost".to_string(),
            make_discovery("test", "test"),
        );

        assert!(matches!(result, Err(CoordinationError::AgentNotFound(_))));
    }

    #[tokio::test]
    async fn test_unregistered_reader_gets_empty_context() {
        let coordinator = AgentCoordinator::new();
        coordinator.register_agent(AgentIdentity::new("writer", AgentRole::Primary));

        coordinator
            .share_discovery(
                &"writer".to_string(),
                make_discovery("test", "test"),
            )
            .expect("should succeed");

        // Unregistered reader gets empty (not error)
        let context = coordinator.get_shared_context(&"unknown".to_string());
        assert!(context.discoveries.is_empty());
    }

    // ===================================================================
    // §7.6 Coordination Events
    // ===================================================================

    #[tokio::test]
    async fn test_assistance_request_emits_event() {
        let coordinator = AgentCoordinator::new();
        coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));

        let mut event_rx = coordinator.subscribe_events();

        let _rx = coordinator
            .request_assistance(
                &"agent_1".to_string(),
                blocking_assistance_request("help needed"),
            )
            .expect("should succeed");

        let event = event_rx.try_recv().expect("should have event");
        assert!(matches!(
            event,
            CoordinationEvent::AssistanceRequested { .. }
        ));
        // No extra events
        assert!(event_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn test_yield_emits_event() {
        let coordinator = AgentCoordinator::new();
        coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));
        coordinator.register_agent(AgentIdentity::new("agent_2", AgentRole::Specialist));

        let mut event_rx = coordinator.subscribe_events();

        let _ = coordinator.yield_to(
            &"agent_1".to_string(),
            None,
            make_yield_context("done"),
        );

        let event = event_rx.try_recv().expect("should have event");
        assert!(matches!(event, CoordinationEvent::AgentYielded { .. }));
        assert!(event_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn test_share_discovery_emits_event() {
        let coordinator = AgentCoordinator::new();
        coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));

        let mut event_rx = coordinator.subscribe_events();

        coordinator
            .share_discovery(
                &"agent_1".to_string(),
                make_discovery("finding", "test"),
            )
            .expect("should succeed");

        let event = event_rx.try_recv().expect("should have event");
        assert!(matches!(event, CoordinationEvent::DiscoveryShared { .. }));
        assert!(event_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn test_multiple_subscribers_all_receive() {
        let coordinator = AgentCoordinator::new();
        coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));

        let mut rx1 = coordinator.subscribe_events();
        let mut rx2 = coordinator.subscribe_events();
        let mut rx3 = coordinator.subscribe_events();

        coordinator
            .share_discovery(
                &"agent_1".to_string(),
                make_discovery("test", "test"),
            )
            .expect("should succeed");

        // All three subscribers receive the event
        assert!(rx1.try_recv().is_ok());
        assert!(rx2.try_recv().is_ok());
        assert!(rx3.try_recv().is_ok());
    }

    #[tokio::test]
    async fn test_failed_actions_do_not_emit_events() {
        let coordinator = AgentCoordinator::new();
        // No agents registered

        let mut event_rx = coordinator.subscribe_events();

        // Assistance from unregistered agent — fails
        let _ = coordinator.request_assistance(
            &"ghost".to_string(),
            blocking_assistance_request("help"),
        );

        // No event should have been emitted
        assert!(event_rx.try_recv().is_err());
    }

    // ===================================================================
    // §7.7 Cleanup on Unregister
    // ===================================================================

    #[tokio::test]
    async fn test_unregister_clears_yielded_state() {
        let coordinator = AgentCoordinator::new();
        coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));
        coordinator.register_agent(AgentIdentity::new("agent_2", AgentRole::Specialist));

        // Yield
        coordinator
            .yield_to(
                &"agent_1".to_string(),
                None,
                make_yield_context("done"),
            )
            .expect("should not error");

        // Unregister
        coordinator.unregister_agent("agent_1");

        // Re-register — should be able to yield again (not AlreadyYielded)
        coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));
        let result = coordinator.yield_to(
            &"agent_1".to_string(),
            None,
            make_yield_context("done again"),
        );
        assert!(matches!(result, Ok(YieldResult::Accepted)));
    }

    #[tokio::test]
    async fn test_unregister_cancels_pending_requests() {
        let coordinator = AgentCoordinator::new();
        coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));

        let rx = coordinator
            .request_assistance(
                &"agent_1".to_string(),
                blocking_assistance_request("need help"),
            )
            .expect("should succeed");

        // Unregister cancels the pending request (removes from map, drops the Sender)
        coordinator.unregister_agent("agent_1");

        // Receiver should get RecvError — sender dropped
        let result = rx.await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_unregister_preserves_discoveries() {
        let coordinator = AgentCoordinator::new();
        coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));
        coordinator.register_agent(AgentIdentity::new("reader", AgentRole::Primary));

        coordinator
            .share_discovery(
                &"agent_1".to_string(),
                make_discovery("important finding", "test"),
            )
            .expect("should succeed");

        // Unregister agent_1
        coordinator.unregister_agent("agent_1");

        // Discoveries survive
        let context = coordinator.get_shared_context(&"reader".to_string());
        assert_eq!(context.discoveries.len(), 1);
        assert_eq!(context.discoveries[0].0, "agent_1");
    }
}
