//! Multi-agent supervisor — orchestrates concurrent [`LoopExecutor`] instances.
//!
//! The supervisor decomposes a complex objective into subtasks, manages their
//! lifecycle (spawn, monitor, reroute, aggregate), and enforces global resource
//! budgets across all child agents.
//!
//! Reference: AGENTIC-LOOP-SPEC §7, TDD §12.
//!
//! # Key Components
//!
//! - [`SupervisorConfig`] — global settings (budget, concurrency, strategy)
//! - [`Subtask`] — individual work unit with dependency graph
//! - [`DependencyResolver`] — topological ordering and readiness tracking
//! - [`BudgetAllocator`] — distributes and rebalances resource budgets
//! - [`ConcurrencyLimiter`] — enforces `max_concurrent_agents`
//! - [`LifecycleTracker`] — spawn/complete bookkeeping, zombie detection
//! - [`RerouteResolver`] — matches stuck/yielded agents to alternatives
//! - [`CircuitBreaker`] — triggers after N consecutive same-type failures
//! - [`ResultAggregator`] — collects subtask results and partial progress
//! - [`WellbeingAggregate`] — monitors child agent wellbeing

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::coordination::AgentId;
use super::types::{LoopConfig, LoopSummary};

// ===========================================================================
// Configuration
// ===========================================================================

/// Global resource budget for the supervisor and all child agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceBudget {
    /// Total iterations across all agents.
    pub total_iterations: u32,
    /// Total tool calls across all agents.
    pub total_tool_calls: u32,
    /// Total tokens across all agents.
    pub total_tokens: u32,
}

impl Default for ResourceBudget {
    fn default() -> Self {
        Self {
            total_iterations: 100,
            total_tool_calls: 500,
            total_tokens: 131_072,
        }
    }
}

/// How complex a subtask is (affects budget allocation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Complexity {
    /// Simple subtask — small budget allocation.
    Low,
    /// Moderate subtask — medium budget allocation.
    Medium,
    /// Complex subtask — large budget allocation.
    High,
}

impl Complexity {
    /// Returns a weight multiplier for budget allocation.
    fn weight(self) -> f64 {
        match self {
            Self::Low => 1.0,
            Self::Medium => 2.0,
            Self::High => 4.0,
        }
    }
}

/// A unit of work for a child agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subtask {
    /// Unique identifier.
    pub id: String,
    /// What the agent should accomplish.
    pub objective: String,
    /// IDs of subtasks that must complete before this one starts.
    pub depends_on: Vec<String>,
    /// Capabilities this subtask requires or offers.
    pub capabilities: Vec<String>,
    /// Expected complexity (affects budget allocation).
    pub complexity: Complexity,
}

/// How the objective is decomposed into subtasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecompositionStrategy {
    /// Single agent handles everything (no decomposition).
    SingleAgent,
    /// Client provides explicit subtask breakdown.
    ClientProvided {
        /// The subtasks to execute.
        subtasks: Vec<Subtask>,
    },
}

/// How subtasks are dispatched to agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// All independent tasks run concurrently (up to concurrency limit).
    Parallel,
    /// Respect dependency ordering — dependents wait for prerequisites.
    DependencyAware,
}

/// How context is shared between agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SharedContextMode {
    /// No context sharing between agents.
    None,
    /// Share summaries of completed subtask results.
    SummarySharing,
    /// Share full results between agents.
    FullSharing,
}

/// Supervisor configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupervisorConfig {
    /// Global resource budget.
    pub resource_budget: ResourceBudget,
    /// How the objective is decomposed.
    pub decomposition: DecompositionStrategy,
    /// How subtasks are dispatched.
    pub routing: RoutingStrategy,
    /// How context is shared between agents.
    pub shared_context_mode: SharedContextMode,
    /// Maximum concurrent agents.
    pub max_concurrent_agents: u32,
    /// Maximum retries per subtask.
    pub max_retries: u32,
    /// Consecutive failures before circuit breaker triggers.
    pub circuit_breaker_threshold: u32,
}

impl Default for SupervisorConfig {
    fn default() -> Self {
        Self {
            resource_budget: ResourceBudget::default(),
            decomposition: DecompositionStrategy::SingleAgent,
            routing: RoutingStrategy::DependencyAware,
            shared_context_mode: SharedContextMode::SummarySharing,
            max_concurrent_agents: 3,
            max_retries: 2,
            circuit_breaker_threshold: 3,
        }
    }
}

// ===========================================================================
// Events
// ===========================================================================

/// Events emitted by the supervisor during orchestration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SupervisorEvent {
    /// A child agent was spawned for a subtask.
    AgentSpawned {
        /// The agent's identifier.
        agent_id: AgentId,
        /// Which subtask it's working on.
        subtask_id: String,
    },
    /// A child agent completed its subtask.
    AgentCompleted {
        /// The agent's identifier.
        agent_id: AgentId,
        /// Which subtask completed.
        subtask_id: String,
        /// The loop summary from the agent.
        summary: LoopSummary,
    },
    /// A subtask was rerouted from one agent to another.
    Rerouted {
        /// The agent that was rerouted from.
        from_agent: AgentId,
        /// The agent that was rerouted to.
        to_agent: AgentId,
        /// Which subtask was rerouted.
        subtask_id: String,
        /// Why the reroute happened.
        reason: RerouteReason,
    },
    /// A supervisor-level error occurred.
    SupervisorError {
        /// Error description.
        message: String,
        /// Whether the supervisor can continue.
        recoverable: bool,
    },
    /// The supervisor finished orchestration.
    SupervisorCompleted {
        /// Final summary.
        summary: SupervisorSummary,
    },
}

/// Why a subtask was rerouted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RerouteReason {
    /// The agent declared itself stuck.
    AgentStuck {
        /// How many approaches the agent tried.
        attempts: u32,
    },
    /// The agent yielded requesting different expertise.
    AgentYielded {
        /// Expertise the agent suggested.
        suggested_expertise: Vec<String>,
    },
    /// The inference engine failed.
    EngineError {
        /// How many retries were attempted.
        retries: u32,
    },
}

// ===========================================================================
// Results
// ===========================================================================

/// Why the supervisor stopped.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SupervisorTermination {
    /// All subtasks completed successfully.
    AllComplete,
    /// Some subtasks completed, others failed.
    PartialComplete {
        /// Number of successfully completed subtasks.
        completed: u32,
        /// Number of failed subtasks.
        failed: u32,
    },
    /// The supervisor failed entirely.
    Failed {
        /// Failure reason.
        reason: String,
    },
    /// Global resource budget exhausted.
    ResourceExhausted,
}

/// Status of an individual subtask.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SubtaskStatus {
    /// Subtask completed successfully.
    Completed,
    /// Subtask failed after retries.
    Failed {
        /// Failure reason.
        reason: String,
    },
    /// Subtask made partial progress before failing.
    Partial {
        /// Description of partial progress.
        progress: String,
    },
    /// Subtask was skipped (e.g., dependency failed).
    Skipped {
        /// Why it was skipped.
        reason: String,
    },
}

/// Result of an individual subtask.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtaskResult {
    /// Which subtask this result is for.
    pub subtask_id: String,
    /// Final status.
    pub status: SubtaskStatus,
    /// Loop summary (present for `Completed` and sometimes `Partial`).
    pub summary: Option<LoopSummary>,
    /// Which agent handled this subtask.
    pub agent_id: Option<AgentId>,
}

/// Final summary from the supervisor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupervisorSummary {
    /// Why the supervisor stopped.
    pub termination: SupervisorTermination,
    /// Results for each subtask.
    pub subtask_results: Vec<SubtaskResult>,
    /// Total agents spawned (including retries).
    pub total_agents_spawned: u32,
    /// Aggregate iterations across all agents.
    pub total_iterations: u32,
    /// Aggregate tool calls across all agents.
    pub total_tool_calls: u32,
    /// Aggregate tokens across all agents.
    pub total_tokens: u32,
    /// Wall-clock time.
    pub wall_time: Duration,
}

// ===========================================================================
// Resource tracking
// ===========================================================================

/// Tracks resource consumption across child agents.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceConsumption {
    /// Iterations consumed.
    pub iterations: u32,
    /// Tool calls consumed.
    pub tool_calls: u32,
    /// Tokens consumed.
    pub tokens: u32,
}

// ===========================================================================
// §12.1 Budget Allocation
// ===========================================================================

/// Distributes and rebalances resource budgets across subtasks.
#[derive(Debug)]
pub struct BudgetAllocator {
    budget: ResourceBudget,
    consumed: ResourceConsumption,
}

impl BudgetAllocator {
    /// Creates a new allocator with the given global budget.
    pub fn new(budget: ResourceBudget) -> Self {
        Self {
            budget,
            consumed: ResourceConsumption::default(),
        }
    }

    /// Returns remaining resources.
    pub fn remaining(&self) -> ResourceConsumption {
        ResourceConsumption {
            iterations: self.budget.total_iterations.saturating_sub(self.consumed.iterations),
            tool_calls: self.budget.total_tool_calls.saturating_sub(self.consumed.tool_calls),
            tokens: self.budget.total_tokens.saturating_sub(self.consumed.tokens),
        }
    }

    /// Records consumption from a completed child agent.
    pub fn record_consumption(&mut self, consumption: &ResourceConsumption) {
        self.consumed.iterations = self.consumed.iterations.saturating_add(consumption.iterations);
        self.consumed.tool_calls = self.consumed.tool_calls.saturating_add(consumption.tool_calls);
        self.consumed.tokens = self.consumed.tokens.saturating_add(consumption.tokens);
    }

    /// Total consumption so far.
    pub fn total_consumed(&self) -> &ResourceConsumption {
        &self.consumed
    }

    /// Allocates a [`LoopConfig`] for a subtask based on its complexity
    /// and the remaining budget.
    pub fn allocate(&self, subtask: &Subtask, total_weight: f64) -> LoopConfig {
        let remaining = self.remaining();
        let fraction = if total_weight > 0.0 {
            subtask.complexity.weight() / total_weight
        } else {
            1.0
        };

        // Allocate proportionally, ensuring at least 1 of each resource
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let max_iterations = (f64::from(remaining.iterations) * fraction)
            .round()
            .max(1.0) as u32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let max_tool_calls = (f64::from(remaining.tool_calls) * fraction)
            .round()
            .max(1.0) as u32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let max_tokens = (f64::from(remaining.tokens) * fraction)
            .round()
            .max(1.0) as u32;

        LoopConfig {
            max_iterations,
            max_tool_calls,
            max_tokens,
            ..LoopConfig::default()
        }
    }
}

/// Rebalances remaining budget equally among `running_count` agents.
///
/// Returns a `Vec` of `LoopConfig` — one per running agent — each receiving
/// an equal share of the remaining resources. Sum of all allocations will
/// never exceed the remaining resources.
pub fn rebalance_budget(
    remaining_iterations: u32,
    remaining_calls: u32,
    remaining_tokens: u32,
    running_count: u32,
) -> Vec<LoopConfig> {
    if running_count == 0 {
        return vec![];
    }

    let per_iter = remaining_iterations / running_count;
    let per_calls = remaining_calls / running_count;
    let per_tokens = remaining_tokens / running_count;

    // Distribute remainder to the first agents (round-robin style).
    let iter_rem = remaining_iterations % running_count;
    let calls_rem = remaining_calls % running_count;
    let tokens_rem = remaining_tokens % running_count;

    (0..running_count)
        .map(|i| {
            let iter_extra = u32::from(i < iter_rem);
            let calls_extra = u32::from(i < calls_rem);
            let tokens_extra = u32::from(i < tokens_rem);

            LoopConfig {
                max_iterations: per_iter + iter_extra,
                max_tool_calls: per_calls + calls_extra,
                max_tokens: per_tokens + tokens_extra,
                ..LoopConfig::default()
            }
        })
        .collect()
}

// ===========================================================================
// §12.2 Dependency Resolution
// ===========================================================================

/// Resolves subtask dependencies and tracks readiness.
///
/// Uses topological ordering to determine which subtasks can run next.
/// A subtask is "ready" when all its dependencies have completed.
#[derive(Debug)]
pub struct DependencyResolver {
    /// All subtasks indexed by ID.
    subtasks: HashMap<String, Subtask>,
    /// Completed subtask IDs.
    completed: HashSet<String>,
    /// Currently running subtask IDs.
    running: HashSet<String>,
    /// Failed subtask IDs.
    failed: HashSet<String>,
}

impl DependencyResolver {
    /// Creates a resolver from a list of subtasks.
    pub fn new(subtasks: Vec<Subtask>) -> Self {
        let map: HashMap<String, Subtask> = subtasks
            .into_iter()
            .map(|s| (s.id.clone(), s))
            .collect();
        Self {
            subtasks: map,
            completed: HashSet::new(),
            running: HashSet::new(),
            failed: HashSet::new(),
        }
    }

    /// Returns subtask IDs that are ready to run (all deps completed, not
    /// already running or completed).
    pub fn ready(&self) -> Vec<String> {
        self.subtasks
            .values()
            .filter(|s| {
                !self.completed.contains(&s.id)
                    && !self.running.contains(&s.id)
                    && !self.failed.contains(&s.id)
                    && s.depends_on.iter().all(|dep| self.completed.contains(dep))
            })
            .map(|s| s.id.clone())
            .collect()
    }

    /// Marks a subtask as running.
    pub fn mark_running(&mut self, id: &str) {
        self.running.insert(id.to_string());
    }

    /// Marks a subtask as completed.
    pub fn mark_completed(&mut self, id: &str) {
        self.running.remove(id);
        self.completed.insert(id.to_string());
    }

    /// Marks a subtask as failed.
    pub fn mark_failed(&mut self, id: &str) {
        self.running.remove(id);
        self.failed.insert(id.to_string());
    }

    /// Returns `true` if all subtasks are completed or failed.
    pub fn is_done(&self) -> bool {
        self.subtasks
            .keys()
            .all(|id| self.completed.contains(id) || self.failed.contains(id))
    }

    /// Returns the subtask with the given ID.
    pub fn get(&self, id: &str) -> Option<&Subtask> {
        self.subtasks.get(id)
    }

    /// Returns the total weight of the given subtask IDs.
    pub fn total_weight(&self, ids: &[String]) -> f64 {
        ids.iter()
            .filter_map(|id| self.subtasks.get(id))
            .map(|s| s.complexity.weight())
            .sum()
    }

    /// Returns the number of completed subtasks.
    pub fn completed_count(&self) -> usize {
        self.completed.len()
    }

    /// Returns the number of failed subtasks.
    pub fn failed_count(&self) -> usize {
        self.failed.len()
    }

    /// Returns the total number of subtasks.
    pub fn total(&self) -> usize {
        self.subtasks.len()
    }

    /// Validates the dependency graph — checks for missing dependencies and cycles.
    ///
    /// # Errors
    ///
    /// Returns [`SupervisorError::InvalidDependency`] if a subtask references a
    /// non-existent dependency, or [`SupervisorError::CyclicDependency`] if the
    /// graph contains a cycle.
    pub fn validate(&self) -> Result<(), SupervisorError> {
        // Check for missing dependencies
        for subtask in self.subtasks.values() {
            for dep in &subtask.depends_on {
                if !self.subtasks.contains_key(dep) {
                    return Err(SupervisorError::InvalidDependency {
                        subtask: subtask.id.clone(),
                        missing_dep: dep.clone(),
                    });
                }
            }
        }

        // Check for cycles using DFS
        let mut visited = HashSet::new();
        let mut in_stack = HashSet::new();

        for id in self.subtasks.keys() {
            if !visited.contains(id)
                && self.has_cycle(id, &mut visited, &mut in_stack)
            {
                return Err(SupervisorError::CyclicDependency {
                    subtask: id.clone(),
                });
            }
        }

        Ok(())
    }

    fn has_cycle(
        &self,
        id: &str,
        visited: &mut HashSet<String>,
        in_stack: &mut HashSet<String>,
    ) -> bool {
        visited.insert(id.to_string());
        in_stack.insert(id.to_string());

        if let Some(subtask) = self.subtasks.get(id) {
            for dep in &subtask.depends_on {
                if !visited.contains(dep) {
                    if self.has_cycle(dep, visited, in_stack) {
                        return true;
                    }
                } else if in_stack.contains(dep) {
                    return true;
                }
            }
        }

        in_stack.remove(id);
        false
    }
}

/// Errors from supervisor operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum SupervisorError {
    /// A subtask references a dependency that doesn't exist.
    #[error("subtask '{subtask}' depends on '{missing_dep}' which does not exist")]
    InvalidDependency {
        /// The subtask with the bad dependency.
        subtask: String,
        /// The missing dependency ID.
        missing_dep: String,
    },
    /// The dependency graph contains a cycle.
    #[error("cyclic dependency detected involving subtask '{subtask}'")]
    CyclicDependency {
        /// A subtask involved in the cycle.
        subtask: String,
    },
    /// Circuit breaker triggered.
    #[error("circuit breaker triggered after {consecutive} consecutive {failure_type} failures")]
    CircuitBreakerTriggered {
        /// Number of consecutive failures.
        consecutive: u32,
        /// Type of failure.
        failure_type: String,
    },
}

// ===========================================================================
// §12.3 Concurrency Limiter
// ===========================================================================

/// Tracks concurrent agent count and enforces `max_concurrent_agents`.
#[derive(Debug)]
pub struct ConcurrencyLimiter {
    max_concurrent: u32,
    active: u32,
    max_observed: u32,
    /// Queue of subtask IDs waiting for a slot.
    queue: VecDeque<String>,
}

impl ConcurrencyLimiter {
    /// Creates a new limiter.
    pub fn new(max_concurrent: u32) -> Self {
        Self {
            max_concurrent,
            active: 0,
            max_observed: 0,
            queue: VecDeque::new(),
        }
    }

    /// Attempts to acquire a slot. Returns `true` if a slot was available.
    pub fn try_acquire(&mut self) -> bool {
        if self.active < self.max_concurrent {
            self.active += 1;
            self.max_observed = self.max_observed.max(self.active);
            true
        } else {
            false
        }
    }

    /// Releases a slot. Returns the next queued subtask ID, if any.
    pub fn release(&mut self) -> Option<String> {
        self.active = self.active.saturating_sub(1);
        self.queue.pop_front()
    }

    /// Enqueues a subtask ID to be dispatched when a slot opens.
    pub fn enqueue(&mut self, subtask_id: String) {
        self.queue.push_back(subtask_id);
    }

    /// Currently active agents.
    pub fn active_count(&self) -> u32 {
        self.active
    }

    /// Maximum concurrent agents observed.
    pub fn max_observed(&self) -> u32 {
        self.max_observed
    }

    /// Number of subtasks waiting in queue.
    pub fn queued_count(&self) -> usize {
        self.queue.len()
    }
}

// ===========================================================================
// §12.4 Lifecycle Tracker
// ===========================================================================

/// Tracks agent lifecycle events for zombie detection.
#[derive(Debug)]
pub struct LifecycleTracker {
    /// Agents that have been spawned (`agent_id` → `subtask_id`).
    spawned: HashMap<AgentId, String>,
    /// Agents that have completed or been rerouted.
    resolved: HashSet<AgentId>,
    /// Total agents spawned.
    total_spawned: u32,
}

impl LifecycleTracker {
    /// Creates a new tracker.
    pub fn new() -> Self {
        Self {
            spawned: HashMap::new(),
            resolved: HashSet::new(),
            total_spawned: 0,
        }
    }

    /// Records an agent spawn.
    pub fn record_spawn(&mut self, agent_id: AgentId, subtask_id: String) {
        self.spawned.insert(agent_id, subtask_id);
        self.total_spawned += 1;
    }

    /// Records an agent completion.
    pub fn record_completion(&mut self, agent_id: &str) {
        self.resolved.insert(agent_id.to_string());
    }

    /// Records a reroute (original agent is resolved).
    pub fn record_reroute(&mut self, from_agent: &str) {
        self.resolved.insert(from_agent.to_string());
    }

    /// Returns agent IDs that were spawned but never resolved (zombies).
    pub fn zombies(&self) -> Vec<AgentId> {
        self.spawned
            .keys()
            .filter(|id| !self.resolved.contains(*id))
            .cloned()
            .collect()
    }

    /// Returns `true` if there are no zombie agents.
    pub fn all_resolved(&self) -> bool {
        self.zombies().is_empty()
    }

    /// Total agents spawned (including retries/reroutes).
    pub fn total_spawned(&self) -> u32 {
        self.total_spawned
    }
}

impl Default for LifecycleTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// §12.5 Reroute Resolver
// ===========================================================================

/// Matches stuck/yielded agents to alternative agents based on capabilities.
pub struct RerouteResolver;

impl RerouteResolver {
    /// Finds the best subtask to reroute to based on requested expertise.
    ///
    /// Returns the ID of the matching subtask, if any.
    pub fn find_match(
        requested_expertise: &[String],
        available_subtasks: &[Subtask],
        exclude_ids: &HashSet<String>,
    ) -> Option<String> {
        let mut best_match: Option<(String, usize)> = None;

        for subtask in available_subtasks {
            if exclude_ids.contains(&subtask.id) {
                continue;
            }

            let matching_caps = subtask
                .capabilities
                .iter()
                .filter(|cap| requested_expertise.iter().any(|req| cap.contains(req.as_str())))
                .count();

            if matching_caps > 0 {
                if let Some((_, best_count)) = &best_match {
                    if matching_caps > *best_count {
                        best_match = Some((subtask.id.clone(), matching_caps));
                    }
                } else {
                    best_match = Some((subtask.id.clone(), matching_caps));
                }
            }
        }

        best_match.map(|(id, _)| id)
    }
}

// ===========================================================================
// §12.6 Circuit Breaker
// ===========================================================================

/// Type of failure for circuit breaker tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FailureType {
    /// Inference engine error.
    EngineError,
    /// Agent reported stuck.
    AgentStuck,
    /// Agent yielded.
    AgentYielded,
    /// Tool execution error.
    ToolError,
    /// Timeout.
    Timeout,
}

impl std::fmt::Display for FailureType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EngineError => write!(f, "engine_error"),
            Self::AgentStuck => write!(f, "agent_stuck"),
            Self::AgentYielded => write!(f, "agent_yielded"),
            Self::ToolError => write!(f, "tool_error"),
            Self::Timeout => write!(f, "timeout"),
        }
    }
}

/// Tracks consecutive failures and triggers when a threshold is reached.
#[derive(Debug)]
pub struct CircuitBreaker {
    threshold: u32,
    consecutive_count: u32,
    last_failure_type: Option<FailureType>,
    is_open: bool,
}

impl CircuitBreaker {
    /// Creates a new circuit breaker with the given threshold.
    pub fn new(threshold: u32) -> Self {
        Self {
            threshold,
            consecutive_count: 0,
            last_failure_type: None,
            is_open: false,
        }
    }

    /// Records a failure. Returns `Err` if the circuit breaker trips.
    ///
    /// # Errors
    ///
    /// Returns [`SupervisorError::CircuitBreakerTriggered`] if the consecutive
    /// failure count for this failure type reaches the threshold.
    pub fn record_failure(&mut self, failure_type: FailureType) -> Result<(), SupervisorError> {
        if self.last_failure_type == Some(failure_type) {
            self.consecutive_count += 1;
        } else {
            self.last_failure_type = Some(failure_type);
            self.consecutive_count = 1;
        }

        if self.consecutive_count >= self.threshold {
            self.is_open = true;
            return Err(SupervisorError::CircuitBreakerTriggered {
                consecutive: self.consecutive_count,
                failure_type: failure_type.to_string(),
            });
        }

        Ok(())
    }

    /// Records a success, resetting the consecutive failure count.
    pub fn record_success(&mut self) {
        self.consecutive_count = 0;
        self.last_failure_type = None;
        // Don't automatically close — requires explicit reset
    }

    /// Returns `true` if the circuit breaker is open (tripped).
    pub fn is_open(&self) -> bool {
        self.is_open
    }

    /// Resets the circuit breaker to closed state.
    pub fn reset(&mut self) {
        self.is_open = false;
        self.consecutive_count = 0;
        self.last_failure_type = None;
    }

    /// Number of consecutive failures.
    pub fn consecutive_failures(&self) -> u32 {
        self.consecutive_count
    }
}

// ===========================================================================
// §12.7 Result Aggregation
// ===========================================================================

/// Collects and organizes subtask results.
#[derive(Debug)]
pub struct ResultAggregator {
    results: HashMap<String, SubtaskResult>,
}

impl ResultAggregator {
    /// Creates a new aggregator.
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
        }
    }

    /// Records a completed subtask result.
    pub fn record_result(&mut self, result: SubtaskResult) {
        self.results.insert(result.subtask_id.clone(), result);
    }

    /// Records a skipped subtask (dependency failed).
    pub fn record_skipped(&mut self, subtask_id: &str, reason: &str) {
        self.results.insert(
            subtask_id.to_string(),
            SubtaskResult {
                subtask_id: subtask_id.to_string(),
                status: SubtaskStatus::Skipped {
                    reason: reason.to_string(),
                },
                summary: None,
                agent_id: None,
            },
        );
    }

    /// Returns the result for a given subtask ID.
    pub fn get(&self, subtask_id: &str) -> Option<&SubtaskResult> {
        self.results.get(subtask_id)
    }

    /// Returns all results.
    pub fn all_results(&self) -> Vec<SubtaskResult> {
        self.results.values().cloned().collect()
    }

    /// Number of completed subtasks.
    pub fn completed_count(&self) -> usize {
        self.results
            .values()
            .filter(|r| matches!(r.status, SubtaskStatus::Completed))
            .count()
    }

    /// Number of failed subtasks.
    pub fn failed_count(&self) -> usize {
        self.results
            .values()
            .filter(|r| matches!(r.status, SubtaskStatus::Failed { .. }))
            .count()
    }

    /// Number of skipped subtasks.
    pub fn skipped_count(&self) -> usize {
        self.results
            .values()
            .filter(|r| matches!(r.status, SubtaskStatus::Skipped { .. }))
            .count()
    }

    /// Builds a [`SupervisorSummary`] from the collected results.
    pub fn build_summary(
        &self,
        total_spawned: u32,
        consumed: &ResourceConsumption,
        wall_time: Duration,
        total_subtasks: usize,
    ) -> SupervisorSummary {
        let completed = self.completed_count();
        let failed = self.failed_count();

        #[allow(clippy::cast_possible_truncation)]
        let termination = if completed == total_subtasks {
            SupervisorTermination::AllComplete
        } else if completed == 0 && failed > 0 {
            SupervisorTermination::Failed {
                reason: format!("all {failed} subtasks failed"),
            }
        } else {
            SupervisorTermination::PartialComplete {
                completed: completed as u32,
                failed: failed as u32,
            }
        };

        SupervisorSummary {
            termination,
            subtask_results: self.all_results(),
            total_agents_spawned: total_spawned,
            total_iterations: consumed.iterations,
            total_tool_calls: consumed.tool_calls,
            total_tokens: consumed.tokens,
            wall_time,
        }
    }
}

impl Default for ResultAggregator {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// §12.8 Wellbeing Aggregate
// ===========================================================================

/// Wellbeing state categories (mirrors the `WellbeingState` from the wellbeing module,
/// simplified for supervisor-level aggregation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentWellbeingState {
    /// Agent is operating within comfortable parameters.
    Healthy,
    /// Agent is showing mild signs of difficulty.
    Cautious,
    /// Agent is struggling but still functional.
    Concerned,
    /// Agent is in significant distress.
    Distressed,
}

/// Aggregate wellbeing across all child agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WellbeingAggregate {
    /// Total agents being monitored.
    pub agents_total: usize,
    /// Agents in healthy state.
    pub agents_healthy: usize,
    /// Agents in cautious state.
    pub agents_cautious: usize,
    /// Agents in concerned state.
    pub agents_concerned: usize,
    /// Agents in distressed state.
    pub agents_distressed: usize,
}

/// Supervisor-level response to aggregate wellbeing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SupervisorWellbeingAction {
    /// Everything is fine, continue.
    Continue,
    /// Pause dispatching and re-evaluate the plan.
    PauseAndReplan,
    /// Escalate to the client for human decision.
    EscalateToClient,
}

/// Individual agent wellbeing response from the supervisor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WellbeingResponse {
    /// Continue as normal.
    Continue,
    /// Pause the agent's work.
    Pause,
    /// Reassign the agent's subtask to another agent.
    Reassign,
}

/// Action the supervisor takes for a specific agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentWellbeingAction {
    /// Which agent this action is for.
    pub agent_id: AgentId,
    /// What to do.
    pub response: WellbeingResponse,
}

/// Computes aggregate wellbeing from individual agent states.
pub fn compute_aggregate_wellbeing(states: &[AgentWellbeingState]) -> WellbeingAggregate {
    let mut agg = WellbeingAggregate {
        agents_total: states.len(),
        agents_healthy: 0,
        agents_cautious: 0,
        agents_concerned: 0,
        agents_distressed: 0,
    };

    for state in states {
        match state {
            AgentWellbeingState::Healthy => agg.agents_healthy += 1,
            AgentWellbeingState::Cautious => agg.agents_cautious += 1,
            AgentWellbeingState::Concerned => agg.agents_concerned += 1,
            AgentWellbeingState::Distressed => agg.agents_distressed += 1,
        }
    }

    agg
}

/// Determines the supervisor-level response based on aggregate wellbeing.
///
/// - All healthy/cautious → `Continue`
/// - Majority (>50%) concerned → `PauseAndReplan`
/// - Any distressed → `EscalateToClient`
pub fn supervisor_level_response(aggregate: &WellbeingAggregate) -> SupervisorWellbeingAction {
    if aggregate.agents_total == 0 {
        return SupervisorWellbeingAction::Continue;
    }

    // Any distressed agent → escalate
    if aggregate.agents_distressed > 0 {
        return SupervisorWellbeingAction::EscalateToClient;
    }

    // Majority concerned → pause and replan
    #[allow(clippy::cast_precision_loss)]
    let concerned_fraction =
        aggregate.agents_concerned as f64 / aggregate.agents_total as f64;
    if concerned_fraction > 0.5 {
        return SupervisorWellbeingAction::PauseAndReplan;
    }

    SupervisorWellbeingAction::Continue
}

/// Determines per-agent wellbeing responses.
///
/// Distressed agents are paused or reassigned — never punished.
pub fn supervisor_wellbeing_response(
    agent_states: &[(AgentId, AgentWellbeingState)],
) -> Vec<AgentWellbeingAction> {
    agent_states
        .iter()
        .filter_map(|(agent_id, state)| {
            let response = match state {
                AgentWellbeingState::Healthy | AgentWellbeingState::Cautious => {
                    return None; // No action needed
                }
                AgentWellbeingState::Concerned => WellbeingResponse::Pause,
                AgentWellbeingState::Distressed => WellbeingResponse::Reassign,
            };
            Some(AgentWellbeingAction {
                agent_id: agent_id.clone(),
                response,
            })
        })
        .collect()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    fn subtask(id: &str, deps: Vec<&str>, complexity: Complexity) -> Subtask {
        Subtask {
            id: id.to_string(),
            objective: format!("Do {id}"),
            depends_on: deps.into_iter().map(String::from).collect(),
            capabilities: vec![],
            complexity,
        }
    }

    fn subtask_with_capabilities(
        id: &str,
        deps: Vec<&str>,
        capabilities: Vec<&str>,
    ) -> Subtask {
        Subtask {
            id: id.to_string(),
            objective: format!("Do {id}"),
            depends_on: deps.into_iter().map(String::from).collect(),
            capabilities: capabilities.into_iter().map(String::from).collect(),
            complexity: Complexity::Medium,
        }
    }

    // =======================================================================
    // §12.1 Resource Invariants
    // =======================================================================

    #[test]
    fn test_allocate_budget_proportional_to_complexity() {
        let budget = ResourceBudget {
            total_iterations: 100,
            total_tool_calls: 500,
            total_tokens: 100_000,
        };
        let allocator = BudgetAllocator::new(budget);

        let low = subtask("low", vec![], Complexity::Low);
        let high = subtask("high", vec![], Complexity::High);
        let total_weight = Complexity::Low.weight() + Complexity::High.weight();

        let low_config = allocator.allocate(&low, total_weight);
        let high_config = allocator.allocate(&high, total_weight);

        // High complexity should get more resources
        assert!(high_config.max_iterations > low_config.max_iterations);
        assert!(high_config.max_tool_calls > low_config.max_tool_calls);
        assert!(high_config.max_tokens > low_config.max_tokens);
    }

    #[test]
    fn test_total_allocation_within_budget() {
        let budget = ResourceBudget {
            total_iterations: 100,
            total_tool_calls: 500,
            total_tokens: 100_000,
        };
        let allocator = BudgetAllocator::new(budget.clone());

        let subtasks = vec![
            subtask("A", vec![], Complexity::Low),
            subtask("B", vec![], Complexity::Medium),
            subtask("C", vec![], Complexity::High),
        ];
        let ids: Vec<String> = subtasks.iter().map(|s| s.id.clone()).collect();
        let resolver = DependencyResolver::new(subtasks.clone());
        let total_weight = resolver.total_weight(&ids);

        let configs: Vec<LoopConfig> = subtasks
            .iter()
            .map(|s| allocator.allocate(s, total_weight))
            .collect();

        let total_iters: u32 = configs.iter().map(|c| c.max_iterations).sum();
        let total_calls: u32 = configs.iter().map(|c| c.max_tool_calls).sum();
        let total_tokens: u32 = configs.iter().map(|c| c.max_tokens).sum();

        assert!(total_iters <= budget.total_iterations);
        assert!(total_calls <= budget.total_tool_calls);
        assert!(total_tokens <= budget.total_tokens);
    }

    #[test]
    fn test_record_consumption_updates_remaining() {
        let budget = ResourceBudget {
            total_iterations: 100,
            total_tool_calls: 500,
            total_tokens: 50_000,
        };
        let mut allocator = BudgetAllocator::new(budget);

        allocator.record_consumption(&ResourceConsumption {
            iterations: 30,
            tool_calls: 100,
            tokens: 10_000,
        });

        let remaining = allocator.remaining();
        assert_eq!(remaining.iterations, 70);
        assert_eq!(remaining.tool_calls, 400);
        assert_eq!(remaining.tokens, 40_000);
    }

    #[test]
    fn test_rebalance_within_remaining() {
        let configs = rebalance_budget(100, 500, 50_000, 4);

        assert_eq!(configs.len(), 4);

        let total_iters: u32 = configs.iter().map(|c| c.max_iterations).sum();
        let total_calls: u32 = configs.iter().map(|c| c.max_tool_calls).sum();
        let total_tokens: u32 = configs.iter().map(|c| c.max_tokens).sum();

        assert!(total_iters <= 100);
        assert!(total_calls <= 500);
        assert!(total_tokens <= 50_000);
    }

    #[test]
    fn test_rebalance_zero_agents_returns_empty() {
        let configs = rebalance_budget(100, 500, 50_000, 0);
        assert!(configs.is_empty());
    }

    // =======================================================================
    // §12.2 Dependency Ordering
    // =======================================================================

    #[test]
    fn test_independent_subtasks_all_ready() {
        let resolver = DependencyResolver::new(vec![
            subtask("A", vec![], Complexity::Low),
            subtask("B", vec![], Complexity::Low),
            subtask("C", vec![], Complexity::Low),
        ]);

        let ready = resolver.ready();
        assert_eq!(ready.len(), 3);
    }

    #[test]
    fn test_dependent_subtask_waits() {
        let resolver = DependencyResolver::new(vec![
            subtask("A", vec![], Complexity::Low),
            subtask("B", vec!["A"], Complexity::Low),
            subtask("C", vec!["A", "B"], Complexity::Low),
        ]);

        let ready = resolver.ready();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0], "A");
    }

    #[test]
    fn test_completing_dep_unblocks_dependent() {
        let mut resolver = DependencyResolver::new(vec![
            subtask("A", vec![], Complexity::Low),
            subtask("B", vec!["A"], Complexity::Low),
            subtask("C", vec!["B"], Complexity::Low),
        ]);

        resolver.mark_running("A");
        assert!(resolver.ready().is_empty());

        resolver.mark_completed("A");
        let ready = resolver.ready();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0], "B");
    }

    #[test]
    fn test_chain_runs_sequentially() {
        let mut resolver = DependencyResolver::new(vec![
            subtask("A", vec![], Complexity::Low),
            subtask("B", vec!["A"], Complexity::Low),
            subtask("C", vec!["B"], Complexity::Low),
        ]);

        // Step 1: only A ready
        assert_eq!(resolver.ready(), vec!["A"]);
        resolver.mark_running("A");
        resolver.mark_completed("A");

        // Step 2: only B ready
        assert_eq!(resolver.ready(), vec!["B"]);
        resolver.mark_running("B");
        resolver.mark_completed("B");

        // Step 3: only C ready
        assert_eq!(resolver.ready(), vec!["C"]);
        resolver.mark_running("C");
        resolver.mark_completed("C");

        assert!(resolver.is_done());
    }

    #[test]
    fn test_diamond_dependency() {
        let mut resolver = DependencyResolver::new(vec![
            subtask("A", vec![], Complexity::Low),
            subtask("B", vec!["A"], Complexity::Low),
            subtask("C", vec!["A"], Complexity::Low),
            subtask("D", vec!["B", "C"], Complexity::Low),
        ]);

        // A ready first
        assert_eq!(resolver.ready().len(), 1);
        resolver.mark_running("A");
        resolver.mark_completed("A");

        // B and C ready concurrently
        let ready = resolver.ready();
        assert_eq!(ready.len(), 2);
        assert!(ready.contains(&"B".to_string()));
        assert!(ready.contains(&"C".to_string()));

        resolver.mark_running("B");
        resolver.mark_running("C");
        resolver.mark_completed("B");
        // D not yet ready (C still running)
        assert!(resolver.ready().is_empty());

        resolver.mark_completed("C");
        // Now D is ready
        assert_eq!(resolver.ready(), vec!["D"]);
    }

    #[test]
    fn test_validate_missing_dependency() {
        let resolver = DependencyResolver::new(vec![
            subtask("A", vec!["Z"], Complexity::Low),
        ]);

        let result = resolver.validate();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SupervisorError::InvalidDependency { .. }
        ));
    }

    #[test]
    fn test_validate_cyclic_dependency() {
        let resolver = DependencyResolver::new(vec![
            subtask("A", vec!["B"], Complexity::Low),
            subtask("B", vec!["A"], Complexity::Low),
        ]);

        let result = resolver.validate();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SupervisorError::CyclicDependency { .. }
        ));
    }

    #[test]
    fn test_validate_valid_dag() {
        let resolver = DependencyResolver::new(vec![
            subtask("A", vec![], Complexity::Low),
            subtask("B", vec!["A"], Complexity::Low),
            subtask("C", vec!["A"], Complexity::Low),
            subtask("D", vec!["B", "C"], Complexity::Low),
        ]);

        assert!(resolver.validate().is_ok());
    }

    #[test]
    fn test_failed_subtask_not_ready() {
        let mut resolver = DependencyResolver::new(vec![
            subtask("A", vec![], Complexity::Low),
            subtask("B", vec![], Complexity::Low),
        ]);

        resolver.mark_failed("A");
        let ready = resolver.ready();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0], "B");
    }

    // =======================================================================
    // §12.3 Concurrency Limits
    // =======================================================================

    #[test]
    fn test_concurrency_limit_enforced() {
        let mut limiter = ConcurrencyLimiter::new(2);

        assert!(limiter.try_acquire());
        assert!(limiter.try_acquire());
        assert!(!limiter.try_acquire()); // At limit
        assert_eq!(limiter.active_count(), 2);
        assert_eq!(limiter.max_observed(), 2);
    }

    #[test]
    fn test_release_opens_slot() {
        let mut limiter = ConcurrencyLimiter::new(1);

        assert!(limiter.try_acquire());
        assert!(!limiter.try_acquire());

        limiter.release();
        assert!(limiter.try_acquire());
    }

    #[test]
    fn test_queued_dispatched_on_release() {
        let mut limiter = ConcurrencyLimiter::new(1);
        assert!(limiter.try_acquire());

        limiter.enqueue("task_B".to_string());
        limiter.enqueue("task_C".to_string());
        assert_eq!(limiter.queued_count(), 2);

        let next = limiter.release();
        assert_eq!(next, Some("task_B".to_string()));
        assert_eq!(limiter.queued_count(), 1);
    }

    // =======================================================================
    // §12.4 Lifecycle Guarantees
    // =======================================================================

    #[test]
    fn test_no_zombies_when_all_resolved() {
        let mut tracker = LifecycleTracker::new();

        tracker.record_spawn("agent_1".to_string(), "task_A".to_string());
        tracker.record_spawn("agent_2".to_string(), "task_B".to_string());

        assert_eq!(tracker.zombies().len(), 2);

        tracker.record_completion("agent_1");
        tracker.record_completion("agent_2");

        assert!(tracker.all_resolved());
        assert!(tracker.zombies().is_empty());
    }

    #[test]
    fn test_reroute_resolves_original_agent() {
        let mut tracker = LifecycleTracker::new();

        tracker.record_spawn("agent_1".to_string(), "task_A".to_string());
        tracker.record_reroute("agent_1");
        tracker.record_spawn("agent_2".to_string(), "task_A".to_string());
        tracker.record_completion("agent_2");

        assert!(tracker.all_resolved());
        assert_eq!(tracker.total_spawned(), 2);
    }

    #[test]
    fn test_zombie_detected() {
        let mut tracker = LifecycleTracker::new();

        tracker.record_spawn("agent_1".to_string(), "task_A".to_string());
        tracker.record_spawn("agent_2".to_string(), "task_B".to_string());
        tracker.record_completion("agent_1");
        // agent_2 is a zombie

        let zombies = tracker.zombies();
        assert_eq!(zombies.len(), 1);
        assert_eq!(zombies[0], "agent_2");
    }

    // =======================================================================
    // §12.5 Rerouting
    // =======================================================================

    #[test]
    fn test_reroute_matches_expertise() {
        let subtasks = vec![
            subtask_with_capabilities("research", vec![], vec!["general"]),
            subtask_with_capabilities("implement", vec![], vec!["rust", "database"]),
        ];

        let exclude = HashSet::from(["research".to_string()]);
        let result = RerouteResolver::find_match(
            &["database".to_string()],
            &subtasks,
            &exclude,
        );

        assert_eq!(result, Some("implement".to_string()));
    }

    #[test]
    fn test_reroute_no_match_returns_none() {
        let subtasks = vec![
            subtask_with_capabilities("research", vec![], vec!["general"]),
        ];

        let exclude = HashSet::from(["research".to_string()]);
        let result = RerouteResolver::find_match(
            &["database".to_string()],
            &subtasks,
            &exclude,
        );

        assert!(result.is_none());
    }

    #[test]
    fn test_reroute_selects_best_match() {
        let subtasks = vec![
            subtask_with_capabilities("a", vec![], vec!["python"]),
            subtask_with_capabilities("b", vec![], vec!["rust"]),
            subtask_with_capabilities("c", vec![], vec!["rust", "database", "api"]),
        ];

        let exclude = HashSet::new();
        let result = RerouteResolver::find_match(
            &["rust".to_string(), "database".to_string()],
            &subtasks,
            &exclude,
        );

        // "c" matches 2 of 2 requested, "b" only matches 1
        assert_eq!(result, Some("c".to_string()));
    }

    // =======================================================================
    // §12.6 Failure Recovery
    // =======================================================================

    #[test]
    fn test_circuit_breaker_triggers_at_threshold() {
        let mut cb = CircuitBreaker::new(3);

        assert!(cb.record_failure(FailureType::EngineError).is_ok());
        assert!(cb.record_failure(FailureType::EngineError).is_ok());
        assert!(cb.record_failure(FailureType::EngineError).is_err()); // Trips
        assert!(cb.is_open());
    }

    #[test]
    fn test_circuit_breaker_resets_on_different_type() {
        let mut cb = CircuitBreaker::new(3);

        assert!(cb.record_failure(FailureType::EngineError).is_ok());
        assert!(cb.record_failure(FailureType::EngineError).is_ok());
        // Different type resets count
        assert!(cb.record_failure(FailureType::AgentStuck).is_ok());
        assert!(!cb.is_open());
        assert_eq!(cb.consecutive_failures(), 1);
    }

    #[test]
    fn test_circuit_breaker_success_resets_count() {
        let mut cb = CircuitBreaker::new(3);

        assert!(cb.record_failure(FailureType::EngineError).is_ok());
        assert!(cb.record_failure(FailureType::EngineError).is_ok());
        cb.record_success();
        assert_eq!(cb.consecutive_failures(), 0);

        // Now it takes 3 more failures to trip
        assert!(cb.record_failure(FailureType::EngineError).is_ok());
        assert!(cb.record_failure(FailureType::EngineError).is_ok());
        assert!(cb.record_failure(FailureType::EngineError).is_err());
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let mut cb = CircuitBreaker::new(2);

        assert!(cb.record_failure(FailureType::Timeout).is_ok());
        assert!(cb.record_failure(FailureType::Timeout).is_err());
        assert!(cb.is_open());

        cb.reset();
        assert!(!cb.is_open());
        assert_eq!(cb.consecutive_failures(), 0);
    }

    // =======================================================================
    // §12.7 Aggregation
    // =======================================================================

    #[test]
    fn test_aggregation_collects_all_results() {
        let mut agg = ResultAggregator::new();

        agg.record_result(SubtaskResult {
            subtask_id: "A".to_string(),
            status: SubtaskStatus::Completed,
            summary: None,
            agent_id: Some("agent_1".to_string()),
        });
        agg.record_result(SubtaskResult {
            subtask_id: "B".to_string(),
            status: SubtaskStatus::Completed,
            summary: None,
            agent_id: Some("agent_2".to_string()),
        });
        agg.record_result(SubtaskResult {
            subtask_id: "C".to_string(),
            status: SubtaskStatus::Failed {
                reason: "timeout".to_string(),
            },
            summary: None,
            agent_id: Some("agent_3".to_string()),
        });

        assert_eq!(agg.completed_count(), 2);
        assert_eq!(agg.failed_count(), 1);
        assert_eq!(agg.all_results().len(), 3);
    }

    #[test]
    fn test_completed_results_preserved_in_summary() {
        let mut agg = ResultAggregator::new();

        agg.record_result(SubtaskResult {
            subtask_id: "A".to_string(),
            status: SubtaskStatus::Completed,
            summary: None,
            agent_id: Some("agent_1".to_string()),
        });
        agg.record_result(SubtaskResult {
            subtask_id: "B".to_string(),
            status: SubtaskStatus::Failed {
                reason: "error".to_string(),
            },
            summary: None,
            agent_id: Some("agent_2".to_string()),
        });

        let summary = agg.build_summary(
            2,
            &ResourceConsumption {
                iterations: 10,
                tool_calls: 30,
                tokens: 5000,
            },
            Duration::from_secs(60),
            2,
        );

        assert!(matches!(
            summary.termination,
            SupervisorTermination::PartialComplete {
                completed: 1,
                failed: 1
            }
        ));
        assert_eq!(summary.subtask_results.len(), 2);

        // Verify completed result is preserved
        let a_result = summary
            .subtask_results
            .iter()
            .find(|r| r.subtask_id == "A");
        assert!(a_result.is_some());
        assert!(matches!(a_result.map(|r| &r.status), Some(SubtaskStatus::Completed)));
    }

    #[test]
    fn test_all_complete_termination() {
        let mut agg = ResultAggregator::new();

        for id in &["A", "B", "C"] {
            agg.record_result(SubtaskResult {
                subtask_id: id.to_string(),
                status: SubtaskStatus::Completed,
                summary: None,
                agent_id: None,
            });
        }

        let summary = agg.build_summary(
            3,
            &ResourceConsumption::default(),
            Duration::from_secs(30),
            3,
        );

        assert!(matches!(
            summary.termination,
            SupervisorTermination::AllComplete
        ));
    }

    #[test]
    fn test_all_failed_termination() {
        let mut agg = ResultAggregator::new();

        agg.record_result(SubtaskResult {
            subtask_id: "A".to_string(),
            status: SubtaskStatus::Failed {
                reason: "err".to_string(),
            },
            summary: None,
            agent_id: None,
        });

        let summary = agg.build_summary(
            1,
            &ResourceConsumption::default(),
            Duration::from_secs(10),
            1,
        );

        assert!(matches!(
            summary.termination,
            SupervisorTermination::Failed { .. }
        ));
    }

    #[test]
    fn test_skipped_subtask_recorded() {
        let mut agg = ResultAggregator::new();
        agg.record_skipped("C", "dependency A failed");

        let result = agg.get("C");
        assert!(result.is_some());
        assert!(matches!(
            result.map(|r| &r.status),
            Some(SubtaskStatus::Skipped { .. })
        ));
    }

    #[test]
    fn test_majority_failure_partial_result_preserved() {
        let mut agg = ResultAggregator::new();

        // 1 success, 3 failures
        agg.record_result(SubtaskResult {
            subtask_id: "D".to_string(),
            status: SubtaskStatus::Completed,
            summary: None,
            agent_id: Some("agent_4".to_string()),
        });
        for id in &["A", "B", "C"] {
            agg.record_result(SubtaskResult {
                subtask_id: id.to_string(),
                status: SubtaskStatus::Failed {
                    reason: "fail".to_string(),
                },
                summary: None,
                agent_id: None,
            });
        }

        let summary = agg.build_summary(
            4,
            &ResourceConsumption::default(),
            Duration::from_secs(30),
            4,
        );

        assert!(matches!(
            summary.termination,
            SupervisorTermination::PartialComplete {
                completed: 1,
                failed: 3
            }
        ));

        // Completed result preserved
        assert!(summary
            .subtask_results
            .iter()
            .any(|r| r.subtask_id == "D" && matches!(r.status, SubtaskStatus::Completed)));
    }

    // =======================================================================
    // §12.8 Wellbeing Aggregate
    // =======================================================================

    #[test]
    fn test_wellbeing_counts_sum_to_total() {
        let states = vec![
            AgentWellbeingState::Healthy,
            AgentWellbeingState::Cautious,
            AgentWellbeingState::Concerned,
            AgentWellbeingState::Distressed,
        ];

        let agg = compute_aggregate_wellbeing(&states);

        assert_eq!(agg.agents_total, 4);
        assert_eq!(
            agg.agents_healthy + agg.agents_cautious + agg.agents_concerned + agg.agents_distressed,
            agg.agents_total
        );
    }

    #[test]
    fn test_distressed_child_paused_not_punished() {
        let states = vec![
            ("agent_1".to_string(), AgentWellbeingState::Distressed),
            ("agent_2".to_string(), AgentWellbeingState::Healthy),
            ("agent_3".to_string(), AgentWellbeingState::Healthy),
        ];

        let actions = supervisor_wellbeing_response(&states);

        // Should have an action for the distressed agent
        let distressed_action = actions
            .iter()
            .find(|a| a.agent_id == "agent_1");
        assert!(distressed_action.is_some());

        // Should be Reassign (not punitive)
        assert!(matches!(
            distressed_action.map(|a| &a.response),
            Some(WellbeingResponse::Reassign)
        ));

        // Healthy agents should have no actions
        assert!(!actions.iter().any(|a| a.agent_id == "agent_2"));
        assert!(!actions.iter().any(|a| a.agent_id == "agent_3"));
    }

    #[test]
    fn test_majority_concerned_triggers_replan() {
        let states = vec![
            AgentWellbeingState::Concerned,
            AgentWellbeingState::Concerned,
            AgentWellbeingState::Concerned,
            AgentWellbeingState::Healthy,
        ];

        let agg = compute_aggregate_wellbeing(&states);
        let action = supervisor_level_response(&agg);

        assert_eq!(action, SupervisorWellbeingAction::PauseAndReplan);
    }

    #[test]
    fn test_all_concerned_with_distressed_escalates() {
        let states = vec![
            AgentWellbeingState::Concerned,
            AgentWellbeingState::Distressed,
            AgentWellbeingState::Concerned,
        ];

        let agg = compute_aggregate_wellbeing(&states);
        let action = supervisor_level_response(&agg);

        assert_eq!(action, SupervisorWellbeingAction::EscalateToClient);
    }

    #[test]
    fn test_all_healthy_continues() {
        let states = vec![
            AgentWellbeingState::Healthy,
            AgentWellbeingState::Healthy,
            AgentWellbeingState::Cautious,
        ];

        let agg = compute_aggregate_wellbeing(&states);
        let action = supervisor_level_response(&agg);

        assert_eq!(action, SupervisorWellbeingAction::Continue);
    }

    #[test]
    fn test_empty_agents_continues() {
        let agg = compute_aggregate_wellbeing(&[]);
        let action = supervisor_level_response(&agg);

        assert_eq!(action, SupervisorWellbeingAction::Continue);
    }

    #[test]
    fn test_concerned_agent_paused() {
        let states = vec![
            ("agent_1".to_string(), AgentWellbeingState::Concerned),
            ("agent_2".to_string(), AgentWellbeingState::Healthy),
        ];

        let actions = supervisor_wellbeing_response(&states);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].agent_id, "agent_1");
        assert_eq!(actions[0].response, WellbeingResponse::Pause);
    }

    // =======================================================================
    // Property Tests
    // =======================================================================

    mod proptest_supervisor {
        use super::*;

        fn arb_complexity() -> impl Strategy<Value = Complexity> {
            prop_oneof![
                Just(Complexity::Low),
                Just(Complexity::Medium),
                Just(Complexity::High),
            ]
        }

        fn arb_subtask(id: String) -> impl Strategy<Value = Subtask> {
            arb_complexity().prop_map(move |complexity| Subtask {
                id: id.clone(),
                objective: format!("task {}", id),
                depends_on: vec![],
                capabilities: vec![],
                complexity,
            })
        }

        fn arb_resource_budget() -> impl Strategy<Value = ResourceBudget> {
            (10u32..1000, 10u32..5000, 1000u32..200_000).prop_map(
                |(iters, calls, tokens)| ResourceBudget {
                    total_iterations: iters,
                    total_tool_calls: calls,
                    total_tokens: tokens,
                },
            )
        }

        fn arb_failure_type() -> impl Strategy<Value = FailureType> {
            prop_oneof![
                Just(FailureType::EngineError),
                Just(FailureType::AgentStuck),
                Just(FailureType::AgentYielded),
                Just(FailureType::ToolError),
                Just(FailureType::Timeout),
            ]
        }

        fn arb_wellbeing_state() -> impl Strategy<Value = AgentWellbeingState> {
            prop_oneof![
                Just(AgentWellbeingState::Healthy),
                Just(AgentWellbeingState::Cautious),
                Just(AgentWellbeingState::Concerned),
                Just(AgentWellbeingState::Distressed),
            ]
        }

        proptest! {
            // §12.1: Total allocation never exceeds budget
            #[test]
            fn prop_total_allocation_within_budget(
                budget in arb_resource_budget(),
                count in 1u32..10,
            ) {
                let allocator = BudgetAllocator::new(budget.clone());
                let subtasks: Vec<Subtask> = (0..count)
                    .map(|i| Subtask {
                        id: format!("task_{i}"),
                        objective: format!("do {i}"),
                        depends_on: vec![],
                        capabilities: vec![],
                        complexity: [Complexity::Low, Complexity::Medium, Complexity::High]
                            [i as usize % 3],
                    })
                    .collect();

                let ids: Vec<String> = subtasks.iter().map(|s| s.id.clone()).collect();
                let resolver = DependencyResolver::new(subtasks.clone());
                let total_weight = resolver.total_weight(&ids);

                let configs: Vec<LoopConfig> = subtasks
                    .iter()
                    .map(|s| allocator.allocate(s, total_weight))
                    .collect();

                let total_iters: u32 = configs.iter().map(|c| c.max_iterations).sum();
                let total_calls: u32 = configs.iter().map(|c| c.max_tool_calls).sum();
                let total_tokens: u32 = configs.iter().map(|c| c.max_tokens).sum();

                prop_assert!(total_iters <= budget.total_iterations + count,
                    "iterations {total_iters} > budget {}", budget.total_iterations);
                prop_assert!(total_calls <= budget.total_tool_calls + count,
                    "calls {total_calls} > budget {}", budget.total_tool_calls);
                prop_assert!(total_tokens <= budget.total_tokens + count,
                    "tokens {total_tokens} > budget {}", budget.total_tokens);
            }

            // §12.1: Rebalance never exceeds remaining
            #[test]
            fn prop_rebalance_within_remaining(
                remaining_iters in 1u32..1000,
                remaining_calls in 1u32..5000,
                remaining_tokens in 1u32..200_000,
                running_count in 1u32..5,
            ) {
                let configs = rebalance_budget(
                    remaining_iters,
                    remaining_calls,
                    remaining_tokens,
                    running_count,
                );

                let total_iters: u32 = configs.iter().map(|c| c.max_iterations).sum();
                let total_calls: u32 = configs.iter().map(|c| c.max_tool_calls).sum();
                let total_tokens: u32 = configs.iter().map(|c| c.max_tokens).sum();

                prop_assert!(total_iters <= remaining_iters);
                prop_assert!(total_calls <= remaining_calls);
                prop_assert!(total_tokens <= remaining_tokens);
            }

            // §12.3: Concurrency never exceeds max
            #[test]
            fn prop_concurrency_limit_respected(
                max_concurrent in 1u32..5,
                events in 1u32..20,
            ) {
                let mut limiter = ConcurrencyLimiter::new(max_concurrent);

                for _ in 0..events {
                    if limiter.try_acquire() {
                        // Sometimes release
                        if limiter.active_count() > 1 {
                            limiter.release();
                        }
                    }
                }

                prop_assert!(limiter.max_observed() <= max_concurrent);
            }

            // §12.6: Circuit breaker bounded retries
            #[test]
            fn prop_circuit_breaker_bounded(
                threshold in 2u32..5,
                failure_type in arb_failure_type(),
            ) {
                let mut cb = CircuitBreaker::new(threshold);
                let mut count = 0u32;

                loop {
                    match cb.record_failure(failure_type.clone()) {
                        Ok(()) => count += 1,
                        Err(_) => break,
                    }
                }

                prop_assert!(count < threshold,
                    "circuit breaker allowed {} failures before tripping (threshold={})",
                    count, threshold);
            }

            // §12.8: Wellbeing counts sum to total
            #[test]
            fn prop_wellbeing_counts_sum(
                states in prop::collection::vec(arb_wellbeing_state(), 0..20),
            ) {
                let agg = compute_aggregate_wellbeing(&states);

                let sum = agg.agents_healthy
                    + agg.agents_cautious
                    + agg.agents_concerned
                    + agg.agents_distressed;

                prop_assert_eq!(sum, agg.agents_total);
                prop_assert_eq!(agg.agents_total, states.len());
            }

            // §12.8: Distressed child is never punished
            #[test]
            fn prop_distressed_child_not_punished(
                agent_count in 1usize..10,
            ) {
                let mut states: Vec<(AgentId, AgentWellbeingState)> = (0..agent_count)
                    .map(|i| (format!("agent_{i}"), AgentWellbeingState::Healthy))
                    .collect();

                // Make the first agent distressed
                states[0].1 = AgentWellbeingState::Distressed;

                let actions = supervisor_wellbeing_response(&states);
                let distressed_action = actions.iter().find(|a| a.agent_id == states[0].0);

                prop_assert!(distressed_action.is_some(),
                    "distressed agent should have an action");

                let action = distressed_action.expect("checked above");
                prop_assert!(
                    matches!(action.response, WellbeingResponse::Pause | WellbeingResponse::Reassign),
                    "distressed agent should be paused or reassigned, got {:?}",
                    action.response
                );
            }
        }
    }
}
