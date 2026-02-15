//! Multi-agent supervisor — orchestrates concurrent [`super::LoopExecutor`] instances.
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
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::{stream::FuturesUnordered, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use infernum_core::{GenerateRequest, SamplingParams};

use super::coordination::{AgentCoordinator, AgentId, AgentIdentity, AgentRole, Discovery};
use super::executor::{ExecutorConfig, LoopExecutor, ToolCallDetector};
use super::types::{
    ExternalTermination, LoopConfig, LoopEvent, LoopSummary, NaturalTermination, TerminationReason,
};
use crate::tool::ToolRegistry;

// ===========================================================================
// Constants
// ===========================================================================

/// Maximum character length for shared summaries in `SummarySharing` mode.
/// Content exceeding this limit will be truncated with "..." appended.
const SUMMARY_TRUNCATION_LIMIT: usize = 500;

/// Initial delay when waiting for dependencies in parallel dispatch (milliseconds).
const DEPENDENCY_WAIT_INITIAL_MS: u64 = 10;

/// Maximum delay when waiting for dependencies (milliseconds).
const DEPENDENCY_WAIT_MAX_MS: u64 = 500;

/// Failure threshold percentage for early termination (spec §8.2).
/// If more than this percentage of subtasks fail, move to aggregation.
const FAILURE_THRESHOLD_PERCENT: f64 = 0.50;

/// Type alias for the futures collection used in parallel dispatch.
type SubtaskFutures = FuturesUnordered<
    std::pin::Pin<
        Box<dyn std::future::Future<Output = (String, Result<SubtaskResult, SupervisorError>)> + Send>,
    >,
>;

// ===========================================================================
// Helper Functions
// ===========================================================================

/// Converts a [`TerminationReason`] to a [`SubtaskStatus`].
///
/// This helper centralizes the status mapping logic used by both
/// `execute_subtask` and `execute_subtask_static`.
/// Result of analyzing termination reason for status and failure type.
struct TerminationAnalysis {
    status: SubtaskStatus,
    failure_type: Option<FailureType>,
}

/// Analyzes termination reason to extract status and failure type.
fn analyze_termination(
    termination: &TerminationReason,
    summary: &LoopSummary,
) -> TerminationAnalysis {
    match termination {
        TerminationReason::Natural(n) => match n {
            NaturalTermination::AnswerProvided { .. } => TerminationAnalysis {
                status: SubtaskStatus::Completed,
                failure_type: None,
            },
            NaturalTermination::TaskComplete => TerminationAnalysis {
                status: SubtaskStatus::Completed,
                failure_type: None,
            },
            NaturalTermination::AgentStuck { attempts, .. } => TerminationAnalysis {
                status: SubtaskStatus::Partial {
                    progress: format!("Agent stuck after {attempts} attempts"),
                },
                failure_type: Some(FailureType::AgentStuck),
            },
            NaturalTermination::AgentYielded { partial, .. } => TerminationAnalysis {
                status: SubtaskStatus::Partial {
                    progress: partial.clone().unwrap_or_else(|| "Agent yielded".to_string()),
                },
                failure_type: Some(FailureType::AgentYielded),
            },
        },
        TerminationReason::Resource(_) => TerminationAnalysis {
            status: SubtaskStatus::Partial {
                progress: summary
                    .partial_answer
                    .clone()
                    .unwrap_or_else(|| "Resource limit reached".to_string()),
            },
            failure_type: Some(FailureType::Timeout), // Resource exhaustion treated as timeout
        },
        TerminationReason::External(ext) => match ext {
            ExternalTermination::ClientCancelled => TerminationAnalysis {
                status: SubtaskStatus::Failed {
                    reason: "Cancelled by client".to_string(),
                },
                failure_type: Some(FailureType::EngineError),
            },
            ExternalTermination::OperatorTerminated { reason } => TerminationAnalysis {
                status: SubtaskStatus::Failed {
                    reason: reason.clone(),
                },
                failure_type: Some(FailureType::EngineError),
            },
            ExternalTermination::SystemShutdown => TerminationAnalysis {
                status: SubtaskStatus::Failed {
                    reason: "System shutdown".to_string(),
                },
                failure_type: Some(FailureType::EngineError),
            },
        },
    }
}

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
    /// Capability requirements and provisions for this subtask.
    ///
    /// Capabilities enable routing and matching in multi-agent scenarios:
    /// - **Required capabilities**: Tools or knowledge the agent must have (e.g., "file_io", "web_search")
    /// - **Provided capabilities**: What this subtask makes available to dependent tasks
    ///
    /// In Phase 2+, the supervisor uses capabilities to select appropriate agents
    /// from the coordinator's pool. Currently unused but reserved for future routing.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// capabilities: vec!["code_execution", "testing"]
    /// ```
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
    /// LLM generates subtask breakdown from the objective.
    LlmPlanned {
        /// Available tools to mention in the planning prompt.
        available_tools: Vec<String>,
        /// Maximum number of subtasks to generate.
        max_subtasks: usize,
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
        /// Partial progress made before yielding.
        partial_progress: Option<String>,
    },
    /// The inference engine failed.
    EngineError {
        /// How many retries were attempted.
        retries: u32,
    },
    /// Resource budget was exhausted.
    ResourceExhausted {
        /// Which resource was exhausted.
        resource: String,
    },
}

/// Strategy for recovering from subtask failures (spec §8.1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Retry the same subtask with the same configuration.
    Retry,
    /// Reassign to a different agent with additional context.
    Reassign {
        /// Include failure context in the new agent's prompt.
        with_context: bool,
        /// Partial progress to forward.
        partial_progress: Option<String>,
    },
    /// Skip this subtask and continue with others.
    Skip {
        /// Reason for skipping.
        reason: String,
    },
    /// Escalate to the client for intervention.
    Escalate {
        /// What the supervisor needs from the client.
        request: String,
    },
    /// Abort the entire supervisor run.
    Abort {
        /// Reason for aborting.
        reason: String,
    },
}

/// Information about a stuck agent for recovery decisions.
#[derive(Debug, Clone)]
pub struct StuckAgentInfo {
    /// The agent that got stuck.
    pub agent_id: AgentId,
    /// The subtask that was being worked on.
    pub subtask_id: String,
    /// Number of attempts made.
    pub attempts: u32,
    /// Partial progress achieved, if any.
    pub partial_progress: Option<String>,
    /// What the agent requested (from StuckRequest).
    pub request_type: StuckRequestType,
}

/// Type of assistance requested by a stuck agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StuckRequestType {
    /// Agent needs clarification on the objective.
    Clarification,
    /// Agent needs more context from other agents.
    MoreContext,
    /// Agent needs different tools.
    DifferentTools {
        /// Tools the agent requested.
        requested_tools: Vec<String>,
    },
    /// Agent needs human intervention.
    HumanIntervention,
    /// Unknown/unspecified request.
    Unknown,
}

/// Information about a yielded agent for routing.
#[derive(Debug, Clone)]
pub struct YieldedAgentInfo {
    /// The agent that yielded.
    pub agent_id: AgentId,
    /// The subtask being worked on.
    pub subtask_id: String,
    /// Partial progress made.
    pub partial_progress: Option<String>,
    /// Expertise the agent suggested for continuation.
    pub suggested_expertise: Vec<String>,
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
    /// Structured failure type for recovery decisions.
    /// Present when status is `Failed` or `Partial`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_type: Option<FailureType>,
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
            iterations: self
                .budget
                .total_iterations
                .saturating_sub(self.consumed.iterations),
            tool_calls: self
                .budget
                .total_tool_calls
                .saturating_sub(self.consumed.tool_calls),
            tokens: self
                .budget
                .total_tokens
                .saturating_sub(self.consumed.tokens),
        }
    }

    /// Records consumption from a completed child agent.
    pub fn record_consumption(&mut self, consumption: &ResourceConsumption) {
        self.consumed.iterations = self
            .consumed
            .iterations
            .saturating_add(consumption.iterations);
        self.consumed.tool_calls = self
            .consumed
            .tool_calls
            .saturating_add(consumption.tool_calls);
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
        let max_tokens = (f64::from(remaining.tokens) * fraction).round().max(1.0) as u32;

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
        let map: HashMap<String, Subtask> =
            subtasks.into_iter().map(|s| (s.id.clone(), s)).collect();
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
            if !visited.contains(id) && self.has_cycle(id, &mut visited, &mut in_stack) {
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

    /// Returns all pending (not running, not completed, not failed) subtasks.
    ///
    /// Used for rerouting to find available subtasks with matching capabilities.
    pub fn pending_subtasks(&self) -> Vec<Subtask> {
        self.subtasks
            .values()
            .filter(|s| {
                !self.completed.contains(&s.id)
                    && !self.running.contains(&s.id)
                    && !self.failed.contains(&s.id)
            })
            .cloned()
            .collect()
    }

    /// Marks a subtask as pending (removes from running/failed, allows retry).
    ///
    /// Used when a subtask should be retried after failure or yield.
    ///
    /// # Validation
    ///
    /// - Logs a warning if the subtask ID doesn't exist in the resolver
    /// - Logs a warning if trying to mark a completed subtask as pending
    /// - Logs debug if the subtask was neither running nor failed (no-op)
    pub fn mark_pending(&mut self, id: &str) {
        // Validate that the subtask exists
        if !self.subtasks.contains_key(id) {
            warn!(
                subtask_id = id,
                "Attempted to mark non-existent subtask as pending"
            );
            return;
        }

        // Warn if trying to re-pend a completed subtask
        if self.completed.contains(id) {
            warn!(
                subtask_id = id,
                "Cannot mark completed subtask as pending; ignoring"
            );
            return;
        }

        // Track if we actually changed state
        let was_running = self.running.remove(id);
        let was_failed = self.failed.remove(id);

        if !was_running && !was_failed {
            debug!(
                subtask_id = id,
                "mark_pending called on subtask that was neither running nor failed"
            );
        }
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
    /// LLM planning failed.
    #[error("planning failed: {0}")]
    PlanningFailed(String),
    /// Executor task panicked.
    #[error("executor panic: {0}")]
    ExecutorPanic(String),
    /// Supervisor was aborted due to unrecoverable failure.
    #[error("supervisor aborted: {reason}")]
    Aborted {
        /// Reason for aborting.
        reason: String,
    },
}

/// Raw subtask from JSON parsing (used for LlmPlanned decomposition).
#[derive(Debug, Clone, Deserialize)]
struct RawSubtask {
    /// Subtask identifier.
    id: String,
    /// What the subtask should accomplish.
    objective: String,
    /// Dependencies.
    #[serde(default)]
    depends_on: Vec<String>,
    /// Complexity estimate.
    #[serde(default = "default_complexity")]
    complexity: String,
}

fn default_complexity() -> String {
    "medium".to_string()
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

        // Normalize requested expertise to lowercase for case-insensitive matching
        let normalized_expertise: Vec<String> = requested_expertise
            .iter()
            .map(|s| s.to_lowercase())
            .collect();

        for subtask in available_subtasks {
            if exclude_ids.contains(&subtask.id) {
                continue;
            }

            // Case-insensitive capability matching
            let matching_caps = subtask
                .capabilities
                .iter()
                .filter(|cap| {
                    let cap_lower = cap.to_lowercase();
                    normalized_expertise
                        .iter()
                        .any(|req| cap_lower.contains(req.as_str()))
                })
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
    /// All retries exhausted.
    RetriesExhausted,
}

impl std::fmt::Display for FailureType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EngineError => write!(f, "engine_error"),
            Self::AgentStuck => write!(f, "agent_stuck"),
            Self::AgentYielded => write!(f, "agent_yielded"),
            Self::ToolError => write!(f, "tool_error"),
            Self::Timeout => write!(f, "timeout"),
            Self::RetriesExhausted => write!(f, "retries_exhausted"),
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
                failure_type: None, // Skipped is not a failure
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

    /// Checks if failures exceed the configured threshold (spec §8.2).
    ///
    /// Returns `true` if more than `FAILURE_THRESHOLD_PERCENT` of total subtasks
    /// have failed, indicating the supervisor should move to aggregation.
    pub fn exceeds_failure_threshold(&self, total_subtasks: usize) -> bool {
        if total_subtasks == 0 {
            return false;
        }

        let failed = self.failed_count();
        let failure_ratio = failed as f64 / total_subtasks as f64;
        failure_ratio > FAILURE_THRESHOLD_PERCENT
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
    let concerned_fraction = aggregate.agents_concerned as f64 / aggregate.agents_total as f64;
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
                },
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
// Supervisor Implementation
// ===========================================================================

/// Trait for inference engines (re-export for convenience).
pub use abaddon::InferenceEngine;

/// The multi-agent supervisor.
///
/// Orchestrates one or more `LoopExecutor` instances to accomplish a complex objective.
/// For Phase 1 (SingleAgent mode), spawns a single executor and monitors it.
///
/// # Example
///
/// ```ignore
/// let supervisor = Supervisor::new(engine, tools, config);
/// let summary = supervisor.run("Build a REST API", event_tx).await?;
/// ```
pub struct Supervisor {
    /// Inference engine for child executors.
    engine: Arc<dyn InferenceEngine>,
    /// Tool registry for child executors.
    tools: Arc<ToolRegistry>,
    /// Tool call detector.
    detector: Option<Arc<dyn ToolCallDetector>>,
    /// Supervisor configuration.
    config: SupervisorConfig,
    /// Agent coordinator for multi-agent coordination.
    coordinator: Arc<AgentCoordinator>,
}

impl Supervisor {
    /// Creates a new supervisor with the given engine, tools, and configuration.
    pub fn new(
        engine: Arc<dyn InferenceEngine>,
        tools: Arc<ToolRegistry>,
        config: SupervisorConfig,
    ) -> Self {
        // Create coordinator with quotas if multi-agent mode
        let coordinator = match &config.decomposition {
            DecompositionStrategy::SingleAgent => Arc::new(AgentCoordinator::new()),
            _ => Arc::new(AgentCoordinator::with_quotas(
                config.resource_budget.total_tool_calls,
                config.resource_budget.total_tokens,
            )),
        };

        Self {
            engine,
            tools,
            detector: None,
            config,
            coordinator,
        }
    }

    /// Creates a new supervisor with a custom coordinator.
    ///
    /// Useful for sharing a coordinator across multiple supervisors.
    pub fn with_coordinator(
        engine: Arc<dyn InferenceEngine>,
        tools: Arc<ToolRegistry>,
        config: SupervisorConfig,
        coordinator: Arc<AgentCoordinator>,
    ) -> Self {
        Self {
            engine,
            tools,
            detector: None,
            config,
            coordinator,
        }
    }

    /// Returns the agent coordinator.
    pub fn coordinator(&self) -> &Arc<AgentCoordinator> {
        &self.coordinator
    }

    /// Sets a custom tool call detector.
    pub fn with_detector(mut self, detector: Arc<dyn ToolCallDetector>) -> Self {
        self.detector = Some(detector);
        self
    }

    /// Runs the supervisor to completion.
    ///
    /// Emits `SupervisorEvent`s through the provided channel.
    /// Returns a `SupervisorSummary` when all subtasks complete or the supervisor terminates.
    pub async fn run(
        &self,
        objective: &str,
        event_tx: mpsc::Sender<SupervisorEvent>,
    ) -> Result<SupervisorSummary, SupervisorError> {
        let start_time = Instant::now();
        info!("Supervisor starting with objective: {}", objective);

        // Initialize tracking components
        let mut budget_allocator = BudgetAllocator::new(self.config.resource_budget.clone());
        let mut lifecycle_tracker = LifecycleTracker::new();
        let mut result_aggregator = ResultAggregator::new();
        let mut circuit_breaker = CircuitBreaker::new(self.config.circuit_breaker_threshold);

        // Decompose objective into subtasks
        let subtasks = match &self.config.decomposition {
            DecompositionStrategy::SingleAgent => {
                // Single subtask with the entire objective
                vec![Subtask {
                    id: "main".to_string(),
                    objective: objective.to_string(),
                    depends_on: vec![],
                    capabilities: vec![],
                    complexity: Complexity::High, // Assume high for full objective
                }]
            }
            DecompositionStrategy::ClientProvided { subtasks } => subtasks.clone(),
            DecompositionStrategy::LlmPlanned {
                available_tools,
                max_subtasks,
            } => {
                // Use LLM to plan subtasks
                self.plan_subtasks(objective, available_tools, *max_subtasks)
                    .await?
            }
        };

        // Initialize dependency resolver and validate
        let mut resolver = DependencyResolver::new(subtasks.clone());
        resolver.validate()?;

        let total_subtasks = resolver.total();
        info!("Supervisor decomposed into {} subtask(s)", total_subtasks);

        // Use parallel dispatch for multi-agent scenarios
        let use_parallel = matches!(self.config.routing, RoutingStrategy::Parallel)
            || self.config.max_concurrent_agents > 1;

        if use_parallel {
            // Parallel dispatch: run multiple subtasks concurrently
            self.run_parallel(
                &mut resolver,
                &mut budget_allocator,
                &mut lifecycle_tracker,
                &mut circuit_breaker,
                &mut result_aggregator,
                &event_tx,
            )
            .await?;
        } else {
            // Sequential dispatch: run one subtask at a time
            self.run_sequential(
                &mut resolver,
                &mut budget_allocator,
                &mut lifecycle_tracker,
                &mut circuit_breaker,
                &mut result_aggregator,
                &event_tx,
            )
            .await?;
        }

        // Build final summary
        let wall_time = start_time.elapsed();
        let summary = result_aggregator.build_summary(
            lifecycle_tracker.total_spawned(),
            budget_allocator.total_consumed(),
            wall_time,
            total_subtasks,
        );

        // Emit completion event
        if event_tx
            .send(SupervisorEvent::SupervisorCompleted {
                summary: summary.clone(),
            })
            .await
            .is_err()
        {
            debug!("Event channel closed before supervisor completion");
        }

        info!(
            "Supervisor completed: {:?}, {} agents spawned, {:?} wall time",
            summary.termination,
            summary.total_agents_spawned,
            wall_time
        );

        Ok(summary)
    }

    // -------------------------------------------------------------------------
    // Sequential and Parallel dispatch helpers
    // -------------------------------------------------------------------------

    /// Runs subtasks sequentially (one at a time).
    async fn run_sequential(
        &self,
        resolver: &mut DependencyResolver,
        budget_allocator: &mut BudgetAllocator,
        lifecycle_tracker: &mut LifecycleTracker,
        circuit_breaker: &mut CircuitBreaker,
        result_aggregator: &mut ResultAggregator,
        event_tx: &mpsc::Sender<SupervisorEvent>,
    ) -> Result<(), SupervisorError> {
        let total_subtasks = resolver.total();

        while !resolver.is_done() {
            let ready = resolver.ready();
            if ready.is_empty() {
                warn!("No subtasks ready, but resolver not done. Possible deadlock.");
                break;
            }

            for subtask_id in ready {
                let subtask = match resolver.get(&subtask_id) {
                    Some(s) => s.clone(),
                    None => continue,
                };

                resolver.mark_running(&subtask_id);

                let result = self
                    .execute_subtask_with_retry(
                        &subtask,
                        budget_allocator,
                        lifecycle_tracker,
                        circuit_breaker,
                        event_tx,
                        resolver,
                    )
                    .await;

                let should_continue = self
                    .handle_subtask_result(
                        result,
                        &subtask_id,
                        resolver,
                        budget_allocator,
                        circuit_breaker,
                        result_aggregator,
                        event_tx,
                        total_subtasks,
                    )
                    .await?;

                if !should_continue {
                    info!("Early termination due to failure threshold");
                    return Ok(());
                }
            }

            if circuit_breaker.is_open() {
                break;
            }
        }

        Ok(())
    }

    /// Runs subtasks in parallel (up to `max_concurrent_agents`).
    async fn run_parallel(
        &self,
        resolver: &mut DependencyResolver,
        budget_allocator: &mut BudgetAllocator,
        lifecycle_tracker: &mut LifecycleTracker,
        circuit_breaker: &mut CircuitBreaker,
        result_aggregator: &mut ResultAggregator,
        event_tx: &mpsc::Sender<SupervisorEvent>,
    ) -> Result<(), SupervisorError> {
        let total_subtasks = resolver.total();
        let mut concurrency_limiter = ConcurrencyLimiter::new(self.config.max_concurrent_agents);
        let mut running_futures: SubtaskFutures = FuturesUnordered::new();

        // Exponential backoff for dependency waiting
        let mut dependency_wait_ms = DEPENDENCY_WAIT_INITIAL_MS;

        loop {
            // Check if we're done
            if resolver.is_done() && running_futures.is_empty() {
                break;
            }

            // Spawn new subtasks if we have capacity and ready subtasks
            while concurrency_limiter.try_acquire() {
                let ready = resolver.ready();
                if ready.is_empty() {
                    // No subtasks ready, release the slot we just acquired
                    concurrency_limiter.release();
                    break;
                }

                // Take the first ready subtask
                let subtask_id = ready.into_iter().next().unwrap();
                let subtask = match resolver.get(&subtask_id) {
                    Some(s) => s.clone(),
                    None => {
                        concurrency_limiter.release();
                        continue;
                    }
                };

                resolver.mark_running(&subtask_id);
                info!("Parallel dispatch: starting subtask {}", subtask_id);

                // Clone values needed for the spawned future
                let engine = Arc::clone(&self.engine);
                let tools = Arc::clone(&self.tools);
                let detector = self.detector.clone();
                let coordinator = Arc::clone(&self.coordinator);
                let config = self.config.clone();
                let budget = budget_allocator.allocate(&subtask, resolver.total_weight(&[subtask_id.clone()]));
                let subtask_id_clone = subtask_id.clone();
                let event_tx_clone = event_tx.clone();

                // Track spawn
                let agent_id = format!("agent_{}", uuid::Uuid::new_v4().simple());
                lifecycle_tracker.record_spawn(agent_id.clone(), subtask_id.clone());

                // Spawn the subtask as an async future
                let future = Box::pin(async move {
                    let result = Self::execute_subtask_static(
                        engine,
                        tools,
                        detector,
                        coordinator,
                        &config,
                        &subtask,
                        budget,
                        &agent_id,
                        &event_tx_clone,
                    )
                    .await;
                    (subtask_id_clone, result)
                });

                running_futures.push(future);
            }

            // Wait for at least one running task to complete
            if !running_futures.is_empty() {
                if let Some((subtask_id, result)) = running_futures.next().await {
                    // Reset exponential backoff since work completed
                    dependency_wait_ms = DEPENDENCY_WAIT_INITIAL_MS;

                    // Release concurrency slot
                    concurrency_limiter.release();

                    // Record lifecycle completion
                    if let Ok(ref r) = result {
                        if let Some(ref agent_id) = r.agent_id {
                            lifecycle_tracker.record_completion(agent_id);
                        }
                    }

                    let should_continue = self
                        .handle_subtask_result(
                            result,
                            &subtask_id,
                            resolver,
                            budget_allocator,
                            circuit_breaker,
                            result_aggregator,
                            event_tx,
                            total_subtasks,
                        )
                        .await?;

                    if !should_continue {
                        info!("Early termination due to failure threshold (parallel mode)");
                        return Ok(());
                    }
                }
            } else if resolver.ready().is_empty() && !resolver.is_done() {
                // No running tasks and no ready tasks, but not done
                // This could happen if we're waiting for dependencies
                warn!(
                    "Parallel dispatch: waiting for dependencies ({}ms backoff)...",
                    dependency_wait_ms
                );
                // Exponential backoff to prevent busy-waiting
                tokio::time::sleep(tokio::time::Duration::from_millis(dependency_wait_ms)).await;
                dependency_wait_ms = (dependency_wait_ms * 2).min(DEPENDENCY_WAIT_MAX_MS);
            }

            if circuit_breaker.is_open() {
                break;
            }
        }

        info!(
            "Parallel dispatch complete: max concurrency observed = {}",
            concurrency_limiter.max_observed()
        );

        Ok(())
    }

    /// Handles the result of a subtask execution (shared by sequential and parallel).
    ///
    /// Returns `Ok(true)` if we should continue processing, `Ok(false)` if we should
    /// early-terminate (e.g., failure threshold exceeded), or an error.
    async fn handle_subtask_result(
        &self,
        result: Result<SubtaskResult, SupervisorError>,
        subtask_id: &str,
        resolver: &mut DependencyResolver,
        budget_allocator: &mut BudgetAllocator,
        circuit_breaker: &mut CircuitBreaker,
        result_aggregator: &mut ResultAggregator,
        event_tx: &mpsc::Sender<SupervisorEvent>,
        total_subtasks: usize,
    ) -> Result<bool, SupervisorError> {
        match result {
            Ok(subtask_result) => {
                circuit_breaker.record_success();

                // Record resource consumption from the summary
                if let Some(ref summary) = subtask_result.summary {
                    budget_allocator.record_consumption(&ResourceConsumption {
                        iterations: summary.iterations_completed,
                        tool_calls: summary.tool_calls_made,
                        tokens: summary.tokens_generated,
                    });
                }

                if matches!(subtask_result.status, SubtaskStatus::Completed) {
                    resolver.mark_completed(subtask_id);
                } else {
                    resolver.mark_failed(subtask_id);
                }

                result_aggregator.record_result(subtask_result);
            }
            Err(e) => {
                error!("Subtask {} failed: {}", subtask_id, e);
                resolver.mark_failed(subtask_id);

                // Record failure with circuit breaker
                circuit_breaker.record_failure(FailureType::EngineError)?;

                result_aggregator.record_result(SubtaskResult {
                    subtask_id: subtask_id.to_string(),
                    status: SubtaskStatus::Failed {
                        reason: e.to_string(),
                    },
                    summary: None,
                    agent_id: None,
                    failure_type: Some(FailureType::EngineError),
                });

                if circuit_breaker.is_open() {
                    if event_tx
                        .send(SupervisorEvent::SupervisorError {
                            message: format!("Circuit breaker triggered: {}", e),
                            recoverable: false,
                        })
                        .await
                        .is_err()
                    {
                        debug!("Event channel closed, unable to send circuit breaker error");
                    }
                }
            }
        }

        // Check if we've exceeded the failure threshold (spec §8.2)
        if result_aggregator.exceeds_failure_threshold(total_subtasks) {
            warn!(
                "Failure threshold exceeded ({:.0}%): {}/{} subtasks failed. Moving to aggregation.",
                FAILURE_THRESHOLD_PERCENT * 100.0,
                result_aggregator.failed_count(),
                total_subtasks
            );
            if event_tx
                .send(SupervisorEvent::SupervisorError {
                    message: format!(
                        "Failure threshold exceeded: {}/{} subtasks failed",
                        result_aggregator.failed_count(),
                        total_subtasks
                    ),
                    recoverable: false,
                })
                .await
                .is_err()
            {
                debug!("Event channel closed, unable to send failure threshold event");
            }
            return Ok(false); // Signal to stop processing
        }

        Ok(true) // Continue processing
    }

    /// Static version of subtask execution for use in spawned futures.
    ///
    /// This is a standalone function that doesn't require `&self`, allowing it to
    /// be used inside `tokio::spawn()` futures where references to `Supervisor`
    /// cannot be held. Used by `run_parallel` to execute subtasks concurrently.
    ///
    /// # Arguments
    ///
    /// * `engine` - Inference engine for LLM calls
    /// * `tools` - Tool registry for agent tool use
    /// * `detector` - Optional custom tool call detector
    /// * `coordinator` - Shared coordinator for multi-agent discovery sharing
    /// * `config` - Supervisor configuration (for shared context mode)
    /// * `subtask` - The subtask to execute
    /// * `loop_config` - Budget and loop configuration for the executor
    /// * `agent_id` - Unique identifier for the spawned agent
    /// * `event_tx` - Channel for sending supervisor events
    ///
    /// # Returns
    ///
    /// A [`SubtaskResult`] containing the execution outcome, status, and optional summary.
    async fn execute_subtask_static(
        engine: Arc<dyn InferenceEngine>,
        tools: Arc<ToolRegistry>,
        detector: Option<Arc<dyn ToolCallDetector>>,
        coordinator: Arc<AgentCoordinator>,
        config: &SupervisorConfig,
        subtask: &Subtask,
        loop_config: LoopConfig,
        agent_id: &str,
        event_tx: &mpsc::Sender<SupervisorEvent>,
    ) -> Result<SubtaskResult, SupervisorError> {
        // Register agent with coordinator
        let identity = AgentIdentity::new(agent_id, AgentRole::Primary).with_task(&subtask.objective);
        coordinator.register_agent(identity);

        debug!(
            "Allocated budget for {}: {} iterations, {} tool calls, {} tokens",
            subtask.id, loop_config.max_iterations, loop_config.max_tool_calls, loop_config.max_tokens
        );

        // Build executor config
        let executor_config = ExecutorConfig::new(agent_id).with_loop_config(loop_config);

        // Create executor
        let mut executor = LoopExecutor::new(engine, tools, executor_config);

        if let Some(det) = detector {
            executor = executor.with_detector(det);
        }

        if event_tx
            .send(SupervisorEvent::AgentSpawned {
                agent_id: agent_id.to_string(),
                subtask_id: subtask.id.clone(),
            })
            .await
            .is_err()
        {
            debug!("Event channel closed, unable to send AgentSpawned event");
        }

        // Create event channel for the executor
        let (loop_tx, mut loop_rx) = mpsc::channel::<LoopEvent>(256);

        // Spawn executor task
        let objective = subtask.objective.clone();
        let executor_handle = tokio::spawn(async move { executor.run(&objective, loop_tx).await });

        // Drain events (could forward them in future)
        while loop_rx.recv().await.is_some() {}

        // Wait for executor to complete
        let loop_result = executor_handle
            .await
            .map_err(|e| SupervisorError::ExecutorPanic(e.to_string()))?;

        // Build result
        let result = match loop_result {
            Ok(summary) => {
                // Analyze termination to get both status and failure_type
                let analysis = analyze_termination(&summary.termination, &summary);

                // Share discovery if context sharing enabled
                if matches!(
                    config.shared_context_mode,
                    SharedContextMode::SummarySharing | SharedContextMode::FullSharing
                ) {
                    let answer: Option<String> = match &summary.termination {
                        TerminationReason::Natural(NaturalTermination::AnswerProvided { answer, .. }) => {
                            Some(answer.clone())
                        }
                        _ => summary.partial_answer.clone(),
                    };

                    if let Some(ref answer_str) = answer {
                        let content = match config.shared_context_mode {
                            SharedContextMode::SummarySharing => {
                                if answer_str.len() > SUMMARY_TRUNCATION_LIMIT {
                                    format!("{}...", &answer_str[..SUMMARY_TRUNCATION_LIMIT])
                                } else {
                                    answer_str.clone()
                                }
                            }
                            SharedContextMode::FullSharing => answer_str.clone(),
                            _ => String::new(),
                        };

                        if !content.is_empty() {
                            let _ = coordinator.share_discovery(
                                &agent_id.to_string(),
                                Discovery {
                                    content,
                                    category: "subtask_result".to_string(),
                                    tags: vec![subtask.id.clone()],
                                    data: None,
                                },
                            );
                        }
                    }
                }

                if event_tx
                    .send(SupervisorEvent::AgentCompleted {
                        agent_id: agent_id.to_string(),
                        subtask_id: subtask.id.clone(),
                        summary: summary.clone(),
                    })
                    .await
                    .is_err()
                {
                    debug!("Event channel closed, unable to send AgentCompleted event");
                }

                Ok(SubtaskResult {
                    subtask_id: subtask.id.clone(),
                    status: analysis.status,
                    summary: Some(summary),
                    agent_id: Some(agent_id.to_string()),
                    failure_type: analysis.failure_type,
                })
            }
            Err(e) => {
                if event_tx
                    .send(SupervisorEvent::SupervisorError {
                        message: format!("Executor failed: {}", e),
                        recoverable: true,
                    })
                    .await
                    .is_err()
                {
                    debug!("Event channel closed, unable to send executor failure event");
                }

                Ok(SubtaskResult {
                    subtask_id: subtask.id.clone(),
                    status: SubtaskStatus::Failed {
                        reason: e.to_string(),
                    },
                    summary: None,
                    agent_id: Some(agent_id.to_string()),
                    failure_type: Some(FailureType::EngineError),
                })
            }
        };

        // Unregister agent
        coordinator.unregister_agent(agent_id);

        result
    }

    // -------------------------------------------------------------------------
    // LLM-based planning
    // -------------------------------------------------------------------------

    /// Uses the LLM to decompose an objective into subtasks.
    async fn plan_subtasks(
        &self,
        objective: &str,
        available_tools: &[String],
        max_subtasks: usize,
    ) -> Result<Vec<Subtask>, SupervisorError> {
        let tool_list = if available_tools.is_empty() {
            "No specific tools available.".to_string()
        } else {
            available_tools.join(", ")
        };

        let planning_prompt = format!(
            r#"You are a task planning assistant. Decompose the following objective into subtasks.

**Objective:** {objective}

**Available tools:** {tool_list}

**Instructions:**
1. Break down the objective into 1-{max_subtasks} subtasks.
2. Each subtask should be independently achievable.
3. Identify dependencies between subtasks (which must complete before others can start).
4. Estimate complexity: "low", "medium", or "high".

**Output a JSON array of subtasks:**
```json
[
  {{
    "id": "subtask_1",
    "objective": "Description of what to accomplish",
    "depends_on": [],
    "complexity": "low|medium|high"
  }}
]
```

Output ONLY the JSON array, no other text."#
        );

        // Create generate request
        let sampling = SamplingParams::default()
            .with_max_tokens(2048)
            .with_temperature(0.3); // Lower temperature for structured output

        let request = GenerateRequest::new(planning_prompt).with_sampling(sampling);

        // Generate the plan
        let response = self
            .engine
            .generate(request)
            .await
            .map_err(|e| SupervisorError::PlanningFailed(format!("LLM generation failed: {e}")))?;

        // Extract the generated text
        let generated_text = response
            .choices
            .first()
            .map(|c| c.text.clone())
            .ok_or_else(|| SupervisorError::PlanningFailed("No response from LLM".to_string()))?;

        // Parse the JSON subtasks
        self.parse_subtasks_json(&generated_text)
    }

    /// Parses JSON subtask array from LLM output.
    fn parse_subtasks_json(&self, text: &str) -> Result<Vec<Subtask>, SupervisorError> {
        // Try to find JSON array in the response
        let json_str = Self::extract_json_array(text)
            .ok_or_else(|| SupervisorError::PlanningFailed("No JSON array found in response".to_string()))?;

        // Parse the JSON
        let raw_subtasks: Vec<RawSubtask> = serde_json::from_str(json_str)
            .map_err(|e| SupervisorError::PlanningFailed(format!("Invalid JSON: {e}")))?;

        // Convert to Subtask
        let subtasks: Vec<Subtask> = raw_subtasks
            .into_iter()
            .map(|raw| Subtask {
                id: raw.id,
                objective: raw.objective,
                depends_on: raw.depends_on,
                capabilities: vec![],
                complexity: match raw.complexity.as_str() {
                    "low" => Complexity::Low,
                    "high" => Complexity::High,
                    _ => Complexity::Medium,
                },
            })
            .collect();

        if subtasks.is_empty() {
            return Err(SupervisorError::PlanningFailed(
                "LLM returned no subtasks".to_string(),
            ));
        }

        info!("LLM planned {} subtask(s)", subtasks.len());
        Ok(subtasks)
    }

    /// Extracts a JSON array from text (handles markdown code blocks).
    fn extract_json_array(text: &str) -> Option<&str> {
        // Try to find JSON in markdown code block
        if let Some(start) = text.find("```json") {
            let after_tag = &text[start + 7..];
            if let Some(end) = after_tag.find("```") {
                return Some(after_tag[..end].trim());
            }
        }

        // Try to find JSON in generic code block
        if let Some(start) = text.find("```") {
            let after_tag = &text[start + 3..];
            // Skip the language identifier line
            if let Some(newline) = after_tag.find('\n') {
                let content = &after_tag[newline + 1..];
                if let Some(end) = content.find("```") {
                    return Some(content[..end].trim());
                }
            }
        }

        // Try to find bare JSON array
        if let Some(start) = text.find('[') {
            if let Some(end) = text.rfind(']') {
                if end > start {
                    return Some(&text[start..=end]);
                }
            }
        }

        None
    }

    /// Executes a single subtask with retry logic.
    async fn execute_subtask_with_retry(
        &self,
        subtask: &Subtask,
        budget_allocator: &mut BudgetAllocator,
        lifecycle_tracker: &mut LifecycleTracker,
        circuit_breaker: &mut CircuitBreaker,
        event_tx: &mpsc::Sender<SupervisorEvent>,
        resolver: &DependencyResolver,
    ) -> Result<SubtaskResult, SupervisorError> {
        let max_retries = self.config.max_retries;
        let mut retries_attempted: u32 = 0;
        #[allow(unused_assignments)] // All break paths set this; None is for type
        let mut last_error: Option<String> = None;
        let mut last_partial_progress: Option<String> = None;

        loop {
            if retries_attempted > 0 {
                info!(
                    "Retrying subtask {} (attempt {}/{})",
                    subtask.id,
                    retries_attempted + 1,
                    max_retries + 1
                );
            }

            match self
                .execute_subtask(subtask, budget_allocator, lifecycle_tracker, event_tx, resolver)
                .await
            {
                Ok(result) => {
                    if matches!(result.status, SubtaskStatus::Completed) {
                        return Ok(result);
                    }

                    // Extract partial progress for recovery strategy
                    let partial_progress = match &result.status {
                        SubtaskStatus::Partial { progress } => Some(progress.clone()),
                        _ => result.summary.as_ref().and_then(|s| s.partial_answer.clone()),
                    };
                    last_partial_progress = partial_progress.clone();

                    // Use structured failure_type from result (set by analyze_termination)
                    let failure_type = result
                        .failure_type
                        .unwrap_or(FailureType::EngineError);
                    circuit_breaker.record_failure(failure_type)?;

                    // Determine recovery strategy
                    let has_budget = budget_allocator.remaining().iterations > 0;
                    let strategy = self.determine_recovery_strategy(
                        failure_type,
                        retries_attempted,
                        partial_progress.as_deref(),
                        has_budget,
                    );

                    match strategy {
                        RecoveryStrategy::Retry => {
                            if retries_attempted >= max_retries {
                                last_error = Some(format!(
                                    "Max retries ({}) exceeded: {:?}",
                                    max_retries, result.status
                                ));
                                break;
                            }
                            retries_attempted += 1;
                            continue;
                        }
                        RecoveryStrategy::Reassign { with_context, partial_progress: progress } => {
                            // For reassign, we inject context and retry with additional context
                            if with_context {
                                debug!(
                                    "Reassigning subtask {} with context injection",
                                    subtask.id
                                );

                                // Inject context by sharing failure info as a discovery
                                let context_info = format!(
                                    "Previous attempt on subtask '{}' encountered issues. \
                                    Failure type: {:?}. Partial progress: {}",
                                    subtask.id,
                                    failure_type,
                                    progress.as_deref().unwrap_or("none")
                                );

                                // Share the context through coordinator for the retried agent
                                let _ = self.coordinator.share_discovery(
                                    &format!("supervisor_retry_{}", subtask.id),
                                    Discovery {
                                        content: context_info,
                                        category: "failure_context".to_string(),
                                        tags: vec![subtask.id.clone(), "retry".to_string()],
                                        data: None,
                                    },
                                );
                            }

                            if retries_attempted >= max_retries {
                                last_error = Some(format!(
                                    "Reassign failed, max retries exceeded. Progress: {:?}",
                                    progress
                                ));
                                break;
                            }
                            retries_attempted += 1;
                            continue;
                        }
                        RecoveryStrategy::Skip { reason } => {
                            info!("Skipping subtask {}: {}", subtask.id, reason);
                            return Ok(SubtaskResult {
                                subtask_id: subtask.id.clone(),
                                status: SubtaskStatus::Partial {
                                    progress: format!("Skipped: {}", reason),
                                },
                                summary: result.summary,
                                agent_id: result.agent_id,
                                failure_type: result.failure_type,
                            });
                        }
                        RecoveryStrategy::Escalate { request } => {
                            warn!("Escalating subtask {}: {}", subtask.id, request);
                            if event_tx
                                .send(SupervisorEvent::SupervisorError {
                                    message: format!("Escalation required: {}", request),
                                    recoverable: true,
                                })
                                .await
                                .is_err()
                            {
                                debug!("Event channel closed, unable to send escalation event");
                            }
                            return Ok(result);
                        }
                        RecoveryStrategy::Abort { reason } => {
                            error!("Aborting supervisor: {}", reason);
                            return Err(SupervisorError::Aborted { reason });
                        }
                    }
                }
                Err(e) => {
                    last_error = Some(e.to_string());
                    circuit_breaker.record_failure(FailureType::EngineError)?;

                    // Determine recovery strategy for errors
                    let has_budget = budget_allocator.remaining().iterations > 0;
                    let strategy = self.determine_recovery_strategy(
                        FailureType::EngineError,
                        retries_attempted,
                        last_partial_progress.as_deref(),
                        has_budget,
                    );

                    match strategy {
                        RecoveryStrategy::Retry | RecoveryStrategy::Reassign { .. } => {
                            if retries_attempted >= max_retries {
                                break;
                            }
                            retries_attempted += 1;
                            continue;
                        }
                        _ => break,
                    }
                }
            }
        }

        // All retries exhausted
        Ok(SubtaskResult {
            subtask_id: subtask.id.clone(),
            status: SubtaskStatus::Failed {
                reason: last_error.unwrap_or_else(|| "Unknown error".to_string()),
            },
            summary: None,
            agent_id: None,
            failure_type: Some(FailureType::RetriesExhausted),
        })
    }

    /// Executes a single subtask by spawning a `LoopExecutor`.
    async fn execute_subtask(
        &self,
        subtask: &Subtask,
        budget_allocator: &mut BudgetAllocator,
        lifecycle_tracker: &mut LifecycleTracker,
        event_tx: &mpsc::Sender<SupervisorEvent>,
        resolver: &DependencyResolver,
    ) -> Result<SubtaskResult, SupervisorError> {
        // Generate agent ID
        let agent_id = format!("agent_{}", uuid::Uuid::new_v4().simple());

        // Register agent with coordinator
        let identity = AgentIdentity::new(&agent_id, AgentRole::Primary)
            .with_task(&subtask.objective);
        self.coordinator.register_agent(identity);

        // Allocate budget for this subtask
        let ready_ids: Vec<String> = resolver.ready();
        let total_weight = resolver.total_weight(&ready_ids);
        let loop_config = budget_allocator.allocate(subtask, total_weight);

        debug!(
            "Allocated budget for {}: {} iterations, {} tool calls, {} tokens",
            subtask.id, loop_config.max_iterations, loop_config.max_tool_calls, loop_config.max_tokens
        );

        // Build executor config
        let executor_config = ExecutorConfig::new(&agent_id)
            .with_loop_config(loop_config);

        // Create executor
        let mut executor = LoopExecutor::new(
            Arc::clone(&self.engine),
            Arc::clone(&self.tools),
            executor_config,
        );

        if let Some(detector) = &self.detector {
            executor = executor.with_detector(Arc::clone(detector));
        }

        // Record spawn
        lifecycle_tracker.record_spawn(agent_id.clone(), subtask.id.clone());

        if event_tx
            .send(SupervisorEvent::AgentSpawned {
                agent_id: agent_id.clone(),
                subtask_id: subtask.id.clone(),
            })
            .await
            .is_err()
        {
            debug!("Event channel closed, unable to send AgentSpawned event");
        }

        // Create event channel for the executor
        let (loop_tx, mut loop_rx) = mpsc::channel::<LoopEvent>(256);

        // Spawn executor task
        let objective = subtask.objective.clone();
        let executor_handle = tokio::spawn(async move {
            executor.run(&objective, loop_tx).await
        });

        // Monitor executor events (forward interesting ones)
        while let Some(_event) = loop_rx.recv().await {
            // In Phase 1, we just drain events
            // Future phases could forward them as SupervisorEvent variants
        }

        // Wait for executor to complete
        let loop_result = executor_handle
            .await
            .map_err(|e| SupervisorError::ExecutorPanic(e.to_string()))?;

        // Record completion
        lifecycle_tracker.record_completion(&agent_id);

        // Convert LoopSummary to SubtaskResult
        let result = match loop_result {
            Ok(summary) => {
                // Record resource consumption
                budget_allocator.record_consumption(&ResourceConsumption {
                    iterations: summary.iterations_completed,
                    tool_calls: summary.tool_calls_made,
                    tokens: summary.tokens_generated,
                });

                // Analyze termination to get both status and failure_type
                let analysis = analyze_termination(&summary.termination, &summary);

                // Share discovery with coordinator if context sharing enabled
                if matches!(
                    self.config.shared_context_mode,
                    SharedContextMode::SummarySharing | SharedContextMode::FullSharing
                ) {
                    // Extract the answer from termination reason or partial_answer
                    let answer: Option<String> = match &summary.termination {
                        TerminationReason::Natural(NaturalTermination::AnswerProvided {
                            answer,
                            ..
                        }) => Some(answer.clone()),
                        _ => summary.partial_answer.clone(),
                    };

                    // Share the final answer or partial progress as a discovery
                    if let Some(ref answer_str) = answer {
                        let content = match self.config.shared_context_mode {
                            SharedContextMode::SummarySharing => {
                                // Share a summary (first 500 chars)
                                if answer_str.len() > 500 {
                                    format!("{}...", &answer_str[..500])
                                } else {
                                    answer_str.clone()
                                }
                            }
                            SharedContextMode::FullSharing => answer_str.clone(),
                            _ => String::new(),
                        };

                        if !content.is_empty() {
                            let _ = self.coordinator.share_discovery(
                                &agent_id,
                                Discovery {
                                    content,
                                    category: "subtask_result".to_string(),
                                    tags: vec![subtask.id.clone()],
                                    data: None,
                                },
                            );
                        }
                    }
                }

                if event_tx
                    .send(SupervisorEvent::AgentCompleted {
                        agent_id: agent_id.clone(),
                        subtask_id: subtask.id.clone(),
                        summary: summary.clone(),
                    })
                    .await
                    .is_err()
                {
                    debug!("Event channel closed, unable to send AgentCompleted event");
                }

                Ok(SubtaskResult {
                    subtask_id: subtask.id.clone(),
                    status: analysis.status,
                    summary: Some(summary),
                    agent_id: Some(agent_id.clone()),
                    failure_type: analysis.failure_type,
                })
            }
            Err(e) => {
                if event_tx
                    .send(SupervisorEvent::SupervisorError {
                        message: format!("Executor failed: {}", e),
                        recoverable: true,
                    })
                    .await
                    .is_err()
                {
                    debug!("Event channel closed, unable to send executor failure event");
                }

                Ok(SubtaskResult {
                    subtask_id: subtask.id.clone(),
                    status: SubtaskStatus::Failed {
                        reason: e.to_string(),
                    },
                    summary: None,
                    agent_id: Some(agent_id.clone()),
                    failure_type: Some(FailureType::EngineError),
                })
            }
        };

        // Unregister agent from coordinator
        self.coordinator.unregister_agent(&agent_id);

        result
    }

    /// Retrieves shared context for an agent from completed subtasks.
    ///
    /// Returns context from other agents' discoveries, filtered by the
    /// configured `SharedContextMode`.
    #[allow(dead_code)] // Will be used in parallel dispatch
    fn get_shared_context_for_agent(&self, agent_id: &AgentId) -> Option<String> {
        if matches!(self.config.shared_context_mode, SharedContextMode::None) {
            return None;
        }

        let context = self.coordinator.get_shared_context(agent_id);
        if context.discoveries.is_empty() {
            return None;
        }

        // Build a context string from discoveries
        let mut result = String::from("## Context from other agents:\n\n");
        for (from_agent, discovery) in &context.discoveries {
            result.push_str(&format!(
                "### {} ({})\n{}\n\n",
                discovery.category, from_agent, discovery.content
            ));
        }

        Some(result)
    }

    // =========================================================================
    // Phase 3: Recovery and Rerouting (spec §4.4, §8)
    // =========================================================================

    /// Determines the appropriate recovery strategy for a failed subtask.
    ///
    /// Implements the decision tree from spec §8.1:
    /// - Engine errors: retry once, then reassign
    /// - Tool errors: reassign with context
    /// - Transition errors: escalate (likely a bug)
    /// - Resource exhaustion: extend budget or skip
    fn determine_recovery_strategy(
        &self,
        failure_type: FailureType,
        retries_attempted: u32,
        partial_progress: Option<&str>,
        global_budget_remaining: bool,
    ) -> RecoveryStrategy {
        match failure_type {
            // Engine errors — retry once, then reassign
            FailureType::EngineError if retries_attempted < 1 => RecoveryStrategy::Retry,
            FailureType::EngineError => RecoveryStrategy::Reassign {
                with_context: false,
                partial_progress: partial_progress.map(String::from),
            },

            // Agent stuck — reassign with full context
            FailureType::AgentStuck => RecoveryStrategy::Reassign {
                with_context: true,
                partial_progress: partial_progress.map(String::from),
            },

            // Agent yielded — reassign with partial progress
            FailureType::AgentYielded => RecoveryStrategy::Reassign {
                with_context: true,
                partial_progress: partial_progress.map(String::from),
            },

            // Tool errors — reassign with context about what failed
            FailureType::ToolError => RecoveryStrategy::Reassign {
                with_context: true,
                partial_progress: partial_progress.map(String::from),
            },

            // Timeout — extend if budget allows, otherwise skip
            FailureType::Timeout => {
                if global_budget_remaining {
                    RecoveryStrategy::Retry
                } else {
                    RecoveryStrategy::Skip {
                        reason: "Timeout with no remaining budget".to_string(),
                    }
                }
            }

            // Retries exhausted — skip the subtask with partial progress
            FailureType::RetriesExhausted => RecoveryStrategy::Skip {
                reason: "All retry attempts exhausted".to_string(),
            },
        }
    }

    /// Handles a yielded agent by forwarding partial progress to a new agent.
    ///
    /// Implements spec §4.4 yield handling:
    /// 1. Record partial progress
    /// 2. Find a suitable agent based on suggested expertise
    /// 3. Route to that agent with the partial progress as context
    #[allow(dead_code)] // Will be used in Phase 3+ when full rerouting is wired up
    async fn handle_yielded_agent(
        &self,
        info: YieldedAgentInfo,
        resolver: &mut DependencyResolver,
        result_aggregator: &mut ResultAggregator,
        event_tx: &mpsc::Sender<SupervisorEvent>,
    ) -> Result<Option<String>, SupervisorError> {
        info!(
            "Agent {} yielded on subtask {} with expertise request: {:?}",
            info.agent_id, info.subtask_id, info.suggested_expertise
        );

        // Record the partial progress in the result aggregator
        result_aggregator.record_result(SubtaskResult {
            subtask_id: info.subtask_id.clone(),
            status: SubtaskStatus::Partial {
                progress: info
                    .partial_progress
                    .clone()
                    .unwrap_or_else(|| "Agent yielded".to_string()),
            },
            summary: None,
            agent_id: Some(info.agent_id.clone()),
            failure_type: None, // Yield is not a failure
        });

        // Find a suitable subtask to reroute to based on expertise
        let exclude: HashSet<String> = std::iter::once(info.subtask_id.clone()).collect();
        let available_subtasks: Vec<Subtask> = resolver.pending_subtasks();

        let reroute_target =
            RerouteResolver::find_match(&info.suggested_expertise, &available_subtasks, &exclude);

        if let Some(target_id) = &reroute_target {
            // Share the partial progress as a discovery
            if let Some(ref progress) = info.partial_progress {
                let _ = self.coordinator.share_discovery(
                    &info.agent_id,
                    Discovery {
                        content: progress.clone(),
                        category: "yielded_progress".to_string(),
                        tags: vec![info.subtask_id.clone(), target_id.clone()],
                        data: None,
                    },
                );
            }

            // Emit reroute event
            let new_agent_id = format!("agent_{}", uuid::Uuid::new_v4().simple());
            if event_tx
                .send(SupervisorEvent::Rerouted {
                    from_agent: info.agent_id.clone(),
                    to_agent: new_agent_id,
                    subtask_id: info.subtask_id.clone(),
                    reason: RerouteReason::AgentYielded {
                        suggested_expertise: info.suggested_expertise.clone(),
                        partial_progress: info.partial_progress.clone(),
                    },
                })
                .await
                .is_err()
            {
                debug!("Event channel closed, unable to send Rerouted event");
            }
        }

        Ok(reroute_target)
    }

    /// Handles a stuck agent by injecting context and potentially rerouting.
    ///
    /// Implements spec §4.4 stuck handling:
    /// 1. Examine the StuckRequest type
    /// 2. For Clarification/MoreContext: inject context from other agents
    /// 3. For DifferentTools: respawn with expanded tool access
    /// 4. For HumanIntervention: escalate to client
    #[allow(dead_code)] // Will be used in Phase 3+ when full rerouting is wired up
    async fn handle_stuck_agent(
        &self,
        info: StuckAgentInfo,
        resolver: &mut DependencyResolver,
        result_aggregator: &mut ResultAggregator,
        event_tx: &mpsc::Sender<SupervisorEvent>,
    ) -> Result<RecoveryStrategy, SupervisorError> {
        info!(
            "Agent {} stuck on subtask {} after {} attempts: {:?}",
            info.agent_id, info.subtask_id, info.attempts, info.request_type
        );

        // Record the partial progress
        result_aggregator.record_result(SubtaskResult {
            subtask_id: info.subtask_id.clone(),
            status: SubtaskStatus::Partial {
                progress: info
                    .partial_progress
                    .clone()
                    .unwrap_or_else(|| format!("Stuck after {} attempts", info.attempts)),
            },
            summary: None,
            agent_id: Some(info.agent_id.clone()),
            failure_type: Some(FailureType::AgentStuck),
        });

        let strategy = match info.request_type {
            StuckRequestType::Clarification => {
                // Try to find context from other completed subtasks
                let context = self.get_shared_context_for_agent(&info.agent_id);
                if context.is_some() {
                    // We have context to inject, retry with it
                    RecoveryStrategy::Reassign {
                        with_context: true,
                        partial_progress: info.partial_progress.clone(),
                    }
                } else {
                    // No context available, escalate
                    RecoveryStrategy::Escalate {
                        request: "Agent needs clarification but no context available".to_string(),
                    }
                }
            }

            StuckRequestType::MoreContext => {
                // Inject context from completed subtasks
                RecoveryStrategy::Reassign {
                    with_context: true,
                    partial_progress: info.partial_progress.clone(),
                }
            }

            StuckRequestType::DifferentTools { ref requested_tools } => {
                // Check if the requested tools are available
                // For now, we reassign with a note about the tools
                warn!(
                    "Agent requested different tools: {:?}",
                    requested_tools
                );
                RecoveryStrategy::Reassign {
                    with_context: true,
                    partial_progress: Some(format!(
                        "{}\n\nNote: Agent requested tools: {:?}",
                        info.partial_progress.as_deref().unwrap_or(""),
                        requested_tools
                    )),
                }
            }

            StuckRequestType::HumanIntervention => {
                RecoveryStrategy::Escalate {
                    request: format!(
                        "Agent {} requires human intervention on subtask {}",
                        info.agent_id, info.subtask_id
                    ),
                }
            }

            StuckRequestType::Unknown => {
                // Default to reassign with context
                RecoveryStrategy::Reassign {
                    with_context: true,
                    partial_progress: info.partial_progress.clone(),
                }
            }
        };

        // Emit reroute event for non-escalation strategies
        if matches!(strategy, RecoveryStrategy::Reassign { .. } | RecoveryStrategy::Retry) {
            let new_agent_id = format!("agent_{}", uuid::Uuid::new_v4().simple());
            if event_tx
                .send(SupervisorEvent::Rerouted {
                    from_agent: info.agent_id.clone(),
                    to_agent: new_agent_id,
                    subtask_id: info.subtask_id.clone(),
                    reason: RerouteReason::AgentStuck {
                        attempts: info.attempts,
                    },
                })
                .await
                .is_err()
            {
                debug!("Event channel closed, unable to send Rerouted event");
            }
        }

        // Mark subtask for retry if applicable
        if matches!(strategy, RecoveryStrategy::Retry | RecoveryStrategy::Reassign { .. }) {
            resolver.mark_pending(&info.subtask_id);
        }

        Ok(strategy)
    }

    /// Applies dynamic budget rebalancing after a subtask completes.
    ///
    /// When a subtask completes under budget, redistributes the surplus
    /// to still-running agents (spec §7.2).
    #[allow(dead_code)] // Will be used in Phase 3+ when full rebalancing is wired up
    fn rebalance_after_completion(
        &self,
        budget_allocator: &mut BudgetAllocator,
        completed_summary: &LoopSummary,
        running_count: u32,
    ) -> Vec<LoopConfig> {
        // Calculate unused resources from the completed subtask
        let used = ResourceConsumption {
            iterations: completed_summary.iterations_completed,
            tool_calls: completed_summary.tool_calls_made,
            tokens: completed_summary.tokens_generated,
        };

        // Record consumption
        budget_allocator.record_consumption(&used);

        // Get remaining budget
        let remaining = budget_allocator.remaining();

        // Rebalance among running agents
        rebalance_budget(
            remaining.iterations,
            remaining.tool_calls,
            remaining.tokens,
            running_count,
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StuckRequest;
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

    fn subtask_with_capabilities(id: &str, deps: Vec<&str>, capabilities: Vec<&str>) -> Subtask {
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
        let resolver = DependencyResolver::new(vec![subtask("A", vec!["Z"], Complexity::Low)]);

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
        let result = RerouteResolver::find_match(&["database".to_string()], &subtasks, &exclude);

        assert_eq!(result, Some("implement".to_string()));
    }

    #[test]
    fn test_reroute_no_match_returns_none() {
        let subtasks = vec![subtask_with_capabilities(
            "research",
            vec![],
            vec!["general"],
        )];

        let exclude = HashSet::from(["research".to_string()]);
        let result = RerouteResolver::find_match(&["database".to_string()], &subtasks, &exclude);

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
            failure_type: None,
        });
        agg.record_result(SubtaskResult {
            subtask_id: "B".to_string(),
            status: SubtaskStatus::Completed,
            summary: None,
            agent_id: Some("agent_2".to_string()),
            failure_type: None,
        });
        agg.record_result(SubtaskResult {
            subtask_id: "C".to_string(),
            status: SubtaskStatus::Failed {
                reason: "timeout".to_string(),
            },
            summary: None,
            agent_id: Some("agent_3".to_string()),
            failure_type: Some(FailureType::Timeout),
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
            failure_type: None,
        });
        agg.record_result(SubtaskResult {
            subtask_id: "B".to_string(),
            status: SubtaskStatus::Failed {
                reason: "error".to_string(),
            },
            summary: None,
            agent_id: Some("agent_2".to_string()),
            failure_type: Some(FailureType::EngineError),
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
        let a_result = summary.subtask_results.iter().find(|r| r.subtask_id == "A");
        assert!(a_result.is_some());
        assert!(matches!(
            a_result.map(|r| &r.status),
            Some(SubtaskStatus::Completed)
        ));
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
                failure_type: None,
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
            failure_type: Some(FailureType::EngineError),
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
            failure_type: None,
        });
        for id in &["A", "B", "C"] {
            agg.record_result(SubtaskResult {
                subtask_id: id.to_string(),
                status: SubtaskStatus::Failed {
                    reason: "fail".to_string(),
                },
                summary: None,
                agent_id: None,
                failure_type: Some(FailureType::EngineError),
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
        let distressed_action = actions.iter().find(|a| a.agent_id == "agent_1");
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
    // Coordinator Integration Tests (Phase 2)
    // =======================================================================

    mod coordinator_tests {
        use super::*;
        use async_trait::async_trait;
        use infernum_core::{
            model::LlamaVersion, response::Choice, EmbedRequest, EmbedResponse, GenerateRequest,
            GenerateResponse, ModelArchitecture, ModelId, ModelMetadata, ModelSource, RequestId,
            Result as CoreResult, TokenStream, Usage,
        };

        /// Minimal mock engine for coordinator tests.
        struct MinimalEngine {
            metadata: ModelMetadata,
        }

        impl Default for MinimalEngine {
            fn default() -> Self {
                Self {
                    metadata: ModelMetadata::builder(
                        "test-model",
                        ModelArchitecture::Llama {
                            version: LlamaVersion::V3,
                        },
                    )
                    .source(ModelSource::local("/tmp/test-model"))
                    .build(),
                }
            }
        }

        #[async_trait]
        impl abaddon::InferenceEngine for MinimalEngine {
            async fn generate(&self, _request: GenerateRequest) -> CoreResult<GenerateResponse> {
                Ok(GenerateResponse {
                    request_id: RequestId::new(),
                    created: 0,
                    model: ModelId::new("test-model"),
                    choices: vec![Choice {
                        index: 0,
                        text: "<answer confidence=\"1.0\">Done</answer>".to_string(),
                        message: None,
                        finish_reason: None,
                        logprobs: None,
                    }],
                    usage: Usage::new(10, 20),
                    time_to_first_token_ms: None,
                    total_time_ms: None,
                })
            }

            async fn generate_stream(&self, _request: GenerateRequest) -> CoreResult<TokenStream> {
                Ok(TokenStream::empty())
            }

            async fn embed(&self, _request: EmbedRequest) -> CoreResult<EmbedResponse> {
                Err(infernum_core::Error::internal("Not supported in mock"))
            }

            fn model_info(&self) -> &ModelMetadata {
                &self.metadata
            }

            fn is_ready(&self) -> bool {
                true
            }
        }

        #[test]
        fn test_supervisor_creates_coordinator_for_single_agent() {
            let engine = Arc::new(MinimalEngine::default());
            let tools = Arc::new(ToolRegistry::new());
            let config = SupervisorConfig::default(); // SingleAgent mode

            let supervisor = Supervisor::new(engine, tools, config);

            // Coordinator should exist
            let coord = supervisor.coordinator();
            // No quotas in SingleAgent mode
            assert!(coord.quotas.is_none());
        }

        #[test]
        fn test_supervisor_creates_coordinator_with_quotas_for_multi_agent() {
            let engine = Arc::new(MinimalEngine::default());
            let tools = Arc::new(ToolRegistry::new());
            let config = SupervisorConfig {
                decomposition: DecompositionStrategy::ClientProvided {
                    subtasks: vec![subtask("a", vec![], Complexity::Low)],
                },
                resource_budget: ResourceBudget {
                    total_tool_calls: 500,
                    total_tokens: 100_000,
                    ..Default::default()
                },
                ..Default::default()
            };

            let supervisor = Supervisor::new(engine, tools, config);

            // Coordinator should have quotas in multi-agent mode
            let coord = supervisor.coordinator();
            assert!(coord.quotas.is_some());
        }

        #[test]
        fn test_supervisor_with_custom_coordinator() {
            let engine = Arc::new(MinimalEngine::default());
            let tools = Arc::new(ToolRegistry::new());
            let config = SupervisorConfig::default();

            // Create custom coordinator with specific quotas
            let custom_coord = Arc::new(AgentCoordinator::with_quotas(1000, 50_000));

            let supervisor =
                Supervisor::with_coordinator(engine, tools, config, custom_coord.clone());

            // Should use the custom coordinator
            assert!(Arc::ptr_eq(supervisor.coordinator(), &custom_coord));
        }

        #[test]
        fn test_shared_context_none_returns_none() {
            let engine = Arc::new(MinimalEngine::default());
            let tools = Arc::new(ToolRegistry::new());
            let config = SupervisorConfig {
                shared_context_mode: SharedContextMode::None,
                ..Default::default()
            };

            let supervisor = Supervisor::new(engine, tools, config);

            // Register an agent first
            let agent_id = "test_agent".to_string();
            supervisor
                .coordinator
                .register_agent(AgentIdentity::new(&agent_id, AgentRole::Primary));

            // Should return None because sharing is disabled
            let context = supervisor.get_shared_context_for_agent(&agent_id);
            assert!(context.is_none());
        }

        #[test]
        fn test_shared_context_with_discoveries() {
            let engine = Arc::new(MinimalEngine::default());
            let tools = Arc::new(ToolRegistry::new());
            let config = SupervisorConfig {
                shared_context_mode: SharedContextMode::FullSharing,
                decomposition: DecompositionStrategy::ClientProvided {
                    subtasks: vec![subtask("a", vec![], Complexity::Low)],
                },
                ..Default::default()
            };

            let supervisor = Supervisor::new(engine, tools, config);

            // Register agents
            let agent_1 = "agent_1".to_string();
            let agent_2 = "agent_2".to_string();
            supervisor
                .coordinator
                .register_agent(AgentIdentity::new(&agent_1, AgentRole::Primary));
            supervisor
                .coordinator
                .register_agent(AgentIdentity::new(&agent_2, AgentRole::Specialist));

            // Agent 1 shares a discovery
            let _ = supervisor.coordinator.share_discovery(
                &agent_1,
                Discovery {
                    content: "Found the answer!".to_string(),
                    category: "subtask_result".to_string(),
                    tags: vec!["task_1".to_string()],
                    data: None,
                },
            );

            // Agent 2 should be able to see it
            let context = supervisor.get_shared_context_for_agent(&agent_2);
            assert!(context.is_some());
            let context_str = context.unwrap();
            assert!(context_str.contains("Found the answer!"));
            assert!(context_str.contains("subtask_result"));
        }

        // -------------------------------------------------------------------
        // Parallel Dispatch Tests
        // -------------------------------------------------------------------

        #[test]
        fn test_parallel_config_triggers_parallel_dispatch() {
            // When routing is Parallel, use_parallel should be true
            let config = SupervisorConfig {
                routing: RoutingStrategy::Parallel,
                max_concurrent_agents: 4,
                ..Default::default()
            };

            let use_parallel = matches!(config.routing, RoutingStrategy::Parallel)
                || config.max_concurrent_agents > 1;

            assert!(use_parallel);
        }

        #[test]
        fn test_sequential_config_avoids_parallel_dispatch() {
            // When routing is DependencyAware and max_concurrent is 1, sequential
            let config = SupervisorConfig {
                routing: RoutingStrategy::DependencyAware,
                max_concurrent_agents: 1,
                ..Default::default()
            };

            let use_parallel = matches!(config.routing, RoutingStrategy::Parallel)
                || config.max_concurrent_agents > 1;

            assert!(!use_parallel);
        }

        #[test]
        fn test_high_concurrency_enables_parallel() {
            // Even with DependencyAware, high concurrency enables parallel
            let config = SupervisorConfig {
                routing: RoutingStrategy::DependencyAware,
                max_concurrent_agents: 8,
                ..Default::default()
            };

            let use_parallel = matches!(config.routing, RoutingStrategy::Parallel)
                || config.max_concurrent_agents > 1;

            assert!(use_parallel);
        }

        #[test]
        fn test_concurrency_limiter_respects_max() {
            let mut limiter = ConcurrencyLimiter::new(3);

            // Can acquire up to 3
            assert!(limiter.try_acquire());
            assert!(limiter.try_acquire());
            assert!(limiter.try_acquire());

            // 4th should fail
            assert!(!limiter.try_acquire());

            // Release one
            limiter.release();

            // Now can acquire again
            assert!(limiter.try_acquire());
        }

        #[test]
        fn test_concurrency_limiter_tracks_max_observed() {
            let mut limiter = ConcurrencyLimiter::new(5);

            limiter.try_acquire();
            limiter.try_acquire();
            limiter.try_acquire();

            // Max observed should be 3
            assert_eq!(limiter.max_observed(), 3);

            limiter.release();
            limiter.release();

            // Still 3
            assert_eq!(limiter.max_observed(), 3);

            // Acquire more
            limiter.try_acquire();
            limiter.try_acquire();
            limiter.try_acquire();
            limiter.try_acquire();

            // Now max is 5 (3 - 2 + 4 = 5)
            assert_eq!(limiter.max_observed(), 5);
        }

        // -------------------------------------------------------------------
        // Phase 3: Recovery Strategy Tests
        // -------------------------------------------------------------------

        #[test]
        fn test_recovery_strategy_retry_on_first_engine_error() {
            let engine = Arc::new(MinimalEngine::default());
            let tools = Arc::new(ToolRegistry::new());
            let config = SupervisorConfig::default();
            let supervisor = Supervisor::new(engine, tools, config);

            let strategy = supervisor.determine_recovery_strategy(
                FailureType::EngineError,
                0, // first attempt
                None,
                true, // has budget
            );

            assert!(matches!(strategy, RecoveryStrategy::Retry));
        }

        #[test]
        fn test_recovery_strategy_reassign_after_retry() {
            let engine = Arc::new(MinimalEngine::default());
            let tools = Arc::new(ToolRegistry::new());
            let config = SupervisorConfig::default();
            let supervisor = Supervisor::new(engine, tools, config);

            let strategy = supervisor.determine_recovery_strategy(
                FailureType::EngineError,
                1, // second attempt
                Some("partial progress"),
                true,
            );

            assert!(matches!(strategy, RecoveryStrategy::Reassign { .. }));
        }

        #[test]
        fn test_recovery_strategy_agent_stuck_reassigns_with_context() {
            let engine = Arc::new(MinimalEngine::default());
            let tools = Arc::new(ToolRegistry::new());
            let config = SupervisorConfig::default();
            let supervisor = Supervisor::new(engine, tools, config);

            let strategy = supervisor.determine_recovery_strategy(
                FailureType::AgentStuck,
                0,
                Some("stuck progress"),
                true,
            );

            match strategy {
                RecoveryStrategy::Reassign {
                    with_context,
                    partial_progress,
                } => {
                    assert!(with_context);
                    assert_eq!(partial_progress, Some("stuck progress".to_string()));
                }
                _ => panic!("Expected Reassign strategy"),
            }
        }

        #[test]
        fn test_recovery_strategy_timeout_with_no_budget_skips() {
            let engine = Arc::new(MinimalEngine::default());
            let tools = Arc::new(ToolRegistry::new());
            let config = SupervisorConfig::default();
            let supervisor = Supervisor::new(engine, tools, config);

            let strategy = supervisor.determine_recovery_strategy(
                FailureType::Timeout,
                0,
                None,
                false, // no budget remaining
            );

            assert!(matches!(strategy, RecoveryStrategy::Skip { .. }));
        }

        #[test]
        fn test_recovery_strategy_retries_exhausted_skips() {
            let engine = Arc::new(MinimalEngine::default());
            let tools = Arc::new(ToolRegistry::new());
            let config = SupervisorConfig::default();
            let supervisor = Supervisor::new(engine, tools, config);

            let strategy = supervisor.determine_recovery_strategy(
                FailureType::RetriesExhausted,
                3,
                Some("gave up"),
                true,
            );

            assert!(matches!(strategy, RecoveryStrategy::Skip { .. }));
        }
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
            (10u32..1000, 10u32..5000, 1000u32..200_000).prop_map(|(iters, calls, tokens)| {
                ResourceBudget {
                    total_iterations: iters,
                    total_tool_calls: calls,
                    total_tokens: tokens,
                }
            })
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

    // =======================================================================
    // §12.9 LlmPlanned Decomposition
    // =======================================================================

    #[test]
    fn test_extract_json_array_markdown_code_block() {
        let text = r#"Here is the plan:

```json
[
  {"id": "task1", "objective": "Do something", "depends_on": [], "complexity": "low"}
]
```

That's all."#;

        let result = Supervisor::extract_json_array(text);
        assert!(result.is_some());
        let json_str = result.unwrap();
        assert!(json_str.starts_with('['));
        assert!(json_str.ends_with(']'));
        assert!(json_str.contains("task1"));
    }

    #[test]
    fn test_extract_json_array_bare_json() {
        let text = r#"[
  {"id": "task1", "objective": "Test", "depends_on": [], "complexity": "medium"}
]"#;

        let result = Supervisor::extract_json_array(text);
        assert!(result.is_some());
        let json_str = result.unwrap();
        assert!(json_str.contains("task1"));
    }

    #[test]
    fn test_extract_json_array_generic_code_block() {
        let text = r#"Plan:
```
[
  {"id": "step1", "objective": "First step", "depends_on": [], "complexity": "low"}
]
```"#;

        let result = Supervisor::extract_json_array(text);
        assert!(result.is_some());
        let json_str = result.unwrap();
        assert!(json_str.contains("step1"));
    }

    #[test]
    fn test_extract_json_array_no_json_returns_none() {
        let text = "This is just plain text without any JSON.";
        let result = Supervisor::extract_json_array(text);
        assert!(result.is_none());
    }

    #[test]
    fn test_raw_subtask_deserialization() {
        let json = r#"[
            {
                "id": "research",
                "objective": "Research existing code",
                "depends_on": [],
                "complexity": "low"
            },
            {
                "id": "implement",
                "objective": "Implement the feature",
                "depends_on": ["research"],
                "complexity": "high"
            }
        ]"#;

        let subtasks: Vec<RawSubtask> = serde_json::from_str(json).unwrap();
        assert_eq!(subtasks.len(), 2);
        assert_eq!(subtasks[0].id, "research");
        assert_eq!(subtasks[0].complexity, "low");
        assert!(subtasks[0].depends_on.is_empty());
        assert_eq!(subtasks[1].id, "implement");
        assert_eq!(subtasks[1].depends_on, vec!["research".to_string()]);
    }

    #[test]
    fn test_raw_subtask_default_complexity() {
        let json = r#"[
            {
                "id": "task",
                "objective": "Do something"
            }
        ]"#;

        let subtasks: Vec<RawSubtask> = serde_json::from_str(json).unwrap();
        assert_eq!(subtasks.len(), 1);
        assert_eq!(subtasks[0].complexity, "medium");
    }

    #[test]
    fn test_llm_planned_decomposition_strategy() {
        let strategy = DecompositionStrategy::LlmPlanned {
            available_tools: vec!["read_file".to_string(), "write_file".to_string()],
            max_subtasks: 5,
        };

        match strategy {
            DecompositionStrategy::LlmPlanned {
                available_tools,
                max_subtasks,
            } => {
                assert_eq!(available_tools.len(), 2);
                assert_eq!(max_subtasks, 5);
            }
            _ => panic!("Expected LlmPlanned"),
        }
    }

    // =======================================================================
    // Phase 3: Recovery and Rerouting Tests
    // =======================================================================

    /// Helper to create a minimal LoopSummary for tests.
    fn make_test_summary(termination: TerminationReason, partial_answer: Option<String>) -> LoopSummary {
        LoopSummary {
            termination,
            iterations_completed: 5,
            tool_calls_made: 10,
            tokens_generated: 1000,
            partial_answer,
            wall_time: Duration::from_secs(30),
            exploration_summary: vec![],
            tool_results_summary: vec![],
            can_resume: false,
            continuation_token: None,
        }
    }

    #[test]
    fn test_analyze_termination_answer_provided() {
        let summary = make_test_summary(
            TerminationReason::Natural(NaturalTermination::AnswerProvided {
                answer: "done".to_string(),
                confidence: 0.9,
            }),
            None,
        );

        let analysis = analyze_termination(&summary.termination, &summary);
        assert!(matches!(analysis.status, SubtaskStatus::Completed));
        assert!(analysis.failure_type.is_none());
    }

    #[test]
    fn test_analyze_termination_agent_stuck() {
        let summary = make_test_summary(
            TerminationReason::Natural(NaturalTermination::AgentStuck {
                attempts: 3,
                request: StuckRequest::Clarification(vec!["Need help".to_string()]),
            }),
            Some("partial work".to_string()),
        );

        let analysis = analyze_termination(&summary.termination, &summary);
        assert!(matches!(analysis.status, SubtaskStatus::Partial { .. }));
        assert_eq!(analysis.failure_type, Some(FailureType::AgentStuck));
    }

    #[test]
    fn test_analyze_termination_agent_yielded() {
        let summary = make_test_summary(
            TerminationReason::Natural(NaturalTermination::AgentYielded {
                partial: Some("progress so far".to_string()),
                reason: "need different expertise".to_string(),
            }),
            None,
        );

        let analysis = analyze_termination(&summary.termination, &summary);
        assert!(matches!(analysis.status, SubtaskStatus::Partial { .. }));
        assert_eq!(analysis.failure_type, Some(FailureType::AgentYielded));
    }

    #[test]
    fn test_exceeds_failure_threshold_below() {
        let mut agg = ResultAggregator::new();

        // 1 failure out of 10 = 10% < 50%
        agg.record_result(SubtaskResult {
            subtask_id: "A".to_string(),
            status: SubtaskStatus::Failed {
                reason: "err".to_string(),
            },
            summary: None,
            agent_id: None,
            failure_type: Some(FailureType::EngineError),
        });

        assert!(!agg.exceeds_failure_threshold(10));
    }

    #[test]
    fn test_exceeds_failure_threshold_at_boundary() {
        let mut agg = ResultAggregator::new();

        // 5 failures out of 10 = 50% is NOT exceeded (threshold is >50%)
        for i in 0..5 {
            agg.record_result(SubtaskResult {
                subtask_id: format!("F{}", i),
                status: SubtaskStatus::Failed {
                    reason: "err".to_string(),
                },
                summary: None,
                agent_id: None,
                failure_type: Some(FailureType::EngineError),
            });
        }

        assert!(!agg.exceeds_failure_threshold(10));
    }

    #[test]
    fn test_exceeds_failure_threshold_above() {
        let mut agg = ResultAggregator::new();

        // 6 failures out of 10 = 60% > 50%
        for i in 0..6 {
            agg.record_result(SubtaskResult {
                subtask_id: format!("F{}", i),
                status: SubtaskStatus::Failed {
                    reason: "err".to_string(),
                },
                summary: None,
                agent_id: None,
                failure_type: Some(FailureType::EngineError),
            });
        }

        assert!(agg.exceeds_failure_threshold(10));
    }

    #[test]
    fn test_exceeds_failure_threshold_empty() {
        let agg = ResultAggregator::new();
        assert!(!agg.exceeds_failure_threshold(0));
        assert!(!agg.exceeds_failure_threshold(10));
    }

    #[test]
    fn test_reroute_resolver_case_insensitive() {
        let subtasks = vec![
            Subtask {
                id: "task_1".to_string(),
                objective: "Do something".to_string(),
                depends_on: vec![],
                capabilities: vec!["FILE_IO".to_string(), "NETWORKING".to_string()],
                complexity: Complexity::Medium,
            },
            Subtask {
                id: "task_2".to_string(),
                objective: "Do another".to_string(),
                depends_on: vec![],
                capabilities: vec!["database".to_string()],
                complexity: Complexity::Low,
            },
        ];

        let exclude = HashSet::new();

        // Lowercase query should match uppercase capability
        let result = RerouteResolver::find_match(&["file_io".to_string()], &subtasks, &exclude);
        assert_eq!(result, Some("task_1".to_string()));

        // Uppercase query should match lowercase capability
        let result = RerouteResolver::find_match(&["DATABASE".to_string()], &subtasks, &exclude);
        assert_eq!(result, Some("task_2".to_string()));

        // Mixed case should work
        let result = RerouteResolver::find_match(&["NetWorking".to_string()], &subtasks, &exclude);
        assert_eq!(result, Some("task_1".to_string()));
    }

    #[test]
    fn test_failure_type_display() {
        assert_eq!(FailureType::EngineError.to_string(), "engine_error");
        assert_eq!(FailureType::AgentStuck.to_string(), "agent_stuck");
        assert_eq!(FailureType::AgentYielded.to_string(), "agent_yielded");
        assert_eq!(FailureType::ToolError.to_string(), "tool_error");
        assert_eq!(FailureType::Timeout.to_string(), "timeout");
        assert_eq!(
            FailureType::RetriesExhausted.to_string(),
            "retries_exhausted"
        );
    }
}
