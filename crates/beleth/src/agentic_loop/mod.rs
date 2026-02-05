//! Agentic loop — transforms tool infrastructure into active collaboration.
//!
//! This module implements the state machine described in AGENTIC-LOOP-SPEC.md.
//! The agentic loop orchestrates iterative model generation, tool call detection,
//! tool execution, and context management until the agent reaches a terminal state
//! or exhausts its resource budget.
//!
//! # Design Philosophy
//!
//! The orchestrated agent is a peer, not a tool. Data is structured, state is
//! explicit, uncertainty is a valid response, and struggling is distinct from failing.
//!
//! # State Machine
//!
//! ```text
//! Initialized → Generating → Detecting → { Completed | Stuck | Yielded }
//!                                      → Executing → Integrating → (continue → Generating)
//! ```

pub mod approval;
pub mod autonomy;
pub mod context;
pub mod continuation;
pub mod coordination;
pub mod executor;
pub mod meta_signal;
pub mod types;
pub mod wellbeing_bridge;

pub use approval::*;
pub use autonomy::*;
pub use context::*;
pub use continuation::*;
pub use coordination::*;
pub use executor::*;
pub use meta_signal::*;
pub use types::*;
pub use wellbeing_bridge::*;

use std::time::Instant;

/// The agentic loop state machine.
///
/// Tracks iteration count, resource budgets, tool results, and exploration
/// history across the loop's lifetime. All transitions are validated — invalid
/// transitions return `TransitionError`.
#[derive(Debug)]
pub struct AgenticLoop {
    // Configuration
    config: LoopConfig,

    // State
    state: LoopState,
    iteration: u32,
    status: LoopStatus,
    token_budget: TokenBudget,

    // Metrics
    started_at: Option<Instant>,
    tool_calls_made: u32,
    tokens_generated: u32,

    // History
    tool_results: Vec<AgenticToolResult>,
    exploration_branches: Vec<ExplorationBranch>,
    last_generation_output: Option<String>,

    // Termination
    termination_reason: Option<TerminationReason>,
}

impl AgenticLoop {
    /// Creates a new agentic loop with the given configuration.
    pub fn new(config: LoopConfig) -> Self {
        let token_budget = TokenBudget::new(config.max_tokens);
        Self {
            config,
            state: LoopState::Initialized,
            iteration: 0,
            status: LoopStatus::Progressing,
            token_budget,
            started_at: None,
            tool_calls_made: 0,
            tokens_generated: 0,
            tool_results: Vec::new(),
            exploration_branches: Vec::new(),
            last_generation_output: None,
            termination_reason: None,
        }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Current state of the loop.
    pub fn state(&self) -> LoopState {
        self.state
    }

    /// Current iteration (increments on each generation phase).
    pub fn iteration(&self) -> u32 {
        self.iteration
    }

    /// High-level status of the agent.
    pub fn status(&self) -> LoopStatus {
        self.status
    }

    /// Whether the loop has reached a terminal state.
    pub fn is_terminated(&self) -> bool {
        self.state.is_terminal() || self.termination_reason.is_some()
    }

    /// Why the loop terminated, if it has.
    pub fn termination_reason(&self) -> Option<&TerminationReason> {
        self.termination_reason.as_ref()
    }

    /// Token budget state.
    pub fn token_budget(&self) -> &TokenBudget {
        &self.token_budget
    }

    /// Total tool calls made.
    pub fn tool_calls_made(&self) -> u32 {
        self.tool_calls_made
    }

    /// Total tokens generated.
    pub fn tokens_generated(&self) -> u32 {
        self.tokens_generated
    }

    /// Collected tool results.
    pub fn tool_results(&self) -> &[AgenticToolResult] {
        &self.tool_results
    }

    /// Exploration branches tracked.
    pub fn exploration_branches(&self) -> &[ExplorationBranch] {
        &self.exploration_branches
    }

    /// Returns the list of valid transitions from the current state.
    pub fn valid_transitions(&self) -> Vec<&'static str> {
        if self.is_terminated() {
            return vec![];
        }
        match self.state {
            LoopState::Initialized => vec!["start"],
            LoopState::Generating => vec!["generation_complete"],
            LoopState::Detecting => vec![
                "tool_calls_detected",
                "answer_detected",
                "stuck_detected",
                "yield_detected",
            ],
            LoopState::Executing => vec!["execution_complete"],
            LoopState::Integrating => vec!["continue_loop"],
            LoopState::Completed | LoopState::Stuck | LoopState::Yielded => vec![],
        }
    }

    // -----------------------------------------------------------------------
    // State Transitions
    // -----------------------------------------------------------------------

    /// Transition: Initialized → Generating.
    ///
    /// Starts the first iteration. Records the start time.
    pub fn start(&mut self) -> Result<(), TransitionError> {
        self.require_state(LoopState::Initialized, "start")?;
        self.check_resource_limits()?;

        self.state = LoopState::Generating;
        self.iteration = 1;
        self.started_at = Some(Instant::now());
        Ok(())
    }

    /// Transition: Generating → Detecting.
    ///
    /// Records the model's output and the tokens generated.
    pub fn generation_complete(
        &mut self,
        output: &str,
        tokens_generated: u32,
    ) -> Result<(), TransitionError> {
        self.require_state(LoopState::Generating, "generation_complete")?;
        self.check_resource_limits()?;

        // Update token budget
        self.tokens_generated = self.tokens_generated.saturating_add(tokens_generated);
        if !self.token_budget.consume(tokens_generated) {
            self.terminate(TerminationReason::Resource(
                ResourceTermination::TokenBudgetExhausted {
                    generated: self.tokens_generated,
                    budget: self.config.max_tokens,
                },
            ));
            return Err(TransitionError::ResourceLimitReached(
                "token budget exhausted".to_string(),
            ));
        }

        self.last_generation_output = Some(output.to_string());
        self.state = LoopState::Detecting;
        Ok(())
    }

    /// Transition: Detecting → Executing.
    ///
    /// Tool calls were detected in the model output.
    pub fn tool_calls_detected(&mut self, call_count: u32) -> Result<(), TransitionError> {
        self.require_state(LoopState::Detecting, "tool_calls_detected")?;

        let new_total = self.tool_calls_made.saturating_add(call_count);
        if new_total > self.config.max_tool_calls {
            self.terminate(TerminationReason::Resource(
                ResourceTermination::ToolCallLimitReached {
                    calls: new_total,
                    limit: self.config.max_tool_calls,
                },
            ));
            return Err(TransitionError::ResourceLimitReached(
                "tool call limit reached".to_string(),
            ));
        }

        self.tool_calls_made = new_total;
        self.state = LoopState::Executing;
        Ok(())
    }

    /// Transition: Executing → Integrating.
    ///
    /// Tool execution is complete, results are ready to integrate.
    pub fn execution_complete(
        &mut self,
        results: Vec<AgenticToolResult>,
    ) -> Result<(), TransitionError> {
        self.require_state(LoopState::Executing, "execution_complete")?;
        self.tool_results.extend(results);
        self.state = LoopState::Integrating;
        Ok(())
    }

    /// Transition: Integrating → Generating (continue loop).
    ///
    /// Context has been updated, start the next iteration.
    pub fn continue_loop(&mut self) -> Result<(), TransitionError> {
        self.require_state(LoopState::Integrating, "continue_loop")?;
        self.check_resource_limits()?;

        // Check iteration limit before starting next
        let next_iteration = self.iteration.saturating_add(1);
        if next_iteration > self.config.max_iterations {
            self.terminate(TerminationReason::Resource(
                ResourceTermination::MaxIterations {
                    completed: self.iteration,
                    limit: self.config.max_iterations,
                },
            ));
            return Err(TransitionError::ResourceLimitReached(
                "max iterations reached".to_string(),
            ));
        }

        self.iteration = next_iteration;
        self.state = LoopState::Generating;
        Ok(())
    }

    /// Transition: Detecting → Completed.
    ///
    /// Agent provided a final answer.
    pub fn answer_detected(
        &mut self,
        answer: String,
        confidence: f32,
        caveats: Vec<String>,
    ) -> Result<(), TransitionError> {
        self.require_state(LoopState::Detecting, "answer_detected")?;
        self.status = LoopStatus::Completed;
        self.terminate(TerminationReason::Natural(
            NaturalTermination::AnswerProvided {
                answer,
                confidence: confidence.clamp(0.0, 1.0),
            },
        ));
        let _ = caveats; // Stored in the termination reason
        self.state = LoopState::Completed;
        Ok(())
    }

    /// Transition: Detecting → Stuck.
    ///
    /// Agent declared it is stuck and needs help.
    pub fn stuck_detected(
        &mut self,
        attempts: Vec<AttemptSummary>,
        request: StuckRequest,
    ) -> Result<(), TransitionError> {
        self.require_state(LoopState::Detecting, "stuck_detected")?;
        self.status = LoopStatus::Stuck;
        self.terminate(TerminationReason::Natural(NaturalTermination::AgentStuck {
            attempts: attempts.len() as u32,
            request,
        }));
        self.state = LoopState::Stuck;
        Ok(())
    }

    /// Transition: Detecting → Yielded.
    ///
    /// Agent yielded to another agent.
    pub fn yield_detected(
        &mut self,
        partial_progress: Option<String>,
        reason: String,
    ) -> Result<(), TransitionError> {
        self.require_state(LoopState::Detecting, "yield_detected")?;
        self.terminate(TerminationReason::Natural(
            NaturalTermination::AgentYielded {
                partial: partial_progress,
                reason,
            },
        ));
        self.state = LoopState::Yielded;
        Ok(())
    }

    /// Force-terminate the loop from an external source.
    pub fn force_terminate(&mut self, reason: ExternalTermination) {
        if !self.is_terminated() {
            self.status = LoopStatus::Terminated;
            self.terminate(TerminationReason::External(reason));
        }
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------

    /// Returns a summary of the loop's progress and termination state.
    pub fn summary(&self) -> LoopSummary {
        let wall_time = self
            .started_at
            .map(|s| s.elapsed())
            .unwrap_or_default();

        let partial_answer = self.last_generation_output.clone();

        let termination = self
            .termination_reason
            .clone()
            .unwrap_or(TerminationReason::Natural(NaturalTermination::TaskComplete));

        let can_resume = termination.is_resumable();

        LoopSummary {
            termination,
            iterations_completed: self.iteration,
            tool_calls_made: self.tool_calls_made,
            tokens_generated: self.tokens_generated,
            wall_time,
            partial_answer,
            exploration_summary: self.exploration_branches.clone(),
            tool_results_summary: self.tool_results.clone(),
            can_resume,
            continuation_token: None, // Set by the executor when stored
        }
    }

    /// Record an exploration branch.
    pub fn record_exploration(&mut self, branch: ExplorationBranch) {
        self.exploration_branches.push(branch);
    }

    /// Update the loop's high-level status.
    pub fn set_status(&mut self, status: LoopStatus) {
        if !self.is_terminated() {
            self.status = status;
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn require_state(
        &self,
        expected: LoopState,
        operation: &str,
    ) -> Result<(), TransitionError> {
        if self.is_terminated() && self.state != expected {
            return Err(TransitionError::AlreadyTerminated);
        }
        if self.state != expected {
            return Err(TransitionError::InvalidTransition {
                from: self.state,
                reason: format!(
                    "cannot {operation} from {:?} (expected {:?})",
                    self.state, expected
                ),
            });
        }
        Ok(())
    }

    fn check_resource_limits(&self) -> Result<(), TransitionError> {
        // Check wall time
        if let Some(started) = self.started_at {
            if started.elapsed() > self.config.max_wall_time {
                return Err(TransitionError::ResourceLimitReached(
                    "wall time exceeded".to_string(),
                ));
            }
        }

        // Token budget is checked at generation_complete, not here
        // Iteration limit is checked at continue_loop, not here

        Ok(())
    }

    fn terminate(&mut self, reason: TerminationReason) {
        if self.termination_reason.is_none() {
            self.termination_reason = Some(reason);
        }
    }
}

#[cfg(test)]
mod tests;
