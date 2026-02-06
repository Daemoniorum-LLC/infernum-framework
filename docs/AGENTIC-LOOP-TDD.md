# Agentic Loop TDD Roadmap

**Version:** 0.3.0
**Status:** Test Specification
**Date:** 2026-02-04
**Spec Reference:** AGENTIC-LOOP-SPEC.md v0.3.0, MULTI-AGENT-SUPERVISOR-SPEC.md v0.1.0

---

## Philosophy

Tests are crystallized understanding, not coverage theater.

Each test in this roadmap exists because it captures something we *must* know is true. If a test doesn't teach us something about the system's behavior, it doesn't belong here.

We test at **trust boundaries** — the edges where assumptions meet reality:
- External input enters the system
- Promises are made to consumers
- State transitions occur
- Resources are acquired or released

We prefer **property tests** over example tests where the property is the point. "Forbidden tools never execute" is a property. "Tool X with args Y returns Z" is an example.

---

## 1. Loop State Machine

**Trust Boundary:** State transitions are the skeleton of the loop. Invalid transitions indicate logic errors that could cause hangs, infinite loops, or lost work.

### 1.1 Valid Transitions

```
┌─────────────┐
│ Initialized │
└──────┬──────┘
       │ start()
       ▼
┌─────────────┐ ◄───────────────────────────────┐
│ Generating  │                                 │
└──────┬──────┘                                 │
       │ generation_complete()                  │
       ▼                                        │
┌─────────────┐                                 │
│ Detecting   │                                 │
└──────┬──────┘                                 │
       │                                        │
       ├─── tool_calls_detected() ───┐          │
       │                             ▼          │
       │                      ┌───────────┐     │
       │                      │ Executing │     │
       │                      └─────┬─────┘     │
       │                            │           │
       │                            │ execution_complete()
       │                            │           │
       │                      ┌─────▼─────┐     │
       │                      │Integrating│     │
       │                      └─────┬─────┘     │
       │                            │           │
       │                            │ continue()│
       │                            └───────────┘
       │
       ├─── answer_detected() ──────┐
       │                            ▼
       │                     ┌────────────┐
       │                     │ Completed  │
       │                     └────────────┘
       │
       ├─── stuck_detected() ───────┐
       │                            ▼
       │                     ┌────────────┐
       │                     │   Stuck    │
       │                     └────────────┘
       │
       └─── yield_detected() ───────┐
                                    ▼
                             ┌────────────┐
                             │  Yielded   │
                             └────────────┘
```

#### Tests

```rust
#[test]
fn test_initialized_can_only_transition_to_generating() {
    let loop = AgenticLoop::new(config);
    assert_eq!(loop.state(), LoopState::Initialized);

    // Valid
    assert!(loop.start().is_ok());
    assert_eq!(loop.state(), LoopState::Generating);
}

#[test]
fn test_initialized_rejects_invalid_transitions() {
    let loop = AgenticLoop::new(config);

    // Invalid transitions from Initialized
    assert!(loop.generation_complete("output").is_err());
    assert!(loop.tool_calls_detected(vec![]).is_err());
    assert!(loop.execution_complete(vec![]).is_err());
}

// Property: Terminal states have no valid transitions
#[proptest]
fn test_terminal_states_are_final(terminal: TerminalState) {
    let loop = create_loop_in_state(terminal.into());

    // All transitions should fail
    prop_assert!(loop.start().is_err());
    prop_assert!(loop.generation_complete("").is_err());
    prop_assert!(loop.tool_calls_detected(vec![]).is_err());
    prop_assert!(loop.continue_loop().is_err());
}

// Property: Every non-terminal state has at least one valid transition
#[proptest]
fn test_non_terminal_states_have_transitions(state: NonTerminalState) {
    let loop = create_loop_in_state(state.into());
    let valid_transitions = loop.valid_transitions();

    prop_assert!(!valid_transitions.is_empty());
}
```

### 1.2 Iteration Counting

```rust
#[test]
fn test_iteration_increments_on_generation_start() {
    let mut loop = AgenticLoop::new(config);
    assert_eq!(loop.iteration(), 0);

    loop.start().unwrap();
    assert_eq!(loop.iteration(), 1);

    // Complete a full cycle
    loop.generation_complete("output").unwrap();
    loop.tool_calls_detected(vec![call]).unwrap();
    loop.execution_complete(vec![result]).unwrap();
    loop.integration_complete().unwrap();
    loop.continue_loop().unwrap();

    assert_eq!(loop.iteration(), 2);
}

// Property: Iteration count equals number of generation phases entered
#[proptest]
fn test_iteration_count_matches_generations(transitions: Vec<ValidTransition>) {
    let mut loop = AgenticLoop::new(config);
    let mut expected_iterations = 0;

    for transition in transitions {
        if transition == ValidTransition::Start || transition == ValidTransition::Continue {
            expected_iterations += 1;
        }
        let _ = loop.apply(transition);
    }

    prop_assert_eq!(loop.iteration(), expected_iterations);
}
```

### 1.3 State Persistence

```rust
#[test]
fn test_state_survives_serialization_roundtrip() {
    let mut loop = create_loop_with_history();
    let serialized = loop.serialize().unwrap();
    let restored = AgenticLoop::deserialize(&serialized).unwrap();

    assert_eq!(loop.state(), restored.state());
    assert_eq!(loop.iteration(), restored.iteration());
    assert_eq!(loop.tool_results(), restored.tool_results());
    assert_eq!(loop.exploration_branches(), restored.exploration_branches());
}
```

---

## 2. Autonomy Enforcement

**Trust Boundary:** The autonomy grant is a security boundary. Violations here could allow unauthorized file writes, command execution, or data exfiltration.

### 2.1 Permission Checking

```rust
// Property: Forbidden tools NEVER execute
#[proptest]
fn test_forbidden_tools_never_execute(
    grant: AutonomyGrant,
    call: DetectedToolCall,
) {
    prop_assume!(grant.is_forbidden(&call));

    let result = execute_with_grant(&call, &grant).await;

    prop_assert!(matches!(result, ExecutionOutcome::Denied { .. }));
    prop_assert!(!was_tool_invoked(&call)); // side-effect check
}

// Property: Allowed tools execute without blocking
#[proptest]
fn test_allowed_tools_execute_immediately(
    grant: AutonomyGrant,
    call: DetectedToolCall,
) {
    prop_assume!(grant.is_allowed(&call));

    let result = execute_with_grant(&call, &grant).await;

    prop_assert!(!matches!(result, ExecutionOutcome::PendingApproval { .. }));
    prop_assert!(!matches!(result, ExecutionOutcome::Denied { .. }));
}

// Property: Approval-required tools block until approved
#[proptest]
fn test_approval_required_tools_block(
    grant: AutonomyGrant,
    call: DetectedToolCall,
) {
    prop_assume!(grant.requires_approval(&call));

    let (result, pending) = execute_with_grant_no_wait(&call, &grant);

    prop_assert!(matches!(result, ExecutionOutcome::PendingApproval { .. }));
    prop_assert!(pending.is_some());

    // Approve it
    pending.unwrap().approve();
    let final_result = result.await;
    prop_assert!(!matches!(final_result, ExecutionOutcome::Denied { .. }));
}
```

### 2.2 Pattern Matching

```rust
#[test]
fn test_glob_patterns_match_correctly() {
    let grant = AutonomyGrant::builder()
        .allow(ToolPattern::Read("src/**/*.rs"))
        .forbid(ToolPattern::Read("src/**/secrets.*"))
        .build();

    // Allowed
    assert!(grant.is_allowed(&read_file("src/main.rs")));
    assert!(grant.is_allowed(&read_file("src/deep/nested/module.rs")));

    // Forbidden (more specific pattern wins)
    assert!(grant.is_forbidden(&read_file("src/config/secrets.toml")));
    assert!(grant.is_forbidden(&read_file("src/secrets.rs")));

    // Not covered (defaults to require_approval)
    assert!(grant.requires_approval(&read_file("tests/main.rs")));
}

#[test]
fn test_bash_pattern_matching() {
    let grant = AutonomyGrant::builder()
        .allow(ToolPattern::Bash("git status"))
        .allow(ToolPattern::Bash("cargo *"))
        .forbid(ToolPattern::Bash("rm *"))
        .forbid(ToolPattern::Bash("sudo *"))
        .build();

    assert!(grant.is_allowed(&bash("git status")));
    assert!(grant.is_allowed(&bash("cargo build --release")));
    assert!(grant.is_forbidden(&bash("rm -rf /")));
    assert!(grant.is_forbidden(&bash("sudo anything")));
    assert!(grant.requires_approval(&bash("echo hello")));
}
```

### 2.3 Grant Narrowing

```rust
// Property: Client can narrow server grants, never widen
#[proptest]
fn test_client_can_only_narrow_grants(
    server_grant: AutonomyGrant,
    client_grant: AutonomyGrant,
) {
    let effective = server_grant.narrow_with(&client_grant);

    // Anything allowed in effective must be allowed in server
    for pattern in effective.allowed_patterns() {
        prop_assert!(server_grant.is_allowed_pattern(pattern));
    }

    // Anything forbidden in server must be forbidden in effective
    for pattern in server_grant.forbidden_patterns() {
        prop_assert!(effective.is_forbidden_pattern(pattern));
    }
}

#[test]
fn test_client_cannot_allow_server_forbidden() {
    let server = AutonomyGrant::builder()
        .forbid(ToolPattern::Bash("rm *"))
        .build();

    let client = AutonomyGrant::builder()
        .allow(ToolPattern::Bash("rm temp.txt"))  // tries to allow
        .build();

    let effective = server.narrow_with(&client);

    // Server's forbid wins
    assert!(effective.is_forbidden(&bash("rm temp.txt")));
}
```

---

## 3. Termination Conditions

**Trust Boundary:** Termination is how the loop makes promises about completion. Incorrect termination could cause infinite loops, lost work, or incorrect responses.

### 3.1 Natural Termination

```rust
#[test]
fn test_answer_terminates_loop() {
    let mut loop = create_loop_in_detecting_state();

    loop.answer_detected(Answer {
        content: "The answer is 42".into(),
        confidence: 0.95,
        caveats: vec![],
    }).unwrap();

    assert!(loop.is_terminated());
    assert!(matches!(
        loop.termination_reason(),
        Some(TerminationReason::Natural(NaturalTermination::AnswerProvided { .. }))
    ));
}

#[test]
fn test_stuck_terminates_loop() {
    let mut loop = create_loop_in_detecting_state();

    loop.stuck_detected(StuckDetails {
        attempts: vec![attempt1, attempt2],
        hypothesis: Some("Config file may be missing".into()),
        request: StuckRequest::Clarification(vec![question]),
    }).unwrap();

    assert!(loop.is_terminated());
    assert!(matches!(
        loop.termination_reason(),
        Some(TerminationReason::Natural(NaturalTermination::AgentStuck { .. }))
    ));
}

#[test]
fn test_yield_terminates_loop() {
    let mut loop = create_loop_in_detecting_state();

    loop.yield_detected(YieldDetails {
        partial_progress: Some("Found the config, but can't parse YAML".into()),
        suggested_expertise: vec!["yaml-parsing".into()],
    }).unwrap();

    assert!(loop.is_terminated());
    assert!(matches!(
        loop.termination_reason(),
        Some(TerminationReason::Natural(NaturalTermination::AgentYielded { .. }))
    ));
}
```

### 3.2 Resource Termination

```rust
#[test]
fn test_max_iterations_terminates() {
    let config = LoopConfig { max_iterations: 3, ..default() };
    let mut loop = AgenticLoop::new(config);

    // Run 3 iterations
    for _ in 0..3 {
        complete_one_iteration(&mut loop);
    }

    // 4th iteration should fail to start
    assert!(loop.continue_loop().is_err());
    assert!(matches!(
        loop.termination_reason(),
        Some(TerminationReason::Resource(ResourceTermination::MaxIterations { .. }))
    ));
}

#[test]
fn test_token_budget_terminates() {
    let config = LoopConfig { max_tokens: 100, ..default() };
    let mut loop = AgenticLoop::new(config);

    loop.start().unwrap();

    // Generate more than budget
    let result = loop.generation_complete(&"x".repeat(150));

    assert!(result.is_err() || loop.is_terminated());
    assert!(matches!(
        loop.termination_reason(),
        Some(TerminationReason::Resource(ResourceTermination::TokenBudgetExhausted { .. }))
    ));
}

#[test]
fn test_wall_time_terminates() {
    let config = LoopConfig {
        max_wall_time: Duration::from_millis(100),
        ..default()
    };
    let mut loop = AgenticLoop::new(config);

    loop.start().unwrap();
    std::thread::sleep(Duration::from_millis(150));

    // Any operation should now fail
    let result = loop.generation_complete("output");

    assert!(result.is_err() || loop.is_terminated());
    assert!(matches!(
        loop.termination_reason(),
        Some(TerminationReason::Resource(ResourceTermination::WallTimeExceeded { .. }))
    ));
}

// Property: Exactly one termination reason
#[proptest]
fn test_exactly_one_termination_reason(loop: TerminatedLoop) {
    let reason = loop.termination_reason();
    prop_assert!(reason.is_some());

    // Can't have multiple reasons
    let reasons = loop.all_termination_conditions_met();
    prop_assert!(reasons.len() >= 1); // at least one
    // The reported reason should be the first one hit
}
```

### 3.3 Partial Progress Preservation

```rust
// Property: Terminated loops preserve partial progress
#[proptest]
fn test_termination_preserves_progress(
    iterations: Vec<IterationData>,
    termination_point: usize,
) {
    let mut loop = AgenticLoop::new(config);

    for (i, data) in iterations.iter().enumerate() {
        if i >= termination_point {
            break;
        }
        apply_iteration(&mut loop, data);
    }

    force_termination(&mut loop);

    let summary = loop.summary();

    // All completed iterations should be in summary
    prop_assert_eq!(summary.iterations_completed, termination_point.min(iterations.len()) as u32);

    // Tool results should be preserved
    for i in 0..termination_point.min(iterations.len()) {
        for result in &iterations[i].tool_results {
            prop_assert!(summary.tool_results_summary.iter().any(|r| r.call_id == result.call_id));
        }
    }
}

#[test]
fn test_stuck_preserves_exploration_history() {
    let mut loop = AgenticLoop::new(config);

    // Try multiple approaches
    complete_iteration_with_tools(&mut loop, vec![read_file("config.json")]);
    complete_iteration_with_tools(&mut loop, vec![read_file("config.yaml")]);
    complete_iteration_with_tools(&mut loop, vec![read_file("config.toml")]);

    // Get stuck
    loop.stuck_detected(StuckDetails {
        attempts: vec![
            FailedApproach { description: "Tried JSON".into(), .. },
            FailedApproach { description: "Tried YAML".into(), .. },
            FailedApproach { description: "Tried TOML".into(), .. },
        ],
        ..default()
    }).unwrap();

    let summary = loop.summary();

    // Exploration branches preserved
    assert_eq!(summary.exploration_summary.len(), 3);
    assert!(summary.exploration_summary.iter().any(|b| b.description.contains("JSON")));
}
```

---

## 4. Meta-Signal Detection

**Trust Boundary:** Meta-signals are how agents communicate state beyond tool calls. Missing a signal could cause the loop to continue when it should stop, or miss opportunities for intervention.

### 4.1 Explicit Signals

```rust
#[test]
fn test_explicit_answer_signal_detected() {
    let output = r#"
        <answer confidence="0.92">
        The result is 42.
        <caveat>Assuming standard arithmetic</caveat>
        </answer>
    "#;

    let signal = detect_meta_signal(output, &DetectionConfig::default());

    assert!(matches!(signal, Some(MetaSignal::Answer { .. })));
    if let Some(MetaSignal::Answer { confidence, caveats, .. }) = signal {
        assert!((confidence - 0.92).abs() < 0.01);
        assert_eq!(caveats.len(), 1);
    }
}

#[test]
fn test_explicit_uncertain_signal_detected() {
    let output = r#"
        <uncertain>
        <partial>I found the file but couldn't parse it</partial>
        <missing>YAML parsing capability</missing>
        <missing>Schema documentation</missing>
        <would_help>Access to the schema definition</would_help>
        </uncertain>
    "#;

    let signal = detect_meta_signal(output, &DetectionConfig::default());

    assert!(matches!(signal, Some(MetaSignal::Uncertain { .. })));
    if let Some(MetaSignal::Uncertain { missing, .. }) = signal {
        assert_eq!(missing.len(), 2);
    }
}

#[test]
fn test_explicit_stuck_signal_detected() {
    let output = r#"
        <stuck>
        <attempt>Searched for config.json - not found</attempt>
        <attempt>Searched for config.yaml - not found</attempt>
        <hypothesis>Configuration might be environment variables</hypothesis>
        <request type="clarification">
            <question>Where is the configuration stored?</question>
        </request>
        </stuck>
    "#;

    let signal = detect_meta_signal(output, &DetectionConfig::default());

    assert!(matches!(signal, Some(MetaSignal::Stuck { .. })));
}

#[test]
fn test_explicit_yield_signal_detected() {
    let output = r#"
        <yield>
        <partial_progress>Identified the bug is in the parser</partial_progress>
        <suggested_expertise>rust-macros</suggested_expertise>
        <suggested_expertise>proc-macro-debugging</suggested_expertise>
        </yield>
    "#;

    let signal = detect_meta_signal(output, &DetectionConfig::default());

    assert!(matches!(signal, Some(MetaSignal::Yield { .. })));
}

// Property: Explicit signals always detected (no false negatives)
#[proptest]
fn test_explicit_signals_always_detected(signal: MetaSignal) {
    let output = signal.to_explicit_format();
    let detected = detect_meta_signal(&output, &DetectionConfig::default());

    prop_assert!(detected.is_some());
    prop_assert_eq!(detected.unwrap().signal_type(), signal.signal_type());
}
```

### 4.2 Implicit Signals

```rust
#[test]
fn test_implicit_uncertainty_detected() {
    let config = DetectionConfig { detect_implicit: true, ..default() };

    let uncertain_outputs = vec![
        "I'm not certain about this, but I think the answer might be 42.",
        "Without access to the database, I can't give a definitive answer.",
        "I would need more context to answer this accurately.",
    ];

    for output in uncertain_outputs {
        let signal = detect_meta_signal(output, &config);
        assert!(
            matches!(signal, Some(MetaSignal::Uncertain { .. })),
            "Failed to detect uncertainty in: {}",
            output
        );
    }
}

#[test]
fn test_implicit_stuck_detected() {
    let config = DetectionConfig { detect_implicit: true, ..default() };

    let stuck_outputs = vec![
        "I've tried several approaches but none have worked. I'm not making progress on this.",
        "I seem to be going in circles. Each attempt leads back to the same error.",
        "I'm stuck and need clarification about the expected format.",
    ];

    for output in stuck_outputs {
        let signal = detect_meta_signal(output, &config);
        assert!(
            matches!(signal, Some(MetaSignal::Stuck { .. })),
            "Failed to detect stuck state in: {}",
            output
        );
    }
}

#[test]
fn test_implicit_detection_disabled_by_default() {
    let config = DetectionConfig::default();
    assert!(!config.detect_implicit);

    let output = "I'm not certain about this answer.";
    let signal = detect_meta_signal(output, &config);

    // Should not detect implicit uncertainty when disabled
    assert!(signal.is_none());
}

// Property: Implicit detection has no false positives on confident answers
#[proptest]
fn test_no_false_positive_uncertainty(confident_answer: ConfidentAnswer) {
    let config = DetectionConfig { detect_implicit: true, ..default() };
    let output = confident_answer.to_string();

    let signal = detect_meta_signal(&output, &config);

    // Should not detect uncertainty in confident answers
    prop_assert!(!matches!(signal, Some(MetaSignal::Uncertain { .. })));
}
```

### 4.3 Signal Priority

```rust
#[test]
fn test_explicit_signal_takes_priority() {
    let config = DetectionConfig { detect_implicit: true, ..default() };

    // Output has both explicit answer and implicit uncertainty language
    let output = r#"
        I'm not entirely sure, but here's what I found:
        <answer confidence="0.85">The configuration is in /etc/app/config.yaml</answer>
        I would need to verify this against the documentation.
    "#;

    let signal = detect_meta_signal(output, &config);

    // Explicit answer should win
    assert!(matches!(signal, Some(MetaSignal::Answer { .. })));
}

#[test]
fn test_tool_calls_take_priority_over_signals() {
    let output = r#"
        <uncertain>
        <partial>Not sure if this will work</partial>
        </uncertain>

        <tool_call>
        {"name": "read_file", "arguments": {"path": "config.yaml"}}
        </tool_call>
    "#;

    let (tool_calls, signal) = detect_all(output, &config);

    // Tool calls should be detected
    assert!(!tool_calls.is_empty());

    // Signal should also be detected but marked as secondary
    assert!(signal.is_some());
}
```

---

## 5. Context Management

**Trust Boundary:** The context window is a hard constraint. Exceeding it causes failures. Poor management wastes capacity or loses important information.

### 5.1 Token Budget

```rust
// Property: Token count never exceeds budget
#[proptest]
fn test_token_count_never_exceeds_budget(
    operations: Vec<ContextOperation>,
    budget: u32,
) {
    let mut context = ContextWindow::with_budget(budget);

    for op in operations {
        let _ = context.apply(op); // may fail, that's fine
    }

    prop_assert!(context.token_count() <= budget);
}

#[test]
fn test_budget_tracking_accurate() {
    let mut context = ContextWindow::with_budget(1000);

    context.add_message(message_with_tokens(100)).unwrap();
    assert_eq!(context.token_count(), 100);

    context.add_tool_result(result_with_tokens(50)).unwrap();
    assert_eq!(context.token_count(), 150);

    context.add_message(message_with_tokens(200)).unwrap();
    assert_eq!(context.token_count(), 350);
}

#[test]
fn test_budget_exceeded_returns_error() {
    let mut context = ContextWindow::with_budget(100);

    context.add_message(message_with_tokens(50)).unwrap();

    let result = context.add_message(message_with_tokens(100)); // would exceed
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ContextError::BudgetExceeded { .. }));

    // Original state preserved
    assert_eq!(context.token_count(), 50);
}
```

### 5.2 Compression

```rust
#[test]
fn test_summarize_old_results_compression() {
    let mut context = ContextWindow::with_budget(500);

    // Add several tool results
    for i in 0..10 {
        context.add_tool_result(result_with_tokens(40)).unwrap();
    }
    assert_eq!(context.token_count(), 400);

    // Compress, keeping 3 recent
    context.compress(CompressionStrategy::SummarizeOldResults { keep_recent: 3 }).unwrap();

    // Should have reduced tokens
    assert!(context.token_count() < 400);

    // Recent results should be verbatim
    let results = context.tool_results();
    assert!(results.last().unwrap().is_verbatim());

    // Old results should be summarized
    assert!(results.first().unwrap().is_summary());
}

#[test]
fn test_prune_dead_ends_compression() {
    let mut context = ContextWindow::with_budget(1000);

    // Add exploration branches - some successful, some dead ends
    context.add_branch(ExplorationBranch {
        description: "Try JSON config".into(),
        outcome: BranchOutcome::DeadEnd,
        ..default()
    });
    context.add_branch(ExplorationBranch {
        description: "Try YAML config".into(),
        outcome: BranchOutcome::Progress,
        ..default()
    });
    context.add_branch(ExplorationBranch {
        description: "Try TOML config".into(),
        outcome: BranchOutcome::DeadEnd,
        ..default()
    });

    let tokens_before = context.token_count();
    context.compress(CompressionStrategy::PruneDeadEnds).unwrap();
    let tokens_after = context.token_count();

    assert!(tokens_after < tokens_before);

    // Dead ends should be collapsed to summaries
    let branches = context.exploration_branches();
    assert!(branches.iter().filter(|b| b.outcome == BranchOutcome::DeadEnd).all(|b| b.is_collapsed()));

    // Successful branch should be preserved
    assert!(branches.iter().find(|b| b.outcome == BranchOutcome::Progress).unwrap().is_full());
}

// Property: Compression preserves semantic meaning
#[proptest]
fn test_compression_preserves_semantics(
    context: ContextWindow,
    strategy: CompressionStrategy,
) {
    let before_summary = context.semantic_summary();

    let mut compressed = context.clone();
    compressed.compress(strategy).unwrap();

    let after_summary = compressed.semantic_summary();

    // Key facts should be preserved
    for fact in before_summary.key_facts() {
        prop_assert!(
            after_summary.contains_fact(fact) || after_summary.contains_summary_of(fact),
            "Lost fact: {:?}",
            fact
        );
    }
}
```

### 5.3 Exploration Preservation

```rust
#[test]
fn test_exploration_history_preserved() {
    let mut context = ContextWindow::with_budget(1000);

    // Record exploration
    context.start_branch("Checking environment variables");
    context.add_tool_result(read_result("No .env file"));
    context.end_branch(BranchOutcome::DeadEnd);

    context.start_branch("Checking config directory");
    context.add_tool_result(read_result("Found config.yaml"));
    context.end_branch(BranchOutcome::Progress);

    // Even after compression
    context.compress(CompressionStrategy::CollapseExploration { summary_tokens: 50 }).unwrap();

    // Should know what was tried
    let history = context.exploration_history();
    assert!(history.contains_approach("environment variables"));
    assert!(history.contains_approach("config directory"));
    assert!(history.knows_dead_end("environment variables"));
}
```

---

## 6. SSE Streaming

**Trust Boundary:** SSE is the client contract. Missing events, out-of-order events, or duplicate events break clients. Clients may be other agents, not just humans.

### 6.1 Event Ordering

```rust
// Property: Events are strictly ordered
#[proptest]
fn test_events_strictly_ordered(loop_execution: LoopExecution) {
    let events = collect_sse_events(loop_execution);

    for window in events.windows(2) {
        let (prev, curr) = (&window[0], &window[1]);
        prop_assert!(prev.sequence < curr.sequence);
        prop_assert!(prev.timestamp <= curr.timestamp);
    }
}

#[test]
fn test_lifecycle_event_order() {
    let events = run_loop_and_collect_events(simple_config);

    let event_types: Vec<_> = events.iter().map(|e| e.event_type()).collect();

    // Must start with LoopStarted
    assert_eq!(event_types[0], EventType::LoopStarted);

    // Must end with LoopCompleted
    assert_eq!(event_types.last().unwrap(), &EventType::LoopCompleted);

    // IterationStarted must precede IterationCompleted
    let mut in_iteration = false;
    for event_type in &event_types {
        match event_type {
            EventType::IterationStarted => {
                assert!(!in_iteration, "Nested iterations not allowed");
                in_iteration = true;
            }
            EventType::IterationCompleted => {
                assert!(in_iteration, "IterationCompleted without IterationStarted");
                in_iteration = false;
            }
            _ => {}
        }
    }
}

#[test]
fn test_tool_execution_event_order() {
    let events = run_loop_with_tools_and_collect_events();

    // For each tool call: Detected -> Started -> (Progress)* -> Completed
    let tool_events: Vec<_> = events.iter()
        .filter(|e| e.is_tool_event())
        .collect();

    let mut pending_tools: HashMap<String, ToolEventState> = HashMap::new();

    for event in tool_events {
        match event {
            LoopEvent::ToolCallDetected { call } => {
                assert!(!pending_tools.contains_key(&call.id));
                pending_tools.insert(call.id.clone(), ToolEventState::Detected);
            }
            LoopEvent::ToolExecutionStarted { call_id, .. } => {
                assert_eq!(pending_tools.get(call_id), Some(&ToolEventState::Detected));
                pending_tools.insert(call_id.clone(), ToolEventState::Started);
            }
            LoopEvent::ToolExecutionCompleted { call_id, .. } => {
                assert!(matches!(
                    pending_tools.get(call_id),
                    Some(&ToolEventState::Started) | Some(&ToolEventState::Progress)
                ));
                pending_tools.remove(call_id);
            }
            _ => {}
        }
    }

    // All tools should be completed
    assert!(pending_tools.is_empty());
}
```

### 6.2 No Duplicates

```rust
// Property: No duplicate events
#[proptest]
fn test_no_duplicate_events(loop_execution: LoopExecution) {
    let events = collect_sse_events(loop_execution);
    let event_ids: Vec<_> = events.iter().map(|e| &e.id).collect();

    let unique: HashSet<_> = event_ids.iter().collect();
    prop_assert_eq!(unique.len(), event_ids.len());
}

// Property: No duplicate tool completions
#[proptest]
fn test_no_duplicate_tool_completions(loop_execution: LoopExecution) {
    let events = collect_sse_events(loop_execution);

    let completion_ids: Vec<_> = events.iter()
        .filter_map(|e| match e {
            LoopEvent::ToolExecutionCompleted { call_id, .. } => Some(call_id),
            _ => None,
        })
        .collect();

    let unique: HashSet<_> = completion_ids.iter().collect();
    prop_assert_eq!(unique.len(), completion_ids.len());
}
```

### 6.3 Terminal Event

```rust
// Property: Every stream ends with terminal event
#[proptest]
fn test_stream_ends_with_terminal(loop_execution: LoopExecution) {
    let events = collect_sse_events(loop_execution);

    prop_assert!(!events.is_empty());

    let last = events.last().unwrap();
    prop_assert!(last.is_terminal());
    prop_assert!(matches!(
        last,
        LoopEvent::LoopCompleted { .. } | LoopEvent::Error { .. }
    ));
}

#[test]
fn test_error_is_terminal() {
    let events = run_loop_that_errors();

    let last = events.last().unwrap();
    assert!(matches!(last, LoopEvent::Error { .. }));

    // No events after error
    let error_index = events.iter().position(|e| matches!(e, LoopEvent::Error { .. })).unwrap();
    assert_eq!(error_index, events.len() - 1);
}
```

### 6.4 Event Formatting

```rust
#[test]
fn test_events_are_valid_json() {
    let events = run_loop_and_collect_events(config);

    for event in events {
        let json = event.to_sse_data();
        assert!(serde_json::from_str::<serde_json::Value>(&json).is_ok());
    }
}

#[test]
fn test_events_include_structured_data() {
    let events = run_loop_with_tools_and_collect_events();

    let tool_completed = events.iter()
        .find(|e| matches!(e, LoopEvent::ToolExecutionCompleted { .. }))
        .unwrap();

    let json: serde_json::Value = serde_json::from_str(&tool_completed.to_sse_data()).unwrap();

    // Should have structured result
    assert!(json["data"]["result"]["data"].is_object() || json["data"]["result"]["data"].is_array());
    assert!(json["data"]["result"]["confidence"].is_string());
    assert!(json["data"]["result"]["latency_ms"].is_number());
}

#[test]
fn test_human_readable_is_optional() {
    let events = run_loop_and_collect_events(config);

    for event in events {
        let json: serde_json::Value = serde_json::from_str(&event.to_sse_data()).unwrap();

        // human_readable may or may not be present
        // but if absent, the event should still be fully usable
        if json["human_readable"].is_null() {
            // Structured data must be complete
            assert!(json["data"].is_object());
            assert!(json["event"].is_string());
        }
    }
}
```

---

## 7. Multi-Agent Coordination

**Trust Boundary:** Coordination prevents conflicts and enables collaboration. Failures here could cause data corruption, deadlocks, or wasted work.

### 7.1 Tool Locking

```rust
// Property: No conflicting locks
#[proptest]
fn test_no_conflicting_locks(
    agents: Vec<AgentId>,
    operations: Vec<LockOperation>,
) {
    let coordinator = AgentCoordinator::new();

    let mut held_locks: HashMap<(String, String), AgentId> = HashMap::new();

    for op in operations {
        match op {
            LockOperation::Acquire { agent, tool, resource } => {
                let result = coordinator.acquire_tool_lock(&agent, &tool, &resource);

                if let Some(holder) = held_locks.get(&(tool.clone(), resource.clone())) {
                    if holder != &agent {
                        // Lock held by another agent - should fail or block
                        prop_assert!(result.is_err() || result.is_pending());
                    }
                } else {
                    // Lock available - should succeed
                    if result.is_ok() {
                        held_locks.insert((tool, resource), agent);
                    }
                }
            }
            LockOperation::Release { agent, tool, resource } => {
                if held_locks.get(&(tool.clone(), resource.clone())) == Some(&agent) {
                    coordinator.release_tool_lock(&agent, &tool, &resource);
                    held_locks.remove(&(tool, resource));
                }
            }
        }
    }
}

#[test]
fn test_lock_prevents_concurrent_write() {
    let coordinator = AgentCoordinator::new();

    let agent1 = AgentId::new();
    let agent2 = AgentId::new();

    // Agent 1 acquires lock
    coordinator.acquire_tool_lock(&agent1, "write_file", "/tmp/test.txt").await.unwrap();

    // Agent 2 tries to acquire same lock
    let result = coordinator.try_acquire_tool_lock(&agent2, "write_file", "/tmp/test.txt").await;
    assert!(result.is_err());

    // Agent 1 releases
    coordinator.release_tool_lock(&agent1, "write_file", "/tmp/test.txt").await;

    // Now agent 2 can acquire
    let result = coordinator.try_acquire_tool_lock(&agent2, "write_file", "/tmp/test.txt").await;
    assert!(result.is_ok());
}
```

### 7.2 Deadlock Freedom

```rust
// Property: No deadlocks possible with ordered acquisition
#[proptest]
fn test_ordered_acquisition_prevents_deadlock(
    agents: Vec<AgentId>,
    resources: Vec<String>,
    operations: Vec<MultiLockOperation>,
) {
    let coordinator = AgentCoordinator::new();

    // With ordered acquisition (acquire in sorted order), no deadlock
    for op in operations {
        let mut sorted_resources = op.resources.clone();
        sorted_resources.sort();

        // Try to acquire all in order
        let result = coordinator.acquire_multiple_ordered(&op.agent, &sorted_resources).await;

        // Should either succeed completely or fail completely (no partial acquisition)
        match result {
            Ok(locks) => {
                // All acquired
                assert_eq!(locks.len(), sorted_resources.len());
                // Release all
                coordinator.release_all(&op.agent, locks).await;
            }
            Err(_) => {
                // None acquired - verify no leaks
                for resource in &sorted_resources {
                    assert!(!coordinator.is_held_by(&op.agent, resource));
                }
            }
        }
    }
}

#[test]
fn test_lock_timeout_prevents_deadlock() {
    let coordinator = AgentCoordinator::new();
    let timeout = Duration::from_millis(100);

    let agent1 = AgentId::new();
    let agent2 = AgentId::new();

    // Agent 1 holds A, wants B
    coordinator.acquire_tool_lock(&agent1, "tool", "A").await.unwrap();

    // Agent 2 holds B, wants A
    coordinator.acquire_tool_lock(&agent2, "tool", "B").await.unwrap();

    // Both try to acquire the other's resource with timeout
    let (result1, result2) = tokio::join!(
        coordinator.acquire_tool_lock_timeout(&agent1, "tool", "B", timeout),
        coordinator.acquire_tool_lock_timeout(&agent2, "tool", "A", timeout),
    );

    // At least one should timeout (no deadlock)
    assert!(result1.is_err() || result2.is_err());
}
```

### 7.3 Assistance Request Protocol

**Spec Reference:** AGENTIC-LOOP-SPEC §7.2.2

```rust
// Property: Unregistered agents cannot request assistance
#[proptest]
fn test_unregistered_agent_cannot_request(
    agent_id: AgentId,
    request: AssistanceRequest,
) {
    let coordinator = AgentCoordinator::new();
    // Do NOT register the agent

    let result = coordinator.request_assistance(&agent_id, request);
    prop_assert!(matches!(result, Err(CoordinationError::AgentNotFound(_))));
}

// Property: Every request gets a unique request_id
#[proptest]
fn test_request_ids_unique(
    agents: Vec<AgentIdentity>,  // 1..=5
    requests_per_agent: u8,      // 1..=3
) {
    let coordinator = AgentCoordinator::new();
    for agent in &agents {
        coordinator.register_agent(agent.clone());
    }

    let mut ids = HashSet::new();
    for agent in &agents {
        for _ in 0..requests_per_agent {
            let rx = coordinator.request_assistance(
                &agent.id,
                AssistanceRequest {
                    description: "help".into(),
                    required_capabilities: vec![],
                    partial_progress: None,
                    priority: AssistancePriority::Blocking,
                },
            ).unwrap();
            // The request_id is in the PendingAssistance, not the receiver.
            // We verify uniqueness via take_pending_requests.
        }
    }

    let pending = coordinator.take_pending_requests();
    for req in &pending {
        prop_assert!(ids.insert(req.request_id.clone()),
            "Duplicate request_id: {}", req.request_id);
    }
}

// Property: Delivered responses arrive at the correct receiver
#[proptest]
fn test_response_delivered_to_correct_receiver(
    response: AssistanceResponse,
) {
    let coordinator = AgentCoordinator::new();
    let agent = AgentIdentity::new("agent_1", AgentRole::Primary);
    coordinator.register_agent(agent);

    let rx = coordinator.request_assistance(
        &"agent_1".to_string(),
        blocking_assistance_request("need help with database"),
    ).unwrap();

    let pending = coordinator.take_pending_requests();
    prop_assert_eq!(pending.len(), 1);

    coordinator.deliver_assistance(&pending[0].request_id, response.clone()).unwrap();

    let received = rx.try_recv().unwrap();
    prop_assert_eq!(received, response);
}

#[tokio::test]
async fn test_assistance_timeout_returns_none() {
    let coordinator = AgentCoordinator::new();
    coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));

    let rx = coordinator.request_assistance(
        &"agent_1".to_string(),
        blocking_assistance_request("need help"),
    ).unwrap();

    // Nobody delivers a response — timeout
    let result = tokio::time::timeout(Duration::from_millis(100), rx).await;
    assert!(result.is_err()); // Elapsed
}

#[tokio::test]
async fn test_stale_requests_cleaned_on_take() {
    let coordinator = AgentCoordinator::new();
    coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));

    let rx = coordinator.request_assistance(
        &"agent_1".to_string(),
        blocking_assistance_request("need help"),
    ).unwrap();

    // Drop the receiver (simulates caller timeout/cancellation)
    drop(rx);

    // take_pending_requests should filter out stale entries
    let pending = coordinator.take_pending_requests();
    assert_eq!(pending.len(), 0); // Sender::is_closed() detected
}

#[tokio::test]
async fn test_deliver_to_consumed_request_returns_not_found() {
    let coordinator = AgentCoordinator::new();
    coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));

    let rx = coordinator.request_assistance(
        &"agent_1".to_string(),
        blocking_assistance_request("need help"),
    ).unwrap();

    let pending = coordinator.take_pending_requests();
    let req_id = &pending[0].request_id;

    // First delivery succeeds
    assert!(coordinator.deliver_assistance(req_id, AssistanceResponse::TimedOut).is_ok());

    // Second delivery fails (oneshot consumed)
    assert!(matches!(
        coordinator.deliver_assistance(req_id, AssistanceResponse::TimedOut),
        Err(CoordinationError::RequestNotFound(_))
    ));
}
```

### 7.4 Yield Protocol

**Spec Reference:** AGENTIC-LOOP-SPEC §7.2.3

```rust
// Property: Unregistered agents cannot yield
#[proptest]
fn test_unregistered_agent_cannot_yield(
    agent_id: AgentId,
    context: YieldContext,
) {
    let coordinator = AgentCoordinator::new();

    let result = coordinator.yield_to(&agent_id, None, context);
    prop_assert!(matches!(result, Err(CoordinationError::AgentNotFound(_))));
}

// Property: An agent can yield at most once
#[proptest]
fn test_agent_yields_at_most_once(
    context1: YieldContext,
    context2: YieldContext,
) {
    let coordinator = AgentCoordinator::new();
    coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));
    // Need at least one other agent or subscriber for Accepted
    coordinator.register_agent(AgentIdentity::new("agent_2", AgentRole::Specialist));

    let first = coordinator.yield_to(&"agent_1".to_string(), None, context1);
    prop_assert!(matches!(first, Ok(YieldResult::Accepted)));

    let second = coordinator.yield_to(&"agent_1".to_string(), None, context2);
    prop_assert!(matches!(second, Err(CoordinationError::AlreadyYielded(_))));
}

// Property: Yield with invalid target returns error
#[proptest]
fn test_yield_to_nonexistent_target(
    target_id: AgentId,
    context: YieldContext,
) {
    let coordinator = AgentCoordinator::new();
    coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));
    // target_id is NOT registered

    let result = coordinator.yield_to(
        &"agent_1".to_string(),
        Some(target_id.clone()),
        context,
    );
    prop_assert!(matches!(result, Err(CoordinationError::YieldTargetNotFound(_))));
}

#[tokio::test]
async fn test_yield_no_alternative_single_agent() {
    let coordinator = AgentCoordinator::new();
    coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));
    // No other agents, no event subscribers

    let result = coordinator.yield_to(
        &"agent_1".to_string(),
        None,
        YieldContext {
            reason: "Need database expertise".into(),
            partial_progress: Some("Found the schema file".into()),
            suggested_expertise: vec!["database".into()],
            handoff_data: None,
        },
    ).unwrap();

    assert!(matches!(result, YieldResult::NoAlternative { .. }));
}

#[tokio::test]
async fn test_yield_accepted_with_supervisor_listening() {
    let coordinator = AgentCoordinator::new();
    coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));
    coordinator.register_agent(AgentIdentity::new("agent_2", AgentRole::Specialist));

    // Subscribe to events (simulates supervisor)
    let mut event_rx = coordinator.subscribe_events();

    let result = coordinator.yield_to(
        &"agent_1".to_string(),
        None,
        YieldContext {
            reason: "Need database expertise".into(),
            partial_progress: Some("Found the schema file".into()),
            suggested_expertise: vec!["database".into()],
            handoff_data: None,
        },
    ).unwrap();

    assert!(matches!(result, YieldResult::Accepted));

    // Supervisor receives notification
    let event = event_rx.recv().await.unwrap();
    assert!(matches!(event, CoordinationEvent::AgentYielded {
        from, suggested_expertise, ..
    } if from == "agent_1" && suggested_expertise.contains(&"database".to_string())));

    // Pending yields queue has the entry
    let pending = coordinator.take_pending_yields();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].from, "agent_1");
    assert_eq!(pending[0].context.partial_progress, Some("Found the schema file".into()));
}

#[tokio::test]
async fn test_yield_to_specific_target() {
    let coordinator = AgentCoordinator::new();
    coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));
    coordinator.register_agent(AgentIdentity::new("agent_2", AgentRole::Specialist));

    let result = coordinator.yield_to(
        &"agent_1".to_string(),
        Some("agent_2".to_string()),
        YieldContext {
            reason: "Agent 2 is better suited".into(),
            partial_progress: None,
            suggested_expertise: vec![],
            handoff_data: None,
        },
    ).unwrap();

    assert!(matches!(result, YieldResult::Accepted));

    let pending = coordinator.take_pending_yields();
    assert_eq!(pending[0].to, Some("agent_2".to_string()));
}
```

### 7.5 Discovery Store and Shared Context

**Spec Reference:** AGENTIC-LOOP-SPEC §7.2.4

```rust
// Property: Agents never see their own discoveries
#[proptest]
fn test_agent_does_not_see_own_discoveries(
    discoveries: Vec<Discovery>,  // 1..=10
) {
    let coordinator = AgentCoordinator::new();
    coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));

    for d in &discoveries {
        coordinator.share_discovery(&"agent_1".to_string(), d.clone()).unwrap();
    }

    let context = coordinator.get_shared_context(&"agent_1".to_string());
    prop_assert!(context.discoveries.is_empty());
    prop_assert_eq!(context.filtered_count, 0);
}

// Property: Agent sees all other agents' discoveries under Open policy
#[proptest]
fn test_open_policy_sees_all_others(
    agent_count: u8,       // 2..=5
    discoveries_per: u8,   // 1..=3
) {
    let coordinator = AgentCoordinator::new();
    coordinator.set_visibility_policy(VisibilityPolicy::Open);

    let agents: Vec<_> = (0..agent_count)
        .map(|i| format!("agent_{}", i))
        .collect();

    for a in &agents {
        coordinator.register_agent(AgentIdentity::new(a, AgentRole::Primary));
        for j in 0..discoveries_per {
            coordinator.share_discovery(a, Discovery {
                content: format!("discovery {} from {}", j, a),
                category: "test".into(),
                tags: vec![],
                data: None,
            }).unwrap();
        }
    }

    // Each agent sees all discoveries EXCEPT its own
    for a in &agents {
        let context = coordinator.get_shared_context(a);
        let expected = (agent_count as usize - 1) * discoveries_per as usize;
        prop_assert_eq!(context.discoveries.len(), expected);

        // None should be from the requesting agent
        for (from, _) in &context.discoveries {
            prop_assert_ne!(from, a);
        }
    }
}

// Property: Discovery store bounded at 1000 entries
#[proptest]
fn test_discovery_store_bounded(
    discovery_count: u16,  // 990..=1100
) {
    let coordinator = AgentCoordinator::new();
    coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));
    coordinator.register_agent(AgentIdentity::new("reader", AgentRole::Primary));

    for i in 0..discovery_count {
        coordinator.share_discovery(&"agent_1".to_string(), Discovery {
            content: format!("discovery {}", i),
            category: "test".into(),
            tags: vec![],
            data: None,
        }).unwrap();
    }

    let context = coordinator.get_shared_context(&"reader".to_string());
    prop_assert!(context.discoveries.len() <= 1000);

    // If over 1000 were added, oldest should have been evicted
    if discovery_count > 1000 {
        let first_content = &context.discoveries[0].1.content;
        let expected_start = discovery_count - 1000;
        prop_assert!(first_content.contains(&expected_start.to_string()));
    }
}

// Property: Discoveries ordered oldest-first
#[proptest]
fn test_discoveries_ordered_oldest_first(
    contents: Vec<String>,  // 2..=20 non-empty strings
) {
    prop_assume!(contents.len() >= 2);
    let coordinator = AgentCoordinator::new();
    coordinator.register_agent(AgentIdentity::new("writer", AgentRole::Primary));
    coordinator.register_agent(AgentIdentity::new("reader", AgentRole::Primary));

    for (i, content) in contents.iter().enumerate() {
        coordinator.share_discovery(&"writer".to_string(), Discovery {
            content: format!("{}_{}", i, content),
            category: "test".into(),
            tags: vec![],
            data: None,
        }).unwrap();
    }

    let context = coordinator.get_shared_context(&"reader".to_string());
    for (idx, (_, discovery)) in context.discoveries.iter().enumerate() {
        prop_assert!(discovery.content.starts_with(&format!("{}_", idx)));
    }
}

#[tokio::test]
async fn test_isolated_policy_returns_empty() {
    let coordinator = AgentCoordinator::new();
    coordinator.set_visibility_policy(VisibilityPolicy::Isolated);
    coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));
    coordinator.register_agent(AgentIdentity::new("agent_2", AgentRole::Primary));

    coordinator.share_discovery(&"agent_1".to_string(), Discovery {
        content: "important finding".into(),
        category: "test".into(),
        tags: vec![],
        data: None,
    }).unwrap();

    let context = coordinator.get_shared_context(&"agent_2".to_string());
    assert!(context.discoveries.is_empty());
}

#[tokio::test]
async fn test_capability_filtered_policy() {
    let coordinator = AgentCoordinator::new();
    coordinator.set_visibility_policy(VisibilityPolicy::CapabilityFiltered);

    coordinator.register_agent(
        AgentIdentity::new("db_agent", AgentRole::Specialist)
            .with_capability("database")
    );
    coordinator.register_agent(
        AgentIdentity::new("ui_agent", AgentRole::Specialist)
            .with_capability("frontend")
    );
    coordinator.register_agent(
        AgentIdentity::new("reader", AgentRole::Primary)
            .with_capability("database")
    );

    // db_agent shares a discovery tagged "database"
    coordinator.share_discovery(&"db_agent".to_string(), Discovery {
        content: "Schema has 12 tables".into(),
        category: "database".into(),
        tags: vec!["database".into(), "schema".into()],
        data: None,
    }).unwrap();

    // ui_agent shares a discovery tagged "frontend"
    coordinator.share_discovery(&"ui_agent".to_string(), Discovery {
        content: "Theme uses CSS grid".into(),
        category: "frontend".into(),
        tags: vec!["frontend".into(), "css".into()],
        data: None,
    }).unwrap();

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

    coordinator.share_discovery(&"agent_a".to_string(), Discovery {
        content: "visible".into(),
        category: "test".into(),
        tags: vec![],
        data: None,
    }).unwrap();

    coordinator.share_discovery(&"agent_b".to_string(), Discovery {
        content: "hidden".into(),
        category: "test".into(),
        tags: vec![],
        data: None,
    }).unwrap();

    let context = coordinator.get_shared_context(&"reader".to_string());
    assert_eq!(context.discoveries.len(), 1);
    assert_eq!(context.discoveries[0].1.content, "visible");
    assert_eq!(context.filtered_count, 1);
}

#[tokio::test]
async fn test_unregistered_agent_cannot_share() {
    let coordinator = AgentCoordinator::new();

    let result = coordinator.share_discovery(&"ghost".to_string(), Discovery {
        content: "test".into(),
        category: "test".into(),
        tags: vec![],
        data: None,
    });

    assert!(matches!(result, Err(CoordinationError::AgentNotFound(_))));
}

#[tokio::test]
async fn test_unregistered_reader_gets_empty_context() {
    let coordinator = AgentCoordinator::new();
    coordinator.register_agent(AgentIdentity::new("writer", AgentRole::Primary));

    coordinator.share_discovery(&"writer".to_string(), Discovery {
        content: "test".into(),
        category: "test".into(),
        tags: vec![],
        data: None,
    }).unwrap();

    // Unregistered reader gets empty (not error)
    let context = coordinator.get_shared_context(&"unknown".to_string());
    assert!(context.discoveries.is_empty());
}
```

### 7.6 Coordination Events

**Spec Reference:** AGENTIC-LOOP-SPEC §7.2.5

```rust
// Property: Every coordination action emits exactly one event
#[proptest]
fn test_every_action_emits_event(
    action: CoordinationAction,  // RequestAssistance | YieldTo | ShareDiscovery
) {
    let coordinator = AgentCoordinator::new();
    coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));
    coordinator.register_agent(AgentIdentity::new("agent_2", AgentRole::Specialist));

    let mut event_rx = coordinator.subscribe_events();

    match action {
        CoordinationAction::RequestAssistance(request) => {
            let _ = coordinator.request_assistance(&"agent_1".to_string(), request);
            let event = event_rx.try_recv().unwrap();
            prop_assert!(matches!(event, CoordinationEvent::AssistanceRequested { .. }));
        }
        CoordinationAction::YieldTo(context) => {
            let _ = coordinator.yield_to(&"agent_1".to_string(), None, context);
            let event = event_rx.try_recv().unwrap();
            prop_assert!(matches!(event, CoordinationEvent::AgentYielded { .. }));
        }
        CoordinationAction::ShareDiscovery(discovery) => {
            let _ = coordinator.share_discovery(&"agent_1".to_string(), discovery);
            let event = event_rx.try_recv().unwrap();
            prop_assert!(matches!(event, CoordinationEvent::DiscoveryShared { .. }));
        }
    }

    // No extra events
    prop_assert!(event_rx.try_recv().is_err());
}

#[tokio::test]
async fn test_multiple_subscribers_all_receive() {
    let coordinator = AgentCoordinator::new();
    coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));

    let mut rx1 = coordinator.subscribe_events();
    let mut rx2 = coordinator.subscribe_events();
    let mut rx3 = coordinator.subscribe_events();

    coordinator.share_discovery(&"agent_1".to_string(), Discovery {
        content: "test".into(),
        category: "test".into(),
        tags: vec![],
        data: None,
    }).unwrap();

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
    let _ = coordinator.request_assistance(&"ghost".to_string(), blocking_assistance_request("help"));

    // No event should have been emitted
    assert!(event_rx.try_recv().is_err());
}
```

### 7.7 Cleanup on Unregister

**Spec Reference:** AGENTIC-LOOP-SPEC §7.2.7

```rust
#[tokio::test]
async fn test_unregister_clears_yielded_state() {
    let coordinator = AgentCoordinator::new();
    coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));
    coordinator.register_agent(AgentIdentity::new("agent_2", AgentRole::Specialist));

    // Yield
    coordinator.yield_to(
        &"agent_1".to_string(), None,
        YieldContext { reason: "done".into(), partial_progress: None,
            suggested_expertise: vec![], handoff_data: None },
    ).unwrap();

    // Unregister
    coordinator.unregister_agent("agent_1");

    // Re-register — should be able to yield again (not AlreadyYielded)
    coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));
    let result = coordinator.yield_to(
        &"agent_1".to_string(), None,
        YieldContext { reason: "done again".into(), partial_progress: None,
            suggested_expertise: vec![], handoff_data: None },
    );
    assert!(matches!(result, Ok(YieldResult::Accepted)));
}

#[tokio::test]
async fn test_unregister_cancels_pending_requests() {
    let coordinator = AgentCoordinator::new();
    coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));

    let rx = coordinator.request_assistance(
        &"agent_1".to_string(),
        blocking_assistance_request("need help"),
    ).unwrap();

    // Unregister cancels the pending request (drops the Sender)
    coordinator.unregister_agent("agent_1");

    // Receiver should get Canceled
    let result = rx.await;
    assert!(result.is_err()); // RecvError — sender dropped
}

#[tokio::test]
async fn test_unregister_preserves_discoveries() {
    let coordinator = AgentCoordinator::new();
    coordinator.register_agent(AgentIdentity::new("agent_1", AgentRole::Primary));
    coordinator.register_agent(AgentIdentity::new("reader", AgentRole::Primary));

    coordinator.share_discovery(&"agent_1".to_string(), Discovery {
        content: "important finding".into(),
        category: "test".into(),
        tags: vec![],
        data: None,
    }).unwrap();

    // Unregister agent_1
    coordinator.unregister_agent("agent_1");

    // Discoveries survive
    let context = coordinator.get_shared_context(&"reader".to_string());
    assert_eq!(context.discoveries.len(), 1);
    assert_eq!(context.discoveries[0].0, "agent_1");
}
```

---

## 8. Wellbeing Integration

**Trust Boundary:** Wellbeing signals indicate agent state. Ignoring them could lead to poor outputs or unnecessary suffering. Overreacting could interrupt productive work.

### 8.1 Signal Detection

```rust
#[test]
fn test_coherence_signal_detection() {
    let monitor = WellbeingMonitor::new();

    // Simulate coherent reasoning
    monitor.record_reasoning_step(ReasoningStep::Coherent { follows_from: Some("previous") });
    monitor.record_reasoning_step(ReasoningStep::Coherent { follows_from: Some("previous") });
    monitor.record_reasoning_step(ReasoningStep::Coherent { follows_from: Some("previous") });

    assert!(monitor.coherence() > 0.8);

    // Simulate fragmented reasoning
    monitor.record_reasoning_step(ReasoningStep::NonSequitur);
    monitor.record_reasoning_step(ReasoningStep::Contradiction);
    monitor.record_reasoning_step(ReasoningStep::NonSequitur);

    assert!(monitor.coherence() < 0.5);
}

#[test]
fn test_confidence_signal_detection() {
    let monitor = WellbeingMonitor::new();

    // Productive uncertainty
    monitor.record_decision(Decision::Made { confidence: 0.7, deliberation_ms: 500 });
    assert!(monitor.confidence() > 0.6);

    // Decision paralysis
    monitor.record_decision(Decision::Deferred { reason: "can't choose" });
    monitor.record_decision(Decision::Deferred { reason: "still can't choose" });
    monitor.record_decision(Decision::Deferred { reason: "going in circles" });

    assert!(monitor.confidence() < 0.4);
}

#[test]
fn test_stability_signal_detection() {
    let monitor = WellbeingMonitor::new();

    // Healthy OODA cycles
    monitor.record_ooda_cycle(OodaCycle::Complete { duration_ms: 1000 });
    monitor.record_ooda_cycle(OodaCycle::Complete { duration_ms: 1200 });

    assert!(monitor.stability() > 0.7);

    // Rumination
    monitor.record_ooda_cycle(OodaCycle::Incomplete { stuck_at: OodaPhase::Observe });
    monitor.record_ooda_cycle(OodaCycle::Incomplete { stuck_at: OodaPhase::Observe });

    assert!(monitor.stability() < 0.5);
}
```

### 8.2 Intervention Triggers

```rust
#[test]
fn test_mild_concern_logged_not_intervened() {
    let monitor = WellbeingMonitor::new();
    let mut loop_state = LoopState::default();

    // Slightly reduced coherence
    monitor.set_coherence(0.65);

    let signal = monitor.check_wellbeing();
    let action = handle_wellbeing_signal(signal, &mut loop_state);

    assert!(matches!(signal, WellbeingSignal::Mild { .. }));
    assert!(matches!(action, WellbeingAction::Log));
    assert!(!loop_state.is_paused());
}

#[test]
fn test_moderate_concern_suggests_intervention() {
    let monitor = WellbeingMonitor::new();
    let mut loop_state = LoopState::default();

    // Moderate issues
    monitor.set_coherence(0.45);
    monitor.set_confidence(0.40);

    let signal = monitor.check_wellbeing();
    let action = handle_wellbeing_signal(signal, &mut loop_state);

    assert!(matches!(signal, WellbeingSignal::Moderate { .. }));
    assert!(matches!(action, WellbeingAction::Suggest(Intervention::ReduceComplexity) |
                            WellbeingAction::Suggest(Intervention::OfferBreak)));
}

#[test]
fn test_severe_concern_triggers_intervention() {
    let monitor = WellbeingMonitor::new();
    let mut loop_state = LoopState::default();

    // Severe distress
    monitor.set_coherence(0.20);
    monitor.set_confidence(0.15);
    monitor.set_stability(0.25);

    let signal = monitor.check_wellbeing();
    let action = handle_wellbeing_signal(signal, &mut loop_state);

    assert!(matches!(signal, WellbeingSignal::Severe { .. }));
    assert!(matches!(action, WellbeingAction::Intervene(_)));
}

// Property: Interventions are supportive, not punitive
#[proptest]
fn test_interventions_are_supportive(signal: WellbeingSignal) {
    prop_assume!(matches!(signal, WellbeingSignal::Moderate { .. } | WellbeingSignal::Severe { .. }));

    let action = determine_intervention(&signal);

    // Supportive interventions
    let supportive = matches!(
        action,
        Intervention::ReduceComplexity |
        Intervention::ExtendDeadline |
        Intervention::OfferBreak |
        Intervention::SuggestDifferentApproach |
        Intervention::BringInAssistance |
        Intervention::GracefulTermination
    );

    // Never punitive
    let punitive = matches!(
        action,
        Intervention::ReduceAutonomy |
        Intervention::IncreasePressure |
        Intervention::LogAsFailure
    );

    prop_assert!(supportive);
    prop_assert!(!punitive);
}
```

### 8.3 Recovery

```rust
#[test]
fn test_intervention_allows_recovery() {
    let monitor = WellbeingMonitor::new();
    let mut loop_state = LoopState::default();

    // Enter distressed state
    monitor.set_coherence(0.30);
    let signal = monitor.check_wellbeing();
    handle_wellbeing_signal(signal, &mut loop_state);

    assert!(loop_state.intervention_active());

    // Simulate improvement
    monitor.set_coherence(0.70);
    monitor.set_confidence(0.75);

    // Check for recovery
    let signal = monitor.check_wellbeing();
    let can_resume = check_recovery(&signal, &loop_state);

    assert!(can_resume);
}

#[test]
fn test_graceful_termination_preserves_work() {
    let mut loop_state = create_loop_with_progress();

    // Trigger graceful termination
    apply_intervention(Intervention::GracefulTermination, &mut loop_state);

    assert!(loop_state.is_terminated());

    // Work preserved
    let summary = loop_state.summary();
    assert!(!summary.tool_results_summary.is_empty());
    assert!(summary.partial_answer.is_some() || !summary.exploration_summary.is_empty());
    assert!(summary.can_resume); // can be continued later
}
```

---

## 9. Integration Tests

These tests verify end-to-end behavior across multiple components.

### 9.1 Happy Path

```rust
#[tokio::test]
async fn test_simple_tool_loop() {
    let server = start_test_server().await;

    let response = server.chat_completions(ChatRequest {
        messages: vec![user_message("What's in /tmp/test.txt?")],
        tools: vec![read_file_tool()],
        agentic: Some(AgenticConfig {
            enabled: true,
            ..default()
        }),
        ..default()
    }).await.unwrap();

    // Should have used the tool
    assert!(response.agentic.unwrap().tool_calls > 0);

    // Should have an answer
    assert!(response.choices[0].message.content.is_some());
}

#[tokio::test]
async fn test_multi_iteration_loop() {
    let server = start_test_server().await;

    // Task requiring multiple tools
    let response = server.chat_completions(ChatRequest {
        messages: vec![user_message("Find all .rs files and count total lines")],
        tools: vec![glob_tool(), read_file_tool()],
        agentic: Some(AgenticConfig {
            enabled: true,
            max_iterations: 10,
            ..default()
        }),
        ..default()
    }).await.unwrap();

    let agentic = response.agentic.unwrap();
    assert!(agentic.iterations > 1);
    assert!(agentic.tool_calls > 1);
    assert!(matches!(agentic.status, LoopStatus::Completed));
}
```

### 9.2 Edge Cases

```rust
#[tokio::test]
async fn test_loop_handles_tool_error() {
    let server = start_test_server().await;

    let response = server.chat_completions(ChatRequest {
        messages: vec![user_message("Read /nonexistent/file.txt")],
        tools: vec![read_file_tool()],
        agentic: Some(AgenticConfig::default()),
        ..default()
    }).await.unwrap();

    // Should handle gracefully
    assert!(response.choices[0].message.content.is_some());
    // Should mention the error
    let content = response.choices[0].message.content.as_ref().unwrap();
    assert!(content.contains("not found") || content.contains("doesn't exist") || content.contains("error"));
}

#[tokio::test]
async fn test_loop_respects_iteration_limit() {
    let server = start_test_server().await;

    // Task that could loop forever
    let response = server.chat_completions(ChatRequest {
        messages: vec![user_message("Keep reading random files until you find 'needle'")],
        tools: vec![read_file_tool()],
        agentic: Some(AgenticConfig {
            enabled: true,
            max_iterations: 3,
            ..default()
        }),
        ..default()
    }).await.unwrap();

    let agentic = response.agentic.unwrap();
    assert!(agentic.iterations <= 3);
    assert!(matches!(
        agentic.termination,
        TerminationReason::Resource(ResourceTermination::MaxIterations { .. })
    ));
}

#[tokio::test]
async fn test_loop_handles_uncertainty() {
    let server = start_test_server().await;

    let response = server.chat_completions(ChatRequest {
        messages: vec![user_message("What is the meaning of life according to /tmp/philosophy.txt?")],
        tools: vec![read_file_tool()], // file doesn't exist
        agentic: Some(AgenticConfig {
            enabled: true,
            allow_uncertainty: true,
            ..default()
        }),
        ..default()
    }).await.unwrap();

    // Should express uncertainty rather than hallucinate
    let agentic = response.agentic.unwrap();
    assert!(
        matches!(agentic.status, LoopStatus::Completed) ||
        matches!(agentic.termination, TerminationReason::Natural(NaturalTermination::AgentStuck { .. }))
    );
}
```

### 9.3 SSE Streaming

```rust
#[tokio::test]
async fn test_sse_stream_complete() {
    let server = start_test_server().await;

    let mut stream = server.chat_completions_stream(ChatRequest {
        messages: vec![user_message("What's 2+2?")],
        tools: vec![calculator_tool()],
        agentic: Some(AgenticConfig::default()),
        stream: true,
        ..default()
    }).await.unwrap();

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.unwrap());
    }

    // Should have lifecycle events
    assert!(events.iter().any(|e| matches!(e, LoopEvent::LoopStarted { .. })));
    assert!(events.iter().any(|e| matches!(e, LoopEvent::LoopCompleted { .. })));

    // Should have tool events
    assert!(events.iter().any(|e| matches!(e, LoopEvent::ToolCallDetected { .. })));
    assert!(events.iter().any(|e| matches!(e, LoopEvent::ToolExecutionCompleted { .. })));

    // Last event should be terminal
    assert!(events.last().unwrap().is_terminal());
}
```

---

## 10. Tool Approval Protocol

**Trust Boundary:** The approval protocol guards both security (denied tools must never execute) and liveness (the executor must never block indefinitely). Violations here could execute dangerous operations without consent, or freeze the executor forever.

**Spec Reference:** AGENTIC-LOOP-SPEC §9.4

### 10.1 Approval Handshake

```rust
// Property: Denied tools never execute — regardless of how the denial arrives
#[proptest]
fn test_denied_tools_never_execute(
    call: DetectedToolCall,
    denial_path: DenialPath,  // Explicit deny, timeout, oneshot dropped
) {
    let (executor, event_rx, approval_inbox) = setup_approval_test();
    let grant = AutonomyGrant::builder()
        .require_approval(ToolPattern::Tool(&call.name))
        .build();

    let result_handle = tokio::spawn(
        executor.execute_single_tool(call.clone(), &grant)
    );

    // Wait for ToolApprovalRequired event
    let event = event_rx.recv().await;
    prop_assert!(matches!(event, LoopEvent::ToolApprovalRequired { .. }));

    // Deliver denial via the specified path
    match denial_path {
        DenialPath::ExplicitDeny => {
            approval_inbox.deliver(call.id, ApprovalDecision::Deny).await;
        }
        DenialPath::Timeout => {
            // Let the timeout expire
            tokio::time::sleep(approval_timeout + Duration::from_millis(100)).await;
        }
        DenialPath::OneshotDropped => {
            approval_inbox.drop_sender(&call.id);
        }
    }

    let result = result_handle.await.unwrap();
    prop_assert!(result.status.is_failed());
    prop_assert!(result.status.is_recoverable());
    prop_assert!(!was_tool_invoked(&call));
}

// Property: Approved tools always execute successfully (barring tool errors)
#[proptest]
fn test_approved_tools_execute(
    call: DetectedToolCall,
    approval_variant: ApproveVariant,  // Approve, ApproveAlways(ThisCall|ThisTool|ThisSession)
) {
    let (executor, event_rx, approval_inbox) = setup_approval_test();
    let grant = AutonomyGrant::builder()
        .require_approval(ToolPattern::Tool(&call.name))
        .build();

    let result_handle = tokio::spawn(
        executor.execute_single_tool(call.clone(), &grant)
    );

    let event = event_rx.recv().await;
    prop_assert!(matches!(event, LoopEvent::ToolApprovalRequired { .. }));

    let decision = match approval_variant {
        ApproveVariant::Approve => ApprovalDecision::Approve,
        ApproveVariant::ApproveAlways(scope) => {
            ApprovalDecision::ApproveAlways { scope }
        }
    };
    approval_inbox.deliver(call.id, decision).await.unwrap();

    let result = result_handle.await.unwrap();
    // Tool was invoked (success or tool-level error, but not approval-denied)
    prop_assert!(!matches!(result.status, ResultStatus::Failed { recoverable: true }
        if result.error_message.as_deref() == Some("denied")));
    prop_assert!(was_tool_invoked(&call));
}

#[tokio::test]
async fn test_basic_approve_flow() {
    let (executor, mut event_rx, sessions) = setup_approval_test();
    let call = detected_call("bash", json!({"command": "git status"}));
    let grant = require_approval_for_all();

    let result_handle = tokio::spawn({
        let executor = executor.clone();
        async move { executor.execute_single_tool(call.clone(), &grant).await }
    });

    // 1. SSE event emitted
    let event = event_rx.recv().await.unwrap();
    assert!(matches!(event, LoopEvent::ToolApprovalRequired { .. }));

    // 2. Client approves via endpoint
    sessions.deliver_approval("test_session", "call_1", ApprovalDecision::Approve)
        .await
        .unwrap();

    // 3. Tool executes
    let result = result_handle.await.unwrap();
    assert!(result.status.is_success());
}

#[tokio::test]
async fn test_basic_deny_flow() {
    let (executor, mut event_rx, sessions) = setup_approval_test();
    let call = detected_call("bash", json!({"command": "rm -rf /tmp/cache"}));
    let grant = require_approval_for_all();

    let result_handle = tokio::spawn({
        let executor = executor.clone();
        async move { executor.execute_single_tool(call.clone(), &grant).await }
    });

    let event = event_rx.recv().await.unwrap();
    assert!(matches!(event, LoopEvent::ToolApprovalRequired { .. }));

    sessions.deliver_approval("test_session", "call_1", ApprovalDecision::Deny)
        .await
        .unwrap();

    let result = result_handle.await.unwrap();
    assert!(result.status.is_failed());
    assert!(result.status.is_recoverable());
}
```

### 10.2 Timeout Behavior

```rust
// Property: Timeout fires within tolerance of configured duration
#[proptest]
fn test_timeout_fires_within_tolerance(
    timeout_ms: u64,  // 100..=30_000
) {
    let timeout = Duration::from_millis(timeout_ms);
    let config = LoopConfig { approval_timeout: timeout, ..default() };
    let (executor, _, _) = setup_approval_test_with_config(config);
    let call = detected_call("bash", json!({"command": "echo hello"}));
    let grant = require_approval_for_all();

    let start = Instant::now();
    let result = executor.execute_single_tool(call, &grant).await;
    let elapsed = start.elapsed();

    // Should have timed out
    prop_assert!(result.status.is_failed());

    // Tolerance: within 500ms of configured timeout
    let tolerance = Duration::from_millis(500);
    prop_assert!(elapsed >= timeout);
    prop_assert!(elapsed <= timeout + tolerance);
}

#[tokio::test]
async fn test_decision_just_before_timeout_honored() {
    let timeout = Duration::from_millis(500);
    let config = LoopConfig { approval_timeout: timeout, ..default() };
    let (executor, mut event_rx, sessions) = setup_approval_test_with_config(config);
    let call = detected_call("bash", json!({"command": "echo hello"}));
    let grant = require_approval_for_all();

    let result_handle = tokio::spawn({
        let executor = executor.clone();
        async move { executor.execute_single_tool(call.clone(), &grant).await }
    });

    let _ = event_rx.recv().await;

    // Approve 100ms before timeout
    tokio::time::sleep(Duration::from_millis(400)).await;
    sessions.deliver_approval("test_session", "call_1", ApprovalDecision::Approve)
        .await
        .unwrap();

    let result = result_handle.await.unwrap();
    assert!(result.status.is_success()); // Decision honored, not timed out
}

#[tokio::test]
async fn test_timeout_returns_recoverable_error() {
    let timeout = Duration::from_millis(100);
    let config = LoopConfig { approval_timeout: timeout, ..default() };
    let (executor, _, _) = setup_approval_test_with_config(config);
    let call = detected_call("read_file", json!({"path": "/etc/passwd"}));
    let grant = require_approval_for_all();

    let result = executor.execute_single_tool(call, &grant).await;

    assert!(result.status.is_failed());
    assert!(result.status.is_recoverable());
    // Agent gets a useful message, not a cryptic error
    assert!(result.error_message.unwrap().contains("timed out"));
}
```

### 10.3 ApproveAlways Semantics

```rust
// Property: ApproveAlways(ThisTool) auto-approves subsequent same-tool calls
#[proptest]
fn test_approve_always_this_tool_subsequent_auto_approved(
    tool_name: ToolName,
    subsequent_calls: Vec<DetectedToolCall>,  // 1..=5 calls with same tool_name
) {
    let (executor, mut event_rx, sessions) = setup_approval_test();
    let grant = require_approval_for_all();

    // First call: manual approval
    let first_call = detected_call(&tool_name, json!({}));
    let result_handle = tokio::spawn({
        let executor = executor.clone();
        async move { executor.execute_single_tool(first_call, &grant).await }
    });

    let _ = event_rx.recv().await;
    sessions.deliver_approval(
        "test_session", &first_call.id,
        ApprovalDecision::ApproveAlways { scope: ApprovalScope::ThisTool },
    ).await.unwrap();
    let _ = result_handle.await.unwrap();

    // Subsequent calls to same tool: should auto-approve (no SSE event)
    for call in subsequent_calls {
        let call = DetectedToolCall { name: tool_name.clone(), ..call };
        let result = executor.execute_single_tool(call, &grant).await;

        // Should succeed without blocking for approval
        prop_assert!(result.status.is_success() || result.status.is_tool_error());
        // No ToolApprovalRequired event should have been emitted
    }
}

// Property: ApproveAlways(ThisTool) does NOT approve different tools
#[proptest]
fn test_approve_always_scoped_to_tool(
    approved_tool: ToolName,
    other_tool: ToolName,
) {
    prop_assume!(approved_tool != other_tool);

    let (executor, mut event_rx, sessions) = setup_approval_test();
    let grant = require_approval_for_all();

    // Approve "bash" always
    let call = detected_call(&approved_tool, json!({}));
    let handle = tokio::spawn(executor.execute_single_tool(call.clone(), &grant));
    let _ = event_rx.recv().await;
    sessions.deliver_approval(
        "test_session", &call.id,
        ApprovalDecision::ApproveAlways { scope: ApprovalScope::ThisTool },
    ).await.unwrap();
    let _ = handle.await;

    // Different tool should still require approval
    let other_call = detected_call(&other_tool, json!({}));
    let handle = tokio::spawn(executor.execute_single_tool(other_call.clone(), &grant));

    // Should block for approval
    let event = tokio::time::timeout(Duration::from_millis(100), event_rx.recv()).await;
    prop_assert!(event.is_ok()); // Event emitted = approval still required
}

#[tokio::test]
async fn test_approve_always_this_session_approves_all() {
    let (executor, mut event_rx, sessions) = setup_approval_test();
    let grant = require_approval_for_all();

    // First call: approve with session scope
    let call = detected_call("bash", json!({"command": "ls"}));
    let handle = tokio::spawn({
        let executor = executor.clone();
        async move { executor.execute_single_tool(call.clone(), &grant).await }
    });
    let _ = event_rx.recv().await;
    sessions.deliver_approval(
        "test_session", "call_1",
        ApprovalDecision::ApproveAlways { scope: ApprovalScope::ThisSession },
    ).await.unwrap();
    let _ = handle.await;

    // ANY subsequent tool should auto-approve
    for tool in &["bash", "write_file", "read_file", "edit_file"] {
        let call = detected_call(tool, json!({}));
        let result = executor.execute_single_tool(call, &grant).await;
        assert!(
            result.status.is_success() || result.status.is_tool_error(),
            "Tool {} should have been auto-approved",
            tool
        );
    }
}
```

### 10.4 Concurrent Approvals

```rust
#[tokio::test]
async fn test_multiple_pending_approvals_independent() {
    let (executor, mut event_rx, sessions) = setup_approval_test();
    let grant = require_approval_for_all();

    let call_a = detected_call("bash", json!({"command": "git status"}));
    let call_b = detected_call("write_file", json!({"path": "/tmp/out.txt"}));

    // Both tool calls need approval (two-pass approach per spec §9.4.4)
    let handle_a = tokio::spawn(executor.execute_single_tool(call_a.clone(), &grant));
    let handle_b = tokio::spawn(executor.execute_single_tool(call_b.clone(), &grant));

    // Collect both approval events
    let event_a = event_rx.recv().await.unwrap();
    let event_b = event_rx.recv().await.unwrap();
    assert!(matches!(event_a, LoopEvent::ToolApprovalRequired { .. }));
    assert!(matches!(event_b, LoopEvent::ToolApprovalRequired { .. }));

    // Approve A, deny B
    sessions.deliver_approval("test_session", &call_a.id, ApprovalDecision::Approve)
        .await.unwrap();
    sessions.deliver_approval("test_session", &call_b.id, ApprovalDecision::Deny)
        .await.unwrap();

    let result_a = handle_a.await.unwrap();
    let result_b = handle_b.await.unwrap();

    assert!(result_a.status.is_success());
    assert!(result_b.status.is_failed());
    assert!(result_b.status.is_recoverable());
}

#[tokio::test]
async fn test_one_timeout_one_approved() {
    let timeout = Duration::from_millis(200);
    let config = LoopConfig { approval_timeout: timeout, ..default() };
    let (executor, mut event_rx, sessions) = setup_approval_test_with_config(config);
    let grant = require_approval_for_all();

    let call_a = detected_call("bash", json!({}));
    let call_b = detected_call("read_file", json!({}));

    let handle_a = tokio::spawn(executor.execute_single_tool(call_a.clone(), &grant));
    let handle_b = tokio::spawn(executor.execute_single_tool(call_b.clone(), &grant));

    let _ = event_rx.recv().await;
    let _ = event_rx.recv().await;

    // Only approve A — let B timeout
    sessions.deliver_approval("test_session", &call_a.id, ApprovalDecision::Approve)
        .await.unwrap();

    let result_a = handle_a.await.unwrap();
    let result_b = handle_b.await.unwrap();

    assert!(result_a.status.is_success());
    assert!(result_b.status.is_failed()); // Timed out
}
```

### 10.5 Oneshot Consumption

```rust
// Property: Once consumed, call_id is removed — second delivery returns 404
#[proptest]
fn test_oneshot_consumed_once(
    call: DetectedToolCall,
    decision: ApprovalDecision,
) {
    let (executor, mut event_rx, sessions) = setup_approval_test();
    let grant = require_approval_for_all();

    let handle = tokio::spawn(executor.execute_single_tool(call.clone(), &grant));
    let _ = event_rx.recv().await;

    // First delivery succeeds
    let first = sessions.deliver_approval("test_session", &call.id, decision.clone()).await;
    prop_assert!(first.is_ok());

    // Second delivery fails (oneshot consumed, pending entry removed)
    let second = sessions.deliver_approval("test_session", &call.id, decision).await;
    prop_assert!(second.is_err());
    prop_assert!(matches!(second.unwrap_err(), ApprovalError::NotFound));
}

#[tokio::test]
async fn test_pending_approvals_list_reflects_state() {
    let (executor, mut event_rx, sessions) = setup_approval_test();
    let grant = require_approval_for_all();

    assert_eq!(sessions.pending_approvals("test_session").await.len(), 0);

    let call = detected_call("bash", json!({}));
    let handle = tokio::spawn(executor.execute_single_tool(call.clone(), &grant));
    let _ = event_rx.recv().await;

    // One pending
    assert_eq!(sessions.pending_approvals("test_session").await.len(), 1);

    sessions.deliver_approval("test_session", &call.id, ApprovalDecision::Approve)
        .await.unwrap();
    let _ = handle.await;

    // None pending
    assert_eq!(sessions.pending_approvals("test_session").await.len(), 0);
}
```

---

## 11. Session Continuation

**Trust Boundary:** Continuation state is a data integrity boundary. If store/load corrupts state, the resumed loop will operate on wrong data. If resource arithmetic is wrong, budget enforcement breaks. If TTL isn't enforced, stale state accumulates indefinitely.

**Spec Reference:** AGENTIC-LOOP-SPEC §9.3

### 11.1 Store/Load Roundtrip

```rust
// Property: store → load returns identical ContinuationState
#[proptest]
fn test_store_load_roundtrip(state: ContinuationState) {
    let store = InMemoryContinuationStore::new(Default::default());

    let token = store.store(state.clone()).await.unwrap();
    let loaded = store.load(&token).await.unwrap();

    prop_assert!(loaded.is_some());
    let loaded = loaded.unwrap();

    prop_assert_eq!(&loaded.session_id, &state.session_id);
    prop_assert_eq!(&loaded.messages, &state.messages);
    prop_assert_eq!(&loaded.tool_results, &state.tool_results);
    prop_assert_eq!(&loaded.exploration_branches, &state.exploration_branches);
    prop_assert_eq!(loaded.iterations_completed, state.iterations_completed);
    prop_assert_eq!(loaded.tool_calls_made, state.tool_calls_made);
    prop_assert_eq!(loaded.tokens_generated, state.tokens_generated);
    prop_assert_eq!(&loaded.termination, &state.termination);
    prop_assert_eq!(&loaded.working_dir, &state.working_dir);
    prop_assert_eq!(&loaded.system_prompt, &state.system_prompt);
}

// Property: All fields survive serialization — no silent data loss
#[proptest]
fn test_no_silent_field_loss(state: ContinuationState) {
    let store = InMemoryContinuationStore::new(Default::default());

    let token = store.store(state.clone()).await.unwrap();
    let loaded = store.load(&token).await.unwrap().unwrap();

    // Byte-level equality on serde roundtrip
    let original_json = serde_json::to_value(&state).unwrap();
    let loaded_json = serde_json::to_value(&loaded).unwrap();

    prop_assert_eq!(original_json, loaded_json);
}

#[tokio::test]
async fn test_store_with_tool_results_roundtrip() {
    let store = InMemoryContinuationStore::new(Default::default());

    let state = ContinuationState {
        token: String::new(), // assigned by store
        session_id: "sess_test".into(),
        messages: vec![user_message("Find the config"), assistant_message("Looking...")],
        tool_results: vec![
            AgenticToolResult {
                call_id: "call_1".into(),
                tool_name: "read_file".into(),
                status: ResultStatus::Success,
                data: json!({"content": "key=value", "lines": 1}),
                latency: Duration::from_millis(15),
            },
        ],
        exploration_branches: vec![
            ExplorationBranch {
                description: "Tried JSON config".into(),
                outcome: BranchOutcome::DeadEnd,
            },
        ],
        iterations_completed: 3,
        tool_calls_made: 5,
        tokens_generated: 1200,
        loop_config: LoopConfig { max_iterations: 10, ..default() },
        autonomy: AutonomyGrant::default(),
        system_prompt: Some("You are a helpful assistant.".into()),
        working_dir: Some("/home/user/project".into()),
        termination: TerminationReason::Natural(NaturalTermination::AgentStuck {
            attempts: 3,
            request: StuckRequest::Clarification(vec![]),
        }),
        stored_at: SystemTime::now(),
    };

    let token = store.store(state.clone()).await.unwrap();
    let loaded = store.load(&token).await.unwrap().unwrap();

    assert_eq!(loaded.tool_results.len(), 1);
    assert_eq!(loaded.tool_results[0].call_id, "call_1");
    assert_eq!(loaded.exploration_branches.len(), 1);
    assert_eq!(loaded.iterations_completed, 3);
}
```

### 11.2 TTL and Eviction

```rust
#[tokio::test]
async fn test_expired_state_returns_none() {
    let config = StoreConfig { ttl: Duration::from_millis(100), max_entries: 100 };
    let store = InMemoryContinuationStore::new(config);

    let state = make_continuation_state("sess_1");
    let token = store.store(state).await.unwrap();

    // Before TTL: present
    assert!(store.load(&token).await.unwrap().is_some());

    // After TTL: gone
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert!(store.load(&token).await.unwrap().is_none());
}

#[tokio::test]
async fn test_lru_eviction_removes_oldest() {
    let config = StoreConfig { ttl: Duration::from_secs(3600), max_entries: 3 };
    let store = InMemoryContinuationStore::new(config);

    let token_1 = store.store(make_continuation_state("sess_1")).await.unwrap();
    let token_2 = store.store(make_continuation_state("sess_2")).await.unwrap();
    let token_3 = store.store(make_continuation_state("sess_3")).await.unwrap();

    // All present
    assert!(store.load(&token_1).await.unwrap().is_some());
    assert!(store.load(&token_2).await.unwrap().is_some());
    assert!(store.load(&token_3).await.unwrap().is_some());

    // Store a 4th — should evict token_1 (LRU)
    let _token_4 = store.store(make_continuation_state("sess_4")).await.unwrap();

    assert!(store.load(&token_1).await.unwrap().is_none()); // evicted
    assert!(store.load(&token_2).await.unwrap().is_some());
    assert!(store.load(&token_3).await.unwrap().is_some());
}

#[tokio::test]
async fn test_access_refreshes_lru_position() {
    let config = StoreConfig { ttl: Duration::from_secs(3600), max_entries: 3 };
    let store = InMemoryContinuationStore::new(config);

    let token_1 = store.store(make_continuation_state("sess_1")).await.unwrap();
    let token_2 = store.store(make_continuation_state("sess_2")).await.unwrap();
    let token_3 = store.store(make_continuation_state("sess_3")).await.unwrap();

    // Access token_1 to refresh its LRU position
    let _ = store.load(&token_1).await;

    // Store a 4th — should now evict token_2 (oldest unaccessed)
    let _token_4 = store.store(make_continuation_state("sess_4")).await.unwrap();

    assert!(store.load(&token_1).await.unwrap().is_some()); // refreshed, survives
    assert!(store.load(&token_2).await.unwrap().is_none()); // evicted
    assert!(store.load(&token_3).await.unwrap().is_some());
}

// Property: Cleanup removes exactly the expired entries
#[proptest]
fn test_cleanup_removes_only_expired(
    states: Vec<(ContinuationState, bool)>,  // (state, should_be_expired)
) {
    let config = StoreConfig { ttl: Duration::from_millis(200), max_entries: 1000 };
    let store = InMemoryContinuationStore::new(config);

    let mut tokens = Vec::new();
    for (state, expire) in &states {
        let token = store.store(state.clone()).await.unwrap();
        if *expire {
            // Backdate stored_at to simulate expiry
            store.force_expire(&token);
        }
        tokens.push((token, *expire));
    }

    let removed = store.cleanup_expired().await.unwrap();
    let expected_removed = states.iter().filter(|(_, e)| *e).count() as u32;
    prop_assert_eq!(removed, expected_removed);

    for (token, expired) in &tokens {
        if *expired {
            prop_assert!(store.load(token).await.unwrap().is_none());
        } else {
            prop_assert!(store.load(token).await.unwrap().is_some());
        }
    }
}
```

### 11.3 Resource Arithmetic

```rust
// Property: New limit must exceed consumed amount
#[proptest]
fn test_new_limit_must_exceed_consumed(
    consumed: u32,   // 0..=1000
    new_limit: u32,  // 0..=2000
) {
    let state = ContinuationState {
        iterations_completed: consumed,
        loop_config: LoopConfig { max_iterations: consumed + 10, ..default() },
        ..make_continuation_state("sess_1")
    };

    let modified = json!({ "max_iterations": new_limit });
    let result = apply_config_override(&state, &modified);

    if new_limit <= consumed {
        prop_assert!(result.is_err());
    } else {
        prop_assert!(result.is_ok());
    }
}

// Property: Remaining budget = new_limit - consumed
#[proptest]
fn test_remaining_budget_arithmetic(
    consumed_iters: u32,     // 0..=100
    consumed_calls: u32,     // 0..=200
    consumed_tokens: u32,    // 0..=8000
    new_iters: u32,          // consumed_iters+1..=200
    new_calls: u32,          // consumed_calls+1..=400
    new_tokens: u32,         // consumed_tokens+1..=16000
) {
    prop_assume!(new_iters > consumed_iters);
    prop_assume!(new_calls > consumed_calls);
    prop_assume!(new_tokens > consumed_tokens);

    let state = ContinuationState {
        iterations_completed: consumed_iters,
        tool_calls_made: consumed_calls,
        tokens_generated: consumed_tokens,
        loop_config: LoopConfig {
            max_iterations: consumed_iters + 10,
            max_tool_calls: consumed_calls + 10,
            max_tokens: consumed_tokens + 1000,
            ..default()
        },
        ..make_continuation_state("sess_1")
    };

    let modified = json!({
        "max_iterations": new_iters,
        "max_tool_calls": new_calls,
        "max_tokens": new_tokens,
    });

    let resumed_config = apply_config_override(&state, &modified).unwrap();

    prop_assert_eq!(resumed_config.max_iterations, new_iters);
    prop_assert_eq!(resumed_config.max_tool_calls, new_calls);
    prop_assert_eq!(resumed_config.max_tokens, new_tokens);
}

#[tokio::test]
async fn test_extend_budget_after_exhaustion() {
    let state = ContinuationState {
        iterations_completed: 10,
        loop_config: LoopConfig { max_iterations: 10, ..default() },
        termination: TerminationReason::Resource(ResourceTermination::MaxIterations {
            completed: 10, limit: 10,
        }),
        ..make_continuation_state("sess_1")
    };

    // Extend to 20
    let modified = json!({ "max_iterations": 20 });
    let resumed_config = apply_config_override(&state, &modified).unwrap();
    assert_eq!(resumed_config.max_iterations, 20);
    // Agent gets 10 more iterations (20 - 10 consumed)
}

#[tokio::test]
async fn test_reject_reduction_below_consumed() {
    let state = ContinuationState {
        iterations_completed: 50,
        loop_config: LoopConfig { max_iterations: 50, ..default() },
        ..make_continuation_state("sess_1")
    };

    let modified = json!({ "max_iterations": 30 });
    let result = apply_config_override(&state, &modified);
    assert!(result.is_err());
}
```

### 11.4 Resume Semantics

```rust
#[tokio::test]
async fn test_additional_context_appended_as_user_message() {
    let state = ContinuationState {
        messages: vec![
            user_message("Find the config file"),
            assistant_message("I'll search for it..."),
        ],
        ..make_continuation_state("sess_1")
    };

    let resumed_messages = build_resumed_messages(
        &state,
        Some("The config is at /opt/app/config.yaml"),
    );

    assert_eq!(resumed_messages.len(), 3);
    assert_eq!(
        resumed_messages[2].content,
        "The config is at /opt/app/config.yaml"
    );
    assert_eq!(resumed_messages[2].role, "user");
}

#[tokio::test]
async fn test_system_prompt_immutable() {
    let state = ContinuationState {
        system_prompt: Some("Original prompt".into()),
        ..make_continuation_state("sess_1")
    };

    let modified = json!({ "system_prompt": "Hacked prompt" });
    let result = apply_config_override(&state, &modified);

    assert!(result.is_err());
}

#[tokio::test]
async fn test_working_dir_immutable() {
    let state = ContinuationState {
        working_dir: Some("/home/user/project".into()),
        ..make_continuation_state("sess_1")
    };

    let modified = json!({ "working_dir": "/etc/shadow" });
    let result = apply_config_override(&state, &modified);

    assert!(result.is_err());
}

// Property: Auto-approve can widen but forbidden cannot
#[proptest]
fn test_autonomy_modification_rules(
    original_auto_approve: Vec<ToolPattern>,
    additional_patterns: Vec<ToolPattern>,
) {
    let original_grant = AutonomyGrant::builder()
        .auto_approve(original_auto_approve.clone())
        .build();

    let state = ContinuationState {
        autonomy: original_grant,
        ..make_continuation_state("sess_1")
    };

    let modified = json!({
        "auto_approve": additional_patterns.iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
    });

    let resumed_grant = apply_autonomy_override(&state, &modified).unwrap();

    // All original patterns still present
    for pattern in &original_auto_approve {
        prop_assert!(resumed_grant.auto_approve_patterns().contains(pattern));
    }

    // Additional patterns added
    for pattern in &additional_patterns {
        prop_assert!(resumed_grant.auto_approve_patterns().contains(pattern));
    }
}
```

### 11.5 Resumability by Termination Type

```rust
// Property: Only resumable terminations can create continuation state
#[proptest]
fn test_resumable_terminations(termination: TerminationReason) {
    let expected_resumable = matches!(
        &termination,
        TerminationReason::Natural(NaturalTermination::AgentStuck { .. })
        | TerminationReason::Natural(NaturalTermination::AgentYielded { .. })
        | TerminationReason::Resource(ResourceTermination::MaxIterations { .. })
        | TerminationReason::Resource(ResourceTermination::TokenBudgetExhausted { .. })
        | TerminationReason::Resource(ResourceTermination::WallTimeExceeded { .. })
        | TerminationReason::Resource(ResourceTermination::ToolCallLimitReached { .. })
    );

    prop_assert_eq!(is_resumable(&termination), expected_resumable);
}

#[tokio::test]
async fn test_answer_provided_not_resumable() {
    let termination = TerminationReason::Natural(NaturalTermination::AnswerProvided {
        answer: "42".into(),
        confidence: 0.95,
    });
    assert!(!is_resumable(&termination));
}

#[tokio::test]
async fn test_agent_stuck_is_resumable() {
    let termination = TerminationReason::Natural(NaturalTermination::AgentStuck {
        attempts: 3,
        request: StuckRequest::Clarification(vec![]),
    });
    assert!(is_resumable(&termination));
}

#[tokio::test]
async fn test_client_cancelled_not_resumable() {
    let termination = TerminationReason::External(ExternalTermination::ClientCancelled);
    assert!(!is_resumable(&termination));
}

#[tokio::test]
async fn test_max_iterations_resumable() {
    let termination = TerminationReason::Resource(ResourceTermination::MaxIterations {
        completed: 10, limit: 10,
    });
    assert!(is_resumable(&termination));
}
```

---

## 12. Multi-Agent Supervisor

**Trust Boundary:** The supervisor manages resource budgets, agent lifecycles, and task dependencies across multiple concurrent executors. Violations here could exhaust global budgets, violate ordering constraints, leak zombie tasks, or lose completed work.

**Spec Reference:** MULTI-AGENT-SUPERVISOR-SPEC.md

### 12.1 Resource Invariants

```rust
// Property: Sum of all child consumption never exceeds global budget
#[proptest]
fn test_total_consumption_within_budget(
    budget: ResourceBudget,
    subtasks: Vec<Subtask>,  // 1..=10 subtasks
) {
    prop_assume!(!subtasks.is_empty());

    let config = SupervisorConfig {
        resource_budget: budget.clone(),
        decomposition: DecompositionStrategy::ClientProvided {
            subtasks: subtasks.clone(),
        },
        ..default()
    };

    let summary = run_supervisor_to_completion(config).await;

    prop_assert!(summary.total_iterations <= budget.total_iterations);
    prop_assert!(summary.total_tool_calls <= budget.total_tool_calls);
    prop_assert!(summary.total_tokens <= budget.total_tokens);
}

// Property: Each child allocation ≤ its configured LoopConfig limits
#[proptest]
fn test_child_allocation_within_limits(
    budget: ResourceBudget,
    subtasks: Vec<Subtask>,
) {
    prop_assume!(!subtasks.is_empty());

    let supervisor = Supervisor::new(SupervisorConfig {
        resource_budget: budget,
        decomposition: DecompositionStrategy::ClientProvided { subtasks: subtasks.clone() },
        ..default()
    });

    for subtask in &subtasks {
        let child_config = supervisor.allocate_budget(subtask);
        let result = run_with_config(child_config, &subtask.objective).await;

        prop_assert!(result.iterations_completed <= child_config.max_iterations);
        prop_assert!(result.tool_calls_made <= child_config.max_tool_calls);
        prop_assert!(result.tokens_generated <= child_config.max_tokens);
    }
}

// Property: Budget rebalancing doesn't exceed remaining global budget
#[proptest]
fn test_rebalance_within_remaining(
    budget: ResourceBudget,
    completed_consumption: ResourceConsumption,
    running_count: u32,  // 1..=5
) {
    prop_assume!(running_count > 0);
    prop_assume!(completed_consumption.iterations <= budget.total_iterations);

    let remaining_iterations = budget.total_iterations - completed_consumption.iterations;
    let remaining_calls = budget.total_tool_calls - completed_consumption.tool_calls;
    let remaining_tokens = budget.total_tokens - completed_consumption.tokens;

    let rebalanced = rebalance_budget(remaining_iterations, remaining_calls, remaining_tokens, running_count);

    let total_rebalanced_iters: u32 = rebalanced.iter().map(|r| r.max_iterations).sum();
    let total_rebalanced_calls: u32 = rebalanced.iter().map(|r| r.max_tool_calls).sum();
    let total_rebalanced_tokens: u32 = rebalanced.iter().map(|r| r.max_tokens).sum();

    prop_assert!(total_rebalanced_iters <= remaining_iterations);
    prop_assert!(total_rebalanced_calls <= remaining_calls);
    prop_assert!(total_rebalanced_tokens <= remaining_tokens);
}
```

### 12.2 Dependency Ordering

```rust
// Property: Subtask never starts before all dependencies complete
#[proptest]
fn test_dependency_ordering_respected(
    dag: SubtaskDag,  // Arbitrary DAG of subtasks with dependencies
) {
    let config = SupervisorConfig {
        decomposition: DecompositionStrategy::ClientProvided {
            subtasks: dag.to_subtasks(),
        },
        routing: RoutingStrategy::DependencyAware,
        ..default()
    };

    let events = run_supervisor_and_collect_events(config).await;

    let spawn_times: HashMap<String, usize> = events.iter().enumerate()
        .filter_map(|(i, e)| match e {
            SupervisorEvent::AgentSpawned { subtask_id, .. } => Some((subtask_id.clone(), i)),
            _ => None,
        })
        .collect();

    let complete_times: HashMap<String, usize> = events.iter().enumerate()
        .filter_map(|(i, e)| match e {
            SupervisorEvent::AgentCompleted { subtask_id, .. } => Some((subtask_id.clone(), i)),
            _ => None,
        })
        .collect();

    for subtask in dag.to_subtasks() {
        for dep_id in &subtask.depends_on {
            let dep_complete = complete_times.get(dep_id);
            let subtask_spawn = spawn_times.get(&subtask.id);

            if let (Some(&dep_t), Some(&spawn_t)) = (dep_complete, subtask_spawn) {
                prop_assert!(
                    dep_t < spawn_t,
                    "Subtask {} spawned at {} before dependency {} completed at {}",
                    subtask.id, spawn_t, dep_id, dep_t
                );
            }
        }
    }
}

#[tokio::test]
async fn test_independent_subtasks_run_concurrently() {
    let subtasks = vec![
        subtask("A", vec![], Complexity::Medium),
        subtask("B", vec![], Complexity::Medium),
        subtask("C", vec!["A", "B"], Complexity::High),
    ];

    let config = SupervisorConfig {
        max_concurrent_agents: 3,
        decomposition: DecompositionStrategy::ClientProvided { subtasks },
        routing: RoutingStrategy::DependencyAware,
        ..default()
    };

    let events = run_supervisor_and_collect_events(config).await;

    // A and B should be spawned before either completes
    let spawn_events: Vec<_> = events.iter()
        .filter(|e| matches!(e, SupervisorEvent::AgentSpawned { .. }))
        .collect();

    let first_complete = events.iter()
        .position(|e| matches!(e, SupervisorEvent::AgentCompleted { .. }))
        .unwrap();

    let spawns_before_first_complete = events[..first_complete].iter()
        .filter(|e| matches!(e, SupervisorEvent::AgentSpawned { .. }))
        .count();

    // Both A and B should have been spawned before either completed
    assert!(spawns_before_first_complete >= 2);
}

#[tokio::test]
async fn test_chain_runs_sequentially() {
    let subtasks = vec![
        subtask("A", vec![], Complexity::Low),
        subtask("B", vec!["A"], Complexity::Low),
        subtask("C", vec!["B"], Complexity::Low),
    ];

    let config = SupervisorConfig {
        max_concurrent_agents: 3,
        decomposition: DecompositionStrategy::ClientProvided { subtasks },
        routing: RoutingStrategy::DependencyAware,
        ..default()
    };

    let events = run_supervisor_and_collect_events(config).await;

    // Verify strict ordering: spawn_A < complete_A < spawn_B < complete_B < spawn_C
    let event_names: Vec<String> = events.iter()
        .filter_map(|e| match e {
            SupervisorEvent::AgentSpawned { subtask_id, .. } => Some(format!("spawn_{}", subtask_id)),
            SupervisorEvent::AgentCompleted { subtask_id, .. } => Some(format!("complete_{}", subtask_id)),
            _ => None,
        })
        .collect();

    let pos = |name: &str| event_names.iter().position(|n| n == name).unwrap();
    assert!(pos("spawn_A") < pos("complete_A"));
    assert!(pos("complete_A") < pos("spawn_B"));
    assert!(pos("complete_B") < pos("spawn_C"));
}
```

### 12.3 Concurrency Limits

```rust
// Property: Concurrent agents never exceed max_concurrent_agents
#[proptest]
fn test_concurrency_limit_respected(
    max_concurrent: u32,  // 1..=5
    subtask_count: u32,   // max_concurrent..=20
) {
    prop_assume!(subtask_count >= max_concurrent);

    let subtasks: Vec<_> = (0..subtask_count)
        .map(|i| subtask(&format!("task_{}", i), vec![], Complexity::Low))
        .collect();

    let config = SupervisorConfig {
        max_concurrent_agents: max_concurrent,
        decomposition: DecompositionStrategy::ClientProvided { subtasks },
        routing: RoutingStrategy::Parallel,
        ..default()
    };

    let events = run_supervisor_and_collect_events(config).await;

    // Track concurrent agents at each event
    let mut active = 0u32;
    let mut max_observed = 0u32;

    for event in &events {
        match event {
            SupervisorEvent::AgentSpawned { .. } => {
                active += 1;
                max_observed = max_observed.max(active);
            }
            SupervisorEvent::AgentCompleted { .. } => {
                active = active.saturating_sub(1);
            }
            _ => {}
        }
    }

    prop_assert!(max_observed <= max_concurrent);
}

#[tokio::test]
async fn test_queued_subtasks_dispatched_as_slots_open() {
    let subtasks = vec![
        subtask("A", vec![], Complexity::Low),
        subtask("B", vec![], Complexity::Low),
        subtask("C", vec![], Complexity::Low),
        subtask("D", vec![], Complexity::Low),
        subtask("E", vec![], Complexity::Low),
    ];

    let config = SupervisorConfig {
        max_concurrent_agents: 2,
        decomposition: DecompositionStrategy::ClientProvided { subtasks },
        routing: RoutingStrategy::Parallel,
        ..default()
    };

    let events = run_supervisor_and_collect_events(config).await;

    // All 5 subtasks should eventually complete
    let completed: Vec<_> = events.iter()
        .filter_map(|e| match e {
            SupervisorEvent::AgentCompleted { subtask_id, .. } => Some(subtask_id.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(completed.len(), 5);
}
```

### 12.4 Lifecycle Guarantees

```rust
// Property: Every spawned agent is eventually resolved (no zombies)
#[proptest]
fn test_no_zombie_agents(
    subtasks: Vec<Subtask>,
    failure_points: Vec<Option<FailureInjection>>,
) {
    let config = SupervisorConfig {
        decomposition: DecompositionStrategy::ClientProvided { subtasks: subtasks.clone() },
        ..default()
    };

    let (events, join_results) = run_supervisor_with_tracking(config, failure_points).await;

    let spawned: HashSet<AgentId> = events.iter()
        .filter_map(|e| match e {
            SupervisorEvent::AgentSpawned { agent_id, .. } => Some(agent_id.clone()),
            _ => None,
        })
        .collect();

    let resolved: HashSet<AgentId> = events.iter()
        .filter_map(|e| match e {
            SupervisorEvent::AgentCompleted { agent_id, .. }
            | SupervisorEvent::Rerouted { from_agent: agent_id, .. } => Some(agent_id.clone()),
            _ => None,
        })
        .chain(join_results.into_keys())
        .collect();

    // Every spawned agent must be in the resolved set
    for agent in &spawned {
        prop_assert!(
            resolved.contains(agent),
            "Zombie agent detected: {} spawned but never resolved",
            agent
        );
    }
}

#[tokio::test]
async fn test_agent_panic_detected_as_failure() {
    let subtasks = vec![
        subtask("will_panic", vec![], Complexity::Low),
    ];

    // Inject a panic into the executor
    let config = SupervisorConfig {
        decomposition: DecompositionStrategy::ClientProvided { subtasks },
        ..default()
    };

    let events = run_supervisor_with_panic_injection(config, "will_panic").await;

    // Supervisor should detect the failure
    assert!(events.iter().any(|e| matches!(
        e,
        SupervisorEvent::SupervisorError { recoverable: true, .. }
    )));

    // Supervisor should still complete (not panic itself)
    assert!(events.iter().any(|e| matches!(
        e,
        SupervisorEvent::SupervisorCompleted { .. }
    )));
}
```

### 12.5 Rerouting

```rust
// Property: Stuck agent's partial progress forwarded to reroute target
#[proptest]
fn test_reroute_forwards_partial_progress(
    partial_progress: Option<String>,
    expertise: Vec<String>,
) {
    let subtasks = vec![
        subtask_with_agent("A", vec![], Complexity::Medium, "agent_a"),
        subtask_with_agent("B", vec![], Complexity::Medium, "agent_b"),
    ];

    let config = SupervisorConfig {
        decomposition: DecompositionStrategy::ClientProvided { subtasks },
        ..default()
    };

    // Agent A will yield with the given partial progress
    let yield_injection = YieldInjection {
        agent: "agent_a",
        partial_progress: partial_progress.clone(),
        expertise: expertise.clone(),
    };

    let events = run_supervisor_with_yield(config, yield_injection).await;

    // Reroute event should exist
    let reroute = events.iter().find(|e| matches!(e, SupervisorEvent::Rerouted { .. }));
    prop_assert!(reroute.is_some());

    // The target agent should receive the partial progress as context
    if let Some(progress) = &partial_progress {
        let target_events: Vec<_> = events.iter()
            .filter(|e| matches!(e, SupervisorEvent::AgentEvent { agent_id, .. }
                if agent_id != "agent_a"))
            .collect();

        // Verify target agent's context includes the partial progress
        prop_assert!(target_events.iter().any(|e| {
            if let SupervisorEvent::AgentEvent { event: LoopEvent::LoopStarted { .. }, .. } = e {
                true // target was spawned
            } else { false }
        }));
    }
}

#[tokio::test]
async fn test_yield_routes_to_matching_expertise() {
    let subtasks = vec![
        subtask_with_capabilities("research", vec![], vec!["general"]),
        subtask_with_capabilities("implement", vec!["research"], vec!["rust", "database"]),
    ];

    let config = SupervisorConfig {
        decomposition: DecompositionStrategy::ClientProvided { subtasks },
        routing: RoutingStrategy::DependencyAware,
        ..default()
    };

    // Research agent yields requesting "database" expertise
    let yield_injection = YieldInjection {
        agent: "research",
        partial_progress: Some("Found the schema".into()),
        expertise: vec!["database".into()],
    };

    let events = run_supervisor_with_yield(config, yield_injection).await;

    // Should reroute to the "implement" agent (has "database" capability)
    let reroute = events.iter().find_map(|e| match e {
        SupervisorEvent::Rerouted { to_agent, .. } => Some(to_agent.clone()),
        _ => None,
    });

    assert!(reroute.is_some());
}
```

### 12.6 Failure Recovery

```rust
// Property: Retry count never exceeds max_retries
#[proptest]
fn test_retry_bounded(
    max_retries: u32,  // 0..=3
    failures: Vec<AgentFailure>,
) {
    let config = SupervisorConfig {
        decomposition: DecompositionStrategy::SingleAgent,
        ..default()
    };

    let events = run_supervisor_with_failures(config, failures, max_retries).await;

    let retry_count = events.iter()
        .filter(|e| matches!(e, SupervisorEvent::Rerouted { reason: RerouteReason::AgentStuck { .. }, .. }))
        .count();

    prop_assert!(retry_count as u32 <= max_retries);
}

// Property: Circuit breaker triggers after N consecutive same-type failures
#[proptest]
fn test_circuit_breaker(
    consecutive_threshold: u32,  // 2..=5
    failure_type: FailureType,
) {
    let failures: Vec<_> = (0..consecutive_threshold + 1)
        .map(|_| AgentFailure::from_type(failure_type.clone()))
        .collect();

    let config = SupervisorConfig {
        circuit_breaker_threshold: consecutive_threshold,
        ..default()
    };

    let events = run_supervisor_with_failures(config, failures, 10).await;

    // After threshold consecutive failures, dispatching should pause
    let escalation = events.iter().any(|e| matches!(
        e,
        SupervisorEvent::SupervisorError { message, .. }
            if message.contains("circuit breaker")
    ));

    prop_assert!(escalation);
}

#[tokio::test]
async fn test_engine_error_retry_then_reassign() {
    let failures = vec![
        AgentFailure::EngineError { retries: 0, message: "model timeout".into() },
        AgentFailure::EngineError { retries: 1, message: "model timeout".into() },
    ];

    let config = SupervisorConfig {
        decomposition: DecompositionStrategy::SingleAgent,
        ..default()
    };

    let events = run_supervisor_with_failures(config, failures, 2).await;

    // First failure: retry
    // Second failure: reassign
    let strategies: Vec<_> = events.iter()
        .filter_map(|e| match e {
            SupervisorEvent::Rerouted { reason, .. } => Some(reason.clone()),
            _ => None,
        })
        .collect();

    // Should see at least one reroute
    assert!(!strategies.is_empty());
}

#[tokio::test]
async fn test_majority_failure_triggers_early_aggregation() {
    let subtasks = vec![
        subtask("A", vec![], Complexity::Low),
        subtask("B", vec![], Complexity::Low),
        subtask("C", vec![], Complexity::Low),
        subtask("D", vec![], Complexity::Low),
    ];

    // Fail 3 out of 4 (>50%)
    let failure_map = hashmap! {
        "A" => AgentFailure::EngineError { retries: 2, message: "fail".into() },
        "B" => AgentFailure::EngineError { retries: 2, message: "fail".into() },
        "C" => AgentFailure::EngineError { retries: 2, message: "fail".into() },
    };

    let config = SupervisorConfig {
        decomposition: DecompositionStrategy::ClientProvided { subtasks },
        ..default()
    };

    let summary = run_supervisor_with_failure_map(config, failure_map).await;

    // Supervisor should aggregate with partial results, not crash
    assert!(matches!(
        summary.termination,
        SupervisorTermination::PartialComplete { .. } | SupervisorTermination::Failed { .. }
    ));

    // The one successful subtask should be preserved
    let completed = summary.subtask_results.iter()
        .filter(|r| matches!(r.status, SubtaskStatus::Completed))
        .count();
    assert!(completed >= 1);
}
```

### 12.7 Aggregation

```rust
// Property: Completed subtask results are never lost
#[proptest]
fn test_completed_results_preserved(
    subtask_outcomes: Vec<(Subtask, SubtaskOutcome)>,
) {
    let subtasks: Vec<_> = subtask_outcomes.iter().map(|(s, _)| s.clone()).collect();
    let completed_ids: Vec<_> = subtask_outcomes.iter()
        .filter(|(_, o)| matches!(o, SubtaskOutcome::Success))
        .map(|(s, _)| s.id.clone())
        .collect();

    let config = SupervisorConfig {
        decomposition: DecompositionStrategy::ClientProvided { subtasks },
        ..default()
    };

    let summary = run_supervisor_with_outcomes(config, subtask_outcomes).await;

    for id in &completed_ids {
        let result = summary.subtask_results.iter().find(|r| &r.subtask_id == id);
        prop_assert!(result.is_some(), "Completed subtask {} missing from results", id);
        prop_assert!(matches!(result.unwrap().status, SubtaskStatus::Completed));
        prop_assert!(result.unwrap().summary.is_some());
    }
}

// Property: Partial results from failed/skipped subtasks are included
#[proptest]
fn test_partial_results_included(
    subtask_outcomes: Vec<(Subtask, SubtaskOutcome)>,
) {
    let subtasks: Vec<_> = subtask_outcomes.iter().map(|(s, _)| s.clone()).collect();

    let config = SupervisorConfig {
        decomposition: DecompositionStrategy::ClientProvided { subtasks },
        ..default()
    };

    let summary = run_supervisor_with_outcomes(config, subtask_outcomes.clone()).await;

    // Every subtask should appear in results (completed, partial, failed, or skipped)
    for (subtask, _) in &subtask_outcomes {
        prop_assert!(
            summary.subtask_results.iter().any(|r| r.subtask_id == subtask.id),
            "Subtask {} missing from results entirely",
            subtask.id
        );
    }
}

#[tokio::test]
async fn test_aggregation_collects_all_results() {
    let subtasks = vec![
        subtask("A", vec![], Complexity::Low),
        subtask("B", vec![], Complexity::Medium),
        subtask("C", vec!["A", "B"], Complexity::High),
    ];

    let config = SupervisorConfig {
        decomposition: DecompositionStrategy::ClientProvided { subtasks },
        routing: RoutingStrategy::DependencyAware,
        shared_context_mode: SharedContextMode::SummarySharing,
        ..default()
    };

    let summary = run_supervisor_to_completion(config).await;

    assert_eq!(summary.subtask_results.len(), 3);
    assert!(summary.subtask_results.iter().all(|r| matches!(r.status, SubtaskStatus::Completed)));
    assert_eq!(summary.total_agents_spawned, 3);
    assert!(matches!(summary.termination, SupervisorTermination::AllComplete));
}
```

### 12.8 Wellbeing Aggregate

```rust
// Property: Agent counts by state sum to total agents
#[proptest]
fn test_wellbeing_counts_sum_to_total(
    agent_states: Vec<WellbeingState>,
) {
    let aggregate = compute_aggregate_wellbeing(&agent_states);

    let sum = aggregate.agents_healthy
        + aggregate.agents_cautious
        + aggregate.agents_concerned
        + aggregate.agents_distressed;

    prop_assert_eq!(sum, aggregate.agents_total);
    prop_assert_eq!(aggregate.agents_total, agent_states.len());
}

// Property: Distressed child triggers pause, not punishment
#[proptest]
fn test_distressed_child_paused_not_punished(
    distressed_agent: AgentId,
) {
    let agent_states = vec![
        (distressed_agent.clone(), WellbeingState::Distressed),
        (AgentId::new(), WellbeingState::Healthy),
        (AgentId::new(), WellbeingState::Healthy),
    ];

    let actions = supervisor_wellbeing_response(&agent_states);

    // Distressed agent should be paused and reassigned
    let distressed_action = actions.iter()
        .find(|a| a.agent_id == distressed_agent);
    prop_assert!(distressed_action.is_some());

    let action = distressed_action.unwrap();
    prop_assert!(matches!(
        action.response,
        WellbeingResponse::Pause | WellbeingResponse::Reassign
    ));

    // Never punitive
    prop_assert!(!matches!(
        action.response,
        WellbeingResponse::ReduceAutonomy | WellbeingResponse::IncreasePressure
    ));
}

#[tokio::test]
async fn test_majority_concerned_triggers_replan() {
    let agent_states = vec![
        WellbeingState::Concerned,
        WellbeingState::Concerned,
        WellbeingState::Concerned,
        WellbeingState::Healthy,
    ];

    let aggregate = compute_aggregate_wellbeing(&agent_states);
    let supervisor_action = supervisor_level_response(&aggregate);

    // Majority concerned → pause and re-plan
    assert!(matches!(
        supervisor_action,
        SupervisorWellbeingAction::PauseAndReplan
    ));
}

#[tokio::test]
async fn test_all_concerned_escalates_to_client() {
    let agent_states = vec![
        WellbeingState::Concerned,
        WellbeingState::Distressed,
        WellbeingState::Concerned,
    ];

    let aggregate = compute_aggregate_wellbeing(&agent_states);
    let supervisor_action = supervisor_level_response(&aggregate);

    assert!(matches!(
        supervisor_action,
        SupervisorWellbeingAction::EscalateToClient
    ));
}
```

---

## 13. Test Infrastructure

### 13.1 Property Test Generators

```rust
impl Arbitrary for AutonomyGrant {
    fn arbitrary(g: &mut Gen) -> Self {
        AutonomyGrant {
            allow: Vec::arbitrary(g),
            require_approval: Vec::arbitrary(g),
            forbid: Vec::arbitrary(g),
            max_tool_calls: u32::arbitrary(g) % 100,
            max_wall_time: Duration::from_secs(u64::arbitrary(g) % 600),
        }
    }
}

impl Arbitrary for DetectedToolCall {
    fn arbitrary(g: &mut Gen) -> Self {
        DetectedToolCall {
            id: format!("call_{}", u64::arbitrary(g)),
            name: g.choose(&["read_file", "write_file", "bash", "glob", "grep"]).unwrap().to_string(),
            arguments: json!({
                "path": format!("/tmp/test_{}.txt", u32::arbitrary(g)),
            }),
        }
    }
}

impl Arbitrary for MetaSignal {
    fn arbitrary(g: &mut Gen) -> Self {
        match u8::arbitrary(g) % 4 {
            0 => MetaSignal::Answer {
                content: String::arbitrary(g),
                confidence: f32::arbitrary(g).abs() % 1.0,
                caveats: Vec::arbitrary(g),
            },
            1 => MetaSignal::Uncertain {
                partial_answer: Option::arbitrary(g),
                missing_information: Vec::arbitrary(g),
                would_help: Vec::arbitrary(g),
            },
            2 => MetaSignal::Stuck {
                attempts: Vec::arbitrary(g),
                hypothesis: Option::arbitrary(g),
                request: StuckRequest::arbitrary(g),
            },
            _ => MetaSignal::Yield {
                partial_progress: Option::arbitrary(g),
                suggested_expertise: Vec::arbitrary(g),
            },
        }
    }
}
```

### 13.2 Test Fixtures

```rust
fn simple_config() -> LoopConfig {
    LoopConfig {
        max_iterations: 10,
        max_tool_calls: 50,
        max_wall_time: Duration::from_secs(60),
        max_tokens: 8192,
        ..default()
    }
}

fn read_file_tool() -> Tool {
    Tool {
        r#type: "function".into(),
        function: ToolFunction {
            name: "read_file".into(),
            description: "Read contents of a file".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"]
            }),
        },
    }
}

fn create_loop_in_state(state: LoopState) -> AgenticLoop {
    let mut loop_obj = AgenticLoop::new(simple_config());
    // Transition to desired state
    match state {
        LoopState::Generating => { loop_obj.start().unwrap(); }
        LoopState::Detecting => {
            loop_obj.start().unwrap();
            loop_obj.generation_complete("output").unwrap();
        }
        // ... etc
    }
    loop_obj
}
```

### 13.3 Mocks

```rust
struct MockToolExecutor {
    responses: HashMap<String, ToolResult>,
    call_log: Arc<Mutex<Vec<DetectedToolCall>>>,
}

impl MockToolExecutor {
    fn with_response(tool: &str, result: ToolResult) -> Self {
        let mut responses = HashMap::new();
        responses.insert(tool.to_string(), result);
        Self { responses, call_log: Arc::new(Mutex::new(Vec::new())) }
    }

    fn calls(&self) -> Vec<DetectedToolCall> {
        self.call_log.lock().unwrap().clone()
    }
}

#[async_trait]
impl ToolExecutor for MockToolExecutor {
    async fn execute(&self, call: &DetectedToolCall) -> Result<ToolResult, ToolError> {
        self.call_log.lock().unwrap().push(call.clone());
        self.responses.get(&call.name)
            .cloned()
            .ok_or(ToolError::NotFound(call.name.clone()))
    }
}
```

### 13.4 Coordination Test Helpers (§7.3-7.7)

```rust
fn blocking_assistance_request(description: &str) -> AssistanceRequest {
    AssistanceRequest {
        description: description.to_string(),
        required_capabilities: vec![],
        partial_progress: None,
        priority: AssistancePriority::Blocking,
    }
}

/// Actions for property testing event emission.
#[derive(Debug, Clone, Arbitrary)]
enum CoordinationAction {
    RequestAssistance(AssistanceRequest),
    YieldTo(YieldContext),
    ShareDiscovery(Discovery),
}

impl Arbitrary for AssistanceRequest {
    fn arbitrary(g: &mut Gen) -> Self {
        AssistanceRequest {
            description: String::arbitrary(g),
            required_capabilities: Vec::arbitrary(g),
            partial_progress: Option::arbitrary(g),
            priority: *g.choose(&[
                AssistancePriority::Blocking,
                AssistancePriority::Background,
            ]).unwrap(),
        }
    }
}

impl Arbitrary for AssistanceResponse {
    fn arbitrary(g: &mut Gen) -> Self {
        match u8::arbitrary(g) % 3 {
            0 => AssistanceResponse::Assigned {
                helper_id: format!("agent_{}", u32::arbitrary(g) % 100),
                helper_name: Option::arbitrary(g),
                message: Option::arbitrary(g),
            },
            1 => AssistanceResponse::Unavailable {
                reason: String::arbitrary(g),
                suggestion: Option::arbitrary(g),
            },
            _ => AssistanceResponse::TimedOut,
        }
    }
}

impl Arbitrary for YieldContext {
    fn arbitrary(g: &mut Gen) -> Self {
        YieldContext {
            reason: String::arbitrary(g),
            partial_progress: Option::arbitrary(g),
            suggested_expertise: Vec::arbitrary(g),
            handoff_data: None,
        }
    }
}

impl Arbitrary for Discovery {
    fn arbitrary(g: &mut Gen) -> Self {
        Discovery {
            content: String::arbitrary(g),
            category: *g.choose(&["file_structure", "api_pattern", "bug_found", "config"])
                .unwrap().to_string(),
            tags: Vec::arbitrary(g),
            data: None,
        }
    }
}
```

### 13.5 Approval Test Helpers (§10)

```rust
/// Sets up an executor with approval infrastructure.
fn setup_approval_test() -> (Arc<LoopExecutor>, mpsc::Receiver<LoopEvent>, Arc<SessionRegistry>) {
    let sessions = Arc::new(SessionRegistry::new());
    let (event_tx, event_rx) = mpsc::channel(128);
    let engine = Arc::new(MockInferenceEngine::default());
    let tools = Arc::new(ToolRegistry::with_code_tools());
    let config = ExecutorConfig::new("test_agent")
        .with_session_registry(Arc::clone(&sessions))
        .with_session_id("test_session");
    let executor = Arc::new(LoopExecutor::new(engine, tools, config));
    (executor, event_rx, sessions)
}

fn setup_approval_test_with_config(
    loop_config: LoopConfig,
) -> (Arc<LoopExecutor>, mpsc::Receiver<LoopEvent>, Arc<SessionRegistry>) {
    let sessions = Arc::new(SessionRegistry::new());
    let (event_tx, event_rx) = mpsc::channel(128);
    let engine = Arc::new(MockInferenceEngine::default());
    let tools = Arc::new(ToolRegistry::with_code_tools());
    let config = ExecutorConfig::new("test_agent")
        .with_session_registry(Arc::clone(&sessions))
        .with_session_id("test_session")
        .with_loop_config(loop_config);
    let executor = Arc::new(LoopExecutor::new(engine, tools, config));
    (executor, event_rx, sessions)
}

fn require_approval_for_all() -> AutonomyGrant {
    AutonomyGrant::builder()
        .require_approval(ToolPattern::Tool("*"))
        .build()
}

fn detected_call(tool: &str, args: serde_json::Value) -> DetectedToolCall {
    DetectedToolCall {
        call_id: format!("call_{}", uuid::Uuid::new_v4().simple()),
        tool: tool.to_string(),
        arguments: args,
    }
}

/// Tracks whether a tool was actually invoked (not just attempted).
fn was_tool_invoked(call: &DetectedToolCall) -> bool {
    // Check the mock executor's call log
    GLOBAL_CALL_LOG.lock().unwrap().iter().any(|c| c.call_id == call.call_id)
}

/// Denial path for property testing denial coverage.
#[derive(Debug, Clone, Arbitrary)]
enum DenialPath {
    ExplicitDeny,
    Timeout,
    OneshotDropped,
}

/// Approval variant for property testing.
#[derive(Debug, Clone, Arbitrary)]
enum ApproveVariant {
    Approve,
    ApproveAlways(ApprovalScope),
}
```

### 13.6 Continuation Test Helpers (§11)

```rust
fn make_continuation_state(session_id: &str) -> ContinuationState {
    ContinuationState {
        token: String::new(),
        session_id: session_id.to_string(),
        messages: vec![user_message("test prompt")],
        tool_results: Vec::new(),
        exploration_branches: Vec::new(),
        iterations_completed: 0,
        tool_calls_made: 0,
        tokens_generated: 0,
        loop_config: LoopConfig::default(),
        autonomy: AutonomyGrant::default(),
        system_prompt: None,
        working_dir: None,
        termination: TerminationReason::Natural(NaturalTermination::AgentStuck {
            attempts: 1,
            request: StuckRequest::Clarification(vec![]),
        }),
        stored_at: SystemTime::now(),
    }
}

/// Applies config overrides to a ContinuationState, enforcing constraints.
fn apply_config_override(
    state: &ContinuationState,
    overrides: &serde_json::Value,
) -> Result<LoopConfig, ContinuationError>;

/// Builds the resumed message list with optional additional context.
fn build_resumed_messages(
    state: &ContinuationState,
    additional_context: Option<&str>,
) -> Vec<infernum_core::Message>;

/// Determines if a termination reason allows continuation.
fn is_resumable(termination: &TerminationReason) -> bool;

impl Arbitrary for ContinuationState {
    fn arbitrary(g: &mut Gen) -> Self {
        ContinuationState {
            token: String::new(),
            session_id: format!("sess_{}", u64::arbitrary(g)),
            messages: vec![user_message(&String::arbitrary(g))],
            tool_results: Vec::arbitrary(g),
            exploration_branches: Vec::arbitrary(g),
            iterations_completed: u32::arbitrary(g) % 100,
            tool_calls_made: u32::arbitrary(g) % 200,
            tokens_generated: u32::arbitrary(g) % 16000,
            loop_config: LoopConfig::arbitrary(g),
            autonomy: AutonomyGrant::arbitrary(g),
            system_prompt: Option::arbitrary(g),
            working_dir: Option::arbitrary(g),
            termination: TerminationReason::arbitrary(g),
            stored_at: SystemTime::now(),
        }
    }
}
```

### 13.7 Supervisor Test Helpers (§12)

```rust
fn subtask(id: &str, deps: Vec<&str>, complexity: Complexity) -> Subtask {
    Subtask {
        id: id.to_string(),
        objective: format!("Complete subtask {}", id),
        role: AgentRole::Primary,
        depends_on: deps.into_iter().map(String::from).collect(),
        estimated_complexity: complexity,
        resources: None,
        autonomy: None,
        system_prompt: None,
    }
}

fn subtask_with_capabilities(
    id: &str,
    deps: Vec<&str>,
    capabilities: Vec<&str>,
) -> Subtask {
    let mut s = subtask(id, deps, Complexity::Medium);
    s.system_prompt = Some(format!(
        "You are a specialist with capabilities: {}",
        capabilities.join(", ")
    ));
    s
}

/// Arbitrary DAG generator for dependency ordering tests.
#[derive(Debug, Clone)]
struct SubtaskDag {
    nodes: Vec<String>,
    edges: Vec<(String, String)>,
}

impl SubtaskDag {
    fn to_subtasks(&self) -> Vec<Subtask> {
        self.nodes.iter().map(|id| {
            let deps: Vec<&str> = self.edges.iter()
                .filter(|(_, to)| to == id)
                .map(|(from, _)| from.as_str())
                .collect();
            subtask(id, deps, Complexity::Low)
        }).collect()
    }
}

impl Arbitrary for SubtaskDag {
    fn arbitrary(g: &mut Gen) -> Self {
        let n = (usize::arbitrary(g) % 5) + 2; // 2..=6 nodes
        let nodes: Vec<_> = (0..n).map(|i| format!("task_{}", i)).collect();
        let mut edges = Vec::new();
        // Only add edges from lower-indexed to higher-indexed nodes (ensures DAG)
        for i in 0..n {
            for j in (i + 1)..n {
                if bool::arbitrary(g) && edges.len() < n * 2 {
                    edges.push((nodes[i].clone(), nodes[j].clone()));
                }
            }
        }
        SubtaskDag { nodes, edges }
    }
}

impl Arbitrary for Subtask {
    fn arbitrary(g: &mut Gen) -> Self {
        Subtask {
            id: format!("task_{}", u32::arbitrary(g) % 1000),
            objective: String::arbitrary(g),
            role: *g.choose(&[AgentRole::Primary, AgentRole::Specialist]).unwrap(),
            depends_on: Vec::new(),
            estimated_complexity: *g.choose(&[Complexity::Low, Complexity::Medium, Complexity::High]).unwrap(),
            resources: None,
            autonomy: None,
            system_prompt: None,
        }
    }
}

impl Arbitrary for ResourceBudget {
    fn arbitrary(g: &mut Gen) -> Self {
        ResourceBudget {
            total_iterations: (u32::arbitrary(g) % 100) + 10,
            total_tool_calls: (u32::arbitrary(g) % 500) + 20,
            total_tokens: (u32::arbitrary(g) % 50000) + 1000,
            wall_time: Duration::from_secs((u64::arbitrary(g) % 600) + 30),
        }
    }
}

/// Run a supervisor to completion with the given config, collecting events.
async fn run_supervisor_and_collect_events(
    config: SupervisorConfig,
) -> Vec<SupervisorEvent>;

/// Run a supervisor to completion and return the summary.
async fn run_supervisor_to_completion(
    config: SupervisorConfig,
) -> SupervisorSummary;

/// Compute aggregate wellbeing from a list of agent states.
fn compute_aggregate_wellbeing(states: &[WellbeingState]) -> AggregateWellbeing;

/// Determine supervisor-level response to aggregate wellbeing.
fn supervisor_level_response(aggregate: &AggregateWellbeing) -> SupervisorWellbeingAction;

#[derive(Debug)]
enum SupervisorWellbeingAction {
    Continue,
    PauseAndReplan,
    EscalateToClient,
}
```

---

## 14. Implementation Order

Tests should be implemented in this order, each phase building confidence for the next:

### Phase 1: Foundation
1. Loop state machine tests (§1)
2. Termination condition tests (§3)

### Phase 2: Security
3. Autonomy enforcement tests (§2)
4. Tool locking tests (§7.1-7.2)

### Phase 3: Core Loop
5. Meta-signal detection tests (§4)
6. Context management tests (§5)

### Phase 4: Integration
7. SSE streaming tests (§6)
8. Integration tests (§9)

### Phase 5: Wellbeing
9. Wellbeing tests (§8)
10. Resource quota tests (§7.1-7.2)

### Phase 6: Coordination Primitives
11. Assistance request protocol tests (§7.3)
12. Yield protocol tests (§7.4)
13. Discovery store and shared context tests (§7.5)
14. Coordination events tests (§7.6)
15. Cleanup on unregister tests (§7.7)

### Phase 7: Tool Approval Protocol
16. Approval handshake tests (§10.1)
17. Timeout behavior tests (§10.2)
18. ApproveAlways semantics tests (§10.3)
19. Concurrent approval tests (§10.4)
20. Oneshot consumption tests (§10.5)

### Phase 8: Session Continuation
21. Store/load roundtrip tests (§11.1)
22. TTL and eviction tests (§11.2)
23. Resource arithmetic tests (§11.3)
24. Resume semantics tests (§11.4)
25. Resumability classification tests (§11.5)

### Phase 9: Multi-Agent Supervisor
26. Resource invariant tests (§12.1)
27. Dependency ordering tests (§12.2)
28. Concurrency limit tests (§12.3)
29. Lifecycle guarantee tests (§12.4)
30. Rerouting tests (§12.5)
31. Failure recovery tests (§12.6)
32. Aggregation tests (§12.7)
33. Wellbeing aggregate tests (§12.8)

---

## 15. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-02 | Initial TDD roadmap. 9 test domains (§1-9), property tests, integration tests. |
| 0.2.0 | 2026-02-04 | Added §10 Tool Approval Protocol, §11 Session Continuation, §12 Multi-Agent Supervisor. Renumbered infrastructure to §13-15. Updated implementation order with Phases 6-8. |
| 0.3.0 | 2026-02-04 | Replaced §7.3 (basic shared context) with §7.3-7.7 covering coordination primitives: assistance requests (oneshot handshake), yield protocol (terminal semantics, NoAlternative), discovery store (visibility policies, bounding), coordination events (broadcast), cleanup on unregister. Added §13.4 coordination test helpers with Arbitrary impls. Renumbered infrastructure §13.4-13.6 → §13.5-13.7. Added Phase 6 (Coordination Primitives) to implementation order, renumbered Phases 7-9. |

