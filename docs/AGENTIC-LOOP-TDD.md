# Agentic Loop TDD Roadmap

**Version:** 0.1.0
**Status:** Test Specification
**Date:** 2026-02-02
**Spec Reference:** AGENTIC-LOOP-SPEC.md v0.1.1

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

### 7.3 Shared Context

```rust
#[test]
fn test_discoveries_shared_between_agents() {
    let coordinator = AgentCoordinator::new();

    let agent1 = AgentId::new();
    let agent2 = AgentId::new();

    // Agent 1 makes a discovery
    coordinator.share_discovery(&agent1, Discovery {
        category: "configuration",
        content: "Config is at /etc/app/config.yaml".into(),
        confidence: 0.95,
    }).await;

    // Agent 2 should see it
    let context = coordinator.get_shared_context(&agent2).await;
    assert!(context.discoveries.iter().any(|d| d.content.contains("config.yaml")));
}

#[test]
fn test_agent_can_filter_shared_context() {
    let coordinator = AgentCoordinator::new();

    let agent1 = AgentId::new();

    // Multiple discoveries
    coordinator.share_discovery(&agent1, discovery("config", "Found config")).await;
    coordinator.share_discovery(&agent1, discovery("database", "DB is postgres")).await;
    coordinator.share_discovery(&agent1, discovery("config", "Config format is YAML")).await;

    // Filter by category
    let context = coordinator.get_shared_context_filtered(&agent1, |d| d.category == "config").await;

    assert_eq!(context.discoveries.len(), 2);
    assert!(context.discoveries.iter().all(|d| d.category == "config"));
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

## 10. Test Infrastructure

### 10.1 Property Test Generators

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

### 10.2 Test Fixtures

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

### 10.3 Mocks

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

---

## 11. Implementation Order

Tests should be implemented in this order, each phase building confidence for the next:

### Phase 1: Foundation (Week 1)
1. Loop state machine tests (§1)
2. Termination condition tests (§3)

### Phase 2: Security (Week 1-2)
3. Autonomy enforcement tests (§2)
4. Tool locking tests (§7.1-7.2)

### Phase 3: Core Loop (Week 2)
5. Meta-signal detection tests (§4)
6. Context management tests (§5)

### Phase 4: Integration (Week 3)
7. SSE streaming tests (§6)
8. Integration tests (§9)

### Phase 5: Wellbeing (Week 3)
9. Wellbeing tests (§8)
10. Shared context tests (§7.3)

---

## 12. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-02 | Initial TDD roadmap. 8 test domains, property tests, integration tests. |

