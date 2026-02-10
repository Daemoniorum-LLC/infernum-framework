//! Agentic loop specification tests.
//!
//! These tests crystallize the requirements from AGENTIC-LOOP-SPEC.md and
//! AGENTIC-LOOP-TDD.md. They test at trust boundaries: state transitions,
//! resource limits, autonomy enforcement, and meta-signal detection.

use std::time::Duration;

use super::*;

// ===========================================================================
// §1. Loop State Machine
// ===========================================================================

mod state_machine {
    use super::*;

    // --- 1.1 Valid Transitions ---

    #[test]
    fn initialized_can_only_transition_to_generating() {
        let mut lp = AgenticLoop::new(LoopConfig::default());
        assert_eq!(lp.state(), LoopState::Initialized);

        assert!(lp.start().is_ok());
        assert_eq!(lp.state(), LoopState::Generating);
    }

    #[test]
    fn initialized_rejects_invalid_transitions() {
        let mut lp = AgenticLoop::new(LoopConfig::default());

        assert!(lp.generation_complete("output", 10).is_err());
        assert!(lp.tool_calls_detected(1).is_err());
        assert!(lp.execution_complete(vec![]).is_err());
        assert!(lp.continue_loop().is_err());
        assert!(lp.answer_detected("answer".into(), 0.9, vec![]).is_err());
    }

    #[test]
    fn generating_transitions_to_detecting() {
        let mut lp = AgenticLoop::new(LoopConfig::default());
        lp.start().unwrap();
        assert_eq!(lp.state(), LoopState::Generating);

        assert!(lp.generation_complete("output", 10).is_ok());
        assert_eq!(lp.state(), LoopState::Detecting);
    }

    #[test]
    fn generating_rejects_invalid_transitions() {
        let mut lp = AgenticLoop::new(LoopConfig::default());
        lp.start().unwrap();

        assert!(lp.start().is_err());
        assert!(lp.tool_calls_detected(1).is_err());
        assert!(lp.execution_complete(vec![]).is_err());
        assert!(lp.continue_loop().is_err());
    }

    #[test]
    fn detecting_can_transition_to_executing() {
        let mut lp = AgenticLoop::new(LoopConfig::default());
        lp.start().unwrap();
        lp.generation_complete("output", 10).unwrap();

        assert!(lp.tool_calls_detected(1).is_ok());
        assert_eq!(lp.state(), LoopState::Executing);
    }

    #[test]
    fn detecting_can_transition_to_completed() {
        let mut lp = AgenticLoop::new(LoopConfig::default());
        lp.start().unwrap();
        lp.generation_complete("answer", 10).unwrap();

        assert!(lp.answer_detected("42".into(), 0.95, vec![]).is_ok());
        assert_eq!(lp.state(), LoopState::Completed);
        assert!(lp.is_terminated());
    }

    #[test]
    fn detecting_can_transition_to_stuck() {
        let mut lp = AgenticLoop::new(LoopConfig::default());
        lp.start().unwrap();
        lp.generation_complete("stuck", 10).unwrap();

        assert!(lp
            .stuck_detected(
                vec![],
                StuckRequest::HumanIntervention {
                    reason: "help".into(),
                }
            )
            .is_ok());
        assert_eq!(lp.state(), LoopState::Stuck);
        assert!(lp.is_terminated());
    }

    #[test]
    fn detecting_can_transition_to_yielded() {
        let mut lp = AgenticLoop::new(LoopConfig::default());
        lp.start().unwrap();
        lp.generation_complete("yield", 10).unwrap();

        assert!(lp.yield_detected(None, "another agent".into()).is_ok());
        assert_eq!(lp.state(), LoopState::Yielded);
        assert!(lp.is_terminated());
    }

    #[test]
    fn executing_transitions_to_integrating() {
        let mut lp = AgenticLoop::new(LoopConfig::default());
        lp.start().unwrap();
        lp.generation_complete("call tool", 10).unwrap();
        lp.tool_calls_detected(1).unwrap();

        assert!(lp.execution_complete(vec![]).is_ok());
        assert_eq!(lp.state(), LoopState::Integrating);
    }

    #[test]
    fn integrating_transitions_back_to_generating() {
        let mut lp = AgenticLoop::new(LoopConfig::default());
        lp.start().unwrap();
        lp.generation_complete("call", 10).unwrap();
        lp.tool_calls_detected(1).unwrap();
        lp.execution_complete(vec![]).unwrap();

        assert!(lp.continue_loop().is_ok());
        assert_eq!(lp.state(), LoopState::Generating);
    }

    #[test]
    fn terminal_states_have_no_valid_transitions() {
        for terminal in [LoopState::Completed, LoopState::Stuck, LoopState::Yielded] {
            let mut lp = AgenticLoop::new(LoopConfig::default());
            lp.start().unwrap();
            lp.generation_complete("out", 10).unwrap();

            match terminal {
                LoopState::Completed => {
                    lp.answer_detected("done".into(), 1.0, vec![]).unwrap();
                },
                LoopState::Stuck => {
                    lp.stuck_detected(
                        vec![],
                        StuckRequest::HumanIntervention {
                            reason: "stuck".into(),
                        },
                    )
                    .unwrap();
                },
                LoopState::Yielded => {
                    lp.yield_detected(None, "yield".into()).unwrap();
                },
                _ => unreachable!(),
            }

            assert!(
                lp.is_terminated(),
                "state {:?} should be terminal",
                terminal
            );
            assert!(
                lp.valid_transitions().is_empty(),
                "terminal state {:?} should have no valid transitions",
                terminal
            );
            assert!(lp.start().is_err());
            assert!(lp.generation_complete("", 0).is_err());
            assert!(lp.tool_calls_detected(1).is_err());
            assert!(lp.continue_loop().is_err());
        }
    }

    #[test]
    fn non_terminal_states_have_at_least_one_transition() {
        let states = [
            LoopState::Initialized,
            LoopState::Generating,
            LoopState::Detecting,
            LoopState::Executing,
            LoopState::Integrating,
        ];
        for state in states {
            let mut lp = AgenticLoop::new(LoopConfig::default());
            // Drive to the desired state
            advance_to_state(&mut lp, state);
            assert!(
                !lp.valid_transitions().is_empty(),
                "non-terminal state {:?} should have transitions",
                state
            );
        }
    }

    // --- 1.2 Iteration Counting ---

    #[test]
    fn iteration_starts_at_zero() {
        let lp = AgenticLoop::new(LoopConfig::default());
        assert_eq!(lp.iteration(), 0);
    }

    #[test]
    fn iteration_increments_on_start() {
        let mut lp = AgenticLoop::new(LoopConfig::default());
        lp.start().unwrap();
        assert_eq!(lp.iteration(), 1);
    }

    #[test]
    fn iteration_increments_on_continue() {
        let mut lp = AgenticLoop::new(LoopConfig::default());
        lp.start().unwrap();
        assert_eq!(lp.iteration(), 1);

        // Complete one full cycle
        lp.generation_complete("call", 10).unwrap();
        lp.tool_calls_detected(1).unwrap();
        lp.execution_complete(vec![]).unwrap();
        lp.continue_loop().unwrap();

        assert_eq!(lp.iteration(), 2);
    }

    #[test]
    fn iteration_count_tracks_generation_phases() {
        let mut lp = AgenticLoop::new(LoopConfig {
            max_iterations: 5,
            ..Default::default()
        });

        lp.start().unwrap();
        assert_eq!(lp.iteration(), 1);

        for expected in 2..=4 {
            complete_one_cycle(&mut lp);
            assert_eq!(lp.iteration(), expected);
        }
    }

    // --- 1.3 Full cycle test ---

    #[test]
    fn full_cycle_generates_detects_executes_integrates_continues() {
        let mut lp = AgenticLoop::new(LoopConfig::default());

        // Initialized → Generating
        lp.start().unwrap();
        assert_eq!(lp.state(), LoopState::Generating);

        // Generating → Detecting
        lp.generation_complete("I'll use tool X", 50).unwrap();
        assert_eq!(lp.state(), LoopState::Detecting);

        // Detecting → Executing
        lp.tool_calls_detected(1).unwrap();
        assert_eq!(lp.state(), LoopState::Executing);

        // Executing → Integrating
        let result = AgenticToolResult {
            call_id: "call_1".into(),
            tool_name: "read_file".into(),
            status: ResultStatus::Success,
            data: serde_json::json!({"content": "file data"}),
            confidence: Confidence::Measured,
            latency_ms: 5,
            truncated: false,
        };
        lp.execution_complete(vec![result]).unwrap();
        assert_eq!(lp.state(), LoopState::Integrating);

        // Integrating → Generating (continue)
        lp.continue_loop().unwrap();
        assert_eq!(lp.state(), LoopState::Generating);
        assert_eq!(lp.iteration(), 2);

        // Second iteration → answer
        lp.generation_complete("The answer is 42", 30).unwrap();
        lp.answer_detected("42".into(), 0.95, vec![]).unwrap();
        assert_eq!(lp.state(), LoopState::Completed);

        // Summary
        let summary = lp.summary();
        assert_eq!(summary.iterations_completed, 2);
        assert_eq!(summary.tool_calls_made, 1);
        assert_eq!(summary.tokens_generated, 80); // 50 + 30
        assert_eq!(summary.tool_results_summary.len(), 1);
    }
}

// ===========================================================================
// §2. Autonomy Enforcement
// ===========================================================================

mod autonomy_enforcement {
    use super::*;

    #[test]
    fn forbidden_tools_are_denied() {
        let grant = AutonomyGrant::builder()
            .forbid(ToolPattern::Bash("rm *".into()))
            .forbid(ToolPattern::Bash("sudo *".into()))
            .build();

        assert!(grant.is_forbidden("bash", Some("rm -rf /")));
        assert!(grant.is_forbidden("bash", Some("sudo anything")));
    }

    #[test]
    fn allowed_tools_pass_without_approval() {
        let grant = AutonomyGrant::builder()
            .allow(ToolPattern::Read("**/*".into()))
            .allow(ToolPattern::Bash("git status".into()))
            .build();

        assert!(grant.is_allowed("read_file", Some("src/main.rs")));
        assert!(grant.is_allowed("bash", Some("git status")));
    }

    #[test]
    fn uncovered_tools_require_approval() {
        let grant = AutonomyGrant::builder()
            .allow(ToolPattern::Read("src/**/*.rs".into()))
            .build();

        // Not covered by any pattern → default to require approval
        assert!(grant.requires_approval("bash", Some("echo hello")));
        assert!(grant.requires_approval("write_file", Some("foo.txt")));
    }

    #[test]
    fn glob_patterns_match_correctly() {
        let grant = AutonomyGrant::builder()
            .allow(ToolPattern::Read("src/**/*.rs".into()))
            .forbid(ToolPattern::Read("src/**/secrets.*".into()))
            .build();

        // Allowed
        assert!(grant.is_allowed("read_file", Some("src/main.rs")));
        assert!(grant.is_allowed("read_file", Some("src/deep/nested/module.rs")));

        // Forbidden (more specific pattern wins due to priority)
        assert!(grant.is_forbidden("read_file", Some("src/config/secrets.toml")));
        assert!(grant.is_forbidden("read_file", Some("src/secrets.rs")));

        // Not covered
        assert!(grant.requires_approval("read_file", Some("tests/main.rs")));
    }

    #[test]
    fn bash_pattern_matching() {
        let grant = AutonomyGrant::builder()
            .allow(ToolPattern::Bash("git status".into()))
            .allow(ToolPattern::Bash("cargo *".into()))
            .forbid(ToolPattern::Bash("rm *".into()))
            .forbid(ToolPattern::Bash("sudo *".into()))
            .build();

        assert!(grant.is_allowed("bash", Some("git status")));
        assert!(grant.is_allowed("bash", Some("cargo build --release")));
        assert!(grant.is_forbidden("bash", Some("rm -rf /")));
        assert!(grant.is_forbidden("bash", Some("sudo anything")));
        assert!(grant.requires_approval("bash", Some("echo hello")));
    }

    #[test]
    fn tool_name_pattern_matching() {
        let grant = AutonomyGrant::builder()
            .allow(ToolPattern::Tool("calculator".into()))
            .forbid(ToolPattern::Tool("dangerous_*".into()))
            .build();

        assert!(grant.is_allowed("calculator", None));
        assert!(grant.is_forbidden("dangerous_tool", None));
        assert!(grant.requires_approval("unknown_tool", None));
    }

    #[test]
    fn forbidden_always_wins_over_allowed() {
        // Even if a pattern matches both allowed and forbidden, forbidden wins
        let grant = AutonomyGrant::builder()
            .allow(ToolPattern::Bash("*".into()))
            .forbid(ToolPattern::Bash("rm *".into()))
            .build();

        assert!(grant.is_forbidden("bash", Some("rm -rf /")));
        assert!(grant.is_allowed("bash", Some("echo hello")));
    }

    // --- 2.3 Grant Narrowing ---

    #[test]
    fn client_cannot_allow_server_forbidden() {
        let server = AutonomyGrant::builder()
            .forbid(ToolPattern::Bash("rm *".into()))
            .build();

        let client = AutonomyGrant::builder()
            .allow(ToolPattern::Bash("rm temp.txt".into()))
            .build();

        let effective = server.narrow_with(&client);

        // Server's forbid wins
        assert!(effective.is_forbidden("bash", Some("rm temp.txt")));
    }

    #[test]
    fn client_can_narrow_allowed_to_require_approval() {
        let server = AutonomyGrant::builder()
            .allow(ToolPattern::Write("**/*".into()))
            .build();

        let client = AutonomyGrant::builder()
            .require_approval(ToolPattern::Write("**/*.rs".into()))
            .build();

        let effective = server.narrow_with(&client);

        // Client's narrowing should add require_approval
        // The .rs files should now require approval since client added that pattern
        assert!(effective.requires_approval("write_file", Some("main.rs")));
    }

    #[test]
    fn client_can_add_new_forbidden_patterns() {
        let server = AutonomyGrant::builder()
            .allow(ToolPattern::Bash("*".into()))
            .build();

        let client = AutonomyGrant::builder()
            .forbid(ToolPattern::Bash("git push *".into()))
            .build();

        let effective = server.narrow_with(&client);
        assert!(effective.is_forbidden("bash", Some("git push origin main")));
    }
}

// ===========================================================================
// §3. Termination Conditions
// ===========================================================================

mod termination {
    use super::*;

    // --- 3.1 Natural Termination ---

    #[test]
    fn answer_terminates_loop() {
        let mut lp = create_loop_in_detecting();

        lp.answer_detected("The answer is 42".into(), 0.95, vec![])
            .unwrap();

        assert!(lp.is_terminated());
        assert!(matches!(
            lp.termination_reason(),
            Some(TerminationReason::Natural(
                NaturalTermination::AnswerProvided { .. }
            ))
        ));
    }

    #[test]
    fn stuck_terminates_loop() {
        let mut lp = create_loop_in_detecting();

        lp.stuck_detected(
            vec![AttemptSummary {
                description: "Tried JSON".into(),
                outcome: "Not found".into(),
            }],
            StuckRequest::Clarification(vec!["Which format?".into()]),
        )
        .unwrap();

        assert!(lp.is_terminated());
        assert!(matches!(
            lp.termination_reason(),
            Some(TerminationReason::Natural(
                NaturalTermination::AgentStuck { .. }
            ))
        ));
    }

    #[test]
    fn yield_terminates_loop() {
        let mut lp = create_loop_in_detecting();

        lp.yield_detected(
            Some("Found config but can't parse YAML".into()),
            "needs yaml expertise".into(),
        )
        .unwrap();

        assert!(lp.is_terminated());
        assert!(matches!(
            lp.termination_reason(),
            Some(TerminationReason::Natural(
                NaturalTermination::AgentYielded { .. }
            ))
        ));
    }

    // --- 3.2 Resource Termination ---

    #[test]
    fn max_iterations_terminates() {
        let mut lp = AgenticLoop::new(LoopConfig {
            max_iterations: 3,
            ..Default::default()
        });

        lp.start().unwrap(); // iteration 1
        complete_one_cycle(&mut lp); // iteration 2
        complete_one_cycle(&mut lp); // iteration 3

        // 4th iteration should fail
        lp.generation_complete("out", 10).unwrap();
        lp.tool_calls_detected(1).unwrap();
        lp.execution_complete(vec![]).unwrap();
        let result = lp.continue_loop();

        assert!(result.is_err());
        assert!(matches!(
            lp.termination_reason(),
            Some(TerminationReason::Resource(
                ResourceTermination::MaxIterations { .. }
            ))
        ));
    }

    #[test]
    fn token_budget_terminates() {
        let mut lp = AgenticLoop::new(LoopConfig {
            max_tokens: 100,
            ..Default::default()
        });

        lp.start().unwrap();

        // Generate more than budget
        let result = lp.generation_complete("x".repeat(200).as_str(), 150);

        assert!(result.is_err());
        assert!(matches!(
            lp.termination_reason(),
            Some(TerminationReason::Resource(
                ResourceTermination::TokenBudgetExhausted { .. }
            ))
        ));
    }

    #[test]
    fn tool_call_limit_terminates() {
        let mut lp = AgenticLoop::new(LoopConfig {
            max_tool_calls: 2,
            ..Default::default()
        });

        lp.start().unwrap();
        lp.generation_complete("call", 10).unwrap();
        lp.tool_calls_detected(2).unwrap();
        lp.execution_complete(vec![]).unwrap();
        lp.continue_loop().unwrap();

        lp.generation_complete("more calls", 10).unwrap();
        let result = lp.tool_calls_detected(1); // Would be 3 total, limit is 2

        assert!(result.is_err());
        assert!(matches!(
            lp.termination_reason(),
            Some(TerminationReason::Resource(
                ResourceTermination::ToolCallLimitReached { .. }
            ))
        ));
    }

    #[test]
    fn wall_time_terminates() {
        let mut lp = AgenticLoop::new(LoopConfig {
            max_wall_time: Duration::from_millis(50),
            ..Default::default()
        });

        lp.start().unwrap();
        std::thread::sleep(Duration::from_millis(80));

        // Next operation should detect wall time exceeded
        let result = lp.generation_complete("output", 10);

        assert!(result.is_err());
    }

    #[test]
    fn external_termination_works() {
        let mut lp = AgenticLoop::new(LoopConfig::default());
        lp.start().unwrap();

        lp.force_terminate(ExternalTermination::ClientCancelled);

        assert!(lp.is_terminated());
        assert!(matches!(
            lp.termination_reason(),
            Some(TerminationReason::External(
                ExternalTermination::ClientCancelled
            ))
        ));
    }

    // --- 3.3 Partial Progress Preservation ---

    #[test]
    fn terminated_loop_preserves_tool_results() {
        let mut lp = AgenticLoop::new(LoopConfig {
            max_iterations: 2,
            ..Default::default()
        });

        lp.start().unwrap();
        lp.generation_complete("call", 10).unwrap();
        lp.tool_calls_detected(1).unwrap();

        let result = AgenticToolResult {
            call_id: "call_1".into(),
            tool_name: "read_file".into(),
            status: ResultStatus::Success,
            data: serde_json::json!({"content": "data"}),
            confidence: Confidence::Measured,
            latency_ms: 5,
            truncated: false,
        };
        lp.execution_complete(vec![result]).unwrap();
        lp.continue_loop().unwrap();

        // Terminate in second iteration
        lp.generation_complete("call again", 10).unwrap();
        lp.tool_calls_detected(1).unwrap();

        let result2 = AgenticToolResult {
            call_id: "call_2".into(),
            tool_name: "bash".into(),
            status: ResultStatus::Success,
            data: serde_json::json!({"output": "ok"}),
            confidence: Confidence::Measured,
            latency_ms: 12,
            truncated: false,
        };
        lp.execution_complete(vec![result2]).unwrap();

        // Can't continue — max iterations
        let _ = lp.continue_loop();

        let summary = lp.summary();
        assert_eq!(summary.tool_results_summary.len(), 2);
        assert!(summary
            .tool_results_summary
            .iter()
            .any(|r| r.call_id == "call_1"));
        assert!(summary
            .tool_results_summary
            .iter()
            .any(|r| r.call_id == "call_2"));
    }

    #[test]
    fn stuck_preserves_exploration_history() {
        let mut lp = AgenticLoop::new(LoopConfig::default());
        lp.start().unwrap();

        // Record exploration branches
        lp.record_exploration(ExplorationBranch {
            description: "Tried JSON config".into(),
            tool_calls: vec!["read_file(config.json)".into()],
            productive: false,
            findings: None,
        });
        lp.record_exploration(ExplorationBranch {
            description: "Tried YAML config".into(),
            tool_calls: vec!["read_file(config.yaml)".into()],
            productive: false,
            findings: None,
        });

        lp.generation_complete("stuck", 10).unwrap();
        lp.stuck_detected(
            vec![
                AttemptSummary {
                    description: "Tried JSON".into(),
                    outcome: "Not found".into(),
                },
                AttemptSummary {
                    description: "Tried YAML".into(),
                    outcome: "Not found".into(),
                },
            ],
            StuckRequest::Clarification(vec!["Which config format?".into()]),
        )
        .unwrap();

        let summary = lp.summary();
        assert_eq!(summary.exploration_summary.len(), 2);
        assert!(summary
            .exploration_summary
            .iter()
            .any(|b| b.description.contains("JSON")));
        assert!(summary
            .exploration_summary
            .iter()
            .any(|b| b.description.contains("YAML")));
    }

    #[test]
    fn summary_reports_correct_metrics() {
        let mut lp = AgenticLoop::new(LoopConfig::default());
        lp.start().unwrap();

        lp.generation_complete("output1", 100).unwrap();
        lp.tool_calls_detected(2).unwrap();
        lp.execution_complete(vec![]).unwrap();
        lp.continue_loop().unwrap();

        lp.generation_complete("output2", 200).unwrap();
        lp.answer_detected("done".into(), 0.9, vec![]).unwrap();

        let summary = lp.summary();
        assert_eq!(summary.iterations_completed, 2);
        assert_eq!(summary.tool_calls_made, 2);
        assert_eq!(summary.tokens_generated, 300);
    }
}

// ===========================================================================
// §4. Meta-Signal Detection
// ===========================================================================

mod meta_signals {
    use super::*;

    // --- 4.1 Explicit Signals ---

    #[test]
    fn explicit_answer_signal_detected() {
        let output = r#"
            <answer confidence="0.92">
            The result is 42.
            <caveat>Assuming standard arithmetic</caveat>
            </answer>
        "#;

        let signal = detect_meta_signal(output, &DetectionConfig::default());

        assert!(matches!(signal, Some(MetaSignal::Answer { .. })));
        if let Some(MetaSignal::Answer {
            confidence,
            caveats,
            content,
            ..
        }) = signal
        {
            assert!((confidence - 0.92).abs() < 0.01);
            assert_eq!(caveats.len(), 1);
            assert!(caveats[0].contains("standard arithmetic"));
            assert!(content.contains("42"));
        }
    }

    #[test]
    fn explicit_uncertain_signal_detected() {
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
        if let Some(MetaSignal::Uncertain {
            partial_answer,
            missing_information,
            would_help,
        }) = signal
        {
            assert!(partial_answer.is_some());
            assert_eq!(missing_information.len(), 2);
            assert_eq!(would_help.len(), 1);
        }
    }

    #[test]
    fn explicit_stuck_signal_detected() {
        let output = r#"
            <stuck>
            <attempt>Tried reading config.json</attempt>
            <attempt>Tried reading config.yaml</attempt>
            <hypothesis>The config might not exist</hypothesis>
            <request>I need clarification on the config location</request>
            </stuck>
        "#;

        let signal = detect_meta_signal(output, &DetectionConfig::default());

        assert!(matches!(signal, Some(MetaSignal::Stuck { .. })));
        if let Some(MetaSignal::Stuck {
            attempts,
            hypothesis,
            ..
        }) = signal
        {
            assert_eq!(attempts.len(), 2);
            assert!(hypothesis.is_some());
            assert!(hypothesis.as_deref().unwrap().contains("might not exist"));
        }
    }

    #[test]
    fn explicit_yield_signal_detected() {
        let output = r#"
            <yield>
            <partial>Found the config structure</partial>
            <expertise>yaml-parsing</expertise>
            <expertise>schema-validation</expertise>
            </yield>
        "#;

        let signal = detect_meta_signal(output, &DetectionConfig::default());

        assert!(matches!(signal, Some(MetaSignal::Yield { .. })));
        if let Some(MetaSignal::Yield {
            partial_progress,
            suggested_expertise,
        }) = signal
        {
            assert!(partial_progress.is_some());
            assert_eq!(suggested_expertise.len(), 2);
        }
    }

    #[test]
    fn explicit_thinking_signal_detected() {
        let output = r#"
            <thinking direction="analyzing dependencies" steps="3">
            Looking at the import graph
            </thinking>
        "#;

        let signal = detect_meta_signal(output, &DetectionConfig::default());

        assert!(matches!(signal, Some(MetaSignal::Thinking { .. })));
        if let Some(MetaSignal::Thinking {
            direction,
            estimated_steps,
        }) = signal
        {
            assert!(direction.contains("dependencies"));
            assert_eq!(estimated_steps, Some(3));
        }
    }

    // --- 4.2 Implicit Signals ---

    #[test]
    fn implicit_uncertainty_detected() {
        let output = "I'm not certain about this, but it might be a configuration issue.";

        let config = DetectionConfig {
            detect_implicit: true,
            ..Default::default()
        };
        let signal = detect_meta_signal(output, &config);

        assert!(matches!(signal, Some(MetaSignal::Uncertain { .. })));
    }

    #[test]
    fn implicit_stuck_detected() {
        let output = "I've tried several approaches but I'm not making progress on this issue.";

        let config = DetectionConfig {
            detect_implicit: true,
            ..Default::default()
        };
        let signal = detect_meta_signal(output, &config);

        assert!(matches!(signal, Some(MetaSignal::Stuck { .. })));
    }

    #[test]
    fn implicit_detection_disabled_returns_none() {
        let output = "I'm not certain about this answer.";

        let config = DetectionConfig {
            detect_implicit: false,
            ..Default::default()
        };
        let signal = detect_meta_signal(output, &config);

        assert!(signal.is_none());
    }

    #[test]
    fn explicit_takes_priority_over_implicit() {
        let output = r#"
            I'm not certain, but here's what I found:
            <answer confidence="0.7">
            The config file is at /etc/app.toml
            </answer>
        "#;

        let signal = detect_meta_signal(output, &DetectionConfig::default());

        // Should detect the explicit answer, not the implicit uncertainty
        assert!(matches!(signal, Some(MetaSignal::Answer { .. })));
    }

    #[test]
    fn no_signal_in_plain_output() {
        let output = "I'll read the file and check the configuration.";

        let signal = detect_meta_signal(output, &DetectionConfig::default());
        assert!(signal.is_none());
    }

    #[test]
    fn answer_confidence_clamped_to_0_1() {
        let output = r#"<answer confidence="1.5">Over-confident answer</answer>"#;
        let signal = detect_meta_signal(output, &DetectionConfig::default());

        if let Some(MetaSignal::Answer { confidence, .. }) = signal {
            assert!(confidence <= 1.0);
        }
    }
}

// ===========================================================================
// §5. Token Budget
// ===========================================================================

mod token_budget {
    use super::*;

    #[test]
    fn new_budget_has_full_remaining() {
        let budget = TokenBudget::new(1000);
        assert_eq!(budget.remaining(), 1000);
        assert!(!budget.is_exhausted());
    }

    #[test]
    fn consume_reduces_remaining() {
        let mut budget = TokenBudget::new(1000);
        assert!(budget.consume(300));
        assert_eq!(budget.remaining(), 700);
    }

    #[test]
    fn consume_past_limit_exhausts_budget() {
        let mut budget = TokenBudget::new(100);
        assert!(!budget.consume(150));
        assert!(budget.is_exhausted());
    }

    #[test]
    fn consume_exact_limit_exhausts() {
        let mut budget = TokenBudget::new(100);
        assert!(!budget.consume(100));
        assert!(budget.is_exhausted());
        assert_eq!(budget.remaining(), 0);
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Advance a loop to the specified state for testing.
fn advance_to_state(lp: &mut AgenticLoop, target: LoopState) {
    match target {
        LoopState::Initialized => {}, // already there
        LoopState::Generating => {
            lp.start().unwrap();
        },
        LoopState::Detecting => {
            lp.start().unwrap();
            lp.generation_complete("output", 10).unwrap();
        },
        LoopState::Executing => {
            lp.start().unwrap();
            lp.generation_complete("output", 10).unwrap();
            lp.tool_calls_detected(1).unwrap();
        },
        LoopState::Integrating => {
            lp.start().unwrap();
            lp.generation_complete("output", 10).unwrap();
            lp.tool_calls_detected(1).unwrap();
            lp.execution_complete(vec![]).unwrap();
        },
        LoopState::Completed | LoopState::Stuck | LoopState::Yielded => {
            panic!("Use terminal state helpers directly");
        },
    }
}

/// Drive a loop through one complete non-terminal cycle.
fn complete_one_cycle(lp: &mut AgenticLoop) {
    lp.generation_complete("output", 10).unwrap();
    lp.tool_calls_detected(1).unwrap();
    lp.execution_complete(vec![]).unwrap();
    lp.continue_loop().unwrap();
}

/// Create a loop already in the Detecting state.
fn create_loop_in_detecting() -> AgenticLoop {
    let mut lp = AgenticLoop::new(LoopConfig::default());
    lp.start().unwrap();
    lp.generation_complete("output", 10).unwrap();
    lp
}
