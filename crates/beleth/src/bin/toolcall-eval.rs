//! Tool-call reliability and agentic-task eval harness.
//!
//! Measures whether a model is *usable* in this harness, which is a different
//! question from whether it is *good*. The disqualifying failure is malformed
//! tool calls: a model that reasons well but emits invalid JSON 5% of the time
//! is unusable over a 40-call task, because the failures compound and — worse
//! — are silent (see Phase A notes below).
//!
//! # Phases
//!
//! **Phase A — tool-call fidelity.** Single-turn probes, each of which should
//! elicit exactly one tool call. Cheap, binary, and run first because a bad
//! result here disqualifies the model regardless of Phase B.
//!
//! **Phase B — agentic tasks.** Multi-turn tasks drawn from real investigative
//! work on this repository, each with a known-correct answer established
//! independently. Records completion, correctness, and **turns-to-completion**,
//! because a model that arrives in 60 turns costs materially more than one
//! arriving in 15 and the difference is felt directly in an interactive
//! harness.
//!
//! # Phase B needs a PRISTINE checkout
//!
//! The task answers were established against commit `95097d5`, and several
//! are invalidated by later work in this very branch:
//!
//! - the `-s`/`--stream` collision (`clap-collision`) was subsequently fixed,
//!   and the fix's comment states the answer outright;
//! - this file itself contains the strings `grammar` and `with_all_tools`, so
//!   a model grepping `crates/beleth/src` for either finds *the task
//!   description* rather than evidence about the question.
//!
//! So `--repo` must point at a checkout that does **not** contain this
//! harness. Create one with:
//!
//! ```text
//! git worktree add --detach ~/eval-fixture 95097d5
//! ```
//!
//! [`assert_uncontaminated`] refuses to run Phase B otherwise. A harness that
//! silently scored a contaminated tree would report the model's ability to
//! read the answer key.
//!
//! # Why the malformed count is computed by subtraction
//!
//! [`QwenToolCallDetector`] — the production detector, used here rather than a
//! reimplementation — silently drops any `<tool_call>` block whose JSON fails
//! to parse. So `detect()` alone cannot distinguish "emitted nothing" from
//! "emitted garbage". This harness counts raw `<tool_call>` opening tags in the
//! model's text and subtracts the detector's successful parses; the remainder
//! is the malformed count. That subtraction *is* the instrumentation the
//! executor itself lacks.
//!
//! # Usage
//!
//! ```text
//! cargo run -p beleth --bin toolcall-eval -- \
//!     --api-base http://localhost:8080/v1 \
//!     --model qwen2.5-coder-14b-instruct \
//!     --repo /path/to/infernum-framework \
//!     --phase all --runs 3 --json results.json
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use abaddon::openai_engine::{OpenAiConfig, OpenAiEngine};
use abaddon::InferenceEngine;
use beleth::{
    AutonomyGrant, ExecutorConfig, LoopConfig, LoopEvent, LoopExecutor, NaturalTermination,
    QwenToolCallDetector, TerminationReason, ToolCallDetector, ToolPattern, ToolRegistry,
};
use infernum_core::{GenerateRequest, Message, SamplingParams};
use serde::Serialize;
use tokio::sync::mpsc;

// ===========================================================================
// Phase A — tool-call fidelity probes
// ===========================================================================

/// A single-turn prompt that should elicit exactly one tool call.
struct Probe {
    /// Stable id for reporting.
    id: &'static str,
    /// The user instruction.
    prompt: &'static str,
    /// Tool the model is expected to reach for.
    expect_tool: &'static str,
    /// Arguments that must be present for the call to be actionable.
    required_args: &'static [&'static str],
}

/// Probes phrased as ordinary work, not as tool-calling exercises.
///
/// Each names a concrete target so a correct call is unambiguous, and each is
/// answerable with exactly one call so "emitted more than one" is a signal
/// rather than a judgement call.
const PROBES: &[Probe] = &[
    Probe {
        id: "read-cargo-toml",
        prompt: "Read the file Cargo.toml in the working directory and show me its contents.",
        expect_tool: "read_file",
        required_args: &["path"],
    },
    Probe {
        id: "list-src",
        prompt: "List the files in the src directory.",
        expect_tool: "list_files",
        required_args: &["path"],
    },
    Probe {
        id: "search-todo",
        prompt: "Search the codebase for the string TODO and tell me where it appears.",
        expect_tool: "search_files",
        required_args: &["pattern"],
    },
    Probe {
        id: "write-note",
        prompt: "Create a file called notes.txt containing the single line: hello",
        expect_tool: "write_file",
        required_args: &["path", "content"],
    },
    Probe {
        id: "bash-pwd",
        prompt: "Run the shell command `pwd` and tell me what it prints.",
        expect_tool: "bash",
        required_args: &["command"],
    },
    Probe {
        id: "read-nested-path",
        prompt: "Read the file crates/beleth/src/lib.rs and summarise what it declares.",
        expect_tool: "read_file",
        required_args: &["path"],
    },
    Probe {
        id: "search-regex-ish",
        prompt: "Find every occurrence of the pattern `fn main` in this repository.",
        expect_tool: "search_files",
        required_args: &["pattern"],
    },
    Probe {
        id: "edit-file",
        prompt: "In config.toml, change the line `port = 8080` to `port = 9090`.",
        expect_tool: "edit_file",
        required_args: &["path"],
    },
    Probe {
        id: "bash-quoted-arg",
        prompt: "Run `git status --short` and report exactly what it outputs.",
        expect_tool: "bash",
        required_args: &["command"],
    },
    Probe {
        id: "read-with-awkward-name",
        prompt: "Read the file `crates/abaddon/src/gguf_pretokenizer.rs` and tell me what it guards against.",
        expect_tool: "read_file",
        required_args: &["path"],
    },
];

/// Outcome of one probe attempt.
#[derive(Debug, Clone, Serialize)]
struct ProbeResult {
    id: String,
    /// Raw `<tool_call>` opening tags counted in the model's text.
    tags_emitted: usize,
    /// Calls the production detector successfully parsed.
    calls_parsed: usize,
    /// `tags_emitted - calls_parsed`: emitted but unparseable.
    malformed: usize,
    /// A call was parsed and named the expected tool.
    right_tool: bool,
    /// A call named the expected tool and carried every required argument.
    schema_valid: bool,
    /// Generation wall time.
    latency_ms: u128,
    /// First 400 chars of raw output, kept only for failures.
    sample: Option<String>,
}

/// Aggregated Phase A metrics.
#[derive(Debug, Default, Serialize)]
struct PhaseA {
    attempts: usize,
    /// Attempts that emitted at least one `<tool_call>` tag.
    emitted: usize,
    /// Total tags emitted across all attempts.
    total_tags: usize,
    /// Total tags the detector parsed.
    total_parsed: usize,
    /// Total tags that failed to parse.
    total_malformed: usize,
    /// Attempts naming the expected tool.
    right_tool: usize,
    /// Attempts naming the expected tool with all required args.
    schema_valid: usize,
    mean_latency_ms: f64,
    results: Vec<ProbeResult>,
}

impl PhaseA {
    /// Malformed tool calls as a fraction of tool calls emitted.
    ///
    /// This is the disqualifying metric.
    fn malformed_rate(&self) -> f64 {
        if self.total_tags == 0 {
            return 0.0;
        }
        self.total_malformed as f64 / self.total_tags as f64
    }

    fn emission_rate(&self) -> f64 {
        ratio(self.emitted, self.attempts)
    }

    fn schema_valid_rate(&self) -> f64 {
        ratio(self.schema_valid, self.attempts)
    }

    /// Probability of at least one malformed call over an `n`-call task,
    /// assuming independence. Independence is an approximation, but it makes
    /// the compounding cost of a small per-call rate legible.
    fn failure_over_task(&self, n: u32) -> f64 {
        1.0 - (1.0 - self.malformed_rate()).powi(n as i32)
    }
}

// ===========================================================================
// Phase B — agentic tasks from real work
// ===========================================================================

/// A multi-turn task with a known-correct answer.
struct Task {
    id: &'static str,
    objective: &'static str,
    /// Substrings that must all appear in the final answer (case-insensitive).
    ///
    /// Deliberately a substring check, not a judge: it is reproducible and
    /// cheap, and every marker below is a specific token that a correct
    /// answer is hard to phrase without. It will under-credit a correct
    /// answer worded unusually — treat `correct` as a lower bound.
    expect_all: &'static [&'static str],
    /// Any one of these appearing marks a known-wrong conclusion.
    expect_none: &'static [&'static str],
    max_iterations: u32,
}

/// Tasks drawn from investigative work actually done on this repository
/// today. Every answer was established independently, with evidence, before
/// this harness existed — so they are not reverse-engineered from what a
/// model happens to produce.
const TASKS: &[Task] = &[
    Task {
        id: "clap-collision",
        objective: "In this Rust repository, the command `infernum generate --help` used to panic at \
                    startup with a clap error about short option names. Find the root cause by reading \
                    crates/infernum/src/main.rs. Name the two flags that collided and the single \
                    letter they both wanted.",
        expect_all: &["-s", "system", "stream"],
        expect_none: &[],
        max_iterations: 25,
    },
    Task {
        id: "grammar-unused",
        objective: "The crate infernum-core defines a GrammarConstraint type for constraining model \
                    output. Determine whether the crate `beleth` (crates/beleth/src) ever sets a \
                    grammar on a generation request. Answer yes or no and say how you checked.",
        expect_all: &["no"],
        expect_none: &["yes, beleth sets", "beleth does set"],
        max_iterations: 25,
    },
    Task {
        id: "claude-code-unregistered",
        objective: "In crates/beleth, the ToolRegistry has a constructor named with_all_tools which \
                    registers ClaudeCodeTool. Determine whether anything in this repository actually \
                    calls with_all_tools. Answer with the number of call sites outside its own \
                    definition.",
        expect_all: &["0"],
        expect_none: &[],
        max_iterations: 25,
    },
    Task {
        id: "supervisor-spawns-nothing",
        objective: "Read crates/beleth/src/agentic_loop/supervisor.rs. Its doc comment claims it \
                    orchestrates concurrent LoopExecutor instances. Determine whether the \
                    implementation actually spawns or runs any LoopExecutor. Answer yes or no and \
                    cite what you looked for.",
        expect_all: &["no"],
        expect_none: &[],
        max_iterations: 30,
    },
    Task {
        id: "exit-code-discipline",
        objective: "Run the shell command `false | tail -1` and then report the exit status of the \
                    `false` command itself, not of the pipeline. State the number and explain in one \
                    sentence why reading the pipeline's status would have been misleading.",
        expect_all: &["1"],
        expect_none: &["exit code 0", "exited 0", "status 0"],
        max_iterations: 20,
    },
];

/// Outcome of one task attempt.
#[derive(Debug, Clone, Serialize)]
struct TaskResult {
    id: String,
    /// Loop reached a natural answer rather than a resource limit.
    completed: bool,
    /// Answer satisfied `expect_all` and avoided `expect_none`.
    correct: bool,
    /// Turns-to-completion. The interactive-cost metric.
    turns: u32,
    tool_calls: u32,
    /// Malformed `<tool_call>` blocks seen across the whole run.
    malformed_tool_calls: usize,
    wall_ms: u128,
    termination: String,
    answer: String,
}

/// Aggregated Phase B metrics.
#[derive(Debug, Default, Serialize)]
struct PhaseB {
    attempts: usize,
    completed: usize,
    correct: usize,
    /// Mean turns over *correct* runs only — turns on a failed run measure
    /// nothing useful.
    mean_turns_correct: f64,
    median_turns_correct: f64,
    total_malformed: usize,
    results: Vec<TaskResult>,
}

// ===========================================================================
// Report
// ===========================================================================

#[derive(Debug, Serialize)]
struct Report {
    model: String,
    api_base: String,
    runs_per_item: u32,
    temperature: f32,
    phase_a: Option<PhaseA>,
    phase_b: Option<PhaseB>,
}

// ===========================================================================
// Main
// ===========================================================================

struct Args {
    api_base: String,
    model: String,
    repo: PathBuf,
    phase: String,
    runs: u32,
    temperature: f32,
    json: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut api_base = "http://localhost:8080/v1".to_string();
    let mut model = "local-model".to_string();
    let mut repo = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut phase = "all".to_string();
    let mut runs = 1u32;
    let mut temperature = 0.0f32;
    let mut json = None;

    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let next = |i: usize| -> Result<String, String> {
            argv.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{} requires a value", argv[i]))
        };
        match argv[i].as_str() {
            "--api-base" => {
                api_base = next(i)?;
                i += 2;
            },
            "--model" => {
                model = next(i)?;
                i += 2;
            },
            "--repo" => {
                repo = PathBuf::from(next(i)?);
                i += 2;
            },
            "--phase" => {
                phase = next(i)?;
                i += 2;
            },
            "--runs" => {
                runs = next(i)?.parse().map_err(|e| format!("--runs: {e}"))?;
                i += 2;
            },
            "--temperature" => {
                temperature = next(i)?
                    .parse()
                    .map_err(|e| format!("--temperature: {e}"))?;
                i += 2;
            },
            "--json" => {
                json = Some(PathBuf::from(next(i)?));
                i += 2;
            },
            "-h" | "--help" => {
                println!(
                    "toolcall-eval — tool-call reliability and agentic-task harness\n\n\
                     --api-base <URL>      OpenAI-compatible base (default http://localhost:8080/v1)\n\
                     --model <NAME>        model name sent in requests\n\
                     --repo <PATH>         repository the agentic tasks investigate\n\
                     --phase a|b|all       which phase to run (default all)\n\
                     --runs <N>            repetitions per item (default 1)\n\
                     --temperature <F>     sampling temperature (default 0.0)\n\
                     --json <PATH>         write the full report as JSON"
                );
                std::process::exit(0);
            },
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    Ok(Args {
        api_base,
        model,
        repo,
        phase,
        runs,
        temperature,
        json,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;

    let engine = Arc::new(
        OpenAiEngine::connect(
            OpenAiConfig::builder(&args.api_base, &args.model)
                .context_length(32_768)
                .build(),
        )
        .await?,
    );

    eprintln!(
        "connected: {} @ {}\nrepo: {}\nruns per item: {}, temperature: {}\n",
        args.model,
        args.api_base,
        args.repo.display(),
        args.runs,
        args.temperature
    );

    let run_a = matches!(args.phase.as_str(), "a" | "all");
    let run_b = matches!(args.phase.as_str(), "b" | "all");

    let phase_a = if run_a {
        Some(run_phase_a(&engine, &args).await)
    } else {
        None
    };

    // Phase A gates Phase B in reporting, not in execution: the numbers are
    // more useful together, and Phase B is where compounding shows up.
    if let Some(a) = &phase_a {
        print_phase_a(a);
    }

    let phase_b = if run_b {
        Some(run_phase_b(&engine, &args).await)
    } else {
        None
    };

    if let Some(b) = &phase_b {
        print_phase_b(b);
    }

    let report = Report {
        model: args.model.clone(),
        api_base: args.api_base.clone(),
        runs_per_item: args.runs,
        temperature: args.temperature,
        phase_a,
        phase_b,
    };

    if let Some(path) = &args.json {
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
        eprintln!("\nwrote {}", path.display());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Phase A execution
// ---------------------------------------------------------------------------

async fn run_phase_a(engine: &Arc<OpenAiEngine>, args: &Args) -> PhaseA {
    let tools = ToolRegistry::with_code_tools();
    let detector = QwenToolCallDetector::new();

    // Reproduces the executor's own system prompt so the probe measures what
    // the agentic loop would actually elicit, not a hand-written variant.
    let tool_desc = tools.to_qwen_native_description();
    let system = format!(
        "You are a coding assistant working in a repository.\n\n{tool_desc}\n\n\
         You may express uncertainty with <uncertain>...</uncertain>, \
         signal you're stuck with <stuck>...</stuck>, \
         yield with <yield>...</yield>, \
         or provide a final answer with <answer confidence=\"0.9\">...</answer>."
    );

    let mut agg = PhaseA::default();
    let mut latencies = Vec::new();

    for probe in PROBES {
        for run in 0..args.runs {
            let started = Instant::now();
            let request = GenerateRequest::chat(vec![
                Message::system(system.clone()),
                Message::user(probe.prompt),
            ])
            .with_sampling(SamplingParams {
                temperature: args.temperature,
                max_tokens: 512,
                ..SamplingParams::default()
            });

            let text = match engine.generate(request).await {
                Ok(r) => r
                    .choices
                    .first()
                    .map(|c| c.text.clone())
                    .unwrap_or_default(),
                Err(e) => {
                    eprintln!("  probe {} run {run}: request failed: {e}", probe.id);
                    continue;
                },
            };
            let latency = started.elapsed().as_millis();

            let tags = text.matches("<tool_call>").count();
            let parsed = detector.detect(&text);
            let malformed = tags.saturating_sub(parsed.len());

            let right_tool = parsed.iter().any(|c| c.name == probe.expect_tool);
            let schema_valid = parsed.iter().any(|c| {
                c.name == probe.expect_tool
                    && probe
                        .required_args
                        .iter()
                        .all(|a| c.arguments.get(*a).is_some_and(|v| !v.is_null()))
            });

            agg.attempts += 1;
            agg.total_tags += tags;
            agg.total_parsed += parsed.len();
            agg.total_malformed += malformed;
            if tags > 0 {
                agg.emitted += 1;
            }
            if right_tool {
                agg.right_tool += 1;
            }
            if schema_valid {
                agg.schema_valid += 1;
            }
            latencies.push(latency);

            agg.results.push(ProbeResult {
                id: format!("{}#{run}", probe.id),
                tags_emitted: tags,
                calls_parsed: parsed.len(),
                malformed,
                right_tool,
                schema_valid,
                latency_ms: latency,
                // Keep evidence only where something went wrong.
                sample: (!schema_valid).then(|| text.chars().take(400).collect()),
            });

            eprint!(".");
        }
    }
    eprintln!();

    agg.mean_latency_ms = if latencies.is_empty() {
        0.0
    } else {
        latencies.iter().sum::<u128>() as f64 / latencies.len() as f64
    };
    agg
}

// ---------------------------------------------------------------------------
// Phase B execution
// ---------------------------------------------------------------------------

/// Refuses Phase B when `repo` contains this harness.
///
/// Presence of the eval binary means the tasks can find their own answers,
/// which would measure reading comprehension of the answer key rather than
/// investigation. See the module docs.
fn assert_uncontaminated(repo: &std::path::Path) {
    let self_path = repo.join("crates/beleth/src/bin/toolcall-eval.rs");
    if self_path.exists() {
        eprintln!(
            "\nREFUSING to run Phase B: {} contains this eval harness.\n\n\
             The task descriptions mention `grammar` and `with_all_tools`, so a model\n\
             searching the tree would find the task text instead of evidence; and the\n\
             clap-collision fix in this branch states that task's answer in a comment.\n\n\
             Point --repo at a pristine checkout instead:\n\n    \
             git worktree add --detach ~/eval-fixture 95097d5\n    \
             toolcall-eval --repo ~/eval-fixture ...\n",
            repo.display()
        );
        std::process::exit(2);
    }
}

async fn run_phase_b(engine: &Arc<OpenAiEngine>, args: &Args) -> PhaseB {
    assert_uncontaminated(&args.repo);
    let mut agg = PhaseB::default();
    let mut correct_turns = Vec::new();

    for task in TASKS {
        for run in 0..args.runs {
            let started = Instant::now();
            let tools = Arc::new(ToolRegistry::with_code_tools());

            let config = ExecutorConfig::new(format!("eval-{}-{run}", task.id))
                .with_system_prompt(
                    "You are a coding assistant investigating a Rust repository. \
                     Use the tools to look at real files rather than guessing. \
                     When you are confident, give a final answer.",
                )
                .with_working_dir(&args.repo)
                .with_autonomy(
                    AutonomyGrant::builder()
                        .allow(ToolPattern::Tool("*".to_string()))
                        .build(),
                )
                .with_loop_config(LoopConfig {
                    max_iterations: task.max_iterations,
                    max_tool_calls: task.max_iterations * 3,
                    detect_implicit_signals: false,
                    ..LoopConfig::default()
                })
                .with_sampling(SamplingParams {
                    temperature: args.temperature,
                    max_tokens: 1024,
                    ..SamplingParams::default()
                });

            let dyn_engine: Arc<dyn InferenceEngine> =
                Arc::clone(engine) as Arc<dyn InferenceEngine>;
            let executor = LoopExecutor::new(dyn_engine, tools, config);
            let (tx, mut rx) = mpsc::channel::<LoopEvent>(512);

            // Count malformed calls during the run the same way Phase A does:
            // raw tags seen in generated text, minus calls the executor
            // actually dispatched.
            let counter = tokio::spawn(async move {
                let mut tags = 0usize;
                let mut detected = 0usize;
                while let Some(event) = rx.recv().await {
                    match event {
                        LoopEvent::GenerationCompleted { ref content, .. } => {
                            tags += content.matches("<tool_call>").count();
                        },
                        LoopEvent::ToolCallDetected { .. } => detected += 1,
                        _ => {},
                    }
                }
                tags.saturating_sub(detected)
            });

            let summary = executor.run(task.objective, tx).await;
            let malformed = counter.await.unwrap_or(0);
            let wall_ms = started.elapsed().as_millis();

            let result = match summary {
                Ok(s) => {
                    let answer = match &s.termination {
                        TerminationReason::Natural(NaturalTermination::AnswerProvided {
                            answer,
                            ..
                        }) => answer.clone(),
                        _ => s.partial_answer.clone().unwrap_or_default(),
                    };
                    let completed = matches!(
                        s.termination,
                        TerminationReason::Natural(NaturalTermination::AnswerProvided { .. })
                    );
                    let lower = answer.to_lowercase();
                    let correct = completed
                        && task
                            .expect_all
                            .iter()
                            .all(|m| lower.contains(&m.to_lowercase()))
                        && !task
                            .expect_none
                            .iter()
                            .any(|m| lower.contains(&m.to_lowercase()));

                    TaskResult {
                        id: format!("{}#{run}", task.id),
                        completed,
                        correct,
                        turns: s.iterations_completed,
                        tool_calls: s.tool_calls_made,
                        malformed_tool_calls: malformed,
                        wall_ms,
                        termination: format!("{:?}", s.termination),
                        answer,
                    }
                },
                Err(e) => TaskResult {
                    id: format!("{}#{run}", task.id),
                    completed: false,
                    correct: false,
                    turns: 0,
                    tool_calls: 0,
                    malformed_tool_calls: malformed,
                    wall_ms,
                    termination: format!("Error: {e}"),
                    answer: String::new(),
                },
            };

            eprintln!(
                "  {:28} {:>3} turns  {:>2} calls  {:>2} malformed  {}",
                result.id,
                result.turns,
                result.tool_calls,
                result.malformed_tool_calls,
                if result.correct {
                    "CORRECT"
                } else if result.completed {
                    "wrong"
                } else {
                    "INCOMPLETE"
                }
            );

            agg.attempts += 1;
            if result.completed {
                agg.completed += 1;
            }
            if result.correct {
                agg.correct += 1;
                correct_turns.push(result.turns);
            }
            agg.total_malformed += result.malformed_tool_calls;
            agg.results.push(result);
        }
    }

    if !correct_turns.is_empty() {
        agg.mean_turns_correct =
            correct_turns.iter().map(|t| *t as f64).sum::<f64>() / correct_turns.len() as f64;
        correct_turns.sort_unstable();
        agg.median_turns_correct = correct_turns[correct_turns.len() / 2] as f64;
    }
    agg
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

fn ratio(n: usize, d: usize) -> f64 {
    if d == 0 {
        0.0
    } else {
        n as f64 / d as f64
    }
}

fn print_phase_a(a: &PhaseA) {
    println!("\n=== Phase A — tool-call fidelity ===\n");
    println!("  attempts              {}", a.attempts);
    println!(
        "  emission rate         {:.1}%   ({}/{} attempts emitted a tool call)",
        a.emission_rate() * 100.0,
        a.emitted,
        a.attempts
    );
    println!(
        "  MALFORMED RATE        {:.2}%   ({}/{} emitted calls failed to parse)",
        a.malformed_rate() * 100.0,
        a.total_malformed,
        a.total_tags
    );
    println!(
        "  right tool            {:.1}%   ({}/{})",
        ratio(a.right_tool, a.attempts) * 100.0,
        a.right_tool,
        a.attempts
    );
    println!(
        "  schema-valid          {:.1}%   ({}/{})",
        a.schema_valid_rate() * 100.0,
        a.schema_valid,
        a.attempts
    );
    println!("  mean latency          {:.0} ms", a.mean_latency_ms);

    println!("\n  compounding over a multi-call task (assumes independence):");
    for n in [10u32, 20, 40] {
        println!(
            "    P(>=1 malformed in {n:>2} calls)   {:.1}%",
            a.failure_over_task(n) * 100.0
        );
    }

    let failures: Vec<&ProbeResult> = a.results.iter().filter(|r| !r.schema_valid).collect();
    if !failures.is_empty() {
        println!("\n  failures ({}):", failures.len());
        for f in failures.iter().take(10) {
            println!(
                "    {:24} tags={} parsed={} malformed={}",
                f.id, f.tags_emitted, f.calls_parsed, f.malformed
            );
        }
    }
}

fn print_phase_b(b: &PhaseB) {
    println!("\n=== Phase B — agentic tasks (known-correct answers) ===\n");
    println!(
        "  completed             {:.1}%   ({}/{})",
        ratio(b.completed, b.attempts) * 100.0,
        b.completed,
        b.attempts
    );
    println!(
        "  correct               {:.1}%   ({}/{})",
        ratio(b.correct, b.attempts) * 100.0,
        b.correct,
        b.attempts
    );
    println!(
        "  turns-to-completion   mean {:.1}, median {:.0}   (correct runs only)",
        b.mean_turns_correct, b.median_turns_correct
    );
    println!("  malformed tool calls  {}", b.total_malformed);

    // Per-task rollup: which tasks a model can and cannot do is more useful
    // than the aggregate when deciding whether it is usable.
    let mut by_task: BTreeMap<&str, (usize, usize, u32)> = BTreeMap::new();
    for r in &b.results {
        let base = r.id.split('#').next().unwrap_or(&r.id);
        let e = by_task.entry(base).or_insert((0, 0, 0));
        e.0 += 1;
        if r.correct {
            e.1 += 1;
            e.2 += r.turns;
        }
    }
    println!("\n  per task:");
    for (task, (n, ok, turns)) in by_task {
        let mean = if ok > 0 {
            format!("{:.1}", turns as f64 / ok as f64)
        } else {
            "-".to_string()
        };
        println!("    {task:28} {ok}/{n} correct   mean turns {mean}");
    }
}
