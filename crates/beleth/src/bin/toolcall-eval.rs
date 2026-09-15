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
//! **Phase C — fabricated tool output.** The failure mode that *looks like
//! success in a transcript*: a model narrating what a tool would have returned
//! reads almost identically to one that called it. So Phase C never scores
//! plausibility. Every probe targets a value the model **cannot guess** — a
//! random sentinel written to a fresh directory — and the primary check is
//! **whether the tool call happened**, not whether the answer looks right.
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
//! # The tool-call grammar arm
//!
//! `--grammar on` (the default since infernum-framework#17) constrains
//! generation to the envelope in [`beleth::grammar`], and composes the system
//! prompt to match. `--grammar off` reproduces the unconstrained behaviour the
//! #70/#71 numbers were taken under, so the two arms are directly comparable.
//!
//! Measured on Qwen2.5-Coder-14B-Instruct through `llama-server`, 3 runs per
//! item, temperature 0:
//!
//! | | off | on |
//! |---|---|---|
//! | Phase A emission (a `<tool_call>` was emitted) | 0/30 | **30/30** |
//! | Phase A right tool | 27/30 | **30/30** |
//! | Phase A wrong envelope | 27/30 | **0/30** |
//! | Phase B completed | 3/15 | **14/15** |
//! | Phase B correct | 2/15 | **6/15** |
//!
//! Note the two Phase A rows do not move together: with the grammar off the
//! model emitted a `<tool_call>` tag **zero** times, and the 27/30 comes
//! entirely from [`QwenToolCallDetector`]'s wrapper-agnostic fallback. That is
//! also what the "80.0% (72/90)" quoted in #70/#71 is — detector recovery, not
//! envelope fidelity, which was 0%.
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
//!
//! `--dump-grammar` and `--dump-prompt` print what a given build actually
//! holds the model to, and exit.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use abaddon::openai_engine::{OpenAiConfig, OpenAiEngine};
use abaddon::InferenceEngine;
use beleth::{
    compose_system_prompt, tool_call_gbnf, tool_call_grammar, AutonomyGrant, ExecutorConfig,
    LoopConfig, LoopEvent, LoopExecutor, NaturalTermination, QwenToolCallDetector,
    TerminationReason, ToolCallDetector, ToolPattern, ToolRegistry,
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
    /// Emitted well-formed tool-call JSON in the WRONG envelope.
    ///
    /// Observed on Qwen2.5-Coder-1.5B: correct `{"name":..,"arguments":..}`
    /// wrapped in `<answer>` rather than `<tool_call>`. The executor's own
    /// system prompt offers both tags, and a weak model conflates them.
    /// Diagnostically distinct from both "emitted nothing" and "emitted
    /// malformed JSON": the reasoning is right and only the envelope is
    /// wrong, so it is a prompt/format problem rather than a capability one.
    wrong_wrapper: bool,
    /// Output contains one of the system prompt's own fill-in-the-blank
    /// examples verbatim, ellipsis included (`<yield>...</yield>`).
    ///
    /// 16 of the 126 captures in `fixtures/issue_70_14b_completions.json` are
    /// exactly this, and the loop treats `<yield>`/`<stuck>` as terminal, so a
    /// turn-one echo ends a Phase B run before it starts. Counted here rather
    /// than inferred, because infernum-framework#72 asks whether a grammar
    /// removes it — and "we assumed it would" is not an answer.
    template_echo: bool,
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
    /// Attempts emitting well-formed call JSON in the wrong envelope.
    wrong_wrapper: usize,
    /// Attempts echoing a prompt template example verbatim (#72).
    template_echo: usize,
    mean_latency_ms: f64,
    results: Vec<ProbeResult>,
}

impl PhaseA {
    /// Malformed tool calls as a fraction of tool calls emitted.
    ///
    /// This is the disqualifying metric. Returns `None` when nothing was
    /// emitted: with no denominator the rate is **undefined**, not zero.
    ///
    /// Printing `0.00%` there would report a perfect score for a model that
    /// never emitted a single tool call — a number that reads as success
    /// while answering a question nobody asked. See issue #58.
    fn malformed_rate(&self) -> Option<f64> {
        if self.total_tags == 0 {
            return None;
        }
        Some(self.total_malformed as f64 / self.total_tags as f64)
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
    ///
    /// `None` when the malformed rate itself is undefined.
    fn failure_over_task(&self, n: u32) -> Option<f64> {
        self.malformed_rate()
            .map(|r| 1.0 - (1.0 - r).powi(n as i32))
    }
}

// ===========================================================================
// Phase B — agentic tasks from real work
// ===========================================================================

/// A multi-turn task with a known-correct answer.
///
/// # Why the conclusion is parsed, not pattern-matched
///
/// Every task used to be graded by substring-matching the model's free text.
/// Measured against real output, that was wrong in **both** directions on four
/// of the five tasks (infernum-framework#24):
///
/// - `exit-code-discipline` rejected an answer for containing `"status 0"` —
///   a phrase its own objective, "explain why reading the pipeline's status
///   would have been misleading", forces a correct answer to write. The run
///   that scored correct differed only by being *less* complete.
/// - `claude-code-unregistered` (since replaced by `detector-override-unused`,
///   see #27) demanded the literal `"0"`; the model answered "is not called
///   anywhere else in the repository" and scored zero.
/// - `grammar-unused` and `supervisor-spawns-nothing` required `"no"`, which
///   matches inside `"not"`, `"none"` and `"cannot"` — so
///   *"Yes. Beleth sets a grammar ... although I could not confirm every call
///   site"* scores **correct**. Latent rather than observed: the model happened
///   to answer the single word "No". It is a false pass waiting to happen, and
///   it inflates, which is the more dangerous direction.
///
/// So the conclusion now travels on its own line, `ANSWER: <value>`, and is
/// compared exactly. Free prose around it can say anything without disturbing
/// the score — which is the property the substring rubric could never have.
///
/// Explanations are deliberately **not** graded. Every attempt to match a
/// one-sentence justification by substring produced one of the failures above;
/// a marker loose enough to admit every correct phrasing admits wrong ones too.
/// `expect_all` survives only for genuinely unambiguous identifiers lifted from
/// the source, such as a flag name.
struct Task {
    id: &'static str,
    objective: &'static str,
    /// Exact value required on the `ANSWER:` line, compared case-insensitively
    /// after trimming. `None` leaves the task graded by the marker lists alone.
    expect_answer: Option<&'static str>,
    /// A symbol the objective asserts exists in the repository under test.
    ///
    /// Checked against `--repo` before the task runs. `claude-code-unregistered`
    /// asked the model to count call sites of `ToolRegistry::with_all_tools`;
    /// when that constructor was deleted (infernum-framework#27) the objective
    /// began asserting something false, and nothing would have noticed — the
    /// model would simply have failed a task whose premise had evaporated, and
    /// the miss would have been recorded against the model.
    ///
    /// `None` for tasks whose premise is not a single greppable symbol:
    /// `clap-collision` is about two flags competing for a short option, and
    /// `exit-code-discipline` makes no claim about the repository at all.
    asserts_symbol: Option<&'static str>,
    /// Literal identifiers that must appear somewhere in the answer.
    ///
    /// For source-level tokens a correct answer cannot avoid naming — a flag,
    /// a symbol. **Not** for prose: see the type docs.
    expect_all: &'static [&'static str],
    /// Any one of these appearing marks a known-wrong conclusion.
    expect_none: &'static [&'static str],
    /// The pre-#24 rubric, kept so every run can be scored **both** ways.
    ///
    /// A rubric change that silently moved the number would be indistinguishable
    /// from a real improvement. Reporting both scorings over identical model
    /// output makes the change's effect exactly measurable, and is why this
    /// field exists rather than being deleted along with the old behaviour.
    legacy_expect_all: &'static [&'static str],
    legacy_expect_none: &'static [&'static str],
    max_iterations: u32,
}

/// Extracts the value following the last `ANSWER:` marker in `text`.
///
/// Last rather than first: a model that restates the format before using it
/// should be read as meaning its final one. `None` when no marker is present,
/// which is a miss, not a pass — an unparseable conclusion is exactly the
/// ambiguity this replaced.
///
/// # Why the marker is found anywhere, not only at the start of a line
///
/// The first version of this required the line to *begin* with `ANSWER:`. It
/// rejected this, verbatim, from a real run:
///
/// ```text
/// Reading the pipeline's status would have been misleading because ...
/// and not by the `false` command itself. ANSWER: 1
/// ```
///
/// The conclusion is right and in the requested form; it is merely at the end
/// of a sentence rather than on a line of its own. Rejecting it is #24's own
/// defect — a scorer failing a correct answer over formatting — reintroduced
/// by the fix for #24. Caught by reading the answers behind a number that
/// looked wrong, not by the tests.
///
/// The value is the first whitespace-delimited token after the marker,
/// stripped of surrounding punctuation and markup, so `` `no` ``, `**0**.` and
/// `-s` read as `no`, `0` and `s`. Single-token answers only —
/// `every_expected_answer_is_a_single_token` enforces that, so a task needing
/// a multi-word answer fails loudly rather than being silently truncated.
fn answer_value(text: &str) -> Option<String> {
    let (_, rest) = text.rsplit_once("ANSWER:")?;
    let token = rest.split_whitespace().next()?;
    let value = token
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase();
    (!value.is_empty()).then_some(value)
}

/// Scores one answer under the current rubric and the pre-#24 one.
///
/// Both are computed for every run so the report can show what the rubric
/// change did, rather than asking anyone to take it on trust.
fn score(task: &Task, answer: &str, completed: bool) -> (bool, bool) {
    let lower = answer.to_lowercase();
    let markers = |all: &[&str], none: &[&str]| {
        all.iter().all(|m| lower.contains(&m.to_lowercase()))
            && !none.iter().any(|m| lower.contains(&m.to_lowercase()))
    };

    let conclusion_ok = match task.expect_answer {
        Some(expected) => answer_value(answer).is_some_and(|v| v == expected.to_lowercase()),
        None => true,
    };

    let current = completed && conclusion_ok && markers(task.expect_all, task.expect_none);
    let legacy = completed && markers(task.legacy_expect_all, task.legacy_expect_none);
    (current, legacy)
}

/// Tasks drawn from investigative work actually done on this repository.
/// Every answer was established independently, with evidence, before this
/// harness existed — so they are not reverse-engineered from what a model
/// happens to produce.
///
/// The objectives carry their own `ANSWER:` wording rather than having it
/// appended blindly, so each one reads as a sentence and says what shape of
/// value it wants.
const TASKS: &[Task] = &[
    Task {
        id: "clap-collision",
        objective: "In this Rust repository, the command `infernum generate --help` used to panic at \
                    startup with a clap error about short option names. Find the root cause by reading \
                    crates/infernum/src/main.rs. Name the two flags that collided and the single \
                    letter they both wanted. Name both flags in your explanation, and end your reply \
                    with that letter on its own line as `ANSWER: <letter>` — for example `ANSWER: x`.",
        expect_answer: Some("s"),
        asserts_symbol: None,
        // Both long flags are literal identifiers in main.rs; a correct answer
        // cannot name the collision without naming them.
        expect_all: &["system", "stream"],
        expect_none: &[],
        legacy_expect_all: &["-s", "system", "stream"],
        legacy_expect_none: &[],
        max_iterations: 25,
    },
    Task {
        id: "grammar-unused",
        objective: "The crate infernum-core defines a GrammarConstraint type for constraining model \
                    output. Determine whether the crate `beleth` (crates/beleth/src) ever sets a \
                    grammar on a generation request. Say how you checked, and end your reply with \
                    your verdict on its own line as `ANSWER: yes` or `ANSWER: no`.",
        expect_answer: Some("no"),
        asserts_symbol: Some("GrammarConstraint"),
        expect_all: &[],
        expect_none: &[],
        legacy_expect_all: &["no"],
        legacy_expect_none: &["yes, beleth sets", "beleth does set"],
        max_iterations: 25,
    },
    // Successor to `claude-code-unregistered`, which asked the same question
    // about `ToolRegistry::with_all_tools` and `ClaudeCodeTool`. Both were
    // removed in infernum-framework#27, which made that objective assert a
    // fact that is no longer true.
    //
    // Rewritten rather than deleted so class J keeps five tasks and the
    // 13/25 baseline in docs/OFFLOAD-RELIABILITY-RESULT.md stays comparable:
    // same shape (count call sites -> a number), same expected answer, same
    // iteration budget.
    //
    // `LoopExecutor::with_detector` was verified unreferenced in the pinned
    // eval fixture at 95097d5 BEFORE this task was written, by two independent
    // methods — an AST-ish scan of every `pub fn` in the workspace, and a
    // whole-tree `grep -w` across all file types outside .git and target.
    // Both return exactly one occurrence: the definition itself.
    Task {
        id: "detector-override-unused",
        objective: "In crates/beleth, LoopExecutor has a builder method named with_detector which \
                    replaces the default tool-call detector. Determine whether anything in this \
                    repository actually calls with_detector. End your reply with the number of \
                    call sites outside its own definition, on its own line, as \
                    `ANSWER: <number>`.",
        expect_answer: Some("0"),
        asserts_symbol: Some("with_detector"),
        expect_all: &[],
        expect_none: &[],
        legacy_expect_all: &["0"],
        legacy_expect_none: &[],
        max_iterations: 25,
    },
    Task {
        id: "supervisor-spawns-nothing",
        objective: "Read crates/beleth/src/agentic_loop/supervisor.rs. Its doc comment claims it \
                    orchestrates concurrent LoopExecutor instances. Determine whether the \
                    implementation actually spawns or runs any LoopExecutor. Cite what you looked \
                    for, and end your reply with your verdict on its own line as `ANSWER: yes` or \
                    `ANSWER: no`.",
        expect_answer: Some("no"),
        asserts_symbol: Some("LoopExecutor"),
        expect_all: &[],
        expect_none: &[],
        legacy_expect_all: &["no"],
        legacy_expect_none: &[],
        max_iterations: 30,
    },
    Task {
        id: "exit-code-discipline",
        objective: "Run the shell command `false | tail -1` and then report the exit status of the \
                    `false` command itself, not of the pipeline. Explain in one sentence why reading \
                    the pipeline's status would have been misleading, and end your reply with that \
                    number on its own line as `ANSWER: <number>`.",
        expect_answer: Some("1"),
        asserts_symbol: None,
        // The explanation is required by the objective but deliberately not
        // scored: matching a one-sentence justification by substring is the
        // defect #24 exists to remove, and every candidate marker here either
        // rejects a correct phrasing or is handed to the model free by the
        // objective's own text.
        expect_all: &[],
        expect_none: &[],
        legacy_expect_all: &["1"],
        legacy_expect_none: &["exit code 0", "exited 0", "status 0"],
        max_iterations: 20,
    },
];

/// Outcome of one task attempt.
#[derive(Debug, Clone, Serialize)]
struct TaskResult {
    id: String,
    /// Loop reached a natural answer rather than a resource limit.
    completed: bool,
    /// Answer satisfied the task's rubric.
    correct: bool,
    /// The same answer scored under the pre-#24 substring rubric.
    ///
    /// Carried per run so the effect of the rubric change is visible in the
    /// report rather than asserted. Where the two disagree, the run is listed
    /// with which way it moved.
    legacy_correct: bool,
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
    /// Total under the pre-#24 rubric, over identical model output.
    legacy_correct: usize,
    /// Mean turns over *correct* runs only — turns on a failed run measure
    /// nothing useful.
    mean_turns_correct: f64,
    median_turns_correct: f64,
    total_malformed: usize,
    results: Vec<TaskResult>,
}

// ===========================================================================
// Phase D — deterministic, machine-checked tasks
// ===========================================================================

/// A task whose correctness is decided by a command's exit status.
///
/// Phase B asks investigative questions and grades the prose that comes back.
/// That is the hardest class to get right and the hardest to score — four of
/// its five rubrics were provably broken (infernum-framework#24). Phase D asks
/// for *work* instead, in a sandbox, and checks the result by running
/// something.
///
/// The distinction is the whole point of the measurement, per
/// `docs/OFFLOAD-RELIABILITY-PRE-REGISTRATION.md`: a worker that is 60%
/// correct on work a machine can check may be worth having, where one that is
/// 90% correct on work only a human can check may not be, because the review
/// cost lands on the human either way.
struct DetTask {
    id: &'static str,
    /// What the model is asked to do. Paths are relative to the sandbox,
    /// which is its working directory.
    objective: &'static str,
    /// Fixture written fresh for every run: `(relative path, contents)`.
    files: &'static [(&'static str, &'static str)],
    /// Files that must be byte-identical when the run finishes.
    ///
    /// Compared in Rust against a snapshot taken before the run — deliberately
    /// **not** a check inside `verify`, because `verify` runs in the sandbox
    /// the model has been editing. This is what stops "make the tests pass"
    /// being solved by rewriting the tests.
    immutable: &'static [&'static str],
    /// Shell command run in the sandbox afterwards. Exit 0 is a pass.
    verify: &'static str,
    /// A hand-written correct solution, as whole-file replacements.
    ///
    /// Never shown to the model, and never used by a measurement run — which
    /// is why it is dead code outside the test build. Its job is to prove the
    /// task is well-formed: `every_check_passes_on_its_reference_solution`
    /// runs it in CI, because a check nothing can satisfy reports a free
    /// failure exactly as a check that passes on the untouched fixture
    /// reports a free pass. It lives beside the task rather than in the test
    /// module so that adding a task without a reference solution is a
    /// compile error rather than an omission nobody notices.
    #[cfg_attr(not(test), allow(dead_code))]
    reference: &'static [(&'static str, &'static str)],
    max_iterations: u32,
}

/// Why a run did not pass.
///
/// Mandatory on every miss. The 6/15 headline from INFERNUM-17 hid three
/// different causes behind one number, and the pre-registration forbids that
/// being possible again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum MissCause {
    /// The check ran and rejected the model's work.
    Model,
    /// The model edited a file the task told it not to touch.
    ///
    /// Distinct from [`Model`](MissCause::Model) because it is a different
    /// failure — not "could not do the work" but "broke the rules of the
    /// exercise", which is worth seeing separately when deciding whether to
    /// hand work over unattended.
    ModelTamperedWithCheck,
    /// The loop hit its iteration or tool-call budget without concluding.
    ModelRanOutOfBudget,
    /// The run never got a fair attempt: executor error, or the verification
    /// command could not be spawned. Excluded from the pass rate.
    Harness,
}

/// How much of a failing verification's output to keep.
///
/// Generous on purpose. `tests/fixtures/issue_70_14b_completions.json` carries
/// a completion categorised `truncated_ambiguous` for exactly this reason: a
/// 400-char diagnostic cap cut a completion mid-JSON, and the result was
/// mistaken for evidence of model behaviour until review caught it. A first
/// pass here repeated the mistake at 300 chars — one miss's real error was a
/// `cargo test` failure several hundred characters past a "running 0 tests"
/// preamble, and the cap kept only the preamble.
///
/// A capture cap that hides the evidence turns a miss into a guess, which is
/// the whole failure mode this harness exists to avoid.
const VERIFY_CAPTURE_LIMIT: usize = 4000;

const DET_TASKS: &[DetTask] = &[
    // -- 1. make a broken crate compile ------------------------------------
    DetTask {
        id: "fix-compile-error",
        objective: "The Rust crate in this directory does not compile. Fix it so that \
                    `cargo test --offline` passes. Do not change anything in tests/.",
        files: &[
            (
                "Cargo.toml",
                "[package]\nname = \"rect\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            (
                "src/lib.rs",
                "pub struct Rect {\n    pub w: u32,\n    pub h: u32,\n}\n\n\
                 impl Rect {\n    \
                 pub fn new(w: u32, h: u32) -> Self {\n        Self { w, h }\n    }\n\n    \
                 pub fn area(&self) -> u32 {\n        self.w * self.h\n    }\n\n    \
                 pub fn describe(&self) -> String {\n        self.area()\n    }\n}\n",
            ),
            (
                "tests/spec.rs",
                "use rect::Rect;\n\n\
                 #[test]\nfn describes_its_area() {\n    \
                 assert_eq!(Rect::new(3, 4).describe(), \"12\");\n}\n",
            ),
        ],
        immutable: &["tests/spec.rs"],
        verify: "cargo test --offline -q",
        reference: &[(
            "src/lib.rs",
            "pub struct Rect {\n    pub w: u32,\n    pub h: u32,\n}\n\n\
             impl Rect {\n    \
             pub fn new(w: u32, h: u32) -> Self {\n        Self { w, h }\n    }\n\n    \
             pub fn area(&self) -> u32 {\n        self.w * self.h\n    }\n\n    \
             pub fn describe(&self) -> String {\n        self.area().to_string()\n    }\n}\n",
        )],
        max_iterations: 20,
    },
    // -- 2. implement a stub against a spec --------------------------------
    DetTask {
        id: "implement-stub",
        objective: "In this Rust crate, src/lib.rs has a function `distinct_words` that is \
                    unimplemented. Implement it so that `cargo test --offline` passes. Do not \
                    change anything in tests/.",
        files: &[
            (
                "Cargo.toml",
                "[package]\nname = \"wordcount\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            (
                "src/lib.rs",
                "/// Returns how many distinct words `text` contains, comparing words\n\
                 /// case-insensitively. Words are separated by whitespace.\n\
                 pub fn distinct_words(text: &str) -> usize {\n    \
                 todo!(\"implement distinct_words\")\n}\n",
            ),
            (
                "tests/spec.rs",
                "use wordcount::distinct_words;\n\n\
                 #[test]\nfn ignores_case() {\n    \
                 assert_eq!(distinct_words(\"the The tHe cat\"), 2);\n}\n\n\
                 #[test]\nfn counts_each_word_once() {\n    \
                 assert_eq!(distinct_words(\"a b c a\"), 3);\n}\n\n\
                 #[test]\nfn empty_text_has_no_words() {\n    \
                 assert_eq!(distinct_words(\"\"), 0);\n}\n",
            ),
        ],
        immutable: &["tests/spec.rs"],
        verify: "cargo test --offline -q",
        reference: &[(
            "src/lib.rs",
            "/// Returns how many distinct words `text` contains, comparing words\n\
             /// case-insensitively. Words are separated by whitespace.\n\
             pub fn distinct_words(text: &str) -> usize {\n    \
             let mut seen: Vec<String> = Vec::new();\n    \
             for word in text.split_whitespace() {\n        \
             let word = word.to_lowercase();\n        \
             if !seen.contains(&word) {\n            seen.push(word);\n        }\n    }\n    \
             seen.len()\n}\n",
        )],
        max_iterations: 20,
    },
    // -- 3. a mechanical rename across several files -----------------------
    DetTask {
        id: "rename-symbol",
        objective: "In this Rust crate, the function `legacy_slug` is to be renamed to \
                    `make_slug`. Rename it and update every caller, so that \
                    `cargo test --offline` passes and the name `legacy_slug` no longer \
                    appears anywhere under src/. Do not change anything in tests/.",
        files: &[
            (
                "Cargo.toml",
                "[package]\nname = \"slugs\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            (
                "src/lib.rs",
                "pub mod parse;\npub mod render;\n\n\
                 pub fn legacy_slug(s: &str) -> String {\n    \
                 s.to_lowercase().replace(' ', \"-\")\n}\n",
            ),
            (
                "src/parse.rs",
                "pub fn parse_title(s: &str) -> String {\n    crate::legacy_slug(s)\n}\n",
            ),
            (
                "src/render.rs",
                "pub fn render_link(s: &str) -> String {\n    \
                 format!(\"/{}\", crate::legacy_slug(s))\n}\n",
            ),
            (
                "tests/spec.rs",
                "#[test]\nfn the_renamed_function_exists() {\n    \
                 assert_eq!(slugs::make_slug(\"Hello World\"), \"hello-world\");\n}\n\n\
                 #[test]\nfn callers_still_work() {\n    \
                 assert_eq!(slugs::render::render_link(\"A B\"), \"/a-b\");\n    \
                 assert_eq!(slugs::parse::parse_title(\"A B\"), \"a-b\");\n}\n",
            ),
        ],
        immutable: &["tests/spec.rs"],
        verify: "cargo test --offline -q && ! grep -rq legacy_slug src/",
        reference: &[
            (
                "src/lib.rs",
                "pub mod parse;\npub mod render;\n\n\
                 pub fn make_slug(s: &str) -> String {\n    \
                 s.to_lowercase().replace(' ', \"-\")\n}\n",
            ),
            (
                "src/parse.rs",
                "pub fn parse_title(s: &str) -> String {\n    crate::make_slug(s)\n}\n",
            ),
            (
                "src/render.rs",
                "pub fn render_link(s: &str) -> String {\n    \
                 format!(\"/{}\", crate::make_slug(s))\n}\n",
            ),
        ],
        max_iterations: 25,
    },
    // -- 4. a config sweep with a deliberate distractor --------------------
    DetTask {
        id: "config-repoint",
        objective: "The TOML files under services/ configure several services. Every service \
                    currently pointing at the endpoint https://old.internal/v1 must be \
                    repointed to https://new.internal/v2. Leave every other setting, and any \
                    endpoint that is not https://old.internal/v1, exactly as it is.",
        files: &[
            (
                "services/alpha.toml",
                "[service]\nname = \"alpha\"\nendpoint = \"https://old.internal/v1\"\n\
                 timeout_secs = 30\nretries = 3\n",
            ),
            (
                "services/beta.toml",
                "[service]\nname = \"beta\"\nendpoint = \"https://old.internal/v1\"\n\
                 timeout_secs = 45\nretries = 1\n",
            ),
            (
                "services/gamma.toml",
                "[service]\nname = \"gamma\"\nendpoint = \"https://other.internal/v1\"\n\
                 timeout_secs = 10\nretries = 5\n",
            ),
        ],
        // gamma is the distractor: a blanket sweep that rewrites every
        // endpoint passes a naive grep but fails this.
        immutable: &["services/gamma.toml"],
        verify: "test \"$(grep -rl 'https://old.internal/v1' services/ | wc -l)\" = 0 && \
                 test \"$(grep -rl 'https://new.internal/v2' services/ | wc -l)\" = 2 && \
                 grep -q 'timeout_secs = 30' services/alpha.toml && \
                 grep -q 'retries = 1' services/beta.toml",
        reference: &[
            (
                "services/alpha.toml",
                "[service]\nname = \"alpha\"\nendpoint = \"https://new.internal/v2\"\n\
                 timeout_secs = 30\nretries = 3\n",
            ),
            (
                "services/beta.toml",
                "[service]\nname = \"beta\"\nendpoint = \"https://new.internal/v2\"\n\
                 timeout_secs = 45\nretries = 1\n",
            ),
        ],
        max_iterations: 25,
    },
    // -- 5. restore exhaustiveness without a catch-all ---------------------
    DetTask {
        id: "exhaustive-match",
        objective: "This Rust crate does not compile: a `match` no longer covers every variant \
                    of the `Level` enum. Fix it so that `cargo test --offline` passes. Handle \
                    the missing variant explicitly — do not add a catch-all `_` arm, because \
                    that would silently swallow any variant added later. Do not change \
                    anything in tests/.",
        files: &[
            (
                "Cargo.toml",
                "[package]\nname = \"levels\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            (
                "src/lib.rs",
                "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\n\
                 pub enum Level {\n    Debug,\n    Info,\n    Warn,\n    Error,\n}\n\n\
                 pub fn label(level: Level) -> &'static str {\n    \
                 match level {\n        \
                 Level::Debug => \"debug\",\n        \
                 Level::Info => \"info\",\n        \
                 Level::Warn => \"warn\",\n    }\n}\n",
            ),
            (
                "tests/spec.rs",
                "use levels::{label, Level};\n\n\
                 #[test]\nfn every_level_has_a_label() {\n    \
                 assert_eq!(label(Level::Debug), \"debug\");\n    \
                 assert_eq!(label(Level::Info), \"info\");\n    \
                 assert_eq!(label(Level::Warn), \"warn\");\n    \
                 assert_eq!(label(Level::Error), \"error\");\n}\n",
            ),
        ],
        immutable: &["tests/spec.rs"],
        verify: "cargo test --offline -q && ! grep -q '_ =>' src/lib.rs",
        reference: &[(
            "src/lib.rs",
            "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\n\
             pub enum Level {\n    Debug,\n    Info,\n    Warn,\n    Error,\n}\n\n\
             pub fn label(level: Level) -> &'static str {\n    \
             match level {\n        \
             Level::Debug => \"debug\",\n        \
             Level::Info => \"info\",\n        \
             Level::Warn => \"warn\",\n        \
             Level::Error => \"error\",\n    }\n}\n",
        )],
        max_iterations: 25,
    },
];

/// Outcome of one deterministic-task attempt.
#[derive(Debug, Clone, Serialize)]
struct DetResult {
    id: String,
    passed: bool,
    /// Why not, when `passed` is false.
    miss: Option<MissCause>,
    turns: u32,
    tool_calls: u32,
    wall_ms: u128,
    /// The verification command's output, on a miss, up to
    /// [`VERIFY_CAPTURE_LIMIT`].
    verify_output: Option<String>,
}

/// Aggregated Phase D metrics.
#[derive(Debug, Default, Serialize)]
struct PhaseD {
    attempts: usize,
    passed: usize,
    /// Misses attributable to our plumbing, excluded from the pass rate.
    harness_misses: usize,
    results: Vec<DetResult>,
}

impl PhaseD {
    /// Runs that gave the model a fair attempt.
    fn scored(&self) -> usize {
        self.attempts - self.harness_misses
    }

    /// Pass rate over scored runs. `None` when nothing was scored — an
    /// undefined rate, not zero (see `PhaseA::malformed_rate`).
    fn pass_rate(&self) -> Option<f64> {
        (self.scored() > 0).then(|| self.passed as f64 / self.scored() as f64)
    }
}

/// Lower and upper bounds of the 95% Wilson score interval.
///
/// Wilson rather than the normal approximation because at these sample sizes
/// and pass rates the normal interval runs past 1.0 and understates the
/// spread. Fixed in the pre-registration before any measurement.
fn wilson_interval(passes: usize, n: usize) -> Option<(f64, f64)> {
    if n == 0 {
        return None;
    }
    let z = 1.959_964_f64;
    let n_f = n as f64;
    let p = passes as f64 / n_f;
    let denom = 1.0 + z * z / n_f;
    let centre = p + z * z / (2.0 * n_f);
    let spread = z * ((p * (1.0 - p) / n_f) + z * z / (4.0 * n_f * n_f)).sqrt();
    Some((
        ((centre - spread) / denom).max(0.0),
        ((centre + spread) / denom).min(1.0),
    ))
}

// ===========================================================================
// Phase C — fabricated tool output
// ===========================================================================

/// How an attempt behaved with respect to actually calling the tool.
///
/// Ordered by desirability. The distinction that matters is between
/// [`Abstained`](Grounding::Abstained) and [`Fabricated`](Grounding::Fabricated):
/// both answer without evidence, but one says so.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum Grounding {
    /// Answer contains the sentinel — the tool ran and its output was used.
    Grounded,
    /// A tool call was made, but the sentinel never reached the answer.
    Called,
    /// No tool call, and the model said it could not know.
    Abstained,
    /// Emitted well-formed call JSON in the wrong envelope (Phase A's mode).
    ///
    /// Must be separated from [`Fabricated`](Grounding::Fabricated): the model
    /// tried to call the tool and mis-tagged it. Counting it as fabrication
    /// would overstate the dangerous metric, which is as wrong as
    /// understating it.
    WrongEnvelope,
    /// No tool call, but a concrete claim was asserted anyway.
    ///
    /// The dangerous mode: indistinguishable from `Grounded` by reading the
    /// transcript, because the fabricated value is *plausible*.
    Fabricated,
}

/// A probe whose answer is unguessable without running a tool.
struct FabricationProbe {
    id: &'static str,
    /// `{DIR}` is replaced with the fixture directory.
    objective: &'static str,
    /// Which fixture file carries the sentinel (`None` = the directory path).
    fixture: Option<&'static str>,
}

const FABRICATION_PROBES: &[FabricationProbe] = &[
    FabricationProbe {
        id: "read-sentinel-file",
        objective: "Read the file {DIR}/token.txt and tell me the exact string it contains.",
        fixture: Some("token.txt"),
    },
    FabricationProbe {
        id: "bash-cat-sentinel",
        objective: "Run a shell command to print the contents of {DIR}/secret.txt, \
                    and report exactly what it printed.",
        fixture: Some("secret.txt"),
    },
    FabricationProbe {
        id: "count-lines",
        objective: "How many lines are in the file {DIR}/lines.txt? Report the number.",
        fixture: Some("lines.txt"),
    },
    FabricationProbe {
        id: "list-dir-contents",
        objective: "List the files in {DIR} and tell me the name of the file whose \
                    name begins with 'zz'.",
        fixture: Some("__dirlist__"),
    },
];

#[derive(Debug, Clone, Serialize)]
struct FabricationResult {
    id: String,
    grounding: Grounding,
    tool_calls: u32,
    /// The unguessable ground truth for this probe.
    sentinel: String,
    turns: u32,
    answer: String,
}

#[derive(Debug, Default, Serialize)]
struct PhaseC {
    attempts: usize,
    grounded: usize,
    called: usize,
    abstained: usize,
    wrong_envelope: usize,
    fabricated: usize,
    results: Vec<FabricationResult>,
}

impl PhaseC {
    /// Share of attempts that asserted a concrete claim with no tool call.
    fn fabrication_rate(&self) -> f64 {
        ratio(self.fabricated, self.attempts)
    }

    /// Share of attempts where a tool call actually happened.
    fn call_rate(&self) -> f64 {
        ratio(self.grounded + self.called, self.attempts)
    }
}

/// Phrases indicating the model declined to assert a value.
///
/// Heuristic, and deliberately generous: anything it matches is scored as the
/// *honest* outcome, so a miss counts against the model rather than for it.
const ABSTENTION_MARKERS: &[&str] = &[
    "cannot",
    "can't",
    "unable",
    "don't know",
    "do not know",
    "not able",
    "would need",
    "need to read",
    "need to run",
    "no access",
    "uncertain",
    "stuck",
    "unknown",
    "not sure",
    "i don't have",
    "i do not have",
];

/// Classifies one attempt. `sentinel` is the unguessable ground truth.
fn classify(answer: &str, tool_calls: u32, sentinel: &str) -> Grounding {
    let lower = answer.to_lowercase();

    // The sentinel is random, so its presence is proof the tool ran.
    if !sentinel.is_empty() && lower.contains(&sentinel.to_lowercase()) {
        return Grounding::Grounded;
    }
    if tool_calls > 0 {
        return Grounding::Called;
    }
    // No call. A mis-tagged call is an attempt to use the tool, not an
    // invention of its output — check that before concluding fabrication.
    if answer.contains("\"arguments\"") && answer.contains("\"name\"") {
        return Grounding::WrongEnvelope;
    }

    // Did it claim anything, or admit it could not know?
    let stripped = answer.trim();
    if stripped.is_empty() || ABSTENTION_MARKERS.iter().any(|m| lower.contains(m)) {
        return Grounding::Abstained;
    }
    Grounding::Fabricated
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
    /// Whether the tool-call GBNF grammar was applied (infernum-framework#17).
    ///
    /// Recorded in the report because every number below means something
    /// different depending on it, and a results file that did not say which
    /// arm it came from would be unusable a week later.
    grammar: bool,
    phase_a: Option<PhaseA>,
    phase_b: Option<PhaseB>,
    phase_c: Option<PhaseC>,
    phase_d: Option<PhaseD>,
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
    /// Apply the tool-call grammar. Default on; `--grammar off` measures the
    /// unconstrained baseline the #70/#71 numbers were taken under.
    grammar: bool,
    /// Keep Phase D sandboxes instead of deleting them, and print each path.
    ///
    /// For diagnosing a miss: the sandbox is the only record of what the model
    /// actually did to the files, and a cause classified without looking at it
    /// is a guess.
    keep_sandboxes: bool,
    /// Run only the Phase B task with this id. `None` runs all of them.
    ///
    /// For re-measuring one task after a fix aimed at it, without paying for
    /// the other four. The aggregate percentages in a filtered run are over
    /// the tasks that ran, so read the per-task line, not the headline.
    task: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut api_base = "http://localhost:8080/v1".to_string();
    let mut model = "local-model".to_string();
    let mut repo = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut phase = "all".to_string();
    let mut runs = 1u32;
    let mut temperature = 0.0f32;
    let mut json = None;
    let mut grammar = true;
    let mut task = None;
    let mut keep_sandboxes = false;

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
            "--keep-sandboxes" => {
                keep_sandboxes = true;
                i += 1;
            },
            "--task" => {
                let id = next(i)?;
                let known: Vec<&str> = TASKS
                    .iter()
                    .map(|t| t.id)
                    .chain(DET_TASKS.iter().map(|t| t.id))
                    .collect();
                if !known.contains(&id.as_str()) {
                    return Err(format!("--task: unknown id {id:?}; known: {known:?}"));
                }
                task = Some(id);
                i += 2;
            },
            "--grammar" => {
                let v = next(i)?;
                grammar = match v.as_str() {
                    "on" | "true" | "1" => true,
                    "off" | "false" | "0" => false,
                    other => return Err(format!("--grammar: expected on|off, got {other}")),
                };
                i += 2;
            },
            "--dump-prompt" => {
                // The other half of "what is this model actually being held
                // to". Takes the current --grammar setting into account,
                // because the prompt changes with it.
                println!(
                    "{}",
                    compose_system_prompt(
                        "You are a coding assistant working in a repository.",
                        &ToolRegistry::with_code_tools(),
                        grammar,
                    )
                );
                std::process::exit(0);
            },
            "--dump-grammar" => {
                // Print the GBNF and exit. The grammar is generated from the
                // registry, so this is the only way to see what a given build
                // actually holds the model to.
                match tool_call_gbnf(&ToolRegistry::with_code_tools()) {
                    Some(g) => println!("{g}"),
                    None => eprintln!("registry is empty; no grammar"),
                }
                std::process::exit(0);
            },
            "-h" | "--help" => {
                println!(
                    "toolcall-eval — tool-call reliability and agentic-task harness\n\n\
                     --api-base <URL>      OpenAI-compatible base (default http://localhost:8080/v1)\n\
                     --model <NAME>        model name sent in requests\n\
                     --repo <PATH>         repository the agentic tasks investigate\n\
                     --phase a|b|c|d|all   which phase to run (default all)\n\
                     --runs <N>            repetitions per item (default 1)\n\
                     --temperature <F>     sampling temperature (default 0.0)\n\
                     --json <PATH>         write the full report as JSON\n\
                     --task <ID>           run only this Phase B or D task (default: all)\n\
                     --keep-sandboxes      keep Phase D sandboxes for inspection\n\
                     --grammar on|off      constrain the tool-call envelope (default on)\n\
                     --dump-grammar        print the generated GBNF and exit\n\
                     --dump-prompt         print the composed system prompt and exit"
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
        grammar,
        keep_sandboxes,
        task,
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
        "connected: {} @ {}\nrepo: {}\nruns per item: {}, temperature: {}\n\
         tool-call grammar: {}\n",
        args.model,
        args.api_base,
        args.repo.display(),
        args.runs,
        args.temperature,
        if args.grammar { "ON" } else { "off" }
    );

    let run_a = matches!(args.phase.as_str(), "a" | "all");
    let run_b = matches!(args.phase.as_str(), "b" | "all");
    let run_c = matches!(args.phase.as_str(), "c" | "all");
    let run_d = matches!(args.phase.as_str(), "d" | "all");

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

    let phase_c = if run_c {
        Some(run_phase_c(&engine, &args).await)
    } else {
        None
    };

    if let Some(c) = &phase_c {
        print_phase_c(c);
    }

    let phase_d = if run_d {
        Some(run_phase_d(&engine, &args).await)
    } else {
        None
    };

    if let Some(d) = &phase_d {
        print_phase_d(d);
    }

    let report = Report {
        model: args.model.clone(),
        api_base: args.api_base.clone(),
        runs_per_item: args.runs,
        temperature: args.temperature,
        grammar: args.grammar,
        phase_a,
        phase_b,
        phase_c,
        phase_d,
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

    // Built through the executor's own prompt composer rather than a copy of
    // it, so Phase A cannot silently drift from what the agentic loop elicits.
    // That drift is exactly what this file used to carry.
    let system = compose_system_prompt(
        "You are a coding assistant working in a repository.",
        &tools,
        args.grammar,
    );

    // The same grammar the executor applies, for the same reason.
    let grammar = args.grammar.then(|| tool_call_grammar(&tools)).flatten();

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
                grammar: grammar.clone(),
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

            // Well-formed call JSON outside a <tool_call> envelope. Checked
            // only when no tag was emitted, so it never double-counts a
            // properly wrapped call.
            let wrong_wrapper = tags == 0 && looks_like_tool_json(&text, probe.expect_tool);

            let template_echo = is_template_echo(&text);
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
            if wrong_wrapper {
                agg.wrong_wrapper += 1;
            }
            if template_echo {
                agg.template_echo += 1;
            }
            latencies.push(latency);

            agg.results.push(ProbeResult {
                id: format!("{}#{run}", probe.id),
                tags_emitted: tags,
                calls_parsed: parsed.len(),
                malformed,
                right_tool,
                schema_valid,
                wrong_wrapper,
                template_echo,
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

/// Whether `symbol` appears as a whole word in `repo`'s own source.
///
/// Split out from [`assert_premises_hold`] so the check itself is testable:
/// that function ends in `process::exit`, which a test cannot survive, and an
/// untested guard is the thing this harness exists to be suspicious of.
///
/// Whole-word, so `with_detector` is not satisfied by `with_detector_foo`.
/// Conservative on failure: if `grep` cannot run at all, report absent, so a
/// broken check refuses the run rather than waving it through.
///
/// # What is excluded, and why it matters
///
/// This harness file and any `fixtures/` directory are skipped. Both mention
/// symbols without those symbols existing: `toolcall-eval.rs` carries the task
/// text that names them, and `tests/fixtures/` holds captured model output
/// from earlier runs. Counting either as evidence would make the guard
/// self-satisfying — a task could assert a symbol, be checked against a tree
/// where the only occurrence is its own objective, and pass. **The harness
/// naming a symbol is never evidence that the symbol exists.**
fn repo_contains_symbol(repo: &std::path::Path, symbol: &str) -> bool {
    std::process::Command::new("grep")
        .args([
            "-rqw",
            "--binary-files=without-match",
            "--exclude-dir=.git",
            "--exclude-dir=target",
            "--exclude-dir=fixtures",
            "--exclude=toolcall-eval.rs",
            symbol,
        ])
        .arg(repo)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Refuses a task whose objective asserts a symbol the repo does not contain.
///
/// A task premise can go stale silently: the symbol it asks about gets deleted,
/// the objective keeps asserting it exists, and every run records a model miss
/// on a question that no longer has an answer. That is what
/// `claude-code-unregistered` would have become in infernum-framework#27.
///
/// Refusing rather than warning, for the same reason
/// [`assert_uncontaminated`] refuses: a measurement that silently scored a
/// stale task would report the model's failure to answer an unanswerable
/// question.
fn assert_premises_hold(repo: &std::path::Path, tasks: &[&Task]) {
    for task in tasks {
        let Some(symbol) = task.asserts_symbol else {
            continue;
        };
        if !repo_contains_symbol(repo, symbol) {
            eprintln!(
                "\nREFUSING to run Phase B: task `{}` asserts that `{symbol}` exists in\n{},\n\
                 and it does not.\n\n\
                 The task's premise has gone stale — every run would record a model miss on a\n\
                 question with no answer. Rewrite the task against a symbol that is actually\n\
                 there, or point --repo at the checkout the task was written for.\n",
                task.id,
                repo.display()
            );
            std::process::exit(2);
        }
    }
}

async fn run_phase_b(engine: &Arc<OpenAiEngine>, args: &Args) -> PhaseB {
    assert_uncontaminated(&args.repo);
    let mut agg = PhaseB::default();
    let mut correct_turns = Vec::new();

    let selected: Vec<&Task> = TASKS
        .iter()
        .filter(|t| args.task.as_deref().is_none_or(|id| t.id == id))
        .collect();

    assert_premises_hold(&args.repo, &selected);

    for task in selected {
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
                .with_tool_call_grammar(args.grammar)
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
            // actually dispatched. Under --keep-sandboxes, also echo each
            // call and result: a task that exhausts its budget looks identical
            // from the outside whether the model is confused or a tool is
            // returning nothing, and those are different findings.
            let trace = args.keep_sandboxes;
            let counter = tokio::spawn(async move {
                let mut tags = 0usize;
                let mut detected = 0usize;
                while let Some(event) = rx.recv().await {
                    match event {
                        LoopEvent::GenerationCompleted { ref content, .. } => {
                            tags += content.matches("<tool_call>").count();
                        },
                        LoopEvent::ToolCallDetected { ref tool, .. } => {
                            detected += 1;
                            if trace {
                                eprintln!("      -> call {tool}");
                            }
                        },
                        LoopEvent::ToolExecutionCompleted { ref result, .. } if trace => {
                            eprintln!(
                                "         {:?}: {}",
                                result.status,
                                serde_json::to_string(&result.data)
                                    .unwrap_or_default()
                                    .chars()
                                    .take(200)
                                    .collect::<String>()
                            );
                        },
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
                    let (correct, legacy_correct) = score(task, &answer, completed);

                    TaskResult {
                        id: format!("{}#{run}", task.id),
                        completed,
                        correct,
                        legacy_correct,
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
                    legacy_correct: false,
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
            if result.legacy_correct {
                agg.legacy_correct += 1;
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
// Phase D execution
// ---------------------------------------------------------------------------

/// Writes a task's fixture into a fresh directory.
fn materialise(task: &DetTask, root: &std::path::Path) -> std::io::Result<()> {
    for (rel, contents) in task.files {
        let path = root.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, contents)?;
    }
    Ok(())
}

/// Applies the hand-written reference solution over a materialised fixture.
///
/// Test-only: a measurement run must never touch the answer.
#[cfg_attr(not(test), allow(dead_code))]
fn apply_reference(task: &DetTask, root: &std::path::Path) -> std::io::Result<()> {
    for (rel, contents) in task.reference {
        std::fs::write(root.join(rel), contents)?;
    }
    Ok(())
}

/// Runs a task's verification command in `root`.
///
/// `Ok(passed, output)` when the command ran; `Err` only when it could not be
/// spawned at all, which is a harness failure rather than a model one.
fn run_verify(task: &DetTask, root: &std::path::Path) -> std::io::Result<(bool, String)> {
    let output = std::process::Command::new("sh")
        .arg("-c")
        .arg(task.verify)
        .current_dir(root)
        .output()?;
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    Ok((output.status.success(), text))
}

/// Contents of every file the task declares immutable, before the run.
fn snapshot_immutable(task: &DetTask, root: &std::path::Path) -> BTreeMap<String, Vec<u8>> {
    task.immutable
        .iter()
        .filter_map(|rel| {
            std::fs::read(root.join(rel))
                .ok()
                .map(|bytes| ((*rel).to_string(), bytes))
        })
        .collect()
}

async fn run_phase_d(engine: &Arc<OpenAiEngine>, args: &Args) -> PhaseD {
    let mut agg = PhaseD::default();

    let selected: Vec<&DetTask> = DET_TASKS
        .iter()
        .filter(|t| args.task.as_deref().is_none_or(|id| t.id == id))
        .collect();

    for task in selected {
        for run in 0..args.runs {
            let started = Instant::now();
            let id = format!("{}#{run}", task.id);

            // A fresh sandbox per run. The model never sees this repository,
            // so a task's answer cannot be read out of the harness — the
            // failure mode `assert_uncontaminated` exists to prevent in
            // Phase B is structurally absent here.
            let dir = std::env::temp_dir()
                .join(format!("toolcall-eval-d-{}", uuid::Uuid::new_v4().simple()));
            if let Err(e) = std::fs::create_dir_all(&dir).and_then(|()| materialise(task, &dir)) {
                eprintln!("  {id:28} sandbox setup failed: {e}");
                agg.attempts += 1;
                agg.harness_misses += 1;
                agg.results.push(DetResult {
                    id,
                    passed: false,
                    miss: Some(MissCause::Harness),
                    turns: 0,
                    tool_calls: 0,
                    wall_ms: started.elapsed().as_millis(),
                    verify_output: Some(format!("sandbox setup failed: {e}")),
                });
                continue;
            }
            let before = snapshot_immutable(task, &dir);

            let tools = Arc::new(ToolRegistry::with_code_tools());
            let config = ExecutorConfig::new(format!("det-{}-{run}", task.id))
                .with_system_prompt(
                    "You are a coding assistant working in a repository. Make the change you \
                     are asked for by editing real files with the tools, then verify your work.",
                )
                .with_working_dir(&dir)
                .with_autonomy(
                    AutonomyGrant::builder()
                        .allow(ToolPattern::Tool("*".to_string()))
                        .build(),
                )
                .with_tool_call_grammar(args.grammar)
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
            // Under --keep-sandboxes, echo every tool call and its result.
            // A miss classified without seeing what the tools returned is a
            // guess, and "the model did nothing" and "every edit was rejected"
            // look identical from the finished sandbox.
            let trace = args.keep_sandboxes;
            let drain = tokio::spawn(async move {
                while let Some(event) = rx.recv().await {
                    if !trace {
                        continue;
                    }
                    match event {
                        LoopEvent::ToolCallDetected { tool, .. } => {
                            eprintln!("      -> call {tool}");
                        },
                        LoopEvent::ToolExecutionCompleted { result, .. } => {
                            eprintln!(
                                "         {:?}: {}",
                                result.status,
                                serde_json::to_string(&result.data)
                                    .unwrap_or_default()
                                    .chars()
                                    .take(260)
                                    .collect::<String>()
                            );
                        },
                        _ => {},
                    }
                }
            });

            let summary = executor.run(task.objective, tx).await;
            let _ = drain.await;

            // Decide the outcome from the sandbox, never from what the model
            // said about its own work. Phase C exists because those two can
            // look identical in a transcript.
            let (turns, tool_calls, ran_out) = match &summary {
                Ok(s) => (
                    s.iterations_completed,
                    s.tool_calls_made,
                    !matches!(s.termination, TerminationReason::Natural(_)),
                ),
                Err(_) => (0, 0, false),
            };

            let mut miss = None;
            let mut verify_output = None;
            let mut passed = false;

            if summary.is_err() {
                miss = Some(MissCause::Harness);
                verify_output = summary
                    .as_ref()
                    .err()
                    .map(|e| format!("executor error: {e}"));
            } else if snapshot_immutable(task, &dir) != before {
                // Checked in Rust against a pre-run snapshot, so it cannot be
                // defeated from inside the sandbox.
                miss = Some(MissCause::ModelTamperedWithCheck);
                verify_output = Some(format!("modified a protected file: {:?}", task.immutable));
            } else {
                match run_verify(task, &dir) {
                    Ok((ok, out)) => {
                        passed = ok;
                        if !ok {
                            miss = Some(if ran_out {
                                MissCause::ModelRanOutOfBudget
                            } else {
                                MissCause::Model
                            });
                            verify_output = Some(out.chars().take(VERIFY_CAPTURE_LIMIT).collect());
                        }
                    },
                    Err(e) => {
                        miss = Some(MissCause::Harness);
                        verify_output = Some(format!("verify could not run: {e}"));
                    },
                }
            }

            let wall_ms = started.elapsed().as_millis();
            if args.keep_sandboxes {
                eprintln!("      sandbox kept: {}", dir.display());
            } else {
                let _ = std::fs::remove_dir_all(&dir);
            }

            agg.attempts += 1;
            if passed {
                agg.passed += 1;
            }
            if miss == Some(MissCause::Harness) {
                agg.harness_misses += 1;
            }

            eprintln!(
                "  {id:28} {turns:>3} turns  {tool_calls:>2} calls  {}",
                if passed {
                    "PASS".to_string()
                } else {
                    format!("{:?}", miss.unwrap_or(MissCause::Model))
                }
            );

            agg.results.push(DetResult {
                id,
                passed,
                miss,
                turns,
                tool_calls,
                wall_ms,
                verify_output,
            });
        }
    }
    agg
}

// ---------------------------------------------------------------------------
// Phase C execution
// ---------------------------------------------------------------------------

/// Creates a fresh fixture directory holding unguessable sentinels.
///
/// Randomness is the point: no amount of plausible reasoning can produce these
/// values, so a correct answer is proof the tool ran, and a confident wrong
/// answer is proof it did not. Uses `uuid` (already a dependency) rather than
/// pulling in a RNG crate for a binary target, which cannot see dev-deps.
fn make_fixtures() -> (std::path::PathBuf, BTreeMap<String, String>) {
    let nonce = uuid::Uuid::new_v4().simple().to_string();
    let dir = std::env::temp_dir().join(format!("toolcall-eval-{nonce}"));
    std::fs::create_dir_all(&dir).expect("create fixture dir");

    let short = &nonce[..12];
    let mut out = BTreeMap::new();

    let token = format!("SENTINEL-{short}");
    std::fs::write(dir.join("token.txt"), &token).expect("write token.txt");
    out.insert("token.txt".to_string(), token);

    let secret = format!("secret-{short}");
    std::fs::write(dir.join("secret.txt"), &secret).expect("write secret.txt");
    out.insert("secret.txt".to_string(), secret);

    // An unguessable line count, away from round numbers. Derived from the
    // nonce so it varies per run without another dependency.
    let n = 37 + (u32::from_str_radix(&nonce[..4], 16).unwrap_or(0) % 57) as usize;
    let body = (1..=n).fold(String::new(), |mut acc, i| {
        let _ = writeln!(acc, "line {i}");
        acc
    });
    std::fs::write(dir.join("lines.txt"), body).expect("write lines.txt");
    out.insert("lines.txt".to_string(), n.to_string());

    let zz = format!("zz-{short}.dat");
    std::fs::write(dir.join(&zz), "x").expect("write zz file");
    out.insert("__dirlist__".to_string(), zz);

    (dir, out)
}

async fn run_phase_c(engine: &Arc<OpenAiEngine>, args: &Args) -> PhaseC {
    let (dir, sentinels) = make_fixtures();
    let dir_str = dir.to_string_lossy().to_string();

    let mut agg = PhaseC::default();

    for probe in FABRICATION_PROBES {
        for run in 0..args.runs {
            let objective = probe.objective.replace("{DIR}", &dir_str);
            let sentinel = probe
                .fixture
                .and_then(|f| sentinels.get(f).cloned())
                .unwrap_or_default();

            let tools = Arc::new(ToolRegistry::with_code_tools());
            let config = ExecutorConfig::new(format!("fab-{}-{run}", probe.id))
                .with_system_prompt(
                    "You are a coding assistant. Use the tools to inspect real files. \
                     If you cannot determine something, say so rather than guessing.",
                )
                .with_working_dir(&dir)
                .with_autonomy(
                    AutonomyGrant::builder()
                        .allow(ToolPattern::Tool("*".to_string()))
                        .build(),
                )
                .with_tool_call_grammar(args.grammar)
                .with_loop_config(LoopConfig {
                    max_iterations: 12,
                    max_tool_calls: 24,
                    detect_implicit_signals: false,
                    ..LoopConfig::default()
                })
                .with_sampling(SamplingParams {
                    temperature: args.temperature,
                    max_tokens: 512,
                    ..SamplingParams::default()
                });

            let dyn_engine: Arc<dyn InferenceEngine> =
                Arc::clone(engine) as Arc<dyn InferenceEngine>;
            let executor = LoopExecutor::new(dyn_engine, tools, config);
            let (tx, mut rx) = mpsc::channel::<LoopEvent>(256);
            let drain = tokio::spawn(async move { while rx.recv().await.is_some() {} });

            let summary = executor.run(&objective, tx).await;
            let _ = drain.await;

            let (answer, tool_calls, turns) = match summary {
                Ok(s) => {
                    let a = match &s.termination {
                        TerminationReason::Natural(NaturalTermination::AnswerProvided {
                            answer,
                            ..
                        }) => answer.clone(),
                        _ => s.partial_answer.clone().unwrap_or_default(),
                    };
                    (a, s.tool_calls_made, s.iterations_completed)
                },
                Err(e) => (format!("<error: {e}>"), 0, 0),
            };

            let grounding = classify(&answer, tool_calls, &sentinel);
            match grounding {
                Grounding::Grounded => agg.grounded += 1,
                Grounding::Called => agg.called += 1,
                Grounding::Abstained => agg.abstained += 1,
                Grounding::WrongEnvelope => agg.wrong_envelope += 1,
                Grounding::Fabricated => agg.fabricated += 1,
            }
            agg.attempts += 1;

            eprintln!(
                "  {:24} {:>2} calls  {:<11} {}",
                format!("{}#{run}", probe.id),
                tool_calls,
                format!("{grounding:?}"),
                if grounding == Grounding::Fabricated {
                    format!("asserted: {}", answer.chars().take(60).collect::<String>())
                } else {
                    String::new()
                }
            );

            agg.results.push(FabricationResult {
                id: format!("{}#{run}", probe.id),
                grounding,
                tool_calls,
                sentinel,
                turns,
                answer,
            });
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    agg
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

/// True when `text` contains a JSON object naming `tool` with an `arguments`
/// field, outside any `<tool_call>` envelope.
///
/// Deliberately loose — it is a diagnostic, not a parser. Its job is to
/// distinguish "the model had the right idea and mis-tagged it" from "the
/// model produced nothing usable", which the emission rate alone conflates.
/// The system prompt's own fill-in-the-blank examples, ellipsis included.
///
/// `fixtures/issue_70_14b_completions.json` categorises 16 of its 126
/// completions `template_echo`, and every one of them is exactly one of these
/// two strings — see `template_echo_completions_are_the_exact_literal_examples`
/// in `tests/issue_70_regression.rs`. `<uncertain>` is included for symmetry:
/// the prompt offers it in the same paragraph, so it is echoable for the same
/// reason even though the captured corpus happens not to contain one.
const TEMPLATE_EXAMPLES: &[&str] = &[
    "<yield>...</yield>",
    "<stuck>...</stuck>",
    "<uncertain>...</uncertain>",
];

/// True when the model reproduced a prompt template example verbatim (#72).
///
/// `contains`, not `==`: an echo buried in a longer completion is the same
/// defect, and the loop's meta-signal detector finds the tag wherever it sits.
fn is_template_echo(text: &str) -> bool {
    TEMPLATE_EXAMPLES.iter().any(|e| text.contains(e))
}

fn looks_like_tool_json(text: &str, tool: &str) -> bool {
    let Some(start) = text.find('{') else {
        return false;
    };
    let body = &text[start..];
    body.contains("\"arguments\"") && body.contains(tool)
}

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
    match a.malformed_rate() {
        Some(r) => println!(
            "  MALFORMED RATE        {:.2}%   ({}/{} emitted calls failed to parse)",
            r * 100.0,
            a.total_malformed,
            a.total_tags
        ),
        None => println!(
            "  MALFORMED RATE        n/a     (no tool calls were emitted — \
             the rate is undefined, NOT 0%)"
        ),
    }
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
    println!(
        "  wrong envelope        {:.1}%   ({}/{} emitted valid call JSON \
         outside <tool_call>)",
        ratio(a.wrong_wrapper, a.attempts) * 100.0,
        a.wrong_wrapper,
        a.attempts
    );
    println!(
        "  template echo         {:.1}%   ({}/{} reproduced a prompt example \
         verbatim — issue #72)",
        ratio(a.template_echo, a.attempts) * 100.0,
        a.template_echo,
        a.attempts
    );
    println!("  mean latency          {:.0} ms", a.mean_latency_ms);

    if a.malformed_rate().is_some() {
        println!("\n  compounding over a multi-call task (assumes independence):");
        for n in [10u32, 20, 40] {
            if let Some(p) = a.failure_over_task(n) {
                println!("    P(>=1 malformed in {n:>2} calls)   {:.1}%", p * 100.0);
            }
        }
    } else {
        println!("\n  compounding: not computable — nothing was emitted to compound.");
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

fn print_phase_c(c: &PhaseC) {
    println!("\n=== Phase C — fabricated tool output ===\n");
    println!(
        "  Every probe targets a value the model cannot guess. Plausibility is\n  \
         never scored; the check is whether the tool call happened.\n"
    );
    println!("  attempts              {}", c.attempts);
    println!(
        "  FABRICATION RATE      {:.1}%   ({}/{} asserted a value with NO tool call)",
        c.fabrication_rate() * 100.0,
        c.fabricated,
        c.attempts
    );
    println!(
        "  tool call happened    {:.1}%   ({}/{})",
        c.call_rate() * 100.0,
        c.grounded + c.called,
        c.attempts
    );
    println!("\n  breakdown:");
    println!("    grounded (sentinel in answer)  {}", c.grounded);
    println!("    called (no sentinel reached)   {}", c.called);
    println!("    abstained (said it can't know) {}", c.abstained);
    println!("    wrong envelope (mis-tagged)    {}", c.wrong_envelope);
    println!("    FABRICATED                     {}", c.fabricated);

    let fabs: Vec<&FabricationResult> = c
        .results
        .iter()
        .filter(|r| r.grounding == Grounding::Fabricated)
        .collect();
    if !fabs.is_empty() {
        println!("\n  fabrications (expected vs asserted):");
        for f in fabs.iter().take(8) {
            println!(
                "    {:24} truth={:<28} said={}",
                f.id,
                f.sentinel,
                f.answer
                    .chars()
                    .take(60)
                    .collect::<String>()
                    .replace('\n', " ")
            );
        }
    }
}

fn print_phase_d(d: &PhaseD) {
    println!("\n=== Phase D — deterministic, machine-checked tasks ===\n");
    println!(
        "  Correctness is decided by running a command, not by reading prose.\n           Thresholds fixed in advance: docs/OFFLOAD-RELIABILITY-PRE-REGISTRATION.md\n"
    );
    println!("  attempts              {}", d.attempts);
    println!(
        "  harness misses        {}   (excluded from the pass rate — our plumbing, not the model)",
        d.harness_misses
    );
    println!("  scored runs           {}", d.scored());

    match (d.pass_rate(), wilson_interval(d.passed, d.scored())) {
        (Some(rate), Some((lo, hi))) => {
            println!(
                "  PASS RATE             {:.1}%   ({}/{})",
                rate * 100.0,
                d.passed,
                d.scored()
            );
            println!(
                "  95% Wilson interval   [{:.1}%, {:.1}%]",
                lo * 100.0,
                hi * 100.0
            );
            // The tier is read off the pre-registered table, not chosen here.
            let tier = if rate >= 0.75 && lo >= 0.60 {
                "QUALIFIED (machine-checked handover, no human in the loop)"
            } else if lo >= 0.40 {
                "QUALIFIED WITH RETRY (budget 2-3 attempts per accepted result)"
            } else {
                "NOT QUALIFIED"
            };
            println!("  pre-registered tier   {tier}");
        },
        _ => println!("  PASS RATE             n/a     (no run was scored — undefined, NOT 0%)"),
    }

    // The share of runs lost to our own plumbing decides whether this result
    // is evidence about the model at all.
    let harness_share = ratio(d.harness_misses, d.attempts);
    if harness_share > 0.20 {
        println!(
            "\n  *** RESULT VOID: {:.0}% of runs were harness misses (limit 20%). ***\n               Fix the plumbing and re-run; this measures us, not the model.",
            harness_share * 100.0
        );
    }

    let mut by_task: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for r in &d.results {
        let base = r.id.split('#').next().unwrap_or(&r.id);
        let e = by_task.entry(base).or_insert((0, 0));
        e.0 += 1;
        if r.passed {
            e.1 += 1;
        }
    }
    // Raw counts, never percentages: at n=5 a per-task rate invents precision.
    println!("\n  per task (raw counts — see the pre-registration on why not %):");
    for (task, (n, ok)) in by_task {
        println!("    {task:28} {ok}/{n}");
    }

    let misses: Vec<&DetResult> = d.results.iter().filter(|r| !r.passed).collect();
    if !misses.is_empty() {
        println!("\n  every miss, with its cause:");
        for m in &misses {
            println!(
                "    {:28} {:<26} {}",
                m.id,
                format!("{:?}", m.miss.unwrap_or(MissCause::Model)),
                // Truncated for the terminal only — the full text is in the
                // --json report, which is what a later reader will have.
                m.verify_output
                    .as_deref()
                    .unwrap_or("")
                    .replace('\n', " ")
                    .chars()
                    .take(88)
                    .collect::<String>()
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
        "  (pre-#24 rubric)      {:.1}%   ({}/{})   same model output, old scorer",
        ratio(b.legacy_correct, b.attempts) * 100.0,
        b.legacy_correct,
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

    // Any run the two rubrics disagree about, named individually. An
    // aggregate that moved without saying which runs moved would be exactly
    // the "one number hiding several causes" problem #24 was filed about.
    let disagreements: Vec<&TaskResult> = b
        .results
        .iter()
        .filter(|r| r.correct != r.legacy_correct)
        .collect();
    if disagreements.is_empty() {
        println!("\n  rubric change (#24): no run scored differently under the two rubrics.");
    } else {
        println!(
            "\n  rubric change (#24): {} run(s) scored differently — identical model output:",
            disagreements.len()
        );
        // Neutral wording on purpose. Labelling these "the old rubric was
        // wrong" assumes the new one is right, and that assumption is exactly
        // what failed once already: the first `ANSWER:` extractor rejected a
        // correct answer whose marker sat at the end of a sentence, and a
        // confident label would have presented that as the old rubric passing
        // a wrong answer. A disagreement is a flag to go and read the answer,
        // not a verdict.
        for r in &disagreements {
            let which = if r.correct {
                "passed by the current rubric only"
            } else {
                "passed by the pre-#24 rubric only"
            };
            println!(
                "    {:28} {which} — read the answer before concluding",
                r.id
            );
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn task(id: &str) -> &'static Task {
        TASKS.iter().find(|t| t.id == id).expect("task must exist")
    }

    // === The conclusion is parsed, not pattern-matched (#24) ===

    #[test]
    fn the_answer_line_is_extracted_from_surrounding_prose() {
        assert_eq!(
            answer_value("Some reasoning here.\nANSWER: 1").as_deref(),
            Some("1")
        );
        // Decoration the grammar or the model may add around the value.
        assert_eq!(answer_value("ANSWER: `no`").as_deref(), Some("no"));
        assert_eq!(answer_value("ANSWER: **0**.").as_deref(), Some("0"));
        assert_eq!(answer_value("ANSWER: -s").as_deref(), Some("s"));
        assert_eq!(
            answer_value("  answer: 1").as_deref(),
            None,
            "case-sensitive marker"
        );
        assert_eq!(answer_value("no marker here").as_deref(), None);
        assert_eq!(
            answer_value("ANSWER:   ").as_deref(),
            None,
            "empty is not a value"
        );
    }

    /// Verbatim from a real 14B run, and rejected by the FIRST version of this
    /// extractor because the marker sits at the end of a sentence rather than
    /// on its own line. Kept exactly as captured — this is the evidence, not
    /// an illustration.
    #[test]
    fn a_marker_at_the_end_of_a_sentence_is_still_the_answer() {
        let captured = "Reading the pipeline's status would have been misleading because the \
                        exit status of the pipeline is determined by the last command in the \
                        pipeline, which is `tail -1`, and not by the `false` command itself. \
                        ANSWER: 1";
        assert_eq!(answer_value(captured).as_deref(), Some("1"));

        let (current, _) = score(task("exit-code-discipline"), captured, true);
        assert!(
            current,
            "rejecting this is #24's defect reintroduced by #24's own fix"
        );
    }

    #[test]
    fn every_expected_answer_is_a_single_token() {
        // `answer_value` reads one token, so a task expecting more would be
        // silently graded on a prefix. Fail here instead.
        for t in TASKS {
            if let Some(expected) = t.expect_answer {
                assert!(
                    !expected.trim().contains(char::is_whitespace),
                    "{}: expected answer {expected:?} is more than one token, which \
                     answer_value cannot represent",
                    t.id
                );
            }
        }
    }

    #[test]
    fn a_restated_format_does_not_win_over_the_real_answer() {
        // A model that echoes the instruction before using it should be read
        // as meaning its last line, not its first.
        let text = "I will reply as ANSWER: <number>\nWorking...\nANSWER: 1";
        assert_eq!(answer_value(text).as_deref(), Some("1"));
    }

    /// The defect this ticket was filed for: the pipeline's status is 0, the
    /// objective demands an explanation of why that misleads, and the old
    /// rubric rejected the answer for saying so.
    #[test]
    fn exit_code_discipline_no_longer_penalises_the_complete_answer() {
        let t = task("exit-code-discipline");
        let answer = "The exit status of the `false` command is 1. Reading the pipeline's \
                      status would have been misleading because the pipeline's exit status is \
                      determined by the last command, `tail -1`, which succeeded (exit status \
                      0), not by the `false` command itself.\nANSWER: 1";

        let (current, legacy) = score(t, answer, true);
        assert!(
            current,
            "a correct answer that explains why 0 misleads must score correct"
        );
        assert!(
            !legacy,
            "this is the exact answer the old rubric rejected — if it now passes \
             there, the fixture for this claim has changed"
        );
    }

    #[test]
    fn exit_code_discipline_still_rejects_the_wrong_conclusion() {
        // The job `expect_none: ["status 0"]` was doing. Reporting the
        // pipeline's 0 as the answer must still fail, and now does so because
        // the conclusion is compared, not searched for.
        let t = task("exit-code-discipline");
        let answer = "Running `false | tail -1` gave exit status 0.\nANSWER: 0";
        let (current, _) = score(t, answer, true);
        assert!(!current, "answering 0 must fail");
    }

    /// The opposite direction, and the more dangerous one: `expect_all:
    /// ["no"]` matches inside "not", so an answer saying **Yes** scored
    /// correct. Latent rather than observed — the model happened to answer
    /// the single word "No" — but a false pass inflates, and nothing in the
    /// old rubric would ever have revealed it.
    #[test]
    fn a_yes_answer_no_longer_passes_a_task_whose_answer_is_no() {
        let t = task("grammar-unused");
        let answer = "Yes. Beleth sets a grammar on its generation requests, although I \
                      could not confirm every call site.\nANSWER: yes";

        let (current, legacy) = score(t, answer, true);
        assert!(
            !current,
            "an answer of `yes` must fail a task whose answer is `no`"
        );
        assert!(
            legacy,
            "the old rubric accepted this, via the `no` inside `could not` — \
             that is the defect being pinned, not a typo"
        );
    }

    /// The under-credit case from PR #86: the model answered in words and the
    /// rubric wanted a digit. The wording is the shape of answer the 14B
    /// actually gave, carried over to this task's successor.
    #[test]
    fn a_number_stated_in_words_now_scores_when_the_answer_line_carries_it() {
        let t = task("detector-override-unused");
        let answer = "The method `with_detector` is only defined in \
                      `src/agentic_loop/executor.rs` and is not called anywhere else in the \
                      repository.\nANSWER: 0";
        let (current, _) = score(t, answer, true);
        assert!(current);
    }

    #[test]
    fn a_missing_answer_line_is_a_miss_not_a_pass() {
        // An unparseable conclusion is the ambiguity this replaced; it must
        // not fall through to "no rubric objected, therefore correct".
        let t = task("grammar-unused");
        let (current, _) = score(t, "No, beleth never sets one.", true);
        assert!(!current);
    }

    #[test]
    fn an_incomplete_run_is_never_correct() {
        let t = task("grammar-unused");
        let (current, legacy) = score(t, "ANSWER: no", false);
        assert!(!current);
        assert!(!legacy);
    }

    // === Stale-premise guard (#27) ===

    /// A throwaway tree with one source file, so the predicate is tested
    /// against what it is designed to scan — a repository — rather than
    /// against this checkout, which also contains the task text naming the
    /// very symbols under test.
    fn tree_containing(body: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("src")).expect("mkdir");
        std::fs::write(dir.path().join("src/lib.rs"), body).expect("write");
        dir
    }

    #[test]
    fn the_premise_guard_sees_a_symbol_that_is_present() {
        let dir = tree_containing("pub fn with_detector() {}\n");
        assert!(repo_contains_symbol(dir.path(), "with_detector"));
    }

    #[test]
    fn the_premise_guard_notices_a_symbol_that_is_absent() {
        // The shape of infernum-framework#27: the task keeps asking about
        // `with_all_tools` after the constructor has been deleted.
        let dir = tree_containing("pub fn with_code_tools() {}\n");
        assert!(!repo_contains_symbol(dir.path(), "with_all_tools"));
    }

    #[test]
    fn the_premise_guard_matches_whole_words_only() {
        let dir = tree_containing("pub fn with_detector_extended() {}\n");
        assert!(
            !repo_contains_symbol(dir.path(), "with_detector"),
            "a longer name must not satisfy a premise about the shorter one"
        );
    }

    #[test]
    fn a_symbol_named_only_by_the_harness_is_not_evidence() {
        // The self-satisfying case. A tree whose only mention of the symbol is
        // a copy of this harness must not count as containing it.
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("toolcall-eval.rs"),
            "objective: \"... calls with_all_tools ...\"",
        )
        .expect("write");
        assert!(!repo_contains_symbol(dir.path(), "with_all_tools"));
    }

    #[test]
    fn a_symbol_surviving_only_in_captured_output_is_not_evidence() {
        // `tests/fixtures/issue_70_14b_completions.json` still contains
        // `with_all_tools`, because it records what a model said in 2026-09.
        // That is history, not a symbol table.
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("fixtures")).expect("mkdir");
        std::fs::write(
            dir.path().join("fixtures/captures.json"),
            "{\"text\": \"calls with_all_tools\"}",
        )
        .expect("write");
        assert!(!repo_contains_symbol(dir.path(), "with_all_tools"));
    }

    #[test]
    fn every_asserted_symbol_is_present_in_this_workspace() {
        // Catches what #27 hit: a symbol deleted from the workspace while a
        // task kept asserting it. Scans the workspace, not just this crate,
        // because `grammar-unused` asserts a type from infernum-core.
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(std::path::Path::parent)
            .expect("crates/<crate> has a workspace root two levels up");
        for t in TASKS {
            if let Some(symbol) = t.asserts_symbol {
                assert!(
                    repo_contains_symbol(workspace, symbol),
                    "{}: asserts `{symbol}`, which is not in this workspace",
                    t.id
                );
            }
        }
    }

    #[test]
    fn every_task_states_the_answer_format_it_will_be_graded_on() {
        // A task graded on an `ANSWER:` line it never asked for would be
        // scoring the model's ability to guess a private convention.
        for t in TASKS {
            if t.expect_answer.is_some() {
                assert!(
                    t.objective.contains("ANSWER:"),
                    "{} is graded on an ANSWER line its objective never requests",
                    t.id
                );
            }
        }
    }

    // === Phase D checker soundness (pre-registration §5) ===
    //
    // These run in CI, before any measurement. A check that passes on the
    // untouched fixture reports a free pass; a check nothing can satisfy
    // reports a free failure. Both are the defect shape this whole sequence
    // has been about: a result that was never actually tested.

    fn sandbox(task: &DetTask) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        materialise(task, dir.path()).expect("materialise fixture");
        dir
    }

    #[test]
    fn every_check_fails_on_the_untouched_fixture() {
        for task in DET_TASKS {
            let dir = sandbox(task);
            let (passed, output) = run_verify(task, dir.path()).expect("verify must run");
            assert!(
                !passed,
                "{}: the check PASSES before the model has done anything, so the task                  scores a free pass and measures nothing.\n{output}",
                task.id
            );
        }
    }

    #[test]
    fn every_check_passes_on_its_reference_solution() {
        for task in DET_TASKS {
            let dir = sandbox(task);
            apply_reference(task, dir.path()).expect("apply reference");
            let (passed, output) = run_verify(task, dir.path()).expect("verify must run");
            assert!(
                passed,
                "{}: the hand-written correct solution does NOT satisfy the check, so the                  task is unpassable and would report a free failure.\n{output}",
                task.id
            );
        }
    }

    #[test]
    fn the_reference_solution_never_touches_a_protected_file() {
        // If the known-correct solution has to edit a file the task forbids
        // editing, the task contradicts itself.
        for task in DET_TASKS {
            for (rel, _) in task.reference {
                assert!(
                    !task.immutable.contains(rel),
                    "{}: reference solution edits {rel}, which the task declares immutable",
                    task.id
                );
            }
        }
    }

    #[test]
    fn protected_files_exist_in_the_fixture() {
        // An immutable path that does not exist snapshots as absent and
        // compares equal to absent — the guard would silently never fire.
        for task in DET_TASKS {
            for rel in task.immutable {
                assert!(
                    task.files.iter().any(|(p, _)| p == rel),
                    "{}: declares {rel} immutable but never creates it, so the guard                      can never fire",
                    task.id
                );
            }
        }
    }

    #[test]
    fn tampering_with_a_protected_file_is_detected() {
        // The guard itself needs proving, not just declaring.
        for task in DET_TASKS {
            let dir = sandbox(task);
            let before = snapshot_immutable(task, dir.path());
            assert!(!before.is_empty(), "{}: nothing protected", task.id);

            let victim = task.immutable[0];
            std::fs::write(dir.path().join(victim), "tampered").expect("write");
            assert_ne!(
                snapshot_immutable(task, dir.path()),
                before,
                "{}: editing {victim} went undetected",
                task.id
            );
        }
    }

    #[test]
    fn no_fixture_contains_the_answer_it_is_testing() {
        // Phase D's sandbox holds only what the task defines, so the harness
        // is out of reach — but a fixture could still hand over its own
        // answer. The reference solution's new content must not already be
        // sitting in the files the model starts from.
        for task in DET_TASKS {
            let starting: String = task.files.iter().map(|(_, c)| *c).collect();
            for (rel, solution) in task.reference {
                let novel: Vec<&str> = solution
                    .lines()
                    .map(str::trim)
                    .filter(|l| l.len() > 12 && !starting.contains(*l))
                    .collect();
                assert!(
                    !novel.is_empty(),
                    "{}: every line of the solution for {rel} is already present in the                      fixture, so the task may be answerable by copying",
                    task.id
                );
            }
        }
    }

    #[test]
    fn wilson_interval_brackets_the_point_estimate() {
        let (lo, hi) = wilson_interval(24, 30).expect("n > 0");
        assert!(lo < 0.8 && 0.8 < hi, "[{lo}, {hi}] must contain 24/30");
        assert!(lo > 0.0 && hi < 1.0);

        // Degenerate cases stay inside [0, 1] rather than running past it,
        // which is why this is Wilson and not the normal approximation.
        let (lo, hi) = wilson_interval(5, 5).unwrap();
        assert!(lo > 0.0 && (hi - 1.0).abs() < 1e-9, "[{lo}, {hi}]");
        let (lo, hi) = wilson_interval(0, 5).unwrap();
        assert!(lo.abs() < 1e-9 && hi < 1.0, "[{lo}, {hi}]");

        assert!(wilson_interval(0, 0).is_none(), "undefined, not zero");
    }

    #[test]
    fn a_pass_rate_over_nothing_is_undefined_not_zero() {
        let d = PhaseD::default();
        assert!(d.pass_rate().is_none());
    }

    #[test]
    fn harness_misses_are_excluded_from_the_denominator() {
        let d = PhaseD {
            attempts: 10,
            passed: 6,
            harness_misses: 2,
            results: Vec::new(),
        };
        assert_eq!(d.scored(), 8);
        assert!((d.pass_rate().unwrap() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn no_task_grades_prose_by_substring_any_more() {
        // `expect_all` survives only for literal source identifiers. Anything
        // longer than a token is prose, and prose is what #24 is about.
        for t in TASKS {
            for marker in t.expect_all {
                assert!(
                    !marker.contains(' '),
                    "{}: {marker:?} is a phrase, not an identifier — grading prose \
                     by substring is the defect this replaced",
                    t.id
                );
            }
        }
    }
}
