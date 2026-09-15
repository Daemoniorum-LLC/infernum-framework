# Lessons Learned

This document captures organizational memory for the Infernum project. Mistakes,
discoveries, and successful patterns are recorded here so future agents don't
repeat failures or lose hard-won insights.

---

## How to Use This Document

**When starting work:**
- Read recent entries relevant to your task
- Note any warnings about approaches that failed

**When ending a session:**
- Document any significant discoveries
- Record mistakes that future agents should avoid
- Note successful patterns worth repeating

---

## 2026-09-15 - The capability was built and never invoked

### Context

Issue #70 reported `0/111` tool-call attempts hitting the required envelope on
Qwen2.5-Coder-14B, and Phase B at `0/15` completed. It read as a model
limitation. INFERNUM-17 set out to constrain the envelope with a GBNF grammar.

### What Happened

The grammar type already existed. `infernum-core`'s `GrammarConstraint` is
complete — `Json`, `Gbnf`, `JsonSchema`, a working `to_gbnf()`, tests.
`llama_cpp_engine.rs:1093` honours it properly: `to_gbnf()` → `LlamaGrammar` →
sampler stage.

Two omissions made it inert:

- `OpenAiEngine::build_body` assembles the request with eleven sampling knobs —
  `top_k`, `min_p`, three penalties, `stop` — each behind a deliberate "only send
  when it carries signal" conditional. **`s.grammar` is not among them.** No
  warning, no error, no test. It is dropped.
- `beleth` never called `with_grammar`. Zero occurrences in `crates/beleth/src/`.

So the engine that worked ignored grammars, and the engine that honoured
grammars was blocked by a separate tokenizer bug (#56). Adding the field and
setting it took the envelope from 0/30 to 30/30 emission, 0.00% malformed, and
Phase B from 0/15 to 15/15 completed.

The same shape was then found one level up: `crates/malphas` is 8,179 lines with
120 tests — priority queues, continuous batching, a service that "wires the
scheduler to the inference engine" — and `crates/legion` is 7,393 more with 112
tests. Both are declared dependencies of `beleth`, `infernum` and
`infernum-server`, and **referenced by zero source files outside themselves.**
The only cross-crate match is the word "legions" in a doc-comment epigraph.

### Root Cause

**Nothing fails when a capability is simply not invoked.** A dropped constraint
produces no error; the caller believes it is constrained and the system behaves
as though the feature does not exist. A dependency edge with no call sites
compiles cleanly forever. Unit tests pass, because the units are correct — they
are just never reached.

### Lesson

**A feature that exists is not a feature that runs.** Before concluding a
capability is missing or a model is deficient, establish that the existing
implementation is actually on the path being measured. `grep` for the call site,
not for the definition.

And the inverse, for reviewers: **green unit tests on an unwired module prove
the units behave and nothing else.** 232 passing tests across `malphas` and
`legion` say nothing about whether either has ever run against a real engine.

### Prevention

A constraint that cannot be honoured must **error**, never drop silently — that
is now the contract in `OpenAiEngine`. When adding an optional field to a
request builder, ask what happens when it is set and the transport cannot carry
it; "nothing" is the wrong answer.

---

## 2026-09-15 - A tool defect costs far more than its size

### Context

Phase B's `exit-code-discipline` scored 0/3. Phase D's `rename-symbol` scored
5/15. Both looked like model failures.

### What Happened

Both were tool defects.

`tools/bash.rs:150` treated a non-zero **command** exit as a **tool** failure:
`false` returned `success: false, error: "Command exited with code 1"`. The model
ran the command correctly, was told its shell was broken, and concluded *"the
tool execution failed repeatedly."* Fixing it moved the task from 2/3 to 3/3
completed — and cut turns from **20 / 8 / 10 to 3 / 3 / 3**.

`tools/search_files.rs` reports match paths relative to the directory searched,
not the agent's working directory, so its output is not valid input to
`edit_file` or `read_file`. The trace shows the model taking `lib.rs` from search,
failing three edits against it, re-running the same search, and giving up.
**The run that PASSED did so by ignoring the search output and guessing
`src/lib.rs` directly.**

### Root Cause

Both tools conflated two states that need opposite responses. `success: false`
should mean *the tool could not run the command* — spawn failure, timeout, bad
working directory. A non-zero exit code is a normal, often expected **result**:
`grep` with no match, a failing test, `git diff --quiet` reporting changes.
Similarly, a path is only useful if it resolves in the frame its consumer uses.

### Lesson

**A model that trusts its tools is the one that fails.** That is a property of
the tool, not the model. When a task scores badly, trace a failing run before
attributing it to the model — `--keep-sandboxes` exists for this.

The turn count is the quieter cost. The model was not merely wrong; it burned
its budget fighting a tool that kept lying to it. On a metered or time-boxed
loop that is the expensive half.

### Prevention

Pin tool contracts with tests that cross tool boundaries: grep for a symbol, then
feed each returned path straight to `read_file`. A per-tool unit test cannot
catch a frame mismatch between two tools that are individually correct.

`bash.rs:217` originally asserted `!result.success` for a non-zero exit — **the
test encoded the bug.** It is now
`a_non_zero_exit_is_a_result_not_a_tool_failure`, carrying its reasoning in a doc
comment: *"If a future change makes this assert `!result.success` again, that is
this bug coming back."* When fixing a defect that a test enshrines, rename the
test to state the invariant and say why, or the next person restores the bug in
good faith.

---

## 2026-09-15 - Measure the harness before you measure the model

### Context

INFERNUM-19 was asked to produce a reliability number to decide what work could
be handed to a local model.

### What Happened

The starting figure was `6/15` — read as "40% correct". Decomposed, of fifteen
runs: six counted correct, three were correct but miscounted by a broken scorer,
three were correct but sabotaged by `bash.rs`, and **three — one task — were
genuine model error.**

Then a third scorer defect was found by asking why the two *passing* tasks
passed: `expect_all: ["no"]` matches inside the word "not", so an answer
beginning *"Yes."* scored **correct** on a task whose answer was "no". Four of
the five original tasks had a broken scorer.

The fix for the scorers then reintroduced the defect it was fixing: the first
`ANSWER:` extractor required the marker at the start of a line, and a correct
answer with it mid-sentence was scored wrong — costing that task 0/5, corrected
to 4/5.

### Root Cause

A substring rubric is a proxy for correctness, and proxies drift from the thing
they stand for in both directions. Under-crediting surfaces eventually, because
someone investigates a surprising failure. **Over-crediting never surfaces on its
own — nobody investigates a pass.**

### Lesson

**Ask why the passing cases pass.** An inflating defect is invisible to every
process that only examines failures, and it is the more dangerous direction: it
produces confidence rather than confusion.

Also: `expect_all: ["1"]` cannot catch a wrong answer when the task's own
objective text contains `tail -1`. **A marker that appears in the question is not
evidence about the answer.**

### Prevention

Pre-registration. `docs/OFFLOAD-RELIABILITY-PRE-REGISTRATION.md` was committed in
`2b1c9d8`, **before** the measurement code in `58d9001` — verifiable by commit
order rather than by assertion. It fixes the thresholds, the two verification
classes and their different bars while the result is still unknown, and states
that corrections are recorded as later changes rather than edited in place.

A threshold chosen after seeing data is not a threshold.

Two guards now enforce the harness's own premises: `assert_uncontaminated`
refuses to run Phase B against a checkout containing the answer key, and
`asserts_symbol` refuses when a task's asserted symbol is absent from the repo —
without it, deleting a symbol leaves every run recording a model miss on a
question with no answer. Note that `repo_contains_symbol` **excludes the harness
file and `fixtures/`**: both mention the symbols tasks ask about, and a guard
satisfied by a task's own objective would be self-satisfying.

---

## 2026-09-15 - A red CI log is a truncated list of problems

### Context

The Documentation job failed on PR #86 after the session had reported. The log
showed two `rustdoc::redundant_explicit_links` violations.

### What Happened

There were three, plus a fourth lint of a different class
(`private_intra_doc_links`). **rustdoc aborts a crate at its first error**, so the
log could only ever reveal a prefix of the problems. Reproducing the job's exact
command locally surfaced the rest:

    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps \
      --features llama-cpp,holotensor,integration-tests

### Root Cause

A failing tool stops at its first failure. Its output is a sample, not a census —
and nothing in the log says so.

### Lesson

**Fix the cause, then re-run the command, rather than fixing what the log
showed.** This is the same family as every other truncation trap: `gh pr view
--json files` silently caps at 100, the REST `pulls/N/files` endpoint caps at
3,000 even with `--paginate`, and a default-branch blob comparison understates
preservation. In each case a partial answer is indistinguishable from a complete
one.

### Prevention

Reproduce CI's exact command locally before pushing a fix for a CI failure.
Where a check has an explicit truncation signal — the git trees API has
`.truncated` — read it. Where it does not, assume the list is partial.

---

## Template for Future Entries

```markdown
## YYYY-MM-DD - [Session/Feature Name]

### Context
What were we trying to do?

### What Happened
What went wrong or right?

### Root Cause (if applicable)
Why did this happen?

### Lesson
What should future agents know?

### Prevention (if applicable)
How do we avoid this in future?
```
