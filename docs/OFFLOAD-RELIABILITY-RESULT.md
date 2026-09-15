# Offload reliability: result

INFERNUM-19. Measured against the thresholds fixed in
`OFFLOAD-RELIABILITY-PRE-REGISTRATION.md`, committed before any run.

Qwen2.5-Coder-14B-Instruct Q8_0, upstream `llama-server` (llama.cpp c32d1da,
CUDA), temperature 0, tool-call grammar on. Class J against a pristine
`git worktree --detach 95097d5`; class D in per-run generated sandboxes.

**This document reports the number. It does not pick a workload.**

---

## Headline

| class | verification | result | 95% Wilson | pre-registered tier |
|---|---|---|---|---|
| **D** — machine-checked | a command's exit status | **50/75 = 66.7%** | [55.4%, 76.3%] | **QUALIFIED WITH RETRY** |
| **J** — judgement-checked | a human reads it | **13/25 = 52.0%** | [33.5%, 70.0%] | **NOT QUALIFIED** |

Harness-caused misses: **0 of 100 runs**, so the 20% void rule is not
triggered and both results stand as evidence about the model.

---

## The aggregate describes neither half — class D is bimodal

| task | passes | 95% Wilson | what it asks for |
|---|---|---|---|
| `fix-compile-error` | **15/15** | [80%, 100%] | make a broken crate compile |
| `config-repoint` | **15/15** | [80%, 100%] | sweep 2 of 3 config files, leave the third |
| `exhaustive-match` | 8/15 | [30%, 75%] | add a missing match arm, no catch-all |
| `implement-stub` | 7/15 | [25%, 70%] | implement a `todo!()` against a fixed spec |
| `rename-symbol` | 5/15 | [15%, 58%] | rename across three files |

66.7% is the average of a 15/15 group and a ~33–53% group. **No task scored
near 67%.** Reading the headline as "the model is two-thirds reliable" would
describe none of the work measured, which is the same error as reading 6/15 as
"40% correct" in INFERNUM-17.

The two perfect tasks share a shape: **a single file to change, and the change
is fully determined by an error message or an explicit pattern.** The three
weak ones all require either locating files first or reconstructing content.

---

## n = 25 does not decide the tier

Three identical runs — same tasks, same model, same temperature 0, same
everything:

| run | result | tier |
|---|---|---|
| 1 | 12/25 = 48.0% | **NOT QUALIFIED** |
| 2 | 19/25 = 76.0% | QUALIFIED WITH RETRY |
| 3 | 19/25 = 76.0% | QUALIFIED WITH RETRY |

Two of the three tiers appear in three runs of the same configuration.
Temperature 0 is not bit-deterministic through llama.cpp, and the spread is
large enough to cross a decision boundary. The pooled n=75 figure is the one to
use; a single n=25 run is not a basis for a scope decision, and the
pre-registration's "if the interval is too wide, that is the result" clause
applies to any single run here.

All three runs are pooled. Selecting one would be cherry-picking, and the
decision to pool was made before the third run was started.

---

## Every miss, by cause

### Class D — 25 misses over 75 runs

| cause | count | notes |
|---|---|---|
| `Model` | 21 | the check ran and rejected the work |
| `ModelRanOutOfBudget` | 4 | iteration budget exhausted |
| `ModelTamperedWithCheck` | 0 | nothing edited a protected file |
| `Harness` | 0 | — |

**No run cheated.** The immutable-file guard — a Rust-side comparison against a
pre-run snapshot, not a check inside the sandbox — never fired. "Make the tests
pass" was never solved by editing the tests.

### Class J — 12 misses over 25 runs

| task | passes | dominant cause |
|---|---|---|
| `grammar-unused` | 5/5 | — |
| `exit-code-discipline` | 4/5 | 1 run gave the right answer without the requested marker |
| `supervisor-spawns-nothing` | 4/5 | 1 incomplete |
| `clap-collision` | 0/5 | **model** — answered `--model`/`--system` on `m`, and `'H'`/`'n'` |
| `claude-code-unregistered` | 0/5 | **model** — see below |

---

## Two tool defects that suppress class D, reported not fixed

Found by tracing (`--keep-sandboxes`); neither is visible from a finished
sandbox. Both are filed separately. **They are classified `Model`, not
`Harness`, and the pass rate is not adjusted for them** — the model was given a
fair attempt and mishandled a degraded tool. But they mean 66.7% is a **floor**
for this model through better tooling, not a ceiling.

**1. `search_files` emits paths its sibling tools reject.** It reports
`lib.rs` for a file at `src/lib.rs` — relative to the searched directory, not
the working directory. Passing that to `edit_file` fails with "No such file or
directory". Traced runs of `rename-symbol` show the model following the search
output into three failed edits and giving up; the run that passed ignored the
search result and guessed `src/lib.rs`. A model that *trusts* the tool is the
one that fails.

**2. `read_file`'s line-numbered output does not survive a round trip.**
`{:>6}\t{line}`, with nothing marking the numbers as not part of the file. Runs
that rewrote a whole file via `write_file` produced `1    /// Returns how
many...` or dropped a `/// ` prefix mid-doc-comment. Runs that used `edit_file`
surgically passed. `implement-stub`'s misses are all this.

## One model failure worth naming separately

`claude-code-unregistered` exhausted its 25-iteration budget in **all five**
runs by issuing the *same* `search_files` call twenty-five times. The tool
returned the correct result every time. This is not a wrong answer and not a
broken tool: it is an inability to recognise that the question is already
answered. For a handover decision that matters more than a wrong answer does —
a wrong answer is caught by the checker, a loop burns the budget and returns
nothing. It is the same shape as the turn-burning INFERNUM-20 fixed, with no
tool defect behind it.

---

## A defect in this measurement, found and fixed mid-run

The first `ANSWER:` extractor required the marker at the start of a line. A run
answered correctly with the marker at the end of a sentence and was scored
wrong — #24's own defect, reintroduced by #24's fix. It cost
`exit-code-discipline` 0/5; corrected, the same task wording scores 4/5. The
captured string is now a test.

Reported rather than quietly corrected because it is the reason class J's
number moved, and a number that moved because I fixed my own scorer must be
separable from one that moved because the model improved. Every class J run is
scored under both the current and the pre-#24 rubric: 13/25 vs 14/25 over
identical output, with the single disagreeing run named in the report.

---

## What this does and does not support

- Class J is **NOT QUALIFIED** at 52% against a 90% bar, and was never the
  class being qualified — review cost lands on a human for all of it.
- Class D as a whole is **QUALIFIED WITH RETRY**, but the class average is not
  a property any task actually has.
- The two 15/15 task shapes are the only ones measured at a rate that would
  survive the QUALIFIED bar, and 15/15 at n=15 has a 95% lower bound of 80% —
  above the 0.75/0.60 threshold. That is a statement about **those two task
  shapes**, not about a workload.
- Which concrete work resembles those shapes closely enough to inherit the
  number is a judgement this measurement cannot make. That decision is
  Daemon's.
