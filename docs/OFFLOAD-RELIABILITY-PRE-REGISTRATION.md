# Offload reliability: pre-registration

**Written and committed BEFORE any measurement it governs.** INFERNUM-19.

Everything in this document is a commitment made while the result is still
unknown. A threshold chosen after seeing data is not a threshold, it is a
rationalisation — and this measurement gates a scope decision, so the
temptation to pick the line that the data happens to clear is exactly the
failure mode to design against.

If a number below turns out to be wrong, the correction goes in a later
document with its reasoning, recorded as a change. It is not edited here.

---

## 1. The question

Daemon, 2026-09-15, on what Infernum should be given:

> "whatever it shows it's capable of doing reliably"

That makes "reliably" the load-bearing word, and it is not self-defining. This
document fixes what it means before the run.

It does **not** pick a workload. It states what a measured *class* of work would
have to score to qualify for handover, and with what kind of checking. Which
concrete work falls into a qualifying class is Daemon's decision.

---

## 2. Why the check matters more than the score

The reasoning this design follows, from Daemon:

> A worker that is 60% correct on work where a machine can tell you WHICH 60%
> is useful. A worker that is 90% correct on work only a human can check may
> not be — because the review cost lands on the human for all 100%, which is
> the cost we are trying to remove.

So a pass rate is meaningless without saying who verifies the output. Two
classes, and they get different thresholds:

| class | verification | who pays for a failure |
|---|---|---|
| **D — machine-checkable** | a command's exit status decides pass/fail, no judgement | nobody; a failure is detected and retried |
| **J — judgement-checkable** | correctness requires reading the answer | a human, for 100% of output, pass or fail |

The existing five Phase B tasks are all class J: investigative root-cause
questions graded by substring match. That is the worst case on both axes. It is
not a criticism of the harness — it was built to answer "is this model usable
at all", and it answered that.

---

## 3. Thresholds, fixed in advance

Let `p` be the pass rate over a class, and `L` the lower bound of its 95%
Wilson score interval. Wilson rather than normal-approximation because at
n≈30 and p near 0.8 the normal interval misbehaves.

### Class D — machine-checkable

| tier | condition | what it means |
|---|---|---|
| **QUALIFIED** | class `p ≥ 0.75` **and** `L ≥ 0.60` | the class can be handed over with an automatic checker and no human in the loop |
| **QUALIFIED WITH RETRY** | `L ≥ 0.40` | handover is viable but budget ~2–3 attempts per accepted result |
| **NOT QUALIFIED** | `L < 0.40` | do not hand this class over |

The arithmetic behind 0.60, so it is not an aesthetic choice: at `p = 0.60`
the expected number of attempts per accepted result is `1/0.6 ≈ 1.7`, and a
3-attempt budget succeeds with probability `1 − 0.4³ = 94%`. That is the point
at which "dispatch it and take the checker's word" is an honest description of
the workflow. At `p = 0.40` the same budget gives `78%`, which still works if
a failure is cheap, but is no longer unattended. Below that the retry loop
costs more supervision than the work saves.

**A tier is claimed only if the checker is sound** — see §5. A high pass rate
against a checker that can accept wrong output measures nothing, and is the
same defect shape as every other finding in this sequence: a green result that
was never actually tested.

### Class J — judgement-checkable

| tier | condition |
|---|---|
| **QUALIFIED** | class `p ≥ 0.90` **and** `L ≥ 0.75` |
| **NOT QUALIFIED** | anything less |

Higher, because review cost lands on a human whether the answer is right or
wrong; the model has to be right often enough that reading its output beats
doing the work. Reported for continuity with #70/#71, but **class J is not the
class being qualified here**, and no workload decision should rest on it.

---

## 4. Sample size and what will be reported

- **n = 5 runs per task**, temperature 0.
- Per-task results are reported as **raw counts**, never as percentages. At
  n = 5 a per-task interval spans most of the unit line; quoting "80%" from
  4/5 would invent precision that is not there.
- The **class aggregate** carries the Wilson interval, and the tier is decided
  on the class, not on any single task.
- If the class interval is too wide to separate two tiers, **that is the
  reported result** — "n does not decide this" — not a nudge toward whichever
  tier is closer.

### Miss classification — mandatory, every miss

The 6/15 figure from INFERNUM-17 hid three different causes behind one number.
Every miss is therefore reported with a cause:

| class | meaning |
|---|---|
| `model` | the checker ran, and rejected the model's work |
| `rubric` | the check is wrong — it rejects correct work or accepts incorrect work |
| `harness` | the run never got a fair attempt: tool error, timeout, sandbox failure, resource limit |

`harness` misses are excluded from the pass rate and reported separately with
their count, because they measure our plumbing rather than the model. If they
exceed **20%** of runs in a class, the class result is declared **void** and
re-run after fixing the plumbing — a measurement mostly reflecting our own
failures is not evidence about the model.

---

## 5. Checker soundness — validated before the run, in tests

Every class D task must satisfy both, asserted in the test suite and run in CI:

1. **The check FAILS on the untouched fixture.** A check that passes before
   the model does anything scores nothing and would report a free pass.
2. **The check PASSES on a reference solution** written by hand, independently
   of any model output. A check nothing can satisfy reports a free failure.

This is the same discipline as verifying a new assertion fails against unfixed
code, applied to the measurement instead of the code.

Additionally, each task names files that must be **byte-identical** after the
run. These are checked by the harness in Rust, not by a shell command inside
the sandbox, so the check cannot be edited by the thing being measured. It
closes the obvious cheat: making `cargo test` pass by rewriting the test.

---

## 6. Task provenance

- Every task's correct answer is established **independently and before the
  model sees it**. Class D answers are established by construction: the
  fixture and its reference solution are written together, and the reference
  is proven to satisfy the check.
- **No task is derived from model output.** A task reverse-engineered from
  what a model happened to produce measures nothing.
- Class J tasks run against a pristine `git worktree add --detach
  ~/eval-fixture 95097d5`; `assert_uncontaminated` refuses otherwise.
- Class D tasks run in a per-run generated sandbox containing only files the
  task defines, so the harness and the repository are both out of reach. A
  test asserts no task's expected answer appears in its own fixture.

---

## 7. Out of scope, deliberately

- **No tuning of the model, prompt, sampling or grammar to improve the
  result.** INFERNUM-17 established the mechanism works (emission 30/30,
  schema-valid 30/30, malformed 0.00%, fabrication 0/12). This measures what
  the model can do *through* that mechanism. If the answer is "not enough",
  that is a finding about model choice, not an invitation to tune.
- **No workload selection.** Report the number and stop.
- **No rubric change in the same pass that reports the number.** The class J
  rubric repair is INFERNUM-24 and is committed separately and earlier; every
  run is additionally scored under the old rubric so the change's effect stays
  separable from the model's.
