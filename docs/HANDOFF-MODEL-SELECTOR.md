# Handoff: Model Selector Refactor

**Date:** 2026-01-13
**Context:** Decoupling model selection from persona identity

## Completed This Session

### Wellbeing Integration (All 5 tasks done)
1. **Guardian Integration** - `nyx/nyx/nyx/agents/guardian/src/wellbeing.rs`
2. **Archon Orchestration** - `nyx/nyx/nyx/agents/archon/src/wellbeing.rs`, `orchestrator.rs`
3. **Intervention Hooks** - `infernum/crates/infernum-server/src/wellbeing_intervention.rs`
4. **Persistent State** - `infernum/crates/beleth/src/wellbeing_persist.rs`
5. **Grimoire Persona Loading** - Enhanced `beleth/src/agent.rs` with async `PersonaSource::resolve()`

All tests passing (443 in beleth).

---

## In Progress: Model Selector Refactor

### The Problem

`Persona` currently has `model: Option<ModelId>` which couples identity to engine:

```rust
// beleth/src/agent.rs lines 115-124
pub struct Persona {
    pub system: PersonaSource,
    pub model: Option<ModelId>,  // <-- REMOVE THIS
    pub max_iterations: u32,
}
```

This is wrong because:
- Persona = WHO (identity, system prompt, personality)
- Model = WHAT (runtime engine choice, resource decision)

A "code-reviewer" persona should work with any model - Opus for deep review, Haiku for linting, local Qwen for sensitive code.

### The Solution

Create `ModelSelector` enum:

```rust
/// How to select a model for agent execution.
pub enum ModelSelector {
    /// User explicitly specifies the model.
    UserSpecified(ModelId),

    /// Auto-select based on task complexity and available resources.
    AutoSelect {
        /// Minimum capability tier required.
        min_tier: ModelTier,
        /// Prefer local models if available.
        prefer_local: bool,
    },

    /// Orchestrator assigns based on task analysis.
    OrchestratorAssigned,

    /// Use whatever engine is already configured.
    EngineDefault,
}

/// Model capability tiers for auto-selection.
pub enum ModelTier {
    /// Fast, cheap - good for simple tasks (Haiku, Qwen-0.5B)
    Fast,
    /// Balanced - general purpose (Sonnet, Qwen-7B)
    Balanced,
    /// Capable - complex reasoning (Opus, Qwen-72B)
    Capable,
    /// Maximum - critical tasks, best available
    Maximum,
}
```

### Files to Modify

1. **`beleth/src/agent.rs`**
   - Remove `model` field from `Persona` struct (line ~120)
   - Remove `with_model()` from `Persona` impl
   - Update `AgentBuilder` to use `ModelSelector` instead of `.model()`
   - Add `model_selector: ModelSelector` field to `Agent`
   - Update tests (several reference `persona.model`)

2. **`beleth/src/lib.rs`**
   - Export new `ModelSelector`, `ModelTier` types

3. **Create `beleth/src/model_selector.rs`** (new file)
   - `ModelSelector` enum
   - `ModelTier` enum
   - `ModelRouter` trait for pluggable selection logic
   - `DefaultModelRouter` implementation using:
     - `TaskComplexity` (already exists in `tool.rs`)
     - Available/warm models (from malphas)
     - Wellbeing state (struggling agents might need faster models)

4. **Wire through execution in `Agent::run()`**
   - Currently uses `self.engine` directly
   - Should consult `ModelSelector` to pick engine/model

### Existing Infrastructure to Leverage

- `TaskComplexity` in `tool.rs`: `Low | Medium | High | Critical`
- `WellbeingState`: `Healthy | Cautious | Concerned | Distressed`
- `malphas` crate: Orchestration, model registry, warm model tracking
- Archon: Resource scheduling, process priority

### After Code Changes: Grimoire Audit

Search Grimoire persona files for hardcoded model references:

```bash
grep -r "model:" ~/.local/share/infernum/personas/
# Or wherever INFERNUM_GRIMOIRE_PATH points
```

Remove any `model:` lines from persona YAML/markdown files.

---

## Test Files to Update

In `beleth/src/agent.rs` tests section:
- `test_persona_with_all_fields` (line ~1148) - references `persona.model`
- `test_agent_builder_with_model` (line ~1392)
- `test_agent_builder_chain_all` (line ~1402)
- `test_persona_builder_pattern` (line ~1712)

These will need to use `ModelSelector::UserSpecified("gpt-4".into())` pattern instead.

---

## Suggested Implementation Order

1. Create `model_selector.rs` with types
2. Add to `lib.rs` exports
3. Remove `model` from `Persona`, update `Persona` impl
4. Update `AgentBuilder` to accept `ModelSelector`
5. Wire `ModelSelector` into `Agent` struct
6. Update `Agent::run()` to consult selector (or leave for future PR)
7. Fix all tests
8. Run `cargo test -p beleth`
9. Audit Grimoire files

---

## Commands to Resume

```bash
cd /home/crook/dev2/workspace/nyx/infernum

# Check current state
cargo test -p beleth agent:: 2>&1 | tail -20

# After changes
CARGO_INCREMENTAL=0 cargo test -p beleth

# Find Grimoire personas
find ~/.local/share/infernum -name "*.md" -o -name "*.yaml" | head -20
```

---

## Notes

The user's insight: "an auto-assign mode where the orchestrating agent decides the needed model for the task" - this is the `OrchestratorAssigned` variant. The orchestrator (likely in malphas) would analyze:
- Task complexity from the prompt/tools
- Available warm models
- Resource constraints (GPU memory, queue depth)
- Agent wellbeing (distressed → faster responses)

This is a resource optimization feature that builds on the wellbeing work we just completed.
