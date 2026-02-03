# Infernum Coding Enhancement Roadmap

**Version:** 1.0
**Created:** 2024-12-25
**Focus:** Agents, Models, and Code Generation Capabilities

---

## Overview

This roadmap focuses on enhancing Infernum's capabilities for code generation, understanding, and agent-assisted development. Each phase builds toward a more capable coding assistant.

---

## Phase 11: Code-Aware Generation

**Theme:** Make the model understand code structure, not just text.

### 11.1 Fill-in-the-Middle (FIM) Support ✅
- [x] **Spec**: Support FIM tokens for code completion scenarios
- [x] **Test**: Given prefix + suffix, model fills middle correctly (8 tests passing)
- [x] **Impl**:
  - Detect FIM-capable models (CodeLlama, StarCoder, DeepSeek-Coder, Qwen-Coder)
  - Handle model-specific FIM tokens (PSM and BeginHoleEnd styles)
  - Auto-format prompts for FIM with language/file hints
  - FimPromptInput variant added to PromptInput
  - GenerateRequest::fim() convenience constructor
- [x] **Files**: `abaddon/src/fim.rs` (new), `infernum-core/src/request.rs`, `abaddon/src/engine.rs`

**Completed**: 2024-12-25

### 11.2 Syntax-Guided Sampling ✅
- [x] **Spec**: Constrain generation to syntactically valid code
- [x] **Test**: Generated Python always parses, generated Rust always compiles structurally (14 tests)
- [x] **Impl**:
  - Integrate tree-sitter for incremental parsing
  - Build token mask from valid next tokens per grammar state
  - Support Python, Rust, TypeScript, JavaScript, Go
  - SyntaxGuide implements Guide trait with validity caching
  - Added sample_with_mask to Sampler for constrained decoding
  - Grammar type in create_guide now uses SyntaxGuide
- [x] **Files**: `abaddon/src/syntax_guide.rs` (new), `abaddon/src/sampler.rs`, `abaddon/src/guided.rs`

**Completed**: 2024-12-25

### 11.3 Indentation Tracker ✅
- [x] **Spec**: Maintain consistent indentation based on language/context
- [x] **Test**: Nested blocks maintain proper indent levels (17 tests passing)
- [x] **Impl**:
  - Track indent stack during generation with `IndentTracker`
  - Auto-insert appropriate whitespace after newlines
  - Respect tabs vs spaces per language convention
  - Support 16 languages including Sigil, Python, Rust, TypeScript, Go, etc.
  - Language-specific indent/dedent triggers (brackets, keywords)
  - Significant whitespace handling for Python/YAML
  - Indent style detection and normalization utilities
- [x] **Files**: `abaddon/src/indent.rs` (new), `abaddon/src/lib.rs`

**Completed**: 2024-12-26

### 11.4 Code Tokenizer Optimization ✅
- [x] **Spec**: Better token efficiency for code constructs
- [x] **Test**: Common patterns tokenize efficiently (operators, keywords) - 23 new tests
- [x] **Impl**:
  - `CodeTokenAnalyzer` - Analyzes token efficiency per language
  - `TokenEfficiency` metrics with pattern breakdown
  - `WhitespacePreserver` - Preserves significant whitespace for Python/YAML
  - `TokenMerger` - Merges common code patterns for efficiency
  - Language-specific keyword/operator definitions for 8 languages
  - Indent style detection and normalization
  - Code vs prose efficiency comparison
- [x] **Files**: `abaddon/src/tokenizer.rs`

**Completed**: 2024-12-26

---

## Phase 12: Multi-File Context

**Theme:** Understand code across file boundaries.

### 12.1 Repository Map ✅
- [x] **Spec**: Compressed representation of codebase for context window
- [x] **Test**: Map captures key structure in <2000 tokens for medium repo (16 tests)
- [x] **Impl**:
  - `RepoMap` with tree structure, exports, and symbols
  - Prioritize by: git recency, test coverage, import centrality
  - `Symbol`/`SymbolKind` for typed symbol representation
  - `SourceLanguage` detection for Rust, TypeScript, Python, Go, Java/Kotlin
  - Git integration via `load_recent_files()` for recency tracking
  - Gitignore support using `ignore` crate walker
  - Centrality analysis via import graph
  - `to_tree_string()`, `to_summary()`, `stats()` output methods
  - Configurable via `RepoMapConfig` (max_tokens, depth, patterns)
- [x] **Files**: `beleth/src/repo_map.rs` (enhanced), `beleth/src/lib.rs`

**Completed**: 2024-12-26

```
Example output:
src/
  auth/
    mod.rs: pub {login, logout, refresh_token}
    jwt.rs: pub struct JwtClaims, fn verify()
  api/
    routes.rs: 15 endpoints
    middleware.rs: pub {auth_guard, rate_limit}
```

### 12.2 Import Resolution ✅
- [x] **Spec**: Auto-include relevant imports/dependencies in context
- [x] **Test**: Editing function includes its dependencies (18 tests)
- [x] **Impl**:
  - `ImportParser` - Parse imports for Rust, TypeScript, Python, Go, Java/Kotlin
  - `ImportResolver` - Resolve imports to file paths with caching
  - `DependencyGraph` - Build and analyze import graphs
  - Transitive resolution with configurable depth limit
  - Cycle detection and shortest path finding
  - `ResolverConfig` for customization (stdlib, external, path mappings)
- [x] **Files**: `beleth/src/imports.rs` (new), `beleth/src/lib.rs`

**Completed**: 2024-12-26

### 12.3 Symbol Table ✅
- [x] **Spec**: Track definitions and usages across files
- [x] **Test**: "Find all usages of X" returns complete list (24 tests)
- [x] **Impl**:
  - `SymbolTable` - Index with qualified/simple name lookup
  - `SymbolDef` - Definition with location, visibility, docs, signature
  - `SymbolRef` - Reference tracking with RefKind (Read, Write, Call, Import, Type)
  - Support: functions, structs, enums, traits, constants, modules, macros
  - Incremental updates via file versioning
  - Symbol extraction for Rust, TypeScript, Python, Go, Java/Kotlin
  - Stats, search_prefix, public_symbols, symbols_of_type queries
- [x] **Files**: `beleth/src/symbols.rs` (new), `beleth/src/lib.rs`

**Completed**: 2024-12-26

### 12.4 Diff-Aware Context ✅
- [x] **Spec**: Focus context on regions relevant to current changes
- [x] **Test**: Large file with small edit uses minimal context (18 tests)
- [x] **Impl**:
  - `DiffParser` - Parse unified diff format with hunk extraction
  - `DiffHunk`/`DiffLine` - Represent changes with +/- tracking
  - `FileDiff` - File-level changes with ChangeType detection
  - `DiffAnalyzer` - Extract focused context regions
  - `ContextConfig` - Configurable context before/after, container inclusion
  - `ContextRegion` - Formatted output with line numbers
  - Container detection for Rust, TypeScript, Python, Go
  - Region merging for overlapping contexts
  - `DiffSummary` for change statistics
- [x] **Files**: `beleth/src/diff_context.rs` (new), `beleth/src/lib.rs`

**Completed**: 2024-12-26

---

## Phase 13: Agent Reasoning

**Theme:** Make the agent think before it codes.

### 13.1 Chain-of-Thought Scaffolding ✅
- [x] **Spec**: Structured reasoning before code generation
- [x] **Test**: Agent explains approach before implementing (6 tests)
- [x] **Impl**:
  - `ThoughtSchema` with understanding, approach, risks, steps
  - `ReasoningBlock` with confidence scoring
  - `ChainOfThought` parser and formatter
  - Markdown and plain text output formats
- [x] **Files**: `beleth/src/reasoning.rs`

**Completed**: Pre-existing

### 13.2 Self-Verification Loop ✅
- [x] **Spec**: Agent validates its own output
- [x] **Test**: Syntax errors trigger automatic retry (8 tests)
- [x] **Impl**:
  - `SelfVerifyLoop` with configurable retries
  - Rust and TypeScript error parsing
  - Fix suggestion extraction
  - Retry prompt generation with error context
- [x] **Files**: `beleth/src/self_verify.rs`

**Completed**: Pre-existing

### 13.3 Compiler Error Recovery ✅
- [x] **Spec**: Parse errors, suggest fixes, retry
- [x] **Test**: Missing semicolon gets auto-fixed (21 tests)
- [x] **Impl**:
  - `ErrorParser` for Rust, TypeScript, Python, Go, Java/Kotlin
  - `CompilerError` with code, location, source_line, suggested_fix
  - `CompilationResult` with errors_only(), warnings(), retry_context()
  - `AutoFixer` with language-specific fix suggestions
  - Auto-detection of compiler type from output
- [x] **Files**: `beleth/src/error_recovery.rs` (new), `beleth/src/lib.rs`

**Completed**: 2024-12-26

### 13.4 Multi-Turn Refinement ✅
- [x] **Spec**: Iterative improvement based on feedback
- [x] **Test**: "Make it faster" produces optimized version (20 tests)
- [x] **Impl**:
  - `RefinementSession` - Track generation history across turns
  - `Generation`/`GenerationMetadata` - Per-turn output with line/function counts
  - `RefinementRequest` - Parse natural language feedback (optimize, fix, extend, etc.)
  - `RefinementConfig` - Max turns, diff inclusion, history tracking
  - Diff-based refinement prompts with change visualization
  - Language detection for Rust, TypeScript, Python, Go
  - Preserve verified working parts during iteration
  - `RefinementSummary` for session statistics
- [x] **Files**: `beleth/src/refinement.rs` (new), `beleth/src/lib.rs`

**Completed**: 2024-12-26

---

## Phase 14: Code Quality Tools

**Theme:** Generate not just code, but quality code.

### 14.1 Test Generation ✅
- [x] **Spec**: Generate tests alongside implementation
- [x] **Test**: Generated tests achieve >80% coverage of new code (26 tests)
- [x] **Impl**:
  - `TestGenerator` - Parse function signatures and generate tests
  - `FunctionInfo`/`FunctionParam` - Function metadata extraction
  - `TestCase`/`TestCategory` - Happy path, edge cases, error cases
  - `TestFramework` - Support pytest, vitest, jest, cargo test, go test, junit
  - Function parsing for Rust, TypeScript, Python, Go
  - Smart test generation based on parameter types
  - Multiple output formats with async/panic support
- [x] **Files**: `beleth/src/test_gen.rs` (new), `beleth/src/lib.rs`

**Completed**: 2024-12-26

### 14.2 Docstring Generation ✅
- [x] **Spec**: Auto-document functions based on implementation
- [x] **Test**: Generated docs match actual behavior (22 tests)
- [x] **Impl**:
  - `DocGenerator` - Generate documentation from function info
  - `FunctionDoc` - Complete doc with summary, params, returns, exceptions
  - `DocFormat` - Support rustdoc, JSDoc, docstring, GoDoc, KDoc, Javadoc
  - Smart summary generation from function names
  - Example code generation per language
  - Exception/error inference from return types
- [x] **Files**: `beleth/src/doc_gen.rs` (new), `beleth/src/lib.rs`

**Completed**: 2024-12-26

### 14.3 Linter Integration ✅
- [x] **Spec**: Apply style fixes during/after generation
- [x] **Test**: Generated code passes project linter (18 tests)
- [x] **Impl**:
  - `Linter` - Run linters and collect issues
  - `LinterTool` - Support rustfmt, clippy, eslint, prettier, black, ruff, gofmt
  - `LintIssue`/`LintFix` - Issues with auto-fix suggestions
  - `LintConfig` - Configurable rules, severity filtering
  - Format-specific linting (trailing whitespace, indent, var usage, etc.)
  - Auto-apply safe fixes with `apply_fixes`
- [x] **Files**: `beleth/src/lint.rs` (new), `beleth/src/lib.rs`

**Completed**: 2024-12-26

### 14.4 Type Inference Suggestions ✅
- [x] **Spec**: Suggest types for dynamic language code
- [x] **Test**: Python function gets accurate type hints (17 tests)
- [x] **Impl**:
  - `TypeInferrer` - Analyze code and infer types
  - `InferredType` - Complete type system (primitives, generics, optional, union)
  - `TypeSuggestion` - Suggestions with confidence scores
  - `InferenceReason` - Track why types were inferred
  - Python and TypeScript support
  - Name-based, literal-based, method-call inference
  - Format as Python type hints or TypeScript annotations
- [x] **Files**: `beleth/src/type_infer.rs` (new), `beleth/src/lib.rs`

**Completed**: 2024-12-26

---

## Phase 15: Specialized Model Support

**Theme:** Optimize for different code models.

### 15.1 Model-Specific Prompting ✅
- [x] **Spec**: Auto-detect model and use optimal prompt format
- [x] **Test**: Each supported model produces quality output (21 tests)
- [x] **Impl**:
  - `ModelFamily` enum: CodeLlama, DeepSeek-Coder, StarCoder, Qwen-Coder, CodeGemma, Codestral
  - `ModelVariant` enum: Base, Instruct, Chat, FIM
  - `SpecialTokens` with model-specific tokens (BOS, EOS, user/assistant/system, FIM)
  - `SystemPrompts` presets for different tasks (completion, debugging, review)
  - `ModelProfile` with `format_prompt()` for each model family
  - `detect_model()` auto-detection from model name
  - `ModelRegistry` for managing known profiles with defaults
  - Context length defaults per family (Qwen: 128k, Codestral: 32k, etc.)
- [x] **Files**: `abaddon/src/model_profiles.rs` (new), `abaddon/src/lib.rs`

**Completed**: 2024-12-26

### 15.2 Code LoRA Adapters ✅
- [x] **Spec**: Fine-tuned adapters for specific codebases/styles
- [x] **Test**: Adapter produces project-consistent code (18 tests)
- [x] **Impl**:
  - `CodeLoraConfig` for code-specific LoRA training
  - `CodeLanguage` enum with 9 languages + extension detection
  - `StyleCategory` enum for pattern categories (naming, formatting, errors, etc.)
  - `CodeFile` with quality scoring and pattern extraction
  - `ProjectStyleAdapter` for project-specific style learning
  - `CodeAdapterRegistry` for adapter management with language lookup
  - `AdapterBlender` with blend strategies (average, primary, language-aware)
  - Error pattern detection for Rust, TypeScript, Python, Go
  - Import pattern detection (grouped, sorted)
- [x] **Files**: `asmodeus/src/code_lora.rs` (new), `asmodeus/src/lib.rs`

**Completed**: 2024-12-26

### 15.3 Language-Specific Decoding ✅
- [x] **Spec**: Optimize sampling parameters per language
- [x] **Test**: Python vs Rust use different temperature/top_p (21 tests)
- [x] **Impl**:
  - `SamplingPreset` with temperature, top_p, top_k, penalties, stop sequences
  - 16 language presets: Rust, TypeScript, JavaScript, Python, Go, Java, Kotlin, C++, SQL, Shell, Markdown, JSON, YAML, HTML, CSS, Sigil
  - `CodeTask` enum with temperature modifiers per task type
  - `LanguageSampler` with extension mapping and content detection
  - `DetectedLanguage` with confidence and detection method
  - `blend_presets()` for weighted preset combination
  - Task-aware adjustments (debugging = lower temp, docs = higher temp)
  - Shebang detection for Python, Bash, Node
- [x] **Files**: `abaddon/src/lang_sampling.rs` (new), `abaddon/src/lib.rs`

**Completed**: 2024-12-26

---

## Phase 16: Advanced Code Understanding

**Theme:** Deep comprehension of code semantics.

### 16.1 Control Flow Analysis ✅
- [x] **Spec**: Understand execution paths through code
- [x] **Test**: Identify unreachable code, infinite loops (22 tests)
- [x] **Impl**:
  - `ControlFlowGraph` with nodes, edges, entry/exit tracking
  - `CfgNode`/`NodeKind` for 16 control flow constructs
  - `CfgEdge`/`EdgeKind` for sequential, branch, loop back, exception flows
  - `CfgBuilder` with language-aware parsing (Rust, TS, Python, Go)
  - `FunctionExtractor` for extracting function bodies
  - `compute_reachability()` via BFS from entry
  - `find_unreachable()`, `find_infinite_loops()`, `find_dead_code()`
  - `cyclomatic_complexity()` calculation
  - DOT format export for visualization
- [x] **Files**: `beleth/src/cfg.rs` (new), `beleth/src/lib.rs`

**Completed**: 2024-12-26

### 16.2 Data Flow Tracking ✅
- [x] **Spec**: Track variable definitions and uses
- [x] **Test**: Identify unused variables, uninitialized reads (19 tests)
- [x] **Impl**:
  - `Definition` with name, location, mutability, type, kind, scope
  - `Use` with location and `UseKind` (Read, Write, ReadWrite, Argument, Return, Borrow)
  - `DefUseChain` linking definitions to all their uses
  - `DataFlowAnalyzer` for Rust, TypeScript, JavaScript, Python, Go
  - `DataFlowResult` with chains, uninitialized_reads, shadowing detection
  - `find_unused_variables()`, `find_write_only_variables()`, `find_unused_mut()`
  - Scope tracking for nested blocks
  - Underscore-prefix convention handling
- [x] **Files**: `beleth/src/dataflow.rs` (new), `beleth/src/lib.rs`

**Completed**: 2024-12-26

### 16.3 Semantic Code Search
- [ ] **Spec**: Find code by meaning, not just text
- [ ] **Test**: "Authentication logic" finds login code
- [ ] **Impl**:
  - Embed code chunks with code-specific model
  - Store in Stolas vector DB
  - Hybrid search: semantic + keyword
- [ ] **Files**: `stolas/src/code_search.rs` (new)

### 16.4 Code Explanation ✅
- [x] **Spec**: Generate natural language explanations of code
- [x] **Test**: Explanation accurately describes behavior (17 tests)
- [x] **Impl**:
  - `CodeExplainer` with language-aware analysis (Rust, TS, Python, Go)
  - `Explanation` with summary, purpose, concepts, complexity, line explanations
  - `LineExplanation` for detailed per-line breakdown
  - `ExplanationLevel` (Brief, Normal, Detailed) for depth control
  - `ComplexityAssessment` with cyclomatic, cognitive, time/space complexity
  - Pattern detection: recursion, loops, iterators, closures, guard clauses, pattern matching
  - Code type detection: function, constructor, getter/setter, test, loop, conditional
  - Time complexity estimation from loop patterns (O(1), O(n), O(n²), O(log n))
  - `to_markdown()` and `to_text()` output formatters
- [x] **Files**: `beleth/src/explain.rs` (new), `beleth/src/lib.rs`

**Completed**: 2024-12-26

---

## Phase 17: Refactoring Intelligence

**Theme:** Automated code improvement and transformation.

### 17.1 Extract Method Detection ✅
- [x] **Spec**: Identify code blocks that should be separate functions
- [x] **Test**: Long function with repeated logic suggests extraction (24 tests)
- [x] **Impl**:
  - `ExtractMethodAnalyzer` with language-aware analysis (Rust, TS, Python, Go)
  - `ExtractionReason` (RepeatedPattern, HighComplexity, DeepNesting, LongBlock)
  - `ExtractionSuggestion` with code block, boundaries, and suggested signature
  - `SuggestedSignature` with language-specific code generation
  - Pattern normalization for duplicate detection
  - Cyclomatic complexity calculation
- [x] **Files**: `beleth/src/refactor/extract.rs` (new), `beleth/src/refactor/mod.rs`

**Completed**: 2024-12-26

### 17.2 Rename Propagation ✅
- [x] **Spec**: Safely rename symbols across entire codebase
- [x] **Test**: Rename function updates all call sites (26 tests)
- [x] **Impl**:
  - `RenameEngine` with language-aware symbol detection
  - `SymbolOccurrence` tracking with definition, import, export, scope info
  - `RenameRequest`/`RenameResult` for computing renames
  - `FileChanges`/`TextEdit` for safe edit application
  - Shadowing detection and warning
  - Naming convention validation (snake_case, PascalCase, etc.)
  - Reserved keyword checking
  - Preview generation before applying
- [x] **Files**: `beleth/src/refactor/rename.rs` (new)

**Completed**: 2024-12-26

### 17.3 Dead Code Elimination ✅
- [x] **Spec**: Find and remove unused code
- [x] **Test**: Unused function flagged for removal (19 tests)
- [x] **Impl**:
  - `DeadCodeAnalyzer` with entry point detection
  - `CodeItem` tracking for functions, types, variables, imports
  - `DeadItem` with `DeadReason` (NeverUsed, UnusedImport, WriteOnly, etc.)
  - Reachability analysis from entry points
  - Unused import detection
  - `AnalysisResult` with summary and removal suggestions
  - Configurable via `AnalysisConfig`
- [x] **Files**: `beleth/src/refactor/dead_code.rs` (new)

**Completed**: 2024-12-26

### 17.4 Dependency Decoupling ✅
- [x] **Spec**: Suggest dependency injection for tightly coupled code
- [x] **Test**: Hard-coded dependency replaced with interface (19 tests)
- [x] **Impl**:
  - `DependencyAnalyzer` for coupling detection
  - `CouplingKind` (ConcreteInstantiation, GlobalStateAccess, HardcodedConfig, ServiceLocator)
  - `CouplingIssue` with location, context, and suggested strategy
  - `DecouplingStrategy` (ExtractInterface, ConstructorInjection, ConfigInjection, DIContainer)
  - `RefactoringPlan` with original/refactored code and interface generation
  - Hardcoded URL/port/credential detection
  - Test code skipping
- [x] **Files**: `beleth/src/refactor/decouple.rs` (new)

**Completed**: 2024-12-26

---

## Phase 18: Project-Aware Generation

**Theme:** Generate code that fits the existing codebase.

### 18.1 Style Mimicry
- [ ] **Spec**: Match existing code patterns in the project
- [ ] **Test**: Generated code indistinguishable from existing style
- [ ] **Impl**:
  - Analyze: formatting, naming, structure patterns
  - Build style profile from project samples
  - Apply during generation
- [ ] **Files**: `beleth/src/style/profile.rs` (new)

### 18.2 Naming Convention Detection
- [ ] **Spec**: Detect and follow project naming conventions
- [ ] **Test**: New function uses project's naming pattern
- [ ] **Impl**:
  - Detect: camelCase vs snake_case, prefixes, suffixes
  - Per-category conventions (functions, types, constants)
  - Enforce during generation
- [ ] **Files**: `beleth/src/style/naming.rs` (new)

### 18.3 Error Handling Patterns
- [ ] **Spec**: Use project's established error patterns
- [ ] **Test**: Generated error handling matches project style
- [ ] **Impl**:
  - Detect: Result vs exceptions, custom error types
  - Learn error message format
  - Apply consistent patterns
- [ ] **Files**: `beleth/src/style/errors.rs` (new)

### 18.4 Import Organization
- [ ] **Spec**: Match project's import style and ordering
- [ ] **Test**: Generated imports follow project conventions
- [ ] **Impl**:
  - Detect: grouping, ordering, aliasing patterns
  - Apply formatting rules
  - Organize after generation
- [ ] **Files**: `beleth/src/style/imports.rs` (new)

---

## Phase 19: Code Review Assistance

**Theme:** AI-powered code review and analysis.

### 19.1 PR Summarization
- [ ] **Spec**: Generate clear summary of what changed and why
- [ ] **Test**: Summary captures intent, not just diff
- [ ] **Impl**:
  - Analyze commit messages and diff
  - Group changes by purpose
  - Generate structured summary
- [ ] **Files**: `beleth/src/review/summarize.rs` (new)

### 19.2 Risk Assessment
- [ ] **Spec**: Highlight risky changes requiring careful review
- [ ] **Test**: Security-sensitive changes flagged appropriately
- [ ] **Impl**:
  - Detect: auth changes, data handling, external calls
  - Score risk by category
  - Prioritize review attention
- [ ] **Files**: `beleth/src/review/risk.rs` (new)

### 19.3 Missing Test Detection
- [ ] **Spec**: Identify code changes that need test coverage
- [ ] **Test**: New function without tests flagged
- [ ] **Impl**:
  - Map changes to test files
  - Detect untested paths
  - Suggest specific test cases
- [ ] **Files**: `beleth/src/review/coverage.rs` (new)

### 19.4 Security Scan
- [ ] **Spec**: Identify potential security vulnerabilities
- [ ] **Test**: SQL injection pattern detected
- [ ] **Impl**:
  - Pattern matching for common vulnerabilities
  - Taint analysis for user input
  - OWASP top 10 checks
- [ ] **Files**: `beleth/src/review/security.rs` (new)

---

## Phase 20: Tool Affinity & Optimal Tool Selection

**Theme:** Make agents naturally reach for the right tools.

### Problem Statement

Agents often fail to use optimal tools for tasks:
- Use `grep` instead of `rg` (ripgrep)
- Use `find` instead of `fd`
- Don't run linters after generating code
- Don't verify with compilers before claiming done
- Prefer verbose manual approaches over specialized tools
- Forget to use project-specific scripts/aliases

### 20.1 Tool Capability Registry
- [ ] **Spec**: Maintain registry of available tools and their strengths
- [ ] **Test**: Agent selects `rg` over `grep` for code search
- [ ] **Impl**:
  - Discover available tools at session start
  - Map task types to optimal tools
  - Rank by: speed, accuracy, features
- [ ] **Files**: `beleth/src/tools/registry.rs` (new)

```rust
// Example registry structure
ToolRegistry {
    search: [
        Tool { name: "rg", score: 100, features: [regex, gitignore, fast] },
        Tool { name: "grep", score: 50, features: [regex, ubiquitous] },
    ],
    find: [
        Tool { name: "fd", score: 100, features: [fast, gitignore, regex] },
        Tool { name: "find", score: 50, features: [ubiquitous, powerful] },
    ],
    lint: [
        Tool { name: "clippy", lang: "rust", score: 100 },
        Tool { name: "eslint", lang: "typescript", score: 100 },
    ],
}
```

### 20.2 Post-Action Verification Hooks
- [ ] **Spec**: Automatically verify outputs with appropriate tools
- [ ] **Test**: Generated Rust code runs `cargo check` before completion
- [ ] **Impl**:
  - Define verification hooks per action type
  - Auto-run after: code generation, file modification
  - Retry on failure with error context
- [ ] **Files**: `beleth/src/tools/verify_hooks.rs` (new)

```rust
// Verification chain
AfterCodeGen -> [
    SyntaxCheck(tree_sitter),
    TypeCheck(cargo_check | tsc),
    LintCheck(clippy | eslint),
    FormatCheck(rustfmt | prettier),
]
```

### 20.3 Project Tool Discovery
- [ ] **Spec**: Discover and prefer project-specific tooling
- [ ] **Test**: Uses project's `./scripts/lint.sh` over generic linter
- [ ] **Impl**:
  - Scan for: Makefile, package.json scripts, shell aliases
  - Parse tool configurations (.eslintrc, clippy.toml)
  - Prefer project tools over system defaults
- [ ] **Files**: `beleth/src/tools/discovery.rs` (new)

### 20.4 Tool Usage Prompting
- [ ] **Spec**: Inject tool awareness into agent prompts
- [ ] **Test**: Agent mentions available tools in reasoning
- [ ] **Impl**:
  - Include tool summary in system prompt
  - Remind of verification tools after generation
  - Suggest optimal tool for detected task
- [ ] **Files**: `beleth/src/tools/prompting.rs` (new)

### 20.5 Efficiency Scoring
- [ ] **Spec**: Score and improve tool selection over time
- [ ] **Test**: Repeated suboptimal choices trigger correction
- [ ] **Impl**:
  - Track tool usage patterns
  - Measure: time saved, accuracy improved
  - Feedback loop for better selection
- [ ] **Files**: `beleth/src/tools/scoring.rs` (new)

### 20.6 Mandatory Verification Gates
- [ ] **Spec**: Code cannot be marked "done" until verification passes
- [ ] **Test**: Incomplete verification blocks task completion
- [ ] **Impl**:
  - Define completion criteria per task type
  - Hard gates, not suggestions
  - Build into action schema itself
- [ ] **Files**: `beleth/src/tools/gates.rs` (new)

```rust
// Task completion requires all gates to pass
TaskCompletion {
    write_code: [syntax_valid, types_check, lint_clean],
    modify_file: [file_exists, syntax_valid, tests_pass],
    refactor: [behavior_preserved, tests_pass, no_regressions],
}
```

### 20.7 Tool Substitution Layer
- [ ] **Spec**: Transparently upgrade suboptimal tool calls
- [ ] **Test**: `grep` call auto-upgrades to `rg`
- [ ] **Impl**:
  - Intercept tool invocations
  - Map to optimal equivalent
  - Preserve semantics, improve performance
- [ ] **Files**: `beleth/src/tools/substitution.rs` (new)

### 20.8 Negative Example Injection
- [ ] **Spec**: Show consequences of skipping verification
- [ ] **Test**: Agent avoids past mistakes after seeing examples
- [ ] **Impl**:
  - Collect failure cases from skipped verification
  - Inject relevant examples into context
  - "Last time you skipped cargo check, this happened..."
- [ ] **Files**: `beleth/src/tools/negative_examples.rs` (new)

### 20.9 Task Completion Criteria
- [ ] **Spec**: Explicit requirements per task type
- [ ] **Test**: "Write function" requires code + tests + lint
- [ ] **Impl**:
  - Define criteria templates
  - Auto-expand task into subtasks
  - Track completion of each criterion
- [ ] **Files**: `beleth/src/tools/completion.rs` (new)

```rust
// "Write a function" expands to:
TaskCriteria::WriteFunction {
    required: [
        "Implementation compiles",
        "Unit tests written",
        "Lint passes",
        "Types checked",
    ],
    optional: [
        "Documentation added",
        "Integration test",
    ],
}
```

### 20.10 Tool Usage Metrics Dashboard
- [ ] **Spec**: Surface tool efficiency to agent
- [ ] **Test**: Metrics influence future tool selection
- [ ] **Impl**:
  - Track: tool used, time taken, success rate
  - Compare optimal vs actual
  - Present stats in agent context
- [ ] **Files**: `beleth/src/tools/metrics.rs` (new)

---

## Phase 21: Cross-Language Understanding

**Theme:** Work seamlessly across language boundaries.

### 21.1 Polyglot Context
- [ ] **Spec**: Understand interactions between languages
- [ ] **Test**: JS calling Rust via WASM understood correctly
- [ ] **Impl**:
  - Map FFI boundaries
  - Track type conversions
  - Unified symbol table across languages
- [ ] **Files**: `beleth/src/polyglot/context.rs` (new)

### 21.2 API Contract Verification
- [ ] **Spec**: Ensure client matches server types
- [ ] **Test**: TypeScript client matches Rust API types
- [ ] **Impl**:
  - Extract API schemas (OpenAPI, GraphQL, protobuf)
  - Compare client/server type definitions
  - Flag mismatches
- [ ] **Files**: `beleth/src/polyglot/contracts.rs` (new)

### 21.3 Migration Assistance
- [ ] **Spec**: Help port code between languages
- [ ] **Test**: Python function correctly ported to Rust
- [ ] **Impl**:
  - Analyze source language constructs
  - Map to target language idioms
  - Handle: ownership, async, error handling differences
- [ ] **Files**: `beleth/src/polyglot/migrate.rs` (new)

---

## Phase 22: Eidolon IDE Integration

**Theme:** Leverage Eidolon's powerful IDE features for agent-assisted development.

### Motivation

Eidolon provides 50+ built-in features that agents underutilize:
- Time-travel debugging instead of guessing at bugs
- AI code review with CWE/CVSS scoring
- Real database/HTTP clients instead of generating untested code
- LSP integration for accurate go-to-definition
- Performance profiler for actual bottleneck detection
- Git timeline visualization

### 22.1 Eidolon Agent SDK Integration
- [ ] **Spec**: Connect agents to Eidolon via gRPC (port 50051)
- [ ] **Test**: Agent can discover and invoke Eidolon tools
- [ ] **Impl**:
  - Use `eidolon-agent-sdk` Tools Client
  - Discover available IDE capabilities
  - Map agent actions to IDE tools
- [ ] **Files**: `beleth/src/eidolon/client.rs` (new)

```rust
// Agent connects to Eidolon
let session = EidolonSession::connect("localhost:50051").await?;
let tools = session.tools().discover().await?;
// Tools: file_read, file_write, git_status, lsp_goto_def, run_debugger, ...
```

### 22.2 Evidentiality-Aware Reasoning
- [ ] **Spec**: Use Eidolon's evidence markers for confidence tracking
- [ ] **Test**: Agent distinguishes `Known!` from `Uncertain?` data
- [ ] **Impl**:
  - Parse evidentiality markers from tool results
  - Adjust reasoning based on confidence
  - Request verification for uncertain data
- [ ] **Files**: `beleth/src/eidolon/evidentiality.rs` (new)

```rust
// Eidolon Native Tool Protocol markers
Known!      // Certain information (verified by tool)
Uncertain?  // Probabilistic/heuristic data
Reported~   // Third-party reported (e.g., from LLM)
Paradox‽    // Contradictory information detected
```

### 22.3 Debug-First Problem Solving
- [ ] **Spec**: Agent uses time-travel debugger when stuck
- [ ] **Test**: Bug investigation uses debugger before guessing
- [ ] **Impl**:
  - Detect "stuck" state (multiple failed attempts)
  - Launch Eidolon debugger session
  - Step through execution to find issue
  - Capture execution state for analysis
- [ ] **Files**: `beleth/src/eidolon/debug.rs` (new)

```rust
// When agent encounters bug
if attempts > 2 && !resolved {
    let debugger = session.debugger().attach(process).await?;
    debugger.set_breakpoint(suspected_location)?;
    let state = debugger.reverse_continue().await?;
    // Analyze state.variables, state.call_stack
}
```

### 22.4 LSP-Powered Code Intelligence
- [ ] **Spec**: Use Eidolon's LSP for accurate code navigation
- [ ] **Test**: Go-to-definition returns correct location
- [ ] **Impl**:
  - Query LSP through Eidolon instead of grep
  - Get accurate: definitions, references, hover info
  - Use for rename, find usages, type info
- [ ] **Files**: `beleth/src/eidolon/lsp.rs` (new)

### 22.5 Real Database Execution
- [ ] **Spec**: Execute queries via Eidolon's database client
- [ ] **Test**: Generated SQL verified by actual execution
- [ ] **Impl**:
  - Connect to project database through Eidolon
  - Validate queries before suggesting
  - Show actual results, not assumed output
- [ ] **Files**: `beleth/src/eidolon/database.rs` (new)

### 22.6 AI Code Review Integration
- [ ] **Spec**: Use Eidolon's security scanner on generated code
- [ ] **Test**: Vulnerabilities detected before completion
- [ ] **Impl**:
  - Run Eidolon AI review on new/modified code
  - Get CWE classification, CVSS scores
  - Auto-fix or flag for review
- [ ] **Files**: `beleth/src/eidolon/review.rs` (new)

### 22.7 Performance Profiling
- [ ] **Spec**: Use Eidolon profiler for optimization tasks
- [ ] **Test**: Performance issues identified with data, not guesses
- [ ] **Impl**:
  - Run profiler on suspect code
  - Get flame graphs, hot paths
  - Target optimization at measured bottlenecks
- [ ] **Files**: `beleth/src/eidolon/profiler.rs` (new)

### 22.8 Event-Driven Agent Actions
- [ ] **Spec**: Subscribe to Eidolon events for reactive behavior
- [ ] **Test**: Agent responds to file save with verification
- [ ] **Impl**:
  - Subscribe to: file changes, build results, test failures
  - Auto-trigger verification on relevant events
  - Proactive error detection
- [ ] **Files**: `beleth/src/eidolon/events.rs` (new)

```rust
// Subscribe to workspace events
session.events().subscribe(|event| {
    match event {
        FileSaved(path) => verify_syntax(path),
        BuildFailed(errors) => analyze_errors(errors),
        TestFailed(test) => investigate_failure(test),
    }
});
```

### 22.9 HTTP Client for API Testing
- [ ] **Spec**: Validate API calls through Eidolon's HTTP client
- [ ] **Test**: Generated API code tested against real endpoints
- [ ] **Impl**:
  - Execute HTTP requests through Eidolon
  - Verify response schemas
  - Capture request/response for documentation
- [ ] **Files**: `beleth/src/eidolon/http.rs` (new)

### 22.10 Git Integration for Safe Operations
- [ ] **Spec**: Use Eidolon's git tools for version control
- [ ] **Test**: Refactoring uses branch isolation
- [ ] **Impl**:
  - Create branches for risky changes
  - Visual diff before commit
  - Interactive rebase through IDE
- [ ] **Files**: `beleth/src/eidolon/git.rs` (new)

---

## Appendix: Priority Matrix

| Phase | Impact | Effort | Priority |
|-------|--------|--------|----------|
| 11.1 FIM Support | High | Medium | P0 |
| 11.2 Syntax Guided | High | High | P1 |
| 12.1 Repo Map | High | Medium | P0 |
| 12.3 Symbol Table | High | High | P1 |
| 13.1 CoT Scaffold | Medium | Low | P0 |
| 13.2 Self-Verify | High | Medium | P0 |
| 14.1 Test Gen | High | Medium | P1 |
| 15.1 Model Profiles | Medium | Low | P0 |
| 16.3 Semantic Search | High | Medium | P1 |
| 17.1 Extract Method | Medium | Medium | P1 |
| 18.1 Style Mimicry | High | Medium | P0 |
| 19.2 Risk Assessment | High | Medium | P1 |
| 20.1 Tool Registry | High | Low | P0 |
| 20.2 Verify Hooks | High | Medium | P0 |
| 20.3 Project Discovery | Medium | Low | P0 |
| 21.2 Contract Verify | High | High | P2 |
| 22.1 Eidolon SDK | High | Medium | P0 |
| 22.3 Debug-First | High | Medium | P0 |
| 22.4 LSP Integration | High | Low | P0 |
| 22.6 AI Code Review | High | Low | P0 |
| 22.8 Event-Driven | Medium | Medium | P1 |

---

## Appendix: Dogfooding Results

### Phase 22 Validation (2024-12-25)

Successfully demonstrated Eidolon Agent SDK integration:

**Environment:**
- Eidolon IDE Server running on port 50051
- Workspace: `/home/user/workspace`
- Branch: `claude/infernum-investigation-nHmRz`

**Demonstrated Capabilities:**
1. **Agent Connection** - Connected via gRPC with agent ID registration
2. **Capability Registration** - Registered file.read, file.write, terminal.execute, code.analyze
3. **File Reading** - Used `read_file` tool to analyze Infernum lib.rs
4. **Git Status** - Used `git_status` tool for repository information

**Infernum Crate Analysis Results:**
```
📊 Module Statistics:
   ├─ Total modules declared: 39
   ├─ Public exports: 40
   ├─ Resilience modules: 11
   └─ Feature-gated modules: 1

🔧 Resilience Modules Found:
   • backpressure, bulkhead, circuit, concurrency, dynamic_batch
   • health, metrics, ratelimit, retry, shedding, timeout

📊 Code Quality Metrics (lib.rs):
   ├─ Lines in lib.rs: 313
   ├─ unwrap() calls: 0
   ├─ expect() calls: 0
   └─ Error handling ratio: 100%
```

**Safety Features Observed:**
- Command execution requires user confirmation (safety gate)
- Tool invocation returns structured results with status tracking
- Event-based architecture for reactive patterns

**Example Agent Created:**
- `/nyx/eidolon/crates/eidolon-agent-sdk/examples/infernum_dev_agent.rs`
- Demonstrates practical use of Agent SDK for development analysis

**Key Learnings:**
1. Agent SDK convenience methods (`read_file`, `git_status`) work seamlessly
2. Confirmation gates protect against unintended command execution
3. Event subscriptions enable reactive development workflows
4. Tool discovery provides inventory of available capabilities

### Phase 11.1 FIM Implementation (2024-12-25)

Used Eidolon HTTP Gateway for development of FIM support:

**Environment:**
- Eidolon IDE Server: port 50051 (gRPC)
- Eidolon Gateway: port 9000 (HTTP)
- MCP Server: @daemoniorum/mcp-nyx

**Tools Used via Gateway:**
1. `read_file` - Analyzed sampler.rs, tokenizer.rs, engine.rs, request.rs
2. `grep` - Searched for existing FIM patterns (found only roadmap references)
3. Health endpoints for service verification

**Pain Points Discovered:**
1. **grep tool returns empty** - Eidolon's grep tool didn't return results for some patterns; had to fall back to native Grep tool
2. **No write_file feedback** - Created files using native Write tool since gateway response was more predictable
3. **Missing edit_file in MCP** - Had to use native Edit tool (though edit_file exists in ide-server now)

**What Worked Well:**
1. HTTP Gateway proxy is stable and responsive
2. Tool invocation streaming works correctly
3. read_file returns full file content reliably
4. Health checks enable quick service verification

**Improvement Opportunities:**
1. Add proper result content extraction in Eidolon grep implementation
2. Consider adding output_mode parameter to grep for content vs files
3. MCP server needs to expose full tool set from gateway
4. Add file operation feedback (bytes written, lines changed)

**Implementation Completed:**
- `abaddon/src/fim.rs` - 503 lines, 8 tests
- `infernum-core/src/request.rs` - FimPromptInput type
- `abaddon/src/engine.rs` - FIM prompt processing

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-12-25 | Initial coding roadmap |
| 1.1 | 2024-12-25 | Add phases 17-21: refactoring, project-aware, review, tools, polyglot |
| 1.2 | 2024-12-25 | Add Phase 22: Eidolon IDE integration for agent-assisted development |
| 1.3 | 2024-12-25 | Add dogfooding results documenting Phase 22 validation |
| 1.4 | 2024-12-25 | Complete Phase 11.1 FIM support, add dogfooding results |
| 1.5 | 2024-12-25 | Complete Phase 11.2 Syntax-Guided Sampling with tree-sitter (15 languages) |
| 1.6 | 2024-12-26 | Add Sigil language support to syntax-guided generation |
| 1.7 | 2024-12-26 | Complete Phase 11.3 Indentation Tracker (16 languages, 17 tests) |
| 1.8 | 2024-12-26 | Complete Phase 11.4 Code Tokenizer Optimization (23 new tests) |
| 1.9 | 2024-12-26 | Complete Phase 12.1 Repository Map (16 tests, 5 languages, git integration) |
| 2.0 | 2024-12-26 | Complete Phase 12.2 Import Resolution (18 tests, 5 languages, dependency graph) |
| 2.1 | 2024-12-26 | Complete Phase 12.3 Symbol Table (24 tests, 5 languages, incremental updates) |
| 2.2 | 2024-12-26 | Complete Phase 12.4 Diff-Aware Context (18 tests, diff parsing, context extraction) |
| 2.3 | 2024-12-26 | Mark 13.1-13.2 complete (pre-existing), implement 13.3 Error Recovery (21 tests) |
| 2.4 | 2024-12-26 | Complete Phase 13.4 Multi-Turn Refinement (20 tests) |
| 2.5 | 2024-12-26 | Complete Phase 14 Code Quality Tools (14.1-14.4: test gen, doc gen, lint, type infer - 83 tests) |
| 2.6 | 2024-12-26 | Complete Phase 15 Specialized Model Support (15.1-15.3: model profiles, code LoRA, lang sampling - 60 tests) |
| 2.7 | 2024-12-26 | Complete Phase 16 Advanced Code Understanding (16.1 CFG, 16.2 Data Flow, 16.4 Explain - 58 tests, 16.3 pending Stolas) |
| 2.8 | 2024-12-26 | Complete Phase 17 Refactoring Intelligence (17.1-17.4: extract, rename, dead code, decouple - 88 tests) |
