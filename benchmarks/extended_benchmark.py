#!/usr/bin/env python3
"""
Infernum Framework Extended Benchmark Suite
Comprehensive comparison of Rust and Sigil implementations including:
- Static code metrics (LOC, file counts, complexity)
- Morpheme pipe usage analysis
- Evidentiality type tracking
- Complexity metrics (nesting depth, cyclomatic complexity)
- A/B testing module (Astaroth) analysis
- Performance simulation benchmarks
- ASCII visualizations
"""

import os
import subprocess
import time
import json
import re
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime

# Paths
RUST_ROOT = Path("/tmp/infernum-framework")
SIGIL_ROOT = Path("/home/user/infernum-complete/infernum-sigil")
RESULTS_DIR = Path("/home/user/infernum-complete/benchmarks/results")

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MorphemeUsage:
    """Tracks Sigil morpheme pipe usage."""
    tau: int = 0          # |τ or |tau - map/transform
    phi: int = 0          # |φ or |phi - filter
    sigma: int = 0        # |σ or |sigma - unwrap_or
    rho: int = 0          # |ρ or |rho - reduce/fold
    await_pipe: int = 0   # |await
    map_err: int = 0      # |map_err
    collect: int = 0      # |collect
    coalesce: int = 0     # ?? operator

@dataclass
class EvidentialityUsage:
    """Tracks Sigil evidentiality type usage."""
    untrusted: int = 0    # ~ suffix
    verified: int = 0     # ! suffix
    inferential: int = 0  # ^ suffix

@dataclass
class ComplexityMetrics:
    """Code complexity metrics."""
    max_nesting_depth: int = 0
    avg_nesting_depth: float = 0.0
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    match_arms: int = 0
    error_handling_points: int = 0

@dataclass
class FileMetrics:
    """Extended metrics for a single file."""
    path: str
    total_lines: int
    code_lines: int
    blank_lines: int
    comment_lines: int
    function_count: int
    struct_count: int
    enum_count: int
    impl_count: int
    trait_count: int = 0
    async_fn_count: int = 0
    test_count: int = 0
    morpheme_usage: MorphemeUsage = field(default_factory=MorphemeUsage)
    evidentiality_usage: EvidentialityUsage = field(default_factory=EvidentialityUsage)
    complexity: ComplexityMetrics = field(default_factory=ComplexityMetrics)

@dataclass
class ModuleMetrics:
    """Aggregated metrics for a module/crate."""
    name: str
    description: str = ""
    files: List[FileMetrics] = field(default_factory=list)

    @property
    def total_lines(self) -> int:
        return sum(f.total_lines for f in self.files)

    @property
    def code_lines(self) -> int:
        return sum(f.code_lines for f in self.files)

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def function_count(self) -> int:
        return sum(f.function_count for f in self.files)

    @property
    def struct_count(self) -> int:
        return sum(f.struct_count for f in self.files)

    @property
    def total_morpheme_usage(self) -> int:
        return sum(
            f.morpheme_usage.tau + f.morpheme_usage.phi + f.morpheme_usage.sigma +
            f.morpheme_usage.rho + f.morpheme_usage.await_pipe + f.morpheme_usage.map_err +
            f.morpheme_usage.collect + f.morpheme_usage.coalesce
            for f in self.files
        )

    @property
    def avg_complexity(self) -> float:
        if not self.files:
            return 0.0
        return sum(f.complexity.cyclomatic_complexity for f in self.files) / len(self.files)

@dataclass
class FrameworkMetrics:
    """Complete metrics for an entire framework."""
    name: str
    language: str
    modules: Dict[str, ModuleMetrics] = field(default_factory=dict)
    compilation_time: float = 0.0
    binary_size: int = 0

    @property
    def total_lines(self) -> int:
        return sum(m.total_lines for m in self.modules.values())

    @property
    def total_code_lines(self) -> int:
        return sum(m.code_lines for m in self.modules.values())

    @property
    def total_files(self) -> int:
        return sum(m.file_count for m in self.modules.values())

    @property
    def total_functions(self) -> int:
        return sum(m.function_count for m in self.modules.values())

    @property
    def total_structs(self) -> int:
        return sum(m.struct_count for m in self.modules.values())

# =============================================================================
# Analysis Functions
# =============================================================================

def count_morpheme_usage(content: str) -> MorphemeUsage:
    """Count Sigil morpheme pipe usage in content."""
    usage = MorphemeUsage()

    # Unicode morphemes
    usage.tau = len(re.findall(r'\|τ\{', content)) + len(re.findall(r'\|tau\{', content))
    usage.phi = len(re.findall(r'\|φ\{', content)) + len(re.findall(r'\|phi\{', content))
    usage.sigma = len(re.findall(r'\|σ\{', content)) + len(re.findall(r'\|sigma\{', content))
    usage.rho = len(re.findall(r'\|ρ', content)) + len(re.findall(r'\|rho', content))
    usage.await_pipe = len(re.findall(r'\|await', content))
    usage.map_err = len(re.findall(r'\|map_err\{', content))
    usage.collect = len(re.findall(r'\|collect', content))
    usage.coalesce = len(re.findall(r'\?\?', content))

    return usage

def count_evidentiality_usage(content: str) -> EvidentialityUsage:
    """Count Sigil evidentiality type usage."""
    usage = EvidentialityUsage()

    # Match type names followed by evidentiality markers
    usage.untrusted = len(re.findall(r'\w+~', content))
    usage.verified = len(re.findall(r'\w+!', content))
    usage.inferential = len(re.findall(r'\w+\^', content))

    return usage

def analyze_complexity(content: str) -> ComplexityMetrics:
    """Analyze code complexity metrics."""
    metrics = ComplexityMetrics()
    lines = content.split('\n')

    current_depth = 0
    max_depth = 0
    total_depth = 0
    depth_samples = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('//'):
            continue

        # Track nesting depth
        opens = stripped.count('{')
        closes = stripped.count('}')
        current_depth += opens - closes
        current_depth = max(0, current_depth)
        max_depth = max(max_depth, current_depth)
        total_depth += current_depth
        depth_samples += 1

        # Count decision points for cyclomatic complexity
        if re.search(r'\bif\b', stripped):
            metrics.cyclomatic_complexity += 1
        if re.search(r'\belse\s+if\b', stripped):
            metrics.cyclomatic_complexity += 1
        if re.search(r'\bmatch\b', stripped):
            metrics.cyclomatic_complexity += 1
        if re.search(r'\bwhile\b', stripped):
            metrics.cyclomatic_complexity += 1
        if re.search(r'\bfor\b', stripped):
            metrics.cyclomatic_complexity += 1
        if re.search(r'\bloop\b', stripped):
            metrics.cyclomatic_complexity += 1
        if stripped.startswith('=> ') or re.search(r'\s=>\s', stripped):
            metrics.match_arms += 1

        # Count error handling
        if '?' in stripped or 'Result' in stripped or 'Error' in stripped:
            metrics.error_handling_points += 1

    metrics.max_nesting_depth = max_depth
    metrics.avg_nesting_depth = total_depth / depth_samples if depth_samples > 0 else 0.0
    metrics.cyclomatic_complexity = max(1, metrics.cyclomatic_complexity + 1)  # Base complexity of 1

    return metrics

def count_rust_metrics(content: str) -> FileMetrics:
    """Count comprehensive metrics for Rust code."""
    lines = content.split('\n')
    total = len(lines)
    blank = 0
    comment = 0
    code = 0
    in_block_comment = False

    fn_pattern = re.compile(r'^\s*(pub\s+)?(async\s+)?fn\s+\w+')
    async_fn_pattern = re.compile(r'^\s*(pub\s+)?async\s+fn\s+\w+')
    struct_pattern = re.compile(r'^\s*(pub\s+)?struct\s+\w+')
    enum_pattern = re.compile(r'^\s*(pub\s+)?enum\s+\w+')
    impl_pattern = re.compile(r'^\s*impl\s+')
    trait_pattern = re.compile(r'^\s*(pub\s+)?trait\s+\w+')
    test_pattern = re.compile(r'#\[test\]|#\[tokio::test\]')

    fn_count = 0
    async_fn_count = 0
    struct_count = 0
    enum_count = 0
    impl_count = 0
    trait_count = 0
    test_count = 0

    for line in lines:
        stripped = line.strip()

        if not stripped:
            blank += 1
            continue

        if in_block_comment:
            comment += 1
            if '*/' in stripped:
                in_block_comment = False
            continue

        if stripped.startswith('/*'):
            comment += 1
            if '*/' not in stripped:
                in_block_comment = True
            continue

        if stripped.startswith('//'):
            comment += 1
            continue

        code += 1

        if fn_pattern.match(line):
            fn_count += 1
        if async_fn_pattern.match(line):
            async_fn_count += 1
        if struct_pattern.match(line):
            struct_count += 1
        if enum_pattern.match(line):
            enum_count += 1
        if impl_pattern.match(line):
            impl_count += 1
        if trait_pattern.match(line):
            trait_count += 1
        if test_pattern.search(line):
            test_count += 1

    complexity = analyze_complexity(content)

    return FileMetrics(
        path="",
        total_lines=total,
        code_lines=code,
        blank_lines=blank,
        comment_lines=comment,
        function_count=fn_count,
        struct_count=struct_count,
        enum_count=enum_count,
        impl_count=impl_count,
        trait_count=trait_count,
        async_fn_count=async_fn_count,
        test_count=test_count,
        complexity=complexity,
    )

def count_sigil_metrics(content: str) -> FileMetrics:
    """Count comprehensive metrics for Sigil code."""
    lines = content.split('\n')
    total = len(lines)
    blank = 0
    comment = 0
    code = 0

    fn_pattern = re.compile(r'^\s*(pub\s+)?(async\s+)?fn\s+\w+')
    async_fn_pattern = re.compile(r'^\s*(pub\s+)?async\s+fn\s+\w+')
    struct_pattern = re.compile(r'^\s*(pub\s+)?struct\s+\w+')
    enum_pattern = re.compile(r'^\s*(pub\s+)?enum\s+\w+')
    impl_pattern = re.compile(r'^\s*impl\s+')
    trait_pattern = re.compile(r'^\s*(pub\s+)?trait\s+\w+')
    test_pattern = re.compile(r'#\[test\]')

    fn_count = 0
    async_fn_count = 0
    struct_count = 0
    enum_count = 0
    impl_count = 0
    trait_count = 0
    test_count = 0

    for line in lines:
        stripped = line.strip()

        if not stripped:
            blank += 1
            continue

        if stripped.startswith('//'):
            comment += 1
            continue

        code += 1

        if fn_pattern.match(line):
            fn_count += 1
        if async_fn_pattern.match(line):
            async_fn_count += 1
        if struct_pattern.match(line):
            struct_count += 1
        if enum_pattern.match(line):
            enum_count += 1
        if impl_pattern.match(line):
            impl_count += 1
        if trait_pattern.match(line):
            trait_count += 1
        if test_pattern.search(line):
            test_count += 1

    morpheme_usage = count_morpheme_usage(content)
    evidentiality_usage = count_evidentiality_usage(content)
    complexity = analyze_complexity(content)

    return FileMetrics(
        path="",
        total_lines=total,
        code_lines=code,
        blank_lines=blank,
        comment_lines=comment,
        function_count=fn_count,
        struct_count=struct_count,
        enum_count=enum_count,
        impl_count=impl_count,
        trait_count=trait_count,
        async_fn_count=async_fn_count,
        test_count=test_count,
        morpheme_usage=morpheme_usage,
        evidentiality_usage=evidentiality_usage,
        complexity=complexity,
    )

def analyze_rust_codebase() -> FrameworkMetrics:
    """Analyze the Rust infernum-framework."""
    metrics = FrameworkMetrics(name="infernum-framework", language="Rust")

    if not RUST_ROOT.exists():
        print(f"Warning: Rust codebase not found at {RUST_ROOT}")
        return metrics

    crates_dir = RUST_ROOT / "crates"
    if not crates_dir.exists():
        return metrics

    module_map = {
        'infernum-core': 'core',
        'infernum-server': 'server',
        'infernum': 'cli',
        'grimoire-loader': 'grimoire_loader',
    }

    for crate_dir in crates_dir.iterdir():
        if not crate_dir.is_dir():
            continue

        crate_name = crate_dir.name
        module_name = module_map.get(crate_name, crate_name)
        module = ModuleMetrics(name=module_name)

        for subdir in ['src', 'benches']:
            search_dir = crate_dir / subdir
            if search_dir.exists():
                for rs_file in search_dir.rglob("*.rs"):
                    try:
                        content = rs_file.read_text()
                        file_metrics = count_rust_metrics(content)
                        file_metrics.path = str(rs_file.relative_to(RUST_ROOT))
                        module.files.append(file_metrics)
                    except Exception as e:
                        print(f"Error reading {rs_file}: {e}")

        if module.files:
            metrics.modules[module_name] = module

    return metrics

def analyze_sigil_codebase() -> FrameworkMetrics:
    """Analyze the Sigil infernum-sigil."""
    metrics = FrameworkMetrics(name="infernum-sigil", language="Sigil")
    src_dir = SIGIL_ROOT / "src"

    if not src_dir.exists():
        print(f"Warning: Sigil codebase not found at {src_dir}")
        return metrics

    module_files = defaultdict(list)

    for sigil_file in src_dir.rglob("*.sigil"):
        rel_path = sigil_file.relative_to(src_dir)
        parts = rel_path.parts

        if len(parts) == 1:
            module_name = parts[0].replace('.sigil', '')
            if module_name == 'lib':
                module_name = 'root'
        else:
            module_name = parts[0]

        module_files[module_name].append(sigil_file)

    # Module descriptions
    descriptions = {
        'core': 'Core types, errors, requests, responses',
        'abaddon': 'Inference engine with LLM backends',
        'malphas': 'Request routing and scheduling',
        'stolas': 'RAG (Retrieval Augmented Generation)',
        'beleth': 'Agent framework with planning',
        'asmodeus': 'Fine-tuning with LoRA',
        'dantalion': 'Observability and tracing',
        'server': 'HTTP server with OpenAI API',
        'grimoire_loader': 'Persona loading from Grimoire',
        'astaroth': 'A/B experimentation framework',
    }

    for module_name, files in module_files.items():
        module = ModuleMetrics(
            name=module_name,
            description=descriptions.get(module_name, '')
        )

        for sigil_file in files:
            try:
                content = sigil_file.read_text()
                file_metrics = count_sigil_metrics(content)
                file_metrics.path = str(sigil_file.relative_to(SIGIL_ROOT))
                module.files.append(file_metrics)
            except Exception as e:
                print(f"Error reading {sigil_file}: {e}")

        metrics.modules[module_name] = module

    return metrics

# =============================================================================
# Visualization Functions
# =============================================================================

def create_bar_chart(data: List[Tuple[str, float, float]], title: str, max_width: int = 40) -> str:
    """Create ASCII horizontal bar chart comparing two values."""
    lines = []
    lines.append(f"\n{title}")
    lines.append("=" * 70)

    if not data:
        return "\n".join(lines)

    max_val = max(max(v1, v2) for _, v1, v2 in data)
    if max_val == 0:
        max_val = 1

    for label, val1, val2 in data:
        bar1_len = int((val1 / max_val) * max_width)
        bar2_len = int((val2 / max_val) * max_width)

        lines.append(f"\n{label}")
        lines.append(f"  Rust:  {'█' * bar1_len}{' ' * (max_width - bar1_len)} {val1:,.0f}")
        lines.append(f"  Sigil: {'▓' * bar2_len}{' ' * (max_width - bar2_len)} {val2:,.0f}")

    lines.append("")
    return "\n".join(lines)

def create_reduction_chart(modules: List[Tuple[str, float]], title: str) -> str:
    """Create ASCII chart showing reduction percentages."""
    lines = []
    lines.append(f"\n{title}")
    lines.append("=" * 70)
    lines.append(f"{'Module':<20} {'Reduction':>10} {'Chart':>40}")
    lines.append("-" * 70)

    for module, reduction in sorted(modules, key=lambda x: -x[1]):
        bar_len = int(reduction / 2.5)  # Scale to ~40 chars for 100%
        bar = '█' * bar_len
        lines.append(f"{module:<20} {reduction:>9.1f}% |{bar}")

    return "\n".join(lines)

def create_morpheme_chart(usage: Dict[str, int], title: str) -> str:
    """Create ASCII chart for morpheme usage."""
    lines = []
    lines.append(f"\n{title}")
    lines.append("=" * 60)

    if not usage or max(usage.values()) == 0:
        lines.append("  No morpheme usage found")
        return "\n".join(lines)

    max_val = max(usage.values())

    for morpheme, count in sorted(usage.items(), key=lambda x: -x[1]):
        bar_len = int((count / max_val) * 30) if max_val > 0 else 0
        lines.append(f"  {morpheme:<15} {'█' * bar_len} {count}")

    return "\n".join(lines)

# =============================================================================
# Report Generation
# =============================================================================

def generate_extended_report(rust: FrameworkMetrics, sigil: FrameworkMetrics) -> str:
    """Generate comprehensive comparison report with visualizations."""
    report = []

    # Header
    report.append("╔" + "═" * 78 + "╗")
    report.append("║" + "INFERNUM FRAMEWORK EXTENDED BENCHMARK REPORT".center(78) + "║")
    report.append("║" + "Rust vs Sigil Implementation Analysis".center(78) + "║")
    report.append("║" + f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(78) + "║")
    report.append("╚" + "═" * 78 + "╝")
    report.append("")

    # Executive Summary
    report.append("┌─────────────────────────────────────────────────────────────────────────────┐")
    report.append("│ EXECUTIVE SUMMARY                                                           │")
    report.append("└─────────────────────────────────────────────────────────────────────────────┘")

    loc_reduction = ((rust.total_lines - sigil.total_lines) / rust.total_lines * 100) if rust.total_lines > 0 else 0
    code_reduction = ((rust.total_code_lines - sigil.total_code_lines) / rust.total_code_lines * 100) if rust.total_code_lines > 0 else 0

    report.append(f"""
┌────────────────────────────────┬────────────────┬────────────────┬────────────────┐
│ Metric                         │ Rust           │ Sigil          │ Reduction      │
├────────────────────────────────┼────────────────┼────────────────┼────────────────┤
│ Total Lines of Code            │ {rust.total_lines:>14,} │ {sigil.total_lines:>14,} │ {loc_reduction:>13.1f}% │
│ Code Lines (excl. blank/cmnt)  │ {rust.total_code_lines:>14,} │ {sigil.total_code_lines:>14,} │ {code_reduction:>13.1f}% │
│ Source Files                   │ {rust.total_files:>14} │ {sigil.total_files:>14} │              - │
│ Functions                      │ {rust.total_functions:>14} │ {sigil.total_functions:>14} │              - │
│ Structs                        │ {rust.total_structs:>14} │ {sigil.total_structs:>14} │              - │
└────────────────────────────────┴────────────────┴────────────────┴────────────────┘
""")

    # LOC Comparison Chart
    module_data = []
    for module in sorted(set(rust.modules.keys()) | set(sigil.modules.keys())):
        if module == 'root':
            continue
        rust_loc = rust.modules.get(module, ModuleMetrics(name=module)).total_lines
        sigil_loc = sigil.modules.get(module, ModuleMetrics(name=module)).total_lines
        module_data.append((module, rust_loc, sigil_loc))

    report.append(create_bar_chart(module_data, "LOC BY MODULE (Rust █ vs Sigil ▓)"))

    # Reduction Analysis
    reductions = []
    for module in sorted(set(rust.modules.keys()) | set(sigil.modules.keys())):
        if module == 'root' or module == 'cli':
            continue
        rust_mod = rust.modules.get(module)
        sigil_mod = sigil.modules.get(module)
        if rust_mod and rust_mod.total_lines > 0:
            rust_loc = rust_mod.total_lines
            sigil_loc = sigil_mod.total_lines if sigil_mod else 0
            reduction = ((rust_loc - sigil_loc) / rust_loc * 100)
            reductions.append((module, reduction))

    report.append(create_reduction_chart(reductions, "CODE REDUCTION BY MODULE"))

    # Per-Module Detailed Comparison
    report.append("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    report.append("│ PER-MODULE DETAILED COMPARISON                                              │")
    report.append("└─────────────────────────────────────────────────────────────────────────────┘")

    report.append(f"\n{'Module':<15} {'Rust LOC':>10} {'Sigil LOC':>10} {'Reduction':>10} {'Files R/S':>10} {'Funcs R/S':>12}")
    report.append("-" * 70)

    all_modules = sorted(set(rust.modules.keys()) | set(sigil.modules.keys()))
    for module in all_modules:
        if module == 'root':
            continue
        rust_mod = rust.modules.get(module)
        sigil_mod = sigil.modules.get(module)

        rust_loc = rust_mod.total_lines if rust_mod else 0
        sigil_loc = sigil_mod.total_lines if sigil_mod else 0
        rust_files = rust_mod.file_count if rust_mod else 0
        sigil_files = sigil_mod.file_count if sigil_mod else 0
        rust_fns = rust_mod.function_count if rust_mod else 0
        sigil_fns = sigil_mod.function_count if sigil_mod else 0

        if rust_loc > 0:
            reduction = ((rust_loc - sigil_loc) / rust_loc * 100)
            reduction_str = f"{reduction:.1f}%"
        elif sigil_loc > 0:
            reduction_str = "NEW"
        else:
            reduction_str = "N/A"

        report.append(f"{module:<15} {rust_loc:>10,} {sigil_loc:>10,} {reduction_str:>10} {rust_files:>4}/{sigil_files:<4} {rust_fns:>5}/{sigil_fns:<5}")

    # Sigil-Only Modules (like Astaroth)
    sigil_only = [m for m in sigil.modules.keys() if m not in rust.modules and m != 'root']
    if sigil_only:
        report.append("\n┌─────────────────────────────────────────────────────────────────────────────┐")
        report.append("│ SIGIL-EXCLUSIVE MODULES (No Rust Equivalent)                                │")
        report.append("└─────────────────────────────────────────────────────────────────────────────┘")

        for module in sigil_only:
            mod = sigil.modules[module]
            report.append(f"\n  {module.upper()}")
            report.append(f"  Description: {mod.description}")
            report.append(f"  Files: {mod.file_count}, LOC: {mod.total_lines:,}, Functions: {mod.function_count}")

            # List files in module
            report.append("  Files:")
            for f in sorted(mod.files, key=lambda x: x.path):
                report.append(f"    - {f.path}: {f.total_lines} lines, {f.function_count} functions")

    # Morpheme Pipe Analysis
    report.append("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    report.append("│ SIGIL MORPHEME PIPE ANALYSIS                                                │")
    report.append("└─────────────────────────────────────────────────────────────────────────────┘")

    total_morphemes = defaultdict(int)
    for module in sigil.modules.values():
        for f in module.files:
            total_morphemes['|τ (map)'] += f.morpheme_usage.tau
            total_morphemes['|φ (filter)'] += f.morpheme_usage.phi
            total_morphemes['|σ (unwrap_or)'] += f.morpheme_usage.sigma
            total_morphemes['|ρ (reduce)'] += f.morpheme_usage.rho
            total_morphemes['|await'] += f.morpheme_usage.await_pipe
            total_morphemes['|map_err'] += f.morpheme_usage.map_err
            total_morphemes['|collect'] += f.morpheme_usage.collect
            total_morphemes['?? (coalesce)'] += f.morpheme_usage.coalesce

    report.append(create_morpheme_chart(dict(total_morphemes), "MORPHEME USAGE DISTRIBUTION"))

    total_usage = sum(total_morphemes.values())
    report.append(f"\n  Total morpheme pipe usages: {total_usage:,}")
    report.append(f"  Estimated lines saved (vs Rust equivalents): ~{int(total_usage * 0.7):,}")

    # Morpheme usage by module
    report.append("\n  Morpheme Usage by Module:")
    report.append(f"  {'Module':<15} {'|τ':>6} {'|φ':>6} {'|σ':>6} {'|ρ':>6} {'|await':>7} {'??':>6} {'Total':>8}")
    report.append("  " + "-" * 65)

    for mod_name in sorted(sigil.modules.keys()):
        if mod_name == 'root':
            continue
        mod = sigil.modules[mod_name]
        tau = sum(f.morpheme_usage.tau for f in mod.files)
        phi = sum(f.morpheme_usage.phi for f in mod.files)
        sigma = sum(f.morpheme_usage.sigma for f in mod.files)
        rho = sum(f.morpheme_usage.rho for f in mod.files)
        await_p = sum(f.morpheme_usage.await_pipe for f in mod.files)
        coal = sum(f.morpheme_usage.coalesce for f in mod.files)
        total = tau + phi + sigma + rho + await_p + coal
        report.append(f"  {mod_name:<15} {tau:>6} {phi:>6} {sigma:>6} {rho:>6} {await_p:>7} {coal:>6} {total:>8}")

    # Evidentiality Type Analysis
    report.append("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    report.append("│ SIGIL EVIDENTIALITY TYPE ANALYSIS                                           │")
    report.append("└─────────────────────────────────────────────────────────────────────────────┘")

    total_evid = defaultdict(int)
    for module in sigil.modules.values():
        for f in module.files:
            total_evid['Untrusted (~)'] += f.evidentiality_usage.untrusted
            total_evid['Verified (!)'] += f.evidentiality_usage.verified
            total_evid['Inferential (^)'] += f.evidentiality_usage.inferential

    report.append("""
  Evidentiality types encode the trustworthiness of data at the type level:
  - Type~  (untrusted): Data from external sources requiring validation
  - Type!  (verified): Data that has been validated/sanitized
  - Type^  (inferential): Data inferred from other sources
""")

    for evid_type, count in sorted(total_evid.items(), key=lambda x: -x[1]):
        report.append(f"  {evid_type:<20}: {count:>6} usages")

    # Complexity Analysis
    report.append("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    report.append("│ COMPLEXITY ANALYSIS                                                         │")
    report.append("└─────────────────────────────────────────────────────────────────────────────┘")

    report.append(f"\n  {'Module':<15} {'Rust CC':>10} {'Sigil CC':>10} {'Rust MaxNest':>12} {'Sigil MaxNest':>13}")
    report.append("  " + "-" * 62)

    for module in sorted(set(rust.modules.keys()) | set(sigil.modules.keys())):
        if module == 'root' or module == 'cli':
            continue
        rust_mod = rust.modules.get(module)
        sigil_mod = sigil.modules.get(module)

        rust_cc = sum(f.complexity.cyclomatic_complexity for f in rust_mod.files) if rust_mod else 0
        sigil_cc = sum(f.complexity.cyclomatic_complexity for f in sigil_mod.files) if sigil_mod else 0
        rust_nest = max((f.complexity.max_nesting_depth for f in rust_mod.files), default=0) if rust_mod else 0
        sigil_nest = max((f.complexity.max_nesting_depth for f in sigil_mod.files), default=0) if sigil_mod else 0

        report.append(f"  {module:<15} {rust_cc:>10} {sigil_cc:>10} {rust_nest:>12} {sigil_nest:>13}")

    # Language Feature Summary
    report.append("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    report.append("│ SIGIL LANGUAGE FEATURE IMPACT                                               │")
    report.append("└─────────────────────────────────────────────────────────────────────────────┘")

    report.append(f"""
  Feature                          Rust Pattern                    Sigil Pattern
  ────────────────────────────────────────────────────────────────────────────────
  Transform/Map                    .map(|x| x.foo())               |tau{{x.foo()}}
  Filter                           .filter(|x| x.valid())          |phi{{x.valid()}}
  Default/Unwrap                   .unwrap_or(default)             |sigma{{default}} or ??
  Reduce/Fold                      .fold(0, |a,b| a+b)             |rho_sum
  Async Await                      future.await                    future|await
  Error Mapping                    .map_err(|e| Error::Foo(e))     |map_err{{Error::Foo}}
  Option Coalescing                opt.unwrap_or(default)          opt ?? default
  Inline Field Defaults            impl Default for Foo {{}}         field: Type = val

  ESTIMATED IMPACT:
  - Total morpheme usages: {total_usage:,}
  - Avg lines saved per morpheme: ~0.7
  - Estimated total lines saved: ~{int(total_usage * 0.7):,}
  - Overall LOC reduction achieved: {loc_reduction:.1f}%
""")

    # Detailed File Listing
    report.append("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    report.append("│ DETAILED FILE METRICS                                                       │")
    report.append("└─────────────────────────────────────────────────────────────────────────────┘")

    report.append("\n### Sigil Files (sorted by LOC)")
    report.append(f"{'File':<50} {'LOC':>6} {'Code':>6} {'Funcs':>6} {'Morph':>6}")
    report.append("-" * 76)

    all_sigil_files = []
    for mod in sigil.modules.values():
        for f in mod.files:
            morph_count = (f.morpheme_usage.tau + f.morpheme_usage.phi +
                          f.morpheme_usage.sigma + f.morpheme_usage.rho +
                          f.morpheme_usage.await_pipe + f.morpheme_usage.coalesce)
            all_sigil_files.append((f.path.replace('src/', ''), f.total_lines, f.code_lines, f.function_count, morph_count))

    for path, loc, code, fns, morph in sorted(all_sigil_files, key=lambda x: -x[1]):
        report.append(f"{path:<50} {loc:>6} {code:>6} {fns:>6} {morph:>6}")

    # Footer
    report.append("\n" + "═" * 80)
    report.append("END OF EXTENDED BENCHMARK REPORT")
    report.append("═" * 80)

    return "\n".join(report)

# =============================================================================
# Performance Simulation
# =============================================================================

def simulate_performance_metrics() -> Dict:
    """Simulate performance benchmarks based on code analysis."""
    # These are simulated metrics based on typical language characteristics
    # In a real scenario, these would come from actual runtime benchmarks

    return {
        "compilation": {
            "rust_release_time_s": 146.45,  # From previous benchmark
            "sigil_transpile_time_s": 12.3,  # Estimated
            "rust_incremental_time_s": 8.5,
            "sigil_incremental_time_s": 1.2,
        },
        "memory": {
            "rust_binary_mb": 15.2,
            "sigil_binary_mb": 14.8,  # Similar after transpilation
        },
        "developer_experience": {
            "avg_chars_per_operation": {
                "rust": {
                    "map_chain": 25,
                    "error_handling": 35,
                    "async_await": 12,
                    "option_unwrap": 22,
                },
                "sigil": {
                    "map_chain": 8,
                    "error_handling": 15,
                    "async_await": 7,
                    "option_unwrap": 6,
                }
            },
            "estimated_typing_reduction": 0.42,  # 42% less typing
        }
    }

def generate_performance_report(perf: Dict) -> str:
    """Generate performance simulation report."""
    report = []

    report.append("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    report.append("│ PERFORMANCE SIMULATION BENCHMARKS                                           │")
    report.append("└─────────────────────────────────────────────────────────────────────────────┘")

    report.append("""
  Note: These metrics are simulated based on typical language characteristics
  and the LOC analysis performed. Actual runtime performance would require
  compilation and execution of both implementations.

  COMPILATION PERFORMANCE
  ═══════════════════════
  Metric                          Rust            Sigil
  ────────────────────────────────────────────────────────
  Full Release Build              {0:.1f}s          ~{1:.1f}s (transpile)
  Incremental Build               ~{2:.1f}s          ~{3:.1f}s

  BINARY SIZE (estimated)
  ═══════════════════════
  Rust binary:  ~{4:.1f} MB
  Sigil binary: ~{5:.1f} MB (after transpilation to Rust)

  DEVELOPER EXPERIENCE METRICS
  ════════════════════════════
  Operation                  Rust (chars)    Sigil (chars)    Reduction
  ───────────────────────────────────────────────────────────────────────
  Map/Transform Chain             ~25              ~8            68%
  Error Handling                  ~35             ~15            57%
  Async/Await                     ~12              ~7            42%
  Option Unwrap                   ~22              ~6            73%

  Overall estimated typing reduction: {6:.0%}
""".format(
        perf["compilation"]["rust_release_time_s"],
        perf["compilation"]["sigil_transpile_time_s"],
        perf["compilation"]["rust_incremental_time_s"],
        perf["compilation"]["sigil_incremental_time_s"],
        perf["memory"]["rust_binary_mb"],
        perf["memory"]["sigil_binary_mb"],
        perf["developer_experience"]["estimated_typing_reduction"],
    ))

    return "\n".join(report)

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the extended benchmark suite."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  INFERNUM EXTENDED BENCHMARK SUITE                            ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()

    print("Analyzing Rust codebase...")
    rust_metrics = analyze_rust_codebase()
    print(f"  Found {rust_metrics.total_files} files, {rust_metrics.total_lines:,} LOC")

    print("Analyzing Sigil codebase...")
    sigil_metrics = analyze_sigil_codebase()
    print(f"  Found {sigil_metrics.total_files} files, {sigil_metrics.total_lines:,} LOC")

    print("Generating performance simulations...")
    perf_metrics = simulate_performance_metrics()

    # Generate main report
    print("Generating extended report...")
    report = generate_extended_report(rust_metrics, sigil_metrics)
    report += generate_performance_report(perf_metrics)

    print("\n" + report)

    # Save report
    report_path = RESULTS_DIR / "extended_benchmark_report.txt"
    report_path.write_text(report)
    print(f"\nReport saved to: {report_path}")

    # Save detailed JSON data
    def metrics_to_dict(m: FrameworkMetrics) -> dict:
        return {
            "name": m.name,
            "language": m.language,
            "total_lines": m.total_lines,
            "total_code_lines": m.total_code_lines,
            "total_files": m.total_files,
            "total_functions": m.total_functions,
            "total_structs": m.total_structs,
            "compilation_time": m.compilation_time,
            "binary_size": m.binary_size,
            "modules": {
                name: {
                    "total_lines": mod.total_lines,
                    "code_lines": mod.code_lines,
                    "file_count": mod.file_count,
                    "function_count": mod.function_count,
                    "description": mod.description,
                    "total_morpheme_usage": mod.total_morpheme_usage,
                    "avg_complexity": mod.avg_complexity,
                    "files": [
                        {
                            "path": f.path,
                            "total_lines": f.total_lines,
                            "code_lines": f.code_lines,
                            "function_count": f.function_count,
                            "struct_count": f.struct_count,
                            "async_fn_count": f.async_fn_count,
                            "test_count": f.test_count,
                            "morpheme_usage": {
                                "tau": f.morpheme_usage.tau,
                                "phi": f.morpheme_usage.phi,
                                "sigma": f.morpheme_usage.sigma,
                                "rho": f.morpheme_usage.rho,
                                "await": f.morpheme_usage.await_pipe,
                                "map_err": f.morpheme_usage.map_err,
                                "collect": f.morpheme_usage.collect,
                                "coalesce": f.morpheme_usage.coalesce,
                            },
                            "evidentiality_usage": {
                                "untrusted": f.evidentiality_usage.untrusted,
                                "verified": f.evidentiality_usage.verified,
                                "inferential": f.evidentiality_usage.inferential,
                            },
                            "complexity": {
                                "max_nesting_depth": f.complexity.max_nesting_depth,
                                "avg_nesting_depth": f.complexity.avg_nesting_depth,
                                "cyclomatic_complexity": f.complexity.cyclomatic_complexity,
                                "match_arms": f.complexity.match_arms,
                                "error_handling_points": f.complexity.error_handling_points,
                            }
                        }
                        for f in mod.files
                    ]
                }
                for name, mod in m.modules.items()
            }
        }

    json_data = {
        "timestamp": datetime.now().isoformat(),
        "rust": metrics_to_dict(rust_metrics),
        "sigil": metrics_to_dict(sigil_metrics),
        "performance": perf_metrics,
    }

    json_path = RESULTS_DIR / "extended_benchmark_data.json"
    json_path.write_text(json.dumps(json_data, indent=2))
    print(f"JSON data saved to: {json_path}")

if __name__ == "__main__":
    main()
