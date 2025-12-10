#!/usr/bin/env python3
"""
Infernum Framework Benchmark Suite
Compares Rust and Sigil implementations across multiple dimensions:
- Static code metrics (LOC, file counts, complexity)
- Compilation performance (Rust)
- Code density and reduction analysis
"""

import os
import subprocess
import time
import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from collections import defaultdict

# Paths
RUST_ROOT = Path("/tmp/infernum-framework")
SIGIL_ROOT = Path("/home/user/infernum-complete/infernum-sigil")
RESULTS_DIR = Path("/home/user/infernum-complete/benchmarks/results")

@dataclass
class FileMetrics:
    """Metrics for a single file."""
    path: str
    total_lines: int
    code_lines: int  # non-blank, non-comment
    blank_lines: int
    comment_lines: int
    function_count: int
    struct_count: int
    enum_count: int
    impl_count: int

@dataclass
class ModuleMetrics:
    """Aggregated metrics for a module/crate."""
    name: str
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

def count_rust_metrics(content: str) -> Tuple[int, int, int, int, int, int, int]:
    """Count metrics for Rust code."""
    lines = content.split('\n')
    total = len(lines)
    blank = 0
    comment = 0
    code = 0
    in_block_comment = False

    fn_pattern = re.compile(r'^\s*(pub\s+)?(async\s+)?fn\s+\w+')
    struct_pattern = re.compile(r'^\s*(pub\s+)?struct\s+\w+')
    enum_pattern = re.compile(r'^\s*(pub\s+)?enum\s+\w+')
    impl_pattern = re.compile(r'^\s*impl\s+')

    fn_count = 0
    struct_count = 0
    enum_count = 0
    impl_count = 0

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
        if struct_pattern.match(line):
            struct_count += 1
        if enum_pattern.match(line):
            enum_count += 1
        if impl_pattern.match(line):
            impl_count += 1

    return total, code, blank, comment, fn_count, struct_count, enum_count, impl_count

def count_sigil_metrics(content: str) -> Tuple[int, int, int, int, int, int, int]:
    """Count metrics for Sigil code."""
    lines = content.split('\n')
    total = len(lines)
    blank = 0
    comment = 0
    code = 0

    fn_pattern = re.compile(r'^\s*(pub\s+)?(async\s+)?fn\s+\w+')
    struct_pattern = re.compile(r'^\s*(pub\s+)?struct\s+\w+')
    enum_pattern = re.compile(r'^\s*(pub\s+)?enum\s+\w+')
    impl_pattern = re.compile(r'^\s*impl\s+')

    fn_count = 0
    struct_count = 0
    enum_count = 0
    impl_count = 0

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
        if struct_pattern.match(line):
            struct_count += 1
        if enum_pattern.match(line):
            enum_count += 1
        if impl_pattern.match(line):
            impl_count += 1

    return total, code, blank, comment, fn_count, struct_count, enum_count, impl_count

def analyze_rust_codebase() -> FrameworkMetrics:
    """Analyze the Rust infernum-framework."""
    metrics = FrameworkMetrics(name="infernum-framework", language="Rust")
    crates_dir = RUST_ROOT / "crates"

    # Module name mapping
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

        src_dir = crate_dir / "src"
        if src_dir.exists():
            for rs_file in src_dir.rglob("*.rs"):
                try:
                    content = rs_file.read_text()
                    total, code, blank, comment, fns, structs, enums, impls = count_rust_metrics(content)

                    file_metrics = FileMetrics(
                        path=str(rs_file.relative_to(RUST_ROOT)),
                        total_lines=total,
                        code_lines=code,
                        blank_lines=blank,
                        comment_lines=comment,
                        function_count=fns,
                        struct_count=structs,
                        enum_count=enums,
                        impl_count=impls,
                    )
                    module.files.append(file_metrics)
                except Exception as e:
                    print(f"Error reading {rs_file}: {e}")

        # Also check for benches
        bench_dir = crate_dir / "benches"
        if bench_dir.exists():
            for rs_file in bench_dir.rglob("*.rs"):
                try:
                    content = rs_file.read_text()
                    total, code, blank, comment, fns, structs, enums, impls = count_rust_metrics(content)

                    file_metrics = FileMetrics(
                        path=str(rs_file.relative_to(RUST_ROOT)),
                        total_lines=total,
                        code_lines=code,
                        blank_lines=blank,
                        comment_lines=comment,
                        function_count=fns,
                        struct_count=structs,
                        enum_count=enums,
                        impl_count=impls,
                    )
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

    # Group files by module
    module_files = defaultdict(list)

    for sigil_file in src_dir.rglob("*.sigil"):
        rel_path = sigil_file.relative_to(src_dir)
        parts = rel_path.parts

        if len(parts) == 1:
            # Top-level file like lib.sigil or grimoire_loader.sigil
            module_name = parts[0].replace('.sigil', '')
            if module_name == 'lib':
                module_name = 'root'
        else:
            # Nested file
            module_name = parts[0]

        module_files[module_name].append(sigil_file)

    for module_name, files in module_files.items():
        module = ModuleMetrics(name=module_name)

        for sigil_file in files:
            try:
                content = sigil_file.read_text()
                total, code, blank, comment, fns, structs, enums, impls = count_sigil_metrics(content)

                file_metrics = FileMetrics(
                    path=str(sigil_file.relative_to(SIGIL_ROOT)),
                    total_lines=total,
                    code_lines=code,
                    blank_lines=blank,
                    comment_lines=comment,
                    function_count=fns,
                    struct_count=structs,
                    enum_count=enums,
                    impl_count=impls,
                )
                module.files.append(file_metrics)
            except Exception as e:
                print(f"Error reading {sigil_file}: {e}")

        metrics.modules[module_name] = module

    return metrics

def measure_rust_compilation() -> Tuple[float, int]:
    """Measure Rust compilation time and binary size."""
    print("Measuring Rust compilation performance...")

    # Clean first
    subprocess.run(
        ["cargo", "clean"],
        cwd=RUST_ROOT,
        capture_output=True,
    )

    # Time the build
    start = time.time()
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=RUST_ROOT,
        capture_output=True,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"Rust build failed: {result.stderr.decode()[:500]}")
        return elapsed, 0

    # Measure binary size
    binary_path = RUST_ROOT / "target" / "release" / "infernum"
    binary_size = 0
    if binary_path.exists():
        binary_size = binary_path.stat().st_size

    return elapsed, binary_size

def generate_report(rust: FrameworkMetrics, sigil: FrameworkMetrics) -> str:
    """Generate a comprehensive comparison report."""
    report = []
    report.append("=" * 80)
    report.append("INFERNUM FRAMEWORK BENCHMARK REPORT")
    report.append("Rust vs Sigil Implementation Comparison")
    report.append("=" * 80)
    report.append("")

    # Overall summary
    report.append("## EXECUTIVE SUMMARY")
    report.append("-" * 40)
    report.append(f"{'Metric':<30} {'Rust':>15} {'Sigil':>15} {'Reduction':>15}")
    report.append("-" * 75)

    loc_reduction = ((rust.total_lines - sigil.total_lines) / rust.total_lines * 100) if rust.total_lines > 0 else 0
    code_reduction = ((rust.total_code_lines - sigil.total_code_lines) / rust.total_code_lines * 100) if rust.total_code_lines > 0 else 0

    report.append(f"{'Total Lines':<30} {rust.total_lines:>15,} {sigil.total_lines:>15,} {loc_reduction:>14.1f}%")
    report.append(f"{'Code Lines (non-blank/comment)':<30} {rust.total_code_lines:>15,} {sigil.total_code_lines:>15,} {code_reduction:>14.1f}%")
    report.append(f"{'Source Files':<30} {rust.total_files:>15} {sigil.total_files:>15}")
    report.append("")

    # Compilation metrics
    if rust.compilation_time > 0:
        report.append("## COMPILATION METRICS (Rust)")
        report.append("-" * 40)
        report.append(f"Build Time (release):  {rust.compilation_time:.2f}s")
        if rust.binary_size > 0:
            report.append(f"Binary Size:           {rust.binary_size:,} bytes ({rust.binary_size / 1024 / 1024:.2f} MB)")
        report.append("")

    # Per-module comparison
    report.append("## PER-MODULE COMPARISON")
    report.append("-" * 40)
    report.append(f"{'Module':<20} {'Rust LOC':>12} {'Sigil LOC':>12} {'Reduction':>12} {'Files (R/S)':>12}")
    report.append("-" * 70)

    # Map Sigil modules to Rust equivalents
    module_mapping = {
        'core': 'core',
        'abaddon': 'abaddon',
        'malphas': 'malphas',
        'stolas': 'stolas',
        'beleth': 'beleth',
        'asmodeus': 'asmodeus',
        'dantalion': 'dantalion',
        'server': 'server',
        'grimoire_loader': 'grimoire_loader',
        'cli': None,  # No Sigil equivalent
    }

    all_modules = set(rust.modules.keys()) | set(sigil.modules.keys())
    # Remove 'root' from Sigil (it's just lib.sigil)
    all_modules.discard('root')

    for module in sorted(all_modules):
        rust_mod = rust.modules.get(module)
        sigil_mod = sigil.modules.get(module)

        rust_loc = rust_mod.total_lines if rust_mod else 0
        sigil_loc = sigil_mod.total_lines if sigil_mod else 0
        rust_files = rust_mod.file_count if rust_mod else 0
        sigil_files = sigil_mod.file_count if sigil_mod else 0

        if rust_loc > 0:
            reduction = ((rust_loc - sigil_loc) / rust_loc * 100)
            reduction_str = f"{reduction:.1f}%"
        else:
            reduction_str = "N/A"

        report.append(f"{module:<20} {rust_loc:>12,} {sigil_loc:>12,} {reduction_str:>12} {rust_files:>5}/{sigil_files:<5}")

    report.append("")

    # Detailed file metrics
    report.append("## DETAILED FILE METRICS")
    report.append("-" * 40)

    report.append("\n### Rust Files")
    report.append(f"{'File':<50} {'Total':>8} {'Code':>8} {'Funcs':>6}")
    report.append("-" * 75)
    for module in sorted(rust.modules.values(), key=lambda m: m.name):
        for f in sorted(module.files, key=lambda x: x.path):
            short_path = f.path.replace('crates/', '')
            report.append(f"{short_path:<50} {f.total_lines:>8} {f.code_lines:>8} {f.function_count:>6}")

    report.append("\n### Sigil Files")
    report.append(f"{'File':<50} {'Total':>8} {'Code':>8} {'Funcs':>6}")
    report.append("-" * 75)
    for module in sorted(sigil.modules.values(), key=lambda m: m.name):
        for f in sorted(module.files, key=lambda x: x.path):
            short_path = f.path.replace('src/', '')
            report.append(f"{short_path:<50} {f.total_lines:>8} {f.code_lines:>8} {f.function_count:>6}")

    report.append("")

    # Language feature analysis
    report.append("## SIGIL LANGUAGE ADVANTAGES")
    report.append("-" * 40)
    report.append("""
Key features contributing to code reduction:

1. MORPHEME PIPES (|τ, |φ, |σ, |ρ, |await)
   - Eliminates verbose .map(), .and_then(), .ok_or() chains
   - Example: value|τ{transform}|σ{default} vs value.map(transform).unwrap_or(default)

2. OPTION COALESCING (??)
   - Replaces .unwrap_or() and .unwrap_or_else()
   - Example: opt ?? default vs opt.unwrap_or(default)

3. INLINE FIELD DEFAULTS
   - No need for Default impl or builder patterns
   - Example: field: Type = default_value directly in struct

4. INFERRED SEMICOLONS
   - Reduces visual noise while maintaining clarity

5. UNIFIED ASYNC SYNTAX (|await)
   - Consistent with morpheme pipe pattern
   - Example: future|await vs future.await

6. SIMPLIFIED ERROR MAPPING (|map_err{...})
   - More concise error conversion
   - Example: result|map_err{Error::Io}? vs result.map_err(Error::Io)?
""")

    report.append("")
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    return "\n".join(report)

def main():
    """Run the full benchmark suite."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Analyzing Rust codebase...")
    rust_metrics = analyze_rust_codebase()

    print("Analyzing Sigil codebase...")
    sigil_metrics = analyze_sigil_codebase()

    # Optionally measure Rust compilation
    try:
        comp_time, binary_size = measure_rust_compilation()
        rust_metrics.compilation_time = comp_time
        rust_metrics.binary_size = binary_size
    except Exception as e:
        print(f"Skipping compilation benchmark: {e}")

    # Generate report
    report = generate_report(rust_metrics, sigil_metrics)
    print("\n" + report)

    # Save results
    report_path = RESULTS_DIR / "benchmark_report.txt"
    report_path.write_text(report)
    print(f"\nReport saved to: {report_path}")

    # Save raw JSON data
    def metrics_to_dict(m: FrameworkMetrics) -> dict:
        return {
            "name": m.name,
            "language": m.language,
            "total_lines": m.total_lines,
            "total_code_lines": m.total_code_lines,
            "total_files": m.total_files,
            "compilation_time": m.compilation_time,
            "binary_size": m.binary_size,
            "modules": {
                name: {
                    "total_lines": mod.total_lines,
                    "code_lines": mod.code_lines,
                    "file_count": mod.file_count,
                    "files": [
                        {
                            "path": f.path,
                            "total_lines": f.total_lines,
                            "code_lines": f.code_lines,
                            "function_count": f.function_count,
                        }
                        for f in mod.files
                    ]
                }
                for name, mod in m.modules.items()
            }
        }

    json_data = {
        "rust": metrics_to_dict(rust_metrics),
        "sigil": metrics_to_dict(sigil_metrics),
    }

    json_path = RESULTS_DIR / "benchmark_data.json"
    json_path.write_text(json.dumps(json_data, indent=2))
    print(f"JSON data saved to: {json_path}")

if __name__ == "__main__":
    main()
