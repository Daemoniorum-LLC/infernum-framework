//! Test runner binary for Rust/Sigil comparison.
//!
//! Runs the shared JSON test suite against the Rust implementation.

use infernum_comparison::test_runner::validate_test_suite as run_test_suite;
use std::path::Path;

fn main() {
    println!("=================================================");
    println!("Infernum Rust/Sigil Comparison - Test Runner");
    println!("=================================================\n");

    let tests_dir = Path::new("tests");
    let mut total_passed = 0;
    let mut total_failed = 0;
    let mut total_tests = 0;

    // Find all JSON test files
    let test_files = [
        "tests/streaming/collect_text.json",
        "tests/streaming/chunk_creation.json",
        "tests/sampling/validation.json",
        "tests/sampling/presets.json",
        "tests/types/usage.json",
    ];

    for file in &test_files {
        let path = Path::new(file);
        if !path.exists() {
            println!("  [SKIP] {} (not found)", file);
            continue;
        }

        match run_test_suite(path) {
            Ok(result) => {
                let status = if result.failed == 0 { "PASS" } else { "FAIL" };
                println!(
                    "  [{}] {} - {}/{} passed",
                    status, result.suite_name, result.passed, result.total
                );

                // Print failures
                for test in &result.results {
                    if !test.passed {
                        println!(
                            "       FAILED: {} - {}",
                            test.name,
                            test.error.as_ref().unwrap_or(&String::new())
                        );
                    }
                }

                total_passed += result.passed;
                total_failed += result.failed;
                total_tests += result.total;
            }
            Err(e) => {
                println!("  [ERROR] {}: {}", file, e);
            }
        }
    }

    println!("\n=================================================");
    println!("Summary: {}/{} tests passed", total_passed, total_tests);
    if total_failed > 0 {
        println!("         {} tests FAILED", total_failed);
    }
    println!("=================================================");

    // Exit with error code if any tests failed
    if total_failed > 0 {
        std::process::exit(1);
    }
}
