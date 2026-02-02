//! CLI Integration Tests for Infernum Studio
//!
//! Tests the `infernum studio` CLI commands end-to-end.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

/// Helper to run the infernum CLI.
fn infernum() -> Command {
    Command::cargo_bin("infernum").unwrap()
}

/// Helper to run the infernum CLI with the given workspace.
fn infernum_with_workspace(workspace: &Path) -> Command {
    let mut cmd = Command::cargo_bin("infernum").unwrap();
    cmd.env("PAIMON_WORKSPACE", workspace);
    cmd
}

/// Helper to set up a temporary workspace for studio commands.
fn setup_workspace() -> TempDir {
    TempDir::new().expect("failed to create temp dir")
}

// =============================================================================
// Studio Stats Tests
// =============================================================================

#[test]
fn test_cli_studio_stats() {
    let temp = setup_workspace();

    // The stats command shows "Infernum Studio" header
    infernum_with_workspace(temp.path())
        .args(["studio", "stats"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Infernum Studio"));
}

// =============================================================================
// Dataset CLI Tests
// =============================================================================

#[test]
fn test_cli_dataset_list_empty() {
    let temp = setup_workspace();

    infernum_with_workspace(temp.path())
        .args(["studio", "dataset", "list"])
        .assert()
        .success();
}

#[test]
fn test_cli_dataset_create() {
    let temp = setup_workspace();

    // Create should succeed
    infernum_with_workspace(temp.path())
        .args(["studio", "dataset", "create", "test-dataset", "-d", "A test dataset"])
        .assert()
        .success();
}

#[test]
fn test_cli_dataset_workflow() {
    let temp = setup_workspace();

    // 1. Create dataset
    infernum_with_workspace(temp.path())
        .args(["studio", "dataset", "create", "workflow-ds", "-d", "Test workflow"])
        .assert()
        .success();

    // 2. List datasets
    infernum_with_workspace(temp.path())
        .args(["studio", "dataset", "list"])
        .assert()
        .success();

    // 3. Create a JSONL file for import
    let jsonl_path = temp.path().join("data.jsonl");
    fs::write(
        &jsonl_path,
        r#"{"input":"What is 2+2?","output":"4"}
{"input":"Hello","output":"Hi there!"}
{"input":"What is the capital of France?","output":"Paris"}"#,
    )
    .expect("failed to write JSONL");

    // 4. Import data
    infernum_with_workspace(temp.path())
        .args([
            "studio",
            "dataset",
            "import",
            "workflow-ds",
            jsonl_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    // 5. Get dataset info
    infernum_with_workspace(temp.path())
        .args(["studio", "dataset", "info", "workflow-ds"])
        .assert()
        .success();

    // 6. Validate dataset
    infernum_with_workspace(temp.path())
        .args(["studio", "dataset", "validate", "workflow-ds"])
        .assert()
        .success();
}

#[test]
fn test_cli_dataset_info_not_found() {
    let temp = setup_workspace();

    // Accessing a nonexistent dataset should fail
    infernum_with_workspace(temp.path())
        .args(["studio", "dataset", "info", "nonexistent-dataset"])
        .assert()
        .failure();
}

// =============================================================================
// Experiment CLI Tests
// =============================================================================

#[test]
fn test_cli_experiment_list_empty() {
    let temp = setup_workspace();

    infernum_with_workspace(temp.path())
        .args(["studio", "experiment", "list"])
        .assert()
        .success();
}

#[test]
fn test_cli_experiment_create() {
    let temp = setup_workspace();

    infernum_with_workspace(temp.path())
        .args(["studio", "experiment", "create", "test-exp", "-d", "A test experiment"])
        .assert()
        .success();
}

#[test]
fn test_cli_experiment_workflow() {
    let temp = setup_workspace();

    // 1. Create experiment
    infernum_with_workspace(temp.path())
        .args(["studio", "experiment", "create", "exp-workflow", "-d", "Workflow test"])
        .assert()
        .success();

    // 2. List experiments
    infernum_with_workspace(temp.path())
        .args(["studio", "experiment", "list"])
        .assert()
        .success();

    // 3. Get experiment info
    infernum_with_workspace(temp.path())
        .args(["studio", "experiment", "info", "exp-workflow"])
        .assert()
        .success();

    // 4. List runs (should be empty initially)
    infernum_with_workspace(temp.path())
        .args(["studio", "experiment", "runs", "exp-workflow"])
        .assert()
        .success();
}

#[test]
fn test_cli_experiment_info_not_found() {
    let temp = setup_workspace();

    infernum_with_workspace(temp.path())
        .args(["studio", "experiment", "info", "nonexistent-experiment"])
        .assert()
        .failure();
}

// =============================================================================
// Prompt CLI Tests
// =============================================================================

#[test]
fn test_cli_prompt_list_empty() {
    let temp = setup_workspace();

    infernum_with_workspace(temp.path())
        .args(["studio", "prompt", "list"])
        .assert()
        .success();
}

#[test]
fn test_cli_prompt_create() {
    let temp = setup_workspace();

    infernum_with_workspace(temp.path())
        .args([
            "studio",
            "prompt",
            "create",
            "greeting-template",
            "Hello, {{name}}! Welcome to {{place}}.",
            "-d",
            "A greeting template",
        ])
        .assert()
        .success();
}

#[test]
fn test_cli_prompt_workflow() {
    let temp = setup_workspace();

    // 1. Create prompt template
    infernum_with_workspace(temp.path())
        .args([
            "studio",
            "prompt",
            "create",
            "qa-template",
            "Question: {{question}}\nAnswer:",
            "-d",
            "Q&A template",
        ])
        .assert()
        .success();

    // 2. List prompts
    infernum_with_workspace(temp.path())
        .args(["studio", "prompt", "list"])
        .assert()
        .success();

    // 3. Show prompt
    infernum_with_workspace(temp.path())
        .args(["studio", "prompt", "show", "qa-template"])
        .assert()
        .success();
}

// =============================================================================
// Registry CLI Tests
// =============================================================================

#[test]
fn test_cli_registry_list_empty() {
    let temp = setup_workspace();

    infernum_with_workspace(temp.path())
        .args(["studio", "registry", "list"])
        .assert()
        .success();
}

#[test]
fn test_cli_registry_register() {
    let temp = setup_workspace();

    // Create a dummy model path
    let model_path = temp.path().join("model");
    fs::create_dir_all(&model_path).expect("create model dir");
    fs::write(model_path.join("config.json"), "{}").expect("write config");

    infernum_with_workspace(temp.path())
        .args([
            "studio",
            "registry",
            "register",
            "test-model",
            model_path.to_str().unwrap(),
            "-d",
            "A test model",
        ])
        .assert()
        .success();
}

// =============================================================================
// Help Tests
// =============================================================================

#[test]
fn test_cli_studio_help() {
    // Help should show all subcommands
    infernum()
        .args(["studio", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("dataset"))
        .stdout(predicate::str::contains("experiment"))
        .stdout(predicate::str::contains("prompt"))
        .stdout(predicate::str::contains("registry"));
}

#[test]
fn test_cli_dataset_help() {
    infernum()
        .args(["studio", "dataset", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("list"))
        .stdout(predicate::str::contains("create"))
        .stdout(predicate::str::contains("import"));
}

#[test]
fn test_cli_experiment_help() {
    infernum()
        .args(["studio", "experiment", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("list"))
        .stdout(predicate::str::contains("create"))
        .stdout(predicate::str::contains("runs"));
}

#[test]
fn test_cli_prompt_help() {
    infernum()
        .args(["studio", "prompt", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("list"))
        .stdout(predicate::str::contains("create"))
        .stdout(predicate::str::contains("show"))
        .stdout(predicate::str::contains("test"));
}

#[test]
fn test_cli_registry_help() {
    infernum()
        .args(["studio", "registry", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("list"))
        .stdout(predicate::str::contains("register"))
        .stdout(predicate::str::contains("promote"));
}
