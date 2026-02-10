//! Integration tests for Paimon LLM Studio.
//!
//! Tests end-to-end workflows across Studio components:
//! datasets, experiments, prompts, registry, and agents.

use std::collections::HashMap;
use std::sync::Arc;

use infernum_paimon::{
    dataset::{DatasetFormat, DatasetStats, IssueSeverity, SplitConfig},
    experiment::{Artifact, ArtifactType, HyperparamValue},
    // Agents
    DataCuratorAgent,
    // Dataset
    Dataset,
    DatasetConfig,
    DatasetManager,
    EvalAnalystAgent,
    Example,
    // Experiment
    Experiment,
    ExperimentConfig,
    ExperimentTracker,
    HyperparamOptimizerAgent,
    // Registry
    Model,
    ModelMetadata,
    ModelRegistry,
    ModelStage,
    // Prompt
    PromptStudio,
    PromptTemplate,
    Run,
    RunStatus,
    // Studio
    Studio,
    StudioConfig,
    // Persistence
    StudioDatabase,
    TrainingCoachAgent,
};
use tempfile::TempDir;

// ============================================================================
// Studio Creation and Configuration Tests
// ============================================================================

#[tokio::test]
async fn test_studio_creation_with_defaults() {
    let temp = TempDir::new().expect("temp dir");
    let config = StudioConfig::with_base_dir(temp.path().to_path_buf());

    let studio = Studio::new(config).await.expect("create studio");

    assert!(studio.agents_enabled());
    assert!(studio.data_curator().is_some());
    assert!(studio.training_coach().is_some());
    assert!(studio.eval_analyst().is_some());
    assert!(studio.hyperparam_optimizer().is_some());
}

#[tokio::test]
async fn test_studio_without_agents() {
    let temp = TempDir::new().expect("temp dir");
    let config = StudioConfig::with_base_dir(temp.path().to_path_buf()).without_agents();

    let studio = Studio::new(config).await.expect("create studio");

    assert!(!studio.agents_enabled());
    assert!(studio.data_curator().is_none());
    assert!(studio.training_coach().is_none());
}

#[tokio::test]
async fn test_studio_with_agent_model() {
    let temp = TempDir::new().expect("temp dir");
    let config =
        StudioConfig::with_base_dir(temp.path().to_path_buf()).with_agent_model("qwen-2.5-7b");

    let studio = Studio::new(config).await.expect("create studio");

    assert_eq!(studio.config().agent_model, Some("qwen-2.5-7b".to_string()));
}

#[tokio::test]
async fn test_studio_stats() {
    let temp = TempDir::new().expect("temp dir");
    let config = StudioConfig::with_base_dir(temp.path().to_path_buf());

    let studio = Studio::new(config).await.expect("create studio");
    let stats = studio.stats().await;

    assert_eq!(stats.datasets_count, 0);
    assert_eq!(stats.experiments_count, 0);
    assert_eq!(stats.models_count, 0);
    assert_eq!(stats.prompts_count, 0);
    assert!(stats.agents_enabled);
}

#[tokio::test]
async fn test_studio_directory_creation() {
    let temp = TempDir::new().expect("temp dir");
    let base = temp.path().to_path_buf();
    let config = StudioConfig::with_base_dir(base.clone());

    let _studio = Studio::new(config).await.expect("create studio");

    // Verify directories were created
    assert!(base.join("datasets").exists());
    assert!(base.join("experiments").exists());
    assert!(base.join("models").exists());
    assert!(base.join("prompts").exists());
}

// ============================================================================
// Dataset Management Integration Tests
// ============================================================================

#[tokio::test]
async fn test_dataset_creation_and_retrieval() {
    let temp = TempDir::new().expect("temp dir");
    let manager = DatasetManager::new(temp.path().to_path_buf());

    let examples = vec![
        Example::new("What is 2+2?", "4"),
        Example::new("What is 3+3?", "6"),
        Example::new("What is 5+5?", "10"),
    ];

    let config = DatasetConfig::new("math-dataset")
        .with_description("Basic math examples")
        .with_format(DatasetFormat::JsonLines)
        .with_tags(vec!["math", "basic"]);

    let dataset = manager.create(config, examples).await.expect("create");

    assert_eq!(dataset.name, "math-dataset");
    assert_eq!(dataset.len(), 3);
    assert!(dataset.validation.is_some());

    // Retrieve
    let retrieved = manager.get(&dataset.id).await.expect("get");
    assert_eq!(retrieved.id, dataset.id);
    assert_eq!(retrieved.examples.len(), 3);
}

#[tokio::test]
async fn test_dataset_validation() {
    let temp = TempDir::new().expect("temp dir");
    let manager = DatasetManager::new(temp.path().to_path_buf());

    // Create dataset with some issues
    let examples = vec![
        Example::new("Good input", "Good output"),
        Example::new("", "Empty input"), // Error: empty input
        Example::new("Short", "x"),      // Warning: very short
        Example::new("Good input", "Another output"), // Duplicate input
    ];

    let config = DatasetConfig::new("problematic-dataset");
    let dataset = manager.create(config, examples).await.expect("create");

    let validation = dataset.validation.expect("validation");
    assert!(!validation.passed); // Should fail due to empty input

    let errors: Vec<_> = validation
        .issues
        .iter()
        .filter(|i| i.severity == IssueSeverity::Error)
        .collect();
    assert!(!errors.is_empty());
}

#[tokio::test]
async fn test_dataset_split() {
    let examples: Vec<Example> = (0..100)
        .map(|i| Example::new(format!("input {}", i), format!("output {}", i)))
        .collect();

    let config = DatasetConfig::new("split-test");
    let dataset = Dataset::new(config, examples);

    // Default split: 80/10/10
    let split = dataset.split(SplitConfig::default());

    assert_eq!(split.train.len(), 80);
    assert_eq!(split.validation.len(), 10);
    assert_eq!(split.test.len(), 10);
    assert_eq!(split.total(), 100);
}

#[tokio::test]
async fn test_dataset_custom_split() {
    let examples: Vec<Example> = (0..100)
        .map(|i| Example::new(format!("input {}", i), format!("output {}", i)))
        .collect();

    let config = DatasetConfig::new("custom-split-test");
    let dataset = Dataset::new(config, examples);

    let split_config = SplitConfig {
        train_ratio: 0.7,
        val_ratio: 0.2,
        shuffle: false,
    };
    let split = dataset.split(split_config);

    assert_eq!(split.train.len(), 70);
    assert_eq!(split.validation.len(), 20);
    assert_eq!(split.test.len(), 10);
}

#[tokio::test]
async fn test_dataset_filter() {
    let examples = vec![
        Example::new("math: 2+2", "4"),
        Example::new("math: 3+3", "6"),
        Example::new("science: H2O", "water"),
        Example::new("science: NaCl", "salt"),
    ];

    let config = DatasetConfig::new("multi-subject");
    let dataset = Dataset::new(config, examples);

    // Filter math only
    let math_only = dataset.filter(|e| e.input.starts_with("math:"));

    assert_eq!(math_only.len(), 2);
    assert!(math_only
        .examples
        .iter()
        .all(|e| e.input.starts_with("math:")));
}

#[tokio::test]
async fn test_dataset_stats_computation() {
    let examples = vec![
        Example::new("input1", "output1"),
        Example::new("input2", "output2").as_synthetic(),
        Example::new("input3", "output3").with_system("system prompt"),
    ];

    let stats = DatasetStats::compute(&examples);

    assert_eq!(stats.example_count, 3);
    assert_eq!(stats.synthetic_count, 1);
    assert_eq!(stats.with_system_count, 1);
    assert!(stats.avg_input_len > 0.0);
    assert!(stats.avg_output_len > 0.0);
}

#[tokio::test]
async fn test_dataset_with_database_persistence() {
    let temp = TempDir::new().expect("temp dir");
    let db = Arc::new(StudioDatabase::in_memory().expect("create db"));
    let manager = DatasetManager::with_database(temp.path().to_path_buf(), db);

    let examples = vec![
        Example::new("Hello", "Hi there!"),
        Example::new("Goodbye", "See you later!"),
    ];

    let config = DatasetConfig::new("db-dataset");
    let dataset = manager.create(config, examples).await.expect("create");

    // Verify count
    assert_eq!(manager.count().await, 1);

    // List
    let list = manager.list().await;
    assert_eq!(list.len(), 1);

    // Delete
    manager.delete(&dataset.id).await.expect("delete");
    assert_eq!(manager.count().await, 0);
}

// ============================================================================
// Experiment Tracking Integration Tests
// ============================================================================

#[tokio::test]
async fn test_experiment_creation_and_runs() {
    let temp = TempDir::new().expect("temp dir");
    let tracker = ExperimentTracker::new(temp.path());

    let config = ExperimentConfig::new("fine-tuning-exp", "llama-3.2-3b", "dataset-123")
        .with_description("Fine-tuning experiment")
        .with_tags(vec!["llama".to_string(), "fine-tuning".to_string()]);

    let experiment = tracker.create_experiment(config).await.expect("create");

    assert!(!experiment.id.is_empty());
    assert_eq!(experiment.config.name, "fine-tuning-exp");
    assert_eq!(experiment.config.base_model, "llama-3.2-3b");
    assert!(experiment.runs.is_empty());
}

#[tokio::test]
async fn test_run_lifecycle() {
    let temp = TempDir::new().expect("temp dir");
    let tracker = ExperimentTracker::new(temp.path());

    let config = ExperimentConfig::new("test-exp", "model", "dataset");
    let experiment = tracker.create_experiment(config).await.expect("create");

    // Start run
    let run = tracker
        .start_run(&experiment.id, Some("run-1".to_string()))
        .expect("start");
    assert_eq!(run.status, RunStatus::Running);

    // Log metrics
    let mut metrics = HashMap::new();
    metrics.insert("loss".to_string(), 1.5);
    metrics.insert("accuracy".to_string(), 0.6);
    tracker
        .log_metrics(&experiment.id, &run.id, 1, metrics.clone())
        .expect("log 1");

    metrics.insert("loss".to_string(), 0.8);
    metrics.insert("accuracy".to_string(), 0.75);
    tracker
        .log_metrics(&experiment.id, &run.id, 2, metrics.clone())
        .expect("log 2");

    // Complete run
    let mut final_metrics = HashMap::new();
    final_metrics.insert("loss".to_string(), 0.3);
    final_metrics.insert("accuracy".to_string(), 0.92);
    tracker
        .complete_run(&experiment.id, &run.id, final_metrics)
        .expect("complete");

    // Verify
    let exp = tracker.get_experiment(&experiment.id).await.expect("get");
    let completed_run = exp.get_run(&run.id).expect("get run");

    assert_eq!(completed_run.status, RunStatus::Completed);
    assert_eq!(completed_run.final_metrics.get("accuracy"), Some(&0.92));
}

#[tokio::test]
async fn test_run_failure() {
    let temp = TempDir::new().expect("temp dir");
    let tracker = ExperimentTracker::new(temp.path());

    let config = ExperimentConfig::new("fail-exp", "model", "dataset");
    let experiment = tracker.create_experiment(config).await.expect("create");

    let run = tracker.start_run(&experiment.id, None).expect("start");
    tracker
        .fail_run(&experiment.id, &run.id, "Out of GPU memory")
        .expect("fail");

    let exp = tracker.get_experiment(&experiment.id).await.expect("get");
    let failed_run = exp.get_run(&run.id).expect("get run");

    assert_eq!(failed_run.status, RunStatus::Failed);
    assert_eq!(
        failed_run.error_message,
        Some("Out of GPU memory".to_string())
    );
}

#[tokio::test]
async fn test_experiment_best_run_tracking() {
    let config = ExperimentConfig::new("best-run-test", "model", "dataset");
    let mut experiment = Experiment::new(config);
    experiment.set_primary_metric("accuracy");

    // Add runs with different accuracies
    for (i, acc) in [0.75, 0.82, 0.91, 0.88].iter().enumerate() {
        let mut run = Run::new(Some(format!("run-{}", i)));
        run.start().expect("start");
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), *acc);
        run.complete(metrics);
        experiment.add_run(run);
    }

    // Best run should be the one with 0.91 accuracy (index 2)
    let best_id = experiment.best_run_id.clone().expect("best run");
    let best_run = experiment.get_run(&best_id).expect("get best");
    assert_eq!(best_run.final_metrics.get("accuracy"), Some(&0.91));
}

#[tokio::test]
async fn test_run_artifacts() {
    let temp = TempDir::new().expect("temp dir");
    let tracker = ExperimentTracker::new(temp.path());

    let config = ExperimentConfig::new("artifact-exp", "model", "dataset");
    let experiment = tracker.create_experiment(config).await.expect("create");
    let run = tracker.start_run(&experiment.id, None).expect("start");

    // Add artifacts
    let checkpoint = Artifact::new("checkpoint-1000", ArtifactType::Checkpoint, "/path/to/ckpt")
        .with_step(1000)
        .with_size(1024 * 1024 * 100)
        .with_metadata("format", "safetensors");

    tracker
        .add_artifact(&experiment.id, &run.id, checkpoint)
        .expect("add artifact");

    let exp = tracker.get_experiment(&experiment.id).await.expect("get");
    let run = exp.get_run(&run.id).expect("get run");

    assert_eq!(run.artifacts.len(), 1);
    assert_eq!(run.artifacts[0].name, "checkpoint-1000");
    assert_eq!(run.artifacts[0].artifact_type, ArtifactType::Checkpoint);
}

#[tokio::test]
async fn test_hyperparameter_tracking() {
    let mut run = Run::new(Some("hyperparam-test".to_string()));

    run.set_hyperparam("learning_rate", HyperparamValue::Float(0.0001));
    run.set_hyperparam("batch_size", HyperparamValue::Int(32));
    run.set_hyperparam("warmup_steps", HyperparamValue::Int(100));
    run.set_hyperparam("optimizer", HyperparamValue::String("adamw".to_string()));
    run.set_hyperparam("use_fp16", HyperparamValue::Bool(true));

    assert_eq!(run.hyperparameters.len(), 5);

    match &run.hyperparameters["learning_rate"] {
        HyperparamValue::Float(v) => assert!((*v - 0.0001).abs() < 0.00001),
        _ => panic!("Expected float"),
    }
}

#[tokio::test]
async fn test_run_comparison() {
    let temp = TempDir::new().expect("temp dir");
    let tracker = ExperimentTracker::new(temp.path());

    // Create two experiments with runs
    let config1 = ExperimentConfig::new("exp-1", "llama-7b", "dataset");
    let exp1 = tracker.create_experiment(config1).await.expect("create 1");

    let run1 = tracker
        .start_run(&exp1.id, Some("run-1".to_string()))
        .expect("start 1");
    let mut metrics1 = HashMap::new();
    metrics1.insert("loss".to_string(), 0.25);
    metrics1.insert("accuracy".to_string(), 0.88);
    tracker
        .complete_run(&exp1.id, &run1.id, metrics1)
        .expect("complete 1");

    let config2 = ExperimentConfig::new("exp-2", "llama-13b", "dataset");
    let exp2 = tracker.create_experiment(config2).await.expect("create 2");

    let run2 = tracker
        .start_run(&exp2.id, Some("run-2".to_string()))
        .expect("start 2");
    let mut metrics2 = HashMap::new();
    metrics2.insert("loss".to_string(), 0.18);
    metrics2.insert("accuracy".to_string(), 0.92);
    tracker
        .complete_run(&exp2.id, &run2.id, metrics2)
        .expect("complete 2");

    // Compare runs
    let comparison = tracker.compare_runs(&[
        (exp1.id.clone(), run1.id.clone()),
        (exp2.id.clone(), run2.id.clone()),
    ]);

    assert_eq!(comparison.runs.len(), 2);
    assert!(comparison.common_metrics.contains(&"loss".to_string()));
    assert!(comparison.common_metrics.contains(&"accuracy".to_string()));

    // Best for accuracy (higher is better)
    let best = comparison.best_for_metric("accuracy", true);
    assert!(best.is_some());
    assert_eq!(best.expect("best").0, "exp-2"); // exp-2 has higher accuracy
}

// ============================================================================
// Prompt Studio Integration Tests
// ============================================================================

#[tokio::test]
async fn test_prompt_template_versioning() {
    let mut template = PromptTemplate::new("code-review")
        .with_description("Code review prompt")
        .with_tags(vec!["code".to_string(), "review".to_string()]);

    // Create first version
    template.create_version("Review this code:\n{{code}}", "Initial version");

    assert_eq!(template.versions.len(), 1);
    assert!(template.active_version_id.is_some());

    // Create second version
    template.create_version(
        "Please review the following {{language}} code and provide feedback:\n```\n{{code}}\n```",
        "Added language support",
    );

    assert_eq!(template.versions.len(), 2);

    // Active is still first
    let active = template.active_version().expect("active");
    assert_eq!(active.version_number, 1);

    // Switch to v2
    let v2_id = template.versions[1].id.clone();
    template.set_active_version(&v2_id).expect("set active");

    let active = template.active_version().expect("active");
    assert_eq!(active.version_number, 2);
}

#[tokio::test]
async fn test_prompt_variable_rendering() {
    let mut template = PromptTemplate::new("greeting");
    template.create_version("Hello {{name}}! Welcome to {{company}}.", "Initial");

    let mut vars = HashMap::new();
    vars.insert("name".to_string(), "Alice".to_string());
    vars.insert("company".to_string(), "Daemoniorum".to_string());

    let rendered = template.render(&vars).expect("render");
    assert_eq!(rendered, "Hello Alice! Welcome to Daemoniorum.");
}

#[tokio::test]
async fn test_prompt_variable_extraction() {
    let mut template = PromptTemplate::new("complex");
    template.create_version(
        "System: {{system_prompt}}\nUser: {{user_input}}\nContext: {{context}}",
        "Multi-variable template",
    );

    let variables = template.variables();

    assert!(variables.contains(&"system_prompt".to_string()));
    assert!(variables.contains(&"user_input".to_string()));
    assert!(variables.contains(&"context".to_string()));
}

#[tokio::test]
async fn test_prompt_studio_crud() {
    let temp = TempDir::new().expect("temp dir");
    let studio = PromptStudio::new(temp.path().to_path_buf());

    // Create template using create_template API
    let template = studio.create_template("qa-prompt").await.expect("create");
    assert!(!template.id.is_empty());

    // Add version
    studio
        .add_version(&template.id, "Answer the question: {{question}}", "v1")
        .await
        .expect("add version");

    // Count
    assert_eq!(studio.count().await, 1);

    // Get template
    let retrieved = studio.get_template(&template.id).await.expect("get");
    assert_eq!(retrieved.name, "qa-prompt");
    assert_eq!(retrieved.versions.len(), 1);

    // List
    let list = studio.list_templates().await;
    assert_eq!(list.len(), 1);

    // Delete
    studio.delete_template(&template.id).expect("delete");
    assert_eq!(studio.count().await, 0);
}

#[tokio::test]
async fn test_prompt_studio_versioning() {
    let temp = TempDir::new().expect("temp dir");
    let studio = PromptStudio::new(temp.path().to_path_buf());

    // Create template
    let template = studio
        .create_template("versioned-prompt")
        .await
        .expect("create");

    // Add multiple versions
    studio
        .add_version(&template.id, "Version 1 content", "Initial")
        .await
        .expect("add v1");
    studio
        .add_version(&template.id, "Version 2 content", "Updated")
        .await
        .expect("add v2");

    // Get and verify
    let retrieved = studio.get_template(&template.id).await.expect("get");
    assert_eq!(retrieved.versions.len(), 2);
    assert_eq!(retrieved.versions[0].version_number, 1);
    assert_eq!(retrieved.versions[1].version_number, 2);
}

// ============================================================================
// Model Registry Integration Tests
// ============================================================================

#[tokio::test]
async fn test_model_registration() {
    let temp = TempDir::new().expect("temp dir");
    let registry = ModelRegistry::new(temp.path().to_path_buf());

    let model = Model::new(
        "llama-finetuned",
        "meta-llama/Llama-3.2-3B",
        "text-generation",
    )
    .with_description("Fine-tuned Llama model for code")
    .with_owner("team-ml")
    .with_tags(vec!["llama".to_string(), "code".to_string()]);

    let model_id = registry.register_model(model).expect("register");

    assert!(!model_id.is_empty());

    // Get model and verify
    let retrieved = registry.get_model(&model_id).expect("get");
    assert_eq!(retrieved.name, "llama-finetuned");
    assert_eq!(retrieved.task_type, "text-generation");
}

#[tokio::test]
async fn test_model_versioning() {
    let temp = TempDir::new().expect("temp dir");
    let registry = ModelRegistry::new(temp.path().to_path_buf());

    let model = Model::new("versioned-model", "base-model", "classification");
    let model_id = registry.register_model(model).expect("register");

    // Add version 1
    let metadata1 = ModelMetadata::new().with_format("safetensors");

    registry
        .create_version(&model_id, metadata1)
        .expect("add v1");

    // Add version 2
    let metadata2 = ModelMetadata::new().with_format("gguf");

    registry
        .create_version(&model_id, metadata2)
        .expect("add v2");

    // Check versions
    let model = registry.get_model(&model_id).expect("get");
    assert_eq!(model.versions.len(), 2);

    let latest = model.latest_version().expect("latest");
    assert_eq!(latest.version, 2);
}

#[tokio::test]
async fn test_model_stage_transitions() {
    let temp = TempDir::new().expect("temp dir");
    let registry = ModelRegistry::new(temp.path().to_path_buf());

    let model = Model::new("staged-model", "base", "generation");
    let model_id = registry.register_model(model).expect("register");

    let metadata = ModelMetadata::new();
    registry
        .create_version(&model_id, metadata)
        .expect("add version");

    // Transition: Development -> Staging
    registry
        .transition_stage(&model_id, 1, ModelStage::Staging, None)
        .expect("to staging");

    let model = registry.get_model(&model_id).expect("get");
    let v1 = model.get_version(1).expect("v1");
    assert_eq!(v1.stage, ModelStage::Staging);

    // Transition: Staging -> Production
    registry
        .transition_stage(&model_id, 1, ModelStage::Production, None)
        .expect("to prod");

    let model = registry.get_model(&model_id).expect("get");
    let v1 = model.get_version(1).expect("v1");
    assert_eq!(v1.stage, ModelStage::Production);
}

#[tokio::test]
async fn test_model_registry_search() {
    let temp = TempDir::new().expect("temp dir");
    let registry = ModelRegistry::new(temp.path().to_path_buf());

    // Register multiple models
    let model1 = Model::new("llama-chat", "llama", "chat")
        .with_tags(vec!["llama".to_string(), "chat".to_string()]);
    registry.register_model(model1).expect("register 1");

    let model2 = Model::new("llama-code", "llama", "code")
        .with_tags(vec!["llama".to_string(), "code".to_string()]);
    registry.register_model(model2).expect("register 2");

    let model3 = Model::new("qwen-chat", "qwen", "chat")
        .with_tags(vec!["qwen".to_string(), "chat".to_string()]);
    registry.register_model(model3).expect("register 3");

    // List all
    let all = registry.list_models();
    assert_eq!(all.len(), 3);

    // Filter by tag
    let llama_models: Vec<_> = all
        .iter()
        .filter(|m| m.tags.contains(&"llama".to_string()))
        .collect();
    assert_eq!(llama_models.len(), 2);

    // Filter by task
    let chat_models: Vec<_> = all.iter().filter(|m| m.task_type == "chat").collect();
    assert_eq!(chat_models.len(), 2);
}

// ============================================================================
// Database Persistence Integration Tests
// ============================================================================

#[tokio::test]
async fn test_database_persistence_workflow() {
    let temp = TempDir::new().expect("temp dir");
    let db_path = temp.path().join("studio.db");

    let dataset_id;

    // Create data with first instance
    {
        let db = Arc::new(StudioDatabase::new(&db_path).expect("create db"));

        let dataset_manager =
            DatasetManager::with_database(temp.path().join("datasets"), db.clone());

        let examples = vec![Example::new("input", "output")];
        let config = DatasetConfig::new("persistent-dataset");
        let dataset = dataset_manager
            .create(config, examples)
            .await
            .expect("create");
        dataset_id = dataset.id.clone();

        let tracker = ExperimentTracker::with_database(temp.path().join("experiments"), db.clone());

        let exp_config = ExperimentConfig::new("persistent-exp", "model", &dataset_id);
        let _experiment = tracker
            .create_experiment(exp_config)
            .await
            .expect("create exp");
    }

    // Verify data persists with new instance
    {
        let db = Arc::new(StudioDatabase::new(&db_path).expect("reopen db"));

        let dataset_manager =
            DatasetManager::with_database(temp.path().join("datasets"), db.clone());

        assert_eq!(dataset_manager.count().await, 1);

        let loaded = dataset_manager.get(&dataset_id).await.expect("get");
        assert_eq!(loaded.name, "persistent-dataset");
    }
}

#[tokio::test]
async fn test_in_memory_database() {
    let db = StudioDatabase::in_memory().expect("create in-memory db");

    // Verify basic operations work
    let examples = vec![Example::new("test", "test")];
    let dataset = Dataset::new(DatasetConfig::new("memory-test"), examples);

    db.save_dataset(&dataset).expect("save");

    let loaded = db.load_dataset(&dataset.id).expect("load").expect("exists");
    assert_eq!(loaded.name, "memory-test");

    db.delete_dataset(&dataset.id).expect("delete");

    let missing = db.load_dataset(&dataset.id).expect("load");
    assert!(missing.is_none());
}

// ============================================================================
// Agent Integration Tests
// ============================================================================

#[test]
fn test_data_curator_agent_creation() {
    let agent = DataCuratorAgent::new(Some("test-model".to_string()));
    // Agent should be created without panicking
    drop(agent);
}

#[test]
fn test_training_coach_agent_creation() {
    let agent = TrainingCoachAgent::new(None);
    // Agent should be created without panicking
    drop(agent);
}

#[test]
fn test_eval_analyst_agent_creation() {
    let agent = EvalAnalystAgent::new(Some("qwen-7b".to_string()));
    // Agent should be created without panicking
    drop(agent);
}

#[test]
fn test_hyperparam_optimizer_agent_creation() {
    let agent = HyperparamOptimizerAgent::new(None);
    // Agent should be created without panicking
    drop(agent);
}

// ============================================================================
// Full Workflow Integration Tests
// ============================================================================

#[tokio::test]
async fn test_complete_fine_tuning_workflow() {
    let temp = TempDir::new().expect("temp dir");
    let config = StudioConfig::with_base_dir(temp.path().to_path_buf()).without_agents(); // Disable agents for faster test

    let studio = Studio::new(config).await.expect("create studio");

    // 1. Create dataset
    let examples: Vec<Example> = (0..50)
        .map(|i| {
            Example::new(
                format!("Explain concept {}", i),
                format!("Concept {} is about...", i),
            )
        })
        .collect();

    let dataset_config = DatasetConfig::new("training-data")
        .with_description("Training examples")
        .with_tags(vec!["training"]);

    let dataset = studio
        .datasets()
        .create(dataset_config, examples)
        .await
        .expect("create dataset");

    assert!(dataset.validation.as_ref().map_or(false, |v| v.passed));

    // 2. Create experiment
    let exp_config = ExperimentConfig::new("fine-tune-exp", "llama-3b", &dataset.id)
        .with_description("Fine-tuning experiment");

    let experiment = studio
        .experiments()
        .create_experiment(exp_config)
        .await
        .expect("create experiment");

    // 3. Run training
    let run = studio
        .experiments()
        .start_run(&experiment.id, Some("run-1".to_string()))
        .expect("start run");

    // Simulate training loop
    for step in 0..10 {
        let mut metrics = HashMap::new();
        metrics.insert("loss".to_string(), 1.0 - (step as f64 * 0.08));
        metrics.insert("accuracy".to_string(), 0.5 + (step as f64 * 0.04));

        studio
            .experiments()
            .log_metrics(&experiment.id, &run.id, step, metrics)
            .expect("log metrics");
    }

    // Complete training
    let mut final_metrics = HashMap::new();
    final_metrics.insert("loss".to_string(), 0.2);
    final_metrics.insert("accuracy".to_string(), 0.9);

    studio
        .experiments()
        .complete_run(&experiment.id, &run.id, final_metrics)
        .expect("complete run");

    // 4. Register model
    let model = Model::new("fine-tuned-llama", "llama-3b", "text-generation")
        .with_description("Fine-tuned on custom data");

    let model_id = studio
        .models()
        .register_model(model)
        .expect("register model");

    let metadata = ModelMetadata::new();
    studio
        .models()
        .create_version(&model_id, metadata)
        .expect("add version");

    // 5. Verify final state
    let stats = studio.stats().await;
    assert_eq!(stats.datasets_count, 1);
    assert_eq!(stats.experiments_count, 1);
    assert_eq!(stats.models_count, 1);
}

#[tokio::test]
async fn test_prompt_ab_testing_workflow() {
    let temp = TempDir::new().expect("temp dir");
    let prompt_studio = PromptStudio::new(temp.path().to_path_buf());

    // Create template with multiple versions for A/B testing
    let template = prompt_studio
        .create_template("summarization")
        .await
        .expect("create");

    // Version A
    prompt_studio
        .add_version(
            &template.id,
            "Summarize the following text:\n{{text}}",
            "Simple summarization prompt",
        )
        .await
        .expect("add v1");

    // Version B
    prompt_studio.add_version(
        &template.id,
        "You are an expert summarizer. Create a concise summary of the following text, highlighting key points:\n\n{{text}}",
        "Expert persona prompt"
    ).await.expect("add v2");

    // Test both versions
    let mut vars = HashMap::new();
    vars.insert(
        "text".to_string(),
        "Long article content here...".to_string(),
    );

    // Get template
    let mut template = prompt_studio.get_template(&template.id).await.expect("get");

    // Render V1 (already active)
    let v1_prompt = template.render(&vars).expect("render v1");
    assert!(v1_prompt.contains("Summarize the following"));

    // Switch to V2 and render
    let v2_id = template.versions[1].id.clone();
    template.set_active_version(&v2_id).expect("set active");
    let v2_prompt = template.render(&vars).expect("render v2");
    assert!(v2_prompt.contains("expert summarizer"));
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test]
async fn test_dataset_not_found() {
    let temp = TempDir::new().expect("temp dir");
    let manager = DatasetManager::new(temp.path().to_path_buf());

    let result = manager.get("nonexistent-id").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_experiment_not_found() {
    let temp = TempDir::new().expect("temp dir");
    let tracker = ExperimentTracker::new(temp.path());

    let result = tracker.start_run("nonexistent-exp", None);
    assert!(result.is_err());
}

#[tokio::test]
async fn test_run_invalid_state_transition() {
    let mut run = Run::new(None);

    // Try to start twice
    run.start().expect("first start");
    let result = run.start();
    assert!(result.is_err());
}

#[tokio::test]
async fn test_prompt_missing_variable() {
    let mut template = PromptTemplate::new("test");
    template.create_version("Hello {{name}}, your order {{order_id}} is ready.", "v1");

    let mut vars = HashMap::new();
    vars.insert("name".to_string(), "Alice".to_string());
    // Missing order_id

    let result = template.render(&vars);
    assert!(result.is_err());
}

// ============================================================================
// Serialization Tests
// ============================================================================

#[test]
fn test_example_serialization() {
    let example = Example::new("What is Rust?", "Rust is a systems programming language.")
        .with_system("You are a programming tutor.")
        .with_metadata("topic", serde_json::json!("programming"));

    let json = serde_json::to_string(&example).expect("serialize");
    let parsed: Example = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(parsed.input, example.input);
    assert_eq!(parsed.output, example.output);
    assert_eq!(parsed.system, example.system);
}

#[test]
fn test_run_serialization() {
    let mut run = Run::new(Some("test-run".to_string()));
    run.set_hyperparam("lr", HyperparamValue::Float(0.001));
    run.log_metric("loss", 1, 0.5);

    let json = serde_json::to_string(&run).expect("serialize");
    let parsed: Run = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(parsed.name, run.name);
    assert_eq!(parsed.hyperparameters.len(), 1);
}

#[test]
fn test_dataset_format_serialization() {
    for format in [
        DatasetFormat::JsonLines,
        DatasetFormat::JsonArray,
        DatasetFormat::Csv,
        DatasetFormat::Alpaca,
        DatasetFormat::ShareGpt,
        DatasetFormat::OpenAI,
    ] {
        let json = serde_json::to_string(&format).expect("serialize");
        let parsed: DatasetFormat = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, format);
    }
}

#[test]
fn test_run_status_serialization() {
    for status in [
        RunStatus::Pending,
        RunStatus::Running,
        RunStatus::Paused,
        RunStatus::Completed,
        RunStatus::Failed,
        RunStatus::Cancelled,
    ] {
        let json = serde_json::to_string(&status).expect("serialize");
        let parsed: RunStatus = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, status);
    }
}

#[test]
fn test_model_stage_serialization() {
    for stage in [
        ModelStage::Development,
        ModelStage::Staging,
        ModelStage::Production,
        ModelStage::Archived,
    ] {
        let json = serde_json::to_string(&stage).expect("serialize");
        let parsed: ModelStage = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, stage);
    }
}
