//! Hyperparameter Optimizer Agent - Suggests optimal training configurations.
//!
//! The Hyperparam Optimizer agent helps with:
//! - Analyzing datasets to suggest appropriate hyperparameters
//! - Recommending LoRA configurations
//! - Providing confidence intervals for suggestions
//! - Learning from past experiments

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::{info, info_span, warn};

use crate::dataset::{Dataset, DatasetStats};
use crate::llm::{LlmClient, GenerateRequest, Message};
use super::{AgentError, Result};

/// System prompt for the Hyperparameter Optimizer agent.
const OPTIMIZER_SYSTEM_PROMPT: &str = r#"You are an expert ML Hyperparameter Optimizer specializing in LLM fine-tuning.

Your role is to analyze datasets and suggest optimal training configurations. Consider:

1. Dataset size and characteristics
2. Model architecture requirements
3. Training stability and convergence
4. Resource constraints

Always respond with valid JSON in this exact format:
{
  "approach": "few_shot" | "lora" | "lora_plus" | "full_fine_tuning",
  "learning_rate": 0.0002,
  "batch_size": 8,
  "epochs": 3,
  "lora_rank": 16,
  "lora_alpha": 32,
  "warmup_ratio": 0.1,
  "weight_decay": 0.01,
  "confidence": 0.85,
  "reasoning": "Explanation of recommendations",
  "key_insights": ["insight 1", "insight 2"],
  "warnings": ["potential issue 1"]
}"#;

/// The Hyperparameter Optimizer agent.
pub struct HyperparamOptimizerAgent {
    /// Optional LLM client for intelligent suggestions.
    llm_client: Option<Arc<dyn LlmClient>>,
}

impl HyperparamOptimizerAgent {
    /// Creates a new Hyperparam Optimizer agent without LLM (uses heuristics only).
    pub fn new(_model: Option<String>) -> Self {
        Self { llm_client: None }
    }

    /// Creates a new Hyperparam Optimizer agent with an LLM client for intelligent suggestions.
    pub fn with_llm(llm_client: Arc<dyn LlmClient>) -> Self {
        Self {
            llm_client: Some(llm_client),
        }
    }

    /// Returns whether this optimizer has LLM capabilities.
    pub fn has_llm(&self) -> bool {
        self.llm_client.is_some()
    }

    /// Suggests hyperparameters based on dataset analysis.
    ///
    /// If an LLM client is available, uses it for intelligent suggestions.
    /// Otherwise, falls back to heuristic-based recommendations.
    pub async fn suggest_hyperparams(
        &self,
        dataset: &Dataset,
        base_model: &str,
    ) -> Result<HyperparamSuggestion> {
        let _span = info_span!(
            "optimizer.suggest",
            dataset = %dataset.name,
            examples = dataset.len(),
            model = %base_model
        ).entered();

        info!("Analyzing dataset for hyperparameter suggestions");

        // Try LLM analysis first if available
        if let Some(ref llm) = self.llm_client {
            match self.suggest_with_llm(llm, dataset, base_model).await {
                Ok(suggestion) => return Ok(suggestion),
                Err(e) => {
                    warn!("LLM suggestion failed, falling back to heuristics: {}", e);
                }
            }
        }

        // Fall back to heuristic suggestions
        self.suggest_heuristic(dataset, base_model).await
    }

    /// Suggests hyperparameters using LLM for intelligent recommendations.
    async fn suggest_with_llm(
        &self,
        llm: &Arc<dyn LlmClient>,
        dataset: &Dataset,
        base_model: &str,
    ) -> Result<HyperparamSuggestion> {
        let prompt = self.build_suggestion_prompt(dataset, base_model);

        let request = GenerateRequest::new(vec![
            Message::system(OPTIMIZER_SYSTEM_PROMPT),
            Message::user(prompt),
        ])
        .with_temperature(0.3)
        .with_max_tokens(1024);

        let response = llm.generate(request).await
            .map_err(|e| AgentError::Llm(format!("LLM request failed: {}", e)))?;

        self.parse_suggestion_response(&response.content, &dataset.stats)
    }

    /// Builds the suggestion prompt from dataset info.
    fn build_suggestion_prompt(&self, dataset: &Dataset, base_model: &str) -> String {
        let stats = &dataset.stats;
        let mut prompt = format!(
            "Suggest optimal hyperparameters for fine-tuning '{}' on this dataset:\n\n",
            base_model
        );

        prompt.push_str(&format!("Dataset: {}\n", dataset.name));
        prompt.push_str(&format!("Examples: {}\n", stats.example_count));
        prompt.push_str(&format!("Average input length: {:.0} chars\n", stats.avg_input_len));
        prompt.push_str(&format!("Average output length: {:.0} chars\n", stats.avg_output_len));
        prompt.push_str(&format!("Total characters: {}\n", stats.total_chars));
        prompt.push_str(&format!("Synthetic examples: {}\n", stats.synthetic_count));
        prompt.push_str(&format!("With system prompts: {}\n", stats.with_system_count));

        prompt.push_str("\nProvide your hyperparameter suggestions as JSON.");
        prompt
    }

    /// Parses the LLM response into HyperparamSuggestion.
    fn parse_suggestion_response(
        &self,
        content: &str,
        stats: &DatasetStats,
    ) -> Result<HyperparamSuggestion> {
        // Extract JSON from response
        let json_str = if let Some(start) = content.find('{') {
            if let Some(end) = content.rfind('}') {
                &content[start..=end]
            } else {
                content
            }
        } else {
            content
        };

        let parsed: LlmSuggestionResponse = serde_json::from_str(json_str)
            .map_err(|e| AgentError::Analysis(format!("Failed to parse LLM response: {}", e)))?;

        // Convert approach string to enum
        let approach = match parsed.approach.as_str() {
            "few_shot" => TrainingApproach::FewShot,
            "lora" => TrainingApproach::LoRA,
            "lora_plus" => TrainingApproach::LoRAPlus,
            "full_fine_tuning" => TrainingApproach::FullFineTuning,
            _ => self.determine_approach(stats), // Fallback
        };

        // Build LoRA config from LLM response
        let lora_config = if approach != TrainingApproach::FewShot && approach != TrainingApproach::FullFineTuning {
            Some(LoRAConfig {
                rank: parsed.lora_rank.unwrap_or(16),
                alpha: parsed.lora_alpha.unwrap_or(32),
                dropout: 0.05,
                target_modules: vec![
                    "q_proj".to_string(),
                    "v_proj".to_string(),
                    "k_proj".to_string(),
                    "o_proj".to_string(),
                ],
            })
        } else {
            None
        };

        // Generate reasoning with LLM insights
        let mut reasoning = parsed.reasoning;
        if !parsed.key_insights.is_empty() {
            reasoning.push_str("\n\nKey insights: ");
            reasoning.push_str(&parsed.key_insights.join(", "));
        }
        if !parsed.warnings.is_empty() {
            reasoning.push_str("\n\nWarnings: ");
            reasoning.push_str(&parsed.warnings.join(", "));
        }

        Ok(HyperparamSuggestion {
            approach,
            learning_rate: parsed.learning_rate,
            batch_size: parsed.batch_size,
            epochs: parsed.epochs,
            lora_config,
            warmup_ratio: parsed.warmup_ratio.unwrap_or(0.1),
            weight_decay: parsed.weight_decay.unwrap_or(0.01),
            gradient_accumulation_steps: self.suggest_gradient_accumulation(parsed.batch_size),
            max_grad_norm: 1.0,
            confidence: parsed.confidence.unwrap_or(0.7),
            reasoning,
            alternatives: self.generate_alternatives(stats),
        })
    }

    /// Heuristic-based hyperparameter suggestion.
    async fn suggest_heuristic(
        &self,
        dataset: &Dataset,
        base_model: &str,
    ) -> Result<HyperparamSuggestion> {
        let stats = &dataset.stats;

        // Determine training approach based on dataset size
        let approach = self.determine_approach(stats);

        // Calculate learning rate
        let learning_rate = self.suggest_learning_rate(stats, &approach);

        // Calculate batch size
        let batch_size = self.suggest_batch_size(stats);

        // Calculate epochs
        let epochs = self.suggest_epochs(stats);

        // LoRA configuration
        let lora_config = self.suggest_lora_config(stats, base_model);

        Ok(HyperparamSuggestion {
            approach,
            learning_rate,
            batch_size,
            epochs,
            lora_config: Some(lora_config),
            warmup_ratio: 0.1,
            weight_decay: 0.01,
            gradient_accumulation_steps: self.suggest_gradient_accumulation(batch_size),
            max_grad_norm: 1.0,
            confidence: self.calculate_confidence(stats),
            reasoning: self.generate_reasoning(stats),
            alternatives: self.generate_alternatives(stats),
        })
    }

    /// Suggests learning rate schedule.
    pub async fn suggest_lr_schedule(
        &self,
        dataset_size: usize,
        epochs: u32,
    ) -> Result<LRScheduleSuggestion> {
        let _span = info_span!("optimizer.lr_schedule", size = dataset_size, epochs = epochs).entered();

        let schedule = if epochs > 5 {
            LRScheduleType::CosineAnnealing
        } else if dataset_size > 10000 {
            LRScheduleType::LinearWarmupDecay
        } else {
            LRScheduleType::Constant
        };

        Ok(LRScheduleSuggestion {
            schedule_type: schedule,
            warmup_steps: (dataset_size as f32 * 0.1) as usize,
            min_lr_ratio: 0.1,
            reasoning: format!(
                "For {} examples over {} epochs, {} schedule provides optimal convergence",
                dataset_size, epochs, schedule.name()
            ),
        })
    }

    /// Analyzes past experiments to improve suggestions.
    pub async fn learn_from_experiments(
        &self,
        experiments: &[ExperimentOutcome],
    ) -> Result<LearnedInsights> {
        let _span = info_span!("optimizer.learn", experiments = experiments.len()).entered();

        info!("Learning from {} past experiments", experiments.len());

        let mut insights = Vec::new();
        let mut best_configs: HashMap<String, HyperparamConfig> = HashMap::new();

        // Find best configurations per metric
        for exp in experiments {
            if exp.success {
                let metric_key = exp.primary_metric.clone();
                let current_best = best_configs.get(&metric_key);

                if current_best.is_none() || exp.metric_value > current_best.map_or(0.0, |c| c.achieved_metric) {
                    best_configs.insert(metric_key, HyperparamConfig {
                        learning_rate: exp.config.learning_rate,
                        batch_size: exp.config.batch_size,
                        epochs: exp.config.epochs,
                        achieved_metric: exp.metric_value,
                    });
                }
            }
        }

        // Generate insights
        if !best_configs.is_empty() {
            let avg_lr: f64 = best_configs.values()
                .map(|c| c.learning_rate)
                .sum::<f64>() / best_configs.len() as f64;

            insights.push(format!(
                "Successful experiments averaged {:.2e} learning rate",
                avg_lr
            ));
        }

        // Analyze failure patterns
        let failures: Vec<_> = experiments.iter()
            .filter(|e| !e.success)
            .collect();

        if !failures.is_empty() {
            let high_lr_failures = failures.iter()
                .filter(|e| e.config.learning_rate > 1e-3)
                .count();

            if high_lr_failures > failures.len() / 2 {
                insights.push("High learning rates (>1e-3) frequently cause failures".to_string());
            }
        }

        Ok(LearnedInsights {
            total_experiments: experiments.len(),
            successful_experiments: experiments.iter().filter(|e| e.success).count(),
            best_configs,
            insights,
        })
    }

    /// Determines the training approach based on dataset size.
    fn determine_approach(&self, stats: &DatasetStats) -> TrainingApproach {
        match stats.example_count {
            0..=100 => TrainingApproach::FewShot,
            101..=1000 => TrainingApproach::LoRA,
            1001..=10000 => TrainingApproach::LoRAPlus,
            _ => TrainingApproach::FullFineTuning,
        }
    }

    /// Suggests learning rate based on dataset characteristics.
    fn suggest_learning_rate(&self, stats: &DatasetStats, approach: &TrainingApproach) -> f64 {
        let base_lr = match approach {
            TrainingApproach::FewShot => 5e-5,
            TrainingApproach::LoRA => 2e-4,
            TrainingApproach::LoRAPlus => 1e-4,
            TrainingApproach::FullFineTuning => 5e-5,
        };

        // Adjust based on dataset size
        let size_factor = if stats.example_count > 5000 {
            0.8
        } else if stats.example_count < 500 {
            1.2
        } else {
            1.0
        };

        base_lr * size_factor
    }

    /// Suggests batch size.
    fn suggest_batch_size(&self, stats: &DatasetStats) -> u32 {
        // Base on average example length
        if stats.avg_input_len + stats.avg_output_len > 1000.0 {
            2 // Long examples need smaller batches
        } else if stats.avg_input_len + stats.avg_output_len > 500.0 {
            4
        } else {
            8
        }
    }

    /// Suggests number of epochs.
    fn suggest_epochs(&self, stats: &DatasetStats) -> u32 {
        match stats.example_count {
            0..=100 => 10,
            101..=500 => 5,
            501..=2000 => 3,
            _ => 2,
        }
    }

    /// Suggests LoRA configuration.
    fn suggest_lora_config(&self, stats: &DatasetStats, _base_model: &str) -> LoRAConfig {
        // Rank based on dataset complexity
        let rank = if stats.example_count > 5000 {
            32
        } else if stats.example_count > 1000 {
            16
        } else {
            8
        };

        LoRAConfig {
            rank,
            alpha: rank * 2,
            dropout: 0.05,
            target_modules: vec![
                "q_proj".to_string(),
                "v_proj".to_string(),
                "k_proj".to_string(),
                "o_proj".to_string(),
            ],
        }
    }

    /// Suggests gradient accumulation steps.
    fn suggest_gradient_accumulation(&self, batch_size: u32) -> u32 {
        // Target effective batch size of 32
        (32 / batch_size).max(1)
    }

    /// Calculates confidence in the suggestion.
    fn calculate_confidence(&self, stats: &DatasetStats) -> f32 {
        let mut confidence: f32 = 0.7; // Base confidence

        // More data = more confidence
        if stats.example_count > 1000 {
            confidence += 0.1;
        }
        if stats.example_count > 5000 {
            confidence += 0.1;
        }

        // Validated data = more confidence
        if stats.example_count > 0 {
            let synthetic_ratio = stats.synthetic_count as f32 / stats.example_count as f32;
            if synthetic_ratio < 0.5 {
                confidence += 0.05;
            }
        }

        confidence.min(0.95)
    }

    /// Generates reasoning for the suggestion.
    fn generate_reasoning(&self, stats: &DatasetStats) -> String {
        format!(
            "Based on {} examples with average input length of {:.0} chars, \
             {} synthetic examples, and {} with system prompts. \
             These parameters balance training efficiency with model quality.",
            stats.example_count,
            stats.avg_input_len,
            stats.synthetic_count,
            stats.with_system_count
        )
    }

    /// Generates alternative configurations.
    fn generate_alternatives(&self, stats: &DatasetStats) -> Vec<AlternativeConfig> {
        let mut alternatives = Vec::new();

        // Conservative alternative
        alternatives.push(AlternativeConfig {
            name: "Conservative".to_string(),
            description: "Lower learning rate, more epochs for stability".to_string(),
            changes: vec![
                ("learning_rate".to_string(), "0.5x".to_string()),
                ("epochs".to_string(), "1.5x".to_string()),
            ],
        });

        // Aggressive alternative
        if stats.example_count > 1000 {
            alternatives.push(AlternativeConfig {
                name: "Aggressive".to_string(),
                description: "Higher learning rate, fewer epochs for speed".to_string(),
                changes: vec![
                    ("learning_rate".to_string(), "2x".to_string()),
                    ("epochs".to_string(), "0.7x".to_string()),
                ],
            });
        }

        alternatives
    }
}

/// Suggested hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparamSuggestion {
    /// Recommended training approach.
    pub approach: TrainingApproach,
    /// Learning rate.
    pub learning_rate: f64,
    /// Batch size.
    pub batch_size: u32,
    /// Number of epochs.
    pub epochs: u32,
    /// LoRA configuration (if applicable).
    pub lora_config: Option<LoRAConfig>,
    /// Warmup ratio.
    pub warmup_ratio: f32,
    /// Weight decay.
    pub weight_decay: f32,
    /// Gradient accumulation steps.
    pub gradient_accumulation_steps: u32,
    /// Max gradient norm.
    pub max_grad_norm: f32,
    /// Confidence in suggestion (0.0 - 1.0).
    pub confidence: f32,
    /// Reasoning behind suggestion.
    pub reasoning: String,
    /// Alternative configurations.
    pub alternatives: Vec<AlternativeConfig>,
}

/// Training approach.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingApproach {
    /// Few-shot learning with minimal data.
    FewShot,
    /// Standard LoRA fine-tuning.
    LoRA,
    /// Enhanced LoRA with higher rank.
    LoRAPlus,
    /// Full model fine-tuning.
    FullFineTuning,
}

/// LoRA configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    /// LoRA rank.
    pub rank: u32,
    /// LoRA alpha (scaling factor).
    pub alpha: u32,
    /// Dropout rate.
    pub dropout: f32,
    /// Target modules.
    pub target_modules: Vec<String>,
}

/// Learning rate schedule suggestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LRScheduleSuggestion {
    /// Schedule type.
    pub schedule_type: LRScheduleType,
    /// Warmup steps.
    pub warmup_steps: usize,
    /// Minimum LR ratio.
    pub min_lr_ratio: f32,
    /// Reasoning.
    pub reasoning: String,
}

/// Types of learning rate schedules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LRScheduleType {
    /// Constant learning rate.
    Constant,
    /// Linear warmup then decay.
    LinearWarmupDecay,
    /// Cosine annealing.
    CosineAnnealing,
    /// Cosine with restarts.
    CosineWithRestarts,
}

impl LRScheduleType {
    /// Returns the schedule name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::LinearWarmupDecay => "linear warmup + decay",
            Self::CosineAnnealing => "cosine annealing",
            Self::CosineWithRestarts => "cosine with restarts",
        }
    }
}

/// Alternative configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativeConfig {
    /// Configuration name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Changes from main suggestion.
    pub changes: Vec<(String, String)>,
}

/// Outcome of a past experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentOutcome {
    /// Experiment ID.
    pub experiment_id: String,
    /// Whether it succeeded.
    pub success: bool,
    /// Primary metric name.
    pub primary_metric: String,
    /// Metric value achieved.
    pub metric_value: f32,
    /// Configuration used.
    pub config: ExperimentConfig,
}

/// Configuration from a past experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// Learning rate used.
    pub learning_rate: f64,
    /// Batch size used.
    pub batch_size: u32,
    /// Epochs trained.
    pub epochs: u32,
}

/// Configuration that achieved a certain metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparamConfig {
    /// Learning rate.
    pub learning_rate: f64,
    /// Batch size.
    pub batch_size: u32,
    /// Epochs.
    pub epochs: u32,
    /// Achieved metric value.
    pub achieved_metric: f32,
}

/// Insights learned from past experiments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedInsights {
    /// Total experiments analyzed.
    pub total_experiments: usize,
    /// Successful experiments.
    pub successful_experiments: usize,
    /// Best configurations per metric.
    pub best_configs: HashMap<String, HyperparamConfig>,
    /// Textual insights.
    pub insights: Vec<String>,
}

/// Response structure for LLM hyperparameter suggestions (internal).
#[derive(Debug, Clone, Deserialize)]
struct LlmSuggestionResponse {
    /// Training approach.
    approach: String,
    /// Learning rate.
    learning_rate: f64,
    /// Batch size.
    batch_size: u32,
    /// Number of epochs.
    epochs: u32,
    /// LoRA rank.
    lora_rank: Option<u32>,
    /// LoRA alpha.
    lora_alpha: Option<u32>,
    /// Warmup ratio.
    warmup_ratio: Option<f32>,
    /// Weight decay.
    weight_decay: Option<f32>,
    /// Confidence level.
    confidence: Option<f32>,
    /// Reasoning explanation.
    reasoning: String,
    /// Key insights.
    #[serde(default)]
    key_insights: Vec<String>,
    /// Potential warnings.
    #[serde(default)]
    warnings: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{DatasetConfig, Example};

    fn sample_dataset() -> Dataset {
        let examples: Vec<Example> = (0..500)
            .map(|i| Example::new(
                format!("Question {}: What is the answer?", i),
                format!("The answer to question {} is 42.", i),
            ))
            .collect();

        Dataset::new(DatasetConfig::new("test"), examples)
    }

    #[tokio::test]
    async fn test_suggest_hyperparams() {
        let optimizer = HyperparamOptimizerAgent::new(None);
        let dataset = sample_dataset();

        let suggestion = optimizer.suggest_hyperparams(&dataset, "llama-7b").await.expect("suggest");

        assert!(suggestion.learning_rate > 0.0);
        assert!(suggestion.batch_size > 0);
        assert!(suggestion.epochs > 0);
        assert!(suggestion.confidence > 0.0);
        assert!(suggestion.lora_config.is_some());
    }

    #[tokio::test]
    async fn test_suggest_lr_schedule() {
        let optimizer = HyperparamOptimizerAgent::new(None);

        let suggestion = optimizer.suggest_lr_schedule(5000, 3).await.expect("schedule");

        assert!(suggestion.warmup_steps > 0);
    }

    #[tokio::test]
    async fn test_learn_from_experiments() {
        let optimizer = HyperparamOptimizerAgent::new(None);

        let experiments = vec![
            ExperimentOutcome {
                experiment_id: "exp1".to_string(),
                success: true,
                primary_metric: "accuracy".to_string(),
                metric_value: 0.85,
                config: ExperimentConfig {
                    learning_rate: 2e-4,
                    batch_size: 8,
                    epochs: 3,
                },
            },
            ExperimentOutcome {
                experiment_id: "exp2".to_string(),
                success: false,
                primary_metric: "accuracy".to_string(),
                metric_value: 0.3,
                config: ExperimentConfig {
                    learning_rate: 1e-2, // Too high
                    batch_size: 8,
                    epochs: 3,
                },
            },
        ];

        let insights = optimizer.learn_from_experiments(&experiments).await.expect("learn");

        assert_eq!(insights.total_experiments, 2);
        assert_eq!(insights.successful_experiments, 1);
    }

    #[tokio::test]
    async fn test_optimizer_with_llm_suggests_hyperparams() {
        use crate::llm::MockLlmClient;

        // Mock LLM returns hyperparameter suggestions
        let llm = Arc::new(MockLlmClient::new().with_json(serde_json::json!({
            "approach": "lora",
            "learning_rate": 0.0002,
            "batch_size": 8,
            "epochs": 3,
            "lora_rank": 16,
            "lora_alpha": 32,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "confidence": 0.85,
            "reasoning": "Based on dataset size and complexity, LoRA is recommended",
            "key_insights": ["Dataset has good diversity", "Medium sequence lengths allow larger batches"],
            "warnings": ["Watch for overfitting after epoch 2"]
        })));

        let optimizer = HyperparamOptimizerAgent::with_llm(llm);
        assert!(optimizer.has_llm());

        let dataset = sample_dataset();
        let suggestion = optimizer.suggest_hyperparams(&dataset, "llama-7b").await.expect("suggest");

        assert_eq!(suggestion.approach, TrainingApproach::LoRA);
        assert!((suggestion.learning_rate - 0.0002).abs() < 0.0001);
        assert_eq!(suggestion.batch_size, 8);
        assert_eq!(suggestion.epochs, 3);
        assert!(suggestion.confidence > 0.8);
        assert!(suggestion.reasoning.contains("LoRA"));
        assert!(suggestion.reasoning.contains("Key insights"));
    }

    #[tokio::test]
    async fn test_optimizer_with_llm_full_fine_tuning() {
        use crate::llm::MockLlmClient;

        // Mock LLM recommends full fine-tuning for large dataset
        let llm = Arc::new(MockLlmClient::new().with_json(serde_json::json!({
            "approach": "full_fine_tuning",
            "learning_rate": 0.00005,
            "batch_size": 4,
            "epochs": 2,
            "confidence": 0.9,
            "reasoning": "Large dataset warrants full fine-tuning for best results",
            "key_insights": ["Sufficient data for full tuning"],
            "warnings": ["Requires significant GPU memory"]
        })));

        let optimizer = HyperparamOptimizerAgent::with_llm(llm);
        let dataset = sample_dataset();
        let suggestion = optimizer.suggest_hyperparams(&dataset, "llama-7b").await.expect("suggest");

        assert_eq!(suggestion.approach, TrainingApproach::FullFineTuning);
        assert!(suggestion.lora_config.is_none());
        assert!(suggestion.reasoning.contains("Warnings"));
    }

    #[tokio::test]
    async fn test_optimizer_fallback_on_llm_error() {
        use crate::llm::MockLlmClient;

        // Mock LLM returns an error
        let llm = Arc::new(MockLlmClient::new().with_error("Connection timeout"));

        let optimizer = HyperparamOptimizerAgent::with_llm(llm);
        let dataset = sample_dataset();

        // Should fall back to heuristic suggestions
        let suggestion = optimizer.suggest_hyperparams(&dataset, "llama-7b").await.expect("suggest");

        assert!(suggestion.learning_rate > 0.0);
        assert!(suggestion.batch_size > 0);
        assert!(suggestion.lora_config.is_some());
    }

    #[tokio::test]
    async fn test_optimizer_without_llm() {
        let optimizer = HyperparamOptimizerAgent::new(None);
        assert!(!optimizer.has_llm());

        let dataset = sample_dataset();
        let suggestion = optimizer.suggest_hyperparams(&dataset, "llama-7b").await.expect("suggest");

        assert!(suggestion.learning_rate > 0.0);
        assert!(suggestion.batch_size > 0);
        assert!(suggestion.epochs > 0);
        assert!(suggestion.lora_config.is_some());
    }
}
