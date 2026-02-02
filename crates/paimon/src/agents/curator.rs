//! Data Curator Agent - Generates and curates training data.
//!
//! The Data Curator agent helps with:
//! - Generating synthetic training examples from seed data
//! - Validating data quality with AI-powered analysis
//! - Suggesting augmentation strategies
//! - Identifying and fixing quality issues

use serde::{Deserialize, Serialize};
use tracing::{info, info_span};

use crate::dataset::Example;
use super::{AgentError, AugmentationStrategy, Difficulty, Result};

/// System prompt for the data curator agent.
const CURATOR_SYSTEM_PROMPT: &str = r#"You are an expert Data Curator AI assistant specializing in training data for large language models.

Your responsibilities:
1. Generate high-quality synthetic training examples that match the style and domain of seed data
2. Identify quality issues in datasets (duplicates, low quality, inconsistencies)
3. Suggest augmentation strategies to improve model performance
4. Ensure diversity and coverage across the training distribution

When generating examples:
- Match the tone, style, and complexity of the seed examples
- Ensure variety in phrasing and scenarios
- Avoid introducing biases or harmful content
- Make outputs factually accurate and helpful

Always respond in valid JSON format as specified."#;

/// The Data Curator agent for dataset management.
pub struct DataCuratorAgent {
    model: Option<String>,
}

impl DataCuratorAgent {
    /// Creates a new Data Curator agent.
    pub fn new(model: Option<String>) -> Self {
        Self { model }
    }

    /// Returns the model to use.
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// Returns the system prompt for the curator agent.
    #[must_use]
    pub fn system_prompt() -> &'static str {
        CURATOR_SYSTEM_PROMPT
    }

    /// Generates synthetic training examples from seed data.
    ///
    /// Uses the LLM to create new examples that match the style
    /// and domain of the provided seed examples.
    pub async fn generate_examples(
        &self,
        seeds: &[Example],
        count: usize,
    ) -> Result<Vec<Example>> {
        let _span = info_span!("curator.generate", seed_count = seeds.len(), target_count = count).entered();

        if seeds.is_empty() {
            return Err(AgentError::Generation("No seed examples provided".to_string()));
        }

        info!("Generating {} synthetic examples from {} seeds", count, seeds.len());

        // Build the generation prompt (will be used when LLM is integrated)
        let _prompt = self.build_generation_prompt(seeds, count);

        // In a real implementation, this would call the LLM
        // For now, we'll generate simple variations
        let mut generated = Vec::with_capacity(count);

        for i in 0..count {
            // Select a seed to base the synthetic example on
            let seed = &seeds[i % seeds.len()];

            // Generate a variation (simplified - real impl would use LLM)
            let synthetic = Example::new(
                format!("{} (variation {})", seed.input, i + 1),
                format!("{}", seed.output),
            )
            .with_metadata("source_seed", serde_json::json!(seed.id))
            .with_metadata("generation_index", serde_json::json!(i))
            .as_synthetic();

            generated.push(synthetic);
        }

        info!(generated = generated.len(), "Generated synthetic examples");
        Ok(generated)
    }

    /// Performs AI-powered quality analysis on examples.
    pub async fn quality_check(&self, examples: &[Example]) -> Result<QualityReport> {
        let _span = info_span!("curator.quality_check", count = examples.len()).entered();

        info!("Performing quality check on {} examples", examples.len());

        let mut issues = Vec::new();
        let mut scores = Vec::new();

        for example in examples {
            let score = self.score_example(example);
            scores.push(score);

            // Check for common issues
            if example.input.len() < 10 {
                issues.push(QualityIssue {
                    example_id: example.id.clone(),
                    issue_type: QualityIssueType::TooShort,
                    description: "Input is very short".to_string(),
                    suggestion: "Consider providing more context".to_string(),
                });
            }

            if example.output.len() < 5 {
                issues.push(QualityIssue {
                    example_id: example.id.clone(),
                    issue_type: QualityIssueType::TooShort,
                    description: "Output is very short".to_string(),
                    suggestion: "Consider providing more detailed responses".to_string(),
                });
            }

            if example.input.to_lowercase() == example.output.to_lowercase() {
                issues.push(QualityIssue {
                    example_id: example.id.clone(),
                    issue_type: QualityIssueType::InputOutputSimilar,
                    description: "Input and output are very similar".to_string(),
                    suggestion: "Ensure the output adds value beyond the input".to_string(),
                });
            }
        }

        let avg_score = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f32>() / scores.len() as f32
        };

        Ok(QualityReport {
            total_examples: examples.len(),
            average_score: avg_score,
            issues,
            high_quality_count: scores.iter().filter(|&&s| s >= 0.8).count(),
            low_quality_count: scores.iter().filter(|&&s| s < 0.5).count(),
        })
    }

    /// Suggests augmentation strategies based on dataset analysis.
    pub async fn suggest_augmentations(
        &self,
        examples: &[Example],
    ) -> Result<Vec<AugmentationStrategy>> {
        let _span = info_span!("curator.suggest_augmentations", count = examples.len()).entered();

        let mut strategies = Vec::new();

        // Analyze dataset characteristics
        let avg_input_len = examples.iter().map(|e| e.input.len()).sum::<usize>() as f32
            / examples.len().max(1) as f32;
        let has_system_prompts = examples.iter().any(|e| e.system.is_some());
        let synthetic_ratio = examples.iter().filter(|e| e.synthetic).count() as f32
            / examples.len().max(1) as f32;

        // Suggest based on analysis
        if examples.len() < 100 {
            strategies.push(AugmentationStrategy {
                name: "Synthetic Generation".to_string(),
                description: "Generate synthetic examples to increase dataset size".to_string(),
                expected_improvement: "Better generalization with more training data".to_string(),
                difficulty: Difficulty::Easy,
                recommended_count: 500 - examples.len(),
            });
        }

        if !has_system_prompts {
            strategies.push(AugmentationStrategy {
                name: "Add System Prompts".to_string(),
                description: "Add consistent system prompts to guide model behavior".to_string(),
                expected_improvement: "More consistent and controllable outputs".to_string(),
                difficulty: Difficulty::Easy,
                recommended_count: examples.len(),
            });
        }

        if avg_input_len < 50.0 {
            strategies.push(AugmentationStrategy {
                name: "Input Elaboration".to_string(),
                description: "Expand inputs with more context and detail".to_string(),
                expected_improvement: "Better understanding of complex queries".to_string(),
                difficulty: Difficulty::Medium,
                recommended_count: examples.len() / 2,
            });
        }

        if synthetic_ratio > 0.8 {
            strategies.push(AugmentationStrategy {
                name: "Add Real Examples".to_string(),
                description: "Balance synthetic data with more real examples".to_string(),
                expected_improvement: "Reduced synthetic bias, more natural outputs".to_string(),
                difficulty: Difficulty::Hard,
                recommended_count: (examples.len() as f32 * 0.3) as usize,
            });
        }

        // Always suggest diversity check
        strategies.push(AugmentationStrategy {
            name: "Diversity Analysis".to_string(),
            description: "Analyze and improve coverage across different scenarios".to_string(),
            expected_improvement: "Better handling of edge cases and varied inputs".to_string(),
            difficulty: Difficulty::Medium,
            recommended_count: 0, // Analysis, not generation
        });

        info!(strategies = strategies.len(), "Generated augmentation suggestions");
        Ok(strategies)
    }

    /// Deduplicates examples based on semantic similarity.
    pub async fn deduplicate(&self, examples: &[Example]) -> Result<DeduplicationResult> {
        let _span = info_span!("curator.deduplicate", count = examples.len()).entered();

        let mut unique = Vec::new();
        let mut duplicates = Vec::new();
        let mut seen_inputs = std::collections::HashSet::new();

        for example in examples {
            // Simple deduplication by exact input match
            // Real implementation would use embeddings for semantic dedup
            let normalized = example.input.to_lowercase().trim().to_string();

            if seen_inputs.contains(&normalized) {
                duplicates.push(example.id.clone());
            } else {
                seen_inputs.insert(normalized);
                unique.push(example.clone());
            }
        }

        info!(
            original = examples.len(),
            unique = unique.len(),
            duplicates = duplicates.len(),
            "Deduplication complete"
        );

        Ok(DeduplicationResult {
            original_count: examples.len(),
            unique_count: unique.len(),
            duplicate_ids: duplicates,
            unique_examples: unique,
        })
    }

    /// Builds the prompt for generating synthetic examples.
    fn build_generation_prompt(&self, seeds: &[Example], count: usize) -> String {
        let seed_examples: String = seeds.iter()
            .take(5) // Use up to 5 seeds as examples
            .enumerate()
            .map(|(i, e)| {
                format!(
                    "Example {}:\nInput: {}\nOutput: {}",
                    i + 1, e.input, e.output
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        format!(
            r#"Based on these seed examples, generate {} new training examples in the same style and domain.

{}

Generate {} new examples in JSON format:
```json
[
  {{"input": "...", "output": "..."}},
  ...
]
```

Requirements:
- Match the style and domain of the seed examples
- Ensure diversity in scenarios and phrasing
- Make outputs helpful and accurate
- Avoid repetition"#,
            count, seed_examples, count
        )
    }

    /// Scores a single example for quality.
    fn score_example(&self, example: &Example) -> f32 {
        let mut score = 1.0f32;

        // Penalize very short inputs/outputs
        if example.input.len() < 10 {
            score -= 0.2;
        }
        if example.output.len() < 10 {
            score -= 0.2;
        }

        // Penalize if input/output are too similar
        if example.input.to_lowercase().contains(&example.output.to_lowercase())
            || example.output.to_lowercase().contains(&example.input.to_lowercase())
        {
            score -= 0.3;
        }

        // Bonus for system prompt
        if example.system.is_some() {
            score += 0.1;
        }

        // Bonus for metadata
        if !example.metadata.is_empty() {
            score += 0.05;
        }

        score.clamp(0.0, 1.0)
    }
}

/// Quality report from the curator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    /// Total examples analyzed.
    pub total_examples: usize,
    /// Average quality score.
    pub average_score: f32,
    /// Quality issues found.
    pub issues: Vec<QualityIssue>,
    /// Count of high quality examples.
    pub high_quality_count: usize,
    /// Count of low quality examples.
    pub low_quality_count: usize,
}

/// A quality issue found in an example.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    /// ID of affected example.
    pub example_id: String,
    /// Type of issue.
    pub issue_type: QualityIssueType,
    /// Description.
    pub description: String,
    /// Suggested fix.
    pub suggestion: String,
}

/// Types of quality issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityIssueType {
    /// Content is too short.
    TooShort,
    /// Content is too long.
    TooLong,
    /// Input and output are too similar.
    InputOutputSimilar,
    /// Potential duplicate.
    PotentialDuplicate,
    /// Low diversity.
    LowDiversity,
    /// Inconsistent formatting.
    InconsistentFormat,
    /// Potential harmful content.
    PotentiallyHarmful,
}

/// Result of deduplication.
#[derive(Debug, Clone)]
pub struct DeduplicationResult {
    /// Original example count.
    pub original_count: usize,
    /// Unique example count.
    pub unique_count: usize,
    /// IDs of duplicate examples.
    pub duplicate_ids: Vec<String>,
    /// The unique examples.
    pub unique_examples: Vec<Example>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_generate_examples() {
        let curator = DataCuratorAgent::new(None);

        let seeds = vec![
            Example::new("What is the capital of France?", "The capital of France is Paris."),
            Example::new("What is 2 + 2?", "2 + 2 equals 4."),
        ];

        let generated = curator.generate_examples(&seeds, 5).await.expect("generate");

        assert_eq!(generated.len(), 5);
        assert!(generated.iter().all(|e| e.synthetic));
    }

    #[tokio::test]
    async fn test_quality_check() {
        let curator = DataCuratorAgent::new(None);

        let examples = vec![
            Example::new("Good input with enough content", "Good output with helpful information"),
            Example::new("Hi", "Hi"), // Short and similar
            Example::new("Normal question here", "x"), // Short output
        ];

        let report = curator.quality_check(&examples).await.expect("quality check");

        assert_eq!(report.total_examples, 3);
        assert!(!report.issues.is_empty());
    }

    #[tokio::test]
    async fn test_suggest_augmentations() {
        let curator = DataCuratorAgent::new(None);

        let examples: Vec<Example> = (0..50)
            .map(|i| Example::new(format!("Q{}", i), format!("A{}", i)))
            .collect();

        let strategies = curator.suggest_augmentations(&examples).await.expect("suggest");

        assert!(!strategies.is_empty());
        // Should suggest synthetic generation for small dataset
        assert!(strategies.iter().any(|s| s.name.contains("Synthetic")));
    }

    #[tokio::test]
    async fn test_deduplicate() {
        let curator = DataCuratorAgent::new(None);

        let examples = vec![
            Example::new("Hello", "World"),
            Example::new("Hello", "Different output"), // Duplicate input
            Example::new("Unique", "Response"),
        ];

        let result = curator.deduplicate(&examples).await.expect("dedupe");

        assert_eq!(result.original_count, 3);
        assert_eq!(result.unique_count, 2);
        assert_eq!(result.duplicate_ids.len(), 1);
    }
}
