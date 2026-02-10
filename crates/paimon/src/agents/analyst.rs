//! Evaluation Analyst Agent - Interprets benchmarks and suggests improvements.
//!
//! The Eval Analyst agent helps with:
//! - Interpreting benchmark results in human-readable terms
//! - Comparing models against baselines
//! - Generating improvement recommendations
//! - Creating evaluation roadmaps

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::{info, info_span, warn};

use super::{AgentError, ImprovementPlan, ImprovementStep, Result};
use crate::llm::{GenerateRequest, LlmClient, Message};

/// System prompt for the Eval Analyst agent.
const ANALYST_SYSTEM_PROMPT: &str = r#"You are an expert ML Evaluation Analyst specializing in benchmark interpretation and model comparison.

Your role is to analyze benchmark results and provide actionable insights. When analyzing:

1. Identify key strengths and weaknesses
2. Compare performance across different benchmarks
3. Provide specific, actionable recommendations

Always respond with valid JSON in this exact format:
{
  "executive_summary": "Brief 1-2 sentence summary",
  "performance_tier": "excellent" | "very_good" | "good" | "acceptable" | "needs_improvement",
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "key_insights": ["insight 1", "insight 2"],
  "recommendations": ["recommendation 1", "recommendation 2"],
  "priority_areas": ["area 1", "area 2"]
}"#;

/// System prompt for competitive analysis.
const COMPETITIVE_ANALYSIS_PROMPT: &str = r#"You are an expert ML Evaluation Analyst comparing model performance against baselines.

Analyze the competitive positioning and provide strategic insights.

Always respond with valid JSON in this exact format:
{
  "overall_assessment": "Brief competitive assessment",
  "competitive_advantages": ["advantage 1", "advantage 2"],
  "competitive_gaps": ["gap 1", "gap 2"],
  "strategic_recommendations": ["recommendation 1", "recommendation 2"],
  "priority_improvements": ["improvement 1", "improvement 2"]
}"#;

/// The Evaluation Analyst agent for benchmark interpretation.
pub struct EvalAnalystAgent {
    /// Optional LLM client for intelligent analysis.
    llm_client: Option<Arc<dyn LlmClient>>,
}

impl EvalAnalystAgent {
    /// Creates a new Eval Analyst agent without LLM (uses heuristics only).
    pub fn new(_model: Option<String>) -> Self {
        Self { llm_client: None }
    }

    /// Creates a new Eval Analyst agent with an LLM client for intelligent analysis.
    pub fn with_llm(llm_client: Arc<dyn LlmClient>) -> Self {
        Self {
            llm_client: Some(llm_client),
        }
    }

    /// Returns whether this analyst has LLM capabilities.
    pub fn has_llm(&self) -> bool {
        self.llm_client.is_some()
    }

    /// Interprets benchmark results into a narrative report.
    ///
    /// If an LLM client is available, uses it for intelligent analysis.
    /// Otherwise, falls back to heuristic-based interpretation.
    pub async fn interpret_results(&self, results: &BenchmarkResults) -> Result<NarrativeReport> {
        let _span = info_span!("analyst.interpret", model = %results.model_name).entered();

        info!("Interpreting benchmark results for {}", results.model_name);

        // Try LLM analysis first if available
        if let Some(ref llm) = self.llm_client {
            match self.interpret_with_llm(llm, results).await {
                Ok(report) => return Ok(report),
                Err(e) => {
                    warn!(
                        "LLM interpretation failed, falling back to heuristics: {}",
                        e
                    );
                },
            }
        }

        // Fall back to heuristic interpretation
        self.interpret_heuristic(results).await
    }

    /// Interprets results using LLM for intelligent insights.
    async fn interpret_with_llm(
        &self,
        llm: &Arc<dyn LlmClient>,
        results: &BenchmarkResults,
    ) -> Result<NarrativeReport> {
        let prompt = self.build_interpret_prompt(results);

        let request = GenerateRequest::new(vec![
            Message::system(ANALYST_SYSTEM_PROMPT),
            Message::user(prompt),
        ])
        .with_temperature(0.3)
        .with_max_tokens(2048);

        let response = llm
            .generate(request)
            .await
            .map_err(|e| AgentError::Llm(format!("LLM request failed: {}", e)))?;

        self.parse_interpret_response(&response.content, results)
    }

    /// Builds the interpretation prompt from benchmark results.
    fn build_interpret_prompt(&self, results: &BenchmarkResults) -> String {
        let mut prompt = format!(
            "Analyze benchmark results for model '{}':\n\n",
            results.model_name
        );

        prompt.push_str(&format!(
            "Overall Score: {:.1}%\n",
            results.overall_score() * 100.0
        ));
        prompt.push_str(&format!(
            "Categories: {}\n\n",
            results.categories().join(", ")
        ));

        prompt.push_str("Benchmark Scores:\n");
        for benchmark in &results.benchmarks {
            prompt.push_str(&format!(
                "- {}: {:.1}%",
                benchmark.name,
                benchmark.score * 100.0
            ));
            if let Some(ref cat) = benchmark.category {
                prompt.push_str(&format!(" ({})", cat));
            }
            prompt.push('\n');
        }

        prompt.push_str("\nProvide your analysis as JSON.");
        prompt
    }

    /// Parses the LLM response into a NarrativeReport.
    fn parse_interpret_response(
        &self,
        content: &str,
        results: &BenchmarkResults,
    ) -> Result<NarrativeReport> {
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

        let parsed: LlmInterpretResponse = serde_json::from_str(json_str)
            .map_err(|e| AgentError::Analysis(format!("Failed to parse LLM response: {}", e)))?;

        // Convert to NarrativeReport
        let mut sections = Vec::new();

        sections.push(ReportSection {
            title: "Executive Summary".to_string(),
            content: parsed.executive_summary,
        });

        if !parsed.strengths.is_empty() {
            sections.push(ReportSection {
                title: "Strengths".to_string(),
                content: parsed
                    .strengths
                    .iter()
                    .map(|s| format!("- {}", s))
                    .collect::<Vec<_>>()
                    .join("\n"),
            });
        }

        if !parsed.weaknesses.is_empty() {
            sections.push(ReportSection {
                title: "Areas for Improvement".to_string(),
                content: parsed
                    .weaknesses
                    .iter()
                    .map(|w| format!("- {}", w))
                    .collect::<Vec<_>>()
                    .join("\n"),
            });
        }

        if !parsed.key_insights.is_empty() {
            sections.push(ReportSection {
                title: "Key Insights".to_string(),
                content: parsed
                    .key_insights
                    .iter()
                    .map(|i| format!("- {}", i))
                    .collect::<Vec<_>>()
                    .join("\n"),
            });
        }

        if !parsed.recommendations.is_empty() {
            sections.push(ReportSection {
                title: "Recommendations".to_string(),
                content: parsed
                    .recommendations
                    .iter()
                    .map(|r| format!("- {}", r))
                    .collect::<Vec<_>>()
                    .join("\n"),
            });
        }

        Ok(NarrativeReport {
            title: format!("Evaluation Report: {}", results.model_name),
            model_name: results.model_name.clone(),
            overall_score: results.overall_score(),
            sections,
            generated_at: chrono::Utc::now(),
        })
    }

    /// Heuristic-based interpretation (fallback when no LLM).
    async fn interpret_heuristic(&self, results: &BenchmarkResults) -> Result<NarrativeReport> {
        let mut sections = Vec::new();

        // Executive summary
        let overall_score = results.overall_score();
        let performance_tier = match overall_score {
            s if s >= 0.9 => "excellent",
            s if s >= 0.8 => "very good",
            s if s >= 0.7 => "good",
            s if s >= 0.6 => "acceptable",
            _ => "needs improvement",
        };

        sections.push(ReportSection {
            title: "Executive Summary".to_string(),
            content: format!(
                "The model '{}' achieved an overall score of {:.1}%, placing it in the '{}' performance tier. \
                 The evaluation covered {} benchmarks across {} categories.",
                results.model_name,
                overall_score * 100.0,
                performance_tier,
                results.benchmarks.len(),
                results.categories().len()
            ),
        });

        // Strengths
        let strengths: Vec<_> = results
            .benchmarks
            .iter()
            .filter(|b| b.score >= 0.8)
            .collect();

        if !strengths.is_empty() {
            let strength_list: String = strengths
                .iter()
                .map(|b| format!("- {}: {:.1}%", b.name, b.score * 100.0))
                .collect::<Vec<_>>()
                .join("\n");

            sections.push(ReportSection {
                title: "Strengths".to_string(),
                content: format!(
                    "The model excels in the following areas:\n{}",
                    strength_list
                ),
            });
        }

        // Weaknesses
        let weaknesses: Vec<_> = results
            .benchmarks
            .iter()
            .filter(|b| b.score < 0.6)
            .collect();

        if !weaknesses.is_empty() {
            let weakness_list: String = weaknesses
                .iter()
                .map(|b| format!("- {}: {:.1}%", b.name, b.score * 100.0))
                .collect::<Vec<_>>()
                .join("\n");

            sections.push(ReportSection {
                title: "Areas for Improvement".to_string(),
                content: format!(
                    "The model shows room for improvement in:\n{}",
                    weakness_list
                ),
            });
        }

        // Recommendations
        sections.push(ReportSection {
            title: "Recommendations".to_string(),
            content: self.generate_recommendations(results),
        });

        Ok(NarrativeReport {
            title: format!("Evaluation Report: {}", results.model_name),
            model_name: results.model_name.clone(),
            overall_score,
            sections,
            generated_at: chrono::Utc::now(),
        })
    }

    /// Compares a model against baselines.
    ///
    /// If an LLM client is available, uses it for intelligent competitive insights.
    /// Otherwise, falls back to heuristic-based comparison.
    pub async fn competitive_analysis(
        &self,
        model_results: &BenchmarkResults,
        baselines: &[BenchmarkResults],
    ) -> Result<CompetitiveReport> {
        let _span = info_span!("analyst.competitive", model = %model_results.model_name).entered();

        info!(
            "Performing competitive analysis: {} vs {} baselines",
            model_results.model_name,
            baselines.len()
        );

        // Try LLM analysis first if available
        if let Some(ref llm) = self.llm_client {
            match self
                .competitive_with_llm(llm, model_results, baselines)
                .await
            {
                Ok(report) => return Ok(report),
                Err(e) => {
                    warn!(
                        "LLM competitive analysis failed, falling back to heuristics: {}",
                        e
                    );
                },
            }
        }

        // Fall back to heuristic analysis
        self.competitive_heuristic(model_results, baselines).await
    }

    /// Competitive analysis using LLM for strategic insights.
    async fn competitive_with_llm(
        &self,
        llm: &Arc<dyn LlmClient>,
        model_results: &BenchmarkResults,
        baselines: &[BenchmarkResults],
    ) -> Result<CompetitiveReport> {
        let prompt = self.build_competitive_prompt(model_results, baselines);

        let request = GenerateRequest::new(vec![
            Message::system(COMPETITIVE_ANALYSIS_PROMPT),
            Message::user(prompt),
        ])
        .with_temperature(0.3)
        .with_max_tokens(2048);

        let response = llm
            .generate(request)
            .await
            .map_err(|e| AgentError::Llm(format!("LLM request failed: {}", e)))?;

        self.parse_competitive_response(&response.content, model_results, baselines)
    }

    /// Builds the competitive analysis prompt.
    fn build_competitive_prompt(
        &self,
        model_results: &BenchmarkResults,
        baselines: &[BenchmarkResults],
    ) -> String {
        let mut prompt = format!(
            "Compare model '{}' (score: {:.1}%) against {} baselines:\n\n",
            model_results.model_name,
            model_results.overall_score() * 100.0,
            baselines.len()
        );

        prompt.push_str("Model benchmarks:\n");
        for b in &model_results.benchmarks {
            prompt.push_str(&format!("- {}: {:.1}%\n", b.name, b.score * 100.0));
        }

        prompt.push_str("\nBaselines:\n");
        for baseline in baselines {
            prompt.push_str(&format!(
                "\n{} (overall: {:.1}%):\n",
                baseline.model_name,
                baseline.overall_score() * 100.0
            ));
            for b in &baseline.benchmarks {
                prompt.push_str(&format!("  - {}: {:.1}%\n", b.name, b.score * 100.0));
            }
        }

        prompt.push_str("\nProvide your competitive analysis as JSON.");
        prompt
    }

    /// Parses the LLM competitive response.
    fn parse_competitive_response(
        &self,
        content: &str,
        model_results: &BenchmarkResults,
        baselines: &[BenchmarkResults],
    ) -> Result<CompetitiveReport> {
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

        let parsed: LlmCompetitiveResponse = serde_json::from_str(json_str)
            .map_err(|e| AgentError::Analysis(format!("Failed to parse LLM response: {}", e)))?;

        // Build comparisons from heuristics (LLM provides insights, we compute stats)
        let comparisons = self.build_comparisons(model_results, baselines);
        let (rank, total) = self.calculate_ranking(model_results, baselines);

        // Combine LLM insights with computed data
        let mut summary = parsed.overall_assessment;
        if let Some(first_rec) = parsed.strategic_recommendations.first() {
            summary.push_str(&format!("\n\nKey recommendation: {}", first_rec));
        }

        Ok(CompetitiveReport {
            model_name: model_results.model_name.clone(),
            comparisons,
            overall_rank: rank,
            total_models: total,
            summary,
        })
    }

    /// Heuristic-based competitive analysis.
    async fn competitive_heuristic(
        &self,
        model_results: &BenchmarkResults,
        baselines: &[BenchmarkResults],
    ) -> Result<CompetitiveReport> {
        let comparisons = self.build_comparisons(model_results, baselines);
        let (rank, total) = self.calculate_ranking(model_results, baselines);

        Ok(CompetitiveReport {
            model_name: model_results.model_name.clone(),
            comparisons,
            overall_rank: rank,
            total_models: total,
            summary: format!(
                "{} ranks #{} out of {} models evaluated",
                model_results.model_name, rank, total
            ),
        })
    }

    /// Builds model comparisons.
    fn build_comparisons(
        &self,
        model_results: &BenchmarkResults,
        baselines: &[BenchmarkResults],
    ) -> Vec<ModelComparison> {
        let mut comparisons = Vec::new();

        for baseline in baselines {
            let model_score = model_results.overall_score();
            let baseline_score = baseline.overall_score();
            let diff = model_score - baseline_score;

            comparisons.push(ModelComparison {
                baseline_name: baseline.model_name.clone(),
                model_score,
                baseline_score,
                difference: diff,
                better_on: model_results
                    .benchmarks
                    .iter()
                    .filter(|b| {
                        baseline
                            .benchmarks
                            .iter()
                            .find(|bb| bb.name == b.name)
                            .map_or(false, |bb| b.score > bb.score)
                    })
                    .map(|b| b.name.clone())
                    .collect(),
                worse_on: model_results
                    .benchmarks
                    .iter()
                    .filter(|b| {
                        baseline
                            .benchmarks
                            .iter()
                            .find(|bb| bb.name == b.name)
                            .map_or(false, |bb| b.score < bb.score)
                    })
                    .map(|b| b.name.clone())
                    .collect(),
            });
        }

        comparisons
    }

    /// Calculates model ranking.
    fn calculate_ranking(
        &self,
        model_results: &BenchmarkResults,
        baselines: &[BenchmarkResults],
    ) -> (usize, usize) {
        let mut all_scores: Vec<_> = baselines
            .iter()
            .map(|b| (b.model_name.clone(), b.overall_score()))
            .collect();
        all_scores.push((
            model_results.model_name.clone(),
            model_results.overall_score(),
        ));
        all_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let rank = all_scores
            .iter()
            .position(|(name, _)| name == &model_results.model_name)
            .map(|p| p + 1)
            .unwrap_or(0);

        (rank, all_scores.len())
    }

    /// Creates an improvement roadmap.
    pub async fn improvement_roadmap(&self, results: &BenchmarkResults) -> Result<ImprovementPlan> {
        let _span = info_span!("analyst.roadmap", model = %results.model_name).entered();

        info!("Creating improvement roadmap for {}", results.model_name);

        let mut steps = Vec::new();
        let mut step_num = 1;

        // Sort benchmarks by score (lowest first)
        let mut sorted_benchmarks = results.benchmarks.clone();
        sorted_benchmarks.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Create improvement steps for weak areas
        for benchmark in sorted_benchmarks.iter().take(3) {
            if benchmark.score < 0.7 {
                steps.push(ImprovementStep {
                    step: step_num,
                    action: format!("Improve {} performance", benchmark.name),
                    rationale: format!(
                        "Current score of {:.1}% is below target. This benchmark tests {}.",
                        benchmark.score * 100.0,
                        benchmark
                            .description
                            .as_deref()
                            .unwrap_or("key capabilities")
                    ),
                    impact: format!(
                        "Expected {:.0}% improvement in {} with targeted training data",
                        (0.8 - benchmark.score) * 100.0,
                        benchmark.name
                    ),
                });
                step_num += 1;
            }
        }

        // Add general improvement steps
        if results.overall_score() < 0.9 {
            steps.push(ImprovementStep {
                step: step_num,
                action: "Expand training data diversity".to_string(),
                rationale: "Broader training data improves generalization across all tasks"
                    .to_string(),
                impact: "5-10% improvement across all benchmarks".to_string(),
            });
            step_num += 1;
        }

        steps.push(ImprovementStep {
            step: step_num,
            action: "Continuous evaluation integration".to_string(),
            rationale: "Regular benchmarking catches regressions early".to_string(),
            impact: "Maintained quality over time".to_string(),
        });

        let estimated_effort = format!("{} weeks of focused development", steps.len());
        let summary = format!(
            "This plan targets {} key areas for improvement, \
             with an expected overall score increase from {:.1}% to {:.1}%",
            steps.len().saturating_sub(1),
            results.overall_score() * 100.0,
            (results.overall_score() + 0.1).min(1.0) * 100.0
        );

        Ok(ImprovementPlan {
            title: format!("Improvement Plan for {}", results.model_name),
            summary,
            steps,
            expected_outcome: "Significant improvement in model capabilities".to_string(),
            estimated_effort,
        })
    }

    /// Generates recommendations based on results.
    fn generate_recommendations(&self, results: &BenchmarkResults) -> String {
        let mut recommendations: Vec<String> = Vec::new();

        let overall = results.overall_score();

        if overall < 0.6 {
            recommendations
                .push("Consider additional fine-tuning on domain-specific data".to_string());
            recommendations.push("Review training data quality and coverage".to_string());
        } else if overall < 0.8 {
            recommendations
                .push("Target specific weak areas with additional training examples".to_string());
            recommendations
                .push("Consider data augmentation for underperforming categories".to_string());
        } else {
            recommendations
                .push("Model is performing well; focus on maintaining quality".to_string());
            recommendations.push("Consider edge case testing for robustness".to_string());
        }

        // Check for specific issues
        let low_performers: Vec<_> = results
            .benchmarks
            .iter()
            .filter(|b| b.score < 0.5)
            .collect();

        if !low_performers.is_empty() {
            recommendations.push(format!(
                "Priority: Address {} benchmarks scoring below 50%",
                low_performers.len()
            ));
        }

        recommendations
            .iter()
            .map(|r| format!("- {}", r))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Results from running benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Model name.
    pub model_name: String,
    /// Individual benchmark results.
    pub benchmarks: Vec<BenchmarkScore>,
    /// When the evaluation was run.
    pub evaluated_at: chrono::DateTime<chrono::Utc>,
}

impl BenchmarkResults {
    /// Calculates the overall score.
    pub fn overall_score(&self) -> f32 {
        if self.benchmarks.is_empty() {
            return 0.0;
        }
        self.benchmarks.iter().map(|b| b.score).sum::<f32>() / self.benchmarks.len() as f32
    }

    /// Returns unique categories.
    pub fn categories(&self) -> Vec<String> {
        let mut cats: Vec<_> = self
            .benchmarks
            .iter()
            .filter_map(|b| b.category.clone())
            .collect();
        cats.sort();
        cats.dedup();
        cats
    }
}

/// Score for a single benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkScore {
    /// Benchmark name.
    pub name: String,
    /// Score (0.0 - 1.0).
    pub score: f32,
    /// Category.
    pub category: Option<String>,
    /// Description.
    pub description: Option<String>,
    /// Number of test cases.
    pub test_cases: Option<usize>,
}

/// A narrative report from the analyst.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeReport {
    /// Report title.
    pub title: String,
    /// Model name.
    pub model_name: String,
    /// Overall score.
    pub overall_score: f32,
    /// Report sections.
    pub sections: Vec<ReportSection>,
    /// Generation timestamp.
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

/// A section in a report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSection {
    /// Section title.
    pub title: String,
    /// Section content.
    pub content: String,
}

/// Competitive analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetitiveReport {
    /// Model being evaluated.
    pub model_name: String,
    /// Comparisons with baselines.
    pub comparisons: Vec<ModelComparison>,
    /// Overall rank.
    pub overall_rank: usize,
    /// Total models compared.
    pub total_models: usize,
    /// Summary statement.
    pub summary: String,
}

/// Comparison between two models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    /// Baseline model name.
    pub baseline_name: String,
    /// Our model's score.
    pub model_score: f32,
    /// Baseline's score.
    pub baseline_score: f32,
    /// Score difference.
    pub difference: f32,
    /// Benchmarks we're better on.
    pub better_on: Vec<String>,
    /// Benchmarks we're worse on.
    pub worse_on: Vec<String>,
}

/// Response structure for LLM interpretation (fields populated by deserialization).
#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
struct LlmInterpretResponse {
    /// Executive summary.
    executive_summary: String,
    /// Performance tier.
    #[serde(default)]
    performance_tier: String,
    /// Identified strengths.
    #[serde(default)]
    strengths: Vec<String>,
    /// Identified weaknesses.
    #[serde(default)]
    weaknesses: Vec<String>,
    /// Key insights.
    #[serde(default)]
    key_insights: Vec<String>,
    /// Recommendations.
    #[serde(default)]
    recommendations: Vec<String>,
    /// Priority areas.
    #[serde(default)]
    priority_areas: Vec<String>,
}

/// Response structure for LLM competitive analysis (fields populated by deserialization).
#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
struct LlmCompetitiveResponse {
    /// Overall assessment.
    overall_assessment: String,
    /// Competitive advantages.
    #[serde(default)]
    competitive_advantages: Vec<String>,
    /// Competitive gaps.
    #[serde(default)]
    competitive_gaps: Vec<String>,
    /// Strategic recommendations.
    #[serde(default)]
    strategic_recommendations: Vec<String>,
    /// Priority improvements.
    #[serde(default)]
    priority_improvements: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_results() -> BenchmarkResults {
        BenchmarkResults {
            model_name: "test-model".to_string(),
            benchmarks: vec![
                BenchmarkScore {
                    name: "MMLU".to_string(),
                    score: 0.75,
                    category: Some("Knowledge".to_string()),
                    description: Some("Multi-task language understanding".to_string()),
                    test_cases: Some(1000),
                },
                BenchmarkScore {
                    name: "HellaSwag".to_string(),
                    score: 0.85,
                    category: Some("Reasoning".to_string()),
                    description: Some("Commonsense reasoning".to_string()),
                    test_cases: Some(500),
                },
                BenchmarkScore {
                    name: "TruthfulQA".to_string(),
                    score: 0.55,
                    category: Some("Truthfulness".to_string()),
                    description: Some("Factual accuracy".to_string()),
                    test_cases: Some(200),
                },
            ],
            evaluated_at: chrono::Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_interpret_results() {
        let analyst = EvalAnalystAgent::new(None);
        let results = sample_results();

        let report = analyst
            .interpret_results(&results)
            .await
            .expect("interpret");

        assert_eq!(report.model_name, "test-model");
        assert!(!report.sections.is_empty());
        assert!(report
            .sections
            .iter()
            .any(|s| s.title == "Executive Summary"));
    }

    #[tokio::test]
    async fn test_competitive_analysis() {
        let analyst = EvalAnalystAgent::new(None);
        let model = sample_results();

        let baseline = BenchmarkResults {
            model_name: "baseline-model".to_string(),
            benchmarks: vec![
                BenchmarkScore {
                    name: "MMLU".to_string(),
                    score: 0.70,
                    category: None,
                    description: None,
                    test_cases: None,
                },
                BenchmarkScore {
                    name: "HellaSwag".to_string(),
                    score: 0.80,
                    category: None,
                    description: None,
                    test_cases: None,
                },
            ],
            evaluated_at: chrono::Utc::now(),
        };

        let report = analyst
            .competitive_analysis(&model, &[baseline])
            .await
            .expect("compare");

        assert_eq!(report.model_name, "test-model");
        assert_eq!(report.comparisons.len(), 1);
    }

    #[tokio::test]
    async fn test_improvement_roadmap() {
        let analyst = EvalAnalystAgent::new(None);
        let results = sample_results();

        let plan = analyst
            .improvement_roadmap(&results)
            .await
            .expect("roadmap");

        assert!(!plan.steps.is_empty());
        assert!(!plan.summary.is_empty());
    }

    #[tokio::test]
    async fn test_analyst_with_llm_interprets_results() {
        use crate::llm::MockLlmClient;

        // Mock LLM returns structured interpretation
        let llm = Arc::new(MockLlmClient::new().with_json(serde_json::json!({
            "executive_summary": "Model shows strong performance overall with room for improvement in truthfulness",
            "performance_tier": "good",
            "strengths": ["Excellent reasoning capabilities", "Strong knowledge recall"],
            "weaknesses": ["Truthfulness needs work", "May hallucinate on edge cases"],
            "key_insights": ["Model excels at structured reasoning tasks"],
            "recommendations": ["Add more fact-checking examples to training data"],
            "priority_areas": ["TruthfulQA improvement"]
        })));

        let analyst = EvalAnalystAgent::with_llm(llm);
        assert!(analyst.has_llm());

        let results = sample_results();
        let report = analyst
            .interpret_results(&results)
            .await
            .expect("interpret");

        assert_eq!(report.model_name, "test-model");
        assert!(report
            .sections
            .iter()
            .any(|s| s.title == "Executive Summary"));
        assert!(report
            .sections
            .iter()
            .any(|s| s.content.contains("strong performance")));
    }

    #[tokio::test]
    async fn test_analyst_with_llm_competitive_analysis() {
        use crate::llm::MockLlmClient;

        // Mock LLM returns competitive insights
        let llm = Arc::new(MockLlmClient::new().with_json(serde_json::json!({
            "overall_assessment": "Model outperforms baseline on reasoning but lags on factual accuracy",
            "competitive_advantages": ["Better reasoning", "Faster inference"],
            "competitive_gaps": ["Lower accuracy on facts", "Weaker long-context handling"],
            "strategic_recommendations": ["Focus on improving factual grounding"],
            "priority_improvements": ["TruthfulQA benchmark improvement"]
        })));

        let analyst = EvalAnalystAgent::with_llm(llm);
        let model = sample_results();

        let baseline = BenchmarkResults {
            model_name: "baseline-model".to_string(),
            benchmarks: vec![BenchmarkScore {
                name: "MMLU".to_string(),
                score: 0.70,
                category: None,
                description: None,
                test_cases: None,
            }],
            evaluated_at: chrono::Utc::now(),
        };

        let report = analyst
            .competitive_analysis(&model, &[baseline])
            .await
            .expect("competitive");

        assert!(!report.summary.is_empty());
        assert!(report.summary.contains("outperforms"));
        assert!(report.summary.contains("Key recommendation"));
    }

    #[tokio::test]
    async fn test_analyst_fallback_on_llm_error() {
        use crate::llm::MockLlmClient;

        // Mock LLM returns an error
        let llm = Arc::new(MockLlmClient::new().with_error("API rate limited"));

        let analyst = EvalAnalystAgent::with_llm(llm);
        let results = sample_results();

        // Should fall back to heuristic analysis
        let report = analyst
            .interpret_results(&results)
            .await
            .expect("interpret");

        assert_eq!(report.model_name, "test-model");
        assert!(report
            .sections
            .iter()
            .any(|s| s.title == "Executive Summary"));
    }

    #[tokio::test]
    async fn test_analyst_without_llm() {
        let analyst = EvalAnalystAgent::new(None);
        assert!(!analyst.has_llm());

        let results = sample_results();
        let report = analyst
            .interpret_results(&results)
            .await
            .expect("interpret");

        assert_eq!(report.model_name, "test-model");
        assert!(report
            .sections
            .iter()
            .any(|s| s.title == "Executive Summary"));
    }
}
