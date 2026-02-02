//! Training Coach Agent - Monitors and guides training runs.
//!
//! The Training Coach agent helps with:
//! - Real-time monitoring of training metrics
//! - Detecting issues (overfitting, divergence, plateau)
//! - Suggesting interventions and adjustments
//! - Providing guidance on when to stop training

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::{info, info_span, warn};

use super::{AgentError, TrainingIssue, TrainingIssueType, Result};
use crate::llm::{LlmClient, GenerateRequest, Message};

/// System prompt for the Training Coach agent.
const COACH_SYSTEM_PROMPT: &str = r#"You are an expert ML Training Coach specializing in deep learning and LLM fine-tuning.

Your role is to analyze training metrics and provide actionable insights. When analyzing runs:

1. Identify the health status: "healthy", "warning", or "critical"
2. Detect issues: overfitting, underfitting, divergence, plateau, gradient problems
3. Provide specific, actionable suggestions

Always respond with valid JSON in this exact format:
{
  "health": "healthy" | "warning" | "critical",
  "issues": [
    {
      "type": "overfitting" | "underfitting" | "divergence" | "plateau" | "gradient_explosion" | "gradient_vanishing",
      "severity": 1-5,
      "evidence": ["evidence point 1", "evidence point 2"]
    }
  ],
  "insights": ["insight 1", "insight 2"],
  "suggestions": ["suggestion 1", "suggestion 2"],
  "should_stop": true | false,
  "estimated_epochs_remaining": number | null
}"#;

/// The Training Coach agent for monitoring training runs.
pub struct TrainingCoachAgent {
    /// Optional LLM client for intelligent analysis.
    llm_client: Option<Arc<dyn LlmClient>>,
}

impl TrainingCoachAgent {
    /// Creates a new Training Coach agent without LLM (uses heuristics only).
    pub fn new(_model: Option<String>) -> Self {
        Self { llm_client: None }
    }

    /// Creates a new Training Coach agent with an LLM client for intelligent analysis.
    pub fn with_llm(llm_client: Arc<dyn LlmClient>) -> Self {
        Self {
            llm_client: Some(llm_client),
        }
    }

    /// Returns whether this coach has LLM capabilities.
    pub fn has_llm(&self) -> bool {
        self.llm_client.is_some()
    }

    /// Analyzes training metrics and returns insights.
    ///
    /// If an LLM client is available, uses it for intelligent analysis.
    /// Otherwise, falls back to heuristic-based analysis.
    pub async fn analyze_run(&self, metrics: &TrainingMetrics) -> Result<RunAnalysis> {
        let _span = info_span!(
            "coach.analyze",
            epoch = metrics.current_epoch,
            loss = metrics.train_loss.unwrap_or(0.0)
        ).entered();

        info!("Analyzing training run at epoch {}", metrics.current_epoch);

        // Try LLM analysis first if available
        if let Some(ref llm) = self.llm_client {
            match self.analyze_with_llm(llm, metrics).await {
                Ok(analysis) => return Ok(analysis),
                Err(e) => {
                    warn!("LLM analysis failed, falling back to heuristics: {}", e);
                }
            }
        }

        // Fall back to heuristic analysis
        self.analyze_heuristic(metrics).await
    }

    /// Analyzes metrics using LLM for intelligent insights.
    async fn analyze_with_llm(
        &self,
        llm: &Arc<dyn LlmClient>,
        metrics: &TrainingMetrics,
    ) -> Result<RunAnalysis> {
        let prompt = self.build_analysis_prompt(metrics);

        let request = GenerateRequest::new(vec![
            Message::system(COACH_SYSTEM_PROMPT),
            Message::user(prompt),
        ])
        .with_temperature(0.3) // Lower temperature for consistent JSON
        .with_max_tokens(1024);

        let response = llm.generate(request).await
            .map_err(|e| AgentError::Llm(format!("LLM request failed: {}", e)))?;

        // Parse the JSON response
        self.parse_llm_response(&response.content)
    }

    /// Builds the analysis prompt from metrics.
    fn build_analysis_prompt(&self, metrics: &TrainingMetrics) -> String {
        let mut prompt = format!(
            "Analyze this training run at epoch {}:\n\n",
            metrics.current_epoch
        );

        if let Some(train_loss) = metrics.train_loss {
            prompt.push_str(&format!("Training loss: {:.6}\n", train_loss));
        }
        if let Some(val_loss) = metrics.val_loss {
            prompt.push_str(&format!("Validation loss: {:.6}\n", val_loss));
        }
        if !metrics.loss_history.is_empty() {
            prompt.push_str(&format!(
                "Loss history (last {} epochs): {:?}\n",
                metrics.loss_history.len().min(10),
                metrics.loss_history.iter().rev().take(10).collect::<Vec<_>>()
            ));
        }
        if let Some(lr) = metrics.learning_rate {
            prompt.push_str(&format!("Learning rate: {:.2e}\n", lr));
        }
        if let Some(grad_norm) = metrics.gradient_norm {
            prompt.push_str(&format!("Gradient norm: {:.4}\n", grad_norm));
        }

        prompt.push_str("\nProvide your analysis as JSON.");
        prompt
    }

    /// Parses the LLM response into a RunAnalysis.
    fn parse_llm_response(&self, content: &str) -> Result<RunAnalysis> {
        // Try to extract JSON from the response
        let json_str = if let Some(start) = content.find('{') {
            if let Some(end) = content.rfind('}') {
                &content[start..=end]
            } else {
                content
            }
        } else {
            content
        };

        let parsed: LlmAnalysisResponse = serde_json::from_str(json_str)
            .map_err(|e| AgentError::Analysis(format!("Failed to parse LLM response: {}", e)))?;

        // Convert LLM response to RunAnalysis
        let health = match parsed.health.as_str() {
            "critical" => RunHealth::Critical,
            "warning" => RunHealth::Warning,
            _ => RunHealth::Healthy,
        };

        Ok(RunAnalysis {
            health,
            insights: parsed.insights,
            recommended_actions: parsed.suggestions,
            should_stop: parsed.should_stop,
            estimated_epochs_remaining: parsed.estimated_epochs_remaining,
        })
    }

    /// Heuristic-based analysis (fallback when no LLM).
    async fn analyze_heuristic(&self, metrics: &TrainingMetrics) -> Result<RunAnalysis> {
        let mut insights = Vec::new();
        let mut health = RunHealth::Healthy;

        // Check loss trend
        if let Some(trend) = self.analyze_loss_trend(&metrics.loss_history) {
            insights.push(trend.clone());
            if trend.contains("diverging") || trend.contains("exploding") {
                health = RunHealth::Critical;
            } else if trend.contains("plateau") {
                health = RunHealth::Warning;
            }
        }

        // Check for overfitting
        if let (Some(train), Some(val)) = (metrics.train_loss, metrics.val_loss) {
            let gap = train - val;
            if gap.abs() > 0.5 && val > train {
                insights.push("Potential overfitting detected: validation loss exceeds training loss".to_string());
                health = RunHealth::Warning;
            }
        }

        // Check learning rate
        if let Some(lr) = metrics.learning_rate {
            if lr < 1e-7 {
                insights.push("Learning rate is very low, training may be ineffective".to_string());
            }
        }

        let recommended_actions = self.get_recommended_actions(&health, &insights);
        let should_stop = health == RunHealth::Critical;
        let estimated_epochs_remaining = self.estimate_remaining_epochs(metrics);

        Ok(RunAnalysis {
            health,
            insights,
            recommended_actions,
            should_stop,
            estimated_epochs_remaining,
        })
    }

    /// Detects training issues from metrics history.
    pub async fn detect_issues(&self, metrics: &TrainingMetrics) -> Result<Vec<TrainingIssue>> {
        let _span = info_span!("coach.detect_issues").entered();

        let mut issues = Vec::new();

        // Check for overfitting
        if let (Some(train), Some(val)) = (metrics.train_loss, metrics.val_loss) {
            if val > train * 1.5 && metrics.current_epoch > 2 {
                issues.push(TrainingIssue {
                    issue_type: TrainingIssueType::Overfitting,
                    description: format!(
                        "Validation loss ({:.4}) is significantly higher than training loss ({:.4})",
                        val, train
                    ),
                    severity: 4,
                    suggested_action: "Consider early stopping, adding regularization, or reducing model capacity".to_string(),
                    evidence: vec![
                        format!("Train loss: {:.4}", train),
                        format!("Val loss: {:.4}", val),
                        format!("Gap: {:.4}", val - train),
                    ],
                });
            }
        }

        // Check for divergence
        if metrics.loss_history.len() >= 3 {
            let recent: Vec<_> = metrics.loss_history.iter().rev().take(3).collect();
            if recent.windows(2).all(|w| w[0] > w[1]) {
                let increase = recent[0] - recent[2];
                if increase > 0.5 {
                    issues.push(TrainingIssue {
                        issue_type: TrainingIssueType::Divergence,
                        description: "Loss is increasing rapidly".to_string(),
                        severity: 5,
                        suggested_action: "Reduce learning rate immediately or restart training".to_string(),
                        evidence: recent.iter().map(|l| format!("{:.4}", l)).collect(),
                    });
                }
            }
        }

        // Check for plateau
        if metrics.loss_history.len() >= 5 {
            let recent: Vec<_> = metrics.loss_history.iter().rev().take(5).collect();
            let variance: f32 = recent.iter()
                .map(|&x| (x - recent[0]).powi(2))
                .sum::<f32>() / 5.0;

            if variance < 0.001 {
                issues.push(TrainingIssue {
                    issue_type: TrainingIssueType::Plateau,
                    description: "Training has plateaued with minimal improvement".to_string(),
                    severity: 3,
                    suggested_action: "Consider learning rate scheduling, or stop if validation metrics are good".to_string(),
                    evidence: vec![format!("Loss variance over 5 epochs: {:.6}", variance)],
                });
            }
        }

        // Check gradient health
        if let Some(grad_norm) = metrics.gradient_norm {
            if grad_norm > 100.0 {
                issues.push(TrainingIssue {
                    issue_type: TrainingIssueType::GradientExplosion,
                    description: format!("Gradient norm is very high: {:.2}", grad_norm),
                    severity: 5,
                    suggested_action: "Apply gradient clipping or reduce learning rate".to_string(),
                    evidence: vec![format!("Gradient norm: {:.2}", grad_norm)],
                });
            } else if grad_norm < 1e-7 {
                issues.push(TrainingIssue {
                    issue_type: TrainingIssueType::GradientVanishing,
                    description: format!("Gradient norm is very low: {:.2e}", grad_norm),
                    severity: 4,
                    suggested_action: "Check model architecture, consider skip connections or different initialization".to_string(),
                    evidence: vec![format!("Gradient norm: {:.2e}", grad_norm)],
                });
            }
        }

        if !issues.is_empty() {
            warn!(issue_count = issues.len(), "Detected training issues");
        }

        Ok(issues)
    }

    /// Suggests interventions for detected issues.
    pub async fn suggest_action(&self, issue: &TrainingIssue) -> Result<SuggestedIntervention> {
        let intervention = match issue.issue_type {
            TrainingIssueType::Overfitting => SuggestedIntervention {
                action: InterventionAction::EarlyStopping,
                description: "Stop training to prevent further overfitting".to_string(),
                urgency: Urgency::High,
                parameters: vec![
                    ("patience".to_string(), "3 epochs".to_string()),
                ],
                alternative_actions: vec![
                    "Increase dropout rate".to_string(),
                    "Add weight decay".to_string(),
                    "Reduce model capacity".to_string(),
                ],
            },
            TrainingIssueType::Divergence => SuggestedIntervention {
                action: InterventionAction::ReduceLearningRate,
                description: "Immediately reduce learning rate to stabilize training".to_string(),
                urgency: Urgency::Critical,
                parameters: vec![
                    ("factor".to_string(), "0.1".to_string()),
                ],
                alternative_actions: vec![
                    "Restart from last checkpoint".to_string(),
                    "Apply gradient clipping".to_string(),
                ],
            },
            TrainingIssueType::Plateau => SuggestedIntervention {
                action: InterventionAction::LearningRateSchedule,
                description: "Apply learning rate decay to escape plateau".to_string(),
                urgency: Urgency::Medium,
                parameters: vec![
                    ("schedule".to_string(), "cosine_annealing".to_string()),
                    ("min_lr".to_string(), "1e-6".to_string()),
                ],
                alternative_actions: vec![
                    "Stop training if validation metrics are satisfactory".to_string(),
                    "Try warmup restart".to_string(),
                ],
            },
            TrainingIssueType::GradientExplosion => SuggestedIntervention {
                action: InterventionAction::GradientClipping,
                description: "Apply gradient clipping to prevent explosion".to_string(),
                urgency: Urgency::Critical,
                parameters: vec![
                    ("max_norm".to_string(), "1.0".to_string()),
                ],
                alternative_actions: vec![
                    "Reduce learning rate".to_string(),
                    "Check for NaN in data".to_string(),
                ],
            },
            TrainingIssueType::GradientVanishing => SuggestedIntervention {
                action: InterventionAction::ArchitectureChange,
                description: "Consider architectural changes to improve gradient flow".to_string(),
                urgency: Urgency::Medium,
                parameters: vec![],
                alternative_actions: vec![
                    "Use residual connections".to_string(),
                    "Try different initialization".to_string(),
                    "Use layer normalization".to_string(),
                ],
            },
            _ => SuggestedIntervention {
                action: InterventionAction::Continue,
                description: "Continue training with monitoring".to_string(),
                urgency: Urgency::Low,
                parameters: vec![],
                alternative_actions: vec![],
            },
        };

        Ok(intervention)
    }

    /// Analyzes loss trend from history.
    fn analyze_loss_trend(&self, history: &[f32]) -> Option<String> {
        if history.len() < 3 {
            return None;
        }

        let recent: Vec<_> = history.iter().rev().take(5).cloned().collect();
        let is_decreasing = recent.windows(2).all(|w| w[0] <= w[1]);
        let is_increasing = recent.windows(2).all(|w| w[0] >= w[1]);

        if is_decreasing {
            Some("Loss is steadily decreasing - training progressing well".to_string())
        } else if is_increasing {
            let rate = (recent[0] - recent.last().unwrap_or(&recent[0])) / recent.len() as f32;
            if rate > 0.1 {
                Some("Loss is diverging rapidly - immediate action needed".to_string())
            } else {
                Some("Loss is slowly increasing - monitor closely".to_string())
            }
        } else {
            let variance: f32 = recent.iter()
                .map(|&x| (x - recent[0]).powi(2))
                .sum::<f32>() / recent.len() as f32;

            if variance < 0.001 {
                Some("Loss has reached a plateau".to_string())
            } else {
                Some("Loss is fluctuating - training may be unstable".to_string())
            }
        }
    }

    /// Gets recommended actions based on health and insights.
    fn get_recommended_actions(&self, health: &RunHealth, _insights: &[String]) -> Vec<String> {
        let mut actions = Vec::new();

        match health {
            RunHealth::Critical => {
                actions.push("Stop training immediately".to_string());
                actions.push("Review recent changes to hyperparameters".to_string());
                actions.push("Consider reverting to last known good checkpoint".to_string());
            }
            RunHealth::Warning => {
                actions.push("Increase monitoring frequency".to_string());
                actions.push("Prepare early stopping criteria".to_string());
            }
            RunHealth::Healthy => {
                actions.push("Continue training".to_string());
            }
        }

        actions
    }

    /// Estimates remaining epochs until convergence.
    fn estimate_remaining_epochs(&self, metrics: &TrainingMetrics) -> Option<u32> {
        if metrics.loss_history.len() < 5 {
            return None;
        }

        // Simple linear extrapolation
        let recent: Vec<_> = metrics.loss_history.iter().rev().take(5).cloned().collect();
        let improvement_rate = (recent.last().unwrap_or(&0.0) - recent[0]) / 5.0;

        if improvement_rate >= 0.0 {
            // Not improving
            return Some(0);
        }

        // Estimate epochs to reach a target loss (e.g., 0.1)
        let current = recent[0];
        let target = 0.1f32;

        if current <= target {
            Some(0)
        } else {
            let epochs = ((current - target) / (-improvement_rate)).ceil() as u32;
            Some(epochs.min(100)) // Cap at 100
        }
    }
}

/// Response structure for LLM analysis (fields populated by deserialization).
#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
struct LlmAnalysisResponse {
    /// Health status.
    health: String,
    /// Detected issues.
    #[serde(default)]
    issues: Vec<LlmIssue>,
    /// Insights from analysis.
    #[serde(default)]
    insights: Vec<String>,
    /// Suggested actions.
    #[serde(default)]
    suggestions: Vec<String>,
    /// Whether training should stop.
    #[serde(default)]
    should_stop: bool,
    /// Estimated epochs remaining.
    #[serde(default)]
    estimated_epochs_remaining: Option<u32>,
}

/// Issue structure from LLM response (fields populated by deserialization).
#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
struct LlmIssue {
    /// Issue type.
    #[serde(rename = "type")]
    issue_type: String,
    /// Severity (1-5).
    severity: u8,
    /// Evidence for the issue.
    #[serde(default)]
    evidence: Vec<String>,
}

/// Training metrics for analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Current epoch number.
    pub current_epoch: u32,
    /// Current training loss.
    pub train_loss: Option<f32>,
    /// Current validation loss.
    pub val_loss: Option<f32>,
    /// Loss history over epochs.
    pub loss_history: Vec<f32>,
    /// Current learning rate.
    pub learning_rate: Option<f32>,
    /// Gradient norm.
    pub gradient_norm: Option<f32>,
    /// Tokens processed per second.
    pub tokens_per_second: Option<f32>,
    /// GPU memory usage (bytes).
    pub gpu_memory_used: Option<u64>,
}

/// Health status of a training run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunHealth {
    /// Training is progressing well.
    Healthy,
    /// Some concerns, needs monitoring.
    Warning,
    /// Serious issues, action needed.
    Critical,
}

/// Analysis of a training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunAnalysis {
    /// Overall health status.
    pub health: RunHealth,
    /// Insights from analysis.
    pub insights: Vec<String>,
    /// Recommended actions.
    pub recommended_actions: Vec<String>,
    /// Whether training should stop.
    pub should_stop: bool,
    /// Estimated epochs remaining.
    pub estimated_epochs_remaining: Option<u32>,
}

/// A suggested intervention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestedIntervention {
    /// The action to take.
    pub action: InterventionAction,
    /// Description.
    pub description: String,
    /// Urgency level.
    pub urgency: Urgency,
    /// Parameters for the action.
    pub parameters: Vec<(String, String)>,
    /// Alternative actions.
    pub alternative_actions: Vec<String>,
}

/// Types of intervention actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterventionAction {
    /// Continue training as-is.
    Continue,
    /// Stop training early.
    EarlyStopping,
    /// Reduce learning rate.
    ReduceLearningRate,
    /// Apply learning rate schedule.
    LearningRateSchedule,
    /// Apply gradient clipping.
    GradientClipping,
    /// Change model architecture.
    ArchitectureChange,
    /// Restart from checkpoint.
    RestartFromCheckpoint,
}

/// Urgency of an intervention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Urgency {
    /// Low urgency, can wait.
    Low,
    /// Medium urgency.
    Medium,
    /// High urgency, act soon.
    High,
    /// Critical, act immediately.
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_analyze_healthy_run() {
        let coach = TrainingCoachAgent::new(None);

        let metrics = TrainingMetrics {
            current_epoch: 5,
            train_loss: Some(0.5),
            val_loss: Some(0.55),
            loss_history: vec![1.0, 0.8, 0.7, 0.6, 0.5],
            learning_rate: Some(1e-4),
            gradient_norm: Some(1.0),
            tokens_per_second: Some(1000.0),
            gpu_memory_used: Some(8_000_000_000),
        };

        let analysis = coach.analyze_run(&metrics).await.expect("analyze");

        assert_eq!(analysis.health, RunHealth::Healthy);
        assert!(!analysis.should_stop);
    }

    #[tokio::test]
    async fn test_detect_overfitting() {
        let coach = TrainingCoachAgent::new(None);

        let metrics = TrainingMetrics {
            current_epoch: 10,
            train_loss: Some(0.1),
            val_loss: Some(0.8), // Much higher than train
            loss_history: vec![0.5, 0.4, 0.3, 0.2, 0.1],
            learning_rate: Some(1e-4),
            gradient_norm: None,
            tokens_per_second: None,
            gpu_memory_used: None,
        };

        let issues = coach.detect_issues(&metrics).await.expect("detect");

        assert!(!issues.is_empty());
        assert!(issues.iter().any(|i| i.issue_type == TrainingIssueType::Overfitting));
    }

    #[tokio::test]
    async fn test_detect_divergence() {
        let coach = TrainingCoachAgent::new(None);

        let metrics = TrainingMetrics {
            current_epoch: 5,
            train_loss: Some(2.0),
            val_loss: Some(2.5),
            loss_history: vec![0.5, 0.8, 1.2, 1.6, 2.0], // Increasing
            learning_rate: Some(1e-3),
            gradient_norm: None,
            tokens_per_second: None,
            gpu_memory_used: None,
        };

        let issues = coach.detect_issues(&metrics).await.expect("detect");

        assert!(issues.iter().any(|i| i.issue_type == TrainingIssueType::Divergence));
    }

    #[tokio::test]
    async fn test_coach_with_llm_detects_overfitting() {
        use crate::llm::MockLlmClient;

        // Mock LLM returns overfitting analysis
        let llm = Arc::new(MockLlmClient::new().with_json(serde_json::json!({
            "health": "warning",
            "issues": [
                {
                    "type": "overfitting",
                    "severity": 4,
                    "evidence": ["validation loss increasing while training loss decreasing"]
                }
            ],
            "insights": ["Model is memorizing training data"],
            "suggestions": ["Add dropout", "Reduce learning rate", "Use early stopping"],
            "should_stop": false,
            "estimated_epochs_remaining": 5
        })));

        let coach = TrainingCoachAgent::with_llm(llm);
        assert!(coach.has_llm());

        let metrics = TrainingMetrics {
            current_epoch: 10,
            train_loss: Some(0.1),
            val_loss: Some(0.8),
            loss_history: vec![0.5, 0.4, 0.3, 0.2, 0.1],
            learning_rate: Some(1e-4),
            gradient_norm: None,
            tokens_per_second: None,
            gpu_memory_used: None,
        };

        let analysis = coach.analyze_run(&metrics).await.expect("analyze");

        assert_eq!(analysis.health, RunHealth::Warning);
        assert!(!analysis.insights.is_empty());
        assert!(analysis.insights.iter().any(|i| i.contains("memorizing")));
        assert!(!analysis.recommended_actions.is_empty());
    }

    #[tokio::test]
    async fn test_coach_with_llm_detects_critical() {
        use crate::llm::MockLlmClient;

        // Mock LLM returns critical analysis
        let llm = Arc::new(MockLlmClient::new().with_json(serde_json::json!({
            "health": "critical",
            "issues": [
                {
                    "type": "divergence",
                    "severity": 5,
                    "evidence": ["loss exploding", "gradient norm > 1000"]
                }
            ],
            "insights": ["Training is diverging rapidly"],
            "suggestions": ["Stop training immediately", "Reduce learning rate by 10x"],
            "should_stop": true,
            "estimated_epochs_remaining": null
        })));

        let coach = TrainingCoachAgent::with_llm(llm);

        let metrics = TrainingMetrics {
            current_epoch: 5,
            train_loss: Some(100.0),
            val_loss: Some(150.0),
            loss_history: vec![0.5, 1.0, 5.0, 20.0, 100.0],
            learning_rate: Some(1e-2),
            gradient_norm: Some(1500.0),
            tokens_per_second: None,
            gpu_memory_used: None,
        };

        let analysis = coach.analyze_run(&metrics).await.expect("analyze");

        assert_eq!(analysis.health, RunHealth::Critical);
        assert!(analysis.should_stop);
    }

    #[tokio::test]
    async fn test_coach_fallback_on_llm_error() {
        use crate::llm::MockLlmClient;

        // Mock LLM returns an error
        let llm = Arc::new(MockLlmClient::new().with_error("Connection failed"));

        let coach = TrainingCoachAgent::with_llm(llm);

        let metrics = TrainingMetrics {
            current_epoch: 5,
            train_loss: Some(0.5),
            val_loss: Some(0.55),
            loss_history: vec![1.0, 0.8, 0.7, 0.6, 0.5],
            learning_rate: Some(1e-4),
            gradient_norm: Some(1.0),
            tokens_per_second: None,
            gpu_memory_used: None,
        };

        // Should fall back to heuristic analysis
        let analysis = coach.analyze_run(&metrics).await.expect("analyze");

        // Heuristic analysis should work
        assert_eq!(analysis.health, RunHealth::Healthy);
    }

    #[tokio::test]
    async fn test_coach_without_llm() {
        let coach = TrainingCoachAgent::new(None);
        assert!(!coach.has_llm());

        let metrics = TrainingMetrics {
            current_epoch: 5,
            train_loss: Some(0.5),
            val_loss: Some(0.55),
            loss_history: vec![1.0, 0.8, 0.7, 0.6, 0.5],
            learning_rate: Some(1e-4),
            gradient_norm: Some(1.0),
            tokens_per_second: None,
            gpu_memory_used: None,
        };

        // Should use heuristic analysis
        let analysis = coach.analyze_run(&metrics).await.expect("analyze");
        assert_eq!(analysis.health, RunHealth::Healthy);
    }
}
