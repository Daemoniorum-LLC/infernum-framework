//! # Asmodeus
//!
//! *"The King of Demons shapes minds"*
//!
//! Asmodeus is the adaptation layer for the Infernum ecosystem,
//! providing model fine-tuning, LoRA training, and prompt optimization.
//!
//! ## Features
//!
//! - **LoRA/QLoRA**: Low-rank adaptation for efficient fine-tuning
//! - **DPO/ORPO**: Preference optimization methods
//! - **Prompt Optimization**: Automatic prompt engineering
//! - **Sigil Specialist**: Training for Sigil-specialized models (Jormungandr)

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![deny(clippy::unwrap_used)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod config;
pub mod gradient;
pub mod lora;
pub mod sigil_specialist;
pub mod trainer;

pub use config::{LoraConfig, TrainingConfig};
pub use lora::{find_target_modules, LoraLayer, LoraModel};
pub use sigil_specialist::{
    CheckpointCollector, SigilDataset, SigilTrainer, SigilTrainingConfig, Specialization,
    TrainingPair, TrainingSource,
};
pub use trainer::{
    AdamW, DataLoader, Dataset, InMemoryDataset, LRScheduler, Trainer, TrainerTrait, TrainingRun,
    TrainingSample, TrainingStatus,
};
pub use gradient::{
    clip_grad_norm, compute_grad_norm, CrossEntropyLoss, DPOLoss, DPOOutput, GradientAccumulator,
    GradientConfig, GradientScaler, Reduction, SFTLoss,
};

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use sigil_specialist::CollectorConfig;
    use std::sync::Arc;

    // === Config Module Tests ===

    #[test]
    fn test_lora_config_default() {
        let config = LoraConfig::default();
        assert_eq!(config.r, 8);
        assert!((config.alpha - 16.0).abs() < 0.01);
        assert!((config.dropout - 0.05).abs() < 0.01);
        assert_eq!(config.target_modules.len(), 2);
    }

    #[test]
    fn test_lora_config_custom() {
        let config = LoraConfig {
            r: 16,
            alpha: 32.0,
            dropout: 0.1,
            target_modules: vec!["q_proj".to_string(), "k_proj".to_string()],
        };
        assert_eq!(config.r, 16);
        assert!((config.alpha - 32.0).abs() < 0.01);
    }

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert!((config.learning_rate - 2e-4).abs() < 1e-6);
        assert_eq!(config.batch_size, 4);
        assert_eq!(config.num_epochs, 3);
        assert!(config.lora.is_some());
    }

    #[test]
    fn test_training_config_custom() {
        let config = TrainingConfig {
            learning_rate: 1e-5,
            batch_size: 8,
            num_epochs: 10,
            warmup_steps: 500,
            weight_decay: 0.0,
            gradient_accumulation_steps: 1,
            max_grad_norm: 0.5,
            lora: None,
        };
        assert!((config.learning_rate - 1e-5).abs() < 1e-7);
        assert_eq!(config.batch_size, 8);
        assert!(config.lora.is_none());
    }

    // === LoRA Module Tests ===

    #[test]
    fn test_find_target_modules_llama() {
        let target = vec!["q_proj".to_string(), "v_proj".to_string()];
        let result = find_target_modules("llama", &target);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_find_target_modules_empty() {
        let target: Vec<String> = vec![];
        let result = find_target_modules("unknown", &target);
        // Returns some default modules
        let _ = result;
    }

    // === Sigil Specialist Module Tests ===

    #[test]
    fn test_sigil_training_config_default() {
        let config = SigilTrainingConfig::default();
        assert!(config.learning_rate > 0.0);
        assert_eq!(config.epochs, 3);
        assert_eq!(config.batch_size, 4);
    }

    #[test]
    fn test_sigil_training_config_lightweight() {
        let config = SigilTrainingConfig::lightweight();
        assert_eq!(config.epochs, 1);
        assert_eq!(config.batch_size, 2);
    }

    #[test]
    fn test_sigil_training_config_full() {
        let config = SigilTrainingConfig::full();
        assert_eq!(config.epochs, 5);
        assert_eq!(config.batch_size, 8);
    }

    #[test]
    fn test_specialization_general() {
        let spec = Specialization::General;
        assert!(matches!(spec, Specialization::General));
        assert!((spec.training_weight() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_specialization_syntax_completion() {
        let spec = Specialization::SyntaxCompletion;
        assert!(matches!(spec, Specialization::SyntaxCompletion));
        assert!((spec.training_weight() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_specialization_evidentiality() {
        let spec = Specialization::EvidentialityInference;
        assert!((spec.training_weight() - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_training_source_synthetic() {
        let source = TrainingSource::Synthetic {
            method: "generated".to_string(),
        };
        assert!(matches!(source, TrainingSource::Synthetic { .. }));
    }

    #[test]
    fn test_training_source_curated() {
        let source = TrainingSource::Curated {
            curator: "human".to_string(),
        };
        assert!(matches!(source, TrainingSource::Curated { .. }));
    }

    #[test]
    fn test_training_source_checkpoint() {
        let source = TrainingSource::Checkpoint {
            checkpoint_id: "cp-001".to_string(),
            project: "test".to_string(),
        };
        assert!(matches!(source, TrainingSource::Checkpoint { .. }));
    }

    #[test]
    fn test_training_pair_new() {
        let pair = TrainingPair::new(
            "test input",
            "test output",
            TrainingSource::Synthetic {
                method: "test".to_string(),
            },
        );
        assert_eq!(pair.input, "test input");
        assert_eq!(pair.output, "test output");
        assert!((pair.quality_score - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_training_pair_with_quality() {
        let pair = TrainingPair::new(
            "input",
            "output",
            TrainingSource::Curated {
                curator: "test".to_string(),
            },
        )
        .with_quality(0.9);
        assert!((pair.quality_score - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_training_pair_with_specialization() {
        let pair = TrainingPair::new(
            "input",
            "output",
            TrainingSource::Synthetic {
                method: "test".to_string(),
            },
        )
        .with_specialization(Specialization::MigrationExpertise);
        assert!(matches!(pair.specialization, Specialization::MigrationExpertise));
    }

    // === Trainer Module Tests ===

    #[test]
    fn test_training_status_pending() {
        let status = TrainingStatus::Pending;
        assert!(matches!(status, TrainingStatus::Pending));
    }

    #[test]
    fn test_training_status_running() {
        let status = TrainingStatus::Running {
            epoch: 1,
            step: 100,
            total_steps: 1000,
            loss: 0.5,
            learning_rate: 1e-4,
        };
        assert!(matches!(status, TrainingStatus::Running { .. }));
    }

    #[test]
    fn test_training_status_completed() {
        let status = TrainingStatus::Completed {
            final_loss: 0.1,
            total_steps: 1000,
            output_path: std::path::PathBuf::from("/output"),
        };
        assert!(matches!(status, TrainingStatus::Completed { .. }));
    }

    #[test]
    fn test_training_status_failed() {
        let status = TrainingStatus::Failed {
            error: "test error".to_string(),
        };
        assert!(matches!(status, TrainingStatus::Failed { .. }));
    }

    #[test]
    fn test_training_sample_construction() {
        let sample = TrainingSample {
            input_ids: vec![1, 2, 3, 4],
            attention_mask: vec![1, 1, 1, 1],
            labels: vec![2, 3, 4, 5],
        };
        assert_eq!(sample.input_ids.len(), 4);
        assert_eq!(sample.attention_mask.len(), 4);
        assert_eq!(sample.labels.len(), 4);
    }

    #[test]
    fn test_lr_scheduler_warmup() {
        let scheduler = LRScheduler::new(1e-4, 100, 1000);
        // At step 0, LR should be 0
        assert!((scheduler.get_lr(0) - 0.0).abs() < 1e-10);
        // At step 50, LR should be 0.5 * base
        assert!((scheduler.get_lr(50) - 5e-5).abs() < 1e-10);
        // At step 100, LR should be base
        assert!((scheduler.get_lr(100) - 1e-4).abs() < 1e-10);
    }

    #[test]
    fn test_adamw_optimizer() {
        // Just verify it constructs
        let _optimizer = AdamW::new(0.001, 0.01);
    }

    #[test]
    fn test_in_memory_dataset() {
        let samples = vec![
            TrainingSample {
                input_ids: vec![1, 2, 3],
                attention_mask: vec![1, 1, 1],
                labels: vec![2, 3, 4],
            },
            TrainingSample {
                input_ids: vec![5, 6, 7],
                attention_mask: vec![1, 1, 1],
                labels: vec![6, 7, 8],
            },
        ];
        let dataset = InMemoryDataset::new(samples);
        assert_eq!(dataset.len(), 2);
        assert!(!dataset.is_empty());
    }

    #[test]
    fn test_in_memory_dataset_get() {
        let samples = vec![TrainingSample {
            input_ids: vec![1, 2, 3],
            attention_mask: vec![1, 1, 1],
            labels: vec![2, 3, 4],
        }];
        let dataset = InMemoryDataset::new(samples);
        let sample = dataset.get(0);
        assert!(sample.is_some());
        assert_eq!(sample.as_ref().map(|s| s.input_ids[0]), Some(1));
    }

    #[test]
    fn test_in_memory_dataset_empty() {
        let dataset = InMemoryDataset::new(vec![]);
        assert!(dataset.is_empty());
        assert_eq!(dataset.len(), 0);
        assert!(dataset.get(0).is_none());
    }

    #[test]
    fn test_data_loader() {
        let samples = vec![
            TrainingSample {
                input_ids: vec![1],
                attention_mask: vec![1],
                labels: vec![2],
            },
            TrainingSample {
                input_ids: vec![3],
                attention_mask: vec![1],
                labels: vec![4],
            },
            TrainingSample {
                input_ids: vec![5],
                attention_mask: vec![1],
                labels: vec![6],
            },
        ];
        let dataset = Arc::new(InMemoryDataset::new(samples));
        let loader = DataLoader::new(dataset, 2, false);
        assert_eq!(loader.num_batches(), 2);
    }

    // === Gradient Module Tests ===

    #[test]
    fn test_gradient_config_default() {
        let config = GradientConfig::default();
        assert_eq!(config.accumulation_steps, 1);
        assert!((config.max_grad_norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_gradient_config_memory_efficient() {
        let config = GradientConfig::memory_efficient();
        assert_eq!(config.accumulation_steps, 4);
        assert!(config.gradient_checkpointing);
    }

    #[test]
    fn test_gradient_config_fast() {
        let config = GradientConfig::fast();
        assert_eq!(config.accumulation_steps, 1);
        assert!(!config.gradient_checkpointing);
    }

    #[test]
    fn test_gradient_scaler() {
        let config = GradientConfig::default();
        let scaler = GradientScaler::new(&config);
        assert!((scaler.scale() - 65536.0).abs() < 0.01);
    }

    #[test]
    fn test_gradient_accumulator() {
        let device = Device::Cpu;
        let accumulator = GradientAccumulator::new(4, device);
        assert!(!accumulator.is_ready());
        assert_eq!(accumulator.steps(), 0);
    }

    #[test]
    fn test_reduction_mean() {
        let reduction = Reduction::Mean;
        assert!(matches!(reduction, Reduction::Mean));
    }

    #[test]
    fn test_reduction_sum() {
        let reduction = Reduction::Sum;
        assert!(matches!(reduction, Reduction::Sum));
    }

    #[test]
    fn test_reduction_none() {
        let reduction = Reduction::None;
        assert!(matches!(reduction, Reduction::None));
    }

    #[test]
    fn test_cross_entropy_loss_new() {
        // Just verify it constructs
        let _loss = CrossEntropyLoss::new();
    }

    #[test]
    fn test_cross_entropy_loss_with_ignore_index() {
        // Just verify builder pattern works
        let _loss = CrossEntropyLoss::new().with_ignore_index(-100);
    }

    #[test]
    fn test_cross_entropy_loss_with_label_smoothing() {
        // Just verify builder pattern works
        let _loss = CrossEntropyLoss::new().with_label_smoothing(0.1);
    }

    #[test]
    fn test_cross_entropy_loss_with_reduction() {
        // Just verify builder pattern works
        let _loss = CrossEntropyLoss::new().with_reduction(Reduction::Sum);
    }

    #[test]
    fn test_sft_loss_new() {
        // Just verify it constructs
        let _loss = SFTLoss::new();
    }

    #[test]
    fn test_sft_loss_with_response_token() {
        // Just verify builder pattern works
        let _loss = SFTLoss::new().with_response_start_token(128000);
    }

    #[test]
    fn test_dpo_loss_new() {
        // Just verify it constructs
        let _loss = DPOLoss::new();
    }

    #[test]
    fn test_dpo_loss_with_beta() {
        // Just verify builder pattern works
        let _loss = DPOLoss::new().with_beta(0.2);
    }

    #[test]
    fn test_dpo_loss_with_label_smoothing() {
        // Just verify builder pattern works
        let _loss = DPOLoss::new().with_label_smoothing(0.1);
    }

    // === Utility Function Tests ===

    #[test]
    fn test_clip_grad_norm_function() {
        // Just verify the function signature is accessible
        let _ = std::any::type_name_of_val(&clip_grad_norm);
    }

    #[test]
    fn test_compute_grad_norm_function() {
        // Just verify the function signature is accessible
        let _ = std::any::type_name_of_val(&compute_grad_norm);
    }

    // === Sigil Dataset Tests ===

    #[test]
    fn test_sigil_dataset_new() {
        let dataset = SigilDataset::new("test-dataset");
        assert_eq!(dataset.pairs.len(), 0);
        assert_eq!(dataset.metadata.name, "test-dataset");
    }

    #[test]
    fn test_sigil_dataset_add_pair() {
        let mut dataset = SigilDataset::new("test");
        let pair = TrainingPair::new(
            "input",
            "output",
            TrainingSource::Curated {
                curator: "test".to_string(),
            },
        );
        dataset.add(pair);
        assert_eq!(dataset.pairs.len(), 1);
    }

    #[test]
    fn test_sigil_dataset_filter_by_quality() {
        let mut dataset = SigilDataset::new("test");
        dataset.add(
            TrainingPair::new("a", "b", TrainingSource::Curated {
                curator: "t".to_string(),
            })
            .with_quality(0.8),
        );
        dataset.add(
            TrainingPair::new("c", "d", TrainingSource::Curated {
                curator: "t".to_string(),
            })
            .with_quality(0.4),
        );

        let filtered = dataset.filter_by_quality(0.6);
        assert_eq!(filtered.pairs.len(), 1);
    }

    #[test]
    fn test_sigil_dataset_stats() {
        let mut dataset = SigilDataset::new("test");
        dataset.add(
            TrainingPair::new("a", "b", TrainingSource::Curated {
                curator: "t".to_string(),
            })
            .with_quality(0.8)
            .with_specialization(Specialization::SyntaxCompletion),
        );

        let stats = dataset.stats();
        assert_eq!(stats.total_pairs, 1);
        assert!((stats.avg_quality - 0.8).abs() < 0.01);
    }

    // === Checkpoint Collector Tests ===

    #[test]
    fn test_checkpoint_collector_new() {
        let config = CollectorConfig::default();
        let collector = CheckpointCollector::new(config);
        let dataset = collector.dataset();
        assert_eq!(dataset.pairs.len(), 0);
    }

    #[test]
    fn test_checkpoint_collector_config_default() {
        let config = CollectorConfig::default();
        assert!((config.min_joy_intensity - 0.5).abs() < 0.01);
        assert!(config.include_frictions);
        assert!(config.include_patterns);
    }

    #[test]
    fn test_checkpoint_collector_collect_from_conversion() {
        let config = CollectorConfig::default();
        let collector = CheckpointCollector::new(config);

        collector.collect_from_conversion(
            "cp-001",
            "test-project",
            "fn hello() {}",
            "fn hello() {}",
        );

        let dataset = collector.dataset();
        assert_eq!(dataset.pairs.len(), 1);
    }

    #[test]
    fn test_checkpoint_collector_collect_from_pattern() {
        let config = CollectorConfig::default();
        let collector = CheckpointCollector::new(config);

        collector.collect_from_pattern("builder", "Builder pattern for APIs", "struct Builder {}");

        let dataset = collector.dataset();
        assert_eq!(dataset.pairs.len(), 1);
    }

    #[test]
    fn test_checkpoint_collector_clear() {
        let config = CollectorConfig::default();
        let collector = CheckpointCollector::new(config);

        collector.collect_from_pattern("test", "desc", "code");
        assert_eq!(collector.dataset().pairs.len(), 1);

        collector.clear();
        assert_eq!(collector.dataset().pairs.len(), 0);
    }

    // === Trainer Tests ===

    #[test]
    fn test_trainer_new() {
        let trainer = Trainer::new("/tmp/output", "test-model");
        assert_eq!(trainer.output_dir(), &std::path::PathBuf::from("/tmp/output"));
    }

    #[test]
    fn test_trainer_with_device() {
        let trainer = Trainer::new("/tmp", "model").with_device(Device::Cpu);
        let _ = format!("{:?}", trainer.output_dir());
    }
}
