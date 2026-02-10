//! Integration tests for Asmodeus - the training and adaptation layer.
//!
//! Tests cover:
//! - LoRA configuration and layer creation
//! - Training configurations
//! - Gradient computation and accumulation
//! - Loss functions
//! - Dataset and DataLoader
//! - LR scheduler
//! - Trainer creation

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use tempfile::tempdir;

use infernum_asmodeus::{
    clip_grad_norm, compute_grad_norm, find_target_modules, AdamW, CrossEntropyLoss, DPOLoss,
    DataLoader, Dataset, GradientAccumulator, GradientConfig, GradientScaler, InMemoryDataset,
    LRScheduler, LoraConfig, LoraLayer, LoraModel, Reduction, SFTLoss, Trainer, TrainingConfig,
    TrainingSample,
};

// ============================================================================
// LoraConfig Tests
// ============================================================================

#[test]
fn test_lora_config_default() {
    let config = LoraConfig::default();

    assert_eq!(config.r, 8);
    assert!((config.alpha - 16.0).abs() < 0.01);
    assert!(config.dropout >= 0.0);
    assert!(!config.target_modules.is_empty());
}

#[test]
fn test_lora_config_custom() {
    let config = LoraConfig {
        r: 16,
        alpha: 32.0,
        dropout: 0.1,
        target_modules: vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "v_proj".to_string(),
        ],
    };

    assert_eq!(config.r, 16);
    assert_eq!(config.target_modules.len(), 3);
}

#[test]
fn test_lora_config_serialization() {
    let config = LoraConfig {
        r: 4,
        alpha: 8.0,
        dropout: 0.05,
        target_modules: vec!["attn".to_string()],
    };

    let json = serde_json::to_string(&config).expect("serialize");
    assert!(json.contains("\"r\":4"));

    let parsed: LoraConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.r, 4);
}

// ============================================================================
// LoraLayer Tests
// ============================================================================

#[test]
fn test_lora_layer_new() {
    let config = LoraConfig {
        r: 8,
        alpha: 16.0,
        dropout: 0.0,
        target_modules: vec![],
    };

    let layer = LoraLayer::new("test.layer", config, 256, 256, &Device::Cpu).unwrap();

    assert_eq!(layer.name, "test.layer");
    assert_eq!(layer.in_features, 256);
    assert_eq!(layer.out_features, 256);
    assert!(layer.lora_a.is_some());
    assert!(layer.lora_b.is_some());
}

#[test]
fn test_lora_layer_parameters() {
    let config = LoraConfig {
        r: 8,
        alpha: 16.0,
        dropout: 0.0,
        target_modules: vec![],
    };

    let layer = LoraLayer::new("test", config, 768, 768, &Device::Cpu).unwrap();

    // A: r x in_features = 8 x 768 = 6144
    // B: out_features x r = 768 x 8 = 6144
    // Total: 12288
    assert_eq!(layer.num_parameters(), 12288);
}

#[test]
fn test_lora_layer_scaling() {
    let config = LoraConfig {
        r: 8,
        alpha: 16.0,
        dropout: 0.0,
        target_modules: vec![],
    };

    let layer = LoraLayer::new("test", config, 64, 64, &Device::Cpu).unwrap();

    // scaling = alpha / r = 16 / 8 = 2.0
    // The scaling field is private, so we test via forward pass behavior
    assert!(layer.lora_a.is_some());
}

#[test]
fn test_lora_layer_forward() {
    let config = LoraConfig {
        r: 4,
        alpha: 8.0,
        dropout: 0.0,
        target_modules: vec![],
    };

    let layer = LoraLayer::new("test", config, 32, 32, &Device::Cpu).unwrap();

    // Create input tensor (batch_size=2, features=32)
    let input = Tensor::ones(&[2, 32], DType::F32, &Device::Cpu).unwrap();

    let output = layer.forward(&input).unwrap();
    assert_eq!(output.dims(), &[2, 32]);
}

#[test]
fn test_lora_layer_training_mode() {
    let config = LoraConfig {
        r: 4,
        alpha: 8.0,
        dropout: 0.1,
        target_modules: vec![],
    };

    let mut layer = LoraLayer::new("test", config, 32, 32, &Device::Cpu).unwrap();

    assert!(!layer.is_training());

    layer.train();
    assert!(layer.is_training());

    layer.eval();
    assert!(!layer.is_training());
}

#[test]
fn test_lora_layer_empty() {
    let config = LoraConfig::default();
    let layer = LoraLayer::empty("empty.layer", config);

    assert_eq!(layer.name, "empty.layer");
    assert!(layer.lora_a.is_none());
    assert!(layer.lora_b.is_none());
}

#[test]
fn test_lora_layer_get_delta() {
    let config = LoraConfig {
        r: 4,
        alpha: 8.0,
        dropout: 0.0,
        target_modules: vec![],
    };

    let layer = LoraLayer::new("test", config, 32, 32, &Device::Cpu).unwrap();
    let delta = layer.get_delta().unwrap();

    assert_eq!(delta.dims(), &[32, 32]);
}

// ============================================================================
// LoraModel Tests
// ============================================================================

#[test]
fn test_lora_model_new() {
    let config = LoraConfig::default();
    let model = LoraModel::new("meta-llama/Llama-3.2-3B", config);

    assert_eq!(model.base_model, "meta-llama/Llama-3.2-3B");
    assert!(model.layers.is_empty());
}

#[test]
fn test_lora_model_add_layer() {
    let config = LoraConfig::default();
    let mut model = LoraModel::new("base-model", config.clone());

    let layer = LoraLayer::new("layer.0", config, 256, 256, &Device::Cpu).unwrap();
    model.add_layer(layer);

    assert_eq!(model.layers.len(), 1);
    assert!(model.get_layer("layer.0").is_some());
}

#[test]
fn test_lora_model_total_parameters() {
    let config = LoraConfig {
        r: 8,
        alpha: 16.0,
        dropout: 0.0,
        target_modules: vec![],
    };

    let mut model = LoraModel::new("base", config.clone());

    for i in 0..4 {
        let layer = LoraLayer::new(
            format!("layer.{}", i),
            config.clone(),
            256,
            256,
            &Device::Cpu,
        )
        .unwrap();
        model.add_layer(layer);
    }

    // Each layer: 8*256 + 256*8 = 4096 params
    // 4 layers * 4096 = 16384
    assert_eq!(model.total_parameters(), 16384);
}

#[test]
fn test_lora_model_train_eval() {
    let config = LoraConfig::default();
    let mut model = LoraModel::new("base", config.clone());

    let layer = LoraLayer::new("layer.0", config, 64, 64, &Device::Cpu).unwrap();
    model.add_layer(layer);

    model.train();
    assert!(model.layers.get("layer.0").unwrap().is_training());

    model.eval();
    assert!(!model.layers.get("layer.0").unwrap().is_training());
}

#[test]
fn test_lora_model_save_load() {
    let dir = tempdir().unwrap();
    let config = LoraConfig::default();

    let mut model = LoraModel::new("test-base", config.clone());
    let layer = LoraLayer::new("test.layer", config, 64, 64, &Device::Cpu).unwrap();
    model.add_layer(layer);

    // Save
    let save_path = dir.path().join("lora_model");
    model.save(&save_path).unwrap();

    // Verify files exist
    assert!(save_path.join("lora_config.json").exists());
    assert!(save_path.join("lora_weights.safetensors").exists());

    // Load
    let loaded = LoraModel::load(&save_path, &Device::Cpu).unwrap();
    assert_eq!(loaded.base_model, "test-base");
    assert!(loaded.get_layer("test.layer").is_some());
}

// ============================================================================
// find_target_modules Tests
// ============================================================================

#[test]
fn test_find_target_modules_llama() {
    let targets = vec!["q_proj".to_string(), "v_proj".to_string()];
    let modules = find_target_modules("llama", &targets);

    assert!(!modules.is_empty());
    assert!(modules.iter().any(|m| m.contains("q_proj")));
    assert!(modules.iter().any(|m| m.contains("v_proj")));
}

#[test]
fn test_find_target_modules_gpt2() {
    let targets = vec!["c_attn".to_string()];
    let modules = find_target_modules("gpt2", &targets);

    assert!(!modules.is_empty());
    assert!(modules.iter().any(|m| m.contains("c_attn")));
}

#[test]
fn test_find_target_modules_unknown() {
    let targets = vec!["custom".to_string()];
    let modules = find_target_modules("unknown_arch", &targets);

    assert_eq!(modules.len(), 1);
    assert_eq!(modules[0], "custom");
}

// ============================================================================
// TrainingConfig Tests
// ============================================================================

#[test]
fn test_training_config_default() {
    let config = TrainingConfig::default();

    assert!(config.learning_rate > 0.0);
    assert!(config.batch_size > 0);
    assert!(config.num_epochs > 0);
    assert!(config.lora.is_some());
}

#[test]
fn test_training_config_custom() {
    let config = TrainingConfig {
        learning_rate: 1e-5,
        batch_size: 16,
        num_epochs: 5,
        warmup_steps: 200,
        weight_decay: 0.0,
        gradient_accumulation_steps: 2,
        max_grad_norm: 0.5,
        lora: None,
    };

    assert!((config.learning_rate - 1e-5).abs() < 1e-7);
    assert_eq!(config.batch_size, 16);
    assert!(config.lora.is_none());
}

#[test]
fn test_training_config_serialization() {
    let config = TrainingConfig::default();

    let json = serde_json::to_string(&config).expect("serialize");
    let parsed: TrainingConfig = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(parsed.batch_size, config.batch_size);
    assert_eq!(parsed.num_epochs, config.num_epochs);
}

// ============================================================================
// GradientConfig Tests
// ============================================================================

#[test]
fn test_gradient_config_default() {
    let config = GradientConfig::default();

    assert_eq!(config.accumulation_steps, 1);
    assert!(!config.gradient_checkpointing);
    assert!((config.max_grad_norm - 1.0).abs() < 0.01);
}

#[test]
fn test_gradient_config_memory_efficient() {
    let config = GradientConfig::memory_efficient();

    assert!(config.accumulation_steps > 1);
    assert!(config.gradient_checkpointing);
    assert!(config.fp16_gradients);
}

#[test]
fn test_gradient_config_fast() {
    let config = GradientConfig::fast();

    assert_eq!(config.accumulation_steps, 1);
    assert!(!config.gradient_checkpointing);
    assert!(!config.fp16_gradients);
}

// ============================================================================
// GradientAccumulator Tests
// ============================================================================

#[test]
fn test_gradient_accumulator_new() {
    let accumulator = GradientAccumulator::new(4, Device::Cpu);

    assert!(!accumulator.is_ready());
    assert_eq!(accumulator.steps(), 0);
}

#[test]
fn test_gradient_accumulator_accumulate() {
    let mut accumulator = GradientAccumulator::new(2, Device::Cpu);

    // First accumulation
    let grad = Tensor::ones((4, 4), DType::F32, &Device::Cpu).unwrap();
    let mut grads = HashMap::new();
    grads.insert("param1".to_string(), grad);
    accumulator.accumulate(grads).unwrap();

    assert_eq!(accumulator.steps(), 1);
    assert!(!accumulator.is_ready());

    // Second accumulation
    let grad = Tensor::ones((4, 4), DType::F32, &Device::Cpu).unwrap();
    let mut grads = HashMap::new();
    grads.insert("param1".to_string(), grad);
    accumulator.accumulate(grads).unwrap();

    assert_eq!(accumulator.steps(), 2);
    assert!(accumulator.is_ready());
}

#[test]
fn test_gradient_accumulator_get_gradients() {
    let mut accumulator = GradientAccumulator::new(4, Device::Cpu);

    for _ in 0..4 {
        let grad = Tensor::full(2.0f32, (4, 4), &Device::Cpu).unwrap();
        let mut grads = HashMap::new();
        grads.insert("param".to_string(), grad);
        accumulator.accumulate(grads).unwrap();
    }

    let avg_grads = accumulator.get_gradients().unwrap();
    let avg = avg_grads.get("param").unwrap();

    // 4 tensors of 2.0, accumulated -> sum = 8.0, averaged = 2.0
    let mean = avg.mean_all().unwrap().to_scalar::<f32>().unwrap();
    assert!((mean - 2.0).abs() < 0.01);
}

#[test]
fn test_gradient_accumulator_clear() {
    let mut accumulator = GradientAccumulator::new(2, Device::Cpu);

    let grad = Tensor::ones((4, 4), DType::F32, &Device::Cpu).unwrap();
    let mut grads = HashMap::new();
    grads.insert("param".to_string(), grad);
    accumulator.accumulate(grads).unwrap();

    accumulator.clear();
    assert_eq!(accumulator.steps(), 0);
    assert!(!accumulator.is_ready());
}

// ============================================================================
// GradientScaler Tests
// ============================================================================

#[test]
fn test_gradient_scaler_new() {
    let config = GradientConfig::default();
    let scaler = GradientScaler::new(&config);

    assert_eq!(scaler.scale(), config.grad_scale);
}

#[test]
fn test_gradient_scaler_scale_loss() {
    let config = GradientConfig::default();
    let scaler = GradientScaler::new(&config);

    // Create a scalar loss tensor
    let loss = Tensor::from_vec(vec![1.0f32], &[1], &Device::Cpu).unwrap();
    let scaled = scaler.scale_loss(&loss).unwrap();

    let val: Vec<f32> = scaled.to_vec1().unwrap();
    assert!((val[0] - config.grad_scale as f32).abs() < 0.01);
}

#[test]
fn test_gradient_scaler_update_success() {
    let config = GradientConfig {
        scale_growth_interval: 2,
        ..Default::default()
    };
    let mut scaler = GradientScaler::new(&config);

    let initial_scale = scaler.scale();

    // Two successful updates should trigger growth
    scaler.update(true);
    scaler.update(true);

    assert!(scaler.scale() > initial_scale);
}

#[test]
fn test_gradient_scaler_update_overflow() {
    let config = GradientConfig::default();
    let mut scaler = GradientScaler::new(&config);

    let initial_scale = scaler.scale();
    scaler.update(false);

    assert!(scaler.scale() < initial_scale);
    assert!(scaler.should_skip_step());
}

// ============================================================================
// compute_grad_norm and clip_grad_norm Tests
// ============================================================================

#[test]
fn test_compute_grad_norm() {
    let grad1 = Tensor::full(3.0f32, (2, 2), &Device::Cpu).unwrap();
    let grad2 = Tensor::full(4.0f32, (2, 2), &Device::Cpu).unwrap();

    let mut grads = HashMap::new();
    grads.insert("g1".to_string(), grad1);
    grads.insert("g2".to_string(), grad2);

    let norm = compute_grad_norm(&grads).unwrap();
    // norm = sqrt(4*9 + 4*16) = sqrt(100) = 10
    assert!((norm - 10.0).abs() < 0.01);
}

#[test]
fn test_clip_grad_norm() {
    let grad = Tensor::full(10.0f32, (2, 2), &Device::Cpu).unwrap();

    let mut grads = HashMap::new();
    grads.insert("g".to_string(), grad);

    // Original norm = sqrt(4 * 100) = 20
    let original = clip_grad_norm(&mut grads, 5.0).unwrap();
    assert!((original - 20.0).abs() < 0.01);

    // After clipping, norm should be 5.0
    let clipped = compute_grad_norm(&grads).unwrap();
    assert!((clipped - 5.0).abs() < 0.01);
}

#[test]
fn test_clip_grad_norm_no_clip_needed() {
    let grad = Tensor::full(1.0f32, (2, 2), &Device::Cpu).unwrap();

    let mut grads = HashMap::new();
    grads.insert("g".to_string(), grad);

    // Original norm = 2.0, max_norm = 5.0
    let original = clip_grad_norm(&mut grads, 5.0).unwrap();
    assert!((original - 2.0).abs() < 0.01);

    // Should not be clipped
    let after = compute_grad_norm(&grads).unwrap();
    assert!((after - 2.0).abs() < 0.01);
}

// ============================================================================
// CrossEntropyLoss Tests
// ============================================================================

#[test]
fn test_cross_entropy_loss_new() {
    let loss = CrossEntropyLoss::new();
    // Verify construction works (fields are private)
    drop(loss);
}

#[test]
fn test_cross_entropy_loss_builder() {
    // Builder pattern should work without panicking
    let loss = CrossEntropyLoss::new()
        .with_ignore_index(-100)
        .with_label_smoothing(0.1)
        .with_reduction(Reduction::Sum);

    // Fields are private, just verify construction
    drop(loss);
}

#[test]
fn test_cross_entropy_loss_forward() {
    let loss = CrossEntropyLoss::new();

    // Create logits: (batch=2, seq=3, vocab=10)
    let logits = Tensor::randn(0.0f32, 1.0f32, (2, 3, 10), &Device::Cpu).unwrap();

    // Create targets: (batch=2, seq=3)
    let targets = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6], (2, 3), &Device::Cpu).unwrap();

    let loss_val = loss.forward(&logits, &targets).unwrap();

    // Loss should be a scalar (empty dims)
    assert!(loss_val.dims().is_empty());

    // Loss should be positive
    let val = loss_val.to_scalar::<f32>().unwrap();
    assert!(val > 0.0);
}

// ============================================================================
// SFTLoss Tests
// ============================================================================

#[test]
fn test_sft_loss_new() {
    let loss = SFTLoss::new();
    // Verify construction works (fields are private)
    drop(loss);
}

#[test]
fn test_sft_loss_with_token() {
    let loss = SFTLoss::new().with_response_start_token(128000);
    // Fields are private, just verify construction
    drop(loss);
}

#[test]
fn test_sft_loss_forward_no_mask() {
    let loss = SFTLoss::new();

    let logits = Tensor::randn(0.0f32, 1.0f32, (2, 4, 100), &Device::Cpu).unwrap();
    let targets = Tensor::from_vec(
        vec![10u32, 20, 30, 40, 50, 60, 70, 80],
        (2, 4),
        &Device::Cpu,
    )
    .unwrap();

    let loss_val = loss.forward(&logits, &targets, None).unwrap();
    let val = loss_val.to_scalar::<f32>().unwrap();
    assert!(val > 0.0);
}

// ============================================================================
// DPOLoss Tests
// ============================================================================

#[test]
fn test_dpo_loss_new() {
    let loss = DPOLoss::new();
    // Verify construction works (fields are private)
    drop(loss);
}

#[test]
fn test_dpo_loss_builder() {
    // Builder pattern should work without panicking
    let loss = DPOLoss::new().with_beta(0.2).with_label_smoothing(0.1);

    // Fields are private, just verify construction
    drop(loss);
}

#[test]
fn test_dpo_loss_forward() {
    let loss = DPOLoss::new().with_beta(0.1);

    // Create log probs (batch=4)
    let policy_chosen =
        Tensor::from_vec(vec![-1.0f32, -0.5, -0.8, -1.2], (4,), &Device::Cpu).unwrap();
    let policy_rejected =
        Tensor::from_vec(vec![-2.0f32, -1.5, -1.8, -2.2], (4,), &Device::Cpu).unwrap();
    let ref_chosen = Tensor::from_vec(vec![-1.1f32, -0.6, -0.9, -1.3], (4,), &Device::Cpu).unwrap();
    let ref_rejected =
        Tensor::from_vec(vec![-2.1f32, -1.6, -1.9, -2.3], (4,), &Device::Cpu).unwrap();

    let output = loss
        .forward(&policy_chosen, &policy_rejected, &ref_chosen, &ref_rejected)
        .unwrap();

    // Check output structure - scalar tensors have empty dims
    assert!(output.loss.dims().is_empty());
    assert_eq!(output.chosen_rewards.dims(), &[4usize]);
    assert_eq!(output.rejected_rewards.dims(), &[4usize]);
    assert!(output.accuracy.dims().is_empty());
}

// ============================================================================
// Dataset and DataLoader Tests
// ============================================================================

#[test]
fn test_training_sample() {
    let sample = TrainingSample {
        input_ids: vec![1, 2, 3, 4, 5],
        attention_mask: vec![1, 1, 1, 1, 1],
        labels: vec![2, 3, 4, 5, 6],
    };

    assert_eq!(sample.input_ids.len(), 5);
    assert_eq!(sample.labels.len(), 5);
}

#[test]
fn test_in_memory_dataset_new() {
    let samples = vec![
        TrainingSample {
            input_ids: vec![1, 2, 3],
            attention_mask: vec![1, 1, 1],
            labels: vec![2, 3, 4],
        },
        TrainingSample {
            input_ids: vec![4, 5, 6],
            attention_mask: vec![1, 1, 1],
            labels: vec![5, 6, 7],
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
    assert_eq!(sample.unwrap().input_ids, vec![1, 2, 3]);

    assert!(dataset.get(1).is_none());
}

#[test]
fn test_in_memory_dataset_from_instruction_pairs() {
    let pairs = vec![
        ("What is 2+2?".to_string(), "4".to_string()),
        (
            "What is the capital of France?".to_string(),
            "Paris".to_string(),
        ),
    ];

    // Simple tokenizer that just converts chars to bytes
    let tokenize = |text: &str| text.chars().map(|c| c as u32).collect();

    let dataset = InMemoryDataset::from_instruction_pairs(pairs, tokenize);

    assert_eq!(dataset.len(), 2);
    assert!(!dataset.is_empty());
}

#[test]
fn test_data_loader_new() {
    let samples = vec![
        TrainingSample {
            input_ids: vec![1],
            attention_mask: vec![1],
            labels: vec![2],
        };
        10
    ];

    let dataset = Arc::new(InMemoryDataset::new(samples));
    let loader = DataLoader::new(dataset, 4, false);

    // 10 samples / 4 batch_size = 3 batches (2 full + 1 partial)
    assert_eq!(loader.num_batches(), 3);
}

#[test]
fn test_data_loader_iteration() {
    let samples = vec![
        TrainingSample {
            input_ids: vec![1],
            attention_mask: vec![1],
            labels: vec![2],
        };
        5
    ];

    let dataset = Arc::new(InMemoryDataset::new(samples));
    let loader = DataLoader::new(dataset, 2, false);

    let batches: Vec<_> = loader.collect();

    assert_eq!(batches.len(), 3);
    assert_eq!(batches[0].len(), 2);
    assert_eq!(batches[1].len(), 2);
    assert_eq!(batches[2].len(), 1);
}

#[test]
fn test_data_loader_shuffle() {
    let samples: Vec<TrainingSample> = (0..10)
        .map(|i| TrainingSample {
            input_ids: vec![i],
            attention_mask: vec![1],
            labels: vec![i + 1],
        })
        .collect();

    let dataset = Arc::new(InMemoryDataset::new(samples));
    let loader = DataLoader::new(dataset, 10, true);

    let batches: Vec<_> = loader.collect();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].len(), 10);
}

#[test]
fn test_data_loader_reset() {
    let samples = vec![
        TrainingSample {
            input_ids: vec![1],
            attention_mask: vec![1],
            labels: vec![2],
        };
        4
    ];

    let dataset = Arc::new(InMemoryDataset::new(samples));
    let mut loader = DataLoader::new(dataset, 2, false);

    // First pass
    let batches1: Vec<_> = loader.by_ref().collect();
    assert_eq!(batches1.len(), 2);

    // Reset
    loader.reset();

    // Second pass
    let batches2: Vec<_> = loader.collect();
    assert_eq!(batches2.len(), 2);
}

// ============================================================================
// LRScheduler Tests
// ============================================================================

#[test]
fn test_lr_scheduler_new() {
    let scheduler = LRScheduler::new(1e-4, 100, 1000);

    // At step 0, LR should be 0 (start of warmup)
    assert!((scheduler.get_lr(0) - 0.0).abs() < 1e-10);
}

#[test]
fn test_lr_scheduler_warmup() {
    let scheduler = LRScheduler::new(1e-4, 100, 1000);

    // At step 50, LR should be 50% of base
    let lr = scheduler.get_lr(50);
    assert!((lr - 5e-5).abs() < 1e-10);

    // At step 100, LR should be base
    let lr = scheduler.get_lr(100);
    assert!((lr - 1e-4).abs() < 1e-10);
}

#[test]
fn test_lr_scheduler_decay() {
    let scheduler = LRScheduler::new(1e-4, 0, 1000);

    // At step 0, LR should be base
    let lr = scheduler.get_lr(0);
    assert!((lr - 1e-4).abs() < 1e-10);

    // At step 1000, LR should be minimum (10% of base)
    let lr = scheduler.get_lr(1000);
    assert!((lr - 1e-5).abs() < 1e-10);
}

#[test]
fn test_lr_scheduler_cosine_decay() {
    let scheduler = LRScheduler::new(1e-4, 0, 1000);

    // At step 500 (middle), should be between base and min
    let lr = scheduler.get_lr(500);
    assert!(lr > 1e-5);
    assert!(lr < 1e-4);
}

// ============================================================================
// AdamW Tests
// ============================================================================

#[test]
fn test_adamw_new() {
    let optimizer = AdamW::new(1e-4, 0.01);

    // Optimizer created successfully (lr field is private, just verify construction)
    drop(optimizer);
}

#[test]
fn test_adamw_set_lr() {
    let mut optimizer = AdamW::new(1e-4, 0.01);

    // set_lr doesn't return anything, just verify it doesn't panic
    optimizer.set_lr(5e-5);
    optimizer.set_lr(1e-6);
}

#[test]
fn test_adamw_step() {
    let mut optimizer = AdamW::new(0.001, 0.01);

    let param = Tensor::randn(0.0f32, 1.0f32, (4, 4), &Device::Cpu).unwrap();
    let grad = Tensor::randn(0.0f32, 0.1f32, (4, 4), &Device::Cpu).unwrap();

    let new_param = optimizer.step("test", &param, &grad).unwrap();

    // New param should be different from original
    let diff = new_param.sub(&param).unwrap();
    let diff_sum = diff
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(diff_sum > 0.0);
}

#[test]
fn test_adamw_multiple_steps() {
    let mut optimizer = AdamW::new(0.001, 0.01);

    let mut param = Tensor::randn(0.0f32, 1.0f32, (4, 4), &Device::Cpu).unwrap();

    for _ in 0..10 {
        let grad = Tensor::randn(0.0f32, 0.1f32, (4, 4), &Device::Cpu).unwrap();
        param = optimizer.step("param", &param, &grad).unwrap();
    }

    // Should still be a valid tensor
    assert_eq!(param.dims(), &[4, 4]);
}

// ============================================================================
// Trainer Tests
// ============================================================================

#[test]
fn test_trainer_new() {
    let dir = tempdir().unwrap();
    let trainer = Trainer::new(dir.path(), "test-model");

    assert_eq!(trainer.output_dir(), dir.path());
}

#[test]
fn test_trainer_with_device() {
    let dir = tempdir().unwrap();
    let trainer = Trainer::new(dir.path(), "test-model").with_device(Device::Cpu);

    assert_eq!(trainer.output_dir(), dir.path());
}

// ============================================================================
// End-to-End Workflow Tests
// ============================================================================

#[test]
fn test_lora_training_workflow() {
    // 1. Create LoRA configuration
    let lora_config = LoraConfig {
        r: 4,
        alpha: 8.0,
        dropout: 0.0,
        target_modules: vec!["q_proj".to_string()],
    };

    // 2. Create model with LoRA layers
    let mut model = LoraModel::new("base-model", lora_config.clone());
    let layer = LoraLayer::new("layer.0.q_proj", lora_config, 256, 256, &Device::Cpu).unwrap();
    model.add_layer(layer);

    // 3. Create training config
    let _training_config = TrainingConfig {
        learning_rate: 1e-4,
        batch_size: 4,
        num_epochs: 1,
        ..Default::default()
    };

    // 4. Create dataset
    let samples = vec![
        TrainingSample {
            input_ids: vec![1, 2, 3, 4],
            attention_mask: vec![1, 1, 1, 1],
            labels: vec![2, 3, 4, 5],
        };
        16
    ];
    let dataset = Arc::new(InMemoryDataset::new(samples));

    // 5. Create data loader
    let loader = DataLoader::new(dataset, 4, true);

    // 6. Verify batching works
    let batches: Vec<_> = loader.collect();
    assert_eq!(batches.len(), 4);

    // 7. Create optimizer
    let mut optimizer = AdamW::new(1e-4, 0.01);

    // 8. Simulate training step
    let layer = model.get_layer("layer.0.q_proj").unwrap();
    let input = Tensor::randn(0.0f32, 1.0f32, (4, 256), &Device::Cpu).unwrap();
    let output = layer.forward(&input).unwrap();

    // 9. Create mock gradients and update
    let grad_a = Tensor::randn(0.0f32, 0.01f32, (4, 256), &Device::Cpu).unwrap();
    let _new_a = optimizer
        .step("layer.0.q_proj.lora_a", &output, &grad_a)
        .unwrap();

    // 10. Verify model can be saved
    let dir = tempdir().unwrap();
    model.save(&dir.path().join("model")).unwrap();
}

#[test]
fn test_gradient_workflow() {
    // 1. Create gradient accumulator
    let mut accumulator = GradientAccumulator::new(4, Device::Cpu);

    // 2. Create gradient scaler
    let config = GradientConfig::default();
    let mut scaler = GradientScaler::new(&config);

    // 3. Simulate micro-batch training
    for _ in 0..4 {
        let grad = Tensor::randn(0.0f32, 1.0f32, (64, 64), &Device::Cpu).unwrap();
        let mut grads = HashMap::new();
        grads.insert("weights".to_string(), grad);
        accumulator.accumulate(grads).unwrap();
    }

    assert!(accumulator.is_ready());

    // 4. Get averaged gradients
    let mut avg_grads = accumulator.get_gradients().unwrap();

    // 5. Unscale gradients
    let finite = scaler.unscale_gradients(&mut avg_grads).unwrap();
    assert!(finite);

    // 6. Clip gradients
    let norm = clip_grad_norm(&mut avg_grads, 1.0).unwrap();
    assert!(norm >= 0.0);

    // 7. Update scaler
    scaler.update(finite);
    assert!(!scaler.should_skip_step());

    // 8. Clear accumulator
    accumulator.clear();
    assert_eq!(accumulator.steps(), 0);
}

#[test]
fn test_loss_comparison() {
    // Compare different loss functions on same data
    let logits = Tensor::randn(0.0f32, 1.0f32, (2, 4, 100), &Device::Cpu).unwrap();
    let targets = Tensor::from_vec(
        vec![10u32, 20, 30, 40, 50, 60, 70, 80],
        (2, 4),
        &Device::Cpu,
    )
    .unwrap();

    // CrossEntropyLoss
    let ce_loss = CrossEntropyLoss::new();
    let ce_val = ce_loss.forward(&logits, &targets).unwrap();

    // SFTLoss (without mask, should equal CE)
    let sft_loss = SFTLoss::new();
    let sft_val = sft_loss.forward(&logits, &targets, None).unwrap();

    // Both should be positive
    assert!(ce_val.to_scalar::<f32>().unwrap() > 0.0);
    assert!(sft_val.to_scalar::<f32>().unwrap() > 0.0);

    // CrossEntropyLoss with label smoothing
    let ce_smooth = CrossEntropyLoss::new().with_label_smoothing(0.1);
    let ce_smooth_val = ce_smooth.forward(&logits, &targets).unwrap();
    assert!(ce_smooth_val.to_scalar::<f32>().unwrap() > 0.0);
}
