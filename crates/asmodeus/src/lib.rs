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

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod config;
pub mod lora;
pub mod trainer;

pub use config::{LoraConfig, TrainingConfig};
pub use lora::{find_target_modules, LoraLayer, LoraModel};
pub use trainer::{
    AdamW, DataLoader, Dataset, InMemoryDataset, LRScheduler, Trainer, TrainerTrait, TrainingRun,
    TrainingSample, TrainingStatus,
};
