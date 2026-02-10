//! GPU metrics collection for Prometheus monitoring.
//!
//! This module provides GPU utilization metrics that can be exposed via the
//! `/metrics` endpoint. It supports both real GPU monitoring (via NVML) and
//! mock implementations for testing and CPU-only environments.
//!
//! # Example
//!
//! ```ignore
//! use infernum_server::gpu_metrics::{GpuMetrics, MockGpuMetrics};
//!
//! let metrics = MockGpuMetrics::new();
//! metrics.set_utilization(0, 0.85);
//! metrics.set_memory_used(0, 8 * 1024 * 1024 * 1024); // 8GB
//!
//! println!("{}", metrics.render_prometheus());
//! ```

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// GPU device information snapshot.
#[derive(Debug, Clone, Default)]
pub struct GpuInfo {
    /// GPU device index.
    pub index: u32,
    /// Device name (e.g., "NVIDIA GeForce RTX 4090").
    pub name: String,
    /// GPU utilization (0.0 - 1.0).
    pub utilization: f64,
    /// Memory used in bytes.
    pub memory_used: u64,
    /// Total memory in bytes.
    pub memory_total: u64,
    /// Temperature in Celsius.
    pub temperature: Option<u32>,
    /// Power usage in watts.
    pub power_watts: Option<f64>,
}

impl GpuInfo {
    /// Creates a new GPU info with basic data.
    #[must_use]
    pub fn new(index: u32, name: impl Into<String>) -> Self {
        Self {
            index,
            name: name.into(),
            ..Default::default()
        }
    }

    /// Returns memory utilization as a ratio (0.0 - 1.0).
    #[must_use]
    pub fn memory_utilization(&self) -> f64 {
        if self.memory_total == 0 {
            0.0
        } else {
            self.memory_used as f64 / self.memory_total as f64
        }
    }
}

/// Trait for GPU metrics providers.
///
/// Implementations can provide real GPU data (NVML) or mock data for testing.
pub trait GpuMetricsProvider: Send + Sync {
    /// Returns the number of GPUs available.
    fn gpu_count(&self) -> u32;

    /// Returns information for a specific GPU.
    fn gpu_info(&self, index: u32) -> Option<GpuInfo>;

    /// Returns information for all GPUs.
    fn all_gpus(&self) -> Vec<GpuInfo> {
        (0..self.gpu_count())
            .filter_map(|i| self.gpu_info(i))
            .collect()
    }

    /// Renders Prometheus-format metrics.
    fn render_prometheus(&self) -> String {
        let gpus = self.all_gpus();

        if gpus.is_empty() {
            return String::new();
        }

        let mut output = String::new();

        // GPU utilization
        output.push_str("# HELP infernum_gpu_utilization GPU utilization percentage (0.0-1.0)\n");
        output.push_str("# TYPE infernum_gpu_utilization gauge\n");
        for gpu in &gpus {
            output.push_str(&format!(
                "infernum_gpu_utilization{{gpu=\"{}\",name=\"{}\"}} {:.4}\n",
                gpu.index, gpu.name, gpu.utilization
            ));
        }
        output.push('\n');

        // Memory used
        output.push_str("# HELP infernum_gpu_memory_used_bytes GPU memory used in bytes\n");
        output.push_str("# TYPE infernum_gpu_memory_used_bytes gauge\n");
        for gpu in &gpus {
            output.push_str(&format!(
                "infernum_gpu_memory_used_bytes{{gpu=\"{}\",name=\"{}\"}} {}\n",
                gpu.index, gpu.name, gpu.memory_used
            ));
        }
        output.push('\n');

        // Memory total
        output.push_str("# HELP infernum_gpu_memory_total_bytes GPU total memory in bytes\n");
        output.push_str("# TYPE infernum_gpu_memory_total_bytes gauge\n");
        for gpu in &gpus {
            output.push_str(&format!(
                "infernum_gpu_memory_total_bytes{{gpu=\"{}\",name=\"{}\"}} {}\n",
                gpu.index, gpu.name, gpu.memory_total
            ));
        }
        output.push('\n');

        // Memory utilization (derived)
        output
            .push_str("# HELP infernum_gpu_memory_utilization GPU memory utilization (0.0-1.0)\n");
        output.push_str("# TYPE infernum_gpu_memory_utilization gauge\n");
        for gpu in &gpus {
            output.push_str(&format!(
                "infernum_gpu_memory_utilization{{gpu=\"{}\",name=\"{}\"}} {:.4}\n",
                gpu.index,
                gpu.name,
                gpu.memory_utilization()
            ));
        }

        // Temperature (if available)
        let has_temp = gpus.iter().any(|g| g.temperature.is_some());
        if has_temp {
            output.push('\n');
            output.push_str("# HELP infernum_gpu_temperature_celsius GPU temperature in Celsius\n");
            output.push_str("# TYPE infernum_gpu_temperature_celsius gauge\n");
            for gpu in &gpus {
                if let Some(temp) = gpu.temperature {
                    output.push_str(&format!(
                        "infernum_gpu_temperature_celsius{{gpu=\"{}\",name=\"{}\"}} {}\n",
                        gpu.index, gpu.name, temp
                    ));
                }
            }
        }

        // Power usage (if available)
        let has_power = gpus.iter().any(|g| g.power_watts.is_some());
        if has_power {
            output.push('\n');
            output.push_str("# HELP infernum_gpu_power_watts GPU power usage in watts\n");
            output.push_str("# TYPE infernum_gpu_power_watts gauge\n");
            for gpu in &gpus {
                if let Some(power) = gpu.power_watts {
                    output.push_str(&format!(
                        "infernum_gpu_power_watts{{gpu=\"{}\",name=\"{}\"}} {:.2}\n",
                        gpu.index, gpu.name, power
                    ));
                }
            }
        }

        output
    }
}

/// Mock GPU metrics provider for testing.
#[derive(Debug, Clone)]
pub struct MockGpuMetrics {
    gpus: Arc<RwLock<HashMap<u32, GpuInfo>>>,
}

impl Default for MockGpuMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl MockGpuMetrics {
    /// Creates a new mock GPU metrics provider with no GPUs.
    #[must_use]
    pub fn new() -> Self {
        Self {
            gpus: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Creates a mock with a single simulated GPU.
    #[must_use]
    pub fn single_gpu() -> Self {
        let mock = Self::new();
        mock.add_gpu(GpuInfo {
            index: 0,
            name: "Mock GPU 0".to_string(),
            utilization: 0.0,
            memory_used: 0,
            memory_total: 16 * 1024 * 1024 * 1024, // 16 GB
            temperature: Some(45),
            power_watts: Some(50.0),
        });
        mock
    }

    /// Creates a mock with multiple simulated GPUs.
    #[must_use]
    pub fn multi_gpu(count: u32) -> Self {
        let mock = Self::new();
        for i in 0..count {
            mock.add_gpu(GpuInfo {
                index: i,
                name: format!("Mock GPU {}", i),
                utilization: 0.0,
                memory_used: 0,
                memory_total: 16 * 1024 * 1024 * 1024,
                temperature: Some(45),
                power_watts: Some(50.0),
            });
        }
        mock
    }

    /// Adds a GPU to the mock.
    pub fn add_gpu(&self, gpu: GpuInfo) {
        if let Ok(mut gpus) = self.gpus.write() {
            gpus.insert(gpu.index, gpu);
        }
    }

    /// Sets utilization for a specific GPU.
    pub fn set_utilization(&self, index: u32, utilization: f64) {
        if let Ok(mut gpus) = self.gpus.write() {
            if let Some(gpu) = gpus.get_mut(&index) {
                gpu.utilization = utilization.clamp(0.0, 1.0);
            }
        }
    }

    /// Sets memory used for a specific GPU.
    pub fn set_memory_used(&self, index: u32, memory_used: u64) {
        if let Ok(mut gpus) = self.gpus.write() {
            if let Some(gpu) = gpus.get_mut(&index) {
                gpu.memory_used = memory_used;
            }
        }
    }

    /// Sets temperature for a specific GPU.
    pub fn set_temperature(&self, index: u32, temperature: u32) {
        if let Ok(mut gpus) = self.gpus.write() {
            if let Some(gpu) = gpus.get_mut(&index) {
                gpu.temperature = Some(temperature);
            }
        }
    }

    /// Sets power usage for a specific GPU.
    pub fn set_power_watts(&self, index: u32, power: f64) {
        if let Ok(mut gpus) = self.gpus.write() {
            if let Some(gpu) = gpus.get_mut(&index) {
                gpu.power_watts = Some(power);
            }
        }
    }
}

impl GpuMetricsProvider for MockGpuMetrics {
    fn gpu_count(&self) -> u32 {
        self.gpus.read().map(|g| g.len() as u32).unwrap_or(0)
    }

    fn gpu_info(&self, index: u32) -> Option<GpuInfo> {
        self.gpus.read().ok()?.get(&index).cloned()
    }
}

/// No-op GPU metrics provider for CPU-only systems.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoGpuMetrics;

impl GpuMetricsProvider for NoGpuMetrics {
    fn gpu_count(&self) -> u32 {
        0
    }

    fn gpu_info(&self, _index: u32) -> Option<GpuInfo> {
        None
    }
}

/// GPU metrics wrapper that can hold any provider.
pub struct GpuMetrics {
    provider: Box<dyn GpuMetricsProvider>,
}

impl GpuMetrics {
    /// Creates GPU metrics with the given provider.
    pub fn new(provider: impl GpuMetricsProvider + 'static) -> Self {
        Self {
            provider: Box::new(provider),
        }
    }

    /// Creates GPU metrics with no GPU support.
    #[must_use]
    pub fn none() -> Self {
        Self::new(NoGpuMetrics)
    }

    /// Creates GPU metrics with a mock provider.
    #[must_use]
    pub fn mock() -> Self {
        Self::new(MockGpuMetrics::single_gpu())
    }

    /// Returns the number of GPUs.
    pub fn gpu_count(&self) -> u32 {
        self.provider.gpu_count()
    }

    /// Returns info for a specific GPU.
    pub fn gpu_info(&self, index: u32) -> Option<GpuInfo> {
        self.provider.gpu_info(index)
    }

    /// Returns info for all GPUs.
    pub fn all_gpus(&self) -> Vec<GpuInfo> {
        self.provider.all_gpus()
    }

    /// Renders Prometheus-format metrics.
    pub fn render_prometheus(&self) -> String {
        self.provider.render_prometheus()
    }
}

impl Default for GpuMetrics {
    fn default() -> Self {
        Self::none()
    }
}

impl std::fmt::Debug for GpuMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuMetrics")
            .field("gpu_count", &self.gpu_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_info_new() {
        let gpu = GpuInfo::new(0, "Test GPU");
        assert_eq!(gpu.index, 0);
        assert_eq!(gpu.name, "Test GPU");
        assert_eq!(gpu.utilization, 0.0);
    }

    #[test]
    fn test_gpu_info_memory_utilization() {
        let gpu = GpuInfo {
            index: 0,
            name: "Test".to_string(),
            memory_used: 4 * 1024 * 1024 * 1024,   // 4 GB
            memory_total: 16 * 1024 * 1024 * 1024, // 16 GB
            ..Default::default()
        };
        assert!((gpu.memory_utilization() - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_gpu_info_memory_utilization_zero_total() {
        let gpu = GpuInfo::default();
        assert_eq!(gpu.memory_utilization(), 0.0);
    }

    #[test]
    fn test_mock_gpu_metrics_new() {
        let mock = MockGpuMetrics::new();
        assert_eq!(mock.gpu_count(), 0);
    }

    #[test]
    fn test_mock_gpu_metrics_single_gpu() {
        let mock = MockGpuMetrics::single_gpu();
        assert_eq!(mock.gpu_count(), 1);
        let gpu = mock.gpu_info(0).expect("GPU 0 should exist");
        assert_eq!(gpu.name, "Mock GPU 0");
    }

    #[test]
    fn test_mock_gpu_metrics_multi_gpu() {
        let mock = MockGpuMetrics::multi_gpu(4);
        assert_eq!(mock.gpu_count(), 4);
        for i in 0..4 {
            let gpu = mock.gpu_info(i).expect("GPU should exist");
            assert_eq!(gpu.name, format!("Mock GPU {}", i));
        }
    }

    #[test]
    fn test_mock_set_utilization() {
        let mock = MockGpuMetrics::single_gpu();
        mock.set_utilization(0, 0.85);
        let gpu = mock.gpu_info(0).expect("GPU 0 should exist");
        assert!((gpu.utilization - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_mock_set_utilization_clamped() {
        let mock = MockGpuMetrics::single_gpu();
        mock.set_utilization(0, 1.5);
        let gpu = mock.gpu_info(0).expect("GPU 0 should exist");
        assert_eq!(gpu.utilization, 1.0);

        mock.set_utilization(0, -0.5);
        let gpu = mock.gpu_info(0).expect("GPU 0 should exist");
        assert_eq!(gpu.utilization, 0.0);
    }

    #[test]
    fn test_mock_set_memory_used() {
        let mock = MockGpuMetrics::single_gpu();
        let mem = 8 * 1024 * 1024 * 1024u64;
        mock.set_memory_used(0, mem);
        let gpu = mock.gpu_info(0).expect("GPU 0 should exist");
        assert_eq!(gpu.memory_used, mem);
    }

    #[test]
    fn test_mock_set_temperature() {
        let mock = MockGpuMetrics::single_gpu();
        mock.set_temperature(0, 75);
        let gpu = mock.gpu_info(0).expect("GPU 0 should exist");
        assert_eq!(gpu.temperature, Some(75));
    }

    #[test]
    fn test_mock_set_power_watts() {
        let mock = MockGpuMetrics::single_gpu();
        mock.set_power_watts(0, 250.5);
        let gpu = mock.gpu_info(0).expect("GPU 0 should exist");
        assert!((gpu.power_watts.unwrap_or(0.0) - 250.5).abs() < 0.01);
    }

    #[test]
    fn test_no_gpu_metrics() {
        let no_gpu = NoGpuMetrics;
        assert_eq!(no_gpu.gpu_count(), 0);
        assert!(no_gpu.gpu_info(0).is_none());
        assert!(no_gpu.all_gpus().is_empty());
        assert!(no_gpu.render_prometheus().is_empty());
    }

    #[test]
    fn test_gpu_metrics_wrapper_none() {
        let metrics = GpuMetrics::none();
        assert_eq!(metrics.gpu_count(), 0);
    }

    #[test]
    fn test_gpu_metrics_wrapper_mock() {
        let metrics = GpuMetrics::mock();
        assert_eq!(metrics.gpu_count(), 1);
    }

    #[test]
    fn test_render_prometheus_single_gpu() {
        let mock = MockGpuMetrics::single_gpu();
        mock.set_utilization(0, 0.85);
        mock.set_memory_used(0, 8 * 1024 * 1024 * 1024);

        let output = mock.render_prometheus();

        assert!(output.contains("infernum_gpu_utilization"));
        assert!(output.contains("infernum_gpu_memory_used_bytes"));
        assert!(output.contains("infernum_gpu_memory_total_bytes"));
        assert!(output.contains("infernum_gpu_memory_utilization"));
        assert!(output.contains("0.8500")); // utilization
        assert!(output.contains("8589934592")); // 8GB in bytes
    }

    #[test]
    fn test_render_prometheus_multi_gpu() {
        let mock = MockGpuMetrics::multi_gpu(2);
        mock.set_utilization(0, 0.50);
        mock.set_utilization(1, 0.75);

        let output = mock.render_prometheus();

        assert!(output.contains("gpu=\"0\""));
        assert!(output.contains("gpu=\"1\""));
        assert!(output.contains("0.5000"));
        assert!(output.contains("0.7500"));
    }

    #[test]
    fn test_render_prometheus_includes_temperature() {
        let mock = MockGpuMetrics::single_gpu();
        mock.set_temperature(0, 72);

        let output = mock.render_prometheus();

        assert!(output.contains("infernum_gpu_temperature_celsius"));
        assert!(output.contains("72"));
    }

    #[test]
    fn test_render_prometheus_includes_power() {
        let mock = MockGpuMetrics::single_gpu();
        mock.set_power_watts(0, 185.5);

        let output = mock.render_prometheus();

        assert!(output.contains("infernum_gpu_power_watts"));
        assert!(output.contains("185.50"));
    }

    #[test]
    fn test_render_prometheus_empty_when_no_gpus() {
        let mock = MockGpuMetrics::new();
        let output = mock.render_prometheus();
        assert!(output.is_empty());
    }

    #[test]
    fn test_all_gpus() {
        let mock = MockGpuMetrics::multi_gpu(3);
        let gpus = mock.all_gpus();
        assert_eq!(gpus.len(), 3);
    }

    #[test]
    fn test_gpu_info_nonexistent() {
        let mock = MockGpuMetrics::single_gpu();
        assert!(mock.gpu_info(99).is_none());
    }

    #[test]
    fn test_gpu_metrics_debug() {
        let metrics = GpuMetrics::mock();
        let debug_str = format!("{:?}", metrics);
        assert!(debug_str.contains("GpuMetrics"));
        assert!(debug_str.contains("gpu_count"));
    }
}
