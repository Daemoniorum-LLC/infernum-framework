//! GPU detection and information gathering.
//!
//! Detects available GPUs and their capabilities via:
//! 1. nvidia-smi (NVIDIA GPUs)
//! 2. rocm-smi (AMD GPUs)
//! 3. System fallback (generic detection)

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors from GPU detection.
#[derive(Debug, Error)]
pub enum GpuDetectionError {
    /// No GPU detected.
    #[error("No GPU detected")]
    NoGpu,

    /// Detection command failed.
    #[error("Detection command failed: {0}")]
    CommandFailed(String),

    /// Failed to parse GPU information.
    #[error("Failed to parse GPU info: {0}")]
    ParseError(String),

    /// Timeout during detection.
    #[error("Detection timed out after {0:?}")]
    Timeout(Duration),
}

/// Result type for GPU detection operations.
pub type Result<T> = std::result::Result<T, GpuDetectionError>;

/// GPU vendor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuVendor {
    /// NVIDIA GPU.
    Nvidia,
    /// AMD GPU.
    Amd,
    /// Intel GPU.
    Intel,
    /// Apple Silicon.
    Apple,
    /// Unknown vendor.
    Unknown,
}

/// Information about a detected GPU.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// Device index (0-based).
    pub device_id: u32,

    /// GPU vendor.
    pub vendor: GpuVendor,

    /// GPU name/model.
    pub name: String,

    /// Total VRAM in bytes.
    pub vram_bytes: u64,

    /// Free VRAM in bytes (at detection time).
    pub vram_free_bytes: u64,

    /// Compute capability (NVIDIA) or architecture info.
    pub compute_capability: Option<String>,

    /// Driver version.
    pub driver_version: Option<String>,
}

impl GpuInfo {
    /// Returns VRAM in gigabytes.
    pub fn vram_gb(&self) -> f64 {
        self.vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Returns free VRAM in gigabytes.
    pub fn vram_free_gb(&self) -> f64 {
        self.vram_free_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Returns VRAM utilization (0.0 - 1.0).
    pub fn vram_utilization(&self) -> f32 {
        if self.vram_bytes == 0 {
            return 0.0;
        }
        let used = self.vram_bytes.saturating_sub(self.vram_free_bytes);
        used as f32 / self.vram_bytes as f32
    }
}

/// GPU detection result.
#[derive(Debug, Clone)]
pub struct GpuDetectionResult {
    /// Detected GPUs.
    pub gpus: Vec<GpuInfo>,

    /// Total VRAM across all GPUs.
    pub total_vram_bytes: u64,

    /// Detection method used.
    pub detection_method: DetectionMethod,
}

impl GpuDetectionResult {
    /// Returns the primary (first) GPU.
    pub fn primary(&self) -> Option<&GpuInfo> {
        self.gpus.first()
    }

    /// Returns total VRAM in GB.
    pub fn total_vram_gb(&self) -> f64 {
        self.total_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Returns true if any GPU was detected.
    pub fn has_gpu(&self) -> bool {
        !self.gpus.is_empty()
    }

    /// Creates an empty result (no GPUs).
    pub fn none() -> Self {
        Self {
            gpus: vec![],
            total_vram_bytes: 0,
            detection_method: DetectionMethod::None,
        }
    }
}

/// Method used for GPU detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionMethod {
    /// NVIDIA nvidia-smi.
    NvidiaSmi,
    /// AMD rocm-smi.
    RocmSmi,
    /// Apple Metal.
    AppleMetal,
    /// Generic system detection.
    System,
    /// No detection performed.
    None,
}

/// GPU detector that tries multiple detection methods.
pub struct GpuDetector {
    /// Timeout for detection commands.
    timeout: Duration,
}

impl Default for GpuDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuDetector {
    /// Creates a new GPU detector with a 5-second timeout.
    pub fn new() -> Self {
        Self {
            timeout: Duration::from_secs(5),
        }
    }

    /// Creates a detector with custom timeout.
    pub fn with_timeout(timeout: Duration) -> Self {
        Self { timeout }
    }

    /// Runs a command with the configured timeout.
    ///
    /// Spawns the process and polls `try_wait` until it exits or the timeout
    /// elapses. On timeout the child is killed and `GpuDetectionError::Timeout`
    /// is returned.
    fn run_with_timeout(&self, cmd: &mut Command) -> Result<std::process::Output> {
        let mut child = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| GpuDetectionError::CommandFailed(e.to_string()))?;

        let start = Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(_)) => {
                    // Child exited — collect stdout/stderr.
                    return child
                        .wait_with_output()
                        .map_err(|e| GpuDetectionError::CommandFailed(e.to_string()));
                }
                Ok(None) => {
                    if start.elapsed() >= self.timeout {
                        let _ = child.kill();
                        let _ = child.wait(); // Reap zombie.
                        return Err(GpuDetectionError::Timeout(self.timeout));
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(e) => {
                    return Err(GpuDetectionError::CommandFailed(e.to_string()));
                }
            }
        }
    }

    /// Detects all available GPUs.
    pub fn detect(&self) -> Result<GpuDetectionResult> {
        // Try NVIDIA first (most common for ML)
        if let Ok(result) = self.detect_nvidia() {
            if result.has_gpu() {
                return Ok(result);
            }
        }

        // Try AMD
        if let Ok(result) = self.detect_amd() {
            if result.has_gpu() {
                return Ok(result);
            }
        }

        // Try Apple Metal on macOS
        #[cfg(target_os = "macos")]
        if let Ok(result) = self.detect_apple() {
            if result.has_gpu() {
                return Ok(result);
            }
        }

        // No GPU found
        Err(GpuDetectionError::NoGpu)
    }

    /// Detects GPUs with fallback to default config on failure.
    pub fn detect_or_default(&self, default_vram_bytes: u64) -> GpuDetectionResult {
        match self.detect() {
            Ok(result) => result,
            Err(_) => GpuDetectionResult {
                gpus: vec![GpuInfo {
                    device_id: 0,
                    vendor: GpuVendor::Unknown,
                    name: "Unknown GPU".to_string(),
                    vram_bytes: default_vram_bytes,
                    vram_free_bytes: default_vram_bytes,
                    compute_capability: None,
                    driver_version: None,
                }],
                total_vram_bytes: default_vram_bytes,
                detection_method: DetectionMethod::None,
            },
        }
    }

    /// Detects NVIDIA GPUs using nvidia-smi.
    fn detect_nvidia(&self) -> Result<GpuDetectionResult> {
        let output = self.run_with_timeout(
            Command::new("nvidia-smi").args([
                "--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ]),
        )?;

        if !output.status.success() {
            return Err(GpuDetectionError::CommandFailed(
                String::from_utf8_lossy(&output.stderr).to_string(),
            ));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut gpus = Vec::new();
        let mut total_vram = 0u64;

        for line in stdout.lines() {
            if line.trim().is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() < 4 {
                continue;
            }

            let device_id = parts[0]
                .parse::<u32>()
                .map_err(|e| GpuDetectionError::ParseError(e.to_string()))?;

            let name = parts[1].to_string();

            // nvidia-smi reports memory in MiB
            let vram_mib = parts[2]
                .parse::<u64>()
                .map_err(|e| GpuDetectionError::ParseError(e.to_string()))?;
            let vram_bytes = vram_mib * 1024 * 1024;

            let vram_free_mib = parts[3]
                .parse::<u64>()
                .map_err(|e| GpuDetectionError::ParseError(e.to_string()))?;
            let vram_free_bytes = vram_free_mib * 1024 * 1024;

            let driver_version = parts.get(4).map(|s| s.to_string());
            let compute_capability = parts.get(5).map(|s| s.to_string());

            total_vram += vram_bytes;

            gpus.push(GpuInfo {
                device_id,
                vendor: GpuVendor::Nvidia,
                name,
                vram_bytes,
                vram_free_bytes,
                compute_capability,
                driver_version,
            });
        }

        Ok(GpuDetectionResult {
            gpus,
            total_vram_bytes: total_vram,
            detection_method: DetectionMethod::NvidiaSmi,
        })
    }

    /// Detects AMD GPUs using rocm-smi.
    fn detect_amd(&self) -> Result<GpuDetectionResult> {
        let output = self.run_with_timeout(
            Command::new("rocm-smi").args(["--showmeminfo", "vram", "--json"]),
        )?;

        if !output.status.success() {
            return Err(GpuDetectionError::CommandFailed(
                String::from_utf8_lossy(&output.stderr).to_string(),
            ));
        }

        // ROCm SMI outputs JSON, but for now we'll return a simple fallback
        // Full implementation would parse the JSON
        let stdout = String::from_utf8_lossy(&output.stdout);

        // Basic parsing - look for memory values
        let mut gpus = Vec::new();
        let mut total_vram = 0u64;

        // Simplified: if rocm-smi succeeded, assume we have at least one AMD GPU
        // A real implementation would parse the JSON properly
        if stdout.contains("card") || stdout.contains("GPU") {
            gpus.push(GpuInfo {
                device_id: 0,
                vendor: GpuVendor::Amd,
                name: "AMD GPU".to_string(),
                vram_bytes: 16 * 1024 * 1024 * 1024, // Default 16GB
                vram_free_bytes: 16 * 1024 * 1024 * 1024,
                compute_capability: None,
                driver_version: None,
            });
            total_vram = 16 * 1024 * 1024 * 1024;
        }

        Ok(GpuDetectionResult {
            gpus,
            total_vram_bytes: total_vram,
            detection_method: DetectionMethod::RocmSmi,
        })
    }

    /// Detects Apple Metal GPUs on macOS.
    #[cfg(target_os = "macos")]
    fn detect_apple(&self) -> Result<GpuDetectionResult> {
        // Use system_profiler to get GPU info
        let output = self.run_with_timeout(
            Command::new("system_profiler").args(["SPDisplaysDataType", "-json"]),
        )?;

        if !output.status.success() {
            return Err(GpuDetectionError::CommandFailed(
                String::from_utf8_lossy(&output.stderr).to_string(),
            ));
        }

        // For Apple Silicon, unified memory is shared
        // We'll estimate GPU portion as ~75% of total RAM
        let sysctl_output = self.run_with_timeout(
            Command::new("sysctl").args(["-n", "hw.memsize"]),
        )?;

        let total_ram = String::from_utf8_lossy(&sysctl_output.stdout)
            .trim()
            .parse::<u64>()
            .unwrap_or(16 * 1024 * 1024 * 1024);

        // Assume 75% of unified memory available for GPU
        let gpu_memory = (total_ram as f64 * 0.75) as u64;

        Ok(GpuDetectionResult {
            gpus: vec![GpuInfo {
                device_id: 0,
                vendor: GpuVendor::Apple,
                name: "Apple Silicon GPU".to_string(),
                vram_bytes: gpu_memory,
                vram_free_bytes: gpu_memory,
                compute_capability: None,
                driver_version: None,
            }],
            total_vram_bytes: gpu_memory,
            detection_method: DetectionMethod::AppleMetal,
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_info_vram_gb() {
        let info = GpuInfo {
            device_id: 0,
            vendor: GpuVendor::Nvidia,
            name: "Test GPU".to_string(),
            vram_bytes: 24 * 1024 * 1024 * 1024, // 24 GB
            vram_free_bytes: 20 * 1024 * 1024 * 1024,
            compute_capability: None,
            driver_version: None,
        };

        assert!((info.vram_gb() - 24.0).abs() < 0.01);
        assert!((info.vram_free_gb() - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_gpu_info_utilization() {
        let info = GpuInfo {
            device_id: 0,
            vendor: GpuVendor::Nvidia,
            name: "Test GPU".to_string(),
            vram_bytes: 10 * 1024 * 1024 * 1024, // 10 GB total
            vram_free_bytes: 4 * 1024 * 1024 * 1024, // 4 GB free = 6 GB used
            compute_capability: None,
            driver_version: None,
        };

        // 6/10 = 60% utilization
        assert!((info.vram_utilization() - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_gpu_info_utilization_zero_vram() {
        let info = GpuInfo {
            device_id: 0,
            vendor: GpuVendor::Unknown,
            name: "Test GPU".to_string(),
            vram_bytes: 0,
            vram_free_bytes: 0,
            compute_capability: None,
            driver_version: None,
        };

        // Should handle zero gracefully
        assert_eq!(info.vram_utilization(), 0.0);
    }

    #[test]
    fn test_detection_result_primary() {
        let result = GpuDetectionResult {
            gpus: vec![
                GpuInfo {
                    device_id: 0,
                    vendor: GpuVendor::Nvidia,
                    name: "GPU 0".to_string(),
                    vram_bytes: 24 * 1024 * 1024 * 1024,
                    vram_free_bytes: 24 * 1024 * 1024 * 1024,
                    compute_capability: Some("8.9".to_string()),
                    driver_version: None,
                },
                GpuInfo {
                    device_id: 1,
                    vendor: GpuVendor::Nvidia,
                    name: "GPU 1".to_string(),
                    vram_bytes: 24 * 1024 * 1024 * 1024,
                    vram_free_bytes: 24 * 1024 * 1024 * 1024,
                    compute_capability: Some("8.9".to_string()),
                    driver_version: None,
                },
            ],
            total_vram_bytes: 48 * 1024 * 1024 * 1024,
            detection_method: DetectionMethod::NvidiaSmi,
        };

        assert!(result.has_gpu());
        assert_eq!(result.primary().map(|g| g.device_id), Some(0));
        assert!((result.total_vram_gb() - 48.0).abs() < 0.01);
    }

    #[test]
    fn test_detection_result_none() {
        let result = GpuDetectionResult::none();

        assert!(!result.has_gpu());
        assert!(result.primary().is_none());
        assert_eq!(result.total_vram_bytes, 0);
    }

    #[test]
    fn test_detector_fallback_on_failure() {
        let detector = GpuDetector::new();
        let default_vram = 8 * 1024 * 1024 * 1024; // 8 GB

        let result = detector.detect_or_default(default_vram);

        // Should always return something, even if detection fails
        assert!(!result.gpus.is_empty());

        // If we're on a machine without a GPU, it should return the default
        if result.detection_method == DetectionMethod::None {
            assert_eq!(result.total_vram_bytes, default_vram);
        }
    }

    #[test]
    fn test_detector_nvidia_parsing() {
        // Test parsing of nvidia-smi output format
        let sample_line = "0, NVIDIA GeForce RTX 4090, 24564, 23000, 545.23.08, 8.9";
        let parts: Vec<&str> = sample_line.split(',').map(|s| s.trim()).collect();

        assert_eq!(parts[0], "0");
        assert_eq!(parts[1], "NVIDIA GeForce RTX 4090");
        assert_eq!(parts[2].parse::<u64>().ok(), Some(24564)); // MiB
        assert_eq!(parts[3].parse::<u64>().ok(), Some(23000)); // MiB free
        assert_eq!(parts[4], "545.23.08");
        assert_eq!(parts[5], "8.9");
    }

    #[test]
    fn test_gpu_vendor_equality() {
        assert_eq!(GpuVendor::Nvidia, GpuVendor::Nvidia);
        assert_ne!(GpuVendor::Nvidia, GpuVendor::Amd);
    }

    #[test]
    fn test_detector_with_timeout() {
        let detector = GpuDetector::with_timeout(Duration::from_secs(10));
        assert_eq!(detector.timeout, Duration::from_secs(10));
    }

    // Integration test that actually calls nvidia-smi (if available)
    #[test]
    fn test_nvidia_detection_real() {
        let detector = GpuDetector::new();

        // This test may pass or fail depending on whether nvidia-smi is available
        match detector.detect_nvidia() {
            Ok(result) => {
                // If NVIDIA GPUs are found, verify the data makes sense
                for gpu in &result.gpus {
                    assert_eq!(gpu.vendor, GpuVendor::Nvidia);
                    assert!(gpu.vram_bytes > 0);
                    assert!(gpu.vram_free_bytes <= gpu.vram_bytes);
                    assert!(!gpu.name.is_empty());
                }
                assert_eq!(result.detection_method, DetectionMethod::NvidiaSmi);
            }
            Err(GpuDetectionError::CommandFailed(_)) => {
                // nvidia-smi not available - acceptable in CI
            }
            Err(e) => {
                panic!("Unexpected error: {}", e);
            }
        }
    }

    #[test]
    fn test_detect_all_graceful() {
        let detector = GpuDetector::new();

        // detect() should either succeed or return NoGpu error
        match detector.detect() {
            Ok(result) => {
                assert!(result.has_gpu());
                assert!(result.total_vram_bytes > 0);
            }
            Err(GpuDetectionError::NoGpu) => {
                // Expected on machines without GPU
            }
            Err(e) => {
                // Other errors should not occur during normal operation
                panic!("Unexpected detection error: {}", e);
            }
        }
    }
}
