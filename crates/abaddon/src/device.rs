//! Device enumeration and selection for compute backends.
//!
//! This module provides utilities for discovering available compute devices
//! (GPUs, NPUs) and selecting the optimal device for inference workloads.

use infernum_core::{DeviceType, Result};
use serde::{Deserialize, Serialize};

/// Information about a compute device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Device type and index.
    pub device_type: DeviceType,
    /// Human-readable device name.
    pub name: String,
    /// Total memory in bytes.
    pub total_memory: usize,
    /// Available/free memory in bytes.
    pub available_memory: usize,
    /// Compute capability (CUDA) or feature level.
    pub compute_capability: Option<(u32, u32)>,
    /// Whether the device has tensor cores.
    pub has_tensor_cores: bool,
    /// Whether the device supports BF16.
    pub supports_bf16: bool,
    /// Recommended for inference.
    pub recommended: bool,
}

/// Enumerates all available compute devices.
///
/// Returns a list of devices sorted by preference (best first).
pub fn enumerate_devices() -> Vec<DeviceInfo> {
    let mut devices = Vec::new();

    // Always add CPU as a fallback
    devices.push(DeviceInfo {
        device_type: DeviceType::Cpu,
        name: cpu_name(),
        total_memory: system_memory(),
        available_memory: available_system_memory(),
        compute_capability: None,
        has_tensor_cores: false,
        supports_bf16: false, // CPU BF16 depends on AVX-512 BF16
        recommended: false,
    });

    // Enumerate CUDA devices
    #[cfg(feature = "cuda")]
    {
        devices.extend(enumerate_cuda_devices());
    }

    // Enumerate Metal devices
    #[cfg(feature = "metal")]
    {
        devices.extend(enumerate_metal_devices());
    }

    // Sort by preference (GPUs with tensor cores first)
    devices.sort_by(|a, b| {
        // Prefer devices with tensor cores
        let a_score = device_score(a);
        let b_score = device_score(b);
        b_score.cmp(&a_score)
    });

    // Mark the best device as recommended
    if let Some(best) = devices.first_mut() {
        best.recommended = true;
    }

    devices
}

/// Calculate a preference score for a device.
fn device_score(device: &DeviceInfo) -> u64 {
    let mut score: u64 = 0;

    // Base score by device type
    match device.device_type {
        DeviceType::Cuda { .. } => score += 1000,
        DeviceType::Metal { .. } => score += 900,
        DeviceType::WebGpu => score += 100,
        DeviceType::Cpu => score += 10,
    }

    // Bonus for tensor cores
    if device.has_tensor_cores {
        score += 500;
    }

    // Bonus for BF16 support
    if device.supports_bf16 {
        score += 200;
    }

    // Bonus for memory (per GB)
    score += (device.total_memory / (1024 * 1024 * 1024)) as u64 * 10;

    // Bonus for compute capability
    if let Some((major, minor)) = device.compute_capability {
        score += (major as u64 * 100) + (minor as u64 * 10);
    }

    score
}

/// Get the best available device for inference.
pub fn best_device() -> DeviceType {
    let devices = enumerate_devices();
    devices
        .into_iter()
        .find(|d| d.recommended)
        .map(|d| d.device_type)
        .unwrap_or(DeviceType::Cpu)
}

/// Get device information for a specific device type.
pub fn device_info(device_type: &DeviceType) -> Option<DeviceInfo> {
    enumerate_devices()
        .into_iter()
        .find(|d| &d.device_type == device_type)
}

/// Print a summary of available devices.
pub fn print_devices() {
    let devices = enumerate_devices();

    eprintln!("\x1b[1mAvailable Compute Devices:\x1b[0m");
    eprintln!();

    for (i, device) in devices.iter().enumerate() {
        let recommended = if device.recommended {
            " \x1b[32m(recommended)\x1b[0m"
        } else {
            ""
        };
        let device_type = match &device.device_type {
            DeviceType::Cpu => "CPU".to_string(),
            DeviceType::Cuda { device_id } => format!("CUDA:{}", device_id),
            DeviceType::Metal { device_id } => format!("Metal:{}", device_id),
            DeviceType::WebGpu => "WebGPU".to_string(),
        };

        eprintln!(
            "  {}. \x1b[1m{}\x1b[0m [{}]{}",
            i + 1,
            device.name,
            device_type,
            recommended
        );

        let mem_gb = device.total_memory as f64 / (1024.0 * 1024.0 * 1024.0);
        let avail_gb = device.available_memory as f64 / (1024.0 * 1024.0 * 1024.0);
        eprintln!(
            "     Memory: {:.1} GB ({:.1} GB available)",
            mem_gb, avail_gb
        );

        if let Some((major, minor)) = device.compute_capability {
            eprintln!("     Compute: {}.{}", major, minor);
        }

        let features: Vec<&str> = [
            device.has_tensor_cores.then_some("Tensor Cores"),
            device.supports_bf16.then_some("BF16"),
        ]
        .into_iter()
        .flatten()
        .collect();

        if !features.is_empty() {
            eprintln!("     Features: {}", features.join(", "));
        }

        eprintln!();
    }
}

// ============================================================================
// CPU Information
// ============================================================================

fn cpu_name() -> String {
    #[cfg(target_os = "linux")]
    {
        // Try to read from /proc/cpuinfo
        if let Ok(contents) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in contents.lines() {
                if line.starts_with("model name") {
                    if let Some(name) = line.split(':').nth(1) {
                        return name.trim().to_string();
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        // Try sysctl
        if let Ok(output) = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        {
            if let Ok(name) = String::from_utf8(output.stdout) {
                return name.trim().to_string();
            }
        }
    }

    "CPU".to_string()
}

fn system_memory() -> usize {
    use sysinfo::System;
    let sys = System::new_all();
    sys.total_memory() as usize
}

fn available_system_memory() -> usize {
    use sysinfo::System;
    let sys = System::new_all();
    sys.available_memory() as usize
}

// ============================================================================
// CUDA Device Enumeration
// ============================================================================

/// Check if running under WSL and provide helpful guidance if CUDA fails.
#[cfg(feature = "cuda")]
fn check_wsl_cuda_guidance() {
    // Check if we're running in WSL
    let is_wsl = std::fs::read_to_string("/proc/version")
        .map(|v| v.to_lowercase().contains("microsoft") || v.to_lowercase().contains("wsl"))
        .unwrap_or(false);

    if !is_wsl {
        return;
    }

    const WSL_CUDA_LIB: &str = "/usr/lib/wsl/lib/libcuda.so.1";

    // Check if WSL CUDA library exists
    if !std::path::Path::new(WSL_CUDA_LIB).exists() {
        return;
    }

    // Check if LD_LIBRARY_PATH includes the WSL CUDA path
    let ld_path = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    if !ld_path.contains("/usr/lib/wsl/lib") {
        tracing::warn!(
            "WSL CUDA detected but LD_LIBRARY_PATH not set. \
             To enable GPU support, run: LD_LIBRARY_PATH=/usr/lib/wsl/lib infernum serve"
        );
    }
}

#[cfg(feature = "cuda")]
fn enumerate_cuda_devices() -> Vec<DeviceInfo> {
    // Safely try to enumerate CUDA devices - may panic if runtime library not available
    match std::panic::catch_unwind(|| enumerate_cuda_devices_inner()) {
        Ok(devices) => {
            tracing::info!(count = devices.len(), "CUDA device enumeration succeeded");
            devices
        },
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            tracing::warn!(error = %msg, "CUDA runtime not available, falling back to CPU");
            // Provide WSL-specific guidance if applicable
            check_wsl_cuda_guidance();
            Vec::new()
        },
    }
}

#[cfg(feature = "cuda")]
fn enumerate_cuda_devices_inner() -> Vec<DeviceInfo> {
    use cudarc::driver::CudaDevice as CudarcDevice;

    let mut devices = Vec::new();

    // Initialize CUDA driver first
    if let Err(e) = cudarc::driver::result::init() {
        tracing::warn!(error = ?e, "Failed to initialize CUDA driver");
        return devices;
    }

    // Try to get device count
    let device_count = match cudarc::driver::result::device::get_count() {
        Ok(count) => {
            tracing::debug!(count, "cudarc device count");
            count as usize
        },
        Err(e) => {
            tracing::warn!(error = ?e, "Failed to get CUDA device count");
            return devices;
        },
    };

    for device_id in 0..device_count {
        if let Ok(cuda_dev) = CudarcDevice::new(device_id) {
            let compute_major = cuda_dev
                .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
                .unwrap_or(0) as u32;

            let compute_minor = cuda_dev
                .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
                .unwrap_or(0) as u32;

            // Get device name
            let name =
                cuda_device_name(device_id).unwrap_or_else(|| format!("CUDA Device {}", device_id));

            // Estimate memory based on compute capability
            let total_memory = estimate_cuda_memory(compute_major, compute_minor);
            let available_memory = (total_memory as f64 * 0.8) as usize;

            let has_tensor_cores = compute_major >= 7;
            let supports_bf16 = compute_major >= 8;

            devices.push(DeviceInfo {
                device_type: DeviceType::Cuda { device_id },
                name,
                total_memory,
                available_memory,
                compute_capability: Some((compute_major, compute_minor)),
                has_tensor_cores,
                supports_bf16,
                recommended: false,
            });
        }
    }

    devices
}

#[cfg(feature = "cuda")]
fn cuda_device_name(device_id: usize) -> Option<String> {
    // Use high-level cudarc API instead of raw sys calls
    // The CudaDevice::new() call handles device selection and provides name access
    match cudarc::driver::CudaDevice::new(device_id) {
        Ok(_device) => {
            // CudaDevice doesn't expose name directly, but we can infer from ordinal
            // For now, return a generic name - the actual GPU info is in DeviceInfo
            Some(format!("CUDA Device {}", device_id))
        },
        Err(_) => None,
    }
}

#[cfg(feature = "cuda")]
fn estimate_cuda_memory(major: u32, minor: u32) -> usize {
    // Common VRAM sizes for different compute capabilities
    match (major, minor) {
        (8, 9) => 24 * 1024 * 1024 * 1024, // RTX 4090/4500: 24GB
        (8, 6) => 12 * 1024 * 1024 * 1024, // RTX 3080: 12GB
        (8, 0) => 40 * 1024 * 1024 * 1024, // A100: 40/80GB
        (7, 5) => 8 * 1024 * 1024 * 1024,  // RTX 2070: 8GB
        (7, 0) => 16 * 1024 * 1024 * 1024, // V100: 16/32GB
        _ => 8 * 1024 * 1024 * 1024,       // Default
    }
}

// ============================================================================
// Metal Device Enumeration
// ============================================================================

#[cfg(feature = "metal")]
fn enumerate_metal_devices() -> Vec<DeviceInfo> {
    // Metal typically has one GPU on Apple Silicon
    let mut devices = Vec::new();

    // Get system info for memory estimate
    let total_memory = unified_memory_estimate();
    let available_memory = (total_memory as f64 * 0.7) as usize; // Reserve 30% for system

    devices.push(DeviceInfo {
        device_type: DeviceType::Metal { device_id: 0 },
        name: apple_gpu_name(),
        total_memory,
        available_memory,
        compute_capability: None,
        has_tensor_cores: true, // Apple Silicon has matrix accelerators
        supports_bf16: true,    // M1+ supports BF16
        recommended: false,
    });

    devices
}

#[cfg(feature = "metal")]
fn apple_gpu_name() -> String {
    // Try to detect Apple Silicon chip
    if let Ok(output) = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
    {
        if let Ok(cpu) = String::from_utf8(output.stdout) {
            let cpu = cpu.trim();
            if cpu.contains("Apple M") {
                // Extract M-series chip name
                for part in cpu.split_whitespace() {
                    if part.starts_with('M')
                        && part.chars().nth(1).map_or(false, |c| c.is_numeric())
                    {
                        return format!("Apple {} GPU", part);
                    }
                }
            }
        }
    }

    "Apple GPU".to_string()
}

#[cfg(feature = "metal")]
fn unified_memory_estimate() -> usize {
    // Apple Silicon uses unified memory
    // Return a portion of system memory as GPU-available
    use sysinfo::System;
    let sys = System::new_all();
    let total = sys.total_memory() as usize;

    // Apple Silicon can use up to ~75% of RAM for GPU
    (total as f64 * 0.75) as usize
}

// ============================================================================
// Device Selection Helpers
// ============================================================================

/// Select the optimal device based on model requirements.
pub fn select_device_for_model(
    model_size_bytes: usize,
    preferred_device: Option<DeviceType>,
) -> Result<DeviceType> {
    // If user specified a device, try to use it
    if let Some(device) = preferred_device {
        if let Some(info) = device_info(&device) {
            if info.available_memory >= model_size_bytes {
                return Ok(device);
            }
            tracing::warn!(
                "Preferred device {} has insufficient memory ({} GB < {} GB needed)",
                info.name,
                info.available_memory / (1024 * 1024 * 1024),
                model_size_bytes / (1024 * 1024 * 1024)
            );
        }
    }

    // Find the best device that can fit the model
    let devices = enumerate_devices();

    for device in devices {
        if device.available_memory >= model_size_bytes {
            tracing::info!(
                "Auto-selected {} for model ({} GB available)",
                device.name,
                device.available_memory / (1024 * 1024 * 1024)
            );
            return Ok(device.device_type);
        }
    }

    // Fall back to CPU (which can use swap)
    tracing::warn!(
        "No GPU with sufficient memory for {} GB model, falling back to CPU",
        model_size_bytes / (1024 * 1024 * 1024)
    );
    Ok(DeviceType::Cpu)
}

/// Check if CUDA is available.
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        cudarc::driver::result::device::get_count()
            .map(|c| c > 0)
            .unwrap_or(false)
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Check if Metal is available.
pub fn metal_available() -> bool {
    #[cfg(feature = "metal")]
    {
        // Metal is always available on macOS with Metal feature
        cfg!(target_os = "macos")
    }
    #[cfg(not(feature = "metal"))]
    {
        false
    }
}

/// Get the number of available CUDA devices.
pub fn cuda_device_count() -> usize {
    #[cfg(feature = "cuda")]
    {
        cudarc::driver::result::device::get_count().unwrap_or(0) as usize
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // DeviceInfo tests
    // ==========================================================================

    #[test]
    fn test_device_info_cpu() {
        let info = DeviceInfo {
            device_type: DeviceType::Cpu,
            name: "Test CPU".to_string(),
            total_memory: 32 * 1024 * 1024 * 1024,
            available_memory: 16 * 1024 * 1024 * 1024,
            compute_capability: None,
            has_tensor_cores: false,
            supports_bf16: false,
            recommended: false,
        };

        assert_eq!(info.device_type, DeviceType::Cpu);
        assert_eq!(info.name, "Test CPU");
        assert!(!info.has_tensor_cores);
    }

    #[test]
    fn test_device_info_cuda() {
        let info = DeviceInfo {
            device_type: DeviceType::Cuda { device_id: 0 },
            name: "RTX 4090".to_string(),
            total_memory: 24 * 1024 * 1024 * 1024,
            available_memory: 20 * 1024 * 1024 * 1024,
            compute_capability: Some((8, 9)),
            has_tensor_cores: true,
            supports_bf16: true,
            recommended: true,
        };

        assert!(matches!(
            info.device_type,
            DeviceType::Cuda { device_id: 0 }
        ));
        assert!(info.has_tensor_cores);
        assert!(info.supports_bf16);
        assert!(info.recommended);
    }

    #[test]
    fn test_device_info_metal() {
        let info = DeviceInfo {
            device_type: DeviceType::Metal { device_id: 0 },
            name: "Apple M3 GPU".to_string(),
            total_memory: 36 * 1024 * 1024 * 1024,
            available_memory: 24 * 1024 * 1024 * 1024,
            compute_capability: None,
            has_tensor_cores: true,
            supports_bf16: true,
            recommended: true,
        };

        assert!(matches!(info.device_type, DeviceType::Metal { .. }));
    }

    #[test]
    fn test_device_info_serialization() {
        let info = DeviceInfo {
            device_type: DeviceType::Cpu,
            name: "Test".to_string(),
            total_memory: 1000,
            available_memory: 500,
            compute_capability: None,
            has_tensor_cores: false,
            supports_bf16: false,
            recommended: false,
        };

        let json = serde_json::to_string(&info).expect("serialize");
        let parsed: DeviceInfo = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.name, "Test");
        assert_eq!(parsed.total_memory, 1000);
    }

    #[test]
    fn test_device_info_clone() {
        let info = DeviceInfo {
            device_type: DeviceType::Cpu,
            name: "Clone Test".to_string(),
            total_memory: 2000,
            available_memory: 1000,
            compute_capability: Some((8, 0)),
            has_tensor_cores: true,
            supports_bf16: true,
            recommended: true,
        };

        let cloned = info.clone();
        assert_eq!(cloned.name, info.name);
        assert_eq!(cloned.total_memory, info.total_memory);
        assert_eq!(cloned.compute_capability, info.compute_capability);
    }

    // ==========================================================================
    // Device scoring tests
    // ==========================================================================

    #[test]
    fn test_device_score_cpu() {
        let cpu = DeviceInfo {
            device_type: DeviceType::Cpu,
            name: "CPU".to_string(),
            total_memory: 32 * 1024 * 1024 * 1024, // 32GB
            available_memory: 16 * 1024 * 1024 * 1024,
            compute_capability: None,
            has_tensor_cores: false,
            supports_bf16: false,
            recommended: false,
        };

        let score = device_score(&cpu);
        // Base score 10 + (32GB * 10) = 10 + 320 = 330
        assert!(score >= 10);
        assert!(score < 500); // Should be lower than any GPU
    }

    #[test]
    fn test_device_score_cuda_prefers_tensor_cores() {
        let without_tc = DeviceInfo {
            device_type: DeviceType::Cuda { device_id: 0 },
            name: "GTX 1080".to_string(),
            total_memory: 8 * 1024 * 1024 * 1024,
            available_memory: 6 * 1024 * 1024 * 1024,
            compute_capability: Some((6, 1)),
            has_tensor_cores: false,
            supports_bf16: false,
            recommended: false,
        };

        let with_tc = DeviceInfo {
            device_type: DeviceType::Cuda { device_id: 0 },
            name: "RTX 4090".to_string(),
            total_memory: 24 * 1024 * 1024 * 1024,
            available_memory: 20 * 1024 * 1024 * 1024,
            compute_capability: Some((8, 9)),
            has_tensor_cores: true,
            supports_bf16: true,
            recommended: false,
        };

        let score_without = device_score(&without_tc);
        let score_with = device_score(&with_tc);

        assert!(score_with > score_without);
    }

    #[test]
    fn test_device_score_cuda_beats_cpu() {
        let cpu = DeviceInfo {
            device_type: DeviceType::Cpu,
            name: "CPU".to_string(),
            total_memory: 128 * 1024 * 1024 * 1024, // 128GB - lots of RAM
            available_memory: 64 * 1024 * 1024 * 1024,
            compute_capability: None,
            has_tensor_cores: false,
            supports_bf16: false,
            recommended: false,
        };

        let cuda = DeviceInfo {
            device_type: DeviceType::Cuda { device_id: 0 },
            name: "RTX 3060".to_string(),
            total_memory: 12 * 1024 * 1024 * 1024, // Only 12GB
            available_memory: 10 * 1024 * 1024 * 1024,
            compute_capability: Some((8, 6)),
            has_tensor_cores: true,
            supports_bf16: true,
            recommended: false,
        };

        let cpu_score = device_score(&cpu);
        let cuda_score = device_score(&cuda);

        // CUDA should still score higher due to base score advantage
        assert!(cuda_score > cpu_score);
    }

    // ==========================================================================
    // Device enumeration tests
    // ==========================================================================

    #[test]
    fn test_enumerate_devices_always_has_cpu() {
        let devices = enumerate_devices();

        // Should always have at least CPU
        assert!(!devices.is_empty());

        let has_cpu = devices
            .iter()
            .any(|d| matches!(d.device_type, DeviceType::Cpu));
        assert!(has_cpu, "CPU should always be available");
    }

    #[test]
    fn test_enumerate_devices_marks_recommended() {
        let devices = enumerate_devices();

        // Exactly one device should be recommended
        let recommended_count = devices.iter().filter(|d| d.recommended).count();
        assert_eq!(recommended_count, 1);
    }

    #[test]
    fn test_best_device_returns_valid() {
        let device = best_device();

        // Should return a valid device type
        match device {
            DeviceType::Cpu => (),
            DeviceType::Cuda { device_id } => assert!(device_id < 100),
            DeviceType::Metal { device_id } => assert!(device_id < 100),
            DeviceType::WebGpu => (),
        }
    }

    #[test]
    fn test_device_info_lookup() {
        let cpu_info = device_info(&DeviceType::Cpu);

        // CPU should always be found
        assert!(cpu_info.is_some());
        let info = cpu_info.unwrap();
        assert!(matches!(info.device_type, DeviceType::Cpu));
    }

    // ==========================================================================
    // Feature availability tests
    // ==========================================================================

    #[test]
    fn test_cuda_available() {
        // Just test that it doesn't panic
        let available = cuda_available();
        // On most test environments, CUDA is not available
        #[cfg(not(feature = "cuda"))]
        assert!(!available);
    }

    #[test]
    fn test_metal_available() {
        let available = metal_available();
        // Metal only on macOS with feature
        #[cfg(not(feature = "metal"))]
        assert!(!available);
    }

    #[test]
    fn test_cuda_device_count() {
        let count = cuda_device_count();
        #[cfg(not(feature = "cuda"))]
        assert_eq!(count, 0);
    }

    // ==========================================================================
    // Device selection tests
    // ==========================================================================

    #[test]
    fn test_select_device_for_small_model() {
        // Small model should fit anywhere
        let device = select_device_for_model(100_000, None).expect("select device");
        // Just verify it returns something valid
        match device {
            DeviceType::Cpu
            | DeviceType::Cuda { .. }
            | DeviceType::Metal { .. }
            | DeviceType::WebGpu => (),
        }
    }

    #[test]
    fn test_select_device_respects_preference() {
        // When preferring CPU with sufficient memory, should use CPU
        let device =
            select_device_for_model(1_000_000, Some(DeviceType::Cpu)).expect("select device");

        assert_eq!(device, DeviceType::Cpu);
    }

    #[test]
    fn test_select_device_huge_model_falls_back_to_cpu() {
        // A model too large for any GPU should fall back to CPU
        let device = select_device_for_model(
            1_000_000_000_000, // 1TB - no GPU has this much
            None,
        )
        .expect("select device");

        // Should fall back to CPU which can use swap
        assert_eq!(device, DeviceType::Cpu);
    }
}
