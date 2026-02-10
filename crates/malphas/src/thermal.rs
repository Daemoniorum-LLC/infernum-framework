//! Thermal management for sustained inference workloads.
//!
//! This module monitors system thermals and adjusts inference parameters
//! to prevent thermal throttling and ensure stable performance on workstations.
//!
//! ## Supported Platforms
//!
//! - Linux: Reads from `/sys/class/thermal` and `/sys/class/hwmon`
//! - NVIDIA GPUs: Uses nvidia-smi or NVML for GPU temperatures
//! - AMD CPUs: Reads from k10temp sensor

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Temperature thresholds in Celsius.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ThermalThresholds {
    /// Target temperature for optimal operation.
    pub target: f32,
    /// Warning threshold - start reducing load.
    pub warning: f32,
    /// Critical threshold - aggressive throttling.
    pub critical: f32,
    /// Emergency shutdown threshold.
    pub emergency: f32,
}

impl Default for ThermalThresholds {
    fn default() -> Self {
        Self {
            target: 75.0,     // Optimal operating temperature
            warning: 85.0,    // Start throttling
            critical: 95.0,   // Aggressive throttling
            emergency: 100.0, // Emergency measures
        }
    }
}

impl ThermalThresholds {
    /// Thresholds for AMD Threadripper PRO.
    pub fn threadripper_pro() -> Self {
        Self {
            target: 70.0,    // Threadripper runs hot, keep lower target
            warning: 80.0,   // Start throttling earlier
            critical: 90.0,  // Aggressive throttling
            emergency: 95.0, // Max safe temp for sustained loads
        }
    }

    /// Thresholds for NVIDIA Ada Lovelace (RTX 4000 series).
    pub fn rtx_4000() -> Self {
        Self {
            target: 70.0,
            warning: 80.0,
            critical: 87.0, // Ada throttles at ~83°C
            emergency: 90.0,
        }
    }

    /// Thresholds for datacenter/workstation (more conservative).
    pub fn workstation() -> Self {
        Self {
            target: 65.0,
            warning: 75.0,
            critical: 85.0,
            emergency: 90.0,
        }
    }
}

/// Power profile for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerProfile {
    /// Maximum performance, no thermal limits.
    Performance,
    /// Balanced performance and thermals.
    Balanced,
    /// Quiet operation, aggressive thermal management.
    Quiet,
    /// Power saver mode.
    PowerSaver,
}

impl Default for PowerProfile {
    fn default() -> Self {
        Self::Balanced
    }
}

/// Thermal state of the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalState {
    /// Normal operation.
    Normal,
    /// Elevated temperatures, monitoring closely.
    Elevated,
    /// Warning threshold reached, reducing load.
    Warning,
    /// Critical threshold reached, aggressive throttling.
    Critical,
    /// Emergency - immediate action required.
    Emergency,
}

/// Temperature reading from a sensor.
#[derive(Debug, Clone)]
pub struct TemperatureReading {
    /// Sensor name.
    pub name: String,
    /// Current temperature in Celsius.
    pub temperature: f32,
    /// Maximum observed temperature.
    pub max_observed: f32,
    /// Timestamp of reading.
    pub timestamp: Instant,
}

/// Thermal manager for inference workloads.
pub struct ThermalManager {
    /// CPU thermal thresholds.
    cpu_thresholds: ThermalThresholds,
    /// GPU thermal thresholds.
    gpu_thresholds: ThermalThresholds,
    /// Current power profile.
    profile: RwLock<PowerProfile>,
    /// Current thermal state.
    state: RwLock<ThermalState>,
    /// Latest temperature readings.
    temperatures: RwLock<HashMap<String, TemperatureReading>>,
    /// Whether monitoring is active.
    monitoring_active: AtomicBool,
    /// Throttle factor (0.0-1.0, 1.0 = full speed).
    throttle_factor: RwLock<f32>,
}

impl ThermalManager {
    /// Create a new thermal manager with default thresholds.
    pub fn new() -> Self {
        Self {
            cpu_thresholds: ThermalThresholds::default(),
            gpu_thresholds: ThermalThresholds::default(),
            profile: RwLock::new(PowerProfile::Balanced),
            state: RwLock::new(ThermalState::Normal),
            temperatures: RwLock::new(HashMap::new()),
            monitoring_active: AtomicBool::new(false),
            throttle_factor: RwLock::new(1.0),
        }
    }

    /// Create a thermal manager configured for workstation use.
    pub fn workstation() -> Self {
        Self {
            cpu_thresholds: ThermalThresholds::threadripper_pro(),
            gpu_thresholds: ThermalThresholds::rtx_4000(),
            profile: RwLock::new(PowerProfile::Balanced),
            state: RwLock::new(ThermalState::Normal),
            temperatures: RwLock::new(HashMap::new()),
            monitoring_active: AtomicBool::new(false),
            throttle_factor: RwLock::new(1.0),
        }
    }

    /// Set the power profile.
    pub async fn set_profile(&self, profile: PowerProfile) {
        *self.profile.write().await = profile;
        self.recalculate_throttle().await;
    }

    /// Get the current power profile.
    pub async fn profile(&self) -> PowerProfile {
        *self.profile.read().await
    }

    /// Get the current thermal state.
    pub async fn state(&self) -> ThermalState {
        *self.state.read().await
    }

    /// Get the current throttle factor (0.0-1.0).
    pub async fn throttle_factor(&self) -> f32 {
        *self.throttle_factor.read().await
    }

    /// Get recommended batch size based on thermals.
    pub async fn recommended_batch_size(&self, max_batch: u32) -> u32 {
        let throttle = self.throttle_factor().await;
        let profile = self.profile().await;

        let profile_factor = match profile {
            PowerProfile::Performance => 1.0,
            PowerProfile::Balanced => 0.9,
            PowerProfile::Quiet => 0.7,
            PowerProfile::PowerSaver => 0.5,
        };

        let effective = throttle * profile_factor;
        ((max_batch as f32 * effective).ceil() as u32).max(1)
    }

    /// Update temperature readings.
    pub async fn update_temperatures(&self) {
        let mut temps = self.temperatures.write().await;

        // Read CPU temperature
        if let Some(cpu_temp) = read_cpu_temperature() {
            let entry = temps
                .entry("cpu".to_string())
                .or_insert(TemperatureReading {
                    name: "CPU".to_string(),
                    temperature: cpu_temp,
                    max_observed: cpu_temp,
                    timestamp: Instant::now(),
                });
            entry.temperature = cpu_temp;
            entry.max_observed = entry.max_observed.max(cpu_temp);
            entry.timestamp = Instant::now();
        }

        // Read GPU temperature
        if let Some(gpu_temp) = read_gpu_temperature() {
            let entry = temps
                .entry("gpu".to_string())
                .or_insert(TemperatureReading {
                    name: "GPU".to_string(),
                    temperature: gpu_temp,
                    max_observed: gpu_temp,
                    timestamp: Instant::now(),
                });
            entry.temperature = gpu_temp;
            entry.max_observed = entry.max_observed.max(gpu_temp);
            entry.timestamp = Instant::now();
        }

        drop(temps);
        self.recalculate_throttle().await;
    }

    /// Recalculate throttle factor based on temperatures.
    async fn recalculate_throttle(&self) {
        let temps = self.temperatures.read().await;

        let cpu_temp = temps.get("cpu").map(|t| t.temperature).unwrap_or(0.0);
        let gpu_temp = temps.get("gpu").map(|t| t.temperature).unwrap_or(0.0);

        // Calculate throttle based on highest temperature
        let cpu_throttle = self.calculate_throttle(cpu_temp, &self.cpu_thresholds);
        let gpu_throttle = self.calculate_throttle(gpu_temp, &self.gpu_thresholds);

        let throttle = cpu_throttle.min(gpu_throttle);

        // Determine thermal state
        let max_temp = cpu_temp.max(gpu_temp);
        let thresholds = if cpu_temp > gpu_temp {
            &self.cpu_thresholds
        } else {
            &self.gpu_thresholds
        };

        let state = if max_temp >= thresholds.emergency {
            ThermalState::Emergency
        } else if max_temp >= thresholds.critical {
            ThermalState::Critical
        } else if max_temp >= thresholds.warning {
            ThermalState::Warning
        } else if max_temp >= thresholds.target {
            ThermalState::Elevated
        } else {
            ThermalState::Normal
        };

        *self.throttle_factor.write().await = throttle;
        *self.state.write().await = state;

        // Log significant state changes
        if state == ThermalState::Warning || state == ThermalState::Critical {
            tracing::warn!(
                cpu_temp = cpu_temp,
                gpu_temp = gpu_temp,
                throttle = throttle,
                ?state,
                "Thermal throttling active"
            );
        }
    }

    /// Calculate throttle factor for a temperature and thresholds.
    fn calculate_throttle(&self, temp: f32, thresholds: &ThermalThresholds) -> f32 {
        if temp <= thresholds.target {
            1.0
        } else if temp <= thresholds.warning {
            // Linear reduction from 1.0 to 0.8
            let range = thresholds.warning - thresholds.target;
            let delta = temp - thresholds.target;
            1.0 - (delta / range) * 0.2
        } else if temp <= thresholds.critical {
            // Linear reduction from 0.8 to 0.4
            let range = thresholds.critical - thresholds.warning;
            let delta = temp - thresholds.warning;
            0.8 - (delta / range) * 0.4
        } else if temp <= thresholds.emergency {
            // Linear reduction from 0.4 to 0.1
            let range = thresholds.emergency - thresholds.critical;
            let delta = temp - thresholds.critical;
            0.4 - (delta / range) * 0.3
        } else {
            // Emergency: minimal operation
            0.1
        }
    }

    /// Start background thermal monitoring.
    pub fn start_monitoring(self: Arc<Self>, interval: Duration) {
        if self.monitoring_active.swap(true, Ordering::SeqCst) {
            return; // Already monitoring
        }

        let manager = self.clone();
        tokio::spawn(async move {
            while manager.monitoring_active.load(Ordering::SeqCst) {
                manager.update_temperatures().await;
                tokio::time::sleep(interval).await;
            }
        });
    }

    /// Stop background thermal monitoring.
    pub fn stop_monitoring(&self) {
        self.monitoring_active.store(false, Ordering::SeqCst);
    }

    /// Get all current temperature readings.
    pub async fn temperatures(&self) -> Vec<TemperatureReading> {
        self.temperatures.read().await.values().cloned().collect()
    }

    /// Print thermal status.
    pub async fn print_status(&self) {
        let temps = self.temperatures.read().await;
        let state = self.state().await;
        let throttle = self.throttle_factor().await;
        let profile = self.profile().await;

        eprintln!("\x1b[1mThermal Status:\x1b[0m");
        eprintln!("  Profile: {:?}", profile);
        eprintln!("  State: {:?}", state);
        eprintln!("  Throttle: {:.0}%", throttle * 100.0);
        eprintln!();

        for (name, reading) in temps.iter() {
            let color = if reading.temperature >= self.cpu_thresholds.critical {
                "\x1b[31m" // Red
            } else if reading.temperature >= self.cpu_thresholds.warning {
                "\x1b[33m" // Yellow
            } else {
                "\x1b[32m" // Green
            };
            eprintln!(
                "  {}: {}{:.1}°C\x1b[0m (max: {:.1}°C)",
                name.to_uppercase(),
                color,
                reading.temperature,
                reading.max_observed
            );
        }
    }
}

impl Default for ThermalManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Platform-specific temperature reading
// ============================================================================

/// Read CPU temperature (platform-specific).
fn read_cpu_temperature() -> Option<f32> {
    #[cfg(target_os = "linux")]
    {
        read_cpu_temperature_linux()
    }

    #[cfg(target_os = "macos")]
    {
        read_cpu_temperature_macos()
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
fn read_cpu_temperature_linux() -> Option<f32> {
    // Try k10temp for AMD CPUs (Threadripper)
    let k10temp_paths = [
        "/sys/class/hwmon/hwmon0/temp1_input",
        "/sys/class/hwmon/hwmon1/temp1_input",
        "/sys/class/hwmon/hwmon2/temp1_input",
    ];

    for path in k10temp_paths {
        if let Ok(contents) = std::fs::read_to_string(path) {
            if let Ok(millidegrees) = contents.trim().parse::<i64>() {
                return Some(millidegrees as f32 / 1000.0);
            }
        }
    }

    // Try coretemp for Intel CPUs
    for entry in std::fs::read_dir("/sys/class/hwmon").ok()? {
        let entry = entry.ok()?;
        let name_path = entry.path().join("name");
        if let Ok(name) = std::fs::read_to_string(&name_path) {
            if name.trim() == "k10temp" || name.trim() == "coretemp" {
                let temp_path = entry.path().join("temp1_input");
                if let Ok(contents) = std::fs::read_to_string(&temp_path) {
                    if let Ok(millidegrees) = contents.trim().parse::<i64>() {
                        return Some(millidegrees as f32 / 1000.0);
                    }
                }
            }
        }
    }

    // Fallback: thermal zone
    if let Ok(contents) = std::fs::read_to_string("/sys/class/thermal/thermal_zone0/temp") {
        if let Ok(millidegrees) = contents.trim().parse::<i64>() {
            return Some(millidegrees as f32 / 1000.0);
        }
    }

    None
}

#[cfg(target_os = "macos")]
fn read_cpu_temperature_macos() -> Option<f32> {
    // macOS doesn't expose CPU temp directly
    // Would need IOKit or a tool like osx-cpu-temp
    None
}

/// Read GPU temperature.
fn read_gpu_temperature() -> Option<f32> {
    #[cfg(target_os = "linux")]
    {
        read_nvidia_gpu_temperature()
    }

    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
fn read_nvidia_gpu_temperature() -> Option<f32> {
    // Try nvidia-smi
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if output.status.success() {
        let temp_str = String::from_utf8(output.stdout).ok()?;
        // Take first GPU's temperature
        let first_line = temp_str.lines().next()?;
        return first_line.trim().parse().ok();
    }

    // Fallback: try hwmon for nvidia (nouveau or nvidia driver)
    for entry in std::fs::read_dir("/sys/class/hwmon").ok()? {
        let entry = entry.ok()?;
        let name_path = entry.path().join("name");
        if let Ok(name) = std::fs::read_to_string(&name_path) {
            if name.trim().contains("nvidia") || name.trim() == "nouveau" {
                let temp_path = entry.path().join("temp1_input");
                if let Ok(contents) = std::fs::read_to_string(&temp_path) {
                    if let Ok(millidegrees) = contents.trim().parse::<i64>() {
                        return Some(millidegrees as f32 / 1000.0);
                    }
                }
            }
        }
    }

    None
}

// ============================================================================
// Power limit management (NVIDIA specific)
// ============================================================================

/// GPU power limit configuration.
#[derive(Debug, Clone, Copy)]
pub struct GpuPowerLimit {
    /// Current power limit in watts.
    pub current: Option<u32>,
    /// Minimum allowed power limit.
    pub min: Option<u32>,
    /// Maximum allowed power limit.
    pub max: Option<u32>,
    /// Default power limit.
    pub default: Option<u32>,
}

/// Get current GPU power limits.
pub fn get_gpu_power_limits() -> Option<GpuPowerLimit> {
    #[cfg(target_os = "linux")]
    {
        get_nvidia_power_limits()
    }

    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
fn get_nvidia_power_limits() -> Option<GpuPowerLimit> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=power.limit,power.min_limit,power.max_limit,power.default_limit",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if output.status.success() {
        let line = String::from_utf8(output.stdout).ok()?;
        let values: Vec<&str> = line.trim().split(',').map(|s| s.trim()).collect();

        if values.len() >= 4 {
            return Some(GpuPowerLimit {
                current: values[0].parse().ok(),
                min: values[1].parse().ok(),
                max: values[2].parse().ok(),
                default: values[3].parse().ok(),
            });
        }
    }

    None
}

/// Set GPU power limit (requires root/admin privileges).
pub fn set_gpu_power_limit(watts: u32) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    {
        let output = std::process::Command::new("nvidia-smi")
            .args(["-pl", &watts.to_string()])
            .output()
            .map_err(|e| format!("Failed to run nvidia-smi: {}", e))?;

        if output.status.success() {
            Ok(())
        } else {
            Err(String::from_utf8_lossy(&output.stderr).to_string())
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        Err("Power limit control not supported on this platform".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === ThermalThresholds Tests ===

    #[test]
    fn test_thermal_thresholds_default() {
        let thresholds = ThermalThresholds::default();
        assert_eq!(thresholds.target, 75.0);
        assert_eq!(thresholds.warning, 85.0);
        assert_eq!(thresholds.critical, 95.0);
        assert_eq!(thresholds.emergency, 100.0);

        // Ensure thresholds are in ascending order
        assert!(thresholds.target < thresholds.warning);
        assert!(thresholds.warning < thresholds.critical);
        assert!(thresholds.critical < thresholds.emergency);
    }

    #[test]
    fn test_thermal_thresholds_threadripper_pro() {
        let thresholds = ThermalThresholds::threadripper_pro();
        assert_eq!(thresholds.target, 70.0);
        assert_eq!(thresholds.warning, 80.0);
        assert_eq!(thresholds.critical, 90.0);
        assert_eq!(thresholds.emergency, 95.0);

        // More conservative than default
        assert!(thresholds.target <= ThermalThresholds::default().target);
    }

    #[test]
    fn test_thermal_thresholds_rtx_4000() {
        let thresholds = ThermalThresholds::rtx_4000();
        assert_eq!(thresholds.target, 70.0);
        assert_eq!(thresholds.warning, 80.0);
        assert_eq!(thresholds.critical, 87.0);
        assert_eq!(thresholds.emergency, 90.0);
    }

    #[test]
    fn test_thermal_thresholds_workstation() {
        let thresholds = ThermalThresholds::workstation();
        assert_eq!(thresholds.target, 65.0);
        assert_eq!(thresholds.warning, 75.0);
        assert_eq!(thresholds.critical, 85.0);
        assert_eq!(thresholds.emergency, 90.0);

        // Most conservative
        assert!(thresholds.target <= ThermalThresholds::threadripper_pro().target);
    }

    // === PowerProfile Tests ===

    #[test]
    fn test_power_profile_default() {
        let profile = PowerProfile::default();
        assert_eq!(profile, PowerProfile::Balanced);
    }

    // === ThermalManager Tests ===

    #[tokio::test]
    async fn test_thermal_manager_default() {
        let manager = ThermalManager::default();
        assert_eq!(manager.profile().await, PowerProfile::Balanced);
        assert_eq!(manager.state().await, ThermalState::Normal);
        assert_eq!(manager.throttle_factor().await, 1.0);
    }

    #[tokio::test]
    async fn test_thermal_manager_workstation() {
        let manager = ThermalManager::workstation();
        // Should have workstation-tuned thresholds
        assert_eq!(manager.profile().await, PowerProfile::Balanced);
        assert_eq!(manager.state().await, ThermalState::Normal);
    }

    #[tokio::test]
    async fn test_set_profile() {
        let manager = ThermalManager::new();

        manager.set_profile(PowerProfile::Performance).await;
        assert_eq!(manager.profile().await, PowerProfile::Performance);

        manager.set_profile(PowerProfile::Quiet).await;
        assert_eq!(manager.profile().await, PowerProfile::Quiet);

        manager.set_profile(PowerProfile::PowerSaver).await;
        assert_eq!(manager.profile().await, PowerProfile::PowerSaver);
    }

    #[tokio::test]
    async fn test_recommended_batch_size_performance() {
        let manager = ThermalManager::new();
        manager.set_profile(PowerProfile::Performance).await;

        // At full throttle, performance profile should give max batch
        let batch = manager.recommended_batch_size(100).await;
        assert_eq!(batch, 100);
    }

    #[tokio::test]
    async fn test_recommended_batch_size_balanced() {
        let manager = ThermalManager::new();
        manager.set_profile(PowerProfile::Balanced).await;

        // Balanced profile reduces by 10%
        let batch = manager.recommended_batch_size(100).await;
        assert_eq!(batch, 90);
    }

    #[tokio::test]
    async fn test_recommended_batch_size_quiet() {
        let manager = ThermalManager::new();
        manager.set_profile(PowerProfile::Quiet).await;

        // Quiet profile reduces by 30%
        let batch = manager.recommended_batch_size(100).await;
        assert_eq!(batch, 70);
    }

    #[tokio::test]
    async fn test_recommended_batch_size_power_saver() {
        let manager = ThermalManager::new();
        manager.set_profile(PowerProfile::PowerSaver).await;

        // Power saver reduces by 50%
        let batch = manager.recommended_batch_size(100).await;
        assert_eq!(batch, 50);
    }

    #[tokio::test]
    async fn test_recommended_batch_size_minimum() {
        let manager = ThermalManager::new();
        manager.set_profile(PowerProfile::PowerSaver).await;

        // Should never return 0
        let batch = manager.recommended_batch_size(1).await;
        assert!(batch >= 1);
    }

    // === Throttle Calculation Tests ===

    #[test]
    fn test_calculate_throttle_below_target() {
        let manager = ThermalManager::new();
        let thresholds = ThermalThresholds::default();

        // Below target: full throttle
        let throttle = manager.calculate_throttle(50.0, &thresholds);
        assert_eq!(throttle, 1.0);

        let throttle = manager.calculate_throttle(thresholds.target, &thresholds);
        assert_eq!(throttle, 1.0);
    }

    #[test]
    fn test_calculate_throttle_at_warning() {
        let manager = ThermalManager::new();
        let thresholds = ThermalThresholds::default();

        // At warning threshold: 0.8 throttle
        let throttle = manager.calculate_throttle(thresholds.warning, &thresholds);
        assert!((throttle - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_calculate_throttle_at_critical() {
        let manager = ThermalManager::new();
        let thresholds = ThermalThresholds::default();

        // At critical threshold: 0.4 throttle
        let throttle = manager.calculate_throttle(thresholds.critical, &thresholds);
        assert!((throttle - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_calculate_throttle_at_emergency() {
        let manager = ThermalManager::new();
        let thresholds = ThermalThresholds::default();

        // At emergency threshold: 0.1 throttle
        let throttle = manager.calculate_throttle(thresholds.emergency, &thresholds);
        assert!((throttle - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_calculate_throttle_above_emergency() {
        let manager = ThermalManager::new();
        let thresholds = ThermalThresholds::default();

        // Above emergency: minimal throttle (0.1)
        let throttle = manager.calculate_throttle(thresholds.emergency + 10.0, &thresholds);
        assert_eq!(throttle, 0.1);
    }

    #[test]
    fn test_calculate_throttle_linear_between_target_and_warning() {
        let manager = ThermalManager::new();
        let thresholds = ThermalThresholds::default();

        // Midpoint between target and warning
        let midpoint = (thresholds.target + thresholds.warning) / 2.0;
        let throttle = manager.calculate_throttle(midpoint, &thresholds);

        // Should be midpoint between 1.0 and 0.8 = 0.9
        assert!((throttle - 0.9).abs() < 0.01);
    }

    // === Monitoring Tests ===

    #[test]
    fn test_stop_monitoring() {
        let manager = ThermalManager::new();

        // Ensure monitoring can be stopped
        manager.stop_monitoring();
        assert!(!manager.monitoring_active.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_temperatures_initially_empty() {
        let manager = ThermalManager::new();
        let temps = manager.temperatures().await;
        assert!(temps.is_empty());
    }

    // === GpuPowerLimit Tests ===

    #[test]
    fn test_gpu_power_limit_struct() {
        let limit = GpuPowerLimit {
            current: Some(300),
            min: Some(100),
            max: Some(450),
            default: Some(350),
        };

        assert_eq!(limit.current, Some(300));
        assert_eq!(limit.min, Some(100));
        assert_eq!(limit.max, Some(450));
        assert_eq!(limit.default, Some(350));
    }

    #[test]
    fn test_gpu_power_limit_optional_fields() {
        let limit = GpuPowerLimit {
            current: None,
            min: None,
            max: None,
            default: None,
        };

        assert!(limit.current.is_none());
        assert!(limit.min.is_none());
        assert!(limit.max.is_none());
        assert!(limit.default.is_none());
    }

    // === ThermalState Tests ===

    #[test]
    fn test_thermal_state_equality() {
        assert_eq!(ThermalState::Normal, ThermalState::Normal);
        assert_ne!(ThermalState::Normal, ThermalState::Warning);
        assert_ne!(ThermalState::Warning, ThermalState::Critical);
        assert_ne!(ThermalState::Critical, ThermalState::Emergency);
    }

    // === TemperatureReading Tests ===

    #[test]
    fn test_temperature_reading_creation() {
        let reading = TemperatureReading {
            name: "CPU".to_string(),
            temperature: 65.5,
            max_observed: 70.0,
            timestamp: Instant::now(),
        };

        assert_eq!(reading.name, "CPU");
        assert_eq!(reading.temperature, 65.5);
        assert_eq!(reading.max_observed, 70.0);
    }
}
