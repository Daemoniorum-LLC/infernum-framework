//! Legion Field - The collective memory substrate.
//!
//! The field is where agent contributions superimpose and interfere,
//! creating the emergent consensus of the collective.
//!
//! Based on the holographic principle: every region of the field
//! contains information about the whole task.

use parking_lot::RwLock;

use crate::quality::{FrequencyBand, QualityCurve};

/// Configuration for the Legion field.
#[derive(Debug, Clone)]
pub struct FieldConfig {
    /// Width of spectral space.
    pub width: usize,
    /// Height of spectral space.
    pub height: usize,
    /// Decay rate per step (0.95 = 5% decay).
    pub decay_rate: f32,
    /// Minimum energy floor (prevents total fade).
    pub energy_floor: f32,
    /// Quality curve for field reconstruction.
    pub quality_curve: QualityCurve,
}

impl Default for FieldConfig {
    fn default() -> Self {
        Self {
            width: 64,
            height: 64,
            decay_rate: 0.95,   // 5% decay per step
            energy_floor: 1e-6, // Never fully forgotten
            quality_curve: QualityCurve::SPECTRAL,
        }
    }
}

/// The Legion Field - collective memory substrate.
///
/// Patterns from agents superimpose here. Through interference,
/// areas of agreement amplify (constructive) while conflicts
/// cancel (destructive).
pub struct LegionField {
    /// Spectral coefficients in frequency domain (DCT-transformed).
    coefficients: RwLock<Vec<f32>>,
    /// Field dimensions.
    width: usize,
    height: usize,
    /// Current total energy.
    energy: std::sync::atomic::AtomicU64, // f32 as bits
    /// Configuration.
    config: FieldConfig,
    /// Step counter for decay.
    step: std::sync::atomic::AtomicU64,
}

impl LegionField {
    /// Creates a new empty field.
    pub fn new(config: FieldConfig) -> Self {
        let size = config.width * config.height;
        Self {
            coefficients: RwLock::new(vec![0.0; size]),
            width: config.width,
            height: config.height,
            energy: std::sync::atomic::AtomicU64::new(0.0f32.to_bits() as u64),
            config,
            step: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Returns field dimensions.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Returns current total energy.
    pub fn energy(&self) -> f32 {
        f32::from_bits(self.energy.load(std::sync::atomic::Ordering::Relaxed) as u32)
    }

    /// Returns current step count.
    pub fn step(&self) -> u64 {
        self.step.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Superimposes a pattern onto the field (⊕ operator).
    ///
    /// The pattern's coefficients are added to the field, weighted
    /// by the agent's emphasis (based on frequency band).
    pub fn superimpose(&self, pattern: &LegionPattern, band: FrequencyBand) {
        let weight = band.emphasis();
        let mut coeffs = self.coefficients.write();

        // Add pattern coefficients, weighted by band emphasis
        for (i, &val) in pattern.coefficients.iter().enumerate() {
            if i < coeffs.len() {
                coeffs[i] += val * weight;
            }
        }

        // Update energy
        let new_energy: f32 = coeffs.iter().map(|c| c * c).sum::<f32>().sqrt();
        self.energy.store(
            new_energy.to_bits() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// Interferes a probe pattern with the field (⫰ operator).
    ///
    /// Returns a resonance map showing where agreement (positive)
    /// and conflict (negative) occur.
    pub fn interfere(&self, probe: &LegionPattern) -> Resonance {
        let coeffs = self.coefficients.read();
        let mut resonance_values = Vec::with_capacity(coeffs.len());

        for (i, &field_val) in coeffs.iter().enumerate() {
            let probe_val = probe.coefficients.get(i).copied().unwrap_or(0.0);
            // Correlation = field * probe
            // Positive = agreement, Negative = conflict
            resonance_values.push(field_val * probe_val);
        }

        // Find peaks and conflicts
        let mut peaks = Vec::new();
        let mut conflicts = Vec::new();

        for (i, &val) in resonance_values.iter().enumerate() {
            if val > 0.5 {
                peaks.push(ResonancePeak {
                    index: i,
                    strength: val,
                });
            } else if val < -0.3 {
                conflicts.push(ResonanceConflict {
                    index: i,
                    strength: val.abs(),
                });
            }
        }

        Resonance {
            values: resonance_values,
            peaks,
            conflicts,
        }
    }

    /// Applies decay to the field (∂ operator).
    ///
    /// Older contributions fade, but never fully disappear.
    pub fn decay(&self) {
        let mut coeffs = self.coefficients.write();

        for coeff in coeffs.iter_mut() {
            *coeff = (*coeff * self.config.decay_rate).max(if *coeff > 0.0 {
                self.config.energy_floor
            } else if *coeff < 0.0 {
                -self.config.energy_floor
            } else {
                0.0
            });
        }

        // Update energy
        let new_energy: f32 = coeffs.iter().map(|c| c * c).sum::<f32>().sqrt();
        self.energy.store(
            new_energy.to_bits() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        self.step.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Clears the field.
    pub fn clear(&self) {
        let mut coeffs = self.coefficients.write();
        coeffs.fill(0.0);
        self.energy.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Extracts a pattern from the field at the current state.
    pub fn extract(&self) -> LegionPattern {
        let coeffs = self.coefficients.read();
        LegionPattern {
            coefficients: coeffs.clone(),
            width: self.width,
            height: self.height,
        }
    }

    /// Returns the coefficient at a specific index.
    pub fn coefficient(&self, index: usize) -> f32 {
        let coeffs = self.coefficients.read();
        coeffs.get(index).copied().unwrap_or(0.0)
    }

    /// Returns quality based on energy level.
    pub fn quality(&self) -> f32 {
        // Higher energy = more agreement = better quality
        let energy = self.energy();
        // Normalize assuming max energy is ~10
        (energy / 10.0).clamp(0.0, 1.0)
    }
}

impl std::fmt::Debug for LegionField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LegionField")
            .field("dimensions", &(self.width, self.height))
            .field("energy", &self.energy())
            .field("step", &self.step())
            .finish()
    }
}

// ==================== Legion Pattern ====================

/// A pattern that can join the Legion field.
///
/// Contains spectral coefficients representing information
/// in the frequency domain.
#[derive(Debug, Clone)]
pub struct LegionPattern {
    /// Spectral coefficients.
    pub coefficients: Vec<f32>,
    /// Pattern width.
    pub width: usize,
    /// Pattern height.
    pub height: usize,
}

impl LegionPattern {
    /// Creates a new pattern with the given dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            coefficients: vec![0.0; width * height],
            width,
            height,
        }
    }

    /// Creates a pattern from raw coefficients.
    pub fn from_coefficients(coefficients: Vec<f32>, width: usize, height: usize) -> Self {
        Self {
            coefficients,
            width,
            height,
        }
    }

    /// Returns the DC component (average/core identity).
    pub fn dc_component(&self) -> f32 {
        self.coefficients.first().copied().unwrap_or(0.0)
    }

    /// Returns the essential coefficients (DC + low frequency).
    ///
    /// These are the "loud voices" that carry most of the information.
    /// ~15% of coefficients, but ~60% of the reconstruction quality.
    pub fn essential(&self) -> EssentialCoefficients {
        let dc = self.dc_component();
        let low_freq_count = (self.coefficients.len() as f32 * 0.15) as usize;
        let low_freq: Vec<f32> = self
            .coefficients
            .iter()
            .skip(1)
            .take(low_freq_count)
            .copied()
            .collect();

        EssentialCoefficients { dc, low_freq }
    }

    /// Returns the detail coefficients (mid/high frequency).
    ///
    /// These are distributed across fragments for progressive loading.
    pub fn detail(&self, start_index: usize) -> DetailCoefficients {
        let essential_count = 1 + (self.coefficients.len() as f32 * 0.15) as usize;
        let detail: Vec<f32> = self
            .coefficients
            .iter()
            .skip(essential_count)
            .copied()
            .collect();

        DetailCoefficients {
            coefficients: detail,
            start_index: essential_count + start_index,
        }
    }

    /// Reconstructs pattern from essential + detail coefficients.
    pub fn reconstruct(
        essential: &EssentialCoefficients,
        detail: Option<&DetailCoefficients>,
        width: usize,
        height: usize,
    ) -> Self {
        let mut coefficients = vec![0.0; width * height];

        // DC component
        if !coefficients.is_empty() {
            coefficients[0] = essential.dc;
        }

        // Low frequency
        for (i, &val) in essential.low_freq.iter().enumerate() {
            if i + 1 < coefficients.len() {
                coefficients[i + 1] = val;
            }
        }

        // Detail (if available)
        if let Some(d) = detail {
            for (i, &val) in d.coefficients.iter().enumerate() {
                let idx = d.start_index + i;
                if idx < coefficients.len() {
                    coefficients[idx] = val;
                }
            }
        }

        Self {
            coefficients,
            width,
            height,
        }
    }

    /// Computes similarity with another pattern (0.0 - 1.0).
    pub fn similarity(&self, other: &LegionPattern) -> f32 {
        if self.coefficients.len() != other.coefficients.len() {
            return 0.0;
        }

        let dot: f32 = self
            .coefficients
            .iter()
            .zip(other.coefficients.iter())
            .map(|(a, b)| a * b)
            .sum();

        let mag_self: f32 = self.coefficients.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_other: f32 = other.coefficients.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_self == 0.0 || mag_other == 0.0 {
            return 0.0;
        }

        (dot / (mag_self * mag_other)).clamp(-1.0, 1.0)
    }
}

// ==================== Essential/Detail Coefficients ====================

/// Essential coefficients: DC + low frequency.
///
/// These carry ~60% of the information with ~15% of the data.
#[derive(Debug, Clone)]
pub struct EssentialCoefficients {
    /// DC component (core identity, replicated everywhere).
    pub dc: f32,
    /// Low frequency coefficients (the "loud voices").
    pub low_freq: Vec<f32>,
}

/// Detail coefficients: mid/high frequency.
///
/// Distributed across fragments for progressive loading.
#[derive(Debug, Clone)]
pub struct DetailCoefficients {
    /// The coefficient values.
    pub coefficients: Vec<f32>,
    /// Starting index in the spectrum.
    pub start_index: usize,
}

// ==================== Resonance ====================

/// Result of interfering a probe with the field.
#[derive(Debug, Clone)]
pub struct Resonance {
    /// Raw resonance values at each position.
    pub values: Vec<f32>,
    /// Peaks where agreement is strong.
    pub peaks: Vec<ResonancePeak>,
    /// Conflicts where disagreement is strong.
    pub conflicts: Vec<ResonanceConflict>,
}

impl Resonance {
    /// Returns total agreement strength.
    pub fn agreement_strength(&self) -> f32 {
        self.peaks.iter().map(|p| p.strength).sum()
    }

    /// Returns total conflict strength.
    pub fn conflict_strength(&self) -> f32 {
        self.conflicts.iter().map(|c| c.strength).sum()
    }

    /// Returns overall confidence based on agreement vs conflict.
    pub fn confidence(&self) -> f32 {
        let agree = self.agreement_strength();
        let conflict = self.conflict_strength();

        if agree + conflict == 0.0 {
            return 0.0;
        }

        agree / (agree + conflict)
    }

    /// Returns true if strong consensus (>80% agreement).
    pub fn is_strong(&self) -> bool {
        self.confidence() >= 0.8
    }
}

/// A peak in the resonance (area of agreement).
#[derive(Debug, Clone, Copy)]
pub struct ResonancePeak {
    /// Index in the spectrum.
    pub index: usize,
    /// Strength of agreement (0.0 - 1.0+).
    pub strength: f32,
}

/// A conflict in the resonance (area of disagreement).
#[derive(Debug, Clone, Copy)]
pub struct ResonanceConflict {
    /// Index in the spectrum.
    pub index: usize,
    /// Strength of conflict (0.0 - 1.0+).
    pub strength: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_creation() {
        let field = LegionField::new(FieldConfig::default());
        assert_eq!(field.dimensions(), (64, 64));
        assert!((field.energy() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_superimpose() {
        let field = LegionField::new(FieldConfig::default());
        let mut pattern = LegionPattern::new(64, 64);
        pattern.coefficients[0] = 1.0;
        pattern.coefficients[1] = 0.5;

        field.superimpose(&pattern, FrequencyBand::Operational);

        assert!(field.energy() > 0.0);
        assert!((field.coefficient(0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_interference() {
        let field = LegionField::new(FieldConfig::default());

        // Add pattern to field
        let mut pattern1 = LegionPattern::new(64, 64);
        pattern1.coefficients[0] = 1.0;
        pattern1.coefficients[1] = 1.0;
        field.superimpose(&pattern1, FrequencyBand::Operational);

        // Interfere with similar probe
        let mut probe = LegionPattern::new(64, 64);
        probe.coefficients[0] = 1.0;
        probe.coefficients[1] = 1.0;

        let resonance = field.interfere(&probe);

        // Should have agreement
        assert!(resonance.agreement_strength() > 0.0);
    }

    #[test]
    fn test_decay() {
        let field = LegionField::new(FieldConfig::default());
        let mut pattern = LegionPattern::new(64, 64);
        pattern.coefficients[0] = 1.0;

        field.superimpose(&pattern, FrequencyBand::Operational);
        let energy_before = field.energy();

        field.decay();
        let energy_after = field.energy();

        assert!(energy_after < energy_before);
        assert!(energy_after > 0.0); // Not fully gone
    }

    #[test]
    fn test_essential_coefficients() {
        let mut pattern = LegionPattern::new(64, 64);
        pattern.coefficients[0] = 1.0; // DC
        pattern.coefficients[1] = 0.8; // Low freq
        pattern.coefficients[2] = 0.6;

        let essential = pattern.essential();
        assert!((essential.dc - 1.0).abs() < 0.001);
        assert!(!essential.low_freq.is_empty());
    }

    #[test]
    fn test_pattern_similarity() {
        let mut p1 = LegionPattern::new(4, 4);
        let mut p2 = LegionPattern::new(4, 4);

        p1.coefficients[0] = 1.0;
        p2.coefficients[0] = 1.0;

        let sim = p1.similarity(&p2);
        assert!((sim - 1.0).abs() < 0.01);

        // Opposite patterns
        let mut p3 = LegionPattern::new(4, 4);
        p3.coefficients[0] = -1.0;

        let sim_neg = p1.similarity(&p3);
        assert!((sim_neg - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_resonance_confidence() {
        let resonance = Resonance {
            values: vec![0.8, 0.9, -0.2],
            peaks: vec![
                ResonancePeak {
                    index: 0,
                    strength: 0.8,
                },
                ResonancePeak {
                    index: 1,
                    strength: 0.9,
                },
            ],
            conflicts: vec![ResonanceConflict {
                index: 2,
                strength: 0.2,
            }],
        };

        // 1.7 / 1.9 = ~0.89
        let conf = resonance.confidence();
        assert!(conf > 0.85 && conf < 0.95);
        assert!(resonance.is_strong());
    }
}
