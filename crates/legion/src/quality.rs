//! Quality management for Legion agents.
//!
//! Implements frequency-band based quality targeting and polynomial
//! quality prediction curves from the Sigil implementation.

use std::fmt;

/// Frequency band for agent operation.
///
/// Based on the holographic principle - each band represents a different
/// perspective on the same task, not a decomposition:
///
/// - **Anima**: Core identity, unchanging (DC component, 0.0)
/// - **Strategic**: High-level planning (ultra-low freq, 0.0-0.1)
/// - **Tactical**: Step-by-step execution (low freq, 0.1-0.3)
/// - **Operational**: Actual work (mid freq, 0.3-0.6)
/// - **Verification**: Quality checking (high freq, 0.6-0.9)
/// - **Reflective**: Meta-cognition (ultra-high freq, 0.9-1.0)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FrequencyBand {
    /// Anima: Core identity, unchanging (DC component).
    /// This is the "soul" of the response - never filtered out.
    Anima,
    /// Strategic: High-level planning and goal-setting.
    /// Ultra-low frequency, sets overall direction.
    Strategic,
    /// Tactical: Step-by-step approach and methodology.
    /// Low frequency, breaks strategy into actionable steps.
    Tactical,
    /// Operational: The actual work and content generation.
    /// Mid frequency, where most output is produced.
    Operational,
    /// Verification: Quality checking and error detection.
    /// High frequency, catches mistakes and validates output.
    Verification,
    /// Reflective: Meta-cognition and self-improvement.
    /// Ultra-high frequency, thinks about thinking.
    Reflective,
}

impl FrequencyBand {
    /// Returns the frequency range for this band as (min, max).
    pub fn frequency_range(&self) -> (f32, f32) {
        match self {
            FrequencyBand::Anima => (0.0, 0.0),       // DC component
            FrequencyBand::Strategic => (0.0, 0.1),   // Ultra-low
            FrequencyBand::Tactical => (0.1, 0.3),    // Low
            FrequencyBand::Operational => (0.3, 0.6), // Mid
            FrequencyBand::Verification => (0.6, 0.9), // High
            FrequencyBand::Reflective => (0.9, 1.0),  // Ultra-high
        }
    }

    /// Returns the emphasis weight for this band in consensus.
    pub fn emphasis(&self) -> f32 {
        match self {
            FrequencyBand::Anima => 1.0,       // Core identity always full weight
            FrequencyBand::Strategic => 0.9,   // Strategy high weight
            FrequencyBand::Tactical => 0.85,   // Tactics moderate-high
            FrequencyBand::Operational => 1.0, // Primary work full weight
            FrequencyBand::Verification => 0.8, // Verification slightly lower
            FrequencyBand::Reflective => 0.6,  // Meta-cognition background
        }
    }

    /// Returns the context fraction (legacy compatibility).
    pub fn context_fraction(&self) -> f32 {
        match self {
            FrequencyBand::Anima => 0.10,      // Core only
            FrequencyBand::Strategic => 0.25,  // Broad strokes
            FrequencyBand::Tactical => 0.50,   // Half context
            FrequencyBand::Operational => 0.75, // Most context
            FrequencyBand::Verification => 0.90, // Nearly all
            FrequencyBand::Reflective => 1.00, // Full context
        }
    }

    /// Returns typical response time multiplier.
    pub fn time_multiplier(&self) -> f32 {
        match self {
            FrequencyBand::Anima => 0.1,       // Instant
            FrequencyBand::Strategic => 0.25,  // Fast
            FrequencyBand::Tactical => 0.5,    // Moderate
            FrequencyBand::Operational => 1.0, // Normal
            FrequencyBand::Verification => 1.25, // Slightly slower
            FrequencyBand::Reflective => 1.5,  // Slowest
        }
    }

    /// Creates a spectral filter that emphasizes this band's frequency range.
    pub fn spectral_filter(&self) -> SpectralFilter {
        let (min, max) = self.frequency_range();
        SpectralFilter {
            band: *self,
            min_freq: min,
            max_freq: max,
            rolloff: 0.1, // 10% rolloff at edges
        }
    }

    /// Creates a frequency band from an agent index.
    pub fn from_index(index: usize, total: usize) -> Self {
        let fraction = (index + 1) as f32 / total as f32;
        if fraction <= 0.10 {
            FrequencyBand::Anima
        } else if fraction <= 0.25 {
            FrequencyBand::Strategic
        } else if fraction <= 0.45 {
            FrequencyBand::Tactical
        } else if fraction <= 0.70 {
            FrequencyBand::Operational
        } else if fraction <= 0.90 {
            FrequencyBand::Verification
        } else {
            FrequencyBand::Reflective
        }
    }

    /// Returns all frequency bands in order.
    pub fn all() -> &'static [FrequencyBand] {
        &[
            FrequencyBand::Anima,
            FrequencyBand::Strategic,
            FrequencyBand::Tactical,
            FrequencyBand::Operational,
            FrequencyBand::Verification,
            FrequencyBand::Reflective,
        ]
    }

    /// Returns the priority for respawn/recovery operations.
    ///
    /// Higher values = more urgent to recover.
    pub fn priority(&self) -> u8 {
        match self {
            FrequencyBand::Anima => 10,       // Core identity - highest priority
            FrequencyBand::Strategic => 8,    // Planning important
            FrequencyBand::Operational => 7,  // Work production
            FrequencyBand::Tactical => 6,     // Step execution
            FrequencyBand::Verification => 4, // Quality checking
            FrequencyBand::Reflective => 2,   // Meta-cognition can wait
        }
    }
}

impl fmt::Display for FrequencyBand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FrequencyBand::Anima => write!(f, "∿ (Anima)"),
            FrequencyBand::Strategic => write!(f, "⟁ (Strategic)"),
            FrequencyBand::Tactical => write!(f, "⟀ (Tactical)"),
            FrequencyBand::Operational => write!(f, "⊕ (Operational)"),
            FrequencyBand::Verification => write!(f, "⫰ (Verification)"),
            FrequencyBand::Reflective => write!(f, "◉ (Reflective)"),
        }
    }
}

/// Spectral filter for emphasizing a frequency band.
#[derive(Debug, Clone, Copy)]
pub struct SpectralFilter {
    /// The band this filter emphasizes.
    pub band: FrequencyBand,
    /// Minimum frequency (0.0 - 1.0).
    pub min_freq: f32,
    /// Maximum frequency (0.0 - 1.0).
    pub max_freq: f32,
    /// Rolloff at edges (0.0 - 1.0).
    pub rolloff: f32,
}

impl SpectralFilter {
    /// Applies the filter to a frequency, returning the weight (0.0 - 1.0).
    pub fn apply(&self, freq: f32) -> f32 {
        if freq < self.min_freq - self.rolloff || freq > self.max_freq + self.rolloff {
            return 0.0;
        }

        // Full pass in band
        if freq >= self.min_freq && freq <= self.max_freq {
            return 1.0;
        }

        // Smooth rolloff at edges
        if freq < self.min_freq {
            let dist = self.min_freq - freq;
            return 1.0 - (dist / self.rolloff).clamp(0.0, 1.0);
        }

        let dist = freq - self.max_freq;
        1.0 - (dist / self.rolloff).clamp(0.0, 1.0)
    }
}

// ==================== Quality Curves ====================

/// Polynomial quality prediction curve.
///
/// Quality as a function of fragments loaded:
/// ```text
/// Q(k/n) = c[0] + c[1]*(k/n) + c[2]*(k/n)² + c[3]*(k/n)³
/// ```
///
/// Where k = fragments loaded, n = total fragments.
#[derive(Debug, Clone)]
pub struct QualityCurve {
    /// Polynomial coefficients [c0, c1, c2, c3].
    pub coefficients: [f32; 4],
    /// Minimum fragments for usable output.
    pub min_fragments: u16,
    /// Fragments needed for sufficient quality (95%+).
    pub sufficient_fragments: u16,
}

impl QualityCurve {
    /// Spectral encoding curve: 60% from first fragment!
    ///
    /// Used for DCT-based holographic encoding where the DC and low-frequency
    /// coefficients are replicated across all fragments.
    pub const SPECTRAL: QualityCurve = QualityCurve {
        coefficients: [0.60, 0.30, 0.08, 0.02],
        min_fragments: 1,
        sufficient_fragments: 6,
    };

    /// Low-rank decomposition curve: Perfect for KV cache.
    ///
    /// Rank-4 approximation sufficient for 99%+ quality.
    pub const LRDF: QualityCurve = QualityCurve {
        coefficients: [0.30, 0.50, 0.15, 0.05],
        min_fragments: 1,
        sufficient_fragments: 4,
    };

    /// Random projection curve: Best for distributed storage.
    ///
    /// Needs more fragments but tolerates high loss.
    pub const RANDOM_PROJECTION: QualityCurve = QualityCurve {
        coefficients: [0.20, 0.40, 0.25, 0.15],
        min_fragments: 2,
        sufficient_fragments: 8,
    };

    /// Linear curve: Traditional (non-holographic) storage.
    pub const LINEAR: QualityCurve = QualityCurve {
        coefficients: [0.0, 1.0, 0.0, 0.0],
        min_fragments: 1,
        sufficient_fragments: 10,
    };

    /// Predicts quality for k fragments out of n total.
    pub fn predict(&self, k: u16, n: u16) -> f32 {
        if n == 0 {
            return 0.0;
        }

        let ratio = k as f32 / n as f32;
        let mut quality = 0.0f32;
        let mut power = 1.0f32;

        for coeff in &self.coefficients {
            quality += coeff * power;
            power *= ratio;
        }

        quality.clamp(0.0, 1.0)
    }

    /// Returns the minimum ratio (k/n) for usable quality (30%+).
    pub fn min_ratio(&self) -> f32 {
        // Binary search for ratio where quality >= 0.3
        let mut low = 0.0f32;
        let mut high = 1.0f32;

        for _ in 0..20 {
            let mid = (low + high) / 2.0;
            let q = self.predict((mid * 100.0) as u16, 100);
            if q < 0.3 {
                low = mid;
            } else {
                high = mid;
            }
        }

        high
    }

    /// Returns the ratio (k/n) for target quality.
    pub fn ratio_for_quality(&self, target_quality: f32) -> f32 {
        let target = target_quality.clamp(0.0, 1.0);

        let mut low = 0.0f32;
        let mut high = 1.0f32;

        for _ in 0..20 {
            let mid = (low + high) / 2.0;
            let q = self.predict((mid * 100.0) as u16, 100);
            if q < target {
                low = mid;
            } else {
                high = mid;
            }
        }

        high
    }
}

// ==================== Quality Targets ====================

/// Quality target for an operation.
#[derive(Debug, Clone)]
pub struct QualityTarget {
    /// Minimum acceptable quality.
    pub minimum: f32,
    /// Target quality.
    pub target: f32,
    /// Maximum useful quality (diminishing returns above this).
    pub maximum: f32,
    /// Preferred frequency band.
    pub preferred_band: Option<FrequencyBand>,
    /// Quality curve to use.
    pub curve: QualityCurve,
}

impl Default for QualityTarget {
    fn default() -> Self {
        Self {
            minimum: 0.5,
            target: 0.8,
            maximum: 1.0,
            preferred_band: None,
            curve: QualityCurve::SPECTRAL,
        }
    }
}

impl QualityTarget {
    /// Creates a target for fast responses.
    pub fn fast() -> Self {
        Self {
            minimum: 0.3,
            target: 0.5,
            maximum: 0.7,
            preferred_band: Some(FrequencyBand::Strategic),
            curve: QualityCurve::SPECTRAL,
        }
    }

    /// Creates a target for balanced responses.
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Creates a target for high-quality responses.
    pub fn quality() -> Self {
        Self {
            minimum: 0.8,
            target: 0.95,
            maximum: 1.0,
            preferred_band: Some(FrequencyBand::Reflective),
            curve: QualityCurve::SPECTRAL,
        }
    }

    /// Checks if a quality level meets the minimum.
    pub fn meets_minimum(&self, quality: f32) -> bool {
        quality >= self.minimum
    }

    /// Checks if a quality level meets the target.
    pub fn meets_target(&self, quality: f32) -> bool {
        quality >= self.target
    }

    /// Returns the number of fragments needed for minimum quality.
    pub fn fragments_for_minimum(&self, total_fragments: u16) -> u16 {
        let ratio = self.curve.ratio_for_quality(self.minimum);
        ((ratio * total_fragments as f32).ceil() as u16).max(1)
    }

    /// Returns the number of fragments needed for target quality.
    pub fn fragments_for_target(&self, total_fragments: u16) -> u16 {
        let ratio = self.curve.ratio_for_quality(self.target);
        ((ratio * total_fragments as f32).ceil() as u16).max(1)
    }
}

// ==================== Quality Metrics ====================

/// Metrics tracking quality over time.
#[derive(Debug, Clone, Default)]
pub struct QualityMetrics {
    /// Total operations.
    pub total_ops: u64,
    /// Operations meeting minimum quality.
    pub met_minimum: u64,
    /// Operations meeting target quality.
    pub met_target: u64,
    /// Average quality achieved.
    pub avg_quality: f64,
    /// Quality by frequency band.
    pub by_band: std::collections::HashMap<FrequencyBand, f64>,
}

impl QualityMetrics {
    /// Records an operation with the given quality and band.
    pub fn record(&mut self, quality: f32, band: FrequencyBand, target: &QualityTarget) {
        self.total_ops += 1;
        if target.meets_minimum(quality) {
            self.met_minimum += 1;
        }
        if target.meets_target(quality) {
            self.met_target += 1;
        }

        // Update running average
        let n = self.total_ops as f64;
        self.avg_quality = self.avg_quality * (n - 1.0) / n + quality as f64 / n;

        // Update band average
        let band_avg = self.by_band.entry(band).or_insert(0.0);
        *band_avg = (*band_avg * (n - 1.0) / n) + quality as f64 / n;
    }

    /// Returns success rate (meeting minimum).
    pub fn success_rate(&self) -> f64 {
        if self.total_ops == 0 {
            return 0.0;
        }
        self.met_minimum as f64 / self.total_ops as f64
    }

    /// Returns target achievement rate.
    pub fn target_rate(&self) -> f64 {
        if self.total_ops == 0 {
            return 0.0;
        }
        self.met_target as f64 / self.total_ops as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_band_context() {
        assert!((FrequencyBand::Anima.context_fraction() - 0.10).abs() < 0.001);
        assert!((FrequencyBand::Reflective.context_fraction() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_frequency_band_ordering() {
        assert!(FrequencyBand::Anima < FrequencyBand::Strategic);
        assert!(FrequencyBand::Strategic < FrequencyBand::Tactical);
        assert!(FrequencyBand::Tactical < FrequencyBand::Operational);
        assert!(FrequencyBand::Operational < FrequencyBand::Verification);
        assert!(FrequencyBand::Verification < FrequencyBand::Reflective);
    }

    #[test]
    fn test_frequency_band_emphasis() {
        assert_eq!(FrequencyBand::Anima.emphasis(), 1.0);
        assert_eq!(FrequencyBand::Operational.emphasis(), 1.0);
        assert!(FrequencyBand::Reflective.emphasis() < FrequencyBand::Operational.emphasis());
    }

    #[test]
    fn test_spectral_filter() {
        let filter = FrequencyBand::Operational.spectral_filter();

        // In band - full pass
        assert!((filter.apply(0.45) - 1.0).abs() < 0.01);

        // Out of band - zero
        assert!((filter.apply(0.0) - 0.0).abs() < 0.01);
        assert!((filter.apply(1.0) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_quality_curve_spectral() {
        let curve = QualityCurve::SPECTRAL;

        // First fragment gives 60% quality
        let q1 = curve.predict(1, 10);
        assert!(q1 > 0.55 && q1 < 0.65, "First fragment should give ~60% quality, got {}", q1);

        // All fragments give 100%
        let qall = curve.predict(10, 10);
        assert!(qall > 0.95, "All fragments should give ~100% quality, got {}", qall);
    }

    #[test]
    fn test_quality_curve_lrdf() {
        let curve = QualityCurve::LRDF;

        // With sufficient_fragments = 4, using n=4 gives near-complete quality
        let q_all = curve.predict(4, 4);
        assert!(q_all > 0.95, "4/4 fragments should give >95% quality, got {}", q_all);

        // Half gives decent quality
        let q_half = curve.predict(2, 4);
        assert!(q_half > 0.50, "2/4 fragments should give >50% quality, got {}", q_half);
    }

    #[test]
    fn test_quality_target_fragments() {
        let target = QualityTarget::fast();
        let min_frags = target.fragments_for_minimum(10);
        let target_frags = target.fragments_for_target(10);

        assert!(min_frags <= target_frags);
        assert!(min_frags >= 1);
    }

    #[test]
    fn test_quality_metrics_recording() {
        let mut metrics = QualityMetrics::default();
        let target = QualityTarget::default();

        metrics.record(0.9, FrequencyBand::Operational, &target);
        metrics.record(0.6, FrequencyBand::Tactical, &target);

        assert_eq!(metrics.total_ops, 2);
        assert_eq!(metrics.met_minimum, 2);
        assert_eq!(metrics.met_target, 1);
    }
}
