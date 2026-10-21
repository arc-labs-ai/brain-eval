//! Statistical primitives for trustworthy metric reporting.
//!
//! A bare point estimate ("recall@1 = 0.86") hides how much of the number is
//! signal and how much is the luck of which questions landed in the sample. Two
//! tools fix that:
//!
//! - [`bootstrap_ci`] — a nonparametric bootstrap 95% confidence interval on
//!   any per-sample metric. Resample the per-question scores with replacement
//!   ~1000× and take the empirical 2.5/97.5 percentiles of the resample means.
//!   No distributional assumption, works for recall / accuracy / judge scores
//!   alike.
//! - [`NoiseBand`] + [`regressed`] — a run-to-run noise band (mean ± `z·stddev`
//!   over repeated trunk runs) so a regression gate fires on *delta beyond
//!   noise*, not a bare threshold. This is what stops the gate from either
//!   flapping on jitter or silently absorbing a real drop.
//!
//! Determinism: [`bootstrap_ci_seeded`] takes an explicit seed so a gate is
//! reproducible; [`bootstrap_ci`] seeds from the sample itself (its length and
//! a digest of the values) so the same input always yields the same interval
//! without a caller-supplied seed.

use serde::{Deserialize, Serialize};

/// A 95%-by-default confidence interval on a metric.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    /// Point estimate — the mean of the observed sample.
    pub point: f64,
    /// Lower bound of the interval.
    pub lower: f64,
    /// Upper bound of the interval.
    pub upper: f64,
    /// Confidence level the bounds were computed at (e.g. `0.95`).
    pub level: f64,
    /// Number of bootstrap resamples used.
    pub resamples: usize,
}

impl ConfidenceInterval {
    /// Half-width of the interval — the `±` a report prints after the point.
    #[must_use]
    pub fn half_width(&self) -> f64 {
        (self.upper - self.lower) / 2.0
    }

    /// Does the interval sit entirely at or above `floor`? A recall gate that
    /// wants "we are confident recall clears the floor" checks `lower >= floor`,
    /// not `point >= floor` — the point can clear a floor the sample can't
    /// actually support.
    #[must_use]
    pub fn clears(&self, floor: f64) -> bool {
        self.lower >= floor
    }
}

/// Default number of bootstrap resamples. 1000 is the field-standard floor for
/// a stable 95% interval; more only tightens the Monte-Carlo error, not the
/// interval itself.
pub const DEFAULT_RESAMPLES: usize = 1000;

/// Nonparametric bootstrap CI at 95%, deterministically seeded from the sample.
///
/// Returns `None` for an empty sample (no interval is meaningful). A
/// single-element sample yields a degenerate `[x, x]` interval, which is
/// correct: one observation carries no spread.
#[must_use]
pub fn bootstrap_ci(sample: &[f64]) -> Option<ConfidenceInterval> {
    bootstrap_ci_with(sample, 0.95, DEFAULT_RESAMPLES, seed_from_sample(sample))
}

/// Nonparametric bootstrap CI with an explicit seed — for reproducible gates
/// and tests. Level and resample count default as in [`bootstrap_ci`].
#[must_use]
pub fn bootstrap_ci_seeded(sample: &[f64], seed: u64) -> Option<ConfidenceInterval> {
    bootstrap_ci_with(sample, 0.95, DEFAULT_RESAMPLES, seed)
}

/// Full-control bootstrap CI: caller picks the confidence `level` (0<level<1),
/// the `resamples` count, and the RNG `seed`.
#[must_use]
pub fn bootstrap_ci_with(
    sample: &[f64],
    level: f64,
    resamples: usize,
    seed: u64,
) -> Option<ConfidenceInterval> {
    if sample.is_empty() || resamples == 0 || !(0.0..1.0).contains(&level) {
        return None;
    }
    let n = sample.len();
    let point = mean(sample);
    if n == 1 {
        return Some(ConfidenceInterval {
            point,
            lower: point,
            upper: point,
            level,
            resamples,
        });
    }

    let mut rng = SplitMix64::new(seed);
    let mut means = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let mut acc = 0.0_f64;
        for _ in 0..n {
            let idx = (rng.next_u64() % n as u64) as usize;
            acc += sample[idx];
        }
        #[allow(clippy::cast_precision_loss)]
        means.push(acc / n as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let tail = (1.0 - level) / 2.0;
    let lower = percentile_sorted(&means, tail);
    let upper = percentile_sorted(&means, 1.0 - tail);
    Some(ConfidenceInterval {
        point,
        lower,
        upper,
        level,
        resamples,
    })
}

/// A run-to-run noise band: the interval a metric is expected to wander within
/// across repeated identical (trunk) runs, absent any real change.
///
/// Built from `k` trunk samples (recommended `k>=5`) as `mean ± z·stddev`,
/// where `z` is the band width in standard deviations (2.0 ≈ 95% for a roughly
/// normal run-to-run distribution). A later run is a regression only when it
/// falls *below* `mean - z·stddev` (see [`regressed`]).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NoiseBand {
    /// Mean of the trunk samples.
    pub mean: f64,
    /// Sample standard deviation of the trunk samples.
    pub stddev: f64,
    /// Band width in standard deviations.
    pub z: f64,
    /// Number of trunk samples the band was built from.
    pub samples: usize,
}

impl NoiseBand {
    /// Lower edge of the band — the floor a run must stay at or above.
    #[must_use]
    pub fn floor(&self) -> f64 {
        self.mean - self.z * self.stddev
    }

    /// Upper edge of the band.
    #[must_use]
    pub fn ceiling(&self) -> f64 {
        self.mean + self.z * self.stddev
    }
}

/// Build a noise band from repeated trunk samples. `z` is the band half-width
/// in standard deviations (2.0 is the usual choice). Returns `None` for fewer
/// than two samples — a band needs spread, and stddev is undefined for `n<2`.
#[must_use]
pub fn noise_band(trunk_samples: &[f64], z: f64) -> Option<NoiseBand> {
    let n = trunk_samples.len();
    if n < 2 {
        return None;
    }
    let mean = mean(trunk_samples);
    let stddev = sample_stddev(trunk_samples, mean);
    Some(NoiseBand {
        mean,
        stddev,
        z,
        samples: n,
    })
}

/// Is `observed` a regression against the band? True only when it drops below
/// the band's floor — inside the band (or above it) is noise/improvement, not a
/// regression. A zero-width band (all trunk runs identical) degrades to a bare
/// `observed < mean` check, which is the correct strict behaviour there.
#[must_use]
pub fn regressed(band: &NoiseBand, observed: f64) -> bool {
    observed < band.floor()
}

/// Arithmetic mean. Callers guarantee non-empty; an empty slice yields `0.0`
/// rather than a NaN so a downstream comparison degrades safely.
#[must_use]
pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let n = xs.len() as f64;
    xs.iter().sum::<f64>() / n
}

/// Bessel-corrected (n-1) sample standard deviation.
#[must_use]
pub fn sample_stddev(xs: &[f64], mean: f64) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let ss: f64 = xs.iter().map(|x| (x - mean).powi(2)).sum();
    #[allow(clippy::cast_precision_loss)]
    let denom = (xs.len() - 1) as f64;
    (ss / denom).sqrt()
}

/// Linear-interpolated percentile over an already-sorted slice. `q` in `[0,1]`.
fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let q = q.clamp(0.0, 1.0);
    #[allow(clippy::cast_precision_loss)]
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor();
    let hi = pos.ceil();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (lo_i, hi_i) = (lo as usize, hi as usize);
    if lo_i == hi_i {
        return sorted[lo_i];
    }
    let frac = pos - lo;
    sorted[lo_i] * (1.0 - frac) + sorted[hi_i] * frac
}

/// Derive a stable seed from the sample so `bootstrap_ci` is deterministic
/// without a caller-supplied seed: two identical samples get identical
/// intervals, different samples almost always differ.
fn seed_from_sample(sample: &[f64]) -> u64 {
    // FNV-1a over the raw bits, mixed with the length.
    let mut h = 0xcbf2_9ce4_8422_2325_u64 ^ (sample.len() as u64);
    for x in sample {
        h ^= x.to_bits();
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

/// Minimal SplitMix64 — a fast, well-distributed PRNG with no external dep.
/// Used only for bootstrap resampling, where cryptographic quality is
/// irrelevant and reproducibility is what matters.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ci_of_constant_sample_is_degenerate() {
        let ci = bootstrap_ci(&[0.5; 20]).expect("non-empty");
        assert!((ci.point - 0.5).abs() < 1e-9);
        assert!((ci.lower - 0.5).abs() < 1e-9);
        assert!((ci.upper - 0.5).abs() < 1e-9);
        assert!((ci.half_width()).abs() < 1e-9);
    }

    #[test]
    fn ci_brackets_the_point_and_is_deterministic() {
        let sample: Vec<f64> = (0..100).map(|i| f64::from(i % 2)).collect(); // mean 0.5
        let a = bootstrap_ci(&sample).expect("non-empty");
        let b = bootstrap_ci(&sample).expect("non-empty");
        assert_eq!(a, b, "seeded-from-sample => reproducible");
        assert!(a.lower <= a.point && a.point <= a.upper);
        assert!(a.lower > 0.2 && a.upper < 0.8, "0/1 sample of n=100: {a:?}");
    }

    #[test]
    fn empty_sample_has_no_interval() {
        assert!(bootstrap_ci(&[]).is_none());
    }

    #[test]
    fn single_sample_is_a_point() {
        let ci = bootstrap_ci(&[0.9]).expect("one element");
        assert!((ci.lower - 0.9).abs() < 1e-9 && (ci.upper - 0.9).abs() < 1e-9);
    }

    #[test]
    fn clears_uses_the_lower_bound() {
        let sample: Vec<f64> = (0..50).map(|i| f64::from(i % 2)).collect();
        let ci = bootstrap_ci(&sample).expect("non-empty");
        assert!(ci.clears(ci.lower - 1e-9));
        assert!(
            !ci.clears(ci.point),
            "point > lower, so point is not cleared"
        );
    }

    #[test]
    fn noise_band_floor_and_regression() {
        let trunk = [0.90, 0.91, 0.89, 0.92, 0.88];
        let band = noise_band(&trunk, 2.0).expect(">=2 samples");
        assert!((band.mean - 0.90).abs() < 1e-9);
        // A run inside the band is not a regression; one below the floor is.
        assert!(!regressed(&band, band.floor() + 1e-6));
        assert!(regressed(&band, band.floor() - 1e-6));
        assert!(
            !regressed(&band, 0.95),
            "an improvement is never a regression"
        );
    }

    #[test]
    fn noise_band_needs_two_samples() {
        assert!(noise_band(&[0.9], 2.0).is_none());
        assert!(noise_band(&[], 2.0).is_none());
    }

    #[test]
    fn zero_width_band_is_strict() {
        let band = noise_band(&[0.8, 0.8, 0.8], 2.0).expect(">=2");
        assert!((band.stddev).abs() < 1e-12);
        assert!(regressed(&band, 0.8 - 1e-9));
        assert!(!regressed(&band, 0.8));
    }

    #[test]
    fn stddev_matches_hand_computation() {
        let xs = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let m = mean(&xs);
        assert!((m - 5.0).abs() < 1e-9);
        // Bessel-corrected stddev of this classic sample is sqrt(32/7).
        assert!((sample_stddev(&xs, m) - (32.0_f64 / 7.0).sqrt()).abs() < 1e-9);
    }
}
