//! Chebyshev Type II filter design, missing from `sci-rs` 0.4 (its
//! `FilterType::ChebyshevII` branch is an unconditional `todo!()`). Mirrors
//! `scipy.signal.cheb2ap` / `scipy.signal.cheby2` exactly, built on the
//! frequency-transform and bilinear-transform primitives `sci-rs` already
//! exports (`lp2bp_zpk_dyn`, `bilinear_zpk_dyn`, `zpk2sos_dyn`) — the same
//! pipeline `iirfilter_dyn` runs internally for Butterworth, with `cheb2ap`
//! substituted for `buttap_dyn`.
//!
//! `sci-rs::lp2hp_zpk_dyn` is *not* reused: its zero/gain mapping disagrees
//! with `scipy.signal.lp2hp_zpk` (verified against a real scipy run — see
//! the `lp2hp_zpk` doc comment below), so the highpass path needs a local,
//! scipy-matching replacement.

use core::f64::consts::PI;

use nalgebra::Complex;
use sci_rs::signal::filter::design::{
    bilinear_zpk_dyn, lp2bp_zpk_dyn, zpk2sos_dyn, FilterBandType, Sos, ZpkFormatFilter,
};

/// Number of zeros a proper transfer function must gain (or, equivalently,
/// zeros the lowpass prototype has "at infinity") when frequency-transformed.
/// `sci-rs` has an identical private helper (`relative_degree_dyn`) that
/// isn't re-exported from `design::*`, so it's duplicated here verbatim.
fn relative_degree(zeros: &[Complex<f64>], poles: &[Complex<f64>]) -> usize {
    poles
        .len()
        .checked_sub(zeros.len())
        .expect("improper transfer function: fewer poles than zeros")
}

/// Product of a slice of complex values, computed by explicit fold (rather
/// than relying on `Iterator::product`, whose `Complex` impl is not
/// guaranteed across `num-complex` versions).
fn complex_product(values: &[Complex<f64>]) -> Complex<f64> {
    values
        .iter()
        .fold(Complex::new(1.0, 0.0), |acc, &value| acc * value)
}

/// Analog lowpass prototype (zeros, poles, gain) for an Nth-order Chebyshev
/// Type II filter with `rs` dB of stopband attenuation, cutoff normalized to
/// 1 rad/s. Equivalent to `scipy.signal.cheb2ap`.
fn cheb2ap(order: usize, rs: f64) -> ZpkFormatFilter<f64> {
    assert!(order > 0, "cheb2ap requires a positive filter order");
    let n = order as isize;
    let n_f = order as f64;

    // Ripple factor (epsilon) and its arcsinh-scaled equivalent.
    let de = 1.0 / (10f64.powf(0.1 * rs) - 1.0).sqrt();
    let mu = (1.0 / de).asinh() / n_f;

    // m1 = -N+1, -N+3, ..., N-1 (N values, symmetric about 0). For odd N
    // this includes 0 (giving one real pole); for even N it never does
    // (all values share the parity of -N+1, which is odd).
    let m1: Vec<f64> = (0..order).map(|k| (-n + 1 + 2 * k as isize) as f64).collect();

    // Poles around the unit circle like Butterworth, then warped into
    // Chebyshev II: p = -1 / sinh(mu + i*theta1).
    let p: Vec<Complex<f64>> = m1
        .iter()
        .map(|&m| {
            let theta1 = PI * m / (2.0 * n_f);
            -Complex::new(1.0, 0.0) / Complex::new(mu, theta1).sinh()
        })
        .collect();

    // Zeros: same m1 sequence with any m == 0 dropped (odd order has one
    // fewer finite zero than pole). z = i / sin(m*pi/2N), purely imaginary.
    let z: Vec<Complex<f64>> = m1
        .iter()
        .filter(|&&m| m != 0.0)
        .map(|&m| Complex::new(0.0, 1.0 / (PI * m / (2.0 * n_f)).sin()))
        .collect();

    let k = (complex_product(&p.iter().map(|&pi| -pi).collect::<Vec<_>>())
        / complex_product(&z.iter().map(|&zi| -zi).collect::<Vec<_>>()))
    .re;

    ZpkFormatFilter::new(z, p, k)
}

/// Lowpass-to-highpass analog frequency transform (`s -> wo/s`), matching
/// `scipy.signal.lp2hp_zpk` exactly:
///
/// ```text
/// z_hp = wo / z            (plus `degree = len(p) - len(z)` zeros at the origin,
/// p_hp = wo / p             carrying over zeros the lowpass prototype had at infinity)
/// k_hp = k * real(prod(-z) / prod(-p))   (using the ORIGINAL z, p — not z_hp/p_hp)
/// ```
///
/// `sci-rs::lp2hp_zpk_dyn` computes something else for `z_hp`/`k_hp` (its own
/// bundled unit test only checks it against its own output, not scipy's —
/// confirmed by re-running that test's exact fixture through real scipy,
/// which returns different `z1`/`k1` values), so it is not used here.
fn lp2hp_zpk(zpk: &ZpkFormatFilter<f64>, wo: f64) -> ZpkFormatFilter<f64> {
    let degree = relative_degree(&zpk.z, &zpk.p);

    let mut z_hp: Vec<Complex<f64>> = zpk.z.iter().map(|&zi| Complex::new(wo, 0.0) / zi).collect();
    let p_hp: Vec<Complex<f64>> = zpk.p.iter().map(|&pi| Complex::new(wo, 0.0) / pi).collect();
    z_hp.extend((0..degree).map(|_| Complex::new(0.0, 0.0)));

    let prod_neg_z = complex_product(&zpk.z.iter().map(|&zi| -zi).collect::<Vec<_>>());
    let prod_neg_p = complex_product(&zpk.p.iter().map(|&pi| -pi).collect::<Vec<_>>());
    let k_hp = zpk.k * (prod_neg_z / prod_neg_p).re;

    ZpkFormatFilter::new(z_hp, p_hp, k_hp)
}

/// Digital Chebyshev Type II filter, second-order-sections form. Equivalent
/// to `scipy.signal.cheby2(order, rs, wn, btype, fs=fs, output="sos")` for
/// `btype` in `{highpass, bandpass}` (the two hifi-decode needs); `wn` is in
/// Hz, `fs` is the sample rate in Hz. Panics if `wn`/`band_type` are
/// inconsistent (single frequency for highpass, a `[low, high]` pair for
/// bandpass) or a frequency does not lie strictly within `(0, fs/2)`.
pub fn cheby2_sos(order: usize, rs: f64, wn: &[f64], band_type: FilterBandType, fs: f64) -> Vec<Sos<f64>> {
    // Nyquist-normalize (1.0 == fs/2), matching scipy's `Wn = 2*Wn/fs`
    // convention when `fs` is passed explicitly.
    let wn_norm: Vec<f64> = wn.iter().map(|&w| 2.0 * w / fs).collect();
    assert!(
        wn_norm.iter().all(|&w| w > 0.0 && w < 1.0),
        "critical frequencies must lie strictly within (0, fs/2)"
    );

    // Prewarp for the bilinear transform, using scipy/sci-rs's internal
    // analog reference rate of 2.0 (not the real `fs`; the real sample rate
    // only enters through the Nyquist normalization above).
    const FS_ANALOG: f64 = 2.0;
    let warped: Vec<f64> = wn_norm
        .iter()
        .map(|&w| 2.0 * FS_ANALOG * (PI * w / FS_ANALOG).tan())
        .collect();

    let prototype = cheb2ap(order, rs);

    let zpk = match band_type {
        FilterBandType::Highpass => {
            assert_eq!(warped.len(), 1, "highpass needs exactly one critical frequency");
            lp2hp_zpk(&prototype, warped[0])
        }
        FilterBandType::Bandpass => {
            assert_eq!(warped.len(), 2, "bandpass needs exactly two critical frequencies");
            let bw = warped[1] - warped[0];
            let wo = (warped[0] * warped[1]).sqrt();
            lp2bp_zpk_dyn(prototype, Some(wo), Some(bw))
        }
        other => panic!("cheby2_sos does not support {other:?}; only highpass and bandpass are implemented"),
    };

    let zpk = bilinear_zpk_dyn(zpk, FS_ANALOG);
    zpk2sos_dyn(order, zpk, None, Some(false)).sos
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Digital frequency response `H(e^{j*2*pi*f/fs})` of an SOS cascade, as
    /// scipy's `sosfreqz` computes it.
    fn sos_response(sos: &[Sos<f64>], freq: f64, fs: f64) -> Complex<f64> {
        let z_inv = Complex::new(0.0, -2.0 * PI * freq / fs).exp();
        sos.iter().fold(Complex::new(1.0, 0.0), |acc, section| {
            let num = Complex::new(section.b[0], 0.0)
                + Complex::new(section.b[1], 0.0) * z_inv
                + Complex::new(section.b[2], 0.0) * z_inv * z_inv;
            let den = Complex::new(section.a[0], 0.0)
                + Complex::new(section.a[1], 0.0) * z_inv
                + Complex::new(section.a[2], 0.0) * z_inv * z_inv;
            acc * (num / den)
        })
    }

    /// Compares against `scipy.signal.cheby2(..., output="sos")` evaluated
    /// with `scipy.signal.sosfreqz` at the same frequencies (see the
    /// hifi-decode port plan, step 2, for the exact Python invocation).
    fn assert_matches_scipy(
        order: usize,
        rs: f64,
        wn: &[f64],
        band_type: FilterBandType,
        fs: f64,
        expected: &[(f64, f64, f64)], // (freq_hz, mag_db, phase_rad)
    ) {
        let sos = cheby2_sos(order, rs, wn, band_type, fs);
        for &(freq, expected_mag_db, expected_phase) in expected {
            let h = sos_response(&sos, freq, fs);
            let mag_db = 20.0 * (h.norm() + 1e-300).log10();
            let phase = h.arg();

            if expected_mag_db < -100.0 {
                // Deep stopband: both implementations are near the f64 noise
                // floor here, so only require "very attenuated" rather than
                // matching scipy's exact residual.
                assert!(
                    mag_db < -80.0,
                    "at {freq} Hz: expected deep attenuation (~{expected_mag_db} dB), got {mag_db} dB"
                );
            } else {
                assert!(
                    (mag_db - expected_mag_db).abs() < 1e-3,
                    "at {freq} Hz: mag_db {mag_db} vs scipy {expected_mag_db}"
                );
                let phase_diff = (phase - expected_phase).rem_euclid(2.0 * PI);
                let phase_diff = phase_diff.min(2.0 * PI - phase_diff);
                assert!(
                    phase_diff < 1e-3,
                    "at {freq} Hz: phase {phase} vs scipy {expected_phase}"
                );
            }
        }
    }

    #[test]
    fn matches_scipy_afe_bandpass() {
        // scipy.signal.cheby2(22, 220, [1400000-371506.25, 1400000+371506.25],
        //                     btype="bandpass", output="sos", fs=40000000)
        assert_matches_scipy(
            22,
            220.0,
            &[1_400_000.0 - 371_506.25, 1_400_000.0 + 371_506.25],
            FilterBandType::Bandpass,
            40_000_000.0,
            &[
                (100_000.0, -224.192844, -0.343047),
                (1_000_000.0, -220.353417, -1.504472),
                (1_300_000.0, -0.000000, 2.905119),
                (1_400_000.0, 0.000000, -2.733416),
                (1_500_000.0, -0.000000, -2.459199),
                (1_771_506.25, -220.000000, -0.740787),
                (2_500_000.0, -220.002113, 0.350776),
                (5_000_000.0, -220.235436, -1.867560),
            ],
        );
    }

    #[test]
    fn matches_scipy_headswitch_highpass() {
        // scipy.signal.cheby2(22, 200, 28000, btype="highpass", output="sos",
        //                     fs=192000)
        assert_matches_scipy(
            22,
            200.0,
            &[28_000.0],
            FilterBandType::Highpass,
            192_000.0,
            &[
                (1_000.0, -202.556043, -0.305026),
                (10_000.0, -209.451848, -3.092793),
                (20_000.0, -205.274000, 2.955935),
                (28_000.0, -200.000000, -0.242647),
                (40_000.0, -13.067454, -1.342006),
                (60_000.0, -0.000000, 0.105240),
                (90_000.0, -0.000000, 0.885835),
            ],
        );
    }
}
