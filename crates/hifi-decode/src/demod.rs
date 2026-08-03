//! FM discriminator (quadrature path). Ports `FMDiscriminator` from
//! `HiFiDecode.py:256-506`, quadrature branch only (`demod_type=quadrature`
//! is hifi-decode's default and, unlike the Hilbert path, needs no IF
//! resample).

use std::f64::consts::PI;

/// Greatest common divisor (Euclid), for `_get_min_iq_length`'s `lcm`.
fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.abs()
}

/// Minimum I/Q oscillator table length that wraps at an exact phase
/// boundary (`FMDiscriminator._get_min_iq_length`, `:280-286`), capped at
/// the caller's block-size hint.
fn min_iq_length(sample_rate: i64, carrier: i64, max_iq_len: usize) -> usize {
    let samples_per_period = sample_rate as f64 / carrier as f64;
    let lcm = (sample_rate / gcd(sample_rate, carrier)) * carrier;
    let min_periods = lcm as f64 / sample_rate as f64;
    // Python's `int()` truncates toward zero, matching `as usize` on a
    // non-negative f64.
    let min_samples = (samples_per_period * min_periods / 2.0) as usize;
    min_samples.min(max_iq_len)
}

/// In-phase/quadrature oscillator tables (`_generate_iq_oscillators`,
/// `:302-310`). Python computes these at `float64` (`DEMOD_DTYPE_NP`) even
/// though the RF/audio buffers around them are `float32` — precision here
/// matters over long tables, so this stays `f64` all the way through.
fn generate_iq_oscillators(carrier: i64, sample_rate: i64, len: usize) -> (Vec<f64>, Vec<f64>) {
    let two_pi_carrier = 2.0 * PI * carrier as f64;
    let mut i_osc = Vec::with_capacity(len);
    let mut q_osc = Vec::with_capacity(len);
    for i in 0..len {
        let t = i as f64 / sample_rate as f64;
        let (sin, cos) = (two_pi_carrier * t).sin_cos();
        i_osc.push(cos);
        q_osc.push(-sin);
    }
    (i_osc, q_osc)
}

/// Complex-conjugate quadrature FM discriminator. Ports
/// `FMDiscriminator.demod_quadrature` (`:434-480`) exactly, including its
/// off-by-one: the source loop is `for i in range(1, rf_len)` writing
/// `out_demod[i-1]`, so the *last* output sample is never written by the
/// Python reference (left as `np.empty` garbage). This port writes `0.0`
/// there instead of leaving it undefined — deterministic, but still not a
/// meaningful demod value, so exclude the last sample when comparing
/// against a Python fixture.
pub struct FmDiscriminator {
    carrier: i64,
    sample_rate: i64,
    deviation: i64,
    i_osc: Vec<f64>,
    q_osc: Vec<f64>,
}

impl FmDiscriminator {
    /// `max_iq_len` should be the caller's per-call input length (Python
    /// sizes it from `initialBlockResampledSize`, the same buffer the
    /// oscillator table is later indexed against).
    pub fn new_quadrature(sample_rate: f64, carrier_center: f64, deviation: f64, max_iq_len: usize) -> Self {
        let sample_rate = sample_rate.round() as i64;
        let carrier = carrier_center.round() as i64;
        let deviation = deviation.round() as i64;
        let iq_len = min_iq_length(sample_rate, carrier, max_iq_len);
        let (i_osc, q_osc) = generate_iq_oscillators(carrier, sample_rate, iq_len);
        FmDiscriminator {
            carrier,
            sample_rate,
            deviation,
            i_osc,
            q_osc,
        }
    }

    /// The rounded carrier center frequency (Hz) this discriminator is
    /// configured for.
    pub fn carrier_hz(&self) -> f64 {
        self.carrier as f64
    }

    pub fn work(&self, in_rf: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; in_rf.len()];
        self.work_into(in_rf, &mut out);
        out
    }

    pub fn work_into(&self, in_rf: &[f32], out_demod: &mut [f32]) {
        assert_eq!(in_rf.len(), out_demod.len());
        let two_pi = 2.0 * PI;
        let phase_scale = self.sample_rate as f64 / (two_pi * self.deviation as f64);
        let iq_len = self.i_osc.len();
        let rf_len = in_rf.len();
        if rf_len < 2 {
            return;
        }

        let mut prev_i = in_rf[0] as f64 * self.i_osc[0];
        let mut prev_q = in_rf[0] as f64 * self.q_osc[0];

        for i in 1..rf_len {
            let iq_index = i % iq_len;
            let sign = if ((i / iq_len) & 1) == 0 { 1.0 } else { -1.0 };
            let rf_signed = in_rf[i] as f64 * sign;

            let i_in = rf_signed * self.i_osc[iq_index];
            let q_in = rf_signed * self.q_osc[iq_index];

            let imag = q_in * prev_i - i_in * prev_q;
            let real = i_in * prev_i + q_in * prev_q;
            let delta = imag.atan2(real);

            let out = delta * phase_scale;
            out_demod[i - 1] = (out as f32).clamp(f32::MIN, f32::MAX);

            prev_i = i_in;
            prev_q = q_in;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity check independent of any Python fixture: FM-modulate a known
    /// tone at a manageable sample rate and confirm the discriminator's
    /// output is dominated by that tone's frequency (checked in the
    /// frequency domain, since the raw pre-resample discriminator output
    /// is scaled by `sample_rate/(2*pi*deviation)` rather than sitting at
    /// unit amplitude — see the fixture-based test for an absolute-value
    /// comparison against the real Python decoder).
    #[test]
    fn recovers_a_synthesized_tone() {
        let fs = 8_000_000.0;
        let carrier = 1_400_000.0;
        let deviation = 150_000.0;
        let tone_hz = 1_000.0;
        let n = 40_000usize; // 5ms

        let dt = 1.0 / fs;
        let mut rf = vec![0.0f32; n];
        let mut phase = 0.0f64;
        for i in 0..n {
            let t = i as f64 * dt;
            let audio = (2.0 * PI * tone_hz * t).sin();
            let inst_freq = carrier + deviation * audio;
            phase += 2.0 * PI * inst_freq * dt;
            rf[i] = phase.cos() as f32;
        }

        let disc = FmDiscriminator::new_quadrature(fs, carrier, deviation, n);
        let demod = disc.work(&rf);

        // Exclude the last sample (never written, see doc comment) and a
        // short settling prefix. Naive DFT at exactly `tone_hz` vs a
        // neighboring off-tone frequency: the tone bin should dominate.
        let usable = &demod[100..n - 1];
        let goertzel = |freq: f64| -> f64 {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for (i, &sample) in usable.iter().enumerate() {
                let angle = -2.0 * PI * freq * (i as f64) / fs;
                re += sample as f64 * angle.cos();
                im += sample as f64 * angle.sin();
            }
            (re * re + im * im).sqrt()
        };
        let at_tone = goertzel(tone_hz);
        let off_tone = goertzel(tone_hz * 1.7);
        assert!(
            at_tone > off_tone * 10.0,
            "tone bin {at_tone} not dominant over off-tone bin {off_tone}"
        );
    }
}
