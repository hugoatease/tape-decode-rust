//! Post-decode audio shaping: DC blocking, de-emphasis, and the
//! companding expander. Ports `DCBlocker`, `Deemphasis`, `Expander`, and
//! `build_shelf_filter` from `HiFiDecode.py`, plus the VHS/8mm chain
//! ordering from `PostProcessor.py`'s `expander_vhs_worker`/
//! `expander_8mm_worker`.
//!
//! Not yet wired to fixtures: the fixture generator
//! (`scripts/hifi-fixtures/generate_reference.py`) only instruments
//! `HiFiDecode.block_decode`, not `PostProcessor`'s separate stage chain.
//! The shelf-filter math itself (`build_shelf_filter`) is verified
//! numerically against `scipy.signal.bilinear` inline (see the module's
//! tests), but the chain orderings below rely on a careful line-by-line
//! reading of `PostProcessor.py:332-502`, not a fixture comparison.

use std::f64::consts::PI;

/// `build_shelf_filter`'s direction argument (`HiFiDecode.py:521,529`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShelfDirection {
    Low,
    High,
}

/// A first-order digital shelf filter: `b[0], b[1]` numerator, `1, a1`
/// denominator (already normalized so `a[0] == 1`).
#[derive(Clone, Copy, Debug)]
struct ShelfFilter {
    b: [f64; 2],
    a1: f64,
}

/// `build_shelf_filter` (`HiFiDecode.py:520-542`): a first-order analog
/// shelf (low: direction toward DC gain; high: toward Nyquist gain),
/// normalized to 0dB at the far end, then bilinear-transformed. The
/// bilinear step below is the closed-form solution for a 2-coefficient
/// `b(s)/a(s)` pair (verified numerically against `scipy.signal.bilinear`
/// in this module's tests) rather than a call into `tape_dsp`'s general
/// polynomial `bilinear`, since first order needs none of that machinery.
fn build_shelf_filter(direction: ShelfDirection, tau1: f64, tau2: f64, fs: f64) -> ShelfFilter {
    let b1 = 1.0 / (tau1 / tau2);
    let (b_analog, a_analog, gain) = match direction {
        ShelfDirection::Low => {
            let b_analog = [tau2 * tau2 / tau1, b1];
            let a_analog = [tau1, 1.0];
            let gain = b_analog[1] / a_analog[1];
            (b_analog, a_analog, gain)
        }
        ShelfDirection::High => {
            let b_analog = [tau2, b1];
            let a_analog = [tau2, 1.0];
            let gain = b_analog[0] / a_analog[0];
            (b_analog, a_analog, gain)
        }
    };
    let b_analog = [b_analog[0] / gain, b_analog[1] / gain];

    // First-order bilinear transform (s = 2*fs*(1-z^-1)/(1+z^-1)), solved
    // in closed form and normalized so the digital a[0] is 1.
    let (b0, b1a) = (b_analog[0], b_analog[1]);
    let (a0, a1a) = (a_analog[0], a_analog[1]);
    let denom = 2.0 * fs * a0 + a1a;
    ShelfFilter {
        b: [(2.0 * fs * b0 + b1a) / denom, (-2.0 * fs * b0 + b1a) / denom],
        a1: (-2.0 * fs * a0 + a1a) / denom,
    }
}

/// `Deemphasis.lfilt_inplace` / `Expander.process`'s weighting-filter call
/// (`HiFiDecode.py:843-860`): in-place first-order Direct-Form-I IIR,
/// `y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1]`, with `(zi_x, zi_y)` carried
/// across calls (block to block).
fn lfilt_inplace(x: &mut [f32], filter: ShelfFilter, zi_x: &mut f64, zi_y: &mut f64) {
    for sample in x.iter_mut() {
        let xi = *sample as f64;
        let yi = filter.b[0] * xi + filter.b[1] * *zi_x - filter.a1 * *zi_y;
        *sample = yi as f32;
        *zi_x = xi;
        *zi_y = yi;
    }
}

/// Three-stage cascaded one-pole DC blocker (`DCBlocker`,
/// `HiFiDecode.py:717-787`). State persists across `process` calls; call
/// `process` once on a throwaway copy of the first real block before the
/// first real call (`PostProcessor.py:280-284`) to avoid a cold-start pop.
pub struct DcBlocker {
    r: f64,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    x3: f64,
    y3: f64,
}

impl DcBlocker {
    pub fn new(sample_rate: f64, cutoff: f64) -> Self {
        const STAGES: f64 = 3.0;
        let scale = 1.0 / (2f64.powf(1.0 / STAGES) - 1.0).sqrt();
        let stage_cutoff = cutoff * scale;
        let r = (-2.0 * PI * stage_cutoff / sample_rate).exp();
        DcBlocker {
            r,
            x1: 0.0,
            y1: 0.0,
            x2: 0.0,
            y2: 0.0,
            x3: 0.0,
            y3: 0.0,
        }
    }

    pub fn process(&mut self, audio: &mut [f32]) {
        for sample in audio.iter_mut() {
            let x = *sample as f64;
            let y1 = x - self.x1 + self.r * self.y1;
            self.x1 = x;
            self.y1 = y1;
            let y2 = y1 - self.x2 + self.r * self.y2;
            self.x2 = y1;
            self.y2 = y2;
            let y3 = y2 - self.x3 + self.r * self.y3;
            self.x3 = y2;
            self.y3 = y3;
            *sample = y3 as f32;
        }
    }
}

/// First-order de-emphasis shelf (`Deemphasis`, `HiFiDecode.py:789-870`).
pub struct Deemphasis {
    filter: ShelfFilter,
    zi_x: f64,
    zi_y: f64,
}

impl Deemphasis {
    pub fn new(audio_rate: f64, low_tau: f64, high_tau: f64) -> Self {
        Deemphasis {
            filter: build_shelf_filter(ShelfDirection::Low, low_tau, high_tau, audio_rate),
            zi_x: 0.0,
            zi_y: 0.0,
        }
    }

    pub fn process(&mut self, audio: &mut [f32]) {
        lfilt_inplace(audio, self.filter, &mut self.zi_x, &mut self.zi_y);
    }
}

/// Envelope detector mode (`--expander_env_detection`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvDetection {
    /// JVC/IEC 60774-2 default.
    Peak,
    /// Panasonic variant.
    Rms,
}

/// Companding expander (`Expander`, `HiFiDecode.py:879-1048`): a
/// diode-capacitor-style envelope detector (peak or RMS) feeding a
/// log-domain VCA. Two constructor parameters Python accepts
/// (`weighting_low_pass`/`weighting_low_pass_transition`) are dropped here:
/// `Expander.__init__` takes them but its `process`/`expand` methods never
/// read them (only `get_response`, a debug-plotting method, references
/// `self.lowpass_iirb`/`self.lowpass_iira` attributes that `__init__`
/// never actually sets — dead code in the Python source).
pub struct Expander {
    gain_db: f64,
    ratio: f64,
    atk_coeff: f64,
    rel_coeff: f64,
    hold_samples: i64,
    hold_state: i64,
    env_lin: f64,
    use_rms: bool,
    weighting: ShelfFilter,
    zi_x: f64,
    zi_y: f64,
}

impl Expander {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        audio_rate: f64,
        gain_db: f64,
        ratio: f64,
        env_detection: EnvDetection,
        attack_tau: f64,
        hold_tau: f64,
        release_tau: f64,
        weighting_low_tau: f64,
        weighting_high_tau: f64,
    ) -> Self {
        let (env_lin, use_rms) = match env_detection {
            EnvDetection::Rms => (1e-12, true),
            EnvDetection::Peak => (0.0, false),
        };
        Expander {
            gain_db,
            ratio,
            atk_coeff: (-1.0 / (attack_tau * audio_rate)).exp(),
            rel_coeff: (-1.0 / (release_tau * audio_rate)).exp(),
            hold_samples: (hold_tau * audio_rate).round() as i64,
            hold_state: 0,
            env_lin,
            use_rms,
            weighting: build_shelf_filter(ShelfDirection::High, weighting_low_tau, weighting_high_tau, audio_rate),
            zi_x: 0.0,
            zi_y: 0.0,
        }
    }

    /// `pre_in` is the sidechain/envelope source (high-pass weighted in
    /// place, then overwritten with the running envelope value — matching
    /// Python's in-place `side_chain` reuse); `audio_out` is gain-modulated
    /// in place.
    pub fn process(&mut self, pre_in: &mut [f32], audio_out: &mut [f32]) {
        assert_eq!(pre_in.len(), audio_out.len());
        lfilt_inplace(pre_in, self.weighting, &mut self.zi_x, &mut self.zi_y);
        self.expand(audio_out, pre_in);
    }

    fn expand(&mut self, audio: &mut [f32], side_chain: &mut [f32]) {
        let epsilon = f64::EPSILON;
        let one_minus_atk = 1.0 - self.atk_coeff;
        let ratio_minus_one = self.ratio - 1.0;
        let mut env_lin = self.env_lin;
        let mut hold_state = self.hold_state;

        if self.use_rms {
            for sc in side_chain.iter_mut() {
                let sc_sq = (*sc as f64) * (*sc as f64);
                if sc_sq > env_lin {
                    env_lin = self.atk_coeff * env_lin + one_minus_atk * sc_sq;
                    hold_state = self.hold_samples;
                } else if hold_state > 0 {
                    hold_state -= 1;
                } else {
                    env_lin = self.rel_coeff * env_lin;
                }
                *sc = (env_lin + epsilon).sqrt() as f32;
            }
        } else {
            for sc in side_chain.iter_mut() {
                let sc_abs = (*sc as f64).abs();
                if sc_abs > env_lin {
                    env_lin = self.atk_coeff * env_lin + one_minus_atk * sc_abs;
                    hold_state = self.hold_samples;
                } else if hold_state > 0 {
                    hold_state -= 1;
                } else {
                    env_lin = self.rel_coeff * env_lin;
                }
                *sc = env_lin as f32;
            }
        }

        for (sample, &sc) in audio.iter_mut().zip(side_chain.iter()) {
            let env_db = 20.0 * (sc as f64).max(epsilon).log10();
            let target_gain_db = ratio_minus_one * env_db + self.gain_db;
            *sample = (*sample as f64 * 10f64.powf(target_gain_db * 0.05)) as f32;
        }

        self.env_lin = env_lin;
        self.hold_state = hold_state;
    }
}

/// Tunable taus/gain shared by `Deemphasis`/`Expander` construction
/// (the CLI's `--expander_*`/`--deemphasis_*`/`--nr_deemphasis_*` group).
#[derive(Clone, Copy, Debug)]
pub struct PostProcessParams {
    pub deemphasis_low_tau: f64,
    pub deemphasis_high_tau: f64,
    pub nr_deemphasis_low_tau: f64,
    pub nr_deemphasis_high_tau: f64,
    pub expander_gain: f64,
    pub expander_ratio: f64,
    pub expander_env_detection: EnvDetection,
    pub expander_attack_tau: f64,
    pub expander_hold_tau: f64,
    pub expander_release_tau: f64,
    pub expander_weighting_low_tau: f64,
    pub expander_weighting_high_tau: f64,
}

/// VHS post-processing chain (`expander_vhs_worker`,
/// `PostProcessor.py:332-420`): both de-emphasis stages run before the
/// expander, one on the sidechain (`pre`) and one on the audio (`post`);
/// then a *second*, NR-specific de-emphasis runs on `post` alone (IEC
/// 60774-2 fig. 2/4/5).
pub struct VhsPostProcess {
    deemphasis_pre_1: Deemphasis,
    deemphasis_pre_2: Deemphasis,
    nr_deemphasis: Deemphasis,
    expander: Expander,
    enable_deemphasis: bool,
    enable_expander: bool,
}

impl VhsPostProcess {
    pub fn new(audio_rate: f64, params: PostProcessParams, enable_deemphasis: bool, enable_expander: bool) -> Self {
        VhsPostProcess {
            deemphasis_pre_1: Deemphasis::new(audio_rate, params.deemphasis_low_tau, params.deemphasis_high_tau),
            deemphasis_pre_2: Deemphasis::new(audio_rate, params.deemphasis_low_tau, params.deemphasis_high_tau),
            nr_deemphasis: Deemphasis::new(audio_rate, params.nr_deemphasis_low_tau, params.nr_deemphasis_high_tau),
            expander: Expander::new(
                audio_rate,
                params.expander_gain,
                params.expander_ratio,
                params.expander_env_detection,
                params.expander_attack_tau,
                params.expander_hold_tau,
                params.expander_release_tau,
                params.expander_weighting_low_tau,
                params.expander_weighting_high_tau,
            ),
            enable_deemphasis,
            enable_expander,
        }
    }

    /// `pre`: dc-blocked demod audio (sidechain source). `post`: the audio
    /// to shape (a copy of `pre`, or the spectral-NR output — spectral NR
    /// is out of scope for this port, so callers should pass a copy of
    /// `pre`). `prime_len`, if given, primes the expander's envelope state
    /// on a throwaway copy of the first `prime_len` samples before the
    /// real pass, matching Python's block-0 cold-start handling.
    ///
    /// `prime_len` must be the size of *one nominal block* (e.g.
    /// `BlockLayout::block_audio_final_size`), not the whole signal:
    /// Python primes using exactly one block's worth of data
    /// (`expander_vhs_worker` runs per block, and only block 0 primes).
    /// Priming over the *whole* concatenated stream — an earlier version
    /// of this port did exactly that — pushes the expander's `env_lin`/
    /// `hold_state` through an entire extra unwanted pass before the real
    /// one, which was diagnosed against a real capture: it produced a
    /// growing amplitude error (correct for the first ~500ms, drifting to
    /// ~2.6x too quiet after) even though the *shape* still correlated
    /// above 0.99 with the Python reference.
    pub fn process(&mut self, pre: &mut [f32], post: &mut [f32], prime_len: Option<usize>) {
        if self.enable_deemphasis {
            self.deemphasis_pre_1.process(pre);
            self.deemphasis_pre_2.process(post);
            self.nr_deemphasis.process(post);
        }
        if self.enable_expander {
            if let Some(len) = prime_len {
                let len = len.min(pre.len()).min(post.len());
                let mut pre_copy = pre[..len].to_vec();
                let mut post_copy = post[..len].to_vec();
                self.expander.process(&mut pre_copy, &mut post_copy);
            }
            self.expander.process(pre, post);
        }
    }
}

/// Video8/Hi8 post-processing chain (`expander_8mm_worker`,
/// `PostProcessor.py:422-502`, IEC 60843-1 fig. 34): a single de-emphasis
/// before the expander, then a second ("NR") de-emphasis *after* it —
/// interleaved with the expander rather than both stages running first, as
/// VHS does. Also note block-0 priming here uses the *real* `pre` buffer
/// (not a copy, unlike the VHS chain) — that's what `PostProcessor.py:494`
/// does, so `pre`'s sidechain weighting filter runs twice on block 0. That
/// asymmetry is preserved deliberately, not a port bug.
pub struct EightMmPostProcess {
    deemphasis_2: Deemphasis,
    deemphasis_1: Deemphasis,
    expander: Expander,
    enable_deemphasis: bool,
    enable_expander: bool,
}

impl EightMmPostProcess {
    pub fn new(audio_rate: f64, params: PostProcessParams, enable_deemphasis: bool, enable_expander: bool) -> Self {
        EightMmPostProcess {
            deemphasis_2: Deemphasis::new(audio_rate, params.deemphasis_low_tau, params.deemphasis_high_tau),
            deemphasis_1: Deemphasis::new(audio_rate, params.nr_deemphasis_low_tau, params.nr_deemphasis_high_tau),
            expander: Expander::new(
                audio_rate,
                params.expander_gain,
                params.expander_ratio,
                params.expander_env_detection,
                params.expander_attack_tau,
                params.expander_hold_tau,
                params.expander_release_tau,
                params.expander_weighting_low_tau,
                params.expander_weighting_high_tau,
            ),
            enable_deemphasis,
            enable_expander,
        }
    }

    /// See `VhsPostProcess::process`'s doc comment for what `prime_len`
    /// must be and why priming over more than one block's worth of
    /// samples is wrong, not just wasteful.
    pub fn process(&mut self, pre: &mut [f32], post: &mut [f32], prime_len: Option<usize>) {
        if self.enable_deemphasis {
            self.deemphasis_2.process(post);
        }
        if self.enable_expander {
            if let Some(len) = prime_len {
                let len = len.min(pre.len()).min(post.len());
                let mut post_copy = post[..len].to_vec();
                // Deliberately the real `pre`'s prefix, not a copy: see
                // the struct doc comment. Scoped to `len` (one block) so
                // only that prefix's envelope state gets corrupted before
                // the real pass, matching Python's block-0 scope exactly.
                self.expander.process(&mut pre[..len], &mut post_copy);
            }
            self.expander.process(pre, post);
        }
        if self.enable_deemphasis {
            self.deemphasis_1.process(post);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `build_shelf_filter`'s bilinear-transform closed form, checked
    /// against `scipy.signal.bilinear` for the three shelf shapes
    /// hifi-decode actually uses (VHS deemphasis, VHS NR deemphasis, 8mm
    /// deemphasis) — see the port session notes for the exact Python
    /// invocation used to derive these expected values.
    #[test]
    fn shelf_filter_matches_scipy_bilinear() {
        let cases: [(ShelfDirection, f64, f64, f64, [f64; 2], f64); 3] = [
            (ShelfDirection::Low, 56e-6, 20e-6, 48_000.0, [0.45796737766624845, -0.14429109159347558], -0.6863237139272271),
            (ShelfDirection::High, 24e-6, 240e-6, 192_000.0, [1.0966079862601976, -0.8819235723486475], -0.978531558608845),
            (ShelfDirection::Low, 75e-6, 27e-6, 48_000.0, [0.43804878048780493, -0.19414634146341467], -0.7560975609756098),
        ];
        for (direction, tau1, tau2, fs, expected_b, expected_a1) in cases {
            let filter = build_shelf_filter(direction, tau1, tau2, fs);
            assert!((filter.b[0] - expected_b[0]).abs() < 1e-9, "{:?}", filter.b);
            assert!((filter.b[1] - expected_b[1]).abs() < 1e-9, "{:?}", filter.b);
            assert!((filter.a1 - expected_a1).abs() < 1e-9, "{}", filter.a1);
        }
    }

    #[test]
    fn dc_blocker_removes_a_constant_offset() {
        let mut blocker = DcBlocker::new(48_000.0, 1.0);
        let mut audio = vec![0.5f32; 48_000]; // 1s of DC offset at 1Hz cutoff
        blocker.process(&mut audio);
        // Should have decayed close to zero by the end of a full second.
        assert!(audio[47_999].abs() < 0.05, "{}", audio[47_999]);
    }

    #[test]
    fn expander_stays_bounded_on_a_tone() {
        let params = PostProcessParams {
            deemphasis_low_tau: 56e-6,
            deemphasis_high_tau: 20e-6,
            nr_deemphasis_low_tau: 240e-6,
            nr_deemphasis_high_tau: 56e-6,
            expander_gain: 30.0,
            expander_ratio: 2.0,
            expander_env_detection: EnvDetection::Peak,
            expander_attack_tau: 6.5e-3,
            expander_hold_tau: 0.0,
            expander_release_tau: 70e-3,
            expander_weighting_low_tau: 240e-6,
            expander_weighting_high_tau: 24e-6,
        };
        let mut chain = VhsPostProcess::new(48_000.0, params, true, true);
        let n = 4800;
        let mut pre: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 1000.0 * i as f64 / 48_000.0).sin() as f32 * 0.5)
            .collect();
        let mut post = pre.clone();
        chain.process(&mut pre, &mut post, Some(n));
        assert!(post.iter().all(|v| v.is_finite() && v.abs() < 100.0));
    }
}
