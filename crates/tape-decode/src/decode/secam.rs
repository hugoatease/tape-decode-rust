//! SECAM method 1 (standard quarter count-down) chroma restoration.
//!
//! Tapes recorded to IEC 60774-1 6.4.1 / annex E figure E1 carry the studio
//! SECAM chroma block band-passed around 4.32 MHz and counted down by 4, so the
//! rest carriers on tape are foB/4 and foR/4 with the FM deviations divided by
//! 4 as well. Restoration is a x4 phase multiplication of the colour-under
//! analytic signal rather than a heterodyne mix: carrier and deviation scale
//! back up together, so tape timebase error self-corrects and there is no
//! conversion LO to servo (unlike ME-SECAM).
//!
//! Ported from the Python implementation in vhs-decode (`vhsdecode/chroma.py`,
//! commit b13ef2a2).

use super::*;

use std::f64::consts::{PI, TAU};

/// Subcarrier rest frequencies and HF ("cloche"/bell) pre-emphasis constants
/// from ITU-R BT.470-6 table 2 / BT.1700: the subcarrier amplitude follows
/// `G = M0 * |1 + j16F| / |1 + j1.26F|` with `F = f/f0 - f0/f`.
const SECAM_FOR: f64 = 4_406_250.0;
const SECAM_FOB: f64 = 4_250_000.0;
const SECAM_BELL_F0: f64 = 4_286_000.0;

/// Minimum fraction of lines that must match the fitted alternation before the
/// fit is allowed to teach the parity flywheel.
const SECAM_IDENT_MIN_CONFIDENCE: f64 = 0.7;

/// Legal carrier excursion (BT.470). The bell gain lookup is clamped to this so
/// noise and carrier switch transients don't get boosted by the bell skirts.
const SECAM_FREQ_MIN: f64 = 3.9e6;
const SECAM_FREQ_MAX: f64 = 4.756e6;

/// Lines below this carry no usable chroma (vertical interval / head switch).
const STARTING_LINE: usize = 16;

/// Relative SECAM subcarrier HF pre-emphasis (bell) gain at the given
/// instantaneous frequency, normalized to 1.0 at f0 (BT.470-6).
fn secam_bell_gain(freq_hz: f64) -> f64 {
    let f = freq_hz / SECAM_BELL_F0;
    let bell_f = f - 1.0 / f;
    let num = 1.0 + (16.0 * bell_f) * (16.0 * bell_f);
    let den = 1.0 + (1.26 * bell_f) * (1.26 * bell_f);
    (num / den).sqrt()
}

fn median_of(values: &[f32]) -> f32 {
    let mut scratch = values.to_vec();
    median_from_values(&mut scratch)
}

fn median_of_f64(values: &[f64]) -> f64 {
    let mut scratch = values.to_vec();
    median_from_values(&mut scratch)
}

/// Raised-cosine ramp of `len` points rising from 0 (exclusive of 1).
fn raised_cosine(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| 0.5 - 0.5 * (PI * i as f64 / len as f64).cos())
        .collect()
}

/// Wrap an angle into (-pi, pi].
fn wrap_pi(angle: f64) -> f64 {
    let wrapped = (angle + PI).rem_euclid(TAU) - PI;
    if wrapped == -PI {
        PI
    } else {
        wrapped
    }
}

/// The colour-under signal decomposed into per-sample phase increments plus an
/// amplitude envelope - everything the restoration needs, and the only form in
/// which the signal is edited.
///
/// Splitting the analysis out from the synthesis is what lets blanking
/// regeneration work on the phase increments (see `patch_blanking_increments`)
/// instead of splicing a synthesized waveform into the colour-under signal:
/// the restored output is `bell * env * cos(mult * cumsum(increments))`, so
/// rewriting a run of increments is a frequency edit whose phase continuity is
/// automatic. Only this struct is expensive to build (band-pass + Hilbert over
/// the whole field); resynthesis after an edit is O(n) arithmetic.
struct ChromaAnalysis {
    /// Per-sample wrapped phase increments of the colour-under carrier, rad.
    increments: Vec<f64>,
    /// Band-pass envelope of the colour-under carrier.
    envelope: Vec<f32>,
    /// Absolute carrier phase at sample 0, rad.
    phase0: f64,
}

/// Band-pass the colour-under chroma and decompose it into phase increments and
/// an amplitude envelope.
fn analyze_under_carrier(
    chroma: &[f32],
    forward_fft: &dyn Fft<f32>,
    inverse_fft: &dyn Fft<f32>,
    under_bpf: &[Sos<f32>],
) -> ChromaAnalysis {
    let filtered = sosfiltfilt_f32(under_bpf, chroma);
    let len = filtered.len();

    // Analytic signal over the whole field so short-window edge effects don't
    // bias the phase.
    let analytic = hilbert_f32(&filtered, forward_fft, inverse_fft);

    let envelope: Vec<f32> = analytic.iter().map(|z| z.norm()).collect();

    // Per-sample wrapped phase increments, taken from the product with the
    // conjugate of the previous sample rather than from a difference of two
    // atan2 results: the carrier advances ~0.38 rad per sample, comfortably
    // inside (-pi, pi], and this keeps the unwrap exact without accumulating an
    // f32 phase ramp whose resolution would swamp the deviation.
    let mut increments = vec![0.0f64; len];
    for i in 1..len {
        let product = analytic[i] * analytic[i - 1].conj();
        increments[i] = (product.im as f64).atan2(product.re as f64);
    }
    if len > 1 {
        increments[0] = increments[1];
    }

    let phase0 = if len > 0 {
        (analytic[0].im as f64).atan2(analytic[0].re as f64)
    } else {
        0.0
    };

    ChromaAnalysis {
        increments,
        envelope,
        phase0,
    }
}

/// Restored-domain instantaneous frequency in Hz, from the colour-under phase
/// increments.
///
/// Central difference plus a short moving average keeps sample-level phase noise
/// from ending up as amplitude noise once the bell is applied; the bell curve
/// itself is smooth so this doesn't blunt legitimate deviation. Also the signal
/// the line identification fit reads.
fn restored_inst_freq(increments: &[f64], samp_rate: f64, carrier_mult: f64) -> Vec<f32> {
    let len = increments.len();
    let freq_scale = carrier_mult * samp_rate / TAU;
    let mut raw_freq = vec![0.0f64; len];
    for i in 0..len {
        let central = if i == 0 {
            increments[(1).min(len - 1)]
        } else if i + 1 >= len {
            increments[i]
        } else {
            (increments[i] + increments[i + 1]) / 2.0
        };
        raw_freq[i] = central * freq_scale;
    }

    const SMOOTH_LEN: usize = 9;
    let half_smooth = SMOOTH_LEN / 2;
    let mut inst_freq = vec![0.0f32; len];
    // Running sum over the centered window, with partial windows at the field
    // edges (numpy's 'same' convolution zero-pads there, which would pull the
    // first and last few samples of the vertical interval toward 0 Hz).
    let mut window_sum = 0.0f64;
    let mut window_len = 0usize;
    for i in 0..len {
        if i == 0 {
            for &value in raw_freq.iter().take(half_smooth + 1) {
                window_sum += value;
                window_len += 1;
            }
        } else {
            if let Some(&entering) = raw_freq.get(i + half_smooth) {
                window_sum += entering;
                window_len += 1;
            }
            if i > half_smooth {
                window_sum -= raw_freq[i - half_smooth - 1];
                window_len -= 1;
            }
        }
        inst_freq[i] =
            (window_sum / window_len as f64).clamp(SECAM_FREQ_MIN, SECAM_FREQ_MAX) as f32;
    }

    inst_freq
}

/// Synthesize the studio SECAM chroma block by multiplying the colour-under
/// carrier phase back up.
///
/// The divider outputs a constant-amplitude signal, so the BT.470 bell
/// pre-emphasis is regenerated here from the restored instantaneous frequency to
/// put the amplitude envelope back on spec for downstream SECAM decoders.
fn synthesize_restored(
    analysis: &ChromaAnalysis,
    inst_freq: &[f32],
    carrier_mult: f64,
    rest_amplitude: f64,
) -> Vec<f32> {
    let len = analysis.increments.len();

    // Scale by the normalized under-carrier envelope (capped just above
    // nominal). Where the carrier is healthy this is ~unity, so the average
    // amplitude stays on the bell curve; where it dips or disappears (dropouts,
    // FM clicks, no colour) the dip is passed through to the output instead of
    // being hard-limited away. Downstream SECAM decoders key their click/dropout
    // concealment off exactly those envelope collapses, so preserving them
    // matters more than emulating the constant-amplitude divider chain of a real
    // deck - and it doubles as the squelch that keeps carrier-free noise from
    // becoming full-scale splatter.
    let env_med = median_of(&analysis.envelope) as f64;

    let mut restored = vec![0.0f32; len];
    // The phase is kept wrapped: cos(carrier_mult * phase) is unchanged by whole
    // turns as long as carrier_mult is an integer, and a bounded argument keeps
    // the cosine's range reduction exact over a whole field.
    let mut phase = analysis.phase0;
    for i in 0..len {
        if i > 0 {
            phase = wrap_pi(phase + analysis.increments[i]);
        }
        let limited = if env_med > 0.0 {
            (analysis.envelope[i] as f64 / env_med).min(1.25)
        } else {
            0.0
        };
        let gain = secam_bell_gain(inst_freq[i] as f64);
        restored[i] = (rest_amplitude * gain * limited * (carrier_mult * phase).cos()) as f32;
    }

    restored
}

/// Fit the field's D'R/D'B line alternation from the active-region median
/// restored frequency of each line: D'R lines sit in the top half of the chroma
/// block, D'B in the bottom.
///
/// The sequence alternates strictly (BT.470), so fit the better of the two
/// possible parities; per-line deviation medians can land on the wrong side on
/// heavily saturated lines, the majority never does.
///
/// Returns `(dr_on_even, confidence)` where confidence is the fraction of lines
/// whose measured identity matches the fitted alternation, or `None` if there
/// are too few lines to fit.
fn fit_secam_line_alternation(
    inst_freq: &[f32],
    linesout: usize,
    outwidth: usize,
    first_line: usize,
    porch_end_px: usize,
) -> Option<(bool, f64)> {
    let n_lines = linesout.checked_sub(first_line)?;
    if n_lines < 32 {
        return None;
    }
    let active_start = porch_end_px + 30;
    let active_end = outwidth.checked_sub(40)?;
    if active_start >= active_end || linesout * outwidth > inst_freq.len() {
        return None;
    }

    let threshold = ((SECAM_FOR + SECAM_FOB) / 2.0) as f32;
    let mut scratch = vec![0.0f32; active_end - active_start];
    let mut even_is_dr = 0usize;
    for line_index in first_line..linesout {
        let base = line_index * outwidth;
        scratch.copy_from_slice(&inst_freq[base + active_start..base + active_end]);
        let is_dr = median_from_values(&mut scratch) > threshold;
        if is_dr == line_index.is_multiple_of(2) {
            even_is_dr += 1;
        }
    }

    let confidence = even_is_dr.max(n_lines - even_is_dr) as f64 / n_lines as f64;
    Some((even_is_dr * 2 >= n_lines, confidence))
}

/// Where a field's resolved parity came from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ParitySource {
    Measured,
    Flywheel,
    Unlocked,
}

impl ParitySource {
    fn as_str(self) -> &'static str {
        match self {
            ParitySource::Measured => "measured",
            ParitySource::Flywheel => "flywheel",
            ParitySource::Unlocked => "unlocked",
        }
    }
}

/// Carry the fitted D'R/D'B alternation across fields.
///
/// Each TBC field is 312.5 line periods, so the alternation phase of consecutive
/// fields walks a strict 4-field cycle:
///
/// ```text
/// dr_on_even(n) = base ^ (((n + 1) >> 1) & 1)
/// ```
///
/// A single bit therefore locks the parity of every field in the recording.
/// Fields whose own alternation fit is confident teach `base`; fields whose
/// content can't be fitted (near-neutral pictures, noisy tape) inherit the
/// predicted parity instead of losing their blanking regeneration.
///
/// The lock requires `MIN_LOCK` agreeing confident fields, expires after
/// `MAX_AGE` fields without confirmation, and a confident contradiction resets
/// it - a dropped field upstream shifts the cycle phase, and re-learning is
/// cheaper than trusting a stale lock.
struct SecamParityFlywheel {
    index: i64,
    last_readloc: Option<u64>,
    base: Option<bool>,
    agree: usize,
    last_confirm: Option<i64>,
}

impl SecamParityFlywheel {
    const MIN_LOCK: usize = 4;
    const MAX_AGE: i64 = 32;

    fn new() -> Self {
        Self {
            index: -1,
            last_readloc: None,
            base: None,
            agree: 0,
            last_confirm: None,
        }
    }

    fn flip(index: i64) -> bool {
        (((index + 1) >> 1) & 1) == 1
    }

    /// Advance to the field identified by `readloc` and resolve its parity.
    fn resolve(&mut self, readloc: u64, fit: Option<(bool, f64)>) -> (Option<bool>, ParitySource) {
        if self.last_readloc != Some(readloc) {
            self.last_readloc = Some(readloc);
            self.index += 1;
        }
        let n = self.index;
        let flip = Self::flip(n);

        if let Some((dr_on_even, confidence)) = fit {
            if confidence >= SECAM_IDENT_MIN_CONFIDENCE {
                let base = dr_on_even ^ flip;
                if self.base == Some(base) {
                    self.agree += 1;
                } else {
                    self.base = Some(base);
                    self.agree = 1;
                }
                self.last_confirm = Some(n);
                return (Some(dr_on_even), ParitySource::Measured);
            }
        }

        if let (Some(base), Some(last_confirm)) = (self.base, self.last_confirm) {
            if self.agree >= Self::MIN_LOCK && n - last_confirm <= Self::MAX_AGE {
                return (Some(base ^ flip), ParitySource::Flywheel);
            }
        }
        (None, ParitySource::Unlocked)
    }
}

/// Geometry of the regenerated blanking interval.
mod blanking {
    /// Crossfade from the measured carrier to the outgoing rest frequency.
    pub(super) const RAMP_LEN: usize = 20;
    /// Median window for the local carrier and amplitude estimates.
    pub(super) const MEAS_LEN: usize = 32;
    /// Rest-to-rest frequency step position within the NEXT line (px from its
    /// start): over the sync tip.
    pub(super) const STEP_START_PX: usize = 8;
    pub(super) const STEP_END_PX: usize = 40;
    /// How far the rewrite extends past the blanking interval at BOTH ends, to
    /// cover the zero-phase under-carrier band-pass smearing the record chain's
    /// transient out into active video. This is the one thing the single-pass
    /// rewrite has to pay for that a second band-pass over a cleaned signal
    /// would have handled for free.
    ///
    /// Both ends need it, and the outgoing one is not optional: the 0.85 us
    /// `blank_start_px` already backs off only moves where the rewrite starts,
    /// while the smear runs backwards from where the transient actually sets
    /// in. Guarding only the incoming end leaves the local carrier estimate
    /// reading smeared samples, which starts the front-porch ramp ~120 kHz off
    /// and shows up as a colour band down the right edge - measured at -50 kHz
    /// of mean bias on LAILA_9 before this was symmetric, against -7 kHz for
    /// the two-pass implementation.
    ///
    /// Measured on a synthetic transient (`smear_guard_covers_the_bandpass_reach`):
    /// the restored frequency error decays from ~122 kHz at the edge of blanking
    /// to the band-pass ripple floor (~8 kHz) by ~17 samples, passing ~12 kHz at
    /// 0.85 us.
    ///
    /// Widening this trades thin neutral strips down both edges of the picture
    /// against tinted ones.
    pub(super) const SMEAR_GUARD_US: f64 = 0.85;

    /// `SMEAR_GUARD_US` in samples at the TBC output rate.
    pub(super) fn smear_guard_px(samp_rate: f64) -> usize {
        (SMEAR_GUARD_US * 1e-6 * samp_rate) as usize
    }
}

/// Rewrite each line's horizontal blanking interval - front porch, sync and back
/// porch in one continuous run - as an undeviated colour-under rest carrier, by
/// replacing the measured phase increments over that run.
///
/// On method 1 tapes the whole blanking interval carries the record chain's
/// divide-by-4 counter settling transient (blanking edges / SECAM subcarrier
/// phase reversals upset the divider), not the undeviated reference BT.470
/// promises. Two things go wrong if it is left in place:
///
/// - the zero-phase filters in this chain (the under-carrier band-pass, and
///   `chroma_filter_final` later) and the linear-phase cloche filters in
///   downstream SECAM decoders smear the end-of-line transient BACKWARDS into
///   the last ~2 us of active video, which demodulates as a magenta band down
///   the right edge of the picture (D'R deviates negative, D'B positive, so the
///   transient reads red on D'R lines and blue on D'B lines);
/// - decoders calibrate their discriminator zeros and line identification from
///   the back porch, and transient energy ringing into that window biases the
///   zeros, which shows up as a full-field colour cast.
///
/// Because the restored output is the cosine of the running sum of these
/// increments, rewriting a run of them is purely a frequency edit: the phase is
/// continuous by construction at both boundaries, with nothing to measure and
/// nothing to close. The synthesized profile ramps from the local measured
/// carrier to the outgoing line's rest frequency across the front porch, steps
/// to the incoming line's rest over the sync tip, and holds it through the back
/// porch, where downstream decoders take their discriminator-zero reference.
/// The envelope is ramped between the two neighbouring levels over the same run
/// so the amplitude carries no step either.
///
/// The rewrite shifts the absolute carrier phase of everything after it, which
/// is immaterial: SECAM is FM, and only the instantaneous frequency survives
/// into the picture.
///
/// Each line's run reads only active video either side of itself and writes only
/// its own blanking interval, so the runs neither overlap nor observe each
/// other: the loop is order-independent.
#[allow(clippy::too_many_arguments)]
fn patch_blanking_increments(
    analysis: &mut ChromaAnalysis,
    samp_rate: f64,
    linesout: usize,
    outwidth: usize,
    blank_start_px: usize,
    porch_end_px: usize,
    first_line: usize,
    dr_on_even: bool,
    carrier_mult: f64,
) {
    use blanking::*;

    let len = analysis.increments.len();
    let guard = smear_guard_px(samp_rate);
    let ramp = raised_cosine(RAMP_LEN);
    let step_len = STEP_END_PX - STEP_START_PX;
    let mid_step = raised_cosine(step_len);
    if blank_start_px >= outwidth || blank_start_px < guard {
        return;
    }
    // Offset of the next line's start within the rewritten run.
    let next_line_p = outwidth - blank_start_px + guard;

    for linenumber in first_line..linesout.saturating_sub(1) {
        let line_is_dr = linenumber.is_multiple_of(2) == dr_on_even;
        let f_out_rest = if line_is_dr { SECAM_FOR } else { SECAM_FOB };
        let f_in_rest = if line_is_dr { SECAM_FOB } else { SECAM_FOR };
        // Colour-under phase advance per sample at each rest carrier.
        let inc_out_rest = TAU * f_out_rest / (carrier_mult * samp_rate);
        let inc_in_rest = TAU * f_in_rest / (carrier_mult * samp_rate);

        // The run is widened by the smear guard at both ends, so the local
        // estimates below read active video the transient has not reached.
        let start = linenumber * outwidth + blank_start_px - guard;
        let end = (linenumber + 1) * outwidth + porch_end_px + guard;
        if end <= start {
            continue;
        }
        let span = end - start;
        // Room for both median windows outside the run, and for the profile
        // inside it.
        if start < 2 * MEAS_LEN || end + 2 * MEAS_LEN > len {
            continue;
        }
        if span < RAMP_LEN + step_len + 2 * MEAS_LEN || next_line_p >= span {
            continue;
        }

        // Local carrier and amplitude, from medians just outside the run.
        // Narrow enough to track the picture at the end of the line, wide
        // enough not to be moved by per-sample phase noise.
        let inc_out = median_of_f64(&analysis.increments[start - MEAS_LEN..start]);
        let amp_out = median_of(&analysis.envelope[start - 2 * MEAS_LEN..start - MEAS_LEN]) as f64;
        let amp_in = median_of(&analysis.envelope[end + MEAS_LEN..end + 2 * MEAS_LEN]) as f64;

        let step0 = (next_line_p + STEP_START_PX)
            .max(RAMP_LEN)
            .min(span - step_len);
        let step1 = step0 + step_len;

        // Frequency profile: measured outgoing -> outgoing rest (over the front
        // porch) -> incoming rest (step over the sync tip) -> incoming rest held
        // through the back porch. Raised-cosine throughout, so the restored
        // instantaneous frequency has no step anywhere.
        //
        // No crossfade back on the incoming side: the run ends inside active
        // video, where the measured increments are already the picture's own,
        // and the 9-tap smoothing in `restored_inst_freq` blends the handover.
        let profile = &mut analysis.increments[start..end];
        for (i, &value) in ramp.iter().enumerate() {
            profile[i] = inc_out + (inc_out_rest - inc_out) * value;
        }
        profile[RAMP_LEN..step0].fill(inc_out_rest);
        for (i, &value) in mid_step.iter().enumerate() {
            profile[step0 + i] = inc_out_rest + (inc_in_rest - inc_out_rest) * value;
        }
        profile[step1..].fill(inc_in_rest);

        // Amplitude ramp across the run: an envelope step at either boundary
        // would read as a click downstream. Measured a little away from the
        // boundaries, where the band-pass smear of the transient still inflates
        // the envelope.
        let amp_step = if span > 1 {
            (amp_in - amp_out) / (span - 1) as f64
        } else {
            0.0
        };
        for (i, sample) in analysis.envelope[start..end].iter_mut().enumerate() {
            *sample = (amp_out + amp_step * i as f64) as f32;
        }
    }
}

/// Per-decode SECAM state carried across fields.
pub(crate) struct SecamState {
    flywheel: SecamParityFlywheel,
}

impl SecamState {
    pub(crate) fn new() -> Self {
        Self {
            flywheel: SecamParityFlywheel::new(),
        }
    }
}

/// SECAM method 1 chroma restoration: x4 phase multiplication instead of a
/// heterodyne mix, plus BT.470 bell amplitude regeneration.
pub(crate) fn process_chroma_secam_method1(
    field: &DecodedField,
    spec: &DecoderSpec,
    state: &mut SecamState,
    chroma: &[f32],
    burstarea: (isize, isize),
    carrier_mult: f64,
) -> Result<Vec<u16>> {
    let under_bpf = spec
        .chroma_filter_secam_under
        .as_ref()
        .context("missing SECAM under-carrier band-pass filter")?;
    let outwidth = field.outlinelen;
    let linesout = field.outlinecount;
    if linesout <= STARTING_LINE || chroma.len() < linesout * outwidth {
        bail!(
            "SECAM field too small to restore: {linesout} lines of {outwidth} against {} samples",
            chroma.len()
        );
    }

    // The restoration runs on the TBC output timebase (outlinelen samples per
    // line period), not on the nominal 4fsc rate.
    let samp_rate = spec.sys_outlinelen as f64 / (spec.sys_line_period * 1e-6);

    // Peak amplitude such that the undeviated carrier lands near the same porch
    // RMS level the other formats' chroma AGC normalizes to.
    let burst_abs_ref = spec.sys_burst_abs_ref.context("missing burst_abs_ref")? as f64;
    let rest_amplitude = burst_abs_ref * std::f64::consts::SQRT_2;

    // The one expensive step: band-pass plus Hilbert over the whole field. The
    // blanking rewrite below edits its output in place, so this runs once even
    // though the restoration is resynthesized afterwards.
    let mut analysis = analyze_under_carrier(
        chroma,
        spec.fft_field_forward_f32.as_ref(),
        spec.fft_field_inverse_f32.as_ref(),
        under_bpf,
    );
    let mut inst_freq = restored_inst_freq(&analysis.increments, samp_rate, carrier_mult);

    // This port has no per-line colour-killer signal, so the first usable line is
    // simply the end of the vertical interval.
    let first_line = STARTING_LINE;
    let porch_end_px = (spec.sys_active_video_us[0] * spec.sys_outfreq) as usize;

    // Give downstream decoders the undeviated blanking-interval reference the
    // standard promises them; what comes off tape there is the record divider's
    // settling transient (see `patch_blanking_increments`).
    let fit = fit_secam_line_alternation(&inst_freq, linesout, outwidth, first_line, porch_end_px);
    let (dr_on_even, parity_source) = state.flywheel.resolve(field.readloc, fit);

    if let Some(dr_on_even) = dr_on_even {
        // The record chain's blanking-edge transient sets in slightly before the
        // nominal end of active video (the source's own blanking edge lands
        // inside the TBC active window), so the rewrite starts a little early.
        let blank_start_px = ((spec.sys_active_video_us[1] - 0.85) * spec.sys_outfreq) as usize;
        patch_blanking_increments(
            &mut analysis,
            samp_rate,
            linesout,
            outwidth,
            blank_start_px,
            porch_end_px,
            first_line,
            dr_on_even,
            carrier_mult,
        );
        inst_freq = restored_inst_freq(&analysis.increments, samp_rate, carrier_mult);
        tracing::debug!(
            "SECAM blanking reference regenerated ({}, fit confidence {})",
            parity_source.as_str(),
            fit.map_or_else(
                || "n/a".to_string(),
                |(_, confidence)| format!("{confidence:.02}")
            )
        );
    } else {
        tracing::debug!(
            "SECAM blanking left as-is (line ident confidence too low, no parity lock)"
        );
    }

    let mut uphet = synthesize_restored(&analysis, &inst_freq, carrier_mult, rest_amplitude);
    uphet.truncate(linesout * outwidth);

    // Block-anchored final band-pass (same band as ME-SECAM).
    uphet = sosfiltfilt_f32(&spec.chroma_filter_final, &uphet);

    // No per-line chroma AGC here: the amplitude envelope was synthesised from
    // the BT.470 bell above, and normalizing every line to its porch level would
    // flatten the intended foR/foB rest amplitude difference. Just blank the
    // vertical interval and log the porch level like `acc` does for the other
    // formats.
    let blanked = (first_line * outwidth).min(uphet.len());
    uphet[..blanked].fill(0.0);

    let (burst_start, burst_end) = burstarea;
    if burst_start >= 0 && burst_end > burst_start && (burst_end as usize) <= outwidth {
        let mut porch_rms_total = 0.0f64;
        for linenumber in STARTING_LINE..linesout {
            let linestart = linenumber * outwidth;
            porch_rms_total +=
                rms(&uphet[linestart + burst_start as usize..linestart + burst_end as usize]);
        }
        tracing::debug!(
            "SECAM chroma porch level: {:.01}",
            porch_rms_total / (linesout - STARTING_LINE) as f64
        );
    }

    Ok(encode_chroma_u16(&uphet))
}

/// Encode restored chroma to the 16-bit output level convention, matching the
/// unity-scale case of `acc` in the heterodyne path: zero signal sits at 32767
/// and out-of-range samples wrap, as they do in the Python decoder's
/// `astype(np.uint16)`.
fn encode_chroma_u16(samples: &[f32]) -> Vec<u16> {
    const SIGNED_SAMPLE_MAX: f32 = 32767.0;
    samples
        .iter()
        .map(|&sample| ((sample + SIGNED_SAMPLE_MAX) as i64) as u16)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustfft::FftPlanner;

    /// TBC output rate for 625-line VHS: outlinelen samples per 64 us line.
    const TEST_SAMP_RATE: f64 = 1135.0 / 64e-6;
    const TEST_OUTWIDTH: usize = 1135;
    const TEST_PORCH_END_PX: usize = 186;
    const TEST_BLANK_START_PX: usize = 1093;
    const CARRIER_MULT: f64 = 4.0;
    /// `blanking::smear_guard_px` at the test rate: 0.85 us -> 15 samples.
    const TEST_GUARD_PX: usize = 15;

    fn under_bandpass() -> Vec<Sos<f32>> {
        let half = TEST_SAMP_RATE / 2.0;
        narrow_sos(
            &butter_sos(3, &[550e3 / half, 1300e3 / half], FilterBandType::Bandpass)
                .expect("under band-pass"),
        )
    }

    fn analyze(chroma: &[f32]) -> ChromaAnalysis {
        let mut planner = FftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(chroma.len());
        let inverse = planner.plan_fft_inverse(chroma.len());
        analyze_under_carrier(
            chroma,
            forward.as_ref(),
            inverse.as_ref(),
            &under_bandpass(),
        )
    }

    /// Run the whole restoration over a buffer, planning FFTs to match its
    /// length.
    fn restore(chroma: &[f32], rest_amplitude: f64) -> (Vec<f32>, Vec<f32>) {
        let analysis = analyze(chroma);
        let inst_freq = restored_inst_freq(&analysis.increments, TEST_SAMP_RATE, CARRIER_MULT);
        let restored = synthesize_restored(&analysis, &inst_freq, CARRIER_MULT, rest_amplitude);
        (inst_freq, restored)
    }

    /// Constant-amplitude colour-under tone at `under_freq`.
    fn under_tone(len: usize, under_freq: f64) -> Vec<f32> {
        (0..len)
            .map(|i| (TAU * under_freq * i as f64 / TEST_SAMP_RATE).cos() as f32)
            .collect()
    }

    #[test]
    fn x4_restores_carrier_and_deviation_together() {
        // A method 1 deck records foR/4 with the deviation divided by 4 as well;
        // playback has to bring both back up by the same factor.
        for deviation in [0.0, 100e3, -100e3, 200e3] {
            let restored_freq = SECAM_FOR + deviation;
            let (inst_freq, _) = restore(&under_tone(16384, restored_freq / CARRIER_MULT), 1.0);
            // Skip the filter transients at both ends.
            let mut middle = inst_freq[4096..12288].to_vec();
            let measured = median_from_values(&mut middle) as f64;
            assert!(
                (measured - restored_freq).abs() < 1e3,
                "deviation {deviation}: restored {measured} Hz, expected {restored_freq} Hz"
            );
        }
    }

    #[test]
    fn x4_restores_the_blue_rest_carrier_too() {
        let (inst_freq, _) = restore(&under_tone(16384, SECAM_FOB / CARRIER_MULT), 1.0);
        let mut middle = inst_freq[4096..12288].to_vec();
        let measured = median_from_values(&mut middle) as f64;
        assert!((measured - SECAM_FOB).abs() < 1e3, "measured {measured} Hz");
        // The two rest carriers must stay 156.25 kHz apart after restoration.
        let (red_freq, _) = restore(&under_tone(16384, SECAM_FOR / CARRIER_MULT), 1.0);
        let mut red_middle = red_freq[4096..12288].to_vec();
        let red_measured = median_from_values(&mut red_middle) as f64;
        assert!(((red_measured - measured) - (SECAM_FOR - SECAM_FOB)).abs() < 1e3);
    }

    #[test]
    fn bell_preemphasis_is_regenerated() {
        // The divider chain outputs constant amplitude; the restored signal has
        // to carry the BT.470 bell envelope again.
        const REST_AMPLITUDE: f64 = 7071.0;
        for restored_freq in [SECAM_FOR, SECAM_FOB, SECAM_FOR + 200e3] {
            let (_, restored) = restore(
                &under_tone(16384, restored_freq / CARRIER_MULT),
                REST_AMPLITUDE,
            );
            let peak = restored[4096..12288]
                .iter()
                .fold(0.0f32, |acc, &value| acc.max(value.abs())) as f64;
            let expected = REST_AMPLITUDE * secam_bell_gain(restored_freq);
            assert!(
                (peak - expected).abs() / expected < 0.02,
                "at {restored_freq} Hz: peak {peak}, expected {expected}"
            );
        }
    }

    /// Colour-under field where active video carries each line's undeviated rest
    /// carrier and the whole blanking interval carries an off-frequency
    /// transient, standing in for the record divider's settling behaviour.
    fn field_with_blanking_transient(linesout: usize, dr_on_even: bool) -> Vec<f32> {
        let mut signal = vec![0.0f32; linesout * TEST_OUTWIDTH];
        let mut phase = 0.0f64;
        for (i, sample) in signal.iter_mut().enumerate() {
            let line = i / TEST_OUTWIDTH;
            let pos = i % TEST_OUTWIDTH;
            let line_is_dr = line.is_multiple_of(2) == dr_on_even;
            let rest = if line_is_dr { SECAM_FOR } else { SECAM_FOB } / CARRIER_MULT;
            let in_active = (TEST_PORCH_END_PX..TEST_BLANK_START_PX).contains(&pos);
            let freq = if in_active { rest } else { rest + 60e3 };
            *sample = phase.cos() as f32;
            phase += TAU * freq / TEST_SAMP_RATE;
        }
        signal
    }

    /// Analyze a synthetic field and rewrite its blanking intervals.
    fn analyze_and_patch(chroma: &[f32], linesout: usize, dr_on_even: bool) -> ChromaAnalysis {
        let mut analysis = analyze(chroma);
        patch_blanking_increments(
            &mut analysis,
            TEST_SAMP_RATE,
            linesout,
            TEST_OUTWIDTH,
            TEST_BLANK_START_PX,
            TEST_PORCH_END_PX,
            1,
            dr_on_even,
            CARRIER_MULT,
        );
        analysis
    }

    #[test]
    fn blanking_patch_restores_the_porch_reference() {
        let linesout = 8usize;
        let dr_on_even = true;
        let chroma = field_with_blanking_transient(linesout, dr_on_even);

        let before = restored_inst_freq(&analyze(&chroma).increments, TEST_SAMP_RATE, CARRIER_MULT);
        let patched = analyze_and_patch(&chroma, linesout, dr_on_even);
        let after = restored_inst_freq(&patched.increments, TEST_SAMP_RATE, CARRIER_MULT);

        // Decoders calibrate their discriminator zeros from roughly 65..5 px
        // before active video. That window must now sit on the incoming line's
        // rest carrier, in the restored domain.
        for line in 3..linesout {
            let line_is_dr = line.is_multiple_of(2) == dr_on_even;
            let rest = if line_is_dr { SECAM_FOR } else { SECAM_FOB };
            let window = line * TEST_OUTWIDTH + TEST_PORCH_END_PX - 65
                ..line * TEST_OUTWIDTH + TEST_PORCH_END_PX - 5;

            let mut before_window = before[window.clone()].to_vec();
            let before_median = median_from_values(&mut before_window) as f64;
            let mut after_window = after[window].to_vec();
            let after_median = median_from_values(&mut after_window) as f64;

            assert!(
                (before_median - rest).abs() > 150e3,
                "line {line}: transient should be present before the rewrite ({before_median} Hz)",
            );
            assert!(
                (after_median - rest).abs() < 20e3,
                "line {line}: porch off rest by {} Hz after the rewrite",
                after_median - rest
            );
        }
    }

    #[test]
    fn blanking_patch_leaves_active_video_alone() {
        let linesout = 8usize;
        let chroma = field_with_blanking_transient(linesout, true);
        let before = analyze(&chroma);
        let after = analyze_and_patch(&chroma, linesout, true);

        // The rewrite spans from `blank_start_px` to `porch_end_px` of the next
        // line, widened by the smear guard at both ends, and must not touch a
        // sample of the picture beyond that - increments or envelope.
        for line in 2..linesout - 1 {
            let base = line * TEST_OUTWIDTH;
            for pos in TEST_PORCH_END_PX + TEST_GUARD_PX..TEST_BLANK_START_PX - TEST_GUARD_PX {
                assert_eq!(
                    before.increments[base + pos],
                    after.increments[base + pos],
                    "line {line} pos {pos}: active video increment was modified"
                );
                assert_eq!(
                    before.envelope[base + pos],
                    after.envelope[base + pos],
                    "line {line} pos {pos}: active video envelope was modified"
                );
            }
        }
    }

    #[test]
    fn smear_guard_covers_the_bandpass_reach() {
        // The single-pass rewrite leaves the zero-phase under-carrier band-pass
        // to smear the tail of the transient forward into active video, so the
        // guard has to outrun it. Measured on the UNPATCHED analysis as the
        // restored-domain error against each line's own rest carrier.
        let linesout = 8usize;
        let dr_on_even = true;
        let chroma = field_with_blanking_transient(linesout, dr_on_even);
        let inst_freq =
            restored_inst_freq(&analyze(&chroma).increments, TEST_SAMP_RATE, CARRIER_MULT);

        let worst_at = |offset: usize| -> f64 {
            (2..linesout - 1)
                .map(|line| {
                    let line_is_dr = line.is_multiple_of(2) == dr_on_even;
                    let rest = if line_is_dr { SECAM_FOR } else { SECAM_FOB };
                    let i = line * TEST_OUTWIDTH + TEST_PORCH_END_PX + offset;
                    (inst_freq[i] as f64 - rest).abs()
                })
                .fold(0.0f64, f64::max)
        };

        // At the nominal end of blanking the smear is still most of the
        // transient; by the guard it has decayed to the band-pass ripple floor.
        assert!(
            worst_at(0) > 100e3,
            "smear at the blanking edge is only {} Hz - the test signal is wrong",
            worst_at(0)
        );
        assert!(
            worst_at(TEST_GUARD_PX) < 12e3,
            "smear is still {} Hz at the guard ({TEST_GUARD_PX} px); widen SMEAR_GUARD_US",
            worst_at(TEST_GUARD_PX)
        );
    }

    #[test]
    fn blanking_patch_introduces_no_frequency_step() {
        // Phase continuity is automatic in the increments domain, so what has to
        // be checked is that the rewrite leaves no step in the restored
        // instantaneous frequency at either boundary or across the profile.
        let linesout = 8usize;
        let chroma = field_with_blanking_transient(linesout, true);
        let patched = analyze_and_patch(&chroma, linesout, true);
        let inst_freq = restored_inst_freq(&patched.increments, TEST_SAMP_RATE, CARRIER_MULT);

        // Scan the region the rewrite actually covers: from line 2 to the end of
        // the last rewritten run. Beyond it lie the final line's own blanking
        // interval (never rewritten - the run needs a next line) and the
        // band-pass edge transient at the end of the buffer, both of which carry
        // the raw transient by design.
        let scan =
            2 * TEST_OUTWIDTH..(linesout - 1) * TEST_OUTWIDTH + TEST_PORCH_END_PX + TEST_GUARD_PX;

        // The steepest intended feature is the rest-to-rest step: 156.25 kHz
        // raised-cosine over 32 samples, so ~7.7 kHz per sample at its centre.
        let max_step = inst_freq[scan]
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(max_step < 12e3, "largest frequency step {max_step} Hz");
    }

    #[test]
    fn blanking_patch_keeps_the_envelope_continuous() {
        let linesout = 8usize;
        let chroma = field_with_blanking_transient(linesout, true);
        let patched = analyze_and_patch(&chroma, linesout, true);

        // A step in the envelope at a rewrite boundary would read as a click
        // downstream. The synthetic field has a flat envelope, so any step here
        // is one the rewrite introduced. Same scan bounds as the frequency
        // check: past the last rewritten run sit the final line's raw blanking
        // and the buffer-edge transient.
        let scan =
            2 * TEST_OUTWIDTH..(linesout - 1) * TEST_OUTWIDTH + TEST_PORCH_END_PX + TEST_GUARD_PX;
        let reference = median_of(&patched.envelope[scan.clone()]) as f64;
        let max_step = patched.envelope[scan]
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .fold(0.0f32, f32::max) as f64;
        assert!(
            max_step < 0.05 * reference,
            "largest envelope step {max_step} against a level of {reference}"
        );
    }

    #[test]
    fn chroma_encode_puts_zero_at_the_u16_midpoint() {
        assert_eq!(encode_chroma_u16(&[0.0]), vec![32767u16]);
        assert_eq!(encode_chroma_u16(&[1.0, -1.0]), vec![32768u16, 32766u16]);
        // Full-scale positive and negative rest carriers stay inside the range.
        assert_eq!(encode_chroma_u16(&[32767.0]), vec![65534u16]);
        assert_eq!(encode_chroma_u16(&[-32767.0]), vec![0u16]);
    }

    #[test]
    fn bell_gain_is_unity_at_reference() {
        assert!((secam_bell_gain(SECAM_BELL_F0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bell_gain_boosts_away_from_reference() {
        // The bell rises on both sides of f0, and both rest carriers sit off it.
        assert!(secam_bell_gain(SECAM_FOR) > 1.0);
        assert!(secam_bell_gain(SECAM_FOB) > 1.0);
        // 16/1.26 is the asymptotic gain; the skirts stay below it.
        assert!(secam_bell_gain(SECAM_FREQ_MAX) < 16.0 / 1.26);
    }

    #[test]
    fn median_matches_numpy_semantics() {
        assert_eq!(median_of(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median_of(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert!(median_of(&[]).is_nan());
        assert_eq!(median_of_f64(&[3.0, 1.0, 2.0]), 2.0);
    }

    #[test]
    fn wrap_pi_folds_into_half_open_turn() {
        assert!((wrap_pi(0.5) - 0.5).abs() < 1e-12);
        assert!((wrap_pi(TAU + 0.5) - 0.5).abs() < 1e-12);
        assert!((wrap_pi(-TAU - 0.5) + 0.5).abs() < 1e-12);
        assert!((wrap_pi(PI).abs() - PI).abs() < 1e-12);
    }

    #[test]
    fn parity_flywheel_walks_the_four_field_cycle() {
        // dr_on_even(n) = base ^ (((n + 1) >> 1) & 1) with base = false gives
        // the FTTF/TFFT alternation seen on real method 1 tapes.
        let expected: Vec<bool> = (0..8).map(SecamParityFlywheel::flip).collect();
        assert_eq!(
            expected,
            vec![false, true, true, false, false, true, true, false]
        );
    }

    #[test]
    fn parity_flywheel_carries_unfittable_fields() {
        let mut flywheel = SecamParityFlywheel::new();
        // Four confident fields in a row teach and lock `base`.
        for index in 0..4u64 {
            let expected = SecamParityFlywheel::flip(index as i64);
            let (parity, source) = flywheel.resolve(index, Some((expected, 1.0)));
            assert_eq!(parity, Some(expected));
            assert_eq!(source, ParitySource::Measured);
        }
        // A field that can't be fitted now inherits the predicted parity.
        let (parity, source) = flywheel.resolve(4, None);
        assert_eq!(source, ParitySource::Flywheel);
        assert_eq!(parity, Some(SecamParityFlywheel::flip(4)));
    }

    #[test]
    fn parity_flywheel_stays_unlocked_before_min_lock() {
        let mut flywheel = SecamParityFlywheel::new();
        flywheel.resolve(0, Some((false, 1.0)));
        let (parity, source) = flywheel.resolve(1, None);
        assert_eq!(parity, None);
        assert_eq!(source, ParitySource::Unlocked);
    }

    #[test]
    fn parity_flywheel_ignores_low_confidence_fits() {
        let mut flywheel = SecamParityFlywheel::new();
        let (parity, source) = flywheel.resolve(0, Some((true, 0.5)));
        assert_eq!(parity, None);
        assert_eq!(source, ParitySource::Unlocked);
    }

    #[test]
    fn line_alternation_fit_reads_the_carrier_pair() {
        let outwidth = 1135usize;
        let linesout = 120usize;
        let first_line = 16usize;
        let porch_end_px = 186usize;
        let mut inst_freq = vec![0.0f32; linesout * outwidth];
        // D'R on even lines.
        for line in 0..linesout {
            let value = if line.is_multiple_of(2) {
                SECAM_FOR as f32
            } else {
                SECAM_FOB as f32
            };
            inst_freq[line * outwidth..(line + 1) * outwidth].fill(value);
        }

        let (dr_on_even, confidence) =
            fit_secam_line_alternation(&inst_freq, linesout, outwidth, first_line, porch_end_px)
                .expect("fit");
        assert!(dr_on_even);
        assert_eq!(confidence, 1.0);

        // Swapping the assignment flips the fit, still at full confidence.
        for line in 0..linesout {
            let value = if line.is_multiple_of(2) {
                SECAM_FOB as f32
            } else {
                SECAM_FOR as f32
            };
            inst_freq[line * outwidth..(line + 1) * outwidth].fill(value);
        }
        let (dr_on_even, confidence) =
            fit_secam_line_alternation(&inst_freq, linesout, outwidth, first_line, porch_end_px)
                .expect("fit");
        assert!(!dr_on_even);
        assert_eq!(confidence, 1.0);
    }

    #[test]
    fn line_alternation_fit_needs_enough_lines() {
        let outwidth = 1135usize;
        let inst_freq = vec![SECAM_FOR as f32; 40 * outwidth];
        assert!(fit_secam_line_alternation(&inst_freq, 40, outwidth, 16, 186).is_none());
    }
}
