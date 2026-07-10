use super::*;

// Native SECAM FM-chroma decoder: demodulates the still-FM-modulated,
// still-÷4-divided color-under carrier directly (as real SECAM VHS decks
// record it), instead of heterodyne-mixing it back up to a standard
// subcarrier the way the PAL/NTSC/MESECAM path does. Mixing cannot invert a
// frequency division, so that path is physically wrong for genuine SECAM
// tapes (it remains correct for MESECAM, which really does mix).
//
// Ported from the Python reference `vhsdecode/secam.py`
// (https://github.com/oyvindln/vhs-decode, branch `secam-native-chroma-decode`).

pub(crate) const UNDER_BPF_HZ: (f64, f64) = (850_000.0, 1_350_000.0);
pub(crate) const UNDER_BPF_ORDER: usize = 4;
pub(crate) const POST_LPF_HZ: f64 = 700_000.0;
pub(crate) const POST_LPF_ORDER: usize = 2;
pub(crate) const PREEMPH_F1_HZ: f64 = 85_000.0;
pub(crate) const PREEMPH_F2_HZ: f64 = 255_000.0;

const RED_CENTER_FREQ: f32 = 4_406_250.0;
const BLUE_CENTER_FREQ: f32 = 4_250_000.0;
const RED_MAX_DEVIATION: f32 = 280_000.0;
const BLUE_MAX_DEVIATION: f32 = 230_000.0;
const RED_COEFF: f32 = -1.902;
const BLUE_COEFF: f32 = 1.505;
const INST_FREQ_CLAMP: (f32, f32) = (3_400_000.0, 4_600_000.0);
const UNDER_FREQ_MULTIPLIER: f32 = 4.0;
const CHROMA_SCALE: f32 = 20_000.0;
const ACTIVE_WINDOW_US: (f64, f64) = (11.0, 62.0);
const VOTE_ROW_RANGE: (usize, usize) = (20, 300);
const AMP_KILL_RATIO: f32 = 0.4;
const CONCEAL_MAX_DISTANCE: i64 = 4;
const LINE_ACTIVE_US: (f64, f64) = (10.2, 62.6);
const TEMPORAL_NR_THRESHOLD: f32 = 0.08;
const TEMPORAL_NR_MAX_WEIGHT: f32 = 0.5;

/// Cross-field state for the motion-adaptive temporal chroma NR. Keyed by
/// (field parity, red/blue line-assignment offset) so a field only ever
/// blends with the field occupying the same slot of SECAM's 4-field colour
/// sequence (~2 frames / 80ms back).
#[derive(Clone, Default)]
pub(crate) struct SecamChromaState {
    previous_native: [Option<Vec<f32>>; 4],
}

impl SecamChromaState {
    fn slot(is_first_field: bool, offset: u8) -> usize {
        (is_first_field as usize) * 2 + offset as usize
    }
}

fn classify_lines(
    freq_by_line: &[f32],
    amp_by_line: &[f32],
    linesout: usize,
    outwidth: usize,
    sample_rate_hz: f64,
) -> (Vec<bool>, u8) {
    let rate_mhz = sample_rate_hz / 1e6;
    let active_lo = ((ACTIVE_WINDOW_US.0 * rate_mhz) as usize).min(outwidth);
    let active_hi = ((ACTIVE_WINDOW_US.1 * rate_mhz) as usize).clamp(active_lo, outwidth);

    let mut scratch = vec![0.0f32; active_hi - active_lo];
    let row_median: Vec<f32> = (0..linesout)
        .map(|line| {
            let start = line * outwidth + active_lo;
            scratch.copy_from_slice(&freq_by_line[start..start + (active_hi - active_lo)]);
            median_from_values(&mut scratch)
        })
        .collect();

    // Amplitude median per line, over the same window, to exclude line pairs
    // where the carrier is too weak to trust from the vote — the same weak
    // regions `conceal_weak_samples` already masks out of the chroma output.
    // Without this, a near-noise-floor carrier makes the frequency median
    // essentially random, which can flip the vote across a 50/50 boundary
    // between independently-started decoders on the same physical field
    // (e.g. two multithreaded-decode workers), producing a categorical
    // red/blue swap since RED_COEFF/BLUE_COEFF have opposite signs.
    let mut amp_scratch = vec![0.0f32; active_hi - active_lo];
    let row_amp: Vec<f32> = (0..linesout)
        .map(|line| {
            let start = line * outwidth + active_lo;
            amp_scratch.copy_from_slice(&amp_by_line[start..start + (active_hi - active_lo)]);
            median_from_values(&mut amp_scratch)
        })
        .collect();
    let mut global_amp_scratch = row_amp.clone();
    let amp_threshold = AMP_KILL_RATIO * median_from_values(&mut global_amp_scratch);

    // VOTE_ROW_RANGE.0 is even by construction (20).
    let vote_hi = VOTE_ROW_RANGE.1.min(linesout.saturating_sub(1));
    let vote = |require_amp: bool| -> (usize, usize) {
        let mut positive = 0usize;
        let mut total = 0usize;
        let mut k = VOTE_ROW_RANGE.0;
        while k + 1 < vote_hi {
            let strong = row_amp[k] >= amp_threshold && row_amp[k + 1] >= amp_threshold;
            if !require_amp || strong {
                if row_median[k + 1] - row_median[k] > 0.0 {
                    positive += 1;
                }
                total += 1;
            }
            k += 2;
        }
        (positive, total)
    };
    // Prefer the amplitude-gated vote; fall back to the ungated one if every
    // pair in the window was too weak to trust, so a field is never left
    // fully unvoted.
    let (positive, total) = match vote(true) {
        (_, 0) => vote(false),
        result => result,
    };
    let odd_is_red_fraction = if total > 0 {
        positive as f64 / total as f64
    } else {
        0.0
    };
    let offset: u8 = if odd_is_red_fraction >= 0.5 { 1 } else { 0 };
    let is_red_line = (0..linesout)
        .map(|i| (i + offset as usize).is_multiple_of(2))
        .collect();
    (is_red_line, offset)
}

/// Patch samples where the FM discriminator was reading bandpass-centered
/// noise (weak/lost carrier) from the nearest same-colour line above, within
/// `CONCEAL_MAX_DISTANCE` same-colour lines; mute to 0.0 beyond that. Run
/// independently per colour parity and per horizontal sample position, since
/// carrier amplitude varies within a line.
fn conceal_weak_samples(
    native: &mut [f32],
    amp_by_line: &[f32],
    is_red_line: &[bool],
    linesout: usize,
    outwidth: usize,
) {
    let lo = 20usize.min(linesout);
    let hi = linesout.saturating_sub(10).max(lo);
    let mut scratch: Vec<f32> = amp_by_line[lo * outwidth..hi * outwidth].to_vec();
    if scratch.is_empty() {
        return;
    }
    let amp_threshold = AMP_KILL_RATIO * median_from_values(&mut scratch);

    for want_red in [true, false] {
        let rows: Vec<usize> = (0..linesout)
            .filter(|&i| is_red_line[i] == want_red)
            .collect();
        let mut donor_group_j = vec![-1i64; outwidth];
        let mut donor_row = vec![0usize; outwidth];
        for (j, &row) in rows.iter().enumerate() {
            let j = j as i64;
            for col in 0..outwidth {
                let idx = row * outwidth + col;
                if amp_by_line[idx] < amp_threshold {
                    let reachable =
                        donor_group_j[col] >= 0 && (j - donor_group_j[col]) <= CONCEAL_MAX_DISTANCE;
                    native[idx] = if reachable {
                        native[donor_row[col] * outwidth + col]
                    } else {
                        0.0
                    };
                } else {
                    donor_group_j[col] = j;
                    donor_row[col] = row;
                }
            }
        }
    }
}

/// Motion-adaptive, non-recursive temporal chroma NR. The buffer stored for
/// next time is always the *unblended* input, never the blended output — a
/// recursive version accumulates sub-threshold noise into drifting colour
/// clouds that stick to moving areas ("dirty window" artifact).
fn temporal_nr(native: &mut [f32], state: &mut SecamChromaState, is_first_field: bool, offset: u8) {
    let slot = SecamChromaState::slot(is_first_field, offset);
    let unblended = native.to_vec();
    if let Some(prev) = &state.previous_native[slot] {
        if prev.len() == native.len() {
            for (n, &p) in native.iter_mut().zip(prev.iter()) {
                let diff = (*n - p).abs();
                let weight =
                    (1.0 - diff / TEMPORAL_NR_THRESHOLD).clamp(0.0, 1.0) * TEMPORAL_NR_MAX_WEIGHT;
                *n = *n * (1.0 - weight) + p * weight;
            }
        }
    }
    state.previous_native[slot] = Some(unblended);
}

pub(super) fn demod_secam_chroma(
    field: &mut DecodedField,
    spec: &DecoderSpec,
    state: &mut SecamChromaState,
    chroma_under: &[f32],
) -> Result<Vec<f32>> {
    let outwidth = field.outlinelen;
    let linesout = field.outlinecount;
    let sample_rate_hz = spec.sys_outfreq * 1e6;

    // Bandpass (850k-1350kHz, order 4, zero-phase) then analytic signal.
    let bandpassed = sosfiltfilt_f32(&spec.secam_chroma_bandpass, chroma_under);
    let analytic: Vec<Complex32> = hilbert_f32(
        &bandpassed,
        spec.fft_field_forward_f32.as_ref(),
        spec.fft_field_inverse_f32.as_ref(),
    );

    // Instantaneous amplitude and frequency.
    let amp_by_line: Vec<f32> = analytic.iter().map(|c| c.norm()).collect();
    let mut freq_by_line = vec![0.0f32; analytic.len()];
    unwrap_angles(&analytic, &mut freq_by_line, sample_rate_hz as f32, 0.0);

    // Undo the real deck's ÷4 recording, clamp to avoid wraparound artifacts.
    for f in &mut freq_by_line {
        *f = (*f * UNDER_FREQ_MULTIPLIER).clamp(INST_FREQ_CLAMP.0, INST_FREQ_CLAMP.1);
    }

    // Edge-line fixup: the first/last line can carry filter-edge artifacts;
    // overwrite from the same-colour line 2 rows away. Alternation is strictly
    // regular, so this is safe before classification runs.
    if linesout >= 3 {
        let (line0, rest) = freq_by_line.split_at_mut(outwidth);
        let line2 = &rest[outwidth..2 * outwidth];
        line0.copy_from_slice(line2);
        let last = linesout - 1;
        let src_start = (last - 2) * outwidth;
        let (head, tail) = freq_by_line.split_at_mut(last * outwidth);
        tail.copy_from_slice(&head[src_start..src_start + outwidth]);
    }

    // Classify red/blue lines.
    let (is_red_line, offset) =
        classify_lines(&freq_by_line, &amp_by_line, linesout, outwidth, sample_rate_hz);
    field.secam_first_line_is_red = is_red_line.first().copied();

    // Deviation from each line's own rest carrier.
    let mut deviation = vec![0.0f32; freq_by_line.len()];
    for line in 0..linesout {
        let center = if is_red_line[line] {
            RED_CENTER_FREQ
        } else {
            BLUE_CENTER_FREQ
        };
        let row = &freq_by_line[line * outwidth..(line + 1) * outwidth];
        let out_row = &mut deviation[line * outwidth..(line + 1) * outwidth];
        for (o, &f) in out_row.iter_mut().zip(row) {
            *o = f - center;
        }
    }

    // Zero deviation outside the active-picture window before filtering, so
    // sync-driven excursions don't ring into active video through the filters.
    let rate_mhz = sample_rate_hz / 1e6;
    let active_lo = ((LINE_ACTIVE_US.0 * rate_mhz) as usize).min(outwidth);
    let active_hi = ((LINE_ACTIVE_US.1 * rate_mhz) as usize).clamp(active_lo, outwidth);
    for line in 0..linesout {
        let row = &mut deviation[line * outwidth..(line + 1) * outwidth];
        row[..active_lo].fill(0.0);
        row[active_hi..].fill(0.0);
    }

    // Deemphasis (causal, whole flattened field, no group-delay compensation —
    // tried and reverted upstream: over-corrects mid frequencies).
    if !spec.secam_chroma_deemphasis.is_empty() {
        deviation = sosfilt_f32(&spec.secam_chroma_deemphasis, &deviation);
    }

    // Post-demod noise lowpass, zero-phase.
    deviation = sosfiltfilt_f32(&spec.secam_chroma_post_lpf, &deviation);

    // Normalize to native Dr/Db, per line.
    let mut native = vec![0.0f32; deviation.len()];
    for line in 0..linesout {
        let (max_dev, coeff) = if is_red_line[line] {
            (RED_MAX_DEVIATION, RED_COEFF)
        } else {
            (BLUE_MAX_DEVIATION, BLUE_COEFF)
        };
        let row = &deviation[line * outwidth..(line + 1) * outwidth];
        let out_row = &mut native[line * outwidth..(line + 1) * outwidth];
        for (o, &d) in out_row.iter_mut().zip(row) {
            *o = (d / max_dev / coeff).clamp(-1.0, 1.0);
        }
    }

    conceal_weak_samples(&mut native, &amp_by_line, &is_red_line, linesout, outwidth);
    temporal_nr(&mut native, state, field.is_first_field.unwrap_or(false), offset);

    for n in &mut native {
        *n *= CHROMA_SCALE;
    }
    Ok(native)
}
