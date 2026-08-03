//! DC cancellation and dropout compensation. Ports `cancelDC_trim`,
//! `_detect_dropouts`, `_check_other_channel`, `_fill`, `_mute_or_fill`,
//! `dropout_compensate`, and `merge_boundaries` from `HiFiDecode.py`.

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};

use crate::stereo::DecodeMode;

/// `cancelDC_trim` (`HiFiDecode.py:1857-1869`): subtract the mean over
/// `[trim, len-trim)` from that same range in place, and zero the `trim`
/// samples on each edge. Returns the DC value subtracted (feeds
/// auto-fine-tune).
pub fn cancel_dc_trim(audio: &mut [f32], trim: usize) -> f32 {
    let n = audio.len();
    assert!(trim * 2 < n, "trim window covers the whole buffer");

    let inner = &audio[trim..n - trim];
    let dc = inner.iter().map(|&v| v as f64).sum::<f64>() / inner.len() as f64;
    let dc = dc as f32;

    for sample in &mut audio[trim..n - trim] {
        *sample -= dc;
    }
    for sample in &mut audio[..trim] {
        *sample = 0.0;
    }
    for sample in &mut audio[n - trim..] {
        *sample = 0.0;
    }
    dc
}

pub(crate) fn mean_stddev(signal: &[f32]) -> (f64, f64) {
    let mean = signal.iter().map(|&v| v as f64).sum::<f64>() / signal.len() as f64;
    let variance = signal.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / signal.len() as f64;
    (mean, variance.sqrt())
}

/// `merge_boundaries` (`HiFiDecode.py:1719-1730`): sort by start, then
/// merge any overlapping/touching `[start, end)` ranges.
pub(crate) fn merge_boundaries(mut boundaries: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    boundaries.sort_by_key(|&(start, _)| start);
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (start, end) in boundaries {
        match merged.last_mut() {
            Some(last) if last.1 >= start => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    merged
}

/// Precomputed constants for dropout detection/compensation, mirroring the
/// `doc_*`/`hs_*`-adjacent fields `HiFiDecode.__init__` sets on `self`
/// (`HiFiDecode.py:1191-1215`). `audio_rate` here is always the fixed
/// 192kHz intermediate rate (`self.doc_audio_rate = self.audio_rate`).
pub struct DropoutParams {
    window_size: usize,
    hop_size: usize,
    amplitude_threshold: f64,
    std_threshold: f64,
    fft_start: usize,
    fft_end: usize,
    fade_samples: usize,
    dc_offset_samples: usize,
    fft_r2c: Arc<dyn RealToComplex<f32>>,
}

impl DropoutParams {
    /// Default DOC constants (`HiFiDecode.py:1194-1204`); `audio_rate`
    /// should be the 192kHz intermediate rate.
    pub fn new(audio_rate: f64) -> Self {
        let window_size = 128usize;
        let cutoff_freq = 40_000.0f64;

        // Mirrors the `doc_fft_start`/`doc_fft_end` derivation loop over
        // `np.fft.fftfreq(window_size, d=1/audio_rate)` exactly
        // (`HiFiDecode.py:1205-1215`), rather than a closed-form shortcut,
        // so an odd `window_size` or non-192kHz rate stays correct too.
        let mut fft_start = 0usize;
        let mut fft_end = 0usize;
        for i in 0..window_size {
            let freq = fftfreq(i, window_size, 1.0 / audio_rate);
            if freq < 0.0 {
                fft_end = i - 1;
                break;
            }
            if freq <= cutoff_freq {
                fft_start = i;
            }
        }

        let fft_r2c = RealFftPlanner::<f32>::new().plan_fft_forward(window_size);

        DropoutParams {
            window_size,
            hop_size: 128,
            amplitude_threshold: 1.0,
            std_threshold: 1.0,
            fft_start,
            fft_end,
            fade_samples: 128,
            dc_offset_samples: 256,
            fft_r2c,
        }
    }
}

/// `numpy.fft.fftfreq(n, d)[i]`.
fn fftfreq(i: usize, n: usize, d: f64) -> f64 {
    let k = if i <= (n - 1) / 2 {
        i as i64
    } else {
        i as i64 - n as i64
    };
    k as f64 / (d * n as f64)
}

/// `_detect_dropouts` (`HiFiDecode.py:1871-1917`): sliding-window FFT over
/// `audio`; a window is "dropped out" when both the magnitude mean and
/// std-dev (over `[fft_start, fft_end)`, i.e. up to `doc_cutoff_freq`)
/// exceed their thresholds — the broadband-noise signature of a lost FM
/// carrier.
fn detect_dropouts(audio: &[f32], params: &DropoutParams) -> Vec<(usize, usize)> {
    let mut dropout_ranges: Vec<(usize, usize)> = Vec::new();
    let mut full_spectrum_ranges_count = 0usize;

    if audio.len() < params.window_size {
        return dropout_ranges;
    }

    let mut start = 0usize;
    while start < audio.len() - params.window_size {
        let end = start + params.window_size;
        let window = &audio[start..end];
        let spectrum = tape_dsp::rfft_f32(window, params.fft_r2c.as_ref());
        let magnitude: Vec<f32> = spectrum[params.fft_start..params.fft_end]
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();
        let (mean, std) = mean_stddev(&magnitude);

        if mean > params.amplitude_threshold && std > params.std_threshold {
            if dropout_ranges.len() == full_spectrum_ranges_count {
                dropout_ranges.push((start, audio.len()));
            }
        } else if dropout_ranges.len() > full_spectrum_ranges_count {
            dropout_ranges[full_spectrum_ranges_count].1 = end;
            full_spectrum_ranges_count += 1;
        }

        start += params.hop_size;
    }

    merge_boundaries(dropout_ranges)
}

/// `_check_other_channel` (`HiFiDecode.py:1919-1943`): classifies each of
/// `gaps_to_fill` into a fillable-from-`source_gaps` part (`Some(fill), None`)
/// and/or a both-channels-lost part that must be muted (`None, Some(mute)`).
fn check_other_channel(
    gaps_to_fill: &[(usize, usize)],
    source_gaps: &[(usize, usize)],
) -> Vec<(Option<(usize, usize)>, Option<(usize, usize)>)> {
    let mut result = Vec::new();
    for &(orig_start, end) in gaps_to_fill {
        let mut start = orig_start;
        let mut overlap_found = false;
        for &(s_start, s_end) in source_gaps {
            if end <= s_start || start >= s_end {
                continue;
            }
            overlap_found = true;
            if start < s_start {
                result.push((Some((start, s_start)), None));
            }
            let overlap_start = start.max(s_start);
            let overlap_end = end.min(s_end);
            result.push((None, Some((overlap_start, overlap_end))));
            start = overlap_end;
        }
        if !overlap_found || start < end {
            result.push((Some((start, end)), None));
        }
    }
    result
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}

/// `_fill` (`HiFiDecode.py:1945-2038`): crossfades `outer[start..end]` with
/// `inner` (or silence, if `mute`), padded by `fade_samples` on each side
/// and DC-corrected by a raised-cosine ramp between the DC level just
/// before and just after the gap.
#[allow(clippy::too_many_arguments)]
fn fill(start: usize, end: usize, outer: &mut [f32], inner: Option<&[f32]>, fade_samples: usize, dc_window: usize, mute: bool, epsilon: f32) {
    let fade_start = start.saturating_sub(fade_samples);
    let fade_start_duration = start - fade_start;

    let dc_before_start = fade_start.saturating_sub(dc_window);
    let dc_before_end = start;
    let dc_before = if dc_before_start < dc_before_end {
        mean(&outer[dc_before_start..dc_before_end])
    } else {
        0.0
    };

    let fade_end = (end + fade_samples).min(outer.len());
    let fade_end_duration = fade_end - end;

    let dc_after_start = end;
    let dc_after_end = (fade_end + dc_window).min(outer.len());
    let dc_after = if dc_after_start < dc_after_end {
        mean(&outer[dc_after_start..dc_after_end])
    } else {
        0.0
    };

    let dc_inner = if mute { 0.0 } else { mean(&inner.expect("inner required when not muting")[fade_start..fade_end]) };

    let dc_total_len = fade_end - fade_start;
    let mut dc_interp_full = vec![0.0f32; dc_total_len];
    if dc_total_len > 1 {
        let denom = (dc_total_len - 1) as f32;
        let delta = dc_after - dc_before;
        for (i, slot) in dc_interp_full.iter_mut().enumerate() {
            let t = i as f32 / denom;
            let smooth = 0.5 * (1.0 - (std::f32::consts::PI * t).cos());
            *slot = dc_before + delta * smooth;
        }
    } else if dc_total_len == 1 {
        dc_interp_full[0] = dc_before;
    }

    for i in 0..fade_start_duration {
        let idx = fade_start + i;
        let dc_idx = idx - fade_start;
        let fade_in_factor = i as f32 / fade_start_duration as f32;
        let fade_out_factor = 1.0 - fade_in_factor;
        let outer_sample = outer[idx] * fade_out_factor;
        let inner_sample = (if mute {
            epsilon
        } else {
            inner.unwrap()[idx] - dc_inner + dc_interp_full[dc_idx]
        }) * fade_in_factor;
        outer[idx] = outer_sample + inner_sample;
    }

    for i in start..end {
        let dc_idx = i - fade_start;
        let inner_sample = if mute { epsilon } else { inner.unwrap()[i] };
        outer[i] = inner_sample - dc_inner + dc_interp_full[dc_idx];
    }

    for i in 0..fade_end_duration {
        let idx = end + i;
        let dc_idx = idx - fade_start;
        let fade_in_factor = (i + 1) as f32 / fade_end_duration as f32;
        let fade_out_factor = 1.0 - fade_in_factor;
        let outer_sample = outer[idx] * fade_in_factor;
        let inner_sample = (if mute {
            epsilon
        } else {
            inner.unwrap()[idx] - dc_inner + dc_interp_full[dc_idx]
        }) * fade_out_factor;
        outer[idx] = outer_sample + inner_sample;
    }
}

/// `_mute_or_fill` (`HiFiDecode.py:2041-2070`).
fn mute_or_fill(
    dropouts: &[(Option<(usize, usize)>, Option<(usize, usize)>)],
    current_channel: &mut [f32],
    other_channel: &[f32],
    fade_samples: usize,
    dc_window: usize,
) {
    // np.finfo(np.float16).eps
    const EPSILON: f32 = 0.000_976_562_5;
    for &(fill_range, mute_range) in dropouts {
        if let Some((s, e)) = fill_range {
            fill(s, e, current_channel, Some(other_channel), fade_samples, dc_window, false, EPSILON);
        }
        if let Some((s, e)) = mute_range {
            fill(s, e, current_channel, Some(other_channel), fade_samples, dc_window, true, EPSILON);
        }
    }
}

/// `dropout_compensate` (`HiFiDecode.py:2072-2125`): detects per-channel
/// carrier-loss dropouts and either fills them from the other channel or
/// mutes them, depending on `decode_mode`/`doc_mute_only`. Mutates
/// `audio_l`/`audio_r` in place, matching the Python signature (which
/// returns nothing and mutates its `np.array` arguments).
pub fn dropout_compensate(audio_l: &mut [f32], audio_r: &mut [f32], params: &DropoutParams, decode_mode: DecodeMode, doc_mute_only: bool) {
    let dropout_boundaries_l = if decode_mode.uses_right_channel() {
        detect_dropouts(audio_l, params)
    } else {
        vec![(0, audio_r.len())]
    };
    let dropout_boundaries_r = if decode_mode.uses_left_channel() {
        detect_dropouts(audio_r, params)
    } else {
        vec![(0, audio_l.len())]
    };

    let mute_all = |gaps: &[(usize, usize)]| -> Vec<(Option<(usize, usize)>, Option<(usize, usize)>)> {
        gaps.iter().map(|&(s, e)| (None, Some((s, e)))).collect()
    };

    let (dropouts_left, dropouts_right) = if decode_mode.is_dual_mono() || doc_mute_only {
        (mute_all(&dropout_boundaries_l), mute_all(&dropout_boundaries_r))
    } else {
        let dropouts_left = if decode_mode.uses_right_channel() {
            check_other_channel(&dropout_boundaries_l, &dropout_boundaries_r)
        } else {
            mute_all(&dropout_boundaries_l)
        };
        let dropouts_right = if decode_mode.uses_left_channel() {
            check_other_channel(&dropout_boundaries_r, &dropout_boundaries_l)
        } else {
            mute_all(&dropout_boundaries_r)
        };
        (dropouts_left, dropouts_right)
    };

    if decode_mode.uses_right_channel() {
        let other = audio_r.to_vec();
        mute_or_fill(&dropouts_left, audio_l, &other, params.fade_samples, params.dc_offset_samples);
    }
    if decode_mode.uses_left_channel() {
        let other = audio_l.to_vec();
        mute_or_fill(&dropouts_right, audio_r, &other, params.fade_samples, params.dc_offset_samples);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancel_dc_trim_zeroes_edges_and_removes_mean() {
        let mut audio = vec![0.0f32; 20];
        for (i, v) in audio.iter_mut().enumerate() {
            *v = 5.0 + i as f32 * 0.01;
        }
        let dc = cancel_dc_trim(&mut audio, 3);
        assert!(dc > 4.9 && dc < 5.2);
        assert!(audio[..3].iter().all(|&v| v == 0.0));
        assert!(audio[17..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn merge_boundaries_merges_overlaps() {
        let merged = merge_boundaries(vec![(0, 5), (10, 15), (3, 8), (20, 25)]);
        assert_eq!(merged, vec![(0, 8), (10, 15), (20, 25)]);
    }
}
