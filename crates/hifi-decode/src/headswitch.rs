//! Head-switching noise detection and interpolation. Ports
//! `headswitch_detect_peaks`, `headswitch_calc_boundaries`,
//! `headswitch_interpolate_boundaries`, `headswitch_remove_noise`, and
//! `smooth` from `HiFiDecode.py:1586-1809`. Unlike the video decoder's
//! head-switch handling, this has nothing to do with field timing — it is
//! purely a noise-removal pass over the demodulated audio.

use sci_rs::signal::filter::design::{FilterBandType, Sos};
use tape_dsp::{cheby2_sos, narrow_sos, sosfiltfilt_f32};

use crate::dropout::{mean_stddev, merge_boundaries};
use crate::find_peaks::{find_peaks, FindPeaksOptions};

/// Fixed constants from `HiFiDecode.__init__` (`:1161-1187`), independent
/// of any CLI option.
pub struct HeadswitchParams {
    hs_sos: Vec<Sos<f32>>,
    signal_rate: f64,
    hz: f64,
    drift_hz: f64,
    peak_prominence_limit: f64,
    interpolation_padding: f64,
    interpolation_neighbor_range: f64,
    passes: usize,
}

impl HeadswitchParams {
    /// `signal_rate` is always the 192kHz intermediate rate
    /// (`headswitch_signal_rate = self.audio_rate`); `field_rate` is 50
    /// (PAL) or 59.94 (NTSC).
    pub fn new(signal_rate: f64, field_rate: f64) -> Self {
        const CUTOFF_FREQ: f64 = 28_000.0;
        let hs_sos_f64 = cheby2_sos(22, 200.0, &[CUTOFF_FREQ], FilterBandType::Highpass, signal_rate);
        HeadswitchParams {
            hs_sos: narrow_sos(&hs_sos_f64),
            signal_rate,
            hz: field_rate,
            drift_hz: field_rate * 0.1,
            peak_prominence_limit: 3.0,
            interpolation_padding: 35e-6,
            interpolation_neighbor_range: 200e-6,
            passes: 1,
        }
    }
}

/// One detected head-switch (or neighboring-noise) event: sample index of
/// the peak center, its interpolated left/right extent at half-prominence
/// height, and its prominence (clamped to `[0, peak_prominence_limit]`).
/// `center` mirrors Python's `peaks` tuple exactly (`HiFiDecode.py:1627-1641`)
/// but, like the Python source, nothing downstream ever reads it back —
/// `headswitch_calc_boundaries` only uses `start`/`end`/`prominence`.
#[allow(dead_code)]
struct DetectedPeak {
    center: f64,
    start: f64,
    end: f64,
    prominence: f64,
}

fn clamp_prominence(prominence: f64, limit: f64) -> f64 {
    prominence.min(limit).max(0.0)
}

/// `headswitch_detect_peaks` (`HiFiDecode.py:1586-1678`).
fn detect_peaks(audio: &[f32], params: &HeadswitchParams) -> Vec<DetectedPeak> {
    let filtered_signal = sosfiltfilt_f32(&params.hs_sos, audio);
    let filtered_abs: Vec<f32> = filtered_signal.iter().map(|v| v.abs()).collect();
    let (mean, std_dev) = mean_stddev(&filtered_signal);

    let peak_distance_seconds = 1.0 / (params.hz + params.drift_hz);
    let distance = peak_distance_seconds * params.signal_rate;

    let primary = find_peaks(
        &filtered_abs,
        FindPeaksOptions {
            distance: Some(distance),
            threshold_min: None,
            prominence_min: None,
        },
    );

    let neighbor_threshold = mean + std_dev;
    let neighbor_search_width = (params.interpolation_neighbor_range * params.signal_rate).round() as usize;

    let mut peaks = Vec::new();
    for peak in &primary {
        peaks.push(DetectedPeak {
            center: peak.index as f64,
            start: peak.left_ips,
            end: peak.right_ips,
            prominence: clamp_prominence(peak.prominence, params.peak_prominence_limit),
        });

        let start_neighbor = (peak.left_ips - neighbor_search_width as f64).floor().max(0.0) as usize;
        let end_neighbor = ((peak.right_ips + neighbor_search_width as f64).ceil() as usize).min(filtered_abs.len());
        if start_neighbor >= end_neighbor {
            continue;
        }

        let neighbor_slice = &filtered_abs[start_neighbor..end_neighbor];
        let neighbors = find_peaks(
            neighbor_slice,
            FindPeaksOptions {
                distance: Some(1.0),
                threshold_min: Some(neighbor_threshold as f64),
                prominence_min: Some(0.25),
            },
        );
        for neighbor in &neighbors {
            peaks.push(DetectedPeak {
                center: neighbor.index as f64 + start_neighbor as f64,
                start: neighbor.left_ips + start_neighbor as f64,
                end: neighbor.right_ips + start_neighbor as f64,
                prominence: clamp_prominence(neighbor.prominence, params.peak_prominence_limit),
            });
        }
    }
    peaks
}

/// `headswitch_calc_boundaries` (`HiFiDecode.py:1680-1702`): widen each
/// peak by `prominence * padding_samples`, then merge overlaps. Boundaries
/// are signed (`isize`) because a peak near a block edge can widen past
/// index 0 or past the buffer end — `interpolate_boundaries` below handles
/// that by clamping.
fn calc_boundaries(peaks: &[DetectedPeak], params: &HeadswitchParams) -> Vec<(isize, isize)> {
    let padding_samples = (params.interpolation_padding * params.signal_rate).round();
    let mut boundaries: Vec<(isize, isize)> = peaks
        .iter()
        .map(|p| {
            let width_padding = p.prominence * padding_samples;
            let start = (p.start - width_padding).floor() as isize;
            let end = (p.end + width_padding).ceil() as isize;
            (start, end)
        })
        .collect();

    // `merge_boundaries` operates on non-negative usize ranges; shift into
    // that domain and back out, rather than duplicating it for isize.
    let shift = boundaries.iter().map(|&(s, _)| s).min().unwrap_or(0).min(0).unsigned_abs();
    let shifted: Vec<(usize, usize)> = boundaries
        .drain(..)
        .map(|(s, e)| ((s + shift as isize) as usize, (e + shift as isize) as usize))
        .collect();
    merge_boundaries(shifted)
        .into_iter()
        .map(|(s, e)| (s as isize - shift as isize, e as isize - shift as isize))
        .collect()
}

/// `smooth` (`HiFiDecode.py:1711-1716`): centered moving average.
fn smooth(data_in: &[f32], half_window: usize) -> Vec<f32> {
    let n = data_in.len();
    (0..n)
        .map(|i| {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(n);
            let slice = &data_in[start..end];
            slice.iter().sum::<f32>() / slice.len() as f32
        })
        .collect()
}

/// `headswitch_interpolate_boundaries` (`HiFiDecode.py:1732-1786`).
///
/// Python builds one global `scipy.interpolate.interp1d` over every
/// sample outside *any* boundary and evaluates it inside each gap; because
/// `calc_boundaries`'s ranges are disjoint and sorted (post-merge), that
/// interpolator's two bracketing knots for a query inside `[start, end)`
/// are always exactly `start-1` and `end` — so this ports it as a direct
/// two-point linear interpolation between those samples, which is
/// equivalent for every in-bounds boundary. The one case this simplifies
/// rather than exactly replicating scipy's `fill_value="extrapolate"` is a
/// boundary touching a buffer edge with fewer than two valid samples on
/// that side to extrapolate a slope from; that falls back to sample-and-
/// hold from the nearest valid sample, same as Python's explicit
/// `start < 0` / `end > len(audio)` branches already do for the more
/// common out-of-bounds case.
fn interpolate_boundaries(audio: &[f32], boundaries: &[(isize, isize)]) -> Vec<f32> {
    let mut out = audio.to_vec();
    let n = audio.len() as isize;

    for &(start, end) in boundaries {
        if start < 0 {
            let hold_value = out[(end.clamp(0, n - 1)) as usize];
            for sample in out[..end.clamp(0, n) as usize].iter_mut() {
                *sample = hold_value;
            }
            continue;
        }
        if end > n {
            let hold_value = out[start.clamp(0, n - 1) as usize];
            for sample in out[start.clamp(0, n) as usize..].iter_mut() {
                *sample = hold_value;
            }
            continue;
        }

        let (start, end) = (start as usize, end as usize);
        let left_knot = if start > 0 { Some((start - 1, audio[start - 1] as f64)) } else { None };
        let right_knot = if end < audio.len() { Some((end, audio[end] as f64)) } else { None };

        for i in start..end {
            let value = match (left_knot, right_knot) {
                (Some((lx, ly)), Some((rx, ry))) => {
                    let t = (i as f64 - lx as f64) / (rx as f64 - lx as f64);
                    ly + (ry - ly) * t
                }
                (Some((_, ly)), None) => ly,
                (None, Some((_, ry))) => ry,
                (None, None) => audio[i] as f64,
            };
            out[i] = value as f32;
        }

        let smoothing_size = 1 + end - start;
        let half_window = smoothing_size.div_ceil(4);
        let smooth_start = start.saturating_sub(smoothing_size);
        let smooth_end = (end + smoothing_size).min(out.len());
        if smooth_start < smooth_end {
            let smoothed = smooth(&out[smooth_start..smooth_end], half_window);
            out[smooth_start..smooth_end].copy_from_slice(&smoothed);
        }
    }

    out
}

/// `headswitch_remove_noise` (`HiFiDecode.py:1788-1809`): detect -> widen
/// -> interpolate, `params.passes` times (always 1 in practice —
/// `headswitch_passes` is hard-coded in `HiFiDecode.__init__`).
pub fn remove_noise(audio: &[f32], params: &HeadswitchParams) -> Vec<f32> {
    let mut audio = audio.to_vec();
    for _ in 0..params.passes {
        let peaks = detect_peaks(&audio, params);
        let boundaries = calc_boundaries(&peaks, params);
        audio = interpolate_boundaries(&audio, &boundaries);
    }
    audio
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smooth_averages_a_centered_window() {
        let data = vec![0.0, 0.0, 10.0, 0.0, 0.0];
        let out = smooth(&data, 1);
        // center sample: mean of [0,10,0] = 3.333...
        assert!((out[2] - 10.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn remove_noise_interpolates_an_injected_spike() {
        let fs = 192_000.0;
        let n = 19_200usize; // 100ms
        let mut audio: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / fs).sin() as f32 * 0.3)
            .collect();
        // Inject a broadband head-switch-like pulse (alternating sign, so
        // it actually has energy above the 28kHz highpass detector -
        // real head-switch noise is broadband, not a DC step) at the
        // field-rate interval (50Hz -> every 3840 samples at 192kHz).
        for center in [3840usize, 7680, 11520, 15360] {
            for (k, i) in (center.saturating_sub(4)..(center + 4).min(n)).enumerate() {
                audio[i] += if k % 2 == 0 { 3.0 } else { -3.0 };
            }
        }

        let params = HeadswitchParams::new(fs, 50.0);
        let cleaned = remove_noise(&audio, &params);

        assert_eq!(cleaned.len(), audio.len());
        // The injected spikes should be substantially attenuated relative
        // to their raw 2.0+ amplitude.
        for center in [3840usize, 7680, 11520, 15360] {
            assert!(cleaned[center].abs() < 1.0, "spike at {center} not interpolated: {}", cleaned[center]);
        }
    }
}
