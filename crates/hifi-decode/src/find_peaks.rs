//! A from-scratch port of the subset of `scipy.signal.find_peaks` that
//! `headswitch_detect_peaks` needs (`HiFiDecode.py:1586-1678`): local
//! maxima with plateau handling, minimum-distance greedy suppression,
//! prominence, and width/`left_ips`/`right_ips` at `rel_height=0.5`.
//! `scipy` has no Rust equivalent, so this reimplements the documented
//! algorithm and is validated numerically against real
//! `scipy.signal.find_peaks` output (see `tests/find_peaks_parity.rs`).
//!
//! Only what `headswitch_detect_peaks` actually reads is implemented:
//! `distance`, `threshold` (minimum-only), `prominence` (minimum-only),
//! `width=1` (minimum-only — narrow single-sample bumps can otherwise
//! slip past the other three filters, confirmed against real scipy
//! output), and `left_ips`/`right_ips`/`prominences`.

/// One detected peak: sample index, its `left_ips`/`right_ips` (the
/// linearly-interpolated crossing points of the half-prominence height,
/// matching scipy's `rel_height=0.5` default), and its prominence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Peak {
    pub index: usize,
    pub left_ips: f64,
    pub right_ips: f64,
    pub prominence: f64,
}

/// `scipy.signal._peak_finding_utils._local_maxima_1d`: every maximal
/// plateau (one or more equal samples) that is strictly greater than the
/// samples immediately outside it on both sides. Returns each plateau's
/// midpoint index (rounded down, matching scipy's `(left+right)/2`
/// integer division) plus its left/right edges.
fn local_maxima_1d(x: &[f32]) -> Vec<(usize, usize, usize)> {
    let mut maxima = Vec::new();
    if x.len() < 3 {
        return maxima;
    }
    let mut i = 1usize;
    let i_max = x.len() - 1;
    while i < i_max {
        if x[i - 1] < x[i] {
            let mut i_ahead = i + 1;
            while i_ahead < i_max && x[i_ahead] == x[i] {
                i_ahead += 1;
            }
            if x[i_ahead] < x[i] {
                let left_edge = i;
                let right_edge = i_ahead - 1;
                let midpoint = (left_edge + right_edge) / 2;
                maxima.push((midpoint, left_edge, right_edge));
                i = i_ahead;
            }
        }
        i += 1;
    }
    maxima
}

/// `scipy.signal._peak_finding_utils._select_by_peak_distance`: greedy
/// suppression, highest-priority (here: signal value at the peak) first,
/// removing any lower-priority peak within `distance` samples.
fn select_by_peak_distance(peaks: &[usize], priority: &[f32], distance: f64) -> Vec<bool> {
    let n = peaks.len();
    let mut keep = vec![true; n];
    // Highest priority first; scipy uses `argsort` (ascending) then walks
    // in reverse, which for ties preserves the *lowest original index*
    // winning among equal priorities (a stable ascending sort reversed).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| priority[a].partial_cmp(&priority[b]).unwrap());

    let distance_ceil = distance.ceil() as isize;
    for &j in order.iter().rev() {
        if !keep[j] {
            continue;
        }
        let mut k = j as isize - 1;
        while k >= 0 && (peaks[j] as isize - peaks[k as usize] as isize) < distance_ceil {
            keep[k as usize] = false;
            k -= 1;
        }
        let mut k = j as isize + 1;
        while (k as usize) < n && (peaks[k as usize] as isize - peaks[j] as isize) < distance_ceil {
            keep[k as usize] = false;
            k += 1;
        }
    }
    keep
}

/// `scipy.signal.peak_prominences` with `wlen=None` (unbounded search):
/// for each peak, extend left/right until a strictly higher sample (or the
/// signal edge), tracking the minimum sample seen in each direction as
/// that side's base. Prominence is `peak_height - max(left_min, right_min)`.
fn peak_prominences(x: &[f32], peaks: &[usize]) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    let mut prominences = Vec::with_capacity(peaks.len());
    let mut left_bases = Vec::with_capacity(peaks.len());
    let mut right_bases = Vec::with_capacity(peaks.len());

    for &peak in peaks {
        let peak_height = x[peak] as f64;

        let mut left_min = peak_height;
        let mut left_base = peak;
        let mut i = peak;
        while i > 0 {
            i -= 1;
            let v = x[i] as f64;
            if v > peak_height {
                break;
            }
            if v < left_min {
                left_min = v;
                left_base = i;
            }
        }

        let mut right_min = peak_height;
        let mut right_base = peak;
        let mut i = peak;
        while i + 1 < x.len() {
            i += 1;
            let v = x[i] as f64;
            if v > peak_height {
                break;
            }
            if v < right_min {
                right_min = v;
                right_base = i;
            }
        }

        prominences.push(peak_height - left_min.max(right_min));
        left_bases.push(left_base);
        right_bases.push(right_base);
    }

    (prominences, left_bases, right_bases)
}

/// `scipy.signal.peak_widths` at a fixed `rel_height` (0.5 = scipy's
/// default, used unconditionally here since that's all `find_peaks`
/// needs): the height `peak - prominence*rel_height` intersected with the
/// signal on each side of the peak (searched only within
/// `[left_base, right_base]`), linearly interpolated for a sub-sample
/// crossing point.
#[allow(clippy::too_many_arguments)]
fn peak_widths_at_half_prominence(
    x: &[f32],
    peaks: &[usize],
    prominences: &[f64],
    left_bases: &[usize],
    right_bases: &[usize],
) -> (Vec<f64>, Vec<f64>) {
    const REL_HEIGHT: f64 = 0.5;
    let mut left_ips = Vec::with_capacity(peaks.len());
    let mut right_ips = Vec::with_capacity(peaks.len());

    for (idx, &peak) in peaks.iter().enumerate() {
        let height = x[peak] as f64 - prominences[idx] * REL_HEIGHT;

        let mut i = peak;
        while i > left_bases[idx] && (x[i] as f64) > height {
            i -= 1;
        }
        let mut left_ip = i as f64;
        if (x[i] as f64) < height {
            // Linear interpolation between i and i+1.
            left_ip += (height - x[i] as f64) / (x[i + 1] as f64 - x[i] as f64);
        }

        let mut i = peak;
        while i < right_bases[idx] && (x[i] as f64) > height {
            i += 1;
        }
        let mut right_ip = i as f64;
        if (x[i] as f64) < height {
            right_ip -= (height - x[i] as f64) / (x[i - 1] as f64 - x[i] as f64);
        }

        left_ips.push(left_ip);
        right_ips.push(right_ip);
    }

    (left_ips, right_ips)
}

/// Options mirroring the subset of `scipy.signal.find_peaks` kwargs
/// `headswitch_detect_peaks` uses. `None` disables that filter, matching
/// scipy's `None` default.
#[derive(Clone, Copy, Debug, Default)]
pub struct FindPeaksOptions {
    pub distance: Option<f64>,
    pub threshold_min: Option<f64>,
    pub prominence_min: Option<f64>,
}

/// `scipy.signal.find_peaks(x, distance=.., threshold=.., prominence=..,
/// width=1)`, staged in scipy's own order: local maxima -> threshold ->
/// distance -> prominence (+ filter) -> width (computed, not filtered —
/// see the module doc comment on why `width=1` is a no-op here).
pub fn find_peaks(x: &[f32], options: FindPeaksOptions) -> Vec<Peak> {
    let maxima = local_maxima_1d(x);
    let mut peaks: Vec<usize> = maxima.iter().map(|&(mid, _, _)| mid).collect();

    if let Some(threshold_min) = options.threshold_min {
        peaks.retain(|&p| {
            if p == 0 || p + 1 >= x.len() {
                return false;
            }
            let left = x[p] as f64 - x[p - 1] as f64;
            let right = x[p] as f64 - x[p + 1] as f64;
            left.min(right) >= threshold_min
        });
    }

    if let Some(distance) = options.distance {
        if distance >= 1.0 && peaks.len() > 1 {
            let priority: Vec<f32> = peaks.iter().map(|&p| x[p]).collect();
            let keep = select_by_peak_distance(&peaks, &priority, distance);
            peaks = peaks
                .into_iter()
                .zip(keep)
                .filter_map(|(p, k)| k.then_some(p))
                .collect();
        }
    }

    let (prominences, left_bases, right_bases) = peak_prominences(x, &peaks);

    let mut kept: Vec<usize> = (0..peaks.len()).collect();
    if let Some(prominence_min) = options.prominence_min {
        kept.retain(|&i| prominences[i] >= prominence_min);
    }

    let kept_peaks: Vec<usize> = kept.iter().map(|&i| peaks[i]).collect();
    let kept_prominences: Vec<f64> = kept.iter().map(|&i| prominences[i]).collect();
    let kept_left_bases: Vec<usize> = kept.iter().map(|&i| left_bases[i]).collect();
    let kept_right_bases: Vec<usize> = kept.iter().map(|&i| right_bases[i]).collect();

    let (left_ips, right_ips) =
        peak_widths_at_half_prominence(x, &kept_peaks, &kept_prominences, &kept_left_bases, &kept_right_bases);

    // `width=1` is always in force at the two `headswitch_detect_peaks`
    // call sites, and — unlike `distance`/`threshold`/`prominence` — it is
    // NOT a no-op: a narrow single-sample bump can have prominence and
    // threshold clearing its bar while still measuring under 1 sample wide
    // at half its own prominence (confirmed against a real scipy run
    // where such a peak was excluded; see find_peaks_parity.rs).
    const WIDTH_MIN: f64 = 1.0;

    kept_peaks
        .into_iter()
        .zip(left_ips)
        .zip(right_ips)
        .zip(kept_prominences)
        .filter_map(|(((index, left_ips), right_ips), prominence)| {
            (right_ips - left_ips >= WIDTH_MIN).then_some(Peak {
                index,
                left_ips,
                right_ips,
                prominence,
            })
        })
        .collect()
}
