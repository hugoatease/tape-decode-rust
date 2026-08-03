//! Sample-rate conversion. Ports the role of Python's `soxr.ResampleStream`
//! (`HiFiDecode.py:1139-1143`) using `rubato` instead — chosen over an FFI
//! binding to libsoxr specifically to keep the workspace free of native
//! dependencies (see the hifi-decode port plan). This is **not** a
//! bit-parity replacement: rubato's windowed-sinc resampler does not
//! reproduce soxr's output sample-for-sample, so validation is by
//! frequency response, not fixture equality.
//!
//! Matches Python's per-block usage exactly in one respect that matters a
//! lot for correctness: `resample_chunk(audio, True)` immediately followed
//! by `.clear()` (`HiFiDecode.py:2195-2196`, `:2229-2230`) means **no
//! resampler state survives across blocks** — each call is independently
//! flushed, and it's block overlap (not resampler continuity) that hides
//! the resulting edge transients. `BlockResampler::resample` mirrors this:
//! it builds a fresh `rubato` resampler per call rather than keeping one
//! alive across calls.

use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

/// Mirrors `--resampler_quality` (`HiFiDecode.py:1126-1137`). Python maps
/// this to soxr presets (VHQ/HQ/MQ/LQ); there's no equivalent preset table
/// in rubato, so this picks `sinc_len`/`oversampling_factor` pairs that
/// trade the same speed/quality axis, documented here rather than assumed
/// perceptually equivalent to soxr's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResamplerQuality {
    High,
    Medium,
    Low,
}

impl ResamplerQuality {
    fn sinc_params(self) -> SincInterpolationParameters {
        let (sinc_len, oversampling_factor) = match self {
            ResamplerQuality::High => (256, 256),
            ResamplerQuality::Medium => (128, 128),
            ResamplerQuality::Low => (64, 64),
        };
        SincInterpolationParameters {
            sinc_len,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor,
            window: WindowFunction::BlackmanHarris2,
        }
    }
}

/// One-shot block resampler: constructs a fresh `rubato::SincFixedIn` sized
/// to the input length on every call, mirroring the Python
/// resample-then-clear pattern described above. `input_rate`/`output_rate`
/// need only be proportionally correct (only their ratio is used).
pub struct BlockResampler {
    ratio: f64,
    quality: ResamplerQuality,
}

impl BlockResampler {
    pub fn new(input_rate: f64, output_rate: f64, quality: ResamplerQuality) -> Self {
        assert!(input_rate > 0.0 && output_rate > 0.0);
        BlockResampler {
            ratio: output_rate / input_rate,
            quality,
        }
    }

    /// Resamples the whole of `input` in one call. Returns an empty vec for
    /// empty input (rubato requires a non-zero chunk size).
    pub fn resample(&self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }
        let mut resampler = SincFixedIn::<f32>::new(
            self.ratio,
            1.0,
            self.quality.sinc_params(),
            input.len(),
            1, // mono, matching Python's per-channel ResampleStream instances
        )
        .expect("resampler construction failed");

        let waveform_in = [input.to_vec()];
        let mut out = resampler
            .process(&waveform_in, None)
            .expect("resample failed");
        out.pop().expect("mono resampler returns one channel")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn downsamples_a_tone_preserving_its_frequency() {
        let fs_in = 192_000.0;
        let fs_out = 48_000.0;
        let tone_hz = 1_000.0;
        let n = 9600usize; // 50ms at fs_in

        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * tone_hz * i as f64 / fs_in).sin() as f32)
            .collect();

        let resampler = BlockResampler::new(fs_in, fs_out, ResamplerQuality::High);
        let output = resampler.resample(&input);

        // Ratio 1:4, so output should be close to n/4 samples.
        let expected_len = (n as f64 * fs_out / fs_in) as usize;
        assert!(
            output.len().abs_diff(expected_len) < 50,
            "output len {} vs expected ~{expected_len}",
            output.len()
        );

        // Goertzel at 1kHz should dominate over a a nearby off-tone bin,
        // skipping the filter's startup transient at the very start.
        let usable = &output[200..output.len() - 1];
        let goertzel = |freq: f64| -> f64 {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for (i, &sample) in usable.iter().enumerate() {
                let angle = -2.0 * PI * freq * (i as f64) / fs_out;
                re += sample as f64 * angle.cos();
                im += sample as f64 * angle.sin();
            }
            (re * re + im * im).sqrt()
        };
        let at_tone = goertzel(tone_hz);
        let off_tone = goertzel(tone_hz * 1.7);
        assert!(
            at_tone > off_tone * 5.0,
            "tone bin {at_tone} not dominant over off-tone bin {off_tone}"
        );
    }
}
