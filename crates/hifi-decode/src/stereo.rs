//! Stereo/dual-mono decode mode and the L/R mixing it drives. Ports the
//! `AUDIO_MODE_*` constants and `mix_for_mode_stereo`
//! (`HiFiDecode.py:2127-2152`).

/// One of hifi-decode's 7 `--audio_mode` values (`constants.py:6-12`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeMode {
    /// `s`: L, R passthrough.
    Stereo,
    /// `ms`: Mid/Side (L+R, L-R).
    StereoMs,
    /// `d`: L, R passthrough, written as two mono files.
    DualMono,
    /// `dms`: Mid/Side on a dual-mono source.
    DualMonoMs,
    /// `l`: both outputs are the left channel.
    MonoL,
    /// `r`: both outputs are the right channel.
    MonoR,
    /// `sum`: both outputs are (L+R)/2.
    MonoSum,
}

impl DecodeMode {
    /// Both `d` and `dms` skip the other-channel dropout fill-in and are
    /// written as two separate mono files downstream.
    pub fn is_dual_mono(self) -> bool {
        matches!(self, DecodeMode::DualMono | DecodeMode::DualMonoMs)
    }

    pub fn uses_right_channel(self) -> bool {
        !matches!(self, DecodeMode::MonoL)
    }

    pub fn uses_left_channel(self) -> bool {
        !matches!(self, DecodeMode::MonoR)
    }
}

/// `mix_for_mode_stereo` (`HiFiDecode.py:2127-2152`). Runs *before*
/// post-processing (deemphasis/expander), so in `ms`/`dms` modes those
/// later stages operate on Mid/Side, not true L/R — that's intentional,
/// not a bug to "fix" when wiring this up.
pub fn mix_for_mode_stereo(l_raw: &[f32], r_raw: &[f32], mode: DecodeMode) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(l_raw.len(), r_raw.len());
    match mode {
        DecodeMode::StereoMs | DecodeMode::DualMonoMs => {
            let l = l_raw.iter().zip(r_raw).map(|(&a, &b)| (a + b) * 0.5).collect();
            let r = l_raw.iter().zip(r_raw).map(|(&a, &b)| (a - b) * 0.5).collect();
            (l, r)
        }
        DecodeMode::MonoL => (l_raw.to_vec(), l_raw.to_vec()),
        DecodeMode::MonoR => (r_raw.to_vec(), r_raw.to_vec()),
        DecodeMode::MonoSum => {
            let mixed: Vec<f32> = l_raw.iter().zip(r_raw).map(|(&a, &b)| (a + b) * 0.5).collect();
            (mixed.clone(), mixed)
        }
        DecodeMode::Stereo | DecodeMode::DualMono => (l_raw.to_vec(), r_raw.to_vec()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stereo_is_passthrough() {
        let l = vec![1.0, 2.0, 3.0];
        let r = vec![4.0, 5.0, 6.0];
        let (gl, gr) = mix_for_mode_stereo(&l, &r, DecodeMode::Stereo);
        assert_eq!(gl, l);
        assert_eq!(gr, r);
    }

    #[test]
    fn ms_mode_matrixes() {
        let l = vec![1.0, 2.0];
        let r = vec![0.5, 0.5];
        let (mid, side) = mix_for_mode_stereo(&l, &r, DecodeMode::StereoMs);
        assert_eq!(mid, vec![0.75, 1.25]);
        assert_eq!(side, vec![0.25, 0.75]);
    }
}
