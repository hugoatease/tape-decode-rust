//! HiFi FM audio decoder core: AFE carrier filtering, FM demodulation,
//! resampling, and post-processing. Port of `vhsdecode.hifi.HiFiDecode` /
//! `PostProcessor` (Python, in the `vhs-decode` repository).

mod afe;
mod demod;
mod dropout;
mod postprocess;
mod resample;
mod stereo;

pub use afe::{field_rate, AfeFilter, AfeOverrides, AfeParams, System, TapeFormat};
pub use demod::FmDiscriminator;
pub use dropout::{cancel_dc_trim, dropout_compensate, DropoutParams};
pub use postprocess::{DcBlocker, Deemphasis, EightMmPostProcess, EnvDetection, Expander, PostProcessParams, VhsPostProcess};
pub use resample::{BlockResampler, ResamplerQuality};
pub use stereo::{mix_for_mode_stereo, DecodeMode};
