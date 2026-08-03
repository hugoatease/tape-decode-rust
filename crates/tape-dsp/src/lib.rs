#![cfg_attr(nightly_portable_simd, feature(portable_simd))]

//! Generic, video-agnostic DSP kernels shared between `tape-decode` (video)
//! and `hifi-decode` (audio): zero-phase SOS filtering, FM-discriminator
//! phase unwrapping, real-FFT helpers, and branch-free vector math.

mod cheby2;
mod fast_math;
mod fft;
mod sosfiltfilt;
mod unwrap_angles;

pub use cheby2::cheby2_sos;
pub use fast_math::{exp_fast, powf_fast_nonneg, sum_algebraic};
pub use fft::{irfft_f32, irfft_owned_f32, rfft_f32, rfft_owned_f32};
pub use sosfiltfilt::{narrow_sos, sosfilt_f32, sosfiltfilt_f32};
pub use unwrap_angles::unwrap_angles;
