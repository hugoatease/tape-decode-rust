//! Single-shot decode orchestration: wires the `hifi-decode` DSP stages
//! together in `HiFiDecode.block_decode` / `PostProcessor`'s order.
//!
//! **Known limitation**: unlike the real pipeline (which streams the input
//! in ~0.5s blocks with overlap, per `HiFiDecode._set_block_overlap`),
//! this decodes the *entire* input as one buffer. That's block-size/
//! overlap chunking, explicitly deferred to the concurrency step of the
//! hifi-decode port plan (it's an orchestration concern, not a DSP one) —
//! this pipeline is what that step will restructure into a streaming,
//! multi-threaded one. For now this means: correct output, but the whole
//! input is held in memory, and there is no block-boundary parallelism.

use anyhow::{bail, Result};
use hifi_decode::{
    cancel_dc_trim, dropout_compensate, headswitch_remove_noise, mix_for_mode_stereo, AfeFilter,
    AfeOverrides, AfeParams, BlockResampler, DcBlocker, DecodeMode, DropoutParams,
    EightMmPostProcess, FmDiscriminator, HeadswitchParams, PostProcessParams, ResamplerQuality,
    System, TapeFormat, VhsPostProcess,
};

const AUDIO_RATE_INTERMEDIATE: f64 = 192_000.0;
const PRE_TRIM: usize = 1000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DemodType {
    Quadrature,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DocMode {
    Full,
    Mute,
    Disabled,
}

pub struct PipelineParams {
    pub input_rate: f64,
    pub format: TapeFormat,
    pub system: System,
    pub afe_overrides: AfeOverrides,
    pub demod_type: DemodType,
    pub resampler_quality: ResamplerQuality,
    pub audio_final_rate: f64,
    pub gain: f64,
    pub mode: DecodeMode,
    pub head_switching_interpolation: bool,
    pub doc_mode: DocMode,
    pub enable_deemphasis: bool,
    pub enable_expander: bool,
    pub post_process: PostProcessParams,
}

/// Decodes the whole of `rf` (one channel of raw RF samples) to a stereo
/// (or mono, for the `l`/`r`/`sum` decode modes) pair of final-rate audio
/// buffers. See the module doc comment for the block-chunking limitation.
pub fn decode(rf: &[f32], params: &PipelineParams) -> Result<(Vec<f32>, Vec<f32>)> {
    if params.demod_type != DemodType::Quadrature {
        bail!("only quadrature demodulation is implemented in this port; the Hilbert path is not yet ported");
    }

    let afe_params = AfeParams::for_format(params.format, params.system, params.afe_overrides);

    // hifi-decode always demodulates both channels, even in mono l/r mode
    // — unlike Python, which skips the unused channel's demod as an
    // optimization (`decode_mode != AUDIO_MODE_MONO_R` guards throughout
    // `block_decode`). Simpler code, correct output, more CPU than
    // strictly necessary in mono mode.
    let afe_l = AfeFilter::design(afe_params.l_carrier_ref, afe_params.l_notch_width, params.input_rate);
    let afe_r = AfeFilter::design(afe_params.r_carrier_ref, afe_params.r_notch_width, params.input_rate);
    let filtered_l = afe_l.work(rf);
    let filtered_r = afe_r.work(rf);

    let disc_l = FmDiscriminator::new_quadrature(params.input_rate, afe_params.l_carrier_ref, afe_params.l_carrier_deviation, filtered_l.len());
    let disc_r = FmDiscriminator::new_quadrature(params.input_rate, afe_params.r_carrier_ref, afe_params.r_carrier_deviation, filtered_r.len());
    let demod_l = disc_l.work(&filtered_l);
    let demod_r = disc_r.work(&filtered_r);

    let audio_resampler_l = BlockResampler::new(params.input_rate, AUDIO_RATE_INTERMEDIATE, params.resampler_quality);
    let audio_resampler_r = BlockResampler::new(params.input_rate, AUDIO_RATE_INTERMEDIATE, params.resampler_quality);
    let mut audio_l = audio_resampler_l.resample(&demod_l);
    let mut audio_r = audio_resampler_r.resample(&demod_r);

    let trim_l = PRE_TRIM.min(audio_l.len() / 2 - 1);
    let trim_r = PRE_TRIM.min(audio_r.len() / 2 - 1);
    cancel_dc_trim(&mut audio_l, trim_l);
    cancel_dc_trim(&mut audio_r, trim_r);

    if params.doc_mode != DocMode::Disabled {
        let doc_params = DropoutParams::new(AUDIO_RATE_INTERMEDIATE);
        dropout_compensate(&mut audio_l, &mut audio_r, &doc_params, params.mode, params.doc_mode == DocMode::Mute);
    }

    let field_rate = hifi_decode::field_rate(params.system);
    let (mut audio_l, mut audio_r) = if params.head_switching_interpolation {
        let hs_params = HeadswitchParams::new(AUDIO_RATE_INTERMEDIATE, field_rate);
        (headswitch_remove_noise(&audio_l, &hs_params), headswitch_remove_noise(&audio_r, &hs_params))
    } else {
        (audio_l, audio_r)
    };

    let final_resampler_l = BlockResampler::new(AUDIO_RATE_INTERMEDIATE, params.audio_final_rate, params.resampler_quality);
    let final_resampler_r = BlockResampler::new(AUDIO_RATE_INTERMEDIATE, params.audio_final_rate, params.resampler_quality);
    if AUDIO_RATE_INTERMEDIATE != params.audio_final_rate {
        audio_l = final_resampler_l.resample(&audio_l);
        audio_r = final_resampler_r.resample(&audio_r);
    }

    let (mut pre_l, mut pre_r) = mix_for_mode_stereo(&audio_l, &audio_r, params.mode);

    if params.gain != 1.0 {
        for sample in pre_l.iter_mut().chain(pre_r.iter_mut()) {
            *sample *= params.gain as f32;
        }
    }

    let mut dc_blocker_l = DcBlocker::new(params.audio_final_rate, 1.0);
    let mut dc_blocker_r = DcBlocker::new(params.audio_final_rate, 1.0);
    dc_blocker_l.process(&mut pre_l);
    dc_blocker_r.process(&mut pre_r);

    let mut post_l = pre_l.clone();
    let mut post_r = pre_r.clone();

    if params.format == TapeFormat::Video8 {
        let mut chain_l = EightMmPostProcess::new(params.audio_final_rate, params.post_process, params.enable_deemphasis, params.enable_expander);
        let mut chain_r = EightMmPostProcess::new(params.audio_final_rate, params.post_process, params.enable_deemphasis, params.enable_expander);
        chain_l.process(&mut pre_l, &mut post_l, true);
        chain_r.process(&mut pre_r, &mut post_r, true);
    } else {
        let mut chain_l = VhsPostProcess::new(params.audio_final_rate, params.post_process, params.enable_deemphasis, params.enable_expander);
        let mut chain_r = VhsPostProcess::new(params.audio_final_rate, params.post_process, params.enable_deemphasis, params.enable_expander);
        chain_l.process(&mut pre_l, &mut post_l, true);
        chain_r.process(&mut pre_r, &mut post_r, true);
    }

    Ok((post_l, post_r))
}
