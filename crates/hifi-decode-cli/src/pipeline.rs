//! Block-parallel decode orchestration: wires the `hifi-decode` DSP stages
//! together in `HiFiDecode.block_decode` / `PostProcessor`'s order, but
//! chunked into overlapping ~0.5s blocks (`hifi_decode::BlockLayout`,
//! ported from `HiFiDecode._set_block_overlap`) that decode in parallel,
//! matching the plan's "threads + channels, one process" architecture —
//! Python's shared-memory/multiprocess machinery has no Rust equivalent
//! needed, since these threads share one address space.
//!
//! What's parallel vs sequential, and why:
//! - AFE filter -> demod -> resample -> DC trim -> DOC -> head-switch ->
//!   final resample -> stereo mix (`decode_one_block`) is **pure per
//!   block**: nothing in it carries state across blocks (the AFE/FM
//!   objects are stateless `work()` calls; DOC and head-switch each look
//!   only within their own block). Runs on a worker-thread pool, one
//!   block per task, order-independent.
//! - Each block's *output* is trimmed to its non-overlap span
//!   (`Block::output_skip`/`output_take`) and blocks are concatenated
//!   **in order** afterward — cheap, and this is where correctness
//!   depends on block order, not on parallel execution order.
//! - `DcBlocker`/`Deemphasis`/`Expander` (`PostProcessParams` chain) carry
//!   real IIR state across the whole stream, so they run **once**, after
//!   concatenation, over the full signal — sequential by construction.
//!   This also sidesteps Python's per-block "prime the state on block 0"
//!   dance entirely: priming the chain once at the very start of the
//!   concatenated signal is equivalent to Python's block-by-block priming,
//!   since both operate on the same contiguous, correctly-ordered stream.

use std::thread;

use anyhow::{bail, Result};
use hifi_decode::{
    cancel_dc_trim, dropout_compensate, headswitch_remove_noise, mix_for_mode_stereo, AfeFilter,
    AfeOverrides, AfeParams, Block, BlockLayout, BlockResampler, DcBlocker, DecodeMode,
    DropoutParams, EightMmPostProcess, FmDiscriminator, HeadswitchParams, PostProcessParams,
    ResamplerQuality, System, TapeFormat, VhsPostProcess,
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

/// Per-block DSP: AFE bandpass -> quadrature demod -> resample to
/// 192kHz -> DC trim -> DOC -> head-switch cleanup -> resample to the
/// final rate -> stereo mix. Everything here is a pure function of
/// `rf_block` plus the shared, read-only, block-independent objects
/// passed in — no state is carried between calls, so this is safe to run
/// concurrently across blocks.
#[allow(clippy::too_many_arguments)]
fn decode_one_block(
    rf_block: &[f32],
    afe_l: &AfeFilter,
    afe_r: &AfeFilter,
    disc_l: &FmDiscriminator,
    disc_r: &FmDiscriminator,
    doc_params: Option<&DropoutParams>,
    hs_params: Option<&HeadswitchParams>,
    params: &PipelineParams,
) -> (Vec<f32>, Vec<f32>) {
    let filtered_l = afe_l.work(rf_block);
    let filtered_r = afe_r.work(rf_block);
    let demod_l = disc_l.work(&filtered_l);
    let demod_r = disc_r.work(&filtered_r);

    let audio_resampler_l = BlockResampler::new(params.input_rate, AUDIO_RATE_INTERMEDIATE, params.resampler_quality);
    let audio_resampler_r = BlockResampler::new(params.input_rate, AUDIO_RATE_INTERMEDIATE, params.resampler_quality);
    let mut audio_l = audio_resampler_l.resample(&demod_l);
    let mut audio_r = audio_resampler_r.resample(&demod_r);

    let trim_l = PRE_TRIM.min(audio_l.len().saturating_sub(1) / 2);
    let trim_r = PRE_TRIM.min(audio_r.len().saturating_sub(1) / 2);
    if trim_l > 0 {
        cancel_dc_trim(&mut audio_l, trim_l);
    }
    if trim_r > 0 {
        cancel_dc_trim(&mut audio_r, trim_r);
    }

    if let Some(doc_params) = doc_params {
        dropout_compensate(&mut audio_l, &mut audio_r, doc_params, params.mode, params.doc_mode == DocMode::Mute);
    }

    let (mut audio_l, mut audio_r) = match hs_params {
        Some(hs_params) => (headswitch_remove_noise(&audio_l, hs_params), headswitch_remove_noise(&audio_r, hs_params)),
        None => (audio_l, audio_r),
    };

    let final_resampler_l = BlockResampler::new(AUDIO_RATE_INTERMEDIATE, params.audio_final_rate, params.resampler_quality);
    let final_resampler_r = BlockResampler::new(AUDIO_RATE_INTERMEDIATE, params.audio_final_rate, params.resampler_quality);
    if AUDIO_RATE_INTERMEDIATE != params.audio_final_rate {
        audio_l = final_resampler_l.resample(&audio_l);
        audio_r = final_resampler_r.resample(&audio_r);
    }

    mix_for_mode_stereo(&audio_l, &audio_r, params.mode)
}

/// Runs `decode_one_block` for every block in `layout.blocks(rf.len())`
/// across a worker-thread pool (bounded by available parallelism), then
/// trims each block's output to its non-overlap span and concatenates in
/// order.
fn decode_blocks_parallel(
    rf: &[f32],
    layout: &BlockLayout,
    afe_l: &AfeFilter,
    afe_r: &AfeFilter,
    disc_l: &FmDiscriminator,
    disc_r: &FmDiscriminator,
    doc_params: Option<&DropoutParams>,
    hs_params: Option<&HeadswitchParams>,
    params: &PipelineParams,
) -> (Vec<f32>, Vec<f32>) {
    let blocks = layout.blocks(rf.len());
    let mut results: Vec<Option<(Vec<f32>, Vec<f32>)>> = (0..blocks.len()).map(|_| None).collect();

    let worker_count = thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(blocks.len().max(1));
    let chunk_size = blocks.len().div_ceil(worker_count.max(1)).max(1);

    thread::scope(|scope| {
        for (block_chunk, result_chunk) in blocks.chunks(chunk_size).zip(results.chunks_mut(chunk_size)) {
            scope.spawn(move || {
                for (block, slot) in block_chunk.iter().zip(result_chunk.iter_mut()) {
                    let rf_block = &rf[block.read_start..block.read_end];
                    *slot = Some(decode_one_block(rf_block, afe_l, afe_r, disc_l, disc_r, doc_params, hs_params, params));
                }
            });
        }
    });

    let mut audio_l = Vec::new();
    let mut audio_r = Vec::new();
    for (block, result) in blocks.iter().zip(results) {
        let (block_l, block_r) = result.expect("every block slot was filled by a worker");
        append_trimmed(&mut audio_l, &block_l, block);
        append_trimmed(&mut audio_r, &block_r, block);
    }
    (audio_l, audio_r)
}

fn append_trimmed(dest: &mut Vec<f32>, block_output: &[f32], block: &Block) {
    let len = block_output.len();
    let skip = block.output_skip.min(len);
    let take = block.output_take.min(len - skip);
    dest.extend_from_slice(&block_output[skip..skip + take]);
}

/// Decodes the whole of `rf` (one channel of raw RF samples) to a stereo
/// (or mono, for the `l`/`r`/`sum` decode modes) pair of final-rate audio
/// buffers.
pub fn decode(rf: &[f32], params: &PipelineParams) -> Result<(Vec<f32>, Vec<f32>)> {
    if params.demod_type != DemodType::Quadrature {
        bail!("only quadrature demodulation is implemented in this port; the Hilbert path is not yet ported");
    }

    let afe_params = AfeParams::for_format(params.format, params.system, params.afe_overrides);
    let layout = BlockLayout::new(params.input_rate, AUDIO_RATE_INTERMEDIATE, params.audio_final_rate);

    // hifi-decode always demodulates both channels, even in mono l/r mode
    // — unlike Python, which skips the unused channel's demod as an
    // optimization (`decode_mode != AUDIO_MODE_MONO_R` guards throughout
    // `block_decode`). Simpler code, correct output, more CPU than
    // strictly necessary in mono mode.
    let afe_l = AfeFilter::design(afe_params.l_carrier_ref, afe_params.l_notch_width, params.input_rate);
    let afe_r = AfeFilter::design(afe_params.r_carrier_ref, afe_params.r_notch_width, params.input_rate);

    // Sized from the nominal (non-last) block, matching Python's
    // FMDiscriminator construction against `initialBlockResampledSize`
    // (built once per channel in `HiFiDecode.__init__`, reused for every
    // block including a shorter last one).
    let disc_l = FmDiscriminator::new_quadrature(params.input_rate, afe_params.l_carrier_ref, afe_params.l_carrier_deviation, layout.block_size);
    let disc_r = FmDiscriminator::new_quadrature(params.input_rate, afe_params.r_carrier_ref, afe_params.r_carrier_deviation, layout.block_size);

    let doc_params = (params.doc_mode != DocMode::Disabled).then(|| DropoutParams::new(AUDIO_RATE_INTERMEDIATE));
    let hs_params = params
        .head_switching_interpolation
        .then(|| HeadswitchParams::new(AUDIO_RATE_INTERMEDIATE, hifi_decode::field_rate(params.system)));

    let (audio_l, audio_r) = decode_blocks_parallel(rf, &layout, &afe_l, &afe_r, &disc_l, &disc_r, doc_params.as_ref(), hs_params.as_ref(), params);

    let (mut pre_l, mut pre_r) = (audio_l, audio_r);

    if params.gain != 1.0 {
        for sample in pre_l.iter_mut().chain(pre_r.iter_mut()) {
            *sample *= params.gain as f32;
        }
    }

    // DC blocking and de-emphasis/expansion carry continuous IIR state, so
    // they run once, sequentially, over the whole concatenated signal —
    // see the module doc comment.
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
