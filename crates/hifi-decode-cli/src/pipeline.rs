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

use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::{bail, Result};
use hifi_decode::{
    cancel_dc_trim, dropout_compensate, headswitch_remove_noise, mix_for_mode_stereo, AfeFilter,
    AfeOverrides, AfeParams, Block, BlockLayout, BlockResampler, DcBlocker, DecodeMode,
    DropoutParams, EightMmPostProcess, FmDiscriminator, HeadswitchParams, PostProcessParams,
    ResamplerQuality, System, TapeFormat, VhsPostProcess,
};
use tape_rf_io::DecodeReader;

use crate::stream::StreamingBlocks;

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

type BlockResult = (usize, Block, Vec<f32>, Vec<f32>);

/// Streams RF blocks from `reader` through a worker-thread pool (bounded
/// by available parallelism), reorders their outputs back into submission
/// order, trims each to its non-overlap span, and concatenates.
///
/// This is a small pipeline, not a pre-sliced parallel loop, specifically
/// so the RF input never has to be fully read into memory: a reader
/// "thread" (actually just this function's caller, driving
/// `StreamingBlocks` — see below) produces blocks one at a time and hands
/// them to workers over a bounded channel; workers decode independently
/// (nothing here carries state across blocks — see `decode_one_block`'s
/// doc comment) and send results back over a second channel; this
/// function's caller thread reorders and trims as results arrive. Only
/// the reordering buffer (`pending`, bounded by how far worker completion
/// order can drift from submission order — in practice a handful of
/// blocks) and the growing decoded-audio output live in memory alongside
/// whatever's in flight; the RF window itself is bounded to a couple of
/// blocks by `StreamingBlocks`.
#[allow(clippy::too_many_arguments)]
fn decode_blocks_streaming(
    reader: &mut DecodeReader,
    layout: &BlockLayout,
    afe_l: &AfeFilter,
    afe_r: &AfeFilter,
    disc_l: &FmDiscriminator,
    disc_r: &FmDiscriminator,
    doc_params: Option<&DropoutParams>,
    hs_params: Option<&HeadswitchParams>,
    params: &PipelineParams,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let worker_count = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    // A handful of blocks of slack per worker: enough that a worker never
    // starves waiting for input, without letting the reader race arbitrarily
    // far ahead of decoding (which would defeat the point of streaming).
    let queue_depth = (worker_count * 2).max(2);

    let (block_tx, block_rx): (SyncSender<(usize, Block, Vec<f32>)>, Receiver<_>) = sync_channel(queue_depth);
    let (result_tx, result_rx): (SyncSender<BlockResult>, Receiver<BlockResult>) = sync_channel(queue_depth);
    let block_rx = Arc::new(Mutex::new(block_rx));

    let mut audio_l = Vec::new();
    let mut audio_r = Vec::new();

    thread::scope(|scope| {
        for _ in 0..worker_count {
            let block_rx = Arc::clone(&block_rx);
            let result_tx = result_tx.clone();
            scope.spawn(move || loop {
                let received = { block_rx.lock().expect("block queue mutex poisoned").recv() };
                let Ok((index, block, rf_block)) = received else {
                    break;
                };
                let (l, r) = decode_one_block(&rf_block, afe_l, afe_r, disc_l, disc_r, doc_params, hs_params, params);
                if result_tx.send((index, block, l, r)).is_err() {
                    break;
                }
            });
        }
        drop(result_tx);

        scope.spawn(move || {
            let mut streamer = StreamingBlocks::new(reader, *layout);
            let mut index = 0usize;
            loop {
                match streamer.next_block() {
                    Ok(Some((block, rf_block))) => {
                        if block_tx.send((index, block, rf_block)).is_err() {
                            break; // workers gone (e.g. panicked) — stop reading
                        }
                        index += 1;
                    }
                    Ok(None) => break,
                    Err(_) => break, // surfaced below via the collector never completing all indices; see note
                }
            }
        });

        // Collector: reorder by index as results arrive, trim, concatenate.
        let mut pending: std::collections::HashMap<usize, (Block, Vec<f32>, Vec<f32>)> = std::collections::HashMap::new();
        let mut next_expected = 0usize;
        while let Ok((index, block, l, r)) = result_rx.recv() {
            pending.insert(index, (block, l, r));
            while let Some((block, l, r)) = pending.remove(&next_expected) {
                append_trimmed(&mut audio_l, &l, &block);
                append_trimmed(&mut audio_r, &r, &block);
                next_expected += 1;
            }
        }
        // `pending` non-empty here would mean a block result never arrived
        // for some index below `next_expected`'s ceiling — in practice
        // unreachable: `tape_rf_io::DecodeReader` never surfaces read
        // errors as `Err` (it logs and treats them as EOF, matching this
        // pipeline's pre-streaming behavior), and a worker panic aborts
        // the whole scope via `thread::scope`'s own propagation before
        // this code runs at all. Kept as a debug assertion rather than
        // silently ignored.
        debug_assert!(pending.is_empty(), "block(s) {:?} never arrived in order", {
            let mut missing: Vec<_> = pending.keys().copied().collect();
            missing.sort_unstable();
            missing
        });
    });

    Ok((audio_l, audio_r))
}

fn append_trimmed(dest: &mut Vec<f32>, block_output: &[f32], block: &Block) {
    let len = block_output.len();
    let skip = block.output_skip.min(len);
    let take = block.output_take.min(len - skip);
    dest.extend_from_slice(&block_output[skip..skip + take]);
}

/// Decodes the whole of `reader`'s RF input to a stereo (or mono, for the
/// `l`/`r`/`sum` decode modes) pair of final-rate audio buffers. Streams
/// the input (see `decode_blocks_streaming`/`crate::stream`) rather than
/// reading it into memory upfront — the only thing this function itself
/// holds in full is the *decoded* audio, which for even an hour-long tape
/// is a few hundred MB, unlike the multi-tens-of-GB raw RF stream.
pub fn decode(reader: &mut DecodeReader, params: &PipelineParams) -> Result<(Vec<f32>, Vec<f32>)> {
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

    let (audio_l, audio_r) = decode_blocks_streaming(reader, &layout, &afe_l, &afe_r, &disc_l, &disc_r, doc_params.as_ref(), hs_params.as_ref(), params)?;

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

    // Prime the expander over exactly one nominal block's worth of
    // samples, matching Python's block-0-only priming scope — see
    // VhsPostProcess::process's doc comment for why priming over the
    // whole stream (this used to pass `true` unconditionally) is a bug,
    // not just wasted work.
    let prime_len = Some(layout.block_audio_final_size);

    if params.format == TapeFormat::Video8 {
        let mut chain_l = EightMmPostProcess::new(params.audio_final_rate, params.post_process, params.enable_deemphasis, params.enable_expander);
        let mut chain_r = EightMmPostProcess::new(params.audio_final_rate, params.post_process, params.enable_deemphasis, params.enable_expander);
        chain_l.process(&mut pre_l, &mut post_l, prime_len);
        chain_r.process(&mut pre_r, &mut post_r, prime_len);
    } else {
        let mut chain_l = VhsPostProcess::new(params.audio_final_rate, params.post_process, params.enable_deemphasis, params.enable_expander);
        let mut chain_r = VhsPostProcess::new(params.audio_final_rate, params.post_process, params.enable_deemphasis, params.enable_expander);
        chain_l.process(&mut pre_l, &mut post_l, prime_len);
        chain_r.process(&mut pre_r, &mut post_r, prime_len);
    }

    Ok((post_l, post_r))
}
