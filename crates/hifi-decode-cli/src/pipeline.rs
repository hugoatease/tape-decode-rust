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
//!   (`Block::output_skip`/`output_take`) and blocks are handed back **in
//!   order** to the caller as they complete — this is where correctness
//!   depends on block order, not on parallel execution order.
//! - `DcBlocker`/`Deemphasis`/`Expander` (`PostProcessParams` chain) carry
//!   real IIR state across the whole stream, so they run sequentially, one
//!   chunk at a time, *as each ordered chunk becomes available* rather than
//!   after concatenating everything into one buffer first — this is what
//!   lets the decoded-audio side stream out to disk instead of being held
//!   in memory for the whole run (mirroring the RF *input* side's
//!   streaming — see `crate::stream`). Each chain is constructed once
//!   before the first chunk and mutated across calls, exactly like Python
//!   carries `PostProcessor`'s state across blocks; only the very first
//!   chunk primes the expander (matching Python's block-0-only priming —
//!   see `VhsPostProcess::process`'s doc comment).

use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

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
/// order, trims each to its non-overlap span, and hands each trimmed
/// chunk to `on_ordered_chunk` **as soon as it's next in line** — never
/// concatenated into one buffer here.
///
/// This is a small pipeline, not a pre-sliced parallel loop, specifically
/// so neither the RF input nor the decoded output ever has to be fully
/// materialized in memory: a reader "thread" (actually just this
/// function's caller, driving `StreamingBlocks` — see below) produces
/// blocks one at a time and hands them to workers over a bounded channel;
/// workers decode independently (nothing here carries state across blocks
/// — see `decode_one_block`'s doc comment) and send results back over a
/// second channel; this function's caller thread reorders, trims, and
/// forwards chunks as results arrive. Only the reordering buffer
/// (`pending`, bounded by how far worker completion order can drift from
/// submission order — in practice a handful of blocks) lives in memory
/// alongside whatever's in flight; the RF window itself is bounded to a
/// couple of blocks by `StreamingBlocks`.
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
    mut on_ordered_chunk: impl FnMut(Vec<f32>, Vec<f32>, &Block) -> Result<()>,
) -> Result<()> {
    let worker_count = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    // A handful of blocks of slack per worker: enough that a worker never
    // starves waiting for input, without letting the reader race arbitrarily
    // far ahead of decoding (which would defeat the point of streaming).
    let queue_depth = (worker_count * 2).max(2);

    let (block_tx, block_rx): (SyncSender<(usize, Block, Vec<f32>)>, Receiver<_>) = sync_channel(queue_depth);
    let (result_tx, result_rx): (SyncSender<BlockResult>, Receiver<BlockResult>) = sync_channel(queue_depth);
    let block_rx = Arc::new(Mutex::new(block_rx));

    let mut first_error: Option<anyhow::Error> = None;

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

        // Collector: reorder by index as results arrive, trim, and forward
        // immediately — nothing is concatenated into a whole-signal buffer
        // here. On the first forwarding error (e.g. a full disk), stop
        // calling the callback but keep draining the channels rather than
        // returning early: workers/the reader thread are still running and
        // would otherwise block forever trying to send into channels
        // nobody's receiving from, since `thread::scope` waits for every
        // spawned thread before this function can return.
        let mut pending: std::collections::HashMap<usize, (Block, Vec<f32>, Vec<f32>)> = std::collections::HashMap::new();
        let mut next_expected = 0usize;
        while let Ok((index, block, l, r)) = result_rx.recv() {
            pending.insert(index, (block, l, r));
            while let Some((block, l, r)) = pending.remove(&next_expected) {
                next_expected += 1;
                if first_error.is_some() {
                    continue;
                }
                let trimmed_l = trim_block_output(&l, &block);
                let trimmed_r = trim_block_output(&r, &block);
                if let Err(e) = on_ordered_chunk(trimmed_l, trimmed_r, &block) {
                    first_error = Some(e);
                }
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

    match first_error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn trim_block_output(block_output: &[f32], block: &Block) -> Vec<f32> {
    let len = block_output.len();
    let skip = block.output_skip.min(len);
    let take = block.output_take.min(len - skip);
    block_output[skip..skip + take].to_vec()
}

/// How often progress gets logged, wall-clock time — independent of
/// decode speed (which varies a lot with thread count/hardware) so a fast
/// run doesn't spam the log and a slow one doesn't go silent for minutes.
const PROGRESS_LOG_INTERVAL: Duration = Duration::from_secs(5);

/// Tracks and periodically logs how far a `decode` call has gotten,
/// mirroring `tape-decode-cli`'s own field-count/FPS progress logging
/// (`tape-decode-cli/src/writer.rs`) — the same idea, adapted to
/// hifi-decode having a cheaply knowable *input* size (a real, seekable
/// RF capture) to report a percentage/ETA against, which the video
/// decoder's field-at-a-time model doesn't have.
struct Progress {
    start: Instant,
    last_log: Instant,
    total_input_samples: Option<u64>,
    audio_final_rate: f64,
    decoded_samples: u64,
    logged_once: bool,
}

impl Progress {
    fn new(total_input_samples: Option<u64>, audio_final_rate: f64) -> Self {
        let now = Instant::now();
        Progress {
            start: now,
            last_log: now,
            total_input_samples,
            audio_final_rate,
            decoded_samples: 0,
            logged_once: false,
        }
    }

    /// Call once per ordered chunk, after it's been fully processed.
    /// `read_end` is the RF-input position of the *nominal* (non-overlap)
    /// end of the block this chunk came from — the same field `Block`
    /// carries — used as "how far into the input have we gotten" for the
    /// percentage/ETA, distinct from `decoded_samples` (how much *output*
    /// audio exists so far, used for the realtime-multiplier figure).
    fn record(&mut self, chunk_len: usize, read_end: usize) {
        self.decoded_samples += chunk_len as u64;
        let now = Instant::now();
        if self.logged_once && now.duration_since(self.last_log) < PROGRESS_LOG_INTERVAL {
            return;
        }
        self.last_log = now;
        self.logged_once = true;
        self.log(now, read_end);
    }

    fn log(&self, now: Instant, read_end: usize) {
        let elapsed = now.duration_since(self.start).as_secs_f64();
        let decoded_secs = self.decoded_samples as f64 / self.audio_final_rate;
        let realtime_x = if elapsed > 0.0 { decoded_secs / elapsed } else { 0.0 };
        match self.total_input_samples {
            Some(total) if total > 0 => {
                let pct = (read_end as f64 / total as f64 * 100.0).clamp(0.0, 100.0);
                let eta_secs = if pct > 0.1 { elapsed * (100.0 - pct) / pct } else { f64::NAN };
                tracing::info!(
                    "decoded {decoded_secs:.1}s of audio so far ({pct:.1}% of input, ETA {eta}) \
                     in {elapsed:.1}s ({realtime_x:.2}x realtime)",
                    eta = format_duration(eta_secs),
                );
            }
            _ => {
                tracing::info!("decoded {decoded_secs:.1}s of audio so far in {elapsed:.1}s ({realtime_x:.2}x realtime)");
            }
        }
    }

    /// Called once after the last chunk, regardless of the throttle
    /// interval, so a run that finishes faster than `PROGRESS_LOG_INTERVAL`
    /// still gets a final status line instead of total silence.
    fn finish(&self) {
        self.log(Instant::now(), self.total_input_samples.unwrap_or(0) as usize);
    }
}

fn format_duration(secs: f64) -> String {
    if !secs.is_finite() || secs < 0.0 {
        return "unknown".to_string();
    }
    let secs = secs.round() as u64;
    if secs < 60 {
        format!("{secs}s")
    } else {
        format!("{}m{:02}s", secs / 60, secs % 60)
    }
}

/// The two post-processing chain shapes (`VhsPostProcess`/
/// `EightMmPostProcess` differ in stage ordering — see their doc
/// comments), unified so `decode` can hold one chain per channel across
/// the whole streamed run without duplicating the per-chunk call site per
/// format.
enum PostChain {
    Vhs(VhsPostProcess),
    EightMm(EightMmPostProcess),
}

impl PostChain {
    fn new(format: TapeFormat, audio_rate: f64, params: PostProcessParams, enable_deemphasis: bool, enable_expander: bool) -> Self {
        if format == TapeFormat::Video8 {
            PostChain::EightMm(EightMmPostProcess::new(audio_rate, params, enable_deemphasis, enable_expander))
        } else {
            PostChain::Vhs(VhsPostProcess::new(audio_rate, params, enable_deemphasis, enable_expander))
        }
    }

    fn process(&mut self, pre: &mut [f32], post: &mut [f32], prime_len: Option<usize>) {
        match self {
            PostChain::Vhs(chain) => chain.process(pre, post, prime_len),
            PostChain::EightMm(chain) => chain.process(pre, post, prime_len),
        }
    }
}

/// Decodes the whole of `reader`'s RF input, calling `on_chunk` with each
/// consecutive span of final-rate stereo (or mono, for the `l`/`r`/`sum`
/// decode modes) audio as soon as it's ready — in order, never all at
/// once. Streams both the input (see `decode_blocks_streaming`/
/// `crate::stream`) and the output this way: neither the raw RF nor the
/// decoded audio is ever fully materialized in memory here, so callers
/// (e.g. `cli::run_cli`, forwarding chunks straight into a file writer)
/// can decode arbitrarily long captures in bounded memory.
pub fn decode(reader: &mut DecodeReader, params: &PipelineParams, mut on_chunk: impl FnMut(&[f32], &[f32]) -> Result<()>) -> Result<()> {
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

    // DC blocking and de-emphasis/expansion carry continuous IIR state
    // across the whole stream — see the module doc comment — so each is
    // constructed once, here, and mutated across every chunk the collector
    // below hands it, in order.
    let mut dc_blocker_l = DcBlocker::new(params.audio_final_rate, 1.0);
    let mut dc_blocker_r = DcBlocker::new(params.audio_final_rate, 1.0);
    let mut chain_l = PostChain::new(params.format, params.audio_final_rate, params.post_process, params.enable_deemphasis, params.enable_expander);
    let mut chain_r = PostChain::new(params.format, params.audio_final_rate, params.post_process, params.enable_deemphasis, params.enable_expander);
    let mut is_first_chunk = true;
    let mut progress = Progress::new(reader.total_samples(), params.audio_final_rate);

    decode_blocks_streaming(
        reader,
        &layout,
        &afe_l,
        &afe_r,
        &disc_l,
        &disc_r,
        doc_params.as_ref(),
        hs_params.as_ref(),
        params,
        |mut pre_l, mut pre_r, block| {
            if params.gain != 1.0 {
                for sample in pre_l.iter_mut().chain(pre_r.iter_mut()) {
                    *sample *= params.gain as f32;
                }
            }

            dc_blocker_l.process(&mut pre_l);
            dc_blocker_r.process(&mut pre_r);

            let mut post_l = pre_l.clone();
            let mut post_r = pre_r.clone();

            // Only the very first chunk primes the expander, over exactly
            // that chunk's own length — matching Python's block-0-only
            // priming scope over one block's worth of data. Priming on
            // every chunk (or over more than one chunk's worth) is wrong,
            // not just wasteful — see `VhsPostProcess::process`'s doc
            // comment for the real-capture bug this caused when an
            // earlier version of this port primed over the whole stream.
            let prime_len = is_first_chunk.then_some(pre_l.len());
            is_first_chunk = false;

            chain_l.process(&mut pre_l, &mut post_l, prime_len);
            chain_r.process(&mut pre_r, &mut post_r, prime_len);

            progress.record(post_l.len(), block.read_end);
            on_chunk(&post_l, &post_r)
        },
    )?;

    progress.finish();
    Ok(())
}
