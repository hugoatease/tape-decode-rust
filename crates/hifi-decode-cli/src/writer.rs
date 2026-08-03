//! Incremental (streaming) output writers. Matches Python's implicit
//! format-from-extension rule (`main.py:998-1025`): `.wav` gets 16-bit
//! PCM, anything else gets 24-bit FLAC. hifi-decode writes no sidecar
//! metadata/JSON, so these are the entire output surface.
//!
//! Both writers accept audio one chunk at a time (`write_chunk`) rather
//! than as one whole-signal buffer, so the pipeline never has to hold the
//! full decoded output in memory before writing it out — for the same
//! reason the RF *input* is streamed rather than read upfront (see
//! `crate::stream`), just at a smaller scale (decoded audio is on the
//! order of a few hundred MB/hour, not tens of GB, but there's no reason
//! to hold even that if the pipeline doesn't have to).

use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::{Context, Result};
use flacenc::component::{BitRepr, Stream, StreamInfo};
use flacenc::error::{Verified, Verify};
use flacenc::source::{Context as Md5Context, Fill, FrameBuf};

fn f32_to_i16(v: f32) -> i32 {
    (v.clamp(-1.0, 1.0) * 32767.0).round() as i32
}

fn f32_to_i24(v: f32) -> i32 {
    (v.clamp(-1.0, 1.0) * 8_388_607.0).round() as i32
}

fn interleave(left: &[f32], right: Option<&[f32]>, widen: impl Fn(f32) -> i32) -> Vec<i32> {
    match right {
        Some(right) => {
            assert_eq!(left.len(), right.len());
            let mut out = Vec::with_capacity(left.len() * 2);
            for (&l, &r) in left.iter().zip(right) {
                out.push(widen(l));
                out.push(widen(r));
            }
            out
        }
        None => left.iter().map(|&l| widen(l)).collect(),
    }
}

/// 16-bit PCM WAV, matching Python's `.wav` output path exactly (fixed
/// `PCM_16`, no user-configurable bit depth). `hound::WavWriter` already
/// writes samples to disk as they're given rather than buffering the
/// whole file, so this is a thin wrapper, not a new streaming mechanism.
pub struct WavSink {
    writer: hound::WavWriter<std::io::BufWriter<File>>,
}

impl WavSink {
    pub fn create(path: &Path, sample_rate: u32, channels: u16) -> Result<Self> {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let writer = hound::WavWriter::create(path, spec)
            .with_context(|| format!("failed to create WAV output {}", path.display()))?;
        Ok(WavSink { writer })
    }

    pub fn write_chunk(&mut self, left: &[f32], right: Option<&[f32]>) -> Result<()> {
        for sample in interleave(left, right, f32_to_i16) {
            // Always in i16 range: interleave()'s widen fn (f32_to_i16)
            // clamps and scales to it before the cast.
            self.writer
                .write_sample(sample as i16)
                .context("failed to write WAV sample")?;
        }
        Ok(())
    }

    pub fn finish(self) -> Result<()> {
        self.writer.finalize().context("failed to finalize WAV output")?;
        Ok(())
    }
}

/// 24-bit FLAC, matching Python's default (non-`.wav`) output path,
/// written one frame at a time directly to disk rather than requiring the
/// whole signal as one in-memory `&[i32]` upfront (what `flacenc`'s batch
/// `MemSource`/`encode_with_fixed_block_size` API needs). Built on
/// `flacenc`'s lower-level, publicly documented manual-`Stream`-building
/// API (`encode_fixed_size_frame` + `StreamInfo::update_frame_info`,
/// which the crate's own docs point to for exactly this use case) instead.
///
/// Strategy: write a placeholder `STREAMINFO` header up front, append one
/// FLAC frame per `write_chunk` call (each chunk directly becomes one
/// frame — larger than FLAC's typical ~4096-sample default, since a
/// chunk here is one whole decode block's worth, but well under
/// `MAX_BLOCK_SIZE` and fully spec-valid), then on `finish` seek back to
/// byte 0 and overwrite the header with final totals.
///
/// This only works because output is always a real, seekable file in this
/// CLI (never stdout) — `hifi-decode` doesn't currently support `-` for
/// output the way `tape-decode` does for `luma`/`chroma-out`.
///
/// The header patch is safe because every `StreamInfo` field is
/// fixed-width regardless of its value (16+16+24+24+20+3+5+36+128 bits =
/// 34 bytes), and the surrounding metadata-block header is fixed at 4
/// bytes too — so the placeholder and final headers are always exactly
/// the same length (42 bytes total, including the 4-byte "fLaC" magic),
/// safe to overwrite in place. MD5 verification is left disabled
/// (`[0u8; 16]`, `flacenc`'s own documented "verification disabled"
/// convention) rather than computed incrementally — real RF-capture FLAC
/// tools already do the same in this investigation's experience (the
/// `flac` reference decoder only warns, doesn't fail, on an unset MD5).
pub struct FlacSink {
    file: File,
    config: Verified<flacenc::config::Encoder>,
    channels: usize,
    info: StreamInfo,
    frame_number: usize,
    /// Samples not yet emitted as a frame, carried across `write_chunk`
    /// calls — see the struct doc comment for why frames are re-chunked
    /// to a fixed size here rather than emitted one-per-input-chunk.
    pending_l: Vec<f32>,
    pending_r: Vec<f32>,
    /// True (unpadded) sample count seen so far, tracked independently of
    /// `info`'s own frame-derived bookkeeping — see `finish`'s doc comment
    /// for why `update_frame_info`'s accounting can't be trusted for this.
    real_sample_count: usize,
    md5_context: Md5Context,
}

/// "fLaC" magic (4 bytes) + one metadata block header (4 bytes) + a
/// STREAMINFO body (34 bytes) — see `FlacSink`'s doc comment for why this
/// is always exactly this many bytes, regardless of field values.
const FLAC_HEADER_LEN: usize = 42;

/// Every FLAC frame this sink emits holds exactly this many samples, with
/// no exceptions — including the last one, zero-padded up to this size if
/// short (see `finish`). `encode_fixed_size_frame` writes frames in
/// "fixed blocksize" mode, where a decoder recovers each frame's absolute
/// sample position as `frame_number * blocksize`; per the FLAC format,
/// that's only a valid stream if *every* frame, including the last, is
/// truly that same size — a genuinely shorter final frame forces
/// `STREAMINFO`'s `min_block_size` below `max_block_size`, and the
/// reference `flac` decoder (1.5.0) then treats the *entire* stream's
/// frame-number bookkeeping as unreliable, re-syncing (and warning: "file
/// might not be seekable") at literally every frame boundary, from the
/// first one on — confirmed by isolating this exact scenario against a
/// minimal reproduction outside this codebase. `flacenc`'s own batch API
/// (`encode_with_fixed_block_size`) sidesteps this the same way: it always
/// reads/pads a full `block_size` per frame internally, and reports the
/// true sample count separately via `StreamInfo::set_total_samples`
/// rather than letting per-frame accounting infer it — the doc comment on
/// `set_total_samples` says as much ("since Frame only knows its frame
/// size, the effective number of samples is not visible after paddings").
/// This sink follows the same pattern by hand, since the streaming
/// per-chunk API doesn't go through that batch path at all. 4096 is a
/// typical FLAC default block size, chosen for no reason beyond "a real
/// FLAC encoder would pick something in this neighborhood".
const FLAC_FRAME_SIZE: usize = 4096;

impl FlacSink {
    pub fn create(path: &Path, sample_rate: u32, channels: usize) -> Result<Self> {
        let info = StreamInfo::new(sample_rate as usize, channels, 24)
            .map_err(|e| anyhow::anyhow!("invalid FLAC stream parameters: {e:?}"))?;
        let config = flacenc::config::Encoder::default()
            .into_verified()
            .map_err(|e| anyhow::anyhow!("invalid FLAC encoder config: {e:?}"))?;

        let mut file = File::create(path)
            .with_context(|| format!("failed to create FLAC output {}", path.display()))?;
        write_header(&mut file, &info)?;

        Ok(FlacSink {
            file,
            config,
            channels,
            info,
            frame_number: 0,
            pending_l: Vec::new(),
            pending_r: Vec::new(),
            real_sample_count: 0,
            md5_context: Md5Context::new(24, channels, FLAC_FRAME_SIZE),
        })
    }

    pub fn write_chunk(&mut self, left: &[f32], right: Option<&[f32]>) -> Result<()> {
        self.real_sample_count += left.len();
        self.pending_l.extend_from_slice(left);
        if self.channels == 2 {
            self.pending_r.extend_from_slice(right.expect("stereo FlacSink requires a right channel"));
        }

        while self.pending_l.len() >= FLAC_FRAME_SIZE {
            let l: Vec<f32> = self.pending_l.drain(..FLAC_FRAME_SIZE).collect();
            let r: Option<Vec<f32>> = (self.channels == 2).then(|| self.pending_r.drain(..FLAC_FRAME_SIZE).collect());
            self.encode_and_write_frame(&l, r.as_deref())?;
        }
        Ok(())
    }

    /// `left`/`right` must be exactly `FLAC_FRAME_SIZE` samples — callers
    /// (`write_chunk`'s drain loop, `finish`'s zero-padded tail) guarantee
    /// this rather than this function handling a variable size, so that
    /// every frame this sink ever writes really is fixed-size, no
    /// exceptions (see the `FLAC_FRAME_SIZE` doc comment for why that
    /// matters).
    fn encode_and_write_frame(&mut self, left: &[f32], right: Option<&[f32]>) -> Result<()> {
        debug_assert_eq!(left.len(), FLAC_FRAME_SIZE);
        let interleaved = interleave(left, right, f32_to_i24);

        self.md5_context
            .fill_interleaved(&interleaved)
            .map_err(|e| anyhow::anyhow!("FLAC md5 context fill failed: {e:?}"))?;

        let mut framebuf = FrameBuf::with_size(self.channels, FLAC_FRAME_SIZE)
            .map_err(|e| anyhow::anyhow!("FLAC frame buffer error: {e:?}"))?;
        framebuf
            .fill_interleaved(&interleaved)
            .map_err(|e| anyhow::anyhow!("FLAC frame buffer fill error: {e:?}"))?;

        let frame = flacenc::encode_fixed_size_frame(&self.config, &framebuf, self.frame_number, &self.info)
            .map_err(|e| anyhow::anyhow!("FLAC frame encode failed: {e:?}"))?;
        self.info.update_frame_info(&frame);
        self.frame_number += 1;

        let mut sink = flacenc::bitsink::ByteSink::new();
        frame
            .write(&mut sink)
            .map_err(|e| anyhow::anyhow!("FLAC frame serialization failed: {e:?}"))?;
        self.file.write_all(sink.as_slice()).context("failed to write FLAC frame")?;
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        if !self.pending_l.is_empty() {
            // Zero-pad the true tail up to a full frame rather than
            // writing it short — see the `FLAC_FRAME_SIZE` doc comment.
            // The padding samples are never played back: `total_samples`
            // below is set from `real_sample_count`, not from this
            // (padded) frame's declared size, so a correct decoder stops
            // exactly at the true audio length.
            self.pending_l.resize(FLAC_FRAME_SIZE, 0.0);
            let l = std::mem::take(&mut self.pending_l);
            let r: Option<Vec<f32>> = (self.channels == 2).then(|| {
                self.pending_r.resize(FLAC_FRAME_SIZE, 0.0);
                std::mem::take(&mut self.pending_r)
            });
            self.encode_and_write_frame(&l, r.as_deref())?;
        }

        self.info.set_total_samples(self.real_sample_count);
        self.info.set_md5_digest(&self.md5_context.md5_digest());

        self.file
            .seek(SeekFrom::Start(0))
            .context("failed to seek back to patch the FLAC header")?;
        write_header(&mut self.file, &self.info)?;
        Ok(())
    }
}

fn write_header(file: &mut File, info: &StreamInfo) -> Result<()> {
    let header_stream = Stream::with_stream_info(info.clone());
    let mut sink = flacenc::bitsink::ByteSink::new();
    header_stream
        .write(&mut sink)
        .map_err(|e| anyhow::anyhow!("FLAC header serialization failed: {e:?}"))?;
    debug_assert_eq!(sink.as_slice().len(), FLAC_HEADER_LEN);
    file.write_all(sink.as_slice()).context("failed to write FLAC header")?;
    Ok(())
}

/// The two output formats hifi-decode supports, unified so callers can
/// feed either one chunk at a time without matching on the format
/// themselves.
pub enum AudioSink {
    Wav(WavSink),
    Flac(FlacSink),
}

impl AudioSink {
    pub fn create(path: &Path, sample_rate: u32, channels: u16, is_wav: bool) -> Result<Self> {
        if is_wav {
            Ok(AudioSink::Wav(WavSink::create(path, sample_rate, channels)?))
        } else {
            Ok(AudioSink::Flac(FlacSink::create(path, sample_rate, channels as usize)?))
        }
    }

    pub fn write_chunk(&mut self, left: &[f32], right: Option<&[f32]>) -> Result<()> {
        match self {
            AudioSink::Wav(sink) => sink.write_chunk(left, right),
            AudioSink::Flac(sink) => sink.write_chunk(left, right),
        }
    }

    pub fn finish(self) -> Result<()> {
        match self {
            AudioSink::Wav(sink) => sink.finish(),
            AudioSink::Flac(sink) => sink.finish(),
        }
    }
}
