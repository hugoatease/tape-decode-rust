//! FLAC input via symphonia. Decodes a mono FLAC stream to `f32` samples behind
//! the same [`SampleSource`] interface as the raw formats. Signed integer FLAC
//! samples are normalized to ~[-1, 1], so a 16-bit FLAC reads identically to the
//! equivalent raw `s16` capture. File-backed input seeks via
//! symphonia's binary search; stdin can only skip forward by decoding.
//!
//! The STREAMINFO `total_samples` field is never handed to symphonia (see
//! [`hide_total_samples`]): it is only 36 bits wide, so long RF captures wrap it
//! (ffmpeg truncates it modulo 2^36 instead of writing 0), and symphonia
//! rejects any seek beyond the declared length.

use std::io::{self, Read, Seek, SeekFrom};

use anyhow::{bail, Context as _, Result};
use symphonia_bundle_flac::{FlacDecoder, FlacReader};
use symphonia_core::audio::{Audio, GenericAudioBufferRef};
use symphonia_core::codecs::audio::{AudioDecoder, AudioDecoderOptions};
use symphonia_core::codecs::CodecParameters;
use symphonia_core::formats::{FormatOptions, FormatReader, SeekMode, SeekTo};
use symphonia_core::io::{MediaSource, MediaSourceStream};
use symphonia_core::units::Timestamp;

use crate::reader::SampleSource;

/// Decode `source` (a seekable file or a forward-only pipe) as a FLAC sample
/// source.
pub fn open(source: Box<dyn MediaSource>) -> Result<Box<dyn SampleSource>> {
    Ok(Box::new(FlacSource::new(hide_total_samples(source))?))
}

/// Bytes of a native FLAC stream needed to cover the STREAMINFO `total_samples`
/// field: `fLaC` (4), block header (4), then 18 bytes of STREAMINFO up to and
/// including the last byte of `total_samples`.
const STREAMINFO_PREFIX_LEN: usize = 26;
/// `total_samples` is 36 bits: the low nibble of byte 21, then bytes 22..26.
const TOTAL_SAMPLES_HIGH_NIBBLE_BYTE: usize = 21;

/// Wrap `source` so that a FLAC STREAMINFO `total_samples` reads as 0 ("unknown").
///
/// The field is 36 bits wide (about 68.7 G samples, 40 minutes at 28.6 MSps).
/// Longer captures wrap: ffmpeg writes the length modulo 2^36, which looks like
/// a valid but far too short length, and symphonia then refuses every seek
/// beyond it. With the field hidden, symphonia binary-searches the frame
/// headers instead, exactly as it does for files that declare 0.
///
/// Anything that does not start with a `fLaC` marker followed by a STREAMINFO
/// block is passed through untouched.
pub(crate) fn hide_total_samples(source: Box<dyn MediaSource>) -> Box<dyn MediaSource> {
    Box::new(HideTotalSamples::new(source))
}

struct HideTotalSamples {
    inner: Box<dyn MediaSource>,
    /// The first bytes of the stream, with `total_samples` already zeroed.
    prefix: Vec<u8>,
    /// Absolute position of the next byte `read` returns.
    pos: u64,
}

impl HideTotalSamples {
    fn new(mut inner: Box<dyn MediaSource>) -> Self {
        let mut prefix = vec![0u8; STREAMINFO_PREFIX_LEN];
        let mut got = 0;
        // A failed or short read just means there is nothing to patch; the
        // bytes already read are still replayed from `prefix`.
        while got < prefix.len() {
            match inner.read(&mut prefix[got..]) {
                Ok(0) | Err(_) => break,
                Ok(n) => got += n,
            }
        }
        prefix.truncate(got);
        if prefix.len() == STREAMINFO_PREFIX_LEN
            && &prefix[..4] == b"fLaC"
            && prefix[4] & 0x7f == 0
        {
            prefix[TOTAL_SAMPLES_HIGH_NIBBLE_BYTE] &= 0xf0;
            prefix[TOTAL_SAMPLES_HIGH_NIBBLE_BYTE + 1..].fill(0);
        }
        Self {
            inner,
            prefix,
            pos: 0,
        }
    }
}

impl Read for HideTotalSamples {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.pos < self.prefix.len() as u64 {
            let start = self.pos as usize;
            let n = buf.len().min(self.prefix.len() - start);
            buf[..n].copy_from_slice(&self.prefix[start..start + n]);
            self.pos += n as u64;
            return Ok(n);
        }
        let n = self.inner.read(buf)?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for HideTotalSamples {
    fn seek(&mut self, to: SeekFrom) -> io::Result<u64> {
        // `inner` sits at the end of the prefix while `pos` is inside it, so a
        // relative seek must be turned into an absolute one first.
        let to = match to {
            SeekFrom::Current(offset) => {
                let target = self.pos.checked_add_signed(offset).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "seek before start of stream")
                })?;
                SeekFrom::Start(target)
            }
            other => other,
        };
        let target = self.inner.seek(to)?;
        self.pos = target;
        if target < self.prefix.len() as u64 {
            // Reads in the prefix are served from memory; keep `inner` aligned
            // with the first byte after it.
            self.inner.seek(SeekFrom::Start(self.prefix.len() as u64))?;
        }
        Ok(target)
    }
}

impl MediaSource for HideTotalSamples {
    fn is_seekable(&self) -> bool {
        self.inner.is_seekable()
    }

    fn byte_len(&self) -> Option<u64> {
        self.inner.byte_len()
    }
}

struct FlacSource {
    reader: FlacReader<'static>,
    decoder: FlacDecoder,
    track_id: u32,
    /// Right shift undoing symphonia's normalization of samples to the full i32
    /// range (`32 - bits_per_sample`), then `scale` normalizes the native signed
    /// integer amplitude to match raw signed PCM input.
    shift: u32,
    scale: f32,
    /// Whether the source supports real seeking (a regular file); pipes can only
    /// skip forward by decoding.
    seekable: bool,
    /// Decoded samples of the current packet not yet returned.
    pending: Vec<f32>,
    pending_pos: usize,
    /// Absolute index of the next sample `read` will return.
    position: u64,
    eof: bool,
}

impl FlacSource {
    fn new(source: Box<dyn MediaSource>) -> Result<Self> {
        let seekable = source.is_seekable();
        let mss = MediaSourceStream::new(source, Default::default());
        let reader =
            FlacReader::try_new(mss, FormatOptions::default()).context("failed to read FLAC")?;

        let (track_id, params) = {
            let track = reader.tracks().first().context("FLAC has no tracks")?;
            let params = track
                .codec_params
                .as_ref()
                .and_then(CodecParameters::audio)
                .context("FLAC track is not audio")?
                .clone();
            (track.id, params)
        };

        let bits = params
            .bits_per_sample
            .context("FLAC stream info missing bits per sample")?;
        if !(1..=32).contains(&bits) {
            bail!("unsupported FLAC bit depth: {bits}");
        }
        let channels = params.channels.as_ref().map_or(1, |c| c.count());
        if channels != 1 {
            bail!("only mono FLAC input is supported, found {channels} channels");
        }

        let decoder = FlacDecoder::try_new(&params, &AudioDecoderOptions::default())
            .context("failed to initialize FLAC decoder")?;

        let scale = 1.0 / ((1u64 << (bits - 1)) as f32);

        Ok(Self {
            reader,
            decoder,
            track_id,
            shift: 32 - bits,
            scale,
            seekable,
            pending: Vec::new(),
            pending_pos: 0,
            position: 0,
            eof: false,
        })
    }

    /// Decode the next packet into `pending`. Returns false at end of input.
    fn decode_next(&mut self) -> Result<bool> {
        let shift = self.shift;
        let scale = self.scale;
        loop {
            let Some(packet) = self.reader.next_packet().context("FLAC read error")? else {
                self.eof = true;
                return Ok(false);
            };
            if packet.track_id != self.track_id {
                continue;
            }
            let decoded = self.decoder.decode(&packet).context("FLAC decode error")?;
            if decoded.frames() == 0 {
                continue;
            }
            let GenericAudioBufferRef::S32(buf) = decoded else {
                bail!("unexpected FLAC sample format (expected signed 32-bit)");
            };
            let plane = buf.plane(0).expect("mono FLAC has one plane");
            self.pending.clear();
            self.pending
                .extend(plane.iter().map(|&s| ((s >> shift) as f32) * scale));
            self.pending_pos = 0;
            return Ok(true);
        }
    }

    /// Ensure `pending` has unconsumed samples, decoding if needed. False at EOF.
    fn fill(&mut self) -> Result<bool> {
        if self.pending_pos < self.pending.len() {
            return Ok(true);
        }
        if self.eof {
            return Ok(false);
        }
        self.decode_next()
    }

    /// Drop samples until `position` reaches `target` (forward only).
    fn skip_to(&mut self, target: u64) -> Result<()> {
        while self.position < target {
            if !self.fill()? {
                bail!("FLAC input ended before reaching sample offset {target}");
            }
            let avail = self.pending.len() - self.pending_pos;
            let n = avail.min((target - self.position) as usize);
            self.pending_pos += n;
            self.position += n as u64;
        }
        Ok(())
    }
}

impl SampleSource for FlacSource {
    fn read(&mut self, out: &mut [f32]) -> Result<usize> {
        let mut written = 0;
        while written < out.len() {
            if !self.fill()? {
                break;
            }
            let avail = &self.pending[self.pending_pos..];
            let n = avail.len().min(out.len() - written);
            out[written..written + n].copy_from_slice(&avail[..n]);
            self.pending_pos += n;
            written += n;
        }
        self.position += written as u64;
        Ok(written)
    }

    fn seek_samples(&mut self, sample: u64) -> Result<()> {
        if self.seekable {
            // Jump to the packet containing `sample` via symphonia's binary
            // search, then drop the handful of samples up to it.
            let ts = Timestamp::try_from(sample).context("seek offset too large")?;
            let seeked = self
                .reader
                .seek(
                    SeekMode::Coarse,
                    SeekTo::Timestamp {
                        ts,
                        track_id: self.track_id,
                    },
                )
                .context("FLAC seek failed")?;
            self.decoder.reset();
            self.pending.clear();
            self.pending_pos = 0;
            self.eof = false;
            self.position = seeked.actual_ts.get() as u64;
        } else if sample < self.position {
            bail!("cannot seek backward on a non-seekable FLAC input (stdin)");
        }
        self.skip_to(sample)
    }
}
