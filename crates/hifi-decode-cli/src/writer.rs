//! Output writers. Matches Python's implicit format-from-extension rule
//! (`main.py:998-1025`): `.wav` gets 16-bit PCM, anything else gets 24-bit
//! FLAC. hifi-decode writes no sidecar metadata/JSON, so these are the
//! entire output surface.

use std::path::Path;

use anyhow::{Context, Result};

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
/// `PCM_16`, no user-configurable bit depth).
pub fn write_wav(path: &Path, sample_rate: u32, left: &[f32], right: Option<&[f32]>) -> Result<()> {
    let channels = if right.is_some() { 2 } else { 1 };
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("failed to create WAV output {}", path.display()))?;
    for sample in interleave(left, right, f32_to_i16) {
        // WavWriter<i16> wants i16 directly; interleave() above always
        // produces values in i16 range for this widen fn.
        writer
            .write_sample(sample as i16)
            .with_context(|| format!("failed to write WAV sample to {}", path.display()))?;
    }
    writer
        .finalize()
        .with_context(|| format!("failed to finalize WAV output {}", path.display()))?;
    Ok(())
}

/// 24-bit FLAC, matching Python's default (non-`.wav`) output path
/// (`PCM_24`, `compression_level=1.0`).
pub fn write_flac(path: &Path, sample_rate: u32, left: &[f32], right: Option<&[f32]>) -> Result<()> {
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;

    let channels = if right.is_some() { 2 } else { 1 };
    let interleaved = interleave(left, right, f32_to_i24);

    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|e| anyhow::anyhow!("invalid FLAC encoder config: {e:?}"))?;
    let source = flacenc::source::MemSource::from_samples(&interleaved, channels, 24, sample_rate as usize);
    let stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|e| anyhow::anyhow!("FLAC encode failed: {e:?}"))?;

    let mut sink = flacenc::bitsink::ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|e| anyhow::anyhow!("FLAC bitstream serialization failed: {e:?}"))?;
    std::fs::write(path, sink.as_slice())
        .with_context(|| format!("failed to write FLAC output {}", path.display()))?;
    Ok(())
}
