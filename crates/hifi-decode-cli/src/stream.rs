//! Bounded-memory RF block source. Produces the same `[read_start,
//! read_end)` spans `hifi_decode::BlockLayout::blocks` would compute for
//! the whole input, but incrementally — reading only as far ahead as the
//! current block needs, without requiring the input's total length
//! upfront and without ever holding more than a couple of blocks' worth
//! of RF samples in memory.
//!
//! This exists because decoding a real capture by first reading the whole
//! file into a `Vec<f32>` needs roughly 4 bytes of RAM per RF sample: a
//! 17GB 8-bit capture becomes ~68GB as f32, which doesn't fit in memory on
//! an ordinary machine. Only the read side needs this treatment — the
//! *decoded* audio (192kHz intermediate, or the final rate) is on the
//! order of a few hundred MB even for an hour-long tape, so it's fine to
//! keep concatenating that in memory as before.

use anyhow::Result;
use hifi_decode::{Block, BlockLayout};
use tape_rf_io::DecodeReader;

/// Pulls RF samples from `reader` in a sliding window and yields one
/// `Block` (plus its RF samples) at a time, in order.
pub struct StreamingBlocks<'a> {
    reader: &'a mut DecodeReader,
    layout: BlockLayout,
    /// RF samples currently buffered, covering
    /// `[window_start, window_start + window.len())`.
    window: Vec<f32>,
    window_start: usize,
    eof: bool,
    next_index: usize,
}

/// Samples read per underlying `DecodeReader::read` call while filling the
/// window.
const READ_CHUNK: usize = 1 << 20;

impl<'a> StreamingBlocks<'a> {
    pub fn new(reader: &'a mut DecodeReader, layout: BlockLayout) -> Self {
        StreamingBlocks {
            reader,
            layout,
            window: Vec::new(),
            window_start: 0,
            eof: false,
            next_index: 0,
        }
    }

    /// Reads more samples into `window` until it covers `want_len` samples
    /// from `window_start`, or until the source is exhausted.
    fn fill_to(&mut self, want_len: usize) -> Result<()> {
        let mut chunk = vec![0.0f32; READ_CHUNK];
        while self.window.len() < want_len && !self.eof {
            let n = self.reader.read(&mut chunk)?;
            if n == 0 {
                self.eof = true;
                break;
            }
            self.window.extend_from_slice(&chunk[..n]);
        }
        Ok(())
    }

    /// Returns the next block's metadata and RF samples, or `None` once
    /// the source is exhausted and there is nothing left to emit.
    pub fn next_block(&mut self) -> Result<Option<(Block, Vec<f32>)>> {
        let index = self.next_index;
        let read_start = self.layout.read_start_for(index);

        // Drop anything before this block's read_start: no later block
        // will ever need it again (spans only overlap with their
        // immediate neighbor).
        if read_start > self.window_start {
            let drop = (read_start - self.window_start).min(self.window.len());
            self.window.drain(..drop);
            self.window_start += drop;
        }

        let want_end = self.layout.read_end_nominal_for(index);
        let want_len = want_end.saturating_sub(self.window_start);
        self.fill_to(want_len)?;

        if self.window.is_empty() {
            return Ok(None);
        }

        let available_end = self.window_start + self.window.len();
        let is_last = self.eof && available_end <= want_end;
        let read_end = available_end.min(want_end).max(read_start);

        let local_end = read_end - self.window_start;
        let rf_samples = self.window[..local_end].to_vec();

        self.next_index += 1;
        let block = self.layout.make_block(index, read_start, read_end, is_last);
        Ok(Some((block, rf_samples)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tape_rf_io::{open_source, SampleFormat};

    /// Streaming through `StreamingBlocks` must produce the exact same
    /// `Block` spans as `BlockLayout::blocks` does with the whole length
    /// known upfront.
    #[test]
    fn matches_whole_buffer_blocks() {
        let sample_rate = 8_000_000.0;
        let layout = BlockLayout::new(sample_rate, 192_000.0, 48_000.0);
        let total_len = 10_000_000usize;

        let samples: Vec<f32> = (0..total_len).map(|i| (i % 100) as f32 / 100.0).collect();
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let dir = std::env::temp_dir();
        let path = dir.join(format!("hifi_decode_stream_test_{}.f32le", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();

        let file = std::fs::File::open(&path).unwrap();
        let mut reader = DecodeReader::new(open_source(file, SampleFormat::F32LE).unwrap());
        let mut streamer = StreamingBlocks::new(&mut reader, layout);

        let mut streamed = Vec::new();
        while let Some((block, rf)) = streamer.next_block().unwrap() {
            assert_eq!(rf.len(), block.read_end - block.read_start);
            streamed.push(block);
        }
        std::fs::remove_file(&path).ok();

        let expected = layout.blocks(total_len);
        assert_eq!(streamed.len(), expected.len());
        for (a, b) in streamed.iter().zip(expected.iter()) {
            assert_eq!(a.read_start, b.read_start);
            assert_eq!(a.read_end, b.read_end);
            assert_eq!(a.output_skip, b.output_skip);
            assert_eq!(a.output_take, b.output_take);
        }
    }
}
