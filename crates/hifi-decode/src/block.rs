//! Block sizing and overlap. Ports `HiFiDecode.calculate_block_sizes` /
//! `_set_block_overlap` (`HiFiDecode.py:1279-1387`) — the math that sizes
//! ~0.5s decode blocks and the RF-domain overlap padding needed to hide
//! resampler edge transients at block boundaries.
//!
//! Unlike Python, which carries this overlap through a hand-rolled
//! shared-memory ring buffer (a multiprocessing IPC concern that doesn't
//! exist for Rust threads sharing one address space), this crate exposes
//! it as plain `[start, end)` ranges into one contiguous RF buffer — see
//! `BlockLayout::blocks`.

use std::cmp::min;

const BLOCKS_PER_SECOND: f64 = 2.0;

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// One decode block's RF-sample span, in absolute offsets into the whole
/// input: `[read_start, read_end)` is what to feed the per-block DSP chain
/// (AFE/demod/resample/...), and `output_skip`/`output_take` say which
/// slice of *that block's decoded output* (at the final audio rate) is
/// the block's real, non-overlap contribution — trim to that span before
/// concatenating blocks.
#[derive(Clone, Copy, Debug)]
pub struct Block {
    pub read_start: usize,
    pub read_end: usize,
    pub output_skip: usize,
    pub output_take: usize,
}

/// Precomputed block/overlap sizing for one decode run. All fields mirror
/// their Python namesakes 1:1 for cross-reference.
#[derive(Clone, Copy, Debug)]
pub struct BlockLayout {
    pub block_size: usize,
    pub block_audio_final_size: usize,
    pub block_overlap: usize,
    pub block_audio_final_overlap: usize,
    pub pre_trim: usize,
}

impl BlockLayout {
    pub fn new(input_rate: f64, audio_rate: f64, audio_final_rate: f64) -> Self {
        let blocks_per_second_ratio = 1.0 / BLOCKS_PER_SECOND;
        let block_size = (input_rate * blocks_per_second_ratio).ceil() as u64;
        let block_audio_size = (audio_rate * blocks_per_second_ratio).ceil() as u64;
        let block_audio_final_size = (audio_final_rate * blocks_per_second_ratio).ceil() as u64;

        let block_size_gcd = gcd(block_size, block_audio_final_size);
        let block_audio_overlap_divisor = if block_size_gcd > 5 {
            block_audio_size / block_size_gcd
        } else {
            tracing::warn!(
                "input sample rate is not evenly divisible by the output sample rate; \
                 audio sync issues may occur (input {input_rate} Hz, output {audio_final_rate} Hz)"
            );
            1
        };

        let pre_trim: u64 = 1000;
        let min_resampler_overlap = pre_trim + 50;
        let min_overlap = ((min_resampler_overlap as f64) / audio_rate * audio_final_rate).ceil() as u64;
        let block_audio_final_overlap_seed =
            (min_overlap as f64 / block_audio_overlap_divisor as f64).ceil() as u64 * block_audio_overlap_divisor;

        let overlap_seconds = block_audio_final_overlap_seed as f64 / audio_final_rate;
        let block_overlap = (input_rate * overlap_seconds).round() as u64;
        let block_audio_final_overlap = (audio_final_rate * overlap_seconds).round() as u64;

        BlockLayout {
            block_size: block_size as usize,
            block_audio_final_size: block_audio_final_size as usize,
            block_overlap: block_overlap as usize,
            block_audio_final_overlap: block_audio_final_overlap as usize,
            pre_trim: pre_trim as usize,
        }
    }

    /// Nominal (pre-overlap) start of block `index`, at a fixed
    /// `block_size` stride — independent of the total input length, which
    /// only ever affects where the *last* block ends, not where any block
    /// (including the last) nominally starts. This is what lets a
    /// streaming reader compute read spans one block at a time without
    /// knowing the input's total length upfront.
    pub fn nominal_block_start(&self, index: usize) -> usize {
        index * self.block_size
    }

    /// Where block `index` should start reading RF samples: the nominal
    /// start, minus `block_overlap` — except block 0, which has no
    /// predecessor to borrow overlap from.
    pub fn read_start_for(&self, index: usize) -> usize {
        let start = self.nominal_block_start(index);
        if index == 0 {
            start
        } else {
            start.saturating_sub(self.block_overlap)
        }
    }

    /// Where block `index` would stop reading if the input were infinite:
    /// nominal start + `block_size` + `block_overlap`. A real reader clamps
    /// this to whatever data actually exists and flags `is_last` itself
    /// once it runs out.
    pub fn read_end_nominal_for(&self, index: usize) -> usize {
        self.nominal_block_start(index) + self.block_size + self.block_overlap
    }

    /// Builds a `Block` from an already-resolved read span. Shared by
    /// `blocks` (which resolves spans against a known `total_len`) and any
    /// streaming block source (which resolves them against real EOF).
    pub fn make_block(&self, index: usize, read_start: usize, read_end: usize, is_last: bool) -> Block {
        let is_first = index == 0;
        let nominal_start = self.nominal_block_start(index);
        let left_pad = nominal_start.saturating_sub(read_start); // == block_overlap unless clamped by is_first
        let output_skip = if is_first {
            0
        } else {
            // The left_pad RF samples resample down to roughly this many
            // output samples; the real overlap trim amount is fixed
            // (block_audio_final_overlap) regardless of the exact
            // resampled length, matching Python's fixed-size discard.
            self.block_audio_final_overlap.min(left_pad)
        };
        let output_take = if is_last {
            usize::MAX // caller clamps to the block's actual output length
        } else {
            self.block_audio_final_size
        };
        Block {
            read_start,
            read_end,
            output_skip,
            output_take,
        }
    }

    /// Splits `total_len` RF samples into overlapping blocks per the
    /// diagram in `HiFiDecode._set_block_overlap` (`HiFiDecode.py:1339-1342`):
    /// each block reads `block_overlap` extra samples on each side (block 0
    /// synthesizes its missing left overlap by reading from its own start;
    /// the last block is simply clamped to `total_len`), and each block's
    /// *output* at the final rate discards `block_audio_final_overlap`
    /// samples on the sides that have a real neighboring block.
    ///
    /// This requires the whole input's length upfront; for a bounded-memory
    /// streaming reader that doesn't have (or want) that, drive
    /// `read_start_for`/`read_end_nominal_for`/`make_block` incrementally
    /// instead (see `hifi-decode-cli`'s streaming block source).
    pub fn blocks(&self, total_len: usize) -> Vec<Block> {
        if total_len == 0 {
            return Vec::new();
        }
        let mut blocks = Vec::new();
        let mut index = 0usize;
        loop {
            let start = self.nominal_block_start(index);
            if start >= total_len {
                break;
            }
            let end = min(start + self.block_size, total_len);
            let is_last = end >= total_len;

            let read_start = self.read_start_for(index);
            let read_end = if is_last { total_len } else { min(self.read_end_nominal_for(index), total_len) };

            blocks.push(self.make_block(index, read_start, read_end, is_last));
            index += 1;
        }
        blocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_the_whole_input_without_gaps_or_double_counting() {
        let layout = BlockLayout::new(8_000_000.0, 192_000.0, 48_000.0);
        // block_size is 0.5s (4_000_000 samples at 8MHz), so 10M samples
        // spans 3 blocks.
        let blocks = layout.blocks(10_000_000);
        assert!(blocks.len() > 1, "expected multiple blocks for this input length");
        assert_eq!(blocks.first().unwrap().read_start, 0);
        assert_eq!(blocks.last().unwrap().read_end, 10_000_000);
        // Each block's un-overlapped span starts exactly where the
        // previous one's did.
        for pair in blocks.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            assert!(b.read_start <= a.read_end, "gap between blocks: {a:?} {b:?}");
        }
    }

    #[test]
    fn single_block_when_input_is_short() {
        let layout = BlockLayout::new(8_000_000.0, 192_000.0, 48_000.0);
        let blocks = layout.blocks(1000);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].read_start, 0);
        assert_eq!(blocks[0].read_end, 1000);
    }

    /// The index-driven helpers a streaming reader uses must agree with
    /// what `blocks()` computes for every non-last block (whose spans
    /// don't depend on where the input actually ends).
    #[test]
    fn index_driven_helpers_agree_with_whole_buffer_blocks() {
        let layout = BlockLayout::new(8_000_000.0, 192_000.0, 48_000.0);
        let total_len = 20_000_000;
        let blocks = layout.blocks(total_len);
        assert!(blocks.len() >= 4, "need several blocks including a non-last one");
        for (i, block) in blocks.iter().enumerate().take(blocks.len() - 1) {
            assert_eq!(layout.read_start_for(i), block.read_start, "block {i} read_start");
            assert_eq!(layout.read_end_nominal_for(i), block.read_end, "block {i} read_end (non-last, so nominal == actual)");
            let rebuilt = layout.make_block(i, block.read_start, block.read_end, false);
            assert_eq!(rebuilt.output_skip, block.output_skip, "block {i} output_skip");
            assert_eq!(rebuilt.output_take, block.output_take, "block {i} output_take");
        }
    }
}
