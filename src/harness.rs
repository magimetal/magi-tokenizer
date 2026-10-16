//! Hooks for the differential harness in `bench/`. Enabled only by the `harness` feature; not a stable API.
//! The settings and counters are process-wide, so use them from one thread.

use crate::bpe;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

static FALLBACK_PIECES: AtomicUsize = AtomicUsize::new(0);
static DISCARDED_TOKENS: AtomicUsize = AtomicUsize::new(0);
static FAIL_WINDOW_MERGE_AFTER_STITCHES: AtomicUsize = AtomicUsize::new(usize::MAX);

/// Pieces that fell back to the whole-piece heap merge since the last reset, in encode or count.
pub fn fallback_pieces() -> usize {
    FALLBACK_PIECES.load(Relaxed)
}

/// Tokens that window merging had already appended or counted before a fallback discarded them.
pub fn discarded_tokens() -> usize {
    DISCARDED_TOKENS.load(Relaxed)
}

pub fn reset_fallback_counters() {
    FALLBACK_PIECES.store(0, Relaxed);
    DISCARDED_TOKENS.store(0, Relaxed);
}

/// Makes window merging give up, as when no stitch point exists, once it has stitched `stitches` windows of a
/// piece. `None` restores normal behavior.
pub fn fail_window_merge_after_stitches(stitches: Option<usize>) {
    FAIL_WINDOW_MERGE_AFTER_STITCHES.store(stitches.unwrap_or(usize::MAX), Relaxed);
}

pub(crate) fn window_merge_forced_to_fail(stitches: usize) -> bool {
    stitches >= FAIL_WINDOW_MERGE_AFTER_STITCHES.load(Relaxed)
}

pub(crate) fn record_fallback(discarded_tokens: usize) {
    FALLBACK_PIECES.fetch_add(1, Relaxed);
    DISCARDED_TOKENS.fetch_add(discarded_tokens, Relaxed);
}

/// Appends the tokens of one piece of arbitrary bytes, through the same dispatch as `encode_ordinary_into`.
pub fn encode_piece(piece: &[u8], out: &mut Vec<u32>) {
    bpe::encode_piece(piece, out);
}

/// Token count of one piece of arbitrary bytes, through the same dispatch as `count_ordinary`.
pub fn count_piece(piece: &[u8]) -> usize {
    bpe::count_piece(piece)
}

/// Whole-piece heap merge with `u32` part indices; panics for pieces of `u32::MAX` bytes or more.
pub fn heap_merge(piece: &[u8]) -> Vec<u32> {
    assert!(piece.len() < u32::MAX as usize, "u32 part indices need pieces under u32::MAX bytes");
    bpe::heap_merge(piece, false)
}

/// Whole-piece heap merge with `usize` part indices, the path for pieces of 4 GiB or more.
pub fn heap_merge_wide_index(piece: &[u8]) -> Vec<u32> {
    bpe::heap_merge(piece, true)
}

/// Window merge alone; None when it gives up and the caller would fall back.
pub fn window_merge(piece: &[u8]) -> Option<Vec<u32>> {
    let mut tokens = Vec::new();
    bpe::merge_in_windows(piece, |batch| tokens.extend_from_slice(batch)).then_some(tokens)
}

/// Stack merge of a piece of at most 128 bytes, without the whole-token shortcut.
pub fn short_merge(piece: &[u8]) -> Option<Vec<u32>> {
    bpe::short_merge(piece)
}
