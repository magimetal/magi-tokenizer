//! Hooks for memory benchmarks in `bench/`. Enabled only by the `harness` feature; not a stable API.
//! The settings and counters are process-wide, so use them from one thread.

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

static FALLBACK_PIECES: AtomicUsize = AtomicUsize::new(0);
static FAIL_WINDOW_MERGE_AFTER_STITCHES: AtomicUsize = AtomicUsize::new(usize::MAX);

/// Pieces that fell back to the whole-piece merge since the last reset, in encode or count.
pub fn fallback_pieces() -> usize {
    FALLBACK_PIECES.load(Relaxed)
}

pub fn reset_fallback_counters() {
    FALLBACK_PIECES.store(0, Relaxed);
}

/// Makes window merging give up, as when no stitch point exists, once it has stitched `stitches` windows of a
/// piece. `None` restores normal behavior.
pub fn fail_window_merge_after_stitches(stitches: Option<usize>) {
    FAIL_WINDOW_MERGE_AFTER_STITCHES.store(stitches.unwrap_or(usize::MAX), Relaxed);
}

pub(crate) fn window_merge_forced_to_fail(stitches: usize) -> bool {
    stitches >= FAIL_WINDOW_MERGE_AFTER_STITCHES.load(Relaxed)
}

pub(crate) fn record_fallback() {
    FALLBACK_PIECES.fetch_add(1, Relaxed);
}
