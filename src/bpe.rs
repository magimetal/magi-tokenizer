// Byte-pair merging of one pretokenized piece. Matches tiktoken exactly: repeatedly merge the adjacent pair whose
// merged token has the lowest rank, leftmost first. Parts are always vocabulary tokens, so "the rank of the
// concatenated bytes" equals the pair table's merged id for (left token, right token).

use crate::tables::{BYTE_TOKENS, NO_MERGE, byte_merge_rank, merge_rank, ordinary_token_bytes, token_id_for_bytes};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Pieces up to this length merge in stack arrays with linear minimum search. Every token is at most this long.
const SHORT_PIECE_BYTES: usize = 128;
/// Long pieces are encoded as overlapping windows of FIRST_WINDOW_BYTES; a window doubles while no stitch point is
/// found, up to MAX_WINDOW_BYTES, after which the whole piece is merged directly.
const FIRST_WINDOW_BYTES: usize = 64;
const MAX_WINDOW_BYTES: usize = 4096;
/// A new window reaches back over the last `window / OVERLAP_DIVISOR` bytes of the encoding so far.
const OVERLAP_DIVISOR: usize = 8;

/// Appends the tokens of `piece` to `out`.
#[inline]
pub(crate) fn encode_piece(piece: &[u8], out: &mut Vec<u32>) {
    if piece.len() <= SHORT_PIECE_BYTES {
        encode_short_piece(piece, out);
    } else {
        encode_long_piece(piece, out);
    }
}

/// Number of tokens `piece` encodes to.
#[inline]
pub(crate) fn count_piece(piece: &[u8]) -> usize {
    if piece.len() > SHORT_PIECE_BYTES {
        let mut tokens = Vec::new();
        encode_long_piece(piece, &mut tokens);
        tokens.len()
    } else if token_id_for_bytes(piece).is_some() {
        1
    } else {
        merge_short_piece(piece, &mut [0u32; SHORT_PIECE_BYTES])
    }
}

#[inline]
fn encode_short_piece(piece: &[u8], out: &mut Vec<u32>) {
    if let Some(id) = token_id_for_bytes(piece) {
        out.push(id);
    } else {
        let mut tokens = [0u32; SHORT_PIECE_BYTES];
        let count = merge_short_piece(piece, &mut tokens);
        out.extend_from_slice(&tokens[..count]);
    }
}

/// Encodes a long piece as overlapping windows, each merged on its own. If the encoding built so far and the next
/// window's encoding contain the same token at the same position, the encoding so far up to that token followed by
/// the window's tokens after it is the exact encoding of the combined text: both sides are exact encodings, and BPE
/// never merges across a token that both sides agree on (see the `bpe` crate README, "Properties of BPE"). Windows
/// with identical bytes reuse an earlier merge.
fn encode_long_piece(piece: &[u8], out: &mut Vec<u32>) {
    let base = out.len();
    // End offset in `piece` of each token in out[base..].
    let mut ends: Vec<usize> = Vec::with_capacity(piece.len() / 2);
    let mut cache: [MergedWindow; WINDOW_CACHE_SLOTS] = Default::default();
    let mut window_bytes = FIRST_WINDOW_BYTES;
    let mut done = 0;
    while done < piece.len() {
        // Start the window where a token of the encoding so far starts: the window's merges then begin from the
        // same position, so its first tokens usually match and stitch immediately.
        let first_overlap = ends.partition_point(|&end| end <= done.saturating_sub(window_bytes / OVERLAP_DIVISOR));
        let start = if first_overlap == 0 { 0 } else { ends[first_overlap - 1] };
        let window = start..(start + window_bytes).min(piece.len());
        // A long first token can pin `start` so the window cannot reach past `done`; grow until it does. Tokens are
        // at most SHORT_PIECE_BYTES long, so this stops well before MAX_WINDOW_BYTES.
        if window.end <= done {
            window_bytes *= 2;
            continue;
        }
        let bytes = &piece[window.clone()];
        let merged = &mut cache[window_cache_slot(bytes)];
        if piece[merged.range.clone()] != *bytes {
            merge_window(bytes, &mut merged.tokens, &mut merged.ends);
            merged.range = window.clone();
        }
        match find_shared_token(&out[base..], &ends, start, &merged.tokens, &merged.ends) {
            Some((done_index, window_index)) => {
                out.truncate(base + done_index + 1);
                ends.truncate(done_index + 1);
                out.extend_from_slice(&merged.tokens[window_index + 1..]);
                ends.extend(merged.ends[window_index + 1..].iter().map(|&end| start + end as usize));
            }
            // A window from the piece start is itself the exact encoding of everything up to its end.
            None if start == 0 => {
                out.truncate(base);
                out.extend_from_slice(&merged.tokens);
                ends.clear();
                ends.extend(merged.ends.iter().map(|&end| end as usize));
            }
            None if window_bytes < MAX_WINDOW_BYTES => {
                window_bytes *= 2;
                continue;
            }
            None => {
                out.truncate(base);
                merge_long_piece(piece, |id| out.push(id));
                return;
            }
        }
        done = window.end;
        // Hard stretches are local: go back to cheap windows once a stitch succeeds.
        window_bytes = FIRST_WINDOW_BYTES;
    }
}

/// Recently merged windows, direct-mapped by content hash. A hit is confirmed by comparing the bytes in full, so
/// periodic text whose windows recur at several phases reuses earlier merges instead of only the latest one.
const WINDOW_CACHE_SLOTS: usize = 16;

#[derive(Default)]
struct MergedWindow {
    /// Where in the piece the merged bytes came from; empty until the slot is filled.
    range: std::ops::Range<usize>,
    tokens: Vec<u32>,
    /// End offset of each token, relative to the window start.
    ends: Vec<u32>,
}

fn window_cache_slot(bytes: &[u8]) -> usize {
    let mut hash = bytes.len() as u64;
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        hash = (hash.rotate_left(5) ^ u64::from_le_bytes(chunk.try_into().unwrap())).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    for &byte in chunks.remainder() {
        hash = (hash.rotate_left(5) ^ byte as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    (hash >> 60) as usize % WINDOW_CACHE_SLOTS
}

/// Exact encoding of `window` into `tokens`, with each token's end offset in `ends`.
fn merge_window(window: &[u8], tokens: &mut Vec<u32>, ends: &mut Vec<u32>) {
    tokens.clear();
    ends.clear();
    if window.len() <= SHORT_PIECE_BYTES {
        encode_short_piece(window, tokens);
    } else {
        merge_long_piece(window, |id| tokens.push(id));
    }
    let mut end = 0;
    ends.extend(tokens.iter().map(|&id| {
        end += ordinary_token_bytes(id).len() as u32;
        end
    }));
}

/// Indices of a token present in both encodings with the same end offset (same id and end means same span).
fn find_shared_token(
    done_tokens: &[u32],
    done_ends: &[usize],
    window_start: usize,
    window_tokens: &[u32],
    window_ends: &[u32],
) -> Option<(usize, usize)> {
    let mut i = done_ends.partition_point(|&end| end <= window_start);
    let mut j = 0;
    while i < done_ends.len() && j < window_ends.len() {
        let done_end = done_ends[i];
        let window_end = window_start + window_ends[j] as usize;
        if done_end == window_end && done_tokens[i] == window_tokens[j] {
            return Some((i, j));
        }
        if done_end <= window_end {
            i += 1;
        }
        if window_end <= done_end {
            j += 1;
        }
    }
    None
}

/// Merges a piece of 2..=SHORT_PIECE_BYTES bytes; writes its tokens to the front of `tokens`, returns the count.
#[inline]
fn merge_short_piece(piece: &[u8], tokens: &mut [u32; SHORT_PIECE_BYTES]) -> usize {
    let mut count = piece.len();
    // ranks[i]: rank of merging tokens[i] and tokens[i + 1]; ranks[count - 1] stays NO_MERGE as a sentinel.
    let mut ranks = [NO_MERGE; SHORT_PIECE_BYTES];
    for i in 0..count {
        tokens[i] = BYTE_TOKENS[piece[i] as usize];
    }
    for i in 0..count - 1 {
        ranks[i] = byte_merge_rank(piece[i], piece[i + 1]);
    }
    loop {
        // Two vectorizable passes beat one branchy pass; `position` keeps the leftmost of tied ranks.
        let live = &ranks[..count - 1];
        let best_rank = live.iter().copied().min().unwrap_or(NO_MERGE);
        if best_rank == NO_MERGE {
            return count;
        }
        let best = live.iter().position(|&rank| rank == best_rank).unwrap_or(0);
        tokens[best] = best_rank;
        tokens.copy_within(best + 2..count, best + 1);
        ranks.copy_within(best + 2..count, best + 1);
        count -= 1;
        ranks[count - 1] = NO_MERGE;
        if best + 1 < count {
            ranks[best] = merge_rank(tokens[best], tokens[best + 1]);
        } else {
            ranks[best] = NO_MERGE;
        }
        if best > 0 {
            ranks[best - 1] = merge_rank(tokens[best - 1], tokens[best]);
        }
        if count == 1 {
            return 1;
        }
    }
}

/// Heap-driven merge for long pieces, O(n log n). Emits tokens in order.
fn merge_long_piece(piece: &[u8], mut emit: impl FnMut(u32)) {
    let len = piece.len();
    const NONE: u32 = u32::MAX;
    // Parts are identified by their first byte index. `next`/`prev` link live parts; `rank[i]` is the current
    // merge rank of part i with its successor, which invalidates stale heap entries.
    let mut tokens: Vec<u32> = piece.iter().map(|&byte| BYTE_TOKENS[byte as usize]).collect();
    let mut next: Vec<u32> = (1..=len as u32).collect();
    let mut prev: Vec<u32> = (0..len as u32).map(|i| i.wrapping_sub(1)).collect();
    next[len - 1] = NONE;
    let mut rank = vec![NO_MERGE; len];
    let mut heap = BinaryHeap::with_capacity(len);
    for i in 0..len - 1 {
        rank[i] = byte_merge_rank(piece[i], piece[i + 1]);
        if rank[i] != NO_MERGE {
            heap.push(Reverse((rank[i], i as u32)));
        }
    }
    while let Some(Reverse((merge, start))) = heap.pop() {
        let start = start as usize;
        if rank[start] != merge {
            continue;
        }
        let right = next[start] as usize;
        tokens[start] = merge;
        let after = next[right];
        next[start] = after;
        if after != NONE {
            prev[after as usize] = start as u32;
        }
        rank[right] = NO_MERGE;
        rank[start] = if after != NONE { merge_rank(merge, tokens[after as usize]) } else { NO_MERGE };
        if rank[start] != NO_MERGE {
            heap.push(Reverse((rank[start], start as u32)));
        }
        let before = prev[start];
        if before != NONE {
            let before = before as usize;
            rank[before] = merge_rank(tokens[before], merge);
            if rank[before] != NO_MERGE {
                heap.push(Reverse((rank[before], before as u32)));
            }
        }
    }
    let mut part = 0u32;
    while part != NONE {
        emit(tokens[part as usize]);
        part = next[part as usize];
    }
}
