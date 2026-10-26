// Byte-pair merging of one pretokenized piece. Matches tiktoken exactly: repeatedly merge the adjacent pair whose
// merged token has the lowest rank, leftmost first. Parts are always vocabulary tokens, so "the rank of the
// concatenated bytes" equals the pair table's merged id for (left token, right token).

use crate::tables::{
    NO_MERGE, Vocabulary, byte_merge_rank, merge_rank, ordinary_token_bytes, premerged_char, token_id_for_bytes,
};

/// Pieces up to this length merge in stack arrays with linear minimum search. Every token is at most this long.
const SHORT_PIECE_BYTES: usize = 128;
/// Long pieces are encoded as overlapping windows of FIRST_WINDOW_BYTES; a window doubles while no stitch point is
/// found, up to MAX_WINDOW_BYTES, after which the whole piece is merged directly. The size sets how windows of
/// periodic text advance, so it decides whether they recur and hit the cache: measured on long runs, 128 beat 64,
/// and 48 or 96 were six times slower.
const FIRST_WINDOW_BYTES: usize = 128;
const MAX_WINDOW_BYTES: usize = 4096;
/// A new window reaches back over the last `window / OVERLAP_DIVISOR` bytes of the encoding so far.
const OVERLAP_DIVISOR: usize = 8;

// Index math below relies on these: short pieces fit stack arrays and u8 links with a power-of-two position mask,
// a window that misses `done` grows past it after one doubling, and window token ends fit u32.
const _: () = {
    assert!(SHORT_PIECE_BYTES.is_power_of_two() && SHORT_PIECE_BYTES <= u8::MAX as usize);
    assert!(crate::table_hash::MAX_TOKEN_BYTES <= SHORT_PIECE_BYTES);
    assert!(FIRST_WINDOW_BYTES >= SHORT_PIECE_BYTES && MAX_WINDOW_BYTES <= u32::MAX as usize);
};

/// Appends the tokens of `piece` to `out`. `buffers` is scratch space reused across the pieces of one text.
#[inline]
pub(crate) fn encode_piece<V: Vocabulary>(piece: &[u8], out: &mut Vec<u32>, buffers: &mut WindowBuffers) {
    if piece.len() <= SHORT_PIECE_BYTES {
        encode_short_piece::<V>(piece, out);
    } else {
        encode_long_piece::<V>(piece, out, buffers);
    }
}

/// Number of tokens `piece` encodes to. Long pieces are counted in bounded memory unless the whole-piece fallback runs.
#[inline]
pub(crate) fn count_piece<V: Vocabulary>(piece: &[u8], buffers: &mut WindowBuffers) -> usize {
    if piece.len() > SHORT_PIECE_BYTES {
        let mut count = 0;
        if !merge_in_windows::<V>(piece, buffers, |tokens| count += tokens.len()) {
            #[cfg(feature = "harness")]
            crate::harness::record_fallback();
            count = 0;
            merge_long_piece::<V>(piece, |_| count += 1);
        }
        count
    } else if token_id_for_bytes::<V>(piece).is_some() {
        1
    } else {
        merge_short_piece::<V>(piece, &mut [0u32; SHORT_PIECE_BYTES])
    }
}

#[inline]
fn encode_short_piece<V: Vocabulary>(piece: &[u8], out: &mut Vec<u32>) {
    if let Some(id) = token_id_for_bytes::<V>(piece) {
        out.push(id);
    } else {
        let mut tokens = [0u32; SHORT_PIECE_BYTES];
        let count = merge_short_piece::<V>(piece, &mut tokens);
        out.extend_from_slice(&tokens[..count]);
    }
}

fn encode_long_piece<V: Vocabulary>(piece: &[u8], out: &mut Vec<u32>, buffers: &mut WindowBuffers) {
    let base = out.len();
    if !merge_in_windows::<V>(piece, buffers, |tokens| out.extend_from_slice(tokens)) {
        #[cfg(feature = "harness")]
        crate::harness::record_fallback();
        out.truncate(base);
        merge_long_piece::<V>(piece, |id| out.push(id));
    }
}

/// How far before `done` a window can start: its overlap plus one token, since it starts at a token boundary.
/// Tokens ending at least this far back are final.
const REACH_BACK_BYTES: usize = MAX_WINDOW_BYTES / OVERLAP_DIVISOR + SHORT_PIECE_BYTES;
/// Pending tokens are checked for final ones once there are this many.
const COMMIT_BATCH_TOKENS: usize = 4096;

/// Scratch space of `merge_in_windows`, kept across the long pieces of one text so each piece reuses the
/// allocations of the ones before it instead of making its own.
#[derive(Default)]
pub(crate) struct WindowBuffers {
    /// Tokens not yet committed, with the end offset in the piece of each.
    tokens: Vec<u32>,
    ends: Vec<usize>,
    cache: [MergedWindow; WINDOW_CACHE_SLOTS],
}

/// Passes the exact encoding of a long piece to `commit`, in order and in batches, as overlapping windows each
/// merged on their own. If the encoding built so far and the next window's encoding contain the same token at the
/// same position, the encoding so far up to that token followed by the window's tokens after it is the exact
/// encoding of the combined text: both sides are exact encodings, and BPE never merges across a token that both
/// sides agree on (see the `bpe` crate README, "Properties of BPE"). Windows with identical bytes reuse an earlier
/// merge.
///
/// Returns false when a stretch has no stitch point within MAX_WINDOW_BYTES. A prefix may already have been
/// committed; the caller discards it and merges the whole piece instead.
pub(crate) fn merge_in_windows<V: Vocabulary>(
    piece: &[u8],
    buffers: &mut WindowBuffers,
    mut commit: impl FnMut(&[u32]),
) -> bool {
    // Only the stretch of pending tokens after a window's start is ever replaced, and windows start at or after
    // `committed_end`. Cached ranges index an earlier piece, so they are emptied: an empty range never matches.
    let WindowBuffers { tokens, ends, cache } = buffers;
    tokens.clear();
    ends.clear();
    for merged in cache.iter_mut() {
        merged.range = 0..0;
    }
    let mut committed_end = 0;
    let mut window_bytes = FIRST_WINDOW_BYTES;
    let mut done = 0;
    #[cfg(feature = "harness")]
    let mut stitches = 0;
    while done < piece.len() {
        #[cfg(feature = "harness")]
        if crate::harness::window_merge_forced_to_fail(stitches) {
            return false;
        }
        // Start the window where a token of the encoding so far starts: the window's merges then begin from the
        // same position, so its first tokens usually match and stitch immediately. Pending tokens all end after
        // `committed_end`, and it is at least REACH_BACK_BYTES behind, so it is the right start when none qualify.
        let first_overlap = ends.partition_point(|&end| end <= done.saturating_sub(window_bytes / OVERLAP_DIVISOR));
        let start = if first_overlap == 0 { committed_end } else { ends[first_overlap - 1] };
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
            merge_window::<V>(bytes, &mut merged.tokens, &mut merged.ends);
            merged.range = window.clone();
        }
        match find_shared_token(tokens, ends, start, &merged.tokens, &merged.ends) {
            Some((done_index, window_index)) => {
                tokens.truncate(done_index + 1);
                ends.truncate(done_index + 1);
                tokens.extend_from_slice(&merged.tokens[window_index + 1..]);
                ends.extend(merged.ends[window_index + 1..].iter().map(|&end| start + end as usize));
            }
            // A window from the piece start is itself the exact encoding of everything up to its end.
            None if start == 0 => {
                tokens.clone_from(&merged.tokens);
                ends.clear();
                ends.extend(merged.ends.iter().map(|&end| end as usize));
            }
            None if window_bytes < MAX_WINDOW_BYTES => {
                window_bytes *= 2;
                continue;
            }
            None => return false,
        }
        done = window.end;
        #[cfg(feature = "harness")]
        {
            stitches += 1;
        }
        // Hard stretches are local: go back to cheap windows once a stitch succeeds.
        window_bytes = FIRST_WINDOW_BYTES;
        if tokens.len() >= COMMIT_BATCH_TOKENS {
            let final_tokens = ends.partition_point(|&end| end + REACH_BACK_BYTES <= done);
            if final_tokens > 0 {
                committed_end = ends[final_tokens - 1];
                commit(&tokens[..final_tokens]);
                tokens.drain(..final_tokens);
                ends.drain(..final_tokens);
            }
        }
    }
    commit(tokens);
    true
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
    let (chunks, remainder) = bytes.as_chunks::<8>();
    for chunk in chunks {
        hash = (hash.rotate_left(5) ^ u64::from_le_bytes(*chunk)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    for &byte in remainder {
        hash = (hash.rotate_left(5) ^ u64::from(byte)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    (hash >> 60) as usize % WINDOW_CACHE_SLOTS
}

/// Exact encoding of `window` into `tokens`, with each token's end offset in `ends`.
fn merge_window<V: Vocabulary>(window: &[u8], tokens: &mut Vec<u32>, ends: &mut Vec<u32>) {
    tokens.clear();
    ends.clear();
    if window.len() <= SHORT_PIECE_BYTES {
        encode_short_piece::<V>(window, tokens);
    } else {
        merge_long_piece::<V>(window, |id| tokens.push(id));
    }
    let mut end = 0;
    ends.extend(tokens.iter().map(|&id| {
        end += ordinary_token_bytes::<V>(id).len() as u32;
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

/// Below this length, shifting parts down after each merge is cheaper than linking them.
const LINKED_MERGE_BYTES: usize = 32;

/// Merges a piece of at most SHORT_PIECE_BYTES bytes; writes its tokens to the front of `tokens`, returns the count.
#[inline]
fn merge_short_piece<V: Vocabulary>(piece: &[u8], tokens: &mut [u32; SHORT_PIECE_BYTES]) -> usize {
    if piece.len() < LINKED_MERGE_BYTES { merge_by_shifting::<V>(piece, tokens) } else { merge_linked::<V>(piece, tokens) }
}

/// A part a merge starts from: `len` bytes at `start` in the piece, which are token `token`.
#[derive(Clone, Copy)]
struct StartingPart {
    start: usize,
    len: usize,
    token: u32,
}

/// Parts a merge starts from, in order: single bytes, except that a character `premerged_char` allows starts
/// whole. Its bytes would merge into it before joining a neighbor anyway, so the result is the same with fewer
/// merges: long pieces of 3-byte characters such as CJK need about a third of the parts.
fn starting_parts<V: Vocabulary>(piece: &[u8]) -> impl Iterator<Item = StartingPart> + '_ {
    let mut start = 0;
    std::iter::from_fn(move || {
        let &byte = piece.get(start)?;
        let (token, len) = premerged_char::<V>(piece, start).unwrap_or((V::byte_tokens()[byte as usize], 1));
        let part = StartingPart { start, len, token };
        start += len;
        Some(part)
    })
}

/// Rank of merging two adjacent starting parts; pairs of single bytes use the dense byte-pair table.
#[inline(always)]
fn starting_merge_rank<V: Vocabulary>(piece: &[u8], left: StartingPart, right: StartingPart) -> u32 {
    if left.len == 1 && right.len == 1 {
        byte_merge_rank::<V>(piece[left.start], piece[right.start])
    } else {
        merge_rank::<V>(left.token, right.token)
    }
}

/// Keeps parts packed at the front: after each merge the parts and ranks above it shift down by one, so each
/// minimum search covers only the live parts. `position` keeps the leftmost of tied ranks.
#[inline]
fn merge_by_shifting<V: Vocabulary>(piece: &[u8], tokens: &mut [u32; SHORT_PIECE_BYTES]) -> usize {
    // ranks[i]: rank of merging tokens[i] and tokens[i + 1].
    let mut ranks = [NO_MERGE; LINKED_MERGE_BYTES];
    let mut count = 0;
    let mut previous: Option<StartingPart> = None;
    for part in starting_parts::<V>(piece) {
        tokens[count] = part.token;
        if let Some(left) = previous {
            ranks[count - 1] = starting_merge_rank::<V>(piece, left, part);
        }
        previous = Some(part);
        count += 1;
    }
    while count > 1 {
        let live = &ranks[..count - 1];
        let best_rank = live.iter().copied().min().unwrap_or(NO_MERGE);
        if best_rank == NO_MERGE {
            break;
        }
        let best = live.iter().position(|&rank| rank == best_rank).unwrap_or(0);
        tokens[best] = best_rank;
        // At most LINKED_MERGE_BYTES elements: a plain loop beats a memmove call.
        for i in best + 1..count - 1 {
            tokens[i] = tokens[i + 1];
            ranks[i] = ranks[i + 1];
        }
        count -= 1;
        ranks[count - 1] = NO_MERGE;
        if best + 1 < count {
            ranks[best] = merge_rank::<V>(tokens[best], tokens[best + 1]);
        }
        if best > 0 {
            ranks[best - 1] = merge_rank::<V>(tokens[best - 1], tokens[best]);
        }
    }
    count
}

/// Parts stay in the slot of their first byte, linked by `next`/`prev`, so a merge moves nothing. `keys[i]` packs
/// the rank of merging part i with its successor above i, so the minimum key is the lowest rank and, among equal
/// ranks, the leftmost part. Merged-away and last parts hold u32::MAX. `block_min` keeps the minimum of each
/// KEY_BLOCK keys: a merge changes at most three keys, so only their blocks are rescanned.
fn merge_linked<V: Vocabulary>(piece: &[u8], tokens: &mut [u32; SHORT_PIECE_BYTES]) -> usize {
    const POSITION_BITS: u32 = SHORT_PIECE_BYTES.trailing_zeros();
    const NO_KEY: u32 = u32::MAX;
    const KEY_BLOCK: usize = 32;
    const BLOCKS: usize = SHORT_PIECE_BYTES / KEY_BLOCK;
    // A packed key (rank below 2^PAIR_ID_BITS) fits u32 and never equals NO_KEY.
    const { assert!(crate::table_hash::PAIR_ID_BITS + POSITION_BITS < 32) };
    const { assert!(SHORT_PIECE_BYTES.is_multiple_of(KEY_BLOCK)) };
    let key = |rank: u32, position: usize| {
        if rank == NO_MERGE { NO_KEY } else { rank << POSITION_BITS | position as u32 }
    };
    let len = piece.len();
    let mut keys = [NO_KEY; SHORT_PIECE_BYTES];
    let mut next = [0u8; SHORT_PIECE_BYTES];
    let mut prev = [0u8; SHORT_PIECE_BYTES];
    let mut count = 0;
    if piece.is_ascii() {
        // No character to start whole: the plain byte loop, which matters for long ASCII runs.
        for i in 0..len {
            tokens[i] = V::byte_tokens()[piece[i] as usize];
            next[i] = i as u8 + 1;
            prev[i] = (i as u8).wrapping_sub(1);
        }
        for i in 0..len - 1 {
            keys[i] = key(byte_merge_rank::<V>(piece[i], piece[i + 1]), i);
        }
        count = len;
    } else {
        let mut previous: Option<StartingPart> = None;
        for part in starting_parts::<V>(piece) {
            tokens[part.start] = part.token;
            next[part.start] = (part.start + part.len) as u8;
            if let Some(left) = previous {
                prev[part.start] = left.start as u8;
                keys[left.start] = key(starting_merge_rank::<V>(piece, left, part), left.start);
            }
            previous = Some(part);
            count += 1;
        }
    }
    let block_of = |keys: &[u32; SHORT_PIECE_BYTES], position: usize| {
        let block = position / KEY_BLOCK * KEY_BLOCK;
        keys[block..block + KEY_BLOCK].iter().copied().min().unwrap_or(NO_KEY)
    };
    let mut block_min = [NO_KEY; BLOCKS];
    for (block, min) in block_min.iter_mut().enumerate() {
        *min = block_of(&keys, block * KEY_BLOCK);
    }
    loop {
        let best = block_min.iter().copied().min().unwrap_or(NO_KEY);
        if best == NO_KEY {
            break;
        }
        let start = (best & (SHORT_PIECE_BYTES as u32 - 1)) as usize;
        let merged = best >> POSITION_BITS;
        let right = next[start] as usize;
        let after = next[right] as usize;
        tokens[start] = merged;
        keys[right] = NO_KEY;
        next[start] = after as u8;
        keys[start] = if after < len {
            prev[after] = start as u8;
            key(merge_rank::<V>(merged, tokens[after]), start)
        } else {
            NO_KEY
        };
        block_min[right / KEY_BLOCK] = block_of(&keys, right);
        block_min[start / KEY_BLOCK] = block_of(&keys, start);
        if start > 0 {
            let before = prev[start] as usize;
            keys[before] = key(merge_rank::<V>(tokens[before], merged), before);
            block_min[before / KEY_BLOCK] = block_of(&keys, before);
        }
        count -= 1;
    }
    // Compact live parts to the front; each part's slot is at or after its output position.
    let mut part = 0;
    for slot in 0..count {
        tokens[slot] = tokens[part];
        part = next[part] as usize;
    }
    count
}

/// Index of a part in `merge_long_piece`: u32 halves link memory, usize handles pieces of 4 GiB and more.
trait PartIndex: Copy + PartialEq {
    const NONE: Self;
    fn from_index(index: usize) -> Self;
    fn index(self) -> usize;
}

impl PartIndex for u32 {
    const NONE: Self = u32::MAX;
    fn from_index(index: usize) -> Self {
        index as u32
    }
    fn index(self) -> usize {
        self as usize
    }
}

impl PartIndex for usize {
    const NONE: Self = usize::MAX;
    fn from_index(index: usize) -> Self {
        index
    }
    fn index(self) -> usize {
        self
    }
}

/// Merge for long pieces, O(n log n), in memory allocated once: five u32 arrays as long as the piece (u32 indices),
/// or two u32 and three usize arrays for pieces of 4 GiB and more. Emits tokens in order.
fn merge_long_piece<V: Vocabulary>(piece: &[u8], emit: impl FnMut(u32)) {
    if piece.len() < u32::MAX as usize {
        merge_parts::<V, u32>(piece, emit);
    } else {
        merge_parts::<V, usize>(piece, emit);
    }
}

fn merge_parts<V: Vocabulary, I: PartIndex>(piece: &[u8], mut emit: impl FnMut(u32)) {
    let len = piece.len();
    if len == 0 {
        return;
    }
    // Parts are identified by their first byte index. `next`/`prev` link live parts; `rank[i]` is the rank of
    // merging part i with its successor, NO_MERGE for the last part and for bytes that start no part.
    let mut tokens = vec![0u32; len];
    let mut next = vec![I::NONE; len];
    let mut prev = vec![I::NONE; len];
    let mut rank = vec![NO_MERGE; len];
    let mut previous: Option<StartingPart> = None;
    for part in starting_parts::<V>(piece) {
        tokens[part.start] = part.token;
        if let Some(left) = previous {
            next[left.start] = I::from_index(part.start);
            prev[part.start] = I::from_index(left.start);
            rank[left.start] = starting_merge_rank::<V>(piece, left, part);
        }
        previous = Some(part);
    }
    let mut lowest = LowestRankTree::<I>::new(&rank);
    loop {
        let start = lowest.lowest_part();
        let merge = rank[start];
        if merge == NO_MERGE {
            break;
        }
        let right = next[start].index();
        tokens[start] = merge;
        let after = next[right];
        next[start] = after;
        if after != I::NONE {
            prev[after.index()] = I::from_index(start);
        }
        rank[right] = NO_MERGE;
        lowest.update(&rank, right);
        rank[start] = if after != I::NONE { merge_rank::<V>(merge, tokens[after.index()]) } else { NO_MERGE };
        lowest.update(&rank, start);
        let before = prev[start];
        if before != I::NONE {
            let before = before.index();
            rank[before] = merge_rank::<V>(tokens[before], merge);
            lowest.update(&rank, before);
        }
    }
    let mut part = 0;
    loop {
        emit(tokens[part]);
        let following = next[part];
        if following == I::NONE {
            return;
        }
        part = following.index();
    }
}

/// Tournament tree over the parts of `merge_parts`, giving the part with the lowest merge rank, leftmost among
/// equal ranks. Leaf i (node len + i) is part i and is not stored; each of nodes 1..len holds the winner of its two
/// children. Every node above 1 has one parent, so node 1 holds the overall winner. Its fixed size, unlike a heap
/// of stale entries, keeps the merge's memory exact.
struct LowestRankTree<I> {
    nodes: Vec<I>,
}

impl<I: PartIndex> LowestRankTree<I> {
    fn new(rank: &[u32]) -> Self {
        let mut tree = Self { nodes: vec![I::NONE; rank.len()] };
        for node in (1..rank.len()).rev() {
            tree.nodes[node] = I::from_index(tree.winner_below(rank, node));
        }
        tree
    }

    /// The part with the lowest rank; its rank is NO_MERGE when no merge is left.
    fn lowest_part(&self) -> usize {
        if self.nodes.len() > 1 { self.nodes[1].index() } else { 0 }
    }

    /// Replays the matches above `part` after its rank changed.
    fn update(&mut self, rank: &[u32], part: usize) {
        let mut node = (rank.len() + part) / 2;
        while node > 0 {
            self.nodes[node] = I::from_index(self.winner_below(rank, node));
            node /= 2;
        }
    }

    fn winner_below(&self, rank: &[u32], node: usize) -> usize {
        let left = self.part_at(rank.len(), 2 * node);
        let right = self.part_at(rank.len(), 2 * node + 1);
        if (rank[left], left) <= (rank[right], right) { left } else { right }
    }

    fn part_at(&self, len: usize, node: usize) -> usize {
        if node >= len { node - len } else { self.nodes[node].index() }
    }
}
