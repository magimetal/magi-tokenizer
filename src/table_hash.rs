// Hash functions and table geometry shared by `build.rs` (table generation) and the library
// (lookups). Both sides must agree bit-for-bit, so they include this one file.

/// Slot count exponent for the vocabulary table (2^19 slots, ~38% load for 199,998 tokens).
pub const VOCAB_TABLE_BITS: u32 = 19;
/// Slot count exponent for the merge-pair table (2^20 slots, ~43% load for 446,189 pairs).
pub const PAIR_TABLE_BITS: u32 = 20;
/// Longest vocabulary token in bytes; longer pieces can never be a single token.
pub const MAX_TOKEN_BYTES: usize = 128;
/// Bytes of a token stored inline in its vocabulary slot (8-byte head + 4-byte tail).
pub const INLINE_TOKEN_BYTES: usize = 12;

/// Vocabulary slot second word layout: bits 0..32 tail bytes 8..12, 32..56 token id, 56..64 length.
pub const VOCAB_SLOT_ID_SHIFT: u32 = 32;
pub const VOCAB_SLOT_LEN_SHIFT: u32 = 56;
pub const VOCAB_SLOT_ID_MASK: u64 = 0x00FF_FFFF << VOCAB_SLOT_ID_SHIFT;

/// Merge-pair slot layout: bits 0..18 merged token id, 18..54 pair key (left << 18 | right).
pub const PAIR_ID_BITS: u32 = 18;
pub const PAIR_EMPTY_SLOT: u64 = u64::MAX;

#[inline(always)]
pub const fn vocab_slot_for(head: u64, tail: u32, len: usize) -> usize {
    let mixed = head.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((tail as u64) | ((len as u64) << 32));
    (mixed.wrapping_mul(0xD6E8_FEB8_6659_FD93) >> (64 - VOCAB_TABLE_BITS)) as usize
}

#[inline(always)]
pub const fn pair_key(left: u32, right: u32) -> u64 {
    ((left as u64) << PAIR_ID_BITS) | right as u64
}

#[inline(always)]
pub const fn pair_slot_for(key: u64) -> usize {
    (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - PAIR_TABLE_BITS)) as usize
}
