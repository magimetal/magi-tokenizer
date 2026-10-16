//! Exact, fast `o200k_base` tokenizer.
//!
//! Token ids are identical to tiktoken's `o200k_base` (`encode_ordinary`, `encode_with_special_tokens`).
//! Tables are built at compile time, so there is no startup parsing and no global initialization.

mod bpe;
mod scanner;
mod table_hash;
mod tables;
mod unicode_classes;

pub use scanner::Pieces;

/// `<|endoftext|>` in `o200k_base`.
pub const END_OF_TEXT: u32 = 199_999;
/// `<|endofprompt|>` in `o200k_base`.
pub const END_OF_PROMPT: u32 = 200_018;
/// Special tokens of `o200k_base`, as (text, id).
pub const SPECIAL_TOKENS: [(&str, u32); 2] = [("<|endoftext|>", END_OF_TEXT), ("<|endofprompt|>", END_OF_PROMPT)];
/// Number of ordinary (non-special) tokens; ordinary ids are `0..ORDINARY_TOKEN_COUNT`.
pub const ORDINARY_TOKEN_COUNT: usize = tables::TOKEN_COUNT;

/// Splits `text` into the pieces the o200k_base pretokenization regex produces.
pub fn pieces(text: &str) -> Pieces<'_> {
    Pieces::new(text)
}

/// Encodes `text`, treating special-token text as ordinary text (tiktoken `encode_ordinary`).
pub fn encode_ordinary(text: &str) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(text.len() / 4 + 4);
    encode_ordinary_into(text, &mut tokens);
    tokens
}

/// Appends the ordinary encoding of `text` to `tokens`.
pub fn encode_ordinary_into(text: &str, tokens: &mut Vec<u32>) {
    for piece in pieces(text) {
        bpe::encode_piece(piece.as_bytes(), tokens);
    }
}

/// Number of tokens `encode_ordinary(text)` returns, without allocating the token list.
pub fn count_ordinary(text: &str) -> usize {
    pieces(text).map(|piece| bpe::count_piece(piece.as_bytes())).sum()
}

/// Encodes `text`, recognizing every special token (tiktoken `encode_with_special_tokens`).
pub fn encode_with_special_tokens(text: &str) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(text.len() / 4 + 4);
    for_each_special_split(text, |segment, special| {
        encode_ordinary_into(segment, &mut tokens);
        tokens.extend(special);
    });
    tokens
}

/// Number of tokens `encode_with_special_tokens(text)` returns.
pub fn count_with_special_tokens(text: &str) -> usize {
    let mut count = 0;
    for_each_special_split(text, |segment, special| count += count_ordinary(segment) + special.is_some() as usize);
    count
}

/// Calls `visit` with each ordinary segment and the special token that follows it (None after the last).
fn for_each_special_split(text: &str, mut visit: impl FnMut(&str, Option<u32>)) {
    let (mut segment_start, mut pos) = (0, 0);
    while let Some(offset) = text[pos..].find("<|") {
        let start = pos + offset;
        if let Some(&(name, id)) = SPECIAL_TOKENS.iter().find(|(name, _)| text[start..].starts_with(name)) {
            visit(&text[segment_start..start], Some(id));
            segment_start = start + name.len();
            pos = segment_start;
        } else {
            pos = start + 2;
        }
    }
    visit(&text[segment_start..], None);
}

/// Bytes of one token, ordinary or special. None for ids that are not tokens.
pub fn token_bytes(id: u32) -> Option<&'static [u8]> {
    if (id as usize) < ORDINARY_TOKEN_COUNT {
        Some(tables::ordinary_token_bytes(id))
    } else {
        SPECIAL_TOKENS.iter().find(|&&(_, special)| special == id).map(|(name, _)| name.as_bytes())
    }
}

/// Concatenated bytes of `tokens`. None if any id is not a token.
pub fn decode_bytes(tokens: &[u32]) -> Option<Vec<u8>> {
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for &id in tokens {
        bytes.extend_from_slice(token_bytes(id)?);
    }
    Some(bytes)
}

/// Decoded text of `tokens`. None if any id is not a token or the bytes are not valid UTF-8.
pub fn decode(tokens: &[u32]) -> Option<String> {
    String::from_utf8(decode_bytes(tokens)?).ok()
}
