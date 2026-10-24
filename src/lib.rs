//! Exact, fast tokenizers for OpenAI's `o200k_base` and (feature `cl100k_base`) `cl100k_base` encodings.
//!
//! Token ids are identical to tiktoken's (`encode_ordinary`, `encode_with_special_tokens`, and `encode` with an
//! allow-list). Each encoding is a module with the same functions; the crate root re-exports [`o200k_base`]. Tables
//! are built at compile time, so there is no startup parsing and no global initialization.

mod bpe;
#[cfg(feature = "harness")]
#[doc(hidden)]
pub mod harness;
mod scanner;
mod table_hash;
mod tables;
mod unicode_classes;

pub use o200k_base::*;
pub use scanner::Pieces;

use scanner::{Scanner, ScannedPieces};
use tables::Vocabulary;

/// Generates one encoding's public functions from its scanner, vocabulary and special tokens.
macro_rules! encoding_api {
    ($name:literal, $scanner:ty, $vocabulary:ty) => {
        use crate::Pieces;

        /// Number of ordinary (non-special) tokens; ordinary ids are `0..ORDINARY_TOKEN_COUNT`.
        pub const ORDINARY_TOKEN_COUNT: usize = <$vocabulary as crate::Vocabulary>::TOKEN_COUNT;

        #[doc = concat!("Splits `text` into the pieces the ", $name, " pretokenization regex produces.")]
        pub fn pieces(text: &str) -> Pieces<'_> {
            Pieces::new::<$scanner>(text)
        }

        /// Encodes `text`, treating special-token text as ordinary text (tiktoken `encode_ordinary`).
        pub fn encode_ordinary(text: &str) -> Vec<u32> {
            let mut tokens = Vec::with_capacity(text.len() / 4 + 4);
            encode_ordinary_into(text, &mut tokens);
            tokens
        }

        /// Appends the ordinary encoding of `text` to `tokens`.
        pub fn encode_ordinary_into(text: &str, tokens: &mut Vec<u32>) {
            crate::encode_scanned_into::<$scanner, $vocabulary>(text, tokens)
        }

        /// Number of tokens `encode_ordinary(text)` returns, without building the token list. Working memory per
        /// piece is bounded, except when a long piece falls back to the whole-piece merge (20 bytes per piece byte).
        pub fn count_ordinary(text: &str) -> usize {
            crate::count_scanned::<$scanner, $vocabulary>(text)
        }

        /// Encodes `text`, recognizing every special token (tiktoken `encode_with_special_tokens`).
        pub fn encode_with_special_tokens(text: &str) -> Vec<u32> {
            encode_with_allowed_special(text, |_| true)
        }

        /// Number of tokens `encode_with_special_tokens(text)` returns.
        pub fn count_with_special_tokens(text: &str) -> usize {
            count_with_allowed_special(text, |_| true)
        }

        /// Encodes `text`, recognizing only the special tokens in `allowed_special`; other special-token text is
        /// encoded as ordinary text (tiktoken-rs `CoreBPE::encode(text, allowed_special)`).
        pub fn encode_allowing_special(text: &str, allowed_special: &[&str]) -> Vec<u32> {
            encode_with_allowed_special(text, |name| allowed_special.contains(&name))
        }

        /// Number of tokens `encode_allowing_special(text, allowed_special)` returns.
        pub fn count_allowing_special(text: &str, allowed_special: &[&str]) -> usize {
            count_with_allowed_special(text, |name| allowed_special.contains(&name))
        }

        /// Python tiktoken's default `encode(text, allowed_special=...)`: an error for the leftmost special-token
        /// text not in `allowed_special`, otherwise `encode_allowing_special(text, allowed_special)`.
        pub fn encode(text: &str, allowed_special: &[&str]) -> Result<Vec<u32>, crate::DisallowedSpecialToken> {
            crate::reject_disallowed_special(text, &SPECIAL_TOKENS, allowed_special)?;
            Ok(encode_allowing_special(text, allowed_special))
        }

        /// Number of tokens `encode(text, allowed_special)` returns, or its error.
        pub fn count(text: &str, allowed_special: &[&str]) -> Result<usize, crate::DisallowedSpecialToken> {
            crate::reject_disallowed_special(text, &SPECIAL_TOKENS, allowed_special)?;
            Ok(count_allowing_special(text, allowed_special))
        }

        fn encode_with_allowed_special(text: &str, is_allowed: impl Fn(&str) -> bool) -> Vec<u32> {
            let mut tokens = Vec::with_capacity(text.len() / 4 + 4);
            crate::for_each_special_split(text, &SPECIAL_TOKENS, is_allowed, |segment, special| {
                encode_ordinary_into(segment, &mut tokens);
                tokens.extend(special);
            });
            tokens
        }

        fn count_with_allowed_special(text: &str, is_allowed: impl Fn(&str) -> bool) -> usize {
            let mut count = 0;
            crate::for_each_special_split(text, &SPECIAL_TOKENS, is_allowed, |segment, special| {
                count += count_ordinary(segment) + special.is_some() as usize
            });
            count
        }

        /// Bytes of one token, ordinary or special. None for ids that are not tokens.
        pub fn token_bytes(id: u32) -> Option<&'static [u8]> {
            crate::token_bytes_in::<$vocabulary>(id, &SPECIAL_TOKENS)
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
    };
}

/// `o200k_base`: GPT-4o, o-series and GPT-5 models. Always enabled; the crate root re-exports it.
pub mod o200k_base {
    /// `<|endoftext|>` in `o200k_base`.
    pub const END_OF_TEXT: u32 = 199_999;
    /// `<|endofprompt|>` in `o200k_base`.
    pub const END_OF_PROMPT: u32 = 200_018;
    /// Special tokens of `o200k_base`, as (text, id).
    pub const SPECIAL_TOKENS: [(&str, u32); 2] = [("<|endoftext|>", END_OF_TEXT), ("<|endofprompt|>", END_OF_PROMPT)];

    encoding_api!("o200k_base", crate::scanner::O200kScanner, crate::tables::O200kBaseVocabulary);
}

/// `cl100k_base`: GPT-4, GPT-3.5 Turbo and `text-embedding-*` models. Requires feature `cl100k_base`.
#[cfg(feature = "cl100k_base")]
pub mod cl100k_base {
    /// `<|endoftext|>` in `cl100k_base`.
    pub const END_OF_TEXT: u32 = 100_257;
    /// `<|fim_prefix|>` in `cl100k_base`.
    pub const FIM_PREFIX: u32 = 100_258;
    /// `<|fim_middle|>` in `cl100k_base`.
    pub const FIM_MIDDLE: u32 = 100_259;
    /// `<|fim_suffix|>` in `cl100k_base`.
    pub const FIM_SUFFIX: u32 = 100_260;
    /// `<|endofprompt|>` in `cl100k_base`.
    pub const END_OF_PROMPT: u32 = 100_276;
    /// Special tokens of `cl100k_base`, as (text, id).
    pub const SPECIAL_TOKENS: [(&str, u32); 5] = [
        ("<|endoftext|>", END_OF_TEXT),
        ("<|fim_prefix|>", FIM_PREFIX),
        ("<|fim_middle|>", FIM_MIDDLE),
        ("<|fim_suffix|>", FIM_SUFFIX),
        ("<|endofprompt|>", END_OF_PROMPT),
    ];

    encoding_api!("cl100k_base", crate::scanner::Cl100kScanner, crate::tables::Cl100kBaseVocabulary);
}

#[inline]
fn encode_scanned_into<S: Scanner, V: Vocabulary>(text: &str, tokens: &mut Vec<u32>) {
    for piece in ScannedPieces::<S>::new(text) {
        bpe::encode_piece::<V>(piece, tokens);
    }
}

#[inline]
fn count_scanned<S: Scanner, V: Vocabulary>(text: &str) -> usize {
    ScannedPieces::<S>::new(text).map(bpe::count_piece::<V>).sum()
}

/// Special-token text that `encode` or `count` found but was not allowed, like Python tiktoken's `ValueError`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DisallowedSpecialToken {
    /// The special token's text, for example `<|endoftext|>`.
    pub token: &'static str,
    /// Byte offset of the token's text in the input.
    pub position: usize,
}

impl std::fmt::Display for DisallowedSpecialToken {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "disallowed special token {:?} at byte {}", self.token, self.position)
    }
}

impl std::error::Error for DisallowedSpecialToken {}

/// First special token at or after byte `pos` for which `is_allowed` holds, as (start, text, id).
/// Every special token contains `<|` only at its start, so matches cannot overlap.
fn find_special_token(
    text: &str,
    mut pos: usize,
    special_tokens: &[(&'static str, u32)],
    is_allowed: impl Fn(&str) -> bool,
) -> Option<(usize, &'static str, u32)> {
    while let Some(offset) = text[pos..].find("<|") {
        let start = pos + offset;
        let at_start = special_tokens.iter().find(|&&(name, _)| text[start..].starts_with(name) && is_allowed(name));
        if let Some(&(name, id)) = at_start {
            return Some((start, name, id));
        }
        pos = start + 2;
    }
    None
}

/// Calls `visit` with each ordinary segment and the allowed special token that follows it (None after the last).
fn for_each_special_split(
    text: &str,
    special_tokens: &[(&'static str, u32)],
    is_allowed: impl Fn(&str) -> bool,
    mut visit: impl FnMut(&str, Option<u32>),
) {
    let mut segment_start = 0;
    while let Some((start, name, id)) = find_special_token(text, segment_start, special_tokens, &is_allowed) {
        visit(&text[segment_start..start], Some(id));
        segment_start = start + name.len();
    }
    visit(&text[segment_start..], None);
}

/// Error for the leftmost special token in `text` that is not in `allowed_special`.
fn reject_disallowed_special(
    text: &str,
    special_tokens: &[(&'static str, u32)],
    allowed_special: &[&str],
) -> Result<(), DisallowedSpecialToken> {
    match find_special_token(text, 0, special_tokens, |name| !allowed_special.contains(&name)) {
        Some((position, token, _)) => Err(DisallowedSpecialToken { token, position }),
        None => Ok(()),
    }
}

fn token_bytes_in<V: Vocabulary>(id: u32, special_tokens: &[(&'static str, u32)]) -> Option<&'static [u8]> {
    if (id as usize) < V::TOKEN_COUNT {
        Some(tables::ordinary_token_bytes::<V>(id))
    } else {
        special_tokens.iter().find(|&&(_, special)| special == id).map(|(name, _)| name.as_bytes())
    }
}
