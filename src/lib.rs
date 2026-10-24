//! Exact, fast tokenizers for OpenAI's `o200k_base`, `o200k_harmony` and, behind features of the same name,
//! `cl100k_base`, `r50k_base` and `p50k_base` (which also enables `p50k_edit`) encodings.
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

/// Generates one encoding's public functions from its scanner, vocabulary and the module's `SPECIAL_TOKENS` and
/// `SPECIAL_TOKEN_TEXT_ORDER`.
macro_rules! encoding_api {
    ($name:literal, $scanner:ty, $vocabulary:ty) => {
        use crate::Pieces;

        const SPECIAL: crate::SpecialTokens = crate::SpecialTokens::new(&SPECIAL_TOKENS, SPECIAL_TOKEN_TEXT_ORDER);

        /// Number of ordinary (non-special) tokens. Their ids are `0..ORDINARY_TOKEN_COUNT`, except in `p50k_base`
        /// and `p50k_edit`, where id 50256 is `<|endoftext|>` and ordinary ids run to 50280.
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
            crate::reject_disallowed_special(text, &SPECIAL, allowed_special)?;
            Ok(encode_allowing_special(text, allowed_special))
        }

        /// Number of tokens `encode(text, allowed_special)` returns, or its error.
        pub fn count(text: &str, allowed_special: &[&str]) -> Result<usize, crate::DisallowedSpecialToken> {
            crate::reject_disallowed_special(text, &SPECIAL, allowed_special)?;
            Ok(count_allowing_special(text, allowed_special))
        }

        fn encode_with_allowed_special(text: &str, is_allowed: impl Fn(&str) -> bool) -> Vec<u32> {
            let mut tokens = Vec::with_capacity(text.len() / 4 + 4);
            crate::for_each_special_split(text, &SPECIAL, is_allowed, |segment, special| {
                encode_ordinary_into(segment, &mut tokens);
                tokens.extend(special);
            });
            tokens
        }

        fn count_with_allowed_special(text: &str, is_allowed: impl Fn(&str) -> bool) -> usize {
            let mut count = 0;
            crate::for_each_special_split(text, &SPECIAL, is_allowed, |segment, special| {
                count += count_ordinary(segment) + special.is_some() as usize
            });
            count
        }

        /// Bytes of one token, ordinary or special. None for ids that are not tokens.
        pub fn token_bytes(id: u32) -> Option<&'static [u8]> {
            crate::token_bytes_in::<$vocabulary>(id, &SPECIAL)
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
    /// Short list: special-token lookup scans it.
    const SPECIAL_TOKEN_TEXT_ORDER: &[u16] = &[];

    encoding_api!("o200k_base", crate::scanner::O200kScanner, crate::tables::O200kBaseVocabulary);
}

/// `o200k_harmony`: `gpt-oss-*` models. Same vocabulary and pretokenization as `o200k_base`, so ordinary encoding
/// and counting give the same ids; its special tokens are ids 199998–201087 (named tokens, the rest
/// `<|reserved_{id}|>`). Shares `o200k_base`'s tables, so it adds only the special-token text.
pub mod o200k_harmony {
    // Named-token constants, `SPECIAL_TOKENS` (by id) and `SPECIAL_TOKEN_TEXT_ORDER`, generated by build.rs.
    include!(concat!(env!("OUT_DIR"), "/o200k_harmony_special_tokens.rs"));

    encoding_api!("o200k_harmony", crate::scanner::O200kScanner, crate::tables::O200kBaseVocabulary);
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
    /// Short list: special-token lookup scans it.
    const SPECIAL_TOKEN_TEXT_ORDER: &[u16] = &[];

    encoding_api!("cl100k_base", crate::scanner::Cl100kScanner, crate::tables::Cl100kBaseVocabulary);
}

/// `r50k_base` (also called `gpt2`): GPT-2 and GPT-3 models such as `davinci`. Requires feature `r50k_base`.
#[cfg(feature = "r50k_base")]
pub mod r50k_base {
    /// `<|endoftext|>` in `r50k_base`.
    pub const END_OF_TEXT: u32 = 50_256;
    /// Special tokens of `r50k_base`, as (text, id).
    pub const SPECIAL_TOKENS: [(&str, u32); 1] = [("<|endoftext|>", END_OF_TEXT)];
    /// Short list: special-token lookup scans it.
    const SPECIAL_TOKEN_TEXT_ORDER: &[u16] = &[];

    encoding_api!("r50k_base", crate::scanner::R50kScanner, crate::tables::R50kBaseVocabulary);
}

/// `p50k_base`: Codex models, `text-davinci-002` and `text-davinci-003`. Requires feature `p50k_base`. Its 50,280
/// ordinary tokens are `r50k_base`'s plus 24 runs of spaces, ids 50257–50280; id 50256 is `<|endoftext|>`.
#[cfg(feature = "p50k_base")]
pub mod p50k_base {
    /// `<|endoftext|>` in `p50k_base`.
    pub const END_OF_TEXT: u32 = 50_256;
    /// Special tokens of `p50k_base`, as (text, id).
    pub const SPECIAL_TOKENS: [(&str, u32); 1] = [("<|endoftext|>", END_OF_TEXT)];
    /// Short list: special-token lookup scans it.
    const SPECIAL_TOKEN_TEXT_ORDER: &[u16] = &[];

    encoding_api!("p50k_base", crate::scanner::R50kScanner, crate::tables::P50kBaseVocabulary);
}

/// `p50k_edit`: edit models such as `text-davinci-edit-001` and `code-davinci-edit-001`. Same vocabulary and
/// pretokenization as `p50k_base`, so ordinary encoding and counting give the same ids; adds fill-in-the-middle
/// special tokens. Requires feature `p50k_base` and shares its tables.
#[cfg(feature = "p50k_base")]
pub mod p50k_edit {
    /// `<|endoftext|>` in `p50k_edit`.
    pub const END_OF_TEXT: u32 = 50_256;
    /// `<|fim_prefix|>` in `p50k_edit`.
    pub const FIM_PREFIX: u32 = 50_281;
    /// `<|fim_middle|>` in `p50k_edit`.
    pub const FIM_MIDDLE: u32 = 50_282;
    /// `<|fim_suffix|>` in `p50k_edit`.
    pub const FIM_SUFFIX: u32 = 50_283;
    /// Special tokens of `p50k_edit`, as (text, id).
    pub const SPECIAL_TOKENS: [(&str, u32); 4] = [
        ("<|endoftext|>", END_OF_TEXT),
        ("<|fim_prefix|>", FIM_PREFIX),
        ("<|fim_middle|>", FIM_MIDDLE),
        ("<|fim_suffix|>", FIM_SUFFIX),
    ];
    /// Short list: special-token lookup scans it.
    const SPECIAL_TOKEN_TEXT_ORDER: &[u16] = &[];

    encoding_api!("p50k_edit", crate::scanner::R50kScanner, crate::tables::P50kBaseVocabulary);
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

/// One encoding's special tokens, checked at compile time: sorted by id, each `<|` text `|>` with no other `|>`.
/// Then the token starting at a `<|` can only be the text up to the first `|>` after it. Long lists find that `|>`
/// within the longest token's length and binary-search the one candidate; short lists compare each token directly.
struct SpecialTokens {
    by_id: &'static [(&'static str, u32)],
    /// Indices into `by_id` in byte order of their text, for binary search; empty for short lists, which are scanned.
    text_order: &'static [u16],
    longest_text: usize,
}

impl SpecialTokens {
    const fn new(by_id: &'static [(&'static str, u32)], text_order: &'static [u16]) -> Self {
        assert!(text_order.is_empty() || text_order.len() == by_id.len());
        let mut longest_text = 0;
        let mut index = 0;
        while index < by_id.len() {
            assert!(index == 0 || by_id[index - 1].1 < by_id[index].1, "special tokens must be sorted by id");
            let text = by_id[index].0.as_bytes();
            let len = text.len();
            assert!(len >= 4 && text[0] == b'<' && text[1] == b'|', "special token must start with `<|`");
            assert!(text[len - 2] == b'|' && text[len - 1] == b'>', "special token must end with `|>`");
            let mut at = 0;
            while at < len - 2 {
                assert!(!(text[at] == b'|' && text[at + 1] == b'>'), "special token has `|>` before its end");
                at += 1;
            }
            if len > longest_text {
                longest_text = len;
            }
            index += 1;
        }
        Self { by_id, text_order, longest_text }
    }

    /// The special token whose text starts at byte `start` of `text`, which holds `<|`, as (text, id).
    #[inline]
    fn starting_at(&self, text: &str, start: usize) -> Option<(&'static str, u32)> {
        let rest = &text[start..];
        if self.text_order.is_empty() {
            return self.by_id.iter().find(|&&(name, _)| rest.starts_with(name)).copied();
        }
        let window = &rest.as_bytes()[2..rest.len().min(self.longest_text)];
        let close = window.windows(2).position(|pair| pair == b"|>")?;
        // `>` is ASCII, so the end is a char boundary.
        let candidate = &rest[..2 + close + 2];
        let found = self.text_order.binary_search_by(|&index| self.by_id[index as usize].0.cmp(candidate)).ok()?;
        Some(self.by_id[self.text_order[found] as usize])
    }

    fn text_of(&self, id: u32) -> Option<&'static str> {
        let index = self.by_id.binary_search_by_key(&id, |&(_, special)| special).ok()?;
        Some(self.by_id[index].0)
    }
}

/// First special token at or after byte `pos` for which `is_allowed` holds, as (start, text, id).
fn find_special_token(
    text: &str,
    mut pos: usize,
    special: &SpecialTokens,
    is_allowed: impl Fn(&str) -> bool,
) -> Option<(usize, &'static str, u32)> {
    while let Some(offset) = text[pos..].find("<|") {
        let start = pos + offset;
        if let Some((name, id)) = special.starting_at(text, start)
            && is_allowed(name)
        {
            return Some((start, name, id));
        }
        pos = start + 2;
    }
    None
}

/// Calls `visit` with each ordinary segment and the allowed special token that follows it (None after the last).
fn for_each_special_split(
    text: &str,
    special: &SpecialTokens,
    is_allowed: impl Fn(&str) -> bool,
    mut visit: impl FnMut(&str, Option<u32>),
) {
    let mut segment_start = 0;
    while let Some((start, name, id)) = find_special_token(text, segment_start, special, &is_allowed) {
        visit(&text[segment_start..start], Some(id));
        segment_start = start + name.len();
    }
    visit(&text[segment_start..], None);
}

/// Error for the leftmost special token in `text` that is not in `allowed_special`.
fn reject_disallowed_special(
    text: &str,
    special: &SpecialTokens,
    allowed_special: &[&str],
) -> Result<(), DisallowedSpecialToken> {
    match find_special_token(text, 0, special, |name| !allowed_special.contains(&name)) {
        Some((position, token, _)) => Err(DisallowedSpecialToken { token, position }),
        None => Ok(()),
    }
}

fn token_bytes_in<V: Vocabulary>(id: u32, special: &SpecialTokens) -> Option<&'static [u8]> {
    // Ids below ORDINARY_ID_END without an ordinary token have empty bytes; they may be special (p50k_base's 50256).
    let ordinary = ((id as usize) < V::ORDINARY_ID_END).then(|| tables::ordinary_token_bytes::<V>(id));
    match ordinary {
        Some(bytes) if !bytes.is_empty() => Some(bytes),
        _ => special.text_of(id).map(str::as_bytes),
    }
}
