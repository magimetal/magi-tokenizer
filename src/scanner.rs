// Hand-written matchers for the pretokenization regexes of each encoding. They return the same pieces as
// tiktoken's fancy-regex search, alternative by alternative. o200k_base:
//
//   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
// | [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
// | \p{N}{1,3}
// |  ?[^\s\p{L}\p{N}]+[\r\n/]*
// | \s*[\r\n]+
// | \s+(?!\S)
// | \s+
//
// cl100k_base:
//
//   '(?i:[sdmt]|ll|ve|re)
// | [^\r\n\p{L}\p{N}]?+\p{L}++
// | \p{N}{1,3}+
// |  ?[^\s\p{L}\p{N}]++[\r\n]*+
// | \s++$
// | \s*[\r\n]
// | \s+(?!\S)
// | \s
//
// Matching order and backtracking follow the scanner design of toks (Apache-2.0, Copyright 2026 Actual
// Computer Inc.), src/core/k3_o200k_c.c and src/core/kernels.h, rewritten in Rust for magi-tokenizer.
// Atoms are Unicode scalars; input is valid UTF-8 because callers pass `&str`.

use crate::unicode_classes::{ASCII_CLASSES, BLOCK_CLASSES, BLOCK_INDEX};
use std::marker::PhantomData;

/// A pretokenizer: where the piece starting at `pos` ends (`pos < text.len()`).
pub(crate) trait Scanner {
    fn piece_end(text: &[u8], pos: usize) -> usize;
}

pub(crate) struct O200kScanner;

impl Scanner for O200kScanner {
    #[inline(always)]
    fn piece_end(text: &[u8], pos: usize) -> usize {
        o200k_piece_end(text, pos)
    }
}

#[cfg(feature = "cl100k_base")]
pub(crate) struct Cl100kScanner;

#[cfg(feature = "cl100k_base")]
impl Scanner for Cl100kScanner {
    #[inline(always)]
    fn piece_end(text: &[u8], pos: usize) -> usize {
        cl100k_piece_end(text, pos)
    }
}

const BASE_MASK: u8 = 0x07;
const PUNCTUATION: u8 = 0;
const LETTER: u8 = 1;
const NUMBER: u8 = 2;
const SPACE: u8 = 3;
/// `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]`
const UPPER: u8 = 0x08;
/// `[\p{Ll}\p{Lm}\p{Lo}\p{M}]`
const LOWER: u8 = 0x10;
const CASED: u8 = UPPER | LOWER;
/// U+017F LATIN SMALL LETTER LONG S, which `(?i:'s)` also matches.
const LONG_S: u32 = 0x017F;

// `atom_at` indexes BLOCK_CLASSES with `block << 8 | low byte`; every block index must name a full 256-entry block.
const _: () = {
    let mut i = 0;
    while i < BLOCK_INDEX.len() {
        assert!((BLOCK_INDEX[i] as usize + 1) * 256 <= BLOCK_CLASSES.len());
        i += 1;
    }
};
const _: () = assert!(BLOCK_INDEX.len() << 8 == 0x110000);

/// Class byte, scalar value and UTF-8 length of the scalar starting at `pos` (`pos < text.len()`).
#[inline(always)]
fn atom_at(text: &[u8], pos: usize) -> (u8, u32, usize) {
    let lead = text[pos];
    if lead < 0x80 {
        return (ASCII_CLASSES[lead as usize], lead as u32, 1);
    }
    let continuation = |offset: usize| (text[pos + offset] & 0x3F) as u32;
    let (scalar, len) = if lead < 0xE0 {
        (((lead & 0x1F) as u32) << 6 | continuation(1), 2)
    } else if lead < 0xF0 {
        (((lead & 0x0F) as u32) << 12 | continuation(1) << 6 | continuation(2), 3)
    } else {
        (((lead & 0x07) as u32) << 18 | continuation(1) << 12 | continuation(2) << 6 | continuation(3), 4)
    };
    let block = BLOCK_INDEX[(scalar >> 8) as usize] as usize;
    (BLOCK_CLASSES[block << 8 | (scalar & 0xFF) as usize], scalar, len)
}

#[inline(always)]
fn class_at(text: &[u8], pos: usize) -> (u8, usize) {
    let lead = text[pos];
    if lead < 0x80 {
        return (ASCII_CLASSES[lead as usize], 1);
    }
    let (class, _, len) = atom_at(text, pos);
    (class, len)
}

/// End of the maximal run of atoms whose class has any bit of `flags`, starting at `from`.
#[inline(always)]
fn flag_run_end(text: &[u8], mut end: usize, flags: u8) -> usize {
    while end < text.len() {
        let (class, len) = class_at(text, end);
        if class & flags == 0 {
            break;
        }
        end += len;
    }
    end
}

/// End of the maximal run of atoms with base class `base`, starting at `from`.
#[inline(always)]
fn base_run_end(text: &[u8], mut end: usize, base: u8) -> usize {
    while end < text.len() {
        let (class, len) = class_at(text, end);
        if class & BASE_MASK != base {
            break;
        }
        end += len;
    }
    end
}

/// End of `(?i:'s|'t|'re|'ve|'m|'ll|'d)` at `pos`, or `pos` when no contraction suffix follows.
#[inline(always)]
fn contraction_end(text: &[u8], pos: usize) -> usize {
    if pos + 1 >= text.len() || text[pos] != b'\'' {
        return pos;
    }
    let (_, first, first_len) = atom_at(text, pos + 1);
    let second_pos = pos + 1 + first_len;
    match first {
        0x73 | 0x53 | 0x74 | 0x54 | 0x6D | 0x4D | 0x64 | 0x44 | LONG_S => second_pos,
        0x72 | 0x52 | 0x76 | 0x56 | 0x6C | 0x4C if second_pos < text.len() => {
            let second = text[second_pos] | 0x20;
            let wanted = if first | 0x20 == 0x6C { b'l' } else { b'e' };
            if second == wanted { second_pos + 1 } else { pos }
        }
        _ => pos,
    }
}

/// Both letter alternatives at `start`, whose atom has class `class` and length `len` and is cased.
/// Returns (first alternative end, second alternative end), 0 meaning no match.
#[inline(always)]
fn letter_alternatives_end(text: &[u8], start: usize, mut class: u8, mut len: usize) -> (usize, usize) {
    let mut upper_end = start;
    let mut last_lower_end = 0;
    while class & UPPER != 0 {
        if class & LOWER != 0 {
            last_lower_end = upper_end + len;
        }
        upper_end += len;
        class = 0;
        if upper_end < text.len() {
            (class, len) = class_at(text, upper_end);
        }
    }
    if class & LOWER != 0 {
        return (contraction_end(text, flag_run_end(text, upper_end + len, LOWER)), 0);
    }
    if last_lower_end != 0 {
        return (contraction_end(text, last_lower_end), 0);
    }
    if upper_end > start {
        return (0, contraction_end(text, upper_end));
    }
    (0, 0)
}

/// `\s*[\r\n]+|\s+(?!\S)|\s+` at `pos`.
#[inline(always)]
fn whitespace_end(text: &[u8], pos: usize) -> usize {
    let mut end = pos;
    let mut last_newline_end = 0;
    let mut last_start = pos;
    while end < text.len() {
        let (class, len) = class_at(text, end);
        match class & BASE_MASK {
            SPACE => {}
            NEWLINE => last_newline_end = end + len,
            _ => break,
        }
        last_start = end;
        end += len;
    }
    if last_newline_end != 0 {
        last_newline_end
    } else if end == text.len() || last_start == pos {
        end
    } else {
        last_start
    }
}
const NEWLINE: u8 = 4;

/// End of `\p{N}{1,3}` whose first atom ends at `second`.
#[inline(always)]
fn number_piece_end(text: &[u8], second: usize) -> usize {
    let mut end = second;
    for _ in 0..2 {
        if end >= text.len() {
            break;
        }
        let (class, len) = class_at(text, end);
        if class & BASE_MASK != NUMBER {
            break;
        }
        end += len;
    }
    end
}

/// End of the o200k_base piece that starts at `pos` (`pos < text.len()`).
#[inline(always)]
fn o200k_piece_end(text: &[u8], pos: usize) -> usize {
    let (class, scalar, first_len) = atom_at(text, pos);
    let second = pos + first_len;
    let end = match class & BASE_MASK {
        LETTER => {
            let (first_alternative, second_alternative) = letter_alternatives_end(text, pos, class, first_len);
            first_alternative.max(second_alternative)
        }
        NUMBER => number_piece_end(text, second),
        base @ (PUNCTUATION | SPACE) => prefixed_piece_end(text, pos, base, class, scalar, second),
        _ => whitespace_end(text, pos),
    };
    // Every alternative above ends past `pos`; the guard keeps the scan loop finite regardless.
    if end > pos { end } else { second }
}

/// A piece starting with a punctuation or whitespace atom, which may prefix a letter piece.
#[inline(always)]
fn prefixed_piece_end(text: &[u8], pos: usize, base: u8, class: u8, scalar: u32, second: usize) -> usize {
    let (second_class, second_len) = if second < text.len() { class_at(text, second) } else { (0xFF, 0) };
    let mut prefixed_second_alternative = 0;
    if second < text.len() && second_class & CASED != 0 {
        let (first_alternative, second_alternative) =
            letter_alternatives_end(text, second, second_class, second_len);
        if first_alternative != 0 {
            return first_alternative;
        }
        prefixed_second_alternative = second_alternative;
    }
    if class & CASED != 0 {
        // A combining mark: the first alternative without the optional prefix always matches.
        let (first_alternative, _) = letter_alternatives_end(text, pos, class, second - pos);
        if first_alternative != 0 {
            return first_alternative;
        }
    }
    if prefixed_second_alternative != 0 {
        return prefixed_second_alternative;
    }
    let mut end = if base == PUNCTUATION {
        base_run_end(text, second, PUNCTUATION)
    } else if scalar == 0x20 && second < text.len() && second_class & BASE_MASK == PUNCTUATION {
        base_run_end(text, second + second_len, PUNCTUATION)
    } else {
        return whitespace_end(text, pos);
    };
    while end < text.len() && matches!(text[end], b'\r' | b'\n' | b'/') {
        end += 1;
    }
    end
}

/// End of the run of `[\r\n]` atoms starting at `end`.
#[cfg(feature = "cl100k_base")]
#[inline(always)]
fn newline_run_end(text: &[u8], mut end: usize) -> usize {
    while end < text.len() && matches!(text[end], b'\r' | b'\n') {
        end += 1;
    }
    end
}

/// End of the cl100k_base piece that starts at `pos` (`pos < text.len()`).
#[cfg(feature = "cl100k_base")]
#[inline(always)]
fn cl100k_piece_end(text: &[u8], pos: usize) -> usize {
    let (class, scalar, first_len) = atom_at(text, pos);
    let second = pos + first_len;
    let base = class & BASE_MASK;
    let end = match base {
        LETTER => base_run_end(text, second, LETTER),
        NUMBER => number_piece_end(text, second),
        PUNCTUATION | SPACE => {
            // `'(?i:[sdmt]|ll|ve|re)` is the same suffix set as o200k_base's contractions.
            let contraction = contraction_end(text, pos);
            if contraction > pos {
                return contraction;
            }
            let (second_class, second_len) = if second < text.len() { class_at(text, second) } else { (0xFF, 0) };
            // `[^\r\n\p{L}\p{N}]?+\p{L}++`: the optional atom is taken whenever it can be, so letters must follow it.
            if second_class & BASE_MASK == LETTER {
                base_run_end(text, second + second_len, LETTER)
            } else if base == PUNCTUATION {
                newline_run_end(text, base_run_end(text, second, PUNCTUATION))
            } else if scalar == 0x20 && second_class & BASE_MASK == PUNCTUATION {
                newline_run_end(text, base_run_end(text, second + second_len, PUNCTUATION))
            } else {
                cl100k_whitespace_end(text, pos)
            }
        }
        _ => cl100k_whitespace_end(text, pos),
    };
    // Every alternative above ends past `pos`; the guard keeps the scan loop finite regardless.
    if end > pos { end } else { second }
}

/// `\s++$|\s*[\r\n]|\s+(?!\S)|\s` at `pos`.
#[cfg(feature = "cl100k_base")]
#[inline(always)]
fn cl100k_whitespace_end(text: &[u8], pos: usize) -> usize {
    let mut end = pos;
    let mut last_newline_end = 0;
    let mut last_start = pos;
    while end < text.len() {
        let (class, len) = class_at(text, end);
        match class & BASE_MASK {
            SPACE => {}
            NEWLINE => last_newline_end = end + len,
            _ => break,
        }
        last_start = end;
        end += len;
    }
    if end == text.len() {
        end
    } else if last_newline_end != 0 {
        last_newline_end
    } else if last_start == pos {
        end
    } else {
        last_start
    }
}

/// Iterator over pretokenized pieces of a string, for the encoding that produced it.
#[derive(Clone, Debug)]
pub struct Pieces<'a> {
    text: &'a str,
    pos: usize,
    piece_end: fn(&[u8], usize) -> usize,
}

impl<'a> Pieces<'a> {
    pub(crate) fn new<S: Scanner>(text: &'a str) -> Self {
        Self { text, pos: 0, piece_end: S::piece_end }
    }
}

impl<'a> Iterator for Pieces<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        if self.pos >= self.text.len() {
            return None;
        }
        let start = self.pos;
        self.pos = (self.piece_end)(self.text.as_bytes(), start);
        // Piece boundaries are scalar boundaries, so the slice is valid UTF-8.
        Some(&self.text[start..self.pos])
    }
}

impl core::iter::FusedIterator for Pieces<'_> {}

/// `Pieces` with the scanner fixed at compile time, for the encoding loops.
pub(crate) struct ScannedPieces<'a, S> {
    text: &'a str,
    pos: usize,
    scanner: PhantomData<S>,
}

impl<'a, S: Scanner> ScannedPieces<'a, S> {
    pub(crate) fn new(text: &'a str) -> Self {
        Self { text, pos: 0, scanner: PhantomData }
    }
}

impl<'a, S: Scanner> Iterator for ScannedPieces<'a, S> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        if self.pos >= self.text.len() {
            return None;
        }
        let start = self.pos;
        self.pos = S::piece_end(self.text.as_bytes(), start);
        Some(&self.text[start..self.pos])
    }
}
