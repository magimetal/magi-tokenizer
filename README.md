# magi-toks

An exact, fast Rust tokenizer for OpenAI's `o200k_base` encoding.

For every input, magi-toks produces the same token IDs as
[tiktoken](https://github.com/openai/tiktoken) (`encode_ordinary` and `encode_with_special_tokens`). Speed is the
goal, but exactness is required. A faster result with one wrong token ID is a failure.

> **Status: experimental, version 0.1.0.** The library passes the full differential verification suite described
> below. It is not published. A portable benchmark harness and several hardening items are still open (see
> [Known limitations](#known-limitations)).

## Contents

- [Scope](#scope)
- [Quick start](#quick-start)
- [API](#api)
- [How it works](#how-it-works)
- [Verification](#verification)
- [Benchmarks](#benchmarks)
- [Repository layout](#repository-layout)
- [Development](#development)
- [Known limitations](#known-limitations)
- [Provenance and licensing](#provenance-and-licensing)

## Scope

| Item | State |
| --- | --- |
| Encoding | `o200k_base`: 199,998 ordinary tokens (IDs 0–199,997) plus special tokens |
| Special tokens | `<\|endoftext\|>` = 199,999, `<\|endofprompt\|>` = 200,018 |
| Models | Upstream tiktoken maps the GPT-4o, o-series and GPT-5 model prefixes to `o200k_base`. GPT-5.6, GPT-6.0, GPT-6.1 and Astra are target models, but their encodings are **not confirmed yet**. The library exposes the encoding, not a model-name router, so it does not claim support for unconfirmed models. |
| Input | `&str` (valid UTF-8). Decoding can return raw bytes for token sequences that are not valid UTF-8. |
| Platforms | Any Rust target. Tables are emitted in the target's byte order. Development and measurement were done on ARM64 macOS. |

## Quick start

The crate is not on crates.io. Use a path or git dependency:

```toml
[dependencies]
magi-toks = { path = "../magi-toks" }
```

```rust
let text = "Hello, world! <|endoftext|>";

// Treat special-token text as ordinary text (tiktoken `encode_ordinary`).
let ids = magi_toks::encode_ordinary(text);
assert_eq!(magi_toks::count_ordinary(text), ids.len());

// Recognise special tokens (tiktoken `encode_with_special_tokens`).
let with_special = magi_toks::encode_with_special_tokens(text);
assert_eq!(*with_special.last().unwrap(), magi_toks::END_OF_TEXT);

assert_eq!(magi_toks::decode(&ids).as_deref(), Some(text));
```

There is no setup step. The first call is as fast as every later call, because all tables are built at compile
time (see [Startup](#startup)).

## API

All functions are free functions. There is no tokenizer object, no global state and no locking, so every function
can be called from any number of threads at once.

| Function | Purpose |
| --- | --- |
| `pieces(text) -> Pieces` | Iterator over the pretokenization pieces (the substrings tiktoken's regex produces). |
| `encode_ordinary(text) -> Vec<u32>` | Token IDs. Special-token text is encoded as ordinary text. |
| `encode_ordinary_into(text, &mut Vec<u32>)` | Same, appended to an existing vector so its allocation can be reused. |
| `count_ordinary(text) -> usize` | Number of tokens `encode_ordinary` returns. |
| `encode_with_special_tokens(text) -> Vec<u32>` | Token IDs with `<\|endoftext\|>` and `<\|endofprompt\|>` recognised. |
| `count_with_special_tokens(text) -> usize` | Number of tokens `encode_with_special_tokens` returns. |
| `token_bytes(id) -> Option<&'static [u8]>` | Bytes of one ordinary or special token; `None` for an unknown ID. |
| `decode_bytes(ids) -> Option<Vec<u8>>` | Concatenated bytes; `None` if an ID is unknown. |
| `decode(ids) -> Option<String>` | Decoded text; `None` if an ID is unknown or the bytes are not valid UTF-8. |

Constants: `END_OF_TEXT`, `END_OF_PROMPT`, `SPECIAL_TOKENS`, `ORDINARY_TOKEN_COUNT`.

## How it works

Tokenization has two stages, and both must match tiktoken exactly:

1. **Pretokenization.** A regex splits the text into pieces: words with an optional leading space or
   punctuation character, groups of 1–3 digits, punctuation runs, whitespace runs and English contractions.
2. **Byte-pair merging.** Each piece starts as single bytes. The adjacent pair whose merged token has the lowest
   rank is merged, the leftmost pair if ranks tie, until no adjacent pair forms a vocabulary token. In
   `o200k_base`, a token's rank is its ID.

magi-toks keeps those semantics and changes how each stage is computed.

### Build-time tables (`build.rs`, `src/table_hash.rs`, `src/tables.rs`)

`build.rs` parses `assets/o200k_base.tiktoken` at compile time. It has no dependencies. It writes aligned binary
table images for the target's byte order:

- **Token bytes and offsets:** `ID -> bytes` for decoding.
- **Vocabulary hash table:** 2^19 slots, about 38% full. Each slot is two `u64`s: the token's first 8 bytes, plus
  4 tail bytes, the length and the ID. Tokens up to 12 bytes are confirmed inside the slot; longer ones are
  compared against the token bytes.
- **Pair table:** 2^20 slots, about 43% full. One `u64` per slot holds `(left ID, right ID) -> merged ID` for all
  446,189 ways to split a vocabulary token into two vocabulary tokens. IDs fit in 18 bits.
- **Byte-pair ranks:** a dense 65,536-entry array for the first merge step, which is the most common lookup.
- **Byte-token IDs:** the 256 single-byte tokens (IDs 0–255 in `o200k_base`).

The build checks the token and pair counts so that a changed vocabulary file fails the build instead of producing
wrong tables. `src/table_hash.rs` holds the hash functions and is included by both `build.rs` and the library, so
both sides use identical code.

### Pretokenizer (`src/scanner.rs`, `src/unicode_classes.rs`)

The o200k regex uses look-ahead (`\s+(?!\S)`), so tiktoken runs it with `fancy-regex`, a backtracking engine. That
is slow. magi-toks replaces it with a hand-written matcher that tries the seven alternatives in the same order and
reproduces the same backtracking results, including:

- case-insensitive contractions, where `'ſ` must not match `'s`;
- the optional leading character that is not a letter, digit, CR or LF;
- whitespace runs that give back their last space to the following word;
- `\s*[\r\n]+` and `\s+(?!\S)`.

The matcher reads Unicode scalar values and looks up one class byte per code point in `src/unicode_classes.rs`
(about 110 KB, generated). The generator, `tools/gen-unicode`, computes every class with the exact `regex` 1.12.4
and `regex-syntax` 0.8.11 versions that tiktoken-rs 0.12.0 uses. As a result, the Unicode data (Unicode 16)
matches the reference tokenizer. A newer Unicode release would quietly change token boundaries, so the versions
are pinned.

The scanner's matching order follows the design of
[toks](https://github.com/actualcomputer/toks) (`src/core/k3_o200k_c.c`), rewritten in Rust. See
[Provenance](#provenance-and-licensing).

### Byte-pair merging (`src/bpe.rs`)

Each piece goes through one of three paths, chosen by its length:

1. **Whole-token lookup.** Most pieces are a single vocabulary token. One vocabulary-table probe returns the ID,
   with no merging.
2. **Short pieces (≤ 128 bytes).** Fixed-size arrays on the stack hold the current tokens and the merge rank of
   each adjacent pair. Each step finds the minimum rank with two simple passes: a minimum, then the first position
   with that rank, which keeps leftmost tie-breaking. The step then merges the pair, shifts the arrays and updates
   the two neighbouring ranks. For short inputs this beats a heap: no allocation, data stays in cache, and the
   compiler vectorises the scans. 128 bytes is also the longest token in `o200k_base`.
3. **Long pieces (> 128 bytes).** Overlapping windows, described next.

#### Long pieces: overlapping windows

A heap-based merge costs O(n log n) per piece and is slow on long repetitive runs, such as 300,000 spaces, `aaaa…`
or `ababab…`. magi-toks encodes a long piece as a series of short windows and joins them. The method rests on a
property of BPE noted in the [`bpe` crate README](https://github.com/github/rust-gems/tree/main/crates/bpe)
("Properties of BPE"):

> If two exact encodings of overlapping text contain the same token at the same byte position, the first encoding
> up to that token, followed by the second encoding after it, is the exact encoding of the combined text.

The loop:

1. Encode the first 64-byte window exactly (short path).
2. Start the next window at a token boundary of the current result, reaching back about 1/8 of the window size.
   Encode it exactly.
3. Look for a **stitch point**: a token with the same ID and the same end offset in both encodings. Both lists of
   token end offsets are sorted, so one merge-style pass finds it.
4. If found, keep the current result up to that token, append the window's tokens after it, and move forward.
5. If not found, double the window size and try again, up to 4,096 bytes. If 4,096 bytes still has no stitch
   point, encode the whole piece with the exact heap merge. This fallback is a safety net.
6. After each successful stitch, the window size goes back to 64 bytes, because hard stretches are usually local.

Two details make this fast on repetitive text:

- **Window cache.** A 16-slot cache, indexed by a hash of the window's bytes, stores recent window encodings. A hit
  counts only after a **full byte comparison**, never on the hash alone. Periodic text such as `"0123456789" × N`
  or `"!@#$%^&*()" × N` reuses earlier merges at every phase of the period. The cache lives for one call only;
  there is no state between calls.
- **Boundary-aligned starts.** A window that starts where a token of the current result starts usually repeats
  the same merges, so the stitch point appears early.

The previous heap-only merge stays in the code. It is the fallback, and it merges windows over 128 bytes.

## Verification

`magi-toks-bench verify` compares magi-toks with **tiktoken-rs 0.12.0** (the reference) and with the original
regex, run through `fancy-regex` 0.17.0. For every text it checks:

- the pieces, against the reference regex;
- `encode_ordinary` IDs and `count_ordinary`;
- `decode` round trip;
- `encode_with_special_tokens` IDs and `count_with_special_tokens`.

It also checks that each valid-UTF-8 vocabulary token that forms a single piece encodes to its own ID.

| Suite | Texts | Bytes | What it covers |
| --- | ---: | ---: | --- |
| `edges` | 883 | 109,014 | Contractions, `ſ`, title-case letters, combining marks, NBSP and ideographic space, CR/LF mixes, special-token fragments, runs of 1–140 characters of six kinds |
| `unicode_sweep` | 6,370 | 52,020,015 | Every Unicode scalar value in six contexts that exercise each regex alternative, plus mixed triples of code points |
| `fuzz` | 200,000 | 14,861,094 | Seeded random strings from a hostile alphabet (contractions, marks, emoji, special-token parts, whitespace kinds) |
| `source_7` / `_48` / `_4096` | 1.2M / 176k / 2k | 8,453,332 each | Real Rust and C source, cut at three chunk sizes |
| `multilingual_256` | 1,921 | 491,200 | CJK, Hangul, Cyrillic, Greek, Arabic, Devanagari, Thai, emoji ZWJ sequences |
| `fixture_200k` | 1 | 938,163 | Shared `o200k_base` comparison fixture |
| `long_runs` | 9 | 1,760,000 | 300k `a`, 300k spaces, repeated digits, `ab`, combining marks, punctuation, CJK, cycles |
| `long_random` | 988 | 2,172,677 | Long single pieces from small alphabets (letters, CJK, marks, punctuation, whitespace) with lengths around the window sizes (129 … 20,000) and runs with one interruption |
| `session_snapshots` | 20 | 11,850,632 | Real coding-agent session transcripts |
| `token_identity` | 198,436 tokens | — | Each single-piece vocabulary token encodes to itself |

Latest result: **0 mismatches in every suite**. The run takes about 18 seconds on an Apple M5 Pro.

The window encoder was also checked separately: its counts match `bpe-openai` on random long pieces of up to
240 KB.

## Benchmarks

`magi-toks-bench bench` measures count and encode throughput for magi-toks, tiktoken-rs 0.12.0 and bpe-openai 0.3.2
on the same inputs. Before timing, it checks that all three return the same total token count. Each figure is the
median of 5 trials of at least 150 ms each.

Latest run, count throughput in MiB/s (the tool prints the label "MB/s", but the unit is MiB/s):

| Workload | MiB | magi-toks | tiktoken-rs | bpe-openai | vs tiktoken | vs bpe-openai |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixture_200k` | 0.895 | 247.9 | 37.3 | 76.9 | 6.65× | 3.23× |
| `source_1MiB` | 1.000 | 244.3 | 37.8 | 78.0 | 6.46× | 3.13× |
| `source_48B_chunks` | 0.250 | 221.4 | 32.7 | 74.8 | 6.77× | 2.96× |
| `source_256B_chunks` | 0.250 | 278.0 | 37.6 | 82.1 | 7.39× | 3.38× |
| `multilingual` | 0.468 | 122.4 | 42.9 | 95.9 | 2.86× | 1.28× |
| `long_runs` | 1.678 | 536.7 | 14.1 | 162.1 | 38.06× | 3.31× |
| `session_snapshots` | 30.748 | 251.5 | 38.5 | 81.2 | 6.53× | 3.10× |

Encode throughput follows the same pattern: 2.84–7.24× tiktoken-rs on ordinary text and 1.32–3.53× bpe-openai.

Before the window encoder, `long_runs` counted at 19.5 MiB/s, about 8× slower than bpe-openai.

### Where magi-toks is still slower

Long single pieces with no repetition, where every window must be merged from scratch (magi-toks / bpe-openai,
milliseconds per call):

| Input | magi-toks | bpe-openai |
| --- | ---: | ---: |
| 240 KB random CJK from 8 characters | 8.2 | 2.5 |
| 200 KB random `a`/`b` | 6.2 | 4.7 |
| 2,000 random `a`/`b` pieces, 129–1,024 bytes | 36.3 | 24.1 |

Random pieces drawn from 26 or 52 letters are already faster than bpe-openai. Speeding up these cases is the next
area of work.

### Startup

magi-toks needs no runtime initialisation. On a contended machine, the first count after process start took
0.107 ms, against 166 ms for tiktoken-rs and 78 ms for bpe-openai (vocabulary loading). Process launch time is not
included. Memory (RSS) has not been measured yet.

### Caveats

- All figures come from one ARM64 Apple M5 Pro with other work running in the background. Treat them as
  directional until they are repeated on a quiet machine with the tokenizers run in alternating order.
- The native `toks` C library has not been benchmarked yet.
- The harness reads fixtures and source trees from hard-coded local paths (see
  [Known limitations](#known-limitations)).

## Repository layout

```text
assets/o200k_base.tiktoken   o200k_base vocabulary (base64 token, rank per line), 3.6 MB
build.rs                     parses the vocabulary, emits binary tables + bindings into OUT_DIR
src/lib.rs                   public API: pieces, encode/count (ordinary and special), decode
src/scanner.rs               hand-written o200k pretokenizer
src/unicode_classes.rs       generated per-code-point class table (do not edit by hand)
src/bpe.rs                   short-piece, window and heap byte-pair merging
src/tables.rs                table lookups: vocabulary, pair ranks, byte ranks, token bytes
src/table_hash.rs            hash functions and table geometry shared with build.rs
bench/                       magi-toks-bench: differential verification and benchmarks
tools/gen-unicode/           standalone generator for src/unicode_classes.rs (pinned regex versions)
```

## Development

Requirements: Rust 1.85 or later (edition 2024). Development used Rust 1.98.1.

```sh
# Library
cargo build --release

# Exact differential verification against tiktoken-rs and fancy-regex (about 18 s)
cargo run --release -p magi-toks-bench -- verify

# Throughput, all three tokenizers; an optional filter matches workload names
cargo run --release -p magi-toks-bench -- bench
cargo run --release -p magi-toks-bench -- bench long

# First-call time in a fresh process
cargo run --release -p magi-toks-bench -- startup magi        # or tiktoken, bpe-openai

# Regenerate the Unicode class table (only when the pinned regex versions change)
cd tools/gen-unicode && cargo run --release
```

The release profile uses fat LTO and one codegen unit. Build scripts are compiled with `opt-level = 3`, so table
generation stays fast in debug builds as well.

The project has no unit tests by design. Correctness comes from the differential `verify` command. Run it after
any change to the scanner, tables or merge code.

## Known limitations

- **Portability of the harness.** `bench/src/main.rs` reads `COMPARISON_ASSETS` and `SOURCE_ROOTS` from absolute
  paths on the author's machine. If the session fixture is missing, its suite is skipped, but the other fixtures
  are required.
- **Benchmark labels.** Throughput is printed as "MB/s" but is MiB/s. The token-identity line counts all
  valid-UTF-8 tokens, but only single-piece tokens are checked.
- **Pieces over 4 GiB.** The heap merge stores byte positions as `u32`.
- **Counting long pieces allocates.** `count_ordinary` builds the full token list for pieces over 128 bytes, then
  takes its length.
- **Slow cases.** Long random pieces from very small alphabets (see above).
- **One encoding.** Only `o200k_base`. Other encodings (`cl100k_base`, `o200k_harmony`) and model-name routing are
  not implemented.

## Provenance and licensing

magi-toks is licensed under the [Apache License 2.0](LICENSE). [NOTICE](NOTICE) carries the attributions below,
including the full MIT notice for the tiktoken vocabulary.

| Component | Source | Licence |
| --- | --- | --- |
| `assets/o200k_base.tiktoken` | OpenAI [tiktoken](https://github.com/openai/tiktoken) public vocabulary | MIT (tiktoken) |
| Scanner design (`src/scanner.rs`) | [toks](https://github.com/actualcomputer/toks) revision `2ebc390`, `src/core/k3_o200k_c.c` and `kernels.h`, Copyright 2026 Actual Computer Inc., rewritten in Rust | Apache-2.0 |
| Window stitching property | [`bpe` crate](https://github.com/github/rust-gems/tree/main/crates/bpe) README, "Properties of BPE". Only the property is used; no code from the crate is included. | MIT |
| Unicode classes | Generated from `regex` 1.12.4 / `regex-syntax` 0.8.11 (Unicode 16 data) | MIT OR Apache-2.0; Unicode License v3 |
| Everything else | magi-toks | Apache-2.0 |
