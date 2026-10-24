# magi-tokenizer

A fast Rust tokenizer for OpenAI's `o200k_base` encoding. It produces the same token IDs as
[tiktoken](https://github.com/openai/tiktoken) for `encode_ordinary` and `encode_with_special_tokens`.

Experimental, version 0.0.2. No automated correctness suite is included. See
[Known limitations](#known-limitations) and [GitHub issues](https://github.com/magimetal/magi-tokenizer/issues).

## Quick start

Add the crate from [crates.io](https://crates.io/crates/magi-tokenizer):

```toml
[dependencies]
magi-tokenizer = "=0.0.2"
```

```rust
let text = "Hello, world! <|endoftext|>";

// Treat special-token text as ordinary text (tiktoken `encode_ordinary`).
let ids = magi_tokenizer::encode_ordinary(text);
assert_eq!(magi_tokenizer::count_ordinary(text), ids.len());

// Recognise special tokens (tiktoken `encode_with_special_tokens`).
let with_special = magi_tokenizer::encode_with_special_tokens(text);
assert_eq!(*with_special.last().unwrap(), magi_tokenizer::END_OF_TEXT);

assert_eq!(magi_tokenizer::decode(&ids).as_deref(), Some(text));
```

No setup or vocabulary load: tables are embedded at compile time. The first call still pays for cold caches
and table pages loaded on demand (see [Startup](#startup)).

## Scope

| Item | State |
| --- | --- |
| Encoding | `o200k_base`: 199,998 ordinary tokens (IDs 0–199,997) plus special tokens |
| Special tokens | `<\|endoftext\|>` = 199,999, `<\|endofprompt\|>` = 200,018 |
| Models | Upstream tiktoken maps the GPT-4o, o-series and GPT-5 model prefixes to `o200k_base`. magi-code v0.97.6 also routes `gpt-6-*` and `gpt-6.1-*` (including `gpt-6-astra`) to `o200k_base`; its maintainer validated that routing in magi-code. The library exposes the encoding, not a model-name router; model routing stays with the application. |
| Input | `&str` (valid UTF-8). Decoding can return raw bytes for token sequences that are not valid UTF-8. |
| Platforms | Any Rust target. Tables are emitted in the target's byte order. CI builds on x86_64 and ARM64 Linux, ARM64 macOS, x86_64 and ARM64 Windows, 32-bit i686 Linux and big-endian s390x Linux. Speed and memory were measured only on ARM64 macOS. |

## Using it in your project

- Pin an exact version and commit `Cargo.lock`. Changes are listed in [CHANGELOG.md](CHANGELOG.md). There is no
  compatibility promise, and the API can change between releases. For git dependencies, pin a full commit hash
  whose CI passed. Do not enable `harness`:
  it exposes memory-benchmark hooks for `bench/`, not a stable API.
- `encode_ordinary` matches tiktoken's `encode_ordinary`. `encode_with_special_tokens` matches
  `encode(text, allowed_special="all")`. To reject special-token text like tiktoken's default `encode()`, or allow
  only some special tokens, check the strings in `SPECIAL_TOKENS` yourself before encoding.
- For counts without a token list, use `count_ordinary` or `count_with_special_tokens`. They match the encode functions' lengths.
- Cap untrusted input before calling the library. Encoding allocates 4 bytes per output token. The whole-piece
  heap fallback uses memory proportional to piece length: 26 to 27 bytes per piece byte in measured runs, not a
  proven bound. No tested input reaches it naturally, but none is known to rule it out. There is no memory-limit setting.
- Embedded tables add about 19 MiB. A release program counting one string was 19.9 MB, versus 0.5 MB empty and
  35.1 MB with bpe-openai (ARM64 macOS). Clean debug and release builds each took about 1.3 s on an Apple M5 Pro.
  Dependents use their own build profiles, not this repository's. If debug table generation is slow, add
  `[profile.dev.build-override] opt-level = 3`.

## API

All functions are free functions, with no tokenizer object, global state or locking. Any number of threads can call them concurrently.

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

### Panics

Functions do not panic or loop forever on any `&str` or `u32` token ID, except for these resource limits:

- Out of memory aborts the process and cannot be caught with `catch_unwind`. Encoding allocates by token count,
  and the whole-piece heap fallback allocates by piece length.
- On 32-bit targets, inputs or pieces of several hundred MiB can request buffers larger than `isize::MAX`, which
  panics. On 64-bit targets, allocation fails first.

Unknown IDs return `None`. The [issue #3 audit](https://github.com/magimetal/magi-tokenizer/issues/3) documents
invariants for each index, `unwrap` and `unsafe` table view. Compile-time assertions check table and index
invariants. No automated correctness suite is included.

## What is different

Every exact `o200k_base` tokenizer must reproduce the same regex splits and lowest-rank adjacent-pair merges.
These three implementations compute them differently:

| Stage | tiktoken-rs | bpe-openai | magi-tokenizer |
| --- | --- | --- | --- |
| Vocabulary setup | Decoded and hashed at run time (166 ms first call) | Built at run time (78 ms first call) | Built by `build.rs` into embedded hash tables; nothing runs at start (0.1 ms first call) |
| Pretokenizer | `fancy-regex` with lookahead | `regex-automata` with a lookahead workaround | Hand-written scanner (derived from `toks`) with Unicode 16 classes generated from the same `regex` crate version as the reference |
| Short pieces | Rescans all pair ranks after each merge | Backtracking search over Aho-Corasick automata of the vocabulary | Stack arrays, no allocation; packed rank+position keys with per-block minimums |
| Long pieces | Same rescan: cost grows with the square of the length (15 MiB/s on long runs) | Same backtracking search | Overlapping windows, stitched and cached (954 MiB/s on long runs) |
| Counting | Encodes, then takes the length | Separate count function | Separate count function; long pieces counted in bounded memory |

On source code, transcripts and the shared fixture, magi-tokenizer is about 6× faster than tiktoken-rs and 3×
faster than bpe-openai. Cached windows make long runs about 64× and 5.6× faster, respectively. Counting avoids
storing output tokens. These results exclude the whole-piece heap fallback.

It supports fewer encodings than either competitor.
All measurements come from one machine under background load (see [Caveats](#caveats)).

## How it works

1. Pretokenization splits text into words with optional leading space or punctuation, groups of 1 to 3 digits,
   punctuation runs, whitespace runs and English contractions.
2. Each piece starts as bytes. Merge the adjacent pair with the lowest vocabulary rank, choosing the leftmost
   pair on ties, until no pair forms a vocabulary token. In `o200k_base`, rank equals token ID.

### Build-time tables (`build.rs`, `src/table_hash.rs`, `src/tables.rs`)

`build.rs` has no dependencies. It parses `assets/o200k_base.tiktoken` and emits aligned binary tables in the target's byte order:

- Token bytes and offsets map IDs to bytes for decoding.
- The vocabulary hash table has 2^19 slots, about 38% full. Each slot uses two `u64`s for the first 8 bytes,
  4 tail bytes, length and ID. Tokens up to 12 bytes are confirmed in the slot. Longer tokens require a byte comparison.
- The pair table has 2^20 slots, about 43% full. Each `u64` maps `(left ID, right ID)` to the merged ID.
  It covers all 446,189 splits of vocabulary tokens into two vocabulary tokens. IDs fit in 18 bits.
- A dense 65,536-entry array holds byte-pair ranks for the most common lookup: the first merge.
- A byte-token table holds the 256 single-byte tokens (IDs 0 to 255).
- A 65,536-entry array, one per 2- and 3-byte character, holds the character's token when merging may start from
  it whole instead of from its bytes (see [Byte-pair merging](#byte-pair-merging-srcbpers)), else 0.

Token and pair counts are checked at build time, so a changed vocabulary fails the build rather than producing
wrong tables. The build script and library share hash functions from `src/table_hash.rs`.

### Pretokenizer (`src/scanner.rs`, `src/unicode_classes.rs`)

The o200k regex uses lookahead (`\s+(?!\S)`), which tiktoken handles with the slower `fancy-regex` backtracking
engine. The hand-written scanner tries all seven alternatives in order and reproduces their backtracking, including:

- case-insensitive contractions, where `'ſ` must not match `'s`;
- an optional leading character that is not a letter, digit, CR or LF;
- whitespace runs that return their last space to the next word;
- `\s*[\r\n]+` and `\s+(?!\S)`.

The scanner reads Unicode scalars and looks up one class byte per code point in the generated
`src/unicode_classes.rs` (about 110 KB). `tools/gen-unicode` uses `regex` 1.12.4 and `regex-syntax` 0.8.11,
matching tiktoken-rs 0.12.0's Unicode 16 data. These versions are pinned: newer Unicode data would change token boundaries.

Matching order follows [toks](https://github.com/actualcomputer/toks) (`src/core/k3_o200k_c.c`), rewritten in Rust.
See [Provenance](#provenance-and-licensing).

### Byte-pair merging (`src/bpe.rs`)

Most pieces are vocabulary tokens: one hash-table probe returns their ID without merging. Other pieces take one of two paths:

- Up to 128 bytes: fixed stack arrays, no allocation. Under 32 bytes, a vectorised minimum finds the lowest rank,
  then its first position breaks ties. Merging shifts the remaining tokens down, which measured faster at this size.
  For 32 to 128 bytes, parts stay in their first-byte slots, linked to neighbours. A 32-bit key packs rank above
  position, so its minimum selects the correct pair without a separate tie check. Each block of 32 keys keeps a
  minimum. A merge changes at most three keys, requiring at most three block rescans and four block-minimum checks.
  The longest `o200k_base` token is also 128 bytes.
- Over 128 bytes: overlapping windows, described below.

Merging starts from single bytes, except for some 2- and 3-byte characters, which start as one part. `build.rs`
marks a character eligible only if its bytes alone merge into its token, and every token that joins part or all of
it to a neighbour ranks above each of those inner merges. Its bytes then always merge first, so the result is
identical with fewer merges. At runtime the byte before must be a continuation byte and the byte after must start a
multibyte character (or the piece ends there), the only neighbours the build check covers. ASCII-only pieces skip
the check. Random CJK and Hangul pieces need about a third of the parts.

#### Long pieces: overlapping windows

A heap merge costs O(n log n) per piece and is slow on repetitive runs such as 300,000 spaces, `aaaa…` or `ababab…`.
Instead, magi-tokenizer encodes short windows and joins them using a property from the
[`bpe` crate README](https://github.com/github/rust-gems/tree/main/crates/bpe), under "Properties of BPE":

> If two exact encodings of overlapping text contain the same token at the same byte position, the first encoding
> up to that token, followed by the second encoding after it, is the exact encoding of the combined text.

The `bpe` crate uses this property in its search algorithm. No code from it is used here. The window algorithm,
cache, boundary alignment, streaming finalization, packed-key merge and all-splits pair table were designed for this project.

1. Encode the first 128-byte window with the short-piece merge.
2. Start the next window at an existing token boundary, overlapping about 1/8 of the window size. Encode it exactly.
3. Find the first matching token ID and end offset in both encodings. A single merge-style pass over sorted offsets finds this stitch point.
4. Keep the existing result through that token and append the window's remaining tokens.
5. Without a stitch point, double the window and retry, up to 4,096 bytes. If that fails, encode the whole piece with the exact heap merge.
6. After a successful stitch, reset to 128 bytes. Difficult stretches are usually local, and each window costs a few microseconds without special data structures.

A 16-slot cache reuses window encodings within each call. Hash hits require full byte comparison, so collisions
cannot change tokens. Repetitive input, including `"!@#$%^&*()" × N`, reuses merges at each phase of its period.
Window size controls alignment: on `long_runs`, 128 bytes beat 64, while 48 and 96 were about six times slower.
Starting at existing token boundaries repeats earlier merges and usually finds a stitch within the first few tokens.
Random small-alphabet text finds stitches just as quickly but never repeats windows, so merging remains its cost.

Windows reach back at most 640 bytes (maximum overlap plus one token). Tokens ending further back are final.
Every 4,096 pending tokens, the encoder appends final tokens to output and the counter only counts them.
Working memory stays bounded regardless of piece length: at most 64 KiB of cached bytes, plus a few hundred KiB
of merge state and pending tokens, usually a few KiB. These are data-structure estimates, excluding allocator overhead.
The exception is whole-piece fallback, which discards that piece's prior output and uses memory proportional to its length.

Peak RSS for one random `a`/`b` piece (`magi-tokenizer-bench memory`, ARM64 macOS, fallback forced through `harness`):

| Piece | `count` via windows | `count` via fallback | `encode` via windows | `encode` via fallback |
| ---: | ---: | ---: | ---: | ---: |
| 16 MiB | 29 MiB, 0.3 s | 421 MiB, 3.3 s | 51 MiB, 0.3 s | 436 MiB, 3.1 s |
| 64 MiB | 77 MiB, 1.1 s | 1,669 MiB, 16.7 s | 164 MiB, 1.1 s | 1,732 MiB, 17.1 s |
| 256 MiB | 269 MiB, 4.5 s | 6,661 MiB, 88.6 s | 619 MiB, 4.4 s | 6,917 MiB, 85.5 s |

RSS includes input at 1 byte per piece byte. Fallback used 26 to 27 bytes per piece byte and took 11 to 20 times
as long as windows, with a growing gap at larger sizes. These are measurements, not worst-case bounds.

The heap merge also handles windows over 128 bytes. It uses `u32` positions, switching to `usize` at 4 GiB.

## Benchmarks

`magi-tokenizer-bench bench` compares count and encode throughput against tiktoken-rs 0.12.0 and bpe-openai 0.3.2.
It checks equal total token counts before timing five jobs: magi count/encode, bpe-openai count/encode and tiktoken
encode. Each runs for at least 150 ms in five rounds, with rotated order. Figures are median rounds.
For tiktoken-rs, which has no separate count, both figures use encode.

Count throughput in MiB/s, commit `629e2e6` (library code as in `49b9a29`), Rust 1.98.1, `toks` source and both private assets:

| Workload | MiB | magi-tokenizer | tiktoken-rs | bpe-openai | vs tiktoken | vs bpe-openai |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixture_200k` | 0.895 | 256.3 | 39.9 | 90.0 | 6.42× | 2.85× |
| `source_1MiB` | 1.000 | 212.7 | 33.8 | 65.0 | 6.30× | 3.27× |
| `source_48B_chunks` | 0.250 | 183.1 | 28.5 | 52.9 | 6.43× | 3.46× |
| `source_256B_chunks` | 0.250 | 207.5 | 31.8 | 63.0 | 6.54× | 3.30× |
| `multilingual` | 0.468 | 137.2 | 44.2 | 100.0 | 3.10× | 1.37× |
| `long_runs` | 1.678 | 953.7 | 14.9 | 170.5 | 64.16× | 5.59× |
| `long_unrepeated` | 0.680 | 57.7 | 14.9 | 45.2 | 3.88× | 1.28× |
| `session_snapshots` | 30.748 | 256.9 | 40.9 | 86.1 | 6.28× | 2.98× |

Encode was 3.02 to 6.49× faster than tiktoken-rs and 1.25 to 3.63× faster than bpe-openai, except on `long_runs`
(65.15× and 6.04×). An immediate second run kept magi count within 4%. bpe-openai ratios moved more:
`fixture_200k` count went from 2.85× to 3.10×, within the roughly 10% variation noted under [Caveats](#caveats).
Source results differ when using this repository instead of `toks`.

### Random CJK

Random CJK pieces never repeat a window, so every window is merged fresh. Starting eligible characters whole
(see [Byte-pair merging](#byte-pair-merging-srcbpers)) roughly halved their cost. A separate probe, with token IDs
checked against bpe-openai, measured milliseconds per call before and after that change:

| Input | magi-tokenizer before | magi-tokenizer after | bpe-openai |
| --- | ---: | ---: | ---: |
| 240 KB random CJK from 8 characters | 4.2 | 2.1 | 2.4 |
| 240 KB random CJK from 400 characters | 3.8 | 3.2 | 4.3 |
| 240 KB random Hangul from 8 characters | 4.4 | 1.3 | 2.4 |
| 240 KB random CJK from `你世一人中大国我` (mostly not eligible) | 4.5 | 2.7 | 2.6 |
| 2,000 random 8-character CJK pieces, 129 to 1,024 bytes | 19.8 | 10.7 | 11.3 |

In the same probe, 200 KB of random `a`/`b` took 3.1 ms versus 4.1 ms, and random punctuation about half
bpe-openai's time, unchanged by that change. `bench` without `toks` or the private assets measured `multilingual`
count at 137 → 178 MiB/s and `long_unrepeated` at 58 → 69 MiB/s; source and `long_runs` stayed within run-to-run
variation. The table above predates the change.

### Startup

On a contended machine, the first count took 0.107 ms, versus 166 ms for tiktoken-rs and 78 ms for bpe-openai's
vocabulary loading. Process launch is excluded. Cold caches and page faults still make the first call slower than later calls.

Counting random `a`/`b` pieces of 16, 64 and 128 MiB peaked at 29, 77 and 141 MiB RSS (`/usr/bin/time -l`):
13 MiB above the input for table pages and buffers. Encoding 64 MiB peaked at 164 MiB, including 87 MiB for
22.9 million IDs. The [memory command](#how-it-works) reproduced the 16 and 64 MiB figures. The 128 MiB figure
came from an earlier probe not in this repository. Whole-text memory and other tokenizers' memory remain unmeasured.

### Reproducing

```sh
MAGI_TOKENIZER_SOURCE=/path/to/toks MAGI_TOKENIZER_ASSETS=/path/to/assets \
  cargo run --release -p magi-tokenizer-bench -- bench          # optional filter, e.g. `bench long`
cargo run --release -p magi-tokenizer-bench -- startup magi     # or tiktoken, bpe-openai; one fresh process each
/usr/bin/time -l target/release/magi-tokenizer-bench memory 64 count window   # Linux: /usr/bin/time -v
```

- `fixture_200k` and `session_snapshots` require `MAGI_TOKENIZER_ASSETS` (see [Development](#development)). Without
  it, they are skipped. Session transcripts are private and unpublished, so others cannot reproduce that row.
- Source workloads read the first 1 MiB (`source_1MiB`) or 256 KiB (chunked workloads) from `MAGI_TOKENIZER_SOURCE`.
- The random CJK probe is not in this repository.
- The bench enables `harness`. A quick check on source text, long runs and random `a`/`b` found count throughput
  unchanged within run-to-run variation when that feature was disabled.

### Caveats

All figures come from one ARM64 Apple M5 Pro under background load. Rotated jobs limit drift, but repeated runs
still vary by about 10%. The table records one run on one commit. Rerun `bench` after scanner or merge changes.
The native `toks` C library has not been benchmarked.

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
bench/                       magi-tokenizer-bench: throughput, startup and memory benchmarks
tools/gen-unicode/           standalone generator for src/unicode_classes.rs (pinned regex versions)
```

## Development

The library builds with Rust 1.88.0 and 1.90.0, both checked in CI. Older versions are untested. The benchmark
harness needs newer standard-library APIs and uses stable in CI. Development used Rust 1.98.1.

```sh
# Library
cargo build --release

# Throughput, all three tokenizers; an optional filter matches workload names
cargo run --release -p magi-tokenizer-bench -- bench
cargo run --release -p magi-tokenizer-bench -- bench long

# First-call time in a fresh process
cargo run --release -p magi-tokenizer-bench -- startup magi        # or tiktoken, bpe-openai

# Regenerate the Unicode class table (only when the pinned regex versions change)
cd tools/gen-unicode && cargo run --release
```

Optional harness inputs:

- `MAGI_TOKENIZER_SOURCE`: source trees in platform path-list syntax. Defaults to this repository. Dot-directories
  and `target` are skipped, with at most about 4 MiB read per tree.
- `MAGI_TOKENIZER_ASSETS`: directory containing `o200k_base-200k.txt` and `session-compacted-500.json`. Unset skips
  those suites. A missing or malformed file when set fails the command.

The release profile uses fat LTO and one codegen unit. Build scripts use `opt-level = 3` for fast debug table generation.

## Known limitations

- Random CJK pieces made mostly of characters not eligible to start whole (for example `你`, `世`) still merge
  from bytes: about level with bpe-openai, not faster.
- If no stitch point exists within 4,096 bytes, whole-piece fallback can use much more time and memory
  (see [measured fallback costs](#how-it-works)). No input is known to rule it out. Measured costs are not proven bounds.
- Only `o200k_base` is implemented, not `cl100k_base`, `o200k_harmony` or model-name routing.
- The `usize` index path has not been exercised on a real piece of 4 GiB or more.
- Speed and memory were measured only on ARM64 macOS. CI does not run correctness comparisons.

## Provenance and licensing

The code uses the [Apache License 2.0](LICENSE). The crate's license expression is
`Apache-2.0 AND MIT AND Unicode-3.0` because it bundles the MIT-licensed tiktoken vocabulary and Unicode-derived
tables. [NOTICE](NOTICE) contains the attributions, full vocabulary MIT notice and full Unicode License v3 text.

| Component | Source | Licence |
| --- | --- | --- |
| `assets/o200k_base.tiktoken` | OpenAI [tiktoken](https://github.com/openai/tiktoken) public vocabulary | MIT (tiktoken) |
| Scanner design (`src/scanner.rs`) | [toks](https://github.com/actualcomputer/toks) revision `2ebc390`, `src/core/k3_o200k_c.c` and `kernels.h`, Copyright 2026 Actual Computer Inc., rewritten in Rust | Apache-2.0 |
| Window stitching property | [`bpe` crate](https://github.com/github/rust-gems/tree/main/crates/bpe) README, "Properties of BPE". Only the property is used; no code from the crate is included. | MIT |
| Unicode classes | Generated from `regex` 1.12.4 / `regex-syntax` 0.8.11 (Unicode 16 data) | MIT OR Apache-2.0; Unicode License v3 |
| Everything else | magi-tokenizer | Apache-2.0 |
