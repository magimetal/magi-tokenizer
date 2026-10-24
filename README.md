# magi-tokenizer

A fast Rust tokenizer for OpenAI's `o200k_base` and (optional feature) `cl100k_base` encodings. It produces the
same token IDs as [tiktoken](https://github.com/openai/tiktoken) for `encode_ordinary`, `encode_with_special_tokens`
and `encode` with an allow-list of special tokens.

Experimental, version 0.0.2. From 0.1.0, releases follow the [compatibility policy](#compatibility). No automated
correctness suite is included. See [Known limitations](#known-limitations) and
[GitHub issues](https://github.com/magimetal/magi-tokenizer/issues).

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

// Like Python tiktoken's default `encode()`: special-token text not in the allow-list is an error.
let error = magi_tokenizer::encode(text, &[]).unwrap_err();
assert_eq!((error.token, error.position), ("<|endoftext|>", 14));
assert_eq!(magi_tokenizer::encode(text, &["<|endoftext|>"]).unwrap(), with_special);

assert_eq!(magi_tokenizer::decode(&ids).as_deref(), Some(text));
```

No setup or vocabulary load: tables are embedded at compile time. The first call still pays for cold caches
and table pages loaded on demand (see [Startup](#startup)).

`cl100k_base` (GPT-4, GPT-3.5 Turbo, `text-embedding-*`) is opt-in:

```toml
magi-tokenizer = { version = "=0.0.2", features = ["cl100k_base"] }
```

```rust
use magi_tokenizer::cl100k_base;
let ids = cl100k_base::encode_ordinary("Hello, world!");
assert_eq!(cl100k_base::count_ordinary("Hello, world!"), ids.len());
```

## Scope

| Item | State |
| --- | --- |
| Encodings | `o200k_base` (always built; module `o200k_base`, also re-exported at the crate root): 199,998 ordinary tokens (IDs 0–199,997). `cl100k_base` (feature `cl100k_base`; module `cl100k_base`): 100,256 ordinary tokens (IDs 0–100,255). Each module has the same functions and constants. |
| Other encodings | `o200k_harmony`: ordinary text only. tiktoken-rs 0.12.0 builds it from the same vocabulary and split pattern as `o200k_base`, so `o200k_base::encode_ordinary` and `count_ordinary` give the same IDs and counts. Its special tokens differ (for example, 200018 is `<\|reserved_200018\|>`, not `<\|endofprompt\|>`), so do not use the special-token APIs for it ([#18](https://github.com/magimetal/magi-tokenizer/issues/18)). The legacy `r50k_base`, `p50k_base` and `p50k_edit` encodings are not implemented ([#19](https://github.com/magimetal/magi-tokenizer/issues/19)). |
| Special tokens | `o200k_base`: `<\|endoftext\|>` = 199,999, `<\|endofprompt\|>` = 200,018. `cl100k_base`: `<\|endoftext\|>` = 100,257, `<\|fim_prefix\|>` = 100,258, `<\|fim_middle\|>` = 100,259, `<\|fim_suffix\|>` = 100,260, `<\|endofprompt\|>` = 100,276 |
| Models | Upstream tiktoken (`tiktoken/model.py`) maps GPT-4o, GPT-4.1, GPT-4.5, GPT-5 and o-series names to `o200k_base`; GPT-4, GPT-3.5 Turbo, `davinci-002`, `babbage-002` and `text-embedding-ada-002`/`-3` to `cl100k_base`; and `gpt-oss-*` to `o200k_harmony`. magi-code v0.97.6 also routes `gpt-6-*` and `gpt-6.1-*` (including `gpt-6-astra`) to `o200k_base`; its maintainer validated that routing in magi-code. The library exposes encodings, not a model-name router: model routing stays with the application by decision (see [Model routing](#model-routing)). |
| Input | `&str` (valid UTF-8) by decision; convert other bytes first, for example with `String::from_utf8_lossy` (see [Byte input](#byte-input)). Decoding can return raw bytes for token sequences that are not valid UTF-8. |
| Platforms | Any Rust target. Tables are emitted in the target's byte order. CI builds on x86_64 and ARM64 Linux, ARM64 macOS, x86_64 and ARM64 Windows, 32-bit i686 Linux and big-endian s390x Linux. Speed and memory were measured only on ARM64 macOS. |

## Using it in your project

- Pin 0.0.x releases exactly (`=0.0.2`) and commit `Cargo.lock`: they have no compatibility promise. From 0.1.0,
  the [compatibility policy](#compatibility) applies. Changes are listed in [CHANGELOG.md](CHANGELOG.md). For git
  dependencies, pin a full commit hash whose CI passed. Do not enable `harness`: it exposes memory-benchmark hooks
  for `bench/` and is outside the policy.
- Special-token modes ([#10](https://github.com/magimetal/magi-tokenizer/issues/10)):
  - `encode_ordinary` matches tiktoken's `encode_ordinary`: special-token text is ordinary text.
  - `encode_with_special_tokens` matches `encode(text, allowed_special="all")`.
  - `encode_allowing_special(text, allowed)` matches tiktoken-rs 0.12.0 `CoreBPE::encode(text, allowed)`: only listed
    special tokens are recognised; other special-token text is ordinary text.
  - `encode(text, allowed)` matches Python tiktoken's default `encode(text, allowed_special=allowed)`: special-token
    text not in `allowed` returns `Err(DisallowedSpecialToken)` with the leftmost such token and its byte offset,
    otherwise the same IDs as `encode_allowing_special`. tiktoken-rs has no such error, so the reference is Python
    tiktoken's check (`disallowed_special="all"`). Pass `&[]` to reject all special-token text from untrusted input.
  - Names in `allowed` that are not special tokens of the encoding are ignored.
- For counts without a token list, use `count_ordinary`, `count_with_special_tokens`, `count_allowing_special` or
  `count`. They match the encode functions' lengths, and `count` returns the same error as `encode`.
- Cap untrusted input before calling the library. Encoding allocates 4 bytes per output token. The whole-piece
  fallback allocates exactly 20 bytes of working memory per piece byte, once, and runs in O(n log n) time
  ([#12](https://github.com/magimetal/magi-tokenizer/issues/12)); measured runs used 21 to 22 bytes of peak memory
  per piece byte, input included. No tested input reaches it naturally, but none is known to rule it out. There is
  no memory-limit setting: an input cap of N bytes bounds fallback working memory at 20 × N bytes.
- Embedded `o200k_base` tables add about 19 MiB. A release program counting one string was 19.9 MB, versus 0.5 MB empty and
  35.1 MB with bpe-openai (ARM64 macOS). Clean debug and release builds each took about 1.3 s on an Apple M5 Pro.
  Dependents use their own build profiles, not this repository's. If debug table generation is slow, add
  `[profile.dev.build-override] opt-level = 3`.
- `cl100k_base` costs only when used. In a fat-LTO release program counting one string with each encoding, the
  binary grew from 20.1 MB (`o200k_base` only) to 30.2 MB (+10.1 MB); enabling the feature without calling it left the
  size unchanged, since unused tables are dropped at link time. Clean release builds of that program took 3.6 s
  without the feature and 4.5 s with it (median of three, Apple M5 Pro).

### Compatibility

Decision ([#13](https://github.com/magimetal/magi-tokenizer/issues/13)): the policy starts with 0.1.0, the next
release. 0.0.1 and 0.0.2 carry no promise. From 0.1.0, versions follow Cargo's SemVer rules: a release that breaks
a promise below gets a new minor version (0.1 → 0.2), which Cargo treats as incompatible; a patch release
(0.1.0 → 0.1.1) keeps every promise. A plain `magi-tokenizer = "0.1"` requirement then takes only compatible releases.

Kept within a minor version:

- Token IDs. For any `&str` input, each encoding returns the same IDs as tiktoken-rs 0.12.0 with its bundled
  vocabularies and Unicode 16 data; the `encode` and `count` error follows Python tiktoken's disallowed-special check.
  Counts equal encode lengths, `pieces` follows the reference split pattern, and `decode` and `token_bytes` return
  vocabulary bytes. A newer reference, Unicode version or vocabulary would move token boundaries, so it needs a new
  minor version. A difference from the reference is a bug: its fix ships in a patch release and is listed under
  "Fixed" in the changelog, because it changes IDs for the affected inputs.
- API. Public items of `magi_tokenizer`, `magi_tokenizer::o200k_base` and `magi_tokenizer::cl100k_base`, the
  crate-root re-export of `o200k_base`, and the `cl100k_base` feature are not removed, renamed or changed
  incompatibly. Additions (functions, constants, encodings, features) can ship in patch releases.
- Minimum Rust version 1.88. Raising it needs a new minor version.

Not promised:

- The `harness` feature and `magi_tokenizer::harness` (hidden from docs) can change or disappear in any release.
  Cargo cannot hide a feature, so the exclusion is by documentation: do not enable it.
- Speed, memory use, binary size and build time can change in any release. The changelog records notable changes.

Release notes: [CHANGELOG.md](CHANGELOG.md) is the record and lists breaking changes under "Breaking". Each GitHub
release repeats its version's changelog section.

### Model routing

Decision ([#9](https://github.com/magimetal/magi-tokenizer/issues/9)): this library does not map model names to
encodings and does not plan to. Keep the mapping in your application, and pick the encoding module from it:

- New models ship more often than this crate. A built-in mapping would lag, and users stay on one minor version
  until they upgrade.
- There is no single source of truth. Upstream tiktoken does not list `gpt-6-*`, which magi-code routes to
  `o200k_base`. The application that knows which models it calls is the right owner.
- Results would be incomplete: `cl100k_base` can be compiled out, `gpt-oss-*` uses `o200k_harmony`, whose
  special tokens are not implemented ([#18](https://github.com/magimetal/magi-tokenizer/issues/18)), and the
  legacy 50k encodings are missing ([#19](https://github.com/magimetal/magi-tokenizer/issues/19)).
- A mapping would be API surface whose behavior changes with every update. Changing a name's encoding changes its
  token counts, which the [compatibility policy](#compatibility) allows only in a new minor version.

A wrong encoding gives wrong counts without any error. Treat unknown model names as an error, or choose a default
deliberately; do not fall back silently. The [Scope](#scope) table lists the upstream mappings to start from.

### Byte input

Decision ([#11](https://github.com/magimetal/magi-tokenizer/issues/11)): encoding takes `&str` only, and there is no
byte-input API. Convert bytes that may not be valid UTF-8 before encoding, usually with `String::from_utf8_lossy`:

- No reference defines tokens for invalid UTF-8 in a public API. tiktoken-rs 0.12.0 encodes only `&str`, and Python
  tiktoken's `encode` takes `str`. Without a reference, byte-input IDs could not be checked exactly.
- Python tiktoken's private `_encode_bytes` is a heuristic, not a definition; upstream calls it a "best guess". It
  splits the valid UTF-8 prefix with the regex, then byte-pair merges that prefix's last piece plus everything from
  the first invalid byte to the end as one piece. One stray byte near the start of a large input would make the rest
  one piece, which could reach the slow whole-piece fallback here.
- Both split patterns are defined over Unicode text. Any other rule for invalid bytes would be this crate's own.
- Byte input needs more than a new signature: the scanner decodes UTF-8 without validation.

`String::from_utf8_lossy` replaces invalid sequences with U+FFFD the same way as Python's
`bytes.decode("utf-8", errors="replace")` (checked on truncated, overlong, surrogate and out-of-range sequences), so
IDs match Python tiktoken for text decoded that way. They do not describe the original bytes: each U+FFFD is the
three bytes `EF BF BD`, and `decode_bytes` returns those, not the replaced bytes.

## API

All functions are free functions, with no tokenizer object, global state or locking. Any number of threads can call
them concurrently. Each encoding module (`magi_tokenizer::o200k_base`, `magi_tokenizer::cl100k_base`) has these
functions; the crate root re-exports `o200k_base`.

| Function | Purpose |
| --- | --- |
| `pieces(text) -> Pieces` | Iterator over the pretokenization pieces (the substrings the encoding's tiktoken regex produces). |
| `encode_ordinary(text) -> Vec<u32>` | Token IDs. Special-token text is encoded as ordinary text. |
| `encode_ordinary_into(text, &mut Vec<u32>)` | Same, appended to an existing vector so its allocation can be reused. |
| `count_ordinary(text) -> usize` | Number of tokens `encode_ordinary` returns. |
| `encode_with_special_tokens(text) -> Vec<u32>` | Token IDs with the encoding's special tokens recognised. |
| `count_with_special_tokens(text) -> usize` | Number of tokens `encode_with_special_tokens` returns. |
| `encode_allowing_special(text, allowed) -> Vec<u32>` | Token IDs with only the special tokens in `allowed: &[&str]` recognised. |
| `count_allowing_special(text, allowed) -> usize` | Number of tokens `encode_allowing_special` returns. |
| `encode(text, allowed) -> Result<Vec<u32>, DisallowedSpecialToken>` | Like `encode_allowing_special`, but an error if the text contains special-token text not in `allowed`. |
| `count(text, allowed) -> Result<usize, DisallowedSpecialToken>` | Number of tokens `encode` returns, or its error. |
| `token_bytes(id) -> Option<&'static [u8]>` | Bytes of one ordinary or special token; `None` for an unknown ID. |
| `decode_bytes(ids) -> Option<Vec<u8>>` | Concatenated bytes; `None` if an ID is unknown. |
| `decode(ids) -> Option<String>` | Decoded text; `None` if an ID is unknown or the bytes are not valid UTF-8. |

Constants: `END_OF_TEXT`, `END_OF_PROMPT`, `SPECIAL_TOKENS`, `ORDINARY_TOKEN_COUNT`. Error type (crate root):
`DisallowedSpecialToken { token, position }`.

### Panics

Functions do not panic or loop forever on any `&str` or `u32` token ID, except for these resource limits:

- Out of memory aborts the process and cannot be caught with `catch_unwind`. Encoding allocates by token count,
  and the whole-piece fallback allocates 20 bytes per piece byte.
- On 32-bit targets, inputs or pieces of several hundred MiB can request buffers larger than `isize::MAX`, which
  panics. On 64-bit targets, allocation fails first.

Unknown IDs return `None`. The [issue #3 audit](https://github.com/magimetal/magi-tokenizer/issues/3) documents
invariants for each index, `unwrap` and `unsafe` table view. Compile-time assertions check table and index
invariants. No automated correctness suite is included.

## What is different

Every exact tokenizer must reproduce the same regex splits and lowest-rank adjacent-pair merges.
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
storing output tokens. These results exclude the whole-piece fallback.

It supports fewer encodings than either competitor: no Harmony special tokens and no legacy 50k encodings.
All measurements come from one machine under background load (see [Caveats](#caveats)).

## How it works

1. Pretokenization splits text into words with optional leading space or punctuation, groups of 1 to 3 digits,
   punctuation runs, whitespace runs and English contractions.
2. Each piece starts as bytes. Merge the adjacent pair with the lowest vocabulary rank, choosing the leftmost
   pair on ties, until no pair forms a vocabulary token. In `o200k_base` and `cl100k_base`, rank equals token ID.

### Build-time tables (`build.rs`, `src/table_hash.rs`, `src/tables.rs`)

`build.rs` has no dependencies. For each enabled encoding it parses `assets/<name>.tiktoken` and emits aligned binary
tables in the target's byte order. `cl100k_base` tables are generated only with its feature. Sizes below are for
`o200k_base`; each hash table gets the smallest power-of-two slot count at least twice its entry count, so
`cl100k_base` uses 2^18 vocabulary and 2^19 pair slots.

- Token bytes and offsets map IDs to bytes for decoding.
- The vocabulary hash table has 2^19 slots, about 38% full. Each slot uses two `u64`s for the first 8 bytes,
  4 tail bytes, length and ID. Tokens up to 12 bytes are confirmed in the slot. Longer tokens require a byte comparison.
- The pair table has 2^20 slots, about 43% full. Each `u64` maps `(left ID, right ID)` to the merged ID.
  It covers all 446,189 splits of vocabulary tokens into two vocabulary tokens (233,378 for `cl100k_base`). IDs fit in 18 bits.
- A dense 65,536-entry array holds byte-pair ranks for the most common lookup: the first merge.
- A byte-token table holds the 256 single-byte tokens (IDs 0 to 255).
- A 65,536-entry array, one per 2- and 3-byte character, holds the character's token when merging may start from
  it whole instead of from its bytes (see [Byte-pair merging](#byte-pair-merging-srcbpers)), else 0.

Token and pair counts are checked at build time, so a changed vocabulary fails the build rather than producing
wrong tables. The build script and library share hash functions from `src/table_hash.rs`. Lookups and merging are
generic over the encoding, so each encoding gets its own compiled copy with its table addresses as constants.

### Pretokenizer (`src/scanner.rs`, `src/unicode_classes.rs`)

The o200k regex uses lookahead (`\s+(?!\S)`), which tiktoken handles with the slower `fancy-regex` backtracking
engine. The hand-written scanner tries all seven alternatives in order and reproduces their backtracking, including:

- case-insensitive contractions, where `'ſ` must not match `'s`;
- an optional leading character that is not a letter, digit, CR or LF;
- whitespace runs that return their last space to the next word;
- `\s*[\r\n]+` and `\s+(?!\S)`.

A second scanner implements the `cl100k_base` regex: standalone contractions, letters with an optional possessive
leading character, 1 to 3 digits, punctuation runs that take trailing CR/LF, and `\s++$`, `\s*[\r\n]`,
`\s+(?!\S)`, `\s`. Both share the class table and digit, contraction and run helpers.

The scanner reads Unicode scalars and looks up one class byte per code point in the generated
`src/unicode_classes.rs` (about 110 KB). `tools/gen-unicode` uses `regex` 1.12.4 and `regex-syntax` 0.8.11,
matching tiktoken-rs 0.12.0's Unicode 16 data. These versions are pinned: newer Unicode data would change token boundaries.

Matching order follows [toks](https://github.com/actualcomputer/toks) (`src/core/k3_o200k_c.c`), rewritten in Rust.
See [Provenance](#provenance-and-licensing).

### Byte-pair merging (`src/bpe.rs`)

Most pieces are vocabulary tokens: one- and two-byte pieces are read from direct byte and byte-pair tables, and longer
ones take one hash-table probe, returning their ID without merging. Other pieces take one of two paths:

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

Merging a whole piece costs O(n log n) and is slow on repetitive runs such as 300,000 spaces, `aaaa…` or `ababab…`.
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
5. Without a stitch point, double the window and retry, up to 4,096 bytes. If that fails, merge the whole piece at once (whole-piece fallback).
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
The exception is whole-piece fallback, which discards that piece's prior output and allocates by its length.

The whole-piece merge allocates five `u32` arrays as long as the piece, once: tokens, both links, ranks and a
tournament tree that picks the lowest rank, leftmost on ties. That is exactly 20 bytes of working memory per piece
byte, independent of content. Each merge updates at most three tree paths, so time is O(n log n). Before
[#12](https://github.com/magimetal/magi-tokenizer/issues/12), a binary heap kept stale entries and peak memory
depended on content.

Peak RSS for one random `a`/`b` piece (`magi-tokenizer-bench memory`, ARM64 macOS, fallback forced through `harness`):

| Piece | `count` via windows | `count` via fallback | `encode` via windows | `encode` via fallback |
| ---: | ---: | ---: | ---: | ---: |
| 16 MiB | 29 MiB, 0.3 s | 341 MiB, 2.6 s | 51 MiB, 0.3 s | 357 MiB, 2.6 s |
| 64 MiB | 77 MiB, 1.1 s | 1,349 MiB, 11.8 s | 165 MiB, 1.1 s | 1,413 MiB, 12.1 s |
| 256 MiB | 269 MiB, 4.2 s | 5,381 MiB, 55.4 s | 619 MiB, 4.3 s | 5,637 MiB, 56.6 s |

RSS includes input at 1 byte per piece byte; `encode` adds its output at 4 bytes per token. Fallback peaks were 21
to 22 bytes per piece byte, against 26 to 27 with the heap, and took 9 to 13 times as long as windows.

The whole-piece merge also handles windows over 128 bytes. It uses `u32` positions, switching to `usize` at 4 GiB,
where the two link arrays and the tree use 8-byte entries: 32 bytes per piece byte.

## Benchmarks

`magi-tokenizer-bench bench` compares count and encode throughput against tiktoken-rs 0.12.0 and bpe-openai 0.3.2
for both encodings. Before timing, it checks that magi-tokenizer returns tiktoken's exact token IDs, counts and
decoded text for every workload, plus special tokens spliced between texts (all allowed, and allow-lists checked
against tiktoken-rs `CoreBPE::encode`, with `encode`'s error checked against Python tiktoken's leftmost-disallowed
rule), a Unicode sweep (planes 0–2 and 14 in
six contexts), short random strings around scanner edge cases and the forced whole-piece fallback. It then times
five jobs: magi count/encode, bpe-openai count/encode and tiktoken encode. Each runs for at least 150 ms in five
rounds, with rotated order. Figures are median rounds. For tiktoken-rs, which has no separate count, both figures use encode.

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

`cl100k_base` count throughput in MiB/s, Rust 1.98.1, this repository's source (`git archive` of `5ac7c1e`), no
private assets. Over six alternating runs of the pre-change and changed benches, median `o200k_base` throughput
moved −1.6% to +1.1% per workload (paired runs −3.0% to +1.9%). This is code placement, not slower code:
`count_ordinary` and `encode_ordinary` compile to the same ARM64 instructions as before; only branch, call and
address targets differ. Elsewhere, 4 instructions differ in the byte comparison that confirms vocabulary tokens
longer than 12 bytes. Rebuilding both versions with 64-byte function alignment moved the gap to −1.7% to +2.7%, in
both directions. The bench now also calls each encoding through a function pointer, which moves its timed code too.

| Workload | MiB | magi-tokenizer | tiktoken-rs | bpe-openai | vs tiktoken | vs bpe-openai |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `source_1MiB` | 0.218 | 250.8 | 12.7 | 51.3 | 19.75× | 4.88× |
| `source_48B_chunks` | 0.218 | 223.3 | 11.6 | 48.5 | 19.26× | 4.61× |
| `source_256B_chunks` | 0.218 | 243.3 | 12.4 | 50.8 | 19.59× | 4.79× |
| `multilingual` | 0.468 | 176.8 | 28.3 | 90.4 | 6.25× | 1.96× |
| `scanner_edges` | 0.262 | 81.3 | 11.6 | 41.6 | 7.03× | 1.96× |
| `long_runs` | 1.678 | 1005.9 | 14.5 | 120.1 | 69.57× | 8.38× |
| `long_unrepeated` | 0.680 | 68.9 | 14.3 | 44.8 | 4.82× | 1.54× |

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
assets/cl100k_base.tiktoken  cl100k_base vocabulary, 1.7 MB (tables built only with feature `cl100k_base`)
build.rs                     parses each enabled vocabulary, emits binary tables + bindings into OUT_DIR
src/lib.rs                   public API: one module per encoding with pieces, encode/count, decode
src/scanner.rs               hand-written o200k and cl100k pretokenizers
src/unicode_classes.rs       generated per-code-point class table (do not edit by hand)
src/bpe.rs                   short-piece, window and whole-piece byte-pair merging
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

# Exact-ID checks, then throughput, all three tokenizers; an optional filter matches encoding/workload names
cargo run --release -p magi-tokenizer-bench -- bench
cargo run --release -p magi-tokenizer-bench -- bench cl100k/long

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
- If no stitch point exists within 4,096 bytes, whole-piece fallback takes 20 bytes of working memory per piece
  byte and about 10 times as long as windows (see [fallback costs](#how-it-works)). No input is known to rule it out.
- `o200k_harmony` special tokens ([#18](https://github.com/magimetal/magi-tokenizer/issues/18)) and the legacy
  `r50k_base`, `p50k_base` and `p50k_edit` encodings ([#19](https://github.com/magimetal/magi-tokenizer/issues/19))
  are not implemented. Model-name routing is left to applications by design ([#9](https://github.com/magimetal/magi-tokenizer/issues/9)).
- Encoding takes `&str` only; byte input is left out by decision ([#11](https://github.com/magimetal/magi-tokenizer/issues/11)).
- The `usize` index path has not been exercised on a real piece of 4 GiB or more.
- Speed and memory were measured only on ARM64 macOS. CI does not run correctness comparisons; run `bench` locally.

## Provenance and licensing

The code uses the [Apache License 2.0](LICENSE). The crate's license expression is
`Apache-2.0 AND MIT AND Unicode-3.0` because it bundles the MIT-licensed tiktoken vocabulary and Unicode-derived
tables. [NOTICE](NOTICE) contains the attributions, full vocabulary MIT notice and full Unicode License v3 text.

| Component | Source | Licence |
| --- | --- | --- |
| `assets/o200k_base.tiktoken`, `assets/cl100k_base.tiktoken` | OpenAI [tiktoken](https://github.com/openai/tiktoken) public vocabularies | MIT (tiktoken) |
| Scanner design (`src/scanner.rs`) | [toks](https://github.com/actualcomputer/toks) revision `2ebc390`, `src/core/k3_o200k_c.c` and `kernels.h`, Copyright 2026 Actual Computer Inc., rewritten in Rust | Apache-2.0 |
| Window stitching property | [`bpe` crate](https://github.com/github/rust-gems/tree/main/crates/bpe) README, "Properties of BPE". Only the property is used; no code from the crate is included. | MIT |
| Unicode classes | Generated from `regex` 1.12.4 / `regex-syntax` 0.8.11 (Unicode 16 data) | MIT OR Apache-2.0; Unicode License v3 |
| Everything else | magi-tokenizer | Apache-2.0 |
