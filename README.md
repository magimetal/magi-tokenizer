# magi-tokenizer

An exact, fast Rust tokenizer for OpenAI's `o200k_base` encoding.

For every input, magi-tokenizer produces the same token IDs as
[tiktoken](https://github.com/openai/tiktoken) (`encode_ordinary` and `encode_with_special_tokens`). Speed is the
goal, but exactness is required. A faster result with one wrong token ID is a failure.

> **Status: experimental, version 0.1.0.** The library passes the full differential verification suite described
> below. It is not published. Open items are listed under [Known limitations](#known-limitations) and as
> [GitHub issues](https://github.com/magimetal/magi-tokenizer/issues). The magi-code pilot plan is in
> [docs/magi-code-pilot.md](docs/magi-code-pilot.md).

## Contents

- [Scope](#scope)
- [What is different](#what-is-different)
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
| Models | Upstream tiktoken maps the GPT-4o, o-series and GPT-5 model prefixes to `o200k_base`. magi-code v0.97.6 also routes `gpt-6-*` and `gpt-6.1-*` (including `gpt-6-astra`) to `o200k_base`; its maintainer validated that routing in magi-code. The library exposes the encoding, not a model-name router; model routing stays with the application. |
| Input | `&str` (valid UTF-8). Decoding can return raw bytes for token sequences that are not valid UTF-8. |
| Platforms | Any Rust target. Tables are emitted in the target's byte order. CI runs `verify` on x86_64 and ARM64 Linux, ARM64 macOS, x86_64 and ARM64 Windows, 32-bit i686 Linux and big-endian s390x Linux (QEMU). Speed and memory were measured only on ARM64 macOS. |

## What is different

Every exact `o200k_base` tokenizer must compute the same result: split text into pieces with the same regex, then
repeatedly merge the lowest-rank adjacent pair in each piece. The projects differ in how they compute it. This
table compares magi-tokenizer with the two Rust tokenizers it is benchmarked against.

| Stage | tiktoken-rs | bpe-openai | magi-tokenizer |
| --- | --- | --- | --- |
| Vocabulary setup | Decoded and hashed at run time (166 ms first call) | Built at run time (78 ms first call) | Built by `build.rs` into embedded hash tables; nothing runs at start (0.1 ms first call) |
| Pretokenizer | `fancy-regex` with lookahead | `regex-automata` with a lookahead workaround | Hand-written scanner (derived from `toks`) with Unicode 16 classes generated from the same `regex` crate version as the reference |
| Short pieces | Rescans all pair ranks after each merge | Backtracking search over Aho-Corasick automata of the vocabulary | Stack arrays, no allocation; packed rank+position keys with per-block minimums |
| Long pieces | Same rescan: cost grows with the square of the length (14 MiB/s on long runs) | Same backtracking search | Overlapping windows, stitched and cached (954 MiB/s on long runs) |
| Counting | Encodes, then takes the length | Separate count function | Separate count function; long pieces counted in bounded memory |

### What is new here

The ideas below were designed for this project. They rest on one published fact, and that source is credited:

- **The stitch property** (not ours). The [`bpe` crate README](https://github.com/github/rust-gems/tree/main/crates/bpe)
  notes that if two exact encodings of overlapping text contain the same token at the same byte position, they can
  be joined at that token and the result is exact. The `bpe` crate uses this inside its own search algorithm. No
  code from it is used here.
- **Independent overlapping windows.** magi-tokenizer applies the property differently: it cuts a long piece into
  short windows, encodes each one with the ordinary short-piece merge, and joins neighbouring windows at the first
  token both agree on. Each window costs a few microseconds and needs no special data structure. If a stretch has
  no join point, the window doubles; past 4,096 bytes the whole piece goes to an exact heap merge, so the result is
  always exact.
- **Window cache with full byte comparison.** Repetitive text (`aaaa…`, `ababab…`, repeated punctuation, 300,000
  spaces) produces the same window bytes again and again. A small cache, cleared on every call, reuses earlier
  window encodings. A hit is accepted only after the bytes are compared in full, so a hash collision cannot produce
  a wrong token. This is why long runs are about 6× faster than bpe-openai and 67× faster than tiktoken-rs.
- **Boundary-aligned windows.** Each window starts where a token of the result so far starts, so its first merges
  repeat ones already done and the join point is usually found within the first few tokens.
- **Streaming finalization.** A window can only reach a fixed distance back (640 bytes), so every token that ends
  further back can never change. Those tokens are handed out in batches, which lets `count_ordinary` count a piece
  of any length in constant extra memory (13 MiB above the text, measured from 16 to 128 MiB pieces).
- **Packed-key short merge.** For pieces of 32–128 bytes, each part stays in place, linked to its neighbours, and
  carries one 32-bit key: merge rank in the high bits, position in the low bits. The smallest key is then the
  correct next merge, including the leftmost-first tie rule, without a separate tie check. A minimum is kept per
  block of 32 keys, and a merge changes at most three keys, so each step rescans at most three blocks plus the
  four block minimums. Shorter pieces use a simple shift-down merge, which measured faster at that size.
- **Compile-time pair table over every split.** `build.rs` records the rank of every way each token can be split
  into two tokens (446,189 pairs), so one probe answers "do these two tokens merge, and into what?" during merging.

### Why it is better

- **Same answer, every time.** Exactness is checked against tiktoken-rs on about 100 MB of text, every Unicode
  scalar, 200,000 random strings and adversarial long pieces, with zero mismatches. No speed-up is kept unless
  this passes.
- **Faster on ordinary text:** about 6× tiktoken-rs and 3× bpe-openai on source code, transcripts and the shared
  fixture.
- **No cliff on the measured long pieces.** Long single pieces that make tiktoken-rs slow down sharply, such as long
  runs of one character or a short pattern, stay fast, and counting them does not need memory for the tokens. Input
  that reaches the whole-piece heap fallback (see [Known limitations](#known-limitations)) is not covered by this.
- **No vocabulary load.** Short-lived processes, such as command-line tools and hooks, do not pay 80–170 ms to load
  a vocabulary. The first call still pays for cold caches and page faults.

### Where it is not better

- Long pieces of random CJK text from a few characters: 4.4 ms against bpe-openai's 2.5 ms per 240 KB.
- Only `o200k_base`. bpe-openai and tiktoken-rs support more encodings.
- Benchmarks come from one machine under background load (see [Caveats](#caveats)).

## Quick start

The crate is not on crates.io. Use a git dependency pinned to a commit, or a path dependency:

```toml
[dependencies]
magi-tokenizer = { git = "https://github.com/magimetal/magi-tokenizer.git", rev = "<commit>" }
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

There is no setup step: all tables are built at compile time and embedded in the binary (see [Startup](#startup)).
The first call still pays for cold caches and for paging in the parts of the tables it touches.

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

### Panics

No function panics or loops forever on any `&str` input or any `u32` token ID. Unknown IDs return `None`.
Exceptions:

- **Allocation failure.** Out of memory aborts the process; `catch_unwind` cannot catch it. Encoding allocates
  output in proportion to the token count, and the whole-piece heap fallback allocates in proportion to the piece
  length.
- **Capacity overflow on 32-bit targets.** A single input or piece of several hundred MiB can request a buffer
  over `isize::MAX` bytes, which panics. On 64-bit targets the allocation fails first.

The audit for [issue #3](https://github.com/magimetal/magi-tokenizer/issues/3) lists the invariant behind each
index, `unwrap` and the `unsafe` table view. Compile-time assertions check the table and index invariants;
`verify` passes with `overflow-checks` and `debug-assertions` enabled in release.

## How it works

Tokenization has two stages, and both must match tiktoken exactly:

1. **Pretokenization.** A regex splits the text into pieces: words with an optional leading space or
   punctuation character, groups of 1–3 digits, punctuation runs, whitespace runs and English contractions.
2. **Byte-pair merging.** Each piece starts as single bytes. The adjacent pair whose merged token has the lowest
   rank is merged, the leftmost pair if ranks tie, until no adjacent pair forms a vocabulary token. In
   `o200k_base`, a token's rank is its ID.

magi-tokenizer keeps those semantics and changes how each stage is computed.

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
is slow. magi-tokenizer replaces it with a hand-written matcher that tries the seven alternatives in the same order and
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
2. **Short pieces (≤ 128 bytes).** Fixed-size arrays on the stack, no allocation. Pieces under 32 bytes keep their
   tokens packed: each step finds the minimum rank with a vectorised minimum and then the first position with that
   rank (leftmost tie-breaking), merges the pair, and shifts the rest down. Longer pieces leave each part in the
   slot of its first byte, linked to its neighbours, so a merge moves nothing. Each part's key packs its merge rank
   above its position, so the smallest key is the lowest rank and, among equal ranks, the leftmost pair. A minimum
   is kept for each block of 32 keys; a merge changes at most three keys, so only their blocks are rescanned.
   128 bytes is also the longest token in `o200k_base`.
3. **Long pieces (> 128 bytes).** Overlapping windows, described next.

#### Long pieces: overlapping windows

A heap-based merge costs O(n log n) per piece and is slow on long repetitive runs, such as 300,000 spaces, `aaaa…`
or `ababab…`. magi-tokenizer encodes a long piece as a series of short windows and joins them. The method rests on a
property of BPE noted in the [`bpe` crate README](https://github.com/github/rust-gems/tree/main/crates/bpe)
("Properties of BPE"):

> If two exact encodings of overlapping text contain the same token at the same byte position, the first encoding
> up to that token, followed by the second encoding after it, is the exact encoding of the combined text.

The loop:

1. Encode the first 128-byte window exactly (short path).
2. Start the next window at a token boundary of the current result, reaching back about 1/8 of the window size.
   Encode it exactly.
3. Look for a **stitch point**: a token with the same ID and the same end offset in both encodings. Both lists of
   token end offsets are sorted, so one merge-style pass finds it.
4. If found, keep the current result up to that token, append the window's tokens after it, and move forward.
5. If not found, double the window size and try again, up to 4,096 bytes. If 4,096 bytes still has no stitch
   point, encode the whole piece with the exact heap merge. This fallback is a safety net.
6. After each successful stitch, the window size goes back to 128 bytes, because hard stretches are usually local.

Two details make this fast on repetitive text:

- **Window cache.** A 16-slot cache, indexed by a hash of the window's bytes, stores recent window encodings. A hit
  counts only after a **full byte comparison**, never on the hash alone. Periodic text such as `ababab…` or
  `"!@#$%^&*()" × N` reuses earlier merges at every phase of the period. The cache lives for one call only;
  there is no state between calls. The window size decides how windows of periodic text line up, so it decides
  whether they recur: on `long_runs`, 128 bytes beat 64, and 48 or 96 bytes were about six times slower.
- **Boundary-aligned starts.** A window that starts where a token of the current result starts usually repeats
  the same merges, so the stitch point appears early.

Random text from a small alphabet finds stitch points just as quickly, but its windows never repeat, so every
window is merged from scratch. That merge cost is what remains in the slow cases below.

**Bounded memory.** A window never starts more than 640 bytes (the largest overlap plus one token) before the
end of the text encoded so far, so tokens ending further back are final. Every 4,096 pending tokens, the final
ones are handed out: appended to the output by `encode_ordinary`, only counted by `count_ordinary`. Working memory
for a long piece is therefore bounded regardless of its length: the 16 cached windows hold at most 64 KiB of bytes,
and the window merge state and pending tokens are a few hundred KiB at most (estimated from the data structures,
not measured including allocator overhead), usually a few KiB. The one exception is the whole-piece heap fallback, which
needs memory proportional to the piece; when it runs, `encode_ordinary` removes the tokens it already appended
for that piece.

Measured peak RSS for one random `a`/`b` piece (`magi-tokenizer-bench memory`, ARM64 macOS, fallback forced through
the `harness` feature):

| Piece | `count` via windows | `count` via fallback | `encode` via windows | `encode` via fallback |
| ---: | ---: | ---: | ---: | ---: |
| 16 MiB | 29 MiB, 0.3 s | 421 MiB, 3.3 s | 51 MiB, 0.3 s | 436 MiB, 3.1 s |
| 64 MiB | 77 MiB, 1.1 s | 1,669 MiB, 16.7 s | 164 MiB, 1.1 s | 1,732 MiB, 17.1 s |
| 256 MiB | 269 MiB, 4.5 s | 6,661 MiB, 88.6 s | 619 MiB, 4.4 s | 6,917 MiB, 85.5 s |

RSS includes the input text (1 byte per piece byte). In these runs the fallback peaked at 26–27 bytes per piece
byte with the input included, and took 11–20 times as long as windows, the gap growing with piece size. These are
measured examples, not worst-case bounds. No text in the `verify` suites reaches the fallback unless forced (see
below).

The heap merge stays in the code as the fallback, and it merges windows over 128 bytes. It stores positions as
`u32` and switches to `usize` for pieces of 4 GiB or more.

## Verification

`magi-tokenizer-bench verify` compares magi-tokenizer with **tiktoken-rs 0.12.0** (the reference) and with the original
regex, run through `fancy-regex` 0.17.0. For every text it checks:

- the pieces, against the reference regex;
- `encode_ordinary` IDs and `count_ordinary`;
- `decode` round trip;
- `encode_with_special_tokens` IDs and `count_with_special_tokens`;
- `encode_ordinary_into` appending to a non-empty vector, keeping its contents.

It also checks that each valid-UTF-8 vocabulary token that forms a single piece (198,424 tokens) encodes to its
own ID, and reports for each suite how many pieces over 128 bytes fell back to the whole-piece heap merge.

Two more checks use hooks from the crate's `harness` feature, which the bench enables and consumers should not:

- `forced_fallback_<n>` reruns the full text checks on the long-piece suites with window merging made to give
  up after `n` stitched windows (0, 1, 16, 1,024). Every long piece that needs more windows than that takes the
  discard-and-retry path, and the 1,024 setting discards tokens already appended or counted. The command fails if
  a setting reaches no fallback or no setting discards tokens. This injects the failure; it does not construct a
  piece that has no stitch point.
- `arbitrary_bytes` checks byte strings that are not regex pieces: random bytes, invalid UTF-8, small byte
  alphabets and valid text cut at any byte, at lengths 0 to 65,536 including 31/32/33, 127/128/129, 4,095/4,096/4,097.
  `encode_piece` (appending), `count_piece`, the stack merge, the window merge, the forced fallback and the heap
  merge with both `u32` and `usize` part indices must all match tiktoken-rs's `byte_pair_split` (pieces up to
  10,000 bytes) or the heap merge (longer ones).

| Suite | Texts | Bytes | What it covers |
| --- | ---: | ---: | --- |
| `edges` | 883 | 109,014 | Contractions, `ſ`, title-case letters, combining marks, NBSP and ideographic space, CR/LF mixes, special-token fragments, runs of 1–140 characters of six kinds |
| `unicode_sweep` | 6,370 | 52,020,015 | Every Unicode scalar value in six contexts that exercise each regex alternative, plus mixed triples of code points |
| `fuzz` | 200,000 | 14,861,094 | Seeded random strings from a hostile alphabet (contractions, marks, emoji, special-token parts, whitespace kinds) |
| `source_7` / `_48` / `_4096` | 869k / 127k / 1.5k | 6,082,527 each | Source files under `MAGI_TOKENIZER_SOURCE` (here the `toks` repository), cut at three chunk sizes |
| `multilingual_256` | 1,921 | 491,200 | CJK, Hangul, Cyrillic, Greek, Arabic, Devanagari, Thai, emoji ZWJ sequences |
| `fixture_200k` | 1 | 938,163 | Shared `o200k_base` comparison fixture |
| `long_runs` | 9 | 1,760,000 | 300k `a`, 300k spaces, repeated digits, `ab`, combining marks, punctuation, CJK, cycles |
| `long_random` | 1,092 | 2,598,617 | Long single pieces from small alphabets (letters, CJK, marks, punctuation, whitespace) with lengths around the window sizes (129 … 20,000) and runs with one interruption |
| `long_unrepeated` | 131 | 713,408 | Seeded random `a`/`b`, CJK and punctuation pieces, 129 bytes to 240 KB: no window ever repeats |
| `session_snapshots` | 20 | 11,850,632 | Real coding-agent session transcripts |
| `token_identity` | 198,424 tokens | — | Each valid-UTF-8, single-piece vocabulary token encodes to itself |
| `forced_fallback_<n>` | 1,233 × 4 | — | Long-piece suites again, with window merging forced to give up after 0, 1, 16 or 1,024 stitches |
| `arbitrary_bytes` | 630 pieces | 1,229,530 | Non-regex byte strings at boundary lengths, every merge path, against tiktoken-rs `byte_pair_split` |

`fixture_200k` and `session_snapshots` run only when `MAGI_TOKENIZER_ASSETS` is set. Latest result, with both
fixtures and the `toks` source tree: **0 mismatches in every suite**, in about 17 seconds on an Apple M5 Pro.

The window encoder was also checked separately: its counts match `bpe-openai` on random long pieces of up to
240 KB.

## Benchmarks

`magi-tokenizer-bench bench` measures count and encode throughput for magi-tokenizer, tiktoken-rs 0.12.0 and bpe-openai 0.3.2
on the same inputs. Before timing, it checks that all three return the same total token count. The five jobs
(magi count, magi encode, bpe-openai count, bpe-openai encode, tiktoken encode) run in five rounds of at least
150 ms each, in a rotated order each round; each figure is the job's median round. tiktoken-rs has no separate
count, so its encode figure is used for both.

Latest run, count throughput in MiB/s:

| Workload | MiB | magi-tokenizer | tiktoken-rs | bpe-openai | vs tiktoken | vs bpe-openai |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixture_200k` | 0.895 | 253.8 | 39.3 | 81.8 | 6.46× | 3.10× |
| `source_1MiB` | 1.000 | 203.1 | 32.6 | 59.5 | 6.24× | 3.41× |
| `source_48B_chunks` | 0.250 | 176.2 | 27.1 | 53.3 | 6.49× | 3.31× |
| `source_256B_chunks` | 0.250 | 212.2 | 29.6 | 58.0 | 7.18× | 3.66× |
| `multilingual` | 0.468 | 130.9 | 43.0 | 96.6 | 3.04× | 1.35× |
| `long_runs` | 1.678 | 954.1 | 14.1 | 163.7 | 67.46× | 5.83× |
| `long_unrepeated` | 0.680 | 54.0 | 14.5 | 43.5 | 3.71× | 1.24× |
| `session_snapshots` | 30.748 | 251.9 | 39.5 | 80.2 | 6.38× | 3.14× |

Encode throughput follows the same pattern: 3.01–6.56× tiktoken-rs and 1.41–3.95× bpe-openai on the ordinary-text
workloads.

Source workloads use the `toks` source tree; with the default source (this repository) the numbers differ.

### Where magi-tokenizer is still slower

Long single pieces of random CJK, where every window must be merged from scratch and each character starts as
three byte-parts (milliseconds per call, separate probe program, counts checked against bpe-openai):

| Input | magi-tokenizer | bpe-openai |
| --- | ---: | ---: |
| 240 KB random CJK from 8 characters | 4.4 | 2.5 |

For comparison, from the same probe run: 200 KB random `a`/`b` took 3.3 ms (bpe-openai 4.2), 2,000 random `a`/`b`
pieces of 129–1,024 bytes took 19.5 ms (bpe-openai 23.6), and random lowercase, letters and punctuation took about
half of bpe-openai's time.

### Startup

magi-tokenizer needs no runtime initialisation. On a contended machine, the first count after process start took
0.107 ms, against 166 ms for tiktoken-rs and 78 ms for bpe-openai (vocabulary loading). Process launch time is not
included, and the first call is still slower than later ones (cold caches, table pages faulted in on first use).

Peak memory, one random `a`/`b` piece (`/usr/bin/time -l`, separate probe program): counting 16, 64 and 128 MiB
pieces peaked at 29, 77 and 141 MiB of RSS, so it adds a constant 13 MiB on top of the text (embedded tables
paged in plus working buffers). Encoding the 64 MiB piece peaked at 164 MiB, because its 22.9 million IDs take
87 MiB. Memory for whole-text workloads and for other tokenizers has not been measured.

### Caveats

- All figures come from one ARM64 Apple M5 Pro with other work running in the background. Rotated job order
  limits drift between tokenizers, but repeated runs still vary by about 10%.
- The native `toks` C library has not been benchmarked yet.

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
bench/                       magi-tokenizer-bench: differential verification and benchmarks
tools/gen-unicode/           standalone generator for src/unicode_classes.rs (pinned regex versions)
```

## Development

Requirements: the library builds with Rust 1.88.0 and 1.90.0 (both checked in CI); older versions are untested. The
benchmark harness uses newer standard-library APIs and runs on stable in CI. Development used Rust 1.98.1.

```sh
# Library
cargo build --release

# Exact differential verification against tiktoken-rs and fancy-regex (about 17 s)
cargo run --release -p magi-tokenizer-bench -- verify

# Throughput, all three tokenizers; an optional filter matches workload names
cargo run --release -p magi-tokenizer-bench -- bench
cargo run --release -p magi-tokenizer-bench -- bench long

# First-call time in a fresh process
cargo run --release -p magi-tokenizer-bench -- startup magi        # or tiktoken, bpe-openai

# Check whole files (issue reproductions): fails if magi-tokenizer differs from tiktoken-rs in pieces, IDs,
# counts, decode or special handling, or if bpe-openai's count differs from tiktoken-rs
cargo run --release -p magi-tokenizer-bench -- check path/to/repro.txt

# Regenerate the Unicode class table (only when the pinned regex versions change)
cd tools/gen-unicode && cargo run --release
```

Harness inputs, both optional:

- `MAGI_TOKENIZER_SOURCE`: source trees to cut into chunks, in the platform's path-list syntax. Default: this
  repository. Files under dot-directories and `target` are skipped; at most about 4 MiB is read per tree.
- `MAGI_TOKENIZER_ASSETS`: a directory containing `o200k_base-200k.txt` and `session-compacted-500.json`. Unset: those
  suites are skipped. Set but a file is missing or malformed: the command fails.

The release profile uses fat LTO and one codegen unit. Build scripts are compiled with `opt-level = 3`, so table
generation stays fast in debug builds as well.

Correctness is checked by the differential `verify` command rather than unit tests. Run it after any change to
the scanner, tables or merge code.

## Known limitations

- **Slow case.** Long random pieces of CJK text from a small set of characters (see above).
- **Heap fallback memory.** If a stretch of a long piece has no stitch point within 4,096 bytes, the whole piece
  is merged with the heap: 26–27 bytes of peak memory per piece byte and 11–20 times slower than windows in the
  measured runs (see [Bounded memory](#how-it-works)); not a proven bound. No piece in the `verify` suites,
  including the session snapshots and the shared fixture, reaches it naturally, but no input is known to be unable
  to reach it.
- **One encoding.** Only `o200k_base`. Other encodings (`cl100k_base`, `o200k_harmony`) and model-name routing are
  not implemented.
- **Testing gaps.** The `usize` index path is checked on small pieces through a harness hook, not on a real piece
  of 4 GiB or more. Speed and memory measured only on ARM64 macOS. CI runs without `MAGI_TOKENIZER_ASSETS`, so the
  shared fixture and session snapshots are checked only locally.

## Provenance and licensing

magi-tokenizer's code is licensed under the [Apache License 2.0](LICENSE). The crate's license expression is
`Apache-2.0 AND MIT AND Unicode-3.0`, because it bundles the MIT-licensed tiktoken vocabulary and tables derived
from the Unicode Character Database. [NOTICE](NOTICE) carries the attributions below, including the full MIT
notice for the vocabulary and the full Unicode License v3 text.

| Component | Source | Licence |
| --- | --- | --- |
| `assets/o200k_base.tiktoken` | OpenAI [tiktoken](https://github.com/openai/tiktoken) public vocabulary | MIT (tiktoken) |
| Scanner design (`src/scanner.rs`) | [toks](https://github.com/actualcomputer/toks) revision `2ebc390`, `src/core/k3_o200k_c.c` and `kernels.h`, Copyright 2026 Actual Computer Inc., rewritten in Rust | Apache-2.0 |
| Window stitching property | [`bpe` crate](https://github.com/github/rust-gems/tree/main/crates/bpe) README, "Properties of BPE". Only the property is used; no code from the crate is included. | MIT |
| Unicode classes | Generated from `regex` 1.12.4 / `regex-syntax` 0.8.11 (Unicode 16 data) | MIT OR Apache-2.0; Unicode License v3 |
| Everything else | magi-tokenizer | Apache-2.0 |
