# magi-tokenizer

A fast Rust tokenizer for OpenAI's `o200k_base` and `o200k_harmony`, with optional `cl100k_base`,
`r50k_base`, `p50k_base` and `p50k_edit` support. Encode text, count tokens without building a token list,
or decode IDs back to bytes or text.

The token IDs match [tiktoken](https://github.com/openai/tiktoken) for ordinary encoding, encoding with
special tokens, and encoding with a special-token allow-list. Vocabulary tables are built into the binary,
so there's nothing to load before the first call.

Current version: 0.1.0. See the [compatibility policy](#compatibility) for release guarantees.
There is no automated correctness suite. Check [Known limitations](#known-limitations) and
[GitHub issues](https://github.com/magimetal/magi-tokenizer/issues) before adopting it.

## Quick start

Add the crate from [crates.io](https://crates.io/crates/magi-tokenizer):

```toml
[dependencies]
magi-tokenizer = "0.1"
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

The first call still pays for cold caches and table pages loaded on demand. See [Startup](#startup) for measurements.

For `cl100k_base` (GPT-4, GPT-3.5 Turbo, `text-embedding-*`), enable its feature:

```toml
magi-tokenizer = { version = "0.1", features = ["cl100k_base"] }
```

```rust
use magi_tokenizer::cl100k_base;
let ids = cl100k_base::encode_ordinary("Hello, world!");
assert_eq!(cl100k_base::count_ordinary("Hello, world!"), ids.len());
```

For legacy GPT-2/GPT-3 encodings, enable `r50k_base` (also called `gpt2`) or `p50k_base`.
The `p50k_base` feature includes `p50k_edit`; they share one vocabulary.

```toml
magi-tokenizer = { version = "0.1", features = ["r50k_base", "p50k_base"] }
```

```rust
use magi_tokenizer::{p50k_edit, r50k_base};
let ids = r50k_base::encode_ordinary("Hello, world!");
assert_eq!(p50k_edit::encode_with_special_tokens("<|fim_prefix|>")[0], p50k_edit::FIM_PREFIX);
```

`o200k_harmony` (`gpt-oss-*`) needs no feature flag. It reuses `o200k_base`'s tables and adds its own special tokens:

```rust
use magi_tokenizer::o200k_harmony;
let ids = o200k_harmony::encode_with_special_tokens("<|start|>user<|message|>Hi<|end|>");
assert_eq!(ids[0], o200k_harmony::START);
```

## Scope

### Encodings

Every module has the same functions and constants. The crate root re-exports `o200k_base`.

| Encoding / module | Cargo feature | Ordinary tokens | Ordinary IDs |
| --- | --- | ---: | --- |
| `o200k_base` | Always built | 199,998 | 0 to 199,997 |
| `o200k_harmony` | Always built | Same as `o200k_base` | Same as `o200k_base` |
| `cl100k_base` | `cl100k_base` | 100,256 | 0 to 100,255 |
| `r50k_base` | `r50k_base` | 50,256 | 0 to 50,255 |
| `p50k_base` | `p50k_base` | 50,280 | 0 to 50,280, except 50,256 (`<\|endoftext\|>`) |
| `p50k_edit` | `p50k_base` | Same as `p50k_base` | Same as `p50k_base` |

`o200k_harmony` uses `o200k_base`'s split pattern, so their ordinary IDs and counts are identical.
Likewise, `p50k_edit` differs from `p50k_base` only in special tokens.
`r50k_base`, `p50k_base` and `p50k_edit` all use the same split pattern.

### Special tokens

| Encoding | Token IDs |
| --- | --- |
| `o200k_base` | `<\|endoftext\|>` = 199,999; `<\|endofprompt\|>` = 200,018 |
| `cl100k_base` | `<\|endoftext\|>` = 100,257; `<\|fim_prefix\|>` = 100,258; `<\|fim_middle\|>` = 100,259; `<\|fim_suffix\|>` = 100,260; `<\|endofprompt\|>` = 100,276 |
| `r50k_base`, `p50k_base` | `<\|endoftext\|>` = 50,256 |
| `p50k_edit` | `<\|endoftext\|>` = 50,256; `<\|fim_prefix\|>` = 50,281; `<\|fim_middle\|>` = 50,282; `<\|fim_suffix\|>` = 50,283 |

`o200k_harmony` has 1,090 special tokens, IDs 199,998 to 201,087, matching tiktoken-rs 0.12.0:

| Token | ID |
| --- | ---: |
| `<\|startoftext\|>` | 199,998 |
| `<\|endoftext\|>` | 199,999 |
| `<\|return\|>` | 200,002 |
| `<\|constrain\|>` | 200,003 |
| `<\|channel\|>` | 200,005 |
| `<\|start\|>` | 200,006 |
| `<\|end\|>` | 200,007 |
| `<\|message\|>` | 200,008 |
| `<\|call\|>` | 200,012 |

Every other Harmony ID is named `<|reserved_{id}|>`. In particular, 200,018 is `<|reserved_200018|>`,
not `<|endofprompt|>`.

### Models

The library exposes encodings, not a model-name router. Your application chooses the encoding
(see [Model routing](#model-routing)). As a starting point, upstream tiktoken's `tiktoken/model.py` maps:

| Models | Encoding |
| --- | --- |
| GPT-4o, GPT-4.1, GPT-4.5, GPT-5, o-series | `o200k_base` |
| GPT-4, GPT-3.5 Turbo, `davinci-002`, `babbage-002`, `text-embedding-ada-002`/`-3` | `cl100k_base` |
| `gpt-oss-*` | `o200k_harmony` |
| Codex models, `text-davinci-002`, `text-davinci-003` | `p50k_base` |
| `text-davinci-edit-001`, `code-davinci-edit-001` | `p50k_edit` |
| GPT-2, `davinci`, other GPT-3 base models | `r50k_base` |

magi-code v0.97.6 also routes `gpt-6-*` and `gpt-6.1-*`, including `gpt-6-astra`, to `o200k_base`.
Its maintainer validated that routing in magi-code; it is not a mapping supplied by this library.

### Input and platforms

Encoding accepts `&str`: valid UTF-8 only. Convert other bytes first, for example with `String::from_utf8_lossy`
(see [Byte input](#byte-input)). Decoding can return raw bytes even when a token sequence isn't valid UTF-8.

The library supports any Rust target and emits tables in the target's byte order. CI builds on x86_64 and ARM64
Linux, ARM64 macOS, x86_64 and ARM64 Windows, 32-bit i686 Linux and big-endian s390x Linux.
Speed and memory measurements cover ARM64 macOS only.

## Using it in your project

### Choose how to handle special tokens

These modes differ in what happens to special-token text such as `<|endoftext|>`
([#10](https://github.com/magimetal/magi-tokenizer/issues/10)):

| Function | What happens to special-token text |
| --- | --- |
| `encode_ordinary(text)` | Encodes it as ordinary text, like tiktoken's `encode_ordinary`. |
| `encode_with_special_tokens(text)` | Recognises every special token, like tiktoken's `encode(text, allowed_special="all")`. |
| `encode_allowing_special(text, allowed)` | Recognises only listed tokens. Treats the rest as ordinary text, like tiktoken-rs 0.12.0 `CoreBPE::encode(text, allowed)`. |
| `encode(text, allowed)` | Recognises listed tokens. Returns `Err(DisallowedSpecialToken)` for any unlisted special-token text, like Python tiktoken's default `encode(text, allowed_special=allowed)`. |

`encode` reports the leftmost disallowed token and its byte offset. With no disallowed tokens, it returns
the same IDs as `encode_allowing_special`. tiktoken-rs has no equivalent error; Python tiktoken's
`disallowed_special="all"` check is the reference. Pass `&[]` to reject all special-token text in untrusted input.
Names in `allowed` that aren't special tokens for the selected encoding are ignored.

If you only need a count, use `count_ordinary`, `count_with_special_tokens`, `count_allowing_special` or `count`.
Each returns the matching encode function's length without storing a token list. `count` returns the same error as `encode`.

### Limit untrusted input

Cap input size before calling the library. There is no memory-limit setting.

Encoding allocates 4 bytes per output token, reserving one token per 3 input bytes up front. The whole-piece
fallback allocates its working memory once: exactly 20 bytes per piece byte (32 for pieces of 4 GiB and more on
64-bit targets), with O(n log n) runtime
([#12](https://github.com/magimetal/magi-tokenizer/issues/12)). Measured peak memory was 21 to 22 bytes per piece byte,
including input. No tested input reaches fallback naturally, but there is no known input condition that rules it out.
An input cap of N bytes bounds fallback working memory at 20 × N bytes, or 32 × N for N of 4 GiB and more.

### Binary size and build time

Embedded `o200k_base` tables add about 19 MiB. On ARM64 macOS, a release program counting one string was 19.9 MB,
compared with 0.5 MB for an empty program and 35.1 MB with bpe-openai. Clean debug and release builds each took
about 1.3 s on an Apple M5 Pro. Your project uses its own build profiles, not this repository's.
If debug table generation is slow, add `[profile.dev.build-override] opt-level = 3`.

Optional encodings add to the binary only when used. In a fat-LTO release program counting one string with
each encoding, `cl100k_base` grew the binary from 20.1 MB to 30.2 MB (+10.1 MB).
Enabling the feature without calling it left the size unchanged: the linker dropped the unused tables.
Clean release builds took 3.6 s without the feature and 4.5 s with it (median of three, Apple M5 Pro).

In the same stripped fat-LTO program, calling either `r50k_base` or `p50k_base` grew the binary from 20.1 MB to
25.4 MB (+5.3 MB). Calling both grew it to 30.4 MB. Enabling either feature without calling it left the size unchanged.
Clean release builds took 3.3 s with neither feature, 3.8 s with one enabled but unused, 3.9 s with one used,
and 4.6 s with both used (median of three, Apple M5 Pro). `p50k_edit` shares `p50k_base`'s tables.

### Pick a release

`magi-tokenizer = "0.1"` accepts only compatible releases under the [compatibility policy](#compatibility).
The 0.0.x releases make no compatibility promise; pin them exactly (`=0.0.2`). For git dependencies, pin a full
commit hash whose CI passed. [CHANGELOG.md](CHANGELOG.md) lists changes.

Don't enable `harness` in your application. It exposes memory-benchmark hooks for `bench/` and sits outside
the compatibility policy.

### Compatibility

The policy starts at 0.1.0 ([#13](https://github.com/magimetal/magi-tokenizer/issues/13)). Versions 0.0.1 and 0.0.2
make no compatibility promise.

From 0.1.0, releases follow Cargo's SemVer rules. Breaking a promise below requires a new minor version
(0.1 → 0.2), which Cargo treats as incompatible. Patch releases (0.1.0 → 0.1.1) keep these promises,
so `magi-tokenizer = "0.1"` accepts only compatible releases.

Within a minor version:

- Token IDs match tiktoken-rs 0.12.0 for every `&str`, using its bundled vocabularies and Unicode 16 data.
  The `encode` and `count` error follows Python tiktoken's disallowed-special check. Counts equal encode lengths,
  `pieces` follows the reference split pattern, and `decode` and `token_bytes` return vocabulary bytes.
  A new reference, Unicode version or vocabulary can move token boundaries and requires a new minor version.
  A mismatch with the current reference is a bug. Its fix ships in a patch release under "Fixed" in the changelog,
  even though the fix changes IDs for affected inputs.
- Public items in `magi_tokenizer` and its encoding modules (`o200k_base`, `o200k_harmony`, `cl100k_base`,
  `r50k_base`, `p50k_base`, `p50k_edit`) won't be removed, renamed or changed incompatibly.
  The same applies to the crate-root `o200k_base` re-export and the `cl100k_base`, `r50k_base` and `p50k_base` features.
  Patch releases may add functions, constants, encodings and features.
- The minimum Rust version stays at 1.88. Raising it requires a new minor version.

The policy does not cover:

- `harness` or `magi_tokenizer::harness`. The module is hidden from docs, but Cargo cannot hide the feature.
  Either can change or disappear in any release. Don't enable it in your application.
- Speed, memory use, binary size or build time. These can change in any release; the changelog records notable changes.

[CHANGELOG.md](CHANGELOG.md) is the release record, with breaking changes under "Breaking".
Each GitHub release repeats its version's changelog section.

### Model routing

Choose the encoding in your application. This library does not map model names to encodings and has no plans to
add a router ([#9](https://github.com/magimetal/magi-tokenizer/issues/9)).

Models arrive more often than crate releases, and applications may stay on one minor version for a while.
A built-in mapping would lag. There isn't one agreed list, either: upstream tiktoken does not list `gpt-6-*`,
which magi-code routes to `o200k_base`. The application knows which models it calls.

A router would also depend on which features you compile: `cl100k_base`, `r50k_base` and `p50k_base` may be absent.
Changing a name's encoding changes its token counts, so every mapping update would affect public API behavior.
The [compatibility policy](#compatibility) allows those count changes only in a new minor version.

A wrong encoding gives wrong counts without an error. Reject unknown model names, or choose a default deliberately.
Don't fall back silently. The [Scope](#scope) section lists the upstream mappings to start from.

### Byte input

Encoding takes `&str` only. There is no byte-input API
([#11](https://github.com/magimetal/magi-tokenizer/issues/11)). If bytes may contain invalid UTF-8, convert them
first, usually with `String::from_utf8_lossy`.

Neither public reference defines tokenization for invalid UTF-8: tiktoken-rs 0.12.0 accepts `&str`, and Python
tiktoken's `encode` accepts `str`. Without a reference, this crate could not check byte-input IDs exactly.

Python tiktoken calls its private `_encode_bytes` a "best guess". It splits the valid UTF-8 prefix with the regex,
then merges that prefix's last piece and everything from the first invalid byte as one piece.
One stray byte near the start of a large input could make the rest a single piece and trigger slow whole-piece fallback here.

The split patterns operate on Unicode text. Any rule for invalid bytes would be specific to this crate.
Supporting byte input would also take more than a new signature: the scanner decodes UTF-8 without validation.

`String::from_utf8_lossy` replaces invalid sequences with U+FFFD, just like Python's
`bytes.decode("utf-8", errors="replace")`. This was checked on truncated, overlong, surrogate and out-of-range sequences.
The resulting IDs match Python tiktoken for text decoded that way, not the original bytes.
Each U+FFFD becomes `EF BF BD`; `decode_bytes` returns those three bytes, not the bytes that were replaced.

## API

There is no tokenizer object to create. All functions are free functions, with no global state or locking,
and any number of threads can call them concurrently.

Each encoding module (`magi_tokenizer::o200k_base`, `magi_tokenizer::o200k_harmony`,
`magi_tokenizer::cl100k_base`, `magi_tokenizer::r50k_base`, `magi_tokenizer::p50k_base`, `magi_tokenizer::p50k_edit`)
has the functions below. The crate root re-exports `o200k_base`.

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

Each module exposes `SPECIAL_TOKENS` (sorted by ID), `ORDINARY_TOKEN_COUNT` and a constant for every named special
token: `END_OF_TEXT`, `END_OF_PROMPT`, `FIM_PREFIX`, and others. Harmony's named constants are `START_OF_TEXT`,
`END_OF_TEXT`, `RETURN`, `CONSTRAIN`, `CHANNEL`, `START`, `END`, `MESSAGE` and `CALL`.
The error type lives at the crate root: `DisallowedSpecialToken { token, position }`.

Special-token lookup doesn't scan the whole list at each `<|`. Every special token starts with `<|` and ends with
`|>`, with no `|>` inside; compile-time checks enforce this. The text up to the next `|>` is the only candidate.
Harmony uses binary search over its 1,090 tokens. Other encodings compare their short lists directly.
`token_bytes` also finds special IDs by binary search.

Calling `o200k_harmony` adds 49,552 bytes to a stripped LTO release binary, measured on ARM64 macOS against
the same program calling `o200k_base`.

### Panics

Functions don't panic or loop forever on any `&str` or `u32` token ID, except at these resource limits:

- Out of memory aborts the process. `catch_unwind` cannot catch it. Encoding allocates by token count,
  and whole-piece fallback allocates 20 bytes per piece byte (32 for pieces of 4 GiB and more on 64-bit targets).
- On 32-bit targets, inputs or pieces of several hundred MiB can request buffers larger than `isize::MAX`,
  which panics. On 64-bit targets, allocation fails first.

Unknown IDs return `None`. The [issue #3 audit](https://github.com/magimetal/magi-tokenizer/issues/3) documents
the invariants behind each index, `unwrap` and `unsafe` table view. Compile-time assertions check table and index
invariants. There is no automated correctness suite.

## What is different

Exact tokenization requires the same regex splits and the same lowest-rank adjacent-pair merges.
The implementations differ in how they get there:

| Stage | tiktoken-rs | bpe-openai | magi-tokenizer |
| --- | --- | --- | --- |
| Vocabulary setup | Decoded and hashed at run time (37 ms first call) | Built at run time (23 ms first call) | Built by `build.rs` into embedded hash tables; nothing runs at start (0.014 ms first call) |
| Pretokenizer | `fancy-regex` with lookahead | `regex-automata` with a lookahead workaround | Hand-written scanner (derived from `toks`) with Unicode 16 classes generated from the same `regex` crate version as the reference |
| Short pieces | Rescans all pair ranks after each merge | Backtracking search over Aho-Corasick automata of the vocabulary | Stack arrays, no allocation; packed rank+position keys with per-block minimums |
| Long pieces | Same rescan: cost grows with the square of the length (14 MiB/s on long runs) | Same backtracking search | Overlapping windows, stitched and cached (924 MiB/s on long runs) |
| Counting | Encodes, then takes the length | Separate count function | Separate count function; long pieces counted in bounded memory |

In these measurements, `o200k_base` counting on source code, transcripts and the shared fixture was about
7× faster than tiktoken-rs and 3.3× faster than bpe-openai. Cached windows pushed long runs to about 66× and 5.7×,
respectively. Counting skips storage for output tokens. These results exclude whole-piece fallback.

magi-tokenizer covers the same OpenAI encodings as tiktoken-rs. bpe-openai has no `r50k_base`, `p50k_base` or `p50k_edit`.
All measurements come from one machine under background load. See [Caveats](#caveats) before comparing them with yours.

## How it works

Tokenization has two steps:

1. Split text into pieces: words with optional leading space or punctuation, digit groups, punctuation runs,
   whitespace runs and English contractions. The exact rules depend on the encoding; o200k uses groups of 1 to 3 digits.
2. Merge adjacent byte pairs within each piece. Always choose the lowest vocabulary rank, breaking ties to the left.
   Stop when no pair forms a vocabulary token. In every encoding, the rank is the token ID.

<details>
<summary>Implementation details: tables, scanning, merging and memory</summary>

### Build-time tables (`build.rs`, `src/table_hash.rs`, `src/tables.rs`)

`build.rs` has no dependencies. It reads `assets/<name>.tiktoken` for each enabled encoding and emits aligned
binary tables in the target's byte order. The `cl100k_base`, `r50k_base` and `p50k_base` tables need their feature flags.

`p50k_base` leaves rank 50,256 for `<|endoftext|>`. The build script records the missing ordinary rank and assigns
it empty bytes. Ordinary lookups never return it, and `token_bytes` falls through to the special token.

The sizes below describe `o200k_base`. Each hash table uses the smallest power-of-two slot count at least twice
its entry count. For comparison, `cl100k_base` uses 2^18 vocabulary slots and 2^19 pair slots.

- Token bytes and offsets map IDs to bytes for decoding.
- The vocabulary hash table has 2^19 slots, about 38% full. Each slot uses two `u64`s for the first 8 bytes,
  4 tail bytes, length and ID. Tokens up to 12 bytes are confirmed in the slot. Longer tokens require a byte comparison.
- The pair table has 2^20 slots, about 43% full. Each `u64` maps `(left ID, right ID)` to the merged ID.
  It covers all 446,189 splits of vocabulary tokens into two vocabulary tokens (233,378 for `cl100k_base`, 108,299 for `r50k_base`, 108,599 for `p50k_base`). IDs fit in 18 bits.
- A dense 65,536-entry array holds byte-pair ranks for the most common lookup: the first merge.
- A byte-token table holds the 256 single-byte tokens (IDs 0 to 255).
- A 65,536-entry array, one per 2- and 3-byte character, holds the character's token when merging may start from
  it whole instead of from its bytes (see [Byte-pair merging](#byte-pair-merging-srcbpers)), else 0.

The build checks token and pair counts. A changed vocabulary fails the build instead of silently producing
wrong tables. Both build script and library use the hash functions in `src/table_hash.rs`.
Lookups and merging are generic over the encoding; the compiler produces a copy for each, with constant table addresses.

### Pretokenizer (`src/scanner.rs`, `src/unicode_classes.rs`)

The o200k regex uses lookahead (`\s+(?!\S)`), which tiktoken handles with the slower `fancy-regex` backtracking
engine. The hand-written scanner tries all seven alternatives in order, with the same backtracking rules:

- Contractions are case-insensitive, but `'ſ` must not match `'s`.
- The optional leading character cannot be a letter, digit, CR or LF.
- Whitespace runs return their last space to the next word.
- `\s*[\r\n]+` and `\s+(?!\S)` retain their regex behavior.

The `cl100k_base` scanner handles standalone contractions, letters with an optional possessive leading character,
groups of 1 to 3 digits, and punctuation runs that take trailing CR/LF. Its whitespace alternatives are
`\s++$`, `\s*[\r\n]`, `\s+(?!\S)` and `\s`.

The third scanner handles `r50k_base`, `p50k_base` and `p50k_edit`. Contractions are case-sensitive.
Letter, digit and punctuation runs have no length limit and allow only U+0020 as an optional leading space.
Whitespace follows `\s++$`, `\s+(?!\S)` and `\s`. All three scanners share the class table and run helpers.

For each Unicode scalar, the scanner looks up one class byte in `src/unicode_classes.rs` (about 110 KB).
The generator, `tools/gen-unicode`, pins `regex` 1.12.4 and `regex-syntax` 0.8.11 to match tiktoken-rs 0.12.0's
Unicode 16 data. Updating those versions would change token boundaries.

Matching order follows [toks](https://github.com/actualcomputer/toks) (`src/core/k3_o200k_c.c`), rewritten in Rust.
See [Provenance](#provenance-and-licensing).

### Byte-pair merging (`src/bpe.rs`)

Most pieces already exist in the vocabulary. One- and two-byte pieces use direct byte and byte-pair tables.
Longer pieces need one hash-table probe. Either way, a match returns the ID without merging.

The remaining pieces follow two paths:

- Up to 128 bytes, merging uses fixed stack arrays without allocation. Below 32 bytes, a vectorised minimum
  finds the lowest rank; its first position breaks ties. Each merge shifts the remaining tokens down,
  which measured faster at this size. From 32 to 128 bytes, parts stay in their first-byte slots, linked to
  their neighbours. A 32-bit key packs rank above position, so the minimum selects the right pair without
  a separate tie check. Each block of 32 keys stores a minimum. A merge changes at most three keys:
  at most three block rescans and four block-minimum checks. The longest `o200k_base` token is also 128 bytes.
- Over 128 bytes, merging uses overlapping windows, described below.

Some 2- and 3-byte characters can start as one part instead of separate bytes. The build checks two conditions:
the character's bytes must merge into its token, and every token joining part or all of it to a neighbour
must rank above each of those internal merges. Its bytes therefore merge first, giving identical tokens with less work.
At runtime, the preceding byte must be a continuation byte,
and the following byte must start a multibyte character, unless the piece ends there.
Those are the neighbours covered by the build check. ASCII-only pieces skip it.
Random CJK and Hangul pieces need about a third as many parts.

One case is checked at runtime: a token joining a character's last byte to the next character's lead byte.
A 4,096-entry table stores the lowest such token for each (continuation, lead) pair, checked against the actual next byte.
In `o200k_base`, such tokens rank below most CJK characters for some lead bytes.
Checking every possible neighbour at build time allowed only 1,037 of its 2,515 single-token characters
in U+4E00 to U+9FFF to start whole. The runtime check allows 2,493.
For `cl100k_base`, it allows 546 of 549, compared with 432 under the build-only check.
The rest still need separate bytes. For example, `删除` (rank 39189) joins a whole character to its neighbour
and ranks below `删` (151816).

#### Long pieces: overlapping windows

Merging a whole piece costs O(n log n). Repetitive input, such as 300,000 spaces, `aaaa…` or `ababab…`, makes
that expensive. magi-tokenizer instead encodes short windows and joins them using this property from the
[`bpe` crate README](https://github.com/github/rust-gems/tree/main/crates/bpe), under "Properties of BPE":

> If two exact encodings of overlapping text contain the same token at the same byte position, the first encoding
> up to that token, followed by the second encoding after it, is the exact encoding of the combined text.

The `bpe` crate uses this property in its search algorithm. This project uses no code from that crate.
The window algorithm, cache, boundary alignment, streaming finalization, packed-key merge and all-splits
pair table were designed for magi-tokenizer.

1. Encode the first 128-byte window with the short-piece merge.
2. Start the next window at an existing token boundary, with about 1/8 of the window overlapping. Encode it exactly.
3. Find the first matching token ID and end offset in both encodings. One merge-style pass over sorted offsets finds this stitch point.
4. Keep the existing result through that token, then append the remaining tokens from the new window.
5. If there is no stitch point, double the window and retry, up to 4,096 bytes. If that also fails, merge the whole piece.
6. After a successful stitch, return to 128-byte windows. Difficult stretches are usually local; each window costs a few microseconds without special data structures.

A 16-slot cache reuses window encodings within each piece. A hash match still requires a full byte comparison,
so collisions cannot change the result. Repetitive input such as `"!@#$%^&*()" × N` reuses merges at each phase
of its period. Window size affects alignment: on `long_runs`, 128 bytes beat 64; 48 and 96 were about six times slower.

Starting at existing token boundaries repeats earlier merges and usually finds a stitch within the first few tokens.
Random small-alphabet text finds stitches just as quickly, but never repeats windows. It still pays for each merge.

Windows reach back at most 640 bytes: the maximum overlap plus one token. Tokens ending further back are final.
Every 4,096 pending tokens, the encoder appends final tokens to its output. The counter counts them without storing them.

Window working memory does not grow with piece length: at most 64 KiB of cached bytes, plus a few hundred KiB
for merge state and pending tokens. Usually it takes a few KiB. These estimates exclude allocator overhead.
One call allocates these buffers once and reuses them for every later long piece.
Whole-piece fallback is the exception: it discards prior output for that piece and allocates by piece length.

Fallback allocates five `u32` arrays once, each as long as the piece: tokens, two links, ranks and a tournament tree.
The tree selects the lowest rank, breaking ties to the left. This costs exactly 20 bytes of working memory per
piece byte, regardless of content; pieces of 4 GiB and more on 64-bit targets use `usize` links and indices,
32 bytes per piece byte. Each merge updates at most three tree paths, giving O(n log n) runtime.
The older binary heap kept stale entries, so its peak memory depended on content
([#12](https://github.com/magimetal/magi-tokenizer/issues/12)).

Peak RSS for one random `a`/`b` piece (`magi-tokenizer-bench memory`, ARM64 macOS, fallback forced through `harness`):

| Piece | `count` via windows | `count` via fallback | `encode` via windows | `encode` via fallback |
| ---: | ---: | ---: | ---: | ---: |
| 16 MiB | 29 MiB, 0.3 s | 341 MiB, 2.6 s | 51 MiB, 0.3 s | 357 MiB, 2.6 s |
| 64 MiB | 77 MiB, 1.1 s | 1,349 MiB, 11.8 s | 165 MiB, 1.1 s | 1,413 MiB, 12.1 s |
| 256 MiB | 269 MiB, 4.2 s | 5,381 MiB, 55.4 s | 619 MiB, 4.3 s | 5,637 MiB, 56.6 s |

RSS includes the input at 1 byte per piece byte. Encoding adds 4 bytes per output token.
Fallback peaked at 21 to 22 bytes per piece byte, compared with 26 to 27 for the heap, and took 9 to 13 times as long as windows.

The whole-piece merge also handles windows over 128 bytes. At 4 GiB, positions switch from `u32` to `usize`.
The two link arrays and tree then use 8-byte entries, raising working memory to 32 bytes per piece byte.

</details>

## Benchmarks

`magi-tokenizer-bench bench` compares counting and encoding against tiktoken-rs 0.12.0 and bpe-openai 0.3.2.
Both libraries cover `o200k_base` and `cl100k_base`. Only tiktoken-rs covers `r50k_base` and `p50k_base`.

Before timing, the benchmark checks exact token IDs, counts and decoded text against tiktoken for:

- Every workload, plus special tokens inserted between texts. All-allowed and allow-list cases use tiktoken-rs
  `CoreBPE::encode`; `encode` errors use Python tiktoken's leftmost-disallowed rule.
- A Unicode sweep over planes 0 to 2 and 14, in six contexts.
- Short random strings around scanner edge cases, plus forced whole-piece fallback.

It then times five jobs: magi count/encode, bpe-openai count/encode and tiktoken encode.
Each job runs for at least 150 ms per round, across five rounds in rotating order. Results are median rounds.
tiktoken-rs has no separate count function, so its count figures use encode.

Harmony rows (`harmony/...`) get the same exact-ID checks against tiktoken-rs `o200k_harmony_singleton()`.
Its ordinary workloads are checked but not timed: they use `o200k_base`'s encoding, and bpe-openai has no Harmony tokenizer.
Likewise, `p50k_edit/...` ordinary workloads are checked but not timed; they use `p50k_base`'s encoding.

For every encoding, `special_tokens` times `encode_with_special_tokens` and `count_with_special_tokens`
against tiktoken-rs. Each 256-byte source chunk is followed by a special token and a stray `<|endof`.

The count tables below use MiB/s. They record commit `7e90235`, Rust 1.98.1, an Apple M5 Pro, `toks` source
at revision `2ebc390`, and both private assets. Speedups use unrounded figures.

| Workload | MiB | magi-tokenizer | tiktoken-rs | bpe-openai | vs tiktoken | vs bpe-openai |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixture_200k` | 0.895 | 252.2 | 38.5 | 76.8 | 6.55× | 3.29× |
| `source_1MiB` | 1.000 | 228.2 | 32.7 | 62.4 | 6.99× | 3.66× |
| `source_48B_chunks` | 0.250 | 199.9 | 27.4 | 54.6 | 7.29× | 3.66× |
| `source_256B_chunks` | 0.250 | 242.2 | 31.1 | 59.5 | 7.78× | 4.07× |
| `multilingual` | 0.468 | 207.1 | 41.9 | 97.4 | 4.94× | 2.13× |
| `scanner_edges` | 0.262 | 81.5 | 17.1 | 42.4 | 4.76× | 1.92× |
| `long_runs` | 1.678 | 924.4 | 14.0 | 163.4 | 66.03× | 5.66× |
| `long_unrepeated` | 0.680 | 76.1 | 15.0 | 44.7 | 5.07× | 1.70× |
| `random_cjk_ineligible` | 0.229 | 245.2 | 22.6 | 90.6 | 10.84× | 2.71× |
| `session_snapshots` | 30.748 | 273.3 | 39.6 | 83.2 | 6.90× | 3.28× |
| `special_tokens` | 0.271 | 223.6 | 32.7 | – | 6.84× | – |

Encode was 4.14 to 10.70× faster than tiktoken-rs and 1.72 to 4.09× faster than bpe-openai, except on `long_runs`
(65.79× and 5.94×). Source results differ when using this repository instead of `toks`.

`cl100k_base` count throughput in MiB/s, same run:

| Workload | MiB | magi-tokenizer | tiktoken-rs | bpe-openai | vs tiktoken | vs bpe-openai |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixture_200k` | 0.895 | 223.7 | 25.6 | 86.9 | 8.75× | 2.57× |
| `source_1MiB` | 1.000 | 244.6 | 21.3 | 66.2 | 11.48× | 3.70× |
| `source_48B_chunks` | 0.250 | 202.8 | 18.3 | 58.8 | 11.05× | 3.45× |
| `source_256B_chunks` | 0.250 | 246.2 | 20.3 | 62.1 | 12.13× | 3.96× |
| `multilingual` | 0.468 | 215.0 | 28.3 | 91.4 | 7.59× | 2.35× |
| `scanner_edges` | 0.262 | 94.7 | 11.9 | 42.4 | 7.97× | 2.24× |
| `long_runs` | 1.678 | 992.8 | 15.9 | 120.9 | 62.52× | 8.21× |
| `long_unrepeated` | 0.680 | 84.9 | 16.2 | 46.3 | 5.23× | 1.83× |
| `random_cjk_ineligible` | 0.229 | 118.2 | 22.1 | 86.3 | 5.36× | 1.37× |
| `session_snapshots` | 30.748 | 229.4 | 26.2 | 86.4 | 8.76× | 2.65× |
| `special_tokens` | 0.271 | 235.0 | 21.3 | – | 11.05× | – |

Encode was 5.22 to 11.40× faster than tiktoken-rs and 1.43 to 4.03× faster than bpe-openai, except on `long_runs`
(61.16× and 8.50×).

`r50k_base` and `p50k_base` count throughput in MiB/s, same run, against tiktoken-rs only:

| Workload | MiB | `r50k_base` | tiktoken-rs | speedup | `p50k_base` | tiktoken-rs | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixture_200k` | 0.895 | 191.9 | 22.4 | 8.56× | 160.7 | 21.4 | 7.52× |
| `source_1MiB` | 1.000 | 207.6 | 21.5 | 9.64× | 203.7 | 21.8 | 9.36× |
| `source_48B_chunks` | 0.250 | 177.5 | 18.2 | 9.77× | 179.3 | 18.5 | 9.71× |
| `source_256B_chunks` | 0.250 | 205.4 | 20.1 | 10.20× | 204.8 | 20.3 | 10.10× |
| `multilingual` | 0.468 | 215.7 | 28.1 | 7.69× | 213.1 | 28.2 | 7.55× |
| `scanner_edges` | 0.262 | 99.3 | 11.3 | 8.80× | 101.1 | 11.4 | 8.91× |
| `long_runs` | 1.678 | 1150.3 | 23.6 | 48.67× | 1194.1 | 16.1 | 74.02× |
| `long_unrepeated` | 0.680 | 82.3 | 20.5 | 4.01× | 80.0 | 20.6 | 3.89× |
| `random_cjk_ineligible` | 0.229 | 121.4 | 48.7 | 2.49× | 120.4 | 48.4 | 2.49× |
| `session_snapshots` | 30.748 | 196.9 | 22.9 | 8.62× | 162.9 | 21.8 | 7.48× |
| `special_tokens` | 0.270 | 197.3 | 21.3 | 9.25× | 194.3 | 21.2 | 9.16× |

Encode was 2.44 to 9.14× (`r50k_base`) and 2.46 to 9.17× (`p50k_base`) faster than tiktoken-rs, except on
`long_runs` (45.76× and 69.91×). `o200k_harmony` and `p50k_edit` special-token workloads ran within 6% of
`o200k_base` and `p50k_base`.

### Random CJK

Random CJK pieces never repeat a window; every window needs a fresh merge. Starting eligible characters whole
(see [Byte-pair merging](#byte-pair-merging-srcbpers)) roughly halved the cost.

<details>
<summary>CJK measurements before and after the character-merging changes</summary>

A separate probe checked token IDs against bpe-openai and measured milliseconds per call:

| Input | magi-tokenizer before | magi-tokenizer after | bpe-openai |
| --- | ---: | ---: | ---: |
| 240 KB random CJK from 8 characters | 4.2 | 2.1 | 2.4 |
| 240 KB random CJK from 400 characters | 3.8 | 3.2 | 4.3 |
| 240 KB random Hangul from 8 characters | 4.4 | 1.3 | 2.4 |
| 240 KB random CJK from `你世一人中大国我` (mostly not eligible) | 4.5 | 2.7 | 2.6 |
| 2,000 random 8-character CJK pieces, 129 to 1,024 bytes | 19.8 | 10.7 | 11.3 |

In the same probe, 200 KB of random `a`/`b` took 3.1 ms versus 4.1 ms for bpe-openai.
Random punctuation took about half bpe-openai's time. The change did not affect either result.
Without `toks` or the private assets, `bench` measured `multilingual` count at 137 → 178 MiB/s and
`long_unrepeated` at 58 → 69 MiB/s. Source and `long_runs` stayed within run-to-run variation.

The `random_cjk_ineligible` workload covers the worst case in `bench`: 240 KB of random `些介传你供像力参`.
Before the runtime last-byte check, none could start whole in either encoding.
This comparison uses the same benchmark before and after the change, with this repository's source,
no private assets, on ARM64 macOS. Units are MiB/s:

| Encoding | before | after | bpe-openai | after vs bpe-openai |
| --- | ---: | ---: | ---: | ---: |
| `o200k_base` | 59.9 | 240.5 | 89.8 | 2.68× |
| `cl100k_base` | 55.8 | 115.7 | 86.1 | 1.34× |

`cl100k_base` gains less. For 7 of the 8 characters, a token joining the last byte to one of the 6 CJK lead bytes
ranks below the character. Before that lead byte, the character must still start as separate bytes.
`参` stays ineligible since `参数` ranks below it.
For `o200k_base`, `long_unrepeated` includes random CJK and rose from 68.6 to 76.0 MiB/s.
Other workloads stayed within run-to-run variation.

</details>

### Allow-lists and long pieces

`encode`, `count`, `encode_allowing_special` and `count_allowing_special` look up the allow-list once per call,
when they meet the first special-token text, instead of scanning it for every candidate. `encode` and `count`
reuse that lookup for their disallowed-token check. Long pieces reuse window buffers across one call.

<details>
<summary>Measurements before and after</summary>

A separate probe, not in this repository, measured the best of 30 calls on ARM64 macOS (release, fat LTO).
Token counts were identical before and after.

| Input and call | before | after |
| --- | ---: | ---: |
| 1.2 MB special-dense `o200k_harmony` text, `count` with all 1,090 names allowed | 55.9 ms | 8.2 ms |
| Same text, `count_allowing_special` with all names | 28.2 ms | 5.0 ms |
| Same text, `count_with_special_tokens` (no allow-list) | 4.2 ms | 4.9 ms |
| 4,000 `o200k_base` pieces of 129 to 192 bytes, `count_ordinary` allocations | 31,510 | 92 |
| Same pieces, `count_ordinary` time | 12.1 ms | 11.6 ms |
| This repository's `src/`, `encode_ordinary` allocations | 3 | 2 |

The source text measured 0.28 tokens per byte, above the old reservation of one token per 4 bytes, so the
token list grew once during encoding. `fixture_200k` and `session_snapshots` stayed within run-to-run variation
for every encoding: they are ordinary text with no allow-list and few pieces over 128 bytes.

</details>

### Startup

The first count took 0.014 ms. tiktoken-rs took 37 ms and bpe-openai took 23 ms to load their vocabularies.
These are medians from three fresh processes per library, excluding process launch.
Cold caches and page faults still make the first call slower than later calls.

Counting random `a`/`b` pieces of 16, 64 and 128 MiB peaked at 29, 77 and 141 MiB RSS (`/usr/bin/time -l`).
That leaves 13 MiB above the input for table pages and buffers. Encoding 64 MiB peaked at 164 MiB,
including 87 MiB for 22.9 million IDs.

The [memory command](#how-it-works) reproduced the 16 and 64 MiB figures.
The 128 MiB figure came from an earlier probe that is not in this repository.
Whole-text memory and other tokenizers' memory have not been measured.

### Reproducing

```sh
MAGI_TOKENIZER_SOURCE=/path/to/toks MAGI_TOKENIZER_ASSETS=/path/to/assets \
  cargo run --release -p magi-tokenizer-bench -- bench          # optional filter, e.g. `bench long`
cargo run --release -p magi-tokenizer-bench -- startup magi     # or tiktoken, bpe-openai; one fresh process each
/usr/bin/time -l target/release/magi-tokenizer-bench memory 64 count window   # Linux: /usr/bin/time -v
```

- `fixture_200k` and `session_snapshots` need `MAGI_TOKENIZER_ASSETS` (see [Development](#development)).
  Without it, both are skipped. The session transcripts are private and unpublished; others cannot reproduce that row.
- Source workloads read the first 1 MiB (`source_1MiB`) or 256 KiB (chunked workloads) from `MAGI_TOKENIZER_SOURCE`.
- The first CJK table uses a probe that is not in this repository. `random_cjk_ineligible` is included.
- The benchmark enables `harness`. Disabling it left count throughput within run-to-run variation in a quick
  check on source text, long runs and random `a`/`b`.

### Caveats

These are results from one ARM64 Apple M5 Pro under background load, not a cross-platform speed guarantee.
Rotating jobs limits drift, but repeated runs still vary by about 10%. The tables record one run on one commit.
Rerun `bench` after scanner or merge changes. The native `toks` C library has not been benchmarked.

## Repository layout

```text
assets/o200k_base.tiktoken   o200k_base vocabulary (base64 token, rank per line), 3.6 MB
assets/cl100k_base.tiktoken  cl100k_base vocabulary, 1.7 MB (tables built only with feature `cl100k_base`)
assets/r50k_base.tiktoken    r50k_base vocabulary, 0.8 MB (tables built only with feature `r50k_base`)
assets/p50k_base.tiktoken    p50k_base vocabulary, 0.8 MB (tables built only with feature `p50k_base`)
build.rs                     parses each enabled vocabulary, emits binary tables + bindings into OUT_DIR
src/lib.rs                   public API: one module per encoding with pieces, encode/count, decode
src/scanner.rs               hand-written o200k, cl100k and r50k/p50k pretokenizers
src/unicode_classes.rs       generated per-code-point class table (do not edit by hand)
src/bpe.rs                   short-piece, window and whole-piece byte-pair merging
src/tables.rs                table lookups: vocabulary, pair ranks, byte ranks, token bytes
src/table_hash.rs            hash functions and table geometry shared with build.rs
bench/                       magi-tokenizer-bench: throughput, startup and memory benchmarks
tools/gen-unicode/           standalone generator for src/unicode_classes.rs (pinned regex versions)
```

## Development

CI checks library builds with Rust 1.88.0 and 1.90.0. Older versions are untested.
The benchmark needs newer standard-library APIs and uses stable in CI. Development used Rust 1.98.1.

```sh
# Library
cargo build --release

# Exact-ID checks, then throughput against tiktoken-rs and bpe-openai; an optional filter matches encoding/workload names
cargo run --release -p magi-tokenizer-bench -- bench
cargo run --release -p magi-tokenizer-bench -- bench cl100k/long

# First-call time in a fresh process
cargo run --release -p magi-tokenizer-bench -- startup magi        # or tiktoken, bpe-openai

# Regenerate the Unicode class table (only when the pinned regex versions change)
cd tools/gen-unicode && cargo run --release
```

The benchmark accepts two optional environment variables:

- `MAGI_TOKENIZER_SOURCE`: source trees in platform path-list syntax. Defaults to this repository.
  It skips dot-directories and `target`, reading at most about 4 MiB per tree.
- `MAGI_TOKENIZER_ASSETS`: a directory containing `o200k_base-200k.txt` and `session-compacted-500.json`.
  If unset, the benchmark skips those suites and prints a notice to stderr.
  If set, a missing or malformed file fails the command.

The release profile uses fat LTO and one codegen unit. Build scripts use `opt-level = 3` to keep debug table generation fast.

### Releasing

CI builds and runs clippy. It does not run exact-ID checks, and it cannot read the private files used by
`fixture_200k` and `session_snapshots`. Before every `cargo publish`, run the full benchmark on the release commit,
with both inputs set and no filter:

```sh
MAGI_TOKENIZER_ASSETS=/path/to/assets cargo run --release -p magi-tokenizer-bench -- bench
```

Publish only after the command finishes with no failed checks. Confirm that the output includes
`fixture_200k` and `session_snapshots` rows for every encoding.
Record the release commit and the successful run in that version's `CHANGELOG.md` entry.

## Known limitations

- Random CJK pieces made mostly of characters that cannot start whole, such as `你` and `世`, still merge from
  bytes. Performance is about level with bpe-openai, not faster.
- Without a stitch point within 4,096 bytes, whole-piece fallback needs 20 bytes of working memory per piece
  byte and takes about 10 times as long as windows (see [fallback costs](#how-it-works)).
  There is no known input condition that rules it out.
- Your application must choose the encoding. There is no model-name router
  ([#9](https://github.com/magimetal/magi-tokenizer/issues/9)).
- Encoding accepts `&str`, not raw bytes ([#11](https://github.com/magimetal/magi-tokenizer/issues/11)).
- The `usize` index path has not been exercised on a real piece of 4 GiB or more.
- Speed and memory measurements cover ARM64 macOS only. CI does not compare correctness; run `bench` locally.

## Provenance and licensing

The code uses the [Apache License 2.0](LICENSE). The crate bundles MIT-licensed tiktoken vocabularies and
Unicode-derived tables, so its license expression is `Apache-2.0 AND MIT AND Unicode-3.0`.
[NOTICE](NOTICE) includes the attributions, full vocabulary MIT notice and full Unicode License v3 text.

| Component | Source | Licence |
| --- | --- | --- |
| `assets/o200k_base.tiktoken`, `assets/cl100k_base.tiktoken`, `assets/r50k_base.tiktoken`, `assets/p50k_base.tiktoken` | OpenAI [tiktoken](https://github.com/openai/tiktoken) public vocabularies | MIT (tiktoken) |
| Scanner design (`src/scanner.rs`) | [toks](https://github.com/actualcomputer/toks) revision `2ebc390`, `src/core/k3_o200k_c.c` and `kernels.h`, Copyright 2026 Actual Computer Inc., rewritten in Rust | Apache-2.0 |
| Window stitching property | [`bpe` crate](https://github.com/github/rust-gems/tree/main/crates/bpe) README, "Properties of BPE". Only the property is used; no code from the crate is included. | MIT |
| Unicode classes | Generated from `regex` 1.12.4 / `regex-syntax` 0.8.11 (Unicode 16 data) | MIT OR Apache-2.0; Unicode License v3 |
| Everything else | magi-tokenizer | Apache-2.0 |
