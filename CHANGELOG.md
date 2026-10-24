# Changelog

Notable changes to magi-tokenizer. Versions are published on [crates.io](https://crates.io/crates/magi-tokenizer).
0.0.x releases have no compatibility promise: pin an exact version. From 0.1.0, releases follow the README's
[compatibility policy](README.md#compatibility), and breaking changes are listed under "Breaking".

## [Unreleased]

### Changed

- README documents the compatibility policy ([#13](https://github.com/magimetal/magi-tokenizer/issues/13)),
  starting with 0.1.0. Within a minor version, token IDs stay equal to tiktoken-rs 0.12.0 with Unicode 16 data,
  public items of the crate root, `o200k_base` and `cl100k_base` and the `cl100k_base` feature are kept, and the
  minimum Rust version stays 1.88. Breaking any of these needs a new minor version; reference mismatches are fixed
  as bugs in patch releases. `harness`, speed, memory, binary size and build time are not covered. GitHub releases
  repeat the changelog section.
- The magi-code pilot plan moved to `docs/archive/magi-code-pilot.md` and is no longer maintained. Remaining gaps are
  tracked in issues #8–#17. The README published with 0.0.2 still links to the old path.
- Added this changelog, backfilled from git history.
- Whole-piece fallback has a fixed memory bound ([#12](https://github.com/magimetal/magi-tokenizer/issues/12)): a
  tournament tree replaces the binary heap, so the merge allocates exactly 20 bytes of working memory per piece byte
  (32 at 4 GiB and above), once, in O(n log n) time. Peak RSS on a forced 64 MiB fallback fell from 1,750 MB to
  1,414 MB and time from 13.6 s to 11.8 s; window timings are unchanged. Token IDs are unchanged: `bench` exact-ID
  checks pass for both encodings, including `forced_fallback`. There is still no memory-limit setting.

- Short-piece speedups for both encodings ([#8](https://github.com/magimetal/magi-tokenizer/issues/8)): one- and
  two-byte pieces skip the vocabulary hash table, merges of pieces under 32 bytes shift parts with a loop instead
  of memmove, and encoding loops scan byte slices. Interleaved runs against `824a427` on one ARM64 macOS machine
  measured +0.5% to +3.2% throughput; token IDs are unchanged.
- README records the decision that model-name routing stays with applications
  ([#9](https://github.com/magimetal/magi-tokenizer/issues/9)), with reasons and guidance for unknown model names.
  The Scope table now lists the full upstream tiktoken mapping, including `gpt-oss-*` → `o200k_harmony`.
- README records the decision that encoding takes `&str` only ([#11](https://github.com/magimetal/magi-tokenizer/issues/11)):
  no reference defines tokens for invalid UTF-8 in a public API. Callers convert bytes with `String::from_utf8_lossy`,
  which replaces invalid sequences as Python's `errors="replace"` does.
- More CJK characters start merging as one part ([#14](https://github.com/magimetal/magi-tokenizer/issues/14)):
  tokens that join only a character's last byte to the next character are checked at runtime against the actual
  next byte, with a new 4,096-entry table per encoding, instead of at build time against every possible neighbour.
  Eligible single-token CJK characters (U+4E00–U+9FFF) rose from 1,037 to 2,493 of 2,515 in `o200k_base` and from
  432 to 546 of 549 in `cl100k_base`. New `bench` workload `random_cjk_ineligible` (240 KB of random CJK, none
  eligible before) went from 59.9 to 240.5 MiB/s count in `o200k_base` (2.68× bpe-openai, was 0.65×) and from 55.8
  to 115.7 MiB/s in `cl100k_base` (1.34×, was 0.64×). `o200k_base` `long_unrepeated`, which includes random CJK, rose
  from 68.6 to 76.0 MiB/s; other workloads stayed within run-to-run variation. Token IDs
  are unchanged: `bench` exact-ID checks pass for both encodings, including `unicode_sweep` and `forced_fallback`.

### Added

- Special-token allow-lists and error mode for both encodings
  ([#10](https://github.com/magimetal/magi-tokenizer/issues/10)): `encode_allowing_special` and
  `count_allowing_special` match tiktoken-rs 0.12.0 `CoreBPE::encode(text, allowed_special)`. `encode` and `count`
  follow Python tiktoken's default `encode()`: special-token text not in the allow-list returns
  `Err(DisallowedSpecialToken)` with the leftmost token and its byte offset. `bench` checks both against the
  references before timing.
- `cl100k_base` encoding behind the `cl100k_base` feature ([#8](https://github.com/magimetal/magi-tokenizer/issues/8)):
  module `magi_tokenizer::cl100k_base` with the same functions and constants as `o200k_base`, a second hand-written
  scanner for its regex, and its own build-time tables. Programs that call it grow by about 10 MB (fat-LTO release,
  ARM64 macOS: 20.1 → 30.2 MB); enabling the feature without calling it does not change size. Clean release builds
  of a one-call program took 3.6 s without the feature and 4.5 s with it.
- Module `magi_tokenizer::o200k_base`; the crate root still re-exports it, so existing code is unchanged.
- `bench` checks exact token IDs, counts, special tokens and decoding against tiktoken-rs for both encodings before
  timing, including a Unicode sweep, scanner edge cases and forced whole-piece fallback.
- README documents that `o200k_harmony` ordinary encoding matches `o200k_base` (special tokens differ, #18), and
  that legacy 50k encodings are not implemented (#19).

### Removed

- Differential verification and file-check commands, their test-only hooks and direct dependencies, and CI test runs.
  Throughput, startup and memory benchmarks remain; CI retains build and clippy coverage.

## [0.0.2] - 2026-10-23

### Changed

- Faster merging for long pieces of 2- and 3-byte characters ([#6](https://github.com/magimetal/magi-tokenizer/issues/6)).
  Merging starts an eligible character as one part instead of separate bytes. `build.rs` marks a character eligible
  only when the result is identical, so token IDs do not change. Adds a 256 KiB embedded table.
  Probe, ms per 240 KB call (0.0.1 → 0.0.2, bpe-openai): random CJK from 8 characters 4.2 → 2.1 (2.4); random
  Hangul 4.4 → 1.3 (2.4); random CJK from mostly ineligible characters 4.5 → 2.7 (2.6).
  `bench`: `multilingual` count 137 → 178 MiB/s, `long_unrepeated` 58 → 69 MiB/s.
- README: random CJK section and known limitations updated to match.

## [0.0.1] - 2026-10-17

First release. Published from commit `23c0680` with the version set to 0.0.1 locally; the repository still said
0.1.0 until 0.0.2.

### Added

- Exact `o200k_base` tokenizer, matching tiktoken's `encode_ordinary` and `encode(text, allowed_special="all")`:
  `encode_ordinary`, `encode_ordinary_into`, `encode_with_special_tokens`, `count_ordinary`,
  `count_with_special_tokens`, `decode`, `decode_bytes`, `token_bytes`, `pieces`, and the `END_OF_TEXT`,
  `END_OF_PROMPT`, `SPECIAL_TOKENS` and `ORDINARY_TOKEN_COUNT` constants.
- Vocabulary and merge tables built at compile time by `build.rs` and embedded in the target's byte order: no
  vocabulary loading at run time.
- Hand-written pretokenizer for the o200k regex, with Unicode 16 classes generated by `tools/gen-unicode`.
- Short pieces merged in fixed stack arrays without allocation. Long pieces merged in overlapping, cached windows,
  with bounded memory when counting, and an exact whole-piece heap fallback.
- `magi-tokenizer-bench`: differential `verify` against tiktoken-rs, `bench` against tiktoken-rs and bpe-openai,
  `startup`, `memory` and `check` commands.
- Fallback forced and measured in `verify`; differential tests of each merge path
  ([#1](https://github.com/magimetal/magi-tokenizer/issues/1), [#2](https://github.com/magimetal/magi-tokenizer/issues/2)).
- Panic audit: compile-time index invariants and a documented panic guarantee
  ([#3](https://github.com/magimetal/magi-tokenizer/issues/3)).
- CI: builds on Rust 1.88 (MSRV) and 1.90; clippy and `verify` on x86_64 and ARM64 Linux, ARM64 macOS, x86_64 and
  ARM64 Windows, i686 Linux and big-endian s390x Linux under QEMU ([#4](https://github.com/magimetal/magi-tokenizer/issues/4)).
- Licensing: license expression `Apache-2.0 AND MIT AND Unicode-3.0`, full Unicode License v3 in `NOTICE`
  ([#5](https://github.com/magimetal/magi-tokenizer/issues/5)).

[Unreleased]: https://github.com/magimetal/magi-tokenizer/compare/v0.0.2...HEAD
[0.0.2]: https://github.com/magimetal/magi-tokenizer/compare/23c0680...v0.0.2
[0.0.1]: https://github.com/magimetal/magi-tokenizer/tree/23c0680
