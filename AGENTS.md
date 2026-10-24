<!-- Generated repository guidance; keep aligned with source and build configuration. -->
# Repository guidance

## Scope and purpose
- Applies throughout this repository unless a nearer `AGENTS.md` overrides it.
- `magi-tokenizer` is a dependency-free Rust implementation of the `o200k_base` tokenizer.
- Library crate: `magi_tokenizer`; Rust edition 2024, minimum Rust 1.88.
- Support only `o200k_base`; do not imply model routing or other vocabularies.

## Source map
- `src/lib.rs`: public encoding, counting, decoding, and piece APIs.
- `src/scanner.rs`: handwritten scanner matching the reference regex behavior.
- `src/bpe.rs`: byte-pair merging, windowed encoding, and whole-piece fallback.
- `src/tables.rs` and `src/table_hash.rs`: runtime table access and shared hashing.
- `src/unicode_classes.rs`: generated Unicode classification tables.
- `src/harness.rs`: unstable memory-benchmark instrumentation behind feature `harness`.
- `build.rs`: vocabulary processing and build-time table generation.
- `assets/o200k_base.tiktoken`: source vocabulary.
- `bench/`: isolated benchmark dependencies; workspace member, not published.
- `tools/gen-unicode/`: standalone generator, excluded from the root workspace.
- `.github/workflows/ci.yml`: authoritative CI configuration.
- `README.md`: architecture and performance; `CHANGELOG.md`: release history.
- `docs/archive/magi-code-pilot.md`: historical material, not current guidance.

## Useful commands
Run from repository root unless shown otherwise. Run only commands relevant to the change.

```sh
cargo build --release -p magi-tokenizer
cargo clippy --release --workspace -- -D warnings
cargo run --release -p magi-tokenizer-bench -- bench
cargo run --release -p magi-tokenizer-bench -- startup magi
cargo run --release -p magi-tokenizer-bench -- memory 64 count window
(cd tools/gen-unicode && cargo run --release)
```

- `bench` accepts substring filters, for example `bench long`.
- `memory` takes `<MiB> <count|encode> <window|fallback>`.
- Benchmark tooling uses newer stable APIs than the library's minimum Rust version.
- Run Unicode generation only for intentional Unicode-table changes.

## Public API boundaries
- Public text inputs are `&str`; internal BPE operations support arbitrary bytes.
- Ordinary encoding treats special-token spellings as ordinary text.
- Special-token APIs recognize `END_OF_TEXT` (199999) and `END_OF_PROMPT` (200018).
- `encode_ordinary_into` appends; preserve existing output.
- Invalid token IDs return `None`; `decode` also rejects invalid UTF-8.
- Keep counting independent of emitting token IDs; keep `harness` out of consumer API promises.

## Encoding invariants
- Preserve scanner alternative ordering, backtracking, Unicode behavior, and long-s U+017F case folding.
- BPE selects the lowest-rank adjacent pair with leftmost tie-breaking; rank equals token ID.
- Pieces up to 128 bytes use stack merging and whole-token lookup.
- Larger pieces use overlapping windows from 128 to 4096 bytes, then whole-piece heap fallback.
- Cache matches require full-byte equality, not hash equality alone.
- A failed window pass may have appended output: roll back only the current piece before fallback.
- Reset the current piece's count before fallback; preserve earlier pieces' output and counts.
- Keep premerged two-/three-byte character optimizations coupled to build/runtime neighbor guards.

## Generated data and portability
- Keep library and build script dependency-free; isolate benchmark dependencies in `bench/`.
- Edit the Unicode generator, not `src/unicode_classes.rs`; preserve scanner class-bit layout.
- Unicode 16 generation pins `regex` 1.12.4 and `regex-syntax` 0.8.11 to match `tiktoken-rs` 0.12.0.
- Generator writes `src/unicode_classes.rs` relative to its manifest, not the working directory.
- `build.rs` emits Rust/binary tables under `OUT_DIR`; do not hand-edit generated outputs.
- Keep build-time and runtime hash geometry synchronized.
- Preserve typed-blob safety assertions, 64-byte alignment, and `CARGO_CFG_TARGET_ENDIAN` handling.
- Preserve 32-bit and big-endian support; expected ordinary-token/pair counts are 199998/446189.
- Preserve vocabulary, scanner, and Unicode attribution in `NOTICE` and package license metadata.

## Benchmark interpretation
- `MAGI_TOKENIZER_SOURCE` is a platform-separated path list; default corpus comes from this repository.
- Corpus collection includes Markdown, so edits to this file affect default measurements; fix corpus inputs for comparisons.
- `MAGI_TOKENIZER_ASSETS` optionally supplies `o200k_base-200k.txt` and `session-compacted-500.json`.
- Unset assets skip those workloads; a configured directory with missing/malformed assets fails.
- Private fixtures are not committed; published ARM64 macOS results are not universally reproducible.
