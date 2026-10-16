# magi-code pilot: handoff

This document is for agents who work in magi-code (`~/Dev/magi-code`). It tells you how to pilot magi-tokenizer as
the `o200k_base` counter, which code to change, which behaviour must stay the same, and how to report problems.
The tracking issue is [#7](https://github.com/magimetal/magi-tokenizer/issues/7).

The author first read magi-code at commit `6f56b655` ("feat(sessions): persist unique short event IDs") and later
checked the routing at `766c6075` (v0.97.6). The author did not change magi-code. Check paths and line numbers again
before you edit, because they can move.

## Contents

- [Goal and scope](#goal-and-scope)
- [What is known about the tokenizer](#what-is-known-about-the-tokenizer)
- [Current magi-code behaviour](#current-magi-code-behaviour)
- [Invariants](#invariants)
- [Plan](#plan)
- [Dependency setup](#dependency-setup)
- [Disagreements and issue reports](#disagreements-and-issue-reports)
- [Rollback](#rollback)
- [Open items](#open-items)

## Goal and scope

Replace bpe-openai with magi-tokenizer for `o200k_base` counting only. Do this only after a shadow phase shows that
both libraries give the same counts in real use.

In scope:

- Text that `src/context/tokens.rs` counts through the `o200k_base` route.

Out of scope. Do not change these in the pilot:

- Model-name routing. magi-tokenizer has one encoding and no router. The app continues to decide which encoding a
  model uses.
- `cl100k_base`. magi-code routes `gpt-4`, `gpt-3.5-turbo` and embedding models to `cl100k_base`. Those routes stay
  on bpe-openai, so bpe-openai stays a dependency.
- Models that use the chars/4 fallback estimate today. Since v0.97.6, `gpt-6-*` and `gpt-6.1-*` route to
  `o200k_base` in `token_encoding_family_for_model`, so they are in scope through that route.
- Other providers. Only `OPENAI_CODEX_PROVIDER` gets a tokenizer today.

## What is known about the tokenizer

These results are from ARM64 macOS with Rust 1.98.1, except where noted:

- The `verify` suite found zero mismatches against tiktoken-rs. It compares token IDs for `encode_ordinary` and
  `encode_with_special_tokens`, and counts. Inputs: edge cases, all Unicode scalar classes, 200,000 fuzz texts,
  source code in 7, 48 and 4,096-byte chunks, multilingual text, long runs, long random and unrepeated pieces, a
  shared fixture and 20 session snapshots.
- Each of the 198,424 valid UTF-8 ordinary tokens encodes to its own ID.
- Counting is about 3× faster than bpe-openai on source code and session snapshots. This was measured on one
  machine under background load, with about ±10% variation.
- There is no vocabulary load at first use, because the build creates the tables at compile time.
- The library builds with Rust 1.88.0 and 1.90.0 and has no runtime dependencies.
- CI runs `verify` (without the private fixture and snapshots) on x86_64 and ARM64 Linux, ARM64 macOS, x86_64 and
  ARM64 Windows, i686 Linux and big-endian s390x Linux under QEMU: zero mismatches on each.

These items are not established yet. Each one has an issue:

| Gap | Issue |
| --- | --- |
| Long untrusted input could reach the whole-piece heap fallback: in measured runs 26–27 bytes of peak memory per piece byte (input included) and 11–20 times slower than windows, not a proven bound. `verify` now forces and checks this path; no verified text reaches it naturally, but no input is known to be unable to reach it. | [#1](https://github.com/magimetal/magi-tokenizer/issues/1) |
| NOTICE does not contain the Unicode license text. | [#5](https://github.com/magimetal/magi-tokenizer/issues/5) |
| Long random CJK pieces are slower than with bpe-openai. | [#6](https://github.com/magimetal/magi-tokenizer/issues/6) |
| Effect on magi-code build time and binary size is not measured. | [#7](https://github.com/magimetal/magi-tokenizer/issues/7) |

Because of #1:

- Do not use the active phase on a platform that is not in the CI `verify` matrix.
- Keep the `off` mode until the gaps that affect your platforms are closed.

## Current magi-code behaviour

All tokenizer counting is in `src/context/tokens.rs`:

| Item | Line (approx.) | Role |
| --- | --- | --- |
| `use bpe_openai::{Tokenizer, cl100k_base, o200k_base}` | 7 | Baseline library |
| `project_provider_request_input_tokens` | 109 | Counts conversation items, JSON tool definitions and the Claude CLI prompt |
| `project_provider_conversation_item_tokens` | 148 | Returns `TokenizerEstimate` or `FallbackEstimate` |
| `project_text_tokens` | 167 | Returns `TokenizerProjection` or `FallbackProjection` |
| `tokenizer_for_provider_model` | 186 | Checks the provider, then maps the family to a `&'static Tokenizer` |
| `token_encoding_family_for_model` | 196 | Lower-cases the model name and matches prefixes to `O200KBase` or `Cl100KBase` |
| `count_supported_provider_conversation_item_tokens` | 223 | Offline measurement. Returns `None` instead of a fallback. |
| `tokenizer_count_text_tokens` | 241 | The only place that counts text: `bpe.count(text)` |
| `TokenCounter`, `BpeCounter`, `FallbackCounter` | 246-274 | Role counting is on for the tokenizer and off for the fallback |
| `count_json_value_tokens` | 276 | Counts keys and values recursively, with `.max(1)` and scalar rules |

Callers outside this file:

- `src/agent/request.rs`
- `src/agent/provider_stream/context_projection.rs` (streaming delta projections, called often)
- `src/tui/controller/app/system_prompt.rs`
- `src/tool_output_measurement.rs` (has its own supported-model check and family lookup)

`src/context/AGENTS.md` requires counting through `Tokenizer::count`, which includes pre-tokenization. It forbids
`.bpe.count`.

## Invariants

The pilot must not change any of these:

1. Ordinary text. Count with `magi_tokenizer::count_ordinary`. Never use `encode_with_special_tokens`. Text such as
   `<|endoftext|>` in user content, files or tool output must count as literal text, the same as with bpe-openai
   `count`.
2. Pre-tokenization. `count_ordinary` runs the `o200k_base` pre-tokenizer itself. The library has no raw BPE entry
   point that can skip it.
3. Routing and the provider check. Keep the same models, providers and fallbacks. Change only the `O200KBase` arm.
4. Provenance. Keep the same `ContextUsageSource` values. A magi count is `TokenizerEstimate` or
   `TokenizerProjection`, the same as a bpe-openai count today.
5. Accounting. Keep role counting, fixed overheads, JSON traversal through `count_json_value_tokens` (not serialized
   JSON), the `.max(1)` rules and `saturating_add`.
6. Offline measurement fails closed. `count_supported_provider_conversation_item_tokens` must return `None` for
   unsupported models in every mode.
7. `cl100k_base` behaviour does not change.

## Plan

### Change point

Change only the counting step for `o200k_base`. One way: let `tokenizer_for_provider_model` return the encoding
family together with the bpe-openai tokenizer, keep both in `BpeCounter`, and choose the library in
`tokenizer_count_text_tokens`:

```rust
fn tokenizer_count_text_tokens(counter: &SelectedTokenizer, text: &str) -> usize {
    match (counter.family, tokenizer_pilot_mode()) {
        (TokenEncodingFamily::O200KBase, TokenizerPilotMode::Active) => magi_tokenizer::count_ordinary(text),
        (TokenEncodingFamily::O200KBase, TokenizerPilotMode::Shadow) => {
            let baseline = counter.bpe.count(text);
            record_shadow_comparison(baseline, magi_tokenizer::count_ordinary(text), text);
            baseline
        }
        _ => counter.bpe.count(text),
    }
}
```

The names are suggestions. If possible, keep the change inside `src/context/tokens.rs` so that all callers get it.

### Mode

The modes are `off`, `shadow` and `active`. Read the mode once, for example into a `OnceLock`, from an environment
variable such as `MAGI_TOKENIZER_PILOT`. Existing flags such as `MAGI_TUI_PERF` are read the same way. Treat an
unknown value as `off`.

| Phase | Default mode | Exit condition |
| --- | --- | --- |
| 1. Dependency | `off` | magi-code builds. `cargo deny check` passes. Build time and binary size are recorded in #7. |
| 2. Shadow | `shadow` for pilot users | Zero disagreements over an agreed amount of real use. No panics, hangs or memory spikes. |
| 3. Active | `active` for `o200k_base` | `off` still works. `cl100k_base` still uses bpe-openai. |

Shadow mode counts `o200k_base` text twice. Streaming projections call the counter often, so check UI latency
during phase 2.

### Shadow comparison

- Keep counters only: comparisons, disagreements and bytes compared. Atomic integers are enough.
- Never log or upload the text.
- Report the counters through an existing output if one fits, for example the `MAGI_TUI_PERF` output in
  `src/tui/perf.rs`. Do not add a logging dependency for this.
- Optional, off by default: if a separate variable such as `MAGI_TOKENIZER_PILOT_DUMP_DIR` is set, write each
  disagreeing text to a file in that directory. The file stays on the user's machine. You can then check it with the
  harness command below.

### Verification in magi-code

Follow magi-code's conventions. At minimum:

1. Build, run clippy and run `cargo deny check`.
2. Run each mode by hand.
3. On a saved session, confirm that `off` gives the same counts as the code before the change.

## Dependency setup

```toml
# magi-code Cargo.toml
magi-tokenizer = { git = "https://github.com/magimetal/magi-tokenizer.git", rev = "c60b63e01f9f88571657126f9854298645badee6" }
```

- Pin a full commit hash, not a branch. Commit `Cargo.lock`. The hash above is the first pushed commit; the
  library code in it is the code that `verify` checked. Use a later commit only if its `verify` run also passed.
- `deny.toml` has `unknown-git = "deny"` and `allow-git = []`. Add this repository:
  `allow-git = ["https://github.com/magimetal/magi-tokenizer"]`.
- The license is Apache-2.0, which `deny.toml` already allows. NOTICE includes the OpenAI MIT notice for the
  vocabulary. The Unicode license text is pending (#5).
- magi-code uses Rust 1.90.0. The library builds on 1.88.0, 1.90.0 and 1.98.1; CI builds it on 1.88.0 and 1.90.0 (#4).
- Dependents do not inherit magi-tokenizer's release profile (fat LTO, optimized build script). Measure debug and
  release build times in magi-code. If the build script is slow in debug builds, add
  `[profile.dev.build-override] opt-level = 3` to magi-code or accept the time.
- The crate embeds about 3.6 MiB of vocabulary and tables. A standalone release probe with magi-tokenizer was smaller
  than the same probe with bpe-openai. magi-code keeps bpe-openai for `cl100k_base`, so its binary will grow. Measure
  the growth.

## Disagreements and issue reports

The repository is public. Do not put prompts, files, tool output, session content, local paths or secrets in issues.

1. Save the disagreeing text locally, for example with the dump directory above.
2. In this repository, run:

   ```sh
   cargo run --release -p magi-tokenizer-bench -- check path/to/text.txt
   ```

   `check` compares magi-tokenizer with tiktoken-rs: pieces, IDs, counts, decode and special-token handling. It also
   compares the bpe-openai count with tiktoken-rs. tiktoken-rs is the reference.

   | Output | Meaning |
   | --- | --- |
   | `matches tiktoken and bpe-openai` | This text has no disagreement. Check that the dumped file is the text that the shadow comparison saw. |
   | `MISMATCH vs tiktoken: ...` | magi-tokenizer bug. Report it. |
   | `matches tiktoken; bpe-openai DIFFERS` | The baseline differs, not magi. Add a note to #7. The active phase will change counts for this kind of text. |

3. Make the text small. Remove parts while `check` still fails.
4. Replace private content with invented text that still fails. Scripts, punctuation, whitespace and lengths can
   all affect the result, so keep them.
5. Open an issue with the `bug` label. Include:
   - the minimized text, or a short program that generates it;
   - the full `check` output line;
   - the magi-tokenizer commit, the platform and the Rust version;
   - the mode in which the disagreement occurred.

## Rollback

- At run time, set the mode to `off`. The bpe-openai path is the code that runs today.
- In code, revert the pilot commit. Routing, provenance and accounting do not change, so nothing else needs
  migration.

## Open items

These are app decisions for magi-code, not for this repository:

- `cl100k_base` routes remain. To remove bpe-openai, first drop or replace those routes.
- Release panic strategy. A panic during counting can end the process. `catch_unwind` does not cover aborts, OOM or
  hangs. Consider this together with #3 before the active phase.

Tokenizer issues: [#1](https://github.com/magimetal/magi-tokenizer/issues/1) to
[#6](https://github.com/magimetal/magi-tokenizer/issues/6).
