// Benchmarks for magi-tokenizer against tiktoken-rs and bpe-openai, for o200k_base, o200k_harmony, cl100k_base,
// r50k_base, p50k_base and p50k_edit.
//
//   cargo run --release -p magi-tokenizer-bench -- bench [filter]  # throughput, all three tokenizers
//   cargo run --release -p magi-tokenizer-bench -- startup <magi|tiktoken|bpe-openai>  # one first-call timing
//   cargo run --release -p magi-tokenizer-bench -- memory <MiB> <count|encode> <window|fallback>  # one long piece;
//       run under `/usr/bin/time -l` (macOS) or `/usr/bin/time -v` (Linux) for peak RSS
//
// `bench` first checks that magi-tokenizer returns tiktoken's exact token ids, counts, special-token ids and
// decoded text for every workload; the filter matches encoding or workload names. o200k_harmony and p50k_edit share
// the ordinary encoding of o200k_base and p50k_base: they are checked against tiktoken-rs but not timed, except for
// the `special_tokens` workload, which times special-token encoding for every encoding. bpe-openai has no r50k_base
// or p50k_base, so those are timed against tiktoken-rs only.
//
// Inputs: MAGI_TOKENIZER_SOURCE lists source trees (path-list syntax, default: this repository) to cut into chunks.
// MAGI_TOKENIZER_ASSETS names a directory with o200k_base-200k.txt and session-compacted-500.json; without it those
// workloads are skipped.
//
// The bench enables magi-tokenizer's `harness` feature for fallback counters and forced fallback, and
// `cl100k_base`, `r50k_base` and `p50k_base` for the other encodings.

use std::collections::HashSet;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use std::{env, fs};

use magi_tokenizer::{DisallowedSpecialToken, harness};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("bench") => bench(args.get(1).map(String::as_str)),
        Some("startup") => startup(args.get(1).map(String::as_str).unwrap_or("magi")),
        Some("memory") if args.len() == 4 => measure_long_piece_memory(&args[1], &args[2], &args[3]),
        _ => Err("usage: magi-tokenizer-bench bench [filter] | \
                  startup <magi|tiktoken|bpe-openai> | memory <MiB> <count|encode> <window|fallback>"
            .into()),
    }
}

// ---- inputs ---------------------------------------------------------------------------------------------

fn chunks(text: &str, size: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let mut end = (start + size).min(text.len());
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        if end == start {
            end = start + text[start..].chars().next().unwrap().len_utf8();
        }
        out.push(text[start..end].to_owned());
        start = end;
    }
    out
}

fn collect_source_text(root: &Path, out: &mut String, limit: usize) {
    let Ok(entries) = fs::read_dir(root) else { return };
    let mut paths: Vec<_> = entries.flatten().map(|entry| entry.path()).collect();
    paths.sort();
    for path in paths {
        if out.len() >= limit {
            return;
        }
        let name = path.file_name().and_then(|name| name.to_str()).unwrap_or("");
        if path.is_dir() {
            if !name.starts_with('.') && name != "target" {
                collect_source_text(&path, out, limit);
            }
        } else if matches!(path.extension().and_then(|e| e.to_str()), Some("rs" | "c" | "h" | "md" | "py" | "toml" | "json"))
            && let Ok(text) = fs::read_to_string(&path)
        {
            out.push_str(&text);
            out.push('\n');
        }
    }
}

fn source_text() -> String {
    let roots: Vec<PathBuf> = match env::var_os("MAGI_TOKENIZER_SOURCE") {
        Some(roots) => env::split_paths(&roots).collect(),
        None => vec![Path::new(env!("CARGO_MANIFEST_DIR")).join("..")],
    };
    let mut text = String::new();
    for root in roots {
        let limit = text.len() + (4 << 20);
        collect_source_text(&root, &mut text, limit);
    }
    text
}

/// `Ok(None)` when MAGI_TOKENIZER_ASSETS is unset; an error when it is set but the asset is missing or malformed.
fn comparison_asset(name: &str) -> Result<Option<Vec<u8>>> {
    let Some(dir) = env::var_os("MAGI_TOKENIZER_ASSETS") else { return Ok(None) };
    let path = PathBuf::from(dir).join(name);
    Ok(Some(fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?))
}

fn fixture_200k() -> Result<Option<String>> {
    let Some(bytes) = comparison_asset("o200k_base-200k.txt")? else { return Ok(None) };
    Ok(Some(String::from_utf8(bytes)?))
}

fn session_snapshots() -> Result<Option<Vec<String>>> {
    let Some(bytes) = comparison_asset("session-compacted-500.json")? else { return Ok(None) };
    let fixture: serde_json::Value = serde_json::from_slice(&bytes)?;
    let snapshots = (|| {
        let segments = fixture["segments"].as_array()?;
        fixture["snapshots"]
            .as_array()?
            .iter()
            .map(|snapshot| {
                let segment = snapshot["segment"].as_u64()? as usize;
                let end = snapshot["end"].as_u64()? as usize;
                Some(segments.get(segment)?.as_str()?.get(..end)?.to_owned())
            })
            .collect::<Option<Vec<_>>>()
    })();
    Ok(Some(snapshots.ok_or("session-compacted-500.json: unexpected structure")?))
}

fn multilingual_text() -> String {
    let lines = [
        "你好世界，这是一个测试。日本語のテキストも含まれています。한국어 문장도 있습니다.",
        "Привет, мир! Это проверка токенизатора. Ελληνικά κείμενα και αριθμοί 12345.",
        "مرحبا بالعالم، هذا اختبار. नमस्ते दुनिया, यह एक परीक्षण है। สวัสดีชาวโลก",
        "Ünïcödé çàsés: ÀÉÎÕÜ àéîõü ßſ ǅǈǋ 'S 'ſ CAN'T WON'T we'LL they'RE e\u{301}\u{300}",
        "🙂👩🏽‍💻🚀 emoji ✨ mixed with code: fn main() { println!(\"{}\", x); } // ok\r\n\r\n",
    ];
    lines.iter().cycle().take(4000).fold(String::new(), |mut text, line| {
        text.push_str(line);
        text.push('\n');
        text
    })
}

/// Encodes or counts one long single piece (random letters `a`/`b`) through windows or the forced whole-piece
/// fallback, so peak RSS can be compared between the two.
fn measure_long_piece_memory(mebibytes: &str, operation: &str, path: &str) -> Result<()> {
    let len = mebibytes.parse::<usize>()?.checked_mul(1 << 20).ok_or("piece size overflows usize")?;
    let fail_after_stitches = match path {
        "window" => None,
        "fallback" => Some(0),
        other => return Err(format!("unknown path {other}").into()),
    };
    if !matches!(operation, "count" | "encode") {
        return Err(format!("unknown operation {operation}").into());
    }
    let mut state = 0x1234_5678_9ABC_DEF1u64;
    let text: String = (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if state & 1 == 0 { 'a' } else { 'b' }
        })
        .collect();
    harness::fail_window_merge_after_stitches(fail_after_stitches);
    harness::reset_fallback_counters();
    let start = Instant::now();
    let tokens = if operation == "count" {
        magi_tokenizer::count_ordinary(&text)
    } else {
        magi_tokenizer::encode_ordinary(&text).len()
    };
    println!(
        "{mebibytes} MiB piece, {operation} via {path}: {tokens} tokens, {} fallbacks, {:.2}s",
        harness::fallback_pieces(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn long_runs() -> Vec<String> {
    vec![
        "a".repeat(300_000),
        " ".repeat(300_000),
        "0123456789".repeat(30_000),
        "ab".repeat(100_000),
        "\u{301}".repeat(50_000),
        "!@#$%^&*()".repeat(20_000),
        "xyzzyplugh".repeat(10_000),
        "你好世界".repeat(20_000),
        (0..20_000).map(|i| char::from(b'a' + (i * 7 % 26) as u8)).collect(),
    ]
}
/// Random text of at least `len` bytes from `alphabet`, advancing the xorshift `state`.
fn random_text(state: &mut u64, len: usize, alphabet: &[&str]) -> String {
    let mut text = String::new();
    while text.len() < len {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        text.push_str(alphabet[(*state % alphabet.len() as u64) as usize]);
    }
    text
}

/// Pieces that defeat window reuse: long random text from small alphabets. Stitch points are found quickly,
/// but every window is new, so the cache never hits. These are the workloads where window merging costs the most.
fn long_unrepeated_pieces() -> Vec<String> {
    let mut state = 0x7777_1111_2222_3333u64;
    let mut texts = vec![
        random_text(&mut state, 200_000, &["a", "b"]),
        random_text(&mut state, 240_000, &["你", "好", "世", "界", "的", "是", "在", "了"]),
        random_text(&mut state, 200_000, &["!", "=", "-", "*", "(", ")", "{", "}", ";", "."]),
    ];
    for len in (129..1024).step_by(7) {
        texts.push(random_text(&mut state, len, &["a", "b"]));
    }
    texts
}

/// One long unrepeated piece of CJK characters that 0.0.2 started merging as separate bytes in both encodings
/// (#14), so every window merges them fresh.
fn random_cjk_ineligible() -> Vec<String> {
    let alphabet = ["些", "介", "传", "你", "供", "像", "力", "参"];
    vec![random_text(&mut 0x1414_1414_1414_1414u64, 240_000, &alphabet)]
}

// ---- benchmarks -----------------------------------------------------------------------------------------

/// Microseconds per call of each `work`, the median of five rounds. Within a round every `work` runs for at least
/// 150ms, in rotating order, so background load and frequency changes hit all tokenizers alike.
fn time_per_call(works: &[&dyn Fn() -> usize]) -> Vec<f64> {
    let mut runs = vec![Vec::new(); works.len()];
    for round in 0..5 {
        for offset in 0..works.len() {
            let index = (round + offset) % works.len();
            let start = Instant::now();
            let mut rounds = 0u32;
            while start.elapsed() < Duration::from_millis(150) {
                black_box(works[index]());
                rounds += 1;
            }
            runs[index].push(start.elapsed().as_secs_f64() * 1e6 / f64::from(rounds));
        }
    }
    runs.into_iter()
        .map(|mut times| {
            times.sort_by(f64::total_cmp);
            times[2]
        })
        .collect()
}

/// One encoding: its magi-tokenizer functions and the reference tokenizers (bpe-openai lacks some encodings).
struct Encoding {
    name: &'static str,
    tiktoken: &'static tiktoken_rs::CoreBPE,
    bpe_openai: Option<&'static bpe_openai::Tokenizer>,
    /// The encoding with the same ordinary encoding that times it, if not this one.
    ordinary_timed_by: Option<&'static str>,
    special_tokens: &'static [(&'static str, u32)],
    encode: fn(&str) -> Vec<u32>,
    count: fn(&str) -> usize,
    encode_with_special_tokens: fn(&str) -> Vec<u32>,
    count_with_special_tokens: fn(&str) -> usize,
    encode_allowing_special: fn(&str, &[&str]) -> Vec<u32>,
    count_allowing_special: fn(&str, &[&str]) -> usize,
    encode_rejecting_special: fn(&str, &[&str]) -> std::result::Result<Vec<u32>, DisallowedSpecialToken>,
    count_rejecting_special: fn(&str, &[&str]) -> std::result::Result<usize, DisallowedSpecialToken>,
    decode: fn(&[u32]) -> Option<String>,
}

/// The `Encoding` of magi-tokenizer module `$module`.
macro_rules! encoding {
    ($name:literal, $module:ident, $tiktoken:expr, $bpe_openai:expr, $ordinary_timed_by:expr) => {
        Encoding {
            name: $name,
            tiktoken: $tiktoken,
            bpe_openai: $bpe_openai,
            ordinary_timed_by: $ordinary_timed_by,
            special_tokens: &magi_tokenizer::$module::SPECIAL_TOKENS,
            encode: magi_tokenizer::$module::encode_ordinary,
            count: magi_tokenizer::$module::count_ordinary,
            encode_with_special_tokens: magi_tokenizer::$module::encode_with_special_tokens,
            count_with_special_tokens: magi_tokenizer::$module::count_with_special_tokens,
            encode_allowing_special: magi_tokenizer::$module::encode_allowing_special,
            count_allowing_special: magi_tokenizer::$module::count_allowing_special,
            encode_rejecting_special: magi_tokenizer::$module::encode,
            count_rejecting_special: magi_tokenizer::$module::count,
            decode: magi_tokenizer::$module::decode,
        }
    };
}

fn encodings() -> [Encoding; 6] {
    [
        encoding!("o200k", o200k_base, tiktoken_rs::o200k_base_singleton(), Some(bpe_openai::o200k_base()), None),
        encoding!("harmony", o200k_harmony, tiktoken_rs::o200k_harmony_singleton(), None, Some("o200k")),
        encoding!("cl100k", cl100k_base, tiktoken_rs::cl100k_base_singleton(), Some(bpe_openai::cl100k_base()), None),
        encoding!("r50k", r50k_base, tiktoken_rs::r50k_base_singleton(), None, None),
        encoding!("p50k", p50k_base, tiktoken_rs::p50k_base_singleton(), None, None),
        encoding!("p50k_edit", p50k_edit, tiktoken_rs::p50k_edit_singleton(), None, Some("p50k")),
    ]
}

/// Short random strings over atoms where the pretokenization alternatives meet: contractions, case, digit runs,
/// punctuation before newlines, and whitespace before text, newlines and the end.
fn scanner_edge_cases() -> Vec<String> {
    let atoms = [
        " ", "  ", "\t", "\n", "\r\n", "\r", "\u{a0}", "\u{2028}", "\u{3000}", "'", "'s", "'S", "'ll", "'LL", "'ve",
        "'Re", "'d", "'m", "'t", "'ſ", "a", "Ab", "CD", "é", "É", "ǅ", "\u{301}", "ß", "1", "234", "٣", "Ⅻ", "½", "!",
        "?!", ".", "...", "-", "<|", "|>", "你", "🙂", "\u{200d}", "\u{feff}",
    ];
    let mut state = 0x2468_ACE0_1357_9BDFu64;
    (0..20_000)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let len = 1 + (state % 12) as usize;
            (0..len).map(|i| atoms[((state >> (8 + 4 * i)) as usize ^ (i * 31)) % atoms.len()]).collect()
        })
        .collect()
}

/// Scalars of the first three planes and plane 14, each in contexts that select different pretokenization
/// alternatives, in texts of 256 scalars.
fn unicode_sweep() -> Vec<String> {
    let contexts = [("a", "b"), (" ", "1"), ("", "'s"), ("\n", " "), ("A", "a"), ("x ", "\r\n")];
    let scalars: Vec<char> = (0..0x3_0000).chain(0xE_0000..0xE_0200).filter_map(char::from_u32).collect();
    contexts
        .iter()
        .flat_map(|(before, after)| {
            scalars.chunks(256).map(move |chunk| chunk.iter().map(|c| format!("{before}{c}{after}")).collect())
        })
        .collect()
}

/// Python tiktoken's default `encode()` check: the leftmost special-token text not in `allowed_special`.
fn python_disallowed_special(
    encoding: &Encoding,
    text: &str,
    allowed_special: &[&str],
) -> Option<DisallowedSpecialToken> {
    text.match_indices("<|").find_map(|(position, _)| {
        let rest = &text[position..];
        let &(token, _) = encoding
            .special_tokens
            .iter()
            .find(|(name, _)| rest.starts_with(name) && !allowed_special.contains(name))?;
        Some(DisallowedSpecialToken { token, position })
    })
}

/// Allow-lists checked against tiktoken-rs `CoreBPE::encode`: none, each token alone and all but each token. Only
/// the first 16 tokens and the last are varied: they cover every named token, and o200k_harmony's 1,090 would
/// make thousands of lists.
fn allow_lists(encoding: &Encoding) -> Vec<Vec<&'static str>> {
    let names: Vec<&'static str> = encoding.special_tokens.iter().map(|(name, _)| *name).collect();
    let varied = names.iter().take(16).chain(names.last().filter(|_| names.len() > 16));
    let mut lists = vec![Vec::new()];
    for name in varied {
        lists.push(vec![*name]);
        lists.push(names.iter().copied().filter(|other| other != name).collect());
    }
    lists
}

/// Chunks of source text, each followed by a special token (striding through all of them) and the `<|` of a
/// special-token text that is not one, so special-token lookup runs often.
fn special_token_text(encoding: &Encoding, source: &str) -> String {
    let tokens = encoding.special_tokens;
    let mut text = String::new();
    for (index, chunk) in chunks(source, 256).iter().enumerate() {
        text.push_str(chunk);
        text.push_str(tokens[index * 7919 % tokens.len()].0);
        text.push_str("<|endof");
    }
    text
}

/// Asserts that magi-tokenizer gives tiktoken's exact ids for `texts`, ordinary and with special tokens, that counts
/// match and that decoding round-trips. Special tokens are spliced between texts so they are matched in context,
/// and allow-lists and Python tiktoken's disallowed-special error are checked on the spliced text.
fn check_exact(encoding: &Encoding, workload: &str, texts: &[String]) {
    let name = format!("{}/{workload}", encoding.name);
    for (index, text) in texts.iter().enumerate() {
        let expected = encoding.tiktoken.encode_ordinary(text);
        assert_eq!((encoding.encode)(text), expected, "{name}: ids differ for text {index}: {text:?}");
        assert_eq!((encoding.count)(text), expected.len(), "{name}: count differs for text {index}");
        assert_eq!((encoding.decode)(&expected).as_deref(), Some(text.as_str()), "{name}: decode differs");
        let rejected = python_disallowed_special(encoding, text, &[]).map_or(Ok(expected), Err);
        let rejected_count = rejected.as_ref().map(Vec::len).map_err(|error| *error);
        assert_eq!((encoding.count_rejecting_special)(text, &[]), rejected_count, "{name}: encode count differs");
        assert_eq!((encoding.encode_rejecting_special)(text, &[]), rejected, "{name}: encode differs for text {index}");
    }
    let joined = texts.iter().take(2000).enumerate().fold(String::new(), |mut joined, (index, text)| {
        joined.push_str(text);
        joined.push_str(encoding.special_tokens[index % encoding.special_tokens.len()].0);
        joined.push_str("<|endof");
        joined
    });
    let expected = encoding.tiktoken.encode_with_special_tokens(&joined);
    assert_eq!((encoding.encode_with_special_tokens)(&joined), expected, "{name}: special-token ids differ");
    assert_eq!((encoding.count_with_special_tokens)(&joined), expected.len(), "{name}: special-token count differs");
    assert_eq!((encoding.decode)(&expected).as_deref(), Some(joined.as_str()), "{name}: special-token decode differs");
    for allowed in allow_lists(encoding) {
        let expected = encoding.tiktoken.encode(&joined, &allowed.iter().copied().collect::<HashSet<_>>()).unwrap().0;
        let ids = (encoding.encode_allowing_special)(&joined, &allowed);
        assert_eq!(ids, expected, "{name}: ids differ allowing {allowed:?}");
        let count = (encoding.count_allowing_special)(&joined, &allowed);
        assert_eq!(count, expected.len(), "{name}: count differs allowing {allowed:?}");
        let rejected = python_disallowed_special(encoding, &joined, &allowed).map_or(Ok(expected), Err);
        let rejected_count = rejected.as_ref().map(Vec::len).map_err(|error| *error);
        let count = (encoding.count_rejecting_special)(&joined, &allowed);
        assert_eq!(count, rejected_count, "{name}: encode count differs allowing {allowed:?}");
        let ids = (encoding.encode_rejecting_special)(&joined, &allowed);
        assert_eq!(ids, rejected, "{name}: encode differs allowing {allowed:?}");
    }
}

fn bench(filter: Option<&str>) -> Result<()> {
    let source = source_text();
    let mut workloads: Vec<(&str, Vec<String>)> = vec![
        ("source_1MiB", vec![source[..source.floor_char_boundary(1 << 20)].to_owned()]),
        ("source_48B_chunks", chunks(&source[..source.floor_char_boundary(256 << 10)], 48)),
        ("source_256B_chunks", chunks(&source[..source.floor_char_boundary(256 << 10)], 256)),
        ("multilingual", vec![multilingual_text()]),
        ("scanner_edges", scanner_edge_cases()),
        ("long_runs", long_runs()),
        ("long_unrepeated", long_unrepeated_pieces()),
        ("random_cjk_ineligible", random_cjk_ineligible()),
    ];
    match (fixture_200k()?, session_snapshots()?) {
        (Some(fixture), Some(snapshots)) => {
            workloads.insert(0, ("fixture_200k", vec![fixture]));
            workloads.push(("session_snapshots", snapshots.into_iter().step_by(10).collect()));
        }
        _ => eprintln!(
            "MAGI_TOKENIZER_ASSETS unset: skipping fixture_200k and session_snapshots; a release needs a full run with them"
        ),
    }
    let selected = |name: &str| filter.is_none_or(|filter| name.contains(filter));
    // Checked but not timed: too slow for tiktoken to time, or measuring a forced path.
    let sweep = unicode_sweep();
    let long_pieces: Vec<String> = long_runs().into_iter().chain(long_unrepeated_pieces()).collect();
    for encoding in &encodings() {
        let name = format!("{}/unicode_sweep", encoding.name);
        if selected(&name) {
            check_exact(encoding, "unicode_sweep", &sweep);
            println!("{name}: exact ids for {} texts", sweep.len());
        }
        let name = format!("{}/forced_fallback", encoding.name);
        if selected(&name) {
            harness::fail_window_merge_after_stitches(Some(0));
            harness::reset_fallback_counters();
            check_exact(encoding, "forced_fallback", &long_pieces);
            harness::fail_window_merge_after_stitches(None);
            assert!(harness::fallback_pieces() > 0, "{name}: no piece fell back");
            println!("{name}: exact ids for {} texts, {} fallbacks", long_pieces.len(), harness::fallback_pieces());
        }
    }
    println!(
        "{:<27} {:>9} | {:>11} {:>11} {:>11} | {:>11} {:>11} {:>11} | speedup vs tiktoken / bpe-openai",
        "encoding/workload", "MiB", "magi cnt", "tiktok cnt", "bpeoai cnt", "magi enc", "tiktok enc", "bpeoai enc"
    );
    for encoding in &encodings() {
        for (workload, texts) in &workloads {
            let name = format!("{}/{workload}", encoding.name);
            if !selected(&name) {
                continue;
            }
            check_exact(encoding, workload, texts);
            if let Some(timed_by) = encoding.ordinary_timed_by {
                println!("{name:<27} exact ids for {} texts; same ordinary encoding as {timed_by}, not timed", texts.len());
                continue;
            }
            let bytes: usize = texts.iter().map(String::len).sum();
            let mib = bytes as f64 / (1 << 20) as f64;
            let throughput = |micros: f64| mib / (micros / 1e6);
            let sum_over = |f: &dyn Fn(&str) -> usize| texts.iter().map(|text| f(black_box(text))).sum::<usize>();
            let Some(bpe_openai) = encoding.bpe_openai else {
                let times = time_per_call(&[
                    &|| sum_over(&|text| (encoding.count)(text)),
                    &|| sum_over(&|text| encoding.tiktoken.encode_ordinary(text).len()),
                    &|| sum_over(&|text| (encoding.encode)(text).len()),
                ]);
                let [magi_cnt, tik, magi_enc] = times[..] else { unreachable!() };
                println!(
                    "{name:<27} {mib:>9.3} | {:>7.1}MiB/s {:>7.1}MiB/s {:>11} | {:>7.1}MiB/s {:>7.1}MiB/s {:>11} | \
                     count {:.2}x encode {:.2}x",
                    throughput(magi_cnt),
                    throughput(tik),
                    "-",
                    throughput(magi_enc),
                    throughput(tik),
                    "-",
                    tik / magi_cnt,
                    tik / magi_enc,
                );
                continue;
            };
            let expected: usize = texts.iter().map(|text| encoding.tiktoken.encode_ordinary(text).len()).sum();
            let bpe_count: usize = texts.iter().map(|text| bpe_openai.count(text.as_str())).sum();
            assert_eq!(bpe_count, expected, "{name}: bpe-openai count differs");
            let times = time_per_call(&[
                &|| sum_over(&|text| (encoding.count)(text)),
                &|| sum_over(&|text| encoding.tiktoken.encode_ordinary(text).len()),
                &|| sum_over(&|text| bpe_openai.count(text)),
                &|| sum_over(&|text| (encoding.encode)(text).len()),
                &|| sum_over(&|text| bpe_openai.encode(text).len()),
            ]);
            // tiktoken has no count; its encode serves as both.
            let [magi_cnt, tik, bpe_cnt, magi_enc, bpe_enc] = times[..] else { unreachable!() };
            println!(
                "{name:<27} {mib:>9.3} | {:>7.1}MiB/s {:>7.1}MiB/s {:>7.1}MiB/s | {:>7.1}MiB/s {:>7.1}MiB/s {:>7.1}MiB/s | \
                 count {:.2}x/{:.2}x encode {:.2}x/{:.2}x",
                throughput(magi_cnt),
                throughput(tik),
                throughput(bpe_cnt),
                throughput(magi_enc),
                throughput(tik),
                throughput(bpe_enc),
                tik / magi_cnt,
                bpe_cnt / magi_cnt,
                tik / magi_enc,
                bpe_enc / magi_enc,
            );
        }
        // Special-token encoding, which bpe-openai does not offer: magi against tiktoken-rs only.
        let name = format!("{}/special_tokens", encoding.name);
        if selected(&name) {
            let text = special_token_text(encoding, &source[..source.floor_char_boundary(256 << 10)]);
            let expected = encoding.tiktoken.encode_with_special_tokens(&text);
            assert_eq!((encoding.encode_with_special_tokens)(&text), expected, "{name}: ids differ");
            assert_eq!((encoding.count_with_special_tokens)(&text), expected.len(), "{name}: count differs");
            let mib = text.len() as f64 / (1 << 20) as f64;
            let throughput = |micros: f64| mib / (micros / 1e6);
            let times = time_per_call(&[
                &|| (encoding.count_with_special_tokens)(black_box(&text)),
                &|| encoding.tiktoken.encode_with_special_tokens(black_box(&text)).len(),
                &|| (encoding.encode_with_special_tokens)(black_box(&text)).len(),
            ]);
            let [magi_cnt, tik, magi_enc] = times[..] else { unreachable!() };
            println!(
                "{name:<27} {mib:>9.3} | {:>7.1}MiB/s {:>7.1}MiB/s {:>11} | {:>7.1}MiB/s {:>7.1}MiB/s {:>11} | \
                 count {:.2}x encode {:.2}x",
                throughput(magi_cnt),
                throughput(tik),
                "-",
                throughput(magi_enc),
                throughput(tik),
                "-",
                tik / magi_cnt,
                tik / magi_enc,
            );
        }
    }
    Ok(())
}

/// Time from process start-up state to the first finished count, including any table construction.
fn startup(which: &str) -> Result<()> {
    let text = "Hello, world! This is the first call after process start.";
    let start = Instant::now();
    let count = match which {
        "magi" => magi_tokenizer::count_ordinary(text),
        "tiktoken" => tiktoken_rs::o200k_base_singleton().encode_ordinary(text).len(),
        "bpe-openai" => bpe_openai::o200k_base().count(text),
        other => return Err(format!("unknown tokenizer {other}").into()),
    };
    println!("{which}: first count {count} tokens in {:.3}ms", start.elapsed().as_secs_f64() * 1e3);
    Ok(())
}
