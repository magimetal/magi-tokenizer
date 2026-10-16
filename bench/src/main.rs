// Differential verification and benchmarks for magi-toks against tiktoken-rs (reference) and bpe-openai.
//
//   cargo run --release -p magi-toks-bench -- verify   # exact pieces/ids/counts/decode vs references
//   cargo run --release -p magi-toks-bench -- bench    # throughput, all three tokenizers
//   cargo run --release -p magi-toks-bench -- startup <magi|tiktoken|bpe-openai>  # one first-call timing

use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, Instant};
use std::{env, fs};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const COMPARISON_ASSETS: &str = "/Users/magimetal/Dev/_third-party/toks-comparison/assets";
const SOURCE_ROOTS: [&str; 2] = ["/Users/magimetal/Dev/magi-code/src", "/Users/magimetal/Dev/_third-party/toks/src"];
/// tiktoken-rs 0.12.0 `O200K_BASE_PAT_STR`, copied verbatim because tiktoken-rs does not export it.
const O200K_PATTERN: &str = concat!(
    r#"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"#,
    "|",
    r#"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"#,
    "|",
    r#"\p{N}{1,3}"#,
    "|",
    r#" ?[^\s\p{L}\p{N}]+[\r\n/]*"#,
    "|",
    r#"\s*[\r\n]+"#,
    "|",
    r#"\s+(?!\S)"#,
    "|",
    r#"\s+"#
);

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("verify") => verify(),
        Some("bench") => bench(args.get(1).map(String::as_str)),
        Some("startup") => startup(args.get(1).map(String::as_str).unwrap_or("magi")),
        _ => Err("usage: magi-toks-bench verify | bench [filter] | startup <magi|tiktoken|bpe-openai>".into()),
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
        if path.is_dir() {
            collect_source_text(&path, out, limit);
        } else if matches!(path.extension().and_then(|e| e.to_str()), Some("rs" | "c" | "h" | "md" | "py" | "toml" | "json"))
            && let Ok(text) = fs::read_to_string(&path)
        {
            out.push_str(&text);
            out.push('\n');
        }
    }
}

fn source_text() -> String {
    let mut text = String::new();
    for root in SOURCE_ROOTS {
        let limit = text.len() + (4 << 20);
        collect_source_text(Path::new(root), &mut text, limit);
    }
    text
}

fn fixture_200k() -> Result<String> {
    Ok(fs::read_to_string(Path::new(COMPARISON_ASSETS).join("o200k_base-200k.txt"))?)
}

fn session_snapshots() -> Result<Vec<String>> {
    let fixture: serde_json::Value =
        serde_json::from_slice(&fs::read(Path::new(COMPARISON_ASSETS).join("session-compacted-500.json"))?)?;
    let segments = fixture["segments"].as_array().ok_or("segments")?;
    let snapshots = fixture["snapshots"].as_array().ok_or("snapshots")?;
    snapshots
        .iter()
        .map(|snapshot| {
            let segment = snapshot["segment"].as_u64().ok_or("segment")? as usize;
            let end = snapshot["end"].as_u64().ok_or("end")? as usize;
            Ok(segments[segment].as_str().and_then(|text| text.get(..end)).ok_or("snapshot text")?.to_owned())
        })
        .collect()
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

/// Deterministic random strings drawn from characters that stress pretokenization and merging.
fn fuzz_strings(count: usize, seed: u64) -> Vec<String> {
    const ALPHABET: &[&str] = &[
        "a", "b", "e", "s", "t", "A", "Z", "'", "'s", "'S", "'re", "'LL", "'ſ", "ſ", " ", "  ", "\t", "\n", "\r",
        "\r\n", "\u{a0}", "\u{3000}", "\u{2028}", "\u{85}", "0", "7", "٣", "Ⅻ", "½", ".", ",", "!", "/", "//", "-",
        "_", "(", "\"", "\u{301}", "\u{300}", "\u{94d}", "\u{200d}", "\u{fe0f}", "ǅ", "ʰ", "ª", "é", "É", "ß", "你",
        "好", "日", "ン", "한", "ж", "Ж", "ع", "क", "ि", "🙂", "🏽", "\0", "<|endoftext|>", "<|", "|>", "the", " the",
        "ing", "tion", " world", "Hello", "HTTP", "JSON", "1234", "    ", "\n\n", " \n", "{\n", "}\n", "=>", "::",
    ];
    let mut state = seed;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..count)
        .map(|_| {
            let len = (next() % 64) as usize;
            (0..len).map(|_| ALPHABET[(next() % ALPHABET.len() as u64) as usize]).collect()
        })
        .collect()
}

/// Every Unicode scalar in contexts that exercise each regex alternative around it.
fn unicode_sweep() -> Vec<String> {
    let mut texts = Vec::new();
    for context in [("a", "B'99\n"), (" ", "x"), ("A", "'s "), ("!", "\u{301}a"), ("1", " \n"), ("\n", "'LL")] {
        let mut text = String::new();
        for c in (0..=0x10FFFFu32).filter_map(char::from_u32) {
            text.push_str(context.0);
            text.push(c);
            text.push_str(context.1);
        }
        texts.extend(chunks(&text, 8192));
    }
    // Pairs of interesting classes adjacent to each other, both orders.
    let probes: Vec<char> = (0..=0x10FFFFu32).step_by(97).filter_map(char::from_u32).collect();
    let mut pairs = String::new();
    for window in probes.windows(3) {
        pairs.extend(window);
        pairs.push(' ');
    }
    texts.extend(chunks(&pairs, 4096));
    texts
}

// ---- verification ---------------------------------------------------------------------------------------

struct References {
    tiktoken: &'static tiktoken_rs::CoreBPE,
    regex: fancy_regex::Regex,
}

impl References {
    fn new() -> Result<Self> {
        Ok(Self {
            tiktoken: tiktoken_rs::o200k_base_singleton(),
            regex: fancy_regex::Regex::new(O200K_PATTERN)?,
        })
    }

    /// Ok(()) when magi-toks agrees exactly with the references on `text`; otherwise a description.
    fn check(&self, text: &str) -> std::result::Result<(), String> {
        let expected_pieces: Vec<&str> = self.regex.find_iter(text).map(|m| m.unwrap().as_str()).collect();
        let pieces: Vec<&str> = magi_toks::pieces(text).collect();
        if pieces != expected_pieces {
            let first = pieces.iter().zip(&expected_pieces).position(|(a, b)| a != b).unwrap_or(pieces.len());
            return Err(format!(
                "pieces differ at {first}: got {:?}, expected {:?}",
                &pieces[first.saturating_sub(1)..(first + 2).min(pieces.len())],
                &expected_pieces[first.saturating_sub(1)..(first + 2).min(expected_pieces.len())]
            ));
        }
        let expected = self.tiktoken.encode_ordinary(text);
        let ids = magi_toks::encode_ordinary(text);
        if ids != expected {
            return Err(format!("ordinary ids differ: got {} tokens, expected {}", ids.len(), expected.len()));
        }
        if magi_toks::count_ordinary(text) != expected.len() {
            return Err("count_ordinary differs".into());
        }
        if magi_toks::decode(&ids).as_deref() != Some(text) {
            return Err("decode does not round-trip".into());
        }
        let expected_special = self.tiktoken.encode_with_special_tokens(text);
        if magi_toks::encode_with_special_tokens(text) != expected_special
            || magi_toks::count_with_special_tokens(text) != expected_special.len()
        {
            return Err("special-token encoding differs".into());
        }
        Ok(())
    }
}

fn verify() -> Result<()> {
    let references = References::new()?;
    let source = source_text();
    let mut suites: Vec<(&str, Vec<String>)> = vec![
        ("edges", edge_cases()),
        ("unicode_sweep", unicode_sweep()),
        ("fuzz", fuzz_strings(200_000, 0x9E37_79B9_7F4A_7C15)),
        ("source_7", chunks(&source, 7)),
        ("source_48", chunks(&source, 48)),
        ("source_4096", chunks(&source, 4096)),
        ("multilingual_256", chunks(&multilingual_text(), 256)),
        ("fixture_200k", vec![fixture_200k()?]),
        ("long_runs", long_runs()),
        ("long_random", long_random_pieces()),
    ];
    if let Ok(snapshots) = session_snapshots() {
        suites.push(("session_snapshots", snapshots.into_iter().step_by(25).collect()));
    }
    let mut failures = 0;
    for (name, texts) in &suites {
        let start = Instant::now();
        let mut suite_failures = 0;
        for text in texts {
            if let Err(error) = references.check(text) {
                suite_failures += 1;
                if suite_failures <= 5 {
                    let preview: String = text.chars().take(80).collect();
                    println!("  MISMATCH [{name}] {error}\n    text: {preview:?}");
                }
            }
        }
        println!(
            "{name:>20}: {:>7} texts, {:>10} bytes, {suite_failures} mismatches ({:.1}s)",
            texts.len(),
            texts.iter().map(String::len).sum::<usize>(),
            start.elapsed().as_secs_f64()
        );
        failures += suite_failures;
    }
    check_every_token_round_trips()?;
    if failures > 0 {
        return Err(format!("{failures} mismatches").into());
    }
    println!("all suites exact");
    Ok(())
}

fn edge_cases() -> Vec<String> {
    let mut texts: Vec<String> = [
        "", "a", " ", "\n", "hello", "Hello, world!", "\r\n\n\t  ", " <|endoftext|> <|fim_prefix|> <|endofprompt|>",
        "<|endoftext|><|endoftext|>", "<|<|endofprompt|>|>", "don't I'M we're 1234567890", "ſ's 'ſ x'ſ X'ſ",
        "\u{301}abc", " \u{301}abc", "!\u{301}", " \u{301}", "\u{301}", "ABC'S", "ABCdef'Re", "aBC", "ABc", "ǅx", "ʰʰ",
        "a  b", "a  \n b", "  \n\n  x", "x \u{a0}y", "x\u{3000}\u{3000}", "a/\n/b", " !!/\r\n/x", "!!! ", " ! a",
        "12345678", "١٢٣٤", "1\u{301}", "a'", "a'l", "a'r", "a'llx", "\u{10FFFF}", "\0\0\0", "\t\t\tx", "x\t\t\t",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    for n in 1..=140 {
        texts.push("a".repeat(n));
        texts.push(format!(" {}", "z".repeat(n)));
        texts.push("ab".repeat(n));
        texts.push("!".repeat(n));
        texts.push("\u{301}".repeat(n));
        texts.push("🙂".repeat(n));
    }
    texts
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
/// Deterministic long single pieces (letter, punctuation, whitespace runs) from small alphabets, so that windows
/// repeat, nearly repeat, and break in many places. Lengths cluster around the window sizes.
fn long_random_pieces() -> Vec<String> {
    const ALPHABETS: &[&[&str]] = &[
        &["a", "b"],
        &["a", "b", "c"],
        &["e", "o", "n"],
        &["a", "a", "a", "a", "a", "a", "a", "b"],
        &["th", "e", "in", "g", "s"],
        &["é", "e", "\u{301}"],
        &["你", "好", "世", "界"],
        &["x", "y", "z", "q", "j", "k", "w", "v"],
        &["!", "=", "-"],
        &["=", "=", "=", "=", "-", "*"],
        &["(", ")", "{", "}", ";", "."],
        &[" ", " ", " ", " ", " ", " ", "\t"],
        &[" ", "\t", "\u{a0}"],
    ];
    const LENGTHS: &[usize] =
        &[129, 130, 150, 191, 192, 193, 255, 256, 257, 300, 511, 512, 513, 1000, 2047, 2048, 4097, 9000, 20_000];
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut texts = Vec::new();
    for alphabet in ALPHABETS {
        for &len in LENGTHS {
            for _ in 0..3 {
                let mut text = String::new();
                while text.len() < len {
                    text.push_str(alphabet[(next() % alphabet.len() as u64) as usize]);
                }
                texts.push(text);
            }
            // A long run of one symbol interrupted once at a random position.
            let mut run: Vec<&str> = vec![alphabet[0]; len / alphabet[0].len()];
            let at = (next() % run.len() as u64) as usize;
            run[at] = alphabet[alphabet.len() - 1];
            texts.push(run.concat());
        }
    }
    texts
}


fn check_every_token_round_trips() -> Result<()> {
    let mut checked = 0;
    for id in 0..magi_toks::ORDINARY_TOKEN_COUNT as u32 {
        let bytes = magi_toks::token_bytes(id).ok_or("missing token")?;
        if let Ok(text) = std::str::from_utf8(bytes) {
            let pieces: Vec<&str> = magi_toks::pieces(text).collect();
            if pieces.len() == 1 && magi_toks::encode_ordinary(text) != [id] {
                return Err(format!("token {id} ({text:?}) does not encode to itself").into());
            }
            checked += 1;
        }
    }
    println!("{:>20}: {checked} UTF-8 single-piece tokens encode to themselves", "token_identity");
    Ok(())
}

// ---- benchmarks -----------------------------------------------------------------------------------------

/// Median of five runs, each at least 150ms, of microseconds per call of `work`.
fn time_per_call(mut work: impl FnMut() -> usize) -> f64 {
    let mut runs = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        let mut rounds = 0u32;
        while start.elapsed() < Duration::from_millis(150) {
            black_box(work());
            rounds += 1;
        }
        runs.push(start.elapsed().as_secs_f64() * 1e6 / f64::from(rounds));
    }
    runs.sort_by(f64::total_cmp);
    runs[2]
}

fn bench(filter: Option<&str>) -> Result<()> {
    let tiktoken = tiktoken_rs::o200k_base_singleton();
    let bpe_openai = bpe_openai::o200k_base();
    let source = source_text();
    let mut workloads: Vec<(&str, Vec<String>)> = vec![
        ("fixture_200k", vec![fixture_200k()?]),
        ("source_1MiB", vec![source[..source.floor_char_boundary(1 << 20)].to_owned()]),
        ("source_48B_chunks", chunks(&source[..source.floor_char_boundary(256 << 10)], 48)),
        ("source_256B_chunks", chunks(&source[..source.floor_char_boundary(256 << 10)], 256)),
        ("multilingual", vec![multilingual_text()]),
        ("long_runs", long_runs()),
    ];
    if let Ok(snapshots) = session_snapshots() {
        workloads.push(("session_snapshots", snapshots.into_iter().step_by(10).collect()));
    }
    println!(
        "{:<20} {:>9} | {:>11} {:>11} {:>11} | {:>11} {:>11} {:>11} | speedup vs tiktoken / bpe-openai",
        "workload", "MiB", "magi cnt", "tiktok cnt", "bpeoai cnt", "magi enc", "tiktok enc", "bpeoai enc"
    );
    for (name, texts) in &workloads {
        if filter.is_some_and(|filter| !name.contains(filter)) {
            continue;
        }
        let expected: usize = texts.iter().map(|text| tiktoken.encode_ordinary(text).len()).sum();
        let magi_count: usize = texts.iter().map(|text| magi_toks::count_ordinary(text)).sum();
        let bpe_count: usize = texts.iter().map(|text| bpe_openai.count(text.as_str())).sum();
        assert_eq!(magi_count, expected, "{name}: magi-toks count differs");
        assert_eq!(bpe_count, expected, "{name}: bpe-openai count differs");
        let bytes: usize = texts.iter().map(String::len).sum();
        let mib = bytes as f64 / (1 << 20) as f64;
        let throughput = |micros: f64| mib / (micros / 1e6);
        let sum_over = |f: &dyn Fn(&str) -> usize| texts.iter().map(|text| f(black_box(text))).sum::<usize>();
        let magi_cnt = time_per_call(|| sum_over(&|text| magi_toks::count_ordinary(text)));
        let tik_cnt = time_per_call(|| sum_over(&|text| tiktoken.encode_ordinary(text).len()));
        let bpe_cnt = time_per_call(|| sum_over(&|text| bpe_openai.count(text)));
        let magi_enc = time_per_call(|| sum_over(&|text| magi_toks::encode_ordinary(text).len()));
        let tik_enc = time_per_call(|| sum_over(&|text| tiktoken.encode_ordinary(text).len()));
        let bpe_enc = time_per_call(|| sum_over(&|text| bpe_openai.encode(text).len()));
        println!(
            "{name:<20} {mib:>9.3} | {:>8.1}MB/s {:>8.1}MB/s {:>8.1}MB/s | {:>8.1}MB/s {:>8.1}MB/s {:>8.1}MB/s | \
             count {:.2}x/{:.2}x encode {:.2}x/{:.2}x",
            throughput(magi_cnt),
            throughput(tik_cnt),
            throughput(bpe_cnt),
            throughput(magi_enc),
            throughput(tik_enc),
            throughput(bpe_enc),
            tik_cnt / magi_cnt,
            bpe_cnt / magi_cnt,
            tik_enc / magi_enc,
            bpe_enc / magi_enc,
        );
    }
    Ok(())
}

/// Time from process start-up state to the first finished count, including any table construction.
fn startup(which: &str) -> Result<()> {
    let text = "Hello, world! This is the first call after process start.";
    let start = Instant::now();
    let count = match which {
        "magi" => magi_toks::count_ordinary(text),
        "tiktoken" => tiktoken_rs::o200k_base_singleton().encode_ordinary(text).len(),
        "bpe-openai" => bpe_openai::o200k_base().count(text),
        other => return Err(format!("unknown tokenizer {other}").into()),
    };
    println!("{which}: first count {count} tokens in {:.3}ms", start.elapsed().as_secs_f64() * 1e3);
    Ok(())
}
