// Benchmarks for magi-tokenizer against tiktoken-rs and bpe-openai.
//
//   cargo run --release -p magi-tokenizer-bench -- bench    # throughput, all three tokenizers
//   cargo run --release -p magi-tokenizer-bench -- startup <magi|tiktoken|bpe-openai>  # one first-call timing
//   cargo run --release -p magi-tokenizer-bench -- memory <MiB> <count|encode> <window|fallback>  # one long piece;
//       run under `/usr/bin/time -l` (macOS) or `/usr/bin/time -v` (Linux) for peak RSS
//
// Inputs: MAGI_TOKENIZER_SOURCE lists source trees (path-list syntax, default: this repository) to cut into chunks.
// MAGI_TOKENIZER_ASSETS names a directory with o200k_base-200k.txt and session-compacted-500.json; without it those
// workloads are skipped.
//
// The bench enables magi-tokenizer's `harness` feature for fallback counters and forced fallback.

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use std::{env, fs};

use magi_tokenizer::harness;
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
/// Pieces that defeat window reuse: long random text from small alphabets. Stitch points are found quickly,
/// but every window is new, so the cache never hits. These are the workloads where window merging costs the most.
fn long_unrepeated_pieces() -> Vec<String> {
    let mut state = 0x7777_1111_2222_3333u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut random_text = |len: usize, alphabet: &[&str]| {
        let mut text = String::new();
        while text.len() < len {
            text.push_str(alphabet[(next() % alphabet.len() as u64) as usize]);
        }
        text
    };
    let mut texts = vec![
        random_text(200_000, &["a", "b"]),
        random_text(240_000, &["你", "好", "世", "界", "的", "是", "在", "了"]),
        random_text(200_000, &["!", "=", "-", "*", "(", ")", "{", "}", ";", "."]),
    ];
    for len in (129..1024).step_by(7) {
        texts.push(random_text(len, &["a", "b"]));
    }
    texts
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

fn bench(filter: Option<&str>) -> Result<()> {
    let tiktoken = tiktoken_rs::o200k_base_singleton();
    let bpe_openai = bpe_openai::o200k_base();
    let source = source_text();
    let mut workloads: Vec<(&str, Vec<String>)> = vec![
        ("source_1MiB", vec![source[..source.floor_char_boundary(1 << 20)].to_owned()]),
        ("source_48B_chunks", chunks(&source[..source.floor_char_boundary(256 << 10)], 48)),
        ("source_256B_chunks", chunks(&source[..source.floor_char_boundary(256 << 10)], 256)),
        ("multilingual", vec![multilingual_text()]),
        ("long_runs", long_runs()),
        ("long_unrepeated", long_unrepeated_pieces()),
    ];
    if let Some(fixture) = fixture_200k()? {
        workloads.insert(0, ("fixture_200k", vec![fixture]));
    }
    if let Some(snapshots) = session_snapshots()? {
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
        let magi_count: usize = texts.iter().map(|text| magi_tokenizer::count_ordinary(text)).sum();
        let bpe_count: usize = texts.iter().map(|text| bpe_openai.count(text.as_str())).sum();
        assert_eq!(magi_count, expected, "{name}: magi-tokenizer count differs");
        assert_eq!(bpe_count, expected, "{name}: bpe-openai count differs");
        let bytes: usize = texts.iter().map(String::len).sum();
        let mib = bytes as f64 / (1 << 20) as f64;
        let throughput = |micros: f64| mib / (micros / 1e6);
        let sum_over = |f: &dyn Fn(&str) -> usize| texts.iter().map(|text| f(black_box(text))).sum::<usize>();
        let times = time_per_call(&[
            &|| sum_over(&|text| magi_tokenizer::count_ordinary(text)),
            &|| sum_over(&|text| tiktoken.encode_ordinary(text).len()),
            &|| sum_over(&|text| bpe_openai.count(text)),
            &|| sum_over(&|text| magi_tokenizer::encode_ordinary(text).len()),
            &|| sum_over(&|text| bpe_openai.encode(text).len()),
        ]);
        // tiktoken has no count; its encode serves as both.
        let [magi_cnt, tik, bpe_cnt, magi_enc, bpe_enc] = times[..] else { unreachable!() };
        println!(
            "{name:<20} {mib:>9.3} | {:>7.1}MiB/s {:>7.1}MiB/s {:>7.1}MiB/s | {:>7.1}MiB/s {:>7.1}MiB/s {:>7.1}MiB/s | \
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
