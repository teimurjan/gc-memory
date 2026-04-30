//! Lightweight, regex-only field extractors for multi-field BM25.
//!
//! The retrieval ablation in `RESEARCH_JOURNEY.md` ("reranker +
//! bi-encoder ablation") established that on conversational long
//! memory BM25 is the load-bearing leg. The next pp lives in giving
//! BM25 *richer signal*, not in upgrading the cross-encoder or the
//! bi-encoder. This module produces two derived views of an entry —
//! `entities` and `title` — that the bench / production pipeline
//! indexes as separate BM25 documents and fuses via RRF.
//!
//! Why regex and not NER: the Claude Code corpus is mostly code,
//! CLI strings, file paths, version numbers, and identifiers — a
//! distribution that off-the-shelf NER (BERT-NER, spaCy default
//! pipeline) under-recovers because it was trained on news text.
//! Curated regex on this distribution has higher precision *and*
//! costs ~zero per chunk.

use std::sync::LazyLock;

use regex::Regex;

/// Capture URLs (http/https). Stops at whitespace or closing paren.
static URL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"https?://[^\s)>\]]+").expect("URL regex"));

/// CamelCase / PascalCase identifiers — typically class names or types.
/// Two or more capitalized chunks: `MemoryStore`, `BiEncoder`,
/// `RetrievalInducedForgetting`.
static CAMEL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b").expect("CamelCase"));

/// Acronyms: 2+ uppercase letters, optional trailing digits.
/// Catches `BM25`, `ONNX`, `RRF`, `NDCG`, `URL`, `CSV`, `IPv4`.
static ACRONYM_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[A-Z]{2,}[0-9]*\b").expect("acronym"));

/// snake_case identifiers — function and variable names.
/// `from_repo_with_pool`, `tokenize_bm25`, `top_k_ids`.
static SNAKE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b").expect("snake_case"));

/// File-like tokens: at least one dot, with alphanumeric segments. Catches
/// `model.onnx`, `bench_bm25.rs`, `package.json`. Excludes IPs/versions
/// (those have their own extractor).
static FILE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[A-Za-z0-9_-]+\.[A-Za-z][A-Za-z0-9]+\b").expect("file"));

/// Path-like tokens: 2+ segments separated by `/`, no spaces.
/// `crates/lethe-core/src/encoders.rs`, `~/.cache/huggingface`.
static PATH_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[A-Za-z0-9_./~-]*/[A-Za-z0-9_./-]+\b").expect("path"));

/// Version strings: `v0.10.0`, `2.0.0-rc.10`, `1.5.2`.
static VERSION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\bv?\d+\.\d+(?:\.\d+)?(?:-[A-Za-z0-9.]+)?\b").expect("version")
});

/// Git hashes / hex IDs (7-40 chars). `4aae737`, `096c64f`,
/// `d34db33fcafe`. Lowercase only — uppercase hex collides with acronyms.
static HASH_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[0-9a-f]{7,40}\b").expect("hash"));

/// Backtick-quoted code spans, capped at 80 chars to skip code blocks.
static BACKTICK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"`([^`\n]{2,80})`").expect("backtick"));

/// Double-quoted strings, same cap.
static DQUOTE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#""([^"\n]{2,80})""#).expect("dquote"));

/// Extract a deduplicated, lowercase list of entity tokens from
/// `text`. The returned `Vec<String>` is the value indexed in the
/// `entities` BM25 leg. Order is insertion order; case-folded for
/// BM25 compatibility (the body BM25 also lowercases).
///
/// Returns at most ~200 entries per chunk to bound the per-document
/// length the entity BM25 sees — long entries don't help retrieval
/// and slow scoring.
#[must_use]
pub fn extract_entities(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut push = |s: &str| {
        let lower = s.to_lowercase();
        if lower.len() >= 2 && seen.insert(lower.clone()) {
            out.push(lower);
        }
    };
    for m in URL_RE.find_iter(text) {
        push(m.as_str());
    }
    for m in CAMEL_RE.find_iter(text) {
        push(m.as_str());
    }
    for m in ACRONYM_RE.find_iter(text) {
        push(m.as_str());
    }
    for m in SNAKE_RE.find_iter(text) {
        push(m.as_str());
    }
    for m in FILE_RE.find_iter(text) {
        push(m.as_str());
    }
    for m in PATH_RE.find_iter(text) {
        push(m.as_str());
    }
    for m in VERSION_RE.find_iter(text) {
        push(m.as_str());
    }
    for m in HASH_RE.find_iter(text) {
        push(m.as_str());
    }
    for cap in BACKTICK_RE.captures_iter(text) {
        if let Some(g) = cap.get(1) {
            push(g.as_str());
        }
    }
    for cap in DQUOTE_RE.captures_iter(text) {
        if let Some(g) = cap.get(1) {
            push(g.as_str());
        }
    }
    if out.len() > 200 {
        out.truncate(200);
    }
    out
}

/// Extract a title-like view: the first non-empty line, capped at
/// 120 characters. Heuristic but cheap — for chat memory entries
/// the first line is usually the question, the topic, or the first
/// sentence of an answer.
#[must_use]
pub fn extract_title(text: &str) -> String {
    for line in text.lines() {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            if trimmed.len() <= 120 {
                return trimmed.to_owned();
            }
            return trimmed.chars().take(120).collect();
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entities_pulls_camel_acronym_snake_path_version_hash() {
        let text = "BiEncoder loads ONNX from `crates/lethe-core/src/encoders.rs` \
                    using from_repo_with_pool, see commit 4aae737, BM25 v0.10.0.";
        let ents = extract_entities(text);
        assert!(ents.contains(&"biencoder".to_owned()), "{ents:?}");
        assert!(ents.contains(&"onnx".to_owned()));
        assert!(ents.contains(&"bm25".to_owned()));
        assert!(ents.contains(&"from_repo_with_pool".to_owned()));
        assert!(
            ents.iter().any(|e| e.contains("encoders.rs")),
            "expected file/path entity in {ents:?}"
        );
        assert!(ents.contains(&"4aae737".to_owned()));
        assert!(ents.iter().any(|e| e.starts_with("v0.10")));
    }

    #[test]
    fn entities_capture_quoted_spans() {
        let text = r#"He said "production is fine" and pointed at `MemoryStore::retrieve`."#;
        let ents = extract_entities(text);
        assert!(ents.contains(&"production is fine".to_owned()), "{ents:?}");
        // The backtick capture is the inner span, lowercased.
        assert!(
            ents.iter().any(|e| e.contains("memorystore")),
            "{ents:?}"
        );
    }

    #[test]
    fn entities_capture_urls() {
        let text = "See https://github.com/foo/bar and visit https://example.com for docs.";
        let ents = extract_entities(text);
        assert!(ents.iter().any(|e| e.contains("github.com")));
        assert!(ents.iter().any(|e| e.contains("example.com")));
    }

    #[test]
    fn entities_dedup_lowercase() {
        let text = "BM25 BM25 bm25 BM25";
        let ents = extract_entities(text);
        assert_eq!(ents.iter().filter(|e| e.as_str() == "bm25").count(), 1);
    }

    #[test]
    fn entities_empty_on_pure_prose() {
        // Plain English with no code-shaped tokens should mostly be empty.
        let ents = extract_entities("the cat sat on the mat");
        assert!(ents.is_empty(), "{ents:?}");
    }

    #[test]
    fn title_first_nonempty_line() {
        assert_eq!(extract_title("hello world\nbody text\n"), "hello world");
        assert_eq!(extract_title("\n\n  first line  \nrest"), "first line");
        assert_eq!(extract_title(""), "");
    }

    #[test]
    fn title_caps_long_first_line() {
        let long = "a".repeat(500);
        let t = extract_title(&long);
        assert!(t.len() <= 120);
    }
}
