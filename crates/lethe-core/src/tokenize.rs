//! BM25 tokenizer — direct port of `src/lethe/vectors.py::tokenize_bm25`.
//!
//! The Python reference is a regex word tokenizer that lowercases input
//! and extracts `[A-Za-z0-9_]+` runs. The shipping benchmark
//! (`benchmarks/results/BENCHMARKS_BM25_TOKENIZER.md`) recorded a
//! +3.68 pp NDCG@10 / +6.79 pp Recall@10 lift over `lower().split()`.
//! See that document before considering changes.

use std::sync::LazyLock;

use regex::Regex;

static WORD_RE: LazyLock<Regex> = LazyLock::new(|| {
    // The `\w` shortcut would be Unicode-aware; the Python reference is
    // ASCII-only. Match it byte-for-byte.
    Regex::new(r"[A-Za-z0-9_]+").expect("static regex must compile")
});

/// Tokenize `text` for BM25 — lowercased ASCII word runs.
///
/// Empty input yields an empty vector (callers like `search_bm25` rely
/// on this to short-circuit on punctuation-only queries).
#[must_use]
pub fn tokenize_bm25(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    // Python lowercases first, then runs the regex; matching that order
    // matters for non-ASCII inputs (the regex won't match non-ASCII
    // anyway, but the lowercase happens first either way).
    let lowered = text.to_ascii_lowercase();
    WORD_RE
        .find_iter(&lowered)
        .map(|m| m.as_str().to_owned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_punctuation_and_lowercases() {
        // Mirrors tests/test_vectors.py::test_tokenize_bm25_strips_punctuation_and_lowercases.
        assert_eq!(tokenize_bm25("MongoDB?"), vec!["mongodb"]);
        assert_eq!(tokenize_bm25("Hello, world!"), vec!["hello", "world"]);
        assert_eq!(tokenize_bm25("can't won't"), vec!["can", "t", "won", "t"]);
        assert!(tokenize_bm25("").is_empty());
    }

    #[test]
    fn punctuation_only_yields_empty() {
        for q in ["???", "...", "!!!", "   "] {
            assert!(tokenize_bm25(q).is_empty(), "expected empty for {q:?}");
        }
    }

    #[test]
    fn keeps_underscores_and_digits() {
        assert_eq!(
            tokenize_bm25("session_42 turn_idx=7"),
            vec!["session_42", "turn_idx", "7"]
        );
    }
}
