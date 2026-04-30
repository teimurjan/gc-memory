# bm25_tokenizer/

Tokenizer ablation for the BM25 path. Sweeps `lower().split()` (the original
checkpoint-8 implementation) against three alternatives: regex word-tokens,
regex + stopword removal, regex + Porter stemming. Surfaces the largest
single-lever quality win after enrichment.

## What it is

A direct response to a Codex code review that flagged `lower().split()`
losing punctuation-adjacent tokens (e.g. `"MongoDB?"` ≠ `"MongoDB"`).
Result: regex `[A-Za-z0-9_]+` lifts the production pipeline 0.3680 →
0.3817 NDCG@10. See RESEARCH_JOURNEY.md "BM25 tokenizer" implementation
note.

## One-time setup

```bash
uv pip install -e research_playground/lethe_reference/[benchmarks]
uv run python research_playground/lethe_reference/scripts/prep_longmemeval.py
```

## Run

```bash
uv run python research_playground/bm25_tokenizer/run.py
```

Writes `results/BENCHMARKS_BM25_TOKENIZER.md`.

## Findings

| Tokenizer | NDCG@10 | Δ vs split |
|-----------|---------|------------|
| baseline (`lower().split()`) | 0.3022 | — |
| **regex `[A-Za-z0-9_]+`** | **0.3390** | **+3.68 pp** |
| regex + stopword removal | 0.3084 | +0.63 pp |
| regex + Porter stemming | 0.3153 | +1.31 pp (17× build cost) |

Stopword removal regresses (function words act as syntactic anchors in
short conversational queries). Porter stemming over-conflates. Production
ships plain regex.
