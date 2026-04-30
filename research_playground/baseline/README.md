# baseline/

LongMemEval baseline comparison — produces the headline numbers cited from
`BENCHMARKS.md` and `RESEARCH_JOURNEY.md` checkpoint 10. Five retrieval configs
(vector only, BM25 only, hybrid RRF, vector+xenc, lethe full) on the full
~200k-turn corpus.

## What it is

The reference Python pipeline that every other research_playground benchmark
either compares against or extends. Mirrored 1-1 by the Rust bench
(`crates/lethe-benchmark`) — see `research_playground/rust_migration/` for
the parity proof.

## One-time setup

```bash
uv pip install -e research_playground/lethe_reference/[benchmarks]
uv run python research_playground/lethe_reference/scripts/prep_longmemeval.py
```

## Run

From the repo root:

```bash
uv run python research_playground/baseline/run.py
```

Writes a results markdown alongside the script (configurable via the script's
own flags). Runtime: tens of minutes on M-series CPU for the full eval.

## Findings

See `RESEARCH_JOURNEY.md` checkpoints 8 (BM25 hybrid) and 10 (integrity audit).
Headline: hybrid BM25+vector+xenc → 0.3680 NDCG@10 on the 200-query eval.
