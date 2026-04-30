# deep_pass/

Re-calibration of the adaptive deep-pass `k_deep` parameter introduced in
checkpoint 6. The original 200 was picked as "large enough to never hurt";
this script sweeps {30, 60, 100, 200} to find the actual ceiling. See
RESEARCH_JOURNEY.md "adaptive deep-pass `k_deep` re-calibration" note.

## What it is

The TUI made the deep-pass user-visible (cross-project search pays
`N × k_deep` rerank cost), so latency optimization graduated from
"would be nice" to "needed for interactive use".

## One-time setup

```bash
uv pip install -e research_playground/lethe_reference/[benchmarks]
uv run python research_playground/lethe_reference/scripts/prep_longmemeval.py
```

## Run

```bash
uv run python research_playground/deep_pass/run.py
```

Writes `results/BENCHMARKS_DEEP_PASS.md`.

## Findings

`k_deep=100` matches `k_deep=200` on NDCG@10 + R@10 while cutting p50
~42% and p95 ~41%. `k_deep=60` costs 1.1 pp NDCG. Production ships 100;
the knob is exposed on `MemoryStore` / `UnionStore` constructors.
