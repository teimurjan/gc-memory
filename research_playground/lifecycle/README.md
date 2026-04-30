# lifecycle/

Tier lifecycle benchmark — exercises the naive → gc → memory tier
transitions with decay and apoptosis, parallel to the standard retrieval
benchmark. From the early phase-1 work on memory management as a separable
concern from retrieval quality.

## What it is

Tracks the cost layer (entry counts per tier, decay rate, apoptosis
events) that the retrieval benchmarks treat as out-of-scope. Useful when
tuning lifecycle constants without re-running the full quality bench.

## One-time setup

```bash
uv pip install -e research_playground/lethe_reference/[benchmarks]
uv run python research_playground/lethe_reference/scripts/prep_longmemeval.py
```

## Run

```bash
uv run python research_playground/lifecycle/run.py
```

Writes `results/BENCHMARKS_LIFECYCLE.md`.

## Findings

Per RESEARCH_JOURNEY.md phase 1 conclusion: the biology-inspired control
loop (adaptive rate, tier lifecycle, decay) is sound engineering for
**memory management** even though it doesn't help **retrieval quality**.
This script measures the management side in isolation.
