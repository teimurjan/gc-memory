# int8/

Speed-up attempt: swap the bi-encoder to `BAAI/bge-small-en-v1.5` int8
(`qdrant/bge-small-en-v1.5-onnx-q`, 67 MB) hoping a smaller, pre-quantized
model would halve cold-start and raise warm throughput. See
RESEARCH_JOURNEY.md "speed-up attempts that didn't pan out" note.

## What it is

Negative result kept for the journey: synthetic-text micro-benchmarks
showed +4.9× throughput; on real LongMemEval turns the same model
regressed ~4× because BGE's 512-token cap produces ~2× the per-item
compute on long conversational turns.

## One-time setup

```bash
uv pip install -e research_playground/lethe_reference/[benchmarks]
uv run python research_playground/lethe_reference/scripts/prep_longmemeval.py
```

## Run

```bash
uv run python research_playground/int8/run.py
```

Writes `results/BENCHMARKS_INT8.md`. Optional embedding cache lives at
`results/int8_embeds/` (gitignored).

## Findings

Synthetic short text: 108 → 532 items/s (4.9×). Real conversational
turns: 47 → 11 items/s (0.23×). Token-throughput differential dwarfs the
int8 tensor-width win. NDCG arm not completed — speed premise gone.
Production sticks with `all-MiniLM-L6-v2` fp32.
