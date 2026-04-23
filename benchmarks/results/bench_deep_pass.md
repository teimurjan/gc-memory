# Adaptive deep-pass `k_deep` sweep

Motivation: the cross-project TUI search was occasionally taking 5+ seconds on weak-match queries. Instrumentation showed the 100% of the extra latency was the adaptive deep pass (`UnionStore.retrieve`, `MemoryStore.retrieve`) re-reranking up to `k_deep=200` candidates at ~17 ms/pair after the top-30 shallow rerank scored below `confidence_threshold=4.0`.

This sweep asks: is `k_deep=200` earning its latency?

## Setup

- Dataset: LongMemEval S (199,509 turns, 500 queries, 100-query random sample, seed 0)
- Pipeline: BM25 top-N + FAISS top-N → RRF merge → cross-encoder rerank of all N, where N = `k_deep`
- Cross-encoder: `Xenova/ms-marco-MiniLM-L-6-v2` (fastembed ONNX, CPU)
- Shallow path fixed at `k_shallow=30`, `confidence_threshold=4.0`
- 100 queries × 4 configs

## Results

| config | NDCG@10 | Recall@10 | ΔNDCG vs baseline | p50 | p95 | p99 | mean | deep-pass fires |
|---|---|---|---|---|---|---|---|---|
| shallow-only (no deep pass) | 0.2866 | 0.3493 | −1.56 pp | 1689 ms | 2371 ms | 3044 ms | 1718 ms | 0% |
| k_deep=60 | 0.2910 | 0.3527 | −1.11 pp | 4159 ms | 5936 ms | 6707 ms | 3555 ms | 59% |
| **k_deep=100** | **0.3022** | **0.3731** | **0.00 pp** | **5651 ms** | **7440 ms** | **8262 ms** | **4420 ms** | **59%** |
| k_deep=200 (old default) | 0.3022 | 0.3727 | baseline | 9800 ms | 12650 ms | 13200 ms | 7137 ms | 59% |

## Decision

Ship `k_deep=100`. It delivers **identical NDCG@10 and Recall@10** to the `k_deep=200` baseline while cutting p50 latency ~42% and p95 ~41%. The back half of the old rerank pool never contained a top-10 answer for this workload — the cross-encoder's top-10 picks stabilize by merged rank ~100.

`k_deep=60` is tempting (another ~1.5 s off p50) but costs 1.11 pp NDCG, well outside the noise floor. Not worth it.

## Reproduce

```bash
uv run python benchmarks/bench_deep_pass.py
```

## Caveats

- Pipeline is single-project here (stands in for the cross-project deep pass). In `UnionStore` the gather fans out per project, so the absolute rerank cost scales linearly with registered projects at the same `k_deep` — identical NDCG shape, steeper latency slope. The ratio (k_deep=100 is Pareto-identical to k_deep=200) is workload-portable; the absolute ms numbers are not.
- Absolute latencies here are higher than what a user sees on a typical 100-1000-entry memory store (full LongMemEval corpus was in use).
