# gc-memory

A memory store for LLM agents with hybrid retrieval and bio-inspired lifecycle management. Combines BM25 + dense vector retrieval with cross-encoder reranking, deduplication, and a germinal-center-inspired tier system for memory lifecycle.

**NDCG@10 = 0.368 on LongMemEval** (+167% over vector-only baseline).

## Quick start

```python
from gc_memory import MemoryStore
from sentence_transformers import SentenceTransformer, CrossEncoder

store = MemoryStore(
    "./my_memories",
    bi_encoder=SentenceTransformer("all-MiniLM-L6-v2"),
    cross_encoder=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"),
)

store.add("I prefer window seats on flights", session_id="trip")
store.add("My wife needs aisle seats", session_id="trip")
store.add("I work at Google as a software engineer", session_id="work")

results = store.retrieve("What are my travel preferences?", k=5)
for entry_id, content, score in results:
    print(f"  [{score:.1f}] {content}")

store.save()
store.close()
```

## Architecture

```
Query
  │
  ├── FAISS top-30 (dense vector similarity, ~1ms)
  ├── BM25 top-30 (sparse keyword match, ~5ms)
  │
  └── Merge + dedup candidates
          │
          └── Cross-encoder rerank → top-k (~300ms)
                    │
                    ├── If max score < threshold → deep search (k=200)
                    │
                    └── Update affinities, check tier transitions
```

### Three storage layers

| Layer | File | Purpose |
|-------|------|---------|
| SQLite | `gc_memory.db` | Entries metadata, rescue cache, stats |
| numpy + FAISS | `embeddings.npz`, `faiss.index` | Vector storage, dense index |
| BM25 | In-memory, rebuilt on startup | Sparse keyword index |

### Entry lifecycle (inspired by germinal center biology)

```
NAIVE → GC → MEMORY
              ↓
         APOPTOTIC
```

- **Naive**: new entries, unproven
- **GC** (germinal center): retrieved 3+ times, actively evaluated
- **Memory**: high affinity + frequently retrieved, stable tier, exempt from decay
- **Apoptotic**: low affinity + idle > 1000 steps, excluded from search

### Deduplication (on add)

1. **Exact**: SHA-256 content hash (free)
2. **Near-duplicate**: cosine similarity > 0.95 (keeps the longer entry)

## Install

```bash
git clone https://github.com/teimurjan/gc-memory && cd gc-memory
uv venv --python 3.12 && uv pip install -e .
```

## Benchmark

```bash
uv run python experiments/data_prep.py --dataset longmemeval
uv run python benchmarks/run_benchmark.py
```

See [BENCHMARKS.md](BENCHMARKS.md) for results.

## How it compares

| System | Approach | NDCG@10 |
|--------|----------|---------|
| Vector only (MiniLM) | Dense retrieval | 0.1376 |
| BM25 only | Sparse keyword | 0.2420 |
| Hybrid BM25+vector RRF | Rank fusion, no reranker | 0.2171 |
| **Hybrid + cross-encoder rerank** | **BM25 + vector + xenc** | **0.3680** |

The retrieval quality comes from combining BM25 keyword matching with dense vector similarity, then reranking with a cross-encoder. This is a well-known IR technique. The GC mechanism provides memory lifecycle management (tiers, decay, dedup) but does not improve retrieval quality. See [BENCHMARKS.md](BENCHMARKS.md) for integrity checks.

Measured on LongMemEval S variant (200k conversation turns, 500 evaluation questions).

## Project structure

```
src/gc_memory/
├── memory_store.py   # Main API: MemoryStore
├── db.py             # SQLite persistence
├── vectors.py        # FAISS + BM25 index management
├── reranker.py       # Cross-encoder + adaptive depth
├── dedup.py          # Hash + cosine deduplication
├── entry.py          # MemoryEntry dataclass + Tier enum
└── config.py         # Hyperparameters

benchmarks/           # Benchmark scripts + results
experiments/          # Research experiment harness
research/             # Original GC mutation research code
```

## Research background

This project started as an experiment porting the immune system's germinal center mechanism to vector memory. Eight approaches were tested: direct embedding mutation, adapter mutation, MLP adapters, text segmentation, co-relevance graphs, rescue caching, adaptive search depth, and GC routing indexes. None improved retrieval quality over the static hybrid+xenc baseline.

What DID work for retrieval: BM25 hybrid retrieval + cross-encoder reranking (a standard IR technique).

What the GC mechanism provides: memory lifecycle management (tier promotion, affinity decay, deduplication). This is useful for long-running agents but doesn't improve retrieval quality on static benchmarks. The full research journey is in [RESEARCH_JOURNEY.md](RESEARCH_JOURNEY.md).
