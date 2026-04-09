# gc-memory

Research prototype testing germinal-center-inspired adaptive mutation-selection over vector embeddings.

## stack

- Python 3.12, managed with `uv`
- FAISS (faiss-cpu) for ANN, sentence-transformers (all-MiniLM-L6-v2, 384-dim)
- numpy for mutation math, matplotlib for plots, pytest for tests
- LongMemEval (S variant) for benchmark data via HuggingFace datasets

## layout

```
src/gc_memory/       # core library (config, entry, mutation, metrics, store, baselines)
experiments/         # data_prep.py, run_experiment.py, analyze.py
tests/               # test_mutation.py, test_store.py, test_metrics.py
results/             # gitignored, populated by experiments
data/                # gitignored, populated by data_prep.py
writeup/             # gitignored, blog post
```

## commands

```bash
uv venv --python 3.12 && uv pip install -e .
uv run pytest tests/ -v                          # run tests
uv run pytest --cov=gc_memory --cov-report=term-missing tests/  # with coverage
uv run mypy --strict src/gc_memory/               # type check
uv run python experiments/data_prep.py             # download + embed LongMemEval (~5 min)
uv run python experiments/run_experiment.py        # run 3-arm experiment (~45 sec)
uv run python experiments/analyze.py               # generate plots + summary
```

## architecture

- `config.py`: single frozen dataclass, all hyperparameters. Every magic number lives here.
- `entry.py`: `Tier` enum (naive/gc/memory/apoptotic), `MemoryEntry` dataclass, `create_entry()` factory.
- `mutation.py`: pure functions. `compute_sigma`, `generate_mutants`, `select_best_mutant`. No state.
- `metrics.py`: pure functions. `ndcg_at_k`, `recall_at_k`, `compute_diversity`, `compute_anchor_drift`.
- `store.py`: `GCMemoryStore` with FAISS index, retrieval with tier weighting, mutation orchestration, affinity EMA, tier transitions, time decay. `_get_sigma()` is the override point for baselines.
- `baselines.py`: `StaticStore` (no-op updates), `RandomMutationStore` (fixed sigma via `_get_sigma` override).

## key formulas (do not change without asking)

- Mutation sigma: `sigma_0 * (1 - affinity) ** gamma`
- Selection: accept if `cos(query, mutant) - cos(query, original) > delta AND cos(mutant, original_embedding) >= theta_anchor`
- Affinity EMA: `(1 - alpha) * affinity + alpha * cos(query, embedding)`
- Time decay: `affinity *= exp(-lambda * interval / 100)` per period, fixed interval

## experiment status

Switched benchmark from NFCorpus to LongMemEval (conversation memory, 19k sessions, 500 questions).
Previous NFCorpus results (Run 2, fixed decay): NEGATIVE. GC +3.3% vs Random, -2.8% vs Static.
LongMemEval run: pending.
