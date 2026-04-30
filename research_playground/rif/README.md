# rif/

Retrieval-Induced Forgetting (Anderson 1994; SAM model — Raaijmakers &
Shiffrin 1981) as a learned candidate-selection mechanism. Suppresses
chronic false positives at the candidate-selection stage when the workload
has recurring retrieval cues (long-term conversational memory). Spans
`RESEARCH_JOURNEY.md` checkpoints 11–14 + 18.

## What's here

| Script | Checkpoint | Notes |
|--------|------------|-------|
| `run.py` | 11 | Global RIF — cue-independent suppression |
| `run_clustered.py` | 12 | Clustered RIF — k-means cue-dependent suppression (production default, n_clusters=30) |
| `run_gap.py` | 13 | Rank-gap competition formula — best retrieval-only |
| `run_gap_nfcorpus.py` | 18 | NFCorpus replication (negative result — out of scope) |
| `run_extended_metrics.py` | 16 | Behavior decomposition: exact_episode, sibling_confusion, wrong_family, stale_fact, abstain |
| `run_enriched.py` | 17 | RIF + LLM write-time enrichment (Haiku) |
| `run_sweep.py` | 12/13 | Hyperparameter sweep harness |
| `run_explore*.py` | 14 | Symmetric rescue mechanism (negative — degrades at scale) |
| `bootstrap_ci.py` | 18 | Bootstrap percentile CIs + paired permutation tests |

## One-time setup

```bash
uv pip install -e research_playground/lethe_reference/[benchmarks]
uv run python research_playground/lethe_reference/scripts/prep_longmemeval.py
```

## Run

From the repo root:

```bash
uv run python research_playground/rif/run_clustered.py    # checkpoint 12 (production default)
uv run python research_playground/rif/run_gap.py          # checkpoint 13
uv run python research_playground/rif/bootstrap_ci.py     # checkpoint 18 stats
```

Each script writes its results table to `research_playground/rif/results/`.

## Findings

- **Clustering is the load-bearing part.** Both clustered variants reject
  the null at Bonferroni-adjusted p<0.01 on LongMemEval-S; both global
  variants do not (`results/rif_gap_ci.md`).
- **Workload-specific.** On NFCorpus three of four variants significantly
  regress (`results/rif_gap_nfcorpus_ci.md`). The mechanism targets the
  chronic-false-positive pattern of accumulating long-term conversation
  memory; it does not generalize to ad-hoc IR.
- **Production**: `crates/lethe-core/src/rif.rs` ships clustered + gap
  RIF. Centroids are query-based (per the implementation note in
  RESEARCH_JOURNEY.md) and persist to DuckDB across sessions.
