# lethe_reference/

The Python reference implementation that the Rust workspace under `crates/`
ports from. Not an experiment — this is the parent codebase, kept around to
back the parity bench (`research_playground/rust_migration/`) and to produce
the published research-journey numbers.

Installed as `lethe-memory-legacy` in dev venvs; not published to PyPI.
The shipping Python wheel is `crates/lethe-py` (PyO3 bindings).

## What's here

```
lethe_reference/
├── lethe/             # Python package (db, encoders, memory_store, rif, …)
├── scripts/           # data prep + LLM enrichment helpers
├── tests/             # 148 production + 8 PyO3 parity tests
├── pyproject.toml
└── uv.lock
```

## One-time setup

```bash
uv pip install -e research_playground/lethe_reference/[dev]
```

## Run tests

```bash
cd research_playground/lethe_reference && uv run pytest tests/ -q
```

148 production + 8 PyO3 parity = 156, ~3 minutes (the PyO3 set loads ONNX
models). No network, no API keys required.

## Why it stays

- `crates/lethe-core/*.rs` files cite this tree in their `//! port of …`
  headers; behavior is held against it.
- `research_playground/rust_migration/` shells into both impls and diffs the
  output (NDCG, components, latency).
- Every benchmark script under `research_playground/{baseline,rif,bm25_tokenizer,
  deep_pass,int8,lifecycle}/` imports from `lethe.*`.
- `release-please` bumps `lethe/__init__.py` and `pyproject.toml` on each
  release so the version stays aligned with the Rust workspace.

## Status

Frozen except for security/dependency updates and bug fixes that surface
during parity checks. New product features go into the Rust workspace.
