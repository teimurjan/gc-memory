# Late chunking on Modal

GPU-accelerated session-level encoding of the LongMemEval corpus,
producing the binary embeddings the local Rust bench reads via
`--late`.

## Why this exists

Local Rust+CPU prep on M-series caps at ~1.5 sessions/sec for the
139M-param `nomic-embed-text-v1.5` int8, putting full prep at ~14 hr.
Modal L40S runs the same job in ~10-15 min for <$1.

The Rust bench already has a `prepare-embeddings-late` subcommand;
this script is the cloud-GPU variant of the same logic. Output format
is byte-for-byte identical (`corpus_embeddings.bin` little-endian f32)
so the local bench can read it after a `cp` / symlink.

## One-time setup

```sh
cd research_playground/late_chunking_modal
uv venv .venv                            # creates the venv
uv pip install modal                     # installs modal into it
uv run modal token new                   # browser auth (one-time)
```

`uv run <cmd>` runs `<cmd>` in this venv without activating it.

## Run

All commands stay in `research_playground/late_chunking_modal/`.

```sh
# 1. Push corpus + IDs to Modal volume (one-time per dataset version)
uv run modal run prep_late.py::upload_inputs

# 2. Quick sanity check (5 sessions, ~30 sec, ~$0.05)
uv run modal run prep_late.py::smoke_test

# 3. Full encode — ~15 min on L40S, costs ~$0.50
uv run modal run prep_late.py::encode_late

# 4. Pull results back to local tmp_data/
uv run modal run prep_late.py::download_outputs

# 5. From the repo root: move outputs to the dir the bench expects, build, run
cd ../..
mv tmp_data/lme_nomic_late_modal tmp_data/lme_nomic-ai_nomic-embed-text-v1.5_late
DUCKDB_DOWNLOAD_LIB=1 cargo build --release --bin lethe-benchmark
DYLD_LIBRARY_PATH=target/release ./target/release/lethe-benchmark \
  --bi-encoder nomic-ai/nomic-embed-text-v1.5 --pooling mean --late \
  --sample-limit 50 longmemeval > /tmp/lethe_bench_nomic_late_50.json
```

## Cost calibration

| Step | GPU | Wall time | Cost @ $1.95/hr |
|---|---|---|---|
| `smoke_test` (5 sessions) | L40S | ~30s incl. cold load | ~$0.02 |
| `encode_late` (19k sessions) | L40S | ~10-15 min | ~$0.40 |
| Cold model load | L40S | ~30-60s first time | flat |

Volume storage (~600 MB inputs + ~600 MB outputs) is ~$0.04/mo —
keep it around for re-runs at different `max_len` / model.

## Knobs

- `--model_repo` (default `nomic-ai/nomic-embed-text-v1.5`): any
  long-context HF model with an `AutoModel` mean-pool fit.
  Try `Alibaba-NLP/gte-base-en-v1.5` (137M, 8k) for an A/B at the
  same param count.
- `--max_len` (default 8192): cap on packed session length. Sessions
  longer than this fall back to per-turn encoding (lose late-chunking
  benefit). LongMemEval p99 ≈ 8k tokens so 8192 covers ~99% of sessions.

## Verifying parity

The Rust bench's own `prepare-embeddings-late` uses the same algorithm
(pack with `[CLS] turn1 [SEP] turn2 [SEP] ...`, mean-pool token spans).
Numerical drift between Rust ort+CPU and Python torch+GPU comes only
from fp32-vs-fp16 and ONNX-vs-PyTorch attention impl. NDCG@10 should
match within ±0.005; if it doesn't, audit the prefix string or pooling
mask first.
