"""Cloud GPU encoder for LongMemEval late chunking.

Runs `nomic-embed-text-v1.5` on a Modal GPU, session-encodes the
LongMemEval corpus with late chunking (single forward pass per
session, mean-pool per turn), and emits the same binary layout the
Rust bench reads from `tmp_data/lme_<sanitized>_late/`.

Layout written:
  corpus_embeddings.bin   f32 little-endian, shape (n_corpus, dim)
  corpus_ids.txt          one id per line
  query_embeddings.bin    f32 LE, shape (n_queries, dim)
  query_ids.txt
  sampled_query_indices.txt  (copied from canonical lme_rust/)
  meta.json               n_corpus / n_queries / dim / model / late=true

Run:
  pip install modal
  modal token new            # one-time browser auth
  modal run prep_late.py     # ~10-20 min on L40S, <$1

Why a separate Python script and not an extension to the Rust
bench: ort+CUDA on Mac doesn't exist, and PyTorch handles long-context
encoders (variable seq, attention masks, fp16) with less ceremony than
ort. The output binary format is identical so the local Rust bench can
read this dir directly via `--bi-encoder <repo> --late`.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import modal

# ---------- Modal app + image -------------------------------------------------

# We bake the Python deps into the image so cold start is fast.
# `transformers` + `torch` + `numpy` is ~2.5GB; the GPU image cache
# this once and reuses across invocations.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.45.2",
        "einops==0.8.0",  # nomic-v1.5 needs einops for its custom layers
        "numpy==1.26.4",
    )
)

app = modal.App("lethe-late-chunking", image=image)

# Volume holds the inputs (corpus JSON, meta, ids) and the outputs
# (binary embeddings). Persistent across invocations so re-runs reuse
# the upload.
volume = modal.Volume.from_name("lethe-late-chunking-data", create_if_missing=True)

# Where the volume mounts inside the Modal function.
DATA_MOUNT = "/data"

# Files we expect on the volume — uploaded once by `upload_inputs`.
REQUIRED_INPUTS = [
    "longmemeval_corpus.json",
    "longmemeval_queries.json",
    "longmemeval_meta.json",
    "lme_rust_corpus_ids.txt",
    "lme_rust_query_ids.txt",
    "lme_rust_sampled_query_indices.txt",
]

# Default model — 137M params, 8k context, mean pooling.
DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
# Nomic v1.5 expects task prefixes; for retrieval the corpus side is
# `search_document: ` and the query side is `search_query: `.
CORPUS_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


# ---------- Utilities (executed locally, no Modal needed) ---------------------


LOCAL_DATA = Path("/Users/teimurgasanov/Projects/lethe/tmp_data")


@app.local_entrypoint()
def upload_inputs():
    """Push the three corpus JSONs + lme_rust IDs onto the Modal volume.

    Idempotent — re-uploading the same bytes is cheap (Modal dedupes
    by content hash). Pass `--force` to clobber if you regenerated
    inputs locally.
    """
    print("[local] reading inputs from", LOCAL_DATA)
    pairs = [
        ("longmemeval_corpus.json", "longmemeval_corpus.json"),
        ("longmemeval_queries.json", "longmemeval_queries.json"),
        ("longmemeval_meta.json", "longmemeval_meta.json"),
        ("lme_rust/corpus_ids.txt", "lme_rust_corpus_ids.txt"),
        ("lme_rust/query_ids.txt", "lme_rust_query_ids.txt"),
        ("lme_rust/sampled_query_indices.txt", "lme_rust_sampled_query_indices.txt"),
    ]
    for src_name, _ in pairs:
        if not (LOCAL_DATA / src_name).exists():
            raise SystemExit(f"missing {LOCAL_DATA / src_name}; run from repo root")
    with volume.batch_upload(force=True) as batch:
        for src_name, dst_name in pairs:
            src = LOCAL_DATA / src_name
            size = src.stat().st_size
            print(f"[local] uploading {dst_name}: {size / 1e6:.1f} MB")
            batch.put_file(str(src), f"/{dst_name}")
    print("[local] upload complete")


# ---------- The actual encoding job ------------------------------------------


@app.function(
    gpu="L40S",  # 48GB VRAM, $1.95/hr — plenty for 137M model
    timeout=60 * 60,  # 1h hard cap
    volumes={DATA_MOUNT: volume},
)
def encode_late(model_repo: str = DEFAULT_MODEL, max_len: int = 8192) -> dict:
    """Encode the LongMemEval corpus session-wise with late chunking.

    Returns a dict of stats; the binary outputs are written to the
    Modal volume at /data/output/. Pull them down with `download_outputs`.
    """
    import time
    from collections import defaultdict
    from pathlib import Path as P

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    data = P(DATA_MOUNT)
    print(f"[gpu] loading model {model_repo}…")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_repo,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # fp16 is plenty for retrieval
    ).to("cuda").eval()
    # Some nomic checkpoints declare a small `model_max_length`; widen it.
    tokenizer.model_max_length = max_len
    print(f"[gpu] model ready in {time.perf_counter() - t0:.1f}s")

    print("[gpu] reading inputs…")
    corpus = json.loads((data / "longmemeval_corpus.json").read_text())
    queries = json.loads((data / "longmemeval_queries.json").read_text())
    meta = json.loads((data / "longmemeval_meta.json").read_text())
    corpus_ids = (data / "lme_rust_corpus_ids.txt").read_text().splitlines()
    query_ids = (data / "lme_rust_query_ids.txt").read_text().splitlines()
    sampled_indices_blob = (data / "lme_rust_sampled_query_indices.txt").read_bytes()

    # Group corpus_ids by session_id, preserving canonical row order.
    sessions: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row, cid in enumerate(corpus_ids):
        m = meta[cid]
        sessions[m["session_id"]].append((m["turn_idx"], row))
    for v in sessions.values():
        v.sort(key=lambda t: t[0])
    print(f"[gpu] {len(sessions)} sessions, {len(corpus_ids)} turns total")

    # Probe dim with a dummy encode.
    dim = model(
        **tokenizer(["probe"], return_tensors="pt", truncation=True, max_length=8).to("cuda")
    ).last_hidden_state.shape[-1]
    print(f"[gpu] dim={dim}")

    n_corpus = len(corpus_ids)
    corpus_embs = np.zeros((n_corpus, dim), dtype=np.float32)

    # Tokenize each turn separately so we can splice them together with
    # known token spans. Add the corpus prefix once per turn.
    n_late = 0
    n_fallback = 0
    n_truncated = 0
    t_enc = time.perf_counter()
    log_every = max(1, len(sessions) // 50)

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    for sess_idx, (sess_id, turns) in enumerate(sessions.items()):
        rows = [r for _, r in turns]
        texts = [CORPUS_PREFIX + corpus[corpus_ids[r]] for r in rows]

        # Tokenize each turn (no special tokens; we add CLS/SEP manually).
        per_turn = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        # Pack: [CLS] turn1 [SEP] turn2 [SEP] ...
        packed: list[int] = [cls_id]
        spans: list[tuple[int, int]] = []
        for ids in per_turn:
            start = len(packed)
            packed.extend(ids)
            spans.append((start, len(packed)))
            packed.append(sep_id)

        if len(packed) > max_len:
            # Session too long for one pass — fall back to per-turn
            # encoding. Loses late-chunking benefit for this session.
            n_fallback += 1
            with torch.inference_mode():
                # Use the standard transformers encode path with truncation.
                tok = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                ).to("cuda")
                out = model(**tok).last_hidden_state
                mask = tok["attention_mask"].unsqueeze(-1).float()
                pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1).float().cpu().numpy()
            for j, r in enumerate(rows):
                corpus_embs[r] = pooled[j]
            continue

        n_late += 1
        input_ids = torch.tensor([packed], dtype=torch.long, device="cuda")
        attention = torch.ones_like(input_ids)
        with torch.inference_mode():
            last_hidden = model(input_ids=input_ids, attention_mask=attention).last_hidden_state[0]
        for (start, end), r in zip(spans, rows):
            if end > start:
                emb = last_hidden[start:end].mean(dim=0)
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1).float().cpu().numpy()
                corpus_embs[r] = emb

        if (sess_idx + 1) % log_every == 0:
            elapsed = time.perf_counter() - t_enc
            rate = (sess_idx + 1) / max(elapsed, 1e-3)
            eta = (len(sessions) - sess_idx - 1) / max(rate, 1e-6)
            print(
                f"[gpu] session {sess_idx + 1}/{len(sessions)} "
                f"({rate:.1f}/s eta {eta:.0f}s late={n_late} fallback={n_fallback})",
                flush=True,
            )

    print(f"[gpu] corpus done: late={n_late} fallback={n_fallback}")

    # Queries — single batch with the query prefix; standard pooling.
    print(f"[gpu] encoding {len(query_ids)} queries…")
    query_texts = [QUERY_PREFIX + queries[q] for q in query_ids]
    query_embs = np.zeros((len(query_ids), dim), dtype=np.float32)
    BATCH = 64
    for i in range(0, len(query_texts), BATCH):
        batch = query_texts[i : i + BATCH]
        tok = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len
        ).to("cuda")
        with torch.inference_mode():
            out = model(**tok).last_hidden_state
            mask = tok["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1).float().cpu().numpy()
        query_embs[i : i + len(batch)] = pooled

    # Write outputs to the volume.
    out_dir = data / "output"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "corpus_embeddings.bin").write_bytes(corpus_embs.tobytes(order="C"))
    (out_dir / "query_embeddings.bin").write_bytes(query_embs.tobytes(order="C"))
    (out_dir / "corpus_ids.txt").write_text("\n".join(corpus_ids))
    (out_dir / "query_ids.txt").write_text("\n".join(query_ids))
    (out_dir / "sampled_query_indices.txt").write_bytes(sampled_indices_blob)
    meta_out = {
        "n_corpus": n_corpus,
        "n_queries": len(query_ids),
        "dim": int(dim),
        "bi_encoder": model_repo,
        "pooling": "mean",
        "max_len": max_len,
        "late_chunking": True,
        "n_sessions_late": n_late,
        "n_sessions_fallback": n_fallback,
        "corpus_prefix": CORPUS_PREFIX,
        "query_prefix": QUERY_PREFIX,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta_out, indent=2))
    volume.commit()
    print(f"[gpu] wrote to volume: {out_dir}")
    return meta_out


@app.local_entrypoint()
def download_outputs(local_dir: str = "tmp_data/lme_nomic_late_modal"):
    """Pull the encoded outputs from the Modal volume to a local dir.

    Layout matches what `lethe-benchmark --late --bi-encoder X` reads:
    `lme_<sanitized>_late/{corpus_embeddings.bin, query_embeddings.bin, …}`.
    Move/symlink it to the actual `lme_<sanitized>_late/` path the bench
    expects (the `nomic-ai_nomic-embed-text-v1.5_late` form).
    """
    import shutil
    import io

    target = Path("/Users/teimurgasanov/Projects/lethe") / local_dir
    target.mkdir(parents=True, exist_ok=True)
    print(f"[local] pulling Modal volume → {target}")
    for name in [
        "corpus_embeddings.bin",
        "query_embeddings.bin",
        "corpus_ids.txt",
        "query_ids.txt",
        "sampled_query_indices.txt",
        "meta.json",
    ]:
        buf = io.BytesIO()
        for chunk in volume.read_file(f"/output/{name}"):
            buf.write(chunk)
        (target / name).write_bytes(buf.getvalue())
        print(f"[local]   {name}: {len(buf.getvalue()) / 1e6:.1f} MB")
    print("[local] done")
    print(f"[local] now: cp -r {target} tmp_data/lme_nomic-ai_nomic-embed-text-v1.5_late/")
    print("        and: lethe-benchmark --bi-encoder nomic-ai/nomic-embed-text-v1.5 --pooling mean --late --sample-limit 50 longmemeval")


# ---------- Smoke test (cheap, 5 sessions) -----------------------------------


@app.function(gpu="L40S", timeout=10 * 60, volumes={DATA_MOUNT: volume})
def smoke_test(model_repo: str = DEFAULT_MODEL) -> dict:
    """Encode the first 5 sessions only — sanity check the pipeline + cost."""
    import time
    from collections import defaultdict
    from pathlib import Path as P

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    data = P(DATA_MOUNT)
    tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_repo, trust_remote_code=True, torch_dtype=torch.float16
    ).to("cuda").eval()

    corpus = json.loads((data / "longmemeval_corpus.json").read_text())
    meta = json.loads((data / "longmemeval_meta.json").read_text())
    corpus_ids = (data / "lme_rust_corpus_ids.txt").read_text().splitlines()

    sessions: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row, cid in enumerate(corpus_ids):
        m = meta[cid]
        sessions[m["session_id"]].append((m["turn_idx"], row))
    sample = list(sessions.items())[:5]

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    t0 = time.perf_counter()
    for sess_id, turns in sample:
        turns.sort(key=lambda t: t[0])
        texts = [CORPUS_PREFIX + corpus[corpus_ids[r]] for _, r in turns]
        per_turn = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
        packed = [cls_id]
        for ids in per_turn:
            packed.extend(ids)
            packed.append(sep_id)
        input_ids = torch.tensor([packed], dtype=torch.long, device="cuda")
        attn = torch.ones_like(input_ids)
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attn).last_hidden_state
        print(f"  session {sess_id}: {len(turns)} turns, {len(packed)} tokens, "
              f"out={tuple(out.shape)}")
    return {"sessions": len(sample), "elapsed_s": time.perf_counter() - t0}
