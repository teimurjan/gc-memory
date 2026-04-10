"""Prepare datasets: NFCorpus (BEIR) and/or LongMemEval (HuggingFace).

Usage:
    python experiments/data_prep.py --dataset nfcorpus
    python experiments/data_prep.py --dataset longmemeval
    python experiments/data_prep.py --dataset both
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]


DATA_DIR = Path("data")
MODEL_NAME = "all-MiniLM-L6-v2"


def prep_nfcorpus(model: SentenceTransformer) -> None:
    """Download and embed NFCorpus via BEIR."""
    from beir import util as beir_util  # type: ignore[import-untyped]
    from beir.datasets.data_loader import GenericDataLoader  # type: ignore[import-untyped]

    dataset = "nfcorpus"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    data_path = beir_util.download_and_unzip(url, str(DATA_DIR))
    print(f"NFCorpus downloaded to {data_path}")

    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    print(f"NFCorpus: {len(corpus)} docs, {len(queries)} queries")

    corpus_ids = list(corpus.keys())
    corpus_texts = [
        f"{corpus[did].get('title', '')} {corpus[did].get('text', '')}".strip()
        for did in corpus_ids
    ]
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    print(f"Embedding {len(corpus_texts)} documents...")
    corpus_embs = model.encode(corpus_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=256)
    print(f"Embedding {len(query_texts)} queries...")
    query_embs = model.encode(query_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=256)

    np.savez(str(DATA_DIR / "nfcorpus_prepared.npz"),
             corpus_ids=np.array(corpus_ids), corpus_embeddings=corpus_embs.astype(np.float32),
             query_ids=np.array(query_ids), query_embeddings=query_embs.astype(np.float32))
    with open(DATA_DIR / "nfcorpus_qrels.json", "w") as f:
        json.dump(qrels, f)
    with open(DATA_DIR / "nfcorpus_corpus.json", "w") as f:
        json.dump({cid: text for cid, text in zip(corpus_ids, corpus_texts)}, f)
    with open(DATA_DIR / "nfcorpus_queries.json", "w") as f:
        json.dump(queries, f)
    # NFCorpus entries don't have session/turn structure, use doc_id as session_id
    meta = {cid: {"session_id": cid, "turn_idx": 0} for cid in corpus_ids}
    with open(DATA_DIR / "nfcorpus_meta.json", "w") as f:
        json.dump(meta, f)
    print(f"NFCorpus saved: {corpus_embs.shape}")


def turn_id(session_id: str, turn_idx: int) -> str:
    return f"{session_id}_t{turn_idx}"


def prep_longmemeval(model: SentenceTransformer) -> None:
    """Download and embed LongMemEval S variant at turn level."""
    from datasets import load_dataset  # type: ignore[import-untyped]

    print("Loading LongMemEval S variant...")
    ds = load_dataset("xiaowu0162/longmemeval-cleaned",
                      data_files="longmemeval_s_cleaned.json", split="train")
    print(f"Loaded {len(ds)} questions")

    corpus: dict[str, str] = {}
    meta: dict[str, dict[str, object]] = {}
    for row in ds:
        for sid, session in zip(row["haystack_session_ids"], row["haystack_sessions"]):
            for ti, turn in enumerate(session):
                tid = turn_id(sid, ti)
                if tid not in corpus:
                    corpus[tid] = f"{turn['role']}: {turn['content']}"
                    meta[tid] = {"session_id": sid, "turn_idx": ti}

    queries: dict[str, str] = {}
    for row in ds:
        queries[row["question_id"]] = row["question"]

    qrels: dict[str, dict[str, int]] = {}
    for row in ds:
        qid = row["question_id"]
        relevant: dict[str, int] = {}
        for sid, session in zip(row["haystack_session_ids"], row["haystack_sessions"]):
            for ti, turn in enumerate(session):
                if turn.get("has_answer") is True:
                    relevant[turn_id(sid, ti)] = 1
        if not relevant:
            for sid, session in zip(row["haystack_session_ids"], row["haystack_sessions"]):
                if sid in row["answer_session_ids"]:
                    for ti, _ in enumerate(session):
                        relevant[turn_id(sid, ti)] = 1
        qrels[qid] = relevant

    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    print(f"LongMemEval: {len(corpus_ids)} turns, {len(query_ids)} queries")

    print(f"Embedding {len(corpus_texts)} turns...")
    corpus_embs = model.encode(corpus_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=256)
    print(f"Embedding {len(query_texts)} questions...")
    query_embs = model.encode(query_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=256)

    np.savez(str(DATA_DIR / "longmemeval_prepared.npz"),
             corpus_ids=np.array(corpus_ids), corpus_embeddings=corpus_embs.astype(np.float32),
             query_ids=np.array(query_ids), query_embeddings=query_embs.astype(np.float32))
    with open(DATA_DIR / "longmemeval_qrels.json", "w") as f:
        json.dump(qrels, f)
    with open(DATA_DIR / "longmemeval_corpus.json", "w") as f:
        json.dump({cid: text for cid, text in zip(corpus_ids, corpus_texts)}, f)
    with open(DATA_DIR / "longmemeval_queries.json", "w") as f:
        json.dump(queries, f)
    with open(DATA_DIR / "longmemeval_meta.json", "w") as f:
        json.dump(meta, f)
    print(f"LongMemEval saved: {corpus_embs.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["nfcorpus", "longmemeval", "both"], default="both")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)

    if args.dataset in ("nfcorpus", "both"):
        prep_nfcorpus(model)
    if args.dataset in ("longmemeval", "both"):
        prep_longmemeval(model)

    print("Done.")


if __name__ == "__main__":
    main()
