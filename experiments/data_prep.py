"""Download NFCorpus via BEIR, embed all documents and queries, save to disk."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from beir import util as beir_util  # type: ignore[import-untyped]
from beir.datasets.data_loader import GenericDataLoader  # type: ignore[import-untyped]
from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]


DATA_DIR = Path("data")
MODEL_NAME = "all-MiniLM-L6-v2"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    # Download NFCorpus
    dataset = "nfcorpus"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    data_path = beir_util.download_and_unzip(url, str(DATA_DIR))
    print(f"Dataset downloaded to {data_path}")

    # Load corpus, queries, qrels (test split)
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    print(f"Corpus: {len(corpus)} docs, Queries: {len(queries)}, Qrels: {len(qrels)}")

    # Build text for each corpus doc
    corpus_ids = list(corpus.keys())
    corpus_texts = [
        f"{corpus[doc_id].get('title', '')} {corpus[doc_id].get('text', '')}".strip()
        for doc_id in corpus_ids
    ]

    # Build query texts
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    # Embed
    model = SentenceTransformer(MODEL_NAME)
    print(f"Embedding {len(corpus_texts)} corpus documents...")
    corpus_embeddings = model.encode(
        corpus_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=256,
    )
    print(f"Embedding {len(query_texts)} queries...")
    query_embeddings = model.encode(
        query_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=256,
    )

    # Save embeddings
    np.savez(
        str(DATA_DIR / "nfcorpus_prepared.npz"),
        corpus_ids=np.array(corpus_ids),
        corpus_embeddings=corpus_embeddings.astype(np.float32),
        query_ids=np.array(query_ids),
        query_embeddings=query_embeddings.astype(np.float32),
    )

    # Save qrels
    with open(DATA_DIR / "nfcorpus_qrels.json", "w") as f:
        json.dump(qrels, f)

    # Save corpus texts for MemoryEntry.content
    corpus_content = {doc_id: text for doc_id, text in zip(corpus_ids, corpus_texts)}
    with open(DATA_DIR / "nfcorpus_corpus.json", "w") as f:
        json.dump(corpus_content, f)

    print(f"Saved to {DATA_DIR}/")
    print(f"  corpus_embeddings: {corpus_embeddings.shape}")
    print(f"  query_embeddings: {query_embeddings.shape}")


if __name__ == "__main__":
    main()
