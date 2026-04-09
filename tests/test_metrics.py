from __future__ import annotations

import math

import numpy as np
import pytest

from gc_memory.entry import MemoryEntry, Tier, create_entry
from gc_memory.metrics import (
    compute_anchor_drift,
    compute_diversity,
    compute_mean_generation,
    compute_tier_distribution,
    ndcg_at_k,
    recall_at_k,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


class TestDiversity:
    def test_identical_embeddings_zero_diversity(self, rng: np.random.Generator) -> None:
        v = np.ones((5, 384), dtype=np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        v = v / norms
        assert compute_diversity(v, 100, rng) == pytest.approx(0.0, abs=1e-6)

    def test_hand_calculated(self, rng: np.random.Generator) -> None:
        """5 orthogonal-ish vectors: pairwise cosine distance should average ~1.0."""
        # Use standard basis vectors in high-dim space — orthogonal means cos=0, dist=1
        embeddings = np.zeros((5, 384), dtype=np.float32)
        for i in range(5):
            embeddings[i, i] = 1.0
        diversity = compute_diversity(embeddings, 1000, rng)
        assert diversity == pytest.approx(1.0, abs=0.01)

    def test_single_entry_returns_zero(self, rng: np.random.Generator) -> None:
        v = np.ones((1, 384), dtype=np.float32)
        v /= np.linalg.norm(v)
        assert compute_diversity(v, 100, rng) == 0.0


class TestAnchorDrift:
    def test_zero_for_unmutated(self, rng: np.random.Generator) -> None:
        entries = []
        for i in range(10):
            v = rng.standard_normal(384).astype(np.float32)
            entries.append(create_entry(f"e{i}", f"content {i}", v))
        drift = compute_anchor_drift(entries, 10, rng)
        assert drift == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_mutated(self, rng: np.random.Generator) -> None:
        v = rng.standard_normal(384).astype(np.float32)
        entry = create_entry("e0", "content", v)
        # Mutate the embedding away from original
        noise = rng.standard_normal(384).astype(np.float32) * 0.5
        mutated = entry.embedding + noise
        entry.embedding = mutated / np.linalg.norm(mutated)
        drift = compute_anchor_drift([entry], 1, rng)
        assert drift > 0.0

    def test_empty_entries(self, rng: np.random.Generator) -> None:
        assert compute_anchor_drift([], 10, rng) == 0.0


class TestTierDistribution:
    def test_counts(self) -> None:
        entries = []
        rng = np.random.default_rng(0)
        for i in range(3):
            e = create_entry(f"n{i}", "c", rng.standard_normal(384).astype(np.float32))
            entries.append(e)
        for i in range(2):
            e = create_entry(f"g{i}", "c", rng.standard_normal(384).astype(np.float32))
            e.tier = Tier.GC
            entries.append(e)
        e = create_entry("m0", "c", rng.standard_normal(384).astype(np.float32))
        e.tier = Tier.MEMORY
        entries.append(e)

        dist = compute_tier_distribution(entries)
        assert dist[Tier.NAIVE] == 3
        assert dist[Tier.GC] == 2
        assert dist[Tier.MEMORY] == 1
        assert dist[Tier.APOPTOTIC] == 0


class TestMeanGeneration:
    def test_excludes_naive(self) -> None:
        rng = np.random.default_rng(0)
        e1 = create_entry("a", "c", rng.standard_normal(384).astype(np.float32))
        e1.tier = Tier.NAIVE
        e1.generation = 10  # should be excluded

        e2 = create_entry("b", "c", rng.standard_normal(384).astype(np.float32))
        e2.tier = Tier.GC
        e2.generation = 4

        e3 = create_entry("c", "c", rng.standard_normal(384).astype(np.float32))
        e3.tier = Tier.MEMORY
        e3.generation = 8

        assert compute_mean_generation([e1, e2, e3]) == pytest.approx(6.0)

    def test_all_naive_returns_zero(self) -> None:
        rng = np.random.default_rng(0)
        e = create_entry("a", "c", rng.standard_normal(384).astype(np.float32))
        assert compute_mean_generation([e]) == 0.0


class TestNDCG:
    def test_perfect_ranking(self) -> None:
        relevance = {"a": 3, "b": 2, "c": 1}
        retrieved = ["a", "b", "c"]
        assert ndcg_at_k(retrieved, relevance, 3) == pytest.approx(1.0)

    def test_inverse_ranking(self) -> None:
        relevance = {"a": 3, "b": 2, "c": 1}
        retrieved = ["c", "b", "a"]
        score = ndcg_at_k(retrieved, relevance, 3)
        assert score < 1.0
        # Hand-calc: DCG = (2^1-1)/log2(2) + (2^2-1)/log2(3) + (2^3-1)/log2(4)
        dcg = 1.0 / 1.0 + 3.0 / math.log2(3) + 7.0 / math.log2(4)
        idcg = 7.0 / 1.0 + 3.0 / math.log2(3) + 1.0 / math.log2(4)
        assert score == pytest.approx(dcg / idcg, abs=1e-6)

    def test_no_relevant_docs(self) -> None:
        assert ndcg_at_k(["x", "y"], {}, 2) == 0.0

    def test_k_truncation(self) -> None:
        relevance = {"a": 3, "b": 2, "c": 1}
        retrieved = ["c", "b", "a", "d"]
        score_k2 = ndcg_at_k(retrieved, relevance, 2)
        score_k3 = ndcg_at_k(retrieved, relevance, 3)
        # k=3 includes "a" (rel=3) so should be higher or equal
        assert score_k3 >= score_k2


class TestRecall:
    def test_perfect_recall(self) -> None:
        relevance = {"a": 1, "b": 2}
        retrieved = ["a", "b", "c"]
        assert recall_at_k(retrieved, relevance, 3) == pytest.approx(1.0)

    def test_partial_recall(self) -> None:
        relevance = {"a": 1, "b": 2, "c": 1}
        retrieved = ["a", "x"]
        assert recall_at_k(retrieved, relevance, 2) == pytest.approx(1 / 3)

    def test_no_relevant(self) -> None:
        assert recall_at_k(["a", "b"], {}, 2) == 0.0
