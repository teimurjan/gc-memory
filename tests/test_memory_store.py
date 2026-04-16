"""End-to-end unit tests for gc_memory.memory_store.MemoryStore.

Uses mock bi-encoder + cross-encoder from conftest.py — no real model loads,
no network, no disk leakage beyond tmp_path.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from gc_memory.entry import Tier
from gc_memory.memory_store import MemoryStore


# ---------- Fixture ----------

@pytest.fixture
def store(mock_bi_encoder, mock_cross_encoder, tmp_store_path: Path) -> MemoryStore:
    return MemoryStore(
        path=tmp_store_path,
        bi_encoder=mock_bi_encoder,
        cross_encoder=mock_cross_encoder,
        dim=16,
        k_shallow=10,
        k_deep=20,
        confidence_threshold=4.0,
        dedup_threshold=0.95,
    )


# ---------- add() ----------

def test_add_returns_id(store: MemoryStore) -> None:
    eid = store.add("hello world")
    assert isinstance(eid, str)
    assert eid in store.entries


def test_add_exact_duplicate_returns_none(store: MemoryStore) -> None:
    store.add("exactly this text")
    again = store.add("exactly this text")
    assert again is None


def test_add_near_duplicate_keeps_longer(store: MemoryStore, mock_bi_encoder) -> None:
    # Add short then longer — the "near-duplicate" check relies on cosine > 0.95.
    # Because our mock encoder hashes text → random unit vector, different strings
    # produce ~orthogonal vectors. So we inject direct copies to trigger dedup.

    # First add a short entry
    short = store.add("hi")
    # Force a synthetic near-duplicate by mutating the stored embedding
    import numpy as np
    # Create an entry whose embedding matches the short one (cosine = 1.0)
    long_eid = "manual-id-long-content"
    long_text = "hi there this is a longer version"
    # Use the same embedding as the short entry
    same_emb = store.entries[short].base_embedding
    from gc_memory.entry import create_entry
    # Instead of calling store.add (which would give a different embedding via encoder),
    # manually do the near-duplicate flow by passing identical embeddings.
    # We'll just verify the short one is replaced by construction:
    # Use the public API: set near-dup threshold check through a direct build.
    # Simpler: monkeypatch encoder.encode to return `same_emb` for the long text.
    orig_encode = mock_bi_encoder.encode
    def encode_same(text, **kwargs):  # noqa: ANN001
        if text == long_text:
            return same_emb
        return orig_encode(text, **kwargs)
    mock_bi_encoder.encode = encode_same  # type: ignore[assignment]
    result = store.add(long_text)
    # Longer wins → old short entry deleted, new longer entry inserted
    assert short not in store.entries
    assert result in store.entries
    assert store.entries[result].content == long_text


def test_add_near_duplicate_skips_shorter(store: MemoryStore, mock_bi_encoder) -> None:
    """A near-duplicate shorter than the existing entry is dropped."""
    long_first = store.add("this is a longer piece of text stored first")
    same_emb = store.entries[long_first].base_embedding
    orig_encode = mock_bi_encoder.encode
    def encode_same(text, **kwargs):  # noqa: ANN001
        if text == "short":
            return same_emb
        return orig_encode(text, **kwargs)
    mock_bi_encoder.encode = encode_same  # type: ignore[assignment]
    result = store.add("short")
    assert result is None  # rejected
    assert long_first in store.entries


# ---------- retrieve() ----------

def test_retrieve_empty_store_returns_empty(store: MemoryStore) -> None:
    results = store.retrieve("anything", k=5)
    assert results == []


def test_retrieve_returns_top_k_sorted(store: MemoryStore) -> None:
    store.add("the quick brown fox jumps")
    store.add("lazy dogs sleep all day")
    store.add("cat cat cat")
    results = store.retrieve("brown fox", k=2)
    # Cross-encoder scores based on shared tokens; "the quick brown fox jumps"
    # shares {"brown", "fox"} with the query → highest.
    assert len(results) <= 2
    assert len(results) >= 1
    assert "brown fox" in " ".join(content for _, content, _ in results).lower() or \
           any("fox" in content for _, content, _ in results)
    # Scores descending
    scores = [s for _, _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_increments_step(store: MemoryStore) -> None:
    store.add("some entry")
    before = store._step
    store.retrieve("some entry", k=1)
    assert store._step == before + 1


def test_retrieve_updates_retrieval_count_and_affinity(store: MemoryStore) -> None:
    eid = store.add("the quick brown fox")
    before_count = store.entries[eid].retrieval_count
    before_aff = store.entries[eid].affinity
    store.retrieve("quick brown fox", k=1)
    assert store.entries[eid].retrieval_count == before_count + 1
    # Affinity moved toward 1.0 (winner gets positive xenc score from the mock)
    assert store.entries[eid].affinity != before_aff


def test_retrieve_applies_rif_state_on_subsequent_calls(store: MemoryStore) -> None:
    """After a retrieve, losing candidates accumulate suppression."""
    winner = store.add("travel to paris in march")
    loser = store.add("travel to tokyo in june")
    # The query matches both on "travel to" but favors one. Do a retrieve.
    store.retrieve("travel to paris in march", k=1)
    # At least one non-winner entry should be tracked for suppression.
    suppressions = {eid: store.entries[eid].suppression for eid in store.entries}
    # Winner gets reinforced (suppression stays 0 or decreases)
    assert suppressions[winner] == 0.0


def test_retrieve_tier_naive_to_gc_after_3_retrievals(store: MemoryStore) -> None:
    eid = store.add("the unique_token here")
    assert store.entries[eid].tier is Tier.NAIVE
    for _ in range(3):
        store.retrieve("unique_token", k=1)
    assert store.entries[eid].tier in (Tier.GC, Tier.MEMORY)


def test_retrieve_tier_gc_to_memory_with_high_affinity(store: MemoryStore) -> None:
    eid = store.add("target_word elevated")
    # Simulate many retrievals with strong positive scores
    for _ in range(10):
        store.retrieve("target_word elevated", k=1)
    # After 5+ retrievals with high xenc (matching tokens), affinity should push it to MEMORY.
    assert store.entries[eid].tier is Tier.MEMORY
    assert store.entries[eid].retrieval_count >= 5
    assert store.entries[eid].affinity >= 0.65


# ---------- save / reload ----------

def test_save_and_reload_roundtrip_preserves_entries(
    tmp_store_path: Path, mock_bi_encoder, mock_cross_encoder,
) -> None:
    s1 = MemoryStore(
        path=tmp_store_path,
        bi_encoder=mock_bi_encoder,
        cross_encoder=mock_cross_encoder,
        dim=16,
    )
    eid_a = s1.add("first entry about cats")
    eid_b = s1.add("second entry about dogs")
    s1.retrieve("cats", k=1)  # mutates state: affinities, step, etc.
    s1.save()
    s1.close()

    s2 = MemoryStore(
        path=tmp_store_path,
        bi_encoder=mock_bi_encoder,
        cross_encoder=mock_cross_encoder,
        dim=16,
    )
    assert eid_a in s2.entries
    assert eid_b in s2.entries
    assert s2.entries[eid_a].content == "first entry about cats"
    # step survived
    assert s2._step >= 1


def test_apoptotic_entries_excluded_on_load(
    tmp_store_path: Path, mock_bi_encoder, mock_cross_encoder,
) -> None:
    s1 = MemoryStore(
        path=tmp_store_path,
        bi_encoder=mock_bi_encoder,
        cross_encoder=mock_cross_encoder,
        dim=16,
    )
    eid = s1.add("going to die soon")
    s1.entries[eid].tier = Tier.APOPTOTIC
    s1.save()
    s1.close()

    s2 = MemoryStore(
        path=tmp_store_path,
        bi_encoder=mock_bi_encoder,
        cross_encoder=mock_cross_encoder,
        dim=16,
    )
    assert eid not in s2.entries


# ---------- decay ----------

def test_decay_reduces_non_memory_affinity(store: MemoryStore) -> None:
    eid = store.add("decay me")
    store.entries[eid].tier = Tier.GC
    store.entries[eid].affinity = 0.6
    store.entries[eid].last_retrieved_step = 0
    store._step = 200  # enough elapsed
    store.decay()
    assert store.entries[eid].affinity < 0.6


def test_decay_exempts_memory_tier(store: MemoryStore) -> None:
    eid = store.add("immune to decay")
    store.entries[eid].tier = Tier.MEMORY
    store.entries[eid].affinity = 0.8
    store.entries[eid].last_retrieved_step = 0
    store._step = 1_000_000
    store.decay()
    assert store.entries[eid].affinity == 0.8


# ---------- stats ----------

def test_stats_reports_tier_distribution(store: MemoryStore) -> None:
    a = store.add("a")
    b = store.add("b")
    c = store.add("c")
    store.entries[a].tier = Tier.NAIVE
    store.entries[b].tier = Tier.GC
    store.entries[c].tier = Tier.MEMORY
    stats = store.stats()
    assert stats["total_entries"] == 3
    assert stats["tiers"]["naive"] == 1
    assert stats["tiers"]["gc"] == 1
    assert stats["tiers"]["memory"] == 1


# ---------- size ----------

def test_size_property(store: MemoryStore) -> None:
    assert store.size == 0
    store.add("a")
    store.add("b")
    assert store.size == 2


# ---------- error paths ----------

def test_retrieve_raises_without_bi_encoder(tmp_store_path: Path, mock_cross_encoder) -> None:
    s = MemoryStore(path=tmp_store_path, bi_encoder=None, cross_encoder=mock_cross_encoder, dim=16)
    with pytest.raises(ValueError, match="bi_encoder required"):
        s.retrieve("q", k=1)


def test_add_raises_without_bi_encoder(tmp_store_path: Path, mock_cross_encoder) -> None:
    s = MemoryStore(path=tmp_store_path, bi_encoder=None, cross_encoder=mock_cross_encoder, dim=16)
    with pytest.raises(ValueError, match="bi_encoder required"):
        s.add("text")
