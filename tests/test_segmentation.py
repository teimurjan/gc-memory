from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from gc_memory.config import Config
from gc_memory.entry import Tier, create_entry
from gc_memory.segmentation import (
    find_merge_candidates,
    merge_entries,
    should_split,
    split_entry,
    split_sentences,
)


@pytest.fixture
def config() -> Config:
    return Config()


def _mock_bi_encoder() -> MagicMock:
    """Mock bi-encoder that returns random unit vectors."""
    mock = MagicMock()
    rng = np.random.default_rng(42)
    def encode(text: str, normalize_embeddings: bool = True) -> np.ndarray:
        v = rng.standard_normal(384).astype(np.float32)
        return v / np.linalg.norm(v)
    mock.encode = encode
    return mock


class TestSplitSentences:
    def test_single_sentence(self) -> None:
        assert split_sentences("Hello world") == ["Hello world"]

    def test_multiple_sentences(self) -> None:
        result = split_sentences("First sentence. Second sentence. Third one.")
        assert len(result) == 3

    def test_splits_on_uppercase_after_period(self) -> None:
        # Simple regex splits on ". [A-Z]" boundaries
        result = split_sentences("First sentence here. Second sentence here.")
        assert len(result) == 2


class TestShouldSplit:
    def test_triggers_on_gc_low_affinity_long_content(self, config: Config) -> None:
        rng = np.random.default_rng(0)
        entry = create_entry("e0", "x" * 300, rng.standard_normal(384).astype(np.float32))
        entry.tier = Tier.GC
        entry.affinity = 0.2
        assert should_split(entry, config) is True

    def test_skips_naive_tier(self, config: Config) -> None:
        rng = np.random.default_rng(0)
        entry = create_entry("e0", "x" * 300, rng.standard_normal(384).astype(np.float32))
        entry.tier = Tier.NAIVE
        entry.affinity = 0.2
        assert should_split(entry, config) is False

    def test_skips_high_affinity(self, config: Config) -> None:
        rng = np.random.default_rng(0)
        entry = create_entry("e0", "x" * 300, rng.standard_normal(384).astype(np.float32))
        entry.tier = Tier.GC
        entry.affinity = 0.8
        assert should_split(entry, config) is False

    def test_skips_short_content(self, config: Config) -> None:
        rng = np.random.default_rng(0)
        entry = create_entry("e0", "Short text.", rng.standard_normal(384).astype(np.float32))
        entry.tier = Tier.GC
        entry.affinity = 0.2
        assert should_split(entry, config) is False


class TestSplitEntry:
    def test_splits_long_entry(self) -> None:
        rng = np.random.default_rng(0)
        content = "This is the first sentence. This is the second sentence. And here is the third one."
        entry = create_entry("e0", content, rng.standard_normal(384).astype(np.float32), session_id="s1", turn_idx=2)
        mock_enc = _mock_bi_encoder()
        result = split_entry(entry, mock_enc)
        assert len(result) == 3
        assert all(r.session_id == "s1" for r in result)

    def test_no_split_for_single_sentence(self) -> None:
        rng = np.random.default_rng(0)
        entry = create_entry("e0", "Just one sentence here", rng.standard_normal(384).astype(np.float32))
        mock_enc = _mock_bi_encoder()
        result = split_entry(entry, mock_enc)
        assert result == []


class TestMergeEntries:
    def test_merge_combines_content(self) -> None:
        rng = np.random.default_rng(0)
        a = create_entry("a", "First turn content.", rng.standard_normal(384).astype(np.float32), session_id="s1", turn_idx=0)
        b = create_entry("b", "Second turn content.", rng.standard_normal(384).astype(np.float32), session_id="s1", turn_idx=1)
        a.affinity = 0.8
        b.affinity = 0.7
        mock_enc = _mock_bi_encoder()
        merged = merge_entries(a, b, mock_enc)
        assert "First turn content." in merged.content
        assert "Second turn content." in merged.content
        assert merged.session_id == "s1"
        assert merged.tier == Tier.GC


class TestFindMergeCandidates:
    def test_finds_adjacent_high_affinity(self, config: Config) -> None:
        rng = np.random.default_rng(0)
        a = create_entry("a", "Turn 1", rng.standard_normal(384).astype(np.float32), session_id="s1", turn_idx=0)
        b = create_entry("b", "Turn 2", rng.standard_normal(384).astype(np.float32), session_id="s1", turn_idx=1)
        a.affinity = 0.8
        b.affinity = 0.7
        retrieved = [(a, 1.0), (b, 0.9)]
        all_entries = {a.id: a, b.id: b}
        candidates = find_merge_candidates(retrieved, all_entries, config)
        assert len(candidates) == 1

    def test_skips_low_affinity(self, config: Config) -> None:
        rng = np.random.default_rng(0)
        a = create_entry("a", "Turn 1", rng.standard_normal(384).astype(np.float32), session_id="s1", turn_idx=0)
        b = create_entry("b", "Turn 2", rng.standard_normal(384).astype(np.float32), session_id="s1", turn_idx=1)
        a.affinity = 0.3
        b.affinity = 0.3
        retrieved = [(a, 1.0), (b, 0.9)]
        all_entries = {a.id: a, b.id: b}
        candidates = find_merge_candidates(retrieved, all_entries, config)
        assert len(candidates) == 0
