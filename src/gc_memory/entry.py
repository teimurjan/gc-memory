from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt


class Tier(Enum):
    NAIVE = "naive"
    GC = "gc"
    MEMORY = "memory"
    APOPTOTIC = "apoptotic"


@dataclass
class MemoryEntry:
    id: str
    content: str
    embedding: npt.NDArray[np.float32]
    original_embedding: npt.NDArray[np.float32]
    affinity: float = 0.5
    retrieval_count: int = 0
    generation: int = 0
    last_retrieved_step: int = 0
    tier: Tier = Tier.NAIVE


def create_entry(
    entry_id: str,
    content: str,
    embedding: npt.NDArray[np.float32],
) -> MemoryEntry:
    """Create a MemoryEntry with unit-normalized embedding and frozen original copy."""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        raise ValueError(f"Zero-norm embedding for entry {entry_id}")
    normalized = (embedding / norm).astype(np.float32)
    return MemoryEntry(
        id=entry_id,
        content=content,
        embedding=normalized.copy(),
        original_embedding=normalized.copy(),
    )
