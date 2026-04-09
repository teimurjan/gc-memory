from __future__ import annotations

import numpy as np
import numpy.typing as npt

from gc_memory.config import Config
from gc_memory.entry import MemoryEntry
from gc_memory.store import GCMemoryStore


class StaticStore(GCMemoryStore):
    """Baseline: no mutation, no decay, no tier transitions."""

    def update_after_retrieval(
        self,
        query: npt.NDArray[np.float32],
        retrieved: list[tuple[MemoryEntry, float]],
        step: int,
    ) -> None:
        pass

    def run_decay(self, step: int) -> None:
        pass


class RandomMutationStore(GCMemoryStore):
    """Control: mutation with fixed sigma=0.025 regardless of affinity."""

    def __init__(
        self,
        entries: list[MemoryEntry],
        config: Config,
        rng: np.random.Generator,
        fixed_sigma: float = 0.025,
    ) -> None:
        super().__init__(entries, config, rng)
        self._fixed_sigma = fixed_sigma

    def _get_sigma(self, affinity: float) -> float:
        return self._fixed_sigma
