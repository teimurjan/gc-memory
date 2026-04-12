"""GC routing index: mutable query-to-entry associations.

Sits on top of the static retrieval pipeline. Learns which entries
are relevant for which query patterns from cross-encoder feedback.
Each (query_cluster, entry) pair has a GC lifecycle: naive → gc → memory.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import numpy.typing as npt


class RouteTier(Enum):
    NAIVE = "naive"
    GC = "gc"
    MEMORY = "memory"
    APOPTOTIC = "apoptotic"


@dataclass
class RouteEntry:
    entry_id: str
    affinity: float = 0.0
    retrieval_count: int = 0
    tier: RouteTier = RouteTier.NAIVE
    last_seen_step: int = 0


class RoutingIndex:
    """GC-managed routing associations between query clusters and entries.

    On retrieval: find nearest query cluster, return associated entry IDs.
    On feedback: reinforce/weaken associations based on cross-encoder scores.
    On exploration: add new associations from deep mining.
    """

    def __init__(
        self,
        max_entries_per_cluster: int = 50,
        promote_gc_threshold: int = 3,
        promote_memory_affinity: float = 0.6,
        apoptosis_affinity: float = 0.1,
        apoptosis_idle_steps: int = 500,
        affinity_alpha: float = 0.3,
        xenc_positive_threshold: float = 0.0,
    ) -> None:
        self.max_entries_per_cluster = max_entries_per_cluster
        self.promote_gc_threshold = promote_gc_threshold
        self.promote_memory_affinity = promote_memory_affinity
        self.apoptosis_affinity = apoptosis_affinity
        self.apoptosis_idle_steps = apoptosis_idle_steps
        self.affinity_alpha = affinity_alpha
        self.xenc_positive_threshold = xenc_positive_threshold

        # Clusters: list of (centroid_embedding, {entry_id: RouteEntry})
        self._centroids: list[npt.NDArray[np.float32]] = []
        self._clusters: list[dict[str, RouteEntry]] = []
        self._step = 0

    def lookup(self, query_emb: npt.NDArray[np.float32], top_k: int = 20) -> list[str]:
        """Return entry IDs from the nearest cluster's gc+memory tier routes."""
        if not self._centroids:
            return []
        cluster_idx = self._nearest_cluster(query_emb)
        if cluster_idx is None:
            return []
        routes = self._clusters[cluster_idx]
        # Return gc and memory tier entries, sorted by affinity
        active = [
            (r.entry_id, r.affinity)
            for r in routes.values()
            if r.tier in (RouteTier.GC, RouteTier.MEMORY)
        ]
        active.sort(key=lambda x: x[1], reverse=True)
        return [eid for eid, _ in active[:top_k]]

    def update(
        self,
        query_emb: npt.NDArray[np.float32],
        scored_entries: list[tuple[str, float]],
    ) -> None:
        """Update routing associations from cross-encoder feedback.

        scored_entries: list of (entry_id, xenc_score) from the reranker.
        """
        self._step += 1
        cluster_idx = self._nearest_cluster(query_emb, create_if_missing=True)
        if cluster_idx is None:
            return
        routes = self._clusters[cluster_idx]

        for entry_id, score in scored_entries:
            if entry_id not in routes:
                if score > self.xenc_positive_threshold:
                    routes[entry_id] = RouteEntry(
                        entry_id=entry_id, affinity=0.0, last_seen_step=self._step,
                    )
                else:
                    continue

            route = routes[entry_id]
            route.retrieval_count += 1
            route.last_seen_step = self._step

            # EMA affinity from normalized xenc score
            norm = 1.0 / (1.0 + math.exp(-score))
            route.affinity = (
                (1.0 - self.affinity_alpha) * route.affinity
                + self.affinity_alpha * norm
            )

        # Tier transitions
        for route in list(routes.values()):
            if route.tier == RouteTier.NAIVE and route.retrieval_count >= self.promote_gc_threshold:
                route.tier = RouteTier.GC
            elif route.tier == RouteTier.GC and route.affinity >= self.promote_memory_affinity:
                route.tier = RouteTier.MEMORY
            if (
                route.tier not in (RouteTier.MEMORY, RouteTier.APOPTOTIC)
                and route.affinity < self.apoptosis_affinity
                and self._step - route.last_seen_step > self.apoptosis_idle_steps
            ):
                route.tier = RouteTier.APOPTOTIC

        # Evict apoptotic and cap size
        routes_list = sorted(routes.values(), key=lambda r: r.affinity, reverse=True)
        self._clusters[cluster_idx] = {
            r.entry_id: r
            for r in routes_list[:self.max_entries_per_cluster]
            if r.tier != RouteTier.APOPTOTIC
        }

        # Update centroid as EMA of query embeddings assigned to this cluster
        alpha = 0.1
        centroid = self._centroids[cluster_idx]
        self._centroids[cluster_idx] = (
            ((1.0 - alpha) * centroid + alpha * query_emb)
            / np.linalg.norm((1.0 - alpha) * centroid + alpha * query_emb)
        ).astype(np.float32)

    def explore(
        self,
        query_emb: npt.NDArray[np.float32],
        deep_scored: list[tuple[str, float]],
    ) -> None:
        """Add new route entries from deep mining results."""
        cluster_idx = self._nearest_cluster(query_emb, create_if_missing=True)
        if cluster_idx is None:
            return
        routes = self._clusters[cluster_idx]
        for entry_id, score in deep_scored:
            if score > self.xenc_positive_threshold and entry_id not in routes:
                routes[entry_id] = RouteEntry(
                    entry_id=entry_id,
                    affinity=1.0 / (1.0 + math.exp(-score)),
                    last_seen_step=self._step,
                )

    def decay(self, factor: float = 0.99) -> None:
        """Decay all non-memory route affinities."""
        for routes in self._clusters:
            for route in routes.values():
                if route.tier != RouteTier.MEMORY:
                    route.affinity *= factor

    def _nearest_cluster(
        self,
        query_emb: npt.NDArray[np.float32],
        create_if_missing: bool = False,
        similarity_threshold: float = 0.7,
    ) -> int | None:
        if not self._centroids:
            if create_if_missing:
                self._centroids.append(query_emb.copy())
                self._clusters.append({})
                return 0
            return None

        sims = np.array([float(np.dot(query_emb, c)) for c in self._centroids])
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= similarity_threshold:
            return best_idx

        if create_if_missing:
            self._centroids.append(query_emb.copy())
            self._clusters.append({})
            return len(self._centroids) - 1

        return best_idx if best_sim > 0.3 else None

    @property
    def num_clusters(self) -> int:
        return len(self._centroids)

    @property
    def num_routes(self) -> int:
        return sum(len(c) for c in self._clusters)

    @property
    def num_memory_routes(self) -> int:
        return sum(
            1 for c in self._clusters
            for r in c.values()
            if r.tier == RouteTier.MEMORY
        )
