"""Compatibility shim for the :mod:`annoy` package.

This module tries to import the real :mod:`annoy` implementation when it is
available in the environment.  If that fails (for instance when the C extension
was not compiled) a lightweight in-memory fallback is provided that mimics the
public API of :class:`annoy.AnnoyIndex` sufficiently for the needs of this
project.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import importlib
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence

__all__ = ["AnnoyIndex"]

try:  # pragma: no cover - exercised when real package installed
    AnnoyIndex = importlib.import_module("annoy").AnnoyIndex  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - fallback used in test environment

    @dataclass
    class _Node:
        index: int
        threshold: float = 0.0
        left: "_Node | None" = None
        right: "_Node | None" = None

    class AnnoyIndex:
        """Simplified in-memory alternative to the real Annoy implementation."""

        def __init__(self, dim: int, metric: str = "angular") -> None:
            self.dim = int(dim)
            self.metric = metric
            self.items: Dict[int, List[float]] = {}
            self._trees: List[_Node] = []
            self._built = False

        # --------------------------------------------------
        def add_item(self, idx: int, vec: Sequence[float]) -> None:
            if len(vec) != self.dim:
                raise ValueError("Vector dimensionality mismatch")
            self.items[int(idx)] = [float(x) for x in vec]

        # --------------------------------------------------
        def build(self, trees: int = 10) -> None:  # noqa: ARG002 - parity with API
            self._trees = [
                self._build_tree(list(self.items.keys()), random.Random(i))
                for i in range(max(1, trees))
            ]
            self._built = True

        def _build_tree(self, indices: List[int], rng: random.Random) -> _Node | None:
            if not indices:
                return None
            idx = rng.choice(indices)
            if len(indices) == 1:
                return _Node(index=idx)
            dist_pairs = [
                (other, self._distance(self.items[idx], self.items[other]))
                for other in indices
                if other != idx
            ]
            dist_pairs.sort(key=lambda x: x[1])
            median = dist_pairs[len(dist_pairs) // 2][1]
            left = [i for i, d in dist_pairs if d <= median]
            right = [i for i, d in dist_pairs if d > median]
            return _Node(
                index=idx,
                threshold=median,
                left=self._build_tree(left, rng),
                right=self._build_tree(right, rng),
            )

        # --------------------------------------------------
        def _serialize_tree(self, node: _Node | None) -> Dict:
            if node is None:
                return {}
            return {
                "index": node.index,
                "threshold": node.threshold,
                "left": self._serialize_tree(node.left),
                "right": self._serialize_tree(node.right),
            }

        def _deserialize_tree(self, data: Dict | None) -> _Node | None:
            if not data:
                return None
            return _Node(
                index=int(data["index"]),
                threshold=float(data.get("threshold", 0.0)),
                left=self._deserialize_tree(data.get("left")),
                right=self._deserialize_tree(data.get("right")),
            )

        def save(self, path: str | Path) -> None:
            data = {
                "dim": self.dim,
                "metric": self.metric,
                "items": self.items,
            }
            if self._built:
                data["trees"] = [self._serialize_tree(t) for t in self._trees]
            with Path(path).open("w", encoding="utf-8") as fh:
                json.dump(data, fh)

        # --------------------------------------------------
        def load(self, path: str | Path) -> None:
            with Path(path).open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.dim = int(data.get("dim", self.dim))
            self.metric = data.get("metric", self.metric)
            self.items = {int(k): list(map(float, v)) for k, v in data.get("items", {}).items()}
            self._trees = [self._deserialize_tree(t) for t in data.get("trees", [])]
            self._built = bool(self._trees)

        # --------------------------------------------------
        def _distance(self, v1: Sequence[float], v2: Sequence[float]) -> float:
            if self.metric == "angular":
                dot = sum(a * b for a, b in zip(v1, v2))
                norm1 = math.sqrt(sum(a * a for a in v1))
                norm2 = math.sqrt(sum(b * b for b in v2))
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                cosine = max(-1.0, min(1.0, dot / (norm1 * norm2)))
                return math.acos(cosine)
            if self.metric == "euclidean":
                return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
            if self.metric == "manhattan":
                return sum(abs(a - b) for a, b in zip(v1, v2))
            if self.metric == "hamming":
                return sum(0 if a == b else 1 for a, b in zip(v1, v2))
            if self.metric == "dot":
                return -sum(a * b for a, b in zip(v1, v2))
            raise ValueError(f"Unsupported metric: {self.metric}")

        # --------------------------------------------------
        def get_nns_by_vector(
            self,
            vector: Sequence[float],
            n: int,
            search_k: int | None = None,
            include_distances: bool = False,
        ) -> List[int] | tuple[List[int], List[float]]:
            if not self._built:
                raise RuntimeError("Index not built")
            if len(vector) != self.dim:
                raise ValueError("Vector dimensionality mismatch")

            search_k = search_k or len(self.items) * 2
            distances = []
            for idx, vec in heapq.nsmallest(
                search_k,
                ((i, self._distance(vector, v)) for i, v in self.items.items()),
                key=lambda x: x[1],
            ):
                distances.append((idx, vec))

            neighbours = [idx for idx, _ in distances[:n]]
            if include_distances:
                return neighbours, [dist for _, dist in distances[:n]]
            return neighbours

        # --------------------------------------------------
        def get_item_vector(self, idx: int) -> List[float]:
            return self.items[int(idx)]

        # --------------------------------------------------
        def unload(self) -> None:  # noqa: D401 - compatibility shim
            """Compatibility shim matching the real Annoy API."""

            self.items.clear()
            self._trees.clear()
            self._built = False

        # --------------------------------------------------
        def __len__(self) -> int:
            return len(self.items)
