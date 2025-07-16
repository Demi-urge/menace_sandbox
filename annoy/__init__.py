"""Compatibility shim for the :mod:`annoy` package.

This module tries to use the real `annoy` implementation when it is available
in the environment.  If that fails (for instance when the C extension was not
compiled) a lightweight in-memory fallback is provided that mimics the public
API of :class:`annoy.AnnoyIndex` sufficiently for the needs of this project.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import importlib
import importlib.machinery
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence


def _try_import_real_annoy() -> "AnnoyIndex | None":
    """Attempt to load the real :mod:`annoy` implementation.

    The local repository also provides a module named ``annoy`` which would
    normally shadow an installed package.  To work around this we manually
    search ``sys.path`` for any other ``annoy`` package and load it if found.
    """

    this_dir = os.path.dirname(__file__)
    for path in sys.path:
        if os.path.abspath(path) == os.path.abspath(os.path.dirname(this_dir)):
            continue
        spec = importlib.machinery.PathFinder.find_spec("annoy", [path])
        if spec and spec.origin and os.path.abspath(spec.origin) != os.path.abspath(__file__):
            module = importlib.util.module_from_spec(spec)
            prev = sys.modules.get("annoy")
            sys.modules["annoy"] = module
            assert spec.loader is not None
            try:
                spec.loader.exec_module(module)  # type: ignore[arg-type]
                return module.AnnoyIndex  # type: ignore[attr-defined]
            except Exception:
                pass
            finally:
                if prev is not None:
                    sys.modules["annoy"] = prev
                else:
                    sys.modules.pop("annoy", None)
            return None
    return None


_REAL_ANNOY = _try_import_real_annoy()


if _REAL_ANNOY is not None:
    AnnoyIndex = _REAL_ANNOY  # pragma: no cover - delegate to real library
else:

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
            if self.metric == "euclidean":
                return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
            if self.metric == "manhattan":
                return sum(abs(a - b) for a, b in zip(v1, v2))
            if self.metric == "hamming":
                return sum(0 if a == b else 1 for a, b in zip(v1, v2))
            if self.metric == "dot":
                return -sum(a * b for a, b in zip(v1, v2))
            # angular / cosine
            dot = sum(a * b for a, b in zip(v1, v2))
            norm1 = math.sqrt(sum(a * a for a in v1))
            norm2 = math.sqrt(sum(b * b for b in v2))
            if not norm1 or not norm2:
                return float("inf")
            cos_sim = dot / (norm1 * norm2)
            return 1 - cos_sim

        # --------------------------------------------------
        def _search_tree(
            self, node: _Node | None, vec: Sequence[float], heap: List[tuple[int, float]], search_k: int
        ) -> None:
            if node is None:
                return
            dist = self._distance(vec, self.items[node.index])
            if len(heap) < search_k:
                heapq.heappush(heap, (dist, node.index))
            elif dist < heap[0][0]:
                heapq.heapreplace(heap, (dist, node.index))
            if node.left is None and node.right is None:
                return
            if dist < node.threshold:
                self._search_tree(node.left, vec, heap, search_k)
                if len(heap) < search_k or node.threshold - dist <= heap[0][0]:
                    self._search_tree(node.right, vec, heap, search_k)
            else:
                self._search_tree(node.right, vec, heap, search_k)
                if len(heap) < search_k or dist - node.threshold <= heap[0][0]:
                    self._search_tree(node.left, vec, heap, search_k)

        # --------------------------------------------------
        def get_nns_by_vector(self, vec: Sequence[float], n: int) -> List[int]:
            if not self.items:
                return []
            if len(vec) != self.dim:
                raise ValueError("Vector dimensionality mismatch")
            if not self._built:
                self.build()
            search_k = max(n * len(self._trees), n)
            heap: List[tuple[int, float]] = []
            for tree in self._trees:
                self._search_tree(tree, vec, heap, search_k)
            heap.sort()
            seen = set()
            result = []
            for dist, idx in heap:
                if idx not in seen:
                    seen.add(idx)
                    result.append(idx)
                if len(result) >= n:
                    break
            return result


__all__ = ["AnnoyIndex"]


