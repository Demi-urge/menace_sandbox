"""Minimal stub of :mod:`networkx` for dependency-light test environments."""
from __future__ import annotations

from typing import Any


class DiGraph:
    def __init__(self) -> None:
        self._edges: set[tuple[Any, Any]] = set()
        self._nodes: set[Any] = set()

    def add_edge(self, u: Any, v: Any) -> None:
        self._nodes.add(u)
        self._nodes.add(v)
        self._edges.add((u, v))

    def edges(self) -> list[tuple[Any, Any]]:
        return list(self._edges)

    def add_node(self, node: Any) -> None:
        self._nodes.add(node)

    @property
    def nodes(self):  # type: ignore[override]
        class _NodeView(dict):
            def __init__(self, nodes: set[Any]):
                super().__init__((node, {}) for node in nodes)

        return _NodeView(self._nodes)


def __getattr__(name: str) -> object:  # pragma: no cover - defensive stub
    raise RuntimeError(
        f"networkx functionality '{name}' is unavailable in this environment"
    )
