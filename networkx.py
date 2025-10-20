"""Minimal stub of :mod:`networkx` for dependency-light test environments."""
from __future__ import annotations

from typing import Any, Iterable, Iterator, Optional


class _EdgeView:
    """Callable edge view that mirrors the subset of ``networkx`` we rely on."""

    def __init__(self, graph: "DiGraph") -> None:
        self._graph = graph

    def _iter_edges(
        self, data: bool = False, default: Optional[Any] = None
    ) -> Iterator[tuple[Any, Any] | tuple[Any, Any, Any]]:
        for u, v in self._graph._edges:
            if data:
                yield (u, v, {} if default is None else default)
            else:
                yield (u, v)

    def __iter__(self) -> Iterator[tuple[Any, Any]]:
        return self._iter_edges()

    def __call__(
        self, nbunch: Optional[Iterable[Any]] = None, data: bool = False, default: Optional[Any] = None
    ) -> Iterator[tuple[Any, Any] | tuple[Any, Any, Any]]:
        del nbunch  # unused in the stub implementation
        return self._iter_edges(data=data, default=default)

    def data(
        self,
        nbunch: Optional[Iterable[Any]] = None,
        data: bool = True,
        default: Optional[Any] = None,
    ) -> Iterator[tuple[Any, Any, Any]]:
        del nbunch  # unused in the stub implementation
        return self._iter_edges(data=data, default=default)  # type: ignore[return-value]


class DiGraph:
    def __init__(self) -> None:
        self._edges: set[tuple[Any, Any]] = set()
        self._nodes: set[Any] = set()
        self._edge_view = _EdgeView(self)

    def add_edge(self, u: Any, v: Any) -> None:
        self._nodes.add(u)
        self._nodes.add(v)
        self._edges.add((u, v))

    def add_node(self, node: Any) -> None:
        self._nodes.add(node)

    @property
    def nodes(self):  # type: ignore[override]
        class _NodeView(dict):
            def __init__(self, nodes: set[Any]):
                super().__init__((node, {}) for node in nodes)

        return _NodeView(self._nodes)

    @property
    def edges(self) -> _EdgeView:  # type: ignore[override]
        return self._edge_view


def __getattr__(name: str) -> object:  # pragma: no cover - defensive stub
    raise RuntimeError(
        f"networkx functionality '{name}' is unavailable in this environment"
    )
