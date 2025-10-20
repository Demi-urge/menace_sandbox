"""Minimal stub of :mod:`networkx` for dependency-light test environments."""
from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Iterable, Iterator, Optional

__path__: list[str] = []


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


class _NodeView(MutableMapping[str, dict[str, Any]]):
    def __init__(self, graph: "DiGraph") -> None:
        self._graph = graph

    def __getitem__(self, node: Any) -> dict[str, Any]:
        if node not in self._graph._nodes:
            raise KeyError(node)
        return self._graph._node_attrs.setdefault(node, {})

    def __setitem__(self, node: Any, value: Iterable[tuple[str, Any]] | dict[str, Any]) -> None:
        self._graph._nodes.add(node)
        if isinstance(value, dict):
            data = dict(value)
        else:
            data = dict(value)
        self._graph._node_attrs[node] = data

    def __delitem__(self, node: Any) -> None:
        self._graph._nodes.discard(node)
        self._graph._node_attrs.pop(node, None)
        self._graph._edges = {edge for edge in self._graph._edges if node not in edge}

    def __iter__(self) -> Iterator[Any]:
        return iter(self._graph._nodes)

    def __len__(self) -> int:
        return len(self._graph._nodes)

    def get(self, node: Any, default: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
        if node not in self._graph._nodes:
            return default
        return self._graph._node_attrs.setdefault(node, {} if default is None else dict(default))

    def setdefault(
        self, node: Any, default: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        if node not in self._graph._nodes:
            self._graph._nodes.add(node)
        return self._graph._node_attrs.setdefault(node, {} if default is None else dict(default))

    def items(self) -> Iterator[tuple[Any, dict[str, Any]]]:
        for node in self._graph._nodes:
            yield node, self._graph._node_attrs.setdefault(node, {})

    def data(
        self,
        data: bool = False,
        default: Optional[dict[str, Any]] = None,
    ) -> Iterator[tuple[Any, dict[str, Any]] | Any]:
        for node in self._graph._nodes:
            if data:
                yield node, self._graph._node_attrs.setdefault(
                    node, {} if default is None else dict(default)
                )
            else:
                yield node


class DiGraph:
    def __init__(self) -> None:
        self._edges: set[tuple[Any, Any]] = set()
        self._nodes: set[Any] = set()
        self._node_attrs: dict[Any, dict[str, Any]] = {}
        self._edge_view = _EdgeView(self)
        self._node_view = _NodeView(self)

    def add_edge(self, u: Any, v: Any) -> None:
        self.add_node(u)
        self.add_node(v)
        self._edges.add((u, v))

    def add_node(self, node: Any) -> None:
        self._nodes.add(node)
        self._node_attrs.setdefault(node, {})

    def clear(self) -> None:
        """Remove all nodes, edges, and associated attributes from the graph."""
        self._edges.clear()
        self._nodes.clear()
        self._node_attrs.clear()

    @property
    def nodes(self) -> _NodeView:  # type: ignore[override]
        return self._node_view

    @property
    def edges(self) -> _EdgeView:  # type: ignore[override]
        return self._edge_view


def __getattr__(name: str) -> object:  # pragma: no cover - defensive stub
    raise RuntimeError(
        f"networkx functionality '{name}' is unavailable in this environment"
    )
