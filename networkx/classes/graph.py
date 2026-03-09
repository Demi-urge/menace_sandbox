"""Graph stub used for pickled networkx compatibility."""
from __future__ import annotations

from typing import Dict, Iterable, Set, Tuple


class Graph:
    def __init__(self) -> None:
        self._nodes: Dict[object, dict] = {}
        self._edges: Set[Tuple[object, object]] = set()
        self._adj: Dict[object, Dict[object, dict]] = {}
        self.graph: dict = {}

    def _ensure_compat_storage(self) -> None:
        """Backfill storage attrs for objects restored from real networkx pickles.

        Real networkx graphs typically use ``_node`` (singular) for node attrs,
        while this lightweight stub stores node attrs in ``_nodes``. When a graph
        object is unpickled from real networkx, ``__init__`` is bypassed and only
        serialized attributes are restored, so ``_nodes`` may be missing.
        """

        if not hasattr(self, "_adj"):
            self._adj = {}
        if not hasattr(self, "_edges"):
            self._edges = set()
        if not hasattr(self, "graph"):
            self.graph = {}

        if not hasattr(self, "_nodes"):
            existing_nodes = getattr(self, "_node", None)
            if isinstance(existing_nodes, dict):
                self._nodes = existing_nodes
            else:
                self._nodes = {}

        # Keep aliases in sync for compatibility with code expecting either name.
        self._node = self._nodes

    @property
    def nodes(self) -> Dict[object, dict]:
        self._ensure_compat_storage()
        return self._nodes

    def add_node(self, node: object, **_attrs: object) -> None:
        self._ensure_compat_storage()
        self._nodes.setdefault(node, {})
        self._adj.setdefault(node, {})

    def add_edge(self, u: object, v: object, **_attrs: object) -> None:
        self._ensure_compat_storage()
        self.add_node(u)
        self.add_node(v)
        adjacency = self._adj.setdefault(u, {})
        edge_attrs = adjacency.setdefault(v, {})
        if _attrs:
            edge_attrs.update(_attrs)
        self._edges.add((u, v))

    def has_edge(self, u: object, v: object) -> bool:
        self._ensure_compat_storage()
        return v in self._adj.get(u, {})

    def __contains__(self, node: object) -> bool:
        self._ensure_compat_storage()
        return node in self._adj

    def __iter__(self):
        self._ensure_compat_storage()
        return iter(self._nodes)

    def has_node(self, node: object) -> bool:
        self._ensure_compat_storage()
        return node in self._adj

    def __len__(self) -> int:
        self._ensure_compat_storage()
        return len(self._nodes)

    def __getitem__(self, node: object) -> Dict[object, dict]:
        self._ensure_compat_storage()
        return self._adj[node]

    def edges(self, data: bool = False):
        self._ensure_compat_storage()
        for u, v in self._edges:
            if data:
                yield (u, v, self._adj.get(u, {}).get(v, {}))
            else:
                yield (u, v)

    def number_of_nodes(self) -> int:
        self._ensure_compat_storage()
        return len(self._nodes)

    def number_of_edges(self) -> int:
        self._ensure_compat_storage()
        return len(self._edges)

    def subgraph(self, nodes: Iterable[object]) -> "Graph":
        self._ensure_compat_storage()
        sub = Graph()
        nodes_set = set(nodes)
        for node in nodes_set:
            sub.add_node(node)
        for u, v in self._edges:
            if u in nodes_set and v in nodes_set:
                attrs = self._adj.get(u, {}).get(v, {}).copy()
                sub.add_edge(u, v, **attrs)
        return sub
