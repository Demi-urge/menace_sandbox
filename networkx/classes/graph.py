"""Graph stub used for pickled networkx compatibility."""
from __future__ import annotations

from typing import Dict, Iterable, Set, Tuple


class Graph:
    def __init__(self) -> None:
        self._nodes: Dict[object, dict] = {}
        self._edges: Set[Tuple[object, object]] = set()
        self._adj: Dict[object, Dict[object, dict]] = {}
        self.graph: dict = {}

    @property
    def nodes(self) -> Dict[object, dict]:
        return self._nodes

    def add_node(self, node: object, **_attrs: object) -> None:
        self._nodes.setdefault(node, {})
        self._adj.setdefault(node, {})

    def add_edge(self, u: object, v: object, **_attrs: object) -> None:
        self.add_node(u)
        self.add_node(v)
        adjacency = self._adj.setdefault(u, {})
        edge_attrs = adjacency.setdefault(v, {})
        if _attrs:
            edge_attrs.update(_attrs)
        self._edges.add((u, v))

    def has_edge(self, u: object, v: object) -> bool:
        return v in self._adj.get(u, {})

    def __contains__(self, node: object) -> bool:
        return node in self._nodes

    def __iter__(self):
        return iter(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def __getitem__(self, node: object) -> Dict[object, dict]:
        return self._adj[node]

    def edges(self, data: bool = False):
        for u, v in self._edges:
            if data:
                yield (u, v, self._adj.get(u, {}).get(v, {}))
            else:
                yield (u, v)

    def number_of_nodes(self) -> int:
        return len(self._nodes)

    def number_of_edges(self) -> int:
        return len(self._edges)

    def subgraph(self, nodes: Iterable[object]) -> "Graph":
        sub = Graph()
        nodes_set = set(nodes)
        for node in nodes_set:
            sub.add_node(node)
        for u, v in self._edges:
            if u in nodes_set and v in nodes_set:
                attrs = self._adj.get(u, {}).get(v, {}).copy()
                sub.add_edge(u, v, **attrs)
        return sub
