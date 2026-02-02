"""Graph stub used for pickled networkx compatibility."""
from __future__ import annotations

from typing import Dict, Iterable, Set, Tuple


class Graph:
    def __init__(self) -> None:
        self._nodes: Dict[object, dict] = {}
        self._edges: Set[Tuple[object, object]] = set()

    @property
    def nodes(self) -> Dict[object, dict]:
        return self._nodes

    def add_node(self, node: object, **_attrs: object) -> None:
        self._nodes.setdefault(node, {})

    def add_edge(self, u: object, v: object, **_attrs: object) -> None:
        self._nodes.setdefault(u, {})
        self._nodes.setdefault(v, {})
        self._edges.add((u, v))

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
                sub.add_edge(u, v)
        return sub
