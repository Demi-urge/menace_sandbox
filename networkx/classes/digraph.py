"""Directed graph stub for networkx."""
from __future__ import annotations

from .graph import Graph


class DiGraph(Graph):
    def predecessors(self, node: object):
        for u, v in self._edges:
            if v == node:
                yield u

    def to_undirected(self) -> Graph:
        undirected = Graph()
        for node in self._nodes:
            undirected.add_node(node)
        for u, v in self._edges:
            attrs = self._adj.get(u, {}).get(v, {}).copy()
            undirected.add_edge(u, v, **attrs)
            undirected.add_edge(v, u, **attrs)
        return undirected
