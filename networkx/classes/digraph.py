"""Directed graph stub for networkx."""
from __future__ import annotations

from .graph import Graph


class DiGraph(Graph):
    def to_undirected(self) -> Graph:
        undirected = Graph()
        for node in self._nodes:
            undirected.add_node(node)
        for u, v in self._edges:
            undirected.add_edge(u, v)
            undirected.add_edge(v, u)
        return undirected
