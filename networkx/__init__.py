"""Minimal networkx stub for sandbox environments without the dependency."""
from __future__ import annotations

from typing import Iterable, List

from .classes.graph import Graph
from .classes.digraph import DiGraph


def number_weakly_connected_components(graph: Graph) -> int:
    return 1 if graph.number_of_nodes() else 0


def connected_components(graph: Graph) -> List[List[object]]:
    nodes = list(getattr(graph, "_nodes", {}).keys())
    return [nodes] if nodes else []


def diameter(_graph: Graph) -> int:
    return 0


def single_source_shortest_path_length(
    graph: Graph, source: object, cutoff: int | None = None, depth: int | None = None
) -> dict[object, int]:
    if source not in getattr(graph, "_nodes", set()):
        return {}
    limit = cutoff if cutoff is not None else depth
    if limit is None:
        limit = 1
    return {source: 0}


__all__ = [
    "DiGraph",
    "Graph",
    "connected_components",
    "diameter",
    "number_weakly_connected_components",
    "single_source_shortest_path_length",
]
