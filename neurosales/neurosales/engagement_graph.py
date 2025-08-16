from __future__ import annotations

import heapq
from typing import Dict, List, Tuple


class EngagementGraph:
    """Track engagement strategies across contexts."""

    def __init__(self) -> None:
        # context -> from_strategy -> to_strategy -> weight
        self.graphs: Dict[str, Dict[str, Dict[str, float]]] = {}

    def record_interaction(
        self,
        context: str,
        from_strategy: str,
        to_strategy: str,
        success: bool,
    ) -> None:
        """Record a transition between strategies in a context."""
        g = self.graphs.setdefault(context, {})
        node = g.setdefault(from_strategy, {})
        node[to_strategy] = node.get(to_strategy, 0.0) + (1.0 if success else -1.0)

    def best_next_strategy(self, context: str, current: str) -> str:
        """Return the next strategy with the highest weight."""
        g = self.graphs.get(context, {})
        node = g.get(current)
        if not node:
            return ""
        return max(node.items(), key=lambda x: x[1])[0]


def pagerank(
    graph: Dict[str, Dict[str, float]],
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1.0e-6,
) -> Dict[str, float]:
    """Compute PageRank scores for a directed weighted graph."""
    nodes = set(graph.keys()) | {v for edges in graph.values() for v in edges}
    n = len(nodes)
    if n == 0:
        return {}
    ranks = {node: 1.0 / n for node in nodes}
    for _ in range(max_iter):
        new_ranks = {}
        for node in nodes:
            rank_sum = 0.0
            for src in nodes:
                edges = graph.get(src, {})
                if node in edges and edges:
                    total = sum(edges.values()) or 1.0
                    rank_sum += ranks[src] * edges[node] / total
            new_ranks[node] = (1.0 - damping) / n + damping * rank_sum
        diff = sum(abs(new_ranks[n] - ranks[n]) for n in nodes)
        ranks = new_ranks
        if diff < tol:
            break
    return ranks


def shortest_path(
    graph: Dict[str, Dict[str, float]], start: str, goal: str
) -> List[str]:
    """Find the lowest cost path using Dijkstra's algorithm."""
    queue: List[Tuple[float, str]] = [(0.0, start)]
    costs = {start: 0.0}
    prev: Dict[str, str] = {}
    visited = set()

    while queue:
        cost, node = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            break
        for nbr, weight in graph.get(node, {}).items():
            new_cost = cost + weight
            if new_cost < costs.get(nbr, float("inf")):
                costs[nbr] = new_cost
                prev[nbr] = node
                heapq.heappush(queue, (new_cost, nbr))

    if goal not in costs:
        return []

    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path
