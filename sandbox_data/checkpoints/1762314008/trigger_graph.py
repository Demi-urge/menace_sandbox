from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class GraphNode:
    """Node storing transition weights to other methods."""

    method: str
    edges: Dict[str, float] = field(default_factory=dict)


class TriggerEffectGraph:
    """Directed graph modeling trigger-effect chains per user."""

    def __init__(self, methods: List[str], ttl_seconds: float = 3600.0) -> None:
        self.methods = methods
        self.ttl_seconds = ttl_seconds
        self.graphs: Dict[str, Dict[str, GraphNode]] = {}
        self.last_reset: Dict[str, float] = {}

    def _init_graph(self) -> Dict[str, GraphNode]:
        return {m: GraphNode(method=m) for m in self.methods}

    def _get_graph(self, user_id: str) -> Dict[str, GraphNode]:
        now = time.time()
        if (
            user_id not in self.graphs
            or now - self.last_reset.get(user_id, 0.0) > self.ttl_seconds
        ):
            self.graphs[user_id] = self._init_graph()
            self.last_reset[user_id] = now
        return self.graphs[user_id]

    def reset_user(self, user_id: str) -> None:
        """Manually reset a user's graph."""
        self.graphs[user_id] = self._init_graph()
        self.last_reset[user_id] = time.time()

    def record_transition(
        self,
        user_id: str,
        from_method: str,
        to_method: str,
        success: bool,
    ) -> None:
        """Update transition weight based on user feedback."""
        graph = self._get_graph(user_id)
        node = graph.setdefault(from_method, GraphNode(method=from_method))
        weight = node.edges.get(to_method, 0.0)
        node.edges[to_method] = weight + (1.0 if success else -1.0)

    def next_method(self, user_id: str, current_method: str) -> str:
        """Return the next method with the highest transition weight."""
        graph = self._get_graph(user_id)
        node = graph.get(current_method)
        if not node or not node.edges:
            return ""
        best = max(node.edges.items(), key=lambda x: x[1])[0]
        return best
