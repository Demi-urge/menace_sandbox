from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, Tuple


@dataclass
class InfluenceNode:
    """Node in the influence graph organised by tier."""

    name: str
    tier: str
    data: Dict[str, str] = field(default_factory=dict)


@dataclass
class InfluenceEdge:
    """Relationship metrics between two nodes."""

    sentiment: float = 0.0
    count: int = 0
    engagement: float = 0.0


class InfluenceGraph:
    """Multi-level learning graph capturing long-term insights."""

    def __init__(self) -> None:
        self.nodes: Dict[str, InfluenceNode] = {}
        self.edges: Dict[Tuple[str, str], InfluenceEdge] = {}

    # ------------------------------------------------------------------
    def _similar(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def add_node(self, name: str, tier: str) -> None:
        if name not in self.nodes:
            self.nodes[name] = InfluenceNode(name=name, tier=tier)

    def _combine_edges(
        self, existing: InfluenceEdge, new: InfluenceEdge
    ) -> InfluenceEdge:
        total = existing.count + new.count
        sentiment = (
            existing.sentiment * existing.count + new.sentiment * new.count
        ) / (total or 1)
        engagement = max(existing.engagement, new.engagement)
        return InfluenceEdge(sentiment=sentiment, count=total, engagement=engagement)

    def add_edge(self, src: str, dst: str, sentiment: float, engagement: float) -> None:
        key = (src, dst)
        edge = self.edges.get(key, InfluenceEdge())
        updated = InfluenceEdge(
            sentiment=sentiment,
            count=1,
            engagement=engagement,
        )
        self.edges[key] = self._combine_edges(edge, updated)

    # ------------------------------------------------------------------
    def add_observation(
        self,
        entity: str,
        intent: str,
        emotion: str,
        *,
        sentiment: float,
        engagement: float = 1.0,
    ) -> None:
        """Ingest a new interaction into the graph."""
        self.add_node(entity, "concept")
        self.add_node(intent, "trigger")
        self.add_node(emotion, "persona")
        self.add_edge(entity, intent, sentiment, engagement)
        self.add_edge(intent, emotion, sentiment, engagement)
        self.add_edge(entity, emotion, sentiment, engagement)

    # ------------------------------------------------------------------
    def merge_redundant_nodes(self, threshold: float = 0.9) -> None:
        names = list(self.nodes.keys())
        for i, n1 in enumerate(names):
            for n2 in names[i + 1 :]:
                if n1 not in self.nodes or n2 not in self.nodes:
                    continue
                if self.nodes[n1].tier != self.nodes[n2].tier:
                    continue
                if self._similar(n1, n2) >= threshold:
                    self._merge_nodes(n2, n1)

    def _merge_nodes(self, src: str, dst: str) -> None:
        for (a, b), edge in list(self.edges.items()):
            if a == src:
                self.edges[(dst, b)] = self._combine_edges(
                    self.edges.get((dst, b), InfluenceEdge()), edge
                )
                del self.edges[(a, b)]
            elif b == src:
                self.edges[(a, dst)] = self._combine_edges(
                    self.edges.get((a, dst), InfluenceEdge()), edge
                )
                del self.edges[(a, b)]
        self.nodes.pop(src, None)

    # ------------------------------------------------------------------
    def split_overdense_clusters(self, max_degree: int = 5) -> None:
        for name in list(self.nodes.keys()):
            degree = sum(1 for (a, b) in self.edges if a == name or b == name)
            if degree > max_degree:
                new_name = f"{name}_1"
                if new_name in self.nodes:
                    continue
                self.nodes[new_name] = InfluenceNode(
                    name=new_name, tier=self.nodes[name].tier
                )
                moved = 0
                for (a, b) in list(self.edges):
                    if moved >= degree // 2:
                        break
                    if a == name or b == name:
                        edge = self.edges.pop((a, b))
                        if a == name:
                            self.edges[(new_name, b)] = edge
                        else:
                            self.edges[(a, new_name)] = edge
                        moved += 1

    # ------------------------------------------------------------------
    def reweight_links(
        self, reinforcement: Dict[Tuple[str, str], float], decay: float = 0.9
    ) -> None:
        for key, edge in self.edges.items():
            edge.sentiment *= decay
            edge.engagement *= decay
            if key in reinforcement:
                edge.sentiment += reinforcement[key]

    def nightly_job(self, reinforcement: Dict[Tuple[str, str], float]) -> None:
        """Run maintenance tasks to evolve the graph."""
        self.merge_redundant_nodes()
        self.split_overdense_clusters()
        self.reweight_links(reinforcement)

    # ------------------------------------------------------------------
    def validate_nodes(self, external_data: Dict[str, str]) -> None:
        """Update node facts using external sources."""
        for name, fact in external_data.items():
            if name in self.nodes:
                self.nodes[name].data["fact"] = fact

