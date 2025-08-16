from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:  # pragma: no cover - optional dep
    GraphDatabase = None  # type: ignore


@dataclass
class ArchetypeNode:
    """Archetype node metadata."""

    name: str
    influence: float = 0.0
    trust_delta: float = 0.0
    reputation: List[str] = field(default_factory=list)
    beliefs: List[float] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)


@dataclass
class RelationshipEdge:
    """Directed weighted edge between archetypes."""

    weight: float
    last_updated: float = field(default_factory=time.time)
    history: List[Tuple[float, float]] = field(default_factory=list)


class ArchetypeGraph:
    """Neo4j-backed archetype relationship graph."""

    def __init__(
        self,
        uri: Optional[str] = None,
        auth: Optional[Tuple[str, str]] = None,
        *,
        decay: float = 0.99,
    ) -> None:
        self.decay = decay
        self.enabled = GraphDatabase is not None and uri and auth
        if self.enabled:
            self.driver = GraphDatabase.driver(uri, auth=auth)  # pragma: no cover
        else:  # pragma: no cover - optional dep missing
            self.driver = None
        self.nodes: Dict[str, ArchetypeNode] = {}
        self.edges: Dict[str, Dict[str, RelationshipEdge]] = {}
        self.ledger: List[Tuple[float, str, str, float]] = []

    # ------------------------------------------------------------------
    def add_archetype(
        self,
        name: str,
        *,
        influence: float = 0.0,
        trust_delta: float = 0.0,
        reputation: Optional[List[str]] = None,
        beliefs: Optional[List[float]] = None,
        triggers: Optional[List[str]] = None,
    ) -> None:
        """Add an archetype node with metadata."""
        node = ArchetypeNode(
            name=name,
            influence=influence,
            trust_delta=trust_delta,
            reputation=reputation or [],
            beliefs=beliefs or [],
            triggers=triggers or [],
        )
        self.nodes[name] = node
        self.edges.setdefault(name, {})
        if self.driver is not None:
            query = (
                "MERGE (a:Archetype {name:$n}) "
                "SET a.influence=$i, a.trust_delta=$t, a.reputation=$r, "
                "a.beliefs=$b, a.triggers=$tr"
            )
            self.driver.session().run(
                query,
                n=name,
                i=influence,
                t=trust_delta,
                r=reputation or [],
                b=beliefs or [],
                tr=triggers or [],
            )

    # ------------------------------------------------------------------
    def connect(
        self, src: str, dst: str, *, weight: float = 0.0, rel_type: str = "trust"
    ) -> None:
        """Create or update a relationship edge."""
        now = time.time()
        edge = self.edges.setdefault(src, {}).get(dst)
        if edge is None:
            edge = RelationshipEdge(weight=weight, last_updated=now, history=[(now, weight)])
            self.edges[src][dst] = edge
        else:
            edge.weight = weight
            edge.last_updated = now
            edge.history.append((now, weight))
        self.ledger.append((now, src, dst, edge.weight))
        if self.driver is not None:
            query = (
                "MERGE (a:Archetype {name:$a}) "
                "MERGE (b:Archetype {name:$b}) "
                "MERGE (a)-[e:REL {type:$t}]->(b) "
                "SET e.weight=$w, e.timestamp=$ts"
            )
            self.driver.session().run(
                query,
                a=src,
                b=dst,
                t=rel_type,
                w=edge.weight,
                ts=now,
            )

    # ------------------------------------------------------------------
    def decay_edges(self) -> None:
        """Apply time decay to all edge weights."""
        now = time.time()
        for s, dsts in self.edges.items():
            for d, edge in dsts.items():
                age = now - edge.last_updated
                new_w = edge.weight * (self.decay ** age)
                if new_w != edge.weight:
                    edge.weight = new_w
                    edge.last_updated = now
                    edge.history.append((now, new_w))
                    self.ledger.append((now, s, d, new_w))

    # ------------------------------------------------------------------
    def update_relationship(
        self,
        src: str,
        dst: str,
        *,
        alignment: float = 0.0,
        frequency: float = 1.0,
        betrayal: bool = False,
        validation: float = 0.0,
    ) -> None:
        """Adjust edge weight based on interactions."""
        if src not in self.nodes or dst not in self.nodes:
            return
        self.decay_edges()
        edge = self.edges.setdefault(src, {}).get(dst)
        if edge is None:
            self.connect(src, dst, weight=alignment * frequency)
            edge = self.edges[src][dst]
        adjust = alignment * frequency + validation
        if betrayal:
            adjust -= abs(edge.weight) * 0.5
        edge.weight += adjust
        edge.last_updated = time.time()
        edge.history.append((edge.last_updated, edge.weight))
        self.ledger.append((edge.last_updated, src, dst, edge.weight))
        if self.driver is not None:
            query = (
                "MATCH (a:Archetype {name:$a})-[e:REL]->(b:Archetype {name:$b}) "
                "SET e.weight=$w, e.timestamp=$ts"
            )
            self.driver.session().run(
                query,
                a=src,
                b=dst,
                w=edge.weight,
                ts=edge.last_updated,
            )

    # ------------------------------------------------------------------
    def alliance_scan(self, threshold: float = 0.8) -> None:
        """Form alliances for similar belief systems."""
        names = list(self.nodes.keys())
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                sim = self._similarity(self.nodes[a].beliefs, self.nodes[b].beliefs)
                if sim >= threshold:
                    self.update_relationship(a, b, alignment=sim)

    # ------------------------------------------------------------------
    def _similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        dot = sum(a[i] * b[i] for i in range(n))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        denom = na * nb or 1.0
        return dot / denom


__all__ = ["ArchetypeGraph", "ArchetypeNode", "RelationshipEdge"]
