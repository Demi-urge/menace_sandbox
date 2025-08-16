from __future__ import annotations

import time
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:  # pragma: no cover - optional dep
    GraphDatabase = None  # type: ignore

try:
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:  # pragma: no cover - optional heavy deps
    faiss = None  # type: ignore
    SentenceTransformer = None  # type: ignore
    np = None  # type: ignore

from .engagement_graph import pagerank, shortest_path


@dataclass
class RuleNode:
    """Psychological if-then rule node."""

    rule_id: str
    condition: str
    response: str
    weight: float = 1.0
    decay: float = 0.01
    last_used: float = field(default_factory=time.time)
    embedding: List[float] = field(default_factory=list)


class PsychologicalGraph:
    """Manage if-then rules with Neo4j persistence and embedding search."""

    def __init__(self, uri: Optional[str] = None, auth: Optional[Tuple[str, str]] = None) -> None:
        uri = uri or os.getenv("NEURO_NEO4J_URI")
        if auth is None:
            user = os.getenv("NEURO_NEO4J_USER")
            pw = os.getenv("NEURO_NEO4J_PASS")
            auth = (user, pw) if user and pw else None

        self.enabled = GraphDatabase is not None and uri and auth
        if self.enabled:
            self.driver = GraphDatabase.driver(uri, auth=auth)  # pragma: no cover
        else:  # pragma: no cover - optional dep missing
            self.driver = None
        self.rules: Dict[str, RuleNode] = {}
        self.edges: Dict[str, Dict[str, float]] = {}
        self._model = None
        self._index = None
        self._ids: List[str] = []
        if SentenceTransformer is not None and faiss is not None:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            dim = self._model.get_sentence_embedding_dimension()
            self._index = faiss.IndexFlatL2(dim)
        
    # ------------------------------------------------------------------
    def _encode(self, text: str) -> List[float]:
        if self._model is None or np is None:
            return []
        vec = self._model.encode(text, convert_to_numpy=True).astype("float32")
        return vec.tolist()

    def _add_to_index(self, rule: RuleNode) -> None:
        if self._index is None or np is None:
            return
        vec = np.array(rule.embedding, dtype="float32").reshape(1, -1)
        self._index.add(vec)
        self._ids.append(rule.rule_id)

    # ------------------------------------------------------------------
    def add_rule(self, condition: str, response: str, weight: float = 1.0, decay: float = 0.01) -> str:
        rule_id = f"r{len(self.rules)}"
        emb = self._encode(condition + " " + response)
        node = RuleNode(rule_id, condition, response, weight, decay, time.time(), emb)
        self.rules[rule_id] = node
        self.edges.setdefault(rule_id, {})
        self._add_to_index(node)
        if self.driver is not None:
            query = (
                "MERGE (r:Rule {id:$id}) "
                "SET r.condition=$c, r.response=$r, r.weight=$w, r.decay=$d, r.last_used=$t"
            )
            self.driver.session().run(
                query,
                id=rule_id,
                c=condition,
                r=response,
                w=weight,
                d=decay,
                t=node.last_used,
            )
        return rule_id

    # ------------------------------------------------------------------
    def link_rules(self, src: str, dst: str, weight: float = 1.0) -> None:
        self.edges.setdefault(src, {})[dst] = weight
        if self.driver is not None:
            query = (
                "MATCH (a:Rule {id:$src}), (b:Rule {id:$dst}) "
                "MERGE (a)-[e:REL]->(b) SET e.weight=$w"
            )
            self.driver.session().run(query, src=src, dst=dst, w=weight)

    # ------------------------------------------------------------------
    def record_interaction(self, rule_id: str, success: bool) -> None:
        node = self.rules.get(rule_id)
        if not node:
            return
        node.weight += 1.0 if success else -1.0
        node.last_used = time.time()

    # ------------------------------------------------------------------
    def decay_weights(self) -> None:
        now = time.time()
        for node in self.rules.values():
            age = now - node.last_used
            node.weight *= (1.0 - node.decay) ** age

    # ------------------------------------------------------------------
    def pagerank_scores(self) -> Dict[str, float]:
        return pagerank(self.edges)

    # ------------------------------------------------------------------
    def shortest_influence_path(self, start: str, end: str) -> List[str]:
        return shortest_path(self.edges, start, end)

    # ------------------------------------------------------------------
    def closest_rule(self, text: str, top_k: int = 1) -> Optional[RuleNode]:
        if self._index is None or self._model is None or np is None:
            return None
        if not self.rules:
            return None
        query = self._model.encode(text, convert_to_numpy=True).astype("float32")
        D, I = self._index.search(query.reshape(1, -1), min(top_k, len(self._ids)))
        if not I.size:
            return None
        return self.rules[self._ids[I[0][0]]]


__all__ = ["PsychologicalGraph", "RuleNode"]
