from __future__ import annotations

"""Utilities for comparing workflow specifications via structural metrics.

The module exposes :class:`WorkflowSynergyComparator` with a ``compare``
method that analyses two workflow specifications and returns a
:class:`WorkflowComparisonResult`.  The comparison builds lightweight
graphs from workflow specs, generates embeddings using either Node2Vec or
``workflow_vectorizer`` and scores:

``efficiency``
    Cosine similarity between the embeddings.
``modularity``
    Jaccard overlap of the modules used in both workflows.
``expandability``
    Average Shannon entropy of the step distributions which acts as a
    proxy for how evenly work is spread across modules and thus how
    easily the workflow might grow.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List
import math

try:  # Optional dependency used for richer graph handling
    import networkx as nx  # type: ignore
    _HAS_NX = True
except Exception:  # pragma: no cover - networkx is optional
    nx = None  # type: ignore
    _HAS_NX = False

try:  # Optional embedding technique
    from node2vec import Node2Vec  # type: ignore
    _HAS_NODE2VEC = True
except Exception:  # pragma: no cover - Node2Vec is optional
    Node2Vec = None  # type: ignore
    _HAS_NODE2VEC = False

try:  # Fallback vectoriser when Node2Vec unavailable
    from workflow_vectorizer import WorkflowVectorizer  # type: ignore
except Exception:  # pragma: no cover - vectoriser is optional
    WorkflowVectorizer = None  # type: ignore


@dataclass
class WorkflowComparisonResult:
    """Container for workflow comparison metrics."""

    efficiency: float
    modularity: float
    expandability: float


class WorkflowSynergyComparator:
    """Compare two workflow specifications using structural heuristics."""

    @staticmethod
    def _build_graph(spec: Dict[str, Any]):
        """Construct a simple dependency graph from ``spec``.

        Nodes correspond to module names.  An edge from ``A`` to ``B`` is
        created when ``A`` produces an output consumed by ``B``.
        """

        steps = spec.get("steps", []) if isinstance(spec, dict) else []
        if _HAS_NX:
            graph = nx.DiGraph()
        else:
            graph = { }  # type: Dict[str, set]
        modules = []
        for step in steps:
            mod = step.get("module")
            if not mod:
                continue
            modules.append(mod)
            if _HAS_NX:
                graph.add_node(mod)
            else:
                graph.setdefault(mod, set())
        for i, a in enumerate(steps):
            mod_a = a.get("module")
            if not mod_a:
                continue
            outputs = set(a.get("outputs") or [])
            if not outputs:
                continue
            for j, b in enumerate(steps):
                if i == j:
                    continue
                mod_b = b.get("module")
                if not mod_b:
                    continue
                inputs = set(b.get("inputs") or [])
                if outputs & inputs:
                    if _HAS_NX:
                        graph.add_edge(mod_a, mod_b)
                    else:
                        graph.setdefault(mod_a, set()).add(mod_b)
        return graph, modules

    @staticmethod
    def _embed_graph(graph: Any, spec: Dict[str, Any]) -> List[float]:
        """Return an embedding for ``graph``/``spec``.

        Attempts to use Node2Vec when available, otherwise falls back to
        :class:`workflow_vectorizer.WorkflowVectorizer` or simple frequency
        counts as a last resort.
        """

        # Node2Vec embedding when networkx and Node2Vec are present
        if _HAS_NX and _HAS_NODE2VEC and isinstance(graph, nx.Graph):
            try:
                node2vec = Node2Vec(graph, dimensions=32, quiet=True, workers=1)
                model = node2vec.fit(window=4, min_count=1, batch_words=4)
                embeddings = [model.wv.get_vector(str(n)) for n in graph.nodes()]
                if embeddings:
                    dim = len(embeddings[0])
                    return [sum(vec[i] for vec in embeddings) / len(embeddings) for i in range(dim)]
            except Exception:  # pragma: no cover - fall back below
                pass

        # ``WorkflowVectorizer`` embedding
        if WorkflowVectorizer is not None:
            wf_record = {"workflow": [s.get("module") for s in spec.get("steps", [])]}
            vec = WorkflowVectorizer().fit([wf_record]).transform(wf_record)
            return vec

        # Count-based fallback
        counts: Dict[str, int] = {}
        for step in spec.get("steps", []):
            mod = step.get("module")
            if mod:
                counts[mod] = counts.get(mod, 0) + 1
        return [counts[k] for k in sorted(counts)]

    @staticmethod
    def _cosine(v1: Iterable[float], v2: Iterable[float]) -> float:
        a = list(v1)
        b = list(v2)
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    @staticmethod
    def _jaccard(a: Iterable[Any], b: Iterable[Any]) -> float:
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union else 0.0

    @staticmethod
    def _entropy(modules: Iterable[Any]) -> float:
        counts: Dict[Any, int] = {}
        for m in modules:
            counts[m] = counts.get(m, 0) + 1
        total = sum(counts.values())
        if not total:
            return 0.0
        ent = 0.0
        for c in counts.values():
            p = c / total
            ent -= p * math.log2(p)
        return ent

    @classmethod
    def compare(cls, a_spec: Dict[str, Any], b_spec: Dict[str, Any]) -> WorkflowComparisonResult:
        """Compare two workflow specifications.

        Parameters
        ----------
        a_spec, b_spec:
            Mappings containing a ``steps`` sequence where each step describes
            ``module``, ``inputs`` and ``outputs`` fields.
        """

        graph_a, modules_a = cls._build_graph(a_spec)
        graph_b, modules_b = cls._build_graph(b_spec)
        vec_a = cls._embed_graph(graph_a, a_spec)
        vec_b = cls._embed_graph(graph_b, b_spec)
        efficiency = cls._cosine(vec_a, vec_b)
        modularity = cls._jaccard(modules_a, modules_b)
        expandability = (cls._entropy(modules_a) + cls._entropy(modules_b)) / 2.0
        return WorkflowComparisonResult(efficiency, modularity, expandability)


__all__ = ["WorkflowSynergyComparator", "WorkflowComparisonResult"]
