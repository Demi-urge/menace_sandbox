"""Workflow comparison utilities.

This module provides :class:`WorkflowSynergyComparator` used to compare two
workflow specifications.  The comparison is intentionally lightweight so it can
operate even in constrained environments.  It attempts to build a small graph
representation of each workflow, generate vector embeddings for the graphs and
then derive a number of heuristics such as cosine similarity, module overlap and
Shannon entropy of the module distribution.

The main entry point is :meth:`WorkflowSynergyComparator.compare` which returns a
:class:`WorkflowComparisonResult`.  Convenience helper
``WorkflowSynergyComparator.is_duplicate`` exposes a simple policy for deciding
whether two workflows are effectively duplicates based on configurable
thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .workflow_metrics import compute_workflow_entropy

try:  # Optional dependency used for richer graph handling
    import networkx as nx  # type: ignore
    _HAS_NX = True
except Exception:  # pragma: no cover - optional dependency
    nx = None  # type: ignore
    _HAS_NX = False

try:  # Optional embedding technique
    from node2vec import Node2Vec  # type: ignore
    _HAS_NODE2VEC = True
except Exception:  # pragma: no cover - optional dependency
    Node2Vec = None  # type: ignore
    _HAS_NODE2VEC = False

try:  # Optional workflow vectoriser
    from workflow_vectorizer import WorkflowVectorizer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    WorkflowVectorizer = None  # type: ignore

try:  # Best effort loader for workflow specs
    from .workflow_lineage import load_specs as _load_specs  # type: ignore
except Exception:  # pragma: no cover - optional
    _load_specs = None  # type: ignore


@dataclass
class WorkflowComparisonResult:
    """Container holding metrics derived from comparing two workflows."""

    similarity: float
    shared_modules: int
    modules_a: int
    modules_b: int
    entropy_a: float
    entropy_b: float
    recommended_winner: Optional[str]


class WorkflowSynergyComparator:
    """Compare two workflow specifications using structural heuristics."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_spec(src: Dict[str, Any] | str | Path) -> Dict[str, Any]:
        """Load a workflow specification from ``src``.

        ``src`` may either be a mapping already representing the specification or
        a path to a ``.json`` file on disk.  When a directory is supplied the
        function attempts to use :func:`workflow_lineage.load_specs` if available
        and returns the first spec found.  Failing all else an empty mapping is
        returned.
        """

        if isinstance(src, dict):
            return src

        path = Path(src)
        if path.is_file():
            try:
                data = json.loads(path.read_text())
                if isinstance(data, dict):
                    return data
            except Exception:  # pragma: no cover - best effort
                return {}

        if path.is_dir() and _load_specs is not None:  # pragma: no cover - IO heavy
            try:
                for spec in _load_specs(path):
                    if isinstance(spec, dict):
                        return spec
            except Exception:
                pass
        return {}

    @staticmethod
    def _build_graph(spec: Dict[str, Any]):
        """Construct a simple dependency graph from ``spec``.

        Nodes correspond to module names.  An edge from ``A`` to ``B`` is created
        when ``A`` produces an output consumed by ``B``.  The returned ``modules``
        list contains the module of each step in the order encountered.
        """

        steps = spec.get("steps", []) if isinstance(spec, dict) else []
        if _HAS_NX:
            graph = nx.DiGraph()
        else:
            graph = {}  # type: Dict[str, set]

        modules: List[str] = []
        for step in steps:
            mod = step.get("module") if isinstance(step, dict) else None
            if not mod:
                continue
            modules.append(mod)
            if _HAS_NX:
                graph.add_node(mod)
            else:
                graph.setdefault(mod, set())

        for i, a in enumerate(steps):
            mod_a = a.get("module") if isinstance(a, dict) else None
            if not mod_a:
                continue
            outputs = set(a.get("outputs") or []) if isinstance(a, dict) else set()
            if not outputs:
                continue
            for j, b in enumerate(steps):
                if i == j:
                    continue
                mod_b = b.get("module") if isinstance(b, dict) else None
                if not mod_b:
                    continue
                inputs = set(b.get("inputs") or []) if isinstance(b, dict) else set()
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
                    return [
                        sum(vec[i] for vec in embeddings) / len(embeddings)
                        for i in range(dim)
                    ]
            except Exception:  # pragma: no cover - fall back below
                pass

        # ``WorkflowVectorizer`` embedding
        if WorkflowVectorizer is not None:
            wf_record = {"workflow": [s.get("module") for s in spec.get("steps", [])]}
            try:
                vec = WorkflowVectorizer().fit([wf_record]).transform(wf_record)
                return list(vec)
            except Exception:  # pragma: no cover - graceful degradation
                pass

        # Count-based fallback
        counts: Dict[str, int] = {}
        for step in spec.get("steps", []):
            mod = step.get("module") if isinstance(step, dict) else None
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @classmethod
    def compare(
        cls,
        a_spec: Dict[str, Any] | str | Path,
        b_spec: Dict[str, Any] | str | Path,
    ) -> WorkflowComparisonResult:
        """Compare two workflow specifications.

        ``a_spec`` and ``b_spec`` may be mappings containing a ``steps`` sequence
        or paths to JSON files storing such mappings.
        """

        spec_a = cls._load_spec(a_spec)
        spec_b = cls._load_spec(b_spec)

        graph_a, modules_a = cls._build_graph(spec_a)
        graph_b, modules_b = cls._build_graph(spec_b)
        vec_a = cls._embed_graph(graph_a, spec_a)
        vec_b = cls._embed_graph(graph_b, spec_b)

        similarity = cls._cosine(vec_a, vec_b)

        set_a = set(modules_a)
        set_b = set(modules_b)
        shared = len(set_a & set_b)

        entropy_a = compute_workflow_entropy(spec_a)
        entropy_b = compute_workflow_entropy(spec_b)

        recommended: Optional[str]
        if entropy_a > entropy_b:
            recommended = "a"
        elif entropy_b > entropy_a:
            recommended = "b"
        else:
            recommended = None

        return WorkflowComparisonResult(
            similarity=similarity,
            shared_modules=shared,
            modules_a=len(set_a),
            modules_b=len(set_b),
            entropy_a=entropy_a,
            entropy_b=entropy_b,
            recommended_winner=recommended,
        )

    @classmethod
    def is_duplicate(
        cls,
        a_spec: Dict[str, Any] | str | Path,
        b_spec: Dict[str, Any] | str | Path,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Heuristic check whether two workflows are near-identical.

        Parameters
        ----------
        thresholds:
            Optional mapping configuring the decision boundaries.  Recognised
            keys are ``similarity`` for cosine similarity, ``overlap`` for module
            overlap ratio and ``entropy`` for maximum absolute entropy delta.
        """

        thresholds = thresholds or {}
        res = cls.compare(a_spec, b_spec)

        sim_thresh = thresholds.get("similarity", 0.95)
        overlap_thresh = thresholds.get("overlap", 0.9)
        entropy_thresh = thresholds.get("entropy", 0.05)

        similarity_ok = res.similarity >= sim_thresh
        overlap_a = res.shared_modules / res.modules_a if res.modules_a else 0.0
        overlap_b = res.shared_modules / res.modules_b if res.modules_b else 0.0
        overlap_ok = min(overlap_a, overlap_b) >= overlap_thresh
        entropy_ok = abs(res.entropy_a - res.entropy_b) <= entropy_thresh

        return similarity_ok and overlap_ok and entropy_ok


__all__ = ["WorkflowSynergyComparator", "WorkflowComparisonResult"]

