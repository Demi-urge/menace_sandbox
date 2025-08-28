"""Utilities for structural workflow comparison.

This module provides :class:`WorkflowSynergyComparator` which compares two
workflow specifications. Workflows are represented as simple graphs and
embedded into a vector space using :mod:`networkx` features or the optional
:mod:`workflow_vectorizer` package. Similarity and structural heuristics are
combined into an aggregate ranking allowing callers to detect duplicated or
closely related workflows.

The public API consists of :meth:`WorkflowSynergyComparator.compare` returning a
:class:`SynergyScores` instance and :meth:`WorkflowSynergyComparator.is_duplicate`
for quick duplicate detection.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:  # pragma: no cover - optional dependency
    import networkx as nx  # type: ignore
    _HAS_NX = True
except Exception:  # pragma: no cover - gracefully degrade
    nx = None  # type: ignore
    _HAS_NX = False

try:  # pragma: no cover - optional dependency
    from node2vec import Node2Vec  # type: ignore
    _HAS_NODE2VEC = True
except Exception:  # pragma: no cover - gracefully degrade
    Node2Vec = None  # type: ignore
    _HAS_NODE2VEC = False

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from workflow_vectorizer import WorkflowVectorizer  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade
    WorkflowVectorizer = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .workflow_graph import WorkflowGraph  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade
    WorkflowGraph = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .roi_tracker import ROITracker  # type: ignore
except BaseException:  # pragma: no cover - gracefully degrade
    ROITracker = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from . import workflow_merger  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade
    workflow_merger = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from . import workflow_run_summary  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade
    workflow_run_summary = None  # type: ignore

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SynergyScores:
    """Structured workflow comparison scores."""

    similarity: float
    """Cosine similarity of workflow embeddings."""

    shared_module_ratio: float
    """Jaccard ratio of modules shared between both workflows."""

    entropy_a: float
    """Entropy of workflow ``A``."""

    entropy_b: float
    """Entropy of workflow ``B``."""

    expandability: float
    """Average entropy representing expandability of both workflows."""

    efficiency: float
    """Latest ROI based efficiency score."""

    modularity: float
    """Structural modularity of the combined workflow graph."""

    aggregate: float
    """Mean of efficiency, modularity and expandability."""


# ---------------------------------------------------------------------------
# Comparator implementation
# ---------------------------------------------------------------------------


class WorkflowSynergyComparator:
    """Compare workflow specifications using structural heuristics."""

    workflow_dir = Path("workflows")
    _embed_cache: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_spec(src: Dict[str, Any] | str | Path) -> Dict[str, Any]:
        """Return a workflow specification for ``src``.

        ``src`` may be a mapping representing the specification directly, a
        path to a JSON file or a workflow identifier.  Workflow identifiers are
        resolved against ``workflow_dir`` following the ``{id}.workflow.json``
        naming scheme.  Errors are swallowed and result in an empty mapping.
        """

        if isinstance(src, dict):
            return src

        path = Path(str(src))
        if path.is_file():
            try:
                return json.loads(path.read_text())
            except Exception:
                return {}

        guess = WorkflowSynergyComparator.workflow_dir / f"{path}.workflow.json"
        if guess.is_file():
            try:
                return json.loads(guess.read_text())
            except Exception:
                return {}
        return {}

    @staticmethod
    def _extract_modules(spec: Dict[str, Any]) -> List[str]:
        mods: List[str] = []
        for step in spec.get("steps", []):
            if isinstance(step, dict):
                mod = step.get("module")
                if mod:
                    mods.append(str(mod))
        return mods

    @classmethod
    def _build_graph(cls, modules: List[str]):
        if _HAS_NX:
            g = nx.DiGraph()
            g.add_nodes_from(modules)
            g.add_edges_from(zip(modules, modules[1:]))
            return g
        # Fallback simple adjacency mapping
        graph: Dict[str, set] = {m: set() for m in modules}
        for a, b in zip(modules, modules[1:]):
            graph.setdefault(a, set()).add(b)
        return graph

    @classmethod
    def _embed_spec(cls, spec: Dict[str, Any]) -> List[float]:
        graph = cls._build_graph(cls._extract_modules(spec))
        return cls._embed_graph(graph, spec)

    @classmethod
    def _embed_graph(cls, graph: Any, spec: Dict[str, Any]) -> List[float]:
        """Return an embedding for ``graph``/``spec`` with caching."""

        key = ",".join(cls._extract_modules(spec))
        if key in cls._embed_cache:
            return cls._embed_cache[key]

        if _HAS_NX and isinstance(graph, nx.Graph):
            if _HAS_NODE2VEC:
                try:  # pragma: no cover - optional dependency
                    n2v = Node2Vec(
                        graph,
                        dimensions=16,
                        walk_length=5,
                        num_walks=20,
                        workers=1,
                        seed=0,
                    )
                    model = n2v.fit(window=5, min_count=1, seed=0)
                    vectors = model.wv
                    if hasattr(vectors, "index_to_key"):
                        nodes = list(vectors.index_to_key)
                    elif hasattr(vectors, "key_to_index"):
                        nodes = list(vectors.key_to_index.keys())
                    else:
                        nodes = list(vectors.keys())  # type: ignore[attr-defined]
                    vec: List[float] = []
                    for n in sorted(nodes):
                        vec.extend([float(v) for v in vectors[n]])
                    cls._embed_cache[key] = vec
                    return vec
                except Exception:
                    pass

            if np is not None:
                try:  # pragma: no cover - optional dependency
                    nodes = sorted(graph.nodes())
                    mat = nx.to_numpy_array(graph, nodelist=nodes)
                    if not np.allclose(mat, mat.T):
                        mat = (mat + mat.T) / 2
                    vals, vecs = np.linalg.eigh(mat)
                    order = np.argsort(vals)[::-1]
                    vecs = vecs[:, order]
                    k = min(len(nodes), 4)
                    emb = vecs[:, :k]
                    vec = emb.flatten().tolist()
                    cls._embed_cache[key] = vec
                    return vec
                except Exception:
                    pass

        if WorkflowVectorizer is not None:
            try:  # pragma: no cover - optional dependency
                wf_record = {"workflow": [s.get("module") for s in spec.get("steps", [])]}
                vec = WorkflowVectorizer().fit([wf_record]).transform(wf_record)
                cls._embed_cache[key] = list(vec)
                return list(vec)
            except Exception:
                pass

        counts: Dict[str, int] = {}
        for step in spec.get("steps", []):
            if isinstance(step, dict):
                mod = step.get("module")
                if mod:
                    counts[str(mod)] = counts.get(str(mod), 0) + 1
        vec = [counts[k] for k in sorted(counts)]
        cls._embed_cache[key] = vec
        return vec

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
    def _shared_ratio(mod_a: Iterable[str], mod_b: Iterable[str]) -> float:
        set_a = set(mod_a)
        set_b = set(mod_b)
        union = set_a | set_b
        return len(set_a & set_b) / len(union) if union else 0.0

    @staticmethod
    def _entropy(spec: Dict[str, Any]) -> float:
        if ROITracker is not None:
            try:  # pragma: no cover - optional dependency
                tracker = ROITracker()
                hist = tracker.metrics_history.get("synergy_shannon_entropy", [])
                if hist:
                    return float(hist[-1])
            except Exception:
                pass
        modules = WorkflowSynergyComparator._extract_modules(spec)
        counts: Dict[str, int] = {}
        for m in modules:
            counts[m] = counts.get(m, 0) + 1
        total = sum(counts.values())
        if not total:
            return 0.0
        return -sum((c / total) * math.log(c / total, 2) for c in counts.values())

    @staticmethod
    def _roi_and_modularity(graph: Any) -> Tuple[float, float]:
        """Return efficiency and modularity for ``graph``.

        Efficiency is derived from :class:`ROITracker`'s ``synergy_efficiency``
        history when available.  Modularity relies on ``networkx`` community
        detection, falling back to ``0.0`` when optional dependencies are
        missing or raise errors.
        """

        efficiency = 0.0
        if ROITracker is not None:
            try:  # pragma: no cover - optional dependency
                tracker = ROITracker()
                eff_hist = tracker.metrics_history.get("synergy_efficiency", [])
                if eff_hist:
                    efficiency = float(eff_hist[-1])
            except Exception:
                pass

        modularity = 0.0
        if _HAS_NX and isinstance(graph, nx.Graph):
            try:  # pragma: no cover - optional dependency
                nx_comm = getattr(getattr(nx, "algorithms", None), "community", None)
                if nx_comm is not None:
                    undirected = graph.to_undirected()
                    communities = list(nx_comm.greedy_modularity_communities(undirected))
                    if communities:
                        modularity = float(nx_comm.modularity(undirected, communities))
            except Exception:
                pass

        return efficiency, modularity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @classmethod
    def compare(
        cls,
        a_spec: Dict[str, Any] | str | Path,
        b_spec: Dict[str, Any] | str | Path,
    ) -> SynergyScores:
        """Compare two workflow specifications or identifiers."""

        spec_a = cls._load_spec(a_spec)
        spec_b = cls._load_spec(b_spec)

        mods_a = cls._extract_modules(spec_a)
        mods_b = cls._extract_modules(spec_b)

        vec_a = cls._embed_spec(spec_a)
        vec_b = cls._embed_spec(spec_b)

        similarity = cls._cosine(vec_a, vec_b)
        shared_ratio = cls._shared_ratio(mods_a, mods_b)
        entropy_a = cls._entropy(spec_a)
        entropy_b = cls._entropy(spec_b)
        expandability = (entropy_a + entropy_b) / 2 if (entropy_a or entropy_b) else 0.0

        # Additional metrics derived from ROITracker and structural communities
        union_graph = cls._build_graph(mods_a + mods_b)
        efficiency, modularity = cls._roi_and_modularity(union_graph)

        aggregate = (efficiency + modularity + expandability) / 3

        return SynergyScores(
            similarity=similarity,
            shared_module_ratio=shared_ratio,
            entropy_a=entropy_a,
            entropy_b=entropy_b,
            expandability=expandability,
            efficiency=efficiency,
            modularity=modularity,
            aggregate=aggregate,
        )

    # ------------------------------------------------------------------
    @classmethod
    def is_duplicate(
        cls, scores: SynergyScores, thresholds: Optional[Dict[str, float]] = None
    ) -> bool:
        """Return ``True`` when two workflows are near-identical.

        ``scores`` must be the result of :meth:`compare`.  ``thresholds`` may
        override the default similarity (``0.95``) and entropy gap (``0.05``)
        limits used to qualify a pair of workflows as duplicates.
        """

        thresholds = thresholds or {}
        sim_thr = thresholds.get("similarity", 0.95)
        ent_thr = thresholds.get("entropy", 0.05)

        ent_gap = abs(scores.entropy_a - scores.entropy_b)
        return scores.similarity >= sim_thr and ent_gap <= ent_thr

    # ------------------------------------------------------------------
    @classmethod
    def merge_duplicate(
        cls,
        base_id: str,
        dup_id: str,
        out_dir: Path | str | None = None,
    ) -> Path | None:
        """Merge ``dup_id`` into ``base_id`` within ``out_dir``.

        This is a thin wrapper around the module level :func:`merge_duplicate`
        so callers can use the class directly.  When ``out_dir`` is ``None`` the
        comparator's :attr:`workflow_dir` is used.
        """

        directory = Path(out_dir) if out_dir is not None else cls.workflow_dir
        return merge_duplicate(base_id, dup_id, directory)


def merge_duplicate(
    base_id: str, dup_id: str, out_dir: str | Path = "workflows"
) -> Path | None:
    """Merge workflow ``dup_id`` into ``base_id`` and return output path.

    The function reads ``{id}.workflow.json`` files from ``out_dir`` for both
    ``base_id`` and ``dup_id``.  The base specification acts as both the merge
    ancestor and branch ``A`` ensuring that only changes from the duplicate are
    applied.  On successful merge the workflow lineage summaries are refreshed
    using :func:`workflow_run_summary.save_all_summaries`.

    Parameters
    ----------
    base_id:
        Identifier of the canonical workflow.
    dup_id:
        Identifier of the duplicate workflow to merge.
    out_dir:
        Directory containing workflow specifications. Defaults to ``"workflows"``.

    Returns
    -------
    Path | None
        Path to the merged workflow specification or ``None`` on failure.
    """

    directory = Path(out_dir)
    base = directory / f"{base_id}.workflow.json"
    dup = directory / f"{dup_id}.workflow.json"
    out = directory / f"{base_id}.merged.json"

    try:
        # Ensure both workflow files exist and contain valid JSON
        json.loads(base.read_text())
        json.loads(dup.read_text())
    except Exception:
        return None

    if workflow_merger is None:
        return None

    try:
        merged = workflow_merger.merge_workflows(base, base, dup, out)
    except Exception:
        return None

    if merged and workflow_run_summary is not None:
        try:
            workflow_run_summary.save_all_summaries(directory)
        except Exception:
            pass

    return merged


__all__ = ["WorkflowSynergyComparator", "SynergyScores", "merge_duplicate"]
