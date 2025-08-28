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

    aggregate: float
    """Mean of efficiency (similarity), modularity and expandability."""


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
        key = ",".join(cls._extract_modules(spec))
        if key in cls._embed_cache:
            return cls._embed_cache[key]
        graph = cls._build_graph(cls._extract_modules(spec))
        vec = cls._embed_graph(graph, spec)
        cls._embed_cache[key] = vec
        return vec

    @staticmethod
    def _embed_graph(graph: Any, spec: Dict[str, Any]) -> List[float]:
        """Return an embedding for ``graph``/``spec``."""

        if _HAS_NX and isinstance(graph, nx.Graph):
            try:  # pragma: no cover - optional dependency
                centrality = nx.degree_centrality(graph)
                return [centrality[n] for n in sorted(centrality)]
            except Exception:
                pass

        if WorkflowVectorizer is not None:
            try:  # pragma: no cover - optional dependency
                wf_record = {"workflow": [s.get("module") for s in spec.get("steps", [])]}
                vec = WorkflowVectorizer().fit([wf_record]).transform(wf_record)
                return list(vec)
            except Exception:
                pass

        counts: Dict[str, int] = {}
        for step in spec.get("steps", []):
            if isinstance(step, dict):
                mod = step.get("module")
                if mod:
                    counts[str(mod)] = counts.get(str(mod), 0) + 1
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
        aggregate = (similarity + shared_ratio + expandability) / 3

        return SynergyScores(
            similarity=similarity,
            shared_module_ratio=shared_ratio,
            entropy_a=entropy_a,
            entropy_b=entropy_b,
            expandability=expandability,
            aggregate=aggregate,
        )

    # ------------------------------------------------------------------
    @classmethod
    def is_duplicate(
        cls,
        scores_or_a: SynergyScores | Dict[str, Any] | str | Path,
        b_spec: Dict[str, Any] | str | Path | None = None,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Return ``True`` when two workflows are near-identical.

        Parameters
        ----------
        scores_or_a:
            Either a :class:`SynergyScores` instance produced by :meth:`compare`
            or the first workflow specification/identifier.
        b_spec:
            Optional second workflow specification/identifier.  Required when
            ``scores_or_a`` is not a :class:`SynergyScores` instance.
        thresholds:
            Optional mapping providing ``similarity`` and ``entropy`` thresholds.
            Defaults to ``{"similarity": 0.95, "entropy": 0.05}``.
        """

        thresholds = thresholds or {}
        sim_thr = thresholds.get("similarity", 0.95)
        ent_thr = thresholds.get("entropy", 0.05)

        if isinstance(scores_or_a, SynergyScores):
            result = scores_or_a
        else:
            if b_spec is None:
                raise TypeError("b_spec must be provided when passing workflow specifications")
            result = cls.compare(scores_or_a, b_spec)

        ent_gap = abs(result.entropy_a - result.entropy_b)
        return result.similarity >= sim_thr and ent_gap <= ent_thr

    # ------------------------------------------------------------------
    @classmethod
    def merge_duplicate(
        cls,
        base_id: str,
        dup_id: str,
        out_dir: Path | str | None = None,
    ) -> Path | None:
        """Merge a duplicate workflow into a canonical base specification.

        The function loads both workflow specifications, writes them to
        temporary files and delegates the actual merge to
        :mod:`workflow_merger`.  Temporary files are cleaned up regardless of
        merge success.
        """

        work_dir = Path(out_dir) if out_dir is not None else cls.workflow_dir

        base_spec = cls._load_spec(work_dir / f"{base_id}.workflow.json")
        dup_spec = cls._load_spec(work_dir / f"{dup_id}.workflow.json")
        if not base_spec or not dup_spec:
            return None

        work_dir.mkdir(parents=True, exist_ok=True)
        base_path = work_dir / f"{base_id}.base.json"
        dup_path = work_dir / f"{dup_id}.dup.json"
        out_path = work_dir / f"{base_id}.merged.json"

        try:
            from . import workflow_merger
        except Exception:
            workflow_merger = None  # type: ignore

        try:
            base_path.write_text(json.dumps(base_spec))
            dup_path.write_text(json.dumps(dup_spec))
            if workflow_merger is None:
                return None
            result = workflow_merger.merge_workflows(
                base_path, base_path, dup_path, out_path
            )
        except Exception:
            result = None
        finally:
            for p in (base_path, dup_path):
                try:
                    p.unlink()
                except Exception:
                    pass

        return result


def merge_duplicate(
    base_id: str, dup_id: str, out_dir: Path | str | None = None
) -> Path | None:
    """Convenience wrapper around :class:`WorkflowSynergyComparator`.

    This allows callers to import :func:`merge_duplicate` directly from the
    module without instantiating :class:`WorkflowSynergyComparator`.
    """

    return WorkflowSynergyComparator.merge_duplicate(base_id, dup_id, out_dir)


__all__ = ["WorkflowSynergyComparator", "SynergyScores", "merge_duplicate"]
