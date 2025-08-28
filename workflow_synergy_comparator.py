"""Workflow similarity and synergy comparison utilities.

The :class:`WorkflowSynergyComparator` class defined in this module provides a
light‑weight way of comparing two workflow specifications.  Workflows are
loaded from the ``workflows/{id}.workflow.json`` directory when an identifier
is supplied and reduced to a sequence of module names.  A small dependency
graph is constructed for each workflow and embedded into a vector space using
``node2vec`` when available.  If :mod:`networkx` or ``node2vec`` are missing we
fall back to the optional :mod:`workflow_vectorizer` package or finally to a
simple frequency based representation.

Cosine similarity between the embeddings together with structural heuristics
such as module overlap and entropy differences provide a cheap signal for
deciding whether two workflows are related.  The helper also attempts to look
up historical ROI and modularity (synergy) scores from
``roi_results_db.ROIResultsDB`` to provide additional context when available.

The main entry point is :meth:`WorkflowSynergyComparator.compare` which returns
an instance of :class:`ComparisonResult`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional third party dependencies -------------------------------------------------
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
    from workflow_vectorizer import WorkflowVectorizer  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade
    WorkflowVectorizer = None  # type: ignore

# Optional internal helpers ---------------------------------------------------------
try:  # pragma: no cover - optional
    from .workflow_graph import WorkflowGraph  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade
    WorkflowGraph = None  # type: ignore

try:  # pragma: no cover - optional
    from .workflow_metrics import compute_workflow_entropy  # type: ignore
except Exception:  # pragma: no cover - graceful fallback
    def compute_workflow_entropy(_spec: Dict[str, Any]) -> float:  # type: ignore
        return 0.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Metrics describing how similar two workflows are."""

    similarity: float
    """Cosine similarity between workflow embeddings."""

    shared_modules: float
    """Number of modules shared between both workflows."""

    entropy_gap: float
    """Absolute difference in Shannon entropy of module distributions."""

    roi_a: float
    roi_b: float
    modularity_a: float
    modularity_b: float

    # Additional context retained for consumers that previously relied on
    # these values.  They are not required by the user instructions but keep
    # backwards compatibility with parts of the codebase and tests.
    entropy_a: float
    entropy_b: float
    modules_a: int
    modules_b: int


# ---------------------------------------------------------------------------
# Comparator implementation
# ---------------------------------------------------------------------------


class WorkflowSynergyComparator:
    """Compare two workflow specifications using structural heuristics."""

    workflow_dir = Path("workflows")

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
                data = json.loads(path.read_text())
                if isinstance(data, dict):
                    return data
            except Exception:  # pragma: no cover - best effort
                return {}

        wf_file = WorkflowSynergyComparator.workflow_dir / f"{path.name}.workflow.json"
        if wf_file.is_file():
            try:
                data = json.loads(wf_file.read_text())
                if isinstance(data, dict):
                    return data
            except Exception:  # pragma: no cover - best effort
                return {}
        return {}

    @staticmethod
    def _extract_modules(spec: Dict[str, Any]) -> List[str]:
        steps = spec.get("steps", []) if isinstance(spec, dict) else []
        modules: List[str] = []
        for step in steps:
            if isinstance(step, dict):
                mod = step.get("module")
                if mod:
                    modules.append(mod)
        return modules

    @staticmethod
    def _build_graph(modules: List[str]):
        """Construct a dependency graph for ``modules``.

        When :mod:`workflow_graph` is available an instance is used as a source
        for potential edges between modules.  Failing that, a simple chain graph
        linking successive modules is produced.
        """

        if _HAS_NX:
            graph = nx.DiGraph()
            graph.add_nodes_from(modules)
            if WorkflowGraph is not None:
                try:  # pragma: no cover - optional behaviour
                    wg = WorkflowGraph()
                    for a in modules:
                        for b in modules:
                            try:
                                if wg.graph.has_edge(a, b):  # type: ignore[attr-defined]
                                    graph.add_edge(a, b)
                            except Exception:
                                pass
                except Exception:
                    pass
            if graph.number_of_edges() == 0:
                graph.add_edges_from(zip(modules, modules[1:]))
        else:
            graph: Dict[str, set] = {m: set() for m in modules}
            for a, b in zip(modules, modules[1:]):
                graph.setdefault(a, set()).add(b)
        return graph

    @staticmethod
    def _embed_graph(graph: Any, spec: Dict[str, Any]) -> List[float]:
        """Return an embedding for ``graph``/``spec``.

        The preferred strategy is ``node2vec`` operating on the constructed
        graph.  If that fails the optional :class:`WorkflowVectorizer` is used.
        As a last resort a simple sorted module frequency vector is returned.
        """

        if _HAS_NX and _HAS_NODE2VEC and isinstance(graph, nx.Graph):
            try:  # pragma: no cover - optional heavy dependency
                node2vec = Node2Vec(graph, dimensions=32, quiet=True, workers=1)
                model = node2vec.fit(window=4, min_count=1, batch_words=4)
                embeddings = [model.wv.get_vector(str(n)) for n in graph.nodes()]
                if embeddings:
                    dim = len(embeddings[0])
                    return [
                        sum(vec[i] for vec in embeddings) / len(embeddings)
                        for i in range(dim)
                    ]
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
    def _shared_ratio(mod_a: Iterable[str], mod_b: Iterable[str]) -> float:
        set_a = set(mod_a)
        set_b = set(mod_b)
        union = set_a | set_b
        return len(set_a & set_b) / len(union) if union else 0.0

    @staticmethod
    def _roi_and_modularity(workflow_id: str) -> Tuple[float, float]:
        """Best effort retrieval of ROI and modularity for ``workflow_id``."""

        try:  # pragma: no cover - optional IO
            from .roi_results_db import ROIResultsDB  # type: ignore

            db = ROIResultsDB()
            cur = db.conn.cursor()
            cur.execute(
                """
                SELECT roi_gain, workflow_synergy_score
                FROM workflow_results
                WHERE workflow_id=?
                ORDER BY timestamp DESC LIMIT 1
                """,
                (workflow_id,),
            )
            row = cur.fetchone()
            if row:
                return float(row[0] or 0.0), float(row[1] or 0.0)
        except Exception:
            pass

        try:  # pragma: no cover - optional heavy dependency
            from .workflow_metrics import compute_workflow_synergy  # type: ignore
            from .roi_tracker import ROITracker  # type: ignore

            tracker = ROITracker()
            return 0.0, float(compute_workflow_synergy(tracker))
        except Exception:
            pass

        return 0.0, 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @classmethod
    def compare(
        cls,
        a_spec: Dict[str, Any] | str | Path,
        b_spec: Dict[str, Any] | str | Path,
    ) -> ComparisonResult:
        """Compare two workflow specifications or identifiers."""

        spec_a = cls._load_spec(a_spec)
        spec_b = cls._load_spec(b_spec)

        modules_a = cls._extract_modules(spec_a)
        modules_b = cls._extract_modules(spec_b)

        graph_a = cls._build_graph(modules_a)
        graph_b = cls._build_graph(modules_b)

        vec_a = cls._embed_graph(graph_a, spec_a)
        vec_b = cls._embed_graph(graph_b, spec_b)

        similarity = cls._cosine(vec_a, vec_b)
        shared = len(set(modules_a) & set(modules_b))

        entropy_a = compute_workflow_entropy(spec_a)
        entropy_b = compute_workflow_entropy(spec_b)
        entropy_gap = abs(entropy_a - entropy_b)

        roi_a, mod_a = cls._roi_and_modularity(str(a_spec))
        roi_b, mod_b = cls._roi_and_modularity(str(b_spec))

        return ComparisonResult(
            similarity=similarity,
            shared_modules=shared,
            entropy_gap=entropy_gap,
            roi_a=roi_a,
            roi_b=roi_b,
            modularity_a=mod_a,
            modularity_b=mod_b,
            entropy_a=entropy_a,
            entropy_b=entropy_b,
            modules_a=len(set(modules_a)),
            modules_b=len(set(modules_b)),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def is_duplicate(
        result: ComparisonResult,
        similarity_threshold: float = 0.95,
        entropy_threshold: float = 0.05,
    ) -> bool:
        """Return ``True`` when ``result`` represents near-identical workflows.

        Parameters
        ----------
        result:
            :class:`ComparisonResult` produced by :meth:`compare`.
        similarity_threshold:
            Minimum cosine similarity to consider workflows duplicates.
        entropy_threshold:
            Maximum allowed entropy delta between workflows.
        """

        return (
            result.similarity >= similarity_threshold
            and result.entropy_gap <= entropy_threshold
        )

    # ------------------------------------------------------------------
    @classmethod
    def merge_duplicate(
        cls, base_id: str, dup_id: str, out_dir: str | Path = "workflows"
    ) -> Path | None:
        """Merge ``dup_id`` into ``base_id`` and refresh lineage information.

        The ``base_id`` workflow acts as the common ancestor and primary branch
        while ``dup_id`` is merged as the secondary branch.  The merged
        specification is written to ``<base_id>.merged.json`` within
        ``out_dir``.  Best-effort hooks update lineage caches and summary
        stores.
        """

        base_path = Path(out_dir) / f"{base_id}.workflow.json"
        dup_path = Path(out_dir) / f"{dup_id}.workflow.json"
        if not base_path.exists() or not dup_path.exists():
            return None

        out_path = Path(out_dir) / f"{base_id}.merged.json"
        try:
            from . import workflow_merger, workflow_lineage

            merged_path = workflow_merger.merge_workflows(
                base_path, base_path, dup_path, out_path
            )
        except Exception:
            return None

        wid = ""
        try:
            data = json.loads(merged_path.read_text())
            wid = str(data.get("metadata", {}).get("workflow_id") or "")
        except Exception:
            pass

        # Refresh lineage structures
        try:
            specs = list(workflow_lineage.load_specs(out_dir))
            workflow_lineage.build_graph(specs)
        except Exception:
            pass

        if wid:
            try:
                from .workflow_run_summary import save_summary

                save_summary(wid, Path(out_dir))
            except Exception:
                pass

            try:
                from .workflow_summary_db import WorkflowSummaryDB

                wid_int = int(wid)
                WorkflowSummaryDB().set_summary(
                    wid_int, f"merged {dup_id} into {base_id}"
                )
            except Exception:
                pass

        return merged_path


__all__ = ["WorkflowSynergyComparator", "ComparisonResult"]

