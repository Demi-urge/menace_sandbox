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

from dataclasses import dataclass, field
import json
import logging
import math
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import Counter

if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parent
    parent = package_root.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    package_name = package_root.name
    pkg = sys.modules.get(package_name)
    if pkg is None:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(package_root)]
        sys.modules[package_name] = pkg
    __package__ = package_name

try:  # pragma: no cover - allow execution without package context
    from .dynamic_path_router import resolve_path
except ImportError:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path  # type: ignore

logger = logging.getLogger(__name__)

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
except ImportError:  # pragma: no cover - fallback when executed directly
    from workflow_graph import WorkflowGraph  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade
    WorkflowGraph = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .roi_tracker import ROITracker  # type: ignore
except ImportError:  # pragma: no cover - fallback when executed directly
    from roi_tracker import ROITracker  # type: ignore
except BaseException:  # pragma: no cover - gracefully degrade
    ROITracker = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .roi_results_db import ROIResultsDB  # type: ignore
except ImportError:  # pragma: no cover - fallback when executed directly
    from roi_results_db import ROIResultsDB  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade
    ROIResultsDB = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from . import workflow_merger  # type: ignore
except ImportError:  # pragma: no cover - fallback when executed directly
    import workflow_merger  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade
    workflow_merger = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from . import workflow_run_summary  # type: ignore
except ImportError:  # pragma: no cover - fallback when executed directly
    import workflow_run_summary  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade
    workflow_run_summary = None  # type: ignore

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class OverfittingReport:
    """Report capturing potential overfitting signals."""

    low_entropy: bool
    """Whether the workflow entropy falls below the threshold."""

    repeated_modules: Dict[str, int]
    """Modules repeated more than allowed with their counts."""

    def is_overfitting(self) -> bool:
        """Return ``True`` if any overfitting signal is present."""
        return self.low_entropy or bool(self.repeated_modules)


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
    """Weighted mean of all synergy metrics."""

    overfit_a: Optional["OverfittingReport"] = None
    """Overfitting analysis for workflow ``A``."""

    overfit_b: Optional["OverfittingReport"] = None
    """Overfitting analysis for workflow ``B``."""

    best_practice_match_a: Tuple[float, List[str]] = field(
        default_factory=lambda: (0.0, [])
    )
    """Best practice sequence match for workflow ``A``."""

    best_practice_match_b: Tuple[float, List[str]] = field(
        default_factory=lambda: (0.0, [])
    )
    """Best practice sequence match for workflow ``B``."""


# ---------------------------------------------------------------------------
# Comparator implementation
# ---------------------------------------------------------------------------


class WorkflowSynergyComparator:
    """Compare workflow specifications using structural heuristics."""

    workflow_dir = resolve_path("workflows")
    _embed_cache: Dict[str, List[float]] = {}
    try:
        best_practices_file: Path = resolve_path("workflow_best_practices.json")
    except FileNotFoundError:
        best_practices_file = workflow_dir.parent / "workflow_best_practices.json"

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

        try:
            path = resolve_path(str(src))
        except FileNotFoundError:
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
    def _embed_spec(
        cls, spec: Dict[str, Any], all_modules: Optional[Iterable[str]] = None
    ) -> List[float]:
        """Return an embedding aligned to ``all_modules``.

        When ``all_modules`` is provided the resulting vector contains one
        contiguous block per module in the sorted union allowing vectors from
        different workflow specifications to be compared directly.  Missing
        modules are represented by zero vectors matching the dimensionality of
        present modules.
        """

        modules = sorted(set(cls._extract_modules(spec)))
        graph = cls._build_graph(modules)
        base_vec = cls._embed_graph(graph, spec)
        dim = len(base_vec) // len(modules) if modules else len(base_vec)
        if modules and (dim == 0 or len(base_vec) % len(modules) != 0):
            counts = Counter(modules)
            base_vec = [counts[m] for m in modules]
            dim = 1

        if not all_modules:
            return base_vec

        union = sorted(set(all_modules))
        aligned: List[float] = []
        index = {m: i for i, m in enumerate(modules)}
        for m in union:
            if m in index:
                start = index[m] * dim
                aligned.extend(base_vec[start:start + dim])
            else:
                aligned.extend([0.0] * dim)
        return aligned

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
                    logger.warning("Node2Vec embedding failed", exc_info=True)

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
                    logger.warning("Spectral embedding failed", exc_info=True)

        if WorkflowVectorizer is not None:
            try:  # pragma: no cover - optional dependency
                wf_record = {"workflow": [s.get("module") for s in spec.get("steps", [])]}
                vec = WorkflowVectorizer().fit([wf_record]).transform(wf_record)
                cls._embed_cache[key] = list(vec)
                return list(vec)
            except Exception:
                logger.warning("WorkflowVectorizer embedding failed", exc_info=True)

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
                logger.warning(
                    "Failed to retrieve entropy from ROITracker", exc_info=True
                )
        modules = WorkflowSynergyComparator._extract_modules(spec)
        counts: Dict[str, int] = {}
        for m in modules:
            counts[m] = counts.get(m, 0) + 1
        total = sum(counts.values())
        if not total:
            return 0.0
        return -sum((c / total) * math.log(c / total, 2) for c in counts.values())

    # ------------------------------------------------------------------
    @classmethod
    def _update_best_practices(cls, modules: List[str]) -> None:
        """Store ``modules`` in the best practices repository if unique."""

        path = cls.best_practices_file
        data: Dict[str, List[List[str]]] = {"sequences": []}
        if path.exists():
            try:
                loaded = json.loads(path.read_text())
                if isinstance(loaded, dict):
                    data.update(loaded)
            except Exception:
                logger.warning(
                    "Failed to read best practices from %s", path, exc_info=True
                )
        seqs = data.setdefault("sequences", [])
        if modules not in seqs:
            seqs.append(modules)
            try:
                path.write_text(json.dumps(data, indent=2))
            except Exception:
                logger.warning(
                    "Failed to write best practices to %s", path, exc_info=True
                )

    @classmethod
    def best_practice_match(cls, spec: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Return the best matching best-practice sequence for ``spec``.

        The method loads stored sequences from :attr:`best_practices_file`,
        embeds each sequence alongside the provided specification and returns
        the highest cosine similarity along with the matching sequence.  When
        no best practices are available ``(0.0, [])`` is returned.
        """

        path = cls.best_practices_file
        if not path.exists():
            return 0.0, []

        try:
            data = json.loads(path.read_text())
        except Exception:
            logger.warning(
                "Failed to load best practices from %s", path, exc_info=True
            )
            return 0.0, []

        sequences = data.get("sequences", []) if isinstance(data, dict) else []
        modules = cls._extract_modules(spec)
        best_score = 0.0
        best_seq: List[str] = []

        for seq in sequences:
            if not isinstance(seq, list):
                continue
            seq_mods = [str(m) for m in seq]
            all_mods = sorted(set(modules) | set(seq_mods))
            vec_spec = cls._embed_spec({"steps": [{"module": m} for m in modules]}, all_mods)
            vec_seq = cls._embed_spec({"steps": [{"module": m} for m in seq_mods]}, all_mods)
            score = cls._cosine(vec_spec, vec_seq)
            if score > best_score:
                best_score = score
                best_seq = seq_mods

        return best_score, best_seq

    # ------------------------------------------------------------------
    @classmethod
    def analyze_overfitting(
        cls,
        spec: Dict[str, Any] | str | Path,
        *,
        entropy_threshold: float = 1.0,
        repeat_threshold: int = 3,
    ) -> OverfittingReport:
        """Flag low entropy or frequent module repetition in ``spec``."""

        data = cls._load_spec(spec)
        modules = cls._extract_modules(data)
        entropy = cls._entropy(data)
        low_entropy = entropy < entropy_threshold
        counts: Dict[str, int] = {}
        for m in modules:
            counts[m] = counts.get(m, 0) + 1
        repeated = {m: c for m, c in counts.items() if c > repeat_threshold}
        report = OverfittingReport(low_entropy=low_entropy, repeated_modules=repeated)
        if not report.is_overfitting():
            cls._update_best_practices(modules)
        return report

    @classmethod
    def _modularity(cls, graph: Any, modules: Iterable[str]) -> float:
        """Return structural modularity for ``graph``.

        When :mod:`networkx` is available the function attempts community
        detection to derive a modularity score.  Otherwise the ratio of unique
        modules to total steps acts as a lightweight heuristic.
        """

        modules = list(modules)
        modularity = 0.0
        if _HAS_NX and isinstance(graph, nx.Graph):
            try:  # pragma: no cover - optional dependency
                nx_comm = getattr(getattr(nx, "algorithms", None), "community", None)
                if nx_comm is not None:
                    undirected = graph.to_undirected()
                    communities = None
                    louvain = getattr(nx_comm, "louvain_communities", None)
                    if callable(louvain):
                        communities = list(louvain(undirected, seed=0))
                    if not communities:
                        greedy = getattr(nx_comm, "greedy_modularity_communities", None)
                        if callable(greedy):
                            communities = list(greedy(undirected))
                    if communities:
                        modularity = float(nx_comm.modularity(undirected, communities))
            except Exception:
                logger.warning(
                    "Community modularity computation failed", exc_info=True
                )

        if not modularity and modules:
            modularity = len(set(modules)) / float(len(modules))

        return modularity

    @staticmethod
    def _workflow_id(src: Dict[str, Any] | str | Path, spec: Dict[str, Any]) -> str:
        """Best effort extraction of a workflow identifier."""

        if isinstance(src, (str, Path)):
            return Path(str(src)).stem
        return str(spec.get("id", ""))

    @staticmethod
    def _efficiency_from_db(workflow_id: str) -> float:
        """Return ROI/runtime ratio for ``workflow_id`` using ``ROIResultsDB``."""

        if workflow_id and ROIResultsDB is not None:
            try:  # pragma: no cover - optional dependency
                db = ROIResultsDB()
                results = db.fetch_results(workflow_id)
                if results:
                    latest = results[-1]
                    if latest.runtime:
                        return float(latest.roi_gain) / float(latest.runtime)
            except Exception:
                logger.warning(
                    "Failed to fetch efficiency from ROIResultsDB", exc_info=True
                )
        if ROITracker is not None:
            try:  # pragma: no cover - optional dependency
                tracker = ROITracker()
                eff_hist = tracker.metrics_history.get("synergy_efficiency", [])
                if eff_hist:
                    return float(eff_hist[-1])
            except Exception:
                logger.warning(
                    "Failed to fetch efficiency from ROITracker", exc_info=True
                )
        return 0.0

    @classmethod
    def _roi_and_modularity(
        cls,
        wid_a: str,
        wid_b: str,
        graph_a: Any,
        graph_b: Any,
        mods_a: Iterable[str],
        mods_b: Iterable[str],
    ) -> Tuple[float, float]:
        """Return efficiency and modularity for the given workflows."""

        eff_a = cls._efficiency_from_db(wid_a)
        eff_b = cls._efficiency_from_db(wid_b)
        efficiency = (eff_a + eff_b) / 2 if (eff_a or eff_b) else 0.0
        mod_a = cls._modularity(graph_a, mods_a)
        mod_b = cls._modularity(graph_b, mods_b)
        modularity = (mod_a + mod_b) / 2 if (mod_a or mod_b) else 0.0
        return efficiency, modularity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @classmethod
    def compare(
        cls,
        a_spec: Dict[str, Any] | str | Path,
        b_spec: Dict[str, Any] | str | Path,
        *,
        weights: Optional[Dict[str, float]] = None,
    ) -> SynergyScores:
        """Compare two workflow specifications or identifiers.

        Parameters
        ----------
        a_spec, b_spec:
            Workflow specifications or identifiers pointing to
            ``{id}.workflow.json`` files.
        weights:
            Optional mapping assigning relative weights to ``similarity``,
            ``shared_modules``, ``entropy`` (expandability), ``efficiency`` and
            ``modularity``.  When omitted each metric is weighted equally.  A
            weight of ``0`` removes the corresponding metric from the aggregate
            score.
        """

        spec_a = cls._load_spec(a_spec)
        spec_b = cls._load_spec(b_spec)

        mods_a = cls._extract_modules(spec_a)
        mods_b = cls._extract_modules(spec_b)
        all_mods = sorted(set(mods_a) | set(mods_b))

        vec_a = cls._embed_spec(spec_a, all_mods)
        vec_b = cls._embed_spec(spec_b, all_mods)

        similarity = cls._cosine(vec_a, vec_b)
        shared_ratio = cls._shared_ratio(mods_a, mods_b)
        entropy_a = cls._entropy(spec_a)
        entropy_b = cls._entropy(spec_b)
        expandability = (entropy_a + entropy_b) / 2 if (entropy_a or entropy_b) else 0.0

        overfit_a = cls.analyze_overfitting(spec_a)
        overfit_b = cls.analyze_overfitting(spec_b)

        best_a = cls.best_practice_match(spec_a)
        best_b = cls.best_practice_match(spec_b)

        # Additional metrics derived from structural communities and ROI history
        graph_a = cls._build_graph(mods_a)
        graph_b = cls._build_graph(mods_b)
        wid_a = cls._workflow_id(a_spec, spec_a)
        wid_b = cls._workflow_id(b_spec, spec_b)
        efficiency, modularity = cls._roi_and_modularity(
            wid_a, wid_b, graph_a, graph_b, mods_a, mods_b
        )

        metrics = {
            "similarity": similarity,
            "shared_modules": shared_ratio,
            "entropy": expandability,
            "efficiency": efficiency,
            "modularity": modularity,
        }
        w = {name: 1.0 for name in metrics}
        if weights:
            for k, v in weights.items():
                if k in w:
                    w[k] = float(v)
        total = sum(val for key, val in w.items() if val > 0)
        weighted_sum = sum(metrics[m] * w[m] for m in metrics if w[m] > 0)
        aggregate = weighted_sum / total if total else 0.0

        return SynergyScores(
            similarity=similarity,
            shared_module_ratio=shared_ratio,
            entropy_a=entropy_a,
            entropy_b=entropy_b,
            expandability=expandability,
            efficiency=efficiency,
            modularity=modularity,
            aggregate=aggregate,
            overfit_a=overfit_a,
            overfit_b=overfit_b,
            best_practice_match_a=best_a,
            best_practice_match_b=best_b,
        )

    # ------------------------------------------------------------------
    @classmethod
    def is_duplicate(
        cls,
        a: SynergyScores | Dict[str, Any] | str | Path,
        b: Optional[Dict[str, Any] | str | Path] = None,
        *,
        similarity_threshold: float = 0.95,
        entropy_threshold: float = 0.05,
    ) -> bool:
        """Return ``True`` when two workflows are near-identical.

        This method accepts either the :class:`SynergyScores` produced by
        :meth:`compare` or two workflow specifications / identifiers.  When
        ``a`` is a :class:`SynergyScores` instance ``b`` must be ``None`` and
        the pre-computed scores are used directly.  Otherwise ``a`` and ``b``
        are forwarded to :meth:`compare` and the result evaluated against the
        provided thresholds.
        """

        if isinstance(a, SynergyScores):
            if b is not None:
                raise ValueError("second argument not allowed when passing SynergyScores")
            scores = a
        else:
            if b is None:
                raise ValueError("two workflow specs or identifiers required")
            scores = cls.compare(a, b)

        ent_gap = abs(scores.entropy_a - scores.entropy_b)
        return scores.similarity >= similarity_threshold and ent_gap <= entropy_threshold

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

        if out_dir is not None:
            try:
                directory = resolve_path(str(out_dir))
            except FileNotFoundError:
                directory = Path(out_dir)
        else:
            directory = cls.workflow_dir
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

    try:
        directory = resolve_path(str(out_dir))
    except FileNotFoundError:
        directory = Path(out_dir)
    base = directory / f"{base_id}.workflow.json"
    dup = directory / f"{dup_id}.workflow.json"
    out = directory / f"{base_id}.merged.json"

    if not base.exists() or not dup.exists():
        return None

    try:
        # Ensure both workflow files contain valid JSON
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
            logger.warning(
                "Failed to save workflow run summaries to %s", directory, exc_info=True
            )

    return merged


__all__ = [
    "WorkflowSynergyComparator",
    "SynergyScores",
    "OverfittingReport",
    "merge_duplicate",
]
