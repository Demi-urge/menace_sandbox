from __future__ import annotations

"""Meta workflow planning utilities with semantic and structural embedding."""

from dataclasses import dataclass, field
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, TYPE_CHECKING
import sys
from statistics import fmean, pvariance
from concurrent.futures import ThreadPoolExecutor

from governed_embeddings import governed_embed, get_embedder

from roi_results_db import ROIResultsDB
from workflow_graph import WorkflowGraph
from vector_utils import persist_embedding, cosine_similarity
from cache_utils import get_cached_chain, set_cached_chain
from logging_utils import get_logger

try:  # pragma: no cover - allow running as script
    from .dynamic_path_router import resolve_path  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path  # type: ignore

logger = get_logger(__name__)

# Decay factor applied when persisting/loading cluster metrics to favor recent runs
_DECAY_FACTOR = 0.9
# Number of consecutive runs without improvement before pruning a chain
_PRUNE_RUNS = 50

try:  # pragma: no cover - compute default path for chain embeddings
    _CHAIN_EMBEDDINGS_PATH = resolve_path("sandbox_data/embeddings.jsonl")
except FileNotFoundError:  # pragma: no cover - file may not exist yet
    _CHAIN_EMBEDDINGS_PATH = resolve_path("sandbox_data") / "embeddings.jsonl"

try:  # pragma: no cover - optional heavy dependency
    from vector_service.retriever import Retriever  # type: ignore
    from vector_service.context_builder import ContextBuilder  # type: ignore
except Exception:  # pragma: no cover - allow running without retriever
    logger.warning("vector_service.retriever import failed; similar search disabled")
    Retriever = None  # type: ignore
    ContextBuilder = None  # type: ignore
from context_builder_util import ensure_fresh_weights

try:  # pragma: no cover - optional heavy dependency
    from roi_tracker import ROITracker  # type: ignore
except Exception:  # pragma: no cover - allow running without ROI tracker
    logger.warning("roi_tracker import failed; using no-op ROITracker fallback")

    class ROITracker:  # type: ignore[no-redef]
        """Deterministic no-op fallback used when ``roi_tracker`` is unavailable."""

        def __init__(self, *_, **__):
            pass

        def record_scenario_delta(self, *_, **__):
            """No-op fallback implementation."""

            logger.debug("ROITracker fallback invoked; no data recorded")

try:  # pragma: no cover - optional persistence helper
    from workflow_stability_db import WorkflowStabilityDB  # type: ignore
except Exception:  # pragma: no cover - allow running without stability db
    logger.warning(
        "workflow_stability_db import failed; stability tracking disabled"
    )
    WorkflowStabilityDB = None  # type: ignore

try:  # pragma: no cover - optional heavy dependency
    import networkx as nx  # type: ignore
    _HAS_NX = True
except Exception:  # pragma: no cover - executed when networkx missing
    logger.warning("networkx import failed; graph features will be limited")
    nx = None  # type: ignore
    _HAS_NX = False

try:  # pragma: no cover - optional heavy dependency
    from workflow_synergy_comparator import WorkflowSynergyComparator  # type: ignore
except Exception:  # pragma: no cover - allow running without comparator
    logger.warning(
        "workflow_synergy_comparator import failed; synergy metrics disabled"
    )

    class WorkflowSynergyComparator:  # type: ignore[no-redef]
        """Fallback comparator returning neutral metrics."""

        @staticmethod
        def compare(*_, **__):
            return type("Result", (), {"aggregate": 0.0})()

        @staticmethod
        def _entropy(*_, **__):
            return 0.0

try:  # pragma: no cover - optional code database
    from code_database import CodeDB  # type: ignore
except Exception:  # pragma: no cover - database unavailable
    logger.warning("code_database import failed; code context features disabled")
    CodeDB = None  # type: ignore
    sys.modules.pop("code_database", None)

try:  # pragma: no cover - optional persistence helper
    from . import synergy_history_db as shd  # type: ignore
except Exception:  # pragma: no cover - fallback when run as script
    import synergy_history_db as shd  # type: ignore

try:  # pragma: no cover - optional clustering dependency
    from sklearn.cluster import DBSCAN  # type: ignore
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - allow running without scikit-learn
    logger.warning("scikit-learn import failed; clustering disabled")
    DBSCAN = None  # type: ignore
    _HAS_SKLEARN = False

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sandbox_runner.workflow_sandbox_runner import WorkflowSandboxRunner


def _get_index(value: Any, mapping: Dict[str, int], max_size: int) -> int:
    """Return index for ``value`` expanding ``mapping`` on demand."""

    val = str(value).lower().strip() or "other"
    if val in mapping:
        return mapping[val]
    if len(mapping) < max_size:
        mapping[val] = len(mapping)
        return mapping[val]
    return mapping["other"]


@dataclass
class MetaWorkflowPlanner:
    """Encode workflows with structural and semantic context."""

    context_builder: ContextBuilder
    max_functions: int = 50
    max_modules: int = 50
    max_tags: int = 50
    max_domains: int = 10
    roi_window: int = 5
    domain_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})
    graph: WorkflowGraph | None = None
    roi_db: ROIResultsDB | None = None
    roi_tracker: ROITracker | None = None
    stability_db: WorkflowStabilityDB | None = None
    code_db: CodeDB | None = None
    # Map of workflow chains to ROI histories for convergence tracking
    cluster_map: Dict[tuple[str, ...], Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.graph is None:
            try:
                self.graph = WorkflowGraph()
            except Exception:
                self.graph = None
        if self.roi_db is None:
            try:
                self.roi_db = ROIResultsDB()
            except Exception:
                self.roi_db = None
        if self.roi_tracker is None and ROITracker is not None:
            try:
                self.roi_tracker = ROITracker(
                    window=self.roi_window, results_db=self.roi_db
                )
            except Exception:
                self.roi_tracker = None
        if self.stability_db is None and WorkflowStabilityDB is not None:
            try:
                self.stability_db = WorkflowStabilityDB()
            except Exception:
                self.stability_db = None
        if self.code_db is None and CodeDB is not None:
            try:
                self.code_db = CodeDB()
            except Exception:
                self.code_db = None
        if self.context_builder is None:
            raise ValueError("context_builder is required")
        if not hasattr(self.context_builder, "build"):
            raise TypeError("context_builder must implement build()")
        try:
            ensure_fresh_weights(self.context_builder)
        except Exception as exc:
            logger.error("context builder refresh failed: %s", exc)
            raise RuntimeError("context builder refresh failed") from exc
        self._load_cluster_map()

    # ------------------------------------------------------------------
    def begin_run(self, workflow_id: str, run_id: str) -> None:
        """Configure trackers for a new sandbox run.

        ``ROITracker`` persists per-module ROI deltas when a run context is
        provided.  Sandbox orchestrators should invoke this hook with the
        workflow identifier and a unique ``run_id`` before recording metrics.
        """

        if self.roi_tracker is not None:
            try:
                self.roi_tracker.set_run_context(workflow_id, run_id, self.roi_db)
            except Exception:
                logger.exception("failed to configure ROITracker run context")

    # ------------------------------------------------------------------
    def encode(self, workflow_id: str, workflow: Mapping[str, Any]) -> List[float]:
        """Return embedding for ``workflow`` and persist it."""

        depth, branching = self._graph_features(workflow_id)
        roi_curve = self._roi_curve(workflow_id)
        (
            funcs,
            mods,
            tags,
            mod_cats,
            depths,
            branchings,
            curves,
            rois,
            failures,
        ) = self._semantic_tokens(workflow, workflow_id)
        d_indices, d_labels = self._workflow_domain(
            workflow_id, {workflow_id: workflow}
        )
        if d_labels:
            tags.extend(d_labels)

        # Merge in module categories from the code database
        mods.extend(mod_cats)

        # Normalize tokens prior to embedding
        norm_funcs = sorted({f.lower().strip() for f in funcs if f})
        norm_mods = sorted({m.lower().strip() for m in mods if m})
        norm_tags = sorted({t.lower().strip() for t in tags if t})

        code_depth = max(depths) if depths else 0.0
        code_branching = max(branchings) if branchings else 0.0
        code_curve = [0.0] * self.roi_window
        if curves:
            for curve in curves:
                for i, val in enumerate(curve[: self.roi_window]):
                    code_curve[i] += float(val)
            code_curve = [v / len(curves) for v in code_curve]

        vec: List[float] = []
        vec.extend([depth, branching])
        vec.extend(roi_curve)
        vec.extend([code_depth, code_branching])
        vec.extend(code_curve)
        func_vec = self._embed_tokens(norm_funcs)
        mod_vec = self._embed_tokens(norm_mods)
        tag_vec = self._embed_tokens(norm_tags)
        vec.extend(func_vec)
        vec.extend(mod_vec)
        vec.extend(tag_vec)
        domain_vec = [0.0] * self.max_domains
        trans_vec = [0.0] * self.max_domains
        if d_indices:
            for idx in d_indices:
                if 0 <= idx < self.max_domains:
                    domain_vec[idx] = 1.0
            trans_probs = self.transition_probabilities()
            for (src, dst), prob in trans_probs.items():
                if src in d_indices and 0 <= dst < self.max_domains:
                    trans_vec[dst] = max(trans_vec[dst], float(prob))
        vec.extend(domain_vec)
        vec.extend(trans_vec)

        code_tags = norm_tags
        embed_dim = len(func_vec)

        try:
            persist_embedding(
                "workflow_meta",
                workflow_id,
                vec,
                origin_db="workflow",
                metadata={
                    "roi_curve": roi_curve,
                    "code_tags": code_tags,
                    "dependency_depth": depth,
                    "branching_factor": branching,
                    "domains": d_labels,
                    "domain_indices": d_indices,
                    "domain_transitions": trans_vec,
                    "vector_schema": {
                        "graph": 2,
                        "roi_curve": self.roi_window,
                        "code_graph": 2,
                        "code_roi_curve": self.roi_window,
                        "function_embedding": embed_dim,
                        "module_embedding": embed_dim,
                        "tag_embedding": embed_dim,
                        "domain": self.max_domains,
                        "domain_transition": self.max_domains,
                    },
                },
            )
        except TypeError:  # pragma: no cover - compatibility shim
            persist_embedding("workflow_meta", workflow_id, vec)
        return vec

    # ------------------------------------------------------------------
    def encode_workflow(self, workflow_id: str, workflow: Mapping[str, Any]) -> List[float]:
        """Return lightweight embedding for ``workflow``.

        The method derives simple structural signals from :class:`WorkflowGraph`
        such as dependency depth and branching factor and augments them with
        basic semantic token presence.  Unlike :meth:`encode`, this variant does
        not persist the resulting vector which makes it suitable for transient
        clustering operations.
        """

        depth, branching = self._graph_features(workflow_id)
        roi_curve = self._roi_curve(workflow_id)
        (
            funcs,
            mods,
            tags,
            mod_cats,
            depths,
            branchings,
            curves,
            rois,
            failures,
        ) = self._semantic_tokens(workflow, workflow_id)
        d_indices, d_labels = self._workflow_domain(
            workflow_id, {workflow_id: workflow}
        )
        if d_labels:
            tags.extend(d_labels)

        mods.extend(mod_cats)

        norm_funcs = sorted({f.lower().strip() for f in funcs if f})
        norm_mods = sorted({m.lower().strip() for m in mods if m})
        norm_tags = sorted({t.lower().strip() for t in tags if t})

        code_depth = max(depths) if depths else 0.0
        code_branching = max(branchings) if branchings else 0.0
        code_curve = [0.0] * self.roi_window
        if curves:
            for curve in curves:
                for i, val in enumerate(curve[: self.roi_window]):
                    code_curve[i] += float(val)
            code_curve = [v / len(curves) for v in code_curve]

        vec: List[float] = [depth, branching]
        vec.extend(roi_curve)
        vec.extend([code_depth, code_branching])
        vec.extend(code_curve)
        vec.extend(self._embed_tokens(norm_funcs))
        vec.extend(self._embed_tokens(norm_mods))
        vec.extend(self._embed_tokens(norm_tags))
        domain_vec = [0.0] * self.max_domains
        trans_vec = [0.0] * self.max_domains
        if d_indices:
            for idx in d_indices:
                if 0 <= idx < self.max_domains:
                    domain_vec[idx] = 1.0
            trans_probs = self.transition_probabilities()
            for (src, dst), prob in trans_probs.items():
                if src in d_indices and 0 <= dst < self.max_domains:
                    trans_vec[dst] = max(trans_vec[dst], float(prob))
        vec.extend(domain_vec)
        vec.extend(trans_vec)
        return vec

    # ------------------------------------------------------------------
    def encode_chain(self, chain: Sequence[str]) -> List[float]:
        """Encode a workflow ``chain`` and persist its embedding.

        The chain is converted into a minimal workflow representation where each
        element is treated as a function step.  The resulting vector is persisted
        to the ``workflow_chain`` database and the planner's ``cluster_map`` is
        updated with the computed embedding.
        """

        chain_id = "->".join(chain)
        cached = get_cached_chain(chain_id)
        if cached is not None:
            vec = list(cached)
        else:
            workflow = {"workflow": [{"function": step} for step in chain]}
            vec = self.encode(chain_id, workflow)
            try:
                persist_embedding(
                    "workflow_chain",
                    chain_id,
                    vec,
                    origin_db="workflow",
                    path=_CHAIN_EMBEDDINGS_PATH,
                )
            except TypeError:  # pragma: no cover - compatibility shim
                persist_embedding("workflow_chain", chain_id, vec, path=_CHAIN_EMBEDDINGS_PATH)
            set_cached_chain(chain_id, vec)
        info = self.cluster_map.setdefault(tuple(chain), {})
        info["embedding"] = vec
        self._save_cluster_map()
        return vec

    # ------------------------------------------------------------------
    def cleanup_chain_embeddings(self, *, path: str | Path = _CHAIN_EMBEDDINGS_PATH) -> None:
        """Remove obsolete chain embeddings from the vector store."""

        active = {"->".join(k) for k in self.cluster_map}
        store = Path(resolve_path(path)) if isinstance(path, str) else Path(path)
        if not store.exists():
            return
        try:  # pragma: no cover - best effort
            lines = store.read_text().splitlines()
            with store.open("w", encoding="utf-8") as fh:
                for line in lines:
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    if data.get("type") != "workflow_chain":
                        fh.write(line + "\n")
                        continue
                    if data.get("id") in active:
                        fh.write(line + "\n")
        except Exception:
            logger.warning("Failed to cleanup chain embeddings", exc_info=True)

    # ------------------------------------------------------------------
    def _failure_entropy_metrics(self, workflow_id: str) -> tuple[float, float]:
        """Return recent failure rate and entropy for ``workflow_id``.

        The method first consults :class:`ROIResultsDB` to fetch the most
        recent result for ``workflow_id`` and derives the failure rate from the
        ``success_rate`` column.  When available, the ``__aggregate__`` entry of
        ``module_deltas`` is inspected for an ``entropy_mean`` field.  Missing
        database information falls back to the planner's ``cluster_map`` where
        historic ``failure_history`` and ``entropy_history`` are maintained for
        evaluated chains.  Both values are clamped to the ``[0.0, 1.0]`` range
        before being returned.
        """

        failure = 0.0
        entropy = 0.0
        if self.roi_db is not None:
            try:
                results = self.roi_db.fetch_results(workflow_id)
                if results:
                    last = results[-1]
                    failure = 1.0 - float(getattr(last, "success_rate", 1.0))
                    agg = getattr(last, "module_deltas", {}).get("__aggregate__", {})
                    entropy = float(agg.get("entropy_mean", agg.get("entropy", 0.0)))
            except Exception:
                failure = 0.0
                entropy = 0.0

        info = self.cluster_map.get((workflow_id,), {})
        if not failure:
            hist = info.get("failure_history") or []
            if hist:
                failure = float(hist[-1])
        if not entropy:
            hist = info.get("entropy_history") or []
            if hist:
                entropy = float(hist[-1])

        failure = max(0.0, min(1.0, failure))
        entropy = max(0.0, min(1.0, entropy))
        return failure, entropy

    # ------------------------------------------------------------------
    def cluster_workflows(
        self,
        workflows: Mapping[str, Mapping[str, Any]],
        *,
        context_builder: ContextBuilder | None = None,
        retriever: Retriever | None = None,
        epsilon: float = 0.5,
        min_samples: int = 2,
    ) -> List[List[str]]:
        """Group ``workflows`` into similarity clusters.

        ``retriever`` must be an initialised :class:`vector_service.retriever.Retriever`
        instance.  Each workflow is encoded via :meth:`encode_workflow`, persisted
        in the ``workflow_meta`` vector database and the embedding is used to
        query the retriever.  The cosine similarity of a query and candidate
        vector is multiplied by ``(1 + ROI)`` for both workflows using weights
        from :func:`_roi_weight_from_db` and further scaled by ``(1 -
        failure_rate) * (1 - entropy)`` for each workflow.  Weighted similarities
        are cached to speed up repeated invocations and then normalised.  When
        scikit-learn is available, the resulting distance matrix is clustered via
        :class:`sklearn.cluster.DBSCAN`.  If scikit-learn is unavailable a
        lightweight similarity-threshold grouping is used instead where
        ``epsilon`` acts as the distance threshold and ``min_samples`` determines
        the minimum component size before a cluster is accepted.  Retrieval
        results are cached via :mod:`cache_utils` to avoid repeated brute-force
        scans across invocations. Missing ROI information results in unweighted
        similarity scores so that the function remains best effort.
        """

        context_builder = context_builder or self.context_builder
        if retriever is None and Retriever is not None:
            try:
                retriever = Retriever(context_builder=context_builder)
            except Exception:
                retriever = None
        if retriever is None or Retriever is None:
            raise ValueError(
                "cluster_workflows requires an initialised Retriever"
            )

        ids = list(workflows.keys())
        if not ids:
            return []

        vecs: Dict[str, List[float]] = {}
        roi_map: Dict[str, float] = {}
        failure_map: Dict[str, float] = {}
        entropy_map: Dict[str, float] = {}
        for wid in ids:
            vec = self.encode_workflow(wid, workflows[wid])
            vecs[wid] = vec
            roi_map[wid] = (
                _roi_weight_from_db(self.roi_db, wid) if self.roi_db is not None else 0.0
            )
            fail, ent = self._failure_entropy_metrics(wid)
            failure_map[wid] = fail
            entropy_map[wid] = ent
            try:
                persist_embedding("workflow_meta", wid, vec, origin_db="workflow")
            except TypeError:  # pragma: no cover - legacy signature
                persist_embedding("workflow_meta", wid, vec)

        sims: Dict[str, Dict[str, float]] = {wid: {} for wid in ids}

        ur = retriever._get_retriever()
        for wid, vec in vecs.items():
            cache_key = json.dumps(vec, sort_keys=True)
            cached_hits = get_cached_chain(cache_key, ["workflow_meta"])
            if cached_hits is None:
                try:  # pragma: no cover - best effort
                    hits, _, _ = ur.retrieve(
                        vec, top_k=len(ids) * 2, dbs=["workflow_meta"]
                    )  # type: ignore[attr-defined]
                    cached_hits = []
                    for h in hits:
                        other = str(
                            getattr(h, "record_id", None)
                            or getattr(getattr(h, "metadata", {}), "get", lambda *_: None)("id")
                            or ""
                        )
                        if not other or other == wid or other not in vecs:
                            continue
                        base_sim = cosine_similarity(vec, vecs[other])
                        cached_hits.append({"record_id": other, "score": base_sim})
                    try:
                        set_cached_chain(cache_key, ["workflow_meta"], cached_hits)
                    except Exception:
                        pass
                except Exception:
                    cached_hits = []

            for hit in cached_hits:
                other = str(hit.get("record_id", ""))
                if not other or other == wid or other not in vecs:
                    continue
                base_sim = float(hit.get("score", 0.0))
                sim = base_sim
                sim *= (1.0 + roi_map[wid]) * (1.0 + roi_map[other])
                sim *= (1.0 - failure_map[wid]) * (1.0 - failure_map[other])
                sim *= (1.0 - entropy_map[wid]) * (1.0 - entropy_map[other])
                if sim > sims[wid].get(other, 0.0):
                    sims[wid][other] = sim
                    sims[other][wid] = sim
        # Normalise similarities and build distance matrix for DBSCAN
        max_sim = max((max(d.values()) for d in sims.values() if d), default=1.0)
        if max_sim <= 0.0:
            max_sim = 1.0

        dist_matrix: List[List[float]] = []
        for wid1 in ids:
            row: List[float] = []
            for wid2 in ids:
                if wid1 == wid2:
                    row.append(0.0)
                else:
                    sim = sims[wid1].get(wid2, 0.0) / max_sim
                    row.append(1.0 - sim)
            dist_matrix.append(row)

        if _HAS_SKLEARN and DBSCAN is not None:
            clustering = DBSCAN(
                eps=epsilon, min_samples=min_samples, metric="precomputed"
            )
            labels = clustering.fit_predict(dist_matrix)

            label_map: Dict[int, List[str]] = defaultdict(list)
            noise: List[List[str]] = []
            for wid, label in zip(ids, labels):
                if int(label) == -1:
                    noise.append([wid])
                else:
                    label_map[int(label)].append(wid)

            clusters: List[List[str]] = list(label_map.values())
            clusters.extend(noise)
            return clusters

        # Fallback when scikit-learn is unavailable.  Workflows whose
        # normalised similarity exceeds ``1 - epsilon`` are linked and connected
        # components are returned as clusters.  Components smaller than
        # ``min_samples`` are treated as noise and returned as single-item
        # clusters.
        threshold = 1.0 - epsilon
        norm_sims: Dict[str, Dict[str, float]] = {
            wid: {other: sims[wid].get(other, 0.0) / max_sim for other in ids if other != wid}
            for wid in ids
        }

        visited: set[str] = set()
        clusters: List[List[str]] = []
        for wid in ids:
            if wid in visited:
                continue
            queue = [wid]
            component: List[str] = []
            visited.add(wid)
            while queue:
                cur = queue.pop()
                component.append(cur)
                for other, sim in norm_sims[cur].items():
                    if sim >= threshold and other not in visited:
                        visited.add(other)
                        queue.append(other)

            if len(component) >= min_samples:
                clusters.append(component)
            else:
                for n in component:
                    clusters.append([n])

        return clusters

    # ------------------------------------------------------------------
    def _workflow_domain(
        self,
        workflow_id: str,
        workflows: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[List[int], List[str]]:
        """Return ``([indices], [labels])`` for ``workflow_id``'s domains.

        Historically a single domain label was returned for a workflow.  The
        planner now supports multiple domains per workflow by deriving all
        available context tags from the :class:`CodeDB` (when available) and
        fallback workflow metadata.  Each discovered domain label is normalised
        to lower case and converted into an index via :func:`_get_index` with
        ``"other"`` acting as a catch-all bucket.
        """

        labels: List[str] = []
        if self.code_db is not None:
            try:
                if hasattr(self.code_db, "get_context_tags"):
                    tags = self.code_db.get_context_tags(workflow_id) or []
                    labels.extend(str(t).lower() for t in tags if t)
                elif hasattr(self.code_db, "context_tags"):
                    tags = self.code_db.context_tags(  # type: ignore[attr-defined]
                        workflow_id
                    ) or []
                    labels.extend(str(t).lower() for t in tags if t)
                elif hasattr(self.code_db, "get_domain"):
                    d = self.code_db.get_domain(workflow_id)
                    if d:
                        labels.append(str(d).lower())
                elif hasattr(self.code_db, "get_platform"):
                    d = self.code_db.get_platform(workflow_id)
                    if d:
                        labels.append(str(d).lower())
            except Exception:
                labels = []

        if not labels and workflows is not None:
            wf = workflows.get(workflow_id, {})
            meta = wf.get("domain") or wf.get("platform") or wf.get("category")
            if isinstance(meta, str):
                labels.append(meta.lower())
            elif isinstance(meta, Iterable):
                labels.extend(str(m).lower() for m in meta if m)

        labels = [lbl for i, lbl in enumerate(labels) if lbl and lbl not in labels[:i]]
        if not labels:
            return [], []
        indices = [
            _get_index(lbl, self.domain_index, self.max_domains) for lbl in labels
        ]
        return indices, labels

    # ------------------------------------------------------------------
    def compose_pipeline(
        self,
        start: str,
        workflows: Mapping[str, Mapping[str, Any]],
        *,
        length: int = 3,
        similarity_weight: float = 1.0,
        synergy_weight: float = 1.0,
        roi_weight: float = 1.0,
        context_builder: ContextBuilder | None = None,
        retriever: Retriever | None = None,
    ) -> List[str]:
        """Compose a workflow pipeline using retrieval to limit candidates.

        ``retriever`` (when provided or available globally) is used via
        :func:`find_synergy_candidates` to obtain a shortlist of potential next
        steps for the current workflow.  Each candidate is then scored using the
        retrieved cosine ``similarity``, structural ``synergy`` and recent ``ROI``
        trends along with historic domain transition probabilities.  Synergy
        scores are sourced from :class:`WorkflowSynergyComparator` when
        available, otherwise any historic ``cluster_map`` pair metrics are used.
        If retrieval fails a best-effort fallback to exhaustive iteration is
        performed.

        The final ranking score is ``(similarity * similarity_weight)`` scaled by
        ``(1 + synergy_weight * synergy)`` and ``(1 + ROI * roi_weight)`` then
        further multiplied by ``1 + transition_prob`` where ``transition_prob``
        reflects empirical ROI deltas between workflow domains.  The method
        stops once ``length`` steps have been selected or no compatible
        candidates remain.
        """

        if start not in workflows:
            return []

        pipeline = [start]
        available = {k for k in workflows.keys() if k != start}
        current = start
        graph = self.graph or WorkflowGraph()

        current_vec = self.encode_workflow(start, workflows[start])
        prev_domains = self._workflow_domain(start, workflows)[0]
        current_roi = (
            _roi_weight_from_db(self.roi_db, start) if self.roi_db is not None else 0.0
        )

        self.cluster_map.setdefault(("__domain_transitions__",), {})
        trans_probs = self.transition_probabilities()

        context_builder = context_builder or self.context_builder
        if retriever is None and Retriever is not None:
            try:  # pragma: no cover - best effort
                retriever = Retriever(context_builder=context_builder)
            except Exception:  # pragma: no cover - fallback to exhaustive search
                retriever = None

        while available and len(pipeline) < length:
            candidates: List[tuple[str, List[float], float, float]] = []
            if retriever is not None:
                try:
                    cands = find_synergy_candidates(
                        current,
                        top_k=len(available),
                        retriever=retriever,
                        context_builder=context_builder,
                        roi_db=self.roi_db,
                        roi_window=self.roi_window,
                        cluster_map=self.cluster_map,
                    )
                except Exception:
                    cands = []
                for cand in cands:
                    wid = cand.get("workflow_id")
                    if wid not in available or not _io_compatible(graph, current, wid):
                        continue
                    cand_vec = self.encode_workflow(wid, workflows[wid])
                    candidates.append(
                        (
                            wid,
                            cand_vec,
                            float(cand.get("similarity", 0.0)),
                            float(cand.get("roi", 0.0)),
                        )
                    )

            if not candidates:
                for wid in list(available):
                    if not _io_compatible(graph, current, wid):
                        continue
                    cand_vec = self.encode_workflow(wid, workflows[wid])
                    cand_roi = (
                        _roi_weight_from_db(self.roi_db, wid)
                        if self.roi_db is not None
                        else 0.0
                    )
                    sim = cosine_similarity(current_vec, cand_vec)
                    candidates.append((wid, cand_vec, sim, cand_roi))

            best_id: str | None = None
            best_score = -1.0
            best_vec: List[float] | None = None
            best_roi = 0.0
            for wid, cand_vec, sim, cand_roi in candidates:
                synergy = 0.0
                if synergy_weight:
                    try:
                        scores = WorkflowSynergyComparator.compare(
                            workflows[current], workflows[wid]
                        )
                        synergy = float(getattr(scores, "aggregate", 0.0))
                    except Exception:
                        synergy = 0.0
                    if synergy == 0.0:
                        synergy = float(
                            self.cluster_map.get((current, wid), {}).get("score", 0.0)
                        )

                score = similarity_weight * sim
                if synergy_weight:
                    score *= 1.0 + synergy_weight * synergy
                roi_avg = (current_roi + cand_roi) / 2.0
                score *= 1.0 + roi_weight * roi_avg

                cand_domains = self._workflow_domain(wid, workflows)[0]
                if prev_domains and cand_domains:
                    prob = 0.0
                    for src in prev_domains:
                        for dst in cand_domains:
                            prob = max(prob, trans_probs.get((src, dst), 0.0))
                    if prob == 0.0:
                        logger.debug(
                            "no transition stats for domains %s -> %s",
                            prev_domains,
                            cand_domains,
                        )
                        score *= 1.0
                    else:
                        score *= 1.0 + prob

                if score > best_score:
                    best_id = wid
                    best_score = score
                    best_vec = cand_vec
                    best_roi = cand_roi

            if best_id is None:
                break
            pipeline.append(best_id)
            available.remove(best_id)
            current = best_id
            prev_domains = self._workflow_domain(current, workflows)[0]
            if best_vec is not None:
                current_vec = best_vec
            current_roi = best_roi

        return pipeline

    # ------------------------------------------------------------------
    def plan_and_validate(
        self,
        target_embedding: Sequence[float],
        workflows: Mapping[str, Callable[[], Any]],
        *,
        top_k: int = 3,
        failure_threshold: int = 0,
        entropy_threshold: float = 2.0,
        runner: WorkflowSandboxRunner | None = None,
        runs: int = 3,
        max_workers: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Suggest and validate workflow chains.

        ``target_embedding`` is clustered via :class:`WorkflowChainSuggester`
        to obtain candidate sequences of workflow identifiers.  Each suggested
        chain undergoes a full sandbox execution via
        :class:`WorkflowSandboxRunner` where ROI gain, failure *rate* and
        entropy are aggregated across multiple runs.  Chains that exceed
        ``failure_threshold`` or ``entropy_threshold`` are discarded.

        Returns a list of dictionaries containing metrics for accepted chains.
        """

        try:
            from workflow_chain_suggester import WorkflowChainSuggester  # type: ignore

            suggester = WorkflowChainSuggester()
            chains = suggester.suggest_chains(target_embedding, top_k=top_k)
        except Exception:
            return []

        if runner is None:
            try:
                from sandbox_runner.workflow_sandbox_runner import (
                    WorkflowSandboxRunner as _Runner,
                )  # type: ignore

                runner = _Runner()
            except Exception:
                runner = None

        results: List[Dict[str, Any]] = []
        for chain in chains:
            record = self._validate_chain(
                chain,
                workflows,
                runner=runner,
                failure_threshold=failure_threshold,
                entropy_threshold=entropy_threshold,
                runs=runs,
                max_workers=max_workers,
            )
            if record:
                results.append(record)

        return results

    # ------------------------------------------------------------------
    def discover_and_persist(
        self,
        workflows: Mapping[str, Callable[[], Any]],
        *,
        runs: int = 3,
        max_workers: int | None = None,
        metrics_db: Any | None = None,
    ) -> List[Dict[str, Any]]:
        """Discover meta-workflows and persist successful chains."""

        target = self.encode("self_improvement", {"workflow": []})
        records = self.plan_and_validate(
            target,
            workflows,
            runs=runs,
            max_workers=max_workers,
        )

        successes: List[Dict[str, Any]] = []
        for record in records:
            chain = record.get("chain") or []
            roi_gain = float(record.get("roi_gain", 0.0))
            if not chain or roi_gain <= 0:
                continue
            chain_id = "->".join(chain)
            if self.stability_db is not None:
                try:
                    self.stability_db.mark_stable(chain_id, roi_gain)
                except Exception:
                    logger.exception("Failed to mark chain %s as stable", chain_id)
            if metrics_db is not None:
                try:
                    metrics_db.log_eval(f"meta:{chain_id}", "roi_gain", float(roi_gain))
                except Exception:
                    logger.warning(
                        "Failed to log ROI gain for chain %s", chain_id, exc_info=True
                    )
            successes.append(record)
        return successes

    # ------------------------------------------------------------------
    def _validate_chain(
        self,
        chain: Sequence[str],
        workflows: Mapping[str, Callable[[], Any]],
        *,
        runner: WorkflowSandboxRunner | None = None,
        failure_threshold: int = 0,
        entropy_threshold: float = 2.0,
        runs: int = 3,
        max_workers: int | None = None,
    ) -> Dict[str, Any] | None:
        """Validate a single chain and update ROI logs and cluster map.

        ``runs`` controls how many times the chain is executed.  Metrics are
        aggregated across runs using their mean and population variance.  When
        ``max_workers`` is greater than 1 the runs are executed concurrently
        using a thread pool for improved throughput.
        """

        funcs: List[Callable[[], Any]] = []
        for wid in chain:
            fn = workflows.get(wid)
            if not callable(fn):
                return None
            funcs.append(fn)

        if not funcs:
            return None

        if runner is None:
            try:
                from sandbox_runner.workflow_sandbox_runner import (
                    WorkflowSandboxRunner as _Runner,
                )  # type: ignore

                runner = _Runner()
            except Exception:
                return None

        comparator = WorkflowSynergyComparator
        try:  # pragma: no cover - allow dynamic replacement via sys.modules
            import sys
            mod = sys.modules.get("workflow_synergy_comparator")
            if mod is not None:
                comparator = getattr(mod, "WorkflowSynergyComparator", comparator)
        except Exception:
            pass
        if comparator is None:
            try:  # pragma: no cover - allow dynamic import for testing
                from workflow_synergy_comparator import (
                    WorkflowSynergyComparator as _WSC,  # type: ignore
                )
                comparator = _WSC
            except Exception:
                comparator = None
        if comparator is None:
            logger.warning(
                "WorkflowSynergyComparator unavailable; entropy metrics will be zero"
            )

        roi_gains: List[float] = []
        failure_rates: List[float] = []
        entropies: List[float] = []
        runtimes: List[float] = []
        success_rates: List[float] = []
        per_run_steps: List[List[Dict[str, Any]]] = []

        run_count = max(1, runs)

        def _single_run() -> tuple[float, float, float, float, float, List[Dict[str, Any]]]:
            metrics = runner.run(funcs)
            failure_count = max(
                metrics.crash_count,
                sum(1 for m in metrics.modules if not m.success),
            )
            module_count = len(metrics.modules)
            failure_rate = failure_count / module_count if module_count else 0.0

            module_names = [m.name for m in metrics.modules]
            spec = {"steps": [{"module": n} for n in module_names]}
            if comparator is not None:
                try:
                    entropy = comparator._entropy(spec)
                except Exception:
                    entropy = 0.0
            else:
                entropy = 0.0

            step_entropies: List[float] = []
            if comparator is not None:
                for i in range(1, len(chain) + 1):
                    try:
                        sub_spec = {"steps": [{"module": m} for m in chain[:i]]}
                        step_entropies.append(comparator._entropy(sub_spec))
                    except Exception:
                        step_entropies.append(0.0)
            else:
                step_entropies = [0.0] * len(chain)

            roi_gain = sum(
                float(m.result)
                for m in metrics.modules
                if isinstance(m.result, (int, float))
            )

            top_metrics = metrics.modules[: len(chain)]
            step_metrics = [
                {
                    "module": m.name,
                    "roi": float(m.result)
                    if isinstance(m.result, (int, float))
                    else 0.0,
                    "failures": 0 if m.success else 1,
                    "entropy": step_entropies[i]
                    if i < len(step_entropies)
                    else 0.0,
                }
                for i, m in enumerate(top_metrics)
            ]
            runtime = sum(m.duration for m in metrics.modules)
            success_rate = 1.0 - failure_rate
            return (
                roi_gain,
                float(failure_rate),
                float(entropy),
                runtime,
                success_rate,
                step_metrics,
            )

        results: List[
            tuple[float, float, float, float, float, List[Dict[str, Any]]]
        ] = []
        if max_workers and max_workers > 1 and run_count > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_single_run) for _ in range(run_count)]
                for fut in futures:
                    results.append(fut.result())
        else:
            for _ in range(run_count):
                results.append(_single_run())

        for (
            roi_gain_val,
            failure_val,
            entropy_val,
            runtime_val,
            success_val,
            steps_val,
        ) in results:
            roi_gains.append(roi_gain_val)
            failure_rates.append(failure_val)
            entropies.append(entropy_val)
            runtimes.append(runtime_val)
            success_rates.append(success_val)
            per_run_steps.append(steps_val)

        roi_gain = fmean(roi_gains)
        failure_rate = fmean(failure_rates)
        entropy = fmean(entropies)
        roi_var = pvariance(roi_gains) if len(roi_gains) > 1 else 0.0
        failure_var = pvariance(failure_rates) if len(failure_rates) > 1 else 0.0
        entropy_var = pvariance(entropies) if len(entropies) > 1 else 0.0
        runtime = fmean(runtimes)
        success_rate = fmean(success_rates)

        # Aggregate per-step metrics across runs
        step_metrics: List[Dict[str, Any]] = []
        if per_run_steps:
            for i in range(len(per_run_steps[0])):
                roi_vals = [s[i].get("roi", 0.0) for s in per_run_steps]
                fail_vals = [s[i].get("failures", 0.0) for s in per_run_steps]
                ent_vals = [s[i].get("entropy", 0.0) for s in per_run_steps]
                step_metrics.append(
                    {
                        "module": per_run_steps[0][i].get("module", ""),
                        "roi": fmean(roi_vals),
                        "failures": fmean(fail_vals),
                        "entropy": fmean(ent_vals),
                    }
                )

        # Penalize improbable domain transitions
        domain_lists = [self._workflow_domain(wid)[0] for wid in chain]
        trans_probs = self.transition_probabilities()
        penalty = 0.0
        for prev, curr in zip(domain_lists, domain_lists[1:]):
            if not prev or not curr:
                continue
            prob = 0.0
            for a in prev:
                for b in curr:
                    prob = max(prob, trans_probs.get((a, b), 0.0))
            penalty += 1.0 - prob
        if penalty:
            roi_gain = max(0.0, roi_gain - penalty)

        chain_id = "->".join(chain)
        prev_roi = (
            self.stability_db.data.get(chain_id, {}).get("roi")
            if self.stability_db is not None
            else 0.0
        )
        roi_delta = roi_gain - float(prev_roi or 0.0)
        if self.roi_tracker is not None:
            try:
                self.roi_tracker.record_scenario_delta(
                    chain_id,
                    roi_delta,
                    metrics_delta={
                        "failures": float(failure_rate),
                        "entropy": float(entropy),
                    },
                )
            except Exception:
                logger.exception(
                    "ROI tracker failed to record scenario delta for %s", chain_id
                )
        if self.stability_db is not None:
            try:
                self.stability_db.record_metrics(
                    chain_id,
                    roi_gain,
                    failure_rate,
                    entropy,
                    roi_delta=roi_delta,
                    roi_var=roi_var,
                    failures_var=failure_var,
                    entropy_var=entropy_var,
                )
            except Exception:
                logger.exception("Failed to record metrics for chain %s", chain_id)

        if failure_rate > failure_threshold or entropy > entropy_threshold:
            return None

        if self.roi_db is not None:
            try:
                self.roi_db.log_result(
                    workflow_id="->".join(chain),
                    run_id="0",
                    runtime=runtime,
                    success_rate=success_rate,
                    roi_gain=roi_gain,
                    workflow_synergy_score=max(0.0, 1.0 - entropy),
                    bottleneck_index=0.0,
                    patchability_score=0.0,
                    module_deltas={
                        **{
                            m["module"]: {
                                "roi_delta": m.get("roi", 0.0),
                                "success_rate": 1.0 - m.get("failures", 0.0),
                            }
                            for m in step_metrics
                        },
                        "__aggregate__": {
                            "roi_gain_var": roi_var,
                            "failures_mean": failure_rate,
                            "failures_var": failure_var,
                            "entropy_mean": entropy,
                            "entropy_var": entropy_var,
                        },
                    },
                )
            except Exception:
                logger.exception("Failed to log ROI result for chain %s", chain_id)

        self._update_cluster_map(
            chain,
            roi_gain,
            failures=failure_rate,
            entropy=entropy,
            step_metrics=step_metrics,
        )

        embeddings = _load_embeddings()
        vecs = [embeddings.get(wid) for wid in chain if embeddings.get(wid)]
        if vecs:
            dims = zip(*vecs)
            chain_vec = [sum(d) / len(vecs) for d in dims]
            try:
                persist_embedding(
                    "workflow_chain",
                    chain_id,
                    chain_vec,
                    metadata={"roi": roi_gain, "entropy": entropy},
                    path=_CHAIN_EMBEDDINGS_PATH,
                )
            except TypeError:  # pragma: no cover - compatibility shim
                persist_embedding("workflow_chain", chain_id, chain_vec, path=_CHAIN_EMBEDDINGS_PATH)

        return {
            "chain": list(chain),
            "roi_gain": roi_gain,
            "roi_var": roi_var,
            "failures": failure_rate,
            "failures_var": failure_var,
            "entropy": entropy,
            "entropy_var": entropy_var,
            "step_metrics": step_metrics,
        }

    # ------------------------------------------------------------------
    def _log_evolution(
        self,
        chain: Sequence[str],
        event: str,
        **extra: Any,
    ) -> None:
        """Persist ``event`` about ``chain`` to ``synergy_history_db``.

        The helper records evolutionary actions such as mutation, halting,
        splitting or remerging.  Entries are best-effort and ignored when the
        history database is unavailable.
        """

        try:  # pragma: no cover - persistence best effort
            rec = getattr(shd, "record", None)
            if not rec:
                return
            entry: Dict[str, Any] = {"chain": "->".join(chain), "event": event}
            entry.update(extra)
            rec(entry)
        except Exception:
            logger.warning("Failed to log evolution event", exc_info=True)

    # ------------------------------------------------------------------
    def mutate_chains(
        self,
        chains: Sequence[Sequence[str]],
        workflows: Mapping[str, Callable[[], Any]],
        *,
        runner: WorkflowSandboxRunner | None = None,
        failure_threshold: int = 0,
        entropy_threshold: float = 2.0,
        runs: int = 3,
        max_workers: int | None = None,
        offspring: int | None = None,
        epsilon: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Mutate ``chains`` using genetic operators and re‑validate offspring.

        Parent chains are selected according to their ``cluster_map['score']``
        weights.  Offspring are generated via single‑point crossover followed by
        a weighted mutation whose rate is inversely proportional to the parents'
        average score.  Each unique offspring is validated and any successful
        records are returned.  ``epsilon`` specifies the minimum ROI improvement
        over the best parent required to continue; when the gain falls below
        this threshold the generation halts and the chain is split for further
        exploration.
        """

        if not chains:
            return []

        wf_ids = list(workflows.keys())
        population = [list(c) for c in chains]
        weights: List[float] = []
        for c in population:
            info = self.cluster_map.get(tuple(c), {})
            weights.append(max(float(info.get("score", 0.0)), 0.0) + 1.0)

        off_count = offspring or max(2, len(population))
        results: List[Dict[str, Any]] = []
        seen: set[tuple[str, ...]] = set()

        for _ in range(off_count):
            parent_a, parent_b = random.choices(population, weights=weights, k=2)
            cut_a = random.randint(1, len(parent_a)) if parent_a else 0
            cut_b = random.randint(1, len(parent_b)) if parent_b else 0
            child = parent_a[:cut_a] + parent_b[cut_b:]

            score_a = weights[population.index(parent_a)] - 1.0
            score_b = weights[population.index(parent_b)] - 1.0
            avg_score = max((score_a + score_b) / 2.0, 0.0)
            mutation_rate = 1.0 / (1.0 + avg_score)

            if random.random() < mutation_rate and wf_ids:
                if child:
                    idx = random.randrange(len(child))
                    replacements = [w for w in wf_ids if w not in child]
                    if replacements:
                        child[idx] = random.choice(replacements)
                else:
                    child = [random.choice(wf_ids)]

            tup = tuple(child)
            if tup in seen:
                continue
            seen.add(tup)

            record = self._validate_chain(
                child,
                workflows,
                runner=runner,
                failure_threshold=failure_threshold,
                entropy_threshold=entropy_threshold,
                runs=runs,
                max_workers=max_workers,
            )
            if not record:
                continue

            results.append(record)
            parents = (parent_a, parent_b)
            parent_rois: List[float] = []
            for p in parents:
                info = self.cluster_map.get(tuple(p), {})
                hist = info.get("roi_history") or []
                if hist:
                    parent_rois.append(float(hist[-1]))
            base_roi = max(parent_rois) if parent_rois else 0.0
            improvement = record.get("roi_gain", 0.0) - base_roi
            if improvement < epsilon:
                self._log_evolution(record.get("chain", []), "halt", improvement=improvement)
                splits = self.split_pipeline(
                    record.get("chain", []),
                    workflows,
                    roi_improvement_threshold=epsilon,
                    runner=runner,
                    failure_threshold=failure_threshold,
                    entropy_threshold=entropy_threshold,
                    runs=runs,
                    max_workers=max_workers,
                )
                results.extend(splits)
                break

            self._log_evolution(record.get("chain", []), "mutate", improvement=improvement)

        return results

    # ------------------------------------------------------------------
    def refine_chains(
        self,
        records: Sequence[Dict[str, Any]],
        workflows: Mapping[str, Callable[[], Any]],
        *,
        roi_threshold: float = 0.0,
        runner: WorkflowSandboxRunner | None = None,
        failure_threshold: int = 0,
        entropy_threshold: float = 2.0,
        runs: int = 3,
        max_workers: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Split underperforming subchains and merge high-ROI chains."""

        low = [
            r["chain"]
            for r in records
            if (
                r.get("roi_gain", 0.0) <= roi_threshold
                or r.get("failures", 0) > failure_threshold
                or r.get("entropy", 0.0) > entropy_threshold
            )
        ]
        high = [
            r["chain"]
            for r in records
            if (
                r.get("roi_gain", 0.0) > roi_threshold
                and r.get("failures", 0) <= failure_threshold
                and r.get("entropy", 0.0) <= entropy_threshold
            )
        ]

        candidates: List[List[str]] = []
        for chain in low:
            if len(chain) > 1:
                mid = len(chain) // 2
                candidates.append(chain[:mid])
                candidates.append(chain[mid:])
        for i in range(len(high)):
            for j in range(i + 1, len(high)):
                merged = high[i] + [w for w in high[j] if w not in high[i]]
                candidates.append(merged)

        results: List[Dict[str, Any]] = []
        for c in candidates:
            record = self._validate_chain(
                c,
                workflows,
                runner=runner,
                failure_threshold=failure_threshold,
                entropy_threshold=entropy_threshold,
                runs=runs,
                max_workers=max_workers,
            )
            if record:
                results.append(record)
        return results

    # ------------------------------------------------------------------
    def mutate_pipeline(
        self,
        pipeline: Sequence[str],
        workflows: Mapping[str, Callable[[], Any]],
        *,
        roi_improvement_threshold: float = 0.0,
        entropy_stability_threshold: float = 1.0,
        runner: WorkflowSandboxRunner | None = None,
        failure_threshold: int = 0,
        entropy_threshold: float = 2.0,
        runs: int = 3,
        max_workers: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Mutate ``pipeline`` when ROI stagnates or entropy drifts.

        The existing pipeline is validated and its recorded ROI delta and
        entropy are inspected.  When the improvement falls below
        ``roi_improvement_threshold`` or the entropy exceeds
        ``entropy_stability_threshold`` a set of mutated variants is generated
        using :meth:`mutate_chains`.  Otherwise the original pipeline metrics
        are returned.
        """

        record = self._validate_chain(
            pipeline,
            workflows,
            runner=runner,
            failure_threshold=failure_threshold,
            entropy_threshold=entropy_threshold,
            runs=runs,
            max_workers=max_workers,
        )
        if not record:
            return []

        chain_id = "->".join(pipeline)
        info = self.cluster_map.get(tuple(pipeline))
        roi_delta = (
            float(info.get("delta_roi", 0.0)) if info else record.get("roi_gain", 0.0)
        )
        failure_delta = (
            float(info.get("delta_failures", 0.0)) if info else record.get("failures", 0)
        )
        entropy_delta = (
            float(info.get("delta_entropy", record.get("entropy", 0.0)))
            if info
            else record.get("entropy", 0.0)
        )

        if (
            roi_delta >= roi_improvement_threshold
            and failure_delta <= 0
            and abs(entropy_delta) <= entropy_stability_threshold
        ):
            return [record]

        results = self.mutate_chains(
            [pipeline],
            workflows,
            runner=runner,
            failure_threshold=failure_threshold,
            entropy_threshold=entropy_threshold,
            runs=runs,
            max_workers=max_workers,
        )

        try:  # pragma: no cover - best effort logging
            from workflow_lineage import log_lineage

            for r in results:
                log_lineage(
                    chain_id,
                    "->".join(r.get("chain", [])),
                    "mutate_pipeline",
                    roi=r.get("roi_gain"),
                )
        except Exception:
            logger.warning(
                "Failed to log lineage for mutated chain %s", chain_id, exc_info=True
            )

        return results

    # ------------------------------------------------------------------
    def manage_pipeline(
        self,
        pipeline: Sequence[str],
        workflows: Mapping[str, Callable[[], Any]],
        *,
        roi_improvement_threshold: float = 0.0,
        entropy_stability_threshold: float = 1.0,
        runner: WorkflowSandboxRunner | None = None,
        failure_threshold: int = 0,
        entropy_threshold: float = 2.0,
        runs: int = 3,
        max_workers: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Monitor ``pipeline`` and split or terminate on stagnating ROI.

        The method validates ``pipeline`` and inspects the recorded ROI delta
        for the chain.  When the improvement drops below
        ``roi_improvement_threshold`` or the entropy exceeds
        ``entropy_stability_threshold`` the pipeline is split into two segments
        using :meth:`split_pipeline`.  If the pipeline consists of a single
        step it is simply terminated (an empty list is returned).  Otherwise the
        original metrics are returned unchanged.
        """

        record = self._validate_chain(
            pipeline,
            workflows,
            runner=runner,
            failure_threshold=failure_threshold,
            entropy_threshold=entropy_threshold,
            runs=runs,
            max_workers=max_workers,
        )
        if not record:
            return []

        info = self.cluster_map.get(tuple(pipeline))
        roi_delta = (
            float(info.get("delta_roi", 0.0)) if info else record.get("roi_gain", 0.0)
        )
        failure_delta = (
            float(info.get("delta_failures", 0.0)) if info else record.get("failures", 0)
        )
        entropy_delta = (
            float(info.get("delta_entropy", record.get("entropy", 0.0)))
            if info
            else record.get("entropy", 0.0)
        )

        if (
            roi_delta >= roi_improvement_threshold
            and failure_delta <= 0
            and abs(entropy_delta) <= entropy_stability_threshold
        ):
            return [record]

        if len(pipeline) > 1:
            return self.split_pipeline(
                pipeline,
                workflows,
                roi_improvement_threshold=roi_improvement_threshold,
                entropy_stability_threshold=entropy_stability_threshold,
                runner=runner,
                failure_threshold=failure_threshold,
                entropy_threshold=entropy_threshold,
                runs=runs,
                max_workers=max_workers,
            )
        return []

    # ------------------------------------------------------------------
    def split_pipeline(
        self,
        pipeline: Sequence[str],
        workflows: Mapping[str, Callable[[], Any]],
        *,
        roi_improvement_threshold: float = 0.0,
        entropy_stability_threshold: float = 1.0,
        runner: WorkflowSandboxRunner | None = None,
        failure_threshold: int = 0,
        entropy_threshold: float = 2.0,
        runs: int = 3,
        max_workers: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Split ``pipeline`` into sub-pipelines when improvement stalls."""

        if len(pipeline) <= 1:
            return []

        chain_id = "->".join(pipeline)
        info = self.cluster_map.get(tuple(pipeline))
        roi_delta = float(info.get("delta_roi", 0.0)) if info else 0.0
        failure_delta = float(info.get("delta_failures", 0.0)) if info else 0.0
        entropy_delta = float(info.get("delta_entropy", 0.0)) if info else 0.0

        record = self._validate_chain(
            pipeline,
            workflows,
            runner=runner,
            failure_threshold=failure_threshold,
            entropy_threshold=entropy_threshold,
            runs=runs,
            max_workers=max_workers,
        )
        if not record:
            return []

        if info is None:
            info = self.cluster_map.get(tuple(pipeline))
            roi_delta = (
                float(info.get("delta_roi", record.get("roi_gain", 0.0)))
                if info
                else record.get("roi_gain", 0.0)
            )
            failure_delta = (
                float(info.get("delta_failures", record.get("failures", 0)))
                if info
                else record.get("failures", 0)
            )
            entropy_delta = (
                float(info.get("delta_entropy", record.get("entropy", 0.0)))
                if info
                else record.get("entropy", 0.0)
            )

        if (
            roi_delta >= roi_improvement_threshold
            and failure_delta <= 0
            and abs(entropy_delta) <= entropy_stability_threshold
        ):
            return [record]

        mid = len(pipeline) // 2
        segments = [list(pipeline[:mid]), list(pipeline[mid:])]
        results: List[Dict[str, Any]] = []
        for seg in segments:
            rec = self._validate_chain(
                seg,
                workflows,
                runner=runner,
                failure_threshold=failure_threshold,
                entropy_threshold=entropy_threshold,
                runs=runs,
                max_workers=max_workers,
            )
            if rec:
                results.append(rec)
                self._log_evolution(rec.get("chain", []), "split", parent=chain_id)
                try:  # pragma: no cover - best effort logging
                    from workflow_lineage import log_lineage

                    log_lineage(chain_id, "->".join(seg), "split_pipeline", roi=rec.get("roi_gain"))
                except Exception:
                    logger.warning(
                        "Failed to log lineage for split segment %s of %s",
                        "->".join(seg),
                        chain_id,
                        exc_info=True,
                    )
        return results

    # ------------------------------------------------------------------
    def remerge_pipelines(
        self,
        pipelines: Sequence[Sequence[str]],
        workflows: Mapping[str, Callable[[], Any]],
        *,
        roi_improvement_threshold: float = 0.0,
        entropy_stability_threshold: float = 1.0,
        similarity_threshold: float = 0.0,
        runner: WorkflowSandboxRunner | None = None,
        failure_threshold: int = 0,
        entropy_threshold: float = 2.0,
        runs: int = 3,
        max_workers: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Merge pipelines that show stable entropy and ROI improvements.

        Pipelines are considered for merging when the cosine similarity of their
        embeddings, adjusted by any ``cluster_map`` pair scores, exceeds
        ``similarity_threshold``.  Only complementary pairs that also satisfy
        the ROI and entropy constraints are revalidated and returned.
        """

        results: List[Dict[str, Any]] = []
        embeddings: Dict[int, List[float]] = {}
        for idx, pipe in enumerate(pipelines):
            vec = self.cluster_map.get(tuple(pipe), {}).get("embedding")
            if vec is None:
                try:
                    vec = self.encode_chain(pipe)
                except Exception:
                    vec = None
            if vec is not None:
                embeddings[idx] = vec

        for i in range(len(pipelines)):
            for j in range(i + 1, len(pipelines)):
                vec_i = embeddings.get(i)
                vec_j = embeddings.get(j)
                sim = 0.0
                if vec_i is not None and vec_j is not None:
                    sim = cosine_similarity(vec_i, vec_j)
                    if pipelines[i] and pipelines[j]:
                        cm_score = float(
                            self.cluster_map
                            .get((pipelines[i][-1], pipelines[j][0]), {})
                            .get("score", 0.0)
                        )
                        sim *= 1.0 + cm_score
                if sim < similarity_threshold:
                    continue
                merged = list(pipelines[i]) + [w for w in pipelines[j] if w not in pipelines[i]]
                rec = self._validate_chain(
                    merged,
                    workflows,
                    runner=runner,
                    failure_threshold=failure_threshold,
                    entropy_threshold=entropy_threshold,
                    runs=runs,
                    max_workers=max_workers,
                )
                if not rec:
                    continue
                chain_id = "->".join(merged)
                roi_delta = 0.0
                entropy_val = rec.get("entropy", 0.0)
                if self.stability_db is not None:
                    entry = self.stability_db.data.get(chain_id, {})
                    roi_delta = float(entry.get("roi_delta", 0.0))
                    entropy_val = float(entry.get("entropy", entropy_val))
                else:
                    roi_delta = rec.get("roi_gain", 0.0)

                if (
                    roi_delta >= roi_improvement_threshold
                    and abs(entropy_val) <= entropy_stability_threshold
                ):
                    results.append(rec)
                    self._log_evolution(
                        rec.get("chain", []),
                        "remerge",
                        parents="|".join(
                            ["->".join(pipelines[i]), "->".join(pipelines[j])]
                        ),
                        similarity=float(sim),
                    )
                    try:  # pragma: no cover - best effort logging
                        from workflow_lineage import log_lineage

                        log_lineage(
                            None,
                            chain_id,
                            "remerge_pipelines",
                            roi=rec.get("roi_gain"),
                        )
                    except Exception:
                        logger.warning(
                            "Failed to log lineage for remerged pipeline %s",
                            chain_id,
                            exc_info=True,
                        )
        return results

    # ------------------------------------------------------------------
    def schedule(
        self,
        workflows: Mapping[str, Callable[[], Any]],
        *,
        runner: "WorkflowSandboxRunner" | None = None,
        failure_threshold: int = 0,
        entropy_threshold: float = 2.0,
        roi_delta_threshold: float = 0.01,
        entropy_delta_threshold: float = 0.01,
        runs: int = 3,
        max_iterations: int = 10,
        max_workers: int | None = None,
        metrics_db: Any | None = None,
    ) -> List[Dict[str, Any]]:
        """Orchestrate discovery, management and mutation cycles.

        The scheduler first invokes :meth:`discover_and_persist` to obtain
        promising chains.  Each chain is then passed through
        :meth:`manage_pipeline` and the resulting pipelines are fed back into
        :meth:`mutate_chains`.  Iteration continues until both ROI and entropy
        deltas fall below the provided thresholds.  Lineage and metrics for
        every produced record are persisted best effort.
        """

        all_records: List[Dict[str, Any]] = []
        discovered = self.discover_and_persist(
            workflows, runs=runs, max_workers=max_workers, metrics_db=metrics_db
        )
        for rec in discovered:
            all_records.append(rec)
            chain_id = "->".join(rec.get("chain", []))
            try:  # pragma: no cover - logging best effort
                from workflow_lineage import log_lineage

                log_lineage(
                    None, chain_id, "discover_and_persist", roi=rec.get("roi_gain")
                )
            except Exception:
                logger.warning(
                    "Failed to log lineage for discovered chain %s",
                    chain_id,
                    exc_info=True,
                )
            self._reinforce(rec)

        active = [r.get("chain", []) for r in discovered if r.get("chain")]
        iteration = 0

        while active and iteration < max_iterations:
            managed: List[Sequence[str]] = []
            for chain in active:
                recs = self.manage_pipeline(
                    chain,
                    workflows,
                    runner=runner,
                    failure_threshold=failure_threshold,
                    entropy_threshold=entropy_threshold,
                    runs=runs,
                    max_workers=max_workers,
                )
                for rec in recs:
                    all_records.append(rec)
                    managed.append(rec.get("chain", []))
                    parent_id = "->".join(chain)
                    child_id = "->".join(rec.get("chain", []))
                    try:  # pragma: no cover - logging best effort
                        from workflow_lineage import log_lineage

                        log_lineage(
                            parent_id, child_id, "manage_pipeline", roi=rec.get("roi_gain")
                        )
                    except Exception:
                        logger.warning(
                            "Failed to log lineage for managed pipeline %s -> %s",
                            parent_id,
                            child_id,
                            exc_info=True,
                        )
                    self._reinforce(rec)

            if not managed:
                break

            mutated = self.mutate_chains(
                managed,
                workflows,
                runner=runner,
                failure_threshold=failure_threshold,
                entropy_threshold=entropy_threshold,
                runs=runs,
                max_workers=max_workers,
            )
            active = []
            for rec in mutated:
                all_records.append(rec)
                chain = rec.get("chain", [])
                active.append(chain)
                child_id = "->".join(chain)
                try:  # pragma: no cover - logging best effort
                    from workflow_lineage import log_lineage

                    log_lineage(None, child_id, "mutate_chains", roi=rec.get("roi_gain"))
                except Exception:
                    logger.warning(
                        "Failed to log lineage for mutated chain %s", child_id, exc_info=True
                    )
                self._reinforce(rec)

            if not active:
                break

            if all(
                abs(float(self.cluster_map.get(tuple(c), {}).get("delta_roi", 0.0)))
                < roi_delta_threshold
                and abs(
                    float(self.cluster_map.get(tuple(c), {}).get("delta_entropy", 0.0))
                )
                < entropy_delta_threshold
                for c in active
            ):
                break

            iteration += 1

        return all_records

    # ------------------------------------------------------------------
    def iterate_pipelines(
        self,
        workflows: Mapping[str, Callable[[], Any]],
        *,
        runner: "WorkflowSandboxRunner" | None = None,
        failure_threshold: int = 0,
        entropy_threshold: float = 2.0,
        roi_improvement_threshold: float = 0.0,
        entropy_stability_threshold: float = 1.0,
        runs: int = 3,
        max_workers: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Evolve converged pipelines based on metric deltas.

        ``cluster_map`` entries flagged as converged are evaluated by
        combining ``delta_roi``, ``delta_failures`` and ``delta_entropy`` into a
        simple fitness score.  Pipelines with negative fitness undergo
        :meth:`mutate_chains` while those with unstable entropy are processed by
        :meth:`split_pipeline`.  Remaining candidates are considered for
        re‑merging.  Winning variants are persisted for reinforcement and
        returned.
        """

        results: List[Dict[str, Any]] = []
        remerge: List[Sequence[str]] = []

        for chain, info in list(self.cluster_map.items()):
            if not info.get("converged"):
                continue

            roi_delta = float(info.get("delta_roi", 0.0))
            fail_delta = float(info.get("delta_failures", 0.0))
            ent_delta = float(info.get("delta_entropy", 0.0))
            fitness = roi_delta - fail_delta - abs(ent_delta)

            if fitness <= roi_improvement_threshold:
                recs = self.mutate_chains(
                    [list(chain)],
                    workflows,
                    runner=runner,
                    failure_threshold=failure_threshold,
                    entropy_threshold=entropy_threshold,
                    runs=runs,
                    max_workers=max_workers,
                )
                if recs:
                    best = max(recs, key=lambda r: r.get("roi_gain", 0.0))
                    results.append(best)
                    self._reinforce(best)
            elif abs(ent_delta) > entropy_stability_threshold:
                recs = self.split_pipeline(
                    list(chain),
                    workflows,
                    roi_improvement_threshold=roi_improvement_threshold,
                    entropy_stability_threshold=entropy_stability_threshold,
                    runner=runner,
                    failure_threshold=failure_threshold,
                    entropy_threshold=entropy_threshold,
                    runs=runs,
                    max_workers=max_workers,
                )
                if recs:
                    best = max(recs, key=lambda r: r.get("roi_gain", 0.0))
                    results.append(best)
                    self._reinforce(best)
            else:
                remerge.append(list(chain))

        if len(remerge) > 1:
            recs = self.remerge_pipelines(
                remerge,
                workflows,
                roi_improvement_threshold=roi_improvement_threshold,
                entropy_stability_threshold=entropy_stability_threshold,
                runner=runner,
                failure_threshold=failure_threshold,
                entropy_threshold=entropy_threshold,
                runs=runs,
                max_workers=max_workers,
            )
            for rec in recs:
                results.append(rec)
                self._reinforce(rec)

        return results

    # ------------------------------------------------------------------
    def _reinforce(self, record: Mapping[str, Any]) -> None:
        """Persist ``record`` metrics for reinforcement."""

        chain = record.get("chain", [])
        roi_gain = float(record.get("roi_gain", 0.0))
        failures = float(record.get("failures", 0.0))
        entropy = float(record.get("entropy", 0.0))
        chain_id = "->".join(chain)

        if self.roi_db is not None:
            try:
                success_rate = (
                    (len(chain) - failures) / len(chain) if chain else 0.0
                )
                self.roi_db.log_result(
                    workflow_id=chain_id,
                    run_id="reinforce",
                    runtime=0.0,
                    success_rate=success_rate,
                    roi_gain=roi_gain,
                    workflow_synergy_score=max(0.0, 1.0 - entropy),
                    bottleneck_index=0.0,
                    patchability_score=0.0,
                    module_deltas={},
                )
            except Exception:
                logger.exception("Failed to log ROI result for chain %s", chain_id)

        if self.stability_db is not None:
            try:
                self.stability_db.record_metrics(
                    chain_id, roi_gain, failures, entropy, roi_delta=roi_gain
                )
            except Exception:
                logger.exception("Failed to record metrics for chain %s", chain_id)

        self._update_cluster_map(
            chain,
            roi_gain,
            failures,
            entropy,
            step_metrics=record.get("step_metrics"),
        )

    # ------------------------------------------------------------------
    def merge_high_performing_variants(
        self,
        records: Sequence[Dict[str, Any]],
        workflows: Mapping[str, Callable[[], Any]],
        *,
        roi_threshold: float = 0.0,
        cluster_epsilon: float = 0.5,
        cluster_min_samples: int = 2,
        runner: WorkflowSandboxRunner | None = None,
        failure_threshold: int = 0,
        entropy_threshold: float = 2.0,
        runs: int = 3,
        context_builder: ContextBuilder | None = None,
        retriever: Retriever | None = None,
        max_workers: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Merge high-ROI variants using clustering results.

        ``records`` should contain ``chain`` and ``roi_gain`` entries.  Chains
        exceeding ``roi_threshold`` are clustered via :meth:`cluster_workflows`
        (using the provided ``retriever``) and each cluster is merged into a
        single pipeline containing the union of steps.  Clustering is controlled
        by ``cluster_epsilon`` and ``cluster_min_samples`` and the merged
        pipelines are validated before being returned.
        """

        high = [r for r in records if r.get("roi_gain", 0.0) > roi_threshold]
        if not high:
            return []

        specs: Dict[str, Dict[str, Any]] = {}
        chain_lookup: Dict[str, Sequence[str]] = {}
        for idx, rec in enumerate(high):
            cid = f"c{idx}"
            chain_lookup[cid] = rec["chain"]
            specs[cid] = {"steps": [{"module": w} for w in rec["chain"]]}

        context_builder = context_builder or self.context_builder
        if retriever is None and Retriever is not None:
            try:
                retriever = Retriever(context_builder=context_builder)
            except Exception:
                retriever = None
        if retriever is None:
            raise ValueError("merge_high_performing_variants requires a Retriever")
        clusters = self.cluster_workflows(
            specs,
            context_builder=context_builder,
            retriever=retriever,
            epsilon=cluster_epsilon,
            min_samples=cluster_min_samples,
        )

        results: List[Dict[str, Any]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            merged: List[str] = []
            for cid in cluster:
                for wid in chain_lookup[cid]:
                    if wid not in merged:
                        merged.append(wid)
            rec = self._validate_chain(
                merged,
                workflows,
                runner=runner,
                failure_threshold=failure_threshold,
                entropy_threshold=entropy_threshold,
                runs=runs,
                max_workers=max_workers,
            )
            if rec:
                results.append(rec)
        return results

    # ------------------------------------------------------------------
    def _save_cluster_map(self) -> None:
        """Persist ``cluster_map`` to ``sandbox_data/meta_clusters.json``."""

        path = resolve_path("sandbox_data") / "meta_clusters.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            now = time.time()
            to_prune: List[tuple[str, ...]] = []
            for key, info in list(self.cluster_map.items()):
                if key == ("__domain_transitions__",):
                    # Apply decay to transition deltas
                    matrix: Dict[tuple[str, str], Dict[str, Any]] = info
                    for entry in matrix.values():
                        entry["delta_roi"] = float(entry.get("delta_roi", 0.0)) * _DECAY_FACTOR
                    continue

                last = float(info.get("ts", now))
                age = max(0.0, now - last)
                decay = _DECAY_FACTOR ** (age / 3600.0)
                info["ts"] = now

                delta_roi = float(info.get("delta_roi", 0.0))
                delta_fail = float(info.get("delta_failures", 0.0))
                delta_ent = float(info.get("delta_entropy", 0.0))

                if delta_roi < 0 or delta_fail > 0 or delta_ent > 0:
                    info["stagnant_runs"] = int(info.get("stagnant_runs", 0)) + 1
                    if info["stagnant_runs"] >= _PRUNE_RUNS:
                        to_prune.append(key)
                else:
                    info["stagnant_runs"] = 0

                info["score"] = float(info.get("score", 0.0)) * decay
                info["delta_roi"] = delta_roi * decay
                info["delta_failures"] = float(info.get("delta_failures", 0.0)) * decay
                info["delta_entropy"] = float(info.get("delta_entropy", 0.0)) * decay

            for key in to_prune:
                self.cluster_map.pop(key, None)

            data = {"|".join(k): v for k, v in self.cluster_map.items()}
            with path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except Exception:
            logger.warning(
                "Failed to save cluster map to %s", path, exc_info=True
            )

    # ------------------------------------------------------------------
    def _load_cluster_map(self) -> None:
        """Load ``cluster_map`` from persistence layers if present."""

        path = resolve_path("sandbox_data") / "meta_clusters.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self.cluster_map = {tuple(k.split("|")): v for k, v in data.items()}
                now = time.time()
                for key, info in list(self.cluster_map.items()):
                    if key == ("__domain_transitions__",):
                        matrix: Dict[tuple[str, str], Dict[str, Any]] = info
                        for entry in matrix.values():
                            entry["delta_roi"] = float(entry.get("delta_roi", 0.0)) * _DECAY_FACTOR
                        continue

                    last = float(info.get("ts", now))
                    age = max(0.0, now - last)
                    decay = _DECAY_FACTOR ** (age / 3600.0)
                    info["score"] = float(info.get("score", 0.0)) * decay
                    info["delta_roi"] = float(info.get("delta_roi", 0.0)) * decay
                    info["delta_failures"] = float(info.get("delta_failures", 0.0)) * decay
                    info["delta_entropy"] = float(info.get("delta_entropy", 0.0)) * decay
                    info["ts"] = now
                    info.setdefault("stagnant_runs", 0)
                    if int(info.get("stagnant_runs", 0)) >= _PRUNE_RUNS and (
                        float(info.get("delta_roi", 0.0)) < 0
                        or float(info.get("delta_failures", 0.0)) > 0
                        or float(info.get("delta_entropy", 0.0)) > 0
                    ):
                        del self.cluster_map[key]
            except Exception:
                self.cluster_map = {}

        # Populate from historic reinforcement stored in ``synergy_history.db``
        try:
            connect = getattr(shd, "connect", None)
            fetch_all = getattr(shd, "fetch_all", None)
            if connect and fetch_all:
                conn = connect()
                try:
                    for entry in fetch_all(conn):
                        chain = entry.get("chain")
                        if not isinstance(chain, str):
                            continue
                        self._update_cluster_map(
                            chain.split("|"),
                            float(entry.get("roi", 0.0)),
                            float(entry.get("failures", 0.0)),
                            float(entry.get("entropy", 0.0)),
                            record_history=False,
                            save=False,
                        )
                finally:
                    try:
                        conn.close()
                    except Exception:  # pragma: no cover - best effort
                        logger.warning(
                            "Failed to close synergy history connection", exc_info=True
                        )
                if self.cluster_map:
                    self._save_cluster_map()
        except Exception:  # pragma: no cover - persistence optional
            logger.warning(
                "Failed to load persistent reinforcement history", exc_info=True
            )

    # ------------------------------------------------------------------
    def _update_cluster_map(
        self,
        chain: Sequence[str],
        roi_gain: float,
        failures: float = 0.0,
        entropy: float = 0.0,
        *,
        step_metrics: Sequence[Mapping[str, Any]] | None = None,
        tol: float = 0.01,
        record_history: bool = True,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Update metric histories for ``chain`` and detect convergence.

        ``failures`` represents the average failure *rate* for the chain in the
        most recent run, allowing reinforcement logic to reason about stability
        irrespective of chain length.
        """

        key = tuple(chain)
        info = self.cluster_map.setdefault(
            key,
            {
                "roi_history": [],
                "failure_history": [],
                "entropy_history": [],
                "step_metrics": [],
                "step_deltas": [],
                "delta_roi": 0.0,
                "delta_failures": 0.0,
                "delta_entropy": 0.0,
                "converged": False,
                "score": 0.0,
                "stagnant_runs": 0,
            },
        )
        info["ts"] = time.time()
        roi_hist = info["roi_history"]
        if roi_hist:
            info["delta_roi"] = roi_gain - roi_hist[-1]
        roi_hist.append(roi_gain)

        fail_hist = info["failure_history"]
        info["delta_failures"] = failures - (fail_hist[-1] if fail_hist else failures)
        fail_hist.append(failures)

        ent_hist = info["entropy_history"]
        info["delta_entropy"] = entropy - (ent_hist[-1] if ent_hist else entropy)
        ent_hist.append(entropy)

        if step_metrics is not None:
            hist = info["step_metrics"]
            if hist:
                prev = hist[-1]
                deltas: List[Dict[str, float]] = []
                for i, step in enumerate(step_metrics):
                    prev_step = prev[i] if i < len(prev) else {}
                    deltas.append(
                        {
                            "roi": step.get("roi", 0.0) - float(prev_step.get("roi", 0.0)),
                            "failures": step.get("failures", 0.0)
                            - float(prev_step.get("failures", 0.0)),
                            "entropy": step.get("entropy", 0.0)
                            - float(prev_step.get("entropy", 0.0)),
                        }
                    )
                info["step_deltas"] = deltas
                if (
                    abs(info["delta_roi"]) < tol
                    and abs(info["delta_failures"]) < tol
                    and abs(info["delta_entropy"]) < tol
                    and all(
                        abs(d["roi"]) < tol
                        and abs(d["failures"]) < tol
                        and abs(d["entropy"]) < tol
                        for d in deltas
                    )
                ):
                    info["converged"] = True
                else:
                    info["converged"] = False
            hist.append(list(step_metrics))

            avg_roi = sum(s.get("roi", 0.0) for s in step_metrics) / len(step_metrics)
            avg_fail = sum(s.get("failures", 0.0) for s in step_metrics) / len(step_metrics)
            avg_ent = sum(s.get("entropy", 0.0) for s in step_metrics) / len(step_metrics)
            new_score = (roi_gain - failures - entropy) + (avg_roi - avg_fail - avg_ent)
        else:
            new_score = roi_gain - failures - entropy

        decay = 0.9
        prev_score = float(info.get("score", 0.0))
        info["score"] = prev_score * decay + new_score * (1.0 - decay)

        matrix = self.cluster_map.setdefault(("__domain_transitions__",), {})
        domain_lists = [self._workflow_domain(wid)[0] for wid in chain]
        step_rois = [m.get("roi", 0.0) for m in step_metrics] if step_metrics else []
        step_fails = [m.get("failures", 0.0) for m in step_metrics] if step_metrics else []
        step_ents = [m.get("entropy", 0.0) for m in step_metrics] if step_metrics else []
        for i, (src_list, dst_list) in enumerate(zip(domain_lists, domain_lists[1:])):
            if not src_list or not dst_list:
                continue
            for a in src_list:
                for b in dst_list:
                    entry = matrix.setdefault(
                        (a, b),
                        {
                            "count": 0,
                            "delta_roi": 0.0,
                            "delta_failures": 0.0,
                            "delta_entropy": 0.0,
                            "last_roi": roi_gain,
                            "last_failures": failures,
                            "last_entropy": entropy,
                        },
                    )
                    entry["count"] += 1
                    if step_rois:
                        roi_a = step_rois[i] if i < len(step_rois) else 0.0
                        roi_b = step_rois[i + 1] if i + 1 < len(step_rois) else roi_gain
                        delta_r = roi_b - roi_a
                        fail_a = step_fails[i] if i < len(step_fails) else failures
                        fail_b = step_fails[i + 1] if i + 1 < len(step_fails) else failures
                        delta_f = fail_b - fail_a
                        ent_a = step_ents[i] if i < len(step_ents) else entropy
                        ent_b = step_ents[i + 1] if i + 1 < len(step_ents) else entropy
                        delta_e = ent_b - ent_a
                    else:
                        prev_roi = float(entry.get("last_roi", roi_gain))
                        entry["last_roi"] = roi_gain
                        delta_r = roi_gain - prev_roi
                        prev_fail = float(entry.get("last_failures", failures))
                        entry["last_failures"] = failures
                        delta_f = failures - prev_fail
                        prev_ent = float(entry.get("last_entropy", entropy))
                        entry["last_entropy"] = entropy
                        delta_e = entropy - prev_ent
                    entry["delta_roi"] += (
                        delta_r - entry.get("delta_roi", 0.0)
                    ) / entry["count"]
                    entry["delta_failures"] += (
                        delta_f - entry.get("delta_failures", 0.0)
                    ) / entry["count"]
                    entry["delta_entropy"] += (
                        delta_e - entry.get("delta_entropy", 0.0)
                    ) / entry["count"]

        if save:
            self._save_cluster_map()

        if record_history:
            try:  # pragma: no cover - best effort persistence
                rec = getattr(shd, "record", None)
                if rec:
                    rec(
                        {
                            "chain": "|".join(chain),
                            "roi": float(roi_gain),
                            "failures": float(failures),
                            "entropy": float(entropy),
                            "score": float(info.get("score", 0.0)),
                        }
                    )
            except Exception:
                logger.warning(
                    "Failed to record reinforcement metrics", exc_info=True
                )

        return info

    # ------------------------------------------------------------------
    def transition_probabilities(self, *, smoothing: float = 0.0) -> Dict[tuple[int, int], float]:
        """Return normalized transition weights as probabilities.

        The transition matrix stored under the special
        ``("__domain_transitions__",)`` key tracks how often workflows move
        between domains along with the average ROI, failure and entropy deltas.
        This method converts those statistics into a probability distribution
        where pairs with higher counts, positive ROI and *lower* failure/
        entropy receive larger weight.  Transitions with non-positive ROI are
        assigned zero probability.

        Parameters
        ----------
        smoothing:
            Optional non-negative value added to every transition weight prior
            to normalisation.  This acts as a mild uniform prior so that unseen
            transitions retain a tiny probability instead of being completely
            discarded.
        """

        matrix = self.cluster_map.get(("__domain_transitions__",), {})
        weights: Dict[tuple[int, int], float] = {}
        for pair, stats in matrix.items():
            count = float(stats.get("count", 0.0))
            roi_delta = float(stats.get("delta_roi", stats.get("roi", 0.0)))
            fail_delta = float(
                stats.get("delta_failures", stats.get("failures", 0.0))
            )
            ent_delta = float(
                stats.get("delta_entropy", stats.get("entropy", 0.0))
            )
            penalty = max(0.0, fail_delta) + max(0.0, ent_delta)
            weight = max(0.0, count * (roi_delta - penalty))
            if weight > 0.0 or smoothing > 0.0:
                weights[pair] = weight + smoothing
            else:
                weights[pair] = 0.0
        total = sum(weights.values())
        if total <= 0:
            return {pair: 0.0 for pair in weights}
        return {pair: w / total for pair, w in weights.items()}

    # ------------------------------------------------------------------
    def _graph_features(self, workflow_id: str) -> List[float] | tuple[float, float]:
        depth = 0.0
        branching = 0.0
        g = getattr(self.graph, "graph", None)
        if g is None:
            return depth, branching
        try:
            if _HAS_NX and hasattr(g, "out_degree"):
                branching = float(g.out_degree(workflow_id)) if g.has_node(workflow_id) else 0.0
                if g.has_node(workflow_id):
                    ancestors = nx.ancestors(g, workflow_id)
                    if ancestors:
                        depth = max(
                            nx.shortest_path_length(g, anc, workflow_id) for anc in ancestors
                        )
            else:
                edges = g.get("edges") if isinstance(g, dict) else None
                if isinstance(edges, dict):
                    branching = float(len(edges.get(workflow_id, {})))
                    reverse: Dict[str, set[str]] = {}
                    for src, dests in edges.items():
                        for dst in dests.keys():
                            reverse.setdefault(dst, set()).add(src)
                    visited = {workflow_id}
                    queue = [(workflow_id, 0)]
                    max_depth = 0
                    while queue:
                        node, d = queue.pop(0)
                        max_depth = max(max_depth, d)
                        for parent in reverse.get(node, ()):  # pragma: no cover - trivial
                            if parent not in visited:
                                visited.add(parent)
                                queue.append((parent, d + 1))
                    depth = float(max_depth)
        except Exception:
            depth = 0.0
            branching = 0.0
        return depth, branching

    # ------------------------------------------------------------------
    def _roi_curve(self, workflow_id: str) -> List[float]:
        history: List[Dict[str, Any]] = []
        if self.roi_db is not None:
            try:
                history = self.roi_db.fetch_trends(workflow_id)
            except Exception:
                history = []
        curve = [float(rec.get("roi_gain", 0.0)) for rec in history[-self.roi_window:]]
        while len(curve) < self.roi_window:
            curve.append(0.0)
        return curve

    # ------------------------------------------------------------------
    def _code_db_context(
        self, func: str
    ) -> tuple[List[str], List[str], float, float, List[float]]:
        if not self.code_db:
            return [], [], 0.0, 0.0, []
        try:
            rows = self.code_db.search(func)
        except Exception:
            return [], [], 0.0, 0.0, []
        mod_set: set[str] = set()
        tag_set: set[str] = set()
        depths: List[float] = []
        branchings: List[float] = []
        curves: List[List[float]] = []
        for r in rows:
            m = r.get("template_type")
            if m:
                mod_set.add(str(m))
            summary = r.get("summary") or ""
            if isinstance(summary, str):
                tag_set.update(summary.split())
            ctags = r.get("context_tags") or r.get("tags") or []
            if isinstance(ctags, str):
                tag_set.update(ctags.split())
            else:
                tag_set.update(str(t) for t in ctags)
            try:
                depths.append(float(r.get("dependency_depth", 0.0) or 0.0))
            except Exception:
                pass
            try:
                branchings.append(float(r.get("branching_factor", 0.0) or 0.0))
            except Exception:
                pass
            rc = r.get("roi_curve") or r.get("roi_curves") or []
            curve_vals: List[float] = []
            if isinstance(rc, str):
                try:
                    curve_vals = [float(x) for x in json.loads(rc)]
                except Exception:
                    try:
                        curve_vals = [float(x) for x in rc.split(",") if x]
                    except Exception:
                        curve_vals = []
            elif isinstance(rc, Iterable):
                try:
                    curve_vals = [float(x) for x in rc]
                except Exception:
                    curve_vals = []
            if curve_vals:
                curves.append(curve_vals)
        depth = fmean(depths) if depths else 0.0
        branching = fmean(branchings) if branchings else 0.0
        curve: List[float] = []
        if curves:
            max_len = max(len(c) for c in curves)
            for i in range(max_len):
                curve.append(fmean(c[i] for c in curves if len(c) > i))
        return list(mod_set), list(tag_set), depth, branching, curve

    def _roi_db_context(self, workflow_id: str | None, func: str) -> tuple[float, float]:
        if not self.roi_db or not hasattr(self.roi_db, "fetch_module_trajectories"):
            return 0.0, 0.0
        if workflow_id is None:
            return 0.0, 0.0
        try:
            data = self.roi_db.fetch_module_trajectories(workflow_id, module=func)
        except Exception:
            return 0.0, 0.0
        stats = data.get(func, [])
        if not stats:
            return 0.0, 0.0
        avg_roi = sum(s.get("roi_delta", 0.0) for s in stats) / len(stats)
        avg_fail = 1.0 - (
            sum(s.get("success_rate", 0.0) for s in stats) / len(stats)
        )
        return float(avg_roi), float(avg_fail)

    # ------------------------------------------------------------------
    def _module_db_tags(self, module: str) -> tuple[List[str], List[str]]:
        """Return module categories and context tags for ``module``.

        The helper consults :class:`code_database.CodeDatabase` if available and
        gracefully falls back to empty lists when the database or methods are
        unavailable.  Implementations may expose ``get_module_categories`` and
        ``get_context_tags`` methods; both are optional.
        """

        if not self.code_db:
            return [], []

        categories: List[str] = []
        ctx_tags: List[str] = []

        try:
            if hasattr(self.code_db, "get_module_categories"):
                categories = list(self.code_db.get_module_categories(module) or [])
            elif hasattr(self.code_db, "module_categories"):
                categories = list(self.code_db.module_categories(module) or [])
        except Exception:
            categories = []

        try:
            if hasattr(self.code_db, "get_context_tags"):
                ctx_tags = list(self.code_db.get_context_tags(module) or [])
            elif hasattr(self.code_db, "context_tags"):
                ctx_tags = list(self.code_db.context_tags(module) or [])
        except Exception:
            ctx_tags = []

        return [str(c) for c in categories], [str(t) for t in ctx_tags]

    # ------------------------------------------------------------------
    def _semantic_tokens(
        self, workflow: Mapping[str, Any], workflow_id: str | None = None
    ) -> tuple[
        List[str],
        List[str],
        List[str],
        List[str],
        List[float],
        List[float],
        List[List[float]],
        List[float],
        List[float],
    ]:
        steps = workflow.get("workflow") or workflow.get("task_sequence") or []
        funcs: List[str] = []
        modules: List[str] = []
        tags: List[str] = []
        module_cats: List[str] = []
        depths: List[float] = []
        branchings: List[float] = []
        curves: List[List[float]] = []
        rois: List[float] = []
        failures: List[float] = []
        if isinstance(workflow.get("category"), str):
            cat = str(workflow["category"])
            modules.append(cat)
            mc, mt = self._module_db_tags(cat)
            module_cats.extend(mc)
            tags.extend(mt)
        if isinstance(workflow.get("platform"), str):
            plat = str(workflow["platform"])
            modules.append(plat)
            mc, mt = self._module_db_tags(plat)
            module_cats.extend(mc)
            tags.extend(mt)
        if isinstance(workflow.get("domain"), str):
            tags.append(str(workflow["domain"]))
        for step in steps:
            fn = None
            mod = None
            platform = None
            domain = None
            stags: Iterable[str] | str | None = []
            if isinstance(step, str):
                fn = step
            elif isinstance(step, Mapping):
                fn = step.get("function") or step.get("call") or step.get("name")
                mod = step.get("module") or step.get("category")
                platform = step.get("platform")
                domain = step.get("domain")
                stags = step.get("context_tags") or step.get("tags") or []
            if fn:
                fname = str(fn)
                funcs.append(fname)
                cmods, ctags, d, b, curve = self._code_db_context(fname)
                module_cats.extend(cmods)
                tags.extend(ctags)
                depths.append(d)
                branchings.append(b)
                curves.append(curve)
                r, f = self._roi_db_context(workflow_id, fname)
                rois.append(r)
                failures.append(f)
            if mod:
                mname = str(mod)
                modules.append(mname)
                mc, mt = self._module_db_tags(mname)
                module_cats.extend(mc)
                tags.extend(mt)
            if platform:
                pname = str(platform)
                modules.append(pname)
                mc, mt = self._module_db_tags(pname)
                module_cats.extend(mc)
                tags.extend(mt)
            if domain:
                tags.append(str(domain))
            if isinstance(stags, str):
                tags.append(stags)
            else:
                tags.extend(str(t) for t in stags)
        return (
            funcs,
            modules,
            tags,
            module_cats,
            depths,
            branchings,
            curves,
            rois,
            failures,
        )

    # ------------------------------------------------------------------
    def _embed_tokens(self, tokens: Iterable[str]) -> List[float]:
        """Return normalized embedding for ``tokens``.

        Tokens are individually embedded using :func:`governed_embed` and the
        resulting vectors averaged.  The mean vector is L2-normalised to unit
        length.  If embedding fails or no tokens are supplied, a zero vector of
        the model's dimensionality is returned.
        """

        vectors: List[List[float]] = []
        for tok in tokens:
            vec = governed_embed(tok)
            if vec:
                vectors.append(list(vec))
        if not vectors:
            embedder = get_embedder()
            if embedder is None:
                return []
            dim = getattr(embedder, "get_sentence_embedding_dimension", lambda: 0)()
            return [0.0] * dim
        dim = len(vectors[0])
        mean = [0.0] * dim
        for vec in vectors:
            if len(vec) != dim:
                continue
            for i, val in enumerate(vec):
                mean[i] += float(val)
        mean = [v / len(vectors) for v in mean]
        norm = math.sqrt(sum(v * v for v in mean)) or 1.0
        return [v / norm for v in mean]


# ---------------------------------------------------------------------------
def _load_embeddings(path: Path = Path("embeddings.jsonl")) -> Dict[str, List[float]]:
    """Return mapping of workflow ids to stored embeddings."""

    embeddings: Dict[str, List[float]] = {}
    if not path.exists():
        return embeddings
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("type") != "workflow_meta":
                continue
            vec = rec.get("vector") or []
            embeddings[str(rec.get("id"))] = [float(x) for x in vec]
    return embeddings


def _load_chain_embeddings(
    path: Path = _CHAIN_EMBEDDINGS_PATH,
) -> List[Dict[str, Any]]:
    """Return stored workflow chain embeddings with metadata."""

    chains: List[Dict[str, Any]] = []
    if not path.exists():
        return chains
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("type") != "workflow_chain":
                continue
            vec = [float(x) for x in rec.get("vector", [])]
            meta = rec.get("metadata", {}) or {}
            chains.append(
                {
                    "id": str(rec.get("id", "")),
                    "vector": vec,
                    "roi": float(meta.get("roi", 0.0)),
                    "entropy": float(meta.get("entropy", 0.0)),
                }
            )
    return chains


# ---------------------------------------------------------------------------
def _roi_scores(tracker: ROITracker) -> Dict[str, float]:
    """Return average ROI gain per workflow from ``tracker``."""

    scores: Dict[str, float] = {}
    for wf_id, hist in getattr(tracker, "final_roi_history", {}).items():
        if hist:
            scores[wf_id] = sum(float(x) for x in hist) / len(hist)
    return scores


# ---------------------------------------------------------------------------
def _roi_weight_from_db(
    db: ROIResultsDB, workflow_id: str, window: int = 5
) -> float:
    """Return average ROI gain for ``workflow_id`` from ``db``.

    The function consults :meth:`ROIResultsDB.fetch_trends` and computes the
    mean ``roi_gain`` over the most recent ``window`` entries.  Any database
    errors simply result in a weight of ``0.0`` so callers can use this in
    best-effort contexts without additional error handling.
    """

    try:
        trends = db.fetch_trends(workflow_id)
    except Exception:
        return 0.0
    if not trends:
        return 0.0
    recent = trends[-window:]
    return sum(float(t.get("roi_gain", 0.0)) for t in recent) / len(recent)


# ---------------------------------------------------------------------------

def find_synergy_candidates(
    query: Sequence[float] | str,
    *,
    top_k: int = 5,
    context_builder: ContextBuilder,
    retriever: Retriever | None = None,
    roi_db: ROIResultsDB | None = None,
    roi_window: int = 5,
    cluster_map: Mapping[tuple[str, ...], Mapping[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """Return top ``top_k`` workflows similar to ``query`` weighted by ROI.

    ``query`` may be either a numeric embedding vector or an identifier of a
    stored workflow embedding.  ``context_builder`` is used to create a
    :class:`Retriever` when one is not supplied explicitly.  Results are ranked
    by cosine similarity scaled by the average ROI gain from ``roi_db``.
    Retrieval failures yield an empty list.  When ``cluster_map`` is provided,
    historic reinforcement scores for ``(query, candidate)`` pairs are used to
    boost similarity prior to ranking.
    """

    try:
        ensure_fresh_weights(context_builder)
    except Exception:
        return []
    if retriever is None and Retriever is not None:
        try:
            retriever = Retriever(context_builder=context_builder)
        except Exception:
            retriever = None
    if retriever is None:
        return []

    roi_db = roi_db or ROIResultsDB()
    embeddings = _load_embeddings()

    query_id: str | None
    if isinstance(query, str):
        query_vec = embeddings.get(query)
        if query_vec is None:
            return []
        exclude = {query}
        query_id = query
    else:
        query_vec = [float(x) for x in query]
        exclude = set()
        query_id = None

    try:  # pragma: no cover - optional path
        ur = retriever._get_retriever()
        hits, _, _ = ur.retrieve(
            query_vec, top_k=top_k * 3, dbs=["workflow_meta"]
        )  # type: ignore[attr-defined]
    except Exception:
        return []

    scored: List[Dict[str, Any]] = []
    for hit in hits:
        wf_id = str(
            getattr(hit, "record_id", None)
            or getattr(getattr(hit, "metadata", {}), "get", lambda *_: None)("id")
            or ""
        )
        if not wf_id or wf_id in exclude:
            continue
        vec = embeddings.get(wf_id)
        if vec is None:
            continue
        sim = cosine_similarity(query_vec, vec)

        if cluster_map is not None and query_id is not None:
            cm_score = float(cluster_map.get((query_id, wf_id), {}).get("score", 0.0))
            sim *= 1.0 + cm_score

        roi = _roi_weight_from_db(roi_db, wf_id, window=roi_window)
        scored.append(
            {
                "workflow_id": wf_id,
                "similarity": sim,
                "roi": roi,
                "score": sim * (1.0 + roi),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
def find_synergistic_workflows(
    workflow_id: str,
    top_k: int = 5,
    *,
    context_builder: ContextBuilder,
    retriever: Retriever | None = None,
) -> List[Dict[str, Any]]:
    """Return workflows synergistic with ``workflow_id`` ranked by ROI."""

    try:
        ensure_fresh_weights(context_builder)
    except Exception:
        return []
    embeddings = _load_embeddings()
    query_vec = embeddings.get(workflow_id)
    if query_vec is None:
        return []

    roi_db = ROIResultsDB()
    results: List[Dict[str, Any]] = []
    try:  # pragma: no cover - optional dependency path
        from annoy import AnnoyIndex  # type: ignore

        index = AnnoyIndex(len(query_vec), "angular")
        ids: List[str] = []
        for idx, (wf_id, vec) in enumerate(embeddings.items()):
            index.add_item(idx, vec)
            ids.append(wf_id)
        index.build(10)
        neighbors = index.get_nns_by_vector(query_vec, top_k * 3)
        for idx in neighbors:
            cand_id = ids[idx]
            if cand_id == workflow_id:
                continue
            vec = embeddings[cand_id]
            sim = cosine_similarity(query_vec, vec)
            roi = _roi_weight_from_db(roi_db, cand_id)
            results.append(
                {
                    "workflow_id": cand_id,
                    "similarity": sim,
                    "roi": roi,
                    "score": sim * (1.0 + roi),
                }
            )
    except Exception:
        if retriever is None and Retriever is not None:
            try:  # pragma: no cover - best effort
                retriever = Retriever(context_builder=context_builder)
            except Exception:
                retriever = None
        if retriever is None:
            return []
        return find_synergy_candidates(
            workflow_id,
            top_k=top_k,
            context_builder=context_builder,
            retriever=retriever,
            roi_db=roi_db,
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

# ---------------------------------------------------------------------------
def find_synergy_chain(
    start_workflow_id: str,
    length: int = 5,
    *,
    context_builder: ContextBuilder,
    cluster_map: Mapping[tuple[str, ...], Mapping[str, Any]] | None = None,
) -> List[str]:
    """Return high-synergy workflow sequence starting from ``start_workflow_id``.

    The chain is biased toward historically successful domain transitions using
    :func:`MetaWorkflowPlanner.transition_probabilities`.  When transition
    statistics are available, moving between domains with higher ROI deltas is
    preferred, e.g. ``YouTube -> Reddit -> Email``.  If a ``cluster_map`` is
    provided (or loaded via :class:`MetaWorkflowPlanner`), historic reinforcement
    scores for ``(current, candidate)`` pairs further boost similarity when
    selecting each step.
    """

    embeddings = _load_embeddings()
    start_vec = embeddings.get(start_workflow_id)
    if start_vec is None:
        return []

    chain_recs = _load_chain_embeddings()
    best_chain: List[str] | None = None
    best_score = 0.0
    for rec in chain_recs:
        steps = rec["id"].split("->") if rec.get("id") else []
        if not steps or steps[0] != start_workflow_id:
            continue
        sim = cosine_similarity(start_vec, rec.get("vector", []))
        score = sim * (1.0 + rec.get("roi", 0.0)) * max(
            0.0, 1.0 - rec.get("entropy", 0.0)
        )
        if score > best_score:
            best_chain = steps
            best_score = score
    if best_chain is not None:
        return best_chain[:length]

    tracker = ROITracker()
    try:
        history_file = resolve_path("roi_history.json")
    except FileNotFoundError:
        history_file = Path("roi_history.json")
    if history_file.exists():
        try:
            tracker.load_history(str(history_file))
        except Exception:
            logger.warning(
                "Failed to load ROI history from %s", history_file, exc_info=True
            )
    roi_scores = _roi_scores(tracker)
    planner = MetaWorkflowPlanner(context_builder=context_builder)
    if cluster_map is None:
        cluster_map = getattr(planner, "cluster_map", {})
    # Ensure the planner's cluster map reflects the provided data so that
    # _failure_entropy_metrics can consult historic metrics.
    planner.cluster_map = dict(cluster_map)

    trans_probs = planner.transition_probabilities()
    prev_domains = planner._workflow_domain(start_workflow_id)[0]

    chain = [start_workflow_id]
    current = start_workflow_id
    for _ in range(max(0, length - 1)):
        current_vec = embeddings[current]
        best_id: str | None = None
        best_score = 0.0
        for wf_id, vec in embeddings.items():
            if wf_id in chain:
                continue
            sim = cosine_similarity(current_vec, vec)
            score = sim * roi_scores.get(wf_id, 0.0)
            # Penalise workflows with higher failure rates or entropy.
            failure, entropy = planner._failure_entropy_metrics(wf_id)
            score *= (1.0 - failure) * (1.0 - entropy)
            if cluster_map is not None:
                cm_score = float(cluster_map.get((current, wf_id), {}).get("score", 0.0))
                score *= 1.0 + cm_score
            cand_domains = planner._workflow_domain(wf_id)[0]
            if prev_domains and cand_domains:
                prob = 0.0
                for src in prev_domains:
                    for dst in cand_domains:
                        prob = max(prob, trans_probs.get((src, dst), 0.0))
                score *= 1.0 + prob
            if score > best_score:
                best_id = wf_id
                best_score = score
        if best_id is None:
            break
        chain.append(best_id)
        current = best_id
        prev_domains = planner._workflow_domain(best_id)[0]
    return chain


# ---------------------------------------------------------------------------
def plan_pipeline(
    start_workflow_id: str,
    workflows: Mapping[str, Mapping[str, Any]],
    *,
    length: int = 5,
) -> List[str]:
    """Plan a workflow pipeline biased by domain transition ROI.

    This convenience wrapper instantiates :class:`MetaWorkflowPlanner` and uses
    :meth:`MetaWorkflowPlanner.compose_pipeline` which consults
    :func:`transition_probabilities` to favour historically profitable domain
    shifts.
    """

    planner = MetaWorkflowPlanner()
    # Ensure transition statistics are loaded before composing the pipeline.
    planner.transition_probabilities()
    return planner.compose_pipeline(start_workflow_id, workflows, length=length)


# ---------------------------------------------------------------------------
def _io_compatible(graph: WorkflowGraph, a: str, b: str) -> bool:
    """Return ``True`` when the outputs of ``a`` satisfy the inputs of ``b``.

    This helper relies on :func:`workflow_graph.get_io_signature` which must
    provide explicit typed ``inputs`` and ``outputs`` mappings.  Each mapping is
    expected to describe the name of the channel and its MIME or schema type.
    Compatibility requires that all of ``a``'s outputs exactly match ``b``'s
    declared inputs (both names and types).  Missing or partially specified
    signatures are treated as incompatible and result in ``False`` being
    returned.
    """

    getter = getattr(graph, "get_io_signature", None)
    if not callable(getter):
        return False
    try:
        sig_a = getter(a)
        sig_b = getter(b)
    except Exception:
        return False

    if not sig_a or not sig_b:
        return False

    out_a = getattr(sig_a, "outputs", None)
    in_b = getattr(sig_b, "inputs", None)
    if out_a is None and isinstance(sig_a, Mapping):
        out_a = sig_a.get("outputs")
    if in_b is None and isinstance(sig_b, Mapping):
        in_b = sig_b.get("inputs")

    if not isinstance(out_a, Mapping) or not isinstance(in_b, Mapping):
        return False

    # Ensure explicit data types are provided for every channel
    if any(not isinstance(v, str) or not v for v in out_a.values()):
        return False
    if any(not isinstance(v, str) or not v for v in in_b.values()):
        return False

    try:
        return dict(out_a) == dict(in_b)
    except Exception:
        return False


# ---------------------------------------------------------------------------
def compose_meta_workflow(
    start_workflow_id: str,
    *,
    length: int = 5,
    graph: WorkflowGraph | None = None,
    context_builder: ContextBuilder | None = None,
) -> Dict[str, Any]:
    """Return ordered meta-workflow specification starting from ``start_workflow_id``.

    The function builds a high-synergy chain via :func:`find_synergy_chain` and
    filters it so consecutive workflows have compatible I/O signatures according
    to :func:`workflow_graph.get_io_signature`.  The resulting meta-workflow is
    expressed as a dictionary containing the ordered ``steps`` and a human
    readable ``chain`` string (e.g. ``scrape->analyze->generate``).
    """

    chain = find_synergy_chain(
        start_workflow_id, length=length, context_builder=context_builder
    )
    if not chain:
        return {}

    graph = graph or WorkflowGraph()

    ordered: List[str] = []
    prev: str | None = None
    for wid in chain:
        if prev is None:
            ordered.append(wid)
            prev = wid
            continue

        if _io_compatible(graph, prev, wid):
            ordered.append(wid)
            prev = wid
    return {
        "chain": "->".join(ordered),
        "steps": [{"workflow_id": wid} for wid in ordered],
    }


# ---------------------------------------------------------------------------
def simulate_meta_workflow(
    meta_spec: Mapping[str, Any],
    workflows: Mapping[str, Callable[[], Any]] | None = None,
    runner: "WorkflowSandboxRunner" | None = None,
) -> Dict[str, Any]:
    """Recursively execute a meta-workflow specification.

    Each step may reference a ``workflow_id`` present in ``workflows`` or embed
    a nested ``steps`` sequence forming a sub meta-workflow.  Every referenced
    workflow is executed using :class:`sandbox_runner.WorkflowSandboxRunner` and
    the resulting ROI gain, failure counts and entropy stability are aggregated
    across all runs.

    Parameters
    ----------
    meta_spec:
        Meta-workflow specification containing ``steps`` entries.
    workflows:
        Optional mapping of workflow identifiers to callables.  When omitted the
        callable must be provided directly in each step under ``workflow`` or
        ``call``.
    runner:
        Optional :class:`sandbox_runner.WorkflowSandboxRunner` instance to
        reuse.  A new instance is created when ``None``.

    Returns
    -------
    Dict[str, Any]
        Mapping with aggregated ``roi_gain``, ``failures`` and average
        ``entropy`` values.
    """

    if runner is None:
        try:
            from sandbox_runner.workflow_sandbox_runner import (
                WorkflowSandboxRunner as _Runner,
            )

            runner = _Runner()
        except Exception:
            return {"roi_gain": 0.0, "failures": 0, "entropy": 0.0}

    comparator = WorkflowSynergyComparator
    try:  # pragma: no cover - allow dynamic replacement via sys.modules
        import sys
        mod = sys.modules.get("workflow_synergy_comparator")
        if mod is not None:
            comparator = getattr(mod, "WorkflowSynergyComparator", comparator)
    except Exception:
        pass
    if comparator is None:
        logger.warning(
            "WorkflowSynergyComparator unavailable; entropy metrics will be zero"
        )

    total_roi = 0.0
    total_failures = 0
    entropies: List[float] = []

    def _run_spec(spec: Mapping[str, Any]) -> None:
        nonlocal total_roi, total_failures, entropies
        for step in spec.get("steps", []):
            if isinstance(step, Mapping) and step.get("steps"):
                _run_spec(step)
                continue

            wid = None
            func: Callable[[], Any] | None = None
            if isinstance(step, Mapping):
                wid = step.get("workflow_id") or step.get("id")
                wf_obj = step.get("workflow") or step.get("call")
                if callable(wf_obj):
                    func = wf_obj  # type: ignore[assignment]

            if func is None and workflows and wid in workflows:
                func = workflows[wid]

            if not callable(func):
                continue

            metrics = runner.run([func])
            roi_gain = sum(
                float(m.result)
                for m in metrics.modules
                if isinstance(m.result, (int, float))
            )
            failures = max(
                metrics.crash_count,
                sum(1 for m in metrics.modules if not m.success),
            )
            total_roi += roi_gain
            total_failures += failures

            entropy = 0.0
            if wid and comparator is not None:
                try:
                    entropy = comparator._entropy({"steps": [{"module": wid}]})
                except Exception:
                    entropy = 0.0
            entropies.append(entropy)

    if isinstance(meta_spec, Mapping):
        _run_spec(meta_spec)

    avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    return {
        "roi_gain": total_roi,
        "failures": total_failures,
        "entropy": avg_entropy,
    }


__all__ = [
    "MetaWorkflowPlanner",
    "simulate_meta_workflow",
    "compose_meta_workflow",
    "plan_pipeline",
    "find_synergy_chain",
    "find_synergy_candidates",
    "find_synergistic_workflows",
]
