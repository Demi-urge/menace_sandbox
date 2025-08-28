from __future__ import annotations

"""Meta workflow planning utilities with semantic and structural embedding."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, TYPE_CHECKING

from roi_results_db import ROIResultsDB
from workflow_graph import WorkflowGraph
from vector_utils import persist_embedding, cosine_similarity

try:  # pragma: no cover - optional heavy dependency
    from vector_service.retriever import Retriever  # type: ignore
except Exception:  # pragma: no cover - allow running without retriever
    Retriever = None  # type: ignore

try:  # pragma: no cover - optional heavy dependency
    from roi_tracker import ROITracker  # type: ignore
except Exception:  # pragma: no cover - allow running without ROI tracker
    ROITracker = None  # type: ignore

try:  # pragma: no cover - optional persistence helper
    from workflow_stability_db import WorkflowStabilityDB  # type: ignore
except Exception:  # pragma: no cover - allow running without stability db
    WorkflowStabilityDB = None  # type: ignore

try:  # pragma: no cover - optional heavy dependency
    import networkx as nx  # type: ignore
    _HAS_NX = True
except Exception:  # pragma: no cover - executed when networkx missing
    nx = None  # type: ignore
    _HAS_NX = False

try:  # pragma: no cover - optional heavy dependency
    from workflow_synergy_comparator import WorkflowSynergyComparator  # type: ignore
except Exception:  # pragma: no cover - allow running without comparator
    WorkflowSynergyComparator = None  # type: ignore

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

    max_functions: int = 50
    max_modules: int = 50
    max_tags: int = 50
    roi_window: int = 5
    function_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})
    module_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})
    tag_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})
    graph: WorkflowGraph | None = None
    roi_db: ROIResultsDB | None = None
    roi_tracker: ROITracker | None = None
    stability_db: WorkflowStabilityDB | None = None
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
                self.roi_tracker = ROITracker(window=self.roi_window)
            except Exception:
                self.roi_tracker = None
        if self.stability_db is None and WorkflowStabilityDB is not None:
            try:
                self.stability_db = WorkflowStabilityDB()
            except Exception:
                self.stability_db = None

    # ------------------------------------------------------------------
    def encode(self, workflow_id: str, workflow: Mapping[str, Any]) -> List[float]:
        """Return embedding for ``workflow`` and persist it."""

        depth, branching = self._graph_features(workflow_id)
        roi_curve = self._roi_curve(workflow_id)
        funcs, mods, tags = self._semantic_tokens(workflow)

        vec: List[float] = []
        vec.extend([depth, branching])
        vec.extend(roi_curve)
        vec.extend(self._encode_tokens(funcs, self.function_index, self.max_functions))
        vec.extend(self._encode_tokens(mods, self.module_index, self.max_modules))
        vec.extend(self._encode_tokens(tags, self.tag_index, self.max_tags))

        try:
            persist_embedding(
                "workflow_meta",
                workflow_id,
                vec,
                origin_db="workflow",
                metadata={
                    "roi_curve": roi_curve,
                    "dependency_depth": depth,
                    "branching_factor": branching,
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
        funcs, mods, tags = self._semantic_tokens(workflow)

        vec: List[float] = [depth, branching]
        vec.extend(self._encode_tokens(funcs, self.function_index, self.max_functions))
        vec.extend(self._encode_tokens(mods, self.module_index, self.max_modules))
        vec.extend(self._encode_tokens(tags, self.tag_index, self.max_tags))
        return vec

    # ------------------------------------------------------------------
    def cluster_workflows(
        self, workflows: Mapping[str, Mapping[str, Any]], *, threshold: float = 0.75
    ) -> List[List[str]]:
        """Group ``workflows`` into similarity clusters.

        The clustering uses :class:`WorkflowSynergyComparator` to score pairs of
        workflows.  Workflows with an aggregate synergy score of at least
        ``threshold`` are placed in the same cluster.  When the comparator is
        unavailable each workflow forms its own cluster.
        """

        remaining = set(workflows.keys())
        clusters: List[List[str]] = []

        while remaining:
            wid = remaining.pop()
            spec_a = workflows[wid]
            cluster = [wid]
            for other in list(remaining):
                if WorkflowSynergyComparator is None:
                    score = 0.0
                else:
                    try:
                        score = WorkflowSynergyComparator.compare(
                            spec_a, workflows[other]
                        ).aggregate
                    except Exception:
                        score = 0.0
                if score >= threshold:
                    cluster.append(other)
                    remaining.remove(other)
            clusters.append(cluster)
        return clusters

    # ------------------------------------------------------------------
    def compose_pipeline(
        self,
        start: str,
        workflows: Mapping[str, Mapping[str, Any]],
        *,
        length: int = 3,
    ) -> List[str]:
        """Compose a high-synergy workflow pipeline.

        Starting from ``start`` the method iteratively selects the workflow with
        the highest synergy score to the current tail using
        :class:`WorkflowSynergyComparator`.  The process stops once ``length``
        steps have been selected or no suitable candidates remain.
        """

        if start not in workflows:
            return []

        pipeline = [start]
        available = {k for k in workflows.keys() if k != start}
        current = start

        while available and len(pipeline) < length:
            best_id: str | None = None
            best_score = -1.0
            for wid in available:
                if WorkflowSynergyComparator is None:
                    score = 0.0
                else:
                    try:
                        score = WorkflowSynergyComparator.compare(
                            workflows[current], workflows[wid]
                        ).aggregate
                    except Exception:
                        score = 0.0
                if score > best_score:
                    best_id = wid
                    best_score = score
            if best_id is None:
                break
            pipeline.append(best_id)
            available.remove(best_id)
            current = best_id

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
    ) -> List[Dict[str, Any]]:
        """Suggest and validate workflow chains.

        ``target_embedding`` is clustered via :class:`WorkflowChainSuggester`
        to obtain candidate sequences of workflow identifiers.  Each suggested
        chain is executed inside a :class:`WorkflowSandboxRunner` and the
        resulting ROI, failure count and entropy are recorded.  Chains that
        exceed ``failure_threshold`` or ``entropy_threshold`` are discarded.

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
            )
            if record:
                results.append(record)

        return results

    # ------------------------------------------------------------------
    def _validate_chain(
        self,
        chain: Sequence[str],
        workflows: Mapping[str, Callable[[], Any]],
        *,
        runner: WorkflowSandboxRunner | None = None,
        failure_threshold: int = 0,
        entropy_threshold: float = 2.0,
    ) -> Dict[str, Any] | None:
        """Validate a single chain and update ROI logs and cluster map."""

        funcs: List[Callable[[], Any]] = []
        for wid in chain:
            fn = workflows.get(wid)
            if not callable(fn):
                return None
            funcs.append(fn)

        if runner is None:
            try:
                from sandbox_runner.workflow_sandbox_runner import (
                    WorkflowSandboxRunner as _Runner,
                )  # type: ignore

                runner = _Runner()
            except Exception:
                return None

        try:
            from workflow_synergy_comparator import WorkflowSynergyComparator  # type: ignore
        except Exception:
            WorkflowSynergyComparator = None  # type: ignore

        metrics = runner.run(funcs)
        failure_count = max(
            metrics.crash_count,
            sum(1 for m in metrics.modules if not m.success),
        )
        spec = {"steps": [{"module": m} for m in chain]}
        if WorkflowSynergyComparator is not None:
            try:
                entropy = WorkflowSynergyComparator._entropy(spec)
            except Exception:
                entropy = 0.0
        else:
            entropy = 0.0
        roi_gain = sum(
            float(m.result)
            for m in metrics.modules
            if isinstance(m.result, (int, float))
        )

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
                    metrics_delta={"failures": float(failure_count), "entropy": float(entropy)},
                )
            except Exception:
                pass
        if self.stability_db is not None:
            try:
                self.stability_db.record_metrics(
                    chain_id,
                    roi_gain,
                    failure_count,
                    entropy,
                    roi_delta=roi_delta,
                )
            except Exception:
                pass

        if failure_count > failure_threshold or entropy > entropy_threshold:
            return None

        if self.roi_db is not None:
            try:
                self.roi_db.log_result(
                    workflow_id="->".join(chain),
                    run_id="0",
                    runtime=sum(m.duration for m in metrics.modules),
                    success_rate=(
                        (len(metrics.modules) - failure_count) / len(metrics.modules)
                        if metrics.modules
                        else 0.0
                    ),
                    roi_gain=roi_gain,
                    workflow_synergy_score=max(0.0, 1.0 - entropy),
                    bottleneck_index=0.0,
                    patchability_score=0.0,
                    module_deltas={
                        m.name: {
                            "roi_delta": float(m.result)
                            if isinstance(m.result, (int, float))
                            else 0.0,
                            "success_rate": 1.0 if m.success else 0.0,
                        }
                        for m in metrics.modules
                    },
                )
            except Exception:
                pass

        self._update_cluster_map(chain, roi_gain)

        return {
            "chain": list(chain),
            "roi_gain": roi_gain,
            "failures": failure_count,
            "entropy": entropy,
        }

    # ------------------------------------------------------------------
    def mutate_chains(
        self,
        chains: Sequence[Sequence[str]],
        workflows: Mapping[str, Callable[[], Any]],
        *,
        runner: WorkflowSandboxRunner | None = None,
        failure_threshold: int = 0,
        entropy_threshold: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """Mutate chains (swap/remove/add) and re-validate."""

        wf_ids = list(workflows.keys())
        candidates: List[List[str]] = []
        for chain in chains:
            chain = list(chain)
            if len(chain) >= 2:
                swapped = chain[:]
                swapped[0], swapped[1] = swapped[1], swapped[0]
                candidates.append(swapped)
            if len(chain) > 1:
                candidates.append(chain[:-1])
            for wid in wf_ids:
                if wid not in chain:
                    candidates.append(chain + [wid])
                    break

        results: List[Dict[str, Any]] = []
        for c in candidates:
            record = self._validate_chain(
                c,
                workflows,
                runner=runner,
                failure_threshold=failure_threshold,
                entropy_threshold=entropy_threshold,
            )
            if record:
                results.append(record)
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
    ) -> List[Dict[str, Any]]:
        """Split underperforming subchains and merge high-ROI chains."""

        low = [r["chain"] for r in records if r.get("roi_gain", 0.0) <= roi_threshold]
        high = [r["chain"] for r in records if r.get("roi_gain", 0.0) > roi_threshold]

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
        )
        if not record:
            return []

        chain_id = "->".join(pipeline)
        roi_delta = 0.0
        entropy_val = record.get("entropy", 0.0)
        if self.stability_db is not None:
            entry = self.stability_db.data.get(chain_id, {})
            roi_delta = float(entry.get("roi_delta", 0.0))
            entropy_val = float(entry.get("entropy", entropy_val))
        else:
            roi_delta = record.get("roi_gain", 0.0)

        if (
            roi_delta >= roi_improvement_threshold
            and abs(entropy_val) <= entropy_stability_threshold
        ):
            return [record]

        results = self.mutate_chains(
            [pipeline],
            workflows,
            runner=runner,
            failure_threshold=failure_threshold,
            entropy_threshold=entropy_threshold,
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
            pass

        return results

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
    ) -> List[Dict[str, Any]]:
        """Split ``pipeline`` into sub-pipelines when improvement stalls."""

        if len(pipeline) <= 1:
            return []

        record = self._validate_chain(
            pipeline,
            workflows,
            runner=runner,
            failure_threshold=failure_threshold,
            entropy_threshold=entropy_threshold,
        )
        if not record:
            return []

        chain_id = "->".join(pipeline)
        roi_delta = 0.0
        entropy_val = record.get("entropy", 0.0)
        if self.stability_db is not None:
            entry = self.stability_db.data.get(chain_id, {})
            roi_delta = float(entry.get("roi_delta", 0.0))
            entropy_val = float(entry.get("entropy", entropy_val))
        else:
            roi_delta = record.get("roi_gain", 0.0)

        if roi_delta >= roi_improvement_threshold or abs(entropy_val) > entropy_stability_threshold:
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
            )
            if rec:
                results.append(rec)
                try:  # pragma: no cover - best effort logging
                    from workflow_lineage import log_lineage

                    log_lineage(chain_id, "->".join(seg), "split_pipeline", roi=rec.get("roi_gain"))
                except Exception:
                    pass
        return results

    # ------------------------------------------------------------------
    def remerge_pipelines(
        self,
        pipelines: Sequence[Sequence[str]],
        workflows: Mapping[str, Callable[[], Any]],
        *,
        roi_improvement_threshold: float = 0.0,
        entropy_stability_threshold: float = 1.0,
        runner: WorkflowSandboxRunner | None = None,
        failure_threshold: int = 0,
        entropy_threshold: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """Merge pipelines that show stable entropy and ROI improvements."""

        results: List[Dict[str, Any]] = []
        for i in range(len(pipelines)):
            for j in range(i + 1, len(pipelines)):
                merged = list(pipelines[i]) + [w for w in pipelines[j] if w not in pipelines[i]]
                rec = self._validate_chain(
                    merged,
                    workflows,
                    runner=runner,
                    failure_threshold=failure_threshold,
                    entropy_threshold=entropy_threshold,
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
                    try:  # pragma: no cover - best effort logging
                        from workflow_lineage import log_lineage

                        log_lineage(
                            None,
                            chain_id,
                            "remerge_pipelines",
                            roi=rec.get("roi_gain"),
                        )
                    except Exception:
                        pass
        return results

    # ------------------------------------------------------------------
    def _update_cluster_map(
        self, chain: Sequence[str], roi_gain: float, *, tol: float = 0.01
    ) -> Dict[str, Any]:
        """Update ROI delta history for ``chain`` and detect convergence."""

        key = tuple(chain)
        info = self.cluster_map.setdefault(
            key, {"roi_history": [], "delta_roi": 0.0, "converged": False}
        )
        history = info["roi_history"]
        if history:
            info["delta_roi"] = roi_gain - history[-1]
            if abs(info["delta_roi"]) < tol:
                info["converged"] = True
        history.append(roi_gain)
        return info

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
    def _semantic_tokens(
        self, workflow: Mapping[str, Any]
    ) -> tuple[List[str], List[str], List[str]]:
        steps = workflow.get("workflow") or workflow.get("task_sequence") or []
        funcs: List[str] = []
        modules: List[str] = []
        tags: List[str] = []
        if isinstance(workflow.get("category"), str):
            modules.append(str(workflow["category"]))
        for step in steps:
            if isinstance(step, str):
                funcs.append(step)
            elif isinstance(step, Mapping):
                fn = step.get("function") or step.get("call") or step.get("name")
                if fn:
                    funcs.append(str(fn))
                mod = step.get("module") or step.get("category")
                if mod:
                    modules.append(str(mod))
                stags = step.get("context_tags") or step.get("tags") or []
                if isinstance(stags, str):
                    tags.append(stags)
                else:
                    tags.extend(str(t) for t in stags)
        return funcs, modules, tags

    # ------------------------------------------------------------------
    def _encode_tokens(
        self, tokens: Iterable[str], mapping: Dict[str, int], max_size: int
    ) -> List[float]:
        vec = [0.0] * max_size
        for tok in {t.lower().strip() for t in tokens if t}:
            idx = _get_index(tok, mapping, max_size)
            if idx < max_size:
                vec[idx] = 1.0
        return vec


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
    retriever: Retriever | None = None,
    roi_db: ROIResultsDB | None = None,
    roi_window: int = 5,
) -> List[Dict[str, Any]]:
    """Return top ``top_k`` workflows similar to ``query`` weighted by ROI.

    ``query`` may be either a numeric embedding vector or an identifier of a
    stored workflow embedding.  Results are ranked by cosine similarity scaled
    by the average ROI gain retrieved from ``roi_db``.  When ``retriever`` is
    provided the underlying :class:`vector_service.retriever.Retriever` is used
    to gather candidate workflow identifiers.  If ``retriever`` is ``None`` or
    the lookup fails the function falls back to scanning all stored
    embeddings.
    """

    roi_db = roi_db or ROIResultsDB()
    embeddings = _load_embeddings()

    if isinstance(query, str):
        query_vec = embeddings.get(query)
        if query_vec is None:
            return []
        exclude = {query}
    else:
        query_vec = [float(x) for x in query]
        exclude: set[str] = set()

    candidates: List[tuple[str, float]] = []

    if retriever is None and Retriever is not None:
        try:  # pragma: no cover - best effort
            retriever = Retriever()
        except Exception:
            retriever = None

    if retriever is not None and Retriever is not None:
        try:  # pragma: no cover - optional path
            ur = retriever._get_retriever()
            hits, _, _ = ur.retrieve(
                query_vec, top_k=top_k * 3, dbs=["workflow_meta"]
            )  # type: ignore[attr-defined]
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
                candidates.append((wf_id, sim))
        except Exception:
            candidates = []

    if not candidates:
        for wf_id, vec in embeddings.items():
            if wf_id in exclude:
                continue
            sim = cosine_similarity(query_vec, vec)
            candidates.append((wf_id, sim))

    scored: List[Dict[str, Any]] = []
    for wf_id, sim in candidates:
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
def find_synergistic_workflows(workflow_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Return workflows synergistic with ``workflow_id`` ranked by ROI."""

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
        return find_synergy_candidates(workflow_id, top_k=top_k, roi_db=roi_db)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
def find_synergy_chain(start_workflow_id: str, length: int = 5) -> List[str]:
    """Return high-synergy workflow sequence starting from ``start_workflow_id``."""

    embeddings = _load_embeddings()
    start_vec = embeddings.get(start_workflow_id)
    if start_vec is None:
        return []

    tracker = ROITracker()
    history_file = Path("roi_history.json")
    if history_file.exists():
        try:
            tracker.load_history(str(history_file))
        except Exception:
            pass
    roi_scores = _roi_scores(tracker)

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
            if score > best_score:
                best_id = wf_id
                best_score = score
        if best_id is None:
            break
        chain.append(best_id)
        current = best_id
    return chain


# ---------------------------------------------------------------------------
def _io_compatible(graph: WorkflowGraph, a: str, b: str) -> bool:
    """Return ``True`` when the outputs of ``a`` feed the inputs of ``b``.

    The check relies on :func:`workflow_graph.get_io_signature`.  When the
    signatures are unavailable or the function is missing, compatibility is
    assumed to avoid false negatives.
    """

    getter = getattr(graph, "get_io_signature", None)
    if not callable(getter):
        return True
    try:
        sig_a = getter(a)
        sig_b = getter(b)
    except Exception:
        return True

    if not sig_a or not sig_b:
        return True

    out_a = getattr(sig_a, "outputs", None)
    in_b = getattr(sig_b, "inputs", None)
    if out_a is None and isinstance(sig_a, Mapping):
        out_a = sig_a.get("outputs")
    if in_b is None and isinstance(sig_b, Mapping):
        in_b = sig_b.get("inputs")

    if not out_a or not in_b:
        return True
    try:
        return bool(set(out_a) & set(in_b))
    except Exception:
        return True


# ---------------------------------------------------------------------------
def compose_meta_workflow(
    start_workflow_id: str,
    *,
    length: int = 5,
    graph: WorkflowGraph | None = None,
) -> Dict[str, Any]:
    """Return ordered meta-workflow specification starting from ``start_workflow_id``.

    The function builds a high-synergy chain via :func:`find_synergy_chain` and
    filters it so consecutive workflows have compatible I/O signatures according
    to :func:`workflow_graph.get_io_signature`.  The resulting meta-workflow is
    expressed as a dictionary containing the ordered ``steps`` and a human
    readable ``chain`` string (e.g. ``scrape->analyze->generate``).
    """

    chain = find_synergy_chain(start_workflow_id, length=length)
    if not chain:
        return {}

    graph = graph or WorkflowGraph()

    ordered: List[str] = []
    prev: str | None = None
    for wid in chain:
        if prev is None or _io_compatible(graph, prev, wid):
            ordered.append(wid)
            prev = wid
        else:
            break

    return {
        "chain": "->".join(ordered),
        "steps": [{"workflow_id": wid} for wid in ordered],
    }


__all__ = [
    "MetaWorkflowPlanner",
    "compose_meta_workflow",
    "find_synergy_chain",
    "find_synergy_candidates",
    "find_synergistic_workflows",
]
