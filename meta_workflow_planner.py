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

try:  # pragma: no cover - optional code database
    from code_database import CodeDB  # type: ignore
except Exception:  # pragma: no cover - database unavailable
    CodeDB = None  # type: ignore

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
                self.roi_tracker = ROITracker(window=self.roi_window)
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
        self._load_cluster_map()

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
        ) = self._semantic_tokens(workflow)
        domain = workflow.get("domain")
        if isinstance(domain, str) and domain:
            tags.append(domain)

        # Merge in module categories from the code database
        mods.extend(mod_cats)

        # Normalize tokens prior to encoding
        funcs = sorted({f.lower().strip() for f in funcs if f})
        mods = sorted({m.lower().strip() for m in mods if m})
        tags = sorted({t.lower().strip() for t in tags if t})

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
        vec.extend(self._encode_tokens(funcs, self.function_index, self.max_functions))
        vec.extend(self._encode_tokens(mods, self.module_index, self.max_modules))
        vec.extend(self._encode_tokens(tags, self.tag_index, self.max_tags))

        code_tags = tags

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
                    "domain": domain,
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
        ) = self._semantic_tokens(workflow)

        mods.extend(mod_cats)

        funcs = sorted({f.lower().strip() for f in funcs if f})
        mods = sorted({m.lower().strip() for m in mods if m})
        tags = sorted({t.lower().strip() for t in tags if t})

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
        vec.extend(self._encode_tokens(funcs, self.function_index, self.max_functions))
        vec.extend(self._encode_tokens(mods, self.module_index, self.max_modules))
        vec.extend(self._encode_tokens(tags, self.tag_index, self.max_tags))
        return vec

    # ------------------------------------------------------------------
    def cluster_workflows(
        self,
        workflows: Mapping[str, Mapping[str, Any]],
        *,
        threshold: float = 0.75,
        retriever: Retriever | None = None,
    ) -> List[List[str]]:
        """Group ``workflows`` into similarity clusters.

        Each workflow is encoded via :meth:`encode_workflow` and the resulting
        vector is used to query a :class:`vector_service.retriever.Retriever`
        for similar candidates.  The cosine similarity of the query and
        candidate vectors is multiplied by ``(1 + ROI)`` for both workflows
        using weights from :func:`_roi_weight_from_db`.  Pairs whose weighted
        similarity meets or exceeds ``threshold`` are placed in the same
        cluster.  When no retriever is available the method falls back to a
        brute-force pairwise comparison.  Missing ROI information results in
        unweighted similarity scores so that the function remains best effort.
        """

        ids = list(workflows.keys())
        if not ids:
            return []

        vecs: Dict[str, List[float]] = {}
        roi_map: Dict[str, float] = {}
        for wid in ids:
            vecs[wid] = self.encode_workflow(wid, workflows[wid])
            roi_map[wid] = (
                _roi_weight_from_db(self.roi_db, wid) if self.roi_db is not None else 0.0
            )

        sims: Dict[str, Dict[str, float]] = {wid: {} for wid in ids}

        if retriever is None and Retriever is not None:
            try:  # pragma: no cover - best effort
                retriever = Retriever()
            except Exception:
                retriever = None

        if retriever is not None and Retriever is not None:
            try:  # pragma: no cover - optional path
                ur = retriever._get_retriever()
                for wid, vec in vecs.items():
                    hits, _, _ = ur.retrieve(
                        vec, top_k=len(ids) * 2, dbs=["workflow_meta"]
                    )  # type: ignore[attr-defined]
                    for hit in hits:
                        other = str(
                            getattr(hit, "record_id", None)
                            or getattr(getattr(hit, "metadata", {}), "get", lambda *_: None)("id")
                            or ""
                        )
                        if not other or other == wid or other not in vecs:
                            continue
                        sim = cosine_similarity(vec, vecs[other])
                        sim *= (1.0 + roi_map[wid]) * (1.0 + roi_map[other])
                        if sim > sims[wid].get(other, 0.0):
                            sims[wid][other] = sim
                            sims[other][wid] = sim
            except Exception:
                retriever = None

        if retriever is None:
            for i, wid in enumerate(ids):
                vec_a = vecs[wid]
                roi_a = roi_map[wid]
                for j in range(i + 1, len(ids)):
                    other = ids[j]
                    vec_b = vecs[other]
                    roi_b = roi_map[other]
                    sim = cosine_similarity(vec_a, vec_b)
                    sim *= (1.0 + roi_a) * (1.0 + roi_b)
                    sims[wid][other] = sim
                    sims[other][wid] = sim

        # Cluster based on thresholded similarities
        remaining = set(ids)
        clusters: List[List[str]] = []
        while remaining:
            wid = remaining.pop()
            cluster = [wid]
            for other in list(remaining):
                if sims[wid].get(other, 0.0) >= threshold:
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
        synergy_weight: float = 1.0,
        roi_weight: float = 1.0,
    ) -> List[str]:
        """Compose a workflow pipeline using embedding similarity.

        At each step the current workflow and every candidate are encoded via
        :meth:`encode_workflow`.  The cosine similarity is multiplied by
        ``(1 + ROI)`` for the candidate workflow with ROI obtained through
        :func:`_roi_weight_from_db`.  ``synergy_weight`` simply scales the
        similarity term while ``roi_weight`` adjusts the influence of the ROI
        multiplier.  The method stops once ``length`` steps have been selected
        or no compatible candidates remain.
        """

        if start not in workflows:
            return []

        pipeline = [start]
        available = {k for k in workflows.keys() if k != start}
        current = start
        graph = self.graph or WorkflowGraph()

        current_vec = self.encode_workflow(start, workflows[start])

        while available and len(pipeline) < length:
            best_id: str | None = None
            best_score = -1.0
            best_vec: List[float] | None = None
            for wid in list(available):
                if not _io_compatible(graph, current, wid):
                    continue
                cand_vec = self.encode_workflow(wid, workflows[wid])
                sim = cosine_similarity(current_vec, cand_vec)
                roi = (
                    _roi_weight_from_db(self.roi_db, wid)
                    if self.roi_db is not None
                    else 0.0
                )
                score = synergy_weight * sim * (1.0 + roi_weight * roi)
                if score > best_score:
                    best_id = wid
                    best_score = score
                    best_vec = cand_vec
            if best_id is None:
                break
            pipeline.append(best_id)
            available.remove(best_id)
            current = best_id
            if best_vec is not None:
                current_vec = best_vec

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
    def discover_and_persist(
        self,
        workflows: Mapping[str, Callable[[], Any]],
        *,
        metrics_db: Any | None = None,
    ) -> List[Dict[str, Any]]:
        """Discover meta-workflows and persist successful chains."""

        target = self.encode("self_improvement", {"workflow": []})
        records = self.plan_and_validate(target, workflows)

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
                    pass
            if metrics_db is not None:
                try:
                    metrics_db.log_eval(f"meta:{chain_id}", "roi_gain", float(roi_gain))
                except Exception:
                    pass
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
        # Track per-step entropy to allow granular convergence checks
        step_entropies: List[float] = []
        if WorkflowSynergyComparator is not None:
            for i in range(1, len(chain) + 1):
                try:
                    sub_spec = {"steps": [{"module": m} for m in chain[:i]]}
                    step_entropies.append(WorkflowSynergyComparator._entropy(sub_spec))
                except Exception:
                    step_entropies.append(0.0)
        else:
            step_entropies = [0.0] * len(chain)
        roi_gain = sum(
            float(m.result)
            for m in metrics.modules
            if isinstance(m.result, (int, float))
        )
        # Capture per-step metrics for cluster tracking
        step_metrics = [
            {
                "module": m.name,
                "roi": float(m.result) if isinstance(m.result, (int, float)) else 0.0,
                "failures": 0 if m.success else 1,
                "entropy": step_entropies[i] if i < len(step_entropies) else 0.0,
            }
            for i, m in enumerate(metrics.modules)
        ]

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

        self._update_cluster_map(
            chain,
            roi_gain,
            failures=failure_count,
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
                )
            except TypeError:  # pragma: no cover - compatibility shim
                persist_embedding("workflow_chain", chain_id, chain_vec)

        return {
            "chain": list(chain),
            "roi_gain": roi_gain,
            "failures": failure_count,
            "entropy": entropy,
            "step_metrics": step_metrics,
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
        """Mutate chains and re-validate.

        The original implementation only swapped the first two steps, trimmed
        the last element or appended an unused workflow.  For richer evolution
        we now support insertion, removal and substitution at arbitrary
        positions.  This keeps the search space intentionally small by only
        inserting or substituting with the first workflow identifier not
        already present in the chain.
        """

        wf_ids = list(workflows.keys())
        seen: set[tuple[str, ...]] = set()
        candidates: List[List[str]] = []

        for chain in chains:
            chain = list(chain)

            # Swap the first two steps when possible
            if len(chain) >= 2:
                swapped = chain[:]
                swapped[0], swapped[1] = swapped[1], swapped[0]
                candidates.append(swapped)

            # Remove each individual step
            for idx in range(len(chain)):
                removed = chain[:idx] + chain[idx + 1:]
                if removed:
                    tup = tuple(removed)
                    if tup not in seen:
                        seen.add(tup)
                        candidates.append(removed)

            # Insert a new workflow at every position
            for idx in range(len(chain) + 1):
                for wid in wf_ids:
                    if wid not in chain:
                        inserted = chain[:idx] + [wid] + chain[idx:]
                        tup = tuple(inserted)
                        if tup not in seen:
                            seen.add(tup)
                            candidates.append(inserted)
                        break

            # Substitute each step with an unused workflow
            for idx, current in enumerate(chain):
                for wid in wf_ids:
                    if wid != current and wid not in chain:
                        substituted = chain[:]
                        substituted[idx] = wid
                        tup = tuple(substituted)
                        if tup not in seen:
                            seen.add(tup)
                            candidates.append(substituted)
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
    def iterate_pipelines(
        self,
        workflows: Mapping[str, Callable[[], Any]],
        *,
        runner: "WorkflowSandboxRunner" | None = None,
        failure_threshold: int = 0,
        entropy_threshold: float = 2.0,
        roi_improvement_threshold: float = 0.0,
        entropy_stability_threshold: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Evolve converged pipelines based on ROI and entropy trends.

        ``cluster_map`` entries marked as converged are inspected.  Chains with
        non‑positive ROI deltas trigger :meth:`mutate_chains`.  Chains whose
        entropy exceeds ``entropy_stability_threshold`` are refined via
        :meth:`split_pipeline`.  Remaining converged chains are considered for
        re‑merging.  Winning variants are persisted for reinforcement and
        returned.
        """

        results: List[Dict[str, Any]] = []
        remerge: List[Sequence[str]] = []

        for chain, info in list(self.cluster_map.items()):
            if not info.get("converged"):
                continue
            chain_id = "->".join(chain)
            roi_delta = float(info.get("delta_roi", 0.0))
            entropy_val = 0.0
            if self.stability_db is not None:
                entry = self.stability_db.data.get(chain_id, {})
                entropy_val = float(entry.get("entropy", 0.0))

            if roi_delta <= roi_improvement_threshold:
                recs = self.mutate_chains(
                    [list(chain)],
                    workflows,
                    runner=runner,
                    failure_threshold=failure_threshold,
                    entropy_threshold=entropy_threshold,
                )
                if recs:
                    best = max(recs, key=lambda r: r.get("roi_gain", 0.0))
                    results.append(best)
                    self._reinforce(best)
            elif abs(entropy_val) > entropy_stability_threshold:
                recs = self.split_pipeline(
                    list(chain),
                    workflows,
                    roi_improvement_threshold=roi_improvement_threshold,
                    entropy_stability_threshold=entropy_stability_threshold,
                    runner=runner,
                    failure_threshold=failure_threshold,
                    entropy_threshold=entropy_threshold,
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
        failures = int(record.get("failures", 0))
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
                pass

        if self.stability_db is not None:
            try:
                self.stability_db.record_metrics(
                    chain_id, roi_gain, failures, entropy, roi_delta=roi_gain
                )
            except Exception:
                pass

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
        cluster_threshold: float = 0.75,
        runner: WorkflowSandboxRunner | None = None,
        failure_threshold: int = 0,
        entropy_threshold: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """Merge high-ROI variants using clustering results.

        ``records`` should contain ``chain`` and ``roi_gain`` entries.  Chains
        exceeding ``roi_threshold`` are clustered via :meth:`cluster_workflows`
        and each cluster is merged into a single pipeline containing the union
        of steps.  The merged pipelines are validated and successful results are
        returned.
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

        clusters = self.cluster_workflows(specs, threshold=cluster_threshold)

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
            )
            if rec:
                results.append(rec)
        return results

    # ------------------------------------------------------------------
    def _save_cluster_map(self) -> None:
        """Persist ``cluster_map`` to ``sandbox_data/meta_clusters.json``."""

        path = Path("sandbox_data/meta_clusters.json")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {"|".join(k): v for k, v in self.cluster_map.items()}
            with path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _load_cluster_map(self) -> None:
        """Load ``cluster_map`` from ``sandbox_data/meta_clusters.json`` if present."""

        path = Path("sandbox_data/meta_clusters.json")
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            self.cluster_map = {tuple(k.split("|")): v for k, v in data.items()}
        except Exception:
            self.cluster_map = {}

    # ------------------------------------------------------------------
    def _update_cluster_map(
        self,
        chain: Sequence[str],
        roi_gain: float,
        failures: int = 0,
        entropy: float = 0.0,
        *,
        step_metrics: Sequence[Mapping[str, Any]] | None = None,
        tol: float = 0.01,
    ) -> Dict[str, Any]:
        """Update metric histories for ``chain`` and detect convergence."""

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
            },
        )
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
            info["score"] = (roi_gain - failures - entropy) + (avg_roi - avg_fail - avg_ent)
        else:
            info["score"] = roi_gain - failures - entropy

        self._save_cluster_map()
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
    def _code_db_context(
        self, func: str
    ) -> tuple[List[str], List[str], float, float, List[float]]:
        if not self.code_db:
            return [], [], 0.0, 0.0, []
        try:
            rows = self.code_db.search(func)
        except Exception:
            return [], [], 0.0, 0.0, []
        mods: List[str] = []
        tags: List[str] = []
        depth = 0.0
        branching = 0.0
        curve: List[float] = []
        for r in rows[:1]:
            m = r.get("template_type")
            if m:
                mods.append(str(m))
            summary = r.get("summary") or ""
            if isinstance(summary, str):
                tags.extend(summary.split())
            ctags = r.get("context_tags") or r.get("tags") or []
            if isinstance(ctags, str):
                tags.extend(ctags.split())
            else:
                tags.extend(str(t) for t in ctags)
            try:
                depth = float(r.get("dependency_depth", 0.0) or 0.0)
            except Exception:
                depth = 0.0
            try:
                branching = float(r.get("branching_factor", 0.0) or 0.0)
            except Exception:
                branching = 0.0
            rc = r.get("roi_curve") or r.get("roi_curves") or []
            if isinstance(rc, str):
                try:
                    curve = [float(x) for x in json.loads(rc)]
                except Exception:
                    try:
                        curve = [float(x) for x in rc.split(",") if x]
                    except Exception:
                        curve = []
            elif isinstance(rc, Iterable):
                curve = [float(x) for x in rc]
        return mods, tags, depth, branching, curve

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
        self, workflow: Mapping[str, Any]
    ) -> tuple[
        List[str],
        List[str],
        List[str],
        List[str],
        List[str],
        List[float],
        List[float],
        List[List[float]],
    ]:
        steps = workflow.get("workflow") or workflow.get("task_sequence") or []
        funcs: List[str] = []
        modules: List[str] = []
        tags: List[str] = []
        module_cats: List[str] = []
        depths: List[float] = []
        branchings: List[float] = []
        curves: List[List[float]] = []
        if isinstance(workflow.get("category"), str):
            cat = str(workflow["category"])
            modules.append(cat)
            mc, mt = self._module_db_tags(cat)
            module_cats.extend(mc)
            tags.extend(mt)
        for step in steps:
            fn = None
            mod = None
            stags: Iterable[str] | str | None = []
            if isinstance(step, str):
                fn = step
            elif isinstance(step, Mapping):
                fn = step.get("function") or step.get("call") or step.get("name")
                mod = step.get("module") or step.get("category")
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
            if mod:
                mname = str(mod)
                modules.append(mname)
                mc, mt = self._module_db_tags(mname)
                module_cats.extend(mc)
                tags.extend(mt)
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
        )

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


def _load_chain_embeddings(path: Path = Path("embeddings.jsonl")) -> List[Dict[str, Any]]:
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
    retriever: Retriever,
    roi_db: ROIResultsDB | None = None,
    roi_window: int = 5,
) -> List[Dict[str, Any]]:
    """Return top ``top_k`` workflows similar to ``query`` weighted by ROI.

    ``query`` may be either a numeric embedding vector or an identifier of a
    stored workflow embedding.  ``retriever`` must be provided to fetch
    candidate workflow identifiers.  Results are ranked by cosine similarity
    scaled by the average ROI gain from ``roi_db``.  Retrieval failures yield an
    empty list.
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
        retr: Retriever | None = None
        if Retriever is not None:
            try:  # pragma: no cover - best effort
                retr = Retriever()
            except Exception:
                retr = None
        if retr is None:
            return []
        return find_synergy_candidates(
            workflow_id, top_k=top_k, retriever=retr, roi_db=roi_db
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
def find_synergy_chain(start_workflow_id: str, length: int = 5) -> List[str]:
    """Return high-synergy workflow sequence starting from ``start_workflow_id``."""

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

    try:  # pragma: no cover - optional dependency
        from workflow_synergy_comparator import WorkflowSynergyComparator  # type: ignore
    except Exception:  # pragma: no cover - when comparator unavailable
        WorkflowSynergyComparator = None  # type: ignore

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
            if wid and WorkflowSynergyComparator is not None:
                try:
                    entropy = WorkflowSynergyComparator._entropy(
                        {"steps": [{"module": wid}]}
                    )
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
    "find_synergy_chain",
    "find_synergy_candidates",
    "find_synergistic_workflows",
]
