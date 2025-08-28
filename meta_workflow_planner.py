from __future__ import annotations

"""Meta workflow planning utilities with semantic and structural embedding."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from roi_results_db import ROIResultsDB
from workflow_graph import WorkflowGraph
from roi_tracker import ROITracker
from vector_utils import persist_embedding, cosine_similarity

try:  # pragma: no cover - optional heavy dependency
    import networkx as nx  # type: ignore
    _HAS_NX = True
except Exception:  # pragma: no cover - executed when networkx missing
    nx = None  # type: ignore
    _HAS_NX = False

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
        curve = [float(rec.get("roi_gain", 0.0)) for rec in history[-self.roi_window :]]
        while len(curve) < self.roi_window:
            curve.append(0.0)
        return curve

    # ------------------------------------------------------------------
    def _semantic_tokens(self, workflow: Mapping[str, Any]) -> tuple[List[str], List[str], List[str]]:
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


__all__ = ["MetaWorkflowPlanner", "find_synergy_chain"]
