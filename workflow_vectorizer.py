from __future__ import annotations

"""Vectorisation utilities for workflow specifications."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping
import json

try:  # pragma: no cover - optional database
    from code_database import CodeDB  # type: ignore
except Exception:  # pragma: no cover - graceful fallback
    CodeDB = None  # type: ignore

try:  # pragma: no cover - optional graph dependency
    from workflow_graph import _HAS_NX  # type: ignore
except Exception:  # pragma: no cover - gracefully handle missing graph
    _HAS_NX = False  # type: ignore
try:  # pragma: no cover - networkx may be heavy
    if _HAS_NX:
        import networkx as nx  # type: ignore
    else:  # pragma: no cover - executed when networkx missing
        nx = None  # type: ignore
except Exception:  # pragma: no cover - networkx not available
    nx = None  # type: ignore
    _HAS_NX = False  # type: ignore

try:  # pragma: no cover - optional vector service
    from vector_service.vectorizer import SharedVectorService  # type: ignore
except Exception:  # pragma: no cover - graceful fallback if service unavailable
    SharedVectorService = None  # type: ignore
from vector_utils import persist_embedding
from dynamic_path_router import resolve_path

_DEFAULT_BOUNDS = {
    "num_steps": 20.0,
    "duration": 10_000.0,
    "estimated_profit": 1_000_000.0,
    "depth": 100.0,
    "branching": 50.0,
    "roi": 10_000.0,
}


def _one_hot(idx: int, length: int) -> List[float]:
    vec = [0.0] * length
    if 0 <= idx < length:
        vec[idx] = 1.0
    return vec


def _get_index(value: Any, mapping: Dict[str, int], max_size: int) -> int:
    val = str(value).lower().strip() or "other"
    if val in mapping:
        return mapping[val]
    if len(mapping) < max_size:
        mapping[val] = len(mapping)
        return mapping[val]
    return mapping["other"]


def _scale(value: Any, bound: float) -> float:
    try:
        f = float(value)
    except Exception:
        return 0.0
    f = max(-bound, min(bound, f))
    return f / bound if bound else 0.0


@dataclass
class WorkflowVectorizer:
    """Encode workflows with structural and semantic context."""

    max_categories: int = 20
    max_status: int = 10
    max_functions: int = 50
    max_modules: int = 50
    max_tags: int = 50
    roi_window: int = 5
    category_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})
    status_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})
    function_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})
    module_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})
    tag_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})
    graph: Any | None = None
    code_db: CodeDB | None = None
    _last_metrics: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.code_db is None and CodeDB is not None:
            try:
                self.code_db = CodeDB()
            except Exception:
                self.code_db = None

    def fit(self, workflows: Iterable[Dict[str, Any]]) -> "WorkflowVectorizer":
        for wf in workflows:
            _get_index(wf.get("category"), self.category_index, self.max_categories)
            _get_index(wf.get("status"), self.status_index, self.max_status)
            funcs, mods, tags, _, _, _ = self._semantic_tokens(wf)
            wid = wf.get("workflow_id") or wf.get("id") or wf.get("record_id")
            if wid is not None:
                gmods, gtags = self._graph_semantics(str(wid))
                mods.extend(gmods)
                tags.extend(gtags)
            for tok in funcs:
                _get_index(tok, self.function_index, self.max_functions)
            for tok in mods:
                _get_index(tok, self.module_index, self.max_modules)
            for tok in tags:
                _get_index(tok, self.tag_index, self.max_tags)
        return self

    @property
    def dim(self) -> int:
        return (
            6
            + self.roi_window
            + self.max_categories
            + self.max_status
            + self.max_functions
            + self.max_modules
            + self.max_tags
        )

    def graph_metrics(self) -> Dict[str, Any]:
        return self._last_metrics

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

    def _semantic_tokens(
        self, workflow: Mapping[str, Any]
    ) -> tuple[
        List[str],
        List[str],
        List[str],
        List[float],
        List[float],
        List[List[float]],
    ]:
        steps = workflow.get("workflow") or workflow.get("task_sequence") or []
        funcs: List[str] = []
        mods: List[str] = []
        tags: List[str] = []
        depths: List[float] = []
        branchings: List[float] = []
        curves: List[List[float]] = []
        if isinstance(workflow.get("category"), str):
            mods.append(str(workflow["category"]))
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
                mods.extend(cmods)
                tags.extend(ctags)
                depths.append(d)
                branchings.append(b)
                curves.append(curve)
            if mod:
                mods.append(str(mod))
            if isinstance(stags, str):
                tags.append(stags)
            else:
                tags.extend(str(t) for t in stags)
        return funcs, mods, tags, depths, branchings, curves

    def _graph_semantics(self, workflow_id: str) -> tuple[List[str], List[str]]:
        mods: List[str] = []
        tags: List[str] = []
        g = getattr(self.graph, "graph", None) if self.graph is not None else None
        if g is None:
            return mods, tags
        try:
            if _HAS_NX and nx is not None:
                data = g.nodes[workflow_id] if g.has_node(workflow_id) else {}
            else:
                data = g.get("nodes", {}).get(workflow_id, {})
        except Exception:
            data = {}
        mod = data.get("module") or data.get("category")
        if mod:
            mods.append(str(mod))
        gt = data.get("context_tags") or data.get("tags") or []
        if isinstance(gt, str):
            tags.append(gt)
        else:
            tags.extend(str(t) for t in gt)
        return mods, tags

    def _graph_features(self, workflow_id: str) -> tuple[float, float]:
        depth = 0.0
        branching = 0.0
        g = getattr(self.graph, "graph", None) if self.graph is not None else None
        if g is None:
            return depth, branching
        try:
            if _HAS_NX and nx is not None and hasattr(g, "out_degree"):
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

    def _code_db_modules(self, func: str) -> List[str]:
        if not self.code_db:
            return []
        try:
            rows = self.code_db.search(func)
        except Exception:
            return []
        mods = []
        for r in rows[:1]:
            m = r.get("template_type")
            if m:
                mods.append(str(m))
        return mods

    def _code_db_tags(self, func: str) -> List[str]:
        if not self.code_db:
            return []
        try:
            rows = self.code_db.search(func)
        except Exception:
            return []
        tags: List[str] = []
        for r in rows[:1]:
            summary = r.get("summary") or ""
            if isinstance(summary, str):
                tags.extend(summary.split())
        return tags

    def _encode_tokens(
        self, tokens: Iterable[str], mapping: Dict[str, int], max_size: int
    ) -> List[float]:
        vec = [0.0] * max_size
        for tok in {t.lower().strip() for t in tokens if t}:
            idx = _get_index(tok, mapping, max_size)
            if idx < max_size:
                vec[idx] = 1.0
        return vec

    def transform(self, wf: Dict[str, Any], workflow_id: str | None = None) -> List[float]:
        wid = workflow_id or wf.get("workflow_id") or wf.get("id") or wf.get("record_id")
        c_idx = _get_index(wf.get("category"), self.category_index, self.max_categories)
        s_idx = _get_index(wf.get("status"), self.status_index, self.max_status)
        steps = wf.get("workflow") or wf.get("task_sequence") or []
        funcs, mods, tags, depths, branchings, curves = self._semantic_tokens(wf)
        if wid is not None:
            gmods, gtags = self._graph_semantics(str(wid))
            mods.extend(gmods)
            tags.extend(gtags)
        depth, branching = self._graph_features(str(wid)) if wid is not None else (0.0, 0.0)
        if depth == 0.0 and depths:
            depth = max(depths)
        if branching == 0.0 and branchings:
            branching = max(branchings)
        agg_curve = [0.0] * self.roi_window
        if curves:
            for curve in curves:
                for i, val in enumerate(curve[: self.roi_window]):
                    agg_curve[i] += float(val)
            agg_curve = [v / len(curves) for v in agg_curve]
        roi = agg_curve[-1] if agg_curve else 0.0
        self._last_metrics = {
            "dependency_depth": depth,
            "branching_factor": branching,
            "roi_curve": agg_curve,
        }
        struct_vec: List[float] = []
        struct_vec.append(_scale(len(steps), _DEFAULT_BOUNDS["num_steps"]))
        struct_vec.append(
            _scale(wf.get("workflow_duration", 0.0), _DEFAULT_BOUNDS["duration"]))
        struct_vec.append(
            _scale(
                wf.get("estimated_profit_per_bot", 0.0),
                _DEFAULT_BOUNDS["estimated_profit"],
            )
        )
        struct_vec.append(_scale(depth, _DEFAULT_BOUNDS["depth"]))
        struct_vec.append(_scale(branching, _DEFAULT_BOUNDS["branching"]))
        struct_vec.append(_scale(roi, _DEFAULT_BOUNDS["roi"]))
        for val in agg_curve:
            struct_vec.append(_scale(val, _DEFAULT_BOUNDS["roi"]))

        sem_vec: List[float] = []
        sem_vec.extend(_one_hot(c_idx, self.max_categories))
        sem_vec.extend(_one_hot(s_idx, self.max_status))
        sem_vec.extend(self._encode_tokens(funcs, self.function_index, self.max_functions))
        sem_vec.extend(self._encode_tokens(mods, self.module_index, self.max_modules))
        sem_vec.extend(self._encode_tokens(tags, self.tag_index, self.max_tags))
        return struct_vec + sem_vec


try:
    _EMBEDDINGS_PATH = resolve_path("embeddings.jsonl").as_posix()
except FileNotFoundError:
    _EMBEDDINGS_PATH = (resolve_path(".") / "embeddings.jsonl").as_posix()


_DEFAULT_VECTORIZER = WorkflowVectorizer()
_DEFAULT_SERVICE = None
if SharedVectorService is not None:  # pragma: no cover - best effort
    try:
        _DEFAULT_SERVICE = SharedVectorService()
    except Exception:
        _DEFAULT_SERVICE = None


def vectorize_and_store(
    record_id: str,
    workflow: Dict[str, Any],
    *,
    path: str = _EMBEDDINGS_PATH,
    origin_db: str = "workflow",
    metadata: Dict[str, Any] | None = None,
) -> List[float]:
    """Vectorise ``workflow`` and persist the embedding."""

    vec = _DEFAULT_VECTORIZER.transform(workflow, workflow_id=record_id)
    meta = {
        **(metadata or {}),
        **_DEFAULT_VECTORIZER.graph_metrics(),
    }
    if _DEFAULT_SERVICE is None:  # pragma: no cover - dependency unavailable
        raise RuntimeError("SharedVectorService unavailable")
    _DEFAULT_SERVICE.vectorise_and_store(
        "workflow",
        record_id,
        workflow,
        origin_db=origin_db,
        metadata=meta,
    )
    return vec


def persist_workflow_embedding(
    record_id: str,
    workflow: Dict[str, Any],
    *,
    path: str = _EMBEDDINGS_PATH,
) -> List[float]:
    """Vectorise ``workflow`` and persist the embedding with metadata."""

    vec = _DEFAULT_VECTORIZER.transform(workflow, workflow_id=record_id)
    meta: Dict[str, Any] = {**_DEFAULT_VECTORIZER.graph_metrics()}
    for key in ("roi", "roi_curve", "failures", "failure_rate", "failure_reason"):
        if key in workflow:
            meta[key] = workflow[key]
    persist_embedding(
        "workflow",
        record_id,
        vec,
        origin_db="workflow",
        metadata=meta,
        path=path,
    )
    return vec


__all__ = ["WorkflowVectorizer", "vectorize_and_store", "persist_workflow_embedding"]
