from __future__ import annotations

"""Vectorisation utilities for workflow specifications."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping

try:  # pragma: no cover - optional heavy deps
    import networkx as nx  # type: ignore
    _HAS_NX = True
except Exception:  # pragma: no cover - executed when networkx missing
    nx = None  # type: ignore
    _HAS_NX = False

try:  # pragma: no cover - optional database
    from workflow_graph import WorkflowGraph  # type: ignore
except Exception:  # pragma: no cover - graceful fallback
    WorkflowGraph = None  # type: ignore

try:  # pragma: no cover - optional database
    from code_database import CodeDB  # type: ignore
except Exception:  # pragma: no cover - graceful fallback
    CodeDB = None  # type: ignore

from vector_utils import persist_embedding

_DEFAULT_BOUNDS = {
    "num_steps": 20.0,
    "duration": 10_000.0,
    "estimated_profit": 1_000_000.0,
    "depth": 10.0,
    "branching": 10.0,
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
    category_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})
    status_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})
    function_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})
    module_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})
    tag_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})
    graph: WorkflowGraph | None = None
    code_db: CodeDB | None = None
    _last_metrics: Dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.graph is None and WorkflowGraph is not None:
            try:
                self.graph = WorkflowGraph()
            except Exception:
                self.graph = None
        if self.code_db is None and CodeDB is not None:
            try:
                self.code_db = CodeDB()
            except Exception:
                self.code_db = None

    def fit(self, workflows: Iterable[Dict[str, Any]]) -> "WorkflowVectorizer":
        for wf in workflows:
            _get_index(wf.get("category"), self.category_index, self.max_categories)
            _get_index(wf.get("status"), self.status_index, self.max_status)
            funcs, mods, tags = self._semantic_tokens(wf)
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
            self.max_categories
            + self.max_status
            + 6
            + self.max_functions
            + self.max_modules
            + self.max_tags
        )

    # ------------------------------------------------------------------
    def _graph_features(self, workflow_id: str) -> tuple[float, float, float]:
        depth = 0.0
        branching = 0.0
        roi = 0.0
        g = getattr(self.graph, "graph", None)
        if not workflow_id or g is None:
            return depth, branching, roi
        try:
            if _HAS_NX and hasattr(g, "out_degree"):
                branching = float(g.out_degree(workflow_id)) if g.has_node(workflow_id) else 0.0
                if g.has_node(workflow_id):
                    roi = float(g.nodes[workflow_id].get("roi", 0.0) or 0.0)
                    ancestors = nx.ancestors(g, workflow_id)
                    if ancestors:
                        depth = max(
                            nx.shortest_path_length(g, anc, workflow_id) for anc in ancestors
                        )
        except Exception:
            depth = 0.0
            branching = 0.0
            roi = 0.0
        return depth, branching, roi

    def graph_metrics(self) -> Dict[str, float]:
        return self._last_metrics

    # ------------------------------------------------------------------
    def _semantic_tokens(self, workflow: Mapping[str, Any]) -> tuple[List[str], List[str], List[str]]:
        steps = workflow.get("workflow") or workflow.get("task_sequence") or []
        funcs: List[str] = []
        mods: List[str] = []
        tags: List[str] = []
        if isinstance(workflow.get("category"), str):
            mods.append(str(workflow["category"]))
        for step in steps:
            if isinstance(step, str):
                fname = step
                funcs.append(fname)
                mods.extend(self._code_db_modules(fname))
                tags.extend(self._code_db_tags(fname))
            elif isinstance(step, Mapping):
                fn = step.get("function") or step.get("call") or step.get("name")
                if fn:
                    fname = str(fn)
                    funcs.append(fname)
                    mods.extend(self._code_db_modules(fname))
                    tags.extend(self._code_db_tags(fname))
                mod = step.get("module") or step.get("category")
                if mod:
                    mods.append(str(mod))
                stags = step.get("context_tags") or step.get("tags") or []
                if isinstance(stags, str):
                    tags.append(stags)
                else:
                    tags.extend(str(t) for t in stags)
        return funcs, mods, tags

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
        wid = workflow_id or str(
            wf.get("workflow_id") or wf.get("id") or wf.get("record_id") or ""
        )
        c_idx = _get_index(wf.get("category"), self.category_index, self.max_categories)
        s_idx = _get_index(wf.get("status"), self.status_index, self.max_status)
        steps = wf.get("workflow") or wf.get("task_sequence") or []
        depth, branching, roi = self._graph_features(wid)
        self._last_metrics = {
            "dependency_depth": depth,
            "branching_factor": branching,
            "roi": roi,
        }
        funcs, mods, tags = self._semantic_tokens(wf)
        vec: List[float] = []
        vec.extend(_one_hot(c_idx, self.max_categories))
        vec.extend(_one_hot(s_idx, self.max_status))
        vec.append(_scale(len(steps), _DEFAULT_BOUNDS["num_steps"]))
        vec.append(_scale(wf.get("workflow_duration", 0.0), _DEFAULT_BOUNDS["duration"]))
        vec.append(
            _scale(
                wf.get("estimated_profit_per_bot", 0.0),
                _DEFAULT_BOUNDS["estimated_profit"],
            )
        )
        vec.append(_scale(depth, _DEFAULT_BOUNDS["depth"]))
        vec.append(_scale(branching, _DEFAULT_BOUNDS["branching"]))
        vec.append(_scale(roi, _DEFAULT_BOUNDS["roi"]))
        vec.extend(self._encode_tokens(funcs, self.function_index, self.max_functions))
        vec.extend(self._encode_tokens(mods, self.module_index, self.max_modules))
        vec.extend(self._encode_tokens(tags, self.tag_index, self.max_tags))
        return vec


_DEFAULT_VECTORIZER = WorkflowVectorizer()


def vectorize_and_store(
    record_id: str,
    workflow: Dict[str, Any],
    *,
    path: str = "embeddings.jsonl",
    origin_db: str = "workflow",
    metadata: Dict[str, Any] | None = None,
) -> List[float]:
    """Vectorise ``workflow`` and persist the embedding."""

    vec = _DEFAULT_VECTORIZER.transform(workflow, workflow_id=record_id)
    meta = {
        **(metadata or {}),
        **_DEFAULT_VECTORIZER.graph_metrics(),
    }
    try:
        persist_embedding(
            "workflow",
            record_id,
            vec,
            path=path,
            origin_db=origin_db,
            metadata=meta,
        )
    except TypeError:  # pragma: no cover - compatibility with older signatures
        persist_embedding("workflow", record_id, vec, path=path)
    return vec

__all__ = ["WorkflowVectorizer", "vectorize_and_store"]
