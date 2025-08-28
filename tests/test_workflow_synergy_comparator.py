from collections import Counter
import json
import math
import sys
import types
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
pkg = types.ModuleType("menace_sandbox")
pkg.__path__ = [str(ROOT / "menace_sandbox")]
sys.modules.setdefault("menace_sandbox", pkg)


def _entropy(spec):
    if isinstance(spec, dict):
        steps = spec.get("steps", [])
    else:
        steps = list(spec)
    modules = [s.get("module") for s in steps if isinstance(s, dict) and s.get("module")]
    total = len(modules)
    if not total:
        return 0.0
    counts = Counter(modules)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

_stub = types.ModuleType("menace_sandbox.workflow_metrics")
_stub.compute_workflow_entropy = _entropy
_prev = sys.modules.get("menace_sandbox.workflow_metrics")
sys.modules["menace_sandbox.workflow_metrics"] = _stub

sys.modules.pop("menace_sandbox.workflow_synergy_comparator", None)
import menace_sandbox.workflow_synergy_comparator as wsc
from menace_sandbox.workflow_metrics import compute_workflow_entropy
import pytest

if _prev is not None:
    sys.modules["menace_sandbox.workflow_metrics"] = _prev
else:
    del sys.modules["menace_sandbox.workflow_metrics"]


def _force_simple(monkeypatch):
    def fake_embed(graph, spec):
        counts = {"a": 0, "b": 0, "c": 0}
        for step in spec.get("steps", []):
            mod = step.get("module")
            if mod in counts:
                counts[mod] += 1
        return [counts["a"], counts["b"], counts["c"]]

    monkeypatch.setattr(wsc.WorkflowSynergyComparator, "_embed_graph", staticmethod(fake_embed))
    monkeypatch.setattr(wsc, "_HAS_NX", False, raising=False)
    monkeypatch.setattr(wsc, "_HAS_NODE2VEC", False, raising=False)
    monkeypatch.setattr(wsc, "WorkflowVectorizer", None, raising=False)
    monkeypatch.setattr(
        wsc.WorkflowSynergyComparator,
        "_roi_and_modularity",
        classmethod(lambda cls, *_: (0.0, 0.0)),
        raising=False,
    )


FIX_DIR = Path(__file__).resolve().parent / "fixtures" / "workflows"


def _load(name: str) -> dict:
    return json.loads((FIX_DIR / name).read_text())


def test_similarity_and_entropy(monkeypatch):
    _force_simple(monkeypatch)
    spec = _load("simple_ab.json")
    result = wsc.WorkflowSynergyComparator.compare(spec, spec)
    assert result.similarity == pytest.approx(1.0)
    assert result.shared_module_ratio == pytest.approx(1.0)
    expected_entropy = compute_workflow_entropy(spec)
    assert result.entropy_a == expected_entropy
    assert result.entropy_b == expected_entropy


def test_shared_modules_detection(monkeypatch):
    _force_simple(monkeypatch)
    spec_a = _load("simple_ab.json")
    spec_b = _load("simple_bc.json")
    result = wsc.WorkflowSynergyComparator.compare(spec_a, spec_b)
    assert result.similarity < 1.0
    assert result.shared_module_ratio == pytest.approx(1 / 3)
    ent_a = compute_workflow_entropy(spec_a)
    ent_b = compute_workflow_entropy(spec_b)
    assert result.entropy_a == ent_a
    assert result.entropy_b == ent_b


def test_duplicate_detection_thresholds(monkeypatch):
    _force_simple(monkeypatch)
    spec_a = _load("simple_ab.json")
    spec_b = _load("simple_bc.json")
    assert wsc.WorkflowSynergyComparator.is_duplicate(
        spec_a, spec_a, {"similarity": 0.95, "entropy": 0.05}
    )

    assert not wsc.WorkflowSynergyComparator.is_duplicate(
        spec_a, spec_b, {"similarity": 0.95, "entropy": 0.05}
    )

    assert wsc.WorkflowSynergyComparator.is_duplicate(
        spec_a, spec_b, {"similarity": 0.49, "entropy": 0.2}
    )


def test_merge_duplicate(monkeypatch, tmp_path):
    _force_simple(monkeypatch)

    spec_a = _load("simple_ab.json")
    spec_b = _load("simple_bc.json")

    # ensure duplicate detection would trigger for identical specs
    assert wsc.WorkflowSynergyComparator.is_duplicate(spec_a, spec_a)

    base_id = "base"
    dup_id = "dup"
    base_file = tmp_path / f"{base_id}.workflow.json"
    dup_file = tmp_path / f"{dup_id}.workflow.json"
    base_file.write_text(json.dumps(spec_a))
    dup_file.write_text(json.dumps(spec_b))

    def fake_merge(base, a, b, out):
        data_a = json.loads(Path(a).read_text())
        data_b = json.loads(Path(b).read_text())
        merged = {
            "steps": data_a.get("steps", []) + data_b.get("steps", []),
            "metadata": {"workflow_id": "merged"},
        }
        out = Path(out)
        out.write_text(json.dumps(merged))
        return out

    monkeypatch.setattr(wsc.workflow_merger, "merge_workflows", fake_merge)

    out_path = wsc.merge_duplicate(base_id, dup_id, tmp_path)
    assert out_path is not None and out_path.exists()
    merged = json.loads(out_path.read_text())
    mods = [s["module"] for s in merged["steps"]]
    assert mods == [
        s["module"] for s in spec_a["steps"]
    ] + [s["module"] for s in spec_b["steps"]]
    assert merged.get("metadata", {}).get("workflow_id") == "merged"
    remaining = sorted(p.name for p in tmp_path.glob("*.json"))
    assert remaining == sorted(
        [
            f"{base_id}.workflow.json",
            f"{dup_id}.workflow.json",
            f"{base_id}.merged.json",
        ]
    )


def test_node2vec_branch(monkeypatch):
    calls = {"count": 0}

    class DummyGraph:
        def __init__(self, nodes: List[str]):
            self._nodes = nodes

        def nodes(self):
            return self._nodes

    class DummyNode2Vec:
        def __init__(self, graph, **kwargs):
            calls["count"] += 1
            self.graph = graph

        def fit(self, **kwargs):
            nodes = list(self.graph.nodes())
            wv = {str(n): [float(i + 1)] for i, n in enumerate(nodes)}
            return types.SimpleNamespace(wv=wv)

    spec = {"steps": [{"module": "x"}, {"module": "y"}]}
    graph = DummyGraph(["x", "y"])

    wsc.WorkflowSynergyComparator._embed_cache.clear()
    monkeypatch.setattr(wsc, "_HAS_NX", True, raising=False)
    monkeypatch.setattr(wsc, "_HAS_NODE2VEC", True, raising=False)
    monkeypatch.setattr(wsc, "Node2Vec", DummyNode2Vec, raising=False)
    monkeypatch.setattr(wsc, "nx", types.SimpleNamespace(Graph=DummyGraph), raising=False)

    result1 = wsc.WorkflowSynergyComparator._embed_graph(graph, spec)
    result2 = wsc.WorkflowSynergyComparator._embed_graph(graph, spec)
    assert result1 == [1.0, 2.0]
    assert result2 == result1
    assert calls["count"] == 1


def test_efficiency_and_modularity(monkeypatch):
    """ROITracker and networkx community metrics feed efficiency/modularity."""

    def fake_embed(graph, spec):
        counts = {"a": 0, "b": 0, "c": 0}
        for step in spec.get("steps", []):
            mod = step.get("module")
            if mod in counts:
                counts[mod] += 1
        return [counts["a"], counts["b"], counts["c"]]

    monkeypatch.setattr(wsc.WorkflowSynergyComparator, "_embed_graph", staticmethod(fake_embed))
    monkeypatch.setattr(wsc, "_HAS_NODE2VEC", False, raising=False)
    monkeypatch.setattr(wsc, "WorkflowVectorizer", None, raising=False)

    class DummyTracker:
        def __init__(self):
            self.metrics_history = {"synergy_efficiency": [0.8]}

    monkeypatch.setattr(wsc, "ROITracker", DummyTracker, raising=False)

    class DummyGraph:
        def add_nodes_from(self, nodes):
            pass

        def add_edges_from(self, edges):
            pass

        def to_undirected(self):
            return self

    def greedy_modularity_communities(g):
        return [{1}, {2}]

    def modularity_func(g, communities):
        return 0.5

    dummy_nx = types.SimpleNamespace(
        DiGraph=DummyGraph,
        Graph=DummyGraph,
        algorithms=types.SimpleNamespace(
            community=types.SimpleNamespace(
                greedy_modularity_communities=greedy_modularity_communities,
                modularity=modularity_func,
            )
        ),
    )

    monkeypatch.setattr(wsc, "_HAS_NX", True, raising=False)
    monkeypatch.setattr(wsc, "nx", dummy_nx, raising=False)

    spec_a = _load("simple_ab.json")
    spec_b = _load("simple_bc.json")

    result = wsc.WorkflowSynergyComparator.compare(spec_a, spec_b)
    ent_a = compute_workflow_entropy(spec_a)
    ent_b = compute_workflow_entropy(spec_b)
    expandability = (ent_a + ent_b) / 2

    assert result.efficiency == pytest.approx(0.8)
    assert result.modularity == pytest.approx(0.5)
    expected_agg = (0.8 + 0.5 + expandability) / 3
    assert result.aggregate == pytest.approx(expected_agg)
