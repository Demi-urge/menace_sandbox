from collections import Counter
import json
import math
import sys
import types
from pathlib import Path
from typing import List

import networkx as nx

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
import menace_sandbox.workflow_synergy_comparator as wsc  # noqa: E402
from menace_sandbox.workflow_metrics import compute_workflow_entropy  # noqa: E402
import pytest  # noqa: E402

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
    monkeypatch.setattr(
        wsc.WorkflowSynergyComparator,
        "best_practices_file",
        Path("/tmp/wsc_best.json"),
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

    scores_same = wsc.WorkflowSynergyComparator.compare(spec_a, spec_a)
    assert wsc.WorkflowSynergyComparator.is_duplicate(scores_same)

    scores_diff = wsc.WorkflowSynergyComparator.compare(spec_a, spec_b)
    assert not wsc.WorkflowSynergyComparator.is_duplicate(scores_diff)

    assert wsc.WorkflowSynergyComparator.is_duplicate(
        scores_diff, similarity_threshold=0.49, entropy_threshold=0.2
    )

    # direct specification invocation
    assert wsc.WorkflowSynergyComparator.is_duplicate(
        spec_a, spec_a, similarity_threshold=0.95, entropy_threshold=0.05
    )
    assert not wsc.WorkflowSynergyComparator.is_duplicate(
        spec_a, spec_b, similarity_threshold=0.95, entropy_threshold=0.05
    )


def test_merge_duplicate(monkeypatch, tmp_path):
    _force_simple(monkeypatch)

    spec_a = _load("simple_ab.json")
    spec_b = _load("simple_bc.json")

    # ensure duplicate detection would trigger for identical specs
    scores_same = wsc.WorkflowSynergyComparator.compare(spec_a, spec_a)
    assert wsc.WorkflowSynergyComparator.is_duplicate(scores_same)

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

    calls = {"count": 0}

    class StubSummary:
        def save_all_summaries(self, directory):
            calls["count"] += 1
            calls["dir"] = Path(directory)

    monkeypatch.setattr(wsc, "workflow_run_summary", StubSummary(), raising=False)

    out_path = wsc.merge_duplicate(base_id, dup_id, tmp_path)
    assert out_path is not None and out_path.exists()
    assert calls["count"] == 1
    assert calls["dir"] == tmp_path
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


def test_merge_duplicate_missing_files(monkeypatch, tmp_path):
    _force_simple(monkeypatch)

    base_id = "base"
    dup_id = "dup"

    def fail_merge(*args, **kwargs):
        raise AssertionError("merge_workflows should not be called")

    monkeypatch.setattr(wsc.workflow_merger, "merge_workflows", fail_merge)
    calls = {"count": 0}

    class StubSummary:
        def save_all_summaries(self, *a, **k):
            calls["count"] += 1

    monkeypatch.setattr(wsc, "workflow_run_summary", StubSummary(), raising=False)

    # Neither file exists
    assert wsc.merge_duplicate(base_id, dup_id, tmp_path) is None
    assert calls["count"] == 0

    # Only base exists
    (tmp_path / f"{base_id}.workflow.json").write_text("{}")
    assert wsc.merge_duplicate(base_id, dup_id, tmp_path) is None
    assert calls["count"] == 0


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


def test_spectral_fallback_cache(monkeypatch):
    wsc.WorkflowSynergyComparator._embed_cache.clear()
    spec = {"steps": [{"module": "a"}, {"module": "b"}]}
    g = nx.DiGraph()
    g.add_edge("a", "b")

    monkeypatch.setattr(wsc, "_HAS_NODE2VEC", False, raising=False)
    called = {"count": 0}
    orig = nx.to_numpy_array

    def counting_to_numpy_array(*args, **kwargs):
        called["count"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(wsc.nx, "to_numpy_array", counting_to_numpy_array)

    result1 = wsc.WorkflowSynergyComparator._embed_graph(g, spec)
    result2 = wsc.WorkflowSynergyComparator._embed_graph(g, spec)
    assert result1 == result2
    assert called["count"] == 1


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

    similarity = result.similarity
    shared = result.shared_module_ratio

    assert result.efficiency == pytest.approx(0.8)
    assert result.modularity == pytest.approx(0.5)
    expected_agg = (similarity + shared + expandability + 0.8 + 0.5) / 5
    assert result.aggregate == pytest.approx(expected_agg)


def test_weighted_aggregate(monkeypatch):
    """Weights exclude similarity/entropy from aggregate."""

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

    weights = {"similarity": 0.0, "shared_modules": 0.0, "entropy": 0.0}
    result = wsc.WorkflowSynergyComparator.compare(spec_a, spec_b, weights=weights)

    assert result.aggregate == pytest.approx((0.8 + 0.5) / 2)


def test_analyze_overfitting(monkeypatch, tmp_path):
    _force_simple(monkeypatch)
    # Directly patch repository path for isolation
    monkeypatch.setattr(
        wsc.WorkflowSynergyComparator,
        "best_practices_file",
        tmp_path / "best.json",
        raising=False,
    )

    # Spec with obvious repetition and low entropy
    over_spec = {"steps": [{"module": "a"}, {"module": "a"}, {"module": "a"}]}
    report = wsc.WorkflowSynergyComparator.analyze_overfitting(
        over_spec, entropy_threshold=1.5, repeat_threshold=2
    )
    assert report.low_entropy
    assert "a" in report.repeated_modules

    # Balanced spec should be added to best practices repository
    good_spec = {"steps": [{"module": "a"}, {"module": "b"}]}
    good_report = wsc.WorkflowSynergyComparator.analyze_overfitting(
        good_spec, entropy_threshold=0.5, repeat_threshold=2
    )
    assert not good_report.low_entropy
    assert good_report.repeated_modules == {}
    data = json.loads((tmp_path / "best.json").read_text())
    assert ["a", "b"] in data.get("sequences", [])


def test_embedding_path_resolution_and_cache(monkeypatch, tmp_path):
    """Workflow identifiers resolve to files and share cached embeddings."""
    monkeypatch.setattr(wsc, "_HAS_NX", False, raising=False)
    monkeypatch.setattr(wsc, "_HAS_NODE2VEC", False, raising=False)
    monkeypatch.setattr(wsc, "WorkflowVectorizer", None, raising=False)
    monkeypatch.setattr(
        wsc.WorkflowSynergyComparator,
        "_roi_and_modularity",
        classmethod(lambda cls, *_: (0.0, 0.0)),
        raising=False,
    )
    monkeypatch.setattr(
        wsc.WorkflowSynergyComparator,
        "workflow_dir",
        tmp_path,
        raising=False,
    )
    monkeypatch.setattr(
        wsc.WorkflowSynergyComparator,
        "best_practices_file",
        tmp_path / "best.json",
        raising=False,
    )

    spec = {"steps": [{"module": "a"}, {"module": "b"}]}
    for name in ("alpha", "beta"):
        (tmp_path / f"{name}.workflow.json").write_text(json.dumps(spec))

    wsc.WorkflowSynergyComparator._embed_cache.clear()
    result = wsc.WorkflowSynergyComparator.compare("alpha", "beta")
    assert result.similarity == pytest.approx(1.0)
    assert len(wsc.WorkflowSynergyComparator._embed_cache) == 1


def test_merge_duplicate_classmethod_delegates(monkeypatch, tmp_path):
    """Class method forwards to module level ``merge_duplicate``."""

    called = {}

    def fake_merge(base, dup, directory):
        called["args"] = (base, dup, directory)
        out = Path(directory) / "merged.json"
        out.write_text("{}")
        return out

    monkeypatch.setattr(wsc, "merge_duplicate", fake_merge)

    out_path = wsc.WorkflowSynergyComparator.merge_duplicate("a", "b", tmp_path)
    assert out_path == tmp_path / "merged.json"
    assert called["args"] == ("a", "b", tmp_path)


def test_compare_returns_overfitting_report(monkeypatch):
    """``compare`` attaches overfitting reports to its result."""
    _force_simple(monkeypatch)
    spec_a = {"steps": [{"module": "a"}, {"module": "a"}, {"module": "a"}]}
    spec_b = {"steps": [{"module": "a"}, {"module": "b"}]}

    scores = wsc.WorkflowSynergyComparator.compare(spec_a, spec_b)
    assert scores.overfit_a and scores.overfit_a.is_overfitting()
    assert scores.overfit_b and not scores.overfit_b.is_overfitting()
