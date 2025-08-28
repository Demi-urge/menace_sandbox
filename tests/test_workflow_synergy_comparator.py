from collections import Counter
import math
import sys
import types
from pathlib import Path

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


def test_similarity_and_expandability(monkeypatch):
    _force_simple(monkeypatch)
    spec = {"steps": [{"module": "a"}, {"module": "b"}]}
    result = wsc.WorkflowSynergyComparator.compare(spec, spec)
    assert result.efficiency == pytest.approx(1.0)
    assert result.modularity == 1.0
    expected_entropy = compute_workflow_entropy(spec)
    assert result.expandability == expected_entropy


def test_shared_modules_detection(monkeypatch):
    _force_simple(monkeypatch)
    spec_a = {"steps": [{"module": "a"}, {"module": "b"}]}
    spec_b = {"steps": [{"module": "b"}, {"module": "c"}]}
    result = wsc.WorkflowSynergyComparator.compare(spec_a, spec_b)
    assert result.efficiency < 1.0
    assert result.modularity == 1 / 3
    ent_a = compute_workflow_entropy(spec_a)
    ent_b = compute_workflow_entropy(spec_b)
    assert result.expandability == (ent_a + ent_b) / 2
