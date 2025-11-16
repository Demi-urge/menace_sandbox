import pytest
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workflow_synergy_comparator import WorkflowSynergyComparator as WSC


def test_embed_spec_aligns_on_union():
    spec_a = {"steps": [{"module": "alpha"}, {"module": "beta"}]}
    spec_b = {"steps": [{"module": "beta"}, {"module": "gamma"}]}
    union = sorted(set(WSC._extract_modules(spec_a)) | set(WSC._extract_modules(spec_b)))
    vec_a = WSC._embed_spec(spec_a, union)
    vec_b = WSC._embed_spec(spec_b, union)
    assert len(vec_a) == len(vec_b) == len(union) * (len(vec_a) // len(union) if union else 0)
    dim = len(vec_a) // len(union)
    assert vec_a[:dim] != [0.0] * dim
    assert vec_b[:dim] == [0.0] * dim
    assert vec_a[dim:2*dim] != [0.0] * dim
    assert vec_b[dim:2*dim] != [0.0] * dim
    assert vec_a[2*dim:3*dim] == [0.0] * dim
    assert vec_b[2*dim:3*dim] != [0.0] * dim


def test_node2vec_used_when_available(monkeypatch):
    import workflow_synergy_comparator as wsc
    if not wsc._HAS_NODE2VEC:
        pytest.skip("node2vec not installed")

    created = {}

    class DummyModel:
        def __init__(self):
            self.wv = {"a": [1.0], "b": [2.0]}

    class DummyN2V:
        def __init__(self, *a, **k):
            created["called"] = True

        def fit(self, *a, **k):
            return DummyModel()

    monkeypatch.setattr(wsc, "Node2Vec", DummyN2V)
    spec = {"steps": [{"module": "a"}, {"module": "b"}]}
    vec = wsc.WorkflowSynergyComparator._embed_spec(spec)
    assert created.get("called")
    assert vec == [1.0, 2.0]
