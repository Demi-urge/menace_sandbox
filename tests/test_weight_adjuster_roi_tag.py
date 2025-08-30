import pytest
import types

from vector_service.weight_adjuster import WeightAdjuster, RoiTag
from vector_service.retriever import PatchRetriever, Retriever


class DummyVectorMetrics:
    def __init__(self):
        self.weights = {}
        self.vector_weights = {}

    def update_db_weight(self, origin, delta):
        weight = self.weights.get(origin, 1.0) + delta
        self.weights[origin] = weight
        return weight

    def update_vector_weight(self, vector_id, delta):
        weight = max(0.0, self.vector_weights.get(vector_id, 0.0) + delta)
        self.vector_weights[vector_id] = weight
        return weight

    def log_ranker_update(self, origin, delta, weight):  # pragma: no cover - optional
        self.last = (origin, delta, weight)


class DummyStore:
    def __init__(self, vectors, meta):
        self.vectors = vectors
        self.ids = [str(i) for i in range(len(vectors))]
        self.meta = meta

    def query(self, vector, top_k=5):
        return [(vid, 0.0) for vid in self.ids[:top_k]]


def test_roi_tag_positive_overrides_score():
    vm = DummyVectorMetrics()
    adj = WeightAdjuster(
        vector_metrics=vm,
        db_success_delta=0.2,
        db_failure_delta=0.2,
        vector_success_delta=0.2,
        vector_failure_delta=0.2,
    )
    adj.adjust([("db", "v1", 0.1, RoiTag.SUCCESS)])
    assert vm.weights["db"] == pytest.approx(1.02)
    assert vm.vector_weights["db:v1"] == pytest.approx(0.02)


def test_roi_tag_negative_overrides_score():
    vm = DummyVectorMetrics()
    adj = WeightAdjuster(
        vector_metrics=vm,
        db_success_delta=0.2,
        db_failure_delta=0.2,
        vector_success_delta=0.2,
        vector_failure_delta=0.2,
    )
    adj.adjust([("db", "v1", 0.9, RoiTag.BUG_INTRODUCED)])
    assert vm.weights["db"] == pytest.approx(0.82)
    assert vm.vector_weights["db:v1"] == pytest.approx(0.0)


def test_patch_retriever_applies_roi_tag_weights():
    vectors = [[1.0, 0.0], [1.0, 0.0]]
    meta = [
        {"origin_db": "patch", "metadata": {"text": "a", "roi_tag": "high-ROI", "enhancement_score": 0.0}},
        {"origin_db": "patch", "metadata": {"text": "b", "roi_tag": "bug-introduced", "enhancement_score": 0.0}},
    ]
    store = DummyStore(vectors, meta)

    def fake_vectorise(kind, record):
        return [1.0, 0.0]

    vec_service = types.SimpleNamespace(vectorise=fake_vectorise)
    pr = PatchRetriever(
        store=store,
        vector_service=vec_service,
        enhancement_weight=0.0,
        roi_tag_weights={"bug-introduced": 0.5, "high-ROI": -0.1},
    )
    results = pr.search("query", top_k=2)
    assert [r["record_id"] for r in results] == ["0", "1"]
    assert results[0]["roi_tag"] == "high-ROI"
    assert results[0]["score"] > results[1]["score"]


def test_parse_hits_applies_roi_tag_penalty(monkeypatch):
    r = Retriever(roi_tag_weights={"low-ROI": 0.4})

    monkeypatch.setattr(
        "vector_service.retriever.govern_retrieval",
        lambda text, metadata, reason=None, max_alert_severity=1.0: (metadata, reason),
    )
    monkeypatch.setattr(r.patch_safety, "evaluate", lambda *a, **k: (True, 0.0, {}))

    hit = types.SimpleNamespace(
        origin_db="patch",
        record_id="1",
        score=0.8,
        reason=None,
        metadata={"redacted": True, "roi_tag": "low-ROI"},
        text="",
    )
    results = r._parse_hits([hit])
    assert results[0]["roi_tag"] == "low-ROI"
    assert results[0]["score"] == pytest.approx(0.4)
