import pytest

from vector_metrics_db import VectorMetricsDB
from vector_service.context_builder import ContextBuilder
from vector_service.cognition_layer import CognitionLayer
from vector_service.patch_logger import PatchLogger


class DummyRetriever:
    def search(self, *_, **__):
        return []


class DummyTracker:
    def __init__(self, deltas):
        self._deltas = deltas
        self.calls = 0

    def origin_db_deltas(self):
        self.calls += 1
        return self._deltas


def _make_layer(tmp_path, tracker):
    vec_db = VectorMetricsDB(tmp_path / "vec.db")
    builder = ContextBuilder(retriever=DummyRetriever(), roi_tracker=tracker)
    patch_logger = PatchLogger(vector_metrics=vec_db, roi_tracker=tracker)
    layer = CognitionLayer(
        retriever=DummyRetriever(),
        context_builder=builder,
        patch_logger=patch_logger,
        vector_metrics=vec_db,
        roi_tracker=tracker,
    )
    return layer, vec_db, builder


def test_update_ranker_applies_tracker_deltas(tmp_path):
    tracker = DummyTracker({"A": [0.5], "B": [-0.25]})
    layer, db, builder = _make_layer(tmp_path, tracker)
    vectors = [("A", "v1", 0.0), ("B", "v2", 0.0)]
    layer.update_ranker(vectors, True)
    weights = db.get_db_weights()
    assert tracker.calls == 1
    assert weights["A"] == pytest.approx(0.5)
    assert weights["B"] == pytest.approx(-0.25)
    assert builder.db_weights["A"] == pytest.approx(0.5)
    assert builder.db_weights["B"] == pytest.approx(-0.25)


def test_update_ranker_aggregates_subsequent_deltas(tmp_path):
    tracker = DummyTracker({"A": [0.3]})
    layer, db, builder = _make_layer(tmp_path, tracker)
    vectors = [("A", "v1", 0.0)]
    layer.update_ranker(vectors, True)
    tracker._deltas = {"A": [0.2]}
    layer.update_ranker(vectors, True)
    weights = db.get_db_weights()
    assert tracker.calls == 2
    assert weights["A"] == pytest.approx(0.5)
    assert builder.db_weights["A"] == pytest.approx(0.5)
