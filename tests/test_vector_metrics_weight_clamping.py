from types import SimpleNamespace

import pytest

from vector_service.cognition_layer import CognitionLayer
from vector_metrics_db import VectorMetricsDB


def test_weights_stay_within_bounds_after_feedback(tmp_path):
    vec_db = VectorMetricsDB(tmp_path / "vm.db")
    builder = SimpleNamespace(refresh_db_weights=lambda *a, **k: None)
    layer = CognitionLayer(context_builder=builder, vector_metrics=vec_db)
    vectors = [("db1", "v1", 0.0), ("db2", "v2", 0.0)]

    for _ in range(5):
        layer.update_ranker(vectors, True, roi_deltas={"db1": 1.0, "db2": -1.0})
    weights = vec_db.get_db_weights()
    assert weights["db1"] == pytest.approx(1.0)
    assert weights["db2"] == pytest.approx(0.0)

    for _ in range(5):
        layer.update_ranker(vectors, True, roi_deltas={"db1": -1.0, "db2": 1.0})
    weights = vec_db.get_db_weights()
    assert weights["db1"] == pytest.approx(0.0)
    assert weights["db2"] == pytest.approx(1.0)
    assert all(0.0 <= w <= 1.0 for w in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0)
