import pytest

from vector_metrics_db import VectorMetricsDB
from weight_adjuster import WeightAdjuster


def test_weight_adjuster_updates_weights(tmp_path):
    db = VectorMetricsDB(tmp_path / "vm.db")
    adj = WeightAdjuster(vector_metrics=db, success_delta=0.2, failure_delta=0.1)
    vectors = [("patch", "v1", 0.0)]

    adj.adjust(vectors, True)
    assert db.get_vector_weight("patch:v1") == pytest.approx(0.2)

    adj.adjust(vectors, False)
    assert db.get_vector_weight("patch:v1") == pytest.approx(0.1)


def test_weight_adjuster_negative_roi(tmp_path):
    db = VectorMetricsDB(tmp_path / "vm2.db")
    adj = WeightAdjuster(vector_metrics=db, success_delta=0.2, failure_delta=0.1)
    vectors = [("patch", "v1", 0.0)]

    adj.adjust(vectors, True, roi_deltas={"patch": -1.0})
    assert db.get_vector_weight("patch:v1") == pytest.approx(0.0)
