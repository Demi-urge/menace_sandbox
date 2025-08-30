import pytest

from vector_metrics_db import VectorMetricsDB
from vector_service.weight_adjuster import WeightAdjuster, RoiTag


def test_weight_adjuster_updates_weights(tmp_path):
    db = VectorMetricsDB(tmp_path / "vm.db")
    adj = WeightAdjuster(
        vector_metrics=db,
        db_success_delta=0.2,
        db_failure_delta=0.1,
        vector_success_delta=0.2,
        vector_failure_delta=0.1,
    )
    vectors = [("patch", "v1", 0.0)]

    adj.adjust(vectors, 1.0, RoiTag.SUCCESS, tests_passed=True)
    assert db.get_vector_weight("patch:v1") == pytest.approx(0.2)

    adj.adjust(vectors, 1.0, RoiTag.LOW_ROI, tests_passed=True)
    assert db.get_vector_weight("patch:v1") == pytest.approx(0.1)


def test_weight_adjuster_scales_by_quality(tmp_path):
    db = VectorMetricsDB(tmp_path / "vm2.db")
    adj = WeightAdjuster(
        vector_metrics=db,
        db_success_delta=0.2,
        db_failure_delta=0.1,
        vector_success_delta=0.2,
        vector_failure_delta=0.1,
    )
    vectors = [("patch", "v1", 1.0)]

    adj.adjust(vectors, 1.0, RoiTag.SUCCESS, tests_passed=True)
    base = db.get_vector_weight("patch:v1")

    adj.adjust(
        vectors,
        1.0,
        RoiTag.SUCCESS,
        tests_passed=False,
        error_trace_count=2,
    )
    assert db.get_vector_weight("patch:v1") == pytest.approx(base - (0.1 / 3))
