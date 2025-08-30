import pytest

from vector_metrics_db import VectorMetricsDB
from vector_service.weight_adjuster import WeightAdjuster, RoiTag


def test_adjust_updates_weights(tmp_path):
    vm = VectorMetricsDB(tmp_path / "vm.db")
    adj = WeightAdjuster(
        vector_metrics=vm,
        db_success_delta=0.2,
        db_failure_delta=0.1,
        vector_success_delta=0.2,
        vector_failure_delta=0.1,
    )
    adj.adjust([("db", "v1", 0.8, RoiTag.HIGH_ROI)])
    assert vm.get_db_weights()["db"] == pytest.approx(0.16)
    assert vm.get_vector_weight("db:v1") == pytest.approx(0.16)


def test_adjust_decreases_on_failure(tmp_path):
    vm = VectorMetricsDB(tmp_path / "vm2.db")
    vm.set_db_weights({"db": 0.5})
    vm.set_vector_weight("db:v1", 0.5)
    adj = WeightAdjuster(
        vector_metrics=vm,
        db_success_delta=0.2,
        db_failure_delta=0.1,
        vector_success_delta=0.2,
        vector_failure_delta=0.1,
    )
    adj.adjust([("db", "v1", 0.8, RoiTag.LOW_ROI)])
    assert vm.get_db_weights()["db"] == pytest.approx(0.42)
    assert vm.get_vector_weight("db:v1") == pytest.approx(0.42)

