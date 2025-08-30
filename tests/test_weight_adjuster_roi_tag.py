import pytest

from vector_service.weight_adjuster import WeightAdjuster


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


def test_roi_tag_positive_overrides_score():
    vm = DummyVectorMetrics()
    adj = WeightAdjuster(vector_metrics=vm, success_delta=0.2, failure_delta=0.2)
    adj.adjust([("db", "v1", 0.5)], 0.1, "success")
    assert vm.weights["db"] == pytest.approx(1.02)
    assert vm.vector_weights["db:v1"] == pytest.approx(0.1)


def test_roi_tag_negative_overrides_score():
    vm = DummyVectorMetrics()
    adj = WeightAdjuster(vector_metrics=vm, success_delta=0.2, failure_delta=0.2)
    adj.adjust([("db", "v1", 0.5)], 0.9, "bug-introduced")
    assert vm.weights["db"] == pytest.approx(0.82)
    assert vm.vector_weights["db:v1"] == pytest.approx(0.0)
