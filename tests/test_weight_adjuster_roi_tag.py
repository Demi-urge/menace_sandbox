import pytest

from vector_service.weight_adjuster import WeightAdjuster


class DummyVectorMetrics:
    def __init__(self):
        self.weights = {}

    def update_db_weight(self, origin, delta):
        weight = self.weights.get(origin, 1.0) + delta
        self.weights[origin] = weight
        return weight

    def log_ranker_update(self, origin, delta, weight):  # pragma: no cover - optional
        self.last = (origin, delta, weight)


def test_roi_tag_positive_overrides_score():
    vm = DummyVectorMetrics()
    adj = WeightAdjuster(vector_metrics=vm, success_delta=0.2, failure_delta=0.2)
    adj.adjust(["db:v1"], 0.1, "pass")
    assert vm.weights["db"] == pytest.approx(1.02)


def test_roi_tag_negative_overrides_score():
    vm = DummyVectorMetrics()
    adj = WeightAdjuster(vector_metrics=vm, success_delta=0.2, failure_delta=0.2)
    adj.adjust(["db:v1"], 0.9, "bug")
    assert vm.weights["db"] == pytest.approx(0.82)
