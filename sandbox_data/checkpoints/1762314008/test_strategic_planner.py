import importlib.util
import pytest

if importlib.util.find_spec("networkx") is None:
    pytest.skip("optional dependencies not installed", allow_module_level=True)

import menace.strategic_planner as sp


class DummyBot:
    def __init__(self):
        self.called = False

    def predict(self, features):
        self.called = True
        return 0.5

    def counter_strategy(self, prob):
        return "plan"


class DummyAutoscaler:
    def __init__(self):
        self.metrics = []

    def scale(self, metrics):
        self.metrics.append(metrics)


class DummyPredictor:
    def predict_future_metrics(self, n):
        class P:
            roi = 0.7
            errors = 0.1
        return P()


def test_plan_cycle():
    planner = sp.StrategicPlanner(DummyBot(), DummyAutoscaler(), DummyPredictor())
    plan = planner.plan_cycle()
    assert plan == "plan"

