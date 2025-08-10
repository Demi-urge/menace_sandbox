import os
import sys
import types
import logging

# ensure light imports
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

# stub heavy modules before importing engine
qe = types.ModuleType("menace.quick_fix_engine")
qe.generate_patch = lambda *a, **k: 0
sys.modules.setdefault("menace.quick_fix_engine", qe)
ml = types.ModuleType("menace.mutation_logger")
ml.log_mutation = lambda *a, **k: 0
ml.record_mutation_outcome = lambda *a, **k: None
sys.modules.setdefault("menace.mutation_logger", ml)
mid = types.ModuleType("menace.module_index_db")
class DummyMID:
    def __init__(self, *a, **k):
        pass
    def refresh(self, *a, **k):
        pass
    def get(self, *a, **k):
        return 0
    def merge_groups(self, *a, **k):
        pass
    def group_id(self, *a, **k):
        return 0
mid.ModuleIndexDB = DummyMID
sys.modules.setdefault("menace.module_index_db", mid)

from tests import test_self_improvement_engine_rl_synergy as base
sie = base.sie


class DummyLearner:
    def __init__(self):
        self.weights = {
            "roi": 1.0,
            "efficiency": 1.0,
            "resilience": 1.0,
            "antifragility": 1.0,
            "reliability": 1.0,
            "maintainability": 1.0,
            "throughput": 1.0,
        }
    def update(self, *a, **k):
        pass


def make_engine():
    eng = object.__new__(sie.SelfImprovementEngine)
    eng.logger = logging.getLogger("test")
    eng.self_coding_engine = None
    eng._last_scenario_metrics = {
        "latency_error_rate": 0.1,
        "hostile_failures": 0.0,
        "concurrency_throughput": 200.0,
    }
    eng.synergy_learner = DummyLearner()
    eng.synergy_weight_roi = eng.synergy_weight_efficiency = eng.synergy_weight_resilience = eng.synergy_weight_antifragility = eng.synergy_weight_reliability = eng.synergy_weight_maintainability = eng.synergy_weight_throughput = 1.0
    eng._last_orphan_metrics = {}
    import types as _t
    eng._metric_delta = _t.MethodType(lambda self, name, window=3: 0.0, eng)
    eng._last_scenario_trend = {}
    eng._scenario_pass_rate = 0.0
    eng._force_rerun = False
    eng._last_mutation_id = None
    return eng


def test_scenario_metric_degradation_triggers_actions(monkeypatch):
    engine = make_engine()
    alerts = []
    patches = []
    monkeypatch.setattr(sie, "dispatch_alert", lambda *a, **k: alerts.append(a))
    monkeypatch.setattr(sie, "generate_patch", lambda *a, **k: patches.append(a))
    engine._evaluate_scenario_metrics(
        {
            "latency_error_rate": 0.3,
            "hostile_failures": 10.0,
            "concurrency_throughput": 50.0,
        }
    )
    assert alerts
    assert patches
    assert engine._force_rerun
    assert engine._scenario_pass_rate < 0
    captured = {}
    def fake_update(roi_delta, deltas, extra=None):
        captured["extra"] = extra
    engine.synergy_learner.update = fake_update  # type: ignore
    engine._update_synergy_weights(0.0)
    assert captured["extra"]["pass_rate"] == engine._scenario_pass_rate
    assert captured["extra"]["avg_roi"] < 0
