import os
import sys
import types
import logging
import importlib.util
import os

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
err = types.ModuleType("error_logger")
err.ErrorLogger = object
sys.modules.setdefault("error_logger", err)

# stub sandbox_runner to avoid heavy imports
sr = types.ModuleType("sandbox_runner")
sr_bootstrap = types.ModuleType("sandbox_runner.bootstrap")
sr_bootstrap.initialize_autonomous_sandbox = lambda *a, **k: None
sys.modules.setdefault("sandbox_runner", sr)
sys.modules.setdefault("sandbox_runner.bootstrap", sr_bootstrap)
sr_env = types.ModuleType("sandbox_runner.environment")
sr_env.load_presets = lambda *a, **k: {}
sr_env.simulate_full_environment = lambda *a, **k: None
sys.modules.setdefault("sandbox_runner.environment", sr_env)
sr_cli = types.ModuleType("sandbox_runner.cli")
sr_cli.main = lambda *a, **k: None
sys.modules.setdefault("sandbox_runner.cli", sr_cli)
sr_cycle = types.ModuleType("sandbox_runner.cycle")
sys.modules.setdefault("sandbox_runner.cycle", sr_cycle)
sr_oi = types.ModuleType("sandbox_runner.orphan_integration")
sys.modules.setdefault("sandbox_runner.orphan_integration", sr_oi)
neurosales_mod = types.ModuleType("neurosales")
sys.modules.setdefault("neurosales", neurosales_mod)

# use real neuroplasticity module so PathwayDB is available
np_spec = importlib.util.spec_from_file_location(
    "menace.neuroplasticity", os.path.join(os.path.dirname(__file__), "..", "neuroplasticity.py")  # path-ignore
)
np_mod = importlib.util.module_from_spec(np_spec)
sys.modules["menace.neuroplasticity"] = np_mod
np_spec.loader.exec_module(np_mod)

from tests import test_self_improvement_engine_rl_synergy as base
sie = base.sie
BaselineTracker = sie.baseline_tracker.BaselineTracker


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
        "latency_error_rate": 0.12,
        "hostile_failures": 0.1,
        "concurrency_throughput": 205.0,
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
    eng.baseline_tracker = BaselineTracker(
        window=3,
        latency_error_rate=[0.1, 0.15, 0.11],
        hostile_failures=[0.0, 0.1, 0.2],
        concurrency_throughput=[200.0, 210.0, 205.0],
    )
    return eng


def test_scenario_metric_degradation_triggers_actions(monkeypatch):
    engine = make_engine()
    alerts: list[tuple] = []
    patches: list[tuple] = []
    monkeypatch.setattr(sie, "dispatch_alert", lambda *a, **k: alerts.append(a))
    monkeypatch.setattr(engine, "_generate_patch_with_memory", lambda *a, **k: (patches.append(a), 0)[1])
    monkeypatch.setattr(engine, "_pre_commit_alignment_check", lambda *a, **k: None)
    monkeypatch.setattr(engine, "_alignment_review_last_commit", lambda *a, **k: None)
    monkeypatch.setattr(engine, "_sandbox_integrate", lambda *a, **k: None)
    monkeypatch.setattr(engine, "_post_round_orphan_scan", lambda *a, **k: None)
    metrics = {
        "latency_error_rate": 0.15,
        "hostile_failures": 0.3,
        "concurrency_throughput": 180.0,
    }
    frac = engine._evaluate_scenario_metrics(metrics)
    engine.baseline_tracker.update(pass_rate=frac, **metrics)
    assert len(alerts) == 1
    assert len(patches) == 1
    assert engine._force_rerun
    assert engine._scenario_pass_rate < 0
    assert engine._pass_rate_delta == engine.baseline_tracker.get("pass_rate_delta")
    captured = {}
    def fake_update(roi_delta, deltas, extra=None):
        captured["extra"] = extra
    engine.synergy_learner.update = fake_update  # type: ignore
    engine._update_synergy_weights(0.0)
    assert captured["extra"]["pass_rate"] == engine._scenario_pass_rate
    assert captured["extra"]["avg_roi"] < 0
