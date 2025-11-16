import os
import sys
import types
import importlib.util
import pytest

# Setup lightweight imports similar to other tests
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

spec = importlib.util.spec_from_file_location(
    "menace", os.path.join(os.path.dirname(__file__), "..", "__init__.py")  # path-ignore
)
menace = importlib.util.module_from_spec(spec)
sys.modules["menace"] = menace
spec.loader.exec_module(menace)

modules = [
    "menace.self_model_bootstrap",
    "menace.research_aggregator_bot",
    "menace.model_automation_pipeline",
    "menace.diagnostic_manager",
    "menace.error_bot",
    "menace.data_bot",
    "menace.code_database",
    "menace.capital_management_bot",
    "menace.learning_engine",
    "menace.unified_event_bus",
    "menace.neuroplasticity",
    "menace.self_coding_engine",
    "menace.action_planner",
    "menace.evolution_history_db",
    "menace.self_improvement_policy",
    "menace.pre_execution_roi_bot",
    "menace.env_config",
]
for name in modules:
    sys.modules.setdefault(name, types.ModuleType(name))

sys.modules["menace.self_model_bootstrap"].bootstrap = lambda *a, **k: 0
sys.modules["menace.model_automation_pipeline"].AutomationResult = object
sys.modules["menace.model_automation_pipeline"].ModelAutomationPipeline = lambda *a, **k: object()
sys.modules["menace.diagnostic_manager"].DiagnosticManager = lambda *a, **k: object()
sys.modules["menace.error_bot"].ErrorBot = lambda *a, **k: object()
sys.modules["menace.error_bot"].ErrorDB = lambda *a, **k: object()
sys.modules["menace.data_bot"].MetricsDB = lambda *a, **k: object()
sys.modules["menace.code_database"].PatchHistoryDB = object
sys.modules["menace.data_bot"].DataBot = lambda *a, **k: object()
sys.modules["menace.capital_management_bot"].CapitalManagementBot = lambda *a, **k: object()
sys.modules["menace.learning_engine"].LearningEngine = object
sys.modules["menace.learning_engine"].LearningEngine = object
sys.modules["menace.unified_event_bus"].UnifiedEventBus = object
sys.modules["menace.unified_event_bus"].UnifiedEventBus = object
sys.modules["menace.neuroplasticity"].PathwayRecord = object
sys.modules["menace.neuroplasticity"].Outcome = object
sys.modules["menace.self_coding_engine"].SelfCodingEngine = object
sys.modules["menace.action_planner"].ActionPlanner = object
sys.modules["menace.evolution_history_db"].EvolutionHistoryDB = object
policy_mod = sys.modules["menace.self_improvement_policy"]
policy_mod.SelfImprovementPolicy = lambda *a, **k: object()
policy_mod.ConfigurableSelfImprovementPolicy = lambda *a, **k: object()
class DummyStrategy:
    def update(self, *a, **k):
        return 0.0

    def predict(self, *_):
        return [0.0] * 7

policy_mod.DQNStrategy = lambda *a, **k: DummyStrategy()
policy_mod.DoubleDQNStrategy = lambda *a, **k: DummyStrategy()
policy_mod.ActorCriticStrategy = lambda *a, **k: DummyStrategy()
policy_mod.torch = None
pre_mod = sys.modules["menace.pre_execution_roi_bot"]
pre_mod.PreExecutionROIBot = object
pre_mod.BuildTask = object
pre_mod.ROIResult = object
env_mod = sys.modules["menace.env_config"]
env_mod.PRE_ROI_SCALE = 1.0
env_mod.PRE_ROI_BIAS = 0.0
env_mod.PRE_ROI_CAP = 1.0
pyd_mod = types.ModuleType("pydantic")
class DummyAgg:
    def __init__(self, *a, **k):
        pass

sys.modules["menace.research_aggregator_bot"].ResearchAggregatorBot = DummyAgg
sys.modules["menace.research_aggregator_bot"].ResearchItem = object
sys.modules["menace.research_aggregator_bot"].InfoDB = lambda *a, **k: object()
sys.modules["menace.patch_score_backend"] = types.ModuleType("menace.patch_score_backend")
sys.modules["menace.patch_score_backend"].PatchScoreBackend = object
sys.modules["menace.patch_score_backend"].backend_from_url = lambda *a, **k: object()


class DummyContextBuilder:
    def refresh_db_weights(self):
        pass


context_builder_util = types.ModuleType("context_builder_util")
context_builder_util.create_context_builder = lambda: DummyContextBuilder()
context_builder_util.ensure_fresh_weights = lambda builder: None
sys.modules.setdefault("context_builder_util", context_builder_util)

pyd_dc = types.ModuleType("dataclasses")
pyd_mod.Field = lambda default=None, **k: default
pyd_dc.dataclass = lambda *a, **k: (lambda cls: cls)
pyd_mod.dataclasses = pyd_dc
pyd_mod.BaseModel = object
sys.modules.setdefault("pydantic", pyd_mod)
sys.modules.setdefault("pydantic.dataclasses", pyd_dc)
ps_mod = types.ModuleType("pydantic_settings")
ps_mod.BaseSettings = object
ps_mod.SettingsConfigDict = dict
sys.modules.setdefault("pydantic_settings", ps_mod)

import menace.self_improvement as sie

sie.ModelAutomationPipeline = lambda *a, **k: object()
sie.ErrorBot = lambda *a, **k: object()
sie.ErrorDB = lambda *a, **k: object()
sie.MetricsDB = lambda *a, **k: object()
sie.DiagnosticManager = lambda *a, **k: object()
sie.ConfigurableSelfImprovementPolicy = lambda *a, **k: object()


class _Rec:
    def __init__(self, r, sr, se, res, af, rel, maint, tp, ts):
        self.roi_delta = r
        self.synergy_roi = sr
        self.synergy_efficiency = se
        self.synergy_resilience = res
        self.synergy_antifragility = af
        self.synergy_reliability = rel
        self.synergy_maintainability = maint
        self.synergy_throughput = tp
        self.ts = ts


class _DummyDB:
    def __init__(self, recs):
        self._recs = list(recs)

    def filter(self):
        return list(self._recs)


class _DummyTracker:
    def __init__(self, metrics):
        self.metrics_history = metrics


def _metric_delta(vals, window=3):
    w = min(window, len(vals))
    current = sum(vals[-w:]) / w
    if len(vals) > w:
        prev_w = min(w, len(vals) - w)
        prev = sum(vals[-w - prev_w : -w]) / prev_w
    elif len(vals) >= 2:
        prev = vals[-2]
    else:
        return float(vals[-1])
    return float(current - prev)


def _expected(records, metrics, window=3):
    base = [
        "synergy_roi",
        "synergy_efficiency",
        "synergy_resilience",
        "synergy_antifragility",
        "synergy_reliability",
        "synergy_maintainability",
        "synergy_throughput",
    ]
    roi_vals = [r.roi_delta for r in records]
    X = [[float(getattr(r, n)) for n in base] for r in records]
    weights = {n: 1.0 / len(base) for n in base}
    stats: dict[str, tuple[float, float]] = {}
    if len(X) >= 2:
        import numpy as np

        arr = np.array(X, dtype=float)
        y = np.array(roi_vals, dtype=float)
        coefs, *_ = np.linalg.lstsq(arr, y, rcond=None)
        coef_abs = np.abs(coefs)
        total = float(coef_abs.sum())
        if total > 0:
            for i, name in enumerate(base):
                weights[name] = coef_abs[i] / total
        for i, name in enumerate(base):
            col = arr[:, i]
            stats[name] = (float(col.mean()), float(col.std() or 1.0))

    def norm(name: str) -> float:
        val = _metric_delta(metrics.get(name, [0.0] * len(records)), window)
        mean, std = stats.get(name, (0.0, 1.0))
        return (val - mean) / (std + 1e-6)

    adj = sum(norm(n) * weights[n] for n in base)
    return adj, weights, stats


def test_synergy_history_and_weight_update():
    records = [
        _Rec(0.5, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, "1"),
        _Rec(0.6, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, "2"),
        _Rec(0.9, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, "3"),
        _Rec(1.2, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, "4"),
    ]
    metrics = {
        "synergy_roi": [0.1, 0.2, 0.3, 0.4],
        "synergy_efficiency": [0.2, 0.1, 0.3, 0.4],
        "synergy_resilience": [0.0, 0.0, 0.0, 0.0],
        "synergy_antifragility": [0.0, 0.0, 0.0, 0.0],
        "synergy_reliability": [0.0, 0.0, 0.0, 0.0],
        "synergy_maintainability": [0.0, 0.0, 0.0, 0.0],
        "synergy_throughput": [0.0, 0.0, 0.0, 0.0],
    }
    engine = sie.SelfImprovementEngine(
        context_builder=DummyContextBuilder(),
        interval=0,
        patch_db=_DummyDB(records),
    )
    engine.tracker = _DummyTracker(metrics)

    expected_adj, weights, stats = _expected(records, metrics)
    result = engine._weighted_synergy_adjustment()
    assert result == pytest.approx(expected_adj)
    for name, val in weights.items():
        assert engine._synergy_cache["weights"][name] == pytest.approx(val)
    hist = [
        {k: getattr(r, k) for k in weights}
        for r in records
    ]
    st = sie.synergy_stats(hist)
    assert st["synergy_roi"]["average"] == pytest.approx(0.25)
    assert st["synergy_roi"]["variance"] == pytest.approx(0.0125)
    ma = sie.synergy_ma(hist, window=2)
    assert ma[-1]["synergy_roi"] == pytest.approx(0.35)

    # test weight learner update using metric deltas
    engine._update_synergy_weights(1.0)
    assert engine.synergy_weight_roi != 1.0

