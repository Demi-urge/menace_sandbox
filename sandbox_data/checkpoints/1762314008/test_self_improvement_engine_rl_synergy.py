import os
import sys
import types
import importlib.util
import pytest

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

spec = importlib.util.spec_from_file_location(
    "menace", os.path.join(os.path.dirname(__file__), "..", "__init__.py")  # path-ignore
)
menace = importlib.util.module_from_spec(spec)
sys.modules["menace"] = menace
spec.loader.exec_module(menace)

# stub heavy dependencies so self_improvement can be imported
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
ra = sys.modules["menace.research_aggregator_bot"]
class DummyAgg:
    def __init__(self, *a, **k):
        pass
ra.ResearchAggregatorBot = DummyAgg
ra.ResearchItem = object
ra.InfoDB = object
map_mod = sys.modules["menace.model_automation_pipeline"]
map_mod.ModelAutomationPipeline = lambda *a, **k: object()
map_mod.AutomationResult = object
sys.modules["menace.diagnostic_manager"].DiagnosticManager = lambda *a, **k: object()
err_mod = sys.modules["menace.error_bot"]
err_mod.ErrorBot = lambda *a, **k: object()
err_mod.ErrorDB = lambda *a, **k: object()
sys.modules["menace.data_bot"].MetricsDB = object
sys.modules["menace.data_bot"].DataBot = object
sys.modules["menace.code_database"].PatchHistoryDB = object
sys.modules["menace.capital_management_bot"].CapitalManagementBot = object
sys.modules["menace.learning_engine"].LearningEngine = object
sys.modules["menace.unified_event_bus"].UnifiedEventBus = object
sys.modules["menace.neuroplasticity"].PathwayRecord = object
sys.modules["menace.neuroplasticity"].Outcome = object
sys.modules["menace.self_coding_engine"].SelfCodingEngine = object
sys.modules["menace.action_planner"].ActionPlanner = object
sys.modules["menace.evolution_history_db"].EvolutionHistoryDB = object
policy_mod = sys.modules["menace.self_improvement_policy"]
policy_mod.SelfImprovementPolicy = object
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

jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)

for name in [
    "cryptography",
    "cryptography.hazmat",
    "cryptography.hazmat.primitives",
    "cryptography.hazmat.primitives.asymmetric",
    "cryptography.hazmat.primitives.asymmetric.ed25519",
    "cryptography.hazmat.primitives.serialization",
]:
    sys.modules.setdefault(name, types.ModuleType(name))

pyd_mod = types.ModuleType("pydantic")
pyd_dc = types.ModuleType("dataclasses")
pyd_dc.dataclass = lambda *a, **k: (lambda f: f)
pyd_mod.dataclasses = pyd_dc
pyd_mod.Field = lambda default=None, **k: default
pyd_mod.BaseModel = object
sys.modules.setdefault("pydantic", pyd_mod)
pyd_settings_mod = types.ModuleType("pydantic_settings")
pyd_settings_mod.BaseSettings = object
pyd_settings_mod.SettingsConfigDict = dict
sys.modules.setdefault("pydantic_settings", pyd_settings_mod)


class DummyContextBuilder:
    def refresh_db_weights(self):
        pass


context_builder_util = types.ModuleType("context_builder_util")
context_builder_util.create_context_builder = lambda: DummyContextBuilder()
context_builder_util.ensure_fresh_weights = lambda builder: None
sys.modules.setdefault("context_builder_util", context_builder_util)

import menace.self_improvement as sie

class _Rec:
    def __init__(self, r, sr):
        self.roi_delta = r
        self.synergy_roi = sr
        self.synergy_efficiency = 0.0
        self.synergy_resilience = 0.0
        self.synergy_antifragility = 0.0
        self.synergy_reliability = 0.0
        self.synergy_maintainability = 0.0
        self.synergy_throughput = 0.0
        self.ts = "1"

class _DummyDB:
    def __init__(self, recs):
        self._recs = list(recs)
    def filter(self):
        return list(self._recs)

class DummyTracker:
    def __init__(self):
        self.metrics_history = {n: [0.0] for n in [
            "synergy_roi",
            "synergy_efficiency",
            "synergy_resilience",
            "synergy_antifragility",
            "synergy_reliability",
            "synergy_maintainability",
            "synergy_throughput",
        ]}
    def advance(self):
        for k in self.metrics_history:
            self.metrics_history[k].append(self.metrics_history[k][-1] + 0.1)


def _make_engine(path: os.PathLike, learner_cls: type) -> sie.SelfImprovementEngine:
    engine = sie.SelfImprovementEngine(
        context_builder=DummyContextBuilder(),
        interval=0,
        patch_db=_DummyDB([]),
        synergy_weights_path=path,
        synergy_weights_lr=0.01,
        synergy_learner_cls=learner_cls,
    )
    engine.tracker = DummyTracker()
    return engine


def _train_cycles(engine: sie.SelfImprovementEngine, cycles: int = 5) -> None:
    for _ in range(cycles):
        engine._update_synergy_weights(1.0)
        engine.tracker.advance()


def _reload_engine(path: os.PathLike, learner_cls: type) -> sie.SelfImprovementEngine:
    return sie.SelfImprovementEngine(
        context_builder=DummyContextBuilder(),
        interval=0,
        patch_db=_DummyDB([]),
        synergy_weights_path=path,
        synergy_learner_cls=learner_cls,
    )


def test_dqn_engine_weights_update(tmp_path):
    torch = pytest.importorskip("torch")
    import importlib
    import menace.self_improvement_policy as sip
    sip = importlib.reload(sip)
    sie = importlib.reload(sys.modules["menace.self_improvement"])

    path = tmp_path / "dqn.json"
    engine = _make_engine(path, sie.DQNSynergyLearner)
    start = engine.synergy_weight_roi
    _train_cycles(engine, cycles=3)
    assert engine.synergy_weight_roi != start
    engine2 = _reload_engine(path, sie.DQNSynergyLearner)
    assert engine2.synergy_weight_roi == pytest.approx(engine.synergy_weight_roi)


def test_sac_engine_weights_update(tmp_path):
    torch = pytest.importorskip("torch")
    import importlib
    import menace.self_improvement_policy as sip
    sip = importlib.reload(sip)
    sie = importlib.reload(sys.modules["menace.self_improvement"])

    path = tmp_path / "sac.json"
    engine = _make_engine(path, sie.SACSynergyLearner)
    start = engine.synergy_weight_roi
    _train_cycles(engine, cycles=3)
    assert engine.synergy_weight_roi != start
    engine2 = _reload_engine(path, sie.SACSynergyLearner)
    assert engine2.synergy_weight_roi == pytest.approx(engine.synergy_weight_roi)


def test_td3_engine_weights_update(tmp_path):
    torch = pytest.importorskip("torch")
    import importlib
    import menace.self_improvement_policy as sip
    sip = importlib.reload(sip)
    sie = importlib.reload(sys.modules["menace.self_improvement"])

    path = tmp_path / "td3.json"
    engine = _make_engine(path, sie.TD3SynergyLearner)
    start = engine.synergy_weight_roi
    _train_cycles(engine, cycles=3)
    assert engine.synergy_weight_roi != start
    engine2 = _reload_engine(path, sie.TD3SynergyLearner)
    assert engine2.synergy_weight_roi == pytest.approx(engine.synergy_weight_roi)


def test_update_failure_dispatches_alert(monkeypatch, tmp_path):
    import importlib
    sie = importlib.reload(sys.modules["menace.self_improvement"])

    path = tmp_path / "w.json"
    engine = _make_engine(path, sie.SynergyWeightLearner)

    alerts: list[tuple] = []

    def fake_alert(*a, **k):
        alerts.append((a, k))

    monkeypatch.setattr(sie, "dispatch_alert", fake_alert)
    monkeypatch.setattr(engine.synergy_learner, "update", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    sie.synergy_weight_update_failures_total.set(0.0)
    sie.synergy_weight_update_alerts_total.set(0.0)

    engine._update_synergy_weights(1.0)

    assert len(alerts) == 1
    assert sie.synergy_weight_update_failures_total._value.get() == 1.0
    assert sie.synergy_weight_update_alerts_total._value.get() == 1.0
