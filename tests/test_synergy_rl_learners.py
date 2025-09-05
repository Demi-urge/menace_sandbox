import os
import sys
import types
import importlib
import importlib.util
import pickle
import pytest


class DummyModel:
    def __init__(self) -> None:
        self.params = [0.0]

    def state_dict(self) -> dict:
        return {"p": list(self.params)}

    def load_state_dict(self, sd: dict) -> None:
        self.params = list(sd["p"])

    def parameters(self):
        return self.params


class DummyStrategy:
    def __init__(self, *, target_sync: int = 1, **_):
        self.model = DummyModel()
        self.target_model = DummyModel()
        self.target_model.load_state_dict(self.model.state_dict())
        self.steps = 0
        self.target_sync = target_sync

    def update(self, table, state, action, reward, next_state, alpha, gamma):
        self.model.params[0] += reward * 0.01
        self.steps += 1
        if self.steps % self.target_sync == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        return self.model.params[0]

    def predict(self, state):
        return [self.model.params[0]] * 7

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

spec = importlib.util.spec_from_file_location(
    "menace", os.path.join(os.path.dirname(__file__), "..", "__init__.py")  # path-ignore
)
menace = importlib.util.module_from_spec(spec)
sys.modules["menace"] = menace
spec.loader.exec_module(menace)

# ------------------------------------------------------------
# helper fixture setting up stub modules for each test

@pytest.fixture
def sie(monkeypatch):
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
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

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
    monkeypatch.setitem(sys.modules, "jinja2", jinja_mod)

    for name in [
        "cryptography",
        "cryptography.hazmat",
        "cryptography.hazmat.primitives",
        "cryptography.hazmat.primitives.asymmetric",
        "cryptography.hazmat.primitives.asymmetric.ed25519",
        "cryptography.hazmat.primitives.serialization",
    ]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    pyd_mod = types.ModuleType("pydantic")
    pyd_dc = types.ModuleType("dataclasses")
    pyd_dc.dataclass = lambda *a, **k: (lambda f: f)
    pyd_mod.dataclasses = pyd_dc
    pyd_mod.Field = lambda default=None, **k: default
    pyd_mod.BaseModel = object
    monkeypatch.setitem(sys.modules, "pydantic", pyd_mod)
    pyd_settings_mod = types.ModuleType("pydantic_settings")
    pyd_settings_mod.BaseSettings = object
    pyd_settings_mod.SettingsConfigDict = dict
    monkeypatch.setitem(sys.modules, "pydantic_settings", pyd_settings_mod)

    policy_mod.DQNStrategy = lambda *a, **k: DummyStrategy(**k)
    policy_mod.DoubleDQNStrategy = lambda *a, **k: DummyStrategy(**k)
    policy_mod.ActorCriticStrategy = lambda *a, **k: DummyStrategy(**k)

    torch_mod = types.ModuleType("torch")
    def save(obj, f):
        if hasattr(f, "write"):
            pickle.dump(obj, f)
        else:
            with open(f, "wb") as fh:
                pickle.dump(obj, fh)

    def load(f):
        if hasattr(f, "read"):
            return pickle.load(f)
        with open(f, "rb") as fh:
            return pickle.load(fh)
    torch_mod.save = save
    torch_mod.load = load
    torch_mod.equal = lambda a, b: a == b
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    policy_mod.torch = torch_mod

    sb3_mod = types.ModuleType("stable_baselines3")
    class DummyAlgo:
        def __init__(self, *a, **k):
            pass
        def learn(self, *a, **k):
            pass
        def predict(self, *_):
            return 0, None
        def save(self, *_):
            pass
        @classmethod
        def load(cls, *_):
            return cls()
    sb3_mod.DQN = sb3_mod.SAC = sb3_mod.TD3 = DummyAlgo
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3_mod)

    sie = importlib.import_module("menace.self_improvement")
    return sie

# ------------------------------------------------------------

DELTAS = {
    "synergy_roi": 1.0,
    "synergy_efficiency": 0.5,
    "synergy_resilience": 0.2,
    "synergy_antifragility": 0.0,
    "synergy_reliability": 0.0,
    "synergy_maintainability": 0.0,
    "synergy_throughput": 0.0,
}

@pytest.mark.parametrize("cls_name", ["DQNSynergyLearner"])
def test_rl_learner_updates_and_sync(tmp_path, sie, cls_name):
    cls = getattr(sie, cls_name)
    path = tmp_path / f"{cls_name}.json"
    learner = cls(path=path, lr=0.1, target_sync=2)
    start = dict(learner.weights)

    learner.update(1.0, DELTAS)
    learner.update(1.0, DELTAS)
    assert learner.strategy.model.params == learner.strategy.target_model.params

    assert learner.weights != start

    learner2 = cls(path=path)
    assert learner2.weights == pytest.approx(learner.weights)
