import os
import sys
import types
import importlib.util
import importlib
import pytest

torch = pytest.importorskip("torch")

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

spec = importlib.util.spec_from_file_location(
    "menace", os.path.join(os.path.dirname(__file__), "..", "__init__.py")  # path-ignore
)
menace = importlib.util.module_from_spec(spec)
sys.modules["menace"] = menace
spec.loader.exec_module(menace)

# stub heavy dependencies
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
    "menace.self_improvement_policy",
    "menace.self_coding_engine",
    "menace.action_planner",
    "menace.evolution_history_db",
    "menace.self_test_service",
    "menace.mutation_logger",
    "menace.pre_execution_roi_bot",
    "menace.env_config",
    "relevancy_radar",
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
sys.modules["menace.neuroplasticity"].PathwayDB = object
sys.modules["menace.self_coding_engine"].SelfCodingEngine = object
sys.modules["menace.action_planner"].ActionPlanner = object
sys.modules["menace.evolution_history_db"].EvolutionHistoryDB = object
sys.modules["menace.evolution_history_db"].EvolutionEvent = object
sys.modules["menace.self_test_service"].SelfTestService = object
sys.modules["menace.mutation_logger"] = types.ModuleType("menace.mutation_logger")
rr = types.ModuleType("relevancy_radar")
rr.tracked_import = __import__
sys.modules["relevancy_radar"] = rr
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
policy_mod.torch = torch

import menace.self_improvement as sie  # noqa: E402


class DummySettings:
    synergy_weights_lr = 0.2
    synergy_train_interval = 1
    synergy_replay_size = 50


def _state_dict_equal(a, b):
    if a.keys() != b.keys():
        return False
    for k in a:
        v1, v2 = a[k], b[k]
        if isinstance(v1, torch.Tensor):
            if not torch.allclose(v1, v2):
                return False
        elif isinstance(v1, dict):
            if not _state_dict_equal(v1, v2):
                return False
        elif isinstance(v1, list):
            if len(v1) != len(v2):
                return False
            for x, y in zip(v1, v2):
                if isinstance(x, torch.Tensor):
                    if not torch.allclose(x, y):
                        return False
                else:
                    if x != y:
                        return False
        else:
            if v1 != v2:
                return False
    return True


def test_weight_convergence_and_persistence(tmp_path):
    path = tmp_path / "weights.json"
    learner = sie.SynergyWeightLearner(path=path, settings=DummySettings())
    deltas = {
        "synergy_roi": 1.0,
        "synergy_efficiency": 0.0,
        "synergy_resilience": 0.0,
        "synergy_antifragility": 0.0,
        "synergy_reliability": 0.0,
        "synergy_maintainability": 0.0,
        "synergy_throughput": 0.0,
    }
    for _ in range(40):
        learner.update(2.0, deltas)
    assert learner.weights["roi"] == pytest.approx(2.0, rel=0.2)
    assert learner.eval_loss >= 0.0
    base = path.with_suffix("")
    assert (tmp_path / (base.name + ".model.pt")).exists()
    assert (tmp_path / (base.name + ".optim.pt")).exists()

    learner2 = sie.SynergyWeightLearner(path=path, settings=DummySettings())
    assert learner2.weights["roi"] == pytest.approx(learner.weights["roi"])
    assert _state_dict_equal(
        learner.model.state_dict(), learner2.model.state_dict()
    )
    assert _state_dict_equal(
        learner.optimizer.state_dict(), learner2.optimizer.state_dict()
    )
