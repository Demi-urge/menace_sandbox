import os
import sys
import types
import importlib.util
import importlib
import pytest
# flake8: noqa

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

# remove torch temporarily
_orig_torch = sys.modules.pop("torch", None)

spec = importlib.util.spec_from_file_location(
    "menace", os.path.join(os.path.dirname(__file__), "..", "__init__.py")  # path-ignore
)
menace = importlib.util.module_from_spec(spec)
sys.modules["menace"] = menace
spec.loader.exec_module(menace)

# stub heavy dependencies needed for self_improvement
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
    "menace.self_test_service",
    "menace.mutation_logger",
    "menace.pre_execution_roi_bot",
    "menace.env_config",
    "adaptive_roi_predictor",
    "sandbox_runner.environment",
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
rr.RelevancyRadar = object
rr.track_usage = lambda *a, **k: None
sys.modules["relevancy_radar"] = rr
pre_mod = sys.modules["menace.pre_execution_roi_bot"]
pre_mod.PreExecutionROIBot = object
pre_mod.BuildTask = object
pre_mod.ROIResult = object
env_mod = sys.modules["menace.env_config"]
env_mod.PRE_ROI_SCALE = 1.0
env_mod.PRE_ROI_BIAS = 0.0
env_mod.PRE_ROI_CAP = 1.0
sr_env = sys.modules["sandbox_runner.environment"]
sr_env.SANDBOX_ENV_PRESETS = {}
sr_env.simulate_full_environment = lambda *a, **k: None
ar_mod = sys.modules["adaptive_roi_predictor"]
ar_mod.load_training_data = lambda *a, **k: []

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

sie = importlib.import_module("menace.self_improvement")


def test_actor_critic_used_without_torch(tmp_path):
    assert "torch" not in sys.modules
    learner = sie.SynergyWeightLearner(path=tmp_path / "weights.json")
    assert isinstance(learner.strategy, sie.ActorCriticStrategy)
    start = dict(learner.weights)
    deltas = {
        "synergy_roi": 0.5,
        "synergy_efficiency": 0.0,
        "synergy_resilience": 0.0,
        "synergy_antifragility": 0.0,
        "synergy_reliability": 0.0,
        "synergy_maintainability": 0.0,
        "synergy_throughput": 0.0,
    }
    learner.update(1.0, deltas)
    assert learner.weights != start
    assert (tmp_path / "weights.json").exists()
    reloaded = sie.SynergyWeightLearner(path=tmp_path / "weights.json")
    assert reloaded.weights == pytest.approx(learner.weights)

    if _orig_torch is not None:
        sys.modules["torch"] = _orig_torch
