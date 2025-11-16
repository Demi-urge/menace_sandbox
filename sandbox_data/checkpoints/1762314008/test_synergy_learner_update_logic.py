import os
import sys
import types
import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import menace
# stub heavy deps so self_improvement can be imported
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
sys.modules["menace.research_aggregator_bot"].ResearchAggregatorBot = object
sys.modules["menace.research_aggregator_bot"].ResearchItem = object
sys.modules["menace.research_aggregator_bot"].InfoDB = object
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

class RLStrategy:
    def __init__(self):
        self.q = [1.0] * 7
        self.actions = []

    def update(self, _policy_state, state, action, reward, next_state, done, gamma):
        self.actions.append(action)
        self.q[action] += reward
        return self.q[action]

    def predict(self, state):
        return list(self.q)

policy_mod.DQNStrategy = lambda *a, **k: RLStrategy()
policy_mod.DoubleDQNStrategy = lambda *a, **k: RLStrategy()
policy_mod.ActorCriticStrategy = lambda *a, **k: RLStrategy()
policy_mod.torch = None
pre_mod = sys.modules["menace.pre_execution_roi_bot"]
pre_mod.PreExecutionROIBot = object
pre_mod.BuildTask = object
pre_mod.ROIResult = object
env_mod = sys.modules["menace.env_config"]
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

env_mod.PRE_ROI_SCALE = 1.0
env_mod.PRE_ROI_BIAS = 0.0
env_mod.PRE_ROI_CAP = 1.0

import menace.self_improvement as sie


def test_weight_learner_rl_update(tmp_path):
    path = tmp_path / "w.json"
    learner = sie.SynergyWeightLearner(path=path, lr=0.1)
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
    after_pos = learner.weights["roi"]
    assert after_pos > start["roi"]
    assert len(learner.strategy.actions) == 7
    learner.update(-1.0, deltas)
    after_neg = learner.weights["roi"]
    assert after_neg < after_pos
    loaded = sie.SynergyWeightLearner(path=path)
    assert loaded.weights == pytest.approx(learner.weights)


def test_dqn_learner_rl_persistence(tmp_path):
    path = tmp_path / "weights.json"
    learner = sie.DQNSynergyLearner(path=path, lr=0.1, target_sync=1)
    deltas = {
        "synergy_roi": 0.4,
        "synergy_efficiency": 0.0,
        "synergy_resilience": 0.0,
        "synergy_antifragility": 0.0,
        "synergy_reliability": 0.0,
        "synergy_maintainability": 0.0,
        "synergy_throughput": 0.0,
    }
    learner.update(1.0, deltas)
    assert learner.weights["roi"] > 1.0
    learner2 = sie.DQNSynergyLearner(path=path, lr=0.1)
    assert learner2.weights == pytest.approx(learner.weights)
