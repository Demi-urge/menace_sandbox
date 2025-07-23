import os
import sys
import types
import importlib.util
import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

spec = importlib.util.spec_from_file_location(
    "menace", os.path.join(os.path.dirname(__file__), "..", "__init__.py")
)
menace = importlib.util.module_from_spec(spec)
sys.modules["menace"] = menace
spec.loader.exec_module(menace)

# stub heavy dependencies so self_improvement_engine can be imported
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
ra.ResearchAggregatorBot = object
ra.ResearchItem = object
ra.InfoDB = object
map_mod = sys.modules["menace.model_automation_pipeline"]
map_mod.ModelAutomationPipeline = object
map_mod.AutomationResult = object
sys.modules["menace.diagnostic_manager"].DiagnosticManager = object
err_mod = sys.modules["menace.error_bot"]
err_mod.ErrorBot = object
err_mod.ErrorDB = object
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
policy_mod.ConfigurableSelfImprovementPolicy = object
policy_mod.DQNStrategy = lambda *a, **k: object()
policy_mod.DoubleDQNStrategy = lambda *a, **k: object()
policy_mod.ActorCriticStrategy = lambda *a, **k: object()
policy_mod.torch = None
pre_mod = sys.modules["menace.pre_execution_roi_bot"]
pre_mod.PreExecutionROIBot = object
pre_mod.BuildTask = object
pre_mod.ROIResult = object
env_mod = sys.modules["menace.env_config"]
env_mod.PRE_ROI_SCALE = 1.0
env_mod.PRE_ROI_BIAS = 0.0
env_mod.PRE_ROI_CAP = 1.0

np_mod = types.ModuleType("numpy")
np_mod.isscalar = lambda x: isinstance(x, (int, float, complex))
np_mod.bool_ = bool
sys.modules.setdefault("numpy", np_mod)

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

import menace.self_improvement_engine as sie


def test_synergy_weight_learner_updates(tmp_path):
    path = tmp_path / "weights.json"
    learner = sie.SynergyWeightLearner(path=path, lr=0.1)
    start = learner.weights["roi"]
    deltas = {
        "synergy_roi": 1.0,
        "synergy_efficiency": 0.0,
        "synergy_resilience": 0.0,
        "synergy_antifragility": 0.0,
        "synergy_reliability": 0.0,
        "synergy_maintainability": 0.0,
        "synergy_throughput": 0.0,
    }
    learner.update(1.0, deltas)
    after_inc = learner.weights["roi"]
    learner.update(-1.0, deltas)
    after_dec = learner.weights["roi"]
    assert after_inc > start
    assert after_dec < after_inc
    learner2 = sie.SynergyWeightLearner(path=path)
    assert learner2.weights["roi"] == pytest.approx(after_dec)


def _expected_weights(start, cycles, lr=0.1):
    """Compute expected weights after applying update cycles."""
    weights = dict(start)
    mapping = {
        "synergy_roi": "roi",
        "synergy_efficiency": "efficiency",
        "synergy_resilience": "resilience",
        "synergy_antifragility": "antifragility",
        "synergy_reliability": "reliability",
        "synergy_maintainability": "maintainability",
        "synergy_throughput": "throughput",
    }
    for roi_delta, deltas in cycles:
        if roi_delta == 0:
            continue
        for name, key in mapping.items():
            d = float(deltas.get(name, 0.0))
            w = float(weights.get(key, 1.0)) + lr * roi_delta * d
            weights[key] = max(0.0, min(w, 10.0))
    return weights


def test_synergy_weight_learner_multi_cycle(tmp_path):
    path = tmp_path / "weights.json"
    lr = 0.1
    learner = sie.SynergyWeightLearner(path=path, lr=lr)

    cycles1 = [
        (1.0, {
            "synergy_roi": 0.4,
            "synergy_efficiency": 0.2,
            "synergy_resilience": 0.1,
            "synergy_antifragility": -0.2,
            "synergy_reliability": 0.0,
            "synergy_maintainability": 0.0,
            "synergy_throughput": 0.0,
        }),
        (0.5, {
            "synergy_roi": 0.2,
            "synergy_efficiency": -0.1,
            "synergy_resilience": 0.0,
            "synergy_antifragility": 0.1,
            "synergy_reliability": 0.0,
            "synergy_maintainability": 0.0,
            "synergy_throughput": 0.0,
        }),
        (-0.3, {
            "synergy_roi": 0.1,
            "synergy_efficiency": 0.2,
            "synergy_resilience": -0.1,
            "synergy_antifragility": 0.0,
            "synergy_reliability": 0.0,
            "synergy_maintainability": 0.0,
            "synergy_throughput": 0.0,
        }),
        (0.2, {
            "synergy_roi": -0.1,
            "synergy_efficiency": 0.0,
            "synergy_resilience": 0.1,
            "synergy_antifragility": 0.2,
            "synergy_reliability": 0.0,
            "synergy_maintainability": 0.0,
            "synergy_throughput": 0.0,
        }),
        (1.2, {
            "synergy_roi": 0.3,
            "synergy_efficiency": 0.1,
            "synergy_resilience": 0.2,
            "synergy_antifragility": 0.1,
            "synergy_reliability": 0.0,
            "synergy_maintainability": 0.0,
            "synergy_throughput": 0.0,
        }),
    ]

    expected1 = _expected_weights(learner.weights, cycles1, lr)
    for roi_delta, deltas in cycles1:
        learner.update(roi_delta, deltas)

    assert learner.weights == pytest.approx(expected1)

    # persistence check after first set of updates
    learner2 = sie.SynergyWeightLearner(path=path, lr=lr)
    assert learner2.weights == pytest.approx(expected1)

    cycles2 = [
        (-0.4, {
            "synergy_roi": 0.2,
            "synergy_efficiency": 0.0,
            "synergy_resilience": 0.1,
            "synergy_antifragility": -0.1,
            "synergy_reliability": 0.0,
            "synergy_maintainability": 0.0,
            "synergy_throughput": 0.0,
        }),
        (0.7, {
            "synergy_roi": -0.3,
            "synergy_efficiency": 0.2,
            "synergy_resilience": 0.0,
            "synergy_antifragility": 0.1,
            "synergy_reliability": 0.0,
            "synergy_maintainability": 0.0,
            "synergy_throughput": 0.0,
        }),
        (0.5, {
            "synergy_roi": 0.1,
            "synergy_efficiency": 0.1,
            "synergy_resilience": 0.2,
            "synergy_antifragility": 0.0,
            "synergy_reliability": 0.0,
            "synergy_maintainability": 0.0,
            "synergy_throughput": 0.0,
        }),
    ]

    expected2 = _expected_weights(expected1, cycles2, lr)
    for roi_delta, deltas in cycles2:
        learner2.update(roi_delta, deltas)

    assert learner2.weights == pytest.approx(expected2)

    # persistence check after all updates
    learner3 = sie.SynergyWeightLearner(path=path, lr=lr)
    assert learner3.weights == pytest.approx(expected2)
