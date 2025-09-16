import os
import sys
import types
import importlib.util
import pytest
# flake8: noqa

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
    "menace.self_test_service",
    "menace.mutation_logger",
    "menace.self_improvement_policy",
    "menace.pre_execution_roi_bot",
    "menace.env_config",
    "relevancy_radar",
    "menace.gpt_memory",
    "gpt_memory",
    "menace.local_knowledge_module",
    "menace.gpt_knowledge_service",
    "gpt_memory_interface",
    "menace.intent_clusterer",
    "sandbox_runner.bootstrap",
    "sandbox_runner.environment",
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
sys.modules["sandbox_runner.bootstrap"].initialize_autonomous_sandbox = lambda *a, **k: None
env_mod = sys.modules["sandbox_runner.environment"]
env_mod.SANDBOX_ENV_PRESETS = [{}]
env_mod.simulate_full_environment = lambda *a, **k: None
env_mod.auto_include_modules = lambda *a, **k: (None, [])
env_mod.run_workflow_simulations = lambda *a, **k: (None, [])
env_mod.try_integrate_into_workflows = lambda *a, **k: None
sys.modules["menace.gpt_memory"].GPTMemoryManager = object
sys.modules["gpt_memory"] = sys.modules["menace.gpt_memory"]
sys.modules["menace.gpt_memory"].STANDARD_TAGS = {}
lk_mod = sys.modules["menace.local_knowledge_module"]
lk_mod.init_local_knowledge = lambda *a, **k: None
lk_mod.LocalKnowledgeModule = object
sys.modules["gpt_memory_interface"].GPTMemoryInterface = object
sys.modules["menace.gpt_knowledge_service"].GPTKnowledgeService = object
sys.modules["menace.intent_clusterer"].IntentClusterer = object
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


class DummyContextBuilder:
    def refresh_db_weights(self):
        pass


context_builder_util = types.ModuleType("context_builder_util")
context_builder_util.create_context_builder = lambda: DummyContextBuilder()
context_builder_util.ensure_fresh_weights = lambda builder: None
sys.modules.setdefault("context_builder_util", context_builder_util)



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

import menace.self_improvement as sie


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
    assert 0.0 <= after_inc <= 10.0
    assert 0.0 <= after_dec <= 10.0
    learner2 = sie.SynergyWeightLearner(path=path)
    assert learner2.weights["roi"] == pytest.approx(after_dec)



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

    vals = []
    for roi_delta, deltas in cycles1:
        learner.update(roi_delta, deltas)
        vals.append(learner.weights["roi"])

    assert all(0.0 <= v <= 10.0 for v in vals)

    # persistence check after first set of updates
    learner2 = sie.SynergyWeightLearner(path=path, lr=lr)
    assert learner2.weights["roi"] == pytest.approx(vals[-1])

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

    vals2 = []
    for roi_delta, deltas in cycles2:
        learner2.update(roi_delta, deltas)
        vals2.append(learner2.weights["roi"])

    assert 0.0 <= vals2[-1] <= 10.0

    # persistence check after all updates
    learner3 = sie.SynergyWeightLearner(path=path, lr=lr)
    assert learner3.weights["roi"] == pytest.approx(vals2[-1])


def test_atomic_save_called(monkeypatch, tmp_path):
    path = tmp_path / "w.json"
    learner = sie.SynergyWeightLearner(path=path)
    called = []

    def fake(path_arg, data, *, binary=False):
        called.append(path_arg)

    monkeypatch.setattr(sie, "_atomic_write", fake)
    learner.save()
    assert path in called


def test_load_invalid_file_uses_defaults(tmp_path, caplog):
    path = tmp_path / "bad.json"
    path.write_text("{\"roi\": 1.0}")
    caplog.set_level("WARNING")
    learner = sie.SynergyWeightLearner(path=path)
    assert learner.weights == sie.get_default_synergy_weights()
    assert "invalid synergy weight data" in caplog.text


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


class _DummyTracker:
    def __init__(self, metrics):
        self.metrics_history = metrics


def test_weights_influence_roi(monkeypatch, tmp_path):
    path = tmp_path / "w.json"
    recs = [_Rec(0.5, 0.1), _Rec(0.6, 0.2)]
    metrics = {"synergy_roi": [0.1, 0.2]}
    engine = sie.SelfImprovementEngine(
        context_builder=DummyContextBuilder(),
        interval=0,
        patch_db=_DummyDB(recs),
        synergy_weights_path=path,
    )
    engine.tracker = _DummyTracker(metrics)
    adj_before = engine._weighted_synergy_adjustment()
    engine._update_synergy_weights(1.0)
    engine._synergy_cache = None
    adj_after = engine._weighted_synergy_adjustment()
    assert adj_before != adj_after


def test_dqn_synergy_learner_multi_cycle_sync(tmp_path):
    torch = pytest.importorskip("torch")
    import importlib

    # reload with real strategies when torch is available
    import menace.self_improvement as sie
    import menace.self_improvement_policy as sip
    sip = importlib.reload(sip)
    sie = importlib.reload(sie)

    path = tmp_path / "weights.json"
    learner = sie.DQNSynergyLearner(path=path, lr=0.01, target_sync=2)

    deltas = {
        "synergy_roi": 1.0,
        "synergy_efficiency": 0.5,
        "synergy_resilience": 0.1,
        "synergy_antifragility": -0.1,
        "synergy_reliability": 0.0,
        "synergy_maintainability": 0.0,
        "synergy_throughput": 0.0,
    }

    start = learner.weights["roi"]
    learner.update(1.0, deltas)

    def _equal(m1, m2):
        return all(torch.equal(a, b) for a, b in zip(m1.parameters(), m2.parameters()))

    # models diverge after first update
    assert not _equal(learner.strategy.model, learner.strategy.target_model)

    learner.update(1.0, deltas)
    # target sync on step 2
    assert _equal(learner.strategy.model, learner.strategy.target_model)

    learner.update(0.5, deltas)
    learner.update(-0.2, deltas)

    assert 0.0 <= learner.weights["roi"] <= 10.0
    assert learner.weights["roi"] != start

    learner2 = sie.DQNSynergyLearner(path=path)
    assert learner2.weights["roi"] == pytest.approx(learner.weights["roi"])


@pytest.mark.parametrize(
    "cls_name",
    [
        "DQNSynergyLearner",
        "DoubleDQNSynergyLearner",
        "SACSynergyLearner",
        "TD3SynergyLearner",
    ],
)
def test_sb3_learners_positive_convergence(tmp_path, cls_name):
    torch = pytest.importorskip("torch")
    import importlib

    import menace.self_improvement as sie
    import menace.self_improvement_policy as sip

    # reload modules with real RL strategies when torch is available
    sip = importlib.reload(sip)
    sie = importlib.reload(sie)

    path = tmp_path / f"{cls_name}.json"
    cls = getattr(sie, cls_name)
    learner = cls(path=path, lr=0.01, target_sync=1)

    deltas = {
        "synergy_roi": 1.0,
        "synergy_efficiency": 0.8,
        "synergy_resilience": 0.6,
        "synergy_antifragility": 0.4,
        "synergy_reliability": 0.2,
        "synergy_maintainability": 0.1,
        "synergy_throughput": 0.3,
    }

    start = learner.weights["roi"]
    for _ in range(5):
        learner.update(1.0, deltas)

    assert 0.0 <= learner.weights["roi"] <= 10.0
    assert learner.weights["roi"] >= start

    learner2 = cls(path=path)
    assert learner2.weights["roi"] == pytest.approx(learner.weights["roi"])


@pytest.mark.parametrize(
    "cls_name",
    [
        "DQNSynergyLearner",
        "DoubleDQNSynergyLearner",
        "SACSynergyLearner",
        "TD3SynergyLearner",
    ],
)
def test_sb3_learners_persistence_and_files(tmp_path, cls_name):
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    import importlib

    import menace.self_improvement as sie
    import menace.self_improvement_policy as sip

    sip = importlib.reload(sip)
    sie = importlib.reload(sie)

    path = tmp_path / f"{cls_name}.json"
    cls = getattr(sie, cls_name)
    learner = cls(path=path, lr=0.01, target_sync=1)

    deltas = {
        "synergy_roi": 0.5,
        "synergy_efficiency": 0.1,
        "synergy_resilience": 0.2,
        "synergy_antifragility": -0.1,
        "synergy_reliability": 0.0,
        "synergy_maintainability": 0.0,
        "synergy_throughput": 0.0,
    }

    start = dict(learner.weights)

    for roi_delta in [0.5, -0.3, 0.7]:
        learner.update(roi_delta, deltas)

    assert learner.weights != start

    learner2 = cls(path=path)
    assert learner2.weights == pytest.approx(learner.weights)

    base = path.with_suffix("")
    policy = base.with_suffix(".policy.pkl")
    model = base.with_suffix(".pt")
    target = base.with_suffix(".target.pt")

    assert policy.exists()
    assert model.exists() or target.exists()


def test_sac_custom_params_override(tmp_path):
    torch = pytest.importorskip("torch")
    import menace.self_improvement as sie

    path = tmp_path / "w.json"
    learner = sie.SACSynergyLearner(
        path=path,
        lr=0.01,
        hidden_sizes=[16, 8],
        noise=0.2,
        batch_size=4,
        target_sync=3,
    )
    assert learner.noise == 0.2
    assert learner.batch_size == 4
    assert learner.target_sync == 3
    assert isinstance(learner.actor[0], torch.nn.Linear)
    assert learner.actor[0].out_features == 16


def test_synergy_weight_logging_info(tmp_path, caplog):
    path = tmp_path / "w.json"
    caplog.set_level("INFO")
    learner = sie.SynergyWeightLearner(path=path)
    assert "loaded synergy weights" not in caplog.text

    caplog.clear()
    learner.save()
    assert "saved synergy weights" in caplog.text

    caplog.clear()
    learner.load()
    assert "loaded synergy weights" in caplog.text

    caplog.clear()
    learner.update(1.0, {"synergy_roi": 1.0})
    text = caplog.text
    assert "updated synergy weights" in text
    assert "saved synergy weights" in text


def test_synergy_weight_checkpoint_failure(tmp_path, monkeypatch, caplog):
    path = tmp_path / "w.json"
    settings = types.SimpleNamespace(
        synergy_weight_file=str(path),
        synergy_checkpoint_interval=1,
        synergy_python_fallback=True,
        synergy_python_max_replay=1000,
        synergy_train_interval=10,
        synergy_replay_size=10,
    )
    learner = sie.SynergyWeightLearner(path=path, lr=0.1, settings=settings)

    def fail_copy(*a, **k):
        raise PermissionError("denied")

    monkeypatch.setattr(sie.shutil, "copy", fail_copy)
    monkeypatch.setattr(sie.time, "sleep", lambda *_: None)

    with caplog.at_level("ERROR"):
        with pytest.raises(PermissionError):
            learner.save()

    assert "failed to checkpoint synergy weights" in caplog.text
