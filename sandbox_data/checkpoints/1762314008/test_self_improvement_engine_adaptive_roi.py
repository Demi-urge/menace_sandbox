import os
import sys
import types
import importlib.util
from pathlib import Path
import asyncio
import pytest

def _load_engine():
    """Load self_improvement with heavy deps stubbed."""
    # Create minimal 'menace' package
    pkg = types.ModuleType("menace")
    sys.modules["menace"] = pkg
    pkg.__path__ = [os.path.dirname(os.path.dirname(__file__))]
    pkg.RAISE_ERRORS = False

    # Basic logging utils stub
    log_mod = types.ModuleType("menace.logging_utils")
    log_mod.log_record = lambda **k: {}
    log_mod.get_logger = lambda name: types.SimpleNamespace(
        info=lambda *a, **k: None,
        exception=lambda *a, **k: None,
        warning=lambda *a, **k: None,
    )
    log_mod.setup_logging = lambda: None
    log_mod.set_correlation_id = lambda _cid: None
    sys.modules["menace.logging_utils"] = log_mod

    modules = [
        "self_model_bootstrap",
        "research_aggregator_bot",
        "model_automation_pipeline",
        "diagnostic_manager",
        "error_bot",
        "data_bot",
        "code_database",
        "capital_management_bot",
        "learning_engine",
        "unified_event_bus",
        "neuroplasticity",
        "self_coding_engine",
        "action_planner",
        "evolution_history_db",
        "self_improvement_policy",
        "pre_execution_roi_bot",
        "env_config",
        "adaptive_roi_predictor",
        "roi_tracker",
        "self_test_service",
        "error_cluster_predictor",
        "quick_fix_engine",
        "error_logger",
        "mutation_logger",
        "synergy_history_db",
        "synergy_weight_cli",
        "patch_score_backend",
        "module_index_db",
    ]
    for name in modules:
        sys.modules[f"menace.{name}"] = types.ModuleType(f"menace.{name}")

    class Dummy:
        """Generic stand-in class."""

    class DummyAgg:
        def __init__(self, *a, **k):
            pass

    m = sys.modules
    m["menace.research_aggregator_bot"].ResearchAggregatorBot = DummyAgg
    m["menace.research_aggregator_bot"].ResearchItem = Dummy
    m["menace.research_aggregator_bot"].InfoDB = Dummy
    m["menace.model_automation_pipeline"].ModelAutomationPipeline = lambda *a, **k: Dummy()
    m["menace.model_automation_pipeline"].AutomationResult = Dummy
    m["menace.diagnostic_manager"].DiagnosticManager = lambda *a, **k: Dummy()
    m["menace.error_bot"].ErrorBot = lambda *a, **k: Dummy()
    m["menace.error_bot"].ErrorDB = lambda *a, **k: Dummy()
    m["menace.data_bot"].MetricsDB = Dummy
    m["menace.data_bot"].DataBot = Dummy
    m["menace.code_database"].PatchHistoryDB = Dummy
    m["menace.capital_management_bot"].CapitalManagementBot = Dummy
    m["menace.learning_engine"].LearningEngine = Dummy
    m["menace.unified_event_bus"].UnifiedEventBus = Dummy
    m["menace.neuroplasticity"].PathwayRecord = Dummy
    m["menace.neuroplasticity"].Outcome = Dummy
    m["menace.self_coding_engine"].SelfCodingEngine = Dummy
    m["menace.action_planner"].ActionPlanner = Dummy
    m["menace.evolution_history_db"].EvolutionHistoryDB = Dummy
    m["menace.self_improvement_policy"].SelfImprovementPolicy = Dummy
    m["menace.self_improvement_policy"].ConfigurableSelfImprovementPolicy = lambda *a, **k: Dummy()
    m["menace.self_improvement_policy"].DQNStrategy = lambda *a, **k: Dummy()
    m["menace.self_improvement_policy"].DoubleDQNStrategy = lambda *a, **k: Dummy()
    m["menace.self_improvement_policy"].ActorCriticStrategy = lambda *a, **k: Dummy()
    m["menace.self_improvement_policy"].sip_torch = None
    m["menace.self_improvement_policy"].torch = None
    m["menace.pre_execution_roi_bot"].PreExecutionROIBot = Dummy
    m["menace.pre_execution_roi_bot"].BuildTask = Dummy
    m["menace.pre_execution_roi_bot"].ROIResult = Dummy
    m["menace.env_config"].PRE_ROI_SCALE = 1.0
    m["menace.env_config"].PRE_ROI_BIAS = 0.0
    m["menace.env_config"].PRE_ROI_CAP = 1.0
    m["menace.self_model_bootstrap"].bootstrap = lambda *a, **k: 0
    m["menace.error_cluster_predictor"].ErrorClusterPredictor = Dummy
    m["menace.quick_fix_engine"].generate_patch = lambda *a, **k: ""
    m["menace.error_logger"].TelemetryEvent = Dummy
    m["menace.mutation_logger"].record_event = lambda *a, **k: None
    m["menace.self_test_service"].SelfTestService = Dummy
    m["menace.adaptive_roi_predictor"].AdaptiveROIPredictor = Dummy
    m["menace.roi_tracker"].ROITracker = Dummy
    m["menace.synergy_history_db"].SynergyHistoryDB = Dummy

    class DummyLearner:
        def __init__(self, *a, **k):
            self.weights = {
                "roi": 1.0,
                "efficiency": 1.0,
                "resilience": 1.0,
                "antifragility": 1.0,
                "reliability": 1.0,
                "maintainability": 1.0,
                "throughput": 1.0,
            }

        def save(self):
            pass

    m["menace.synergy_weight_cli"].SynergyWeightLearner = DummyLearner
    m["menace.patch_score_backend"].PatchScoreBackend = Dummy
    m["menace.patch_score_backend"].backend_from_url = lambda *a, **k: Dummy()
    m["menace.module_index_db"].ModuleIndexDB = lambda *a, **k: Dummy()

    # top-level helpers
    sandbox_settings = types.ModuleType("sandbox_settings")
    sandbox_settings.SandboxSettings = lambda: types.SimpleNamespace(
        sandbox_score_db="db",
        synergy_weight_roi=1.0,
        synergy_weight_efficiency=1.0,
        synergy_weight_resilience=1.0,
        synergy_weight_antifragility=1.0,
        synergy_weight_reliability=1.0,
        synergy_weight_maintainability=1.0,
        synergy_weight_throughput=1.0,
        roi_ema_alpha=0.1,
        synergy_weights_lr=0.1,
        adaptive_roi_prioritization=True,
        growth_multiplier_exponential=1.2,
        growth_multiplier_linear=1.0,
        growth_multiplier_marginal=0.8,
        sandbox_data_dir=".",
        patch_retries=3,
        patch_retry_delay=0.1,
    )
    sys.modules["sandbox_settings"] = sandbox_settings

    sys.modules["sandbox_runner"] = types.ModuleType("sandbox_runner")
    sys.modules["sandbox_runner.environment"] = types.ModuleType("environment")
    boot_mod = types.ModuleType("sandbox_runner.bootstrap")
    boot_mod.initialize_autonomous_sandbox = lambda *a, **k: None
    sys.modules["sandbox_runner.bootstrap"] = boot_mod
    sys.modules["sandbox_runner.cycle"] = types.ModuleType("sandbox_runner.cycle")
    sys.modules["analytics"] = types.ModuleType("analytics")
    sys.modules["analytics.adaptive_roi_model"] = types.ModuleType(
        "analytics.adaptive_roi_model"
    )
    ss_mod = types.ModuleType("sandbox_settings")
    class SandboxSettingsStub:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            return 1

    ss_mod.SandboxSettings = SandboxSettingsStub
    ss_mod.load_sandbox_settings = lambda *a, **k: SandboxSettingsStub()
    class ROISettings:
        def __init__(self, *a, **k):
            pass

    ss_mod.ROISettings = ROISettings
    sys.modules["sandbox_settings"] = ss_mod
    sys.modules["menace.sandbox_settings"] = ss_mod

    spec = importlib.util.spec_from_file_location(
        "menace.self_improvement",
        os.path.join(os.path.dirname(__file__), "..", "self_improvement.py"),  # path-ignore
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_predictor_invoked_in_feature_collection():
    sie = _load_engine()

    class Predictor:
        def __init__(self):
            self.called = False

        def predict(self, feats, horizon=None):
            self.called = True
            return [0.0], "linear", [], []

    eng = sie.SelfImprovementEngine.__new__(sie.SelfImprovementEngine)
    eng.roi_predictor = Predictor()
    eng.roi_history = [0.5]
    eng._collect_action_features()
    assert eng.roi_predictor.called


def test_drift_metrics_trigger_retraining(monkeypatch):
    from menace_sandbox import adaptive_roi_predictor as arp

    monkeypatch.setattr(arp.AdaptiveROIPredictor, "train", lambda self, *a, **k: setattr(self, "_model", object()))
    predictor = arp.AdaptiveROIPredictor(cv=0, param_grid={})

    called = {"train": False}

    def _train(self, *a, **k):
        called["train"] = True

    monkeypatch.setattr(predictor, "train", _train.__get__(predictor, arp.AdaptiveROIPredictor))

    class Tracker:
        def evaluate_model(self, **_):
            return 0.5, 0.2  # low accuracy, high MAE

    predictor.evaluate_model(Tracker(), accuracy_threshold=0.6, mae_threshold=0.1)
    assert called["train"]


def test_growth_class_alters_action_selection():
    sie = _load_engine()

    class Predictor:
        def __init__(self, growth):
            self.growth = growth

        def predict(self, feats, horizon=None):
            return [0.0], self.growth, [], []

    async def run_once(growth):
        eng = sie.SelfImprovementEngine.__new__(sie.SelfImprovementEngine)
        eng.interval = 0
        eng._cycle_running = False
        eng.capital_bot = None
        eng.use_adaptive_roi = True
        eng.roi_predictor = Predictor(growth)
        eng._collect_action_features = lambda: [[0.5, 0.0]]
        eng.baseline_margin = 0.0
        _BT_SPEC = importlib.util.spec_from_file_location(
            "baseline_tracker", Path(__file__).resolve().parents[1] / "self_improvement" / "baseline_tracker.py"  # path-ignore
        )
        baseline_tracker = importlib.util.module_from_spec(_BT_SPEC)
        assert _BT_SPEC and _BT_SPEC.loader
        _BT_SPEC.loader.exec_module(baseline_tracker)  # type: ignore[attr-defined]
        BaselineTracker = baseline_tracker.BaselineTracker
        eng.baseline_tracker = BaselineTracker(window=3, energy=[0.5])
        called = {"run": False}

        def run_cycle(self, *, energy=1):
            called["run"] = True

        eng.run_cycle = types.MethodType(run_cycle, eng)
        eng.logger = types.SimpleNamespace(info=lambda *a, **k: None, exception=lambda *a, **k: None)

        class Stop:
            def __init__(self):
                self.n = 0

            def is_set(self):
                self.n += 1
                return self.n > 1

        eng._stop_event = Stop()
        await eng._schedule_loop(energy=0.5)
        return called["run"]

    exp = asyncio.run(run_once("exponential"))
    marg = asyncio.run(run_once("marginal"))
    assert not exp and marg
