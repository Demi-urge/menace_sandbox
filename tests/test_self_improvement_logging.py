import os
import sys
import types
import importlib.util
import builtins
import pytest

# Create a minimal fake 'menace' package with required submodules
pkg = types.ModuleType("menace")
sys.modules["menace"] = pkg
pkg.__path__ = [os.path.dirname(os.path.dirname(__file__))]
pkg.RAISE_ERRORS = False

stub = types.ModuleType("menace.logging_utils")
stub.log_record = lambda **k: {}
sys.modules["menace.logging_utils"] = stub

# Simple stubs for modules referenced in self_improvement_engine
for name in [
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
    "self_test_service",
    "quick_fix_engine",
    "mutation_logger",
    "adaptive_roi_predictor",
    "adaptive_roi_dataset",
    "roi_tracker",
    "self_improvement_policy",
    "pre_execution_roi_bot",
    "env_config",
]:
    mod = types.ModuleType(f"menace.{name}")
    sys.modules[f"menace.{name}"] = mod

# minimal class factory
class _Dummy:
    pass

sys.modules["menace.research_aggregator_bot"].ResearchAggregatorBot = _Dummy
sys.modules["menace.research_aggregator_bot"].ResearchItem = _Dummy
sys.modules["menace.research_aggregator_bot"].InfoDB = _Dummy

sys.modules["menace.model_automation_pipeline"].ModelAutomationPipeline = _Dummy
sys.modules["menace.model_automation_pipeline"].AutomationResult = _Dummy

sys.modules["menace.diagnostic_manager"].DiagnosticManager = _Dummy
sys.modules["menace.error_bot"].ErrorBot = _Dummy
sys.modules["menace.error_bot"].ErrorDB = _Dummy
sys.modules["menace.data_bot"].MetricsDB = _Dummy
sys.modules["menace.data_bot"].DataBot = _Dummy
sys.modules["menace.code_database"].PatchHistoryDB = _Dummy
sys.modules["menace.capital_management_bot"].CapitalManagementBot = _Dummy
sys.modules["menace.learning_engine"].LearningEngine = _Dummy
sys.modules["menace.unified_event_bus"].UnifiedEventBus = _Dummy
sys.modules["menace.neuroplasticity"].PathwayRecord = _Dummy
sys.modules["menace.neuroplasticity"].Outcome = _Dummy
sys.modules["menace.self_coding_engine"].SelfCodingEngine = _Dummy
sys.modules["menace.action_planner"].ActionPlanner = _Dummy
sys.modules["menace.evolution_history_db"].EvolutionHistoryDB = _Dummy
sys.modules["menace.self_test_service"].SelfTestService = _Dummy
sys.modules["menace.quick_fix_engine"].generate_patch = lambda *a, **k: None
sys.modules["menace.mutation_logger"].log_patch = lambda *a, **k: None
sys.modules["menace.adaptive_roi_predictor"].AdaptiveROIPredictor = _Dummy
sys.modules["menace.adaptive_roi_predictor"].load_training_data = lambda *a, **k: None
sys.modules["menace.adaptive_roi_dataset"].build_dataset = lambda *a, **k: []
sys.modules["menace.roi_tracker"].ROITracker = _Dummy

# Populate needed attributes
sys.modules["menace.self_model_bootstrap"].bootstrap = lambda: None
sys.modules["menace.self_improvement_policy"].SelfImprovementPolicy = object
sys.modules["menace.self_improvement_policy"].ConfigurableSelfImprovementPolicy = object
class _DummyStrategy:
    def __init__(self, *a, **k):
        self.model = None
        self.target_model = None

    def update(self, *a, **k):
        return 0.0

    def predict(self, *_):
        return [0.0]

sys.modules["menace.self_improvement_policy"].DQNStrategy = _DummyStrategy
sys.modules["menace.self_improvement_policy"].DoubleDQNStrategy = _DummyStrategy
sys.modules["menace.self_improvement_policy"].ActorCriticStrategy = _DummyStrategy
sys.modules["menace.self_improvement_policy"].sip_torch = None
sys.modules["menace.self_improvement_policy"].torch = None

sys.modules["menace.pre_execution_roi_bot"].PreExecutionROIBot = object
sys.modules["menace.pre_execution_roi_bot"].BuildTask = object
sys.modules["menace.pre_execution_roi_bot"].ROIResult = object

sys.modules["menace.env_config"].PRE_ROI_SCALE = 1.0
sys.modules["menace.env_config"].PRE_ROI_BIAS = 0.0
sys.modules["menace.env_config"].PRE_ROI_CAP = 1.0

# top-level module not under menace
sandbox_settings = types.ModuleType("sandbox_settings")
sandbox_settings.SandboxSettings = lambda: types.SimpleNamespace(
    sandbox_score_db="db", synergy_weight_roi=1.0, synergy_weight_efficiency=1.0,
    synergy_weight_resilience=1.0, synergy_weight_antifragility=1.0,
    roi_ema_alpha=0.1, synergy_weights_lr=0.1
)
sys.modules["sandbox_settings"] = sandbox_settings


def _load_engine():
    spec = importlib.util.spec_from_file_location(
        "menace.self_improvement_engine",
        os.path.join(os.path.dirname(__file__), "..", "self_improvement_engine.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_synergy_weight_save_logs(monkeypatch, tmp_path, caplog):
    sie = _load_engine()
    learner = sie.SynergyWeightLearner(path=tmp_path / "w.json")

    def fail(*a, **k):
        raise OSError("boom")

    monkeypatch.setattr(sie.tempfile, "NamedTemporaryFile", fail)
    caplog.set_level("ERROR")
    learner.save()
    assert "failed to save synergy weights" in caplog.text


class DummyModel:
    def state_dict(self):
        return {}

    def load_state_dict(self, *_):
        raise OSError("load fail")


class DummyStrategy:
    def __init__(self):
        self.model = DummyModel()
        self.target_model = DummyModel()

    def _ensure_model(self, *_):
        pass

    def update(self, *a, **k):
        return 0.0

    def predict(self, *_):
        return [0.0] * 7


def test_dqn_load_logs(monkeypatch, tmp_path, caplog):
    sie = _load_engine()
    path = tmp_path / "w.json"
    path.write_text("{}")
    learner = sie.DQNSynergyLearner(path=path)
    learner.strategy = DummyStrategy()

    monkeypatch.setattr(sie.os.path, "exists", lambda p: True)
    monkeypatch.setattr(sie, "sip_torch", types.SimpleNamespace(load=lambda *a, **k: (_ for _ in ()).throw(OSError("torch fail"))))
    monkeypatch.setattr(sie.pickle, "load", lambda *a, **k: (_ for _ in ()).throw(OSError("pickle fail")))

    caplog.set_level("ERROR")
    learner.load()
    text = caplog.text
    assert "failed to load DQN models" in text
    assert "failed to load strategy pickle" in text


def test_dqn_update_logs(monkeypatch, caplog):
    sie = _load_engine()
    learner = sie.DQNSynergyLearner()
    learner.strategy = DummyStrategy()
    learner.target_sync = 1
    caplog.set_level("ERROR")
    learner.update(1.0, {"synergy_roi": 0.0, "synergy_efficiency": 0.0, "synergy_resilience": 0.0, "synergy_antifragility": 0.0})
    assert "target model sync failed" in caplog.text


def test_load_invalid_weights_logs_warning(tmp_path, caplog):
    sie = _load_engine()
    path = tmp_path / "w.json"
    path.write_text("{\"roi\": 1.0, \"efficiency\": \"bad\"}")
    caplog.set_level("WARNING")
    learner = sie.SynergyWeightLearner(path=path)
    assert learner.weights == sie.DEFAULT_SYNERGY_WEIGHTS
    assert "invalid synergy weight data" in caplog.text


def test_load_corrupted_weights_logs_warning(tmp_path, caplog):
    sie = _load_engine()
    path = tmp_path / "bad.json"
    path.write_text("{broken}")
    caplog.set_level("WARNING")
    learner = sie.SynergyWeightLearner(path=path)
    assert learner.weights == sie.DEFAULT_SYNERGY_WEIGHTS
    assert "failed to load synergy weights" in caplog.text


def test_flag_patch_alignment_logs_event(monkeypatch, tmp_path):
    sie = _load_engine()
    diff = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,2 +1 @@\n"
        "-logging.info('hi')\n"
        "+pass\n"
    )
    events = []

    class Bus:
        def publish(self, topic, record):
            events.append((topic, record))

    engine = types.SimpleNamespace(
        alignment_flagger=sie.HumanAlignmentFlagger(),
        cycle_logs=[],
        _cycle_count=0,
        event_bus=Bus(),
        logger=types.SimpleNamespace(exception=lambda *a, **k: None),
    )

    def fake_run(cmd, capture_output, text, check):
        if cmd == ["git", "show", "HEAD"]:
            return types.SimpleNamespace(stdout=diff)
        if cmd == ["git", "rev-parse", "HEAD"]:
            return types.SimpleNamespace(stdout="abc123\n")
        raise AssertionError("unexpected command")

    monkeypatch.setattr(sie.subprocess, "run", fake_run)
    import pathlib

    orig_path = pathlib.Path
    monkeypatch.setattr(sie, "Path", lambda p: orig_path(tmp_path / p))

    sie.SelfImprovementEngine._flag_patch_alignment(engine, 1, {})

    assert events and events[0][0] == "alignment:flag"
    assert engine.cycle_logs and engine.cycle_logs[0]["report"]["issues"]
    log_file = tmp_path / "sandbox_data" / "alignment_flags.jsonl"
    assert log_file.exists()


def teardown_module(module):
    for name in list(sys.modules):
        if name.startswith("menace"):
            sys.modules.pop(name, None)
    sys.modules.pop("sandbox_settings", None)
