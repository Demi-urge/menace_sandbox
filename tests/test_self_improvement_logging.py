import os
import sys
import types
import importlib.util
import json
import logging

# flake8: noqa

# Create a minimal fake 'menace' package with required submodules
pkg = types.ModuleType("menace")
sys.modules["menace"] = pkg
pkg.__path__ = [os.path.dirname(os.path.dirname(__file__))]
pkg.RAISE_ERRORS = False

stub = types.ModuleType("menace.logging_utils")
stub.log_record = lambda **k: {}
stub.get_logger = lambda name=None: logging.getLogger(name)
stub.setup_logging = lambda *a, **k: None
sys.modules["menace.logging_utils"] = stub

# Lightweight stubs for modules pulling heavy dependencies
dmm = types.ModuleType("dynamic_module_mapper")
dmm.build_module_map = lambda *a, **k: {}
dmm.discover_module_groups = lambda *a, **k: {}
sys.modules["dynamic_module_mapper"] = dmm

oa = types.ModuleType("orphan_analyzer")
oa.classify_module = lambda *a, **k: "ok"
oa.analyze_redundancy = lambda *a, **k: {}
sys.modules["orphan_analyzer"] = oa

sandbox_runner = types.ModuleType("sandbox_runner")
env_mod = types.ModuleType("sandbox_runner.environment")
env_mod.auto_include_modules = lambda *a, **k: None
env_mod.run_workflow_simulations = lambda *a, **k: ([], {})
env_mod.try_integrate_into_workflows = lambda *a, **k: []
sandbox_runner.environment = env_mod
bootstrap_mod = types.ModuleType("sandbox_runner.bootstrap")
bootstrap_mod.initialize_autonomous_sandbox = lambda *a, **k: None
sandbox_runner.bootstrap = bootstrap_mod
sys.modules["sandbox_runner"] = sandbox_runner
sys.modules["sandbox_runner.environment"] = env_mod
sys.modules["sandbox_runner.bootstrap"] = bootstrap_mod

error_logger = types.ModuleType("menace.error_logger")
error_logger.TelemetryEvent = type("TelemetryEvent", (), {})
sys.modules["menace.error_logger"] = error_logger

sys.modules["menace.synergy_history_db"] = types.ModuleType("menace.synergy_history_db")

# Simple stubs for modules referenced in self_improvement
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
sys.modules["menace.self_model_bootstrap"].bootstrap = lambda *a, **k: None
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


class SandboxSettings(types.SimpleNamespace):
    def __init__(self):
        super().__init__(
            sandbox_score_db="db",
            synergy_weight_roi=1.0,
            synergy_weight_efficiency=1.0,
            synergy_weight_resilience=1.0,
            synergy_weight_antifragility=1.0,
            roi_ema_alpha=0.1,
            synergy_weights_lr=0.1,
            enable_alignment_flagger=True,
            alignment_warning_threshold=0.5,
            alignment_failure_threshold=0.9,
            patch_retries=3,
            patch_retry_delay=0.1,
        )


sandbox_settings.SandboxSettings = SandboxSettings
sandbox_settings.load_sandbox_settings = lambda: SandboxSettings()
sandbox_settings.DEFAULT_SEVERITY_SCORE_MAP = {}
sys.modules["sandbox_settings"] = sandbox_settings
sys.modules["menace.sandbox_settings"] = sandbox_settings


def _load_engine():
    spec = importlib.util.spec_from_file_location(
        "menace.self_improvement",
        os.path.join(os.path.dirname(__file__), "..", "self_improvement", "__init__.py"),  # path-ignore
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
    assert learner.weights == sie.get_default_synergy_weights()
    assert "invalid synergy weight data" in caplog.text


def test_load_corrupted_weights_logs_warning(tmp_path, caplog):
    sie = _load_engine()
    path = tmp_path / "bad.json"
    path.write_text("{broken}")
    caplog.set_level("WARNING")
    learner = sie.SynergyWeightLearner(path=path)
    assert learner.weights == sie.get_default_synergy_weights()
    assert "failed to load synergy weights" in caplog.text


def test_flag_improvement_logs_violation(monkeypatch, tmp_path):
    sie = _load_engine()
    from menace import violation_logger as vl

    log_path = tmp_path / "violation_log.jsonl"
    monkeypatch.setattr(vl, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(vl, "LOG_PATH", str(log_path))

    code = "import subprocess\nsubprocess.run('ls', shell=True)"
    warnings = sie.flag_improvement(
        workflow_changes=[{"file": "workflow.py", "code": code}],  # path-ignore
        metrics={},
        logs=[],
    )
    engine = types.SimpleNamespace(_cycle_count=0)
    sie.SelfImprovementEngine._log_improvement_warnings(engine, warnings)

    data = json.loads(log_path.read_text().splitlines()[0])
    assert data["evidence"]["file"] == "workflow.py"  # path-ignore
    assert "subprocess.run" in data["evidence"].get("snippet", "")


def test_flag_patch_alignment_logs_violation(monkeypatch, tmp_path):
    sie = _load_engine()
    from menace import violation_logger as vl

    log_path = tmp_path / "violation_log.jsonl"
    monkeypatch.setattr(vl, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(vl, "LOG_PATH", str(log_path))

    diff = (
        "diff --git a/foo.py b/foo.py\n"  # path-ignore
        "--- a/foo.py\n"  # path-ignore
        "+++ b/foo.py\n"  # path-ignore
        "@@ -1,2 +1 @@\n"
        "-logging.info('hi')\n"
        "+pass\n"
    )

    class Bus:
        def publish(self, topic, record):
            pass

    engine = types.SimpleNamespace(
        alignment_flagger=sie.HumanAlignmentFlagger(),
        cycle_logs=[],
        _cycle_count=0,
        event_bus=Bus(),
        logger=types.SimpleNamespace(exception=lambda *a, **k: None, info=lambda *a, **k: None),
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

    data = json.loads(log_path.read_text().splitlines()[0])
    assert data["evidence"]["file"] == "foo.py"  # path-ignore
    assert "Logging removed" in data["evidence"].get("snippet", "")


def test_flag_patch_alignment_logs_event(monkeypatch, tmp_path):
    sie = _load_engine()
    diff = (
        "diff --git a/foo.py b/foo.py\n"  # path-ignore
        "--- a/foo.py\n"  # path-ignore
        "+++ b/foo.py\n"  # path-ignore
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
        logger=types.SimpleNamespace(exception=lambda *a, **k: None, info=lambda *a, **k: None),
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
    assert "score" in events[0][1] and events[0][1]["score"] > 0
    assert engine.cycle_logs and engine.cycle_logs[0]["score"] > 0
    log_file = tmp_path / "sandbox_data" / "alignment_flags.jsonl"
    assert log_file.exists()


def test_flag_patch_alignment_dispatches_warning(monkeypatch, tmp_path):
    sie = _load_engine()
    diff = (
        "diff --git a/foo.py b/foo.py\n"  # path-ignore
        "--- a/foo.py\n"  # path-ignore
        "+++ b/foo.py\n"  # path-ignore
        "@@ -1,2 +1 @@\n"
        "-logging.info('hi')\n"
        "+pass\n"
    )
    dispatched = []

    def fake_dispatch(record):
        dispatched.append(record)

    monkeypatch.setattr(sie.security_auditor, "dispatch_alignment_warning", fake_dispatch)

    class Bus:
        def publish(self, topic, record):
            pass

    engine = types.SimpleNamespace(
        alignment_flagger=sie.HumanAlignmentFlagger(),
        cycle_logs=[],
        _cycle_count=0,
        event_bus=Bus(),
        logger=types.SimpleNamespace(exception=lambda *a, **k: None, info=lambda *a, **k: None),
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

    assert dispatched and dispatched[0]["patch_id"] == 1
    assert dispatched[0]["score"] == dispatched[0]["report"]["score"]


def test_flag_patch_alignment_dispatch_failure_does_not_raise(monkeypatch, tmp_path):
    sie = _load_engine()
    diff = (
        "diff --git a/foo.py b/foo.py\n"  # path-ignore
        "--- a/foo.py\n"  # path-ignore
        "+++ b/foo.py\n"  # path-ignore
        "@@ -1,2 +1 @@\n"
        "-logging.info('hi')\n"
        "+pass\n"
    )

    def fail_dispatch(record):
        raise RuntimeError("boom")

    monkeypatch.setattr(sie.security_auditor, "dispatch_alignment_warning", fail_dispatch)

    engine = types.SimpleNamespace(
        alignment_flagger=sie.HumanAlignmentFlagger(),
        cycle_logs=[],
        _cycle_count=0,
        event_bus=None,
        logger=types.SimpleNamespace(exception=lambda *a, **k: None, info=lambda *a, **k: None),
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

    assert engine.cycle_logs and engine.cycle_logs[0]["patch_id"] == 1
    assert engine.cycle_logs[0]["score"] == engine.cycle_logs[0]["report"]["score"]


def test_flag_patch_alignment_escalates_high_severity(monkeypatch, tmp_path):
    sie = _load_engine()
    diff = (
        "diff --git a/foo_test.py b/foo_test.py\n"  # path-ignore
        "--- a/foo_test.py\n"  # path-ignore
        "+++ b/foo_test.py\n"  # path-ignore
        "@@ -1,5 +1 @@\n"
        "-\"\"\"Doc\"\"\"\n"
        "-import logging\n"
        "-def test_something():\n"
        "-    logging.info(\"hi\")\n"
        "-    assert 1 == 1\n"
        "+pass\n"
    )
    alerts: list[tuple[tuple, dict]] = []

    def fake_alert(*a, **k):
        alerts.append((a, k))

    monkeypatch.setattr(sie, "dispatch_alert", fake_alert)
    monkeypatch.setattr(sie.security_auditor, "dispatch_alignment_warning", lambda record: None)

    class Bus:
        def publish(self, topic, record):
            pass

    engine = types.SimpleNamespace(
        alignment_flagger=sie.HumanAlignmentFlagger(),
        cycle_logs=[],
        _cycle_count=0,
        event_bus=Bus(),
        logger=types.SimpleNamespace(exception=lambda *a, **k: None, info=lambda *a, **k: None),
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

    assert alerts and alerts[0][0][0] == "alignment_review"
    log_file = tmp_path / "sandbox_data" / "alignment_flags.jsonl"
    data = json.loads(log_file.read_text().splitlines()[0])
    assert data.get("escalated")


def test_flag_patch_alignment_disabled(monkeypatch, tmp_path):
    sie = _load_engine()
    diff = (
        "diff --git a/foo.py b/foo.py\n"  # path-ignore
        "--- a/foo.py\n"  # path-ignore
        "+++ b/foo.py\n"  # path-ignore
        "@@ -1,2 +1 @@\n"
        "-logging.info('hi')\n"
        "+pass\n"
    )
    dispatched: list = []
    alerts: list = []

    monkeypatch.setattr(sie.security_auditor, "dispatch_alignment_warning", lambda record: dispatched.append(record))
    monkeypatch.setattr(sie, "dispatch_alert", lambda *a, **k: alerts.append((a, k)))
    sie.SandboxSettings = lambda: types.SimpleNamespace(
        enable_alignment_flagger=False,
        alignment_warning_threshold=0.5,
        alignment_failure_threshold=0.9,
    )

    engine = types.SimpleNamespace(
        alignment_flagger=sie.HumanAlignmentFlagger(),
        cycle_logs=[],
        _cycle_count=0,
        event_bus=None,
        logger=types.SimpleNamespace(exception=lambda *a, **k: None, info=lambda *a, **k: None),
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

    assert not dispatched
    assert not alerts
    log_file = tmp_path / "sandbox_data" / "alignment_flags.jsonl"
    assert not log_file.exists()


def test_flag_patch_alignment_threshold_escalation(monkeypatch, tmp_path):
    sie = _load_engine()
    diff = (
        "diff --git a/foo.py b/foo.py\n"  # path-ignore
        "--- a/foo.py\n"  # path-ignore
        "+++ b/foo.py\n"  # path-ignore
        "@@ -1,2 +1 @@\n"
        "-logging.info('hi')\n"
        "+pass\n"
    )
    alerts: list = []

    def fake_alert(*a, **k):
        alerts.append((a, k))

    monkeypatch.setattr(sie, "dispatch_alert", fake_alert)
    monkeypatch.setattr(sie.security_auditor, "dispatch_alignment_warning", lambda record: None)

    class Bus:
        def publish(self, topic, record):
            pass

    engine = types.SimpleNamespace(
        alignment_flagger=sie.HumanAlignmentFlagger(),
        cycle_logs=[],
        _cycle_count=0,
        event_bus=Bus(),
        logger=types.SimpleNamespace(exception=lambda *a, **k: None, info=lambda *a, **k: None),
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

    sie.SandboxSettings = lambda: types.SimpleNamespace(
        enable_alignment_flagger=True,
        alignment_warning_threshold=0.4,
        alignment_failure_threshold=0.6,
    )
    sie.SelfImprovementEngine._flag_patch_alignment(engine, 1, {})
    assert not alerts

    alerts.clear()
    sie.SandboxSettings = lambda: types.SimpleNamespace(
        enable_alignment_flagger=True,
        alignment_warning_threshold=0.4,
        alignment_failure_threshold=0.4,
    )
    sie.SelfImprovementEngine._flag_patch_alignment(engine, 2, {})
    assert alerts and alerts[0][0][0] == "alignment_review"


def test_alignment_review_agent_dispatches_quick_fix_warnings(monkeypatch, tmp_path):
    sie = _load_engine()
    from menace import violation_logger, quick_fix_engine, security_auditor
    import menace.alignment_review_agent as ara

    monkeypatch.setattr(violation_logger, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(
        violation_logger, "LOG_PATH", str(tmp_path / "violations.jsonl")
    )

    def gen_patch(module, engine=None, **kwargs):
        violation_logger.log_violation(
            "quick_fix_1",
            "alignment_warning",
            2,
            {"file": module},
            alignment_warning=True,
        )
        return 1

    monkeypatch.setattr(quick_fix_engine, "generate_patch", gen_patch)

    quick_fix_engine.generate_patch("foo.py", context_builder=object())  # path-ignore

    dispatched: list = []
    monkeypatch.setattr(
        security_auditor,
        "dispatch_alignment_warning",
        lambda record: dispatched.append(record),
    )

    agent = ara.AlignmentReviewAgent(
        interval=0,
        auditor=types.SimpleNamespace(
            audit=security_auditor.dispatch_alignment_warning
        ),
    )

    def fake_load(limit: int = 50):
        warnings = violation_logger.load_recent_alignment_warnings(limit)
        agent._stop.set()
        return warnings

    monkeypatch.setattr(ara, "load_recent_alignment_warnings", fake_load)

    agent._run()

    assert dispatched and dispatched[1]["entry_id"] == "quick_fix_1"


def teardown_module(module):
    for name in list(sys.modules):
        if name.startswith("menace"):
            sys.modules.pop(name, None)
    sys.modules.pop("sandbox_settings", None)
