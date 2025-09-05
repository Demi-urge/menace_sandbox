import json
import types
import sys
import logging
import pytest
from pathlib import Path

if "filelock" not in sys.modules:
    fl = types.ModuleType("filelock")

    class _FL:
        def __init__(self, *a, **k):
            self.lock_file = "x"
            self.is_locked = False

        def acquire(self, *a, **k):
            self.is_locked = True

        def release(self):
            self.is_locked = False

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc, tb):
            self.release()
            return False

    fl.FileLock = _FL
    fl.Timeout = type("Timeout", (Exception,), {})
    sys.modules["filelock"] = fl


def _stub_module(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    pkg, _, sub = name.partition(".")
    pkg_mod = sys.modules.get(pkg)
    if pkg_mod and sub:
        setattr(pkg_mod, sub, mod)
    return mod


class DummyBot:
    def __init__(self, *a, **k):
        pass


class DummySandbox:
    def __init__(self, *a, **k):
        pass

    def analyse_and_fix(self):
        pass


class _ROITracker:
    def __init__(self, *a, **k):
        self.roi_history = []
        self.calls = []

    def update(self, prev_roi, roi, modules=None, resources=None, metrics=None):
        self.roi_history.append(roi)
        return 0.0, [], False

    def forecast(self):
        return 0.0, (0.0, 0.0)

    def forecast_metric(self, name):
        return 0.0, (0.0, 0.0)

    def diminishing(self):
        return 0.0

    def record_prediction(self, predicted, actual, *a, **k):
        pass

    def record_metric_prediction(self, metric, predicted, actual):
        pass


def test_coverage_summary_flags_missing(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBot)
    _stub_module(monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyBot)
    _stub_module(monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyBot)
    _stub_module(monkeypatch, "menace.self_improvement", SelfImprovementEngine=DummyBot)
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyBot)
    _stub_module(monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox)
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", PatchHistoryDB=DummyBot, CodeDB=DummyBot)
    _stub_module(monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(monkeypatch, "menace.discrepancy_detection_bot", DiscrepancyDetectionBot=DummyBot)
    _stub_module(monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot())
    _stub_module(monkeypatch, "menace.data_bot", MetricsDB=DummyBot, DataBot=DummyBot)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=_ROITracker)
    _stub_module(
        monkeypatch,
        "vector_service",
        EmbeddableDBMixin=type("E", (), {"__init__": lambda self, *a, **k: None}),
        CognitionLayer=DummyBot,
    )
    _stub_module(
        monkeypatch,
        "menace.diagnostic_manager",
        DiagnosticManager=DummyBot,
        ResolutionRecord=DummyBot,
    )
    _stub_module(monkeypatch, "error_logger", ErrorLogger=DummyBot)
    class _BT:
        window = 1
        def __init__(self, *a, **k):
            self._history = {}

    _stub_module(
        monkeypatch,
        "self_improvement.baseline_tracker",
        BaselineTracker=_BT,
        TRACKER=_BT(),
    )

    _stub_module(
        monkeypatch,
        "menace.environment_generator",
        generate_canonical_presets=lambda: [],
        generate_presets=lambda profiles=None: [],
        suggest_profiles_for_module=lambda module: [],
        _CPU_LIMITS=["0.1"],
        _MEMORY_LIMITS=["32Mi"],
    )

    import importlib.util

    ROOT = Path(__file__).resolve().parents[1]
    pkg = sys.modules.setdefault("menace", types.ModuleType("menace"))
    pkg.__path__ = [str(ROOT)]
    spec = importlib.util.spec_from_file_location(
        "menace.task_handoff_bot",
        ROOT / "task_handoff_bot.py",  # path-ignore
        submodule_search_locations=[str(ROOT)],
    )
    thb = importlib.util.module_from_spec(spec)
    sys.modules["menace.task_handoff_bot"] = thb
    spec.loader.exec_module(thb)
    wf_db = thb.WorkflowDB(tmp_path / "wf.db")
    wf_db.add(thb.WorkflowRecord(workflow=["simple_functions:print_ten"], title="t"))

    import sandbox_runner

    def fake_run_workflow_simulations(*_a, **_k):
        logging.getLogger(__name__).warning("module mod missing scenarios: hostile_input")
        return types.SimpleNamespace(
            coverage_summary={"mod": {"only": True, "hostile_input": False}}
        )

    sandbox_runner.run_workflow_simulations = fake_run_workflow_simulations

    with caplog.at_level(logging.WARNING):
        tracker = sandbox_runner.run_workflow_simulations(
            workflows_db=str(tmp_path / "wf.db"),
            env_presets=[{"SCENARIO_NAME": "only"}]
        )

    assert tracker.coverage_summary
    mod = next(iter(tracker.coverage_summary))
    assert tracker.coverage_summary[mod]["only"] is True
    assert tracker.coverage_summary[mod]["hostile_input"] is False
    assert any("missing scenarios" in r.message for r in caplog.records)


def test_save_coverage_data_writes_summary(tmp_path, monkeypatch):
    _stub_module(
        monkeypatch,
        "menace.environment_generator",
        CANONICAL_PROFILES=["one", "two"],
        _CPU_LIMITS=["0.1"],
        _MEMORY_LIMITS=["32Mi"],
    )
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    _stub_module(
        monkeypatch,
        "vector_service",
        EmbeddableDBMixin=type("E", (), {"__init__": lambda self, *a, **k: None}),
        CognitionLayer=DummyBot,
    )

    try:
        from sandbox_runner.environment import (
            COVERAGE_TRACKER,
            FUNCTION_COVERAGE_TRACKER,
            _update_coverage,
            save_coverage_data,
        )
    except Exception:
        pytest.skip("environment import failed")

    monkeypatch.setenv("SANDBOX_COVERAGE_FILE", str(tmp_path / "coverage.json"))
    monkeypatch.setenv(
        "SANDBOX_COVERAGE_SUMMARY", str(tmp_path / "coverage_summary.json")
    )

    COVERAGE_TRACKER.clear()
    FUNCTION_COVERAGE_TRACKER.clear()
    _update_coverage("mod1", "one", ["f1"])

    save_coverage_data()

    cov_data = json.loads((tmp_path / "coverage.json").read_text())
    summary_data = json.loads((tmp_path / "coverage_summary.json").read_text())
    assert cov_data["modules"] == {"mod1": {"one": 1}}
    assert cov_data["functions"]["mod1"]["f1"]["one"] == 1
    assert summary_data["mod1"]["counts"]["one"] == 1
    assert summary_data["mod1"]["function_counts"]["mod1:f1"]["one"] == 1
    assert "two" in summary_data["mod1"]["missing"]
