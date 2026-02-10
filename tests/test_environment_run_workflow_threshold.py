import pytest
import importlib.util
import sys
import types
from pathlib import Path


def _load_env():
    if "filelock" not in sys.modules:
        class DummyLock:
            def __init__(self):
                self.is_locked = False
                self.lock_file = ""

            def acquire(self):
                self.is_locked = True

            def release(self):
                self.is_locked = False

            def __enter__(self):
                self.acquire()
                return self

            def __exit__(self, exc_type, exc, tb):
                self.release()

        sys.modules["filelock"] = types.SimpleNamespace(
            FileLock=lambda *a, **k: DummyLock(), Timeout=Exception
        )

    pkg_path = Path(__file__).resolve().parents[1] / "sandbox_runner"
    if "sandbox_runner" not in sys.modules:
        pkg = types.ModuleType("sandbox_runner")
        pkg.__path__ = [str(pkg_path)]
        sys.modules["sandbox_runner"] = pkg
    path = pkg_path / "environment.py"
    spec = importlib.util.spec_from_file_location("sandbox_runner.environment", path)
    env = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner.environment"] = env
    assert spec and spec.loader
    spec.loader.exec_module(env)  # type: ignore[attr-defined]
    return env


class _NoGetTracker:
    def __init__(self):
        self.metrics_history = {"synergy": [2.0, 4.0, 6.0]}
        self.roi_history = []

    def get(self, *_a, **_k):
        raise AssertionError("tracker.get should not be used for synergy threshold")

    def std(self, *_a, **_k):
        raise AssertionError("tracker.std should not be used for synergy threshold")

    def diminishing(self):
        return 0.0

    def forecast(self):
        return 0.0, (0.0, 0.0)

    def record_metric_prediction(self, *a, **k):
        return None

    def update(self, *a, **k):
        self.roi_history.append(0.0)
        return 0.0, [], False


class _WorkflowRecord:
    def __init__(self, workflow, wid=1, **_k):
        self.workflow = workflow
        self.wid = wid


class _WorkflowDB:
    def __init__(self, *_a, **_k):
        pass

    def fetch(self):
        return [_WorkflowRecord(["simple_functions:print_ten"], wid=1)]


class _ContextBuilder:
    def refresh_db_weights(self):
        return None


def test_run_workflow_simulations_uses_synergy_history_not_get(monkeypatch, tmp_path):
    env = _load_env()

    monkeypatch.setitem(
        sys.modules,
        "menace.task_handoff_bot",
        types.SimpleNamespace(WorkflowDB=_WorkflowDB, WorkflowRecord=_WorkflowRecord),
    )
    monkeypatch.setitem(
        sys.modules,
        "menace.self_debugger_sandbox",
        types.SimpleNamespace(SelfDebuggerSandbox=type("S", (), {"analyse_and_fix": lambda self: None})),
    )
    monkeypatch.setitem(
        sys.modules,
        "menace.self_coding_engine",
        types.SimpleNamespace(SelfCodingEngine=object),
    )
    monkeypatch.setitem(sys.modules, "menace.code_database", types.SimpleNamespace(CodeDB=object))
    monkeypatch.setitem(
        sys.modules,
        "menace.menace_memory_manager",
        types.SimpleNamespace(MenaceMemoryManager=object),
    )

    async def fake_section_worker(snippet, env_input, threshold, runner_config=None):
        return {"exit_code": 0}, [], {}

    monkeypatch.setattr(env, "simulate_execution_environment", lambda *_a, **_k: {"risk_flags_triggered": []})
    monkeypatch.setattr(env, "get_error_logger", lambda *_a, **_k: None)
    monkeypatch.setattr(env, "_section_worker", fake_section_worker)
    monkeypatch.setattr(env, "_resolve_roi_tracker_class", lambda: _NoGetTracker)

    tracker = _NoGetTracker()
    result = env.run_workflow_simulations(
        workflows_db=str(tmp_path / "wf.db"),
        env_presets=[{"SCENARIO_NAME": "dev"}],
        tracker=tracker,
        module_threshold=None,
        context_builder=_ContextBuilder(),
    )

    assert result is tracker


def test_environment_self_debugger_import_prefers_package(monkeypatch):
    env = _load_env()

    class PackageSandbox:
        pass

    monkeypatch.setitem(
        sys.modules,
        "menace.self_debugger_sandbox",
        types.SimpleNamespace(SelfDebuggerSandbox=PackageSandbox),
    )
    monkeypatch.delitem(sys.modules, "self_debugger_sandbox", raising=False)

    resolved = env._resolve_self_debugger_sandbox_class()
    assert resolved is PackageSandbox


def test_environment_self_debugger_import_falls_back_to_flat(monkeypatch):
    env = _load_env()

    class FlatSandbox:
        pass

    monkeypatch.delitem(sys.modules, "menace.self_debugger_sandbox", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "self_debugger_sandbox",
        types.SimpleNamespace(SelfDebuggerSandbox=FlatSandbox),
    )

    resolved = env._resolve_self_debugger_sandbox_class()
    assert resolved is FlatSandbox


def test_environment_self_debugger_import_error_includes_both_paths(monkeypatch):
    env = _load_env()

    monkeypatch.setitem(sys.modules, "menace.self_debugger_sandbox", None)
    monkeypatch.setitem(sys.modules, "self_debugger_sandbox", None)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        env._resolve_self_debugger_sandbox_class()

    msg = str(exc_info.value)
    assert "menace.self_debugger_sandbox" in msg
    assert "self_debugger_sandbox" in msg
