import argparse
import importlib.util
import sys
import subprocess
from pathlib import Path

import types

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "sandbox_recovery_manager", str(ROOT / "sandbox_recovery_manager.py")
)
srm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srm)


def test_restart_on_failure(monkeypatch):
    calls = []

    def fail_then_ok(preset, args):
        calls.append("call")
        if len(calls) == 1:
            raise RuntimeError("boom")
        return "ok"

    monkeypatch.setattr(srm.time, "sleep", lambda s: None)
    mgr = srm.SandboxRecoveryManager(fail_then_ok, retry_delay=0)
    result = mgr.run({}, argparse.Namespace())
    assert result == "ok"
    assert len(calls) == 2


def test_run_autonomous_integration(monkeypatch, tmp_path):
    # Stub heavy imports before loading modules
    sc_mod = types.ModuleType("menace.startup_checks")
    sc_mod.verify_project_dependencies = lambda: ["pkg"]
    monkeypatch.setitem(sys.modules, "menace.startup_checks", sc_mod)

    tracker_mod = types.ModuleType("menace.roi_tracker")
    class DummyTracker:
        def __init__(self, *a, **k):
            self.module_deltas = {}
            self.metrics_history = {}
        def load_history(self, path):
            pass
        def diminishing(self):
            return 0.0

    tracker_mod.ROITracker = DummyTracker
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", tracker_mod)

    calls = []

    def fail_then_ok(preset, args):
        calls.append("call")
        if len(calls) == 1:
            raise RuntimeError("boom")
        return types.SimpleNamespace()

    sr_stub = types.ModuleType("sandbox_runner")
    sr_stub._sandbox_main = fail_then_ok
    cli_stub = types.ModuleType("sandbox_runner.cli")
    cli_stub.full_autonomous_run = lambda args: sr_stub._sandbox_main({}, args)
    cli_stub._diminishing_modules = lambda *a, **k: set()
    sr_stub.cli = cli_stub
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_stub)

    path = ROOT / "run_autonomous.py"
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    run_autonomous = importlib.util.module_from_spec(spec)
    sys.modules["run_autonomous"] = run_autonomous
    spec.loader.exec_module(run_autonomous)

    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))
    monkeypatch.setattr(run_autonomous.shutil, "which", lambda c: f"/usr/bin/{c}")

    pip_calls = []

    def fake_call(cmd, **kwargs):
        pip_calls.append(list(cmd))
        return 0

    monkeypatch.setattr(run_autonomous.subprocess, "check_call", fake_call)
    monkeypatch.setattr(run_autonomous, "generate_presets", lambda n: [{}])
    monkeypatch.setattr(run_autonomous, "full_autonomous_run", cli_stub.full_autonomous_run)

    run_autonomous.main([])
    assert len(calls) == 2
    assert [run_autonomous.sys.executable, "-m", "pip", "install", "pkg"] in pip_calls


def test_run_autonomous_multiple_runs(monkeypatch):
    sc_mod = types.ModuleType("menace.startup_checks")
    sc_mod.verify_project_dependencies = lambda: ["pkg"]
    monkeypatch.setitem(sys.modules, "menace.startup_checks", sc_mod)

    tracker_mod = types.ModuleType("menace.roi_tracker")
    class DummyTracker:
        def __init__(self, *a, **k):
            self.module_deltas = {}
            self.metrics_history = {}
        def load_history(self, path):
            pass
        def diminishing(self):
            return 0.0

    tracker_mod.ROITracker = DummyTracker
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", tracker_mod)

    runs = []

    cli_stub = types.ModuleType("sandbox_runner.cli")
    cli_stub.full_autonomous_run = lambda args: runs.append("run")
    cli_stub._diminishing_modules = lambda *a, **k: set()
    sr_stub = types.ModuleType("sandbox_runner")
    sr_stub._sandbox_main = lambda p, a: None
    sr_stub.cli = cli_stub
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_stub)

    path = ROOT / "run_autonomous.py"
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    run_autonomous = importlib.util.module_from_spec(spec)
    sys.modules["run_autonomous"] = run_autonomous
    spec.loader.exec_module(run_autonomous)

    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))
    monkeypatch.setattr(run_autonomous.shutil, "which", lambda c: f"/usr/bin/{c}")

    pip_calls = []

    def fake_call(cmd, **kwargs):
        pip_calls.append(list(cmd))
        return 0

    monkeypatch.setattr(run_autonomous.subprocess, "check_call", fake_call)
    monkeypatch.setattr(run_autonomous, "generate_presets", lambda n: [{}])
    monkeypatch.setattr(run_autonomous, "full_autonomous_run", cli_stub.full_autonomous_run)

    run_autonomous.main(["--runs", "2"])
    assert runs == ["run", "run"]
    assert [run_autonomous.sys.executable, "-m", "pip", "install", "pkg"] in pip_calls
