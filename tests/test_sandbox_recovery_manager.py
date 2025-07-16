import argparse
import importlib.util
import sys
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
    sc_mod.verify_project_dependencies = lambda: []
    monkeypatch.setitem(sys.modules, "menace.startup_checks", sc_mod)

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
    sr_stub.cli = cli_stub
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_stub)

    path = ROOT / "run_autonomous.py"
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    run_autonomous = importlib.util.module_from_spec(spec)
    sys.modules["run_autonomous"] = run_autonomous
    spec.loader.exec_module(run_autonomous)

    monkeypatch.setattr(run_autonomous, "_check_dependencies", lambda: None)
    monkeypatch.setattr(run_autonomous, "generate_presets", lambda n: [{}])
    monkeypatch.setattr(run_autonomous, "full_autonomous_run", cli_stub.full_autonomous_run)

    run_autonomous.main([])
    assert len(calls) == 2


def test_run_autonomous_multiple_runs(monkeypatch):
    sc_mod = types.ModuleType("menace.startup_checks")
    sc_mod.verify_project_dependencies = lambda: []
    monkeypatch.setitem(sys.modules, "menace.startup_checks", sc_mod)

    runs = []

    cli_stub = types.ModuleType("sandbox_runner.cli")
    cli_stub.full_autonomous_run = lambda args: runs.append("run")
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

    monkeypatch.setattr(run_autonomous, "_check_dependencies", lambda: None)
    monkeypatch.setattr(run_autonomous, "generate_presets", lambda n: [{}])
    monkeypatch.setattr(run_autonomous, "full_autonomous_run", cli_stub.full_autonomous_run)

    run_autonomous.main(["--runs", "2"])
    assert runs == ["run", "run"]
