import argparse
import importlib.util
import sys
import subprocess
import json
import pytest
from pathlib import Path

import types

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "sandbox_recovery_manager", str(ROOT / "sandbox_recovery_manager.py")  # path-ignore
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


def test_run_with_context_builder(monkeypatch):
    builder = object()
    seen = {}

    def succeed(preset, args, ctx):
        seen["ctx"] = ctx
        return "ok"

    monkeypatch.setattr(srm.time, "sleep", lambda s: None)
    mgr = srm.SandboxRecoveryManager(succeed, retry_delay=0)
    result = mgr.run({}, argparse.Namespace(), builder)

    assert result == "ok"
    assert seen["ctx"] is builder


def test_logging_and_callback(monkeypatch, tmp_path):
    calls = []

    def fail_then_ok(preset, args):
        calls.append("call")
        if len(calls) == 1:
            raise RuntimeError("boom")
        return "ok"

    cb_calls = []

    def cb(exc, runtime):
        cb_calls.append((str(exc), runtime))

    monotonic_counter = [0]

    def fake_monotonic():
        monotonic_counter[0] += 1
        return monotonic_counter[0]

    monkeypatch.setattr(srm.time, "sleep", lambda s: None)
    monkeypatch.setattr(srm.time, "monotonic", fake_monotonic)

    mgr = srm.SandboxRecoveryManager(fail_then_ok, retry_delay=0, on_retry=cb)
    args = argparse.Namespace(sandbox_data_dir=str(tmp_path))
    result = mgr.run({}, args)

    assert result == "ok"
    assert len(calls) == 2
    assert cb_calls and "boom" in cb_calls[0][0]
    log_content = (tmp_path / "recovery.log").read_text()
    assert "RuntimeError: boom" in log_content


def test_metrics_property(monkeypatch):
    def fail_then_ok(preset, args):
        if not getattr(fail_then_ok, "called", False):
            fail_then_ok.called = True
            raise RuntimeError("boom")
        return "ok"

    monkeypatch.setattr(srm.time, "sleep", lambda s: None)
    mgr = srm.SandboxRecoveryManager(fail_then_ok, retry_delay=0)
    result = mgr.run({}, argparse.Namespace())
    assert result == "ok"
    metrics = mgr.metrics
    assert metrics["sandbox_restart_total"] == 1.0
    assert isinstance(metrics["sandbox_last_failure_ts"], float)


def test_gauge_updates_on_failure(monkeypatch, tmp_path):
    stub = types.ModuleType("metrics_exporter")
    stub._USING_STUB = False

    class DummyGauge:
        def __init__(self, *a, **k):
            self.values = {}
            self.unlabelled = []

        def set(self, v):
            self.unlabelled.append(v)

        def labels(self, **labels):
            key = tuple(sorted(labels.items()))
            self.values.setdefault(key, [])
            def _set(val):
                self.values[key].append(val)
            return types.SimpleNamespace(set=_set)

    stub.Gauge = DummyGauge
    stub.CollectorRegistry = object
    stub.sandbox_restart_total = DummyGauge()
    stub.sandbox_last_failure_ts = DummyGauge()
    monkeypatch.setitem(sys.modules, "metrics_exporter", stub)

    calls = []

    def fail_twice_then_ok(preset, args):
        calls.append("call")
        if len(calls) <= 2:
            raise RuntimeError("boom")
        return "ok"

    monkeypatch.setattr(srm.time, "sleep", lambda s: None)
    mgr = srm.SandboxRecoveryManager(fail_twice_then_ok, retry_delay=0)
    mgr.run({}, argparse.Namespace(sandbox_data_dir=str(tmp_path)))

    key = (("reason", "RuntimeError"), ("service", "fail_twice_then_ok"))
    assert stub.sandbox_restart_total.values[key] == [1.0, 2.0]
    assert len(stub.sandbox_last_failure_ts.unlabelled) == 2
    assert not (tmp_path / "recovery.json").exists()


def test_json_fallback_and_cli(monkeypatch, tmp_path, capsys):
    stub = types.ModuleType("metrics_exporter")
    stub._USING_STUB = True
    class DummyGauge:
        def __init__(self, *a, **k):
            pass
        def set(self, v):
            pass
    stub.Gauge = DummyGauge
    stub.CollectorRegistry = object
    stub.sandbox_restart_total = DummyGauge()
    stub.sandbox_last_failure_ts = DummyGauge()
    monkeypatch.setitem(sys.modules, "metrics_exporter", stub)

    def fail_then_ok(preset, args):
        if not getattr(fail_then_ok, "called", False):
            fail_then_ok.called = True
            raise RuntimeError("boom")
        return "ok"

    monkeypatch.setattr(srm.time, "sleep", lambda s: None)
    mgr = srm.SandboxRecoveryManager(fail_then_ok, retry_delay=0)
    args = argparse.Namespace(sandbox_data_dir=str(tmp_path))
    mgr.run({}, args)

    data = json.loads((tmp_path / "recovery.json").read_text())
    assert data["sandbox_restart_total"] == 1.0
    assert isinstance(data["sandbox_last_failure_ts"], float)

    metrics = srm.load_metrics(tmp_path / "recovery.json")
    assert metrics["sandbox_restart_total"] == 1.0
    assert isinstance(metrics["sandbox_last_failure_ts"], float)


def test_json_metrics_multiple_failures(monkeypatch, tmp_path):
    stub = types.ModuleType("metrics_exporter")
    stub._USING_STUB = True

    class DummyGauge:
        def __init__(self, *a, **k):
            pass

        def set(self, v):
            pass

    stub.Gauge = DummyGauge
    stub.CollectorRegistry = object
    stub.sandbox_restart_total = DummyGauge()
    stub.sandbox_last_failure_ts = DummyGauge()
    monkeypatch.setitem(sys.modules, "metrics_exporter", stub)

    calls = []

    def fail_twice_then_ok(preset, args):
        calls.append("call")
        if len(calls) <= 2:
            raise RuntimeError("boom")
        return "ok"

    monkeypatch.setattr(srm.time, "sleep", lambda s: None)
    mgr = srm.SandboxRecoveryManager(fail_twice_then_ok, retry_delay=0)
    args = argparse.Namespace(sandbox_data_dir=str(tmp_path))
    mgr.run({}, args)

    data = json.loads((tmp_path / "recovery.json").read_text())
    assert data["sandbox_restart_total"] == 2.0
    assert isinstance(data["sandbox_last_failure_ts"], float)


def test_run_autonomous_integration(monkeypatch, tmp_path):
    pytest.skip("requires full environment")
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    pkg = types.ModuleType("menace")
    pkg.__path__ = [str(ROOT / "menace")]
    sys.modules["menace"] = pkg
    # Stub heavy imports before loading modules
    sc_mod = types.ModuleType("menace.startup_checks")
    sc_mod.verify_project_dependencies = lambda: ["pkg"]
    sc_mod._parse_requirement = lambda r: "pkg"
    monkeypatch.setitem(sys.modules, "menace.startup_checks", sc_mod)

    env_mod = types.ModuleType("menace.environment_generator")
    env_mod.generate_presets = lambda n=None: [{}]
    monkeypatch.setitem(sys.modules, "menace.environment_generator", env_mod)

    env_mod = types.ModuleType("menace.environment_generator")
    env_mod.generate_presets = lambda n=None: [{}]
    monkeypatch.setitem(sys.modules, "menace.environment_generator", env_mod)

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
    cli_stub._diminishing_modules = lambda *a, **k: (set(), None)
    cli_stub._ema = lambda seq: (0.0, [])
    cli_stub._adaptive_threshold = lambda *a, **k: 0.0
    cli_stub._adaptive_synergy_threshold = lambda *a, **k: 0.0
    cli_stub._synergy_converged = lambda *a, **k: (True, 0.0, {})
    sr_stub.cli = cli_stub
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_stub)

    path = ROOT / "run_autonomous.py"  # path-ignore
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    run_autonomous = importlib.util.module_from_spec(spec)
    sys.modules["run_autonomous"] = run_autonomous
    auto_env = types.ModuleType("menace.auto_env_setup")
    auto_env.ensure_env = lambda path=None: None
    monkeypatch.setitem(sys.modules, "menace.auto_env_setup", auto_env)
    import pydantic.class_validators as cv
    orig_validator = cv.validator
    def _validator(*fields, **kw):
        kw.setdefault("allow_reuse", True)
        return orig_validator(*fields, **kw)
    monkeypatch.setattr(cv, "validator", _validator)
    spec.loader.exec_module(run_autonomous)
    run_autonomous.validate_presets = lambda p: p
    run_autonomous.validate_presets = lambda p: p

    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))
    monkeypatch.setattr(run_autonomous.shutil, "which", lambda c: f"/usr/bin/{c}")
    monkeypatch.setattr(run_autonomous.importlib, "import_module", lambda n: types.ModuleType(n))

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
    pytest.skip("requires full environment")
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    pkg = types.ModuleType("menace")
    pkg.__path__ = [str(ROOT / "menace")]
    sys.modules["menace"] = pkg
    sc_mod = types.ModuleType("menace.startup_checks")
    sc_mod.verify_project_dependencies = lambda: ["pkg"]
    sc_mod._parse_requirement = lambda r: "pkg"
    monkeypatch.setitem(sys.modules, "menace.startup_checks", sc_mod)

    env_mod = types.ModuleType("menace.environment_generator")
    env_mod.generate_presets = lambda n=None: [{}]
    monkeypatch.setitem(sys.modules, "menace.environment_generator", env_mod)

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
    cli_stub._diminishing_modules = lambda *a, **k: (set(), None)
    cli_stub._ema = lambda seq: (0.0, [])
    cli_stub._adaptive_threshold = lambda *a, **k: 0.0
    cli_stub._adaptive_synergy_threshold = lambda *a, **k: 0.0
    cli_stub._synergy_converged = lambda *a, **k: (True, 0.0, {})
    sr_stub = types.ModuleType("sandbox_runner")
    sr_stub._sandbox_main = lambda p, a: None
    sr_stub.cli = cli_stub
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_stub)

    path = ROOT / "run_autonomous.py"  # path-ignore
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    run_autonomous = importlib.util.module_from_spec(spec)
    sys.modules["run_autonomous"] = run_autonomous
    auto_env = types.ModuleType("menace.auto_env_setup")
    auto_env.ensure_env = lambda path=None: None
    monkeypatch.setitem(sys.modules, "menace.auto_env_setup", auto_env)
    import pydantic.class_validators as cv
    orig_validator = cv.validator
    def _validator(*fields, **kw):
        kw.setdefault("allow_reuse", True)
        return orig_validator(*fields, **kw)
    monkeypatch.setattr(cv, "validator", _validator)
    spec.loader.exec_module(run_autonomous)

    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))
    monkeypatch.setattr(run_autonomous.shutil, "which", lambda c: f"/usr/bin/{c}")
    monkeypatch.setattr(run_autonomous.importlib, "import_module", lambda n: types.ModuleType(n))

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
