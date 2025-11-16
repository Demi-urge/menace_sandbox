import importlib.util
import json
import sys
import types
import shutil
from pathlib import Path
from dynamic_path_router import resolve_path

import pytest


def _load_run_autonomous(monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    pyd = types.ModuleType("pydantic")

    class _Base:
        @classmethod
        def parse_obj(cls, obj):
            inst = cls()
            inst.__root__ = obj
            return inst

    pyd.BaseModel = _Base
    pyd.ValidationError = Exception
    pyd.validator = lambda *a, **k: (lambda f: f)
    pyd.Field = lambda default=None, **k: default
    pyd.BaseSettings = object
    monkeypatch.setitem(sys.modules, "pydantic", pyd)

    ps = types.ModuleType("pydantic_settings")
    ps.BaseSettings = object
    ps.SettingsConfigDict = dict
    monkeypatch.setitem(sys.modules, "pydantic_settings", ps)

    fl = types.ModuleType("filelock")

    class _Lock:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    fl.FileLock = _Lock
    fl.Timeout = RuntimeError
    monkeypatch.setitem(sys.modules, "filelock", fl)

    auto_env = types.ModuleType("menace.auto_env_setup")
    auto_env.ensure_env = lambda p: None
    monkeypatch.setitem(sys.modules, "menace.auto_env_setup", auto_env)

    sc = types.ModuleType("menace.startup_checks")
    sc.verify_project_dependencies = lambda: []
    sc._parse_requirement = lambda r: r
    monkeypatch.setitem(sys.modules, "menace.startup_checks", sc)

    env_gen = types.ModuleType("menace.environment_generator")
    env_gen.generate_presets = lambda n=None: [{"CPU_LIMIT": "1"}]
    env_gen._CPU_LIMITS = ["1"]
    env_gen._MEMORY_LIMITS = ["1"]
    monkeypatch.setitem(sys.modules, "menace.environment_generator", env_gen)
    monkeypatch.setitem(sys.modules, "menace_sandbox.environment_generator", env_gen)

    tracker = types.ModuleType("menace.roi_tracker")

    class DummyTracker:
        def __init__(self, *a, **k):
            self.module_deltas = {}
            self.metrics_history = {}
            self.roi_history = []

    tracker.ROITracker = DummyTracker
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", tracker)
    monkeypatch.setitem(sys.modules, "menace_sandbox.roi_tracker", tracker)

    syn_mod = types.ModuleType("menace.synergy_exporter")
    syn_mod.SynergyExporter = object
    monkeypatch.setitem(sys.modules, "menace.synergy_exporter", syn_mod)
    monkeypatch.setitem(sys.modules, "menace_sandbox.synergy_exporter", syn_mod)

    rec = types.ModuleType("sandbox_recovery_manager")
    rec.SandboxRecoveryManager = object
    monkeypatch.setitem(sys.modules, "sandbox_recovery_manager", rec)

    env_stub = types.ModuleType("sandbox_runner.environment")
    env_stub.SANDBOX_ENV_PRESETS = [{}]
    env_stub.simulate_full_environment = lambda preset: DummyTracker()
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_stub)

    cli_spec = importlib.util.spec_from_file_location(
        "sandbox_runner.cli", str(resolve_path("sandbox_runner/cli.py"))  # path-ignore
    )
    cli_mod = importlib.util.module_from_spec(cli_spec)
    sys.modules["sandbox_runner.cli"] = cli_mod
    cli_spec.loader.exec_module(cli_mod)

    sr_pkg = types.ModuleType("sandbox_runner")
    sr_pkg.cli = cli_mod
    sr_pkg._sandbox_main = lambda p, a: DummyTracker()
    sr_pkg.__path__ = ["sandbox_runner"]
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_pkg)

    monkeypatch.setattr(shutil, "which", lambda x: "/usr/bin/" + x)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: types.SimpleNamespace())

    spec = importlib.util.spec_from_file_location(
        "run_autonomous", str(resolve_path("run_autonomous.py"))  # path-ignore
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_autonomous"] = mod
    spec.loader.exec_module(mod)
    return mod, cli_mod


def test_auto_threshold_convergence(monkeypatch, tmp_path):
    mod, cli = _load_run_autonomous(monkeypatch)
    hist_vals = [0.1, 0.05, 0.02, 0.01, 0.005]
    history = [{"synergy_roi": v} for v in hist_vals]
    data_file = Path(tmp_path) / "synergy_history.json"
    data_file.write_text(json.dumps(history))

    hist, ma = mod.load_previous_synergy(tmp_path)
    exp_ma = []
    vals = []
    for entry in history:
        vals.append(entry["synergy_roi"])
        ema, _ = cli._ema(vals)
        exp_ma.append({"synergy_roi": ema})
    assert hist == history
    assert ma == exp_ma

    thr = cli._adaptive_synergy_threshold(hist, 3)
    ok_auto, ema_auto, conf_auto = cli.adaptive_synergy_convergence(
        hist, 3, threshold=None, threshold_window=3
    )
    ok_fixed, ema_fixed, conf_fixed = cli.adaptive_synergy_convergence(
        hist, 3, threshold=thr, threshold_window=3
    )
    assert ok_auto is True and ok_fixed is True
    assert abs(ema_auto - ema_fixed) < 1e-9
    assert conf_auto == pytest.approx(conf_fixed)
    assert conf_auto > 0.9
