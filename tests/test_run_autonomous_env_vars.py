import importlib.util
import sys
import types
from pathlib import Path
import logging
import shutil
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module(monkeypatch):
    # stub menace package and required submodules before import
    pkg = types.ModuleType("menace")
    pkg.__path__ = []
    auto_env = types.ModuleType("menace.auto_env_setup")
    auto_env.ensure_env = lambda p: None
    env_gen = types.ModuleType("menace.environment_generator")
    env_gen.generate_presets = lambda n=None: [{}]
    env_gen.generate_presets_from_history = lambda *a, **k: [{}]
    startup = types.ModuleType("menace.startup_checks")
    startup.verify_project_dependencies = lambda: []
    dep_inst = types.ModuleType("menace.dependency_installer")
    dep_inst.install_packages = lambda *a, **k: None
    roi_mod = types.ModuleType("menace.roi_tracker")
    roi_mod.ROITracker = lambda *a, **k: None

    monkeypatch.setitem(sys.modules, "menace", pkg)
    monkeypatch.setitem(sys.modules, "menace.auto_env_setup", auto_env)
    monkeypatch.setitem(sys.modules, "menace.environment_generator", env_gen)
    monkeypatch.setitem(sys.modules, "menace.startup_checks", startup)
    monkeypatch.setitem(sys.modules, "menace.dependency_installer", dep_inst)
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", roi_mod)
    synergy_exp = types.ModuleType("menace.synergy_exporter")
    synergy_exp.SynergyExporter = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "menace.synergy_exporter", synergy_exp)
    shd_mod = types.ModuleType("menace.synergy_history_db")
    shd_mod.migrate_json_to_db = lambda *a, **k: None
    shd_mod.insert_entry = lambda *a, **k: None
    shd_mod.connect_locked = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "menace.synergy_history_db", shd_mod)
    sym_mon = types.ModuleType("synergy_monitor")
    sym_mon.ExporterMonitor = lambda *a, **k: types.SimpleNamespace(start=lambda: None, stop=lambda: None, restart_count=0)
    sym_mon.AutoTrainerMonitor = lambda *a, **k: types.SimpleNamespace(start=lambda: None, stop=lambda: None, restart_count=0)
    monkeypatch.setitem(sys.modules, "synergy_monitor", sym_mon)
    srm = types.ModuleType("sandbox_recovery_manager")
    srm.SandboxRecoveryManager = lambda *a, **k: types.SimpleNamespace(run=lambda *a, **k: None, sandbox_main=None)
    monkeypatch.setitem(sys.modules, "sandbox_recovery_manager", srm)

    sr_mod = types.ModuleType("sandbox_runner")
    cli_mod = types.ModuleType("sandbox_runner.cli")
    cli_mod.full_autonomous_run = lambda args, **k: None
    sr_mod.cli = cli_mod
    sr_mod._sandbox_main = lambda p, a: None
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_mod)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_mod)

    if "filelock" not in sys.modules:
        fl = types.ModuleType("filelock")

        class DummyLock:
            def __init__(self, *a, **k):
                pass

            def acquire(self, timeout=0):
                pass

            def release(self):
                pass

        fl.FileLock = DummyLock
        fl.Timeout = RuntimeError
        monkeypatch.setitem(sys.modules, "filelock", fl)
    if "dotenv" not in sys.modules:
        dmod = types.ModuleType("dotenv")
        dmod.dotenv_values = lambda *a, **k: {}
        monkeypatch.setitem(sys.modules, "dotenv", dmod)
    if "pydantic" not in sys.modules:
        pyd = types.ModuleType("pydantic")
        pyd.BaseModel = object
        class _Root(object):
            @classmethod
            def __class_getitem__(cls, item):
                return cls

        pyd.RootModel = _Root
        pyd.ValidationError = type("ValidationError", (Exception,), {})
        pyd.validator = lambda *a, **k: (lambda f: f)
        pyd.BaseSettings = object
        pyd.Field = lambda default=None, **k: default
        monkeypatch.setitem(sys.modules, "pydantic", pyd)
    if "pydantic_settings" not in sys.modules:
        ps = types.ModuleType("pydantic_settings")
        ps.BaseSettings = object
        ps.SettingsConfigDict = dict
        monkeypatch.setitem(sys.modules, "pydantic_settings", ps)

    path = ROOT / "run_autonomous.py"
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "run_autonomous", mod)
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/true")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setitem(
        sys.modules,
        "menace.audit_trail",
        types.SimpleNamespace(AuditTrail=lambda *a, **k: None),
    )
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "_check_dependencies", lambda: True)
    return mod


def test_invalid_roi_cycles_warns(monkeypatch, tmp_path, caplog):
    mod = _load_module(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ROI_CYCLES", "foo")
    caplog.set_level(logging.WARNING)
    mod.main(["--runs", "0"])
    assert "Invalid ROI_CYCLES value: foo" in caplog.text


def test_invalid_synergy_cycles_warns(monkeypatch, tmp_path, caplog):
    mod = _load_module(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SYNERGY_CYCLES", "bar")
    caplog.set_level(logging.WARNING)
    mod.main(["--runs", "0"])
    assert "Invalid SYNERGY_CYCLES value: bar" in caplog.text


def test_main_exits_when_required_env_missing(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VISUAL_AGENT_TOKEN", raising=False)
    monkeypatch.delenv("SANDBOX_REPO_PATH", raising=False)
    with pytest.raises(SystemExit) as exc:
        mod.main(["--runs", "0"])
    msg = str(exc.value)
    assert "VISUAL_AGENT_TOKEN" in msg
    assert "SANDBOX_REPO_PATH" in msg
