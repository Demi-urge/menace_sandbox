import importlib.util
import argparse
import os
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
    env_gen.adapt_presets = types.SimpleNamespace(last_actions=[])
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
    sym_mon.ExporterMonitor = lambda *a, **k: types.SimpleNamespace(
        start=lambda: None, stop=lambda: None, restart_count=0
    )
    sym_mon.AutoTrainerMonitor = lambda *a, **k: types.SimpleNamespace(
        start=lambda: None, stop=lambda: None, restart_count=0
    )
    monkeypatch.setitem(sys.modules, "synergy_monitor", sym_mon)
    srm = types.ModuleType("sandbox_recovery_manager")
    srm.SandboxRecoveryManager = lambda *a, **k: types.SimpleNamespace(
        run=lambda *a, **k: None, sandbox_main=None
    )
    monkeypatch.setitem(sys.modules, "sandbox_recovery_manager", srm)

    sr_mod = types.ModuleType("sandbox_runner")
    sr_mod.__path__ = []
    cli_mod = types.ModuleType("sandbox_runner.cli")
    cli_mod.full_autonomous_run = lambda args, **k: None
    sr_mod.cli = cli_mod
    sr_mod._sandbox_main = lambda p, a: None
    br_mod = types.ModuleType("sandbox_runner.bootstrap")
    br_mod.bootstrap_environment = lambda s, v: s
    br_mod._verify_required_dependencies = lambda: None
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_mod)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_mod)
    monkeypatch.setitem(sys.modules, "sandbox_runner.bootstrap", br_mod)

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

    path = ROOT / "run_autonomous.py"  # path-ignore
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
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)
    monkeypatch.setattr(mod, "validate_presets", lambda p: p)
    return mod


def _load_module_capture(monkeypatch, capture):
    mod = _load_module(monkeypatch)
    sr_mod = sys.modules["sandbox_runner"]

    class DummyTester:
        def __init__(self, *a, **kw):
            capture.update(kw)

    def _sandbox_main(preset, args):
        DummyTester(
            None,
            include_orphans=False,
            discover_orphans=False,
            discover_isolated=os.getenv("SANDBOX_DISCOVER_ISOLATED") == "1",
            recursive_orphans=os.getenv("SANDBOX_RECURSIVE_ORPHANS") == "1",
            recursive_isolated=os.getenv("SANDBOX_RECURSIVE_ISOLATED") == "1",
        )

    sr_mod._sandbox_main = _sandbox_main
    sr_mod.cli.full_autonomous_run = lambda args, **k: _sandbox_main({}, args)
    monkeypatch.setattr(mod, "validate_presets", lambda p: p)
    return mod


def test_invalid_roi_cycles_warns(monkeypatch, caplog):
    stub_env = types.ModuleType("sandbox_runner.environment")
    stub_env.SANDBOX_ENV_PRESETS = [{}]
    stub_env.simulate_full_environment = lambda preset: None
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", stub_env)
    from sandbox_runner.cli import full_autonomous_run
    args = argparse.Namespace(max_iterations=0)
    monkeypatch.setenv("ROI_CYCLES", "foo")
    caplog.set_level(logging.WARNING, logger="sandbox_runner.cli")
    full_autonomous_run(args)
    assert "invalid ROI_CYCLES" in caplog.text


def test_invalid_synergy_cycles_warns(monkeypatch, caplog):
    stub_env = types.ModuleType("sandbox_runner.environment")
    stub_env.SANDBOX_ENV_PRESETS = [{}]
    stub_env.simulate_full_environment = lambda preset: None
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", stub_env)
    from sandbox_runner.cli import full_autonomous_run
    args = argparse.Namespace(max_iterations=0)
    monkeypatch.setenv("SYNERGY_CYCLES", "bar")
    caplog.set_level(logging.WARNING, logger="sandbox_runner.cli")
    full_autonomous_run(args)
    assert "invalid SYNERGY_CYCLES" in caplog.text


def test_main_exits_when_required_env_missing(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SANDBOX_REPO_PATH", raising=False)
    with pytest.raises(SystemExit) as exc:
        mod.main(["--runs", "0"])
    msg = str(exc.value)
    assert "SANDBOX_REPO_PATH" in msg


def test_recursion_defaults_enabled(monkeypatch, tmp_path):
    capture = {}
    mod = _load_module_capture(monkeypatch, capture)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("VISUAL_AGENT_TOKEN", "x")
    monkeypatch.setenv("VISUAL_AGENT_AUTOSTART", "0")
    monkeypatch.delenv("SANDBOX_RECURSIVE_ISOLATED", raising=False)
    monkeypatch.delenv("SELF_TEST_RECURSIVE_ISOLATED", raising=False)
    mod.main(["--runs", "0", "--check-settings"])
    sys.modules["sandbox_runner"]._sandbox_main({}, argparse.Namespace())
    assert capture.get("discover_isolated") is True
    assert capture.get("recursive_orphans") is True
    assert capture.get("recursive_isolated") is True
    assert os.getenv("SANDBOX_RECURSIVE_ORPHANS") == "1"
    assert os.getenv("SELF_TEST_RECURSIVE_ORPHANS") == "1"
    assert os.getenv("SANDBOX_RECURSIVE_ISOLATED") == "1"
    assert os.getenv("SELF_TEST_RECURSIVE_ISOLATED") == "1"
    assert os.getenv("SANDBOX_DISCOVER_ISOLATED") == "1"


def test_no_recursive_flags_disable_recursion(monkeypatch, tmp_path):
    capture = {}
    mod = _load_module_capture(monkeypatch, capture)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("VISUAL_AGENT_TOKEN", "x")
    monkeypatch.setenv("VISUAL_AGENT_AUTOSTART", "0")
    mod.main([
        "--runs",
        "0",
        "--check-settings",
        "--no-recursive-include",
        "--no-recursive-isolated",
    ])
    sys.modules["sandbox_runner"]._sandbox_main({}, argparse.Namespace())
    assert capture.get("discover_isolated") is True
    assert capture.get("recursive_orphans") is False
    assert capture.get("recursive_isolated") is False
    assert os.getenv("SANDBOX_RECURSIVE_ORPHANS") == "0"
    assert os.getenv("SELF_TEST_RECURSIVE_ORPHANS") == "0"
    assert os.getenv("SANDBOX_RECURSIVE_ISOLATED") == "0"
    assert os.getenv("SELF_TEST_RECURSIVE_ISOLATED") == "0"


def test_recursive_isolated_flag_enables_recursion(monkeypatch, tmp_path):
    capture = {}
    mod = _load_module_capture(monkeypatch, capture)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("VISUAL_AGENT_TOKEN", "x")
    monkeypatch.setenv("VISUAL_AGENT_AUTOSTART", "0")
    monkeypatch.setenv("SANDBOX_RECURSIVE_ISOLATED", "0")
    monkeypatch.setenv("SELF_TEST_RECURSIVE_ISOLATED", "0")
    mod.main(["--runs", "0", "--check-settings", "--recursive-isolated"])
    sys.modules["sandbox_runner"]._sandbox_main({}, argparse.Namespace())
    assert capture.get("recursive_isolated") is True
    assert os.getenv("SANDBOX_RECURSIVE_ISOLATED") == "1"
    assert os.getenv("SELF_TEST_RECURSIVE_ISOLATED") == "1"


def test_auto_include_isolated_sets_flags(monkeypatch, tmp_path):
    capture = {}
    mod = _load_module_capture(monkeypatch, capture)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("VISUAL_AGENT_TOKEN", "x")
    monkeypatch.setenv("VISUAL_AGENT_AUTOSTART", "0")
    monkeypatch.delenv("SANDBOX_RECURSIVE_ISOLATED", raising=False)
    monkeypatch.delenv("SELF_TEST_RECURSIVE_ISOLATED", raising=False)
    monkeypatch.setenv("SANDBOX_AUTO_INCLUDE_ISOLATED", "1")
    mod.main(["--runs", "0", "--check-settings"])
    sys.modules["sandbox_runner"]._sandbox_main({}, argparse.Namespace())
    assert capture.get("recursive_isolated") is True
    assert os.getenv("SANDBOX_DISCOVER_ISOLATED") == "1"
    assert os.getenv("SANDBOX_RECURSIVE_ISOLATED") == "1"
    assert os.getenv("SELF_TEST_AUTO_INCLUDE_ISOLATED") == "1"
    assert os.getenv("SELF_TEST_DISCOVER_ISOLATED") == "1"
    assert os.getenv("SELF_TEST_RECURSIVE_ISOLATED") == "1"


def test_auto_include_isolated_flag(monkeypatch, tmp_path):
    capture = {}
    mod = _load_module_capture(monkeypatch, capture)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("VISUAL_AGENT_TOKEN", "x")
    monkeypatch.setenv("VISUAL_AGENT_AUTOSTART", "0")
    monkeypatch.delenv("SANDBOX_RECURSIVE_ISOLATED", raising=False)
    monkeypatch.delenv("SELF_TEST_RECURSIVE_ISOLATED", raising=False)
    mod.main(["--runs", "0", "--check-settings", "--auto-include-isolated"])
    sys.modules["sandbox_runner"]._sandbox_main({}, argparse.Namespace())
    assert os.getenv("SANDBOX_AUTO_INCLUDE_ISOLATED") == "1"
    assert capture.get("recursive_isolated") is True
    assert os.getenv("SANDBOX_DISCOVER_ISOLATED") == "1"
    assert os.getenv("SANDBOX_RECURSIVE_ISOLATED") == "1"
    assert os.getenv("SELF_TEST_AUTO_INCLUDE_ISOLATED") == "1"
    assert os.getenv("SELF_TEST_DISCOVER_ISOLATED") == "1"
    assert os.getenv("SELF_TEST_RECURSIVE_ISOLATED") == "1"


def test_no_discover_isolated_flag_disables(monkeypatch, tmp_path):
    capture = {}
    mod = _load_module_capture(monkeypatch, capture)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("VISUAL_AGENT_TOKEN", "x")
    monkeypatch.setenv("VISUAL_AGENT_AUTOSTART", "0")
    mod.main(["--runs", "0", "--check-settings", "--no-discover-isolated"])
    sys.modules["sandbox_runner"]._sandbox_main({}, argparse.Namespace())
    assert capture.get("discover_isolated") is False
    assert os.getenv("SANDBOX_DISCOVER_ISOLATED") == "0"


def test_discover_isolated_flag_overrides_env(monkeypatch, tmp_path):
    capture = {}
    mod = _load_module_capture(monkeypatch, capture)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("VISUAL_AGENT_TOKEN", "x")
    monkeypatch.setenv("VISUAL_AGENT_AUTOSTART", "0")
    monkeypatch.setenv("SANDBOX_DISCOVER_ISOLATED", "0")
    mod.main(["--runs", "0", "--check-settings", "--discover-isolated"])
    sys.modules["sandbox_runner"]._sandbox_main({}, argparse.Namespace())
    assert capture.get("discover_isolated") is True
    assert os.getenv("SANDBOX_DISCOVER_ISOLATED") == "1"
