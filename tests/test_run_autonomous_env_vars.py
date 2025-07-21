import importlib.util
import sys
import types
from pathlib import Path
import logging

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

    sr_mod = types.ModuleType("sandbox_runner")
    cli_mod = types.ModuleType("sandbox_runner.cli")
    cli_mod.full_autonomous_run = lambda args: None
    sr_mod.cli = cli_mod
    sr_mod._sandbox_main = lambda p, a: None
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_mod)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_mod)

    path = ROOT / "run_autonomous.py"
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "run_autonomous", mod)
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
