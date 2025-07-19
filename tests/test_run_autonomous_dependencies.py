import importlib.util
import sys
import subprocess
import types
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "run_autonomous.py"
    sys.modules.pop("menace", None)
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_autonomous"] = mod
    spec.loader.exec_module(mod)
    return mod


def setup_startup(monkeypatch, missing_pkg="pkg"):
    sc_mod = types.ModuleType("menace.startup_checks")
    sc_mod.verify_project_dependencies = lambda: [missing_pkg]
    sc_mod._parse_requirement = lambda r: missing_pkg
    monkeypatch.setitem(sys.modules, "menace.startup_checks", sc_mod)
    eg_mod = types.ModuleType("menace.environment_generator")
    eg_mod.generate_presets = lambda n=None: [{}]
    monkeypatch.setitem(sys.modules, "menace.environment_generator", eg_mod)
    tracker_mod = types.ModuleType("menace.roi_tracker")
    class DummyTracker:
        def __init__(self, *a, **k):
            self.module_deltas = {}
            self.metrics_history = {}
        def load_history(self, p):
            pass
        def diminishing(self):
            return 0.0

    tracker_mod.ROITracker = DummyTracker
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", tracker_mod)


def test_retry_on_install_failure(monkeypatch):
    setup_startup(monkeypatch)
    mod = load_module()
    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))
    monkeypatch.setattr(mod.shutil, "which", lambda c: f"/usr/bin/{c}")
    monkeypatch.setattr(mod.importlib, "import_module", lambda n: types.ModuleType(n))
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    calls = []

    def failing_call(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, cmd)
        return 0

    monkeypatch.setattr(mod.subprocess, "check_call", failing_call)
    mod._check_dependencies()
    expected = [mod.sys.executable, "-m", "pip", "install", "pkg"]
    assert calls.count(expected) == 2


def test_abort_after_failed_installs(monkeypatch):
    setup_startup(monkeypatch)
    mod = load_module()
    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))
    monkeypatch.setattr(mod.shutil, "which", lambda c: f"/usr/bin/{c}")
    monkeypatch.setattr(mod.importlib, "import_module", lambda n: (_ for _ in ()).throw(ImportError()))
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(mod.subprocess, "check_call", lambda *a, **k: (_ for _ in ()).throw(subprocess.CalledProcessError(1, a[0])))
    with pytest.raises(RuntimeError):
        mod._check_dependencies()


def test_lock_file_used(monkeypatch, tmp_path):
    setup_startup(monkeypatch)
    lock = tmp_path / "uv.lock"
    lock.write_text("")
    monkeypatch.chdir(tmp_path)
    mod = load_module()
    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))
    monkeypatch.setattr(mod.shutil, "which", lambda c: f"/usr/bin/{c}")
    monkeypatch.setattr(mod.importlib, "import_module", lambda n: types.ModuleType(n))
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    calls = []
    monkeypatch.setattr(
        mod.subprocess,
        "check_call",
        lambda cmd, **k: calls.append(cmd) or 0,
    )
    mod._check_dependencies()
    assert [mod.sys.executable, "-m", "pip", "install", "-r", str(lock)] in calls

