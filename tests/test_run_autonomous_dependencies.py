import importlib.util
import sys
import subprocess
import types
from pathlib import Path
import collections
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "run_autonomous.py"  # path-ignore
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
    eg_mod = types.ModuleType("menace.environment_generator")
    eg_mod.generate_presets = lambda n=None: [{"CPU_LIMIT": "1", "MEMORY_LIMIT": "1"}]
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

    pkg = types.ModuleType("menace")
    pkg.__path__ = []
    pkg.startup_checks = sc_mod
    pkg.environment_generator = eg_mod
    pkg.roi_tracker = tracker_mod
    monkeypatch.setitem(sys.modules, "menace", pkg)
    monkeypatch.setitem(sys.modules, "menace.startup_checks", sc_mod)
    monkeypatch.setitem(sys.modules, "menace.environment_generator", eg_mod)
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", tracker_mod)

    sr_stub = types.ModuleType("sandbox_runner")
    cli_stub = types.ModuleType("sandbox_runner.cli")
    cli_stub.full_autonomous_run = lambda args: None
    cli_stub._diminishing_modules = lambda *a, **k: set()
    cli_stub.adaptive_synergy_convergence = lambda *a, **k: (True, 0.0, {})
    sr_stub._sandbox_main = lambda p, a: None
    sr_stub.cli = cli_stub
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_stub)


def test_retry_on_install_failure(monkeypatch):
    setup_startup(monkeypatch)
    mod = load_module()
    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))
    _patch_tools(monkeypatch, mod)
    di = sys.modules["menace.dependency_installer"]
    monkeypatch.setattr(di.importlib, "import_module", lambda n: types.ModuleType(n))
    monkeypatch.setattr(di.time, "sleep", lambda s: None)
    calls = []

    def failing_call(cmd, **kwargs):
        calls.append(cmd)
        if cmd[-1] == "pkg" and calls.count(cmd) == 1:
            raise subprocess.CalledProcessError(1, cmd)
        return 0

    monkeypatch.setattr(di.subprocess, "check_call", failing_call)
    mod._check_dependencies()
    expected = [mod.sys.executable, "-m", "pip", "install", "pkg"]
    assert calls.count(expected) == 2


def test_abort_after_failed_installs(monkeypatch):
    setup_startup(monkeypatch)
    mod = load_module()
    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))
    _patch_tools(monkeypatch, mod)
    di = sys.modules["menace.dependency_installer"]
    monkeypatch.setattr(di.importlib, "import_module", lambda n: (_ for _ in ()).throw(ImportError()))
    monkeypatch.setattr(di.time, "sleep", lambda s: None)
    monkeypatch.setattr(di.subprocess, "check_call", lambda *a, **k: (_ for _ in ()).throw(subprocess.CalledProcessError(1, a[0])))
    assert not mod._check_dependencies()


def test_lock_file_used(monkeypatch, tmp_path):
    setup_startup(monkeypatch)
    lock = tmp_path / "uv.lock"
    lock.write_text("")
    monkeypatch.chdir(tmp_path)
    mod = load_module()
    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))
    _patch_tools(monkeypatch, mod)
    di = sys.modules["menace.dependency_installer"]
    monkeypatch.setattr(di.importlib, "import_module", lambda n: types.ModuleType(n))
    monkeypatch.setattr(di.time, "sleep", lambda s: None)
    calls = []
    monkeypatch.setattr(
        di.subprocess,
        "check_call",
        lambda cmd, **k: calls.append(cmd) or 0,
    )
    mod._check_dependencies()
    assert [mod.sys.executable, "-m", "pip", "install", "-r", lock.name] in calls


def test_offline_mode_skips_install(monkeypatch):
    setup_startup(monkeypatch)
    mod = load_module()
    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))
    _patch_tools(monkeypatch, mod, git=False)
    di = sys.modules["menace.dependency_installer"]
    monkeypatch.setattr(di.importlib, "import_module", lambda n: (_ for _ in ()).throw(ImportError()))
    pip_calls = []
    monkeypatch.setattr(di.subprocess, "check_call", lambda *a, **k: pip_calls.append(a))
    apt_calls = []
    monkeypatch.setattr(mod.subprocess, "run", lambda cmd, check=False: apt_calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    monkeypatch.setenv("MENACE_OFFLINE_INSTALL", "1")
    monkeypatch.delenv("MENACE_WHEEL_DIR", raising=False)
    assert not mod._check_dependencies()
    assert pip_calls == []
    assert apt_calls == []


def test_offline_mode_wheel_dir(monkeypatch, tmp_path):
    setup_startup(monkeypatch)
    mod = load_module()
    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))
    _patch_tools(monkeypatch, mod, git=False)
    di = sys.modules["menace.dependency_installer"]
    monkeypatch.setattr(di.importlib, "import_module", lambda n: types.ModuleType(n))
    pip_calls = []
    monkeypatch.setattr(di.subprocess, "check_call", lambda cmd, **k: pip_calls.append(cmd) or 0)
    apt_calls = []
    monkeypatch.setattr(mod.subprocess, "run", lambda cmd, check=False: apt_calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    monkeypatch.setenv("MENACE_OFFLINE_INSTALL", "1")
    monkeypatch.setenv("MENACE_WHEEL_DIR", str(tmp_path))
    assert not mod._check_dependencies()
    assert pip_calls
    assert all("--find-links" in call for call in pip_calls)
    assert apt_calls == []


def test_install_missing_system_tool(monkeypatch):
    mod = _setup_base(monkeypatch)
    _patch_tools(monkeypatch, mod, git=False)
    calls = []

    def record(cmd, check=False):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(mod.subprocess, "run", record)
    mod._check_dependencies()
    assert ["apt-get", "install", "-y", "git"] in calls


def _setup_base(monkeypatch):
    """Return loaded module with startup stubs and patched dependencies."""
    setup_startup(monkeypatch)
    mod = load_module()
    monkeypatch.setattr(mod, "verify_project_dependencies", lambda: [])
    monkeypatch.setattr(mod, "install_packages", lambda *a, **k: {})
    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))
    return mod


def _patch_tools(monkeypatch, mod, *, git=True, pytest_tool=True, docker_group=True):
    def which(cmd):
        if cmd == "git" and not git:
            return None
        if cmd == "pytest" and not pytest_tool:
            return None
        return f"/usr/bin/{cmd}"

    monkeypatch.setattr(mod.shutil, "which", which)

    gid = 100
    grp_stub = types.SimpleNamespace(
        getgrnam=lambda n: types.SimpleNamespace(
            gr_mem=["user"] if docker_group else [], gr_gid=gid
        )
    )
    monkeypatch.setitem(sys.modules, "grp", grp_stub)
    monkeypatch.setitem(
        sys.modules, "getpass", types.SimpleNamespace(getuser=lambda: "user")
    )
    monkeypatch.setattr(mod.os, "getgid", lambda: gid if docker_group else gid + 1)


def test_python_version_requirement(monkeypatch):
    mod = _setup_base(monkeypatch)
    _patch_tools(monkeypatch, mod)
    VersionInfo = collections.namedtuple(
        "VersionInfo", "major minor micro releaselevel serial"
    )
    monkeypatch.setattr(mod.sys, "version_info", VersionInfo(3, 9, 0, "final", 0))
    assert not mod._check_dependencies()


def test_missing_system_tools(monkeypatch):
    mod = _setup_base(monkeypatch)
    _patch_tools(monkeypatch, mod, git=False, pytest_tool=False)
    VersionInfo = collections.namedtuple(
        "VersionInfo", "major minor micro releaselevel serial"
    )
    monkeypatch.setattr(mod.sys, "version_info", VersionInfo(3, 10, 0, "final", 0))
    assert not mod._check_dependencies()


def test_docker_group_membership(monkeypatch):
    mod = _setup_base(monkeypatch)
    _patch_tools(monkeypatch, mod, docker_group=False)
    VersionInfo = collections.namedtuple(
        "VersionInfo", "major minor micro releaselevel serial"
    )
    monkeypatch.setattr(mod.sys, "version_info", VersionInfo(3, 10, 0, "final", 0))
    assert not mod._check_dependencies()

