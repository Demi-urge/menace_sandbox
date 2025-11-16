import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(monkeypatch):
    path = ROOT / "run_autonomous.py"  # path-ignore
    sys.modules.pop("menace", None)
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/true")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_autonomous"] = mod
    spec.loader.exec_module(mod)
    return mod


def setup_stubs(monkeypatch):
    import types

    sc_mod = types.ModuleType("menace.startup_checks")
    sc_mod.verify_project_dependencies = lambda: []
    sc_mod._parse_requirement = lambda r: r
    eg_mod = types.ModuleType("menace.environment_generator")
    eg_mod.generate_presets = lambda n=None: [{}]
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
    monkeypatch.setitem(sys.modules, "menace.startup_checks", sc_mod)
    monkeypatch.setitem(sys.modules, "menace.environment_generator", eg_mod)
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", tracker_mod)

    sr_stub = types.ModuleType("sandbox_runner")
    cli_stub = types.ModuleType("sandbox_runner.cli")
    cli_stub.full_autonomous_run = lambda args, **k: None
    cli_stub._diminishing_modules = lambda *a, **k: (set(), None)
    cli_stub._ema = lambda seq: (0.0, [])
    cli_stub._adaptive_threshold = lambda *a, **k: 0.0
    cli_stub._adaptive_synergy_threshold = lambda *a, **k: 0.0
    cli_stub._synergy_converged = lambda *a, **k: (True, 0.0, {})
    sr_stub._sandbox_main = lambda p, a: None
    sr_stub.cli = cli_stub
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_stub)


def _setup_dashboard(monkeypatch, started):
    import types
    dash_mod = types.ModuleType("menace.metrics_dashboard")

    class DummyDash:
        def __init__(self, *a, **k):
            pass

        def run(self, port=0):
            started["port"] = port

    dash_mod.MetricsDashboard = DummyDash
    monkeypatch.setitem(sys.modules, "menace.metrics_dashboard", dash_mod)

    import threading

    class DummyThread:
        def __init__(self, target=None, kwargs=None, daemon=None):
            self.target = target
            self.kwargs = kwargs or {}

        def start(self):
            if self.target:
                self.target(**self.kwargs)

    monkeypatch.setattr(threading, "Thread", DummyThread)


def test_env_dashboard(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    mod = load_module(monkeypatch)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)
    monkeypatch.setattr(mod, "generate_presets", lambda n=None: [{}])
    monkeypatch.setattr(mod, "validate_presets", lambda p: p)
    monkeypatch.setattr(mod, "full_autonomous_run", lambda args, **k: None)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    class DummySettings:
        def __init__(self):
            self.sandbox_data_dir = str(tmp_path)
            self.sandbox_env_presets = None
            self.auto_dashboard_port = None
            self.save_synergy_history = True
            self.roi_cycles = None
            self.synergy_cycles = None
            self.roi_threshold = None
            self.synergy_threshold = None
            self.roi_confidence = None
            self.synergy_confidence = None
            self.synergy_threshold_window = None
            self.synergy_threshold_weight = None
            self.synergy_ma_window = None
            self.synergy_stationarity_confidence = None
            self.synergy_std_threshold = None
            self.synergy_variance_confidence = None
    monkeypatch.setattr(mod, "SandboxSettings", DummySettings)
    started = {}
    _setup_dashboard(monkeypatch, started)
    monkeypatch.setenv("AUTO_DASHBOARD_PORT", "1234")

    mod.main(["--no-recursive-orphans", "--no-recursive-isolated"])

    assert started.get("port") == 1234


def test_cli_overrides_env(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    mod = load_module(monkeypatch)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)
    monkeypatch.setattr(mod, "generate_presets", lambda n=None: [{}])
    monkeypatch.setattr(mod, "validate_presets", lambda p: p)
    monkeypatch.setattr(mod, "full_autonomous_run", lambda args, **k: None)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    class DummySettings:
        def __init__(self):
            self.sandbox_data_dir = str(tmp_path)
            self.sandbox_env_presets = None
            self.auto_dashboard_port = None
            self.save_synergy_history = True
            self.roi_cycles = None
            self.synergy_cycles = None
            self.roi_threshold = None
            self.synergy_threshold = None
            self.roi_confidence = None
            self.synergy_confidence = None
            self.synergy_threshold_window = None
            self.synergy_threshold_weight = None
            self.synergy_ma_window = None
            self.synergy_stationarity_confidence = None
            self.synergy_std_threshold = None
            self.synergy_variance_confidence = None
    monkeypatch.setattr(mod, "SandboxSettings", DummySettings)
    started = {}
    _setup_dashboard(monkeypatch, started)
    monkeypatch.setenv("AUTO_DASHBOARD_PORT", "1111")

    mod.main(["--dashboard-port", "9999", "--no-recursive-orphans", "--no-recursive-isolated"])

    assert started.get("port") == 9999


def test_no_dashboard(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    mod = load_module(monkeypatch)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)
    monkeypatch.setattr(mod, "generate_presets", lambda n=None: [{}])
    monkeypatch.setattr(mod, "validate_presets", lambda p: p)
    monkeypatch.setattr(mod, "full_autonomous_run", lambda args, **k: None)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    class DummySettings:
        def __init__(self):
            self.sandbox_data_dir = str(tmp_path)
            self.sandbox_env_presets = None
            self.auto_dashboard_port = None
            self.save_synergy_history = True
            self.roi_cycles = None
            self.synergy_cycles = None
            self.roi_threshold = None
            self.synergy_threshold = None
            self.roi_confidence = None
            self.synergy_confidence = None
            self.synergy_threshold_window = None
            self.synergy_threshold_weight = None
            self.synergy_ma_window = None
            self.synergy_stationarity_confidence = None
            self.synergy_std_threshold = None
            self.synergy_variance_confidence = None
    monkeypatch.setattr(mod, "SandboxSettings", DummySettings)
    started = {}
    _setup_dashboard(monkeypatch, started)

    mod.main(["--no-recursive-orphans", "--no-recursive-isolated"])

    assert started == {}
