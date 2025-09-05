import importlib.util
import os
import sys
import types
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(monkeypatch):
    path = ROOT / "run_autonomous.py"  # path-ignore
    sys.modules.pop("menace", None)
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_autonomous"] = mod
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/true")
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
        pmod = types.ModuleType("pydantic")
        class BaseModel:
            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, v)
            def dict(self, *a, **k):
                return self.__dict__
            @classmethod
            def parse_obj(cls, obj):
                return cls(**obj)
        class BaseSettings:
            pass
        class ValidationError(Exception):
            pass
        def Field(default, **kw):
            return default
        def validator(*a, **k):
            def wrap(fn):
                return fn
            return wrap
        pmod.BaseModel = BaseModel
        class RootModel(BaseModel):
            @classmethod
            def __class_getitem__(cls, item):
                return cls
        pmod.RootModel = RootModel
        pmod.BaseSettings = BaseSettings
        pmod.ValidationError = ValidationError
        pmod.Field = Field
        pmod.validator = validator
        monkeypatch.setitem(sys.modules, "pydantic", pmod)
    if "pydantic_settings" not in sys.modules:
        ps_mod = types.ModuleType("pydantic_settings")
        ps_mod.BaseSettings = BaseSettings
        monkeypatch.setitem(sys.modules, "pydantic_settings", ps_mod)
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
            self.roi_history = []
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
    cli_stub.adaptive_synergy_convergence = lambda *a, **k: (True, 0.0, {})
    sr_stub._sandbox_main = lambda p, a: None
    sr_stub.cli = cli_stub
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_stub)
    monkeypatch.setenv("SAVE_SYNERGY_HISTORY", "0")


class DummyMonitor:
    def __init__(self, *a, **k):
        pass

    def start(self):
        pass

    def stop(self):
        pass


class DummyManager:
    def __init__(self):
        self.calls = []
        self.process = types.SimpleNamespace(poll=lambda: None, returncode=0)

    def restart_with_token(self, token):
        self.calls.append(token)

    def shutdown(self, timeout=5.0):
        pass


def test_token_rotate(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    mod = load_module(monkeypatch)
    import contextlib
    shd_stub = types.SimpleNamespace(
        connect_locked=lambda p: types.SimpleNamespace(
            execute=lambda *a, **k: None,
            fetchall=lambda: [],
            close=lambda: None,
        )
    )
    monkeypatch.setattr(mod, "shd", shd_stub, raising=False)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)
    monkeypatch.setattr(mod, "VisualAgentMonitor", DummyMonitor)
    monkeypatch.setattr(mod, "_visual_agent_running", lambda u: True)
    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)

    mgr = DummyManager()
    vamod = types.ModuleType("visual_agent_manager")
    vamod.VisualAgentManager = lambda *a, **k: mgr
    monkeypatch.setitem(sys.modules, "visual_agent_manager", vamod)

    monkeypatch.setenv("VISUAL_AGENT_AUTOSTART", "1")
    monkeypatch.setenv("VISUAL_AGENT_TOKEN", "tok1")
    monkeypatch.setenv("VISUAL_AGENT_TOKEN_ROTATE", "tok2")
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    mod.main([
        "--max-iterations",
        "1",
        "--runs",
        "2",
        "--preset-count",
        "1",
        "--sandbox-data-dir",
        str(tmp_path),
    ])

    assert mgr.calls == ["tok2"]
    assert os.environ.get("VISUAL_AGENT_TOKEN") == "tok2"
    assert os.environ.get("VISUAL_AGENT_TOKEN_ROTATE") is None
