import importlib.util
import sys
import types
import os
import shutil
from pathlib import Path

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

from tests.test_menace_master import _setup_mm_stubs, DummyBot, _stub_module

scipy_mod = types.ModuleType("scipy")
stats_stub = types.SimpleNamespace(pearsonr=lambda *a, **k: (0.0, 0.0), t=lambda *a, **k: 0.0)
scipy_mod.stats = stats_stub
sys.modules.setdefault("scipy", scipy_mod)
sys.modules.setdefault("scipy.stats", stats_stub)
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
yaml_mod = types.ModuleType("yaml")
yaml_mod.safe_load = lambda *a, **k: {}
yaml_mod.safe_dump = lambda *a, **k: ""
sys.modules.setdefault("yaml", yaml_mod)
np_mod = types.ModuleType("numpy")
np_mod.array = lambda *a, **k: []
sys.modules.setdefault("numpy", np_mod)
sys.modules.setdefault("networkx", types.ModuleType("networkx"))


class _Policy:
    def __init__(self, *a, **k):
        pass

    def save(self):
        pass


class _Audit:
    def __init__(self, *a, **k):
        pass

    def record(self, *a, **k):
        pass


class DummyTracker:
    def __init__(self, *a, **k):
        self.roi_history = []
        self.metrics_history = {}
        self.module_deltas = {}
        self.saved_paths = []

    def update(self, prev, roi, modules=None, resources=None, metrics=None):
        self.roi_history.append(roi)
        if metrics:
            for k, v in metrics.items():
                self.metrics_history.setdefault(k, []).append(v)
        return 0.0, [], False

    def forecast(self):
        return 0.0, (0.0, 0.0)

    def diminishing(self):
        return 0.0

    def record_prediction(self, *a, **k):
        pass

    def rolling_mae(self, window=None):
        return 0.0

    def load_history(self, path):
        pass

    def save_history(self, path):
        self.saved_paths.append(path)
        Path(path).write_text("{}")



ROOT = Path(__file__).resolve().parents[1]


def load_module(monkeypatch=None):
    path = ROOT / "run_autonomous.py"  # path-ignore
    sys.modules.pop("menace", None)
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_autonomous"] = mod
    if monkeypatch is not None:
        monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/true")
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
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
                def dict(self):
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
            pmod.BaseSettings = BaseSettings
            pmod.ValidationError = ValidationError
            pmod.Field = Field
            pmod.validator = validator
            monkeypatch.setitem(sys.modules, "pydantic", pmod)
        if "pydantic_settings" not in sys.modules:
            ps_mod = types.ModuleType("pydantic_settings")
            ps_mod.BaseSettings = BaseSettings
            monkeypatch.setitem(sys.modules, "pydantic_settings", ps_mod)
        br_mod = types.ModuleType("sandbox_runner.bootstrap")
        br_mod.bootstrap_environment = lambda s, v: s
        br_mod._verify_required_dependencies = lambda: None
        monkeypatch.setitem(sys.modules, "sandbox_runner.bootstrap", br_mod)
    spec.loader.exec_module(mod)
    return mod


def test_autonomous_full_loop(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)

    # stub modules required by sandbox_runner
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=_Audit)
    _stub_module(monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot)
    _stub_module(monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=_Policy)
    _stub_module(monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot())
    _stub_module(monkeypatch, "menace.data_bot", MetricsDB=DummyBot, DataBot=DummyBot)

    preset_calls = []
    adapt_calls = []

    def fake_generate(n=None):
        preset_calls.append(n)
        return [{"CPU_LIMIT": "1"}]

    def fake_adapt(tracker, presets):
        adapt_calls.append(list(presets))
        for p in presets:
            p["ADAPTED"] = True
        return presets

    eg_mod = types.ModuleType("menace.environment_generator")
    eg_mod.generate_presets = fake_generate
    eg_mod.adapt_presets = fake_adapt
    monkeypatch.setitem(sys.modules, "menace.environment_generator", eg_mod)

    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=DummyTracker)

    run_calls = []

    class DummyImprover:
        def run_cycle(self):
            run_calls.append(True)
            return types.SimpleNamespace(roi=types.SimpleNamespace(roi=0.0))

        def _policy_state(self):
            return ()

    def fake_cycle(ctx, sec, snip, tracker, scenario=None):
        ctx.improver.run_cycle()
        tracker.update(0.0, 0.1, metrics={"security_score": 80})

    sr_stub = types.ModuleType("sandbox_runner")
    cli_stub = types.ModuleType("sandbox_runner.cli")
    cli_stub.full_autonomous_run = lambda args: sr_stub._sandbox_main({}, args)
    cli_stub._diminishing_modules = lambda *a, **k: (set(), None)
    cli_stub._ema = lambda seq: (0.0, [])
    cli_stub._adaptive_threshold = lambda *a, **k: 0.0
    cli_stub._adaptive_synergy_threshold = lambda *a, **k: 0.0
    cli_stub._synergy_converged = lambda *a, **k: (True, 0.0, {})

    def fake_sandbox_main(preset, args):
        class Ctx:
            def __init__(self):
                self.tracker = DummyTracker()
                self.improver = DummyImprover()
                self.roi_history_file = Path(args.sandbox_data_dir) / "roi_history.json"
        ctx = Ctx()
        fake_cycle(ctx, None, None, ctx.tracker)
        from menace.environment_generator import adapt_presets as _adapt
        _adapt(ctx.tracker, [preset])
        ctx.tracker.save_history(str(ctx.roi_history_file))
        return ctx.tracker

    sr_stub._sandbox_main = fake_sandbox_main
    sr_stub._sandbox_cycle_runner = fake_cycle
    sr_stub.SelfImprovementEngine = lambda *a, **k: DummyImprover()
    sr_stub.cli = cli_stub
    sys.modules["sandbox_runner"] = sr_stub
    sys.modules["sandbox_runner.cli"] = cli_stub

    mod = load_module(monkeypatch)

    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)

    popen_calls = []
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: popen_calls.append(a))
    monkeypatch.setenv("VISUAL_AGENT_AUTOSTART", "0")
    monkeypatch.chdir(tmp_path)

    mod.main(["--max-iterations", "1", "--runs", "1", "--preset-count", "1", "--sandbox-data-dir", str(tmp_path)])

    roi_file = tmp_path / "roi_history.json"
    assert roi_file.exists()
    assert preset_calls
    assert adapt_calls
    assert run_calls
    assert not popen_calls


