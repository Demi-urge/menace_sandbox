import importlib.util
import argparse
import sys
import os
from pathlib import Path
import asyncio
import pytest
from dynamic_path_router import resolve_dir, resolve_path, path_for_prompt

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
import types
audit_mod = types.ModuleType("menace.audit_trail")
class AuditTrail:
    def __init__(self, *a, **k):
        pass
audit_mod.AuditTrail = AuditTrail
sys.modules.setdefault("menace.audit_trail", audit_mod)
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
if "filelock" not in sys.modules:
    fl = types.ModuleType("filelock")
    class _FL:
        def __init__(self, *a, **k):
            self.lock_file = "x"
            self.is_locked = False
        def acquire(self, *a, **k):
            self.is_locked = True
        def release(self):
            self.is_locked = False
        def __enter__(self):
            self.acquire()
            return self
        def __exit__(self, exc_type, exc, tb):
            self.release()
            return False
    fl.FileLock = _FL
    fl.Timeout = type("Timeout", (Exception,), {})
    sys.modules["filelock"] = fl

# stub modules used during sandbox_runner import
meta_mod = types.ModuleType("menace.meta_logging")
sys.modules.setdefault("menace.meta_logging", meta_mod)
adv_mod = types.ModuleType("menace.advanced_error_management")
class _ARM:
    pass
adv_mod.AutomatedRollbackManager = _ARM
class FormalVerifier:
    pass
adv_mod.FormalVerifier = FormalVerifier
sys.modules.setdefault("menace.advanced_error_management", adv_mod)

class _Dummy:
    def __init__(self, *a, **k):
        pass

for name, attr in {
    "menace.unified_event_bus": "UnifiedEventBus",
    "menace.menace_orchestrator": "MenaceOrchestrator",
    "menace.self_improvement_policy": "SelfImprovementPolicy",
    "menace.self_improvement": "SelfImprovementEngine",
    "menace.self_test_service": "SelfTestService",
    "menace.code_database": ["PatchHistoryDB", "CodeDB"],
    "menace.error_bot": ["ErrorBot", "ErrorDB"],
    "menace.data_bot": ["MetricsDB", "DataBot"],
    "menace.metrics_plugins": ["load_metrics_plugins", "collect_plugin_metrics", "discover_metrics_plugins"],
    "sandbox_runner.metrics_plugins": ["load_metrics_plugins", "collect_plugin_metrics", "discover_metrics_plugins"],
    "menace.discrepancy_detection_bot": "DiscrepancyDetectionBot",
    "menace.pre_execution_roi_bot": "PreExecutionROIBot",
    "menace.menace_memory_manager": "MenaceMemoryManager",
}.items():
    mod = types.ModuleType(name)
    if isinstance(attr, list):
        for a in attr:
            if a == "load_metrics_plugins":
                setattr(mod, a, lambda: [])
            elif a == "collect_plugin_metrics":
                setattr(mod, a, lambda *b, **k: {})
            elif "DB" in a:
                setattr(mod, a, lambda *b, **k: _Dummy())
            else:
                setattr(mod, a, _Dummy)
    else:
        setattr(mod, attr, _Dummy)
    sys.modules.setdefault(name, mod)

for mod_name in [
    "menace.unified_event_bus",
    "menace.menace_orchestrator",
    "menace.self_improvement_policy",
    "menace.self_improvement",
    "menace.self_test_service",
    "menace.code_database",
    "menace.error_bot",
    "menace.data_bot",
    "menace.metrics_plugins",
    "sandbox_runner.metrics_plugins",
    "menace.discrepancy_detection_bot",
    "menace.pre_execution_roi_bot",
    "menace.menace_memory_manager",
]:
    sys.modules.setdefault(mod_name, types.ModuleType(mod_name))

from tests.test_menace_master import _setup_mm_stubs, DummyBot, _stub_module


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


class _ROITracker:
    instance = None

    def __init__(self, *a, **k):
        _ROITracker.instance = self
        self.calls = []
        self.module_deltas = {}
        self.synergy_history = []
        self.scenario_synergy = {}

    def get_scenario_synergy(self, name):
        return self.scenario_synergy.get(name, [])

    def update(self, prev_roi, roi, modules=None, resources=None, metrics=None):
        self.calls.append(metrics)
        return 0.0, [], False

    def forecast(self):
        return 0.0, (0.0, 0.0)

    def diminishing(self):
        return 0.0

    def record_prediction(self, predicted, actual, *a, **k):
        pass

    def record_metric_prediction(self, metric, predicted, actual):
        pass

    def rolling_mae(self, window=None):
        return 0.0

    def load_history(self, path):
        pass

    def save_history(self, path):
        pass


def test_run_sandbox_uses_temporary_env(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)

    # Additional stubs required by _run_sandbox imports
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=_Audit)
    _stub_module(monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot)
    _stub_module(monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=_Policy)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=_ROITracker)
    _stub_module(monkeypatch, "menace.db_router", init_db_router=lambda *a, **k: None)
    _stub_module(monkeypatch, "menace.error_bot", ErrorDB=lambda p: DummyBot(), ErrorBot=DummyBot)
    sandbox_runner = types.ModuleType("sandbox_runner")
    sandbox_runner.subprocess = types.SimpleNamespace(run=lambda *a, **k: None)
    sandbox_runner.shutil = types.SimpleNamespace(copy2=lambda *a, **k: None)
    def _sandbox_init(config, args, context_builder=None):
        bus = sandbox_runner.UnifiedEventBus(persist_path="menace_sandbox_dummy.db")
        return types.SimpleNamespace(bus=bus, tracker=_ROITracker())
    def _sandbox_cycle_runner(ctx, a, b, tracker):
        tracker.update(None, None, metrics={})
    def _sandbox_cleanup(ctx):
        ctx.bus.close()
    sandbox_runner._sandbox_init = _sandbox_init
    sandbox_runner._sandbox_cycle_runner = _sandbox_cycle_runner
    sandbox_runner._sandbox_cleanup = _sandbox_cleanup
    sys.modules["sandbox_runner"] = sandbox_runner


    bus_paths = []

    class DummyBus:
        def __init__(self, persist_path=None, **kw):
            bus_paths.append(persist_path)
        def close(self):
            bus_paths.append("closed")

    monkeypatch.setattr(sandbox_runner, "UnifiedEventBus", DummyBus, raising=False)

    def fake_run(cmd, check=False):
        Path(cmd[-1]).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sandbox_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(sandbox_runner.shutil, "copy2", lambda *a, **k: None)

    class DummyOrch:
        def create_oversight(self, *a, **k):
            pass
        def run_cycle(self, *a, **k):
            class R:
                roi = None
            return R()

    class DummyImprover:
        def run_cycle(self):
            class Res:
                roi = None
            return Res()

        def _policy_state(self):
            return ()

    class DummyTester:
        def __init__(self, *a, **k):
            pass

        def _run_once(self):
            pass

    class DummySandbox:
        def __init__(self, *a, **k):
            pass

        def analyse_and_fix(self):
            pass
    monkeypatch.setattr(sandbox_runner, "MenaceOrchestrator", DummyOrch, raising=False)
    monkeypatch.setattr(
        sandbox_runner, "SelfImprovementEngine", lambda *a, **k: DummyImprover(), raising=False
    )
    monkeypatch.setattr(sandbox_runner, "SelfTestService", DummyTester, raising=False)
    monkeypatch.setattr(sys.modules["menace.self_debugger_sandbox"], "SelfDebuggerSandbox", DummySandbox)
    monkeypatch.setattr(sys.modules["menace.data_bot"], "MetricsDB", DummyBot)

    orig_db = tmp_path / "orig.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{orig_db}")
    monkeypatch.setenv("BOT_DB_PATH", str(tmp_path / "bots.db"))
    monkeypatch.setenv("BOT_PERFORMANCE_DB", str(tmp_path / "perf.db"))
    monkeypatch.setenv("MAINTENANCE_DB", str(tmp_path / "maint.db"))

    ctx = sandbox_runner._sandbox_init({}, argparse.Namespace(), object())
    sandbox_runner._sandbox_cycle_runner(ctx, None, None, ctx.tracker)
    sandbox_runner._sandbox_cleanup(ctx)

    assert os.environ["DATABASE_URL"] == f"sqlite:///{orig_db}"
    assert os.environ["BOT_DB_PATH"] == str(tmp_path / "bots.db")
    assert not orig_db.exists()
    assert bus_paths and bus_paths[-1] == "closed"
    assert "menace_sandbox_" in bus_paths[0]
    assert isinstance(_ROITracker.instance.calls[0], dict)


def test_run_sandbox_merges_module_index(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)

    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=_Audit)
    _stub_module(monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot)
    _stub_module(monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=_Policy)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=_ROITracker)
    _stub_module(monkeypatch, "menace.db_router", init_db_router=lambda *a, **k: None)
    _stub_module(monkeypatch, "menace.error_bot", ErrorDB=lambda p: DummyBot(), ErrorBot=DummyBot)

    sandbox_runner = types.ModuleType("sandbox_runner")
    sandbox_runner.subprocess = types.SimpleNamespace(run=lambda *a, **k: None)
    sandbox_runner.shutil = types.SimpleNamespace(copy2=lambda *a, **k: None)
    def _sandbox_init(config, args, context_builder=None):
        bus = sandbox_runner.UnifiedEventBus(persist_path="menace_sandbox_dummy.db")
        return types.SimpleNamespace(bus=bus, tracker=_ROITracker())
    def _sandbox_cycle_runner(ctx, a, b, tracker):
        tracker.update(None, None, metrics={})
    def _sandbox_cleanup(ctx):
        ctx.bus.close()
    sandbox_runner._sandbox_init = _sandbox_init
    sandbox_runner._sandbox_cycle_runner = _sandbox_cycle_runner
    sandbox_runner._sandbox_cleanup = _sandbox_cleanup
    sys.modules["sandbox_runner"] = sandbox_runner

    import shutil as real_shutil
    orig_copy = real_shutil.copy2
    def safe_copy(src, dst, *a, **k):
        if Path(src).exists():
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            orig_copy(src, dst, *a, **k)

    monkeypatch.setattr(sandbox_runner.shutil, "copy2", safe_copy)

    monkeypatch.setattr(sandbox_runner.subprocess, "run", lambda cmd, check=False: Path(cmd[-1]).mkdir(parents=True, exist_ok=True))

    class DummyBus:
        def __init__(self, persist_path=None, **kw):
            pass
        def close(self):
            pass

    monkeypatch.setattr(sandbox_runner, "UnifiedEventBus", DummyBus, raising=False)

    import json

    class DummyImprover:
        def __init__(self, *a, **kw):
            self.db = kw.get("module_index")

        def run_cycle(self):
            if self.db:
                self.db.get("new.py")  # path-ignore
            class Res:
                roi = None
            return Res()

        def _policy_state(self):
            return ()

    class DummyOrch:
        def create_oversight(self, *a, **k):
            pass

        def run_cycle(self, *a, **k):
            class R:
                roi = None
            return R()

    class DummyTester:
        def __init__(self, *a, **k):
            pass

        def _run_once(self):
            pass

    class DummySandbox:
        def __init__(self, *a, **k):
            pass

        def analyse_and_fix(self):
            pass

    monkeypatch.setattr(sandbox_runner, "MenaceOrchestrator", DummyOrch, raising=False)
    monkeypatch.setattr(
        sandbox_runner,
        "SelfImprovementEngine",
        lambda *a, **kw: DummyImprover(*a, **kw),
        raising=False,
    )
    monkeypatch.setattr(sandbox_runner, "SelfTestService", DummyTester, raising=False)
    monkeypatch.setattr(sys.modules["menace.self_debugger_sandbox"], "SelfDebuggerSandbox", DummySandbox)
    monkeypatch.setattr(sys.modules["menace.data_bot"], "MetricsDB", DummyBot)
    class DummyContextBuilder:
        def refresh_db_weights(self):
            pass
    monkeypatch.setattr(
        sandbox_runner, "ContextBuilder", DummyContextBuilder, raising=False
    )

    module_map = tmp_path / "module_map.json"
    module_map.write_text("{\"old.py\": 42}")  # path-ignore
    ctx = sandbox_runner._sandbox_init(
        {}, argparse.Namespace(sandbox_data_dir=str(tmp_path)), sandbox_runner.ContextBuilder()
    )
    sandbox_runner._sandbox_cycle_runner(ctx, None, None, ctx.tracker)
    sandbox_runner._sandbox_cleanup(ctx)

    assert module_map.exists()
    assert isinstance(_ROITracker.instance.calls[0], dict)


def test_module_map_clusters_have_same_index(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")  # path-ignore
    (pkg / "a.py").write_text("from . import b\nb.f()\n")  # path-ignore
    (pkg / "b.py").write_text("def f():\n    pass\n")  # path-ignore
    from scripts.generate_module_map import generate_module_map
    generate_module_map(tmp_path / "sandbox_data" / "module_map.json", root=tmp_path)
    from module_index_db import ModuleIndexDB
    db = ModuleIndexDB(tmp_path / "sandbox_data" / "module_map.json")
    assert db.get("pkg/a") == db.get("pkg/b")


def test_section_short_circuit(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)

    from tests.test_menace_master import _stub_module, DummyBot

    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot)
    _stub_module(monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=_Policy)
    _stub_module(monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p=None: DummyBot())
    _stub_module(monkeypatch, "menace.data_bot", MetricsDB=DummyBot, DataBot=DummyBot)

    class DummyTracker:
        def __init__(self, *a, **k):
            self.module_deltas = {}

        def update(self, *a, **k):
            return 0.0, [], False

        def forecast(self):
            return 0.0, (0.0, 0.0)

        def diminishing(self):
            return 0.02

        def record_prediction(self, *a, **k):
            pass

        def rolling_mae(self, window=None):
            return 0.0

        def load_history(self, path):
            pass

        def save_history(self, path):
            pass

        def rankings(self):
            return []

    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=DummyTracker)

    class DummyImprover:
        def __init__(self):
            self.val = 0.0
            self.calls = 0

        def run_cycle(self):
            self.val += 0.01
            self.calls += 1
            return types.SimpleNamespace(roi=types.SimpleNamespace(roi=self.val))

        def _policy_state(self):
            return ()

    improver = DummyImprover()

    class DummyBus:
        def __init__(self, persist_path=None, **kw):
            pass

        def close(self):
            pass

    class DummySandbox:
        def __init__(self, *a, **k):
            pass

        def analyse_and_fix(self):
            pass

    class DummyTester:
        def __init__(self, *a, **k):
            pass

        def _run_once(self):
            pass

    class DummyOrch:
        def create_oversight(self, *a, **k):
            pass

        def run_cycle(self, *a, **k):
            class R:
                roi = None

            return R()

    monkeypatch.setattr(sandbox_runner, "UnifiedEventBus", DummyBus)
    monkeypatch.setattr(sandbox_runner.subprocess, "run", lambda cmd, check=False: Path(cmd[-1]).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(sandbox_runner.shutil, "copy2", lambda *a, **k: None)

    monkeypatch.setattr(sandbox_runner, "MenaceOrchestrator", DummyOrch)
    monkeypatch.setattr(sandbox_runner, "SelfImprovementEngine", lambda *a, **k: improver)
    monkeypatch.setattr(sandbox_runner, "SelfTestService", DummyTester)
    monkeypatch.setattr(sys.modules["menace.self_debugger_sandbox"], "SelfDebuggerSandbox", DummySandbox, raising=False)
    monkeypatch.setattr(sys.modules["menace.data_bot"], "MetricsDB", DummyBot, raising=False)

    class TrackLogger(sandbox_runner._SandboxMetaLogger):
        instance = None

        def __init__(self, path):
            super().__init__(path)
            TrackLogger.instance = self

    monkeypatch.setattr(sandbox_runner, "_SandboxMetaLogger", TrackLogger)
    monkeypatch.setattr(
        sandbox_runner,
        "scan_repo_sections",
        lambda path, modules=None: {"mod.py": {"sec": ["pass"]}},  # path-ignore
    )

    monkeypatch.setenv("SANDBOX_CYCLES", "5")
    ctx = sandbox_runner._sandbox_init(
        {}, argparse.Namespace(sandbox_data_dir=str(tmp_path)), sandbox_runner.ContextBuilder()
    )
    sandbox_runner._sandbox_cycle_runner(ctx, None, None, ctx.tracker)
    sandbox_runner._sandbox_cleanup(ctx)

    assert improver.calls < 5
    assert "mod.py:sec" in TrackLogger.instance.flagged_sections  # path-ignore


def test_failure_modes_timeout(monkeypatch):
    _setup_mm_stubs(monkeypatch)
    import sandbox_runner

    res, _ = asyncio.run(
        sandbox_runner._section_worker(
            "import time; time.sleep(1)",
            {"FAILURE_MODES": "timeout"},
            0.0,
        )
    )

    assert res["exit_code"] != 0


def test_failure_modes_disk_corruption(monkeypatch):
    _setup_mm_stubs(monkeypatch)
    import sandbox_runner

    code = "with open('f','w') as f: f.write('ok')\nprint(open('f').read())"
    res, _ = asyncio.run(
        sandbox_runner._section_worker(
            code,
            {"FAILURE_MODES": ["disk_corruption"]},
            0.0,
        )
    )

    assert "CORRUPTED" in res["stdout"]


def test_failure_modes_network(monkeypatch):
    _setup_mm_stubs(monkeypatch)
    import sandbox_runner

    code = "import socket; s=socket.socket(); s.connect(('example.com',80))"
    res, _ = asyncio.run(
        sandbox_runner._section_worker(
            code,
            {"FAILURE_MODES": "network"},
            0.0,
        )
    )

    assert res["exit_code"] != 0


def test_failure_modes_network_partition(monkeypatch):
    _setup_mm_stubs(monkeypatch)
    import sandbox_runner

    code = "import socket; s=socket.socket(); s.connect(('example.com',80))"
    res, _ = asyncio.run(
        sandbox_runner._section_worker(
            code,
            {"FAILURE_MODES": "network_partition"},
            0.0,
        )
    )

    assert res["exit_code"] != 0


def test_failure_modes_disk(monkeypatch):
    _setup_mm_stubs(monkeypatch)
    import sandbox_runner

    code = "with open('f','w') as f: f.write('ok')\nprint(open('f').read())"
    res, _ = asyncio.run(
        sandbox_runner._section_worker(
            code,
            {"FAILURE_MODES": "disk"},
            0.0,
        )
    )

    assert res["stdout"].strip() == "ok"


def test_workflow_run_called(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)
    import shutil, importlib.util
    monkeypatch.setattr(shutil, "which", lambda *a, **k: "/usr/bin/true")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    class DummyTracker:
        def __init__(self, **kw):
            self.updated = []
            self.roi_history = []
            self.metrics_history = {}
        def update(self, prev, roi, modules=None, resources=None, metrics=None):
            self.updated.append(modules)
            self.roi_history.append(roi)
            return 0, [], False
        def forecast(self):
            return 0.0, (0.0, 0.0)
        def diminishing(self):
            return 0.0
        def record_prediction(self, *a, **k):
            pass
        def register_metrics(self, *a, **k):
            pass
        def save_history(self, *a, **k):
            pass

        def rankings(self):
            return []

        def rankings(self):
            return []

    _stub_module(monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot())
    _stub_module(monkeypatch, "menace.data_bot", MetricsDB=DummyBot, DataBot=DummyBot)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=DummyTracker)
    _stub_module(monkeypatch, "menace.quick_fix_engine", QuickFixEngine=DummyBot)
    import importlib.util
    import sys
    import argparse

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner",
        str(resolve_path("sandbox_runner.py")),  # path-ignore
        submodule_search_locations=[str(resolve_dir("sandbox_runner"))],
    )
    sandbox_runner = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner"] = sandbox_runner
    spec.loader.exec_module(sandbox_runner)

    tracker = DummyTracker()


    class DummyMeta:
        def __init__(self):
            self.flagged_sections = set()
        def rankings(self):
            return {}
        def diminishing(self, threshold=None, consecutive=3, entropy_threshold=None):
            return []

    class DummyCtx:
        def __init__(self):
            self.sections = {"m.py": {"sec": ["pass"]}}  # path-ignore
            self.all_section_names = {"m.py:sec"}  # path-ignore
            self.meta_log = DummyMeta()
            self.tracker = tracker
            self.res_db = None
            self.prev_roi = 0.0
            self.predicted_roi = None
            self.roi_history_file = tmp_path / "roi.json"
            self.synergy_needed = False
            self.best_roi = 0.0
            self.best_synergy_metrics = {}
            self.settings = types.SimpleNamespace(
                entropy_plateau_threshold=None, entropy_plateau_consecutive=None
            )
            self.sandbox = types.SimpleNamespace(graph=None)

    monkeypatch.setattr(
        sandbox_runner, "_sandbox_init", lambda preset, args, context_builder: DummyCtx()
    )
    monkeypatch.setattr(sandbox_runner, "_sandbox_cleanup", lambda ctx: None)
    monkeypatch.setattr(
        sandbox_runner,
        "_sandbox_cycle_runner",
        lambda ctx, sec, snip, t, scenario=None: t.update(0.0, 0.1, modules=[sec] if sec else []),
    )

    calls = []

    def fake_run(db, presets=None, *, return_details=False, tracker=None):
        calls.append(tracker)
        if tracker:
            tracker.update(0.0, 0.2, modules=["workflow"])
        return tracker

    monkeypatch.setattr(sandbox_runner, "run_workflow_simulations", fake_run)

    class DummyContextBuilder:
        def refresh_db_weights(self):
            pass

    monkeypatch.setattr(sandbox_runner, "ContextBuilder", DummyContextBuilder)

    args = argparse.Namespace(
        workflow_db=str(tmp_path / "wf.db"),
        sandbox_data_dir=str(tmp_path),
        no_workflow_run=False,
    )

    result = sandbox_runner._sandbox_main({}, args)
    assert calls and calls[0] is result


def test_no_workflow_run_option(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)
    class DummyTracker:
        def __init__(self):
            self.updated = []
            self.roi_history = []
            self.metrics_history = {}
        def update(self, prev, roi, modules=None, resources=None, metrics=None):
            self.updated.append(modules)
            self.roi_history.append(roi)
            return 0, [], False
        def forecast(self):
            return 0.0, (0.0, 0.0)
        def diminishing(self):
            return 0.0
        def record_prediction(self, *a, **k):
            pass
        def register_metrics(self, *a, **k):
            pass
        def save_history(self, *a, **k):
            pass

    _stub_module(monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot())
    _stub_module(monkeypatch, "menace.data_bot", MetricsDB=DummyBot, DataBot=DummyBot)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=DummyTracker)
    import importlib.util
    import sys
    import argparse

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner",
        str(resolve_path("sandbox_runner.py")),  # path-ignore
        submodule_search_locations=[str(resolve_dir("sandbox_runner"))],
    )
    sandbox_runner = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner"] = sandbox_runner
    spec.loader.exec_module(sandbox_runner)

    class DummyCtx:
        def __init__(self):
            self.sections = {}
            self.all_section_names = set()
            self.meta_log = type("M", (), {"flagged_sections": set(), "rankings": lambda self: {}, "diminishing": lambda self, threshold=None: []})()
            self.tracker = DummyTracker()
            self.res_db = None
            self.prev_roi = 0.0
            self.predicted_roi = None
            self.roi_history_file = tmp_path / "roi.json"
            self.synergy_needed = False
            self.best_roi = 0.0
            self.best_synergy_metrics = {}

    monkeypatch.setattr(
        sandbox_runner, "_sandbox_init", lambda preset, args, context_builder: DummyCtx()
    )
    monkeypatch.setattr(sandbox_runner, "_sandbox_cleanup", lambda ctx: None)
    monkeypatch.setattr(sandbox_runner, "_sandbox_cycle_runner", lambda *a, **k: None)

    class DummyContextBuilder:
        def refresh_db_weights(self):
            pass

    monkeypatch.setattr(sandbox_runner, "ContextBuilder", DummyContextBuilder)

    called = []
    monkeypatch.setattr(sandbox_runner, "run_workflow_simulations", lambda *a, **k: called.append(None))

    args = argparse.Namespace(
        workflow_db=str(tmp_path / "wf.db"),
        sandbox_data_dir=str(tmp_path),
        no_workflow_run=True,
    )

    sandbox_runner._sandbox_main({}, args)
    assert not called


def test_dynamic_workflows_build_from_repo(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)

    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=_Audit)
    _stub_module(monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot)
    _stub_module(monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=_Policy)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=_ROITracker)

    import sandbox_runner
    import sandbox_runner.environment as env
    import types, sys
    from pathlib import Path as _P
    import importlib.util

    ROOT = _P(__file__).resolve().parents[1]
    pkg = sys.modules.setdefault("menace", types.ModuleType("menace"))
    pkg.__path__ = [str(ROOT)]
    spec = importlib.util.spec_from_file_location(
        "menace.task_handoff_bot",
        ROOT / "task_handoff_bot.py",  # path-ignore
        submodule_search_locations=[str(ROOT)],
    )
    thb = importlib.util.module_from_spec(spec)
    sys.modules["menace.task_handoff_bot"] = thb
    spec.loader.exec_module(thb)

    groups = {"1": ["simple_functions", "sandbox_runner.cli"], "2": ["sandbox_runner.environment"]}
    dmm = types.ModuleType("dynamic_module_mapper")
    dmm.discover_module_groups = lambda *a, **k: groups
    dmm.dotify_groups = lambda g: g
    monkeypatch.setitem(sys.modules, "dynamic_module_mapper", dmm)

    snippets = []

    async def fake_worker(snippet, env_input, threshold):
        snippets.append(snippet)
        return {"exit_code": 0}, [(0.0, 1.0, {})]

    monkeypatch.setattr(env, "_section_worker", fake_worker)
    monkeypatch.setattr(sandbox_runner, "simulate_execution_environment", lambda s, e: {})

    tracker = sandbox_runner.run_workflow_simulations(
        workflows_db=str(tmp_path / "wf.db"),
        env_presets=[{}],
        dynamic_workflows=True,
    )

    assert len(snippets) == 3
    assert any("import simple_functions" in s for s in snippets)
    assert any("import sandbox_runner.cli" in s for s in snippets)
    assert any("import sandbox_runner.environment" in s for s in snippets)


def test_run_sandbox_cli_recursion_default(monkeypatch):
    from .test_sandbox_runner_cli_recursion import _load_cli, _capture_run

    capture = {}
    cli = _load_cli(monkeypatch)
    _capture_run(monkeypatch, cli, capture)
    monkeypatch.delenv("SANDBOX_RECURSIVE_ORPHANS", raising=False)
    cli.main([])
    assert capture.get("recursive_orphans") is True
    assert os.getenv("SANDBOX_RECURSIVE_ORPHANS") == "1"


def test_cli_help_resolves_orphan_path(monkeypatch, capsys):
    import dynamic_path_router as dpr

    dpr.clear_cache()
    monkeypatch.delenv("MENACE_ROOTS", raising=False)
    import sandbox_runner.cli as cli

    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    expected = f"{path_for_prompt('sandbox_data')}/orphan_modules.json"
    assert expected in out
