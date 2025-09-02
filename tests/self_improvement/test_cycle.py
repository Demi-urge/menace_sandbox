import asyncio
import sys
import types
import importlib.util
from pathlib import Path

import pytest
from sandbox_settings import SandboxSettings
import self_improvement.baseline_tracker as baseline_tracker


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def _stub_deps():
    sys.modules.setdefault("quick_fix_engine", types.ModuleType("quick_fix_engine"))
    sys.modules.setdefault("error_logger", types.ModuleType("error_logger"))
    sr_pkg = types.ModuleType("sandbox_runner")
    sr_pkg.__path__ = []
    sys.modules.setdefault("sandbox_runner", sr_pkg)
    boot = types.ModuleType("sandbox_runner.bootstrap")
    boot.initialize_autonomous_sandbox = lambda *a, **k: None
    sys.modules.setdefault("sandbox_runner.bootstrap", boot)
    sys.modules.setdefault(
        "sandbox_runner.orphan_integration",
        types.ModuleType("sandbox_runner.orphan_integration"),
    )


def test_self_improvement_cycle_runs(tmp_path, monkeypatch, in_memory_dbs):
    _stub_deps()
    menace_pkg = types.ModuleType("menace")
    menace_pkg.__path__ = []
    sys.modules["menace"] = menace_pkg
    si_pkg = types.ModuleType("menace.self_improvement")
    si_pkg.__path__ = [str(Path("self_improvement"))]
    sys.modules["menace.self_improvement"] = si_pkg

    bootstrap = types.ModuleType("sandbox_runner.bootstrap")
    bootstrap.initialize_autonomous_sandbox = lambda s: None
    sys.modules["sandbox_runner.bootstrap"] = bootstrap

    logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        exception=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    logging_utils = types.SimpleNamespace(
        get_logger=lambda name: logger,
        log_record=lambda **k: k,
        setup_logging=lambda: None,
    )
    sys.modules["menace.logging_utils"] = logging_utils

    InMemoryROI, InMemoryStability = in_memory_dbs
    roi_calls = {"count": 0}
    orig_log = InMemoryROI.log_result

    def spy(self, **kw):
        roi_calls["count"] += 1
        return orig_log(self, **kw)

    monkeypatch.setattr(InMemoryROI, "log_result", spy)
    baseline_calls: dict[str, int] = {}

    def track(metric: str) -> float:
        baseline_calls[metric] = baseline_calls.get(metric, 0) + 1
        return 0.0

    class DummyLock:
        def __init__(self, *a, **k):
            pass

        def acquire(self, *a, **k):
            class Ctx:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return Ctx()

    sys.modules["menace.lock_utils"] = types.SimpleNamespace(
        SandboxLock=DummyLock, Timeout=Exception, LOCK_TIMEOUT=1
    )
    sys.modules["menace.unified_event_bus"] = types.SimpleNamespace(UnifiedEventBus=None)
    sys.modules["menace.meta_workflow_planner"] = types.SimpleNamespace(
        MetaWorkflowPlanner=None
    )

    import sandbox_settings as sandbox_settings_module

    sys.modules["menace.sandbox_settings"] = sandbox_settings_module

    init_module = _load_module("menace.self_improvement.init", Path("self_improvement/init.py"))
    meta_planning = _load_module(
        "menace.self_improvement.meta_planning", Path("self_improvement/meta_planning.py")
    )

    monkeypatch.setattr(baseline_tracker.TRACKER, "get", track)
    monkeypatch.setattr(meta_planning.BASELINE_TRACKER, "get", track)

    monkeypatch.setattr(init_module, "verify_dependencies", lambda auto_install=False: None)

    settings = SandboxSettings()
    settings.sandbox_data_dir = str(tmp_path)
    settings.sandbox_central_logging = False

    monkeypatch.setattr(init_module, "load_sandbox_settings", lambda: settings)
    init_module.init_self_improvement()

    calls = {"count": 0}

    class DummyPlanner:
        def __init__(self):
            self.roi_db = InMemoryROI()
            self.stability_db = InMemoryStability()
            self.cluster_map = {}

        def discover_and_persist(self, workflows):
            calls["count"] += 1
            return [
                {"chain": ["w"], "roi_gain": 1.0, "failures": 0, "entropy": 0.1}
            ]

    monkeypatch.setattr(meta_planning, "_FallbackPlanner", DummyPlanner)
    monkeypatch.setattr(meta_planning, "MetaWorkflowPlanner", None)

    async def run_cycle():
        task = asyncio.create_task(
            meta_planning.self_improvement_cycle({"w": lambda: None}, interval=0)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_cycle())
    meta_planning.BASELINE_TRACKER.get("roi_delta")
    assert calls["count"] > 0
    assert roi_calls["count"] > 0
    assert InMemoryStability.instances[0].data
    assert baseline_calls.get("roi_delta", 0) > 0


def test_self_improvement_cycle_handles_db_errors(tmp_path, monkeypatch, in_memory_dbs):
    menace_pkg = types.ModuleType("menace")
    menace_pkg.__path__ = []
    sys.modules["menace"] = menace_pkg
    si_pkg = types.ModuleType("menace.self_improvement")
    si_pkg.__path__ = [str(Path("self_improvement"))]
    sys.modules["menace.self_improvement"] = si_pkg

    bootstrap = types.ModuleType("sandbox_runner.bootstrap")
    bootstrap.initialize_autonomous_sandbox = lambda s: None
    sys.modules["sandbox_runner.bootstrap"] = bootstrap

    logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        exception=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    logging_utils = types.SimpleNamespace(
        get_logger=lambda name: logger,
        log_record=lambda **k: k,
        setup_logging=lambda: None,
    )
    sys.modules["menace.logging_utils"] = logging_utils

    InMemoryROI, InMemoryStability = in_memory_dbs
    monkeypatch.setattr(
        InMemoryROI, "log_result", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    class DummyLock:
        def __init__(self, *a, **k):
            pass

        def acquire(self, *a, **k):
            class Ctx:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return Ctx()

    sys.modules["menace.lock_utils"] = types.SimpleNamespace(
        SandboxLock=DummyLock, Timeout=Exception, LOCK_TIMEOUT=1
    )
    sys.modules["menace.unified_event_bus"] = types.SimpleNamespace(UnifiedEventBus=None)
    sys.modules["menace.meta_workflow_planner"] = types.SimpleNamespace(
        MetaWorkflowPlanner=None
    )

    import sandbox_settings as sandbox_settings_module

    sys.modules["menace.sandbox_settings"] = sandbox_settings_module

    init_module = _load_module("menace.self_improvement.init", Path("self_improvement/init.py"))
    meta_planning = _load_module(
        "menace.self_improvement.meta_planning", Path("self_improvement/meta_planning.py")
    )

    monkeypatch.setattr(init_module, "verify_dependencies", lambda auto_install=False: None)

    settings = SandboxSettings()
    settings.sandbox_data_dir = str(tmp_path)
    settings.sandbox_central_logging = False

    monkeypatch.setattr(init_module, "load_sandbox_settings", lambda: settings)
    init_module.init_self_improvement()

    class DummyPlanner:
        def __init__(self):
            self.roi_db = InMemoryROI()
            self.stability_db = InMemoryStability()
            self.cluster_map = {}

        def discover_and_persist(self, workflows):
            return [
                {"chain": ["w"], "roi_gain": 1.0, "failures": 0, "entropy": 0.1}
            ]

    monkeypatch.setattr(meta_planning, "_FallbackPlanner", DummyPlanner)
    monkeypatch.setattr(meta_planning, "MetaWorkflowPlanner", None)

    async def run_cycle():
        task = asyncio.create_task(
            meta_planning.self_improvement_cycle({"w": lambda: None}, interval=0)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_cycle())
    # error in ROI logging should not prevent stability metrics from recording
    assert InMemoryStability.instances[0].data


def test_start_self_improvement_cycle_dependency_failure(tmp_path, monkeypatch, in_memory_dbs):
    menace_pkg = types.ModuleType("menace")
    menace_pkg.__path__ = []
    sys.modules["menace"] = menace_pkg
    si_pkg = types.ModuleType("menace.self_improvement")
    si_pkg.__path__ = [str(Path("self_improvement"))]
    sys.modules["menace.self_improvement"] = si_pkg

    logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        exception=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    logging_utils = types.SimpleNamespace(
        get_logger=lambda name: logger,
        log_record=lambda **k: k,
        setup_logging=lambda: None,
    )
    sys.modules["menace.logging_utils"] = logging_utils

    class BoomROI:
        def __init__(self, *a, **k):
            raise OSError("no db")

    # WorkflowStabilityDB may still be instantiated successfully
    InMemoryROI, InMemoryStability = in_memory_dbs

    sys.modules["menace.lock_utils"] = types.SimpleNamespace(
        SandboxLock=lambda *a, **k: types.SimpleNamespace(
            __enter__=lambda self: self, __exit__=lambda self, exc_type, exc, tb: False
        ),
        Timeout=Exception,
        LOCK_TIMEOUT=1,
    )
    sys.modules["menace.unified_event_bus"] = types.SimpleNamespace(UnifiedEventBus=None)
    sys.modules["menace.meta_workflow_planner"] = types.SimpleNamespace(
        MetaWorkflowPlanner=None
    )

    import sandbox_settings as sandbox_settings_module
    sys.modules["menace.sandbox_settings"] = sandbox_settings_module

    init_module = _load_module("menace.self_improvement.init", Path("self_improvement/init.py"))
    meta_planning = _load_module(
        "menace.self_improvement.meta_planning", Path("self_improvement/meta_planning.py")
    )
    monkeypatch.setattr(init_module, "load_sandbox_settings", lambda: SandboxSettings())
    monkeypatch.setattr(init_module, "verify_dependencies", lambda auto_install=False: None)
    init_module.init_self_improvement()

    monkeypatch.setattr(meta_planning, "ROIResultsDB", BoomROI)

    with pytest.raises(RuntimeError):
        meta_planning.start_self_improvement_cycle({"w": lambda: None}, interval=0)
