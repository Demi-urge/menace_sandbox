import asyncio
import sys
import types
import importlib.util
from pathlib import Path

import pytest
from sandbox_settings import SandboxSettings


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def test_self_improvement_cycle_runs(tmp_path, monkeypatch):
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

    class DummyDB:
        def __init__(self, *a, **k):
            pass

        def log_result(self, *a, **k):
            pass

        def record_metrics(self, *a, **k):
            pass

    sys.modules["menace.workflow_stability_db"] = types.SimpleNamespace(
        WorkflowStabilityDB=DummyDB
    )
    sys.modules["menace.roi_results_db"] = types.SimpleNamespace(ROIResultsDB=DummyDB)

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

    settings = SandboxSettings()
    settings.sandbox_data_dir = str(tmp_path)
    settings.sandbox_central_logging = False

    monkeypatch.setattr(init_module, "load_sandbox_settings", lambda: settings)
    init_module.init_self_improvement()

    calls = {"count": 0}

    class DummyPlanner:
        def __init__(self):
            self.roi_db = None
            self.stability_db = None
            self.cluster_map = {}

        def discover_and_persist(self, workflows):
            calls["count"] += 1
            return []

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
    assert calls["count"] > 0
