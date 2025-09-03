import ast
import asyncio
import logging
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable, Mapping
import pytest


def _load_meta_planning():
    src = Path("self_improvement/meta_planning.py").read_text()
    tree = ast.parse(src)
    wanted = {"_get_entropy_threshold", "_should_encode", "self_improvement_cycle"}
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in wanted
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias("annotations", None)], level=0
    )
    module = ast.Module([future] + nodes, type_ignores=[])
    module = ast.fix_missing_locations(module)
    ns = {
        "asyncio": asyncio,
        "PLANNER_INTERVAL": 0.0,
        "SandboxSettings": object,
        "WorkflowStabilityDB": object,
        "Any": Any,
        "Callable": Callable,
        "Mapping": Mapping,
        "BASELINE_TRACKER": types.SimpleNamespace(
            get=lambda m: 0.0, std=lambda m: 0.0, momentum=1.0
        ),
    }
    exec(compile(module, "<ast>", "exec"), ns)
    return ns


@pytest.mark.parametrize(
    "cfg_value, base, std, dev, expected",
    [
        (0.7, 0.1, 0.2, 2.0, 0.7),
        (None, 0.2, 0.05, 2.0, 0.3),
        (None, 0.0, 0.0, 1.0, 0.0),
    ],
)
def test_get_entropy_threshold(cfg_value, base, std, dev, expected):
    meta = _load_meta_planning()

    class Cfg:
        meta_entropy_threshold = cfg_value
        entropy_deviation = dev

    class Tracker:
        def get(self, metric):
            return base

        def std(self, metric):
            return std

    assert meta["_get_entropy_threshold"](Cfg(), Tracker()) == pytest.approx(expected)


def test_should_encode_requires_positive_roi_and_low_entropy():
    meta = _load_meta_planning()
    should_encode = meta["_should_encode"]

    tracker = types.SimpleNamespace(get=lambda m: 0.0, momentum=1.0)

    assert should_encode(
        {"roi_gain": 0.1, "entropy": 0.1}, tracker, entropy_threshold=0.2
    )
    assert not should_encode(
        {"roi_gain": 0.0, "entropy": 0.1}, tracker, entropy_threshold=0.2
    )
    assert should_encode(
        {"roi_gain": 0.1, "entropy": 0.3}, tracker, entropy_threshold=0.2
    )


def test_cycle_uses_fallback_planner_when_missing():
    meta = _load_meta_planning()

    calls = {"count": 0}

    class DummyPlanner:
        roi_db = None
        stability_db = None

        def __init__(self):
            self.cluster_map = {}

        def discover_and_persist(self, workflows):
            calls["count"] += 1
            return []

    meta.update(
        {
            "MetaWorkflowPlanner": None,
            "_FallbackPlanner": DummyPlanner,
            "get_logger": lambda name: types.SimpleNamespace(
                warning=lambda *a, **k: None, exception=lambda *a, **k: None
            ),
            "log_record": lambda **kw: kw,
            "load_sandbox_settings": lambda: types.SimpleNamespace(
                meta_mutation_rate=0.0,
                meta_roi_weight=0.0,
                meta_domain_penalty=0.0,
                meta_entropy_threshold=0.2,
                enable_meta_planner=False,
            ),
            "_init": types.SimpleNamespace(
                settings=types.SimpleNamespace(
                    meta_mutation_rate=0.0,
                    meta_roi_weight=0.0,
                    meta_domain_penalty=0.0,
                    meta_entropy_threshold=0.2,
                    enable_meta_planner=False,
                )
            ),
            "get_stable_workflows": lambda: types.SimpleNamespace(),
        }
    )

    async def run():
        task = asyncio.create_task(meta["self_improvement_cycle"]({}, interval=0))
        await asyncio.sleep(0.05)
        assert calls["count"] > 0
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())


def test_cycle_fails_when_enabled_but_missing():
    meta = _load_meta_planning()

    meta.update(
        {
            "MetaWorkflowPlanner": None,
            "get_logger": lambda name: types.SimpleNamespace(),
            "log_record": lambda **kw: kw,
            "load_sandbox_settings": lambda: types.SimpleNamespace(
                meta_mutation_rate=0.0,
                meta_roi_weight=0.0,
                meta_domain_penalty=0.0,
                meta_entropy_threshold=0.2,
                enable_meta_planner=True,
            ),
            "_init": types.SimpleNamespace(
                settings=types.SimpleNamespace(
                    meta_mutation_rate=0.0,
                    meta_roi_weight=0.0,
                    meta_domain_penalty=0.0,
                    meta_entropy_threshold=0.2,
                    enable_meta_planner=True,
                )
            ),
            "get_stable_workflows": lambda: types.SimpleNamespace(),
            "_FallbackPlanner": object,
        }
    )

    with pytest.raises(RuntimeError):
        asyncio.run(meta["self_improvement_cycle"]({}))


def test_cycle_thread_logs_cancellation(monkeypatch, caplog):
    import importlib.util

    path = Path("self_improvement/meta_planning.py")
    spec = importlib.util.spec_from_file_location(
        "menace_sandbox.self_improvement.meta_planning", path
    )
    mp = importlib.util.module_from_spec(spec)
    root_pkg = types.ModuleType("menace_sandbox")
    root_pkg.__path__ = [str(Path("."))]
    sub_pkg = types.ModuleType("menace_sandbox.self_improvement")
    sub_pkg.__path__ = [str(path.parent)]
    monkeypatch.setitem(sys.modules, "menace_sandbox", root_pkg)
    monkeypatch.setitem(sys.modules, "menace_sandbox.self_improvement", sub_pkg)
    stub_settings = types.SimpleNamespace(
        meta_entropy_threshold=0.2,
        meta_mutation_rate=0.0,
        meta_roi_weight=0.0,
        meta_domain_penalty=0.0,
        meta_entropy_weight=0.0,
        meta_search_depth=1,
        meta_beam_width=1,
    )
    monkeypatch.setitem(
        sys.modules,
        "menace_sandbox.self_improvement.init",
        types.SimpleNamespace(settings=stub_settings),
    )
    monkeypatch.setitem(
        sys.modules,
        "menace_sandbox.workflow_stability_db",
        types.SimpleNamespace(WorkflowStabilityDB=lambda *a, **k: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "menace_sandbox.roi_results_db",
        types.SimpleNamespace(ROIResultsDB=lambda *a, **k: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "menace_sandbox.lock_utils",
        types.SimpleNamespace(SandboxLock=object, Timeout=Exception, LOCK_TIMEOUT=1),
    )
    import logging_utils as real_logging_utils

    monkeypatch.setitem(
        sys.modules,
        "menace_sandbox.logging_utils",
        real_logging_utils,
    )
    monkeypatch.setitem(
        sys.modules,
        "menace_sandbox.sandbox_settings",
        types.SimpleNamespace(SandboxSettings=object),
    )
    monkeypatch.setitem(
        sys.modules, "menace_sandbox.self_improvement.meta_planning", mp
    )
    assert spec.loader is not None
    spec.loader.exec_module(mp)

    class Counter:
        def __init__(self) -> None:
            self.reason = None
            self.count = 0

        def labels(self, reason: str):  # pragma: no cover - simple stub
            self.reason = reason

            def inc() -> None:
                self.count += 1

            return types.SimpleNamespace(inc=inc)

    me = types.SimpleNamespace(self_improvement_failure_total=Counter())
    monkeypatch.setitem(sys.modules, "menace_sandbox.metrics_exporter", me)
    monkeypatch.setitem(sys.modules, "metrics_exporter", me)

    async def dummy_cycle(workflows, interval, event_bus=None, stop_event=None):
        await asyncio.Event().wait()

    monkeypatch.setattr(mp, "self_improvement_cycle", dummy_cycle)

    with caplog.at_level(logging.INFO):
        thread = mp.start_self_improvement_cycle({})
        thread.start()
        for _ in range(100):
            if thread._task is not None:
                break
            time.sleep(0.01)
        assert thread._task is not None
        thread._loop.call_soon_threadsafe(lambda: thread._task.cancel())
        thread.join()

    assert me.self_improvement_failure_total.count > 0
