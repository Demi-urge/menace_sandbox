import ast
import asyncio
import logging
import sys
import threading
import time
import types
from pathlib import Path
from dynamic_path_router import resolve_path
from typing import Any, Callable, Mapping, Sequence
from self_improvement.baseline_tracker import BaselineTracker
import pytest


def _load_meta_planning():
    src = resolve_path("self_improvement/meta_planning.py").read_text()
    tree = ast.parse(src)
    wanted = {
        "_get_entropy_threshold",
        "_get_overfit_thresholds",
        "_should_encode",
        "self_improvement_cycle",
        "_evaluate_cycle",
        "_percentile",
    }
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
        "Sequence": Sequence,
        "TelemetryEvent": object,
        "DEFAULT_SEVERITY_SCORE_MAP": {},
        "BASELINE_TRACKER": types.SimpleNamespace(
            get=lambda m: 0.0, std=lambda m: 0.0, momentum=1.0, update=lambda **kw: None
        ),
    }
    exec(compile(module, "<ast>", "exec"), ns)
    ns["REQUIRED_METRICS"] = ("roi", "pass_rate", "entropy")
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


def test_overfit_thresholds_adjust_with_baseline():
    meta = _load_meta_planning()

    class Cfg:
        max_allowed_errors = None
        entropy_overfit_threshold = None

    tracker = BaselineTracker(window=4)
    tracker.update(error_count=1, entropy=0.0)
    tracker.update(error_count=1, entropy=0.0)
    tracker.update(error_count=1, entropy=0.1)
    max1, ent1 = meta["_get_overfit_thresholds"](Cfg(), tracker)

    tracker.update(error_count=5, entropy=0.5)
    max2, ent2 = meta["_get_overfit_thresholds"](Cfg(), tracker)

    assert max2 > max1
    assert ent2 > ent1


def test_should_encode_requires_positive_roi_and_low_entropy():
    meta = _load_meta_planning()
    should_encode = meta["_should_encode"]

    tracker1 = BaselineTracker(window=3)
    tracker1.update(roi=0.0, pass_rate=1.0, entropy=0.1)
    rec1 = {"roi_gain": 0.1, "entropy": 0.1, "failures": 0}
    tracker1.update(roi=rec1["roi_gain"], pass_rate=1.0, entropy=rec1["entropy"])
    ok1, reason1 = should_encode(rec1, tracker1, entropy_threshold=0.2)
    assert ok1 and reason1 == "improved"

    tracker2 = BaselineTracker(window=3)
    tracker2.update(roi=0.1, pass_rate=1.0, entropy=0.1)
    rec2 = {"roi_gain": 0.0, "entropy": 0.1, "failures": 0}
    tracker2.update(roi=rec2["roi_gain"], pass_rate=1.0, entropy=rec2["entropy"])
    ok2, reason2 = should_encode(rec2, tracker2, entropy_threshold=0.2)
    assert not ok2 and reason2 == "no_delta"

    tracker3 = BaselineTracker(window=3)
    tracker3.update(roi=0.0, pass_rate=1.0, entropy=0.1)
    rec3 = {"roi_gain": 0.1, "entropy": 0.31, "failures": 0}
    tracker3.update(roi=rec3["roi_gain"], pass_rate=1.0, entropy=rec3["entropy"])
    ok3, reason3 = should_encode(rec3, tracker3, entropy_threshold=0.2)
    assert not ok3 and reason3 == "entropy_spike"


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
                warning=lambda *a, **k: None,
                exception=lambda *a, **k: None,
                debug=lambda *a, **k: None,
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
        await asyncio.sleep(0.2)
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

    path = resolve_path("self_improvement/meta_planning.py")
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
        types.SimpleNamespace(SandboxSettings=object, DEFAULT_SEVERITY_SCORE_MAP={}),
    )
    monkeypatch.setitem(
        sys.modules,
        "menace_sandbox.error_logger",
        types.SimpleNamespace(TelemetryEvent=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "audit",
        types.SimpleNamespace(log_db_access=lambda *a, **k: None),
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


def test_overfitting_fallback_runs_when_entropy_shift_high():
    meta = _load_meta_planning()
    logs: list[tuple[str, Any]] = []
    stop_event = threading.Event()

    class DummyPlanner:
        roi_db = None
        stability_db = None

        def __init__(self) -> None:
            self.cluster_map = {}

        def discover_and_persist(self, workflows):
            stop_event.set()
            return [
                {
                    "chain": ["a"],
                    "roi_gain": 0.1,
                    "failures": 0,
                    "entropy": 0.0,
                    "errors": [],
                }
            ]

    meta.update(
        {
            "MetaWorkflowPlanner": None,
            "_FallbackPlanner": DummyPlanner,
            "get_logger": lambda name: types.SimpleNamespace(
                debug=lambda msg, extra=None: logs.append((msg, extra)),
                warning=lambda *a, **k: None,
                exception=lambda *a, **k: None,
            ),
            "log_record": lambda **kw: kw,
            "_init": types.SimpleNamespace(
                settings=types.SimpleNamespace(
                    meta_mutation_rate=0.0,
                    meta_roi_weight=0.0,
                    meta_domain_penalty=0.0,
                    meta_entropy_threshold=0.2,
                    overfitting_entropy_threshold=0.05,
                )
            ),
            "get_stable_workflows": lambda: types.SimpleNamespace(),
            "_evaluate_cycle": lambda tracker, errors: (
                "skip",
                {"reason": "test"},
            ),
            "BASELINE_TRACKER": types.SimpleNamespace(
                get=lambda m: 0.05 if m == "entropy_delta" else 0.0,
                update=lambda **kw: None,
                delta=lambda m: 0.1 if m == "entropy" else 0.0,
                to_dict=lambda: {"entropy_delta": [0.05, 0.1], "error_count": [0]},
                std=lambda m: 0.05,
                entropy_delta=0.1,
                _history={},
            ),
        }
    )

    asyncio.run(meta["self_improvement_cycle"]({"a": lambda: None}, stop_event=stop_event))

    assert stop_event.is_set()
