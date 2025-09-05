import ast
import asyncio
import threading
import types
import queue
from pathlib import Path
from typing import Any, Callable, Mapping
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    src = (ROOT / "self_improvement" / "meta_planning.py").read_text()  # path-ignore
    tree = ast.parse(src)
    wanted = {"start_self_improvement_cycle", "self_improvement_cycle", "_evaluate_cycle"}
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in wanted
    ]
    module = ast.Module(nodes, type_ignores=[])
    module = ast.fix_missing_locations(module)
    ns = {
        "asyncio": asyncio,
        "threading": threading,
        "queue": queue,
        "PLANNER_INTERVAL": 0.0,
        "Mapping": Mapping,
        "Callable": Callable,
        "Any": Any,
        "ROIResultsDB": lambda *a, **k: None,
        "WorkflowStabilityDB": lambda *a, **k: None,
        "UnifiedEventBus": object,
        "get_logger": lambda name: types.SimpleNamespace(info=lambda *a, **k: None),
        "log_record": lambda **kw: kw,
        "_evaluate_cycle": lambda tracker, errors: ("run", {"reason": "noop"}),
        "BaselineTracker": type("BaselineTracker", (), {}),
        "_init": types.SimpleNamespace(
            settings=types.SimpleNamespace(
                meta_planning_interval=0.0,
                meta_mutation_rate=0,
                meta_roi_weight=0,
                meta_domain_penalty=0,
                meta_entropy_threshold=None,
            )
        ),
    }
    exec(compile(module, "<ast>", "exec"), ns)
    return ns


def test_start_self_improvement_cycle_runs_background_thread(monkeypatch):
    mod = _load_module()
    cycle_ran = threading.Event()
    workflow_calls: list[str] = []

    class DummyBus:
        def publish(self, topic: str, payload: dict) -> None:  # pragma: no cover - stub
            pass

    class DummyRunner:
        def run(self, wfs, safe_mode=True):
            for wf in wfs:
                wf()
            return types.SimpleNamespace(crash_count=0)

    async def fake_cycle(workflows: Mapping[str, Callable[[], Any]], interval=0.0, event_bus=None):
        runner = DummyRunner()
        runner.run(list(workflows.values()), safe_mode=True)
        cycle_ran.set()

    mod["self_improvement_cycle"] = fake_cycle

    def workflow():
        workflow_calls.append("ran")
        return "ok"

    thread = mod["start_self_improvement_cycle"](
        {"wf": workflow}, event_bus=DummyBus(), interval=0.0
    )

    thread.start()
    thread.join(timeout=1.0)
    assert cycle_ran.is_set()
    assert workflow_calls


def test_stop_cancels_cycle(monkeypatch):
    mod = _load_module()
    stopped = threading.Event()

    async def fake_cycle(workflows: Mapping[str, Callable[[], Any]], interval=0.0, event_bus=None):
        try:
            while True:
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            stopped.set()
            raise

    mod["self_improvement_cycle"] = fake_cycle

    thread = mod["start_self_improvement_cycle"]({"wf": lambda: None}, interval=0.0)
    thread.start()
    thread.stop()
    assert stopped.is_set()


def test_cycle_exception_propagated(monkeypatch):
    mod = _load_module()

    async def fake_cycle(workflows: Mapping[str, Callable[[], Any]], interval=0.0, event_bus=None):
        raise RuntimeError("boom")

    mod["self_improvement_cycle"] = fake_cycle

    thread = mod["start_self_improvement_cycle"]({"wf": lambda: None}, interval=0.0)
    thread.start()
    with pytest.raises(RuntimeError):
        thread.join(timeout=1.0)


def test_start_cycle_requires_evaluator():
    mod = _load_module()
    with pytest.raises(ValueError):
        mod["start_self_improvement_cycle"](
            {"wf": lambda: None}, interval=0.0, evaluate_cycle=None
        )
