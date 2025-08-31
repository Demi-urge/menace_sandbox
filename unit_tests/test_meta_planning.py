import ast
import asyncio
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
        "DEFAULT_ENTROPY_THRESHOLD": 0.2,
    }
    exec(compile(module, "<ast>", "exec"), ns)
    return ns


@pytest.mark.parametrize(
    "cfg_value, db_data, expected",
    [
        (0.7, {}, 0.7),
        (None, {"a": {"entropy": 0.2}, "b": {"entropy": -0.5}}, 0.5),
        (None, {}, 0.2),
    ],
)
def test_get_entropy_threshold(cfg_value, db_data, expected):
    meta = _load_meta_planning()

    class Cfg:
        meta_entropy_threshold = cfg_value

    class DB:
        data = db_data

    assert meta["_get_entropy_threshold"](Cfg(), DB()) == expected


def test_should_encode_requires_positive_roi_and_low_entropy():
    meta = _load_meta_planning()
    should_encode = meta["_should_encode"]

    assert should_encode({"roi_gain": 0.1, "entropy": 0.1}, entropy_threshold=0.2)
    assert not should_encode({"roi_gain": 0.0, "entropy": 0.1}, entropy_threshold=0.2)
    assert not should_encode({"roi_gain": 0.1, "entropy": 0.3}, entropy_threshold=0.2)


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
