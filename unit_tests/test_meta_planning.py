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


def test_cycle_logs_when_planner_missing():
    meta = _load_meta_planning()

    messages: list[str] = []
    meta.update(
        {
            "MetaWorkflowPlanner": None,
            "get_logger": lambda name: types.SimpleNamespace(
                warning=lambda msg: messages.append(msg)
            ),
            "log_record": lambda **kw: kw,
            "SandboxSettings": lambda: types.SimpleNamespace(
                meta_mutation_rate=None,
                meta_roi_weight=None,
                meta_domain_penalty=None,
                meta_entropy_threshold=0.2,
            ),
            "STABLE_WORKFLOWS": types.SimpleNamespace(),
        }
    )

    asyncio.run(meta["self_improvement_cycle"]({}))
    assert any("MetaWorkflowPlanner unavailable" in m for m in messages)
