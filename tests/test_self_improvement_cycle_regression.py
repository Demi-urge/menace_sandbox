from __future__ import annotations

import ast
import asyncio
import json
import types
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "regression"
ROOT = Path(__file__).resolve().parent.parent


def _load_meta_planning():
    src = (ROOT / "self_improvement" / "meta_planning.py").read_text()
    tree = ast.parse(src)
    wanted = {"_get_entropy_threshold", "_should_encode", "self_improvement_cycle"}
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in wanted
    ]
    module = ast.Module(nodes, type_ignores=[])
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
        "load_sandbox_settings": lambda: None,
    }
    exec(compile(module, "<ast>", "exec"), ns)
    return ns


def test_self_improvement_cycle_matches_fixture():
    meta = _load_meta_planning()

    class DummyROI:
        def __init__(self):
            self.logged = []

        def log_result(self, **kw):
            self.logged.append(kw)

    class DummyStability:
        def __init__(self):
            self.recorded = []

        def record_metrics(self, wf, roi, failures, entropy, roi_delta=None):
            self.recorded.append((wf, roi, entropy))

    class DummyPlanner:
        def __init__(self):
            self.roi_db = DummyROI()
            self.stability_db = DummyStability()
            self.cluster_map = {}

        def discover_and_persist(self, workflows):
            return [
                {"chain": ["wf"], "roi_gain": 0.5, "failures": 0, "entropy": 0.0}
            ]

    class DummyBus:
        def __init__(self):
            self.events: list[tuple[str, dict]] = []

        def publish(self, topic: str, payload: dict):
            self.events.append((topic, payload))

    bus = DummyBus()

    meta.update(
        {
            "MetaWorkflowPlanner": DummyPlanner,
            "get_logger": lambda name: types.SimpleNamespace(
                warning=lambda *a, **k: None, exception=lambda *a, **k: None
            ),
            "log_record": lambda **kw: kw,
            "STABLE_WORKFLOWS": DummyStability(),
            "load_sandbox_settings": lambda: types.SimpleNamespace(
                meta_mutation_rate=None,
                meta_roi_weight=None,
                meta_domain_penalty=None,
                meta_entropy_threshold=None,
            ),
        }
    )

    async def run_cycle():
        await meta["self_improvement_cycle"]({"wf": lambda: None}, interval=10, event_bus=bus)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(asyncio.wait_for(run_cycle(), timeout=0.01))

    expected = json.loads((FIXTURES / "self_improvement_events.json").read_text())
    assert [list(e) for e in bus.events] == expected
