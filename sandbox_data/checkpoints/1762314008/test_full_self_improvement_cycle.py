import ast
import asyncio
import importlib.util
import sys
import types
from typing import Any, Callable, Mapping

import pytest

from dynamic_path_router import resolve_dir, resolve_path, repo_root


ROOT = repo_root()


def _load_meta_planning():
    src = resolve_path("self_improvement/meta_planning.py").read_text()  # path-ignore
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
            get=lambda m: 0.0, std=lambda m: 0.0, momentum=1.0, update=lambda **kw: None
        ),
        "_init": types.SimpleNamespace(
            settings=types.SimpleNamespace(
                meta_mutation_rate=0,
                meta_roi_weight=0,
                meta_domain_penalty=0,
                meta_entropy_threshold=None,
            )
        ),
    }
    exec(compile(module, "<ast>", "exec"), ns)
    return ns


# Dynamically load WorkflowSandboxRunner without importing the full package
package_path = resolve_dir("sandbox_runner")
package = types.ModuleType("sandbox_runner")
package.__path__ = [str(package_path)]
sys.modules["sandbox_runner"] = package
spec = importlib.util.spec_from_file_location(
    "sandbox_runner.workflow_sandbox_runner", resolve_path("workflow_sandbox_runner.py")  # path-ignore
)
wsr = importlib.util.module_from_spec(spec)
assert spec.loader
sys.modules[spec.name] = wsr
spec.loader.exec_module(wsr)
WorkflowSandboxRunner = wsr.WorkflowSandboxRunner


def test_full_self_improvement_cycle(monkeypatch):
    meta = _load_meta_planning()

    class DummyROI:
        def __init__(self):
            self.logged = []

        def log_result(self, **kw):
            self.logged.append(kw)

    class DummyStability:
        def __init__(self):
            self.recorded = []
            self.data = {}

        def record_metrics(self, wf, roi, failures, entropy, roi_delta=None):
            self.recorded.append((wf, roi, entropy))

    planner_instances: list[object] = []

    class DummyPlanner:
        def __init__(self):
            self.roi_db = DummyROI()
            self.stability_db = DummyStability()
            self.cluster_map = {}
            planner_instances.append(self)

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
    global_db = DummyStability()

    meta.update(
        {
            "MetaWorkflowPlanner": DummyPlanner,
            "get_logger": lambda name: types.SimpleNamespace(
                warning=lambda *a, **k: None, exception=lambda *a, **k: None
            ),
            "log_record": lambda **kw: kw,
            "_init": types.SimpleNamespace(
                settings=types.SimpleNamespace(
                    meta_mutation_rate=None,
                    meta_roi_weight=None,
                    meta_domain_penalty=None,
                    meta_entropy_threshold=None,
                )
            ),
            "get_stable_workflows": lambda: global_db,
        }
    )

    # Step 1: execute a workflow inside the sandbox
    runner = WorkflowSandboxRunner()

    def wf():
        return "ok"

    metrics = runner.run([wf], safe_mode=True)
    assert metrics.crash_count == 0

    # Step 2: run a single iteration of the self-improvement cycle
    async def run_cycle():
        await meta["self_improvement_cycle"]({"wf": wf}, interval=0.0, event_bus=bus)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(asyncio.wait_for(run_cycle(), timeout=0.01))

    planner = planner_instances[0]
    assert planner.roi_db.logged
    assert planner.stability_db.recorded
    assert global_db.recorded
    assert bus.events
