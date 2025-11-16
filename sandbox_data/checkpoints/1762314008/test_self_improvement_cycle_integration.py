import asyncio
import ast
import sys
import types
from pathlib import Path

import pytest
from sandbox_settings import SandboxSettings

ROOT = Path(__file__).resolve().parent.parent


def _load_meta_planning():
    src = (ROOT / "self_improvement" / "meta_planning.py").read_text()  # path-ignore
    tree = ast.parse(src)
    wanted = {"_get_entropy_threshold", "_should_encode", "self_improvement_cycle", "_evaluate_cycle"}
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in wanted
    ]
    future = ast.ImportFrom(module="__future__", names=[ast.alias("annotations", None)], level=0)
    module = ast.Module([future] + nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {
        "asyncio": asyncio,
        "PLANNER_INTERVAL": 0.0,
        "SandboxSettings": SandboxSettings,
        "WorkflowStabilityDB": object,
        "Any": object,
        "Callable": object,
        "Mapping": object,
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


def test_self_improvement_cycle_runs_with_patch_and_orphans(tmp_path):
    meta = _load_meta_planning()

    patch_calls = {"count": 0}
    integrate_calls = {"count": 0}

    def fake_generate_patch(*a, **k):
        patch_calls["count"] += 1
        return "patch"

    def fake_integrate(repo, router=None):
        integrate_calls["count"] += 1
        return []

    qfe = types.SimpleNamespace(generate_patch=fake_generate_patch)
    sys.modules["quick_fix_engine"] = qfe
    sys.modules["sandbox_runner.orphan_integration"] = types.SimpleNamespace(
        integrate_orphans=fake_integrate
    )

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
            self.data[wf] = {"roi": roi}

        def is_stable(self, wf, current_roi=None, threshold=None):
            return True

    planner_instances = []

    class DummyPlanner:
        def __init__(self):
            self.roi_db = DummyROI()
            self.stability_db = DummyStability()
            self.cluster_map = {}
            planner_instances.append(self)

        def discover_and_persist(self, workflows):
            import quick_fix_engine
            from sandbox_runner.orphan_integration import integrate_orphans

            quick_fix_engine.generate_patch("repo", object(), context_builder=object())
            integrate_orphans(tmp_path)
            return [
                {"chain": ["wf"], "roi_gain": 1.0, "failures": 0, "entropy": 0.0}
            ]

    meta.update(
        {
            "MetaWorkflowPlanner": DummyPlanner,
            "get_logger": lambda name: types.SimpleNamespace(
                warning=lambda *a, **k: None,
                exception=lambda *a, **k: None,
                debug=lambda *a, **k: None,
                error=lambda *a, **k: None,
            ),
            "log_record": lambda **kw: kw,
            "get_stable_workflows": lambda: DummyStability(),
        }
    )

    async def run_cycle():
        task = asyncio.create_task(
            meta["self_improvement_cycle"]({"wf": lambda: None}, interval=0.0)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_cycle())

    planner = planner_instances[0]
    assert planner.roi_db.logged
    assert planner.stability_db.recorded
    assert patch_calls["count"] > 0
    assert integrate_calls["count"] > 0
