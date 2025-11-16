import sys
import types
import importlib.util

import pytest

from dynamic_path_router import resolve_path
from meta_workflow_planner import MetaWorkflowPlanner

spec = importlib.util.spec_from_file_location(
    "workflow_sandbox_runner", resolve_path("workflow_sandbox_runner.py")  # path-ignore
)
wsr = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
assert spec.loader is not None
sys.modules[spec.name] = wsr
spec.loader.exec_module(wsr)  # type: ignore[misc]
WorkflowSandboxRunner = wsr.WorkflowSandboxRunner


class StubComparator:
    @staticmethod
    def _entropy(spec):
        return 0.0


def test_validate_chain_penalizes_improbable_domain_transitions(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "workflow_synergy_comparator",
        types.SimpleNamespace(WorkflowSynergyComparator=StubComparator),
    )
    planner = MetaWorkflowPlanner()
    planner.cluster_map = {("__domain_transitions__",): {(0, 0): {"count": 1.0, "delta_roi": 1.0}}}
    monkeypatch.setattr(
        planner,
        "_workflow_domain",
        lambda wid, workflows=None: ([0], ["alpha"]) if wid == "a" else ([1], ["beta"]),
    )

    def a():
        return 1.0

    def b():
        return 1.0

    workflows = {"a": a, "b": b}
    runner = WorkflowSandboxRunner()
    record = planner._validate_chain(["a", "b"], workflows, runner=runner, runs=1)
    assert record is not None
    assert record["roi_gain"] == pytest.approx(1.0)


def test_multi_domain_pipeline_records_transitions(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "workflow_synergy_comparator",
        types.SimpleNamespace(WorkflowSynergyComparator=StubComparator),
    )
    planner = MetaWorkflowPlanner()
    planner.domain_index = {
        "other": 0,
        "youtube": 1,
        "reddit": 2,
        "email": 3,
        "social": 4,
    }

    def domain_stub(wid, workflows=None):
        mapping = {
            "yt": ([1], ["youtube"]),
            "rd": ([2, 4], ["reddit", "social"]),
            "em": ([3], ["email"]),
        }
        return mapping[wid]

    monkeypatch.setattr(planner, "_workflow_domain", domain_stub)

    def step():
        return 1.0

    workflows = {"yt": step, "rd": step, "em": step}
    runner = WorkflowSandboxRunner()
    record = planner._validate_chain(["yt", "rd", "em"], workflows, runner=runner, runs=1)
    assert record is not None
    matrix = planner.cluster_map.get(("__domain_transitions__",), {})
    assert matrix[(1, 2)]["count"] == 1
    assert matrix[(2, 3)]["count"] == 1
    assert matrix[(1, 4)]["count"] == 1
    assert matrix[(4, 3)]["count"] == 1


class ModuleMetric:
    def __init__(self, name: str, result: float, success: bool = True) -> None:
        self.name = name
        self.result = result
        self.success = success
        self.duration = 0.0


class DummyRunner:
    def run(self, funcs):
        modules = []
        for fn in funcs:
            try:
                res = fn()
                modules.append(
                    ModuleMetric(
                        fn.__name__, res if isinstance(res, (int, float)) else 0.0, True
                    )
                )
            except Exception:
                modules.append(ModuleMetric(fn.__name__, 0.0, False))
                continue
            if isinstance(res, list):
                for sub in res:
                    try:
                        r = sub()
                        modules.append(
                            ModuleMetric(
                                sub.__name__,
                                r if isinstance(r, (int, float)) else 0.0,
                                True,
                            )
                        )
                    except Exception:
                        modules.append(ModuleMetric(sub.__name__, 0.0, False))
        crash_count = sum(1 for m in modules if not m.success)
        return types.SimpleNamespace(modules=modules, crash_count=crash_count)


def test_validate_chain_recursive_execution(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "workflow_synergy_comparator",
        types.SimpleNamespace(WorkflowSynergyComparator=StubComparator),
    )
    planner = MetaWorkflowPlanner()
    planner.cluster_map = {}
    monkeypatch.setattr(planner, "_workflow_domain", lambda wid, workflows=None: ([], []))

    def child_ok():
        return 1.0

    def child_fail():
        raise RuntimeError("boom")

    def parent():
        return [child_ok, child_fail]

    workflows = {"parent": parent, "child_ok": child_ok, "child_fail": child_fail}
    runner = DummyRunner()
    record = planner._validate_chain(
        ["parent"], workflows, runner=runner, runs=1, failure_threshold=10
    )
    assert record is not None
    assert record["roi_gain"] == pytest.approx(1.0)
    assert record["failures"] == pytest.approx(1 / 3)


def test_validate_chain_skips_empty_chain():
    planner = MetaWorkflowPlanner()

    class DummyRunner:
        def run(self, *_a, **_kw):  # pragma: no cover - should not be called
            raise AssertionError("runner.run should not be invoked")

    record = planner._validate_chain([], {}, runner=DummyRunner(), runs=1)
    assert record is None
