import asyncio
import importlib
import importlib.util
import json
import sys
import types
from dataclasses import dataclass as _dc
from pathlib import Path

import pytest
from prometheus_client import REGISTRY

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Provide minimal stubs for optional dependencies

import pydantic as _pydantic
sub = types.ModuleType("pydantic.dataclasses")
sub.dataclass = _dc
_pydantic.dataclasses = sub  # type: ignore[attr-defined]
sys.modules.setdefault("pydantic", _pydantic)
sys.modules.setdefault("pydantic.dataclasses", sub)

jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: object()
sys.modules.setdefault("jinja2", jinja_mod)

yaml_mod = types.ModuleType("yaml")
yaml_mod.safe_load = lambda *a, **k: {}
yaml_mod.dump = lambda *a, **k: ""
sys.modules.setdefault("yaml", yaml_mod)

REGISTRY._names_to_collectors.clear()

# real metrics exporter so gauge values can be inspected
import metrics_exporter
sys.modules["menace.metrics_exporter"] = metrics_exporter
sys.modules["metrics_exporter"] = metrics_exporter

# lightweight stubs for heavy dependencies
env_mod = types.ModuleType("menace.environment_generator")
env_mod._CPU_LIMITS = {}
env_mod._MEMORY_LIMITS = {}
sys.modules["menace.environment_generator"] = env_mod

alert_mod = types.ModuleType("alert_dispatcher")
alert_mod.dispatch_alert = lambda *a, **k: None
sys.modules["alert_dispatcher"] = alert_mod

dmm = types.ModuleType("dynamic_module_mapper")
dmm.build_module_map = lambda *a, **k: {}
dmm.discover_module_groups = lambda *a, **k: {}
sys.modules["dynamic_module_mapper"] = dmm

map_mod = types.ModuleType("menace.model_automation_pipeline")
class _Dummy: ...
map_mod.ModelAutomationPipeline = _Dummy
map_mod.ResearchAggregatorBot = _Dummy
map_mod.InformationSynthesisBot = _Dummy
map_mod.TaskValidationBot = _Dummy
map_mod.BotPlanningBot = _Dummy
map_mod.HierarchyAssessmentBot = _Dummy
map_mod.ResourcePredictionBot = _Dummy
map_mod.DataBot = _Dummy
map_mod.CapitalManagementBot = _Dummy
map_mod.PreExecutionROIBot = _Dummy
map_mod.AutomationResult = _Dummy
sys.modules["menace.model_automation_pipeline"] = map_mod

# placeholder environment module for sandbox_runner import
sys.modules["sandbox_runner.environment"] = types.ModuleType("sandbox_runner.environment")

# load menace modules from source
menace_pkg = sys.modules.setdefault("menace", types.ModuleType("menace"))
menace_pkg.__path__ = [str(ROOT)]

spec_sts = importlib.util.spec_from_file_location(
    "menace.self_test_service", ROOT / "self_test_service.py"  # path-ignore
)
sts = importlib.util.module_from_spec(spec_sts)
sys.modules["menace.self_test_service"] = sts
spec_sts.loader.exec_module(sts)

spec_sie = importlib.util.spec_from_file_location(
    "menace.self_improvement", ROOT / "self_improvement.py"  # path-ignore
)
sie = importlib.util.module_from_spec(spec_sie)
sys.modules["menace.self_improvement"] = sie
spec_sie.loader.exec_module(sie)

# ---------------------------------------------------------------------------

def _build_repo(tmp_path: Path):
    (tmp_path / "a.py").write_text("import b\nimport fail\n")  # path-ignore
    (tmp_path / "b.py").write_text("import c\nimport helper\n")  # path-ignore
    (tmp_path / "c.py").write_text("import red\n")  # path-ignore
    (tmp_path / "helper.py").write_text("VALUE = 1\n")  # path-ignore
    (tmp_path / "red.py").write_text("# deprecated\nVALUE = 2\n")  # path-ignore
    (tmp_path / "fail.py").write_text("VALUE = 3\n")  # path-ignore
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    map_path = data_dir / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))
    return data_dir, map_path

# ---------------------------------------------------------------------------


def test_orphan_pipelines_prune_and_record_metrics(tmp_path, monkeypatch):
    """End-to-end test exercising self_test_service and self_improvement."""

    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    data_dir, map_path = _build_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    sr = importlib.import_module("sandbox_runner")
    sr.discover_recursive_orphans(str(tmp_path), module_map=str(map_path))

    for g in (
        metrics_exporter.orphan_modules_tested_total,
        metrics_exporter.orphan_modules_reintroduced_total,
        metrics_exporter.orphan_modules_failed_total,
        metrics_exporter.orphan_modules_legacy_total,
    ):
        g.set(0)

    calls: dict[str, list[list[str]]] = {"include": [], "workflows": []}

    def fake_auto_include(mods, recursive=False, validate=True, context_builder=None):
        calls["include"].append(sorted(mods))
        return [1]

    def fake_try(mods, context_builder=None):
        calls["workflows"].append(sorted(mods))

    env = types.SimpleNamespace(
        auto_include_modules=fake_auto_include,
        try_integrate_into_workflows=fake_try,
    )
    monkeypatch.setattr(sie, "environment", env)

    class DummySTS:
        def __init__(self, *a, pytest_args: str = "", **k):
            self._mods = pytest_args.split()
            self.results: dict[str, object] = {}
        async def _run_once(self) -> None:
            passed = [m for m in self._mods if "fail.py" not in m and "red.py" not in m]  # path-ignore
            self.results = {
                "orphan_passed": passed,
                "orphan_redundant": [m for m in self._mods if "red.py" in m],  # path-ignore
                "failed": 0,
            }

    monkeypatch.setitem(sys.modules, "self_test_service", types.SimpleNamespace(SelfTestService=DummySTS))

    class DummyIndex:
        def __init__(self):
            self.refreshed: list[list[str]] = []
        def refresh(self, mods, force=True):
            self.refreshed.append(sorted(mods))
        def get(self, mod):
            return 1
        def save(self):
            pass

    class DummyTracker:
        def __init__(self):
            self.roi_history = [0.0]
            self.metrics: list[dict[str, float] | None] = []

        def register_metrics(self, *names):
            self.registered = names

        def update(self, before, after, modules=None, resources=None, metrics=None):
            self.metrics.append(metrics)

    class DummyPreRoi:
        def predict_model_roi(self, model, tasks):
            return types.SimpleNamespace(roi=1.0)

    engine = types.SimpleNamespace(
        module_index=DummyIndex(),
        module_clusters={},
        logger=types.SimpleNamespace(info=lambda *a, **k: None, exception=lambda *a, **k: None),
        data_bot=types.SimpleNamespace(metrics_db=types.SimpleNamespace(log_eval=lambda *a, **k: None)),
        orphan_traces={},
        _update_orphan_modules=lambda: None,
        pre_roi_bot=DummyPreRoi(),
        tracker=DummyTracker(),
    )

    mods = [str(tmp_path / m) for m in ["a.py", "b.py", "c.py", "helper.py", "red.py", "fail.py"]]  # path-ignore
    passing = sie.SelfImprovementEngine._test_orphan_modules(engine, mods)
    integrated = sie.SelfImprovementEngine._integrate_orphans(engine, passing)

    assert passing == {"a.py", "b.py", "c.py", "helper.py"}  # path-ignore
    assert passing <= integrated
    assert calls["include"] and set(["a.py", "b.py", "c.py", "helper.py"]).issubset(set(calls["include"][-1]))  # path-ignore
    assert calls["workflows"] and set(["a.py", "b.py", "c.py", "helper.py"]).issubset(set(calls["workflows"][0]))  # path-ignore
    assert metrics_exporter.orphan_modules_tested_total._value.get() == 6
    assert metrics_exporter.orphan_modules_reintroduced_total._value.get() >= 8
    assert metrics_exporter.orphan_modules_failed_total._value.get() == 1
    assert metrics_exporter.orphan_modules_legacy_total._value.get() == 2
    assert engine._last_orphan_metrics["pass_rate"] == pytest.approx(4 / 6)
    assert engine._last_orphan_metrics["avg_roi"] == 1.0
    assert engine.tracker.metrics and engine.tracker.metrics[-1]["orphan_pass_rate"] == pytest.approx(4 / 6)
    assert engine.tracker.metrics[-1]["orphan_avg_roi"] == 1.0


def test_side_effect_metric_increments(monkeypatch, tmp_path):
    """`try_integrate_into_workflows` should record skipped modules."""

    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path / "sandbox_data"))
    (tmp_path / "sandbox_data").mkdir()
    mod = tmp_path / "skip.py"  # path-ignore
    mod.write_text("VALUE = 1\n")

    stub_mod = types.ModuleType("module_index_db")
    stub_mod.ModuleIndexDB = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "module_index_db", stub_mod)
    stub_th = types.ModuleType("menace.task_handoff_bot")
    stub_th.WorkflowDB = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "menace.task_handoff_bot", stub_th)
    stub_sts = types.ModuleType("self_test_service")
    stub_sts.SelfTestService = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "self_test_service", stub_sts)

    metrics_exporter.orphan_modules_side_effects_total.set(0)

    monkeypatch.delitem(sys.modules, "sandbox_runner.environment", raising=False)
    env = importlib.import_module("sandbox_runner.environment")

    res = env.try_integrate_into_workflows(
        [str(mod)], side_effects={"skip.py": 11}  # path-ignore
    )

    assert res == []
    assert metrics_exporter.orphan_modules_side_effects_total._value.get() == 1
