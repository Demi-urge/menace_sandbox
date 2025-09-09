import asyncio
import importlib
import importlib.util
import json
import sys
import types
from dataclasses import dataclass as _dc
from pathlib import Path

import pytest

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

from prometheus_client import REGISTRY
REGISTRY._names_to_collectors.clear()

# load core menace modules from source
spec_sts = importlib.util.spec_from_file_location(
    "menace.self_test_service", ROOT / "self_test_service.py"  # path-ignore
)
sts = importlib.util.module_from_spec(spec_sts)
menace_pkg = sys.modules.setdefault("menace", types.ModuleType("menace"))
sys.modules["menace.self_test_service"] = sts
spec_sts.loader.exec_module(sts)


class DummyBuilder:
    def refresh_db_weights(self):
        pass

    def build_context(self, *a, **k):
        return "", "", {}

env_mod = types.ModuleType("menace.environment_generator")
env_mod._CPU_LIMITS = {}
env_mod._MEMORY_LIMITS = {}
sys.modules["menace.environment_generator"] = env_mod

# lightweight stubs for heavy dependencies
alert_mod = types.ModuleType("alert_dispatcher")
alert_mod.dispatch_alert = lambda *a, **k: None
sys.modules["alert_dispatcher"] = alert_mod

def _counter():
    class C:
        def inc(self, *a, **k):
            pass
        def set(self, *a, **k):
            pass
    return C()

metrics = types.ModuleType("metrics_exporter")
metrics.synergy_weight_updates_total = _counter()
metrics.synergy_weight_update_failures_total = _counter()
metrics.synergy_weight_update_alerts_total = _counter()
metrics.orphan_modules_reintroduced_total = _counter()
metrics.orphan_modules_passed_total = _counter()
metrics.orphan_modules_tested_total = _counter()
metrics.orphan_modules_failed_total = _counter()
metrics.orphan_modules_reclassified_total = _counter()
metrics.orphan_modules_redundant_total = _counter()
metrics.orphan_modules_legacy_total = _counter()
metrics.learning_cv_score = _counter()
metrics.learning_holdout_score = _counter()
sys.modules["menace.metrics_exporter"] = metrics
sys.modules["metrics_exporter"] = metrics

dmm = types.ModuleType("dynamic_module_mapper")
dmm.build_module_map = lambda *a, **k: {}
dmm.discover_module_groups = lambda *a, **k: {}
sys.modules["dynamic_module_mapper"] = dmm

map_mod = types.ModuleType("menace.model_automation_pipeline")
class _Dummy:
    pass
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

sys.modules["sandbox_runner.environment"] = types.ModuleType(
    "sandbox_runner.environment"
)

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


def test_discover_recursive_orphans_classification(tmp_path, monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    data_dir, map_path = _build_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    sr = importlib.import_module("sandbox_runner")
    mapping = sr.discover_recursive_orphans(str(tmp_path), module_map=str(map_path))
    assert mapping == {
        "a": {"parents": [], "classification": "candidate", "redundant": False},
        "b": {"parents": ["a"], "classification": "candidate", "redundant": False},
        "fail": {"parents": ["a"], "classification": "candidate", "redundant": False},
        "c": {"parents": ["b"], "classification": "candidate", "redundant": False},
        "helper": {"parents": ["b"], "classification": "candidate", "redundant": False},
        "red": {"parents": ["c"], "classification": "legacy", "redundant": True},
    }


def test_self_test_service_executes_and_cleans(tmp_path, monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    data_dir, map_path = _build_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    sr = importlib.import_module("sandbox_runner")
    sr.discover_recursive_orphans(str(tmp_path), module_map=str(map_path))

    generated: list[list[str]] = []

    def fake_generate(mods, workflows_db="workflows.db"):
        generated.append(list(mods))
        return [1]

    env = types.ModuleType("sandbox_runner.environment")
    env.generate_workflows_for_modules = fake_generate
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env)

    async def fake_exec(*cmd, **kwargs):
        path = None
        mod = str(cmd[-1]) if cmd else ""
        for i, a in enumerate(cmd):
            s = str(a)
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
                break
        if path:
            failed = "fail.py" in mod  # path-ignore
            Path(path).write_text(
                json.dumps({"summary": {"passed": 0 if failed else 1, "failed": 1 if failed else 0}})
            )
        class P:
            returncode = 0
            async def communicate(self):
                return b"", b""
            async def wait(self):
                return None
        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setenv("SANDBOX_CLEAN_ORPHANS", "1")

    def integrate(mods: list[str]) -> None:
        mods[:] = [m for m in mods if Path(m).name != "fail.py"]  # path-ignore
        data = json.loads(map_path.read_text())
        for m in mods:
            name = Path(m).name
            data["modules"][name] = 1
            env.generate_workflows_for_modules([name])
        map_path.write_text(json.dumps(data))

    svc = sts.SelfTestService(
        include_orphans=True,
        recursive_orphans=True,
        clean_orphans=True,
        integration_callback=integrate,
        context_builder=DummyBuilder(),
    )
    svc.run_once()

    assert [sorted(g) for g in generated] == [["a.py"], ["b.py"], ["c.py"], ["helper.py"]]  # path-ignore
    data = json.loads(map_path.read_text())
    assert set(data["modules"]) == {"a.py", "b.py", "c.py", "helper.py"}  # path-ignore
    orphan_list = json.loads((data_dir / "orphan_modules.json").read_text())
    assert orphan_list == ["red.py"]  # path-ignore


def test_self_improvement_integration(tmp_path, monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    data_dir, map_path = _build_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    sr = importlib.import_module("sandbox_runner")
    sr.discover_recursive_orphans(str(tmp_path), module_map=str(map_path))

    calls: list[list[str]] = []

    def fake_auto_include(mods, recursive=False, validate=True):
        calls.append(sorted(mods))
        return [1]

    env = types.ModuleType("sandbox_runner.environment")
    env.auto_include_modules = fake_auto_include
    env.try_integrate_into_workflows = lambda mods: None
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

    monkeypatch.setitem(
        sys.modules,
        "self_test_service",
        types.SimpleNamespace(SelfTestService=DummySTS),
    )

    class DummyIndex:
        def __init__(self):
            self.refreshed: list[list[str]] = []
        def refresh(self, mods, force=True):
            self.refreshed.append(sorted(mods))
        def get(self, mod):
            return 1
        def save(self):
            pass

    class DummyMetricsDB:
        def __init__(self):
            self.records: list[tuple[str, float]] = []
        def log_eval(self, cycle, metric, value):
            self.records.append((metric, value))

    class DummyDataBot:
        def __init__(self):
            self.metrics_db = DummyMetricsDB()

    engine = types.SimpleNamespace(
        module_index=DummyIndex(),
        module_clusters={},
        logger=types.SimpleNamespace(info=lambda *a, **k: None, exception=lambda *a, **k: None),
        data_bot=DummyDataBot(),
        orphan_traces={},
        _update_orphan_modules=lambda: None,
    )

    mods = ["a.py", "b.py", "c.py", "helper.py", "red.py", "fail.py"]  # path-ignore
    passing = sie.SelfImprovementEngine._test_orphan_modules(engine, mods)
    assert passing == {"a.py", "b.py", "c.py", "helper.py"}  # path-ignore

    integrated = sie.SelfImprovementEngine._integrate_orphans(engine, passing)
    assert integrated == {"a.py", "b.py", "c.py", "helper.py"}  # path-ignore
    assert calls and calls[0] == sorted(["a.py", "b.py", "c.py", "helper.py"])  # path-ignore
    assert set(engine.module_clusters) == {"a.py", "b.py", "c.py", "helper.py"}  # path-ignore
    orphan_list = json.loads((data_dir / "orphan_modules.json").read_text())
    assert orphan_list == ["fail.py"]  # path-ignore
