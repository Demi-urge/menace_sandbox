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

pyd = types.ModuleType("pydantic")
sub = types.ModuleType("pydantic.dataclasses")
sub.dataclass = _dc
pyd.BaseModel = type("BaseModel", (), {})
pyd.dataclasses = sub
sys.modules.setdefault("pydantic", pyd)
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

# load SelfTestService from source
spec = importlib.util.spec_from_file_location(
    "menace.self_test_service", ROOT / "self_test_service.py"  # path-ignore
)
sts = importlib.util.module_from_spec(spec)
sys.modules.setdefault("menace", types.ModuleType("menace"))
sys.modules["menace.self_test_service"] = sts
spec.loader.exec_module(sts)


class DummyBuilder:
    def refresh_db_weights(self):
        pass

    def build_context(self, *a, **k):
        return "", "", {}


# ---------------------------------------------------------------------------

def test_recursive_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    # create two orphan chains in separate directories
    (tmp_path / "one").mkdir()
    (tmp_path / "one" / "__init__.py").write_text("\n")  # path-ignore
    (tmp_path / "one" / "a.py").write_text("import one.b\nVALUE = 1\n")  # path-ignore
    (tmp_path / "one" / "b.py").write_text("VALUE = 2\n")  # path-ignore
    (tmp_path / "two").mkdir()
    (tmp_path / "two" / "__init__.py").write_text("\n")  # path-ignore
    (tmp_path / "two" / "a.py").write_text("import two.b\nVALUE = 3\n")  # path-ignore
    (tmp_path / "two" / "b.py").write_text("VALUE = 4\n")  # path-ignore

    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    map_path = data_dir / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))

    monkeypatch.chdir(tmp_path)

    sr = importlib.import_module("sandbox_runner")
    mapping = sr.discover_recursive_orphans(str(tmp_path), module_map=str(map_path))
    assert set(mapping.keys()) == {"one.a", "one.b", "two.a", "two.b"}
    assert mapping["one.b"]["parents"] == ["one.a"]
    assert mapping["two.b"]["parents"] == ["two.a"]

    calls: list[list[str]] = []

    def fake_auto_include(mods, recursive=False, validate=False, context_builder=None):
        calls.append(sorted(mods))
        return [1]

    env_mod = types.ModuleType("sandbox_runner.environment")
    env_mod.auto_include_modules = fake_auto_include
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)

    def integrate(mods: list[str]) -> None:
        data = json.loads(map_path.read_text())
        for m in mods:
            data["modules"][Path(m).as_posix()] = 1
        map_path.write_text(json.dumps(data))
        env_mod.auto_include_modules(mods, recursive=True, context_builder=None)

    svc = sts.SelfTestService(
        include_orphans=True,
        recursive_orphans=True,
        clean_orphans=True,
        integration_callback=integrate,
        context_builder=DummyBuilder(),
    )
    mods = [Path(*name.split(".")).with_suffix(".py").as_posix() for name in mapping]  # path-ignore
    svc.integration_callback(mods)

    assert calls and calls[0] == [
        "one/a.py",  # path-ignore
        "one/b.py",  # path-ignore
        "two/a.py",  # path-ignore
        "two/b.py",  # path-ignore
    ]
    data = json.loads(map_path.read_text())
    assert set(data["modules"].keys()) == {"one/a.py", "one/b.py", "two/a.py", "two/b.py"}  # path-ignore

    # ------------------------------------------------------------------
    # SelfImprovementEngine should also integrate using full paths

    sie_calls: list[list[str]] = []

    def fake_auto_include_sie(mods, recursive=False, validate=True):
        sie_calls.append(sorted(mods))
        return [1]

    env_mod.auto_include_modules = fake_auto_include_sie

    from tests.test_self_improvement_logging import _load_engine
    log_stub = sys.modules["menace.logging_utils"]
    class _Log:
        def info(self, *a, **k):
            pass

        def exception(self, *a, **k):
            pass

    log_stub.get_logger = lambda name=None: _Log()
    log_stub.setup_logging = lambda *a, **k: None
    log_stub.set_correlation_id = lambda *a, **k: None
    sys.modules["menace"].RAISE_ERRORS = False

    class DummySTS:
        def __init__(self, *a, pytest_args: str = "", **k):
            self._mods = pytest_args.split()
            self.results: dict[str, object] = {}

        async def _run_once(self) -> None:
            self.results = {
                "orphan_passed": self._mods,
                "orphan_redundant": [],
                "failed": 0,
            }

    monkeypatch.setitem(
        sys.modules,
        "self_test_service",
        types.SimpleNamespace(SelfTestService=DummySTS),
    )

    sie = _load_engine()

    class DummyMetricsDB:
        def __init__(self):
            self.records: list[tuple[str, str, float]] = []

        def log_eval(self, cycle, metric, value):
            self.records.append((cycle, metric, value))

    class DummyDataBot:
        def __init__(self):
            self.metrics_db = DummyMetricsDB()

    engine = types.SimpleNamespace(
        logger=types.SimpleNamespace(
            info=lambda *a, **k: None, exception=lambda *a, **k: None
        ),
        data_bot=DummyDataBot(),
        orphan_traces={},
    )

    mods = ["one/a.py", "one/b.py", "two/a.py", "two/b.py"]  # path-ignore
    result = sie.SelfImprovementEngine._test_orphan_modules(engine, mods)
    assert result == set(mods)
    assert sie_calls and sie_calls[0] == sorted(mods)
