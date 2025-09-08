import asyncio
import importlib.util
import json
import sys
import types
import time
from pathlib import Path
from typing import Iterable

from prometheus_client import REGISTRY

ROOT = Path(__file__).resolve().parents[1]

# provide minimal pydantic stub for optional dependencies
pyd_mod = types.ModuleType("pydantic")
sub = types.ModuleType("pydantic.dataclasses")
pyd_mod.BaseModel = type("BaseModel", (), {})
from dataclasses import dataclass as _dc
sub.dataclass = _dc
pyd_mod.dataclasses = sub
# stub jinja2 Template to avoid heavy dependency
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: object()
yaml_mod = types.ModuleType("yaml")
yaml_mod.safe_load = lambda *a, **k: {}
yaml_mod.dump = lambda *a, **k: ""

sys.modules.setdefault("pydantic", pyd_mod)
sys.modules.setdefault("pydantic.dataclasses", sub)
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("yaml", yaml_mod)

# reset prometheus registry to avoid duplicate metric errors when importing
REGISTRY._names_to_collectors.clear()

# load SelfTestService from source
spec = importlib.util.spec_from_file_location(
    "menace.self_test_service", ROOT / "self_test_service.py"  # path-ignore
)
sts = importlib.util.module_from_spec(spec)
sys.modules["menace.self_test_service"] = sts
spec.loader.exec_module(sts)

# load ModuleIndexDB
spec = importlib.util.spec_from_file_location(
    "module_index_db", ROOT / "module_index_db.py"  # path-ignore
)
mod_db = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod_db)


class DummyBuilder:
    def refresh_db_weights(self):
        pass

    def build_context(self, *a, **k):
        if k.get("return_metadata"):
            return "", {}
        return ""

class DummyLogger:
    def __init__(self) -> None:
        self.exc: list[str] = []

    def exception(self, msg: str, exc: Exception | None = None) -> None:
        self.exc.append(msg)

    def info(self, *a, **k) -> None:
        pass


class DummyEngine:
    def __init__(self, index: mod_db.ModuleIndexDB) -> None:
        self.module_index = index
        self.module_clusters: dict[str, int] = {}
        self.logger = DummyLogger()
        self._last_map_refresh = 0.0
        self.auto_refresh_map = True
        self.meta_logger = None

    def _integrate_orphans(self, paths: Iterable[str]) -> None:
        if not self.module_index:
            return
        mods = {Path(p).name for p in paths}
        unknown = [m for m in mods if m not in self.module_clusters]
        if not unknown:
            return
        try:
            self.module_index.refresh(mods, force=True)
            grp_map = {m: self.module_index.get(m) for m in mods}
            self.module_clusters.update(grp_map)
            self.module_index.save()
            self._last_map_refresh = time.time()
            from sandbox_runner import environment as _env
            _env.generate_workflows_for_modules(sorted(mods))
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.exception("orphan integration failed: %s", exc)

    def _refresh_module_map(self, modules: list[str] | None = None) -> None:
        if modules:
            self._integrate_orphans(modules)
            return


def test_recursive_inclusion_cleanup(tmp_path, monkeypatch):
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("x = 1\n")  # path-ignore
    (data_dir / "orphan_modules.json").write_text(json.dumps(["a.py"]))  # path-ignore

    commands: list[str] = []

    async def fake_exec(*cmd, **kwargs):
        commands.append(" ".join(map(str, cmd)))
        path = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
                break
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"summary": {"passed": 1, "failed": 0}}, fh)

        class P:
            returncode = 0

            async def communicate(self):
                return b"", b""

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.chdir(tmp_path)

    env_mod = types.ModuleType("environment")
    pkg = types.ModuleType("sandbox_runner")
    pkg.environment = env_mod
    pkg.discover_recursive_orphans = (
        lambda repo, module_map=None: {"a": [], "b": []}
    )
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg)
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)

    map_path = tmp_path / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))
    index = mod_db.ModuleIndexDB(map_path)
    engine = DummyEngine(index)

    def fake_refresh(self, modules=None, force=False):
        for m in modules or []:
            self._map[m] = 1
            self._groups[str(1)] = 1
        self.save()

    monkeypatch.setattr(mod_db.ModuleIndexDB, "refresh", fake_refresh)

    svc = sts.SelfTestService(
        include_orphans=True,
        integration_callback=engine._refresh_module_map,
        clean_orphans=True,
        context_builder=DummyBuilder(),
    )
    svc.run_once()

    assert any("a.py" in c for c in commands)  # path-ignore
    assert any("b.py" in c for c in commands)  # path-ignore
    assert svc.results.get("orphan_total") == 2
    assert svc.results.get("orphan_failed") == 0
    assert sorted(svc.results.get("orphan_passed", [])) == ["a.py", "b.py"]  # path-ignore

    data = json.loads(map_path.read_text())
    assert "a.py" in data.get("modules", {})  # path-ignore
    assert "b.py" in data.get("modules", {})  # path-ignore
    assert engine.module_clusters.get("a.py") == 1  # path-ignore
    assert engine.module_clusters.get("b.py") == 1  # path-ignore

    cleaned = json.loads((data_dir / "orphan_modules.json").read_text())
    assert cleaned == []


def test_local_import_inclusion_non_recursive(tmp_path, monkeypatch):
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("import c\n")  # path-ignore
    (tmp_path / "c.py").write_text("x = 1\n")  # path-ignore

    commands: list[str] = []

    async def fake_exec(*cmd, **kwargs):
        commands.append(" ".join(map(str, cmd)))
        path = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
                break
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"summary": {"passed": 1, "failed": 0}}, fh)

        class P:
            returncode = 0

            async def communicate(self):
                return b"", b""

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.chdir(tmp_path)

    def fake_find(root: Path):
        assert root == Path.cwd()
        return [Path("a.py")]  # path-ignore

    monkeypatch.setattr(
        "scripts.find_orphan_modules.find_orphan_modules", fake_find
    )
    monkeypatch.setattr(sts, "analyze_redundancy", lambda p: False)

    svc = sts.SelfTestService(
        include_orphans=True,
        discover_orphans=False,
        discover_isolated=False,
        recursive_orphans=False,
        context_builder=DummyBuilder(),
    )
    svc.run_once()

    assert any("a.py" in c for c in commands)  # path-ignore
    assert any("b.py" in c for c in commands)  # path-ignore
    assert any("c.py" in c for c in commands)  # path-ignore
    assert svc.results.get("orphan_total") == 3
    assert svc.results.get("orphan_failed") == 0
    assert sorted(svc.results.get("orphan_passed", [])) == [
        "a.py",  # path-ignore
        "b.py",  # path-ignore
        "c.py",  # path-ignore
    ]
