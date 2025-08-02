import json
import types
import time
import sys
from pathlib import Path
from typing import Iterable

from module_index_db import ModuleIndexDB
from orphan_analyzer import analyze_redundancy
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "menace.self_test_service", ROOT / "self_test_service.py"
)
sts = importlib.util.module_from_spec(spec)
pkg = sys.modules.get("menace")
if pkg is not None:
    pkg.__path__ = [str(ROOT)]
from prometheus_client import REGISTRY
REGISTRY._names_to_collectors.clear()
spec.loader.exec_module(sts)


class DummyLogger:
    def __init__(self) -> None:
        self.info_msgs: list[str] = []
        self.exc: list[str] = []

    def info(self, msg: str, extra=None) -> None:
        self.info_msgs.append(msg)

    def exception(self, msg: str, *a) -> None:
        self.exc.append(msg)


class DummyEngine:
    def __init__(self, index: ModuleIndexDB) -> None:
        self.module_index = index
        self.module_clusters: dict[str, int] = {}
        self.logger = DummyLogger()
        self._last_map_refresh = 0.0

    def _integrate_orphans(self, paths: Iterable[str]) -> None:
        if not self.module_index:
            return
        mods = set()
        for p in paths:
            path = Path(p)
            try:
                if analyze_redundancy(path):
                    self.logger.info("redundant module skipped", extra={"module": path.name})
                    continue
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("redundancy analysis failed", exc)
            mods.add(path.name)
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
            self.logger.exception("orphan integration failed", exc)



def _fake_refresh(self, modules=None, force=False):
    for m in modules or []:
        self._map[m] = 1
        self._groups[str(1)] = 1
    self.save()


def test_redundant_module_skipped(tmp_path, monkeypatch):
    map_path = tmp_path / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))
    index = ModuleIndexDB(map_path)
    engine = DummyEngine(index)

    monkeypatch.setattr(ModuleIndexDB, "refresh", _fake_refresh)

    def fake_analyze(path: Path) -> bool:
        return path.name == "dup.py"

    monkeypatch.setattr(sys.modules[__name__], "analyze_redundancy", fake_analyze)
    env_mod = types.ModuleType("env")
    wf_calls: list[list[str]] = []
    env_mod.generate_workflows_for_modules = lambda mods: wf_calls.append(mods)
    pkg = types.ModuleType("sandbox_runner")
    pkg.environment = env_mod
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg)
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)

    engine._integrate_orphans([str(tmp_path / "dup.py")])

    data = json.loads(map_path.read_text())
    assert "dup.py" not in data.get("modules", {})
    assert "redundant module skipped" in engine.logger.info_msgs
    assert not wf_calls


def test_module_integrated(tmp_path, monkeypatch):
    map_path = tmp_path / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))
    index = ModuleIndexDB(map_path)
    engine = DummyEngine(index)

    monkeypatch.setattr(ModuleIndexDB, "refresh", _fake_refresh)
    monkeypatch.setattr(sys.modules[__name__], "analyze_redundancy", lambda p: False)
    env_mod = types.ModuleType("env")
    env_mod.generate_workflows_for_modules = lambda mods: None
    pkg = types.ModuleType("sandbox_runner")
    pkg.environment = env_mod
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg)
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)

    engine._integrate_orphans([str(tmp_path / "ok.py")])

    assert engine.module_clusters.get("ok.py") == 1


def test_update_orphan_modules_filters(monkeypatch, tmp_path):
    from tests.test_recursive_orphans import _load_methods

    _, update, _, _ = _load_methods()

    sr = types.ModuleType("sandbox_runner")
    sr.discover_recursive_orphans = lambda repo, module_map=None: {"foo": []}
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr)

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SANDBOX_RECURSIVE_ORPHANS", "0")

    calls: list[Path] = []

    def fake_analyze(path: Path) -> bool:
        calls.append(path)
        return True

    update.__globals__["analyze_redundancy"] = fake_analyze

    eng = types.SimpleNamespace(logger=DummyLogger())

    update(eng)

    mod_file = tmp_path / "orphan_modules.json"
    data = json.loads(mod_file.read_text()) if mod_file.exists() else []
    assert "foo.py" not in data
    assert calls and calls[0].name == "foo.py"
    assert "redundant module skipped" in eng.logger.info_msgs
    assert "redundant modules skipped" in eng.logger.info_msgs


def test_discover_orphans_filters_recursive(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    svc = sts.SelfTestService(discover_isolated=False)
    svc.logger = DummyLogger()

    sr = types.ModuleType("sandbox_runner")
    sr.discover_recursive_orphans = (
        lambda repo, module_map=None: {"foo": [], "dup": []}
    )
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr)

    calls: list[Path] = []

    def fake_analyze(p: Path) -> bool:
        calls.append(p)
        return p.stem == "dup"

    monkeypatch.setattr(sts, "analyze_redundancy", fake_analyze)

    mods = svc._discover_orphans()

    assert mods == [str(Path("foo.py"))]
    assert sorted(c.name for c in calls) == ["dup.py", "foo.py"]
    assert "redundant module skipped" in svc.logger.info_msgs


def test_discover_orphans_filters_non_recursive(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    svc = sts.SelfTestService(discover_isolated=False, recursive_orphans=False)
    svc.logger = DummyLogger()

    def fake_find(root: Path):
        return [Path("foo.py"), Path("dup.py")]

    monkeypatch.setattr(
        "scripts.find_orphan_modules.find_orphan_modules", fake_find
    )

    calls: list[Path] = []

    def fake_analyze(p: Path) -> bool:
        calls.append(p)
        return p.stem == "dup"

    monkeypatch.setattr(sts, "analyze_redundancy", fake_analyze)

    mods = svc._discover_orphans()

    assert mods == [str(Path("foo.py"))]
    assert sorted(c.name for c in calls) == ["dup.py", "foo.py"]
    assert "redundant module skipped" in svc.logger.info_msgs

