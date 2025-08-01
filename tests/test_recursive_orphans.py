import ast
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_methods():
    path = ROOT / "self_improvement_engine.py"
    src = path.read_text()
    tree = ast.parse(src)
    methods = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SelfImprovementEngine":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in {
                    "_integrate_orphans",
                    "_update_orphan_modules",
                    "_refresh_module_map",
                }:
                    methods.append(item)
    mod_dict = {
        "os": os,
        "json": json,
        "Path": Path,
        "time": __import__("time"),
        "datetime": __import__("datetime").datetime,
        "Iterable": __import__("typing").Iterable,
        "build_module_map": lambda repo, ignore=None: {},
        "generate_workflows_for_modules": lambda mods: None,
        "log_record": lambda **k: {},
    }
    ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    code = ast.Module(body=methods, type_ignores=[])
    exec(compile(code, str(path), "exec"), mod_dict)
    return (
        mod_dict["_integrate_orphans"],
        mod_dict["_update_orphan_modules"],
        mod_dict["_refresh_module_map"],
    )


_integrate_orphans, _update_orphan_modules, _refresh_module_map = _load_methods()


class DummyIndex:
    def __init__(self, path: Path | None = None) -> None:
        self._map = {}
        self._groups = {}
        self.path = path

    def refresh(self, modules=None, force=False):
        for m in modules or []:
            self._map[Path(m).name] = 1
            self._groups[str(1)] = 1

    def get(self, name):
        return 1

    def save(self):
        if self.path:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump({"modules": self._map, "groups": self._groups}, fh)


class DummyLogger:
    def exception(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass


def test_update_orphan_modules_recursive(monkeypatch, tmp_path):
    called = {}
    sr = types.ModuleType("sandbox_runner")

    def discover(repo_path: str, module_map=None):
        called["used"] = True
        called["repo"] = repo_path
        called["map"] = module_map
        return ["foo.bar"]

    sr.discover_recursive_orphans = discover
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr)

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SANDBOX_RECURSIVE_ORPHANS", "1")

    eng = types.SimpleNamespace(logger=DummyLogger())

    _update_orphan_modules(eng)

    assert called.get("used") is True
    assert Path(called["repo"]) == tmp_path
    assert Path(called["map"]).resolve() == (tmp_path / "module_map.json").resolve()
    data = json.loads((tmp_path / "orphan_modules.json").read_text())
    assert "foo/bar.py" in data


def test_refresh_module_map_triggers_update(monkeypatch, tmp_path):
    index = DummyIndex()
    eng = types.SimpleNamespace(
        module_index=index,
        module_clusters={},
        logger=DummyLogger(),
        _last_map_refresh=0.0,
        auto_refresh_map=True,
        meta_logger=None,
        patch_db=None,
        data_bot=None,
    )

    calls = {"update": 0}

    def fake_update(self):
        calls["update"] += 1

    eng._update_orphan_modules = types.MethodType(fake_update, eng)
    eng._integrate_orphans = types.MethodType(_integrate_orphans, eng)

    monkeypatch.setenv("SANDBOX_RECURSIVE_ORPHANS", "1")

    _refresh_module_map(eng, ["foo.py"])

    assert calls["update"] == 1


def test_isolated_modules_refresh_map(monkeypatch, tmp_path):
    index = DummyIndex(tmp_path / "module_map.json")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "iso.py").write_text("pass\n")

    mod = types.ModuleType("scripts.discover_isolated_modules")

    def discover(path):
        assert Path(path) == repo
        return ["iso.py"]

    mod.discover_isolated_modules = discover
    pkg = types.ModuleType("scripts")
    pkg.discover_isolated_modules = mod
    monkeypatch.setitem(sys.modules, "scripts", pkg)
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", mod)

    eng = types.SimpleNamespace(
        module_index=index,
        module_clusters={},
        logger=DummyLogger(),
    )
    eng._integrate_orphans = types.MethodType(_integrate_orphans, eng)
    eng._refresh_module_map = types.MethodType(_refresh_module_map, eng)

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SANDBOX_DISCOVER_ISOLATED", "1")

    _update_orphan_modules(eng)

    data = json.loads((tmp_path / "module_map.json").read_text())
    assert data["modules"].get("iso.py") == 1


def test_recursive_isolated(monkeypatch, tmp_path):
    index = DummyIndex(tmp_path / "module_map.json")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "iso.py").write_text("pass\n")

    called = {}
    mod = types.ModuleType("scripts.discover_isolated_modules")

    def discover(path, *, recursive=False):
        called["recursive"] = recursive
        assert Path(path) == repo
        return ["iso.py"]

    mod.discover_isolated_modules = discover
    pkg = types.ModuleType("scripts")
    pkg.discover_isolated_modules = mod
    monkeypatch.setitem(sys.modules, "scripts", pkg)
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", mod)

    eng = types.SimpleNamespace(
        module_index=index,
        module_clusters={},
        logger=DummyLogger(),
    )
    eng._integrate_orphans = types.MethodType(_integrate_orphans, eng)
    eng._refresh_module_map = types.MethodType(_refresh_module_map, eng)

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SANDBOX_DISCOVER_ISOLATED", "1")
    monkeypatch.setenv("SANDBOX_RECURSIVE_ISOLATED", "1")

    _update_orphan_modules(eng)

    data = json.loads((tmp_path / "module_map.json").read_text())
    assert data["modules"].get("iso.py") == 1
    assert called.get("recursive") is True
