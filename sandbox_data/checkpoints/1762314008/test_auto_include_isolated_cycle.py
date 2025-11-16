import json
import types
from pathlib import Path
import sys

from tests.test_recursive_orphans import (
    _load_methods,
    DummyIndex,
    DummyLogger,
)

ROOT = Path(__file__).resolve().parents[1]

_integrate_orphans, _update_orphan_modules, _refresh_module_map, _test_orphan_modules = _load_methods()

def test_auto_include_isolated_cycle(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "iso.py").write_text("import dep\n")  # path-ignore
    (repo / "dep.py").write_text("x = 1\n")  # path-ignore

    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()

    monkeypatch.chdir(repo)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SANDBOX_AUTO_INCLUDE_ISOLATED", "1")

    called = {}
    mod = types.ModuleType("scripts.discover_isolated_modules")

    def discover(path, *, recursive=True):
        called["recursive"] = recursive
        assert Path(path) == repo
        return ["iso.py", "dep.py"]  # path-ignore

    mod.discover_isolated_modules = discover
    pkg = types.ModuleType("scripts")
    pkg.discover_isolated_modules = mod
    monkeypatch.setitem(sys.modules, "scripts", pkg)
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", mod)

    index = DummyIndex(data_dir / "module_map.json")
    engine = types.SimpleNamespace(
        module_index=index,
        module_clusters={},
        logger=DummyLogger(),
    )
    engine._integrate_orphans = types.MethodType(_integrate_orphans, engine)
    engine._refresh_module_map = types.MethodType(_refresh_module_map, engine)
    engine._test_orphan_modules = types.MethodType(_test_orphan_modules, engine)
    engine._collect_recursive_modules = lambda mods: set(mods)

    _update_orphan_modules(engine)

    data = json.loads((data_dir / "module_map.json").read_text())
    assert set(data.get("modules", {})) == {"iso.py", "dep.py"}  # path-ignore
    orphans = json.loads((data_dir / "orphan_modules.json").read_text())
    assert orphans == []
    assert called.get("recursive") is True
