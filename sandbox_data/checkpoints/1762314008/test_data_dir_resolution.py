import importlib
import sys
import types
from pathlib import Path

def _reload(module_name: str):
    module = sys.modules.get(module_name)
    if module is not None and not isinstance(module, types.ModuleType):
        del sys.modules[module_name]
        return importlib.import_module(module_name)
    if module is not None:
        return importlib.reload(module)
    return importlib.import_module(module_name)

def test_module_index_db_resolves_paths(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    data_dir = repo / "relative"
    data_dir.mkdir()
    (data_dir / "module_map.json").write_text("{}")
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", "relative")
    monkeypatch.setenv("SANDBOX_AUTO_MAP", "0")
    _reload("dynamic_path_router")
    mid = _reload("module_index_db")
    db = mid.ModuleIndexDB(auto_map=False)
    assert db.path == data_dir / "module_map.json"
