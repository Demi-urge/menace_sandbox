import sys
import importlib.util
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def _load_dpr():
    spec = importlib.util.spec_from_file_location(
        "dynamic_path_router", ROOT / "dynamic_path_router.py"  # path-ignore
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules["dynamic_path_router"] = module
    return module


def test_patch_suggestion_db_resolves_after_repo_move(tmp_path, monkeypatch):
    dpr = _load_dpr()
    repo = tmp_path / "repo"
    sandbox = repo / "sandbox_data"
    sandbox.mkdir(parents=True)
    (repo / "logs").mkdir()
    cfg_dir = repo / "config"
    cfg_dir.mkdir()
    shutil.copy(ROOT / "config" / "db_router_tables.json", cfg_dir / "db_router_tables.json")
    dpr.clear_cache()
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    data_dir = dpr.resolve_path("sandbox_data")
    from patch_suggestion_db import PatchSuggestionDB

    db = PatchSuggestionDB()
    try:
        assert db.path == data_dir / "suggestions.db"  # path-ignore
    finally:
        db.conn.close()
