import importlib
import shutil
import sys


def test_snippet_cache_dir_created_with_sandbox_repo_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    sys.modules.pop("dynamic_path_router", None)
    sys.modules.pop("chunking", None)
    chunking = importlib.import_module("chunking")

    cache_dir = chunking.SNIPPET_CACHE_DIR
    assert cache_dir == tmp_path / "chunk_summary_cache"
    assert cache_dir.is_dir()

    shutil.rmtree(cache_dir)
    assert not cache_dir.exists()

    chunking._store_snippet_summary("deadbeef", "summary")
    assert cache_dir.is_dir()
    assert (cache_dir / "deadbeef.json").exists()
