import importlib

utils = importlib.import_module("menace.self_improvement.utils")


def test_remove_import_cache_files(tmp_path):
    root = tmp_path / "pkg"
    pycache = root / "__pycache__"
    pycache.mkdir(parents=True)
    (pycache / "mod.pyc").write_text("data")
    utils.remove_import_cache_files(root)
    assert not pycache.exists()
