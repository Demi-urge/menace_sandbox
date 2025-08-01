import importlib
import sys


def test_recursive_orphan_import(tmp_path, monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    sys.modules.pop("sandbox_runner", None)
    sr = importlib.import_module("sandbox_runner")
    assert hasattr(sr, "discover_recursive_orphans")

    (tmp_path / "a.py").write_text("import b\n")
    (tmp_path / "b.py").write_text("x = 1\n")

    res = sr.discover_recursive_orphans(str(tmp_path))
    assert sorted(res) == ["a", "b"]
