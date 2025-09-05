import importlib
import sys

def test_light_import_orphan_utils(tmp_path, monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    sys.modules.pop("sandbox_runner", None)
    sys.modules.pop("sandbox_runner.environment", None)
    sr = importlib.import_module("sandbox_runner")
    assert "sandbox_runner.environment" not in sys.modules

    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("x = 1\n")  # path-ignore

    orphans = sr.discover_orphan_modules(str(tmp_path), recursive=False)
    assert orphans == ["a"]

    rec = sr.discover_recursive_orphans(str(tmp_path))
    assert rec == {
        "a": {"parents": [], "redundant": False},
        "b": {"parents": ["a"], "redundant": False},
    }
