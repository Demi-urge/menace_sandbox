import importlib
import json
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


def test_recursive_orphan_import_chain(tmp_path, monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    sys.modules.pop("sandbox_runner", None)
    sr = importlib.import_module("sandbox_runner")
    assert hasattr(sr, "discover_recursive_orphans")

    # a -> b -> c should discover all modules
    (tmp_path / "a.py").write_text("import b\n")
    (tmp_path / "b.py").write_text("import c\n")
    (tmp_path / "c.py").write_text("x = 1\n")

    res = sr.discover_recursive_orphans(str(tmp_path))
    assert sorted(res) == ["a", "b", "c"]


def test_recursive_orphan_import_skips_known(tmp_path, monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    sys.modules.pop("sandbox_runner", None)
    sr = importlib.import_module("sandbox_runner")
    assert hasattr(sr, "discover_recursive_orphans")

    (tmp_path / "a.py").write_text("import b\n")
    (tmp_path / "b.py").write_text("import c\n")
    (tmp_path / "c.py").write_text("x = 1\n")

    sd = tmp_path / "sandbox_data"
    sd.mkdir()
    (sd / "module_map.json").write_text(json.dumps({"modules": {"b.py": 1}}))

    res = sr.discover_recursive_orphans(str(tmp_path))
    # 'b' is already tracked in module_map so traversal stops before reaching 'c'
    assert sorted(res) == ["a"]
