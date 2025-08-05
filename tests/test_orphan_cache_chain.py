import importlib
import json
import sys

import orphan_analyzer


def test_orphan_chain_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    sys.modules.pop("sandbox_runner", None)
    sr = importlib.import_module("sandbox_runner")
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (data_dir / "module_map.json").write_text(json.dumps({"modules": {}}))
    monkeypatch.setattr(orphan_analyzer, "analyze_redundancy", lambda p: p.name == "c.py")
    (tmp_path / "a.py").write_text("import b\n")
    (tmp_path / "b.py").write_text("x = 1\n")
    sr.discover_recursive_orphans(str(tmp_path), module_map=data_dir / "module_map.json")
    cache_path = data_dir / "orphan_modules.json"
    first = json.loads(cache_path.read_text())
    assert first == {
        "a.py": {"parents": [], "redundant": False},
        "b.py": {"parents": ["a"], "redundant": False},
    }
    (tmp_path / "b.py").write_text("import c\n")
    (tmp_path / "c.py").write_text("pass\n")
    sr.discover_recursive_orphans(str(tmp_path), module_map=data_dir / "module_map.json")
    cached = json.loads(cache_path.read_text())
    assert cached == {
        "a.py": {"parents": [], "redundant": False},
        "b.py": {"parents": ["a"], "redundant": False},
        "c.py": {"parents": ["b"], "redundant": True},
    }
