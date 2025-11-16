import importlib
import json
import sys


def test_recursive_orphan_import(tmp_path, monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    sys.modules.pop("sandbox_runner", None)
    sr = importlib.import_module("sandbox_runner")
    assert hasattr(sr, "discover_recursive_orphans")

    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("x = 1\n")  # path-ignore

    res = sr.discover_recursive_orphans(str(tmp_path))
    assert res == {
        "a": {
            "parents": [],
            "classification": "candidate",
            "redundant": False,
        },
        "b": {
            "parents": ["a"],
            "classification": "candidate",
            "redundant": False,
        },
    }


def test_recursive_orphan_import_chain(tmp_path, monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    sys.modules.pop("sandbox_runner", None)
    sr = importlib.import_module("sandbox_runner")
    assert hasattr(sr, "discover_recursive_orphans")

    # a -> b -> c should discover all modules
    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("import c\n")  # path-ignore
    (tmp_path / "c.py").write_text("x = 1\n")  # path-ignore

    res = sr.discover_recursive_orphans(str(tmp_path))
    assert res == {
        "a": {
            "parents": [],
            "classification": "candidate",
            "redundant": False,
        },
        "b": {
            "parents": ["a"],
            "classification": "candidate",
            "redundant": False,
        },
        "c": {
            "parents": ["b"],
            "classification": "candidate",
            "redundant": False,
        },
    }


def test_recursive_orphan_import_skips_known(tmp_path, monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    sys.modules.pop("sandbox_runner", None)
    sr = importlib.import_module("sandbox_runner")
    assert hasattr(sr, "discover_recursive_orphans")

    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("import c\n")  # path-ignore
    (tmp_path / "c.py").write_text("x = 1\n")  # path-ignore

    sd = tmp_path / "sandbox_data"
    sd.mkdir()
    (sd / "module_map.json").write_text(json.dumps({"modules": {"b.py": 1}}))  # path-ignore

    res = sr.discover_recursive_orphans(str(tmp_path))
    # 'b' is tracked in the module map but its dependency 'c' should still be discovered
    assert res == {
        "a": {
            "parents": [],
            "classification": "candidate",
            "redundant": False,
        },
        "c": {
            "parents": ["b"],
            "classification": "candidate",
            "redundant": False,
        },
    }
