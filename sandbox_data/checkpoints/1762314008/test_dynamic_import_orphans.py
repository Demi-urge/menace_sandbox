import importlib
import json
import orphan_analyzer


def test_dynamic_import_discovery(tmp_path, monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    # create dynamic import chain a -> b -> c
    (tmp_path / "a.py").write_text("import importlib as il\nil.import_module('b')\n")  # path-ignore
    (tmp_path / "b.py").write_text("__import__('c')\n")  # path-ignore
    (tmp_path / "c.py").write_text("VALUE = 1\n")  # path-ignore

    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    map_path = data_dir / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))

    monkeypatch.chdir(tmp_path)

    def fake_classify(path, *, include_meta=False, classifiers=None):
        return ("candidate", {}) if include_meta else "candidate"

    monkeypatch.setattr(orphan_analyzer, "classify_module", fake_classify)

    sr = importlib.import_module("sandbox_runner")
    result = sr.discover_recursive_orphans(str(tmp_path), module_map=str(map_path))

    assert result["a"]["parents"] == [] and not result["a"]["redundant"]
    assert result["b"]["parents"] == ["a"] and not result["b"]["redundant"]
    assert result["c"]["parents"] == ["b"] and not result["c"]["redundant"]

