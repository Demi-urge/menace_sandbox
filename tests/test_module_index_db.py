import json
import pytest
from module_index_db import ModuleIndexDB


def test_merge_groups(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"a": 0, "b": 1}))
    db = ModuleIndexDB(path)
    old_a = db.get("a")
    groups = {"a": 0, "b": 0, "c": 0}
    db.merge_groups(groups)
    assert db.get("a") == old_a
    # existing index for b should remain unchanged
    assert db.get("b") == 1
    # new module shares index with the first known group member
    assert db.get("c") == old_a


def test_get_resolves_suffix(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"foo": 7}))
    db = ModuleIndexDB(path)
    assert db.get("foo.py") == 7  # path-ignore


def test_distinct_paths(tmp_path):
    db = ModuleIndexDB(tmp_path / "map.json")
    idx_a = db.get("a/foo.py")  # path-ignore
    idx_b = db.get("b/foo.py")  # path-ignore
    assert idx_a != idx_b



def test_auto_map(monkeypatch, tmp_path):
    generated = {}
    def fake_generate(output, *, root, algorithm, threshold, semantic, exclude=None):
        generated.update(dict(output=output, root=root, algorithm=algorithm, threshold=threshold, semantic=semantic))
        output.write_text(json.dumps({"mod": 3}))
        return {"mod": 3}
    monkeypatch.setattr("module_index_db.generate_module_map", fake_generate)
    monkeypatch.setenv("SANDBOX_AUTO_MAP", "1")
    monkeypatch.setenv("SANDBOX_MODULE_ALGO", "label")
    monkeypatch.setenv("SANDBOX_MODULE_THRESHOLD", "0.3")
    monkeypatch.setenv("SANDBOX_SEMANTIC_MODULES", "1")
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))

    path = tmp_path / "map.json"
    db = ModuleIndexDB(path, auto_map=None)
    db.refresh(force=True)
    assert generated["output"] == path
    assert generated["root"] == tmp_path
    assert generated["algorithm"] == "label"
    assert generated["threshold"] == 0.3
    assert generated["semantic"]
    assert db.get("mod") == 3


def test_autodiscover_deprecated(monkeypatch, tmp_path):
    generated = {}

    def fake_generate(output, *, root, algorithm, threshold, semantic, exclude=None):
        generated.update(dict(output=output))
        output.write_text(json.dumps({"m": 1}))
        return {"m": 1}

    monkeypatch.setattr("module_index_db.generate_module_map", fake_generate)
    monkeypatch.setenv("SANDBOX_AUTODISCOVER_MODULES", "1")
    path = tmp_path / "map.json"
    with pytest.warns(UserWarning):
        db = ModuleIndexDB(path, auto_map=None)
    db.refresh(force=True)
    assert generated["output"] == path
    assert db.get("m") == 1


def test_refresh_generates_when_unknown(monkeypatch, tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"modules": {"old.py": 0}, "groups": {"0": 0}}))  # path-ignore
    db = ModuleIndexDB(path)
    called = {}

    def fake_generate(output, *, root, algorithm, threshold, semantic, exclude=None):
        called["called"] = True
        output.write_text(json.dumps({"old.py": 0, "new.py": 1}))  # path-ignore
        return {"old.py": 0, "new.py": 1}  # path-ignore

    monkeypatch.setattr("module_index_db.generate_module_map", fake_generate)

    db.refresh(["new.py"])  # path-ignore
    assert called.get("called")
    assert db.get("new.py") == 1  # path-ignore

    called.clear()
    db.refresh(["old.py"])  # path-ignore
    assert not called

