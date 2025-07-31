import json
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
    assert db.get("foo.py") == 7



def test_auto_map(monkeypatch, tmp_path):
    generated = {}
    def fake_generate(output, *, root, algorithm, threshold, semantic):
        generated.update(dict(output=output, root=root, algorithm=algorithm, threshold=threshold, semantic=semantic))
        output.write_text(json.dumps({"mod": 3}))
        return {"mod": 3}
    monkeypatch.setattr("module_index_db.generate_module_map", fake_generate)
    monkeypatch.setenv("SANDBOX_AUTODISCOVER_MODULES", "1")
    monkeypatch.setenv("SANDBOX_MODULE_ALGO", "label")
    monkeypatch.setenv("SANDBOX_MODULE_THRESHOLD", "0.3")
    monkeypatch.setenv("SANDBOX_MODULE_SEMANTIC", "1")
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    path = tmp_path / "map.json"
    db = ModuleIndexDB(path, auto_map=None)
    assert generated["output"] == path
    assert generated["root"] == tmp_path
    assert generated["algorithm"] == "label"
    assert generated["threshold"] == 0.3
    assert generated["semantic"]
    assert db.get("mod") == 3
