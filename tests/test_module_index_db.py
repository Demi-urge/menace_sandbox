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
