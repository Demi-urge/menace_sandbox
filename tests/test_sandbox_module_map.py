import json
from pathlib import Path
from scripts.generate_module_map import generate_module_map


def test_generate_map_call_edges(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "__init__.py").write_text("")  # path-ignore
    (tmp_path / "b" / "c.py").write_text("def foo():\n    pass\n")  # path-ignore
    (tmp_path / "a.py").write_text("from b import c\n\ndef run():\n    c.foo()\n")  # path-ignore
    out = tmp_path / "module_map.json"
    mapping = generate_module_map(out, root=tmp_path)
    data = json.loads(out.read_text())
    assert mapping == data
    assert mapping["a"] == mapping["b/c"]


def test_generate_map_semantic(tmp_path):
    (tmp_path / "a.py").write_text('"""DB utils"""\n\ndef a():\n    pass\n')  # path-ignore
    (tmp_path / "b.py").write_text('"""DB utils"""\n\ndef b():\n    pass\n')  # path-ignore
    out = tmp_path / "map.json"
    plain = generate_module_map(out, root=tmp_path, algorithm="label")
    assert plain["a"] != plain["b"]
    mapping = generate_module_map(out, root=tmp_path, algorithm="label", semantic=True)
    assert mapping["a"] == mapping["b"]


def test_generate_map_exclude(tmp_path):
    skip = tmp_path / "skip"
    skip.mkdir()
    (skip / "x.py").write_text("pass\n")  # path-ignore
    (tmp_path / "a.py").write_text("pass\n")  # path-ignore

    out = tmp_path / "map.json"
    mapping = generate_module_map(out, root=tmp_path, exclude=["skip"])
    assert "a" in mapping
    assert "skip/x" not in mapping
