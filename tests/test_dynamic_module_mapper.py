import json
from pathlib import Path

import pytest

pytest.importorskip("networkx")

import dynamic_module_mapper as dmm


def _read_map(path: Path) -> dict:
    return json.loads(path.read_text())


def test_mutual_imports_grouped(tmp_path):
    (tmp_path / "a.py").write_text("import b\n")
    (tmp_path / "b.py").write_text("import a\n")
    mapping = dmm.build_module_map(tmp_path)
    assert mapping["a"] == mapping["b"]


def test_cli_writes_map(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("import b\n")
    (tmp_path / "b.py").write_text("import a\n")
    out = tmp_path / "sandbox_data" / "module_map.json"
    dmm.main([str(tmp_path)])
    data = _read_map(out)
    assert data["a"] == data["b"]


def test_semantic_group_without_static_import(tmp_path):
    doc = "\"\"\"Perform operations with the same semantic meaning\"\"\""
    (tmp_path / "a.py").write_text(
        doc + "\n\n" "def a_func():\n    __import__('b').b_func()\n"
    )
    (tmp_path / "b.py").write_text(
        doc + "\n\n" "def b_func():\n    __import__('a').a_func()\n"
    )

    plain = dmm.build_module_map(tmp_path, algorithm="label")
    assert plain["a"] != plain["b"]

    mapping = dmm.build_module_map(tmp_path, algorithm="label", use_semantic=True)
    assert mapping["a"] == mapping["b"]


def test_call_links_modules(tmp_path):
    pkg = tmp_path / "b"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "c.py").write_text("def foo():\n    pass\n")

    (tmp_path / "a.py").write_text(
        "from b import c\n\n" "def run():\n    c.foo()\n"
    )

    mapping = dmm.build_module_map(tmp_path)
    assert mapping["a"] == mapping["b/c"]


def test_semantic_group_no_imports(tmp_path):
    (tmp_path / "a.py").write_text(
        '"""Database utilities."""\n\ndef a():\n    pass\n'
    )
    (tmp_path / "b.py").write_text(
        '"""Database utilities."""\n\ndef b():\n    pass\n'
    )

    plain = dmm.build_module_map(tmp_path, algorithm="label")
    assert plain["a"] != plain["b"]

    mapping = dmm.build_module_map(tmp_path, algorithm="label", use_semantic=True)
    assert mapping["a"] == mapping["b"]




def test_semantic_fixture_grouping(tmp_path):
    src = Path(__file__).parent / "fixtures" / "semantic"
    import shutil
    shutil.copytree(src, tmp_path / "mods")
    plain = dmm.build_module_map(tmp_path / "mods", algorithm="label")
    idxs = {plain["a"], plain["b"], plain["c"]}
    assert len(idxs) > 1

    mapping = dmm.build_module_map(tmp_path / "mods", algorithm="label", use_semantic=True)
    assert mapping["a"] == mapping["b"] == mapping["c"]


def test_cli_option_parsing(monkeypatch):
    calls = {}

    def fake_build(repo, *, algorithm, threshold, use_semantic):
        calls.update(repo=repo, algorithm=algorithm, threshold=threshold, semantic=use_semantic)
        return {}

    monkeypatch.setattr(dmm, "build_module_map", fake_build)
    dmm.main(["src", "--algorithm", "label", "--threshold", "0.5", "--semantic"])
    assert calls == {
        "repo": "src",
        "algorithm": "label",
        "threshold": 0.5,
        "semantic": True,
    }
