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

    plain = dmm.build_module_map(tmp_path)
    assert plain["a"] != plain["b"]

    mapping = dmm.build_module_map(tmp_path, use_semantic=True)
    assert mapping["a"] == mapping["b"]
