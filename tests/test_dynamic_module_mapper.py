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
    groups = dmm.discover_module_groups(tmp_path)
    group_a = next((g for g, mods in groups.items() if "a" in mods), None)
    group_b = next((g for g, mods in groups.items() if "b" in mods), None)
    assert group_a is not None
    assert group_a == group_b


def test_cli_writes_map(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("import b\n")
    (tmp_path / "b.py").write_text("import a\n")
    out = tmp_path / "sandbox_data" / "module_map.json"
    dmm.main([str(tmp_path)])
    data = _read_map(out)
    group = next((g for g, mods in data.items() if "a" in mods), None)
    assert group is not None
    assert "b" in data[group]


def test_semantic_group_without_static_import(tmp_path):
    doc = "\"\"\"Perform operations with the same semantic meaning\"\"\""
    (tmp_path / "a.py").write_text(
        doc + "\n\n" "def a_func():\n    __import__('b').b_func()\n"
    )
    (tmp_path / "b.py").write_text(
        doc + "\n\n" "def b_func():\n    __import__('a').a_func()\n"
    )

    plain = dmm.discover_module_groups(tmp_path)
    group_a_plain = next((g for g, mods in plain.items() if "a" in mods), None)
    group_b_plain = next((g for g, mods in plain.items() if "b" in mods), None)
    assert group_a_plain != group_b_plain

    groups = dmm.discover_module_groups(tmp_path, use_semantic=True)
    group_a = next((g for g, mods in groups.items() if "a" in mods), None)
    group_b = next((g for g, mods in groups.items() if "b" in mods), None)
    assert group_a is not None
    assert group_a == group_b
