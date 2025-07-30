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
