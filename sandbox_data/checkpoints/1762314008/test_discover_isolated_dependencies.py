import json
from pathlib import Path

import orphan_analyzer
from scripts.discover_isolated_modules import discover_isolated_modules


def test_dependency_of_isolated_module_included(tmp_path):
    """ensure dependencies of isolated modules are listed once"""

    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("x = 1\n")  # path-ignore

    res = discover_isolated_modules(tmp_path, recursive=True)
    assert res == ["a.py", "b.py"]  # path-ignore

    data = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert isinstance(data, dict)


def test_known_modules_skipped(tmp_path):
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (tmp_path / "a.py").write_text("pass\n")  # path-ignore
    (tmp_path / "b.py").write_text("pass\n")  # path-ignore
    (data_dir / "module_map.json").write_text(json.dumps({"modules": {"a.py": 1}}))  # path-ignore

    res = discover_isolated_modules(tmp_path, recursive=True)
    assert res == ["b.py"]  # path-ignore


def test_redundant_modules_omitted(tmp_path, monkeypatch):
    (tmp_path / "c.py").write_text("pass\n")  # path-ignore
    (tmp_path / "d.py").write_text("pass\n")  # path-ignore

    def fake_classify(path):
        return "redundant" if path.name == "d.py" else "candidate"  # path-ignore

    monkeypatch.setattr(
        orphan_analyzer, "classify_module", fake_classify
    )

    res = discover_isolated_modules(tmp_path, recursive=True)
    assert res == ["c.py"]  # path-ignore

