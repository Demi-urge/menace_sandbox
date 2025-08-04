import json
from pathlib import Path

from scripts.discover_isolated_modules import discover_isolated_modules


def test_dependency_of_isolated_module_included(tmp_path, monkeypatch):
    """ensure dependencies of isolated modules are listed once"""

    # create isolated module ``a`` which depends on another isolated module ``b``
    (tmp_path / "a.py").write_text("import b\n")
    (tmp_path / "b.py").write_text("x = 1\n")

    # make ``find_orphan_modules`` return only ``a.py`` so the dependency is
    # discovered via ``discover_recursive_orphans``
    from scripts import discover_isolated_modules as mod

    monkeypatch.setattr(
        mod,
        "find_orphan_modules",
        lambda root, recursive=False: [Path("a.py")],
    )

    res = discover_isolated_modules(tmp_path, recursive=True)
    assert sorted(res) == ["a.py", "b.py"]

    data = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert data == sorted(res)


def test_known_and_duplicate_modules_skipped(tmp_path, monkeypatch):
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (tmp_path / "a.py").write_text("pass\n")
    (tmp_path / "b.py").write_text("pass\n")
    (tmp_path / "c.py").write_text("pass\n")
    (data_dir / "module_map.json").write_text(json.dumps({"a": 1, "b": 2}))

    from scripts import discover_isolated_modules as mod

    monkeypatch.setattr(
        mod,
        "find_orphan_modules",
        lambda root, recursive=False: [Path("a.py"), Path("c.py")],
    )
    monkeypatch.setattr(mod, "analyze_redundancy", lambda p: False)
    mod._discover_import_orphans = lambda root, recursive=False: ["b", "c"]

    res = discover_isolated_modules(tmp_path, recursive=False)
    assert res == ["c.py"]


def test_redundant_modules_omitted(tmp_path, monkeypatch):
    (tmp_path / "c.py").write_text("pass\n")
    (tmp_path / "d.py").write_text("pass\n")

    from scripts import discover_isolated_modules as mod

    monkeypatch.setattr(
        mod,
        "find_orphan_modules",
        lambda root, recursive=False: [Path("c.py"), Path("d.py")],
    )
    mod._discover_import_orphans = None
    monkeypatch.setattr(mod, "analyze_redundancy", lambda p: p.name == "d.py")

    res = discover_isolated_modules(tmp_path, recursive=False)
    assert res == ["c.py"]

