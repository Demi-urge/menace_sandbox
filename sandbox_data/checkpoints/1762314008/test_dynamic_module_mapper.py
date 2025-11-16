import json
from pathlib import Path

import json
from pathlib import Path

import pytest

pytest.importorskip("networkx")

import dynamic_module_mapper as dmm
from dynamic_path_router import resolve_path


def _read_map(path: Path) -> dict:
    return json.loads(path.read_text())


def test_mutual_imports_grouped(tmp_path):
    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("import a\n")  # path-ignore
    mapping = dmm.build_module_map(tmp_path)
    assert mapping["a"] == mapping["b"]


def test_cli_writes_map(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("import a\n")  # path-ignore
    out = tmp_path / "sandbox_data" / "module_map.json"
    dmm.main([str(tmp_path)])
    data = _read_map(out)
    assert data["a"] == data["b"]


def test_semantic_group_without_static_import(tmp_path):
    doc = "\"\"\"Perform operations with the same semantic meaning\"\"\""
    (tmp_path / "a.py").write_text(  # path-ignore
        doc + "\n\n" "def a_func():\n    __import__('b').b_func()\n"
    )
    (tmp_path / "b.py").write_text(  # path-ignore
        doc + "\n\n" "def b_func():\n    __import__('a').a_func()\n"
    )

    plain = dmm.build_module_map(tmp_path, algorithm="label")
    assert plain["a"] != plain["b"]

    mapping = dmm.build_module_map(tmp_path, algorithm="label", use_semantic=True)
    assert mapping["a"] == mapping["b"]


def test_call_links_modules(tmp_path):
    pkg = tmp_path / "b"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")  # path-ignore
    (pkg / "c.py").write_text("def foo():\n    pass\n")  # path-ignore

    (tmp_path / "a.py").write_text(  # path-ignore
        "from b import c\n\n" "def run():\n    c.foo()\n"
    )

    mapping = dmm.build_module_map(tmp_path)
    assert mapping["a"] == mapping["b/c"]


def test_exclude_patterns(tmp_path):
    skip = tmp_path / "skip"
    skip.mkdir()
    (skip / "x.py").write_text("pass\n")  # path-ignore
    (tmp_path / "a.py").write_text("pass\n")  # path-ignore

    mapping = dmm.build_module_map(tmp_path, ignore=["skip"])
    assert "a" in mapping
    assert "skip/x" not in mapping


def test_semantic_group_no_imports(tmp_path):
    (tmp_path / "a.py").write_text(  # path-ignore
        '"""Database utilities."""\n\ndef a():\n    pass\n'
    )
    (tmp_path / "b.py").write_text(  # path-ignore
        '"""Database utilities."""\n\ndef b():\n    pass\n'
    )

    plain = dmm.build_module_map(tmp_path, algorithm="label")
    assert plain["a"] != plain["b"]

    mapping = dmm.build_module_map(tmp_path, algorithm="label", use_semantic=True)
    assert mapping["a"] == mapping["b"]




def test_semantic_fixture_grouping(tmp_path):
    import shutil

    src = resolve_path("tests/fixtures/semantic")
    shutil.copytree(src, tmp_path / "mods")
    plain = dmm.build_module_map(tmp_path / "mods", algorithm="label")
    idxs = {plain["a"], plain["b"], plain["c"]}
    assert len(idxs) > 1

    mapping = dmm.build_module_map(tmp_path / "mods", algorithm="label", use_semantic=True)
    assert mapping["a"] == mapping["b"] == mapping["c"]


def test_semantic_tfidf_similarity(tmp_path):
    (tmp_path / "a.py").write_text('"""Database helper utilities."""\n')  # path-ignore
    (tmp_path / "b.py").write_text('"""Utilities for working with database"""\n')  # path-ignore

    plain = dmm.build_module_map(tmp_path, algorithm="label", threshold=0.2)
    assert plain["a"] != plain["b"]

    mapping = dmm.build_module_map(
        tmp_path, algorithm="label", threshold=0.2, use_semantic=True
    )
    assert mapping["a"] == mapping["b"]


def test_redundant_module_excluded(tmp_path):
    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("import a\n")  # path-ignore
    (tmp_path / "c.py").write_text("pass\n")  # path-ignore

    mapping = dmm.build_module_map(tmp_path)
    assert "c" not in mapping
    assert "a" in mapping and "b" in mapping


def test_cli_option_parsing(monkeypatch):
    calls = {}

    def fake_build(repo, *, algorithm, threshold, use_semantic, ignore):
        calls.update(
            repo=repo,
            algorithm=algorithm,
            threshold=threshold,
            semantic=use_semantic,
            ignore=ignore,
        )
        return {}

    monkeypatch.setattr(dmm, "build_module_map", fake_build)
    dmm.main([
        "src",
        "--algorithm",
        "label",
        "--threshold",
        "0.5",
        "--semantic",
        "--exclude",
        "tests",
    ])
    assert calls == {
        "repo": "src",
        "algorithm": "label",
        "threshold": 0.5,
        "semantic": True,
        "ignore": ["tests"],
    }


def _write_cluster_project(path: Path, heavy: bool) -> None:
    """Create four modules with optional repeated calls."""
    path.mkdir(parents=True, exist_ok=True)
    calls = "".join("    b.x()\n" for _ in range(9)) if heavy else ""
    (path / "a.py").write_text(  # path-ignore
        "import b\nimport c\nimport d\n\n" f"def go():\n{calls}"
    )
    (path / "b.py").write_text(  # path-ignore
        "import a\nimport c\nimport d\n\n" "def x():\n    pass\n"
    )
    calls_cd = "".join("    d.y()\n" for _ in range(9)) if heavy else ""
    (path / "c.py").write_text(  # path-ignore
        "import a\nimport b\nimport d\n\n" f"def go():\n{calls_cd}"
    )
    (path / "d.py").write_text(  # path-ignore
        "import a\nimport b\nimport c\n\n" "def y():\n    pass\n"
    )


def test_repeated_calls_strengthen_clustering(tmp_path):
    base = tmp_path / "base"
    weighted = tmp_path / "weighted"
    _write_cluster_project(base, heavy=False)
    _write_cluster_project(weighted, heavy=True)

    plain = dmm.build_module_map(base)
    assert len({plain["a"], plain["b"], plain["c"], plain["d"]}) == 1

    mapping = dmm.build_module_map(weighted)
    groups = {mapping["a"], mapping["b"], mapping["c"], mapping["d"]}
    assert len(groups) == 2
    assert mapping["a"] == mapping["b"]
    assert mapping["c"] == mapping["d"]
    assert mapping["a"] != mapping["c"]


def test_hdbscan_algorithm(tmp_path):
    pytest.importorskip("hdbscan")
    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("import a\n")  # path-ignore
    (tmp_path / "c.py").write_text("pass\n")  # path-ignore

    mapping = dmm.build_module_map(tmp_path, algorithm="hdbscan")
    assert mapping["a"] == mapping["b"]
    assert mapping["c"] != mapping["a"]

