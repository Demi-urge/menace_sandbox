import os
from pathlib import Path
from sandbox_runner.dependency_utils import collect_local_dependencies


def test_collect_local_dependencies_respects_max_depth(tmp_path, monkeypatch):
    a = tmp_path / "a.py"  # path-ignore
    b = tmp_path / "b.py"  # path-ignore
    c = tmp_path / "c.py"  # path-ignore
    a.write_text("import b\n")
    b.write_text("import c\n")
    c.write_text("x = 1\n")
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    deps = collect_local_dependencies([str(a)], max_depth=1)
    assert "b.py" in deps  # path-ignore
    assert "c.py" not in deps  # path-ignore
