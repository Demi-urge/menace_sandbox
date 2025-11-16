from __future__ import annotations

from pathlib import Path

from sandbox_runner import orphan_discovery


def _setup_repo(tmp_path: Path) -> None:
    (tmp_path / "mod1.py").write_text("import mod2\n")  # path-ignore
    (tmp_path / "mod2.py").write_text("# empty\n")  # path-ignore
    (tmp_path / "mod3.py").write_text("# orphan\n")  # path-ignore


def test_parallel_vs_sequential(tmp_path, monkeypatch):
    _setup_repo(tmp_path)

    monkeypatch.setenv("SANDBOX_DISCOVERY_WORKERS", "1")
    seq = orphan_discovery.discover_recursive_orphans(str(tmp_path))

    monkeypatch.setenv("SANDBOX_DISCOVERY_WORKERS", "2")
    par = orphan_discovery.discover_recursive_orphans(str(tmp_path))

    assert seq == par
    assert {"mod1", "mod2", "mod3"} == set(seq)
