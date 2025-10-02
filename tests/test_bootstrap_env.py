from __future__ import annotations

import os
from pathlib import Path

import pytest

import scripts.bootstrap_env as bootstrap_env


@pytest.mark.parametrize("existing_entries", ["C\\\\Windows\\System32", ""])
def test_windows_compatibility_injects_scripts(monkeypatch, tmp_path, existing_entries):
    """Ensure the Windows bootstrap helper injects the Scripts directories."""

    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir()

    monkeypatch.setattr(bootstrap_env, "_is_windows", lambda: True)
    monkeypatch.setattr(bootstrap_env.os, "pathsep", ";", raising=False)
    monkeypatch.setattr(bootstrap_env.sys, "executable", str(tmp_path / "python.exe"))
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path))
    monkeypatch.setenv("PATH", existing_entries)
    monkeypatch.delenv("PATHEXT", raising=False)
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)

    bootstrap_env._ensure_windows_compatibility()

    resulting_path = os.environ["PATH"].split(os.pathsep)
    normalized = [entry.lower() for entry in resulting_path]
    assert str(scripts_dir).lower() in normalized
    assert os.environ["PYTHONUTF8"] == "1"
    assert os.environ["PYTHONIOENCODING"].lower() == "utf-8"

    pathext = {ext.upper() for ext in os.environ["PATHEXT"].split(os.pathsep)}
    assert {".COM", ".EXE", ".BAT", ".CMD", ".PY", ".PYW"}.issubset(pathext)


def test_windows_scripts_candidates_include_sysconfig(monkeypatch, tmp_path):
    """The candidate discovery should include the interpreter scripts path."""

    scripts_path = tmp_path / "alt_scripts"
    scripts_path.mkdir()

    monkeypatch.setattr(bootstrap_env, "_is_windows", lambda: True)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)

    def fake_get_path(name: str) -> str:
        if name == "scripts":
            return str(scripts_path)
        raise KeyError(name)

    monkeypatch.setattr(bootstrap_env.sysconfig, "get_path", fake_get_path)

    candidates = list(
        bootstrap_env._iter_windows_script_candidates(Path(tmp_path / "python.exe"))
    )

    assert scripts_path in candidates


def test_repo_root_injected_without_duplicates(monkeypatch, tmp_path):
    """The bootstrap script should normalise sys.path entries on Windows."""

    repo_root = tmp_path / "RepoRoot"
    repo_root.mkdir()

    original_path = [
        "C:/Temp/Libs",
        str(repo_root).upper(),
        str(tmp_path / "other"),
    ]

    monkeypatch.setattr(
        bootstrap_env.os.path,
        "normcase",
        lambda value: str(value).lower(),
        raising=False,
    )
    monkeypatch.setattr(bootstrap_env.sys, "path", list(original_path))

    bootstrap_env._ensure_repo_root_on_path(repo_root)

    assert bootstrap_env.sys.path[0] == str(repo_root)
    assert bootstrap_env.sys.path.count(str(repo_root)) == 1
    assert len(bootstrap_env.sys.path) == len(original_path)
