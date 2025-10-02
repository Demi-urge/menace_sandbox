from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

import scripts.bootstrap_env as bootstrap_env


@pytest.fixture(autouse=True)
def restore_environment(monkeypatch):
    original_path = os.environ.get("PATH")
    original_Path = os.environ.get("Path")
    original_pathext = os.environ.get("PATHEXT")
    try:
        yield
    finally:
        if original_path is None:
            monkeypatch.delenv("PATH", raising=False)
        else:
            monkeypatch.setenv("PATH", original_path)
        if original_Path is None:
            monkeypatch.delenv("Path", raising=False)
        else:
            monkeypatch.setenv("Path", original_Path)
        if original_pathext is None:
            monkeypatch.delenv("PATHEXT", raising=False)
        else:
            monkeypatch.setenv("PATHEXT", original_pathext)


def _prime_windows_environment(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(bootstrap_env, "_is_windows", lambda: True)
    executable = tmp_path / "python.exe"
    executable.write_text("", encoding="utf-8")
    monkeypatch.setattr(bootstrap_env.sys, "executable", str(executable))
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir()
    bootstrap_env._windows_path_normalizer.cache_clear()

    def _candidates(_executable: Path):
        return [scripts_dir]

    monkeypatch.setattr(bootstrap_env, "_iter_windows_script_candidates", _candidates)
    return scripts_dir


def test_windows_path_is_augmented_with_scripts_directory(tmp_path, monkeypatch, caplog):
    scripts_dir = _prime_windows_environment(monkeypatch, tmp_path)

    monkeypatch.delenv("Path", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    monkeypatch.setenv("PATHEXT", ".COM;.EXE")

    caplog.set_level(logging.INFO)
    bootstrap_env._ensure_windows_compatibility()

    path_value = os.environ["PATH"]
    assert os.environ["Path"] == path_value
    segments = path_value.split(os.pathsep)
    assert segments[0] == str(scripts_dir)
    assert segments.count(str(scripts_dir)) == 1

    pathext = os.environ["PATHEXT"].split(os.pathsep)
    assert {ext.upper() for ext in pathext} >= {".COM", ".EXE", ".BAT", ".CMD", ".PY", ".PYW"}

    assert any("Scripts directories" in record.getMessage() for record in caplog.records)


def test_windows_path_deduplicates_entries(tmp_path, monkeypatch, caplog):
    scripts_dir = _prime_windows_environment(monkeypatch, tmp_path)

    duplicate_entries = os.pathsep.join([str(scripts_dir), str(scripts_dir).upper()])
    monkeypatch.setenv("PATH", duplicate_entries)

    caplog.set_level(logging.INFO)
    bootstrap_env._ensure_windows_compatibility()

    path_segments = os.environ["PATH"].split(os.pathsep)
    assert path_segments == [str(scripts_dir)]
    assert any("duplicate entries" in record.getMessage() for record in caplog.records)
