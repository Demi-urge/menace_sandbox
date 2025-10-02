"""Unit tests for Windows specific helpers in ``scripts.bootstrap_env``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts import bootstrap_env


@pytest.fixture(autouse=True)
def _restore_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure PATH/PATHEXT manipulations are isolated per test."""

    original_environ = os.environ.copy()
    yield
    for key in set(os.environ) - set(original_environ):
        monkeypatch.delenv(key, raising=False)
    for key, value in original_environ.items():
        os.environ[key] = value


def test_gather_existing_path_entries_normalizes_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "pathsep", ";")
    monkeypatch.setattr(bootstrap_env.os, "pathsep", ";")
    path_value = ";".join(
        [
            r"C:\Tools",
            r' "C:\Tools" ',
            r'"C:\Program Files\Python\Scripts"',
            r"C:\Program Files\Python\Scripts",
        ]
    )
    monkeypatch.setenv("PATH", path_value)
    ordered, seen, dedup = bootstrap_env._gather_existing_path_entries()

    assert dedup is True
    normalizer = bootstrap_env._windows_path_normalizer()
    normalized = normalizer(bootstrap_env._strip_windows_quotes(ordered[0]))
    assert normalized in seen
    # The script should retain a single canonical entry for the Scripts directory
    scripts_entries = [entry for entry in ordered if "Program Files" in entry]
    assert len(scripts_entries) == 1
    assert scripts_entries[0].startswith('"') and scripts_entries[0].endswith('"')


def test_ensure_windows_compatibility_injects_scripts_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(os, "pathsep", ";")
    monkeypatch.setattr(bootstrap_env.os, "pathsep", ";")
    scripts_dir = tmp_path / "Python Tools" / "Scripts"
    scripts_dir.mkdir(parents=True)

    monkeypatch.setenv("PATH", r"C:\Existing")
    monkeypatch.setenv("PATHEXT", ".EXE")
    monkeypatch.setattr(bootstrap_env, "_is_windows", lambda: True)
    monkeypatch.setattr(
        bootstrap_env,
        "_iter_windows_script_candidates",
        lambda executable: [scripts_dir],
    )
    monkeypatch.setattr(bootstrap_env.sys, "executable", str(tmp_path / "python.exe"))

    bootstrap_env._ensure_windows_compatibility()

    updated_path = os.environ["PATH"].split(";")
    # The new Scripts directory should be prepended and quoted because of the space.
    assert updated_path[0] == f'"{scripts_dir}"'
    # PATHEXT should be augmented with common Python extensions.
    pathext_entries = {ext.upper() for ext in os.environ["PATHEXT"].split(";")}
    assert {".PY", ".PYW"}.issubset(pathext_entries)
