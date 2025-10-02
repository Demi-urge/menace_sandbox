from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest


bootstrap_env = importlib.import_module("scripts.bootstrap_env")


def _reset_windows_normalizer() -> None:
    bootstrap_env._windows_path_normalizer.cache_clear()  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MENACE_ENV_FILE", raising=False)
    monkeypatch.delenv("MENACE_SKIP_STRIPE_ROUTER", raising=False)
    monkeypatch.delenv("PATHEXT", raising=False)
    monkeypatch.delenv("Path", raising=False)
    monkeypatch.delenv("PATH", raising=False)
    _reset_windows_normalizer()


def test_expand_environment_path_supports_windows_style(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    expanded = bootstrap_env._expand_environment_path(r"%USERPROFILE%\\config.env")
    normalized = Path(expanded.replace("\\", os.sep))
    assert normalized == tmp_path / "config.env"


def test_expand_environment_path_rejects_unresolved_tokens() -> None:
    with pytest.raises(bootstrap_env.BootstrapError) as excinfo:
        bootstrap_env._expand_environment_path(r"%UNDEFINED_TOKEN%\\settings.env")
    assert "%UNDEFINED_TOKEN%" in str(excinfo.value)


def test_resolved_env_file_rejects_directory(tmp_path: Path) -> None:
    config = bootstrap_env.BootstrapConfig(env_file=tmp_path)
    with pytest.raises(bootstrap_env.BootstrapError):
        config.resolved_env_file()


def test_ensure_windows_compatibility_injects_scripts_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir()
    python_executable = tmp_path / "python.exe"
    python_executable.write_text("", encoding="utf-8")

    monkeypatch.setattr(bootstrap_env, "_is_windows", lambda: True)
    monkeypatch.setattr(bootstrap_env.sys, "executable", str(python_executable), raising=False)
    monkeypatch.setattr(bootstrap_env.sys, "prefix", str(tmp_path), raising=False)
    monkeypatch.setattr(bootstrap_env.sys, "base_prefix", str(tmp_path), raising=False)
    monkeypatch.setattr(bootstrap_env.sys, "exec_prefix", str(tmp_path), raising=False)
    monkeypatch.setattr(
        bootstrap_env.sysconfig,
        "get_path",
        lambda key: str(scripts_dir) if key == "scripts" else None,
        raising=False,
    )
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path))
    monkeypatch.setenv("PATH", os.pathsep.join(["C:/existing/bin"]))

    bootstrap_env._ensure_windows_compatibility()

    expected = bootstrap_env._format_windows_path_entry(str(scripts_dir.resolve()))
    path_entries = os.environ["PATH"].split(os.pathsep)
    assert expected in path_entries
    assert os.environ["PATHEXT"]
