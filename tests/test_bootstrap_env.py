from __future__ import annotations

import importlib
import json
import os
import subprocess
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


def test_collect_docker_diagnostics_missing_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap_env, "_is_windows", lambda: False)
    monkeypatch.setattr(bootstrap_env, "_is_wsl", lambda: False)
    monkeypatch.setattr(bootstrap_env.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        bootstrap_env,
        "_detect_runtime_context",
        lambda: bootstrap_env.RuntimeContext(
            platform="linux",
            is_windows=False,
            is_wsl=False,
            inside_container=False,
            container_runtime=None,
            container_indicators=(),
            is_ci=False,
            ci_indicators=(),
        ),
    )

    result = bootstrap_env._collect_docker_diagnostics(timeout=0.1)

    assert result.cli_path is None
    assert result.available is False
    assert any("Docker CLI" in error for error in result.errors)


def test_collect_docker_diagnostics_reports_server_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_cli = tmp_path / "docker"
    fake_cli.write_text("", encoding="utf-8")

    monkeypatch.setattr(bootstrap_env, "_is_windows", lambda: False)
    monkeypatch.setattr(bootstrap_env, "_is_wsl", lambda: False)
    monkeypatch.setattr(bootstrap_env.shutil, "which", lambda _: str(fake_cli))
    monkeypatch.setattr(
        bootstrap_env,
        "_detect_runtime_context",
        lambda: bootstrap_env.RuntimeContext(
            platform="linux",
            is_windows=False,
            is_wsl=False,
            inside_container=False,
            container_runtime=None,
            container_indicators=(),
            is_ci=False,
            ci_indicators=(),
        ),
    )

    def _fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if "version" in cmd:
            payload = json.dumps({"Client": {"Version": "26.0", "ApiVersion": "1.45"}, "Server": {}})
            return subprocess.CompletedProcess(cmd, 0, payload, "")
        return subprocess.CompletedProcess(cmd, 1, "", "error during connect")

    monkeypatch.setattr(bootstrap_env.subprocess, "run", _fake_run)

    result = bootstrap_env._collect_docker_diagnostics(timeout=0.1)

    assert result.cli_path == fake_cli
    assert result.available is False
    assert any("daemon" in error.lower() or "server" in error.lower() for error in result.errors)
    assert result.metadata["cli_path"] == str(fake_cli)


def test_extract_json_document_tolerates_prefixed_warnings() -> None:
    payload = json.dumps({"Client": {"Version": "27.0"}})
    stdout = "WARNING: worker stalled; restarting\n\n" + payload
    stderr = "warning: docker desktop resources low"

    extracted, warnings = bootstrap_env._extract_json_document(stdout, stderr)

    assert extracted is not None
    assert json.loads(extracted)["Client"]["Version"] == "27.0"
    assert any("worker stalled" in warning.lower() for warning in warnings)
    assert any("resources" in warning.lower() for warning in warnings)


def test_parse_docker_json_surfaces_non_json_output() -> None:
    completed = subprocess.CompletedProcess(
        ["docker", "info"],
        0,
        "WARNING: diagnostics disabled",
        "",
    )

    data, warnings = bootstrap_env._parse_docker_json(completed, "info")

    assert data is None
    assert any("no json" in warning.lower() for warning in warnings)
