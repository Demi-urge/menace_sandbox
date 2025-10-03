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

    extracted, warnings, metadata = bootstrap_env._extract_json_document(stdout, stderr)

    assert extracted is not None
    assert json.loads(extracted)["Client"]["Version"] == "27.0"
    assert any("docker desktop reported" in warning.lower() for warning in warnings)
    assert any("resources" in warning.lower() for warning in warnings)
    assert metadata.get("docker_worker_health") == "flapping"


def test_normalise_docker_warning_handles_worker_stall_variants() -> None:
    message = "WARNING[0012]: worker stalled; restarting (background-sync)"
    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "docker desktop reported" in cleaned.lower()
    assert cleaned.lower().endswith("background-sync.")
    assert metadata["docker_worker_health"] == "flapping"
    assert metadata["docker_worker_context"] == "background-sync"


def test_normalise_docker_warning_extracts_key_value_context() -> None:
    message = (
        'time="2024-05-03T08:13:37-07:00" level=warning msg="worker stalled; restarting" '
        'context="buildkitd" id="buildkitd"'
    )
    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "docker desktop reported" in cleaned.lower()
    assert "worker stalled" not in cleaned.lower()
    assert metadata["docker_worker_health"] == "flapping"
    assert metadata["docker_worker_context"] == "buildkitd"


def test_normalise_docker_warning_extracts_context_after_restarting_clause() -> None:
    message = "WARNING: worker stalled; restarting worker moby-buildkit (pid=1234)"
    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "docker desktop reported" in cleaned.lower()
    assert metadata["docker_worker_health"] == "flapping"
    assert metadata["docker_worker_context"] == "moby-buildkit"


def test_normalise_docker_warning_enriches_restart_metadata() -> None:
    message = (
        "time=\"2024-05-05T00:01:02Z\" level=warning msg=\"worker stalled; restarting\" "
        "module=buildkit worker=buildkitd restarts=7 backoff=20s err=\"context canceled\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert metadata["docker_worker_health"] == "flapping"
    assert metadata["docker_worker_context"] == "buildkitd"
    assert metadata["docker_worker_restart_count"] == "7"
    assert metadata["docker_worker_backoff"] == "20s"
    assert metadata["docker_worker_last_error"] == "context canceled"
    assert "additional context" in cleaned.lower()
    assert "7 restart" in cleaned
    assert "context canceled" in cleaned


def test_parse_wsl_distribution_table_handles_complex_rows() -> None:
    payload = (
        "  NAME                   STATE           VERSION\n"
        "* Ubuntu-22.04           Running         2\n"
        "  docker-desktop         Stopped         2\n"
        "  docker-desktop-data    Running         2\n"
        "  Custom Distro Name     Installing      1\n"
    )

    rows = bootstrap_env._parse_wsl_distribution_table(payload)

    assert any(row["is_default"] for row in rows)
    desktop_entry = next(row for row in rows if row["name"] == "docker-desktop")
    assert desktop_entry["state"] == "Stopped"
    assert desktop_entry["version"] == "2"
    custom_entry = next(row for row in rows if row["name"] == "Custom Distro Name")
    assert custom_entry["state"] == "Installing"
    assert custom_entry["version"] == "1"


def test_parse_docker_json_surfaces_non_json_output() -> None:
    completed = subprocess.CompletedProcess(
        ["docker", "info"],
        0,
        "WARNING: diagnostics disabled",
        "",
    )

    data, warnings, metadata = bootstrap_env._parse_docker_json(completed, "info")

    assert data is None
    assert any("no json" in warning.lower() for warning in warnings)
    assert metadata == {}


def test_collect_windows_virtualization_insights_reports_inactive_components(monkeypatch: pytest.MonkeyPatch) -> None:
    payload_status = "Default Version: 2\nWSL version: 1.2.5.0"
    payload_list = (
        "  NAME                   STATE           VERSION\n"
        "* Ubuntu-22.04           Stopped         2\n"
        "  docker-desktop         Stopped         2\n"
        "  docker-desktop-data    Running         2\n"
    )

    def _fake_run(command: list[str], *, timeout: float) -> tuple[subprocess.CompletedProcess[str] | None, str | None]:
        if command[:2] == ["wsl.exe", "--status"]:
            return subprocess.CompletedProcess(command, 0, payload_status, ""), None
        if command[:3] == ["wsl.exe", "-l", "-v"]:
            return subprocess.CompletedProcess(command, 0, payload_list, ""), None
        if "Microsoft-Hyper-V-All" in command[-1]:
            return subprocess.CompletedProcess(command, 0, "Enabled\n", ""), None
        if "VirtualMachinePlatform" in command[-1]:
            return subprocess.CompletedProcess(command, 0, "Enabled\n", ""), None
        if "vmcompute" in command[-1]:
            return subprocess.CompletedProcess(command, 0, "Stopped\n", ""), None
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(bootstrap_env, "_run_command", _fake_run)

    warnings, errors, metadata = bootstrap_env._collect_windows_virtualization_insights(timeout=0.1)

    assert any("default wsl distribution" in warning.lower() for warning in warnings)
    assert any("docker desktop" in error.lower() for error in errors)
    assert any("vmcompute" in error.lower() for error in errors)
    assert metadata["wsl_distro_docker_desktop_state"] == "Stopped"
    assert metadata["vmcompute_status"].lower() == "stopped"
