"""Regression tests for Docker diagnostics in ``scripts/bootstrap_env``."""

from __future__ import annotations

import json
import os
import subprocess
import types
from pathlib import Path

import pytest

from scripts import bootstrap_env


def _windows_context() -> bootstrap_env.RuntimeContext:
    """Return a RuntimeContext representing a Windows host."""

    return bootstrap_env.RuntimeContext(
        platform="win32",
        is_windows=True,
        is_wsl=False,
        inside_container=False,
        container_runtime=None,
        container_indicators=(),
        is_ci=False,
        ci_indicators=(),
    )


def test_windows_path_translates_to_wsl_mount_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Windows drive paths should translate into the configured WSL mount root."""

    mount_root = tmp_path / "wsl-root"
    mount_root.mkdir()

    bootstrap_env._get_wsl_host_mount_root.cache_clear()
    monkeypatch.setenv("WSL_HOST_MOUNT_ROOT", str(mount_root))
    monkeypatch.setattr(bootstrap_env, "_is_windows", lambda: False)
    monkeypatch.setattr(bootstrap_env, "_is_wsl", lambda: True)

    windows_path = Path(r"C:\Program Files\Docker\Docker\resources\cli")
    translated = bootstrap_env._translate_windows_host_path(windows_path)

    expected = (
        mount_root
        / "c"
        / "Program Files"
        / "Docker"
        / "Docker"
        / "resources"
        / "cli"
    )

    assert translated == expected


def test_resolve_command_path_discovers_windows_cli_from_wsl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``_resolve_command_path`` should locate Windows CLIs exposed to WSL."""

    bootstrap_env._resolve_command_path.cache_clear()
    bootstrap_env._get_wsl_host_mount_root.cache_clear()

    mount_root = tmp_path / "wsl-root"
    system32 = mount_root / "c" / "Windows" / "System32"
    system32.mkdir(parents=True)

    shim = system32 / "wsl.exe"
    shim.write_text("#!/bin/sh\necho status\n", encoding="utf-8")
    shim.chmod(0o755)

    monkeypatch.setenv("WSL_HOST_MOUNT_ROOT", str(mount_root))
    monkeypatch.setenv("SystemRoot", r"C:\\Windows")
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(bootstrap_env, "_is_windows", lambda: False)
    monkeypatch.setattr(bootstrap_env, "_is_wsl", lambda: True)

    resolved = bootstrap_env._resolve_command_path("wsl.exe")

    assert resolved == str(shim)


def test_run_command_executes_resolved_windows_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``_run_command`` should execute CLIs discovered via Windows fallbacks."""

    bootstrap_env._resolve_command_path.cache_clear()
    bootstrap_env._get_wsl_host_mount_root.cache_clear()

    mount_root = tmp_path / "wsl-root"
    system32 = mount_root / "c" / "Windows" / "System32"
    system32.mkdir(parents=True)

    shim = system32 / "bcdedit.exe"
    shim.write_text("#!/bin/sh\necho run-ok\n", encoding="utf-8")
    shim.chmod(0o755)

    monkeypatch.setenv("WSL_HOST_MOUNT_ROOT", str(mount_root))
    monkeypatch.setenv("SystemRoot", r"C:\\Windows")
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(bootstrap_env, "_is_windows", lambda: False)
    monkeypatch.setattr(bootstrap_env, "_is_wsl", lambda: True)

    completed, failure = bootstrap_env._run_command(["bcdedit.exe"], timeout=2.0)

    assert failure is None
    assert completed is not None
    assert "run-ok" in completed.stdout


def test_get_wsl_mount_root_respects_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The automount root from configuration files should drive translation."""

    bootstrap_env._get_wsl_host_mount_root.cache_clear()

    custom_root = tmp_path / "wsl-root"
    (custom_root / "c").mkdir(parents=True)

    config_path = tmp_path / "wsl.conf"
    config_path.write_text("[automount]\nroot = %s\n" % custom_root.as_posix(), encoding="utf-8")

    monkeypatch.delenv("WSL_HOST_MOUNT_ROOT", raising=False)
    monkeypatch.setattr(bootstrap_env, "_is_windows", lambda: False)
    monkeypatch.setattr(bootstrap_env, "_is_wsl", lambda: True)
    monkeypatch.setattr(
        bootstrap_env,
        "_iter_wsl_configuration_paths",
        lambda: (config_path,),
    )
    monkeypatch.setattr(
        bootstrap_env,
        "_iter_wsl_mount_roots_from_proc",
        lambda: (),
    )

    root = bootstrap_env._get_wsl_host_mount_root()

    assert root == custom_root


def test_get_wsl_mount_root_prefers_proc_mounts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mount candidates discovered via ``/proc`` should be preferred."""

    bootstrap_env._get_wsl_host_mount_root.cache_clear()

    proc_root = tmp_path / "drvfs"
    (proc_root / "c").mkdir(parents=True)

    monkeypatch.delenv("WSL_HOST_MOUNT_ROOT", raising=False)
    monkeypatch.setattr(bootstrap_env, "_is_windows", lambda: False)
    monkeypatch.setattr(bootstrap_env, "_is_wsl", lambda: True)
    monkeypatch.setattr(bootstrap_env, "_iter_wsl_configuration_paths", lambda: ())
    monkeypatch.setattr(
        bootstrap_env,
        "_iter_wsl_mount_roots_from_proc",
        lambda: (proc_root,),
    )

    root = bootstrap_env._get_wsl_host_mount_root()

    assert root == proc_root


def test_translate_windows_path_uses_detected_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Windows path translation should honour the detected automount root."""

    bootstrap_env._get_wsl_host_mount_root.cache_clear()

    custom_root = tmp_path / "wslaut"
    target = custom_root / "c" / "Program Files"
    target.mkdir(parents=True)

    config_path = tmp_path / "wsl.conf"
    config_path.write_text("[automount]\nroot = %s\n" % custom_root.as_posix(), encoding="utf-8")

    monkeypatch.delenv("WSL_HOST_MOUNT_ROOT", raising=False)
    monkeypatch.setattr(bootstrap_env, "_is_windows", lambda: False)
    monkeypatch.setattr(bootstrap_env, "_is_wsl", lambda: True)
    monkeypatch.setattr(
        bootstrap_env,
        "_iter_wsl_configuration_paths",
        lambda: (config_path,),
    )
    monkeypatch.setattr(
        bootstrap_env,
        "_iter_wsl_mount_roots_from_proc",
        lambda: (),
    )

    translated = bootstrap_env._translate_windows_host_path(Path(r"C:\\Program Files"))

    assert translated == target


def test_docker_cli_discovery_uses_wsl_translation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLI discovery should resolve Windows paths via WSL mount translations."""

    bootstrap_env._get_wsl_host_mount_root.cache_clear()
    mount_root = tmp_path / "wsl-root"
    target_dir = (
        mount_root
        / "c"
        / "Program Files"
        / "Docker"
        / "Docker"
        / "resources"
        / "cli"
    )
    target_dir.mkdir(parents=True)

    cli_path = target_dir / "docker.exe"
    cli_path.write_text("@echo off\n")
    cli_path.chmod(0o755)

    monkeypatch.setenv("WSL_HOST_MOUNT_ROOT", str(mount_root))
    monkeypatch.delenv("ProgramFiles", raising=False)
    monkeypatch.delenv("ProgramW6432", raising=False)
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)
    monkeypatch.delenv("ProgramData", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)

    monkeypatch.setattr(bootstrap_env, "_is_windows", lambda: False)
    monkeypatch.setattr(bootstrap_env, "_is_wsl", lambda: True)
    monkeypatch.setattr(bootstrap_env.shutil, "which", lambda executable: None)

    resolved, warnings = bootstrap_env._discover_docker_cli()

    assert resolved == cli_path
    assert warnings
    assert any("WSL interop" in warning for warning in warnings)


def test_windows_docker_directory_variants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ensure Windows Docker discovery scans modern CLI bundle locations."""

    program_files = tmp_path / "Program Files"
    monkeypatch.setenv("ProgramFiles", str(program_files))
    monkeypatch.delenv("ProgramW6432", raising=False)
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)

    program_data = tmp_path / "ProgramData"
    monkeypatch.setenv("ProgramData", str(program_data))

    local_appdata = tmp_path / "LocalAppData"
    monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))

    directories = list(bootstrap_env._iter_windows_docker_directories())
    directory_set = set(directories)

    expected = {
        program_files / "Docker" / "Docker" / "resources" / "cli",
        program_files / "Docker" / "Docker" / "resources" / "cli-wsl",
        program_files / "Docker" / "Docker" / "resources" / "cli-linux",
        program_files / "Docker" / "Docker" / "resources" / "cli-bin",
        program_files / "Docker" / "Docker" / "resources" / "docker-cli",
        program_files / "Docker" / "Docker" / "resources" / "cli-arm",
        program_files / "Docker" / "Docker" / "resources" / "cli-arm64",
        program_files / "Docker" / "Docker" / "cli",
        program_data / "DockerDesktop" / "cli",
        program_data / "DockerDesktop" / "cli-bin",
        program_data / "DockerDesktop" / "cli-tools",
        local_appdata / "DockerDesktop" / "cli",
        local_appdata / "DockerDesktop" / "cli-bin",
        local_appdata / "DockerDesktop" / "cli-tools",
        local_appdata / "DockerDesktop" / "cli-arm",
        local_appdata / "DockerDesktop" / "cli-arm64",
    }

    assert expected.issubset(directory_set)


def test_windows_docker_directory_includes_arm_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Docker discovery should probe Program Files (Arm) when available."""

    arm_root = tmp_path / "Program Files (Arm)"
    arm_root.mkdir()

    monkeypatch.delenv("ProgramFiles", raising=False)
    monkeypatch.delenv("ProgramW6432", raising=False)
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)
    monkeypatch.setenv("ProgramFiles(Arm)", str(arm_root))

    directories = list(bootstrap_env._iter_windows_docker_directories())
    directory_set = {Path(path) for path in directories}

    expected = {
        arm_root / "Docker" / "Docker" / "resources" / "cli-arm",
        arm_root / "Docker" / "Docker" / "resources" / "cli-arm64",
        arm_root / "Docker" / "Docker" / "cli",
    }

    assert expected.issubset(directory_set)


def test_windows_docker_directory_includes_versioned_app_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Docker discovery should surface versioned DockerDesktop app bundles."""

    program_files = tmp_path / "Program Files"
    program_files.mkdir()
    monkeypatch.setenv("ProgramFiles", str(program_files))
    monkeypatch.delenv("ProgramW6432", raising=False)
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)
    monkeypatch.delenv("ProgramFiles(Arm)", raising=False)

    program_data = tmp_path / "ProgramData"
    program_data.mkdir()
    monkeypatch.setenv("ProgramData", str(program_data))

    local_appdata = tmp_path / "LocalAppData"
    app_dir = local_appdata / "DockerDesktop" / "app-4.33.0"
    (app_dir / "resources" / "bin").mkdir(parents=True)
    (app_dir / "resources" / "cli").mkdir(parents=True)
    (app_dir / "cli-bin").mkdir(parents=True)
    monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))

    directories = {Path(entry) for entry in bootstrap_env._iter_windows_docker_directories()}

    expected = {
        app_dir / "resources" / "bin",
        app_dir / "resources" / "cli",
        app_dir / "cli-bin",
    }

    assert expected.issubset(directories)


def test_worker_warning_sanitization_removes_raw_banner() -> None:
    """Ensure ``worker stalled`` banners are rewritten into guidance."""

    cleaned, metadata = bootstrap_env._normalise_docker_warning(
        "WARNING: worker stalled; restarting"
    )

    assert cleaned  # sanity check that a message is produced
    assert "worker stalled; restarting" not in cleaned.lower()
    assert "Docker Desktop recovered from transient worker stalls" in cleaned
    assert metadata["docker_worker_health"] == "flapping"


def test_plural_worker_banner_is_normalised() -> None:
    """Plural ``workers stalled`` banners should be normalised into guidance."""

    message = (
        "WARNING: workers stalled; restarting component=\"vpnkit\" restartCount=3"
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled; restarting" not in cleaned.lower()
    assert metadata["docker_worker_health"] == "flapping"
    assert metadata["docker_worker_restart_count"] == "3"
    assert metadata["docker_worker_context"] == "vpnkit"


def test_parenthetical_plural_worker_banner_is_normalised() -> None:
    """``worker(s) stalled`` phrasing should also be rewritten into guidance."""

    message = (
        "WARNING: worker(s) stalled; restarting backoff=\"PT45S\" "
        "component=\"vpnkit\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled; restarting" not in cleaned.lower()
    assert metadata["docker_worker_backoff"] == "45s"
    assert metadata["docker_worker_context"] == "vpnkit"


def test_camelcase_worker_banner_is_normalised() -> None:
    """CamelCase ``workerStalled`` banners should be rewritten into guidance."""

    message = "WARNING: workerStalled;Restarting component=\"vpnkit\" restartCount=3"

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert cleaned
    assert "worker stalled" not in cleaned.lower()
    assert metadata["docker_worker_health"] == "flapping"
    assert metadata["docker_worker_restart_count"] == "3"
    assert metadata["docker_worker_context"] == "vpnkit"


def test_worker_stalls_variant_is_normalised() -> None:
    """``worker stalls`` (present tense) banners should be harmonised."""

    message = (
        "WARNING: worker stalls; restarting component=\"vpnkit\" restartCount=5"
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert cleaned
    assert "worker stalls" not in cleaned.lower()
    assert metadata["docker_worker_health"] == "flapping"
    assert metadata["docker_worker_restart_count"] == "5"
    assert metadata["docker_worker_context"] == "vpnkit"


def test_worker_has_stalled_phrase_is_sanitized() -> None:
    """Phrases like ``worker has stalled`` should be harmonised into guidance."""

    message = "WARNING: worker has stalled; restarting due to IO pressure"

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert cleaned
    assert "worker stalled; restarting" not in cleaned.lower()
    assert "io pressure" in cleaned.lower()
    assert metadata["docker_worker_health"] == "flapping"


def test_worker_warning_extracts_last_healthy_marker() -> None:
    """Docker worker warnings should surface the last healthy timestamp."""

    message = (
        "WARNING: worker stalled; restarting component=\"vpnkit\" "
        "lastHealthy=2024-10-03T17:45:00Z restartCount=2"
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert cleaned
    assert metadata["docker_worker_last_healthy"] == "2024-10-03T17:45:00Z"
    assert "Docker Desktop recovered from transient worker stalls" in cleaned


def test_post_process_virtualization_insights_for_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Virtualization diagnostics should run even for warning level telemetry."""

    called = types.SimpleNamespace(count=0)

    def fake_collect(timeout: float) -> tuple[list[str], list[str], dict[str, str]]:
        called.count += 1
        return (
            ["WSL kernel update is available"],
            ["WSL default version is set to 1"],
            {"wsl_default_version": "1"},
        )

    monkeypatch.setattr(
        bootstrap_env,
        "_collect_windows_virtualization_insights",
        fake_collect,
    )

    warnings, errors, metadata = bootstrap_env._post_process_docker_health(
        metadata={"docker_worker_health": "flapping"},
        context=_windows_context(),
        timeout=0.01,
    )

    assert called.count == 1
    assert errors == []
    assert any("Virtualization issue detected" in warning for warning in warnings)
    assert metadata["docker_worker_health_severity"] == "warning"
    assert "docker_worker_health_reasons" not in metadata
    assert metadata["wsl_default_version"] == "1"
    summary = metadata["docker_worker_health_summary"]
    assert summary
    assert "worker stalled" not in summary.lower()
    assert summary in warnings


def test_post_process_virtualization_insights_for_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Virtualization diagnostics feed into errors for severe flapping."""

    def fake_collect(timeout: float) -> tuple[list[str], list[str], dict[str, str]]:
        return (
            ["Docker Desktop WSL distribution is stopped"],
            ["WSL integration is disabled"],
            {"wsl_integration": "disabled"},
        )

    monkeypatch.setattr(
        bootstrap_env,
        "_collect_windows_virtualization_insights",
        fake_collect,
    )

    warnings, errors, metadata = bootstrap_env._post_process_docker_health(
        metadata={
            "docker_worker_health": "flapping",
            "docker_worker_restart_count": "8",
            "docker_worker_context": "vpnkit",
        },
        context=_windows_context(),
        timeout=0.01,
    )

    assert any("Docker Desktop WSL distribution is stopped" in warning for warning in warnings)
    assert any(error == "WSL integration is disabled" for error in errors)
    assert metadata["docker_worker_health_severity"] == "error"
    reasons = metadata["docker_worker_health_reasons"]
    assert "six worker restarts" in reasons
    assert metadata["wsl_integration"] == "disabled"
    summary = metadata["docker_worker_health_summary"]
    assert summary
    assert "worker stalled" not in summary.lower()
    assert summary in errors


def test_transient_worker_flaps_downgraded_to_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benign worker stalls should be reported as informational guidance."""

    message = (
        "WARNING: worker stalled; restarting component=\"vpnkit\" "
        "restartCount=3 backoff=PT30S "
        "errCode=VPNKIT_BACKGROUND_SYNC_STALLED "
        "lastError=\"vpnkit background sync stalled\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled; restarting" not in cleaned.lower()
    metadata["docker_worker_warning_occurrences"] = "2"

    telemetry = bootstrap_env.WorkerRestartTelemetry.from_metadata(metadata)
    assessment = bootstrap_env._classify_worker_flapping(telemetry, _windows_context())

    assert assessment.severity == "info"
    assert "transient worker stalls" in assessment.headline.lower()
    assert assessment.metadata.get("docker_worker_health_state") == "stabilising"

    monkeypatch.setattr(
        bootstrap_env,
        "_collect_windows_virtualization_insights",
        lambda timeout: ([], [], {}),
    )

    warnings, errors, extra_metadata = bootstrap_env._post_process_docker_health(
        metadata=dict(metadata),
        context=_windows_context(),
        timeout=0.01,
    )

    assert warnings == []
    assert errors == []
    assert extra_metadata["docker_worker_health_severity"] == "info"
    summary = extra_metadata["docker_worker_health_summary"].lower()
    assert "worker stalled" not in summary
    assert "recovered" in summary or "transient" in summary


def test_worker_warning_wsl_vm_suspended_errcode() -> None:
    """Explicit WSL suspension error codes should produce virtualization guidance."""

    message = (
        "WARNING: worker stalled; restarting component=\"vm\" "
        "restartCount=5 backoff=PT90S "
        "errCode=WSL_VM_SUSPENDED "
        "lastError=\"WSL virtual machine suspended during host sleep\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled; restarting" not in cleaned.lower()
    assert metadata["docker_worker_last_error_code"] == "WSL_VM_SUSPENDED"
    summary = metadata["docker_worker_health_summary"].lower()
    assert "suspend" in summary or "wsl" in summary
    guidance = metadata["docker_worker_health_remediation"]
    assert "wsl --shutdown" in guidance


def test_worker_warning_hung_variant() -> None:
    """Hung worker phrasing should normalise to the canonical guidance."""

    message = (
        "WARNING: worker hung; restarting component=\"vpnkit\" "
        "restartCount=4 errCode=VPNKIT_UNRESPONSIVE"
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker hung" not in cleaned.lower()
    assert metadata["docker_worker_last_error_code"] == "VPNKIT_UNRESPONSIVE"
    assert "vpnkit" in metadata["docker_worker_health_details"].lower()


def test_worker_warning_frozen_variant() -> None:
    """Frozen worker phrasing should also collapse into guidance."""

    message = (
        "WARN[0042] moby/buildkit: worker frozen; restarting due to host sleep cycle"
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker frozen" not in cleaned.lower()
    assert metadata["docker_worker_health"] == "flapping"
    assert "restart" in metadata["docker_worker_health_summary"].lower()


def test_worker_warning_wsl_vm_suspended_inferred() -> None:
    """Suspension phrasing without explicit codes should infer WSL suspension faults."""

    message = (
        "WARN[0042] moby/buildkit: worker stalled; restarting due to WSL "
        "virtual machine suspended after hibernation"
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled; restarting" not in cleaned.lower()
    assert metadata["docker_worker_last_error_code"] == "WSL_VM_SUSPENDED"
    details = metadata["docker_worker_health_details"].lower()
    assert "suspend" in details or "hibern" in details


def test_worker_warning_hyperv_not_running_inferred() -> None:
    """Hyper-V outage phrasing should map to the not-running directive."""

    message = (
        "WARNING: worker stalled; restarting component=\"vm\" "
        "lastError=\"Hyper-V hypervisor not running; hypervisorlaunchtype off\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled; restarting" not in cleaned.lower()
    assert metadata["docker_worker_last_error_code"] == "HCS_E_HYPERV_NOT_RUNNING"
    remediation = metadata["docker_worker_health_remediation"].lower()
    assert "hyper-v" in remediation
    assert "hypervisor" in remediation or "hypervisorlaunchtype" in remediation


def test_worker_warning_hyperv_not_present_errcode() -> None:
    """Explicit Hyper-V missing codes should surface installation guidance."""

    message = (
        "WARNING: worker stalled; restarting component=\"vm\" "
        "errCode=HCS_E_HYPERV_NOT_PRESENT "
        "lastError=\"Hyper-V features are not installed\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled; restarting" not in cleaned.lower()
    assert metadata["docker_worker_last_error_code"] == "HCS_E_HYPERV_NOT_PRESENT"
    details = metadata["docker_worker_health_details"].lower()
    assert "hyper-v" in details or "virtual machine platform" in details
    remediation = metadata["docker_worker_health_remediation"].lower()
    assert "optionalfeatures" in remediation or "enable" in remediation

def test_collect_docker_diagnostics_virtualization_without_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Virtualization diagnostics should surface when the Docker CLI is missing."""

    context = _windows_context()

    monkeypatch.setattr(bootstrap_env, "_detect_runtime_context", lambda: context)
    monkeypatch.setattr(
        bootstrap_env,
        "_discover_docker_cli",
        lambda: (None, []),
    )
    monkeypatch.setattr(
        bootstrap_env,
        "_infer_missing_docker_skip_reason",
        lambda ctx: None,
    )

    calls = types.SimpleNamespace(count=0)

    def fake_virtualization(timeout: float) -> tuple[list[str], list[str], dict[str, str]]:
        calls.count += 1
        return (
            ["Hyper-V optional feature is disabled"],
            ["Virtual Machine Platform is disabled"],
            {"hyper_v_state": "Disabled"},
        )

    monkeypatch.setattr(
        bootstrap_env,
        "_collect_windows_virtualization_insights",
        fake_virtualization,
    )

    result = bootstrap_env._collect_docker_diagnostics(timeout=0.25)

    assert calls.count == 1
    assert result.cli_path is None
    assert not result.available
    assert any("hyper-v" in warning.lower() for warning in result.warnings)
    assert any("virtual machine platform" in error.lower() for error in result.errors)
    assert "hyper_v_state" in result.metadata
    assert all("worker stalled" not in warning.lower() for warning in result.warnings)
    assert all("worker stalled" not in error.lower() for error in result.errors)


def test_worker_error_code_guidance_enriches_classification() -> None:
    """Known worker error codes should drive actionable remediation guidance."""

    telemetry = bootstrap_env.WorkerRestartTelemetry.from_metadata(
        {
            "docker_worker_context": "vpnkit",
            "docker_worker_health": "flapping",
            "docker_worker_last_error_code": "WSL_KERNEL_OUTDATED",
            "docker_worker_last_error": "WSL kernel outdated",
        }
    )

    assessment = bootstrap_env._classify_worker_flapping(telemetry, _windows_context())

    assert any(
        "windows subsystem for linux" in reason.lower()
        for reason in assessment.reasons
    )
    assert any("wsl --update" in step.lower() for step in assessment.remediation)
    guidance_key = "docker_worker_last_error_guidance_wsl_kernel_outdated"
    assert assessment.metadata[guidance_key].startswith("Update the WSL kernel")


def test_virtualization_followups_triggered_by_worker_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """WSL/Hyper-V worker errors should trigger deeper virtualization diagnostics."""

    fake_cli = tmp_path / "docker.exe"
    fake_cli.write_text("", encoding="utf-8")

    monkeypatch.setattr(bootstrap_env, "_discover_docker_cli", lambda: (fake_cli, []))
    monkeypatch.setattr(bootstrap_env, "_detect_runtime_context", lambda: _windows_context())

    def fake_probe(
        cli_path: Path, timeout: float
    ) -> tuple[dict[str, str], list[str], list[str]]:
        assert cli_path == fake_cli
        return ({"server_version": "26.0.0"}, [], [])

    monkeypatch.setattr(bootstrap_env, "_probe_docker_environment", fake_probe)

    def fake_post_process(
        *, metadata: dict[str, str], context: bootstrap_env.RuntimeContext, timeout: float = 6.0
    ) -> tuple[list[str], list[str], dict[str, str]]:
        metadata.update(
            {
                "docker_worker_last_error_code": "WSL_VM_CRASHED",
                "docker_worker_health_severity": "error",
                "docker_worker_health_summary": "Docker Desktop reported its WSL VM crashed.",
            }
        )
        return [], [], {}

    monkeypatch.setattr(bootstrap_env, "_post_process_docker_health", fake_post_process)

    calls: dict[str, float] = {}

    def fake_virtualization(timeout: float) -> tuple[list[str], list[str], dict[str, str]]:
        calls["timeout"] = timeout
        return (
            ["WSL virtualization check warning"],
            ["WSL virtualization check error"],
            {"wsl_status_raw": "Status: Running"},
        )

    monkeypatch.setattr(
        bootstrap_env,
        "_collect_windows_virtualization_insights",
        fake_virtualization,
    )

    result = bootstrap_env._collect_docker_diagnostics(timeout=0.5)

    assert calls["timeout"] == pytest.approx(0.5, rel=1e-6)
    assert result.cli_path == fake_cli
    assert not result.available
    assert any("virtualization check warning" in warning.lower() for warning in result.warnings)
    assert any("virtualization check error" in error.lower() for error in result.errors)
    assert result.metadata["wsl_status_raw"].startswith("Status:")


def test_collect_diagnostics_redacts_residual_worker_banner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Metadata should not leak verbatim ``worker stalled`` banners."""

    fake_cli = tmp_path / "docker.exe"
    fake_cli.write_text("", encoding="utf-8")

    monkeypatch.setattr(bootstrap_env, "_discover_docker_cli", lambda: (fake_cli, []))
    monkeypatch.setattr(bootstrap_env, "_detect_runtime_context", lambda: _windows_context())

    def fake_probe(
        cli_path: Path, timeout: float
    ) -> tuple[dict[str, str], list[str], list[str]]:
        assert cli_path == fake_cli
        return ({"server_version": "26.1.0"}, [], [])

    monkeypatch.setattr(bootstrap_env, "_probe_docker_environment", fake_probe)

    def fake_post_process(
        *, metadata: dict[str, str], context: bootstrap_env.RuntimeContext, timeout: float = 6.0
    ) -> tuple[list[str], list[str], dict[str, str]]:
        metadata.update(
            {
                "docker_worker_last_error_banner_raw": "WARNING: worker stalled; restarting",
                "docker_worker_last_error_banner_preserved": "worker stalled; restarting",
            }
        )
        return [], [], {}

    monkeypatch.setattr(bootstrap_env, "_post_process_docker_health", fake_post_process)

    result = bootstrap_env._collect_docker_diagnostics(timeout=0.5)

    banner_raw = result.metadata["docker_worker_last_error_banner_raw"]
    preserved = result.metadata["docker_worker_last_error_banner_preserved"]

    assert "worker stalled; restarting" not in banner_raw.lower()
    assert "worker stalled; restarting" not in preserved.lower()
    assert result.metadata["docker_worker_last_error_banner_raw_fingerprint"].startswith(
        bootstrap_env._WORKER_STALLED_SIGNATURE_PREFIX
    )


def test_virtualization_followups_skipped_for_non_virtualization_issue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Non-virtualization worker issues should not trigger Windows diagnostics."""

    fake_cli = tmp_path / "docker.exe"
    fake_cli.write_text("", encoding="utf-8")

    monkeypatch.setattr(bootstrap_env, "_discover_docker_cli", lambda: (fake_cli, []))
    monkeypatch.setattr(bootstrap_env, "_detect_runtime_context", lambda: _windows_context())

    monkeypatch.setattr(
        bootstrap_env,
        "_probe_docker_environment",
        lambda cli_path, timeout: ({"server_version": "26.0.0"}, [], []),
    )

    def fake_post_process(
        *, metadata: dict[str, str], context: bootstrap_env.RuntimeContext, timeout: float = 6.0
    ) -> tuple[list[str], list[str], dict[str, str]]:
        metadata.update(
            {
                "docker_worker_last_error_code": "VPNKIT_UNRESPONSIVE",
                "docker_worker_health_severity": "warning",
                "docker_worker_health_summary": "Docker Desktop reported networking restarts.",
            }
        )
        return [], [], {}

    monkeypatch.setattr(bootstrap_env, "_post_process_docker_health", fake_post_process)

    def virtualization_not_expected(timeout: float) -> tuple[list[str], list[str], dict[str, str]]:
        raise AssertionError("Virtualization diagnostics should not run for networking stalls")

    monkeypatch.setattr(
        bootstrap_env,
        "_collect_windows_virtualization_insights",
        virtualization_not_expected,
    )

    result = bootstrap_env._collect_docker_diagnostics(timeout=0.5)

    assert result.cli_path == fake_cli
    assert result.available
    assert result.warnings == ()
    assert result.errors == ()


def test_collect_windows_service_health_flags_service_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Critical Docker Desktop services should raise actionable findings."""

    payload = json.dumps(
        [
            {"Name": "vmcompute", "Status": "Stopped", "StartType": "Automatic"},
            {"Name": "hns", "Status": "Running", "StartType": "Automatic"},
            {"Name": "LxssManager", "Status": "Running", "StartType": "Automatic"},
            {"Name": "vmms", "Status": "Stopped", "StartType": "Manual"},
            {"Name": "com.docker.service", "Status": "Missing", "StartType": "Unknown"},
        ]
    )

    def fake_run_command(
        command: list[str], *, timeout: float
    ) -> tuple[subprocess.CompletedProcess[str] | None, str | None]:
        assert command[0].lower().endswith("powershell.exe")
        return subprocess.CompletedProcess(command, 0, payload, ""), None

    monkeypatch.setattr(bootstrap_env, "_run_command", fake_run_command)

    warnings, errors, metadata = bootstrap_env._collect_windows_service_health(timeout=0.25)

    assert any("hyper-v host compute service" in message.lower() for message in errors)
    assert any("docker desktop service" in message.lower() for message in errors)
    assert any("virtual machine management service" in message.lower() for message in warnings)
    assert metadata["windows_service_vmcompute_status"] == "Stopped"
    assert metadata["windows_service_com_docker_service_status"] == "Missing"
    assert metadata["vmcompute_status"] == "Stopped"


def test_collect_windows_service_health_handles_command_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failures to execute PowerShell should downgrade into warnings."""

    def fake_run_command(
        command: list[str], *, timeout: float
    ) -> tuple[subprocess.CompletedProcess[str] | None, str | None]:
        return None, "Executable 'powershell.exe' is not available on PATH"

    monkeypatch.setattr(bootstrap_env, "_run_command", fake_run_command)

    warnings, errors, metadata = bootstrap_env._collect_windows_service_health(timeout=0.25)

    assert warnings and "powershell.exe" in warnings[0].lower()
    assert errors == []
    assert metadata == {}


def test_summarize_docker_command_failure_sanitizes_worker_banner() -> None:
    """Non-zero docker exit codes should surface cleaned warnings and metadata."""

    payload_stdout = "WARNING: worker stalled; restarting"
    payload_stderr = (
        "warning: worker stalled; restarting component=\"vpnkit\" restartCount=3\n"
        "Error: context deadline exceeded"
    )

    completed = subprocess.CompletedProcess(
        ["docker", "info"],
        1,
        payload_stdout,
        payload_stderr,
    )

    message, warnings, metadata = bootstrap_env._summarize_docker_command_failure(
        completed,
        "info",
    )

    assert "worker stalled; restarting" not in message.lower()
    assert "context deadline exceeded" in message
    assert any(
        "docker desktop recovered from transient worker stalls" in warning.lower()
        for warning in warnings
    )
    assert metadata["docker_worker_health"] == "flapping"


def test_summarize_docker_command_failure_handles_whitespace_separator() -> None:
    """Stall banners separated only by whitespace should be rewritten."""

    payload_stdout = "WARNING: worker stalled    restarting"
    payload_stderr = ""

    completed = subprocess.CompletedProcess(
        ["docker", "info"],
        1,
        payload_stdout,
        payload_stderr,
    )

    message, warnings, metadata = bootstrap_env._summarize_docker_command_failure(
        completed,
        "info",
    )

    normalized = " ".join(message.split())
    assert "worker stalled; restarting" not in normalized.lower()
    assert warnings
    assert all("worker stalled; restarting" not in warning.lower() for warning in warnings)
    assert metadata["docker_worker_health"] == "flapping"


def test_worker_warning_with_hyphenated_restart_is_normalised() -> None:
    """Hyphenated restart tokens should still be detected as worker stall banners."""

    message = (
        "WARNING: worker stalled; re-starting component=\"vpnkit\" "
        "restartCount=2 lastError=\"worker stalled; re-starting\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled; re-starting" not in cleaned.lower()
    assert metadata["docker_worker_health"] == "flapping"


def test_worker_warning_without_explicit_restart_is_detected() -> None:
    """Worker stall diagnostics lacking the word restart should still be sanitised."""

    message = (
        "WARN[0032] moby/buildkit: worker stalled (component=\"vpnkitCore\") "
        "status=stalled background=sync"
    )

    normalized, metadata = bootstrap_env._normalize_warning_collection([message])

    assert normalized, "expected sanitized worker warning"
    assert all("worker stalled" not in entry.lower() for entry in normalized)
    assert metadata["docker_worker_health"] == "flapping"


def test_enforce_worker_banner_sanitization_removes_literal_banner() -> None:
    """A final sanitisation pass should eliminate raw worker stall banners."""

    metadata: dict[str, str] = {}
    warnings = ["WARNING: worker stalled; restarting"]

    harmonised = bootstrap_env._enforce_worker_banner_sanitization(warnings, metadata)

    assert harmonised, "expected sanitized worker warning"
    assert all("worker stalled; restarting" not in entry.lower() for entry in harmonised)
    assert metadata["docker_worker_health"] == "flapping"


def test_enforce_worker_banner_sanitization_handles_colon_separator() -> None:
    """Windows Docker builds occasionally surface colon separated worker stalls."""

    metadata: dict[str, str] = {}
    warnings = ["WARNING worker stalled : restarting (errCode=VPNKIT_VSOCK_TIMEOUT)"]

    harmonised = bootstrap_env._enforce_worker_banner_sanitization(warnings, metadata)

    assert harmonised, "expected sanitized worker warning"
    assert all("worker stalled" not in entry.lower() for entry in harmonised)
    assert metadata["docker_worker_health"] == "flapping"
    assert metadata["docker_worker_last_error_code"] == "VPNKIT_VSOCK_TIMEOUT"


def test_enforce_worker_banner_sanitization_handles_dash_separator() -> None:
    """Unicode dash separators from Docker Desktop logs should be normalised."""

    metadata: dict[str, str] = {}
    warnings = [
        "WARNING worker stalled – re-starting component=\"vpnkit\" restartCount=3",
    ]

    harmonised = bootstrap_env._enforce_worker_banner_sanitization(warnings, metadata)

    assert harmonised, "expected sanitized worker warning"
    assert all("worker stalled" not in entry.lower() for entry in harmonised)
    assert metadata["docker_worker_health"] == "flapping"


def test_guarantee_worker_banner_suppression_handles_spaced_semicolon() -> None:
    """Final guard should rewrite banners with padded delimiters."""

    metadata: dict[str, str] = {}
    warnings = [
        "WARNING worker stalled ; restarting component=\"vpnkit\" restartCount=2",
    ]

    safeguarded = bootstrap_env._guarantee_worker_banner_suppression(warnings, metadata)

    assert safeguarded, "expected sanitized worker warning"
    assert all("worker stalled" not in entry.lower() for entry in safeguarded)
    assert metadata["docker_worker_health"] == "flapping"


def test_guarantee_worker_banner_suppression_handles_arrow_separator() -> None:
    """Arrow separators occasionally appear in Windows event streams."""

    metadata: dict[str, str] = {}
    warnings = [
        "WARN[0042] moby/buildkit: worker stalled -> restarting due to IO pressure",
    ]

    safeguarded = bootstrap_env._guarantee_worker_banner_suppression(warnings, metadata)

    assert safeguarded, "expected sanitized worker warning"
    assert all("worker stalled" not in entry.lower() for entry in safeguarded)
    assert metadata["docker_worker_health"] == "flapping"


def test_guarantee_worker_banner_suppression_handles_comma_separator() -> None:
    """Comma-separated worker restarts should be rewritten into guidance."""

    metadata: dict[str, str] = {}
    warnings = [
        "WARN[0015] moby/buildkit: worker stalled, restarting component=\"vpnkit\" restartCount=2",
    ]

    safeguarded = bootstrap_env._guarantee_worker_banner_suppression(warnings, metadata)

    assert safeguarded, "expected sanitized worker warning"
    assert all("worker stalled" not in entry.lower() for entry in safeguarded)
    assert metadata["docker_worker_health"] == "flapping"


def test_guarantee_worker_banner_suppression_preserves_guidance() -> None:
    """Guidance messages should flow through untouched."""

    metadata: dict[str, str] = {}
    guidance = bootstrap_env._WORKER_STALLED_PRIMARY_NARRATIVE

    safeguarded = bootstrap_env._guarantee_worker_banner_suppression([guidance], metadata)

    assert safeguarded == [guidance]
    assert metadata == {}


def test_enforce_worker_banner_sanitization_handles_stuck_synonym() -> None:
    """Stuck phrasing introduced by newer Docker builds should be normalised."""

    metadata: dict[str, str] = {}
    warnings = [
        "WARN[0042] moby/buildkit: worker stuck; restarting component=\"vpnkit\" restartCount=2",
    ]

    harmonised = bootstrap_env._enforce_worker_banner_sanitization(warnings, metadata)

    assert harmonised, "expected sanitized worker warning"
    assert all("worker stuck; restarting" not in entry.lower() for entry in harmonised)
    assert metadata["docker_worker_health"] == "flapping"


def test_contains_worker_stall_signal_detects_stuck_banner() -> None:
    """Heuristics should treat stuck workers as stall incidents."""

    message = "WARNING: worker stuck; restarting component=\"vpnkit\""

    assert bootstrap_env._contains_worker_stall_signal(message)


def test_enforce_worker_banner_sanitization_handles_resetting_variants() -> None:
    """Resetting phrasing should be harmonised like restart diagnostics."""

    metadata: dict[str, str] = {}
    warnings = [
        "WARNING worker stall detection triggered; resetting component=\"vpnkit\" "
        "errCode=VPNKIT_VSOCK_TIMEOUT",
    ]

    harmonised = bootstrap_env._enforce_worker_banner_sanitization(warnings, metadata)

    assert harmonised, "expected sanitized worker warning"
    assert all("worker stalled" not in entry.lower() for entry in harmonised)
    assert all("resetting" not in entry.lower() for entry in harmonised)
    assert metadata["docker_worker_health"] == "flapping"
    assert metadata.get("docker_worker_last_error_code") == "VPNKIT_VSOCK_TIMEOUT"


def test_worker_banner_raw_metadata_is_redacted() -> None:
    """Raw banner metadata should be rewritten and fingerprinted."""

    message = (
        "WARNING: worker stalled; restarting component=\"vpnkit\" "
        "restartCount=2 lastError=\"worker stalled; restarting\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled" not in cleaned.lower()
    assert metadata["docker_worker_health"] == "flapping"

    sensitive_keys = [
        "docker_worker_last_error_banner_preserved_raw",
        "docker_worker_last_error_banner_preserved_raw_samples",
        "docker_worker_last_error_banner_raw",
        "docker_worker_last_error_banner_raw_samples",
    ]

    for key in sensitive_keys:
        value = metadata.get(key, "")
        if value:
            assert "worker stalled" not in value.lower()
            fingerprint_key = f"{key}_fingerprint"
            assert metadata.get(fingerprint_key), f"missing fingerprint for {key}"


def test_worker_metadata_value_collapses_restart_suffix() -> None:
    """Metadata sanitiser should strip ``; restarting`` fragments."""

    value = (
        "Docker Desktop automatically restarted a background worker after it stalled; restarting"
    )

    sanitized, digest = bootstrap_env._sanitize_worker_metadata_value(value)

    assert sanitized == (
        "Docker Desktop automatically restarted a background worker after it stalled"
    )
    assert digest
    assert "stalled; restarting" not in sanitized.lower()


def test_worker_metadata_value_preserves_context_while_collapsing_restart_suffix() -> None:
    """Composite metadata keeps auxiliary fields when trimming restart suffixes."""

    value = (
        'context="background-sync"; restartCount=3; '
        "Docker Desktop automatically restarted a background worker after it stalled; restarting in 45s"
    )

    sanitized, _ = bootstrap_env._sanitize_worker_metadata_value(value)

    assert "context=\"background-sync\"" in sanitized
    assert "restartcount=3" in sanitized.lower()
    assert "restart in 45s" in sanitized.lower()
    assert "stalled; restarting" not in sanitized.lower()


def test_worker_metadata_value_rewrites_embedded_json_banner() -> None:
    """JSON fragments embedded in metadata should be rewritten without raw banners."""

    value = json.dumps(
        {
            "component": "vpnkit",
            "status": "worker stalled; restarting",
            "details": {
                "lastError": "worker stalled; restarting due to IO pressure",
                "restartCount": 3,
            },
        }
    )

    sanitized, digest = bootstrap_env._sanitize_worker_metadata_value(value)

    assert sanitized is not None
    assert digest
    assert "worker stalled" not in sanitized.lower()

    parsed = json.loads(sanitized)
    assert parsed["component"] == "vpnkit"
    assert parsed["details"]["restartCount"] == 3
    assert parsed["status"].startswith(
        "Docker Desktop automatically restarted a background worker"
    )
    assert parsed["details"]["lastError"].startswith(
        "Docker Desktop automatically restarted a background worker"
    )


def test_worker_primary_metadata_is_redacted() -> None:
    """Primary worker error metadata should not leak the raw stall banner."""

    narrative = "vpnkit background sync worker stalled; restarting due to IO pressure"
    metadata: dict[str, str] = {
        "docker_worker_last_error": narrative,
        "docker_worker_last_error_original": narrative,
        "docker_worker_last_error_raw": narrative,
        "docker_worker_last_error_banner": narrative,
        "docker_worker_last_error_samples": "; ".join([narrative, "Recovered automatically"]),
        "docker_worker_last_error_original_samples": narrative,
        "docker_worker_last_error_raw_samples": narrative,
        "docker_worker_last_error_banner_samples": narrative,
    }

    bootstrap_env._redact_worker_banner_artifacts(metadata)

    expected = bootstrap_env._WORKER_STALLED_PRIMARY_NARRATIVE.lower()
    sanitized_keys = [
        "docker_worker_last_error",
        "docker_worker_last_error_original",
        "docker_worker_last_error_raw",
        "docker_worker_last_error_banner",
        "docker_worker_last_error_samples",
        "docker_worker_last_error_original_samples",
        "docker_worker_last_error_raw_samples",
        "docker_worker_last_error_banner_samples",
    ]

    for key in sanitized_keys:
        value = metadata.get(key, "")
        if not value:
            continue
        assert "worker stalled" not in value.lower()
        assert expected in value.lower()
        fingerprint_key = f"{key}_fingerprint"
        assert metadata.get(fingerprint_key), f"expected fingerprint for {key}"


def test_worker_banner_unknown_metadata_is_redacted() -> None:
    """Previously unseen metadata keys should be sanitized proactively."""

    metadata = {
        "customTelemetry": (
            "WARNING: worker stalled; restarting component=\"vpnkit\" "
            "restartCount=2"
        )
    }

    bootstrap_env._redact_worker_banner_artifacts(metadata)

    sanitized = metadata.get("customTelemetry", "")
    assert sanitized, "sanitized metadata should retain guidance"
    assert "worker stalled" not in sanitized.lower()

    fingerprint_key = "customTelemetry_fingerprint"
    fingerprint = metadata.get(fingerprint_key)
    assert fingerprint, "sanitized metadata should emit a fingerprint"
    assert fingerprint.startswith(bootstrap_env._WORKER_STALLED_SIGNATURE_PREFIX)


def test_enforce_worker_banner_sanitization_handles_period_separator() -> None:
    """Worker stall banners that use periods as separators should be rewritten."""

    metadata: dict[str, str] = {}
    warnings = ["WARNING: worker stalled. restarting component=\"vpnkit\" backoff=\"5s\""]

    harmonised = bootstrap_env._enforce_worker_banner_sanitization(warnings, metadata)

    assert harmonised, "expected sanitized worker warning"
    assert all("worker stalled" not in entry.lower() for entry in harmonised)
    assert metadata["docker_worker_health"] == "flapping"


def test_enforce_worker_banner_sanitization_handles_slash_separator() -> None:
    """Forward-slash separators should not leak raw worker stall banners."""

    metadata: dict[str, str] = {}
    warnings = [
        "WARNING worker stalled / restarting component=\"vpnkit\" restartCount=2",
    ]

    harmonised = bootstrap_env._enforce_worker_banner_sanitization(warnings, metadata)

    assert harmonised, "expected sanitized worker warning"
    assert all("worker stalled" not in entry.lower() for entry in harmonised)
    assert metadata["docker_worker_health"] == "flapping"


def test_enforce_worker_banner_sanitization_handles_question_separator() -> None:
    """Question mark separators from verbose Docker logs should be normalised."""

    metadata: dict[str, str] = {}
    warnings = [
        "WARNING worker stalled ? restarting component=\"vpnkit\" errCode=VPNKIT_VSOCK_TIMEOUT",
    ]

    harmonised = bootstrap_env._enforce_worker_banner_sanitization(warnings, metadata)

    assert harmonised, "expected sanitized worker warning"
    assert all("worker stalled" not in entry.lower() for entry in harmonised)
    assert metadata["docker_worker_health"] == "flapping"


def test_enforce_worker_banner_sanitization_handles_fullwidth_punctuation() -> None:
    """Full-width punctuation from localized Windows builds should be normalised."""

    metadata: dict[str, str] = {}
    warnings = [
        "WARNING：worker stalled； restarting component=\"vpnkit\" errCode=VPNKIT_VSOCK_TIMEOUT",
    ]

    harmonised = bootstrap_env._enforce_worker_banner_sanitization(warnings, metadata)

    assert harmonised, "expected sanitized worker warning"
    assert all("worker stalled" not in entry.lower() for entry in harmonised)
    assert metadata["docker_worker_health"] == "flapping"
    assert metadata["docker_worker_last_error_code"] == "VPNKIT_VSOCK_TIMEOUT"


def test_enforce_worker_banner_sanitization_strips_invisible_characters() -> None:
    """Zero-width characters from Windows consoles should not leak raw banners."""

    metadata: dict[str, str] = {}
    warnings = [
        "WARNING\u200b: worker\u202fstalled;\u2060 restarting component=\"vpnkit\" restartCount=2",
    ]

    harmonised = bootstrap_env._enforce_worker_banner_sanitization(warnings, metadata)

    assert harmonised, "expected sanitized worker warning"
    assert all("worker stalled" not in entry.lower() for entry in harmonised)
    assert metadata["docker_worker_health"] == "flapping"


def test_errcode_field_feeds_worker_error_guidance() -> None:
    """errCode metadata should be interpreted as an actionable worker error code."""

    message = (
        "WARNING: worker stalled; restarting component=\"vpnkit\" "
        "restartCount=6 errCode=WSL_KERNEL_OUTDATED "
        "lastError=\"WSL kernel outdated\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled; restarting" not in cleaned.lower()
    assert metadata["docker_worker_health"] == "flapping"


def test_worker_warning_with_become_synonym_is_normalised() -> None:
    """Worker stall banners using "has become" should collapse cleanly."""

    message = (
        "WARNING: worker has become stuck; restarting soon (component=\"vpnkit\")"
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert cleaned is not None
    assert "worker has become stuck" not in cleaned.lower()
    assert metadata["docker_worker_health"] == "flapping"


def test_worker_warning_with_remains_synonym_is_normalised() -> None:
    """Worker stall banners using "remains" should be collapsed into guidance."""

    message = "WARNING: worker remains stalled; restarting"

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert cleaned is not None
    assert "worker remains stalled" not in cleaned.lower()
    assert metadata["docker_worker_health"] == "flapping"
    assert metadata["docker_worker_last_error_code"] == "stalled_restart"
    assert metadata["docker_worker_health_severity"] == "info"
    remediation = metadata.get("docker_worker_health_remediation", "").lower()
    assert "monitor docker desktop" in remediation


def test_inline_errcode_banner_is_not_misclassified_as_context() -> None:
    """Inline errCode markers should not masquerade as worker context."""

    message = "WARNING: worker stalled; restarting (errCode=WSL_KERNEL_OUTDATED)"

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled; restarting" not in cleaned.lower()
    assert "affected component" not in cleaned.lower()
    assert "docker_worker_context" not in metadata
    assert metadata["docker_worker_last_error_code"] == "WSL_KERNEL_OUTDATED"

    telemetry = bootstrap_env.WorkerRestartTelemetry.from_metadata(metadata)
    assessment = bootstrap_env._classify_worker_flapping(telemetry, _windows_context())

    assert assessment.severity == "error"
    assert any("wsl kernel" in detail.lower() for detail in assessment.details)


def test_unknown_wsl_errcode_generates_generic_guidance() -> None:
    """Unrecognised WSL errCodes should still surface actionable guidance."""

    message = (
        "WARNING: worker stalled; restarting component=\"vpnkit\" "
        "restartCount=3 errCode=WSL_VM_STOPPED "
        "lastError=\"WSL VM stopped unexpectedly\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled; restarting" not in cleaned.lower()
    assert metadata["docker_worker_last_error_code"] == "WSL_VM_STOPPED"

    telemetry = bootstrap_env.WorkerRestartTelemetry.from_metadata(metadata)
    assessment = bootstrap_env._classify_worker_flapping(telemetry, _windows_context())

    assert assessment.severity == "error"
    assert any("wsl" in reason.lower() for reason in assessment.reasons)
    assert any("wsl" in detail.lower() for detail in assessment.details)
    assert any("wsl --status" in step.lower() for step in assessment.remediation)

    guidance_key = "docker_worker_last_error_guidance_wsl_vm_stopped"
    assert guidance_key in assessment.metadata
    assert "docker desktop" in assessment.metadata[guidance_key].lower()


def test_virtualization_errcode_generic_guidance() -> None:
    """Virtualization-centric errCodes should yield virtualization remediation."""

    message = (
        "WARNING: worker stalled; restarting component=\"vm\" "
        "restartCount=5 errCode=HCS_E_ACCESS_DENIED "
        "lastError=\"Access denied while starting VM\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled; restarting" not in cleaned.lower()
    assert metadata["docker_worker_last_error_code"] == "HCS_E_ACCESS_DENIED"

    telemetry = bootstrap_env.WorkerRestartTelemetry.from_metadata(metadata)
    assessment = bootstrap_env._classify_worker_flapping(telemetry, _windows_context())

    assert assessment.severity == "error"
    assert any("virtualization" in reason.lower() for reason in assessment.reasons)
    assert any(
        "hyper-v" in detail.lower() or "host compute" in detail.lower()
        for detail in assessment.details
    )
    assert any(
        "hyper-v" in step.lower() or "virtualization" in step.lower()
        for step in assessment.remediation
    )

    guidance_key = "docker_worker_last_error_guidance_hcs_e_access_denied"
    assert guidance_key in assessment.metadata
    summary = assessment.metadata[guidance_key].lower()
    assert "virtual" in summary or "hyper-v" in summary


def test_vpnkit_network_jitter_errcode_surfaces_actionable_guidance() -> None:
    """Newer vpnkit jitter errCodes should never leak raw stall banners."""

    message = (
        "WARNING: worker stalled; restarting component=\"vpnkit\" "
        "restartCount=4 errCode=VPNKIT_BACKGROUND_SYNC_NETWORK_JITTER "
        "lastError=\"worker stalled; restarting due to network jitter\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert cleaned is not None
    assert "worker stalled; restarting" not in cleaned.lower()
    assert metadata["docker_worker_last_error_code"] == "VPNKIT_BACKGROUND_SYNC_NETWORK_JITTER"
    assert metadata["docker_worker_health"] == "flapping"

    guidance_key = "docker_worker_last_error_guidance_vpnkit_background_sync_network_jitter"
    assert guidance_key in metadata
    assert "network" in metadata[guidance_key].lower()

    telemetry = bootstrap_env.WorkerRestartTelemetry.from_metadata(metadata)
    assessment = bootstrap_env._classify_worker_flapping(telemetry, _windows_context())

    assert assessment.severity in {"warning", "error"}
    assert any("network" in reason.lower() for reason in assessment.reasons)
    assert any("vpnkit" in detail.lower() for detail in assessment.details)
    assert any("restart docker desktop" in step.lower() for step in assessment.remediation)


def test_vpnkit_io_throttled_errcode_includes_disk_guidance() -> None:
    """The IO throttled errCode should surface storage remediation steps."""

    message = (
        "WARNING: worker stalled; restarting component=\"vpnkit\" "
        "restartCount=3 errCode=VPNKIT_BACKGROUND_SYNC_IO_THROTTLED "
        "lastError=\"worker stalled; restarting while host IO was throttled\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert cleaned is not None
    assert "worker stalled; restarting" not in cleaned.lower()
    assert metadata["docker_worker_last_error_code"] == "VPNKIT_BACKGROUND_SYNC_IO_THROTTLED"

    guidance_key = "docker_worker_last_error_guidance_vpnkit_background_sync_io_throttled"
    assert guidance_key in metadata
    assert "io" in metadata[guidance_key].lower() or "throttling" in metadata[guidance_key].lower()

    telemetry = bootstrap_env.WorkerRestartTelemetry.from_metadata(metadata)
    assessment = bootstrap_env._classify_worker_flapping(telemetry, _windows_context())

    assert any("i/o" in reason.lower() or "io" in reason.lower() for reason in assessment.reasons)
    assert any("disk" in detail.lower() or "storage" in detail.lower() for detail in assessment.details)
    assert any("restart docker desktop" in step.lower() for step in assessment.remediation)


def test_wsl_kernel_missing_guidance() -> None:
    """Missing WSL kernel codes should surface installation guidance."""

    message = (
        "WARNING: worker stalled; restarting component=\"vm\" "
        "restartCount=2 errCode=WSL_KERNEL_MISSING "
        "lastError=\"WSL kernel not installed\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled; restarting" not in cleaned.lower()
    assert metadata["docker_worker_last_error_code"] == "WSL_KERNEL_MISSING"

    telemetry = bootstrap_env.WorkerRestartTelemetry.from_metadata(metadata)
    assessment = bootstrap_env._classify_worker_flapping(telemetry, _windows_context())

    assert any("kernel" in detail.lower() for detail in assessment.details)
    assert any("wsl" in step.lower() for step in assessment.remediation)
    guidance_key = "docker_worker_last_error_guidance_wsl_kernel_missing"
    assert guidance_key in assessment.metadata
    assert "wsl" in assessment.metadata[guidance_key].lower()


def test_worker_unresponsive_banner_is_normalized() -> None:
    """Unresponsive worker banners should be rewritten into guidance."""

    message = (
        "WARNING: worker unresponsive; restarting component=\"vpnkit\" "
        "restartCount=3 errCode=VPNKIT_UNRESPONSIVE "
        "lastError=\"vpnkit background sync worker unresponsive; restarting due to IO pressure\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert cleaned is not None
    lowered = cleaned.lower()
    assert "worker unresponsive; restarting" not in lowered
    assert "worker stalled; restarting" not in lowered
    assert (
        "automatically restarted" in lowered
        or "repeatedly restarting" in lowered
        or "restart" in lowered
    )

    assert metadata["docker_worker_last_error_code"] == "VPNKIT_UNRESPONSIVE"
    assert metadata["docker_worker_health"] == "flapping"
    assert metadata["docker_worker_health_severity"] in {"warning", "error", "info"}


def test_worker_timeout_banner_is_normalized() -> None:
    """Timeout-oriented worker banners should be harmonised."""

    message = (
        "WARN[0045] moby/buildkit: worker timed out; restarting component=\"vpnkit\" "
        "restartCount=2 errCode=VPNKIT_BACKGROUND_SYNC_TIMEOUT "
        "lastError=\"worker timed out; restarting after 45s\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert cleaned is not None
    lowered = cleaned.lower()
    assert "worker timed out; restarting" not in lowered
    assert "worker stalled; restarting" not in lowered
    assert "automatically restarted" in lowered

    assert metadata["docker_worker_last_error_code"] == "VPNKIT_BACKGROUND_SYNC_TIMEOUT"
    assert metadata["docker_worker_health"] == "flapping"


def test_wsl2_prefixed_errcode_surfaces_wsl_guidance() -> None:
    """WSL2-prefixed error codes should be treated as WSL guidance triggers."""

    message = (
        "WARNING: worker stalled; restarting component=\"vpnkit\" "
        "restartCount=4 backoff=120s errCode=WSL2_KERNEL_OUTDATED "
        "lastError=\"WSL2 kernel requires an update\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled; restarting" not in cleaned.lower()
    assert metadata["docker_worker_last_error_code"] == "WSL2_KERNEL_OUTDATED"

    telemetry = bootstrap_env.WorkerRestartTelemetry.from_metadata(metadata)
    assessment = bootstrap_env._classify_worker_flapping(telemetry, _windows_context())

    assert assessment.severity == "error"
    guidance_key = "docker_worker_last_error_guidance_wsl2_kernel_outdated"
    assert guidance_key in assessment.metadata
    assert "wsl 2" in assessment.metadata[guidance_key].lower()
    assert any("wsl 2" in detail.lower() or "windows subsystem for linux" in detail.lower() for detail in assessment.details)
    assert any("wsl --update" in step.lower() for step in assessment.remediation)


def test_vpnkit_errcode_guidance_addresses_network_stalls() -> None:
    """vpnkit-specific error codes should surface targeted networking guidance."""

    message = (
        "WARNING[0032]: worker stalled; restarting component=\"vpnkit\" "
        "restartCount=4 backoff=30s errCode=VPNKIT_HEALTHCHECK_FAILED "
        "lastError=\"vpnkit health check timed out\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled; restarting" not in cleaned.lower()
    assert metadata["docker_worker_health"] == "flapping"
    assert metadata["docker_worker_last_error_code"] == "VPNKIT_HEALTHCHECK_FAILED"

    telemetry = bootstrap_env.WorkerRestartTelemetry.from_metadata(metadata)
    assessment = bootstrap_env._classify_worker_flapping(telemetry, _windows_context())

    assert any("vpnkit" in reason.lower() for reason in assessment.reasons)
    assert any("vpnkit" in detail.lower() for detail in assessment.details)
    assert any(
        "vpnkit" in step.lower() or "network" in step.lower()
        for step in assessment.remediation
    )
    guidance_key = "docker_worker_last_error_guidance_vpnkit_healthcheck_failed"
    assert assessment.metadata[guidance_key].startswith("Restart Docker Desktop")


def test_vpnkit_background_sync_guidance() -> None:
    """Background sync stalls should surface vpnkit-specific remediation."""

    message = (
        "WARNING: worker stalled; restarting component=\"vpnkit\" "
        "restartCount=4 errCode=VPNKIT_BACKGROUND_SYNC_STALLED "
        "lastError=\"vpnkit background sync worker stalled; restarting\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled; restarting" not in cleaned.lower()
    assert metadata["docker_worker_last_error_code"] == "VPNKIT_BACKGROUND_SYNC_STALLED"

    telemetry = bootstrap_env.WorkerRestartTelemetry.from_metadata(metadata)
    assessment = bootstrap_env._classify_worker_flapping(telemetry, _windows_context())

    assert any("background sync" in detail.lower() for detail in assessment.details)
    assert any("vpnkit" in step.lower() for step in assessment.remediation)

    guidance_key = "docker_worker_last_error_guidance_vpnkit_background_sync_stalled"
    assert guidance_key in assessment.metadata
    assert "background sync" in assessment.metadata[guidance_key].lower()


def test_structured_warning_combines_status_and_detail_fields() -> None:
    """Structured worker payloads split across fields should be normalised."""

    payload = {
        "status": "worker stalled",
        "statusDetailText": (
            "restarting component=\"vpnkit\" restartCount=4 backoff=\"PT45S\" "
            "errCode=VPNKIT_BACKGROUND_SYNC_NETWORK_SATURATION"
        ),
        "component": "vpnkit",
        "restartCount": "4",
        "backoff": "PT45S",
        "errCode": "VPNKIT_BACKGROUND_SYNC_NETWORK_SATURATION",
    }

    rendered = bootstrap_env._stringify_structured_warning(payload, ("warnings", "0"))

    assert rendered is not None
    assert "worker stalled; restarting" not in rendered.lower()

    cleaned, metadata = bootstrap_env._normalise_docker_warning(rendered)

    assert cleaned is not None
    assert "worker stalled; restarting" not in cleaned.lower()
    assert metadata["docker_worker_last_error_code"] == "VPNKIT_BACKGROUND_SYNC_NETWORK_SATURATION"

    guidance_key = (
        "docker_worker_last_error_guidance_vpnkit_background_sync_network_saturation"
    )
    assert guidance_key in metadata
    assert "network" in metadata[guidance_key].lower()


def test_worker_flapping_metadata_enrichment_handles_structured_payload() -> None:
    """Structured worker telemetry should be extracted from the warning payload."""

    message = (
        "WARNING[0000]: worker stalled; restarting component=\"vpnkit\" "
        "restartCount=7 backoff=\"PT45S\" last_error='context deadline exceeded'"
        " last_restart=2024-05-01T10:15:00Z"
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "vpnkit" in cleaned
    assert metadata["docker_worker_health"] == "flapping"
    assert metadata["docker_worker_context"] == "vpnkit"
    assert metadata["docker_worker_restart_count"] == "7"
    assert metadata["docker_worker_backoff"] == "45s"
    assert metadata["docker_worker_last_error"].lower().startswith("context deadline")
    assert metadata["docker_worker_last_restart"] == "2024-05-01T10:15:00Z"


def test_vpnkit_hns_guidance_mentions_hns() -> None:
    """vpnkit HNS error codes should highlight Host Network Service remediation."""

    message = (
        "WARN[0045] moby/buildkit: worker stalled; restarting component=\"vpnkit\" "
        "restartCount=6 errCode=VPNKIT_HNS_UNAVAILABLE backoff=45s "
        "lastError=\"vpnkit lost connectivity with HNS\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled; restarting" not in cleaned.lower()
    assert metadata["docker_worker_last_error_code"] == "VPNKIT_HNS_UNAVAILABLE"

    telemetry = bootstrap_env.WorkerRestartTelemetry.from_metadata(metadata)
    assessment = bootstrap_env._classify_worker_flapping(telemetry, _windows_context())

    assert any("hns" in detail.lower() for detail in assessment.details)
    assert any("hns" in step.lower() for step in assessment.remediation)

    guidance_key = "docker_worker_last_error_guidance_vpnkit_hns_unavailable"
    assert guidance_key in assessment.metadata
    assert "hns" in assessment.metadata[guidance_key].lower()


def test_worker_warning_infers_vsock_timeout_guidance() -> None:
    """vsock timeouts without explicit codes should map to actionable guidance."""

    message = (
        "WARN[0039] moby/buildkit: worker stalled; restarting component=\"vpnkit\" "
        "restartCount=4 lastError=\"vsock handshake timed out after 30s\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled" not in cleaned.lower()
    assert metadata["docker_worker_last_error_code"] == "VPNKIT_VSOCK_TIMEOUT"

    telemetry = bootstrap_env.WorkerRestartTelemetry.from_metadata(metadata)
    assessment = bootstrap_env._classify_worker_flapping(telemetry, _windows_context())

    assert any("vsock" in detail.lower() for detail in assessment.details)

    guidance_key = "docker_worker_last_error_guidance_vpnkit_vsock_timeout"
    assert guidance_key in assessment.metadata
    assert "vsock" in assessment.metadata[guidance_key].lower()


def test_worker_warning_explicit_vsock_guidance() -> None:
    """Explicit vsock error codes should surface dedicated remediation steps."""

    message = (
        "WARN[0032] moby/buildkit: worker stalled; restarting component=\"vpnkit\" "
        "restartCount=5 errCode=VPNKIT_VSOCK_UNRESPONSIVE "
        "lastError=\"vsock connection to com.docker.backend refused\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled" not in cleaned.lower()
    assert metadata["docker_worker_last_error_code"] == "VPNKIT_VSOCK_UNRESPONSIVE"

    telemetry = bootstrap_env.WorkerRestartTelemetry.from_metadata(metadata)
    assessment = bootstrap_env._classify_worker_flapping(telemetry, _windows_context())

    assert any("vsock" in detail.lower() for detail in assessment.details)

    guidance_key = "docker_worker_last_error_guidance_vpnkit_vsock_unresponsive"
    assert guidance_key in assessment.metadata
    assert "vsock" in assessment.metadata[guidance_key].lower()


def test_worker_warning_vpnkit_io_pressure_guidance() -> None:
    """Host I/O pressure codes should translate into actionable vpnkit guidance."""

    message = (
        "WARNING: worker stalled; restarting component=\"vpnkit\" "
        "restartCount=6 backoff=PT90S errCode=VPNKIT_BACKGROUND_SYNC_IO_PRESSURE "
        "lastError=\"vpnkit background sync worker stalled; restarting due to IO pressure\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled; restarting" not in cleaned.lower()
    assert metadata["docker_worker_last_error_code"] == "VPNKIT_BACKGROUND_SYNC_IO_PRESSURE"

    telemetry = bootstrap_env.WorkerRestartTelemetry.from_metadata(metadata)
    assessment = bootstrap_env._classify_worker_flapping(telemetry, _windows_context())

    guidance_key = "docker_worker_last_error_guidance_vpnkit_background_sync_io_pressure"
    assert guidance_key in assessment.metadata
    assert "pressure" in assessment.metadata[guidance_key].lower()
    assert any("pressure" in segment.lower() for segment in assessment.details)


def test_worker_warning_vpnkit_disk_pressure_guidance() -> None:
    """Disk pressure codes should surface vpnkit-specific remediation steps."""

    message = (
        "WARNING: worker stalled; restarting component=\"vpnkit\" "
        "restartCount=4 backoff=PT75S errCode=VPNKIT_BACKGROUND_SYNC_DISK_PRESSURE "
        "lastError=\"vpnkit background sync worker stalled; restarting due to disk pressure\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled; restarting" not in cleaned.lower()
    assert metadata["docker_worker_last_error_code"] == "VPNKIT_BACKGROUND_SYNC_DISK_PRESSURE"

    telemetry = bootstrap_env.WorkerRestartTelemetry.from_metadata(metadata)
    assessment = bootstrap_env._classify_worker_flapping(telemetry, _windows_context())

    guidance_key = "docker_worker_last_error_guidance_vpnkit_background_sync_disk_pressure"
    assert guidance_key in assessment.metadata
    assert "disk" in assessment.metadata[guidance_key].lower()
    assert any("disk" in segment.lower() or "storage" in segment.lower() for segment in assessment.details)


def test_generic_pressure_error_guidance() -> None:
    """Unrecognised pressure-oriented error codes should still provide guidance."""

    directive = bootstrap_env._derive_generic_error_code_guidance("host_disk_pressure_alert")

    assert directive is not None
    assert "pressure" in directive.reason.lower()
    assert "pressure" in (directive.detail or "").lower()
    assert any("pressure" in hint.lower() for hint in directive.remediation)


def test_worker_warning_parenthetical_metadata_is_normalised() -> None:
    """Parenthetical worker telemetry should not leak closing punctuation."""

    message = (
        "WARNING: worker stalled; restarting in 5s (context=background-sync) "
        "(lastError=EOF)"
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "background-sync" in cleaned
    assert metadata["docker_worker_context"] == "background-sync"
    assert metadata["docker_worker_backoff"] == "5s"
    assert metadata["docker_worker_last_error"] == "EOF"
    assert metadata["docker_worker_last_error_raw"] == "EOF)"


def test_worker_warning_next_restart_metadata_is_normalised() -> None:
    """Derived ``next restart`` fields should collapse into canonical backoff hints."""

    message = (
        "WARNING: worker stalled; restarting component=\"vpnkit\" "
        "restart_delay_ms=45000 next_retry_in=30s retry_after_seconds=90 "
        "next_restart_seconds=120"
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert cleaned
    assert "worker stalled; restarting" not in cleaned.lower()
    assert metadata["docker_worker_context"] == "vpnkit"
    assert metadata["docker_worker_backoff"] == "45s"

    backoff_options = metadata.get("docker_worker_backoff_options", "")
    if backoff_options:
        normalized_options = {option.strip() for option in backoff_options.split(",")}
        assert "30s" in normalized_options or "about 30s" in normalized_options
        assert any(
            candidate in normalized_options
            for candidate in {"90s", "1m30s"}
        )


def test_normalize_warning_collection_aggregates_last_healthy_markers() -> None:
    """Aggregated worker metadata should retain last healthy timestamps."""

    warnings = [
        "WARNING: worker stalled; restarting component=\"vpnkit\" "
        "lastHealthy=2024-10-03T17:40:00Z restartCount=2",
        "WARNING: worker stalled; restarting component=\"vpnkit\" "
        "lastHealthy=2024-10-03T17:45:00Z restartCount=3",
    ]

    normalized, metadata = bootstrap_env._normalize_warning_collection(warnings)

    assert normalized
    assert metadata["docker_worker_last_healthy"] == "2024-10-03T17:45:00Z"
    samples = metadata.get("docker_worker_last_healthy_samples", "")
    assert "2024-10-03T17:40:00Z" in samples


def test_normalize_warning_collection_parses_iso_backoff_metadata() -> None:
    """ISO-8601 backoff tokens should be normalised to human readable durations."""

    warnings = [
        "WARNING: worker stalled; restarting backoff=\"PT45S\" restartCount=2"
    ]

    normalized, metadata = bootstrap_env._normalize_warning_collection(warnings)

    assert normalized and "worker stalled" not in normalized[0].lower()
    assert metadata["docker_worker_backoff"] == "45s"
    assert metadata["docker_worker_restart_count"] == "2"


def test_normalize_warnings_interprets_mapping_components() -> None:
    """Structured warning mappings should propagate worker context metadata."""

    payload = {
        "vpnkit": {
            "status": "worker stalled; restarting",
            "restartCount": 4,
            "backoffSeconds": 45,
            "lastError": "context deadline exceeded",
        }
    }

    warnings, metadata = bootstrap_env._normalize_docker_warnings(payload)

    assert len(warnings) == 1
    message = warnings[0]
    assert "worker stalled; restarting" not in message.lower()
    assert "vpnkit" in message.lower()
    assert metadata["docker_worker_context"] == "vpnkit"
    assert metadata["docker_worker_restart_count"] == "4"
    assert metadata["docker_worker_backoff"].startswith("45") or metadata[
        "docker_worker_backoff"
    ].startswith("1m")
    assert metadata["docker_worker_last_error"].lower().startswith("context deadline")


def test_normalize_warnings_handles_worker_collections() -> None:
    """Nested worker collections should yield independent diagnostics."""

    payload = {
        "workers": [
            {
                "name": "vpnkit",
                "status": "worker stalled; restarting",
                "restartAttempts": 3,
                "backoffIntervalMs": 30000,
            },
            {"name": "buildkit", "status": "running"},
        ]
    }

    warnings, metadata = bootstrap_env._normalize_docker_warnings(payload)

    assert len(warnings) == 1
    message = warnings[0]
    assert "vpnkit" in message.lower()
    assert metadata["docker_worker_context"] == "vpnkit"
    assert metadata["docker_worker_restart_count"] == "3"
    backoff = metadata.get("docker_worker_backoff", "")
    assert backoff.endswith("s")


def test_worker_error_fallback_sanitizes_worker_stalls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback worker error handling should still purge stall banners."""

    monkeypatch.setattr(
        bootstrap_env,
        "_contains_worker_stall_signal",
        lambda message: False,
    )

    narrative, detail, metadata = bootstrap_env._normalise_worker_error_message(
        "WARNING: worker stalled; restarting"
    )

    assert narrative == bootstrap_env._WORKER_STALLED_PRIMARY_NARRATIVE
    assert "worker stalled; restarting" not in detail.lower()
    assert all(
        "worker stalled; restarting" not in str(value).lower()
        for key, value in metadata.items()
        if "banner_raw" not in key and "banner_preserved" not in key
    )


def test_structured_warning_infers_context_from_display_name() -> None:
    """Display name and status message fields should map to actionable guidance."""

    payload = {
        "diagnostics": {
            "components": [
                {
                    "componentDisplayName": "VPNKit Service",
                    "statusMessage": "worker stalled; restarting (errCode=VPNKIT_UNRESPONSIVE)",
                    "err_code": "VPNKIT_UNRESPONSIVE",
                    "lastErrorMessage": "vpnkit stopped responding to health checks",
                }
            ]
        }
    }

    warnings, metadata = bootstrap_env._normalize_docker_warnings(payload)

    assert warnings, "expected a synthesized warning"
    message = warnings[0]
    assert "worker stalled; restarting" not in message.lower()
    assert "vpnkit" in message.lower()
    assert "vpnkit" in metadata["docker_worker_context"].lower()
    assert metadata["docker_worker_last_error_code"] == "VPNKIT_UNRESPONSIVE"
    assert metadata["docker_worker_last_error"].lower().startswith(
        "vpnkit stopped responding"
    )


def test_component_statuses_payloads_are_interpreted() -> None:
    """Docker Desktop component status feeds should yield cleaned warnings."""

    payload = {
        "componentStatuses": [
            {
                "displayName": "VPNKit Background Sync",
                "statusShortMessage": "worker stalled; restarting (errCode=VPNKIT_HNS_UNAVAILABLE)",
                "shortErrorMessage": "worker stalled; restarting",
                "errCode": "VPNKIT_HNS_UNAVAILABLE",
                "restartCount": 6,
                "backoffSeconds": 75,
                "lastError": "vpnkit lost connectivity with the Windows Host Network Service",
            }
        ]
    }

    warnings, metadata = bootstrap_env._normalize_docker_warnings(payload)

    assert warnings, "expected synthesized warning output"
    message = warnings[0]
    assert "worker stalled; restarting" not in message.lower()
    assert "vpnkit" in message.lower()
    assert metadata["docker_worker_context"].lower().startswith("vpnkit")
    assert metadata["docker_worker_restart_count"] == "6"
    assert metadata["docker_worker_backoff"].endswith("s")
    assert metadata["docker_worker_last_error_code"] == "VPNKIT_HNS_UNAVAILABLE"
    assert "lost connectivity" in metadata["docker_worker_last_error"].lower()


def test_normalize_warnings_handles_worker_stall_detected_banner() -> None:
    """Structured payloads using ``worker stall`` phrasing are canonicalised."""

    payload = {
        "vpnkit": {
            "status": "worker stall detected; restarting",
            "restartCount": 5,
            "backoffSeconds": 10,
        }
    }

    warnings, metadata = bootstrap_env._normalize_docker_warnings(payload)

    assert len(warnings) == 1
    banner = warnings[0].lower()
    assert "worker stall" not in banner
    assert "vpnkit" in banner
    assert metadata["docker_worker_health"] == "flapping"
    assert metadata["docker_worker_context"] == "vpnkit"
    assert metadata["docker_worker_restart_count"] == "5"
    assert metadata["docker_worker_backoff"] == "10s"


def test_structured_warning_component_name_context() -> None:
    """CamelCase component keys should still drive context detection."""

    payload = {
        "level": "warning",
        "message": "worker stalled; restarting",
        "componentName": "vpnkitCore",
        "restartCount": 5,
        "backoffSeconds": 45,
        "lastError": "vpnkit health-check timed out",
    }

    warnings, metadata = bootstrap_env._normalize_docker_warnings(payload)

    assert warnings
    assert metadata["docker_worker_context"] == "vpnkitCore"
    assert any("vpnkitcore" in warning.lower() for warning in warnings)
    assert "worker stalled; restarting" not in warnings[0].lower()


def test_structured_warning_component_identifier_context() -> None:
    """Structured payloads using identifier fields should expose context."""

    payload = {
        "message": "worker stalled; restarting",
        "componentIdentifier": "vpnkit-background-sync",
        "restartCount": 3,
        "backoffSeconds": 30,
    }

    warnings, metadata = bootstrap_env._normalize_docker_warnings(payload)

    assert warnings
    assert metadata["docker_worker_context"] == "vpnkit-background-sync"
    assert metadata["docker_worker_restart_count"] == "3"
    assert metadata["docker_worker_backoff"] == "30s"
    assert any("vpnkit-background-sync" in warning.lower() for warning in warnings)
    assert "worker stalled; restarting" not in warnings[0].lower()


def test_structured_warning_prefers_status_message_over_status() -> None:
    """Status message fields should outrank generic status strings."""

    payload = {
        "components": [
            {
                "name": "vpnkit",
                "status": {
                    "status": "degraded",
                    "statusMessage": "worker stalled; restarting",
                    "restartCount": 4,
                    "backoff": "PT30S",
                    "lastError": {
                        "code": "VPNKIT_BACKGROUND_SYNC_STALLED",
                        "message": "vpnkit background sync worker stalled; restarting",
                    },
                },
            }
        ]
    }

    warnings, metadata = bootstrap_env._normalize_docker_warnings(payload)

    assert warnings
    assert any(
        "docker desktop worker processes are repeatedly restarting" in warning.lower()
        for warning in warnings
    )
    assert all(
        "worker stalled; restarting" not in warning.lower()
        for warning in warnings
    )
    assert metadata["docker_worker_context"] == "vpnkit"
    assert metadata["docker_worker_restart_count"] == "4"
    assert metadata["docker_worker_backoff"] == "30s"
    assert metadata["docker_worker_last_error_code"] == "VPNKIT_BACKGROUND_SYNC_STALLED"

    rewritten, aggregated = bootstrap_env._scrub_residual_worker_warnings(warnings)

    assert rewritten
    assert "worker stalled; restarting" not in rewritten[0].lower()
    assert "vpnkit" in rewritten[0].lower()
    assert metadata["docker_worker_health"] == "flapping"
    assert metadata["docker_worker_context"] == "vpnkit"
    assert metadata["docker_worker_restart_count"] == "4"
    assert metadata["docker_worker_backoff"] == "30s"
    assert metadata["docker_worker_last_error_code"] == "VPNKIT_BACKGROUND_SYNC_STALLED"
    assert aggregated == {}


def test_structured_warning_friendly_name_and_long_message() -> None:
    """Docker Desktop friendly names and long-form messages should drive guidance."""

    payload = {
        "diagnostics": {
            "components": [
                {
                    "componentFriendlyName": "VPNKit Background Sync",
                    "componentName": "vpnkitSync",
                    "statusShortMessage": "worker stalled; restarting",
                    "statusLongMessage": (
                        "worker stalled; restarting due to errCode=VPNKIT_BACKGROUND_SYNC_STALLED"
                    ),
                    "metadata": {
                        "restartAttemptsTotal": 5,
                        "backoffIntervalSeconds": 75,
                        "lastErrorMessage": (
                            "vpnkit background sync worker stalled; restarting because of IO pressure"
                        ),
                        "lastErrorCode": "VPNKIT_BACKGROUND_SYNC_STALLED",
                    },
                }
            ]
        }
    }

    warnings, metadata = bootstrap_env._normalize_docker_warnings(payload)

    assert warnings
    banner = " ".join(warnings).lower()
    assert "worker stalled; restarting" not in banner
    assert "vpnkit" in banner

    context = metadata.get("docker_worker_context", "").lower()
    assert "vpnkit" in context
    assert metadata.get("docker_worker_restart_count") == "5"
    assert metadata.get("docker_worker_backoff") in {"75s", "1m15s"}
    assert metadata.get("docker_worker_last_error_code") == "VPNKIT_BACKGROUND_SYNC_STALLED"
    last_error = metadata.get("docker_worker_last_error", "").lower()
    assert last_error
    assert "worker stalled" not in last_error

def test_structured_notifications_with_text_field() -> None:
    """Docker Desktop notifications using ``text`` should become actionable."""

    payload = {
        "diagnostics": {
            "notifications": [
                {
                    "source": "VPNKit Background Sync",
                    "title": "Background worker restart",
                    "text": "WARNING: worker stalled; restarting component=\"vpnkit\"",
                    "metadata": {
                        "restartCount": 4,
                        "backoff": "PT45S",
                        "lastErrorCode": "VPNKIT_SYNC_TIMEOUT",
                        "lastErrorMessage": "vpnkit background sync worker stalled; restarting",
                    },
                }
            ]
        }
    }

    warnings, metadata = bootstrap_env._normalize_docker_warnings(payload)

    assert warnings
    banner = " ".join(warnings).lower()
    assert "worker stalled; restarting" not in banner
    assert "vpnkit" in metadata.get("docker_worker_context", "").lower()
    assert metadata.get("docker_worker_restart_count") == "4"
    assert metadata.get("docker_worker_backoff") == "45s"
    assert metadata.get("docker_worker_last_error_code") == "VPNKIT_SYNC_TIMEOUT"
    assert "worker stalled" not in metadata.get("docker_worker_last_error", "").lower()
    assert "vpnkit background sync" in metadata.get(
        "docker_worker_last_error_banner_preserved", ""
    ).lower()
    assert "worker stalled; restarting" not in metadata.get(
        "docker_worker_last_error_banner_raw", ""
    ).lower()


def test_structured_notifications_with_localized_messages() -> None:
    """Localized Docker Desktop payloads should sanitise worker stall banners."""

    payload = {
        "diagnostics": {
            "notifications": [
                {
                    "source": "VPNKit Background Sync",
                    "localizedMessage": (
                        "WARNING: worker stalled; restarting component=\"vpnkit\" restartCount=3"
                    ),
                    "localizedShortMessage": "worker stalled; restarting component=\"vpnkit\"",
                    "localizedStatusMessage": (
                        "worker stalled; restarting (errCode=VPNKIT_UNRESPONSIVE)"
                    ),
                    "metadata": {
                        "restartCount": 3,
                        "backoffSeconds": 45,
                        "context": "vpnkit",
                        "lastErrorCode": "VPNKIT_UNRESPONSIVE",
                        "lastErrorMessage": (
                            "vpnkit background sync worker stalled; restarting due to IO pressure"
                        ),
                    },
                }
            ]
        }
    }

    warnings, metadata = bootstrap_env._normalize_docker_warnings(payload)

    assert warnings
    banner = " ".join(warnings).lower()
    assert "worker stalled; restarting" not in banner
    assert "docker desktop reported" in banner

    context = metadata.get("docker_worker_context", "").lower()
    assert "vpnkit" in context
    assert metadata.get("docker_worker_restart_count") == "3"
    assert metadata.get("docker_worker_backoff") == "45s"
    assert metadata.get("docker_worker_last_error_code") == "VPNKIT_UNRESPONSIVE"
    assert metadata.get("docker_worker_health") == "flapping"


def test_worker_stall_json_error_payload_enriched() -> None:
    """JSON ``lastError`` payloads should surface actionable metadata."""

    payload = json.dumps(
        {
            "code": "VPNKIT_UNRESPONSIVE",
            "message": "worker stalled; restarting",
            "detail": "vpnkit lost connectivity to the network stack",
        }
    )

    message = (
        "WARNING: worker stalled; restarting component=\"vpnkit\" "
        "restartCount=3 backoff=45s "
        f"lastError={payload}"
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert cleaned
    assert "worker stalled; restarting" not in cleaned.lower()
    assert metadata["docker_worker_last_error_code"] == "VPNKIT_UNRESPONSIVE"
    assert (
        metadata["docker_worker_last_error_structured_message"]
        == "vpnkit lost connectivity to the network stack"
    )
    assert metadata["docker_worker_backoff"] == "45s"


def test_worker_stall_json_payload_sanitises_primary_error_fields() -> None:
    """Structured worker payloads should override truncated log fragments."""

    message = (
        "WARNING: worker stalled; restarting component=\"vpnkit\" "
        "lastError={\"message\":\"worker stalled; restarting\"}"
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert cleaned
    primary_error = metadata["docker_worker_last_error"]
    assert primary_error == bootstrap_env._WORKER_STALLED_PRIMARY_NARRATIVE
    assert metadata["docker_worker_last_error_banner"] == primary_error
    assert metadata["docker_worker_last_error_raw"] == primary_error
    assert metadata["docker_worker_last_error_original"] == primary_error
    assert "\"message\"" not in metadata["docker_worker_health_summary"].lower()
    assert "worker stalled; restarting" not in metadata["docker_worker_health_summary"].lower()


def test_structured_worker_message_sanitised_when_only_banner() -> None:
    """Structured ``lastError`` payloads should not leak the raw banner."""

    payload = json.dumps(
        {
            "code": "VPNKIT_BACKGROUND_SYNC_STALLED",
            "message": "worker stalled; restarting",
            "detail": "worker stalled; restarting due to IO pressure",
        }
    )

    message = (
        "WARN[0042] moby/buildkit: worker stalled; restarting "
        f"component=\"vpnkit\" lastError={payload}"
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert cleaned
    structured = metadata["docker_worker_last_error_structured_message"]
    assert structured
    assert structured == (
        "Docker Desktop automatically restarted a background worker after it stalled due to IO pressure"
    )
    assert "worker stalled" not in structured.lower()
    fingerprint = metadata.get(
        "docker_worker_last_error_structured_message_fingerprint"
    )
    assert fingerprint
    assert fingerprint.startswith(bootstrap_env._WORKER_STALLED_SIGNATURE_PREFIX)
    raw_value = metadata.get("docker_worker_last_error_structured_message_raw")
    assert raw_value
    assert "worker stalled" not in raw_value.lower()


def test_finalize_metadata_sanitises_nested_json_payloads() -> None:
    """Nested JSON metadata values should be rewritten without leaking stall banners."""

    metadata: dict[str, str] = {
        "docker_worker_json_snapshot": json.dumps(
            {
                "status": "worker stalled; restarting component=\"vpnkit\"",
                "history": [
                    "worker stalled; restarting due to IO pressure",
                    {
                        "message": "worker stalled; restarting",
                        "detail": "worker stalled; restarting (errCode=VPNKIT_VSOCK_TIMEOUT)",
                    },
                ],
            }
        )
    }

    bootstrap_env._finalize_worker_banner_metadata(metadata)

    snapshot = metadata["docker_worker_json_snapshot"]
    parsed = json.loads(snapshot)

    serialised = json.dumps(parsed)
    assert "worker stalled; restarting" not in serialised.lower()
    assert "Docker Desktop automatically restarted" in serialised

    fingerprint_key = "docker_worker_json_snapshot_fingerprint"
    assert fingerprint_key in metadata

def test_error_code_guidance_covers_hns_service_unavailable() -> None:
    """Explicit HNS error codes should surface actionable remediation steps."""

    details: list[str] = []
    remediation: list[str] = []
    reasons: dict[str, str] = {}

    def _register(key: str, message: str) -> None:
        reasons.setdefault(key, message)

    metadata = bootstrap_env._apply_error_code_guidance(
        ["HNS_SERVICE_UNAVAILABLE"],
        register_reason=_register,
        detail_collector=details,
        remediation_collector=remediation,
    )

    assert any("host network service" in reason.lower() for reason in reasons.values())
    assert any("hns" in detail.lower() for detail in details)
    assert any("winsock" in step.lower() for step in remediation)
    assert (
        metadata["docker_worker_last_error_guidance_hns_service_unavailable"]
        .lower()
        .startswith("restart the host network service")
    )


def test_generic_error_code_guidance_handles_hns_variants() -> None:
    """Unknown HNS-prefixed error codes should still map to HNS remediation."""

    directive = bootstrap_env._derive_generic_error_code_guidance(
        "HNS_NETWORK_STORE_CORRUPT"
    )

    assert directive is not None
    assert "host network service" in directive.reason.lower()
    assert any("restart-service" in step.lower() for step in directive.remediation)
    assert "docker_worker_last_error_guidance_hns_network_store_corrupt" in directive.metadata


def test_normalize_warnings_interprets_hns_error_codes() -> None:
    """HNS worker stalls emitted by Docker Desktop should be rewritten."""

    payload = {
        "diagnostics": {
            "components": [
                {
                    "componentDisplayName": "Host Network Service",
                    "statusMessage": "WARNING: worker stalled; restarting (errCode=HNS_SERVICE_UNAVAILABLE)",
                    "errCode": "HNS_SERVICE_UNAVAILABLE",
                    "lastError": "host network service failed during start",
                }
            ]
        }
    }

    warnings, metadata = bootstrap_env._normalize_docker_warnings(payload)

    assert warnings, "expected worker stall guidance"
    banner = warnings[0].lower()
    assert "worker stalled; restarting" not in banner
    assert "host network service" in banner
    assert metadata["docker_worker_last_error_code"] == "HNS_SERVICE_UNAVAILABLE"
    assert metadata["docker_worker_last_error"].lower().startswith(
        "host network service failed"
    )


def test_worker_context_extraction_ignores_backoff_tokens() -> None:
    """Backoff metadata should not masquerade as the affected worker name."""

    message = (
        "WARNING: worker stalled; restarting backoff=45s "
        "lastError=\"vpnkit health-check timed out\" restartCount=5"
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert cleaned
    assert metadata["docker_worker_backoff"] == "45s"
    assert "docker_worker_context" not in metadata


def test_worker_context_extraction_ignores_errcode_colon_delimiter() -> None:
    """Colon-delimited ``errCode`` metadata should not leak into the context."""

    message = "WARNING: worker stalled; restarting errCode:WSL_VM_PAUSED"

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert cleaned
    assert "Code:" not in cleaned
    assert metadata["docker_worker_last_error_code"] == "WSL_VM_PAUSED"
    assert "docker_worker_context" not in metadata


def test_worker_stall_detected_variant_is_normalised() -> None:
    """Windows stall detection banners should be collapsed into canonical phrasing."""

    message = (
        "WARNING: worker stall detected; restarting due to virtualization pressure"
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert cleaned
    assert "worker stall detected" not in cleaned.lower()
    assert "worker stalled; restarting" not in cleaned.lower()
    assert metadata["docker_worker_health"] == "flapping"
