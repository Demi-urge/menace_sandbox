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


def test_docker_cli_discovery_uses_wsl_translation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLI discovery should resolve Windows paths via WSL mount translations."""

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
        program_files / "Docker" / "Docker" / "cli",
        program_data / "DockerDesktop" / "cli",
        program_data / "DockerDesktop" / "cli-bin",
        program_data / "DockerDesktop" / "cli-tools",
        local_appdata / "DockerDesktop" / "cli",
        local_appdata / "DockerDesktop" / "cli-bin",
        local_appdata / "DockerDesktop" / "cli-tools",
    }

    assert expected.issubset(directory_set)


def test_worker_warning_sanitization_removes_raw_banner() -> None:
    """Ensure ``worker stalled`` banners are rewritten into guidance."""

    cleaned, metadata = bootstrap_env._normalise_docker_warning(
        "WARNING: worker stalled; restarting"
    )

    assert cleaned  # sanity check that a message is produced
    assert "worker stalled; restarting" not in cleaned.lower()
    assert "Docker Desktop reported repeated restarts" in cleaned
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
    assert metadata["docker_worker_health"] == "flapping"


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
    assert any("docker desktop reported" in warning.lower() for warning in warnings)
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
    assert metadata["docker_worker_last_error_code"] == "WSL_KERNEL_OUTDATED"

    telemetry = bootstrap_env.WorkerRestartTelemetry.from_metadata(metadata)
    context = _windows_context()
    assessment = bootstrap_env._classify_worker_flapping(telemetry, context)

    assert assessment.severity == "error"
    assert any("wsl kernel" in detail.lower() for detail in assessment.details)
    assert any("wsl --update" in step.lower() for step in assessment.remediation)


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
        "docker desktop reported repeated restarts" in warning.lower()
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
