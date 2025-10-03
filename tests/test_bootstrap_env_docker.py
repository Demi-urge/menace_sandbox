"""Regression tests for Docker diagnostics in ``scripts/bootstrap_env``."""

from __future__ import annotations

import subprocess
import types

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


def test_worker_warning_sanitization_removes_raw_banner() -> None:
    """Ensure ``worker stalled`` banners are rewritten into guidance."""

    cleaned, metadata = bootstrap_env._normalise_docker_warning(
        "WARNING: worker stalled; restarting"
    )

    assert cleaned  # sanity check that a message is produced
    assert "worker stalled; restarting" not in cleaned.lower()
    assert "Docker Desktop reported repeated restarts" in cleaned
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
    assert metadata["docker_worker_backoff"].lower().startswith("pt45")
    assert metadata["docker_worker_last_error"].lower().startswith("context deadline")
    assert metadata["docker_worker_last_restart"] == "2024-05-01T10:15:00Z"


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
