"""Regression tests for Docker diagnostics in ``scripts/bootstrap_env``."""

from __future__ import annotations

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
