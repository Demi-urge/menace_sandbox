"""Tests for sanitising Docker worker stall banners within bootstrap diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
import json

from scripts import bootstrap_env


def _flatten_values(value: object) -> Iterable[object]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _flatten_values(child)
        return

    if isinstance(value, (list, tuple, set, frozenset)):
        for child in value:
            yield from _flatten_values(child)
        return

    yield value


def test_nested_structures_are_scrubbed_of_worker_stall_banners() -> None:
    metadata: dict[str, object] = {
        "docker_worker_nested_payload": {
            "raw_messages": [
                "Worker stalled; restarting due to host suspend",
                {"inner": "worker stalled; restarting (context=db)"},
            ],
            "tuple_data": (
                "worker stalled; restarting soon",
                "worker stuck; restarting",
                "background worker stabilised",
            ),
        },
        "docker_worker_set_payload": {"worker stalled; restarting backoff 5s"},
    }

    bootstrap_env._redact_worker_banner_artifacts(metadata)  # type: ignore[arg-type]

    flattened = list(_flatten_values(metadata))
    for item in flattened:
        if isinstance(item, str):
            lowered = item.casefold()
            assert "worker stalled; restarting" not in lowered

    nested_fingerprints = metadata.get("docker_worker_nested_banner_fingerprints")
    assert isinstance(nested_fingerprints, str)
    assert "worker-banner:" in nested_fingerprints


def test_worker_banner_subject_skips_severity_prefixes() -> None:
    message = "WARN[0042] moby/buildkit: worker stalled; restarting due to IO pressure"

    sanitized = bootstrap_env._sanitize_worker_banner_text(message)  # type: ignore[attr-defined]

    assert sanitized == (
        "Docker Desktop automatically restarted the moby/buildkit worker after it stalled"
    )


def test_worker_banner_ignores_structured_log_metadata() -> None:
    message = (
        'time="2024-05-03T08:13:37-07:00" level=warning msg="worker stalled; restarting"'
    )

    sanitized = bootstrap_env._sanitize_worker_banner_text(message)  # type: ignore[attr-defined]

    assert sanitized == bootstrap_env._WORKER_STALLED_PRIMARY_NARRATIVE


def test_worker_banner_treats_warning_prefix_as_noise() -> None:
    message = "warning: worker stalled; restarting"

    sanitized = bootstrap_env._sanitize_worker_banner_text(message)  # type: ignore[attr-defined]

    assert sanitized == bootstrap_env._WORKER_STALLED_PRIMARY_NARRATIVE


def test_worker_banner_final_guard_rewrites_literal_phrase() -> None:
    messages = [
        "worker stalled; restarting",
        "no action required",
    ]
    metadata: dict[str, str] = {}

    safeguarded = bootstrap_env._guarantee_worker_banner_suppression(messages, metadata)  # type: ignore[attr-defined]

    assert all("worker stalled; restarting" not in entry.casefold() for entry in safeguarded if isinstance(entry, str))
    assert metadata["docker_worker_health"] == "flapping"


def test_worker_banner_json_payload_is_sanitized() -> None:
    payload = json.dumps(
        {
            "time": "2024-10-01T08:00:00-07:00",
            "level": "warning",
            "msg": "worker stalled; restarting",
            "context": "vpnkit",
            "diagnostic": {"error": "vm paused"},
        }
    )
    metadata: dict[str, str] = {}

    harmonised = bootstrap_env._enforce_worker_banner_sanitization([payload], metadata)  # type: ignore[attr-defined]

    assert len(harmonised) == 1
    message = harmonised[0]
    assert isinstance(message, str)
    assert "worker stalled; restarting" not in message.lower()
    assert "vpnkit" in message.lower()
    assert metadata.get("docker_worker_context") == "vpnkit"
    assert metadata.get("docker_worker_health") == "flapping"


def test_worker_banner_structured_mapping_is_scrubbed() -> None:
    payload = {
        "level": "warn",
        "message": "worker stalled; restarting",
        "component": "com.docker.backend",
        "context": "vpnkit",
    }

    rewritten, metadata = bootstrap_env._scrub_residual_worker_warnings([payload])  # type: ignore[attr-defined]

    assert rewritten
    assert isinstance(rewritten[0], str)
    assert "worker stalled; restarting" not in rewritten[0].lower()
    assert metadata.get("docker_worker_context") == "vpnkit"


def test_worker_banner_nested_json_is_rewritten_by_guard() -> None:
    payload = json.dumps(
        {
            "warnings": [
                {
                    "message": "worker stalled; restarting",
                    "module": "vpnkit",
                    "extra": {"retry": "5s"},
                }
            ]
        }
    )
    metadata: dict[str, str] = {}

    safeguarded = bootstrap_env._guarantee_worker_banner_suppression([payload], metadata)  # type: ignore[attr-defined]

    assert safeguarded
    assert isinstance(safeguarded[0], str)
    assert "worker stalled; restarting" not in safeguarded[0].lower()
    assert metadata.get("docker_worker_context") == "vpnkit"


def test_worker_banner_split_across_lines_is_stitched_and_rewritten() -> None:
    messages = [
        "WARNING: worker stalled;",
        'restarting component="vpnkit" restartCount=2',
        "lastError=\"worker stalled; restarting\"",
    ]

    metadata: dict[str, str] = {}

    safeguarded = bootstrap_env._guarantee_worker_banner_suppression(messages, metadata)  # type: ignore[attr-defined]

    assert all(
        "worker stalled" not in entry.lower()
        for entry in safeguarded
        if isinstance(entry, str)
    )
    assert metadata.get("docker_worker_health") == "flapping"
