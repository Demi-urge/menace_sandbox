"""Tests for sanitising Docker worker stall banners within bootstrap diagnostics."""

from __future__ import annotations

import base64
from collections.abc import Iterable
import json

import pytest

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


def test_worker_metadata_bytes_are_sanitized() -> None:
    """Byte-oriented metadata should be normalised and fingerprinted."""

    payload = b"WARNING: worker stalled; restarting component=\"vpnkit\" restartCount=2"
    metadata = {"docker_worker_last_error_banner_raw": payload}

    bootstrap_env._redact_worker_banner_artifacts(metadata)  # type: ignore[arg-type]

    sanitized = metadata.get("docker_worker_last_error_banner_raw")
    assert isinstance(sanitized, str)
    assert "worker stalled; restarting" not in sanitized.lower()

    fingerprint = metadata.get("docker_worker_last_error_banner_raw_fingerprint")
    assert fingerprint
    assert fingerprint.startswith(bootstrap_env._WORKER_STALLED_SIGNATURE_PREFIX)


def test_worker_metadata_base64_payload_is_sanitized() -> None:
    """Base64 encoded payloads should be decoded before sanitisation."""

    raw = b"WARNING: worker stalled; restarting component=\"vpnkit\" restartCount=2"
    payload = base64.b64encode(raw).decode("ascii")
    metadata = {"docker_worker_last_error_banner_raw": payload}

    bootstrap_env._redact_worker_banner_artifacts(metadata)  # type: ignore[arg-type]

    sanitized = metadata.get("docker_worker_last_error_banner_raw")
    assert isinstance(sanitized, str)
    assert "worker stalled; restarting" not in sanitized.lower()
    assert "vpnkit" in sanitized.lower()

    fingerprint = metadata.get("docker_worker_last_error_banner_raw_fingerprint")
    assert fingerprint
    assert fingerprint.startswith(bootstrap_env._WORKER_STALLED_SIGNATURE_PREFIX)


def test_worker_metadata_utf16_bytes_are_sanitized() -> None:
    """UTF-16 encoded payloads from Windows event logs should be decoded cleanly."""

    payload = "WARNING: worker stalled; restarting component=\"vpnkit\"".encode("utf-16-le")
    metadata = {"docker_worker_last_error_banner_raw": payload}

    bootstrap_env._redact_worker_banner_artifacts(metadata)  # type: ignore[arg-type]

    sanitized = metadata.get("docker_worker_last_error_banner_raw")
    assert isinstance(sanitized, str)
    assert "worker stalled; restarting" not in sanitized.lower()
    assert "vpnkit" in sanitized.lower()

    fingerprint = metadata.get("docker_worker_last_error_banner_raw_fingerprint")
    assert fingerprint
    assert fingerprint.startswith(bootstrap_env._WORKER_STALLED_SIGNATURE_PREFIX)


def test_worker_metadata_memoryview_is_sanitized() -> None:
    """``memoryview`` payloads should be decoded and rewritten."""

    payload = memoryview(b"worker stalled; restarting component=\"vm\"")

    sanitized, digest = bootstrap_env._sanitize_worker_metadata_value(payload)  # type: ignore[arg-type]

    assert sanitized
    assert "worker stalled; restarting" not in sanitized.lower()
    assert digest
    assert digest.startswith(bootstrap_env._WORKER_STALLED_SIGNATURE_PREFIX)


def test_nested_bytes_payload_is_sanitized() -> None:
    """Nested byte payloads should be converted into guidance."""

    metadata = {
        "docker_worker_nested_payload": [memoryview(b"worker stalled; restarting due to IO pressure")]
    }

    bootstrap_env._redact_worker_banner_artifacts(metadata)  # type: ignore[arg-type]

    nested = metadata["docker_worker_nested_payload"]
    assert isinstance(nested, list)
    assert nested
    first = nested[0]
    assert isinstance(first, str)
    assert "worker stalled; restarting" not in first.lower()

    fingerprints = metadata.get("docker_worker_nested_banner_fingerprints", "")
    assert isinstance(fingerprints, str)
    assert bootstrap_env._WORKER_STALLED_SIGNATURE_PREFIX in fingerprints


def test_worker_banner_subject_skips_severity_prefixes() -> None:
    message = "WARN[0042] moby/buildkit: worker stalled; restarting due to IO pressure"

    sanitized = bootstrap_env._sanitize_worker_banner_text(message)  # type: ignore[attr-defined]

    assert sanitized == (
        "Docker Desktop automatically restarted the moby/buildkit worker after it stalled due to IO pressure"
    )


@pytest.mark.parametrize(
    "phrase",
    (
        "reinitializing",
        "re-initializing",
        "reinitialising",
        "relaunching",
        "re-launching",
        "reiniting",
    ),
)
def test_worker_banner_recovery_synonyms_are_sanitized(phrase: str) -> None:
    message = (
        f'WARNING: worker stalled; {phrase} component="vpnkit" restartCount=2 '
        "lastError=\"vpnkit background sync stalled\""
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    lowered = cleaned.lower()
    assert "worker stalled; restarting" not in lowered
    assert phrase.lower() not in lowered
    assert metadata.get("docker_worker_health") == "flapping"
    details = metadata.get("docker_worker_health_details", "").lower()
    assert "automatically restarted" in details
    assert "vpnkit" in details


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


def test_worker_banner_html_entity_semicolon_is_decoded() -> None:
    message = "worker stalled&#59; restarting after host sleep"

    sanitized = bootstrap_env._sanitize_worker_banner_text(message)  # type: ignore[attr-defined]

    assert sanitized == bootstrap_env._WORKER_STALLED_PRIMARY_NARRATIVE


def test_fullwidth_worker_banner_is_normalized() -> None:
    message = "ＷＡＲＮＩＮＧ： Ｗｏｒｋｅｒ ｓｔａｌｌｅｄ； ｒｅｓｔａｒｔｉｎｇ"

    cleaned, _ = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled" not in cleaned.lower()
    assert "docker desktop" in cleaned.lower()


def test_worker_banner_errcode_cpu_pressure_guidance() -> None:
    message = (
        "WARNING: worker stalled; restarting component=\"vpnkit\" "
        "errCode=VPNKIT_BACKGROUND_SYNC_CPU_PRESSURE restartCount=3"
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    lowered = cleaned.lower()
    assert "worker stalled" not in lowered
    assert "cpu pressure" in lowered
    assert "vpnkit" in lowered

    fingerprint_key = "docker_worker_last_error_guidance_vpnkit_background_sync_cpu_pressure"
    assert fingerprint_key in metadata
    assert "cpu" in metadata[fingerprint_key].lower()


def test_worker_unresponsive_banner_is_sanitized() -> None:
    message = "WARNING: worker unresponsive; restarting because heartbeat lost"

    sanitized = bootstrap_env._sanitize_worker_banner_text(message)  # type: ignore[attr-defined]

    assert "worker unresponsive; restarting" not in sanitized.lower()
    assert "heartbeat" in sanitized.lower()
    assert "automatically restarted" in sanitized.lower()


def test_worker_timeout_metadata_is_sanitized() -> None:
    metadata = {
        "docker_worker_last_error_banner_raw": "worker timed out; restarting",
        "docker_worker_last_error_banner": "worker timed out; restarting",
    }

    bootstrap_env._redact_worker_banner_artifacts(metadata)  # type: ignore[arg-type]

    for key in ("docker_worker_last_error_banner_raw", "docker_worker_last_error_banner"):
        value = metadata.get(key)
        assert isinstance(value, str)
        assert "worker timed out; restarting" not in value.lower()
        assert "automatically restarted" in value.lower()


def test_worker_banner_errcode_vsock_signal_guidance() -> None:
    message = (
        "WARN[0032] moby/buildkit: worker stalled; restarting "
        "component=\"vpnkit\" errCode=VPNKIT_VSOCK_SIGNAL_LOST"
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    lowered = cleaned.lower()
    assert "worker stalled" not in lowered
    assert "vsock" in lowered
    assert "vpnkit" in lowered

    fingerprint_key = "docker_worker_last_error_guidance_vpnkit_vsock_signal_lost"
    assert fingerprint_key in metadata
    assert "vsock" in metadata[fingerprint_key].lower()


def test_format_worker_restart_reason_strips_prefixes() -> None:
    reason = "because of lingering IO pressure "

    formatted = bootstrap_env._format_worker_restart_reason(reason)  # type: ignore[attr-defined]

    assert formatted == "lingering IO pressure"


def test_format_worker_restart_reason_rejects_worker_tokens() -> None:
    reason = "due to worker stalled; restarting"

    formatted = bootstrap_env._format_worker_restart_reason(reason)  # type: ignore[attr-defined]

    assert formatted is None


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


def test_worker_metadata_keys_are_sanitized() -> None:
    payload = {
        "WARNING: worker stalled; restarting component=\"vpnkit\"": "worker stalled; restarting",
        "nested": {
            "worker stalled; restarting (errCode=VPNKIT_BACKGROUND_SYNC_CPU)": "vpnkit background sync worker stalled; restarting",
        },
    }

    sanitized, digests, mutated = bootstrap_env._scrub_nested_worker_artifacts(payload)  # type: ignore[attr-defined]

    assert mutated
    assert digests, "expected worker banner fingerprints to be propagated"
    assert all("worker stalled" not in key.lower() for key in sanitized.keys())

    nested = sanitized.get("nested")
    assert isinstance(nested, dict)
    assert all("worker stalled" not in key.lower() for key in nested.keys())
    assert all("worker stalled" not in str(value).lower() for value in nested.values())


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


def test_worker_hibernation_error_code_guidance() -> None:
    message = "WARNING: worker stalled; restarting (errCode=WSL_VM_HIBERNATED)"

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled" not in cleaned.lower()
    assert metadata.get("docker_worker_last_error_code") == "WSL_VM_HIBERNATED"
    guidance = metadata.get("docker_worker_last_error_guidance_wsl_vm_hibernated")
    assert guidance
    assert "fast startup" in guidance.lower()


def test_hibernation_inferred_from_context() -> None:
    message = "worker stalled; restarting after Windows resumed from hibernation"

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled" not in cleaned.lower()
    assert metadata.get("docker_worker_last_error_code") == "WSL_VM_HIBERNATED"


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


@pytest.mark.parametrize(
    "connector",
    [
        "；",
        "：",
        "，",
        "。",
        "！",
        "？",
        "!",
        "?",
        "！？",
        "？！",
        "!?",
        "?!",
        "‧",
        "・",
        "／",
        "＼",
    ],
)
def test_worker_banner_localised_connectors_are_sanitized(connector: str) -> None:
    """Non-ASCII separators should still trigger stall banner rewriting."""

    message = f"WARN[0001] moby/buildkit: worker stalled{connector} restarting component=\"vpnkit\""
    metadata: dict[str, str] = {}

    safeguarded = bootstrap_env._guarantee_worker_banner_suppression([message], metadata)  # type: ignore[attr-defined]

    assert safeguarded
    assert all(
        "worker stalled; restarting" not in entry.casefold()
        for entry in safeguarded
        if isinstance(entry, str)
    )
    assert metadata.get("docker_worker_health") == "flapping"


def test_worker_banner_whitespace_separator_is_sanitized() -> None:
    """Pure whitespace separators should still trigger stall rewriting."""

    message = "WARNING: worker stalled    restarting component=\"vpnkit\""
    metadata: dict[str, str] = {}

    safeguarded = bootstrap_env._guarantee_worker_banner_suppression([message], metadata)  # type: ignore[attr-defined]

    assert safeguarded
    assert all(
        "worker stalled; restarting" not in entry.casefold()
        for entry in safeguarded
        if isinstance(entry, str)
    )
    assert metadata.get("docker_worker_health") == "flapping"


def test_worker_banner_period_separator_is_sanitized() -> None:
    """A literal period separator should be treated as a stall banner delimiter."""

    message = "WARNING: worker stalled. restarting component=\"vpnkit\" restartCount=2"
    metadata: dict[str, str] = {}

    safeguarded = bootstrap_env._guarantee_worker_banner_suppression([message], metadata)  # type: ignore[attr-defined]

    assert safeguarded
    assert all(
        "worker stalled" not in entry.casefold()
        for entry in safeguarded
        if isinstance(entry, str)
    )
    assert metadata.get("docker_worker_health") == "flapping"


def test_finalize_sequences_strip_carriage_return_variants() -> None:
    """Final sanitisation should catch carriage-return variants of the banner."""

    message = "WARNING: worker stalled;\r restarting component=\"vpnkit\""
    metadata: dict[str, str] = {}

    sanitized = bootstrap_env._finalize_worker_banner_sequences([message], metadata)  # type: ignore[attr-defined]

    assert sanitized
    assert all(
        "worker stalled" not in entry.lower()
        for entry in sanitized
        if isinstance(entry, str)
    )
    assert metadata.get("docker_worker_health") == "flapping"
