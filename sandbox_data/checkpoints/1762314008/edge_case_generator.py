"""Utilities for generating edge case test inputs.

This module emits a set of payloads covering common edge cases used during
sandbox testing.  The payloads intentionally contain malformed JSON, timeout
sentinels, null/empty strings and other invalid formats.  Network/API
anomalies are also provided via mock URLs allowing callers to stub out
HTTP responses without performing real network requests.
"""

from __future__ import annotations

from typing import Dict, Any
import os
import json
from pathlib import Path
import yaml

from dynamic_path_router import resolve_path


def malformed_json() -> str:
    """Return a JSON string with broken syntax."""
    return '{"broken": "json",}'  # trailing comma makes this invalid


def timeout_sentinel() -> str:
    """Return a marker representing a simulated timeout."""
    return "__TIMEOUT__"


def null_or_empty() -> list[str | None]:
    """Return a sequence containing ``None`` and an empty string."""
    return [None, ""]


def invalid_format() -> str:
    """Return data in an unexpected format."""
    return "not a valid format"


def malformed_json_response() -> str:
    """Return a network response containing malformed JSON."""
    return malformed_json()


def http_timeout_marker() -> str:
    """Return a marker representing an HTTP timeout."""
    return timeout_sentinel()


def empty_payload() -> str:
    """Return an empty network payload."""
    return ""


def null_payload() -> None:
    """Return a null network payload."""
    none, _ = null_or_empty()
    return none


def invalid_payload_format() -> str:
    """Return network data in an unexpected format."""
    return invalid_format()


def network_anomalies() -> Dict[str, Any]:
    """Return mapping of URLs to anomalous network responses.

    Keys are mock URLs consumed by network interceptors in tests.  Each value
    mirrors the corresponding local edge case variant.
    """
    none_val, empty_val = null_or_empty()
    return {
        "http://edge-case.test/malformed": malformed_json_response(),
        "http://edge-case.test/timeout": http_timeout_marker(),
        "http://edge-case.test/empty": empty_val,
        "http://edge-case.test/null": none_val,
        "http://edge-case.test/invalid": invalid_payload_format(),
    }


def _load_extra_payloads() -> Dict[str, Any]:
    """Load additional payloads from environment or config file."""
    data: Dict[str, Any] = {}

    raw = os.getenv("SANDBOX_HOSTILE_PAYLOADS")
    if raw:
        try:
            extra = json.loads(raw)
        except Exception:
            try:
                extra = yaml.safe_load(raw)
            except Exception:
                extra = {}
        if isinstance(extra, dict):
            data.update(extra)

    file_path = os.getenv("SANDBOX_HOSTILE_PAYLOADS_FILE")
    if file_path:
        try:
            path = Path(resolve_path(file_path))
        except FileNotFoundError:
            path = Path(file_path)
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8")
                extra = json.loads(content)
            except Exception:
                try:
                    extra = yaml.safe_load(content)
                except Exception:
                    extra = {}
            if isinstance(extra, dict):
                data.update(extra)

    return data


def generate_edge_cases() -> Dict[str, Any]:
    """Return a mapping of filenames and URLs to edge case payloads.

    The set can be extended via ``SANDBOX_HOSTILE_PAYLOADS`` or
    ``SANDBOX_HOSTILE_PAYLOADS_FILE``.

    Network mock keys:

    ``http://edge-case.test/malformed``
        Malformed JSON response.
    ``http://edge-case.test/timeout``
        Timeout sentinel payload.
    ``http://edge-case.test/empty``
        Empty string payload.
    ``http://edge-case.test/null``
        ``None`` payload.
    ``http://edge-case.test/invalid``
        Unexpected format payload.
    """
    none_value, empty_value = null_or_empty()
    cases = {
        "malformed.json": malformed_json(),
        "timeout": timeout_sentinel(),
        "empty.txt": empty_value,
        "null.txt": none_value,
        "invalid.bin": invalid_format(),
    }
    cases.update(network_anomalies())
    cases.update(_load_extra_payloads())
    return cases
