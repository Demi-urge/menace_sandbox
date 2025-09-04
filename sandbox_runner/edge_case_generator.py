"""Utilities for generating edge case test inputs.

This module emits a set of payloads covering common edge cases used during
sandbox testing. The payloads intentionally contain malformed JSON, timeout
sentinels, null/empty strings and other invalid formats.
"""

from __future__ import annotations

from typing import Dict, Any


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


def generate_edge_cases() -> Dict[str, Any]:
    """Return a mapping of filenames to edge case payloads."""
    none_value, empty_value = null_or_empty()
    return {
        "malformed.json": malformed_json(),
        "timeout": timeout_sentinel(),
        "empty.txt": empty_value,
        "null.txt": none_value,
        "invalid.bin": invalid_format(),
    }
