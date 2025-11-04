"""Tests for preset validation in :mod:`run_autonomous`."""

from __future__ import annotations

from typing import Any

import importlib

import pytest
from pydantic import ValidationError


def _base_preset(**extra: Any) -> dict[str, Any]:
    preset: dict[str, Any] = {
        "CPU_LIMIT": "1",
        "MEMORY_LIMIT": "512Mi",
    }
    preset.update(extra)
    return preset


def test_validate_presets_accepts_profile_specific_fields() -> None:
    run_autonomous = importlib.reload(importlib.import_module("run_autonomous"))
    presets = run_autonomous.validate_presets(
        [
            _base_preset(
                API_LATENCY_MS="500",
                SANDBOX_STUB_STRATEGY="misuse",
                INVALID_CONFIG="true",
                CONCURRENCY_LEVEL="100",
                SCENARIO_NAME="user_misuse",
            )
        ]
    )

    assert presets[0]["API_LATENCY_MS"] == 500
    assert presets[0]["CONCURRENCY_LEVEL"] == 100
    assert presets[0]["INVALID_CONFIG"] is True
    assert presets[0]["SANDBOX_STUB_STRATEGY"] == "misuse"


def test_validate_presets_rejects_unexpected_attribute_names() -> None:
    run_autonomous = importlib.reload(importlib.import_module("run_autonomous"))
    with pytest.raises(ValidationError):
        run_autonomous.validate_presets([_base_preset(customField=1)])
