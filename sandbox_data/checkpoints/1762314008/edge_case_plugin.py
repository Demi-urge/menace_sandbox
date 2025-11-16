from __future__ import annotations

import json
import os
import pytest

from .edge_case_generator import generate_edge_cases


def _load_edge_cases() -> dict[str, object]:
    raw = os.getenv("SANDBOX_EDGE_CASES")
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
    return generate_edge_cases()


@pytest.fixture
def hostile_payloads() -> dict[str, object]:
    """Return edge case payloads for tests."""
    return _load_edge_cases()
