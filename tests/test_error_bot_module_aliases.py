import importlib
import sys

import pytest


def test_error_bot_legacy_and_canonical_module_alias_same_object():
    try:
        canonical = importlib.import_module("menace_sandbox.error_bot")
    except Exception as exc:  # pragma: no cover - optional dependency environments
        pytest.skip(f"error_bot import unavailable in this environment: {exc}")

    legacy = importlib.import_module("menace.error_bot")

    assert canonical is legacy
    assert sys.modules["menace_sandbox.error_bot"] is sys.modules["menace.error_bot"]
