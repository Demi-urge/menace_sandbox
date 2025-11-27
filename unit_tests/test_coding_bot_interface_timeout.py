import importlib
import sys
import types


def _load_coding_bot_interface_module():
    sys.modules.pop("coding_bot_interface", None)
    sys.modules.pop("menace_sandbox.coding_bot_interface", None)

    menace_stub = types.ModuleType("menace_sandbox")
    menace_stub.__path__ = []  # type: ignore[attr-defined]
    sys.modules["menace_sandbox"] = menace_stub

    return importlib.import_module("coding_bot_interface")


def test_watchdog_timeout_clamped_to_standard_floor():
    coding_bot_interface = _load_coding_bot_interface_module()
    start_time = 100.0
    deadline = start_time + 30.0

    normalized_deadline, normalized_timeout, escalated = (
        coding_bot_interface._normalize_watchdog_timeout(
            deadline, start_time=start_time, vector_heavy=False
        )
    )

    assert escalated is True
    assert normalized_timeout == coding_bot_interface._MIN_STAGE_TIMEOUT
    assert normalized_deadline == start_time + coding_bot_interface._MIN_STAGE_TIMEOUT


def test_watchdog_timeout_clamped_to_vector_floor():
    coding_bot_interface = _load_coding_bot_interface_module()
    start_time = 200.0
    deadline = start_time + 30.0

    normalized_deadline, normalized_timeout, escalated = (
        coding_bot_interface._normalize_watchdog_timeout(
            deadline, start_time=start_time, vector_heavy=True
        )
    )

    assert escalated is True
    assert normalized_timeout == coding_bot_interface._MIN_STAGE_TIMEOUT_VECTOR
    assert normalized_deadline == start_time + coding_bot_interface._MIN_STAGE_TIMEOUT_VECTOR
