import importlib
import os
import sys
import time
import types


def _load_coding_bot_interface_module():
    sys.modules.pop("coding_bot_interface", None)
    sys.modules.pop("menace_sandbox.coding_bot_interface", None)

    menace_stub = types.ModuleType("menace_sandbox")
    menace_stub.__path__ = []  # type: ignore[attr-defined]
    sys.modules["menace_sandbox"] = menace_stub

    return importlib.import_module("coding_bot_interface")


def test_watchdog_timeout_uses_history_and_host_scale(monkeypatch):
    coding_bot_interface = _load_coding_bot_interface_module()
    start_time = 100.0
    deadline = start_time + 500.0

    coding_bot_interface._PREPARE_PIPELINE_WATCHDOG["stages"] = [
        {"label": "bootstrap", "elapsed": 5.0},
        {"label": "prepare", "elapsed": 7.0},
    ]

    monkeypatch.setattr(os, "getloadavg", lambda: (4.0, 0.0, 0.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 2)

    normalized_deadline, normalized_timeout, telemetry = (
        coding_bot_interface._normalize_watchdog_timeout(
            deadline, start_time=start_time, vector_heavy=False
        )
    )

    assert telemetry["history_mean"] == 6.0
    assert telemetry["host_scale"] != 1.0
    assert normalized_timeout == telemetry["stage_budget"]
    assert normalized_deadline == start_time + normalized_timeout
    assert telemetry["staged_readiness"] is False


def test_watchdog_timeout_enters_staged_readiness(monkeypatch):
    coding_bot_interface = _load_coding_bot_interface_module()
    start_time = time.perf_counter()
    deadline = start_time + 5.0

    coding_bot_interface._PREPARE_PIPELINE_WATCHDOG["stages"] = [
        {"label": "bootstrap", "elapsed": 120.0}
    ]

    monkeypatch.setattr(os, "getloadavg", lambda: (1.0, 0.0, 0.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 1)

    normalized_deadline, normalized_timeout, telemetry = (
        coding_bot_interface._normalize_watchdog_timeout(
            deadline, start_time=start_time, vector_heavy=True
        )
    )

    assert telemetry["staged_readiness"] is True
    assert normalized_timeout == telemetry["remaining_window"]
    assert normalized_deadline == start_time + normalized_timeout
