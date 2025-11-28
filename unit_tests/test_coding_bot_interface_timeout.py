import importlib
import os
import sys
import time
import types


def _reset_watchdog_state(module):
    module._PREPARE_PIPELINE_WATCHDOG["stages"] = []
    module._PREPARE_PIPELINE_WATCHDOG["timeouts"] = 0
    module._PREPARE_PIPELINE_WATCHDOG.pop("staged_ready", None)
    module._PREPARE_PIPELINE_WATCHDOG["staged_ready_event"] = module.threading.Event()


def _load_coding_bot_interface_module():
    sys.modules.pop("coding_bot_interface", None)
    sys.modules.pop("menace_sandbox.coding_bot_interface", None)

    menace_stub = types.ModuleType("menace_sandbox")
    menace_stub.__path__ = []  # type: ignore[attr-defined]
    sys.modules["menace_sandbox"] = menace_stub

    module = importlib.import_module("coding_bot_interface")
    _reset_watchdog_state(module)
    return module


def test_watchdog_timeout_uses_history_and_host_scale(monkeypatch):
    monkeypatch.delenv("PREPARE_PIPELINE_STAGE_BUDGETS", raising=False)
    monkeypatch.delenv("PREPARE_PIPELINE_VECTOR_STAGE_BUDGETS", raising=False)
    monkeypatch.delenv("PREPARE_PIPELINE_VECTOR_GRACE_SECS", raising=False)
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
            deadline, start_time=start_time, vector_heavy=False, stage_label="prepare"
        )
    )

    assert telemetry["history_mean"] == 6.0
    assert telemetry["host_scale"] != 1.0
    assert normalized_timeout == telemetry["stage_budget"]
    assert normalized_deadline == start_time + normalized_timeout
    assert telemetry["staged_readiness"] is False
    assert coding_bot_interface._PREPARE_PIPELINE_WATCHDOG.get("staged_ready") is None
    assert not coding_bot_interface._PREPARE_PIPELINE_WATCHDOG[
        "staged_ready_event"
    ].is_set()


def test_watchdog_timeout_enters_staged_readiness(monkeypatch):
    monkeypatch.setenv("PREPARE_PIPELINE_VECTOR_GRACE_SECS", "2")
    coding_bot_interface = _load_coding_bot_interface_module()
    start_time = time.perf_counter()
    deadline = start_time + 5.0

    coding_bot_interface._PREPARE_PIPELINE_WATCHDOG["stages"] = [
        {"label": "bootstrap", "elapsed": 10.0}
    ]

    monkeypatch.setattr(coding_bot_interface, "_MIN_STAGE_TIMEOUT_VECTOR", 5.0)
    monkeypatch.setattr(coding_bot_interface, "_VECTOR_STAGE_GRACE_PERIOD", 2.0)
    monkeypatch.setattr(os, "getloadavg", lambda: (1.0, 0.0, 0.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 1)

    normalized_deadline, normalized_timeout, telemetry = (
        coding_bot_interface._normalize_watchdog_timeout(
            deadline,
            start_time=start_time,
            vector_heavy=True,
            stage_label="vector_warmup",
        )
    )

    assert telemetry["staged_readiness"] is True
    assert telemetry["deadline_extended"] is True
    assert telemetry["extension_seconds"] > 0
    assert normalized_deadline == start_time + normalized_timeout
    assert coding_bot_interface._PREPARE_PIPELINE_WATCHDOG["staged_ready"][
        "stage"
    ] == "vector_warmup"
    assert coding_bot_interface._PREPARE_PIPELINE_WATCHDOG["staged_ready_event"].is_set()
