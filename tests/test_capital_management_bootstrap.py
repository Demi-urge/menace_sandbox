"""Tests for CapitalManagementBot bootstrap edge cases."""

from __future__ import annotations

from typing import List


def test_manager_initialisation_defers_during_nested_import(monkeypatch) -> None:
    """Self-coding bootstrap should defer when imports are still active."""

    import menace_sandbox.capital_management_bot as module

    load_calls: List[str] = []
    retry_calls: List[object] = []

    timer = getattr(module, "_manager_retry_timer", None)
    if timer is not None:
        try:
            timer.cancel()
        except Exception:
            pass

    monkeypatch.setattr(module, "manager", None)
    monkeypatch.setattr(module, "_manager_retry_count", 0)
    monkeypatch.setattr(module, "_manager_retry_timer", None)
    monkeypatch.setattr(module, "self_coding_import_depth", lambda: 2)

    def fake_schedule(delay=None) -> None:
        retry_calls.append(delay)
        module._manager_retry_count += 1

    monkeypatch.setattr(module, "_schedule_manager_retry", fake_schedule)
    monkeypatch.setattr(module, "_load_pipeline_instance", lambda: load_calls.append("pipeline"))
    monkeypatch.setattr(module, "_load_thresholds", lambda: load_calls.append("thresholds"))

    module._initialise_self_coding_manager()

    assert module.manager is None
    assert load_calls == []
    assert retry_calls == [None]
