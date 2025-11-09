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


def test_manager_disabled_when_orchestrator_missing(monkeypatch) -> None:
    """Self-coding bootstrap should fall back to the disabled manager."""

    import menace_sandbox.capital_management_bot as module

    timer = getattr(module, "_manager_retry_timer", None)
    if timer is not None:
        try:
            timer.cancel()
        except Exception:
            pass

    monkeypatch.setattr(module, "manager", None)
    monkeypatch.setattr(module, "_manager_retry_count", 0)
    monkeypatch.setattr(module, "_manager_retry_timer", None)
    monkeypatch.setattr(module, "self_coding_import_depth", lambda: 0)

    class DummyThresholds:
        roi_drop = 0.1
        error_increase = 0.2
        test_failure_increase = 0.3

    monkeypatch.setattr(module, "_load_pipeline_instance", lambda: object())
    monkeypatch.setattr(module, "_load_thresholds", lambda: DummyThresholds())
    monkeypatch.setattr(module, "_get_engine", lambda: object())

    registry_calls = {}

    class DummyRegistry:
        def register_bot(self, *args, **kwargs):
            registry_calls["args"] = args
            registry_calls["kwargs"] = kwargs

    registry = DummyRegistry()
    monkeypatch.setattr(module, "_get_registry", lambda: registry)
    data_bot = object()
    monkeypatch.setattr(module, "_get_data_bot", lambda: data_bot)

    disabled_instances = []

    class DummyDisabled:
        def __init__(self, *, bot_registry, data_bot):
            disabled_instances.append((bot_registry, data_bot))

    monkeypatch.setattr(module, "_DisabledSelfCodingManager", DummyDisabled)

    monkeypatch.setattr(module, "_load_evolution_orchestrator", lambda: None)

    def fail_internalize(*args, **kwargs):  # pragma: no cover - should not run
        raise AssertionError("internalize_coding_bot should not be called")

    monkeypatch.setattr(module, "internalize_coding_bot", fail_internalize)

    retry_calls: list[object] = []

    def fake_schedule(delay=None):
        retry_calls.append(delay)

    monkeypatch.setattr(module, "_schedule_manager_retry", fake_schedule)
    monkeypatch.setattr(module, "_get_capital_management_bot_class", lambda: type("Dummy", (), {}))

    module._initialise_self_coding_manager()

    assert isinstance(module.manager, DummyDisabled)
    assert disabled_instances == [(registry, data_bot)]
    assert "args" in registry_calls
    assert retry_calls == []
    assert module._manager_retry_timer is None
    assert module._manager_retry_count == 0
