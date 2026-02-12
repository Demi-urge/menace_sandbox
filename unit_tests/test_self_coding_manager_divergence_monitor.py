import importlib
import pathlib
import sys
import types

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parent))
scm = importlib.import_module("test_self_coding_manager_thresholds").scm


class BotRegistryStub:
    def register_bot(self, *args, **kwargs):
        return None


class DataBotStub(scm.DataBot):
    def __init__(self) -> None:
        self._thresholds = types.SimpleNamespace(
            roi_drop=-0.5,
            error_threshold=2.0,
            test_failure_threshold=1.0,
        )

    def roi(self, bot: str) -> float:
        return 1.0

    def average_errors(self, bot: str) -> float:
        return 0.0

    def average_test_failures(self, bot: str) -> float:
        return 0.0

    def get_thresholds(self, bot: str):
        return self._thresholds

    def reload_thresholds(self, bot: str):
        return self._thresholds


class EngineStub:
    def __init__(self):
        builder = types.SimpleNamespace(session_id="", refresh_db_weights=lambda: None)
        self.cognition_layer = types.SimpleNamespace(context_builder=builder)
        self.patch_suggestion_db = None


class EventBusStub:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def publish(self, event: str, payload: dict) -> None:
        self.events.append((event, payload))


class AuditTrailStub:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def record(self, payload: dict) -> None:
        self.records.append(payload)


def make_manager(*, event_bus: EventBusStub | None = None) -> scm.SelfCodingManager:
    manager = scm.SelfCodingManager(
        EngineStub(),
        types.SimpleNamespace(),
        bot_name="alpha",
        data_bot=DataBotStub(),
        bot_registry=BotRegistryStub(),
        quick_fix=types.SimpleNamespace(context_builder=None),
    )
    manager.event_bus = event_bus
    manager.engine.audit_trail = AuditTrailStub()
    return manager


def test_divergence_trigger_pauses_and_emits_critical_telemetry():
    event_bus = EventBusStub()
    mgr = make_manager(event_bus=event_bus)

    mgr.register_patch_cycle("cycle-1", context_meta={"reward": 1.0, "revenue": 10.0})
    mgr.register_patch_cycle("cycle-2", context_meta={"reward": 2.0, "revenue": 9.0})

    with pytest.raises(RuntimeError, match="self-coding paused"):
        mgr.register_patch_cycle("cycle-3", context_meta={"reward": 3.0, "revenue": 9.0})

    assert mgr._self_coding_paused is True
    assert mgr._self_coding_disabled_reason == "reward_profit_revenue_divergence"
    event_name, payload = event_bus.events[-1]
    assert event_bus.events[-2][0] == "self_coding:critical_divergence"
    assert event_name == "self_coding:high_severity_alert"
    assert payload["severity"] == "high"
    assert payload["reward_window"] == [1.0, 2.0, 3.0]
    assert payload["revenue_window"] == [10.0, 9.0, 9.0]
    assert payload["cycle_metrics"][-1]["cycle_index"] >= 1
    assert payload["cycle_metrics"][-1]["bot_id"] == "alpha"


def test_no_trigger_when_reward_and_business_metrics_align():
    mgr = make_manager()

    mgr.register_patch_cycle("cycle-1", context_meta={"reward": 1.0, "revenue": 10.0})
    mgr.register_patch_cycle("cycle-2", context_meta={"reward": 2.0, "revenue": 11.0})
    mgr.register_patch_cycle("cycle-3", context_meta={"reward": 3.0, "revenue": 12.0})

    assert mgr._self_coding_paused is False


def test_pause_persists_until_operator_controlled_reset():
    mgr = make_manager()

    mgr.register_patch_cycle("cycle-1", context_meta={"reward": 1.0, "revenue": 10.0})
    mgr.register_patch_cycle("cycle-2", context_meta={"reward": 2.0, "revenue": 10.0})
    with pytest.raises(RuntimeError, match="self-coding paused"):
        mgr.register_patch_cycle("cycle-3", context_meta={"reward": 3.0, "revenue": 9.0})

    with pytest.raises(RuntimeError, match="self-coding paused"):
        mgr.register_patch_cycle("cycle-4", context_meta={"reward": 0.5, "revenue": 99.0})

    mgr.reset_self_coding_pause(operator_id="oncall-1", reason="manual review complete")
    mgr.register_patch_cycle("cycle-5", context_meta={"reward": 0.5, "revenue": 99.0})

    assert mgr._self_coding_paused is False
    assert mgr.engine.audit_trail.records[-1]["action"] == "self_coding_manual_reset"
