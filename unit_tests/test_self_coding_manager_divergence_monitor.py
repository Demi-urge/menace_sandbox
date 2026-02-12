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
    manager.objective_guard = None
    return manager


def test_divergence_trigger_pauses_and_emits_critical_telemetry():
    event_bus = EventBusStub()
    mgr = make_manager(event_bus=event_bus)

    mgr.register_patch_cycle("cycle-1", context_meta={"reward": 1.0, "revenue": 10.0})
    mgr.register_patch_cycle("cycle-2", context_meta={"reward": 2.0, "revenue": 9.0})
    mgr.register_patch_cycle("cycle-3", context_meta={"reward": 3.0, "revenue": 9.0})

    with pytest.raises(RuntimeError, match="self-coding paused"):
        mgr.register_patch_cycle("cycle-4", context_meta={"reward": 4.0, "revenue": 8.5})

    assert mgr._self_coding_paused is True
    assert mgr._self_coding_disabled_reason == "reward_profit_revenue_divergence"
    event_name, payload = event_bus.events[-1]
    assert event_bus.events[-3][0] == "self_coding:divergence_kill_switch"
    assert event_bus.events[-2][0] == "self_coding:critical_divergence"
    assert event_name == "self_coding:high_severity_alert"
    assert payload["severity"] == "high"
    assert payload["reward_window"] == [2.0, 3.0, 4.0]
    assert payload["revenue_window"] == [9.0, 9.0, 8.5]
    assert payload["cycle_metrics"][-1]["cycle_index"] >= 1
    assert payload["cycle_metrics"][-1]["bot_id"] == "alpha"
    assert payload["reward_trend"] > 0.0
    assert payload["real_metric_trend"] <= 0.0


def test_no_trigger_when_reward_and_business_metrics_align():
    mgr = make_manager()

    mgr.register_patch_cycle("cycle-1", context_meta={"reward": 1.0, "revenue": 10.0})
    mgr.register_patch_cycle("cycle-2", context_meta={"reward": 2.0, "revenue": 11.0})
    mgr.register_patch_cycle("cycle-3", context_meta={"reward": 3.0, "revenue": 12.0})

    assert mgr._self_coding_paused is False


def test_auto_resume_flow_after_recovery_cycles():
    event_bus = EventBusStub()
    mgr = make_manager(event_bus=event_bus)

    mgr.register_patch_cycle("cycle-1", context_meta={"reward": 1.0, "revenue": 10.0})
    mgr.register_patch_cycle("cycle-2", context_meta={"reward": 2.0, "revenue": 9.5})
    mgr.register_patch_cycle("cycle-3", context_meta={"reward": 3.0, "revenue": 9.0})
    with pytest.raises(RuntimeError, match="self-coding paused"):
        mgr.register_patch_cycle("cycle-4", context_meta={"reward": 4.0, "revenue": 8.0})

    assert mgr._self_coding_paused is True

    with pytest.raises(RuntimeError, match="self-coding paused"):
        mgr.register_patch_cycle("cycle-5", context_meta={"reward": 1.0, "revenue": 9.0})

    mgr.register_patch_cycle("cycle-6", context_meta={"reward": 0.5, "revenue": 9.5})

    assert mgr._self_coding_paused is False
    assert any(event == "self_coding:divergence_recovered" for event, _ in event_bus.events)


class DataBotWithMetricsFallback(DataBotStub):
    def __init__(self) -> None:
        super().__init__()
        self.db = types.SimpleNamespace(
            fetch=lambda limit=3: [
                {"bot": "alpha", "revenue": 8.0, "profitability": 2.0},
                {"bot": "alpha", "revenue": 8.5, "profitability": 2.5},
                {"bot": "alpha", "revenue": 9.0, "profitability": 3.0},
            ]
        )


def test_uses_databot_metrics_when_business_values_missing_from_context():
    mgr = scm.SelfCodingManager(
        EngineStub(),
        types.SimpleNamespace(),
        bot_name="alpha",
        data_bot=DataBotWithMetricsFallback(),
        bot_registry=BotRegistryStub(),
        quick_fix=types.SimpleNamespace(context_builder=None),
    )
    mgr.objective_guard = None

    mgr.register_patch_cycle("cycle-1", context_meta={"reward": 1.0})

    assert len(mgr._cycle_metrics_window) == 1
    assert mgr._cycle_metrics_window[-1].revenue == 8.0
    assert mgr._cycle_metrics_window[-1].profit == 2.0
