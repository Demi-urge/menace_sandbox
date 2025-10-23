import importlib
import sys

from menace_sandbox import coding_bot_interface
from menace_sandbox import data_bot as data_bot_module
from menace_sandbox import shared_event_bus
from menace_sandbox.unified_event_bus import UnifiedEventBus


class _StubTemplateDB:
    def query(self, task: str):  # pragma: no cover - simple stub
        return type("_Empty", (), {"empty": True})()


def _listener_count(bus: UnifiedEventBus, topic: str) -> int:
    callbacks = getattr(bus, "_subs", {}).get(topic, [])  # type: ignore[attr-defined]
    return len(callbacks)


def test_resource_prediction_bot_reuses_shared_data_bot(monkeypatch):
    test_bus = UnifiedEventBus()
    monkeypatch.setattr(shared_event_bus, "event_bus", test_bus, raising=False)
    monkeypatch.setattr(data_bot_module, "_SHARED_EVENT_BUS", test_bus, raising=False)

    data_bot_module.reset_shared_data_bot_for_testing()
    shared_instance = data_bot_module.get_shared_data_bot(
        event_bus=test_bus, start_server=False
    )
    monkeypatch.setattr(data_bot_module, "data_bot", shared_instance, raising=False)

    def _noop_decorator(**_kwargs):  # pragma: no cover - simple test shim
        return lambda cls: cls

    monkeypatch.setattr(coding_bot_interface, "self_coding_managed", _noop_decorator)

    sys.modules.pop("menace_sandbox.resource_prediction_bot", None)
    import menace_sandbox.resource_prediction_bot as resource_module

    first_bot = resource_module.ResourcePredictionBot(db=_StubTemplateDB())
    assert first_bot.data_bot is shared_instance

    listeners_before = _listener_count(test_bus, "bot:updated")
    assert listeners_before > 0

    resource_module = importlib.reload(resource_module)
    second_bot = resource_module.ResourcePredictionBot(db=_StubTemplateDB())

    listeners_after = _listener_count(test_bus, "bot:updated")
    assert listeners_after == listeners_before
    assert second_bot.data_bot is first_bot.data_bot
