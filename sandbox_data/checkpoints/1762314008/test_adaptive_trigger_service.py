import types
import pytest

pytest.importorskip("pandas")

from menace.adaptive_trigger_service import AdaptiveTriggerService
from menace.unified_event_bus import UnifiedEventBus


def _stop_after_first(svc: AdaptiveTriggerService):
    def inner(_: float) -> None:
        svc.running = False
        raise SystemExit
    return inner


def test_service_publishes(monkeypatch):
    bus = UnifiedEventBus()
    events = []
    bus.subscribe("evolve:self_improve", lambda t, e: events.append(("improve", e)))
    bus.subscribe("evolve:system", lambda t, e: events.append(("system", e)))
    data_bot = types.SimpleNamespace(db=types.SimpleNamespace(fetch=lambda limit=30: [{"errors": 5}]))
    cap_bot = types.SimpleNamespace(energy_score=lambda **k: 0.2)
    svc = AdaptiveTriggerService(data_bot, cap_bot, bus, interval=0, error_threshold=0.1, energy_threshold=0.3)
    svc.running = True
    monkeypatch.setattr("menace.adaptive_trigger_service.time.sleep", _stop_after_first(svc))
    with pytest.raises(SystemExit):
        svc._loop()
    kinds = {ev[0] for ev in events}
    assert "improve" in kinds and "system" in kinds


def test_loop_logs_exception(monkeypatch, caplog):
    bus = UnifiedEventBus()
    data_bot = types.SimpleNamespace(
        db=types.SimpleNamespace(fetch=lambda limit=30: (_ for _ in ()).throw(RuntimeError("fail1")))
    )
    cap_bot = types.SimpleNamespace(
        energy_score=lambda **k: (_ for _ in ()).throw(RuntimeError("fail2"))
    )
    svc = AdaptiveTriggerService(data_bot, cap_bot, bus, interval=0)
    svc.running = True
    monkeypatch.setattr("menace.adaptive_trigger_service.time.sleep", _stop_after_first(svc))
    caplog.set_level("ERROR")
    with pytest.raises(SystemExit):
        svc._loop()
    text = caplog.text
    assert "fetch errors failed" in text and "energy score failed" in text
    assert svc.failure_count == 2
