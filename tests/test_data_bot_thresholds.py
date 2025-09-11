import sys
import types

from menace.unified_event_bus import UnifiedEventBus


def test_threshold_event_published(monkeypatch, tmp_path):
    stub = types.ModuleType("vector_metrics_db")
    monkeypatch.setitem(sys.modules, "menace.vector_metrics_db", stub)
    sys.modules.setdefault("sandbox_settings", sys.modules["menace.sandbox_settings"])
    import menace.data_bot as db  # noqa: WPS433
    monkeypatch.setattr(db, "psutil", None)
    settings = types.SimpleNamespace(
        self_coding_roi_drop=-0.1,
        self_coding_error_increase=1.0,
        bot_thresholds={
            "bot": types.SimpleNamespace(roi_drop=-5.0, error_threshold=5.0)
        },
    )
    bus = UnifiedEventBus()
    mdb = db.MetricsDB(tmp_path / "m.db")
    bot = db.DataBot(mdb, event_bus=bus, settings=settings)
    events: list[dict] = []
    degraded: list[dict] = []
    bus.subscribe("bot:degraded", lambda _t, e: degraded.append(e))
    bot.subscribe_threshold_breaches(lambda e: events.append(e))
    bot.collect("bot", revenue=10.0, expense=0.0, errors=0)
    bot.collect("bot", revenue=0.0, expense=0.0, errors=10)
    assert events and events[0]["roi_breach"] and events[0]["error_breach"]
    assert degraded and degraded[0]["roi_breach"]


def test_check_degradation_callback(monkeypatch, tmp_path):
    stub = types.ModuleType("vector_metrics_db")
    monkeypatch.setitem(sys.modules, "menace.vector_metrics_db", stub)
    sys.modules.setdefault("sandbox_settings", sys.modules["menace.sandbox_settings"])
    import menace.data_bot as db  # noqa: WPS433
    monkeypatch.setattr(db, "psutil", None)
    settings = types.SimpleNamespace(
        self_coding_roi_drop=-0.1,
        self_coding_error_increase=1.0,
        bot_thresholds={},
    )
    mdb = db.MetricsDB(tmp_path / "m.db")
    bot = db.DataBot(mdb, settings=settings)
    events: list[dict] = []
    bot.check_degradation("bot", roi=10.0, errors=0.0)
    bot.check_degradation("bot", roi=0.0, errors=10.0, callback=lambda e: events.append(e))
    assert events and events[0]["roi_breach"] and events[0]["error_breach"]
