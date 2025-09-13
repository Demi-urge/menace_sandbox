import sys
import types

stub_cbi = types.ModuleType("menace.coding_bot_interface")
stub_cbi.self_coding_managed = lambda cls: cls
stub_cbi.manager_generate_helper = lambda *_a, **_k: None
sys.modules["menace.coding_bot_interface"] = stub_cbi

class UnifiedEventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list] = {}

    def subscribe(self, topic: str, handler):
        self._subs.setdefault(topic, []).append(handler)

    def publish(self, topic: str, payload):
        for h in self._subs.get(topic, []):
            h(topic, payload)


def test_threshold_event_published(monkeypatch, tmp_path):
    stub = types.ModuleType("vector_metrics_db")
    monkeypatch.setitem(sys.modules, "menace.vector_metrics_db", stub)
    sys.modules.setdefault("sandbox_settings", sys.modules["menace.sandbox_settings"])
    import menace.data_bot as db  # noqa: WPS433
    monkeypatch.setattr(db, "psutil", None)
    settings = types.SimpleNamespace(
        self_coding_roi_drop=-0.1,
        self_coding_error_increase=1.0,
        self_coding_test_failure_increase=0.0,
        bot_thresholds={
            "bot": types.SimpleNamespace(roi_drop=-5.0, error_threshold=5.0)
        },
    )
    bus = UnifiedEventBus()
    mdb = db.MetricsDB(tmp_path / "m.db")
    bot = db.DataBot(
        mdb,
        event_bus=bus,
        settings=settings,
        roi_drop_threshold=-5.0,
        error_threshold=5.0,
    )
    events: list[dict] = []
    degraded: list[dict] = []
    bus.subscribe("bot:degraded", lambda _t, e: degraded.append(e))
    bot.subscribe_threshold_breaches(lambda e: events.append(e))
    bot.collect("bot", revenue=10.0, expense=0.0, errors=0)
    bot.collect("bot", revenue=0.0, expense=0.0, errors=10)
    assert len(events) == 1
    assert events[0]["roi_breach"] and events[0]["error_breach"]
    assert not events[0]["test_failure_breach"]
    assert degraded and degraded[0]["roi_breach"]
    # Degradation events provide summary metrics and deltas for downstream consumers
    assert "roi_drop" in degraded[0]
    assert "error_rate" in degraded[0]
    assert "tests_failed" in degraded[0]
    assert "delta_roi" in degraded[0]
    assert "delta_errors" in degraded[0]
    assert "delta_tests_failed" in degraded[0]


def test_check_degradation_callback(monkeypatch, tmp_path):
    stub = types.ModuleType("vector_metrics_db")
    monkeypatch.setitem(sys.modules, "menace.vector_metrics_db", stub)
    sys.modules.setdefault("sandbox_settings", sys.modules["menace.sandbox_settings"])
    import menace.data_bot as db  # noqa: WPS433
    monkeypatch.setattr(db, "psutil", None)
    settings = types.SimpleNamespace(
        self_coding_roi_drop=-0.1,
        self_coding_error_increase=1.0,
        self_coding_test_failure_increase=0.0,
        bot_thresholds={},
    )
    mdb = db.MetricsDB(tmp_path / "m.db")
    bot = db.DataBot(
        mdb,
        settings=settings,
        roi_drop_threshold=-0.1,
        error_threshold=1.0,
    )
    events: list[dict] = []
    bot.check_degradation("bot", roi=10.0, errors=0.0)
    bot.check_degradation("bot", roi=0.0, errors=10.0, callback=lambda e: events.append(e))
    assert events and events[0]["roi_breach"] and events[0]["error_breach"]
    assert "bot" in bot._thresholds


def test_forecasting_model_detection(monkeypatch, tmp_path):
    stub = types.ModuleType("vector_metrics_db")
    monkeypatch.setitem(sys.modules, "menace.vector_metrics_db", stub)
    sys.modules.setdefault("sandbox_settings", sys.modules["menace.sandbox_settings"])
    import menace.data_bot as db  # noqa: WPS433
    monkeypatch.setattr(db, "psutil", None)
    settings = types.SimpleNamespace(
        self_coding_roi_drop=-0.1,
        self_coding_error_increase=1.0,
        self_coding_test_failure_increase=0.0,
        bot_thresholds={},
    )
    mdb = db.MetricsDB(tmp_path / "m.db")
    bot = db.DataBot(
        mdb,
        settings=settings,
        roi_drop_threshold=-0.1,
        error_threshold=1.0,
    )
    for _ in range(5):
        assert not bot.check_degradation("bot", roi=10.0, errors=0.0, test_failures=0.0)
    # Slight changes stay within the model's confidence interval
    assert not bot.check_degradation("bot", roi=9.95, errors=0.1, test_failures=0.0)
    # Significant deviation triggers degradation
    assert bot.check_degradation("bot", roi=0.0, errors=5.0, test_failures=3.0)
    hist = bot._forecast_history["bot"]
    assert len(hist["roi"]) >= 7
