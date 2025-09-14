import types
import sys

# provide stubs for optional dependencies
stub_cbi = types.ModuleType("coding_bot_interface")
stub_cbi.self_coding_managed = lambda *a, **k: (lambda cls: cls)
stub_cbi.manager_generate_helper = lambda *_a, **_k: None
sys.modules["coding_bot_interface"] = stub_cbi
sys.modules["menace.coding_bot_interface"] = stub_cbi

class UnifiedEventBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def publish(self, topic: str, payload: dict) -> None:
        self.events.append((topic, payload))

    def subscribe(self, topic: str, handler):
        pass


def _setup(monkeypatch, tmp_path):
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
    bus = UnifiedEventBus()
    bot = db.DataBot(mdb, event_bus=bus, settings=settings)
    return db, bot, bus


def test_reload_thresholds_recalibrates(monkeypatch, tmp_path):
    db, bot, bus = _setup(monkeypatch, tmp_path)
    tracker = db.BaselineTracker(window=5)
    for roi, err, fail in zip([1, 0, 1, 0, 1], [0, 3, 0, 3, 0], [0, 1, 0, 1, 0]):
        tracker.update(roi=roi, errors=err, tests_failed=fail)
    bot._baseline["bot"] = tracker
    raw = db.SelfCodingThresholds(
        roi_drop=-0.1,
        error_increase=1.0,
        test_failure_increase=0.0,
        auto_recalibrate=True,
    )
    monkeypatch.setattr(db, "load_sc_thresholds", lambda *a, **k: raw)
    monkeypatch.setattr(db, "_load_sc_thresholds", lambda *a, **k: raw)
    calls: list[tuple] = []
    monkeypatch.setattr(db, "persist_sc_thresholds", lambda *a, **k: calls.append((a, k)))
    rt = bot.reload_thresholds("bot")
    assert rt.roi_drop < -0.1
    assert rt.error_threshold > 1.0
    assert rt.test_failure_threshold > 0.0
    assert calls
    recal = [e for t, e in bus.events if t == "data:thresholds_recalibrated"]
    assert recal and recal[0]["bot"] == "bot"


def test_reload_thresholds_respects_toggle(monkeypatch, tmp_path):
    db, bot, bus = _setup(monkeypatch, tmp_path)
    tracker = db.BaselineTracker(window=5)
    for roi, err, fail in zip([1, 0, 1, 0, 1], [0, 3, 0, 3, 0], [0, 1, 0, 1, 0]):
        tracker.update(roi=roi, errors=err, tests_failed=fail)
    bot._baseline["bot"] = tracker
    raw = db.SelfCodingThresholds(
        roi_drop=-0.1,
        error_increase=1.0,
        test_failure_increase=0.0,
        auto_recalibrate=False,
    )
    monkeypatch.setattr(db, "load_sc_thresholds", lambda *a, **k: raw)
    monkeypatch.setattr(db, "_load_sc_thresholds", lambda *a, **k: raw)
    bot.reload_thresholds("bot")
    recal = [e for t, e in bus.events if t == "data:thresholds_recalibrated"]
    assert not recal
