import logging
import os

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import pytest

from menace.unified_event_bus import UnifiedEventBus


def test_persist_and_replay(tmp_path):
    db_path = tmp_path / "events.db"
    bus1 = UnifiedEventBus(str(db_path))
    bus1.publish("t", {"v": 1})
    bus1.publish("t", {"v": 2})

    events: list[dict] = []
    bus2 = UnifiedEventBus(str(db_path))
    bus2.subscribe("t", lambda topic, ev: events.append(ev))
    bus2.replay()

    assert events == [{"v": 1}, {"v": 2}]

    bus2.publish("t", {"v": 3})
    assert events[-1] == {"v": 3}


def test_callback_error_logged_and_collected(caplog):
    bus = UnifiedEventBus(collect_errors=True)

    def boom(t, e):
        raise RuntimeError("boom")

    bus.subscribe("x", boom)
    caplog.set_level(logging.ERROR)
    bus.publish("x", {})
    assert "subscriber failed" in caplog.text
    assert isinstance(bus.callback_errors[0], RuntimeError)


def test_callback_error_rethrow():
    bus = UnifiedEventBus(rethrow_errors=True)

    def boom(t, e):
        raise RuntimeError("boom")

    bus.subscribe("x", boom)
    with pytest.raises(RuntimeError):
        bus.publish("x", {})


def test_persist_error_logged(monkeypatch, caplog):
    bus = UnifiedEventBus(":memory:")

    class BadDB:
        def execute(self, *a, **k):
            raise RuntimeError("fail")

        def commit(self):
            pass

    bus._persist = BadDB()
    caplog.set_level(logging.ERROR)
    bus.publish("x", {"v": 1})
    assert "failed persisting event" in caplog.text


def test_flag_for_review_processing():
    bus = UnifiedEventBus()
    bus.flag_for_review("b1")
    import time
    time.sleep(0.05)
    assert bus.last_review_event == {"bot_id": "b1"}


class DummyReviewer:
    def __init__(self) -> None:
        self.events: list[object] = []

    def handle(self, event: object) -> None:
        self.events.append(event)


def test_automated_reviewer_called():
    reviewer = DummyReviewer()
    bus = UnifiedEventBus(reviewer=reviewer)
    bus.flag_for_review("b2")
    assert reviewer.events[-1] == {"bot_id": "b2"}
