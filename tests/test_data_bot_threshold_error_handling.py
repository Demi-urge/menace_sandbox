import logging

import menace.data_bot as db
from menace.data_bot import DataBot, MetricsDB


class DummyBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def publish(self, topic: str, payload: object) -> None:
        self.events.append((topic, payload))

    def subscribe(self, _topic: str, _handler) -> None:  # pragma: no cover - simple stub
        pass


def _has_event(bus: DummyBus, bot: str, error: str) -> bool:
    return any(
        topic == "data:threshold_update_failed"
        and payload.get("bot") == bot
        and error in payload.get("error", "")
        for topic, payload in bus.events
    )


def test_reload_thresholds_failure_emits_log_and_event(tmp_path, monkeypatch, caplog):
    db_path = tmp_path / "metrics.db"
    metrics = MetricsDB(path=db_path)
    bus = DummyBus()
    bot = DataBot(db=metrics, event_bus=bus, start_server=False)

    def bad_reload(name: str) -> None:
        raise RuntimeError("reload boom")

    monkeypatch.setattr(bot, "reload_thresholds", bad_reload)
    with caplog.at_level(logging.WARNING, logger=db.__name__):
        bot.check_degradation("alpha", 0.0, 0.0)
    assert any("reload_thresholds failed" in m for m in caplog.messages)
    assert _has_event(bus, "alpha", "reload boom")


def test_save_thresholds_failure_emits_log_and_event(tmp_path, monkeypatch, caplog):
    db_path = tmp_path / "metrics.db"
    metrics = MetricsDB(path=db_path)
    bus = DummyBus()
    bot = DataBot(db=metrics, event_bus=bus, start_server=False)

    monkeypatch.setattr(bot, "reload_thresholds", lambda _b: None)

    def bad_save(*_a, **_k):
        raise RuntimeError("save boom")

    monkeypatch.setattr(db, "persist_sc_thresholds", bad_save)
    with caplog.at_level(logging.WARNING, logger=db.__name__):
        bot.check_degradation("beta", 0.0, 0.0)
    assert any("update_thresholds failed" in m for m in caplog.messages)
    assert _has_event(bus, "beta", "save boom")


def test_reload_thresholds_recursion_guard(tmp_path, monkeypatch):
    db_path = tmp_path / "metrics.db"
    metrics = MetricsDB(path=db_path)
    bot = DataBot(db=metrics, start_server=False)

    def boom(*_a, **_k):
        raise RuntimeError("persist exploded")

    monkeypatch.setattr(db, "persist_sc_thresholds", boom)

    class ReentrantLogger:
        def __init__(self, target):
            self.target = target
            self.calls = 0

        def exception(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                self.target.reload_thresholds("alpha")

    logger_stub = ReentrantLogger(bot)
    bot.logger = logger_stub  # type: ignore[assignment]

    rt = bot.reload_thresholds("alpha")
    assert isinstance(rt, db.ROIThresholds)
    assert logger_stub.calls == 1
