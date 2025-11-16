import time

from menace import mutation_logger as ml
from menace.evolution_history_db import EvolutionHistoryDB


class DummyBus:
    def __init__(self):
        self.published = []

    def publish(self, topic, event):
        self.published.append((topic, event))


def test_log_mutation_publishes_event(monkeypatch, tmp_path):
    db = EvolutionHistoryDB(tmp_path / "e.db")
    monkeypatch.setattr(ml, "_history_db", db)

    class ImmediateThread:
        def __init__(self, target, daemon=True):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(ml, "Thread", ImmediateThread)

    bus = DummyBus()
    ml.set_event_bus(bus)
    event_id = ml.log_mutation("change", "reason", "trigger", 1.0, workflow_id=1)
    assert bus.published[0][0] == "mutation_recorded"
    assert bus.published[0][1]["event_id"] == event_id

    ml.set_event_bus(None)


def test_log_mutation_bus_failure_non_blocking(monkeypatch, tmp_path):
    db = EvolutionHistoryDB(tmp_path / "e.db")
    monkeypatch.setattr(ml, "_history_db", db)

    class SlowFailBus:
        def publish(self, topic, event):
            time.sleep(0.2)
            raise RuntimeError("boom")

    bus = SlowFailBus()
    ml.set_event_bus(bus)

    def fake_publish_with_retry(bus_arg, topic, event, **kwargs):
        bus_arg.publish(topic, event)

    monkeypatch.setattr(ml, "publish_with_retry", fake_publish_with_retry)
    start = time.perf_counter()
    ml.log_mutation("change", "reason", "trigger", 1.0, workflow_id=1)
    duration = time.perf_counter() - start
    assert duration < 0.1
    ml.set_event_bus(None)
