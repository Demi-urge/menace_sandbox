import os

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

from menace import config


class DummyBus:
    def __init__(self) -> None:
        self.events = []

    def publish(self, topic: str, payload: object) -> None:
        self.events.append((topic, payload))


def test_reload_emits_event(monkeypatch):
    bus = DummyBus()
    config.set_event_bus(bus)

    # ensure initial config is loaded
    config.get_config()

    # change logging verbosity to trigger a diff
    monkeypatch.setattr(config, "_OVERRIDES", {"logging": {"verbosity": "INFO"}})

    config.reload()

    assert bus.events, "expected a config:reload event"
    topic, payload = bus.events[-1]
    assert topic == "config:reload"
    assert payload["config"]["logging"]["verbosity"] == "INFO"
    assert payload["diff"]["logging"]["verbosity"] == "INFO"

    # cleanup
    config.set_event_bus(None)
