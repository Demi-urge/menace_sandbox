from menace_sandbox.gpt_memory import GPTMemoryManager


class DummyBus:
    def __init__(self):
        self.events = []

    def publish(self, topic, payload):
        self.events.append((topic, payload))


def test_log_interaction_publishes_event():
    bus = DummyBus()
    mgr = GPTMemoryManager(":memory:", event_bus=bus)
    mgr.log_interaction("p", "r", tags=["x"])
    assert bus.events
    topic, payload = bus.events[0]
    assert topic == "memory:new"
    assert payload["prompt"] == "p"
    assert payload["tags"] == ["x"]
    assert "response" not in payload
    assert "ts" not in payload

