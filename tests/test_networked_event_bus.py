import subprocess
import time
import pytest

pytest.importorskip("pika")

from menace.networked_event_bus import NetworkedEventBus


def start_server():
    subprocess.run(["service", "rabbitmq-server", "start"], check=False)


def test_multiple_topics():
    start_server()
    bus = NetworkedEventBus()

    events_a: list[dict] = []
    events_b: list[dict] = []

    bus.subscribe("a", lambda t, e: events_a.append(e))
    bus.subscribe("b", lambda t, e: events_b.append(e))

    bus.publish("a", {"v": 1})
    bus.publish("b", {"v": 2})
    time.sleep(0.5)

    assert {"v": 1} in events_a
    assert {"v": 2} in events_b
    bus.close()


def test_no_events_after_close():
    start_server()
    bus = NetworkedEventBus()

    events: list[dict] = []

    bus.subscribe("c", lambda t, e: events.append(e))
    bus.publish("c", {"v": 1})
    time.sleep(0.5)
    bus.close()

    bus2 = NetworkedEventBus()
    bus2.publish("c", {"v": 2})
    time.sleep(0.5)

    assert events == [{"v": 1}]
    bus2.close()
