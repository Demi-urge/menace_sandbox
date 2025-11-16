import types
import queue as queue_module
import sys
import time

import pytest

# Build fake pika with in-memory queues
class FakeChannel:
    def __init__(self, bus):
        self.bus = bus
        self.closed = False
    def queue_declare(self, queue=None, durable=True):
        self.bus.setdefault(queue, queue_module.Queue())
    def basic_publish(self, exchange, routing_key, body, properties=None):
        q = self.bus.setdefault(routing_key, queue_module.Queue())
        q.put(body)
    def consume(self, queue=None, inactivity_timeout=1):
        q = self.bus.setdefault(queue, queue_module.Queue())
        while True:
            try:
                body = q.get(timeout=inactivity_timeout)
                yield types.SimpleNamespace(delivery_tag=1), None, body
            except queue_module.Empty:
                yield None, None, None
    def basic_ack(self, tag):
        pass
    def close(self):
        self.closed = True

class FakeConnection:
    def __init__(self, params):
        self.bus = FakePika.bus
    def channel(self):
        return FakeChannel(self.bus)
    def close(self):
        pass

class FakeParameters:
    def __init__(self, host=None):
        self.host = host

class FakeProperties:
    def __init__(self, delivery_mode=2):
        self.delivery_mode = delivery_mode

class FakePika(types.ModuleType):
    bus = {}
    BlockingConnection = FakeConnection
    ConnectionParameters = FakeParameters
    BasicProperties = FakeProperties


def test_unified_event_bus_delegates(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pika', FakePika('pika'))
    from menace.unified_event_bus import UnifiedEventBus
    bus1 = UnifiedEventBus(rabbitmq_host='localhost')
    bus2 = UnifiedEventBus(rabbitmq_host='localhost')
    events = []
    bus2.subscribe('x', lambda t, e: events.append(e))
    bus1.publish('x', {'v': 1})
    time.sleep(0.1)
    assert {'v': 1} in events
    bus1.close()
    bus2.close()
