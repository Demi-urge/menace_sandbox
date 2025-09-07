import json
import types
import menace.error_logger as elog


class DummyBuilder:
    def refresh_db_weights(self):
        pass


class StubReplicator:
    instances = []

    def __init__(self, topic="menace.telemetry", hosts=None, *, sentry=None, disk_path=None):
        self.topic = topic
        self.hosts = hosts
        self.events = []
        StubReplicator.instances.append(self)

    def replicate(self, event):
        self.events.append(event)

    def flush(self):
        pass


def test_log_roi_cap_emits_roibottleneck_event(monkeypatch):
    monkeypatch.setenv("KAFKA_HOSTS", "broker:9092")
    monkeypatch.setattr(elog, "TelemetryReplicator", StubReplicator)
    monkeypatch.setattr(elog, "get_embedder", lambda: None)
    db = types.SimpleNamespace(add_telemetry=lambda e: None)
    logger = elog.ErrorLogger(db, context_builder=DummyBuilder())
    profile = {
        "weights": {
            "profitability": 0.25,
            "efficiency": 0.2,
            "reliability": 0.15,
            "resilience": 0.1,
            "maintainability": 0.1,
            "security": 0.1,
            "latency": -0.05,
            "energy": -0.05,
        },
        "veto": {"security": {"min": 0.4}},
    }
    metrics = {
        "profitability": 1.0,
        "efficiency": 1.0,
        "reliability": 1.0,
        "resilience": 1.0,
        "maintainability": 1.0,
        "security": 0.2,
        "latency": 1.0,
        "energy": 0.5,
    }
    suggestions = logger.log_roi_cap("wf1", metrics, profile)
    inst = StubReplicator.instances[0]
    assert inst.events
    event = inst.events[0]
    assert event.root_cause == "ROIBottleneck"
    assert event.task_id == "wf1"
    assert suggestions
    payload = json.loads(event.stack_trace.split(": ", 1)[1])
    assert payload["metrics"]["security"] == 0.2
    assert [tuple(s) for s in payload["suggestions"]] == suggestions


def test_log_roi_cap_logs_when_no_replicator(monkeypatch):
    monkeypatch.delenv("KAFKA_HOSTS", raising=False)
    monkeypatch.setattr(elog, "get_embedder", lambda: None)
    db = types.SimpleNamespace(add_telemetry=lambda e: None)
    logger = elog.ErrorLogger(db, context_builder=DummyBuilder())

    class StubLogger:
        def __init__(self):
            self.messages = []

        def info(self, msg):
            self.messages.append(msg)

        def error(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

    stub = StubLogger()
    logger.logger = stub
    profile = {
        "weights": {
            "profitability": 0.25,
            "efficiency": 0.2,
            "reliability": 0.15,
            "resilience": 0.1,
            "maintainability": 0.1,
            "security": 0.1,
            "latency": -0.05,
            "energy": -0.05,
        },
        "veto": {"security": {"min": 0.4}},
    }
    metrics = {
        "profitability": 1.0,
        "efficiency": 1.0,
        "reliability": 1.0,
        "resilience": 1.0,
        "maintainability": 1.0,
        "security": 0.2,
        "latency": 1.0,
        "energy": 0.5,
    }
    suggestions = logger.log_roi_cap("wf2", metrics, profile)
    assert stub.messages
    payload = json.loads(stub.messages[0].split(": ", 1)[1])
    assert payload["workflow_id"] == "wf2"
    assert [tuple(s) for s in payload["suggestions"]] == suggestions
