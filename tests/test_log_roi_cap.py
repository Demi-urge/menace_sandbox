import json
import types
import menace.error_logger as elog


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
    logger = elog.ErrorLogger(db)
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
